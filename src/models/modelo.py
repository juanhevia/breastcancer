import torch
import torch.nn as nn


class BreastCNN(nn.Module):
    """
    CNN 2D desde cero para predecir pCR a partir de un corte DCE-MRI
    de 3 fases (PRE, EARLY, LATE) apiladas como canales.

    Entrada:  (batch, 3, 256, 256)
    Salida:   (batch,)  -> un logit por muestra (sin sigmoide,
              se aplica dentro de BCEWithLogitsLoss durante el
              entrenamiento, y manualmente en inferencia).
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1: 256x256 -> 128x128
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 2: 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 3: 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 32x32 -> 1x1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x.squeeze(1)


class BreastCNNV2(nn.Module):
    """CNN propia, sin preentrenamiento, con más contexto espacial que el baseline.

    Conserva PRE/EARLY/LATE y la salida de un logit. GroupNorm no depende de
    estadísticas del lote; media y máximo espacial resumen señales difusas y
    localizadas. Estos cambios son hipótesis que hay que validar, no garantías.
    """

    def __init__(self, canales_entrada=3):
        super().__init__()
        bloques = []
        entrada = canales_entrada
        for canales in (16, 32, 64, 128):
            bloques.extend([
                nn.Conv2d(entrada, canales, 3, padding=1, bias=False),
                nn.GroupNorm(8, canales),
                nn.ReLU(),
                nn.Conv2d(canales, canales, 3, padding=1, bias=False),
                nn.GroupNorm(8, canales),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
            ])
            entrada = canales
        self.features = nn.Sequential(*bloques)
        self.media = nn.AdaptiveAvgPool2d(1)
        self.maximo = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.cat((self.media(x), self.maximo(x)), dim=1)
        return self.classifier(x).squeeze(1)


class RealceDCE(nn.Module):
    """Cambio de base fijo: PRE, EARLY-PRE, LATE-EARLY, sin dividir por PRE.

    Conserva toda la información de las tres fases: EARLY = PRE + diferencia1
    y LATE = EARLY + diferencia2. Los cambios negativos son válidos.
    """

    def forward(self, x):
        return torch.cat((x[:, 0:1], x[:, 1:2] - x[:, 0:1],
                          x[:, 2:3] - x[:, 1:2]), dim=1)


class BreastCNNV3(nn.Module):
    """CNN propia pequeña con diferencias temporales y resumen espacial 2×2.

    Entrada externa idéntica: PRE/EARLY/LATE divididas entre 255. El cambio de
    base está dentro de la red, también en inferencia; no usa datos clínicos.
    """

    def __init__(self):
        super().__init__()
        self.realce = RealceDCE()
        bloques = []
        entrada = 3
        for canales in (16, 24, 48, 64):
            bloques.extend([
                nn.Conv2d(entrada, canales, 3, padding=1, bias=False),
                nn.BatchNorm2d(canales), nn.ReLU(),
                nn.Conv2d(canales, canales, 3, padding=1, bias=False),
                nn.BatchNorm2d(canales), nn.ReLU(), nn.MaxPool2d(2),
            ])
            if canales >= 48:
                bloques.append(nn.Dropout2d(0.1))
            entrada = canales
        self.features = nn.Sequential(*bloques)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(64 * 2 * 2, 32),
            nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(self.realce(x))
        return self.classifier(self.pool(x)).squeeze(1)


class BreastCNNV5(BreastCNNV2):
    """V2 a media resolución, con media 2×2 antes de las convoluciones.

    Recibe las mismas fases originales /255. La reducción está dentro del
    modelo para usarla también en inferencia. No recorta regiones ni selecciona
    lesiones. Busca menor coste y sensibilidad al detalle fino; puede perder
    información útil y no garantiza mejorar la AUC.
    """

    def __init__(self):
        super().__init__()
        self.reducir = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return super().forward(self.reducir(x))


class ContrasteConjuntoDCE(nn.Module):
    """Vista auxiliar con una única ventana de intensidad para las tres fases.

    Percentiles 5/95 de píxeles >0 del corte completo (PRE/EARLY/LATE juntos).
    No estima estadísticas del lote ni usa etiquetas. Ganancia limitada a 3
    para evitar amplificar excesivamente cortes de bajo contraste. Se recortan
    las colas a [0,1]; por eso V6 conserva también los canales originales.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("percentiles", torch.tensor([.05, .95]))
        self.register_buffer("ganancia_maxima", torch.tensor(3.0))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Se espera (batch, 3, alto, ancho) PRE/EARLY/LATE.")
        # float32 para quantile incluso si más adelante se activa precisión mixta.
        valores = x.detach().float().flatten(1)
        sin_fondo = valores.masked_fill(valores <= 0, float("nan"))
        limites = torch.nanquantile(sin_fondo, self.percentiles.float(), dim=1)
        bajo, alto = limites[0], limites[1]
        validos = torch.isfinite(bajo) & torch.isfinite(alto) & ((alto - bajo) > 1e-6)
        bajo = torch.nan_to_num(bajo).reshape(-1, 1, 1, 1)
        rango = torch.nan_to_num(alto - limites[0]).clamp_min(
            1 / self.ganancia_maxima).reshape(-1, 1, 1, 1)
        ajustado = ((x - bajo) / rango).clamp(0, 1)
        # Cortes vacíos/constantes: conservarlos, sin NaN ni contraste artificial.
        return torch.where(validos.reshape(-1, 1, 1, 1), ajustado, x).to(x.dtype)


class BreastCNNV6(BreastCNNV2):
    """V2 con seis canales internos: fases originales + contraste conjunto.

    Entrada externa: tres PNG /255, a resolución completa. Preprocesado fijo
    dentro del modelo, idéntico en train, validación e inferencia. CNN propia
    desde cero; no incorpora imágenes nuevas, pesos preentrenados ni clínica.
    """

    def __init__(self):
        super().__init__(canales_entrada=6)
        self.contraste = ContrasteConjuntoDCE()

    def preparar_entrada(self, x):
        return torch.cat((x, self.contraste(x)), dim=1)

    def forward(self, x):
        return super().forward(self.preparar_entrada(x))


def contar_parametros(modelo):
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    return entrenables, total


if __name__ == "__main__":
    modelo = BreastCNN()

    # Verificación de forma con un batch de prueba
    x = torch.randn(4, 3, 256, 256)
    y = modelo(x)
    print(f"Entrada:  {tuple(x.shape)}")
    print(f"Salida:   {tuple(y.shape)}")

    entrenables, total = contar_parametros(modelo)
    print(f"Parámetros entrenables: {entrenables:,}")
    print(f"Parámetros totales:     {total:,}")
