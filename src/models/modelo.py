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

    def __init__(self):
        super().__init__()
        bloques = []
        entrada = 3
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
