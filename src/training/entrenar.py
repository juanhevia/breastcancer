"""Bucle de entrenamiento y CLI: python -m src.training.entrenar --help."""

import argparse
import json
import math
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data import utils_caso as uc
from src.data.aumentos import AumentosDCE
from src.models.modelo import BreastCNN, BreastCNNV2, contar_parametros


# Los argumentos explícitos de la CLI prevalecen sobre estos valores.
EXPERIMENTOS = {
    "baseline": dict(modelo="baseline", epocas=5, lr=1e-4, paciencia=7,
                     optimizador="adam", weight_decay=0.0, aumentos=False,
                     reducir_lr=False, seleccion="loss"),
    "v2": dict(modelo="v2", epocas=40, lr=1e-4, paciencia=12,
               optimizador="adamw", weight_decay=1e-3, aumentos=True,
               reducir_lr=True, seleccion="auc"),
}


def guardar_json(ruta, contenido):
    """Las métricas indefinidas se guardan como null, no como NaN inválido."""
    def limpiar(valor):
        if isinstance(valor, dict):
            return {k: limpiar(v) for k, v in valor.items()}
        if isinstance(valor, list):
            return [limpiar(v) for v in valor]
        if isinstance(valor, float) and not math.isfinite(valor):
            return None
        return valor
    Path(ruta).write_text(json.dumps(limpiar(contenido), indent=2, ensure_ascii=False))


def elegir_dispositivo():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def entrenar(
    modelo, train_ds, val_ds, val_samples, epocas=40, lr=1e-4,
    batch_size=32, pos_weight=None, paciencia=7, dispositivo=None,
    metodo_agregacion="mean", umbral=0.5, nombre="modelo", num_workers=0,
    salida=None, semilla=42, optimizador="adam", weight_decay=0.0,
    reducir_lr=False, seleccion="loss",
):
    """Devuelve (historial, mejor_state_dict, mejor_epoca, resultado_paciente).

    Cada época recorre TODOS los cortes de train_ds y después valida.
    Selección/parada: mínima pérdida o máxima AUC por paciente en validación.
    val_samples debe coincidir exactamente con el orden de val_ds.
    """
    if min(epocas, batch_size, paciencia) < 1 or lr <= 0 or num_workers < 0:
        raise ValueError("Épocas, batch, paciencia y lr deben ser positivos; workers >= 0.")
    if not math.isfinite(lr) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("lr y weight_decay deben ser finitos; weight_decay >= 0.")
    if optimizador not in ("adam", "adamw") or seleccion not in ("loss", "auc"):
        raise ValueError("Optimizador o criterio de selección no válido.")
    if not len(train_ds) or not len(val_ds):
        raise ValueError("Entrenamiento y validación deben contener muestras.")
    if len(val_ds) != len(val_samples):
        raise ValueError("val_ds y val_samples no tienen la misma longitud.")
    if seleccion == "auc" and val_samples.pCR.nunique() != 2:
        raise ValueError("Seleccionar por AUC requiere las dos clases en validación.")
    if hasattr(val_ds, "filas"):
        columnas = ["sample_id", "patient_id", "pCR"]
        if not val_ds.filas[columnas].reset_index(drop=True).equals(
            val_samples[columnas].reset_index(drop=True)
        ):
            raise ValueError("val_samples no coincide con el orden de val_ds.")
    if pos_weight is not None and (not math.isfinite(pos_weight) or pos_weight <= 0):
        raise ValueError("pos_weight debe ser un número positivo finito.")
    if not 0 <= umbral <= 1 or metodo_agregacion not in uc.METODOS_AGREGACION:
        raise ValueError("Umbral o método de agregación no válido.")

    dispositivo = torch.device(dispositivo or elegir_dispositivo())
    modelo.to(dispositivo)
    salida = Path(salida) if salida is not None else None
    if salida:
        salida.mkdir(parents=True, exist_ok=True)
    generador = torch.Generator().manual_seed(semilla)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=generador)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    pw = (torch.tensor(float(pos_weight), device=dispositivo)
          if pos_weight is not None else None)
    criterio = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    clase_optimizador = torch.optim.AdamW if optimizador == "adamw" else torch.optim.Adam
    optimizador = clase_optimizador(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    planificador = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="max" if seleccion == "auc" else "min",
        factor=0.5, patience=3, threshold=0.0, threshold_mode="abs",
        min_lr=min(1e-6, lr),
    ) if reducir_lr else None)
    historial = {k: [] for k in (
        "epoca", "train_loss", "val_loss", "val_auc", "val_auc_paciente", "segundos",
        "val_accuracy", "val_sensibilidad", "val_especificidad", "val_precision", "val_f1", "lr"
    )}
    mejor_valor = -float("inf") if seleccion == "auc" else float("inf")
    mejor_state, mejor_epoca = None, 0
    epocas_sin_mejora = 0

    def validar():
        modelo.eval()
        perdida, probabilidades, etiquetas = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dispositivo), y.to(dispositivo).float()
                logits = modelo(x)
                perdida += criterio(logits, y).item() * len(x)
                probabilidades.append(torch.sigmoid(logits).cpu().numpy())
                etiquetas.append(y.cpu().numpy())
        return perdida / len(val_ds), np.concatenate(probabilidades), np.concatenate(etiquetas)

    for epoca in range(1, epocas + 1):
        t0 = time.perf_counter()
        lr_epoca = optimizador.param_groups[0]["lr"]
        modelo.train()
        loss_acumulada = 0.0
        for lote, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(dispositivo), y.to(dispositivo).float()
            optimizador.zero_grad(set_to_none=True)
            loss = criterio(modelo(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Pérdida no finita; revisa datos y tasa de aprendizaje.")
            loss.backward()
            optimizador.step()
            loss_acumulada += loss.item() * len(x)
            if lote % 50 == 0:
                print(f"[{nombre}] Época {epoca}: lote {lote}/{len(train_loader)}", flush=True)

        train_loss = loss_acumulada / len(train_ds)
        val_loss, probs, y_true = validar()
        if not math.isfinite(val_loss):
            raise RuntimeError("Pérdida de validación no finita.")
        val_auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else float("nan")
        res = uc.evaluar_por_paciente(probs, val_samples, umbral=umbral, metodo=metodo_agregacion)
        dt = time.perf_counter() - t0
        cm = res["matriz_confusion"]
        denominador_f1 = 2 * cm["VP"] + cm["FP"] + cm["FN"]
        f1 = 2 * cm["VP"] / denominador_f1 if denominador_f1 else float("nan")
        valores = (epoca, train_loss, val_loss, val_auc, res["auc"], dt,
                   res["accuracy"], res["sensibilidad"], res["especificidad"], res["precision"], f1, lr_epoca)
        for clave, valor in zip(historial, valores):
            historial[clave].append(valor)
        print(f"[{nombre}] Época {epoca:02d}/{epocas} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} AUC_corte={val_auc:.4f} "
              f"AUC_paciente={res['auc']:.4f} lr={lr_epoca:.2g} ({dt:.1f}s)", flush=True)

        valor = res["auc"] if seleccion == "auc" else val_loss
        if not math.isfinite(valor):
            raise RuntimeError("La métrica de selección no es finita.")
        mejora = valor > mejor_valor if seleccion == "auc" else valor < mejor_valor
        if mejora:
            mejor_valor, mejor_epoca = valor, epoca
            mejor_state = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            epocas_sin_mejora = 0
            if salida:
                torch.save(mejor_state, salida / "mejor_modelo.pt")
                guardar_json(salida / "mejor_validacion.json", {
                    "epoca": epoca, "seleccion": seleccion, "valor_seleccion": valor,
                    "val_loss": val_loss, **res,
                })
        else:
            epocas_sin_mejora += 1
        if salida:
            pd.DataFrame(historial).to_csv(salida / "historial.csv", index=False)
        if planificador:
            planificador.step(valor)
            siguiente_lr = optimizador.param_groups[0]["lr"]
            if siguiente_lr < lr_epoca:
                print(f"[{nombre}] lr para la siguiente época: {siguiente_lr:.2g}", flush=True)
        if epocas_sin_mejora >= paciencia:
            print(f"[{nombre}] Parada anticipada; mejor época: {mejor_epoca}.", flush=True)
            break

    modelo.load_state_dict(mejor_state)
    _, probs_val, _ = validar()
    resultado = uc.evaluar_por_paciente(probs_val, val_samples, umbral=umbral, metodo=metodo_agregacion)
    if salida:
        tabla = val_samples[["sample_id", "patient_id", "pCR"]].copy()
        tabla["probabilidad"] = probs_val
        tabla.to_csv(salida / "predicciones_validacion.csv", index=False)
        guardar_json(salida / "metricas.json", {
            "mejor_epoca": mejor_epoca, "seleccion": seleccion,
            "valor_seleccion": mejor_valor, **resultado,
        })
    return historial, mejor_state, mejor_epoca, resultado


def dibujar_curvas(historial, ruta):
    # Conserva el import usado en versiones anteriores del proyecto.
    from src.training.graficas import dibujar_curvas as dibujar
    dibujar(historial, ruta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experimento", choices=EXPERIMENTOS, default="baseline",
                   help="baseline conserva el entrenamiento original; v2 activa las mejoras")
    p.add_argument("--modelo", choices=("baseline", "v2"), default=None)
    p.add_argument("--epocas", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--paciencia", type=int, default=None)
    p.add_argument("--optimizador", choices=("adam", "adamw"), default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--seleccion", choices=("loss", "auc"), default=None,
                   help="Checkpoint y parada: menor BCE o mayor AUC por paciente")
    p.add_argument("--aumentos", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--reducir-lr", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--perdida", choices=("normal", "ponderada", "ambas"), default="ambas")
    p.add_argument("--dispositivo", choices=("cpu", "cuda", "mps"), default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--hilos", type=int, default=4, help="Hilos de CPU de PyTorch")
    p.add_argument("--umbral", type=float, default=0.5)
    p.add_argument("--agregacion", choices=uc.METODOS_AGREGACION, default="mean")
    p.add_argument("--datos", type=Path, default=uc.RAIZ)
    p.add_argument("--salida", type=Path, default=uc.RAIZ / "models")
    p.add_argument("--prueba", action="store_true", help="Solo 2 pacientes por clase y partición; NO mide rendimiento")
    args = p.parse_args()
    for clave, valor in EXPERIMENTOS[args.experimento].items():
        if getattr(args, clave) is None:
            setattr(args, clave, valor)
    if args.hilos < 1:
        p.error("--hilos debe ser positivo")
    torch.set_num_threads(args.hilos)
    train_df, val_df = uc.particion(uc.cargar_samples(args.datos), fold_val=args.fold)
    if args.prueba:
        def reducir(df):
            pacientes = df[["patient_id", "pCR"]].drop_duplicates()
            ids = pacientes.groupby("pCR", group_keys=False).sample(n=2, random_state=args.semilla).patient_id
            return df[df.patient_id.isin(ids)]
        train_df, val_df = reducir(train_df), reducir(val_df)
    for df in (train_df, val_df):
        faltan = [str(args.datos / ruta) for c in ("path_pre", "path_early", "path_late")
                  for ruta in df[c] if not (args.datos / ruta).is_file()]
        if faltan:
            p.error(f"Faltan {len(faltan)} imágenes; primera: {faltan[0]}")
    dispositivo = args.dispositivo or elegir_dispositivo()
    nombre = (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{args.experimento}"
              + ("_prueba" if args.prueba else ""))
    carpeta = args.salida / nombre
    carpeta.mkdir(parents=True, exist_ok=False)
    print(f"Dispositivo: {dispositivo} | resultados: {carpeta}", flush=True)
    print(f"Modelo: {args.modelo} | aumentos: {args.aumentos} | selección: {args.seleccion}", flush=True)
    print(f"Train: {len(train_df)} cortes / {train_df.patient_id.nunique()} pacientes; "
          f"validación: {len(val_df)} cortes / {val_df.patient_id.nunique()} pacientes", flush=True)
    perdidas = ("normal", "ponderada") if args.perdida == "ambas" else (args.perdida,)
    resumen = []
    for perdida in perdidas:
        random.seed(args.semilla)
        np.random.seed(args.semilla)
        torch.manual_seed(args.semilla)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.semilla)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        modelo = BreastCNNV2() if args.modelo == "v2" else BreastCNN()
        pw = uc.pos_weight(train_df) if perdida == "ponderada" else None
        salida = carpeta / perdida
        salida.mkdir()
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(dispositivo=dispositivo, perdida=perdida, pos_weight=pw,
                      python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__, pandas=pd.__version__,
                      sistema=platform.platform(), arquitectura=str(modelo),
                      parametros=contar_parametros(modelo)[0],
                      optimizador="AdamW" if args.optimizador == "adamw" else "Adam",
                      inicializacion="Valores por defecto de PyTorch, misma semilla en ambas pérdidas",
                      criterio_parada=("Máxima AUC por paciente en validación interna"
                                       if args.seleccion == "auc" else "Mínima pérdida de validación interna"),
                      planificador=(dict(nombre="ReduceLROnPlateau", factor=0.5, paciencia=3,
                                         min_lr=min(1e-6, args.lr), threshold=0.0)
                                    if args.reducir_lr else None),
                      aumento_entrenamiento=("Reflejo horizontal p=0.5; giro ±10 grados, "
                                             "bilineal, fondo cero; mismas transformaciones en las tres fases"
                                             if args.aumentos else "Ninguno"),
                      normalizacion="PRE, EARLY, LATE / 255; sin normalización ImageNet",
                      train_cortes=len(train_df), val_cortes=len(val_df),
                      train_pacientes=train_df.patient_id.nunique(),
                      val_pacientes=val_df.patient_id.nunique(),
                      clases_train={str(k): int(v) for k,v in train_df.pCR.value_counts().items()})
        guardar_json(salida / "configuracion.json", config)
        pd.concat([train_df.assign(uso="train"), val_df.assign(uso="validacion")])[
            ["sample_id", "patient_id", "pCR", "fold", "uso"]
        ].to_csv(salida / "particion.csv", index=False)
        if dispositivo == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        hist, _, epoca, res = entrenar(
            modelo, uc.BreastDCEDataset(train_df, args.datos,
                                       transform=AumentosDCE() if args.aumentos else None),
            uc.BreastDCEDataset(val_df, args.datos),
            val_df, epocas=args.epocas, lr=args.lr, batch_size=args.batch_size, pos_weight=pw,
            paciencia=args.paciencia, dispositivo=dispositivo, metodo_agregacion=args.agregacion,
            umbral=args.umbral, nombre=perdida, num_workers=args.num_workers,
            salida=salida, semilla=args.semilla,
            optimizador=args.optimizador, weight_decay=args.weight_decay,
            reducir_lr=args.reducir_lr, seleccion=args.seleccion,
        )
        duracion = time.perf_counter() - t0
        guardar_json(salida / "ejecucion.json", {
            "segundos_total_entrenamiento_y_evaluacion": duracion,
            "epocas_completadas": len(hist["epoca"]),
            "memoria_gpu_max_bytes": torch.cuda.max_memory_allocated() if dispositivo == "cuda" else None,
            "nota_memoria": "Pico CUDA registrado; pico de memoria CPU/MPS no medido",
        })
        from src.training.graficas import generar_graficas
        # Actualiza las mismas tres figuras al terminar cada modelo.
        generar_graficas(carpeta)
        resumen.append({"perdida": perdida, "mejor_epoca": epoca, **res})
        guardar_json(carpeta / "comparacion.json", resumen)
        print(f"Mejor época de {perdida}: {epoca}; AUC por paciente: {res['auc']:.4f}", flush=True)
    print(f"Resultados guardados en {carpeta}", flush=True)


if __name__ == "__main__":
    main()
