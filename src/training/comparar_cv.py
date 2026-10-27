"""Comparación v2/v7: cinco folds externos y un fold interno para selección.

Solo usa split=train. No consulta imágenes ni resultados del test oficial.
python -m src.training.comparar_cv --dispositivo cuda --perdida ambas
"""

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from src.data import utils_caso as uc
from src.data.aumentos import AumentosDCE
from src.models.modelo import BreastCNNV2, BreastCNNV7
from src.training.entrenar import entrenar, guardar_json


MODELOS = {"v2": BreastCNNV2, "v7": BreastCNNV7}
AVISO = (
    "CV de desarrollo: cinco folds externos con selección en un fold interno. "
    "No es test final ni CV anidada exhaustiva. La desviación entre folds no es un IC. "
    "Los modelos ya se diseñaron usando parte de estas pacientes; existe sesgo de selección previo."
)


def validar_samples(samples):
    claves = ["sample_id", "patient_id", "pCR", "fold", "split"]
    if samples[claves].isna().any().any() or samples.sample_id.duplicated().any():
        raise ValueError("Metadatos ausentes o sample_id duplicados.")
    if not samples.pCR.isin([0, 1]).all() or not samples.split.isin(["train", "test"]).all():
        raise ValueError("Etiquetas o splits no válidos.")
    for clave in ("pCR", "fold", "split"):
        if (samples.groupby("patient_id")[clave].nunique() != 1).any():
            raise ValueError(f"Una paciente tiene valores contradictorios de {clave}.")
    desarrollo = samples[samples.split == "train"]
    if not desarrollo.fold.isin(range(5)).all() or set(desarrollo.fold) != set(range(5)):
        raise ValueError("Desarrollo debe tener exactamente los cinco folds 0..4.")
    for fold in range(5):
        if desarrollo[desarrollo.fold == fold].pCR.nunique() != 2:
            raise ValueError(f"Fold {fold}: se necesitan ambas clases.")
    return desarrollo


def particiones(samples, fold_externo, prueba=False, semilla=42):
    """Tres grupos disjuntos; el externo nunca controla la parada ni el umbral."""
    if fold_externo not in range(5):
        raise ValueError("Fold externo fuera de 0..4.")
    desarrollo = validar_samples(samples)
    interno = (fold_externo + 1) % 5
    grupos = (
        desarrollo[~desarrollo.fold.isin([fold_externo, interno])],
        desarrollo[desarrollo.fold == interno],
        desarrollo[desarrollo.fold == fold_externo],
    )
    if prueba:
        reducidos = []
        for grupo in grupos:
            pacientes = grupo[["patient_id", "pCR"]].drop_duplicates()
            ids = pacientes.groupby("pCR", group_keys=False).sample(n=2, random_state=semilla).patient_id
            reducidos.append(grupo[grupo.patient_id.isin(ids)])
        grupos = tuple(reducidos)
    return tuple(g.reset_index(drop=True) for g in grupos)


def tabla_pacientes(predicciones):
    """Media de probabilidades por paciente, nunca votación por corte."""
    if not np.isfinite(predicciones.probabilidad).all() or not predicciones.probabilidad.between(0, 1).all():
        raise ValueError("Probabilidades no válidas.")
    if (predicciones.groupby("patient_id").pCR.nunique() != 1).any():
        raise ValueError("Etiquetas contradictorias de una paciente.")
    return predicciones.groupby("patient_id").agg(
        pCR=("pCR", "first"), probabilidad=("probabilidad", "mean")
    ).reset_index()


def elegir_umbral(predicciones_internas):
    """F1 macro interno; empates: umbral más próximo a 0.5, después mayor."""
    tabla = tabla_pacientes(predicciones_internas)
    if tabla.pCR.nunique() != 2:
        raise ValueError("Elegir umbral requiere ambas clases internas.")
    candidatos = np.unique(np.r_[0., .5, 1., tabla.probabilidad.values])
    return float(max(candidatos, key=lambda t: (
        f1_score(tabla.pCR, tabla.probabilidad >= t, average="macro", zero_division=0),
        -abs(t - .5), t,
    )))


def metricas(predicciones, umbral):
    tabla = tabla_pacientes(predicciones)
    resultado = uc.evaluar_por_paciente(tabla.probabilidad, tabla, umbral=umbral)
    resultado["f1_macro"] = float(f1_score(
        tabla.pCR, tabla.probabilidad >= umbral, average="macro", zero_division=0))
    resultado["por_clase"] = classification_report(
        tabla.pCR, (tabla.probabilidad >= umbral).astype(int), labels=[0, 1],
        target_names=["sin_pCR", "con_pCR"], output_dict=True, zero_division=0)
    return resultado


def predecir(modelo, filas, datos, dispositivo, batch_size, workers):
    loader = DataLoader(uc.BreastDCEDataset(filas, datos), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    modelo.eval()
    probabilidades = []
    with torch.no_grad():
        for x, _ in loader:
            probabilidades.extend(torch.sigmoid(modelo(x.to(dispositivo))).cpu().tolist())
    resultado = filas[["sample_id", "patient_id", "pCR"]].copy()
    resultado["probabilidad"] = probabilidades
    return resultado


def ejecutar_fold(args, samples, fold, arquitectura, perdida, destino):
    lr = getattr(args, "lr", 1e-4)
    train, interna, externa = particiones(samples, fold, args.prueba, args.semilla)
    random.seed(args.semilla)
    np.random.seed(args.semilla)
    torch.manual_seed(args.semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.semilla)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    modelo = MODELOS[arquitectura]()
    destino.mkdir(parents=True, exist_ok=False)
    seleccion = destino / "seleccion_interna"
    seleccion.mkdir()
    config = dict(modelo=arquitectura, perdida=perdida, fold_externo=fold,
                  fold_interno=(fold + 1) % 5, prueba=args.prueba, semilla=args.semilla,
                  epocas=args.epocas, paciencia=9, lr=lr, weight_decay=.001,
                  seleccion="auc", agregacion="mean", umbral=.5, evaluar_train=True,
                  batch_size=args.batch_size, normalizacion="PRE, EARLY, LATE / 255",
                  criterio_umbral="Máximo F1 macro en selección interna; no criterio clínico",
                  python_torch=str(torch.__version__), dispositivo=args.dispositivo)
    guardar_json(seleccion / "configuracion.json", config)
    pd.concat([g.assign(uso=uso) for g, uso in zip(
        (train, interna, externa), ("train", "seleccion_interna", "evaluacion_externa"))]
    )[["sample_id", "patient_id", "pCR", "fold", "uso"]].to_csv(destino / "particion.csv", index=False)
    inicio = time.perf_counter()
    hist, _, mejor_epoca, _ = entrenar(
        modelo, uc.BreastDCEDataset(train, args.datos, transform=AumentosDCE()),
        uc.BreastDCEDataset(interna, args.datos), interna,
        epocas=args.epocas, lr=lr, batch_size=args.batch_size,
        pos_weight=uc.pos_weight(train) if perdida == "ponderada" else None,
        paciencia=9, dispositivo=args.dispositivo, salida=seleccion, semilla=args.semilla,
        num_workers=args.num_workers, optimizador="adamw", weight_decay=.001,
        reducir_lr=True, seleccion="auc", min_delta=0.,
        train_eval_ds=uc.BreastDCEDataset(train, args.datos), train_samples=train,
        nombre=f"fold {fold} / {arquitectura} / {perdida}",
    )
    pred_interna = pd.read_csv(seleccion / "predicciones_validacion.csv")
    umbral = elegir_umbral(pred_interna)
    guardar_json(destino / "seleccion_congelada.json", {
        "mejor_epoca": mejor_epoca, "umbral": umbral,
        "criterio": config["criterio_umbral"], "fold_interno": (fold + 1) % 5,
    })
    # entrenar restaura el mejor checkpoint interno. El externo se predice SOLO aquí.
    pred_externa = predecir(modelo, externa, args.datos, args.dispositivo,
                           args.batch_size, args.num_workers)
    pred_externa.to_csv(destino / "predicciones_externas.csv", index=False)
    evaluacion = metricas(pred_externa, umbral)
    fijo = metricas(pred_externa, .5)
    baseline = metricas(pred_externa.assign(probabilidad=0.), .5)
    seleccion_metricas = json.loads((seleccion / "metricas.json").read_text())
    resultado = dict(
        modelo=arquitectura, perdida=perdida, fold=fold, prueba=args.prueba,
        mejor_epoca=mejor_epoca, epocas_completadas=len(hist["epoca"]), umbral=umbral,
        auc_interno=seleccion_metricas["auc"],
        auc_train=seleccion_metricas["entrenamiento"]["auc"],
        externo=evaluacion, externo_umbral_05=fijo, siempre_sin_pcr=baseline,
        segundos=time.perf_counter() - inicio, aviso=AVISO,
    )
    # Marcador de finalización: solo aparece después de evaluación y escritura completas.
    temporal = destino / "resultado.json.tmp"
    guardar_json(temporal, resultado)
    temporal.replace(destino / "resultado.json")
    return resultado


def resumir(resultados, salida):
    filas = []
    for r in resultados:
        filas.append(dict(modelo=r["modelo"], perdida=r["perdida"], fold=r["fold"],
                          mejor_epoca=r["mejor_epoca"], umbral=r["umbral"],
                          auc=r["externo"]["auc"], f1_macro=r["externo"]["f1_macro"],
                          sensibilidad=r["externo"]["sensibilidad"],
                          especificidad=r["externo"]["especificidad"],
                          auc_interno=r["auc_interno"], auc_train=r["auc_train"]))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(salida / "resultados_por_fold.csv", index=False)
    resumen = []
    for (modelo, perdida), g in tabla.groupby(["modelo", "perdida"]):
        item = dict(modelo=modelo, perdida=perdida, folds_completados=len(g),
                    completo=set(g.fold) == set(range(5)))
        for metrica in ("auc", "f1_macro", "sensibilidad", "especificidad"):
            item[metrica + "_media"] = float(g[metrica].mean())
            item[metrica + "_desviacion"] = float(g[metrica].std(ddof=1))
        resumen.append(item)
    guardar_json(salida / "resumen.json", dict(aviso=AVISO, prueba=any(r["prueba"] for r in resultados),
                                              modelos=resumen))
    pareadas = tabla.pivot(index=["perdida", "fold"], columns="modelo", values="auc")
    if {"v2", "v7"}.issubset(pareadas.columns):
        pareadas = pareadas.dropna(subset=["v2", "v7"])
        pareadas["diferencia_v7_menos_v2"] = pareadas.v7 - pareadas.v2
        pareadas.to_csv(salida / "diferencias_pareadas.csv")
    lineas = ["# Validación cruzada de desarrollo", "", AVISO, "",
              "PRUEBA REDUCIDA: no mide rendimiento." if any(r["prueba"] for r in resultados)
              else "AUC externa por paciente; umbral elegido exclusivamente en el fold interno.", "",
              "| Modelo | Pérdida | Folds | AUC media ± desviación |", "|---|---|---|---|"]
    for r in resumen:
        sd = f"{r['auc_desviacion']:.3f}" if r["folds_completados"] > 1 else "no estimable"
        lineas.append(f"| {r['modelo']} | {r['perdida']} | {r['folds_completados']}/5 | "
                      f"{r['auc_media']:.3f} ± {sd} |")
    lineas += ["", "No declarar ganador con folds incompletos. Una diferencia pequeña no demuestra superioridad.",
               "F1, sensibilidad y especificidad usan el umbral interno; resultado.json también conserva el umbral 0.5."]
    (salida / "COMPARACION.md").write_text("\n".join(lineas) + "\n")
    return resumen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos", type=Path, default=uc.RAIZ)
    p.add_argument("--salida", type=Path, default=uc.RAIZ / "models" / "comparacion_v2_v7")
    p.add_argument("--dispositivo", choices=["cpu", "cuda", "mps"], default="cuda")
    p.add_argument("--perdida", choices=["normal", "ponderada", "ambas"], default="ambas")
    p.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS),
                   help="Arquitecturas que se ejecutan; por defecto v2 y v7")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Tasa inicial de aprendizaje; usar una salida nueva para cada ensayo")
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--hilos", type=int, default=4)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--prueba", action="store_true", help="Dos pacientes por clase y grupo; NO mide rendimiento")
    p.add_argument("--solo-plan", action="store_true", help="Verifica metadatos sin entrenar ni crear resultados")
    p.add_argument("--reanudar", action="store_true", help="Omite trabajos terminados; reinicia los incompletos desde cero")
    args = p.parse_args()
    if min(args.epocas, args.batch_size, args.hilos) < 1 or args.num_workers < 0:
        p.error("Épocas, batch e hilos deben ser positivos; workers >= 0.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error("lr debe ser finito y positivo.")
    if len(set(args.modelos)) != len(args.modelos):
        p.error("No repitas arquitecturas en --modelos.")
    samples = uc.cargar_samples(args.datos)
    desarrollo = validar_samples(samples)
    perdidas = ["normal", "ponderada"] if args.perdida == "ambas" else [args.perdida]
    protocolo = dict(version=1, modelos=args.modelos, perdidas=perdidas, epocas=args.epocas,
                     paciencia=9, lr=args.lr, weight_decay=.001, semilla=args.semilla,
                     batch_size=args.batch_size, num_workers=args.num_workers, hilos=args.hilos,
                     dispositivo=args.dispositivo, prueba=args.prueba, datos=str(args.datos.resolve()),
                     metadata_sha256=hashlib.sha256(samples.to_csv(index=False).encode()).hexdigest(),
                     seleccion="AUC interna; umbral interno por F1 macro; externo una vez", aviso=AVISO)
    fuentes = [Path(__file__), Path(__file__).with_name("entrenar.py"),
               uc.RAIZ / "src/models/modelo.py", uc.RAIZ / "src/data/utils_caso.py",
               uc.RAIZ / "src/data/aumentos.py"]
    protocolo["codigo_sha256"] = hashlib.sha256(b"".join(f.read_bytes() for f in fuentes)).hexdigest()
    for fold in range(5):
        grupos = particiones(samples, fold, args.prueba, args.semilla)
        print(f"Fold {fold}: train/interno/externo = "
              f"{[g.patient_id.nunique() for g in grupos]} pacientes", flush=True)
    print(f"Modelos: {', '.join(args.modelos)} | pérdidas: {', '.join(perdidas)} | lr inicial: {args.lr:g}", flush=True)
    print(f"{5 * len(args.modelos) * len(perdidas)} entrenamientos; máximo {args.epocas} épocas cada uno; "
          f"paciencia 9. Test oficial excluido ({desarrollo.patient_id.nunique()} pacientes de desarrollo).", flush=True)
    if args.solo_plan:
        return
    if args.dispositivo == "cuda" and not torch.cuda.is_available():
        p.error("CUDA/ROCm no disponible. Verifica el entorno; no se cambia a CPU automáticamente.")
    for columna in ("path_pre", "path_early", "path_late"):
        for ruta in desarrollo[columna]:
            if not (args.datos / ruta).is_file():
                p.error(f"Falta imagen de desarrollo: {ruta}")
    manifiesto = args.salida / "protocolo.json"
    if args.reanudar:
        if not manifiesto.exists() or json.loads(manifiesto.read_text()) != protocolo:
            p.error("No coincide el protocolo/dataset/código. Usa otra carpeta para una comparación distinta.")
    elif args.salida.exists():
        p.error("La carpeta ya existe. Usa --reanudar o una --salida nueva.")
    else:
        args.salida.mkdir(parents=True)
        guardar_json(manifiesto, protocolo)
    torch.set_num_threads(args.hilos)
    resultados = []
    try:
        for fold in range(5):
            for perdida in perdidas:
                for arquitectura in args.modelos:
                    carpeta = args.salida / f"fold_{fold}" / arquitectura / perdida
                    marcador = carpeta / "resultado.json"
                    if marcador.exists():
                        resultado = json.loads(marcador.read_text())
                        print(f"Ya completado: {carpeta}", flush=True)
                    else:
                        if carpeta.exists():
                            carpeta.rename(carpeta.with_name(perdida + "_incompleto_" +
                                           datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
                        resultado = ejecutar_fold(args, samples, fold, arquitectura, perdida, carpeta)
                    resultados.append(resultado)
                    resumir(resultados, args.salida)
    except KeyboardInterrupt:
        print("Interrumpido. --reanudar conserva los trabajos completos y reinicia el incompleto.", flush=True)
        raise SystemExit(130)
    print(f"Comparación terminada: {args.salida / 'COMPARACION.md'}", flush=True)


if __name__ == "__main__":
    main()
