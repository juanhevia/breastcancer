"""Auditoría y prueba de memorización; nunca evalúa imágenes de test.

python -m src.training.diagnostico --memorizacion --modelo v2
La AUC de memorización se mide sobre las mismas imágenes entrenadas: NO es
una estimación de generalización ni un resultado para comparar con AUC 0,70.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.data import utils_caso as uc
from src.models.modelo import BreastCNNV2, BreastCNNV5
from src.training.entrenar import guardar_json


def auditar_metadatos(samples, patients, fold=0):
    errores = []
    if not samples.pCR.isin([0, 1]).all() or not patients.pCR.isin([0, 1]).all():
        errores.append("Hay etiquetas pCR distintas de 0/1.")
    if samples.sample_id.isna().any() or samples.sample_id.duplicated().any():
        errores.append("Hay sample_id ausentes o duplicados.")
    if patients.pid.isna().any() or patients.pid.duplicated().any():
        errores.append("Hay pid ausentes o duplicados en patients.csv.")
    if samples[["patient_id", "pCR", "split", "fold"]].isna().any().any():
        errores.append("Faltan metadatos de muestras.")
    if patients[["pCR", "dataset", "split"]].isna().any().any():
        errores.append("Faltan metadatos de pacientes.")
    for columna in ("pCR", "split", "fold"):
        if (samples.groupby("patient_id")[columna].nunique() != 1).any():
            errores.append(f"Hay pacientes con valores contradictorios de {columna}.")
    if not samples.split.isin(["train", "test"]).all():
        errores.append("Hay splits desconocidos.")
    if not samples.loc[samples.split == "train", "fold"].isin(range(5)).all():
        errores.append("Hay folds de train fuera de 0..4.")
    if not (samples.loc[samples.split == "test", "fold"] == -1).all():
        errores.append("El fold de test debe ser -1.")
    faltan = set(samples.patient_id) - set(patients.pid)
    if faltan:
        errores.append(f"Faltan {len(faltan)} pacientes en patients.csv.")
    if errores:
        return {"errores": errores}
    unido = samples.merge(patients[["pid", "pCR", "dataset", "split"]],
                          left_on="patient_id", right_on="pid", suffixes=("", "_paciente"),
                          validate="many_to_one")
    for columna in ("pCR", "split"):
        if (unido[columna] != unido[columna + "_paciente"]).any():
            errores.append(f"{columna} no coincide entre los dos CSV.")
    tr, va = uc.particion(samples, fold)
    conteos = []
    for uso, filas in (("train", tr), ("validacion", va)):
        tabla = filas[["patient_id", "pCR"]].drop_duplicates().merge(
            patients[["pid", "dataset"]], left_on="patient_id", right_on="pid", validate="one_to_one")
        for cohorte, g in tabla.groupby("dataset"):
            conteos.append(dict(uso=uso, cohorte=cohorte, pacientes=len(g),
                                pcr=int(g.pCR.sum()), prevalencia=float(g.pCR.mean())))
    return dict(errores=errores, pacientes=samples.patient_id.nunique(), cortes=len(samples),
                pacientes_compartidas=len(set(tr.patient_id) & set(va.patient_id)), cohortes=conteos,
                limites="Audita metadatos. No detecta duplicados de píxeles ni errores clínicos de las etiquetas.")


def resumir_predicciones(pred, patients, esperadas):
    """Requiere exactamente la validación declarada, con sus etiquetas y cortes."""
    claves = ["sample_id", "patient_id", "pCR"]
    a = pred[claves].sort_values("sample_id").reset_index(drop=True)
    b = esperadas[claves].sort_values("sample_id").reset_index(drop=True)
    if not a.equals(b):
        raise ValueError("Las predicciones no coinciden con todos los cortes y etiquetas de validación.")
    if not np.isfinite(pred.probabilidad).all() or not pred.probabilidad.between(0, 1).all():
        raise ValueError("Probabilidades inválidas.")
    tabla = pred.groupby("patient_id").agg(pCR=("pCR", "first"), prob=("probabilidad", "mean"))
    tabla = tabla.reset_index().merge(patients[["pid", "dataset"]],
                                     left_on="patient_id", right_on="pid", how="left", validate="one_to_one")
    if tabla.dataset.isna().any():
        raise ValueError("Falta la cohorte de alguna paciente de validación.")
    resumen = []
    for cohorte, g in [("todas", tabla), *list(tabla.groupby("dataset"))]:
        positivo = g.pCR == 1
        resumen.append(dict(cohorte=cohorte, pacientes=len(g), pcr=int(positivo.sum()),
                            auc=float(roc_auc_score(g.pCR, g.prob)) if g.pCR.nunique() == 2 else None,
                            prob_media=float(g.prob.mean()),
                            prob_media_pcr=float(g.loc[positivo, "prob"].mean()),
                            prob_media_no_pcr=float(g.loc[~positivo, "prob"].mean()),
                            vp_umbral_05=int((positivo & (g.prob >= .5)).sum()),
                            fp_umbral_05=int((~positivo & (g.prob >= .5)).sum())))
    return resumen


def seleccionar_memorizacion(samples, fold=0, pacientes=8, semilla=42):
    if pacientes < 2 or pacientes % 2:
        raise ValueError("El número de pacientes debe ser par y >= 2.")
    tr, _ = uc.particion(samples, fold)
    ids = tr[["patient_id", "pCR"]].drop_duplicates()
    if len(ids[ids.pCR == 0]) < pacientes // 2 or len(ids[ids.pCR == 1]) < pacientes // 2:
        raise ValueError("No hay suficientes pacientes de cada clase en train.")
    elegidas = ids.groupby("pCR", group_keys=False).sample(n=pacientes // 2, random_state=semilla)
    filas = []
    for pid in elegidas.patient_id:
        cortes = tr[tr.patient_id == pid].sort_values("slice_index")
        filas.append(cortes.iloc[len(cortes)//2])
    return pd.DataFrame(filas).reset_index(drop=True)


def probar_memorizacion(samples, raiz, salida, modelo="v2", fold=0, pacientes=8,
                       semilla=42, pasos=100, max_segundos=240, lr=1e-3):
    if pasos < 1 or max_segundos <= 0 or not np.isfinite(max_segundos) or lr <= 0 or not np.isfinite(lr):
        raise ValueError("Pasos, tiempo máximo y lr deben ser positivos y finitos.")
    filas = seleccionar_memorizacion(samples, fold, pacientes, semilla)
    filas.to_csv(salida / "muestras_memorizacion.csv", index=False)
    torch.manual_seed(semilla)
    red = {"v2": BreastCNNV2, "v5": BreastCNNV5}[modelo]()
    # Retirar regularización SOLO en esta prueba para detectar si puede ajustar.
    for capa in red.modules():
        if isinstance(capa, (torch.nn.Dropout, torch.nn.Dropout2d)):
            capa.p = 0.0
    ds = uc.BreastDCEDataset(filas, raiz)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor(filas.pCR.to_numpy(), dtype=torch.float32)
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Imágenes fuera de rango o no finitas.")
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    criterio = torch.nn.BCEWithLogitsLoss()
    historial, estables = [], 0
    inicio = time.perf_counter()
    for paso in range(pasos + 1):
        if paso:
            red.train()
            opt.zero_grad(set_to_none=True)
            loss = criterio(red(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Pérdida de memorización no finita.")
            loss.backward()
            opt.step()
        tiempo_agotado = time.perf_counter() - inicio >= max_segundos
        if paso % 10 == 0 or paso == pasos or tiempo_agotado:
            red.eval()
            with torch.no_grad():
                z = red(x)
                prob = z.sigmoid()
                perdida = float(criterio(z, y))
                acierto = float(((prob >= .5) == y.bool()).float().mean())
            registro = dict(paso=paso, loss=perdida, accuracy=acierto,
                            auc_sobre_train=float(roc_auc_score(y.numpy(), prob.numpy())),
                            segundos=time.perf_counter() - inicio)
            historial.append(registro)
            pd.DataFrame(historial).to_csv(salida / "memorizacion.csv", index=False)
            print(f"Memorización, paso {paso}: loss={perdida:.4f}, aciertos={acierto:.0%} "
                  "(MISMAS imágenes de train; no es validación)", flush=True)
            estables = estables + 1 if acierto == 1 and perdida < .05 else 0
            if estables >= 2 or tiempo_agotado:
                break
    resultado = dict(modelo=modelo, pacientes=pacientes, cortes=pacientes, fold_excluido=fold,
                     semilla=semilla, lr=lr, dropout=0, weight_decay=0, aumentos=False,
                     pasos_maximos=pasos, max_segundos=max_segundos, ultima_medicion=historial[-1],
                     ajuste_logrado=estables >= 2,
                     significado="Ajustar estas mismas imágenes comprueba capacidad de optimización. NO mide generalización ni valida las etiquetas clínicas.")
    guardar_json(salida / "memorizacion.json", resultado)
    return resultado


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos", type=Path, default=uc.RAIZ)
    p.add_argument("--salida", type=Path, default=uc.RAIZ / "models")
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--memorizacion", action="store_true")
    p.add_argument("--modelo", choices=("v2", "v5"), default="v2")
    p.add_argument("--pacientes", type=int, default=8)
    p.add_argument("--pasos", type=int, default=100)
    p.add_argument("--max-segundos", type=float, default=240)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()
    torch.set_num_threads(4)
    carpeta = args.salida / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_diagnostico")
    carpeta.mkdir(parents=True, exist_ok=False)
    samples = uc.cargar_samples(args.datos)
    patients = uc.cargar_patients(args.datos)
    auditoria = auditar_metadatos(samples, patients, args.fold)
    guardar_json(carpeta / "auditoria.json", auditoria)
    if auditoria["errores"]:
        raise SystemExit("Corregir metadatos antes de entrenar: " + "; ".join(auditoria["errores"]))
    print("Metadatos sin contradicciones en las comprobaciones realizadas.", flush=True)
    import json
    comparaciones, omitidos = {}, []
    for ruta in sorted((args.datos / "models").glob("*/*/predicciones_validacion.csv")):
        config = ruta.parent / "configuracion.json"
        if not config.exists():
            continue
        c = json.loads(config.read_text())
        if c.get("prueba", False):
            continue
        _, va = uc.particion(samples, c.get("fold", 0))
        try:
            comparaciones[str(ruta.parent.relative_to(args.datos))] = resumir_predicciones(
                pd.read_csv(ruta), patients, va)
        except ValueError as error:
            omitidos.append(dict(ruta=str(ruta), motivo=str(error)))
    guardar_json(carpeta / "cohortes.json", dict(modelos=comparaciones, omitidos=omitidos))
    if args.memorizacion:
        probar_memorizacion(samples, args.datos, carpeta, args.modelo, args.fold,
                            args.pacientes, args.semilla, args.pasos, args.max_segundos, args.lr)
    print(f"Diagnóstico guardado en {carpeta}", flush=True)


if __name__ == "__main__":
    main()
