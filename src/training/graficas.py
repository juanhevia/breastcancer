"""Gráficas de validación: python -m src.training.graficas models/<ejecucion>."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)

from src.data import utils_caso as uc


METRICAS = {
    "val_accuracy": "Accuracy", "val_sensibilidad": "Sensibilidad",
    "val_especificidad": "Especificidad", "val_precision": "Precisión",
    "val_f1": "F1",
}


def guardar(fig, ruta):
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)


def configurar_epocas(eje, epocas):
    valores = np.unique(epocas)
    if len(valores) == 1:
        eje.set_xlim(valores[0] - .5, valores[0] + .5)
    if len(valores) <= 20:
        eje.set_xticks(valores)
    else:
        eje.xaxis.set_major_locator(MaxNLocator(integer=True))


def dibujar_curvas(historial, ruta, contexto="", mejor_epoca=None):
    """Representa el historial completo; la mejor época es solo una referencia."""
    ruta = Path(ruta)
    numero_epocas = len(historial["epoca"])
    titulo = ("Historial completo · 1 época registrada (sin evolución temporal)"
              if numero_epocas == 1 else f"Historial completo · {numero_epocas} épocas registradas")
    if contexto:
        titulo = contexto + "\n" + titulo

    def terminar_eje(eje):
        configurar_epocas(eje, historial["epoca"])
        if mejor_epoca is not None:
            eje.axvline(mejor_epoca, color="gray", linestyle="--", alpha=.7,
                        label=f"Mejor época = {mejor_epoca}")
        eje.legend(loc="best")
        eje.grid(alpha=.2)

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ejes[0].plot(historial["epoca"], historial["train_loss"], label="Entrenamiento", marker="o")
    ejes[0].plot(historial["epoca"], historial["val_loss"], label="Validación interna", marker="o")
    ejes[0].set(ylabel="BCE", xlabel="Época", title="Pérdida")
    ejes[1].plot(historial["epoca"], historial["val_auc_paciente"], label="Por paciente", marker="o")
    ejes[1].plot(historial["epoca"], historial["val_auc"], label="Por corte", marker="o", alpha=.6)
    ejes[1].set(ylabel="ROC-AUC", xlabel="Época", ylim=(-.02, 1.02), title="Validación interna")
    for eje in ejes:
        terminar_eje(eje)
    fig.suptitle(titulo)
    guardar(fig, ruta)
    presentes = [k for k in METRICAS if k in historial]
    if presentes:
        fig, eje = plt.subplots(figsize=(7, 4), layout="constrained")
        for k in presentes:
            eje.plot(historial["epoca"], historial[k], marker="o", label=METRICAS[k])
        eje.set(xlabel="Época", ylabel="Valor", ylim=(-.02, 1.02),
                title="Métricas por paciente · validación interna")
        terminar_eje(eje)
        fig.suptitle(titulo)
        guardar(fig, ruta.with_name("metricas_epocas.png"))


def preparar_pacientes(predicciones, metodo="mean", umbral=.5):
    """Agrega primero los cortes: cada paciente cuenta una sola vez."""
    requeridas = {"patient_id", "pCR", "probabilidad"}
    if not requeridas.issubset(predicciones.columns) or predicciones.empty:
        raise ValueError("Se necesitan predicciones con patient_id, pCR y probabilidad.")
    if predicciones[list(requeridas)].isna().any().any():
        raise ValueError("Las predicciones contienen valores ausentes.")
    if not predicciones.pCR.isin([0, 1]).all():
        raise ValueError("pCR debe contener etiquetas binarias 0/1.")
    if not np.isfinite(predicciones.probabilidad).all() or not predicciones.probabilidad.between(0, 1).all():
        raise ValueError("Las probabilidades deben ser finitas y estar entre 0 y 1.")
    if not 0 <= umbral <= 1:
        raise ValueError("El umbral debe estar entre 0 y 1.")
    grupos = predicciones.groupby("patient_id")
    if (grupos.pCR.nunique() != 1).any():
        raise ValueError("Hay etiquetas contradictorias dentro de una paciente.")
    tabla = uc.agregar_por_paciente(predicciones.patient_id, predicciones.probabilidad, metodo, umbral)
    tabla = tabla.join(grupos.pCR.first()).join(grupos.size().rename("cortes"))
    tabla["prediccion"] = (tabla.prob >= umbral).astype(int)
    return tabla


def datos_curvas(tabla):
    """ROC y PR requieren ambas clases; AP no es el área trapezoidal de PR."""
    y, prob = tabla.pCR.to_numpy(), tabla.prob.to_numpy()
    if len(np.unique(y)) != 2:
        return None
    fpr, tpr, _ = roc_curve(y, prob)
    precision, recall, _ = precision_recall_curve(y, prob)
    return dict(fpr=fpr, tpr=tpr, precision=precision, recall=recall,
                auc=float(roc_auc_score(y, prob)), ap=float(average_precision_score(y, prob)))


def dibujar_evaluacion(predicciones, salida, metodo="mean", umbral=.5, contexto=""):
    """Gráficas del mejor modelo, usando solo validación interna."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    tabla = preparar_pacientes(predicciones, metodo, umbral)
    tabla.to_csv(salida / "predicciones_pacientes.csv")
    y, prob = tabla.pCR.to_numpy(dtype=int), tabla.prob.to_numpy()
    datos = datos_curvas(tabla)
    titulo = f"Validación interna · {len(tabla)} pacientes · {metodo}"
    if contexto:
        titulo += f"\n{contexto}"

    for tipo in ("roc", "precision_recall"):
        fig, eje = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
        if tipo == "roc":
            eje.plot([0, 1], [0, 1], "--", color="gray", label="Referencia aleatoria")
            if datos:
                eje.plot(datos["fpr"], datos["tpr"], label=f"ROC-AUC = {datos['auc']:.3f}")
            eje.set(xlabel="Tasa de falsos positivos (1 − especificidad)", ylabel="Sensibilidad",
                    title="Curva ROC\n" + titulo)
        else:
            eje.axhline(y.mean(), ls="--", color="gray", label=f"Prevalencia pCR = {y.mean():.3f}")
            if datos:
                eje.step(datos["recall"], datos["precision"], where="post", label=f"AP = {datos['ap']:.3f}")
            eje.set(xlabel="Recall (sensibilidad)", ylabel="Precisión (valor predictivo positivo)",
                    title="Precisión–recall\n" + titulo)
        if datos is None:
            eje.text(.5, .45, "Curva no estimable:\nsolo hay una clase", ha="center", transform=eje.transAxes)
        eje.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        eje.grid(alpha=.2)
        eje.legend(loc="best")
        guardar(fig, salida / f"{tipo}.png")

    observada, predicha = calibration_curve(y, prob, n_bins=10, strategy="uniform")
    brier = float(brier_score_loss(y, prob))
    fig, eje = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    eje.plot([0, 1], [0, 1], "--", color="gray", label="Calibración ideal")
    eje.plot(predicha, observada, "o-", label=f"Brier = {brier:.3f}")
    eje.set(xlabel="Puntuación predicha media (10 intervalos)", ylabel="Fracción observada de pCR",
            title="Calibración (sin recalibrar el modelo)\n" + titulo,
            xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    eje.legend()
    eje.grid(alpha=.2)
    guardar(fig, salida / "calibracion.png")

    matriz = confusion_matrix(y, tabla.prediccion, labels=[0, 1])
    fig, eje = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    im = eje.imshow(matriz, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            eje.text(j, i, str(matriz[i, j]), ha="center", va="center", fontsize=18,
                     color="white" if matriz[i, j] > matriz.max()/2 else "black")
    eje.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No pCR", "pCR"],
            yticklabels=["No pCR", "pCR"], xlabel="Predicción", ylabel="Etiqueta real",
            title=f"Matriz de confusión · umbral {umbral:g}\n" + titulo)
    fig.colorbar(im, ax=eje, label="Pacientes")
    guardar(fig, salida / "matriz_confusion.png")

    filas = []
    for t in np.unique(np.r_[np.linspace(0, 1, 101), prob, umbral]):
        pred = prob >= t
        vp, vn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 0))
        fp, fn = np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
        def div(a, b):
            return float(a / b) if b else float("nan")
        filas.append(dict(umbral=t, sensibilidad=div(vp, vp+fn), especificidad=div(vn, vn+fp),
                          precision=div(vp, vp+fp), f1=div(2*vp, 2*vp+fp+fn), accuracy=div(vp+vn, len(y))))
    umbrales = pd.DataFrame(filas)
    umbrales.to_csv(salida / "metricas_umbrales.csv", index=False)
    fig, eje = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    for k, label in (("sensibilidad", "Sensibilidad"), ("especificidad", "Especificidad"),
                     ("precision", "Precisión"), ("f1", "F1"), ("accuracy", "Accuracy")):
        eje.plot(umbrales.umbral, umbrales[k], label=label)
    eje.axvline(umbral, ls="--", color="black", label=f"Umbral actual = {umbral:g}")
    eje.set(xlabel="Umbral aplicado a la puntuación agregada", ylabel="Valor",
            title="Métricas según el umbral\n" + titulo, ylim=(-.02, 1.02))
    eje.legend(loc="best")
    eje.grid(alpha=.2)
    guardar(fig, salida / "umbrales.png")

    fig, eje = plt.subplots(figsize=(7, 5), layout="constrained")
    for clase in (0, 1):
        eje.hist(prob[y == clase], bins=np.linspace(0, 1, 11), alpha=.55,
                 label=f"pCR={clase} (n={np.sum(y == clase)})")
    eje.axvline(umbral, color="black", ls="--", label=f"Umbral = {umbral:g}")
    eje.set(xlabel="Puntuación agregada", ylabel="Número de pacientes", title="Distribución de predicciones\n" + titulo)
    eje.yaxis.set_major_locator(MaxNLocator(integer=True))
    eje.legend()
    guardar(fig, salida / "distribucion_predicciones.png")
    return tabla


def generar_graficas(carpeta):
    """Reutiliza los CSV guardados; no carga imágenes, pesos ni ejecuta entrenamiento."""
    carpeta = Path(carpeta)
    config = json.loads((carpeta / "configuracion.json").read_text())
    metricas = json.loads((carpeta / "metricas.json").read_text())
    contexto = config.get('perdida', carpeta.name)
    if config.get("prueba"):
        contexto += " · PRUEBA REDUCIDA"
    historial = pd.read_csv(carpeta / "historial.csv")
    dibujar_curvas(historial, carpeta / "curvas.png", contexto=contexto,
                   mejor_epoca=metricas["mejor_epoca"])
    return dibujar_evaluacion(pd.read_csv(carpeta / "predicciones_validacion.csv"), carpeta,
                             metodo=metricas["metodo"], umbral=metricas["umbral"],
                             contexto=f"{contexto} · mejor época {metricas['mejor_epoca']}")


def comparar_perdidas(carpeta):
    """Superpone ROC y PR solo si ambas pérdidas validaron las mismas pacientes."""
    carpeta = Path(carpeta)
    tablas, curvas = {}, {}
    prueba = False
    for nombre in ("normal", "ponderada"):
        origen = carpeta / nombre
        metricas = json.loads((origen / "metricas.json").read_text())
        config = json.loads((origen / "configuracion.json").read_text())
        prueba = prueba or config.get("prueba", False)
        tablas[nombre] = preparar_pacientes(
            pd.read_csv(origen / "predicciones_validacion.csv"),
            metodo=metricas["metodo"], umbral=metricas["umbral"],
        )
        curvas[nombre] = datos_curvas(tablas[nombre])
    if not tablas["normal"].pCR.equals(tablas["ponderada"].pCR):
        raise ValueError("No se pueden comparar pérdidas con pacientes o etiquetas diferentes.")
    fig, ejes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    ejes[0].plot([0, 1], [0, 1], "--", color="gray", label="Referencia aleatoria")
    prevalencia = tablas["normal"].pCR.mean()
    ejes[1].axhline(prevalencia, ls="--", color="gray", label=f"Prevalencia = {prevalencia:.3f}")
    for nombre, datos in curvas.items():
        if datos is not None:
            ejes[0].plot(datos["fpr"], datos["tpr"], label=f"{nombre}: AUC = {datos['auc']:.3f}")
            ejes[1].step(datos["recall"], datos["precision"], where="post",
                         label=f"{nombre}: AP = {datos['ap']:.3f}")
    ejes[0].set(xlabel="Tasa de falsos positivos", ylabel="Sensibilidad", title="Curva ROC")
    ejes[1].set(xlabel="Recall (sensibilidad)", ylabel="Precisión", title="Precisión–recall")
    for eje in ejes:
        if curvas["normal"] is None:
            eje.text(.5, .45, "Curva no estimable:\nsolo hay una clase", ha="center", transform=eje.transAxes)
        eje.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        eje.grid(alpha=.2)
        eje.legend(loc="best")
    titulo = f"Mejores modelos · validación interna · {len(tablas['normal'])} pacientes"
    if prueba:
        titulo += "\nPRUEBA REDUCIDA"
    fig.suptitle(titulo)
    guardar(fig, carpeta / "comparacion_roc_pr.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("carpeta", type=Path, help="Carpeta de una ejecución o su subcarpeta normal/ponderada")
    args = p.parse_args()
    carpetas = ([args.carpeta] if (args.carpeta / "predicciones_validacion.csv").is_file()
                else [c for c in (args.carpeta / "normal", args.carpeta / "ponderada")
                      if (c / "predicciones_validacion.csv").is_file()])
    if not carpetas:
        p.error("No se encontraron predicciones de validación en esa carpeta.")
    for carpeta in carpetas:
        generar_graficas(carpeta)
        print(f"Gráficas guardadas en {carpeta}")
    if len(carpetas) == 2:
        comparar_perdidas(args.carpeta)
        print(f"Comparación guardada en {args.carpeta / 'comparacion_roc_pr.png'}")


if __name__ == "__main__":
    main()
