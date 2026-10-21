"""Figuras individuales para presentar el entrenamiento y su validación interna.

Uso: python -m src.training.graficas models/<ejecucion> [--detalle]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)

from src.data import utils_caso as uc


COMPUESTAS_ANTIGUAS = (
    "01_evolucion_del_aprendizaje.png",
    "02_capacidad_para_distinguir_pcr.png",
    "03_aciertos_y_errores.png",
)
PRINCIPALES = (
    "01_perdida.png", "02_auc_por_epoca.png", "03_roc.png",
    "04_precision_sensibilidad.png", "05_matriz_confusion.png",
)
NOMBRES = {"normal": "BCE normal", "ponderada": "BCE ponderada"}
METRICAS = {
    "val_accuracy": "Exactitud (aciertos totales)", "val_sensibilidad": "Sensibilidad",
    "val_especificidad": "Especificidad", "val_precision": "Precisión (VPP)", "val_f1": "F1",
}
DEFINICION = "pCR = respuesta patológica completa; sin pCR = ausencia de respuesta completa."


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


def pie(fig, texto):
    fig.text(.5, .015, texto, ha="center", va="bottom", fontsize=9)


def titulo_general(registros):
    prueba = any(r["config"].get("prueba", False) for r in registros)
    return "PRUEBA REDUCIDA · no estima el rendimiento real" if prueba else "Validación interna · test reservado"


def criterio_seleccion(registro):
    criterio = registro["metricas"].get("seleccion", registro["config"].get("seleccion", "loss"))
    return "mayor AUC por paciente" if criterio == "auc" else "menor pérdida de validación"


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


def calcular_umbrales(tabla, umbral):
    y, prob = tabla.pCR.to_numpy(), tabla.prob.to_numpy()
    filas = []
    for t in np.unique(np.r_[np.linspace(0, 1, 101), prob, umbral]):
        pred = prob >= t
        vp, vn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 0))
        fp, fn = np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))

        def div(a, b):
            return float(a / b) if b else float("nan")

        filas.append(dict(umbral=t, sensibilidad=div(vp, vp+fn), especificidad=div(vn, vn+fp),
                          precision=div(vp, vp+fp), f1=div(2*vp, 2*vp+fp+fn), accuracy=div(vp+vn, len(y))))
    return pd.DataFrame(filas)


def dibujar_evolucion(registros, ruta):
    n = len(registros)
    fig, ejes = plt.subplots(n, 2, figsize=(13, 3.5*n+2), squeeze=False)
    fig.subplots_adjust(top=.81 if n == 1 else .86, bottom=.23 if n == 1 else .16,
                        hspace=.7, wspace=.3)
    for r, (perdida, auc) in zip(registros, ejes):
        hist = r["historial"]
        epocas = hist["epoca"]
        total = len(epocas)
        nota = f"{total} épocas registradas" if total != 1 else "1 época: todavía no hay evolución"
        comparable = "train_eval_loss" in hist
        perdida.plot(epocas, hist["train_eval_loss"] if comparable else hist["train_loss"],
                     label="Entrenamiento (evaluación)" if comparable else "Entrenamiento", marker="o", markersize=3)
        perdida.plot(epocas, hist["val_loss"], label="Validación interna", marker="o", markersize=3)
        unidad = r["config"].get("unidad_perdidas_historial", "corte")
        perdida.set(title=f"{r['nombre']} · pérdida (menor es mejor)", xlabel="Época",
                    ylabel=f"Pérdida BCE por {unidad}")
        if "train_auc_paciente" in hist:
            auc.plot(epocas, hist["train_auc_paciente"], marker="o", markersize=3,
                     label="AUC en entrenamiento por paciente", color="tab:blue")
        auc.plot(epocas, hist["val_auc_paciente"], marker="o", markersize=3,
                 label="AUC en validación por paciente", color="tab:green")
        auc.axhline(.5, color="gray", ls=":", label="Referencia aleatoria = 0,50")
        auc.set(title=f"{r['nombre']} · AUC (mayor es mejor)", xlabel="Época",
                ylabel="ROC-AUC por paciente", ylim=(-.02, 1.02))
        for eje in (perdida, auc):
            configurar_epocas(eje, epocas)
            if r["metricas"].get("mejor_epoca") is not None:
                eje.axvline(r["metricas"]["mejor_epoca"], color="gray", ls="--",
                            label=f"Época elegida = {r['metricas']['mejor_epoca']}")
            eje.legend(fontsize=8)
            eje.grid(alpha=.2)
        perdida.text(0, 1.12, nota, transform=perdida.transAxes, fontsize=9)
        auc.text(0, 1.12, f"Selección: {criterio_seleccion(r)}", transform=auc.transAxes, fontsize=9)
    fig.suptitle("1. ¿Cómo aprende el modelo a lo largo de las épocas?\n" + titulo_general(registros), fontsize=14)
    nota_train = ("Train y validación se evalúan sin aumentos ni dropout; la pérdida durante la optimización se conserva en el CSV."
                  if all("train_eval_loss" in r["historial"] for r in registros)
                  else "Si la leyenda dice «evaluación», train se mide sin aumentos ni dropout; en historiales antiguos es la pérdida de optimización.")
    pie(fig, "Izquierda: error de entrenamiento y validación (unidad indicada en el eje). Derecha: capacidad de ordenar las pacientes según pCR.\n"
        "Se muestran TODAS las épocas; la línea vertical solo señala los pesos elegidos. Las BCE normal y ponderada tienen escalas distintas.\n"
        + nota_train + "\n" + DEFINICION)
    guardar(fig, ruta)


def dibujar_curvas(historial, ruta, contexto="", mejor_epoca=None):
    """Compatibilidad: una figura de evolución, sin generar copias por métrica."""
    dibujar_evolucion([dict(historial=historial, nombre=contexto or "Modelo", config={},
                            metricas={"mejor_epoca": mejor_epoca})], ruta)


def dibujar_discriminacion(registros, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(top=.77, bottom=.26, wspace=.3)
    ejes[0].plot([0, 1], [0, 1], "--", color="gray", label="Referencia aleatoria (AUC = 0,50)")
    prevalencia = registros[0]["tabla"].pCR.mean()
    ejes[1].axhline(prevalencia, ls="--", color="gray", label=f"Proporción real con pCR = {prevalencia:.1%}")
    for indice, r in enumerate(registros):
        datos = datos_curvas(r["tabla"])
        if datos is None:
            continue
        etiqueta = f"{r['nombre']} · época {r['metricas']['mejor_epoca']}"
        estilo = "-" if indice == 0 else "--"
        ejes[0].plot(datos["fpr"], datos["tpr"], ls=estilo, label=f"{etiqueta} · AUC={datos['auc']:.3f}")
        ejes[1].step(datos["recall"], datos["precision"], where="post",
                     ls=estilo, label=f"{etiqueta} · AP={datos['ap']:.3f}")
    ejes[0].set(title="ROC: detección frente a falsas alarmas",
                xlabel="Falsas alarmas entre las pacientes SIN pCR\n(1 − especificidad)",
                ylabel="Pacientes CON pCR detectadas\n(sensibilidad)")
    ejes[1].set(title="Precisión–sensibilidad: fiabilidad de los positivos",
                xlabel="Pacientes CON pCR detectadas (sensibilidad)",
                ylabel="Predicciones de pCR que son correctas\n(precisión o VPP)")
    for eje in ejes:
        if datos_curvas(registros[0]["tabla"]) is None:
            eje.text(.5, .4, "No se puede estimar esta curva:\nfalta una de las dos clases", ha="center", transform=eje.transAxes)
        eje.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        eje.xaxis.set_major_formatter(PercentFormatter(1))
        eje.yaxis.set_major_formatter(PercentFormatter(1))
        eje.legend(fontsize=8, loc="best")
        eje.grid(alpha=.2)
    n = len(registros[0]["tabla"])
    fig.suptitle(f"2. ¿Distingue el modelo entre pacientes con y sin pCR?\n{titulo_general(registros)} · {n} pacientes", fontsize=14)
    pie(fig, "Cada curva usa el modelo de su época elegida y recorre distintos umbrales; NO representa el paso de las épocas.\n"
        "ROC: mejor hacia arriba e izquierda. Precisión–sensibilidad: mejor hacia arriba y derecha.\n"
        "AUC resume la ROC; AP resume precisión–sensibilidad (mayor es mejor en ambas). Curvas iguales quedan superpuestas.\n" + DEFINICION)
    guardar(fig, ruta)


def dibujar_aciertos(registros, ruta):
    fig, ejes = plt.subplots(1, len(registros), figsize=(6.5*len(registros), 6.5), squeeze=False)
    fig.subplots_adjust(top=.75, bottom=.23, wspace=.35)
    for r, eje in zip(registros, ejes[0]):
        tabla = r["tabla"]
        matriz = confusion_matrix(tabla.pCR, tabla.prediccion, labels=[0, 1])
        totales = matriz.sum(axis=1, keepdims=True)
        porcentajes = np.divide(matriz, totales, out=np.full((2, 2), np.nan), where=totales != 0)
        eje.imshow(np.ma.masked_invalid(porcentajes), cmap="Blues", vmin=0, vmax=1)
        nombres = [["Acierto: sin pCR", "Error: falsa pCR"], ["Error: pCR no detectada", "Acierto: con pCR"]]
        for i in range(2):
            for j in range(2):
                porcentaje = (f"{porcentajes[i, j]:.1%} de esta fila" if totales[i, 0] else "Sin pacientes de esta clase")
                eje.text(j, i, f"{nombres[i][j]}\n{matriz[i, j]} pacientes\n{porcentaje}",
                         ha="center", va="center", fontsize=10,
                         color="white" if porcentajes[i, j] > .5 else "black")
        eje.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Sin pCR", "Con pCR"],
                yticklabels=[f"Sin pCR (n={totales[0, 0]})", f"Con pCR (n={totales[1, 0]})"],
                xlabel="Lo que predice el modelo", ylabel="Resultado real",
                title=f"{r['nombre']} · época {r['metricas']['mejor_epoca']}\nUmbral de decisión = {r['metricas']['umbral']:g}")
    fig.suptitle("3. ¿A cuántas pacientes clasifica bien y en qué se equivoca?\n" + titulo_general(registros), fontsize=14)
    pie(fig, "Se predice pCR cuando la puntuación por paciente es mayor o igual al umbral indicado.\n"
        "Cada fila suma el 100 % de su clase real; los números son pacientes, no cortes.\n"
        "Una falsa pCR es predecir respuesta completa cuando no la hubo.\n" + DEFINICION)
    guardar(fig, ruta)


def dibujar_detalle(r, ruta):
    """Una sola hoja opcional por modelo, con diagnósticos complementarios."""
    tabla, hist = r["tabla"], r["historial"]
    y, prob = tabla.pCR.to_numpy(), tabla.prob.to_numpy()
    umbral = r["metricas"]["umbral"]
    fig, ejes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(top=.85, bottom=.17, hspace=.65, wspace=.3)
    observada, predicha = calibration_curve(y, prob, n_bins=10, strategy="uniform")
    eje = ejes[0, 0]
    eje.plot([0, 1], [0, 1], "--", color="gray", label="Calibración ideal")
    eje.plot(predicha, observada, "o-", label=f"Brier = {brier_score_loss(y, prob):.3f} (menor es mejor)")
    eje.set(title="¿Coinciden puntuaciones y frecuencias reales?",
            xlabel="Puntuación media por intervalo", ylabel="Fracción real con pCR", xlim=(0, 1), ylim=(-.02, 1.02))
    eje = ejes[0, 1]
    umbrales = r["umbrales"]
    for k, nombre in (("sensibilidad", "Sensibilidad"), ("especificidad", "Especificidad"), ("precision", "Precisión (VPP)")):
        eje.plot(umbrales.umbral, umbrales[k], label=nombre)
    eje.axvline(umbral, ls="--", color="black", label=f"Umbral actual = {umbral:g}")
    eje.set(title="¿Qué cambia al mover el umbral?", xlabel="Umbral sobre la puntuación por paciente",
            ylabel="Proporción", ylim=(-.02, 1.02))
    eje = ejes[1, 0]
    for clase, nombre in ((0, "Sin pCR"), (1, "Con pCR")):
        eje.hist(prob[y == clase], bins=np.linspace(0, 1, 11), alpha=.55, label=f"{nombre} (n={np.sum(y == clase)})")
    eje.axvline(umbral, ls="--", color="black", label="Umbral actual")
    eje.set(title="¿Qué puntuaciones recibe cada clase real?", xlabel="Puntuación por paciente", ylabel="Número de pacientes")
    eje.yaxis.set_major_locator(MaxNLocator(integer=True))
    eje = ejes[1, 1]
    for clave, nombre in METRICAS.items():
        if clave in hist:
            eje.plot(hist["epoca"], hist[clave], marker="o", markersize=3, label=nombre)
    if not any(k in hist for k in METRICAS):
        eje.text(.5, .5, "Este historial antiguo no guardaba\nestas métricas por época", ha="center", transform=eje.transAxes)
    configurar_epocas(eje, hist["epoca"])
    eje.set(title=f"Métricas de validación con umbral fijo {umbral:g}", xlabel="Época", ylabel="Proporción", ylim=(-.02, 1.02))
    for eje in ejes.flat:
        eje.grid(alpha=.2)
        if eje.get_legend_handles_labels()[0]:
            eje.legend(fontsize=8)
    fig.suptitle(f"Diagnóstico adicional · {r['nombre']} · {len(tabla)} pacientes\n{titulo_general([r])}", fontsize=14)
    pie(fig, "Los tres primeros paneles usan la época elegida; el último usa todas las épocas registradas.\n"
        "Calibración: solo aparecen intervalos con pacientes; pocos puntos NO significan pocas épocas. No se recalibra el modelo.\n"
        "Las puntuaciones (especialmente con BCE ponderada o voto) no garantizan probabilidades calibradas.\n" + DEFINICION)
    guardar(fig, ruta)


def cargar_registros(carpeta):
    carpeta = Path(carpeta)
    origenes = ([carpeta] if (carpeta / "historial.csv").is_file() else
                [c for c in (carpeta / "normal", carpeta / "ponderada")
                 if (c / "historial.csv").is_file()])
    if not origenes:
        raise ValueError("No se encontraron historiales de entrenamiento en esa carpeta.")
    registros = []
    for origen in origenes:
        config = json.loads((origen / "configuracion.json").read_text())
        ruta_metricas = origen / "metricas.json"
        if not ruta_metricas.exists():
            ruta_metricas = origen / "mejor_validacion.json"
        metricas = json.loads(ruta_metricas.read_text()) if ruta_metricas.exists() else {}
        metricas.setdefault("mejor_epoca", metricas.get("epoca"))
        metricas.setdefault("umbral", config.get("umbral", .5))
        metricas.setdefault("metodo", config.get("agregacion", "mean"))
        predicciones = origen / "predicciones_validacion.csv"
        tabla = (preparar_pacientes(pd.read_csv(predicciones),
                                   metodo=metricas["metodo"], umbral=metricas["umbral"])
                 if predicciones.exists() else None)
        nombre = config.get("perdida", origen.name)
        registros.append(dict(origen=origen, config=config, metricas=metricas, tabla=tabla,
                               nombre=NOMBRES.get(nombre, nombre),
                               historial=pd.read_csv(origen / "historial.csv"),
                               umbrales=calcular_umbrales(tabla, metricas["umbral"]) if tabla is not None else None))
    con_predicciones = [r for r in registros if r["tabla"] is not None]
    if any(not r["tabla"].pCR.equals(con_predicciones[0]["tabla"].pCR) for r in con_predicciones[1:]):
        raise ValueError("No se pueden comparar pérdidas con pacientes o etiquetas diferentes.")
    return registros


def figura_individual(r, titulo, xlabel, ylabel):
    fig, eje = plt.subplots(figsize=(8, 5), layout="constrained")
    contexto = "Prueba reducida" if r["config"].get("prueba") else "Validación interna"
    eje.set(title=f"{titulo} · {r['nombre']}\n{contexto}", xlabel=xlabel, ylabel=ylabel)
    eje.title.set_fontsize(14)
    eje.tick_params(labelsize=11)
    eje.xaxis.label.set_size(12)
    eje.yaxis.label.set_size(12)
    eje.spines[["top", "right"]].set_visible(False)
    eje.grid(alpha=.18)
    return fig, eje


def guardar_presentacion(fig, ruta):
    """PNG para diapositivas y PDF vectorial; una sola gráfica en cada archivo."""
    ruta = Path(ruta)
    if fig.axes[0].get_legend_handles_labels()[0]:
        fig.axes[0].legend(fontsize=10)
    fig.savefig(ruta.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def marcar_epoca(eje, r):
    configurar_epocas(eje, r["historial"]["epoca"])
    epoca = r["metricas"].get("mejor_epoca")
    if epoca is not None:
        eje.axvline(epoca, color=".5", ls="--", lw=1, label=f"Época elegida: {epoca}")


def dibujar_presentacion(r, salida, detalle=False):
    salida.mkdir(parents=True, exist_ok=True)
    hist, tabla = r["historial"], r["tabla"]
    unidad = r["config"].get("unidad_perdidas_historial", "corte")
    fig, eje = figura_individual(r, "Pérdida por época", "Época", f"Pérdida BCE por {unidad}")
    comparable = "train_eval_loss" in hist
    eje.plot(hist.epoca, hist.train_eval_loss if comparable else hist.train_loss,
             "o-", ms=3, label="Entrenamiento (evaluación)" if comparable else "Entrenamiento", color="#2271B2")
    eje.plot(hist.epoca, hist.val_loss, "o-", ms=3, label="Validación", color="#D55E00")
    marcar_epoca(eje, r)
    guardar_presentacion(fig, salida / PRINCIPALES[0])
    fig, eje = figura_individual(r, "AUC por época", "Época", "ROC-AUC por paciente")
    if "train_auc_paciente" in hist:
        eje.plot(hist.epoca, hist.train_auc_paciente, "o-", ms=3, label="Entrenamiento", color="#2271B2")
    eje.plot(hist.epoca, hist.val_auc_paciente, "o-", ms=3, label="Validación", color="#D55E00")
    eje.axhline(.5, ls=":", color=".5", label="Azar: 0,50")
    eje.set_ylim(0, 1.02)
    marcar_epoca(eje, r)
    guardar_presentacion(fig, salida / PRINCIPALES[1])
    if tabla is None:
        return
    curvas = datos_curvas(tabla)
    for indice, titulo, xlabel, ylabel in (
        (2, "Curva ROC por paciente", "1 − especificidad", "Sensibilidad"),
        (3, "Precisión–sensibilidad por paciente", "Sensibilidad (recall)", "Precisión (VPP)"),
    ):
        fig, eje = figura_individual(r, titulo, xlabel, ylabel)
        if curvas is None:
            eje.text(.5, .5, "No estimable: solo hay una clase", ha="center", transform=eje.transAxes)
        elif indice == 2:
            eje.plot(curvas["fpr"], curvas["tpr"], lw=2, color="#2271B2", label=f"AUC = {curvas['auc']:.3f}")
            eje.plot([0, 1], [0, 1], "--", color=".5", label="Azar")
        else:
            eje.step(curvas["recall"], curvas["precision"], where="post", lw=2,
                     color="#2271B2", label=f"AP = {curvas['ap']:.3f}")
            eje.axhline(tabla.pCR.mean(), ls="--", color=".5", label=f"Prevalencia: {tabla.pCR.mean():.1%}")
        eje.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        guardar_presentacion(fig, salida / PRINCIPALES[indice])
    fig, eje = figura_individual(r, "Matriz de confusión por paciente", "Predicción", "Resultado real")
    matriz = confusion_matrix(tabla.pCR, tabla.prediccion, labels=[0, 1])
    totales = matriz.sum(axis=1, keepdims=True)
    porcentajes = np.divide(matriz, totales, out=np.full((2, 2), np.nan), where=totales != 0)
    eje.grid(False)
    eje.imshow(np.ma.masked_invalid(porcentajes), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            porcentaje = f"{porcentajes[i,j]:.1%}" if totales[i,0] else "—"
            eje.text(j, i, f"{matriz[i,j]}\n({porcentaje})", ha="center", va="center", fontsize=18,
                     color="white" if porcentajes[i,j] > .5 else "black")
    eje.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Sin pCR", "Con pCR"], yticklabels=["Sin pCR", "Con pCR"],
            xlabel=f"Predicción · umbral {r['metricas']['umbral']:g}", ylabel="Resultado real · % por fila")
    guardar_presentacion(fig, salida / PRINCIPALES[4])
    if detalle:
        dibujar_detalle_individual(r, salida / "detalle")


def dibujar_detalle_individual(r, salida):
    salida.mkdir(exist_ok=True)
    tabla, hist = r["tabla"], r["historial"]
    y, prob = tabla.pCR.to_numpy(), tabla.prob.to_numpy()
    umbral = r["metricas"]["umbral"]
    fig, eje = figura_individual(r, "Calibración por paciente", "Puntuación media", "Fracción con pCR")
    observada, predicha = calibration_curve(y, prob, n_bins=10, strategy="uniform")
    eje.plot([0, 1], [0, 1], "--", color=".5", label="Ideal")
    eje.plot(predicha, observada, "o-", label=f"Brier = {brier_score_loss(y, prob):.3f}")
    eje.set(xlim=(0, 1), ylim=(0, 1))
    guardar_presentacion(fig, salida / "06_calibracion.png")
    fig, eje = figura_individual(r, "Efecto del umbral", "Umbral", "Proporción")
    for clave in ("sensibilidad", "especificidad", "precision"):
        eje.plot(r["umbrales"].umbral, r["umbrales"][clave], label=clave.capitalize())
    eje.axvline(umbral, ls="--", color=".5", label=f"Umbral: {umbral:g}")
    eje.set_ylim(0, 1.02)
    guardar_presentacion(fig, salida / "07_umbral.png")
    fig, eje = figura_individual(r, "Puntuaciones por clase", "Puntuación por paciente", "Pacientes")
    for clase, nombre in ((0, "Sin pCR"), (1, "Con pCR")):
        eje.hist(prob[y == clase], bins=np.linspace(0, 1, 11), alpha=.6, label=nombre)
    eje.axvline(umbral, ls="--", color=".5", label=f"Umbral: {umbral:g}")
    eje.yaxis.set_major_locator(MaxNLocator(integer=True))
    guardar_presentacion(fig, salida / "08_puntuaciones.png")
    fig, eje = figura_individual(r, "Métricas por época", "Época", "Proporción")
    for clave, nombre in METRICAS.items():
        if clave in hist:
            eje.plot(hist.epoca, hist[clave], label=nombre)
    eje.set_ylim(0, 1.02)
    marcar_epoca(eje, r)
    guardar_presentacion(fig, salida / "09_metricas.png")


def generar_graficas(carpeta, detalle=False):
    """Cinco figuras independientes por pérdida, en PNG y PDF."""
    carpeta = Path(carpeta)
    registros = cargar_registros(carpeta)
    salida = carpeta / "graficas"
    salida.mkdir(exist_ok=True)
    # Preserva los antiguos paneles, fuera de las carpetas para presentación.
    for nombre in COMPUESTAS_ANTIGUAS:
        anterior = salida / nombre
        if anterior.exists():
            archivo = salida / "formato_anterior"
            archivo.mkdir(exist_ok=True)
            destino = archivo / nombre
            if not destino.exists():
                anterior.rename(destino)
    lineas = ["# Gráficas para presentación", "", titulo_general(registros), "", DEFINICION, "",
              "Cada pérdida tiene su carpeta: normal/ o ponderada/. Cada figura tiene PNG a 300 dpi y PDF vectorial.",
              "Las explicaciones se conservan aquí para dejar las diapositivas despejadas.", "",
              "- 01_perdida: todas las épocas; train_eval_loss si existe (sin aumentos/dropout), si no train_loss de optimización.",
              "- 02_auc_por_epoca: AUC por paciente en todas las épocas; train solo cuando se registró. Línea vertical: época elegida.",
              "- 03_roc y 04_precision_sensibilidad: validación del mejor modelo; recorren umbrales, no épocas. AP no es área trapezoidal.",
              "- 05_matriz_confusion: pacientes; porcentaje dentro de cada clase real (fila).",
              "Las pérdidas normal/ponderada no tienen la misma escala. No se suavizan las curvas.",
              "Si faltan predicciones, solo se generan pérdida y AUC; no se inventan curvas ROC.", "",
              "## Procedencia de las figuras", ""]
    for r in registros:
        if r["tabla"] is not None:
            r["tabla"].to_csv(r["origen"] / "predicciones_pacientes.csv")
            r["umbrales"].to_csv(r["origen"] / "metricas_umbrales.csv", index=False)
        dibujar_presentacion(r, salida / r["origen"].name, detalle)
        m = r["metricas"]
        lineas.append(f"- {r['nombre']}: época elegida {m.get('mejor_epoca')}; criterio {criterio_seleccion(r)}; "
                      f"agregación {m['metodo']}; umbral {m['umbral']:g}. "
                      + (f"{len(r['tabla'])} pacientes." if r['tabla'] is not None else "Sin predicciones disponibles."))
    lineas += ["", "Validación interna usada para elegir el modelo, no test. Pruebas reducidas no estiman rendimiento real.",
               "Una sola clase impide estimar ROC/PR. Una sola época produce un punto.",
               "Con voto, las curvas mantienen fijo el umbral de los cortes y varían el umbral sobre la fracción de votos.",
               "--detalle exporta cuatro figuras adicionales separadas: calibración, umbral, puntuaciones y métricas por época.",
               "Calibración agrupa puntuaciones, no épocas; no se recalibra el modelo. Las puntuaciones no garantizan probabilidades calibradas.", ""]
    (salida / "LEEME_GRAFICAS.md").write_text("\n".join(lineas))
    return registros[0]["tabla"] if len(registros) == 1 else {r["origen"].name: r["tabla"] for r in registros}


def comparar_perdidas(carpeta):
    """Compatibilidad: exporta las figuras de ambas pérdidas."""
    return generar_graficas(carpeta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("carpeta", type=Path, help="Carpeta de la ejecución o de un único modelo")
    p.add_argument("--detalle", action="store_true", help="Añade cuatro figuras de diagnóstico por modelo")
    args = p.parse_args()
    try:
        generar_graficas(args.carpeta, detalle=args.detalle)
    except ValueError as error:
        p.error(str(error))
    print(f"Figuras individuales PNG/PDF y su guía: {args.carpeta / 'graficas'}")


if __name__ == "__main__":
    main()
