"""Tres figuras para entender el entrenamiento y su validación interna.

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


PRINCIPALES = (
    "01_evolucion_del_aprendizaje.png",
    "02_capacidad_para_distinguir_pcr.png",
    "03_aciertos_y_errores.png",
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
    origenes = ([carpeta] if (carpeta / "predicciones_validacion.csv").is_file() else
                [c for c in (carpeta / "normal", carpeta / "ponderada")
                 if (c / "predicciones_validacion.csv").is_file()])
    if not origenes:
        raise ValueError("No se encontraron predicciones de validación en esa carpeta.")
    registros = []
    for origen in origenes:
        config = json.loads((origen / "configuracion.json").read_text())
        metricas = json.loads((origen / "metricas.json").read_text())
        tabla = preparar_pacientes(pd.read_csv(origen / "predicciones_validacion.csv"),
                                  metodo=metricas["metodo"], umbral=metricas["umbral"])
        nombre = config.get("perdida", origen.name)
        registros.append(dict(origen=origen, config=config, metricas=metricas, tabla=tabla,
                               nombre=NOMBRES.get(nombre, nombre),
                               historial=pd.read_csv(origen / "historial.csv"),
                               umbrales=calcular_umbrales(tabla, metricas["umbral"])))
    if any(not r["tabla"].pCR.equals(registros[0]["tabla"].pCR) for r in registros[1:]):
        raise ValueError("No se pueden comparar pérdidas con pacientes o etiquetas diferentes.")
    return registros


def generar_graficas(carpeta, detalle=False):
    """Tres figuras compartidas por ejecución; los CSV completos se conservan."""
    carpeta = Path(carpeta)
    registros = cargar_registros(carpeta)
    salida = carpeta / "graficas"
    salida.mkdir(exist_ok=True)
    for r in registros:
        r["tabla"].to_csv(r["origen"] / "predicciones_pacientes.csv")
        r["umbrales"].to_csv(r["origen"] / "metricas_umbrales.csv", index=False)
    dibujar_evolucion(registros, salida / PRINCIPALES[0])
    dibujar_discriminacion(registros, salida / PRINCIPALES[1])
    dibujar_aciertos(registros, salida / PRINCIPALES[2])
    if detalle:
        adicionales = salida / "detalle"
        adicionales.mkdir(exist_ok=True)
        for r in registros:
            dibujar_detalle(r, adicionales / f"{r['origen'].name}_diagnostico.png")
    lineas = ["# Cómo leer estas tres figuras", "", titulo_general(registros), "", DEFINICION, "",
              "1. **Evolución del aprendizaje:** pérdida (unidad indicada en el eje) y AUC por paciente durante TODAS las épocas. "
              "La línea vertical identifica los pesos elegidos. Menor pérdida y mayor AUC son preferibles; "
              "sus evoluciones no tienen por qué coincidir. Las BCE normal y ponderada no comparten escala.", "",
              "2. **Capacidad para distinguir pCR:** ROC y precisión–sensibilidad de la época elegida. "
              "Cada curva recorre umbrales, no épocas. AUC resume la ROC; AP es average precision, "
              "no el área trapezoidal. La referencia de precisión es la proporción real de pCR.", "",
              "3. **Aciertos y errores:** recuentos por paciente con el umbral indicado. Cada fila corresponde "
              "a una clase real y sus porcentajes suman el 100 %. Cambiar el umbral cambia estos aciertos y errores; "
              "no cambia la ROC-AUC de las puntuaciones ya calculadas.", "",
              "Sensibilidad: proporción de pCR reales detectadas. Especificidad: proporción de pacientes sin pCR "
              "identificadas como tales. Precisión (VPP): proporción de predicciones de pCR que son correctas.", "",
              "## Modelos incluidos", ""]
    for r in registros:
        m = r["metricas"]
        lineas.append(f"- {r['nombre']}: {len(r['tabla'])} pacientes; agregación `{m['metodo']}`; "
                      f"época {m['mejor_epoca']} elegida por {criterio_seleccion(r)}; umbral {m['umbral']:g}.")
        unidad = r["config"].get("unidad_perdidas_historial", "corte")
        lineas.append(f"  Las pérdidas de entrenamiento y validación se calculan por **{unidad}**. "
                      "No comparar directamente BCE por corte con BCE por paciente entre experimentos.")
        if m.get("brecha_auc") is not None:
            lineas.append(f"  AUC de entrenamiento en esa época: {m['entrenamiento']['auc']:.4f}; "
                          f"diferencia AUC train − validación: {m['brecha_auc']:.4f}. "
                          "Entrenamiento evaluado sin aumentos ni dropout; no interviene en la selección.")
    lineas += ["", "Si existe train_eval_loss, la figura 1 compara las pérdidas de train y validación "
               "medidas en modo evaluación, sin aumentos ni dropout, con los mismos pesos al final de cada época. "
               "La columna train_loss conserva la pérdida durante la optimización. Si se registró "
               "train_auc_paciente, se representa junto a la AUC de validación; los historiales antiguos "
               "no permiten reconstruir esta AUC sin pesos e imágenes."]
    lineas += ["", "Las métricas son de validación interna, usada para seleccionar el modelo; no son una evaluación en test. "
               "Una sola época produce un punto, y una sola clase impide estimar ROC y precisión–sensibilidad.", "",
               "Con agregación `voto`, el umbral configurado convierte primero los cortes en votos. "
               "Las curvas posteriores recorren umbrales sobre la fracción de votos, manteniendo fijos esos votos.", "",
               "El diagnóstico opcional (`--detalle`) reúne calibración, efecto del umbral, distribución de "
               "puntuaciones y otras métricas por época en una hoja por modelo. Los datos numéricos completos "
               "siguen en historial.csv, predicciones_pacientes.csv y metricas_umbrales.csv.", ""]
    (salida / "LEEME_GRAFICAS.md").write_text("\n".join(lineas))
    return registros[0]["tabla"] if len(registros) == 1 else {r["origen"].name: r["tabla"] for r in registros}


def comparar_perdidas(carpeta):
    """Compatibilidad: la comparación ahora forma parte de las tres figuras."""
    return generar_graficas(carpeta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("carpeta", type=Path, help="Carpeta de la ejecución o de un único modelo")
    p.add_argument("--detalle", action="store_true", help="Añade una hoja de diagnóstico por modelo")
    args = p.parse_args()
    try:
        generar_graficas(args.carpeta, detalle=args.detalle)
    except ValueError as error:
        p.error(str(error))
    print(f"Tres figuras principales y su guía: {args.carpeta / 'graficas'}")


if __name__ == "__main__":
    main()
