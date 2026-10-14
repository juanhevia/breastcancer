import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.training.graficas import (
    comparar_perdidas, datos_curvas, dibujar_curvas, generar_graficas,
    guardar, preparar_pacientes,
)


class GraficasTests(unittest.TestCase):
    def test_historial_muestra_todas_las_epocas_aunque_la_mejor_sea_intermedia(self):
        historial = {
            "epoca": [1, 2, 3, 4, 5],
            "train_loss": [.8, .6, .5, .4, .3],
            "val_loss": [.9, .55, .6, .7, .8],
            "val_auc": [.5, .7, .65, .6, .55],
            "val_auc_paciente": [.55, .75, .7, .65, .6],
            "val_accuracy": [.5, .8, .7, .6, .5],
        }
        lineas = {}
        titulos = []

        def inspeccionar(fig, ruta):
            titulos.append(fig._suptitle.get_text())
            for eje in fig.axes:
                for linea in eje.lines:
                    lineas[linea.get_label()] = (linea.get_xdata(), linea.get_ydata())
            guardar(fig, ruta)

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.training.graficas.guardar", side_effect=inspeccionar):
                dibujar_curvas(historial, Path(tmp) / "curvas.png", "normal", mejor_epoca=2)
            for etiqueta, clave in (("Entrenamiento", "train_loss"), ("Validación interna", "val_loss"),
                                    ("Por paciente", "val_auc_paciente"), ("Por corte", "val_auc"),
                                    ("Accuracy", "val_accuracy")):
                x, y = lineas[etiqueta]
                np.testing.assert_array_equal(x, historial["epoca"])
                np.testing.assert_array_equal(y, historial[clave])
            self.assertIn("Mejor época = 2", lineas)
            self.assertTrue(all("5 épocas registradas" in t for t in titulos))

    def predicciones(self):
        # Distinto número de cortes por paciente: el AUC debe pesar pacientes.
        return pd.DataFrame({
            "patient_id": ["a"]*5 + ["b", "c", "c", "d"],
            "pCR": [0]*6 + [1, 1, 1],
            "probabilidad": [.1]*5 + [.6, .3, .5, .8],
        })

    def test_curvas_se_calculan_por_paciente(self):
        tabla = preparar_pacientes(self.predicciones())
        self.assertEqual(len(tabla), 4)
        self.assertAlmostEqual(tabla.loc["c", "prob"], .4)
        self.assertEqual(tabla.loc["a", "cortes"], 5)
        curvas = datos_curvas(tabla)
        self.assertAlmostEqual(curvas["auc"], .75)
        self.assertAlmostEqual(curvas["ap"], 5/6)
        np.testing.assert_array_equal(tabla.prediccion, [0, 1, 0, 1])

    def test_respeta_agregacion_y_umbral_incluyendo_igualdad(self):
        tabla = preparar_pacientes(self.predicciones(), "max", .5)
        self.assertEqual(tabla.loc["c", "prediccion"], 1)
        tabla = preparar_pacientes(self.predicciones(), "voto", .5)
        self.assertEqual(tabla.loc["c", "prob"], .5)
        self.assertEqual(tabla.loc["c", "prediccion"], 1)

    def test_rechaza_etiquetas_contradictorias(self):
        df = self.predicciones()
        df.loc[0, "pCR"] = 1
        with self.assertRaisesRegex(ValueError, "contradictorias"):
            preparar_pacientes(df)

    def test_comparacion_exige_las_mismas_pacientes(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            for nombre in ("normal", "ponderada"):
                carpeta = raiz / nombre
                carpeta.mkdir()
                self.predicciones().to_csv(carpeta / "predicciones_validacion.csv", index=False)
                (carpeta / "configuracion.json").write_text(json.dumps({"prueba": True}))
                (carpeta / "metricas.json").write_text(json.dumps({"metodo": "mean", "umbral": .5}))
            comparar_perdidas(raiz)
            with Image.open(raiz / "comparacion_roc_pr.png") as imagen:
                imagen.verify()
            distintas = self.predicciones().replace({"patient_id": {"d": "otra"}})
            distintas.to_csv(raiz / "ponderada" / "predicciones_validacion.csv", index=False)
            with self.assertRaisesRegex(ValueError, "pacientes o etiquetas diferentes"):
                comparar_perdidas(raiz)

    def test_genera_desde_csv_antiguos_y_con_una_clase(self):
        for solo_clase in (None, 0, 1):
            with self.subTest(clase=solo_clase), tempfile.TemporaryDirectory() as tmp:
                carpeta = Path(tmp)
                df = self.predicciones()
                if solo_clase is not None:
                    df = df[df.pCR == solo_clase]
                df.to_csv(carpeta / "predicciones_validacion.csv", index=False)
                # Historial antiguo: no tenía las nuevas métricas por época.
                pd.DataFrame({"epoca": [1], "train_loss": [.7], "val_loss": [.8],
                              "val_auc": [.6], "val_auc_paciente": [.75]}).to_csv(carpeta / "historial.csv", index=False)
                (carpeta / "configuracion.json").write_text(json.dumps({"perdida": "normal", "prueba": True}))
                (carpeta / "metricas.json").write_text(json.dumps({"mejor_epoca": 1, "metodo": "mean", "umbral": .5}))
                tabla = generar_graficas(carpeta)
                for nombre in ("curvas", "roc", "precision_recall", "calibracion",
                               "matriz_confusion", "umbrales", "distribucion_predicciones"):
                    with Image.open(carpeta / f"{nombre}.png") as im:
                        im.verify()
                self.assertEqual(len(pd.read_csv(carpeta / "predicciones_pacientes.csv")), len(tabla))
                umbrales = pd.read_csv(carpeta / "metricas_umbrales.csv")
                central = umbrales.loc[np.isclose(umbrales.umbral, .5)].iloc[0]
                if solo_clase is None:
                    self.assertAlmostEqual(central.sensibilidad, .5)
                    self.assertAlmostEqual(central.especificidad, .5)
                    self.assertAlmostEqual(central.f1, .5)
                else:
                    self.assertIsNone(datos_curvas(tabla))


if __name__ == "__main__":
    unittest.main()
