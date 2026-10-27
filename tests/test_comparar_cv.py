import json
import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.training.comparar_cv import (
    elegir_umbral, ejecutar_fold, main, metricas, particiones, resumir, validar_samples,
)


class ComparacionCVTests(unittest.TestCase):
    def samples(self):
        filas = []
        for fold in range(5):
            for paciente in range(4):
                for corte in range(2):
                    filas.append(dict(sample_id=f"{fold}_{paciente}_{corte}",
                                      patient_id=f"{fold}_{paciente}", pCR=paciente % 2,
                                      fold=fold, split="train"))
        filas.append(dict(sample_id="test", patient_id="test", pCR=1, fold=-1, split="test"))
        return pd.DataFrame(filas)

    def test_cada_paciente_se_evalua_una_vez_y_test_no_entra(self):
        samples = self.samples()
        evaluadas = []
        for fold in range(5):
            tr, interna, externa = particiones(samples, fold)
            grupos = [set(g.patient_id) for g in (tr, interna, externa)]
            self.assertFalse(grupos[0] & grupos[1] or grupos[0] & grupos[2] or grupos[1] & grupos[2])
            self.assertEqual(set.union(*grupos), set(samples[samples.split == "train"].patient_id))
            self.assertEqual(set(interna.fold), {(fold + 1) % 5})
            evaluadas.extend(externa.patient_id.unique())
        self.assertEqual(len(evaluadas), len(set(evaluadas)))
        self.assertEqual(len(evaluadas), 20)

    def test_rechaza_fuga_y_folds_incompletos(self):
        s = self.samples()
        s.loc[0, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "contradictorios"):
            validar_samples(s)
        s = self.samples()
        with self.assertRaisesRegex(ValueError, "cinco folds"):
            validar_samples(s[s.fold != 4])

    def test_umbral_y_metricas_usan_media_por_paciente(self):
        pred = pd.DataFrame(dict(patient_id=["a", "a", "b", "c", "d"],
                                 pCR=[0, 0, 0, 1, 1], probabilidad=[.1, .3, .25, .35, .4]))
        self.assertEqual(elegir_umbral(pred), .35)
        # Repetir todos los cortes de una paciente no cambia su peso en las métricas.
        repetida = pd.concat([pred, pred[pred.patient_id == "a"]] * 3)
        self.assertAlmostEqual(elegir_umbral(repetida), .35)
        self.assertEqual(metricas(pred, .35)["f1_macro"], 1.)
        self.assertEqual(metricas(repetida, .35)["pacientes"], 4)
        self.assertEqual(metricas(pred, .5)["sensibilidad"], 0.)

    def test_externo_solo_se_predice_despues_de_congelar_seleccion(self):
        with tempfile.TemporaryDirectory() as tmp:
            salida = Path(tmp) / "normal"
            args = Namespace(datos=Path(tmp), prueba=False, semilla=42, epocas=2,
                             batch_size=2, dispositivo="cpu", num_workers=0, lr=3e-4)
            samples = self.samples()
            pasos = []

            def entrenar_falso(modelo, tr, va, filas, **kw):
                pasos.append("entrenar")
                self.assertEqual(kw["lr"], 3e-4)
                config = json.loads((kw["salida"] / "configuracion.json").read_text())
                self.assertEqual(config["lr"], kw["lr"])
                self.assertEqual(kw["paciencia"], 9)
                self.assertFalse(set(filas.patient_id) & set(samples[samples.fold == 0].patient_id))
                self.assertFalse(set(kw["train_samples"].patient_id) & set(samples[samples.fold == 0].patient_id))
                pred = filas[["sample_id", "patient_id", "pCR"]].assign(probabilidad=filas.pCR * .2 + .1)
                pred.to_csv(kw["salida"] / "predicciones_validacion.csv", index=False)
                (kw["salida"] / "metricas.json").write_text(json.dumps({"auc": 1., "entrenamiento": {"auc": 1.}}))
                return {"epoca": [1, 2]}, {}, 1, {}

            def predecir_falso(modelo, filas, *args):
                pasos.append("externo")
                self.assertTrue((salida / "seleccion_congelada.json").exists())
                self.assertEqual(set(filas.fold), {0})
                self.assertFalse((salida / "resultado.json").exists())
                return filas[["sample_id", "patient_id", "pCR"]].assign(probabilidad=filas.pCR * .2 + .1)

            with patch("src.training.comparar_cv.entrenar", side_effect=entrenar_falso), \
                    patch("src.training.comparar_cv.predecir", side_effect=predecir_falso):
                r = ejecutar_fold(args, samples, 0, "v2", "normal", salida)
            self.assertEqual(pasos, ["entrenar", "externo"])
            self.assertEqual(r["externo"]["auc"], 1.)
            self.assertEqual(r["externo"]["f1_macro"], 1.)
            self.assertEqual(r["externo_umbral_05"]["sensibilidad"], 0.)
            self.assertTrue((salida / "resultado.json").exists())

    def test_incompleto_no_se_marca_completado(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(datos=Path(tmp), prueba=False, semilla=42, epocas=2,
                             batch_size=2, dispositivo="cpu", num_workers=0)
            salida = Path(tmp) / "normal"
            with patch("src.training.comparar_cv.entrenar", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    ejecutar_fold(args, self.samples(), 0, "v7", "normal", salida)
            self.assertFalse((salida / "resultado.json").exists())

    def test_resumen_pareado_y_desviacion_muestral_no_confunde_auc_interno(self):
        resultados = []
        for fold in range(5):
            for modelo, ventaja in (("v2", 0.), ("v7", .02)):
                resultados.append(dict(modelo=modelo, perdida="normal", fold=fold,
                                       mejor_epoca=2, umbral=.35, prueba=False,
                                       auc_train=.99, auc_interno=.95,
                                       externo=dict(auc=.5 + fold * .1 + ventaja, f1_macro=.5,
                                                    sensibilidad=.6, especificidad=.6)))
        with tempfile.TemporaryDirectory() as tmp:
            salida = Path(tmp)
            resumen = resumir(resultados, salida)
            self.assertTrue(all(r["completo"] for r in resumen))
            self.assertAlmostEqual(resumen[0]["auc_media"], .7)
            self.assertAlmostEqual(resumen[0]["auc_desviacion"], .158113883)
            pares = pd.read_csv(salida / "diferencias_pareadas.csv")
            self.assertTrue(((pares.diferencia_v7_menos_v2 - .02).abs() < 1e-8).all())
            parcial = resumir(resultados[:2], salida)
            self.assertFalse(parcial[0]["completo"])
            self.assertIsNone(json.loads((salida / "resumen.json").read_text())["modelos"][0]["auc_desviacion"])

    def test_reanudar_omite_completos_reinicia_incompleto_y_rechaza_otro_protocolo(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            (raiz / "metadata").mkdir()
            (raiz / "imagen.png").touch()
            samples = self.samples()
            for c in ("path_pre", "path_early", "path_late"):
                samples[c] = "imagen.png"
            # El test oficial no tiene imágenes: tampoco deben comprobarse ni abrirse.
            samples.loc[samples.split == "test", "path_pre"] = "test_no_disponible.png"
            samples.to_csv(raiz / "metadata/samples.csv", index=False)
            salida = raiz / "resultados"
            argv = ["comparar_cv", "--datos", str(raiz), "--salida", str(salida),
                    "--dispositivo", "cpu", "--perdida", "normal"]
            llamadas = []

            def ejecutar_falso(args, samples, fold, modelo, perdida, destino):
                llamadas.append((fold, modelo))
                destino.mkdir(parents=True)
                if len(llamadas) == 2:
                    (destino / "parcial.txt").write_text("incompleto")
                    raise KeyboardInterrupt
                r = dict(modelo=modelo, perdida=perdida, fold=fold, prueba=False,
                         mejor_epoca=1, umbral=.3, auc_interno=.7, auc_train=.8,
                         externo=dict(auc=.6, sensibilidad=.5, especificidad=.7, f1_macro=.5))
                (destino / "resultado.json").write_text(json.dumps(r))
                return r

            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("src.training.comparar_cv.ejecutar_fold", side_effect=ejecutar_falso):
                with patch("sys.argv", argv), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 130)
                with patch("sys.argv", argv + ["--reanudar"]):
                    main()
                self.assertEqual(llamadas.count((0, "v2")), 1)
                self.assertEqual(llamadas.count((0, "v7")), 2)
                self.assertEqual(len(list(salida.glob("fold_*/v*/normal/resultado.json"))), 10)
                self.assertEqual(len(list(salida.glob("fold_0/v7/normal_incompleto_*/parcial.txt"))), 1)
                with patch("sys.argv", argv + ["--reanudar", "--batch-size", "8"]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

                # Ensayo aislado: solo cinco v7, sin volver a ejecutar v2 ni
                # reutilizar por accidente resultados con otra tasa.
                llamadas.clear()
                nuevo = raiz / "v7_lr"
                argv_lr = ["comparar_cv", "--datos", str(raiz), "--salida", str(nuevo),
                           "--dispositivo", "cpu", "--perdida", "normal",
                           "--modelos", "v7", "--lr", "0.0003"]

                def cinco_folds(args, samples, fold, modelo, perdida, destino):
                    self.assertEqual(args.lr, .0003)
                    self.assertEqual(modelo, "v7")
                    destino.mkdir(parents=True)
                    llamadas.append((fold, modelo))
                    r = dict(modelo=modelo, perdida=perdida, fold=fold, prueba=False,
                             mejor_epoca=1, umbral=.3, auc_interno=.7, auc_train=.8,
                             externo=dict(auc=.6, sensibilidad=.5, especificidad=.7, f1_macro=.5))
                    (destino / "resultado.json").write_text(json.dumps(r))
                    return r

                with patch("src.training.comparar_cv.ejecutar_fold", side_effect=cinco_folds), \
                        patch("sys.argv", argv_lr):
                    main()
                self.assertEqual(llamadas, [(f, "v7") for f in range(5)])
                protocolo = json.loads((nuevo / "protocolo.json").read_text())
                self.assertEqual(protocolo["modelos"], ["v7"])
                self.assertEqual(protocolo["lr"], .0003)
                self.assertEqual(len(list(salida.glob("fold_*/v*/normal/resultado.json"))), 10)
                with patch("sys.argv", argv_lr + ["--reanudar", "--lr", "0.0001"]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_rechaza_lr_invalido_antes_de_leer_datos(self):
        for lr in ("0", "-0.1", "nan", "inf"):
            with self.subTest(lr=lr), patch("sys.argv", ["comparar_cv", "--lr", lr]), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    patch("src.training.comparar_cv.uc.cargar_samples") as cargar, \
                    self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            cargar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
