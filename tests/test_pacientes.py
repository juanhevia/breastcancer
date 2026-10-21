import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.entrenar import EXPERIMENTOS, entrenar
from src.training.pacientes import (
    CortesPorPaciente, grupos_pacientes, juntar_pacientes,
    logits_media_paciente, perdida_pacientes,
)


class PacientesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def datos(self):
        # Pacientes intercaladas con 2 y 3 cortes; índice no consecutivo.
        filas = pd.DataFrame({
            "sample_id": list("abcde"), "patient_id": ["a", "b", "a", "b", "b"],
            "pCR": [0, 1, 0, 1, 1],
        }, index=[11, 12, 30, 40, 50])
        x = torch.tensor([[-2.], [1.], [-1.], [2.], [3.]])
        y = torch.tensor(filas.pCR.to_numpy(), dtype=torch.float32)
        return TensorDataset(x, y), filas

    def test_cada_corte_una_vez_y_paciente_completa_en_cada_lote(self):
        ds, filas = self.datos()
        for batch in (1, 2):
            loader = DataLoader(CortesPorPaciente(ds, filas), batch_size=batch,
                                shuffle=True, collate_fn=juntar_pacientes,
                                generator=torch.Generator().manual_seed(42))
            vistos, pacientes = [], 0
            for x, y, tamanos in loader:
                vistos.extend(x.flatten().tolist())
                pacientes += len(y)
                for cortes, etiqueta in zip(torch.split(x, tamanos), y):
                    self.assertEqual(len(cortes), 3 if etiqueta == 1 else 2)
                    self.assertTrue(((cortes > 0) == bool(etiqueta)).all())
            self.assertEqual(pacientes, 2)
            self.assertEqual(sorted(vistos), sorted(ds.tensors[0].flatten().tolist()))

    def test_media_probabilidades_y_gradientes_no_media_logits(self):
        z = torch.tensor([-3., 2., .2, 1., 5.], dtype=torch.float64, requires_grad=True)
        agregado = logits_media_paciente(z, [2, 3])
        esperado = torch.stack([z[:2].sigmoid().mean(), z[2:].sigmoid().mean()])
        torch.testing.assert_close(agregado.sigmoid(), esperado)
        self.assertFalse(torch.allclose(agregado[0], z[:2].mean()))
        etiquetas = torch.tensor([0., 1.], dtype=z.dtype)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(agregado, etiquetas)
        grad = torch.autograd.grad(loss, z, retain_graph=True)[0]
        directa = torch.nn.functional.binary_cross_entropy(esperado, etiquetas)
        torch.testing.assert_close(loss, directa)
        torch.testing.assert_close(grad, torch.autograd.grad(directa, z)[0])

    def test_logits_extremos_mantienen_perdida_y_gradientes_finitos(self):
        z = torch.tensor([-1000., -1000., 1000., 1000.], requires_grad=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_media_paciente(z, [2, 2]), torch.tensor([1., 0.]))
        loss.backward()
        self.assertAlmostEqual(loss.item(), 1000., places=3)
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue((z.grad.abs() > 0).all())

    def test_bce_ponderada_pesa_pacientes_no_numero_de_cortes(self):
        ds, filas = self.datos()
        z = ds.tensors[0].flatten()
        p = torch.stack([z[[0, 2]].sigmoid().mean(), z[[1, 3, 4]].sigmoid().mean()])
        esperada = (-torch.log1p(-p[0]) - 2.5 * p[1].log()) / 2
        torch.testing.assert_close(perdida_pacientes(z, filas, 2.5), esperada)
        # Repetir todos los cortes de una paciente no aumenta su peso en la BCE.
        extra = filas.iloc[[0, 2]].assign(sample_id=["f", "g"])
        duplicada = pd.concat([filas, extra])
        torch.testing.assert_close(perdida_pacientes(torch.cat([z, z[[0, 2]]]), duplicada, 2.5), esperada)

    def test_rechaza_etiquetas_contradictorias_y_cortes_duplicados(self):
        ds, filas = self.datos()
        contradictorias = filas.copy()
        contradictorias.loc[30, "pCR"] = 1
        with self.assertRaisesRegex(ValueError, "contradictorias"):
            grupos_pacientes(contradictorias)
        with self.assertRaisesRegex(ValueError, "duplicados"):
            grupos_pacientes(pd.concat([filas, filas.iloc[:1]]))
        mal_ds = TensorDataset(ds.tensors[0], 1-ds.tensors[1])
        with self.assertRaisesRegex(ValueError, "no coinciden"):
            CortesPorPaciente(mal_ds, filas)[0]

    def test_entrenamiento_y_evaluacion_usan_bce_paciente_y_guardan_mejor(self):
        ds, tr = self.datos()
        va = tr.assign(patient_id=tr.patient_id + "_val", sample_id=tr.sample_id + "_val")
        for peso in (None, 2.5):
            torch.manual_seed(42)
            modelo = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Flatten(0))
            with tempfile.TemporaryDirectory() as tmp:
                hist, state, epoca, res = entrenar(
                    modelo, ds, ds, va, train_samples=tr, train_eval_ds=ds,
                    unidad_entrenamiento="paciente", batch_pacientes=2, batch_size=2,
                    epocas=2, dispositivo="cpu", salida=tmp, pos_weight=peso,
                    seleccion="auc", optimizador="adamw",
                )
                with torch.no_grad():
                    z = modelo(ds.tensors[0])
                    esperada = perdida_pacientes(z, tr, peso).item()
                self.assertEqual(len(hist["epoca"]), 2)
                self.assertEqual(res["pacientes"], 2)
                self.assertAlmostEqual(hist["val_loss"][epoca-1], esperada, places=6)
                self.assertAlmostEqual(hist["train_eval_loss"][epoca-1], esperada, places=6)
                guardado = torch.load(Path(tmp) / "mejor_modelo.pt", weights_only=True)
                for clave, valor in state.items():
                    torch.testing.assert_close(valor, guardado[clave])
                    torch.testing.assert_close(valor, modelo.state_dict()[clave])
                pred = pd.read_csv(Path(tmp) / "predicciones_validacion.csv")
                np.testing.assert_allclose(pred.probabilidad, z.sigmoid().numpy(), rtol=1e-6)
                metricas = json.loads((Path(tmp) / "metricas.json").read_text())
                self.assertAlmostEqual(metricas["train_eval_loss"], esperada, places=6)

    def test_modo_paciente_sin_diagnostico_y_rechaza_agregacion_distinta(self):
        ds, tr = self.datos()
        va = tr.assign(patient_id=tr.patient_id + "_val", sample_id=tr.sample_id + "_val")
        modelo = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Flatten(0))
        hist, _, _, _ = entrenar(modelo, ds, ds, va, train_samples=tr,
                                 unidad_entrenamiento="paciente", epocas=1, dispositivo="cpu")
        self.assertNotIn("train_eval_loss", hist)
        with self.assertRaisesRegex(ValueError, "agregación mean"):
            entrenar(modelo, ds, ds, va, train_samples=tr,
                     unidad_entrenamiento="paciente", metodo_agregacion="max")
        with self.assertRaisesRegex(ValueError, "compartidas"):
            entrenar(modelo, ds, ds, tr, train_samples=tr, unidad_entrenamiento="paciente")

    def test_v4_conserva_arquitectura_v2_y_activa_diagnostico(self):
        self.assertEqual(EXPERIMENTOS["v4"]["modelo"], "v2")
        self.assertEqual(EXPERIMENTOS["v4"]["unidad_entrenamiento"], "paciente")
        self.assertTrue(EXPERIMENTOS["v4"]["evaluar_train"])


if __name__ == "__main__":
    unittest.main()
