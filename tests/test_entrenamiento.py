import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.data import utils_caso as uc
from src.data.aumentos import AumentosDCE
from src.training.entrenar import entrenar
from src.models.modelo import BreastCNN, BreastCNNV2, contar_parametros


class EntrenamientoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forma_y_gradientes_con_un_solo_corte(self):
        modelo = BreastCNN()
        logits = modelo(torch.rand(1, 3, 256, 256))
        self.assertEqual(tuple(logits.shape), (1,))
        torch.nn.BCEWithLogitsLoss()(logits, torch.ones(1)).backward()
        self.assertTrue(torch.isfinite(modelo.features[0].weight.grad).all())
        self.assertEqual(contar_parametros(modelo)[0], 93825)

    def test_particiones_sin_fuga_ni_test(self):
        samples = uc.cargar_samples()
        for fold in range(5):
            tr, va = uc.particion(samples, fold)
            self.assertFalse(set(tr.patient_id) & set(va.patient_id))
            self.assertTrue((tr.split == "train").all())
            self.assertTrue((va.split == "train").all())
            self.assertEqual(len(tr) + len(va), (samples.split == "train").sum())

    def test_v2_acepta_un_corte_y_propaga_gradientes(self):
        modelo = BreastCNNV2()
        logits = modelo(torch.rand(1, 3, 256, 256))
        self.assertEqual(tuple(logits.shape), (1,))
        torch.nn.BCEWithLogitsLoss()(logits, torch.ones(1)).backward()
        for parametro in modelo.parameters():
            self.assertIsNotNone(parametro.grad)
            self.assertTrue(torch.isfinite(parametro.grad).all())
        modelo.eval()
        x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            self.assertTrue(torch.equal(modelo(x), modelo(x)))

    def test_aumentos_conservan_alineacion_fases_y_semilla(self):
        # Fases proporcionales: cualquier desalineación rompe esta relación.
        base = torch.zeros(32, 32)
        base[5:18, 12:23] = 0.25
        imagen = torch.stack([base, base * 2, base * 3])
        original = imagen.clone()
        torch.manual_seed(11)
        aumentado = AumentosDCE()(imagen)
        torch.manual_seed(11)
        self.assertTrue(torch.equal(aumentado, AumentosDCE()(imagen)))
        self.assertTrue(torch.equal(imagen, original))
        self.assertFalse(torch.equal(aumentado, imagen))
        self.assertTrue(torch.allclose(aumentado[1], aumentado[0] * 2))
        self.assertTrue(torch.allclose(aumentado[2], aumentado[0] * 3, atol=1e-6))
        self.assertGreaterEqual(aumentado.min().item(), 0)
        self.assertLessEqual(aumentado.max().item(), 1)

    def test_auc_selecciona_pesos_y_reduce_lr_antes_de_parar(self):
        torch.manual_seed(9)
        x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = torch.tensor([0., 0., 1., 1.])
        ds = TensorDataset(x, y)
        filas = pd.DataFrame({"sample_id": list("abcd"), "patient_id": list("abcd"), "pCR": y.numpy()})
        modelo = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Flatten(0))
        aucs = iter([0.6, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        evaluar_real = uc.evaluar_por_paciente
        pesos_epocas = []

        def evaluar_controlado(*args, **kwargs):
            res = evaluar_real(*args, **kwargs)
            if len(pesos_epocas) < 8:
                pesos_epocas.append({k: v.detach().clone() for k, v in modelo.state_dict().items()})
                res["auc"] = next(aucs)
            return res

        with tempfile.TemporaryDirectory() as tmp, patch.object(uc, "evaluar_por_paciente", side_effect=evaluar_controlado):
            hist, state, epoca, res = entrenar(
                modelo, ds, ds, filas, epocas=20, batch_size=4, lr=1e-2,
                paciencia=6, dispositivo="cpu", salida=tmp, seleccion="auc",
                optimizador="adamw", weight_decay=1e-3, reducir_lr=True,
            )
            self.assertEqual(epoca, 2)
            self.assertEqual(len(hist["epoca"]), 8)
            self.assertEqual(hist["lr"][5], 1e-2)
            self.assertEqual(hist["lr"][6], 5e-3)
            guardado = torch.load(Path(tmp) / "mejor_modelo.pt", weights_only=True)
            for k, valor in modelo.state_dict().items():
                self.assertTrue(torch.equal(valor, pesos_epocas[1][k]))
                self.assertTrue(torch.equal(valor, state[k]))
                self.assertTrue(torch.equal(valor, guardado[k]))
            probs = torch.sigmoid(modelo(x)).detach().numpy()
            np.testing.assert_allclose(pd.read_csv(Path(tmp) / "predicciones_validacion.csv").probabilidad, probs)
            self.assertEqual(res["auc"], evaluar_real(probs, filas)["auc"])

    def test_seleccion_auc_rechaza_validacion_sin_ambas_clases(self):
        ds = TensorDataset(torch.zeros(2, 2), torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "dos clases"):
            entrenar(torch.nn.Linear(2, 1), ds, ds, pd.DataFrame({"pCR": [0, 0]}), seleccion="auc")

    def test_guardado_y_restauracion_del_mejor_modelo(self):
        torch.manual_seed(7)
        x, y = torch.rand(4, 2), torch.tensor([0., 0., 1., 1.])
        ds = TensorDataset(x, y)
        filas = pd.DataFrame({"sample_id": ["a1", "a2", "b1", "b2"], "patient_id": ["a", "a", "b", "b"], "pCR": y.numpy()})
        modelo = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Flatten(0))
        with tempfile.TemporaryDirectory() as tmp:
            hist, state, epoca, res = entrenar(modelo, ds, ds, filas,
                epocas=2, batch_size=3, dispositivo="cpu", salida=tmp)
            self.assertEqual(epoca, int(np.argmin(hist["val_loss"])) + 1)
            self.assertEqual(res["pacientes"], 2)
            for clave in ("val_accuracy", "val_sensibilidad", "val_especificidad", "val_precision", "val_f1"):
                self.assertEqual(len(hist[clave]), 2)
            cm = res["matriz_confusion"]
            self.assertAlmostEqual(hist["val_accuracy"][epoca-1], res["accuracy"])
            self.assertAlmostEqual(hist["val_f1"][epoca-1],
                                   2*cm["VP"] / (2*cm["VP"] + cm["FP"] + cm["FN"]))
            guardado = torch.load(Path(tmp) / "mejor_modelo.pt", weights_only=True)
            for k, valor in modelo.state_dict().items():
                self.assertTrue(torch.equal(valor, state[k]))
                self.assertTrue(torch.equal(valor, guardado[k]))
            probs = torch.sigmoid(modelo(x)).detach().numpy()
            esperado = uc.evaluar_por_paciente(probs, filas)
            self.assertEqual(res["matriz_confusion"], esperado["matriz_confusion"])
            self.assertEqual(len(pd.read_csv(Path(tmp) / "historial.csv")), 2)

    def test_rechaza_cero_epocas(self):
        ds = TensorDataset(torch.zeros(2, 2), torch.zeros(2))
        with self.assertRaises(ValueError):
            entrenar(torch.nn.Linear(2, 1), ds, ds, pd.DataFrame(index=range(2)), epocas=0)


if __name__ == "__main__":
    unittest.main()
