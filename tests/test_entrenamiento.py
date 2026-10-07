import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.data import utils_caso as uc
from src.training.entrenar import entrenar
from src.models.modelo import BreastCNN, contar_parametros


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
