import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import pandas as pd
from torch.utils.data import TensorDataset

from src.data import utils_caso as uc
from src.data.aumentos import AumentosDCE
from src.models.modelo import BreastCNNV2, BreastCNNV5, contar_parametros
from src.training.entrenar import EXPERIMENTOS, entrenar


class V5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_modelo_reduce_dentro_del_forward_y_preserva_fases(self):
        torch.manual_seed(42)
        modelo = BreastCNNV5()
        x = torch.rand(1, 3, 256, 256)
        reducida = modelo.reducir(x)
        self.assertEqual(tuple(reducida.shape), (1, 3, 128, 128))
        esperada = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2]
                    + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 4
        torch.testing.assert_close(reducida, esperada)
        v2 = BreastCNNV2()
        v2.load_state_dict(modelo.state_dict())
        modelo.eval()
        v2.eval()
        with torch.no_grad():
            torch.testing.assert_close(modelo(x), v2(reducida))
        self.assertEqual(contar_parametros(modelo)[0], 310513)

    def test_gradientes_y_checkpoint_v5_recargable(self):
        modelo = BreastCNNV5()
        x = torch.rand(1, 3, 256, 256)
        logits = modelo(x)
        self.assertEqual(tuple(logits.shape), (1,))
        torch.nn.BCEWithLogitsLoss()(logits, torch.ones(1)).backward()
        for parametro in modelo.parameters():
            self.assertIsNotNone(parametro.grad)
            self.assertTrue(torch.isfinite(parametro.grad).all())
        nuevo = BreastCNNV5()
        nuevo.load_state_dict(modelo.state_dict())
        modelo.eval()
        nuevo.eval()
        with torch.no_grad():
            torch.testing.assert_close(modelo(x), nuevo(x), rtol=0, atol=0)

    def test_giro90_conjunto_sin_permutar_fases(self):
        base = torch.zeros(32, 32)
        base[3:12, 7:20] = .2
        x = torch.stack([base, 2*base, 3*base])
        with patch("torch.randint", return_value=torch.tensor(1)), patch(
            "torch.rand", side_effect=[torch.tensor(.9), torch.tensor(.5)]
        ):
            salida = AumentosDCE(rotaciones90=True)(x)
        torch.testing.assert_close(salida, torch.rot90(x, 1, dims=(-2, -1)))
        torch.testing.assert_close(salida[1], 2*salida[0])
        torch.testing.assert_close(salida[2], 3*salida[0])
        torch.manual_seed(7)
        primero = AumentosDCE(rotaciones90=True)(x)
        torch.manual_seed(7)
        torch.testing.assert_close(primero, AumentosDCE(rotaciones90=True)(x), rtol=0, atol=0)

    def ejecutar_aucs(self, aucs, paciencia):
        x = torch.tensor([[0.], [1.], [2.], [3.]])
        y = torch.tensor([0., 0., 1., 1.])
        ds = TensorDataset(x, y)
        filas = pd.DataFrame({"sample_id": list("abcd"), "patient_id": list("abcd"), "pCR": y.numpy()})
        modelo = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Flatten(0))
        evaluar_real = uc.evaluar_por_paciente
        estados = []

        def evaluar(*args, **kwargs):
            r = evaluar_real(*args, **kwargs)
            if len(estados) < len(aucs):
                r["auc"] = aucs[len(estados)]
                estados.append({k: v.detach().clone() for k, v in modelo.state_dict().items()})
            return r

        with tempfile.TemporaryDirectory() as tmp, patch.object(uc, "evaluar_por_paciente", side_effect=evaluar):
            h, state, epoca, _ = entrenar(
                modelo, ds, ds, filas, epocas=len(aucs), paciencia=paciencia,
                min_delta=.002, seleccion="auc", dispositivo="cpu", salida=tmp,
                batch_size=4, reducir_lr=True,
            )
            self.assertEqual(epoca, 1 + aucs.index(max(aucs[:len(h["epoca"])])))
            for k, v in state.items():
                torch.testing.assert_close(v, estados[epoca-1][k])
                torch.testing.assert_close(v, modelo.state_dict()[k])
            info = json.loads((Path(tmp) / "mejor_validacion.json").read_text())
            self.assertEqual(info["epoca"], epoca)
            return h

    def test_micro_mejoras_no_reinician_paciencia_pero_se_guardan(self):
        hist = self.ejecutar_aucs([.600, .601, .6015, .6018, .6019], paciencia=3)
        self.assertEqual(hist["epoca"], [1, 2, 3, 4])
        self.assertEqual(hist["epocas_sin_mejora"], [0, 1, 2, 3])
        self.assertEqual(hist["referencia_paciencia"], [.600]*4)

    def test_mejoras_pequenas_acumuladas_si_reinician_paciencia(self):
        hist = self.ejecutar_aucs([.600, .601, .6025, .603, .6035, .604, .6041], paciencia=3)
        self.assertEqual(hist["epoca"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(hist["epocas_sin_mejora"], [0, 1, 0, 1, 2, 3])

    def test_scheduler_ignora_micro_mejoras_y_reduce_lr(self):
        hist = self.ejecutar_aucs([.600, .6001, .6002, .6003, .6004, .6005], paciencia=5)
        self.assertEqual(hist["lr"][:5], [1e-4]*5)
        self.assertEqual(hist["lr"][5], 5e-5)

    def test_rechaza_delta_invalido(self):
        ds = TensorDataset(torch.zeros(2, 1), torch.tensor([0., 1.]))
        for delta in (-.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "min_delta"):
                entrenar(torch.nn.Linear(1, 1), ds, ds, pd.DataFrame(index=range(2)), min_delta=delta)

    def test_preset_v5_vuelve_a_cortes_y_evalua_train(self):
        v5 = EXPERIMENTOS["v5"]
        self.assertEqual(v5["modelo"], "v5")
        self.assertEqual(v5["unidad_entrenamiento"], "corte")
        self.assertTrue(v5["rotaciones90"] and v5["evaluar_train"])
        self.assertEqual(v5["min_delta"], .002)


if __name__ == "__main__":
    unittest.main()
