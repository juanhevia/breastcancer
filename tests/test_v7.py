import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.data import utils_caso as uc
from src.models.modelo import BloqueResidualDCE, BreastCNNV2, BreastCNNV7, contar_parametros
from src.training.entrenar import EXPERIMENTOS, entrenar


class V7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_atajo_transmite_senal_y_gradiente_sin_rama(self):
        bloque = BloqueResidualDCE(16, 16)
        for p in bloque.rama.parameters():
            p.data.zero_()
        x = torch.ones(1, 16, 8, 8, requires_grad=True)
        y = bloque(x)
        torch.testing.assert_close(y, x)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_forma_gradientes_y_checkpoint_recargable(self):
        torch.manual_seed(42)
        red = BreastCNNV7()
        x = torch.rand(1, 3, 256, 256)
        original = x.clone()
        y = red(x)
        self.assertEqual(tuple(y.shape), (1,))
        torch.nn.BCEWithLogitsLoss()(y, torch.ones(1)).backward()
        for p in red.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        torch.testing.assert_close(x, original)
        self.assertLess(contar_parametros(red)[0], contar_parametros(BreastCNNV2())[0])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'modelo.pt'
            torch.save(red.state_dict(), p)
            copia = BreastCNNV7()
            copia.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
            red.eval()
            copia.eval()
            with torch.no_grad():
                torch.testing.assert_close(red(x), copia(x), rtol=0, atol=0)

    def test_preset_nueve_y_optimizacion_v2(self):
        base, nuevo = EXPERIMENTOS['v2'], EXPERIMENTOS['v7']
        self.assertEqual(nuevo['modelo'], 'v7')
        self.assertEqual(nuevo['paciencia'], 9)
        self.assertEqual(nuevo['min_delta'], 0)
        for clave in base:
            if clave not in {'modelo', 'paciencia', 'evaluar_train'}:
                self.assertEqual(nuevo[clave], base[clave])

    def test_parada_tras_nueve_empates_y_reinicio_con_mejora(self):
        ds = TensorDataset(torch.tensor([[0., 0.], [1., 1.]]), torch.tensor([0., 1.]))
        filas = pd.DataFrame({'sample_id': ['a', 'b'], 'patient_id': ['a', 'b'], 'pCR': [0, 1]})
        red = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Flatten(0))
        aucs = iter([.6, .7] + [.7]*8 + [.71] + [.71]*9)
        evaluar = uc.evaluar_por_paciente

        def controlado(*args, **kwargs):
            r = evaluar(*args, **kwargs)
            r['auc'] = next(aucs, .71)
            return r

        with tempfile.TemporaryDirectory() as tmp, patch.object(uc, 'evaluar_por_paciente', side_effect=controlado):
            hist, _, mejor, _ = entrenar(red, ds, ds, filas, epocas=30, paciencia=9,
                                        min_delta=0, seleccion='auc', dispositivo='cpu', salida=tmp)
            self.assertEqual(hist['epoca'][-1], 20)
            self.assertEqual(mejor, 11)
            self.assertEqual(hist['epocas_sin_mejora'][9], 8)
            self.assertEqual(hist['epocas_sin_mejora'][10], 0)
            self.assertEqual(hist['epocas_sin_mejora'][-1], 9)
            self.assertEqual(len(list((Path(tmp)/'epocas').glob('*.pt'))), 20)


if __name__ == '__main__':
    unittest.main()
