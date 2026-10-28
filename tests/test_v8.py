import io
import unittest

import torch

from src.models.modelo import BreastCNNV7, BreastCNNV8, contar_parametros
from src.training.comparar_cv import MODELOS
from src.training.entrenar import EXPERIMENTOS


class V8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fases_originales_y_realce_negativo_sin_recortar(self):
        x = torch.tensor([.2, .8, .3]).view(1, 3, 1, 1).expand(2, 3, 8, 8).clone()
        original = x.clone()
        entrada = BreastCNNV8.preparar_entrada(x)
        torch.testing.assert_close(entrada[:, :3], original)
        torch.testing.assert_close(entrada[:, 3], torch.full((2, 8, 8), .6))
        torch.testing.assert_close(entrada[:, 4], torch.full((2, 8, 8), -.5))
        torch.testing.assert_close(x, original)
        with self.assertRaises(ValueError):
            BreastCNNV8.preparar_entrada(torch.ones(1, 5, 8, 8))

    def test_gradientes_y_misma_inferencia_tras_recargar(self):
        torch.manual_seed(42)
        model = BreastCNNV8()
        x = torch.rand(2, 3, 64, 64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss = torch.nn.BCEWithLogitsLoss()(model(x), torch.tensor([0., 1.]))
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        optimizer.step()
        model.eval()
        with torch.no_grad():
            full = torch.rand(1, 3, 256, 256)
            expected = model(full)
            self.assertEqual(expected.shape, (1,))
            data = io.BytesIO()
            torch.save(model.state_dict(), data)
            data.seek(0)
            restored = BreastCNNV8().eval()
            restored.load_state_dict(torch.load(data, weights_only=True, map_location="cpu"))
            torch.testing.assert_close(expected, restored(full), rtol=0, atol=0)

    def test_protocolo_y_compatibilidad_v7(self):
        self.assertEqual(MODELOS["v8"], BreastCNNV8)
        self.assertEqual(EXPERIMENTOS["v8"], dict(EXPERIMENTOS["v7"], modelo="v8"))
        self.assertEqual(contar_parametros(BreastCNNV7())[0], 232513)
        self.assertEqual(contar_parametros(BreastCNNV8())[0], 232833)
        self.assertEqual(BreastCNNV7().features[0].rama[0].in_channels, 3)


if __name__ == "__main__":
    unittest.main()
