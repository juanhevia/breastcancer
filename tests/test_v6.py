import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models.modelo import BreastCNNV2, BreastCNNV6, ContrasteConjuntoDCE, contar_parametros
from src.training.entrenar import EXPERIMENTOS


class V6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_contraste_aplica_la_misma_ventana_a_las_tres_fases(self):
        x = torch.linspace(.02, .7, 48).reshape(1, 3, 4, 4)
        x[0, :, 0, 0] = 0
        original = x.clone()
        bajo, alto = np.quantile(x.numpy()[x.numpy() > 0], [.05, .95])
        esperado = np.clip((x.numpy() - bajo) / max(alto - bajo, 1/3), 0, 1)
        ajustado = ContrasteConjuntoDCE()(x)
        np.testing.assert_allclose(ajustado.numpy(), esperado, atol=1e-6)
        torch.testing.assert_close(x, original)
        # Donde no se recortan extremos, las diferencias temporales se escalan igual.
        dentro = ((ajustado > 0) & (ajustado < 1)).all(dim=1)
        diferencia = ajustado[:, 1] - ajustado[:, 0]
        torch.testing.assert_close(diferencia[dentro], ((x[:, 1] - x[:, 0]) / (alto-bajo))[dentro])

    def test_no_depende_del_lote_ni_del_modo_y_conserva_originales(self):
        red = BreastCNNV6()
        x = torch.rand(2, 3, 32, 32)
        preparado = red.preparar_entrada(x)
        self.assertEqual(tuple(preparado.shape), (2, 6, 32, 32))
        torch.testing.assert_close(preparado[:, :3], x, rtol=0, atol=0)
        individual = torch.cat([red.preparar_entrada(m[None]) for m in x])
        torch.testing.assert_close(preparado, individual, rtol=0, atol=0)
        red.eval()
        torch.testing.assert_close(preparado, red.preparar_entrada(x), rtol=0, atol=0)

    def test_vacios_constantes_y_contraste_minimo(self):
        ajuste = ContrasteConjuntoDCE()
        x = torch.stack([torch.zeros(3, 16, 16), torch.full((3, 16, 16), .2)])
        torch.testing.assert_close(ajuste(x), x)
        pequeno = torch.linspace(.1, .11, 768).reshape(1, 3, 16, 16)
        resultado = ajuste(pequeno)
        self.assertTrue(torch.isfinite(resultado).all())
        self.assertLessEqual(resultado.max().item(), .030001)
        self.assertGreaterEqual(resultado.min().item(), 0)

    def test_gradientes_y_pesos_recargables_incluyen_preprocesado(self):
        torch.manual_seed(42)
        red = BreastCNNV6()
        x = torch.rand(2, 3, 32, 32)
        salida = red(x)
        self.assertEqual(tuple(salida.shape), (2,))
        torch.nn.BCEWithLogitsLoss()(salida, torch.tensor([0., 1.])).backward()
        for parametro in red.parameters():
            self.assertIsNotNone(parametro.grad)
            self.assertTrue(torch.isfinite(parametro.grad).all())
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / 'epoca.pt'
            torch.save(red.state_dict(), ruta)
            nueva = BreastCNNV6()
            nueva.load_state_dict(torch.load(ruta, weights_only=True, map_location='cpu'))
            red.eval()
            nueva.eval()
            with torch.no_grad():
                torch.testing.assert_close(red(x), nueva(x), rtol=0, atol=0)
        self.assertEqual(contar_parametros(red)[0] - contar_parametros(BreastCNNV2())[0], 432)

    def test_preset_conserva_optimizacion_v2(self):
        v2, v6 = EXPERIMENTOS['v2'], EXPERIMENTOS['v6']
        for clave in v2:
            if clave not in {'modelo', 'evaluar_train'}:
                self.assertEqual(v6[clave], v2[clave])
        self.assertEqual(v6['modelo'], 'v6')
        self.assertTrue(v6['evaluar_train'])

    @unittest.skipUnless(torch.cuda.is_available(), 'Requiere GPU CUDA/ROCm')
    def test_gpu_produce_el_mismo_preprocesado(self):
        x = torch.rand(2, 3, 32, 32)
        torch.testing.assert_close(ContrasteConjuntoDCE()(x),
                                   ContrasteConjuntoDCE().cuda()(x.cuda()).cpu(), rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
