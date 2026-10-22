import unittest
from src.training.entrenar import EXPERIMENTOS


class V2CortoTests(unittest.TestCase):
    def test_conserva_v2_con_corte_y_comparacion_train_eval(self):
        original, corto = EXPERIMENTOS['v2'], EXPERIMENTOS['v2_corto']
        self.assertEqual(corto['modelo'], 'v2')
        self.assertEqual(corto['epocas'], 15)
        self.assertEqual(corto['paciencia'], 9)
        self.assertEqual(corto['min_delta'], 0)
        self.assertTrue(corto['evaluar_train'])
        for clave in original:
            if clave not in {'epocas', 'paciencia', 'evaluar_train'}:
                self.assertEqual(corto[clave], original[clave])
        self.assertEqual(original['epocas'], 40)
        self.assertEqual(original['paciencia'], 12)


if __name__ == '__main__':
    unittest.main()
