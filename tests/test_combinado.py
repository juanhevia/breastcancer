import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.servidor import crear_app
from src.models.combinado import Complemento, matriz, probabilidad_json, validar_variables
from tests.test_app import archivos


class ComplementoTests(unittest.TestCase):
    def paquete(self, ruta):
        modelos = {}
        for nombre, con_imagen in [('solo_variables', False), ('combinado', True)]:
            x = matriz([30, 40, np.nan, 60, 50, 70], ['TripleNeg', 'HER2pos', 'No disponible', 'HRposHER2neg', 'TripleNeg', 'HER2pos'], [.1, .2, .3, .4, .5, .6], 50, con_imagen)
            scaler = StandardScaler().fit(x)
            m = LogisticRegression().fit(scaler.transform(x), [0, 1, 0, 1, 0, 1])
            parametros = dict(media=scaler.mean_.tolist(), escala=scaler.scale_.tolist(), coeficientes=m.coef_[0].tolist(), intercepto=float(m.intercept_[0]), umbral=.4)
            np.testing.assert_allclose(probabilidad_json(parametros, x), m.predict_proba(scaler.transform(x))[:, 1], atol=1e-12)
            modelos[nombre] = parametros
        ruta.write_text(json.dumps(dict(version=1, sha256_cnn='abc', variables=['age', 'HR_HER2_STATUS'], mediana_edad=50, modelos=modelos, advertencia='Prueba')))
        return Complemento(ruta, 'abc')

    def test_json_equivalente_y_checkpoint_verificado(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp)/'modelo.json'
            c = self.paquete(ruta)
            with self.assertRaises(ValueError):
                Complemento(ruta, 'otra CNN')
            r = c.predecir(.3, {'HR_HER2_STATUS': 'HER2pos'})
            self.assertTrue(r['edad_imputada'])
            self.assertEqual(r, c.predecir(.3, {'HR_HER2_STATUS': 'HER2pos'}))
            with self.assertRaises(ValueError):
                c.predecir(float('nan'), {'age': 50})

    def test_variables_invalidas(self):
        for v in ({}, {'age': -1}, {'age': True}, {'age': float('nan')}, {'age': '50'}, {'age': 50, 'pCR': 1}, {'age': 50, 'tum_vol': 3}, {'HR_HER2_STATUS': 'inventado'}):
            with self.subTest(v=v), self.assertRaises(ValueError):
                validar_variables(v)

    def test_api_preserva_cnn_y_rechaza_variables_antes_de_inferencia(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.paquete(Path(tmp)/'modelo.json')
            pred = Mock()
            pred.predecir.return_value = dict(probabilidad=.32, umbral=.356, clase=0)
            with TestClient(crear_app(pred, c)) as client:
                self.assertTrue(client.get('/api/complemento').json()['disponible'])
                payload = dict(files=archivos(), variables={'age': 50, 'HR_HER2_STATUS': 'HER2pos'})
                baseline = client.post('/api/predecir', json=payload).json()
                r = client.post('/api/predecir_combinado', json=payload)
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()['solo_imagenes'], baseline)
                self.assertEqual(r.json()['combinado'], c.predecir(.32, payload['variables'])['combinado'])
                pred.predecir.reset_mock()
                payload['variables'] = {'age': 50, 'pCR': 1}
                self.assertEqual(client.post('/api/predecir_combinado', json=payload).status_code, 422)
                pred.predecir.assert_not_called()
            with TestClient(crear_app(pred)) as client:
                self.assertFalse(client.get('/api/complemento').json()['disponible'])
                self.assertEqual(client.post('/api/predecir_combinado', json={}).status_code, 503)


if __name__ == '__main__':
    unittest.main()
