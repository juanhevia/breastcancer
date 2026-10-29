import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from fastapi.testclient import TestClient

from app.inferencia import Predictor, leer_paciente, vista_previa
from app.servidor import crear_app
from src.data.utils_caso import cargar_imagen
from src.models.modelo import BreastCNNV7, BreastCNNV8


def archivo(nombre, valor=128, size=(256, 256), modo='L', formato='PNG'):
    memoria = io.BytesIO()
    Image.new(modo, size, valor).save(memoria, format=formato)
    return dict(name=nombre, content=base64.b64encode(memoria.getvalue()).decode())


def archivos():
    return [archivo(f'PACIENTE_z{c:03d}_{f}.png', v+c) for c in (1, 2)
            for f,v in [('PRE', 20), ('EARLY', 160), ('LATE', 100)]]


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_preprocesado_identico_al_dataset_y_orden_independiente(self):
        files = archivos()
        paciente = leer_paciente(list(reversed(files)))
        with tempfile.TemporaryDirectory() as tmp:
            for f in files:
                (Path(tmp) / f['name']).write_bytes(base64.b64decode(f['content']))
            from types import SimpleNamespace
            fila = SimpleNamespace(**{f'path_{f.lower()}':f'PACIENTE_z001_{f}.png' for f in ('PRE','EARLY','LATE')})
            np.testing.assert_array_equal(paciente['tensor'][0].numpy(), cargar_imagen(fila, tmp))
        self.assertEqual(paciente['cortes'], [1, 2])
        self.assertEqual(paciente['tensor'].shape, (2,3,256,256))

    def test_rechaza_entradas_invalidas_y_pacientes_mezcladas(self):
        f = archivos()
        casos = [f[:-1], f+[f[0]], [dict(f[0],name='../PACIENTE_z001_PRE.png')]+f[1:],
                 [dict(f[0],name='OTRA_z001_PRE.png')]+f[1:],
                 [dict(f[0],content='no base64')]+f[1:],
                 [archivo(f[0]['name'],size=(32,32))]+f[1:],
                 [archivo(f[0]['name'],modo='RGB')]+f[1:],
                 [archivo(f[0]['name'],formato='JPEG')]+f[1:],
                 [dict(f[0],content=base64.b64encode(b'corrupto').decode())]+f[1:]]
        for i,caso in enumerate(casos):
            with self.subTest(caso=i), self.assertRaises(ValueError):
                leer_paciente(caso)

    def test_inspeccion_funciona_sin_modelo_y_no_inventa_prediccion(self):
        with TestClient(crear_app()) as cliente:
            self.assertEqual(cliente.get('/').status_code,200)
            self.assertFalse(cliente.get('/api/modelo').json()['disponible'])
            self.assertEqual(cliente.post('/api/predecir',json={'files':archivos()}).status_code,503)
            r=cliente.post('/api/inspeccionar',json={'files':archivos()})
            self.assertEqual(r.status_code,200)
            self.assertEqual(r.json()['numero_cortes'],2)
            self.assertEqual(set(r.json()['vistas'][0]),{'corte','pre','early','late','realce'})
            self.assertEqual(cliente.post('/api/inspeccionar',json={'files':archivos()[:-1]}).status_code,422)
            with patch('app.servidor.MAX_PETICION',20):
                self.assertEqual(cliente.post('/api/inspeccionar',json={'files':archivos()}).status_code,413)

    def test_inferencia_igual_a_pipeline_y_reproducible(self):
        for nombre, clase in [('v7', BreastCNNV7), ('v8', BreastCNNV8)]:
            with tempfile.TemporaryDirectory() as tmp:
                ruta=Path(tmp)
                torch.manual_seed(42)
                model=clase().eval()
                torch.save(model.state_dict(),ruta/'mejor_modelo.pt')
                (ruta/'configuracion.json').write_text(json.dumps(dict(modelo=nombre,agregacion='mean',normalizacion='PRE, EARLY, LATE / 255',prueba=True)))
                (ruta/'metricas.json').write_text(json.dumps(dict(auc=.5,umbral=.5)))
                manifest=dict(umbral=.35,origen_umbral='validacion_interna',criterio_umbral='Prueba técnica',
                              sha256=hashlib.sha256((ruta/'mejor_modelo.pt').read_bytes()).hexdigest())
                (ruta/'app_manifest.json').write_text(json.dumps(manifest))
                predictor=Predictor(ruta)
                paciente=leer_paciente(archivos())
                with torch.inference_mode():
                    esperado=torch.sigmoid(model(paciente['tensor'])).numpy().astype(np.float64).mean()
                resultado=predictor.predecir(paciente)
                self.assertAlmostEqual(resultado['probabilidad'],esperado,places=7)
                self.assertEqual(resultado['clase'],int(esperado>=.35))
                self.assertEqual(predictor.predecir(paciente)['probabilidad'],resultado['probabilidad'])
                with TestClient(crear_app(predictor)) as cliente:
                    r=cliente.post('/api/predecir',json={'files':archivos()})
                    self.assertEqual(r.status_code,200)
                    self.assertEqual(r.json()['probabilidad'],resultado['probabilidad'])
                    self.assertEqual(cliente.get('/api/modelo').json()['metricas_internas']['umbral'],.5)
                manifest['sha256']='incorrecto'
                (ruta/'app_manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'checksum'):
                    Predictor(ruta)

    def test_realce_no_modifica_entrada(self):
        paciente=leer_paciente(archivos()); antes=paciente['tensor'].clone()
        vista_previa(paciente)
        self.assertTrue(torch.equal(antes,paciente['tensor']))


if __name__=='__main__':
    unittest.main()
