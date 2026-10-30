"""Servidor local: python -m uvicorn app.servidor:app --host 127.0.0.1 --port 8000."""

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.inferencia import Predictor, leer_paciente, vista_previa
from src.models.combinado import Complemento, validar_variables

RAIZ = Path(__file__).resolve().parent
MAX_PETICION = 16 * 1024 * 1024


def crear_app(predictor=None, complemento=None):
    servicio = FastAPI(title='BreastDCEDL · Demo educativa', docs_url=None, redoc_url=None)
    servicio.state.predictor = predictor
    servicio.state.complemento = complemento
    servicio.mount('/static', StaticFiles(directory=RAIZ / 'static'), name='static')

    @servicio.get('/')
    def inicio():
        return FileResponse(RAIZ / 'static/index.html')

    @servicio.get('/api/modelo')
    def modelo():
        pred = servicio.state.predictor
        return pred.info if pred else dict(disponible=False, mensaje='Modelo pendiente de configurar. Puedes inspeccionar las imágenes.')

    @servicio.get('/api/complemento')
    def info_complemento():
        c = servicio.state.complemento
        return dict(disponible=c is not None, metricas=c.config.get('metricas_internas') if c else None,
                    alcance=c.config['advertencia'] if c else 'Complemento pendiente de configurar.')

    async def cargar(request, con_variables=False):
        # Límite real durante lectura, incluso si falta Content-Length. Sin persistencia de imágenes.
        contenido = bytearray()
        async for chunk in request.stream():
            if len(contenido) + len(chunk) > MAX_PETICION:
                raise HTTPException(413, 'La carga completa supera 16 MiB.')
            contenido.extend(chunk)
        try:
            cuerpo = json.loads(contenido)
            if not isinstance(cuerpo, dict):
                raise ValueError('Se esperaba un grupo de imágenes.')
            if con_variables:
                validar_variables(cuerpo.get('variables'))
            paciente = await run_in_threadpool(leer_paciente, cuerpo.get('files'))
            return (paciente, cuerpo['variables']) if con_variables else paciente
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @servicio.post('/api/inspeccionar')
    async def inspeccionar(request: Request):
        paciente = await cargar(request)
        return await run_in_threadpool(vista_previa, paciente)

    @servicio.post('/api/predecir')
    async def predecir(request: Request):
        if servicio.state.predictor is None:
            raise HTTPException(503, 'Todavía no hay un modelo configurado para predecir.')
        paciente = await cargar(request)
        try:
            return await run_in_threadpool(servicio.state.predictor.predecir, paciente)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @servicio.post('/api/predecir_combinado')
    async def predecir_combinado(request: Request):
        if servicio.state.predictor is None or servicio.state.complemento is None:
            raise HTTPException(503, 'El complemento y su CNN deben estar configurados.')
        paciente, variables = await cargar(request, con_variables=True)
        try:
            imagen = await run_in_threadpool(servicio.state.predictor.predecir, paciente)
            extra = servicio.state.complemento.predecir(imagen['probabilidad'], variables)
            return dict(solo_imagenes=imagen, **extra)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return servicio


predictor = None
if os.environ.get('BREAST_MODEL_DIR'):
    try:
        predictor = Predictor(os.environ['BREAST_MODEL_DIR'], batch_size=int(os.environ.get('BREAST_INFERENCE_BATCH', '8')))
    except Exception:
        logging.exception('No se pudo cargar el modelo configurado; inferencia desactivada.')
complemento = None
ruta_complemento = Path(os.environ.get('BREAST_COMPLEMENTO', str(RAIZ / 'paquetes/complemento_v8.json')))
if predictor and ruta_complemento.is_file():
    try:
        complemento = Complemento(ruta_complemento, predictor.info['sha256'])
    except Exception:
        logging.exception('Complemento incompatible; la CNN sigue disponible.')
app = crear_app(predictor, complemento)
