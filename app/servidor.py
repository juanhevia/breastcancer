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

RAIZ = Path(__file__).resolve().parent
MAX_PETICION = 16 * 1024 * 1024


def crear_app(predictor=None):
    servicio = FastAPI(title='BreastDCEDL · Demo educativa', docs_url=None, redoc_url=None)
    servicio.state.predictor = predictor
    servicio.mount('/static', StaticFiles(directory=RAIZ / 'static'), name='static')

    @servicio.get('/')
    def inicio():
        return FileResponse(RAIZ / 'static/index.html')

    @servicio.get('/api/modelo')
    def modelo():
        pred = servicio.state.predictor
        return pred.info if pred else dict(disponible=False, mensaje='Modelo pendiente de configurar. Puedes inspeccionar las imágenes.')

    async def cargar(request):
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
            return await run_in_threadpool(leer_paciente, cuerpo.get('files'))
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

    return servicio


predictor = None
if os.environ.get('BREAST_MODEL_DIR'):
    try:
        predictor = Predictor(os.environ['BREAST_MODEL_DIR'])
    except Exception:
        logging.exception('No se pudo cargar el modelo configurado; inferencia desactivada.')
app = crear_app(predictor)
