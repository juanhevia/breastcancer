"""Arranque de la demo en Render: python -m app.render."""
import os
from pathlib import Path


def main():
    import asyncio
    # Un corte cada vez y un proceso: reduce memoria sin cambiar pesos ni agregación.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('BREAST_INFERENCE_BATCH', '1')
    os.environ.setdefault('BREAST_MODEL_DIR', str(Path(__file__).resolve().parent / 'paquetes/v8'))
    import torch
    import uvicorn
    torch.set_num_threads(1)
    from app.servidor import app
    if app.state.predictor is None or app.state.complemento is None:
        raise RuntimeError('No se ha cargado el paquete v8 y su complemento. Revisa los archivos del despliegue.')
    analisis = asyncio.Semaphore(1)

    @app.middleware('http')
    async def limitar_analisis(request, call_next):
        # Serializa también la lectura de imágenes para no multiplicar memoria.
        if request.method == 'POST':
            async with analisis:
                return await call_next(request)
        return await call_next(request)

    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '10000')),
                workers=1, limit_concurrency=8, timeout_keep_alive=2)


if __name__ == '__main__':
    main()
