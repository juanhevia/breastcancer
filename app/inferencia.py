"""Inferencia CPU por paciente; no entrena ni consulta etiquetas externas."""

import base64
import hashlib
import io
import json
import math
import re
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from src.models.modelo import BreastCNNV2, BreastCNNV7, BreastCNNV8, contar_parametros

FASES = ("PRE", "EARLY", "LATE")
NOMBRE = re.compile(r"(?P<paciente>[A-Za-z0-9_-]+)_z(?P<corte>[0-9]+)_(?P<fase>PRE|EARLY|LATE)\.png", re.I)
MAX_ARCHIVOS = 90
MAX_ARCHIVO = 512 * 1024
NORMALIZACION = "PRE, EARLY, LATE / 255"


def leer_paciente(archivos):
    """Valida PNG gris de 8 bits, correspondencia de fases y una sola paciente."""
    if not isinstance(archivos, list) or not 3 <= len(archivos) <= MAX_ARCHIVOS:
        raise ValueError("Carga entre 3 y 90 PNG: tres fases por corte, de una sola paciente.")
    grupos, pacientes = {}, set()
    for archivo in archivos:
        if not isinstance(archivo, dict):
            raise ValueError("Formato de archivo no válido.")
        nombre = archivo.get("name", "")
        match = NOMBRE.fullmatch(nombre) if isinstance(nombre, str) else None
        if not match:
            raise ValueError("Nombres esperados: PACIENTE_z012_PRE.png, _EARLY.png y _LATE.png.")
        contenido = archivo.get("content")
        if not isinstance(contenido, str) or len(contenido) > 4 * ((MAX_ARCHIVO + 2) // 3):
            raise ValueError("Cada PNG debe ocupar como máximo 512 KiB.")
        try:
            datos = base64.b64decode(contenido, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Contenido de archivo no válido.") from exc
        if not datos or len(datos) > MAX_ARCHIVO:
            raise ValueError("Archivo vacío o mayor de 512 KiB.")
        try:
            with Image.open(io.BytesIO(datos)) as png:
                if png.format != "PNG" or png.size != (256, 256) or png.mode != "L" or getattr(png, "n_frames", 1) != 1:
                    raise ValueError("Se requieren PNG estáticos de 256 × 256, en gris de 8 bits.")
                png.verify()
            with Image.open(io.BytesIO(datos)) as png:
                # Mismo orden, tipo y escala que utils_caso.cargar_imagen.
                matriz = np.asarray(png.convert("L"), dtype=np.float32) / 255.0
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ValueError("Hay un PNG corrupto o ilegible.") from exc
        if not np.isfinite(matriz).all():
            raise ValueError("La imagen contiene valores no finitos.")
        paciente, corte, fase = match['paciente'], int(match['corte']), match['fase'].upper()
        pacientes.add(paciente)
        if len(pacientes) != 1:
            raise ValueError("Selecciona archivos de una sola paciente.")
        grupo = grupos.setdefault(corte, {})
        if fase in grupo:
            raise ValueError(f"Fase {fase} duplicada en el corte {corte}.")
        grupo[fase] = matriz
    for corte, grupo in grupos.items():
        if set(grupo) != set(FASES):
            raise ValueError(f"Faltan fases en el corte {corte}: {', '.join(sorted(set(FASES) - set(grupo)))}.")
    indices = sorted(grupos)
    tensor = torch.from_numpy(np.stack([np.stack([grupos[c][f] for f in FASES]) for c in indices]))
    return dict(paciente=next(iter(pacientes)), cortes=indices, tensor=tensor)


def vista_previa(paciente):
    def png_url(matriz):
        memoria = io.BytesIO()
        Image.fromarray(matriz).save(memoria, format="PNG")
        return "data:image/png;base64," + base64.b64encode(memoria.getvalue()).decode()

    vistas = []
    for indice, tensor in zip(paciente['cortes'], paciente['tensor'].numpy()):
        fases = {f.lower(): png_url(np.rint(m * 255).astype(np.uint8)) for f, m in zip(FASES, tensor)}
        # Escala fija [-1,1], azul negativo, blanco cero, rojo positivo. Solo visualización.
        realce = tensor[1] - tensor[0]
        rgb = np.ones((256, 256, 3), dtype=np.float32)
        rgb[:, :, 0] -= np.maximum(-realce, 0)
        rgb[:, :, 1] -= np.abs(realce)
        rgb[:, :, 2] -= np.maximum(realce, 0)
        vistas.append(dict(corte=indice, **fases, realce=png_url(np.rint(rgb * 255).astype(np.uint8))))
    return dict(paciente=paciente['paciente'], numero_cortes=len(vistas), vistas=vistas)


class Predictor:
    def __init__(self, carpeta):
        carpeta = Path(carpeta)
        config = json.loads((carpeta / 'configuracion.json').read_text())
        manifiesto = json.loads((carpeta / 'app_manifest.json').read_text())
        if config.get('modelo') not in ('v2', 'v7', 'v8'):
            raise ValueError("La demo admite los candidatos v2, v7 y v8.")
        if config.get('agregacion') != 'mean' or not config.get('normalizacion', '').startswith(NORMALIZACION):
            raise ValueError("El modelo no declara la normalización y agregación compatibles.")
        umbral = manifiesto.get('umbral')
        if isinstance(umbral, bool) or not isinstance(umbral, (int, float)) or not math.isfinite(umbral) or not 0 <= umbral <= 1:
            raise ValueError("Falta un umbral válido en app_manifest.json.")
        if manifiesto.get('origen_umbral') != 'validacion_interna' or not manifiesto.get('criterio_umbral'):
            raise ValueError("Documenta el criterio del umbral, elegido en validación interna.")
        pesos = carpeta / 'mejor_modelo.pt'
        checksum = hashlib.sha256(pesos.read_bytes()).hexdigest()
        if checksum != manifiesto.get('sha256'):
            raise ValueError("El checksum no corresponde a los pesos declarados.")
        modelo = {'v2': BreastCNNV2, 'v7': BreastCNNV7, 'v8': BreastCNNV8}[config['modelo']]()
        # Solo pesos suministrados por el autor, nunca archivos de pesos subidos por visitantes.
        estado = torch.load(pesos, map_location='cpu', weights_only=True)
        modelo.load_state_dict(estado, strict=True)
        if any(not torch.isfinite(t).all() for t in modelo.state_dict().values()):
            raise ValueError("Los pesos contienen valores no finitos.")
        self.modelo, self.umbral = modelo.eval(), float(umbral)
        self.lock = threading.Lock()
        metricas = json.loads((carpeta / 'metricas.json').read_text())
        ajustadas = carpeta / 'metricas_internas_umbral.json'
        if ajustadas.is_file():
            internas = json.loads(ajustadas.read_text())
            if internas.get('umbral') != self.umbral or internas.get('auc') != metricas.get('auc'):
                raise ValueError('Las métricas internas no corresponden al umbral y modelo del paquete.')
            metricas.update(internas)
        self.info = dict(disponible=True, modelo=config['modelo'], sha256=checksum, umbral=self.umbral,
                         estado_modelo=manifiesto.get('estado_modelo', 'Candidato provisional'),
                         parametros=contar_parametros(modelo)[0],
                         criterio_umbral=manifiesto['criterio_umbral'], dispositivo='CPU',
                         agregacion='Media de probabilidades de los cortes cargados',
                         prueba=bool(config.get('prueba', False)),
                         entrenamiento={k: config.get(k) for k in ('perdida', 'epocas', 'paciencia', 'lr', 'batch_size', 'semilla')},
                         metricas_internas={k: metricas.get(k) for k in ('auc', 'mejor_epoca', 'pacientes', 'sensibilidad', 'especificidad', 'umbral')})
        # El umbral de las métricas históricas puede ser distinto al desplegado: se conserva explícito.
        json.dumps(self.info, allow_nan=False)

    def predecir(self, paciente):
        inicio = time.perf_counter()
        with self.lock, torch.inference_mode():
            self.modelo.eval()
            probs = torch.cat([torch.sigmoid(self.modelo(x)) for x in paciente['tensor'].split(8)])
        if not torch.isfinite(probs).all():
            raise ValueError("El modelo produjo valores no finitos; no se muestra una predicción.")
        valores = probs.numpy().astype(np.float64)
        probabilidad = float(valores.mean())
        return dict(paciente=paciente['paciente'], numero_cortes=len(valores), probabilidad=probabilidad,
                    clase=int(probabilidad >= self.umbral), umbral=self.umbral, sha256=self.info['sha256'],
                    dispositivo='CPU', segundos=time.perf_counter() - inicio,
                    alcance='Un solo corte; no representa todos los cortes de la paciente.' if len(valores) == 1 else
                            'Media de los cortes cargados. Comprueba que has incluido todos los disponibles.',
                    cortes=[dict(corte=c, probabilidad=float(p)) for c,p in zip(paciente['cortes'], valores)])
