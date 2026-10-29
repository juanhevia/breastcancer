"""Prepara un paquete local con pesos y umbral fijados en validación interna."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from app.inferencia import Predictor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origen', type=Path, required=True)
    p.add_argument('--destino', type=Path, default=Path('models/app_modelo'))
    p.add_argument('--umbral', type=float, required=True)
    p.add_argument('--criterio', required=True)
    args = p.parse_args()
    if not math.isfinite(args.umbral) or not 0 <= args.umbral <= 1 or not args.criterio.strip():
        p.error('Indica un umbral entre 0 y 1 y su criterio de validación interna.')
    nombres = ('mejor_modelo.pt', 'configuracion.json', 'metricas.json')
    for nombre in nombres:
        if not (args.origen / nombre).is_file():
            p.error(f'Falta {nombre} en la carpeta de origen.')
    if args.destino.exists():
        p.error('El destino ya existe; utiliza una carpeta nueva.')
    args.destino.mkdir(parents=True)
    for nombre in nombres:
        shutil.copy2(args.origen / nombre, args.destino / nombre)
    manifiesto = dict(umbral=args.umbral, origen_umbral='validacion_interna', criterio_umbral=args.criterio,
                     sha256=hashlib.sha256((args.destino / 'mejor_modelo.pt').read_bytes()).hexdigest())
    (args.destino / 'app_manifest.json').write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False))
    Predictor(args.destino)
    print(f'Paquete local verificado: {args.destino.resolve()}')
    print('Todavía no se ha publicado ni desplegado la aplicación.')


if __name__ == '__main__':
    main()
