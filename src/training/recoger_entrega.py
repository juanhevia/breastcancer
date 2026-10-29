"""Recoge resultados existentes y prepara la app. No entrena ni evalúa imágenes."""
import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raiz', type=Path, default=Path(__file__).resolve().parents[2])
    args = p.parse_args()
    raiz = args.raiz.resolve()
    origen = raiz / 'models/final_train_v8/normal'
    test = raiz / 'models/test_final_v8'
    paquete = raiz / 'models/app_modelo_v8'
    archivo = raiz / 'models/entrega_v8.zip'
    requeridos = [origen / n for n in ('mejor_modelo.pt', 'configuracion.json',
        'metricas.json', 'historial.csv', 'seleccion_congelada.json', 'metricas_internas_umbral.json')]
    requeridos += [test / n for n in ('protocolo_fijado.json', 'metricas_test.json',
        'metricas_cohortes.json', 'predicciones_pacientes.csv', 'predicciones_cortes.csv',
        'roc_test.png', 'matriz_confusion_test.png', 'calibracion_test.png')]
    for ruta in requeridos:
        if not ruta.is_file():
            p.error(f'Falta {ruta}. No repitas el test: comparte el error para recuperar los resultados.')
    leer = lambda ruta: json.loads(ruta.read_text())
    seleccion = leer(origen / 'seleccion_congelada.json')
    protocolo = leer(test / 'protocolo_fijado.json')
    config = leer(origen / 'configuracion.json')
    sha = hashlib.sha256((origen / 'mejor_modelo.pt').read_bytes()).hexdigest()
    if (config.get('modelo') != 'v8' or config.get('modo') != 'final_train' or
            protocolo.get('sha256') != sha or protocolo.get('seleccion') != seleccion or
            protocolo.get('umbral') != seleccion['umbral']):
        p.error('El checkpoint o la selección no coinciden con los usados en test.')
    if archivo.exists() or paquete.exists():
        p.error('El paquete o el ZIP ya existen. Se conservan; no se sobrescriben.')
    manifiesto = dict(umbral=seleccion['umbral'], origen_umbral='validacion_interna',
        criterio_umbral=seleccion['criterio'], sha256=sha,
        estado_modelo='V8 seleccionado · evaluación final realizada')
    # Construir en temporal evita paquetes parciales si falla una copia.
    with tempfile.TemporaryDirectory(dir=raiz / 'models') as tmp:
        carpeta = Path(tmp) / 'app_modelo_v8'
        carpeta.mkdir()
        for nombre in ('mejor_modelo.pt', 'configuracion.json', 'metricas.json'):
            shutil.copy2(origen / nombre, carpeta / nombre)
        (carpeta / 'app_manifest.json').write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2))
        carpeta.rename(paquete)
    fuentes = [paquete, origen.parent, test]
    with zipfile.ZipFile(archivo, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for fuente in fuentes:
            for ruta in sorted(fuente.rglob('*')):
                if ruta.is_file() and 'epocas' not in ruta.relative_to(fuente).parts and (
                        ruta.suffix in ('.csv', '.json', '.png', '.md') or ruta.name == 'mejor_modelo.pt'):
                    z.write(ruta, ruta.relative_to(raiz))
        for ensayo in ('comparacion_v2_v7', 'cv_v7_lr3e4', 'cv_v8_realce'):
            for nombre in ('resultados_por_fold.csv', 'resumen.json', 'protocolo.json'):
                ruta = raiz / 'models' / ensayo / nombre
                if ruta.is_file():
                    z.write(ruta, ruta.relative_to(raiz))
    print(f'ZIP para adjuntar: {archivo}')
    print(f'Paquete para la app: {paquete}')
    print('No se ha ejecutado entrenamiento ni inferencia. No contiene imágenes del dataset.')


if __name__ == '__main__':
    main()
