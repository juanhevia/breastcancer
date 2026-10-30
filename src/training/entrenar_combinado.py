"""Ajusta dos regresiones logísticas pequeñas con predicciones YA guardadas.

No entrena la CNN ni abre imágenes o resultados del test. --solo-plan no ajusta.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.combinado import SUBTIPOS, matriz, probabilidad_json
from src.training.comparar_cv import elegir_umbral, metricas, tabla_pacientes


def preparar(origen, metadata):
    config = json.loads((origen / 'configuracion.json').read_text())
    if config.get('modelo') != 'v8' or config.get('modo') != 'final_train' or config.get('prueba'):
        raise ValueError('Se requiere el entrenamiento final v8, no una prueba o un fold CV.')
    # Se filtra desarrollo inmediatamente; nunca se usa una etiqueta de test.
    clinica = pd.read_csv(metadata)
    clinica = clinica.loc[clinica.split.eq('train'), ['pid', 'pCR', 'age', 'HR_HER2_STATUS']].rename(columns={'pid': 'patient_id'})
    if clinica.patient_id.duplicated().any():
        raise ValueError('Identificadores duplicados en desarrollo.')
    part = pd.read_csv(origen.parent / 'particion_pacientes.csv')
    if part.patient_id.duplicated().any():
        raise ValueError('Solapamiento en la partición.')
    tablas = []
    for nombre, uso in [('entrenamiento', 'train'), ('validacion', 'seleccion_interna')]:
        cortes = pd.read_csv(origen / f'predicciones_{nombre}.csv')
        if cortes.sample_id.duplicated().any() or cortes.groupby('patient_id').pCR.nunique().ne(1).any():
            raise ValueError('Predicciones duplicadas o etiquetas inconsistentes.')
        t = tabla_pacientes(cortes)
        if set(t.patient_id) != set(part.loc[part.uso.eq(uso), 'patient_id']):
            raise ValueError('Las predicciones no coinciden con la partición.')
        t = t.merge(clinica, on='patient_id', how='left', validate='one_to_one', suffixes=('', '_meta'), indicator=True)
        if not t['_merge'].eq('both').all() or not t.pCR.eq(t.pCR_meta).all():
            raise ValueError('Una paciente no pertenece a desarrollo o su etiqueta no coincide.')
        t['age'] = pd.to_numeric(t.age, errors='raise')
        if not t.age.dropna().between(0, 120, inclusive='right').all():
            raise ValueError('Edad fuera de rango.')
        t['HR_HER2_STATUS'] = t.HR_HER2_STATUS.fillna('No disponible')
        if not t.HR_HER2_STATUS.isin(SUBTIPOS).all() or not np.isfinite(t.probabilidad).all() or not t.probabilidad.between(0, 1).all():
            raise ValueError('Subtipos o probabilidades inválidos.')
        tablas.append(t)
    if len(tablas[0]) != 878 or len(tablas[1]) != 219 or set(tablas[0].patient_id) & set(tablas[1].patient_id):
        raise ValueError('Se esperan 878/219 pacientes sin solapamiento.')
    return tablas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origen', type=Path, default=Path('models/final_train_v8/normal'))
    p.add_argument('--metadata', type=Path, default=Path('metadata/patients.csv'))
    p.add_argument('--salida', type=Path, default=Path('models/complemento_v8'))
    p.add_argument('--solo-plan', action='store_true')
    args = p.parse_args()
    train, val = preparar(args.origen, args.metadata)
    print(f'Complemento: {len(train)} entrenamiento; {len(val)} validación. Variables: edad y subtipo. Test excluido.')
    if args.solo_plan:
        return
    if args.salida.exists():
        p.error('La salida ya existe: no se sobrescribe.')
    mediana = float(train.age.median())
    if not np.isfinite(mediana):
        p.error('No hay edades observadas en entrenamiento.')
    sha = hashlib.sha256((args.origen / 'mejor_modelo.pt').read_bytes()).hexdigest()
    seleccion = json.loads((args.origen / 'seleccion_congelada.json').read_text())
    modelos, resultados = {}, {'solo_imagenes': metricas(val, seleccion['umbral'])}
    predicciones = val[['patient_id', 'pCR', 'probabilidad']].rename(columns={'probabilidad': 'solo_imagenes'})
    for nombre, con_imagen in [('solo_variables', False), ('combinado', True)]:
        matrices = [matriz(t.age, t.HR_HER2_STATUS, t.probabilidad, mediana, con_imagen) for t in (train, val)]
        scaler = StandardScaler().fit(matrices[0])
        modelo = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
        modelo.fit(scaler.transform(matrices[0]), train.pCR)
        if int(modelo.n_iter_.max()) >= 2000:
            raise ValueError('El ajuste no ha convergido; no se exporta.')
        m = dict(media=scaler.mean_.tolist(), escala=scaler.scale_.tolist(), coeficientes=modelo.coef_[0].tolist(), intercepto=float(modelo.intercept_[0]))
        probs = probabilidad_json(m, matrices[1])
        np.testing.assert_allclose(probs, modelo.predict_proba(scaler.transform(matrices[1]))[:, 1], atol=1e-12)
        evaluacion = val.assign(probabilidad=probs)
        m['umbral'] = elegir_umbral(evaluacion)
        modelos[nombre], resultados[nombre] = m, metricas(evaluacion, m['umbral'])
        predicciones[nombre] = probs
    salida = dict(version=1, sha256_cnn=sha, variables=['age', 'HR_HER2_STATUS'], mediana_edad=mediana,
                  modelos=modelos, metricas_internas=resultados, entrenamiento=878, validacion=219,
                  advertencia='Análisis exploratorio posterior al test v8. CNN congelada; sus puntuaciones de entrenamiento son internas a su ajuste. Validación compartida con la selección CNN; no estima rendimiento independiente.',
                  configuracion=dict(C=1., semilla=42, max_iter=2000, transformacion_imagen='logit recortado a [1e-6,1-1e-6]', imputacion='Mediana de entrenamiento más indicador de ausencia', umbral='Máximo F1 macro en validación interna'))
    args.salida.mkdir(parents=True)
    for nombre, objeto in [('combinado.json', salida), ('metricas_internas.json', resultados)]:
        (args.salida / nombre).write_text(json.dumps(objeto, ensure_ascii=False, indent=2, allow_nan=False))
    predicciones.to_csv(args.salida / 'predicciones_validacion.csv', index=False)
    pd.concat([train.assign(uso='train'), val.assign(uso='seleccion_interna')])[['patient_id', 'uso']].to_csv(args.salida / 'particion.csv', index=False)
    for nombre, m in resultados.items():
        print(f"{nombre}: AUC interna {m['auc']:.3f}; F1 macro {m['f1_macro']:.3f}")
    print(f'Paquete adicional: {args.salida / "combinado.json"}. CNN intacta. No se ha evaluado test.')


if __name__ == '__main__':
    main()
