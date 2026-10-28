"""Evalúa una vez el test reservado, con checkpoint y umbral internos ya fijados."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.data import utils_caso as uc
from src.training.comparar_cv import MODELOS, metricas, predecir, tabla_pacientes
from src.training.entrenar import guardar_json


def intervalo_auc(tabla, repeticiones=2000):
    """Bootstrap estratificado por paciente; IC percentil 95 %, no por corte."""
    rng = np.random.default_rng(42)
    negativos = np.flatnonzero(tabla.pCR.to_numpy() == 0)
    positivos = np.flatnonzero(tabla.pCR.to_numpy() == 1)
    if not len(negativos) or not len(positivos):
        return None
    y, prob = tabla.pCR.to_numpy(), tabla.probabilidad.to_numpy()
    valores = []
    for _ in range(repeticiones):
        indices = np.r_[rng.choice(negativos, len(negativos)), rng.choice(positivos, len(positivos))]
        valores.append(roc_auc_score(y[indices], prob[indices]))
    return [float(v) for v in np.quantile(valores, [.025, .975])]


def graficas(tabla, umbral, salida):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

    y, prob = tabla.pCR, tabla.probabilidad
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y, prob):.3f}', color='#17697b')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='1 − especificidad', ylabel='Sensibilidad', title='ROC · test por paciente')
    ax.legend()
    fig.savefig(salida / 'roc_test.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    ConfusionMatrixDisplay.from_predictions(y, prob >= umbral, labels=[0, 1],
        display_labels=['Sin pCR', 'Con pCR'], colorbar=False, cmap='Blues', ax=ax)
    ax.set(xlabel='Predicción', ylabel='Clase real', title='Matriz de confusión · test')
    fig.savefig(salida / 'matriz_confusion_test.png', dpi=160)
    plt.close(fig)
    real, estimada = calibration_curve(y, prob, n_bins=5, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    ax.plot(estimada, real, 'o-', color='#17697b', label='5 grupos por cuantiles')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Probabilidad media estimada',
           ylabel='Proporción observada con pCR', title='Calibración · test por paciente')
    ax.legend()
    fig.savefig(salida / 'calibracion_test.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origen', type=Path, required=True)
    p.add_argument('--salida', type=Path, required=True)
    p.add_argument('--datos', type=Path, default=uc.RAIZ)
    p.add_argument('--dispositivo', choices=['cpu', 'cuda', 'mps'], default='cuda')
    args = p.parse_args()
    if args.salida.exists():
        p.error('La salida ya existe: conserva la evaluación guardada, no repitas el test.')
    leer = lambda nombre: json.loads((args.origen / nombre).read_text())
    config, seleccion, internas = leer('configuracion.json'), leer('seleccion_congelada.json'), leer('metricas.json')
    if (config.get('modo') != 'final_train' or config.get('prueba') or
            config.get('agregacion') != 'mean' or
            config.get('normalizacion') != 'PRE, EARLY, LATE / 255' or
            seleccion['mejor_epoca'] != internas['mejor_epoca'] or
            seleccion['fold_interno'] != config['fold_interno']):
        p.error('Configuración o selección interna incompatibles.')
    umbral = seleccion['umbral']
    if isinstance(umbral, bool) or not isinstance(umbral, (int, float)) or not math.isfinite(umbral) or not 0 <= umbral <= 1:
        p.error('Umbral interno inválido.')
    if args.dispositivo == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA/ROCm no disponible.')
    torch.set_num_threads(config.get('hilos', 4))
    modelo = MODELOS[config['modelo']]()
    pesos = args.origen / 'mejor_modelo.pt'
    estado = torch.load(pesos, map_location='cpu', weights_only=True)
    epoca = torch.load(args.origen / 'epocas' / f"epoca_{seleccion['mejor_epoca']:03d}.pt",
                       map_location='cpu', weights_only=True)
    if estado.keys() != epoca.keys() or any(not torch.equal(estado[k], epoca[k]) for k in estado):
        p.error('El mejor checkpoint no coincide con la época seleccionada.')
    modelo.load_state_dict(estado, strict=True)
    if any(not torch.isfinite(t).all() for t in modelo.state_dict().values()):
        p.error('Pesos no finitos.')
    modelo.to(args.dispositivo).eval()
    samples = uc.cargar_samples(args.datos)
    test = samples.loc[samples.split == 'test'].reset_index(drop=True)
    particion = pd.read_csv(args.origen.parent / 'particion_pacientes.csv')
    ids_test = set(test.patient_id)
    if len(ids_test) != 176 or test.sample_id.duplicated().any() or test.groupby('patient_id').pCR.nunique().ne(1).any():
        p.error('Se esperaban 176 pacientes de test con metadatos coherentes.')
    if ids_test & (set(particion.patient_id) | set(samples.loc[samples.split == 'train', 'patient_id'])):
        p.error('Solapamiento entre test y desarrollo.')
    if not test.pCR.isin([0, 1]).all() or test.pCR.nunique() != 2:
        p.error('Etiquetas de test inválidas.')
    args.salida.mkdir(parents=True, exist_ok=False)
    guardar_json(args.salida / 'protocolo_fijado.json', dict(
        modelo=config['modelo'], seleccion=seleccion, umbral=umbral,
        sha256=hashlib.sha256(pesos.read_bytes()).hexdigest(),
        agregacion='mean', normalizacion=config['normalizacion'], pacientes_test=176,
        dispositivo=args.dispositivo, torch=str(torch.__version__),
        test_metadata_sha256=hashlib.sha256(test.to_csv(index=False).encode()).hexdigest()))
    print('Evaluando test con modelo y umbral fijados; no se entrena ni se ajusta el umbral.', flush=True)
    pred = predecir(modelo, test, args.datos, args.dispositivo, config.get('batch_size', 16), 0)
    pred.to_csv(args.salida / 'predicciones_cortes.csv', index=False)
    tabla = tabla_pacientes(pred)
    tabla['prediccion'] = (tabla.probabilidad >= umbral).astype(int)
    tabla['error'] = tabla.prediccion != tabla.pCR
    tabla.to_csv(args.salida / 'predicciones_pacientes.csv', index=False)
    resultado = metricas(pred, umbral)
    resultado.update(auc_ic95=intervalo_auc(tabla),
        metodo_ic='Bootstrap estratificado por paciente, 2000 remuestreos, semilla 42',
        brier=float(brier_score_loss(tabla.pCR, tabla.probabilidad)),
        siempre_sin_pcr=metricas(pred.assign(probabilidad=0.), .5))
    guardar_json(args.salida / 'metricas_test.json', resultado)
    # La cohorte solo se usa para describir resultados, nunca como entrada del modelo.
    cohortes = pd.read_csv(args.datos / 'metadata/patients.csv', usecols=['pid', 'dataset']).rename(columns={'pid':'patient_id'})
    union = tabla.merge(cohortes, on='patient_id', how='left', validate='one_to_one')
    union['dataset'] = union['dataset'].fillna('Sin cohorte')
    guardar_json(args.salida / 'metricas_cohortes.json', {
        str(nombre): metricas(grupo, umbral) for nombre, grupo in union.groupby('dataset')})
    graficas(tabla, umbral, args.salida)
    print(f"Test terminado: AUC={resultado['auc']:.3f}; resultados en {args.salida}")


if __name__ == '__main__':
    main()
