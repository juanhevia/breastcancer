"""Exporta ROC comparada y matriz interna desde predicciones guardadas; sin test."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origen', type=Path, default=Path('models/complemento_v8'))
    args = parser.parse_args()
    datos = pd.read_csv(args.origen / 'predicciones_validacion.csv')
    metricas = json.loads((args.origen / 'metricas_internas.json').read_text())
    assert len(datos) == 219 and not datos.patient_id.duplicated().any()
    destino = Path(__file__).resolve().parent / 'static/evidencia/v8'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(7, 4.4), layout='constrained')
    for nombre, etiqueta, color in [('solo_imagenes', 'Imágenes', '#185c8a'), ('solo_variables', 'Edad + subtipo', '#b67a24'), ('combinado', 'Imágenes + variables', '#147d73')]:
        auc = roc_auc_score(datos.pCR, datos[nombre])
        assert abs(auc - metricas[nombre]['auc']) < 1e-9
        fpr, tpr, _ = roc_curve(datos.pCR, datos[nombre])
        ax.plot(fpr, tpr, color=color, label=f'{etiqueta} · AUC {auc:.3f}', lw=2)
    ax.plot([0, 1], [0, 1], '--', color='#8d99a4', lw=1)
    ax.set(title='ROC · Validación interna (219 pacientes)', xlabel='1 − especificidad', ylabel='Sensibilidad para pCR', xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    fig.savefig(destino / 'roc_interna_comparada.png', dpi=180)
    plt.close(fig)
    m = metricas['solo_imagenes']
    cm = confusion_matrix(datos.pCR, datos.solo_imagenes.ge(m['umbral']), labels=[0, 1])
    esperado = m['matriz_confusion']
    assert cm.tolist() == [[esperado['VN'], esperado['FP']], [esperado['FN'], esperado['VP']]]
    fig, ax = plt.subplots(figsize=(5.5, 4.4), layout='constrained')
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=22, color='white' if cm[i, j] > cm.max()/2 else '#172e46')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Sin pCR', 'Con pCR'], yticklabels=['Sin pCR', 'Con pCR'], xlabel='Predicción', ylabel='Real', title='V8 · Matriz de validación interna')
    fig.savefig(destino / 'matriz_interna.png', dpi=180)
    plt.close(fig)
    print('ROC interna comparada y matriz interna verificadas. No se ha evaluado test.')


if __name__ == '__main__':
    main()
