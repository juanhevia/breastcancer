"""Exporta curvas agregadas para la web, sin identificadores ni imágenes de pacientes.

Ejecutar desde la raíz: python -m app.exportar_evidencia.
La evidencia histórica v7 queda separada del modelo conectado a inferencia.
"""
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

RAIZ = Path(__file__).resolve().parents[1]


def main():
    destino = RAIZ / 'app/static/evidencia'
    destino.mkdir(parents=True, exist_ok=True)
    origen = RAIZ / 'models/importados_chat_20261027/v7_original/normal/predicciones_validacion.csv'
    filas = pd.read_csv(origen)
    if not filas.groupby('patient_id').pCR.nunique().eq(1).all():
        raise ValueError('Etiquetas inconsistentes por paciente.')
    pacientes = filas.groupby('patient_id').agg(pCR=('pCR', 'first'), probabilidad=('probabilidad', 'mean'))
    auc = roc_auc_score(pacientes.pCR, pacientes.probabilidad)
    fpr, tpr, _ = roc_curve(pacientes.pCR, pacientes.probabilidad)
    if len(pacientes) != 219 or abs(auc - .5793346774193548) > 1e-8:
        raise ValueError('La fuente no corresponde al v7 original documentado.')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
        ax.plot(fpr, tpr, color='#17697b', linewidth=2.5, label=f'V7 · AUC {auc:.3f}'.replace('.', ','))
        ax.plot([0, 1], [0, 1], '--', color='#8b99a7', label='Referencia aleatoria')
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Tasa de falsos positivos (1 − especificidad)',
               ylabel='Sensibilidad', title='ROC · validación interna de v7')
        ax.grid(alpha=.18)
        ax.legend(loc='lower right', frameon=False)
        fig.savefig(destino / 'roc_v7.png', dpi=160)
        plt.close(fig)
    fuentes = [origen]
    for nombre in ('auc_cv.png', 'auc_v7.png', 'perdida_v7.png'):
        fuente = RAIZ / 'docs/memoria/figuras' / nombre
        shutil.copy2(fuente, destino / nombre)
        fuentes.append(fuente)
    historial = RAIZ / 'models/importados_chat_20261023/v7/normal/historial.csv'
    history = pd.read_csv(historial)
    fuentes.append(historial)
    exportado = {
        'version': 1, 'fecha': '2026-10-28', 'modelo': 'v7',
        'alcance': 'Evidencia histórica de desarrollo; no es el test final ni cambia con el modelo conectado.',
        'auc_interna': float(auc), 'pacientes_validacion': len(pacientes),
        'mejor_epoca': 6,
        'historial': history[['epoca', 'train_eval_loss', 'val_loss', 'train_auc_paciente', 'val_auc_paciente']].to_dict('records'),
        'fuentes': [{'archivo': str(p.relative_to(RAIZ)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in fuentes],
    }
    (destino / 'datos.json').write_text(json.dumps(exportado, ensure_ascii=False, indent=2, allow_nan=False))
    print(f'Evidencia agregada exportada: {destino}')


if __name__ == '__main__':
    main()
