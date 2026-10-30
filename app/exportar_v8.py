"""Publica evidencia agregada de resultados guardados; no entrena ni infiere.

python -m app.exportar_v8 --origen models/importados_entrega_v8/models
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]


def graficar_auc(hist, epoca, destino):
    """Escala completa del AUC; conserva los valores registrados por paciente."""
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(8, 3.6), layout='constrained')
        ax.plot(hist.epoca, hist.train_auc_paciente, label='Entrenamiento', color='#185c8a', lw=2)
        ax.plot(hist.epoca, hist.val_auc_paciente, label='Validación interna', color='#c96525', lw=2)
        ax.axhline(.5, color='#9a9a9a', ls='--', lw=1, label='Referencia AUC = 0,5')
        ax.axvline(epoca, ls=':', color='#666666', label=f'Época elegida {epoca}')
        ax.set(title='V8 · AUC por época', xlabel='Época', ylabel='ROC AUC por paciente',
               xlim=(1, len(hist)), ylim=(0, 1))
        ax.set_xticks([1, 3, 6, 9, 12, 15])
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=9, loc='lower right')
        fig.savefig(destino / 'auc.png', dpi=180)
        plt.close(fig)


def graficar_perdida(hist, epoca, destino):
    """Escala 0,50–0,70 para el historial final v8, sin alterar las pérdidas."""
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(8, 3.6), layout='constrained')
        ax.plot(hist.epoca, hist.train_eval_loss, label='Entrenamiento', color='#185c8a', lw=2)
        ax.plot(hist.epoca, hist.val_loss, label='Validación interna', color='#c96525', lw=2)
        ax.axvline(epoca, ls=':', color='#666666', label=f'Época elegida {epoca}')
        # Amplía el rango si un historial futuro excede estos límites: nunca recorta datos.
        minimo = min(.50, float(hist[['train_eval_loss', 'val_loss']].min().min()) - .01)
        maximo = max(.70, float(hist[['train_eval_loss', 'val_loss']].max().max()) + .01)
        ax.set(title='V8 · Pérdida por época', xlabel='Época', ylabel='Pérdida BCE por corte',
               xlim=(1, len(hist)), ylim=(minimo, maximo))
        ax.set_xticks([1, 3, 6, 9, 12, 15])
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=9, loc='upper right')
        fig.savefig(destino / 'perdida.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origen', type=Path, default=RAIZ / 'models')
    args = parser.parse_args()
    origen = args.origen.resolve()
    final, test = origen / 'final_train_v8/normal', origen / 'test_final_v8'
    leer = lambda p: json.loads(p.read_text())
    protocolo = leer(test / 'protocolo_fijado.json')
    seleccion = leer(final / 'seleccion_congelada.json')
    metricas = leer(test / 'metricas_test.json')
    internas = leer(final / 'metricas_internas_umbral.json')
    sha = hashlib.sha256((final / 'mejor_modelo.pt').read_bytes()).hexdigest()
    assert sha == protocolo['sha256'] and seleccion == protocolo['seleccion']
    assert seleccion['umbral'] == metricas['umbral'] == internas['umbral']
    particion = pd.read_csv(origen / 'final_train_v8/particion_pacientes.csv')
    assert not particion.patient_id.duplicated().any()
    train = set(particion.loc[particion.uso.eq('train'), 'patient_id'])
    val = set(particion.loc[particion.uso.eq('seleccion_interna'), 'patient_id'])
    assert len(train) == 878 and len(val) == 219 and not train & val
    preds = pd.read_csv(test / 'predicciones_pacientes.csv')
    assert len(preds) == preds.patient_id.nunique() == 176
    assert not (train | val) & set(preds.patient_id)
    for carpeta, esperado, ids in [(final, internas, val), (test, metricas, set(preds.patient_id))]:
        cortes = pd.read_csv(carpeta / ('predicciones_validacion.csv' if carpeta == final else 'predicciones_cortes.csv'))
        assert not cortes.sample_id.duplicated().any()
        assert cortes.groupby('patient_id').pCR.nunique().eq(1).all()
        pacientes = cortes.groupby('patient_id').agg(pCR=('pCR', 'first'), probabilidad=('probabilidad', 'mean'))
        assert set(pacientes.index) == ids
        y, p = pacientes.pCR, pacientes.probabilidad
        pred = p.ge(seleccion['umbral'])
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        assert dict(VP=int(tp), VN=int(tn), FP=int(fp), FN=int(fn)) == esperado['matriz_confusion']
        assert abs(roc_auc_score(y, p) - esperado['auc']) < 1e-9
        assert abs(f1_score(y, pred, average='macro') - esperado['f1_macro']) < 1e-9
        if carpeta == test:
            guardadas = preds.set_index('patient_id').loc[pacientes.index]
            np.testing.assert_allclose(p, guardadas.probabilidad, atol=1e-9, rtol=0)
            np.testing.assert_array_equal(y, guardadas.pCR)
            np.testing.assert_array_equal(pred, guardadas.prediccion)
    hist = pd.read_csv(final / 'historial.csv')
    epoca = seleccion['mejor_epoca']
    assert hist.loc[hist.val_auc_paciente.idxmax(), 'epoca'] == epoca
    assert len(hist) == seleccion['epocas_completadas'] == 15
    destino = RAIZ / 'app/static/evidencia/v8'
    destino.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    graficar_auc(hist, epoca, destino)
    graficar_perdida(hist, epoca, destino)
    comparaciones = []
    for ensayo in ('comparacion_v2_v7', 'cv_v7_lr3e4', 'cv_v8_realce'):
        df = pd.read_csv(origen / ensayo / 'resultados_por_fold.csv')
        for (modelo, perdida), grupo in df.groupby(['modelo', 'perdida'], sort=False):
            assert set(grupo.fold) == set(range(5)) and len(grupo) == 5
            comparaciones.append(dict(nombre=f'{modelo.upper()} {perdida}' + (' · lr 0,0003' if ensayo == 'cv_v7_lr3e4' else ''),
                media=float(grupo.auc.mean()), de=float(grupo.auc.std(ddof=1)), folds=grupo.auc.tolist()))
    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    for i, row in enumerate(comparaciones):
        ax.scatter(row['folds'], i + np.linspace(-.12, .12, 5), color='#7093ab', s=22)
        ax.errorbar(row['media'], i, xerr=row['de'], fmt='D', color='#185c8a', capsize=4)
    ax.set_yticks(range(len(comparaciones)), [r['nombre'] for r in comparaciones])
    ax.invert_yaxis()
    ax.axvline(.5, ls='--', color='#777777')
    ax.set(title='AUC en cinco folds de desarrollo', xlabel='ROC AUC por paciente', xlim=(.40, .70))
    ax.grid(axis='x', alpha=.18)
    fig.savefig(destino / 'comparacion_cv.png', dpi=180)
    plt.close(fig)
    for nombre in ('roc_test.png', 'matriz_confusion_test.png', 'calibracion_test.png'):
        shutil.copy2(test / nombre, destino / nombre)
    fuentes = [final / 'historial.csv', final / 'metricas_internas_umbral.json',
               test / 'metricas_test.json', test / 'metricas_cohortes.json', test / 'protocolo_fijado.json',
               origen / 'final_train_v8/particion_pacientes.csv']
    fuentes += [origen / n / 'resultados_por_fold.csv' for n in ('comparacion_v2_v7', 'cv_v7_lr3e4', 'cv_v8_realce')]
    salida = dict(modelo='v8', sha256=sha, mejor_epoca=epoca, umbral=seleccion['umbral'],
                  pacientes_entrenamiento=len(train), pacientes_validacion=len(val), test=metricas,
                  internas=internas, cohortes=leer(test / 'metricas_cohortes.json'), comparaciones=comparaciones,
                  particiones_disjuntas=True, auditoria='Métricas contrastadas con predicciones guardadas; sin repetir inferencia de test.',
                  historial=hist[['epoca', 'train_eval_loss', 'val_loss', 'train_auc_paciente', 'val_auc_paciente']].to_dict('records'),
                  fuentes=[dict(archivo=str(p.relative_to(RAIZ)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in fuentes])
    (destino / 'datos.json').write_text(json.dumps(salida, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'Evidencia verificada y exportada: {destino}')


if __name__ == '__main__':
    main()
