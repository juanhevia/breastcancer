"""Reproduce la figura de dos errores, sin inferencia ni cambios al modelo.

Ejecutar desde la raíz tras importar entrega_v8.zip; requiere los PNG locales.
Criterio posterior al test: FP máximo y FN mínimo, corte central disponible.
Fuente: BreastDCEDL, adaptación docente, CC BY-NC 4.0.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
root=Path('models/importados_entrega_v8/models')
p=pd.read_csv(root/'test_final_v8/predicciones_pacientes.csv');samples=pd.read_csv('metadata/samples.csv')
fig,axs=plt.subplots(2,3,figsize=(8,5.5),layout='constrained')
casos=[]
for i,(tipo,mask,asc) in enumerate([('Falso positivo',(p.pCR==0)&(p.prediccion==1),False),('Falso negativo',(p.pCR==1)&(p.prediccion==0),True)]):
 r=p[mask].sort_values(['probabilidad','patient_id'],ascending=[asc,True]).iloc[0]
 grupo=samples[samples.patient_id==r.patient_id].sort_values('slice_index');c=grupo.iloc[len(grupo)//2]
 for j,fase in enumerate(['PRE','EARLY','LATE']):
  axs[i,j].imshow(np.asarray(Image.open(c[f'path_{fase.lower()}'])),cmap='gray',vmin=0,vmax=255)
  axs[i,j].set_xticks([]);axs[i,j].set_yticks([]);axs[i,j].set_title(fase,fontsize=11)
  if j==0: axs[i,j].set_ylabel(f'{tipo}\np = {r.probabilidad:.3f}',fontsize=10)
 casos.append(dict(tipo=tipo,probabilidad=float(r.probabilidad),clase_real=int(r.pCR),corte=int(c.slice_index),numero_cortes=len(grupo)))
fig.savefig('app/static/evidencia/v8/errores_test.png',dpi=160);plt.close(fig)
Path('app/static/evidencia/v8/casos.json').write_text(json.dumps({'seleccion':'FP de mayor puntuación y FN de menor puntuación; corte central disponible de cada paciente. Selección posterior al test solo descriptiva, sin modificar modelo.', 'casos':casos},ensure_ascii=False,indent=2))
