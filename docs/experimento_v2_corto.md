# V2 original con un límite de 15 épocas

## Por qué se fija este corte

Se usa el historial **normal** de `laboratorio_v2_recibido_20261014` (fold 0).
El corte orientativo es la época 15, última antes de una separación sostenida
entre la pérdida de entrenamiento y la de validación desde la época 16.

| Época histórica | Pérdida train | Pérdida validación | Diferencia val − train | AUC por paciente |
|---|---:|---:|---:|---:|
| 14 | 0,5969 | 0,5986 | 0,0017 | 0,5508 |
| 15 | 0,5970 | 0,5990 | 0,0021 | 0,5681 |
| 16 | 0,5945 | 0,6126 | 0,0181 | 0,5267 |
| 17 | 0,5929 | 0,6050 | 0,0121 | 0,5501 |
| 18 | 0,5853 | 0,6316 | 0,0463 | 0,5341 |
| 24 | 0,5579 | 0,6145 | 0,0566 | 0,6123 |

Son métricas del entrenamiento antiguo, **no resultados de una ejecución nueva**.
El corte no maximiza el AUC: la época 24 lo tenía mayor. Que las pérdidas estén
cerca no garantiza generalización ni demuestra por sí solo ausencia de sobreajuste.

La pérdida train antigua se medía durante la optimización, con aumentos y dropout,
y la de validación al final de la época en modo evaluación. Por tanto, no eran
medidas en condiciones idénticas. El nuevo ensayo añade `train_eval_loss` y AUC
train al final de cada época, sin aumentos ni dropout; son las curvas que se
compararán ahora. `train_loss` se conserva igualmente en el CSV.

El límite 15 es **fijo y orientativo**, elegido del historial disponible. No es
un algoritmo que detecte automáticamente dónde empiezan a separarse dos curvas
nuevas. No se borran ni se recortan los resultados históricos del entrenamiento
original: las gráficas de cada ejecución muestran todas sus épocas completadas.

## Configuración

Preset `v2_corto`, misma arquitectura `BreastCNNV2` (310.513 parámetros) e
hiperparámetros de optimización de v2: AdamW, lr 0,0001, weight decay 0,001,
aumentos geométricos conjuntos de las tres fases, selección por AUC de validación
por paciente, scheduler, semilla 42 y batch recomendado 16. Sin contraste de v6
ni bloques residuales de v7. Se entrena desde cero, no desde un `.pt` anterior.

Cambios: máximo **15 épocas**, paciencia **9** solicitada y diagnóstico train
en modo evaluación. Puede acabar antes de la 15 si acumula nueve épocas sin
mejorar el AUC. Se guardan todos los `.pt` de las épocas que se completen.
Las mismas semillas no garantizan idénticos valores entre versiones o equipos.

El corte se justifica para pérdida normal. La ponderada tuvo otra evolución;
no se deduce de este análisis que deba cortarse también en 15. El comando de
abajo selecciona explícitamente normal.

## Entrenar en el laboratorio

En la terminal integrada de VS Code, dentro del repositorio:

```bash
git pull --ff-only
conda activate cancer
python -m src.training.entrenar --experimento v2_corto --perdida normal --dispositivo cuda --batch-size 16
```

Conserva la instalación ROCm existente. Los resultados se crean en una carpeta
nueva `models/<fecha_hora>_v2_corto/`; no sustituyen el entrenamiento original.

```text
normal/
  configuracion.json
  historial.csv
  metricas.json
  mejor_modelo.pt
  epocas/
    epoca_001.pt
    epoca_002.pt
    ...
    epoca_015.pt    # solo si se completa la época 15
```

`mejor_modelo.pt` representa la mejor AUC dentro de esta ejecución corta y
puede corresponder a una época anterior a la última. Para usar exactamente
los pesos del corte, cargar `epocas/epoca_015.pt` si existe; consultar siempre
`historial.csv`. Los pesos de épocas antiguas nunca guardadas no se recuperan.

## Usar posteriormente imágenes de prueba

Cada `.pt` guarda un `state_dict` de BreastCNNV2, que se carga con:

```python
import torch
from src.models.modelo import BreastCNNV2

modelo = BreastCNNV2()
modelo.load_state_dict(torch.load(ruta_pt, map_location="cpu", weights_only=True))
modelo.eval()
```

La entrada debe conservar PRE/EARLY/LATE de cada corte, mismo orden, tamaño y
normalización /255. No usar aumentos en inferencia. Una paciente con varios
cortes se evalúa promediando sus probabilidades; un corte aislado no equivale
a la evaluación completa de esa paciente. Las puntuaciones no garantizan
probabilidades calibradas, y el umbral se fija con validación.

Se podrán comparar checkpoints usando **validación interna**. Si «imágenes de
prueba» se refiere al test reservado del caso, fijar primero modelo, época y
umbral, y evaluarlo una sola vez al final. Probar todas las épocas en test y
quedarse con la que mejor funcione convertiría test en otro conjunto de selección.
En esta preparación no se consulta test ni se lanza entrenamiento completo.

## Verificación de funcionamiento

Se superaron cinco pruebas dirigidas al preset, al guardado/restauración de pesos
y al contador de paciencia. Terminó una prueba integrada en CPU de dos épocas
con un subconjunto de cuatro pacientes de entrenamiento y cuatro de validación.
Se verificaron arquitectura v2, evaluación train sin aumentos, ambos `.pt`
recargables y las cinco figuras individuales PNG/PDF. Esa prueba pequeña no
estima el rendimiento del entrenamiento completo de 15 épocas.
