# Diagnóstico del aprendizaje y siguiente comparación

## Resultados comprobados el 17 de octubre de 2026

Se revisaron los metadatos locales y las predicciones disponibles, y se realizó
una prueba de ajuste sobre ocho imágenes de entrenamiento. No se entrenó un
nuevo modelo sobre el conjunto completo ni se evaluaron imágenes de test.

### Integridad básica

- 1.273 pacientes y 12.703 cortes en el índice total.
- Sin sample_id duplicados ni pacientes con etiquetas, splits o folds contradictorios.
- pCR y split coinciden entre samples.csv y patients.csv.
- Sin pacientes compartidas entre entrenamiento y validación.
- Las rutas PRE/EARLY/LATE corresponden al sample_id, a la paciente y a la fase
  indicada: no se detectaron intercambios en los nombres de archivo.

Esto no confirma que todas las etiquetas clínicas sean correctas ni descarta
imágenes duplicadas de contenido: no se compararon hashes de píxeles.
Los campos clínicos no se incorporaron a la CNN. `dataset` se utilizó para
describir grupos y analizar resultados, no como predictor.

### Capacidad de aprender imágenes de entrenamiento

Se eligieron cuatro pacientes por clase, semilla 42, excluyendo fold 0 y test,
y un corte central por paciente. Con BreastCNNV2, lr=0,001, sin aumentos,
dropout ni weight decay:

| Paso de optimización | Pérdida en las mismas 8 imágenes | Aciertos |
|---|---:|---:|
| 0 | 0,6795 | 50 % |
| 10 | 0,2469 | 100 % |
| 20 | 0,0021 | 100 % |
| 30 | 0,000117 | 100 % |

La prueba duró unos 42 segundos de optimización/evaluación. Confirma que esta
arquitectura y este camino de carga/optimización pueden ajustar imágenes reales.
**No es validación, no mide generalización y su AUC=1 no es un logro del objetivo
0,70.** Al desactivar regularización y usar otro lr, tampoco permite atribuir
el problema del entrenamiento completo a un único hiperparámetro.

Artefactos locales: `models/20261017_142530_202686_diagnostico/`, con auditoría,
selección exacta de cortes, historial y resultado de memorización.

### Resultados por cohorte en validación interna

Se promediaron los cortes por paciente antes de calcular las métricas.

| Cohorte | Pacientes | Con pCR | AUC v3 normal | AUC v4 normal |
|---|---:|---:|---:|---:|
| Duke | 42 | 11 | 0,686 | 0,607 |
| I-SPY1 | 21 | 4 | 0,824 | 0,485 |
| I-SPY2 | 156 | 49 | 0,480 | 0,537 |
| Todas | 219 | 64 | 0,550 | 0,558 |

I-SPY2 contiene la mayor parte de las pacientes de validación y es un punto
débil de estos dos modelos. El AUC de I-SPY1 tiene muy pocos positivos; no se
debe presentar 0,824 como evidencia de un modelo general con AUC >0,70.
Estas diferencias no prueban que la cohorte sea la causa del error, y los AUC
de grupos pequeños tienen alta incertidumbre. No se han calculado intervalos
de confianza en esta auditoría.

V2 sigue siendo la mejor referencia global recibida (AUC 0,6123). Localmente
no están sus predicciones completas ni su checkpoint del laboratorio, por lo
que no se ha podido hacer la misma comparación por cohorte ni un análisis
pareado de errores contra v2. V5 se interrumpió antes de exportar las
predicciones de validación con la versión anterior del código; su historial
temprano no permite concluir cómo habría terminado.

## Qué se ha corregido en el entrenamiento

Antes se guardaban pesos e historial durante el entrenamiento, pero las
predicciones completas y metricas.json se exportaban al finalizar. Ahora se
guardan también al elegir cada nuevo mejor checkpoint. Si se pulsa Ctrl+C
después de una época completa, la CLI genera las tres figuras con los datos
guardados, escribe interrupcion.json y termina sin iniciar la siguiente pérdida.
El estado no se presenta como entrenamiento completo ni como reanudable.

Esta mejora se aplica a futuras ejecuciones; no reconstruye automáticamente
las predicciones que faltan de ejecuciones antiguas. Durante una interrupción
antes de la primera época completa todavía no existe informe de validación.

## Siguiente ensayo: v2_lr, una variable de optimización

Se conserva BreastCNNV2 a 256×256, GroupNorm, los aumentos originales ±10° y
reflejo, BCE por corte, evaluación por media de probabilidades por paciente,
semilla 42, AdamW y weight decay=0,001. **El cambio de optimización es lr:
0,0001 → 0,0003.** Se activa además el diagnóstico completo de train; no se usa
para seleccionar pesos ni para la parada y su DataLoader tiene un generador
separado del de entrenamiento.

La hipótesis es probar si un paso de optimización mayor ayuda al aprendizaje
con la configuración completa. La prueba de memorización solo motiva estudiar
la optimización: **no demuestra que lr=0,0003 sea mejor**. Este control no añade
una arquitectura nueva, no incorpora los giros de 90° de v5 ni el objetivo por
paciente de v4. No se ha ejecutado el entrenamiento completo de este ensayo.

Mantiene el máximo de 40 épocas, paciencia 12 y min_delta=0 de v2 para no
mezclar el cambio de lr con otro criterio de parada. Puede durar muchas horas
en CPU. La comparación preferible es en el mismo entorno y con batch 16,
igual que el v2 del laboratorio. Una mejora pequeña necesita confirmación;
no se reinterpretará como éxito simplemente por tener una época algo mayor.

Después de sincronizar el código, con el entorno `cancer` del laboratorio:

```bash
python -m src.training.entrenar --experimento v2_lr --perdida ambas --dispositivo cuda --batch-size 16
```

La carpeta se llamará `models/<fecha_hora>_v2_lr/`. El comando entrena normal y
después ponderada desde cero; no continúa checkpoints anteriores. Para una
prueba de funcionamiento, añadir `--prueba --epocas 2`. No reinstalar el
PyTorch ROCm que ya funciona en el laboratorio.

Si se decide entrenar en el Mac, se puede usar `python3` y `--dispositivo cpu`,
manteniendo batch 16 para comparar. No hay una mejora de velocidad como en v5:
la resolución vuelve a ser 256×256. No se ha lanzado ese entrenamiento aquí.

## Cómo evitar seguir cambiando modelos sin una comparación clara

1. Evaluar el control con los mismos criterios que v2: AUC por paciente,
   sensibilidad, especificidad y brecha train–validación. Revisar cohortes.
2. Si merece continuar, fijar la configuración y contrastarla con los otros
   folds internos; no elegir solo el fold que dé mejor resultado. Resumir los
   cinco con media y desviación. No se lanzan automáticamente desde esta guía.
3. Elegir y justificar el umbral con validación. Cambiarlo modifica sensibilidad
   y especificidad, pero no mejora el AUC de las puntuaciones.
4. Reservar test para la evaluación final. Probar repetidamente variantes en
   fold 0 puede adaptar las decisiones a esa validación.

El trabajo de referencia usa un ViT y describe además combinaciones con
variables clínicas; no es una reproducción de la CNN propia desde cero exigida
en este caso. Sus cifras no garantizan el resultado de nuestra configuración.
[Artículo original, método 2.5](https://arxiv.org/html/2506.12190v1).

## Repetir el diagnóstico

Solo metadatos y predicciones ya guardadas, sin entrenar:

```bash
python3 -m src.training.diagnostico
```

Añadir la prueba pequeña, independiente de los entrenamientos completos:

```bash
python3 -m src.training.diagnostico --memorizacion --modelo v2 --pasos 100 --max-segundos 240
```

Guarda resultados en una carpeta nueva `models/<fecha_hora>_diagnostico/`.
El límite de tiempo se comprueba entre pasos; no interrumpe una operación a
mitad. La prueba carga ocho imágenes en memoria y no guarda un modelo para uso
predictivo. No sustituye la evaluación completa ni la validación cruzada.
