# V5: CNN de v2 a media resolución y con más orientaciones

## Punto de partida y revisión de imágenes

V2 normal sigue siendo la referencia: AUC interna por paciente de 0,6123.
V3 normal obtuvo 0,5496. V4 normal terminó con máximo 0,5583; la ponderada
tenía máximo 0,5633 en la última revisión, sin haber finalizado normalmente.
Los intentos anteriores no han demostrado una mejora sobre v2.

Antes de definir v5 se revisaron las tres fases de seis ejemplos de entrenamiento
y se midieron cortes centrales de 90 pacientes de train elegidas con semilla 42,
excluyendo fold 0 y test. Las imágenes muestran orientaciones diversas y detalle
fino/ruido. En esa muestra la media de intensidad conjunta varió aproximadamente
de 0,105 a 0,353. Esta exploración no demuestra qué rasgos predicen pCR ni
justifica localizar automáticamente un tumor. No se han utilizado anotaciones
clínicas, etiquetas de validación ni imágenes de test para definir el preprocesado.

Hipótesis: una representación a menor resolución, junto a aumentos de orientación,
puede reducir la dependencia del detalle fino y la orientación, con menor coste
de cómputo. **También puede perder información útil y empeorar el resultado.**
V5 es un experimento nuevo; no se ha demostrado que supere v2 ni que alcance 0,70.

## Arquitectura y entrada

`BreastCNNV5` conserva las capas entrenables de v2 y añade una media espacial 2×2
antes de la primera convolución. La entrada externa sigue siendo PRE/EARLY/LATE
de 256×256, divididas entre 255. La media se calcula independientemente en cada
fase: no mezcla canales, no recorta regiones y no selecciona lesiones.

| Etapa | Salida por corte |
|---|---|
| Entrada PRE/EARLY/LATE /255 | 3 × 256 × 256 |
| AvgPool 2×2, stride 2 | 3 × 128 × 128 |
| Dos Conv 3×3 + GroupNorm + ReLU, MaxPool, Dropout2d 0,1 | 16 × 64 × 64 |
| Mismo tipo de bloque | 32 × 32 × 32 |
| Mismo tipo de bloque | 64 × 16 × 16 |
| Mismo tipo de bloque | 128 × 8 × 8 |
| Media global y máximo global concatenados | 256 |
| Dropout 0,4, Linear 256→64, ReLU, Dropout 0,3, Linear 64→1 | Un logit |

Tiene **310.513 parámetros**, como v2. No son menos parámetros: son mapas
espaciales con la cuarta parte de posiciones en las convoluciones. Eso reduce
el trabajo de esas capas, pero no divide necesariamente por cuatro el tiempo
total: cargar PNG, aplicar aumentos y evaluar también cuestan tiempo.

Se entrena desde cero, sin cargar checkpoints ni pesos preentrenados. Se usa
GroupNorm como en v2; no se reintroduce la BatchNorm de v3. Para inferencia,
instanciar `BreastCNNV5` y proporcionar las imágenes originales /255: **no
reducirlas antes**, porque la reducción ya está dentro de `forward`. Los pesos
comparten nombres y tamaños con v2, por lo que es esencial respetar `modelo=v5`
en configuracion.json al cargarlos.

## Aumentos y entrenamiento

V5 añade un giro equiprobable de 0°, 90°, 180° o 270°, sin interpolación en ese
paso, antes del reflejo horizontal y del giro suave ±10° ya usados en v2.
Las tres fases reciben las mismas transformaciones. Solo se aplican a train;
validación e inferencia usan las imágenes originales y la reducción determinista.
No se tratan las fases como RGB ni se normalizan con estadísticas ImageNet.

Se vuelve a **BCE por corte**, como en v2. La predicción por paciente sigue
siendo la media de las probabilidades de TODOS sus cortes, nunca la media de
los logits. Se entrenan normal y ponderada por separado con la misma semilla.
El peso de clase se calcula exclusivamente con los cortes de train.

AdamW, lr=0,0001, weight decay=0,001, semilla 42, batch predeterminado 16.
Se evalúan train y validación completos cada época sin aumentos ni dropout;
se guardan pérdidas comparables y ambas AUC por paciente. El máximo es de
40 épocas por pérdida. No hay un cambio de tasa de aprendizaje inicial ni de
partición respecto a v2.

Hay dos cambios experimentales respecto a v2: resolución y aumentos. Si mejora,
no se puede atribuir la mejora a uno solo sin un control adicional. Para separar
los efectos, se pueden usar `--experimento v5 --no-rotaciones90` (solo resolución)
o `--experimento v5 --modelo v2` (solo nuevos giros). No se ejecutan automáticamente.

## Parada sin prolongar el entrenamiento por avances diminutos

En v5, paciencia=8 y min_delta=0,002. Son dos decisiones separadas:

- **Checkpoint:** se guarda siempre la mayor AUC observada, aunque el aumento
  sea diminuto. Si hay empate se conserva la primera época.
- **Paciencia:** solo se reinicia cuando la AUC supera en más de 0,002 la última
  referencia significativa. Los avances pequeños pueden acumularse: desde
  0,600, pasar por 0,601 y llegar a 0,603 sí reinicia el contador.

Ese 0,002 es una tolerancia práctica de parada, **no un criterio de significación
estadística**. La reducción del lr usa la misma tolerancia absoluta; factor 0,5
tras cuatro épocas sin superarla, con lr mínimo 0,000001.

La consola y el CSV muestran `epocas_sin_mejora` y `referencia_paciencia`.
Por ello, ya no se debe calcular la parada restando la época del mejor checkpoint
a la época actual. Si el contador llega a 8, se recupera el mejor checkpoint
absoluto y se pasa a la otra pérdida. Baseline/v2/v3/v4 mantienen min_delta=0
por defecto: sus comandos anteriores conservan el criterio original.

## Ejecutar desde VS Code en el Mac

Desde la raíz del proyecto, usando el Python con el que ya entrenas:

```bash
python3 -m src.training.entrenar --experimento v5 --perdida ambas --dispositivo cpu --batch-size 8
```

Primero normal, después ponderada: hasta 40 épocas cada una, con parada
anticipada independiente. Son pesos nuevos; el comando **no reanuda** modelos
anteriores. Los resultados se guardan en `models/<fecha_hora>_v5/`.
Batch 8 facilita usar el Mac; es distinto del batch 16 del v2 del laboratorio
y debe tenerse en cuenta al comparar. En el laboratorio se puede mantener 16.

Para una prueba breve de funcionamiento, sin valor como estimación de AUC:

```bash
python3 -m src.training.entrenar --experimento v5 --perdida ambas --prueba --epocas 2 --dispositivo cpu --batch-size 8
```

Antes de lanzarlo, detener la ejecución anterior si sigue activa, para no
competir por CPU y memoria. Mantener `caffeinate -i` en otra terminal si se
quiere evitar reposo por inactividad, con el Mac enchufado y la tapa abierta.
El entrenamiento completo sigue pudiendo durar horas; la prueba breve no mide
el tiempo exacto de una época completa.

En el laboratorio, después de sincronizar el código, con `cancer` activado:

```bash
python -m src.training.entrenar --experimento v5 --perdida ambas --dispositivo cuda --batch-size 16
```

No se añaden dependencias ni se necesita reinstalar PyTorch ROCm.

## Cómo decidir si sirve

Se mantienen las tres figuras principales: evolución por época de pérdida/AUC,
ROC y precisión–sensibilidad, y matrices de confusión. Normal y ponderada se
reúnen en las mismas figuras. El máximo AUC elegido, la sensibilidad con umbral
0,5 y la brecha train–validación deben interpretarse conjuntamente.

La primera comparación es con v2 (0,6123), usando validación por paciente.
Alcanzar 0,70 en fold 0 sería una observación interna, no una garantía en test.
Como ya se han probado varias variantes en fold 0, la elección puede adaptarse
demasiado a esa partición. La configuración fijada deberá contrastarse con
los otros cuatro folds y resumirse con media y desviación antes del test final.
No se evalúa test durante este entrenamiento. El umbral justificado, intervalos
de confianza y análisis por cohorte siguen pendientes.

## Verificación realizada

El 17 de octubre de 2026 pasaron las 37 pruebas automáticas del proyecto,
incluidas la alineación de fases tras girar, gradientes/checkpoint de v5 y
la separación entre mejor checkpoint y referencia de paciencia.
Se ejecutaron dos épocas por pérdida con cuatro pacientes de train y cuatro
de validación en modo prueba. Ambas terminaron y generaron sus checkpoints,
CSV y las tres figuras conjuntas. Esta ejecución no estima rendimiento real.

En una medición local de CPU con cuatro hilos, lote sintético de ocho cortes,
cinco repeticiones alternadas tras calentamiento y los mismos pesos iniciales,
la mediana de forward+BCE+backward fue **1,816 s en v2 y 0,413 s en v5**
(factor aproximado 4,4). No incluye lectura de PNG, aumentos ni validación,
no mide una época completa y puede variar con la carga y temperatura del Mac.
Las mediciones se guardaron localmente en
`models/diagnostico_v5/benchmark_cpu.json`. No se lanzó entrenamiento completo.
