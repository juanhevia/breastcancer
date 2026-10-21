# V4: aprender sobre la predicción de cada paciente

## Por qué este experimento

En fold 0, v2 normal alcanzó AUC interna 0,6123 y v3 normal 0,5496.
V3 presentó sobreajuste: en la última época, AUC de train 0,8685 frente a
0,4798 en validación. Más épocas de ese entrenamiento no solucionaron el problema.

V4 **reutiliza exactamente BreastCNNV2**, con 310.513 parámetros y GroupNorm,
pero entrena nuevos pesos desde cero con una pérdida por paciente. Es un nuevo
experimento de aprendizaje, no una arquitectura nueva ni una mejora demostrada.
La hipótesis es que exigir la etiqueta pCR a cada corte por separado puede ser
demasiado restrictivo: la etiqueta corresponde a la paciente, no a una anotación
independiente de cada corte. Ahora se optimiza la misma media que se evalúa.
Esta hipótesis no garantiza superar v2 ni alcanzar AUC 0,70.

## Qué cambia exactamente

Para una paciente con n cortes y logits z_i:

```text
p_paciente = media(sigmoid(z_1), ..., sigmoid(z_n))
loss_paciente = -w*y*log(p_paciente) - (1-y)*log(1-p_paciente)
loss_lote = media(loss_paciente de cada paciente del lote)
```

Cada paso usa **2 pacientes completas**, normalmente unos 20 cortes. Se baraja
el orden de las pacientes cada época. No se recortan, duplican ni muestrean sus
cortes: cada corte participa una vez por época. Las pacientes con más cortes
no reciben por ello más peso en la pérdida. El último lote también se utiliza.

La media es de **probabilidades**, no de logits. Se implementa mediante
log-sigmoid y logsumexp para calcular el logit de esa media sin recortar
probabilidades ni perder gradientes al saturar. Después se usa BCEWithLogitsLoss.
No se añade atención aprendida ni se cambia la agregación de evaluación.

- Normal: w=1.
- Ponderada: w=número de pacientes sin pCR / número con pCR, calculado solo
  en train. Con fold 0 completo es 620/258, aproximadamente 2,4031.
- Los cortes conservan PRE/EARLY/LATE /255 y los aumentos geométricos conjuntos
  de las tres fases. No se añaden variables clínicas ni preentrenamiento.
- Se mantienen AdamW, lr=0,0001, weight decay=0,001, semilla 42, máximo 40
  épocas por pérdida, paciencia 12 y selección por máxima AUC de validación
  por paciente. Se conserva la primera época en caso de empate.
- La evaluación de train y validación usa todos los cortes, sin aumentos ni
  dropout. Sus pérdidas también son por paciente; ambas AUC usan la media.

Cambiar la unidad de entrenamiento también cambia la composición del lote y
el número de pasos: con fold 0 y 2 pacientes por lote son 439 pasos por época,
frente a 548 lotes de 16 cortes en v2. Esto forma parte del nuevo protocolo;
una diferencia observada no puede atribuirse únicamente a la fórmula de la BCE.
No se modifican varias arquitecturas a la vez.

## Ejecutar desde Visual Studio Code en el Mac

En la terminal integrada, desde la raíz del proyecto, con el Python que ya
funciona. Primero, una prueba pequeña opcional (no estima rendimiento):

```bash
python3 -m src.training.entrenar --experimento v4 --perdida ambas --prueba --epocas 2 --dispositivo cpu --batch-size 8
```

En `--prueba` se escogen dos pacientes por clase: el peso de clase es 1,
por lo que normal y ponderada pueden coincidir exactamente. En el conjunto
completo, el desbalance de train produce un peso distinto de 1.

Entrenamiento completo:

```bash
python3 -m src.training.entrenar --experimento v4 --perdida ambas --dispositivo cpu --batch-size 8
```

Se ejecutan **normal y después ponderada**, cada una desde cero, hasta 40 épocas
por pérdida. Puede haber hasta 80 épocas en total. La parada anticipada de la
normal no cancela la ponderada: al acabar la primera comienza la segunda.
En CPU puede volver a tardar horas; v4 no promete ser más rápido que v3.

`--batch-size 8` controla los cortes por lote **durante evaluación** en v4.
El entrenamiento usa `--batch-pacientes 2`. Si falta memoria, reducirlo a 1
disminuye los cortes simultáneos, pero cambia el protocolo; registrarlo al comparar.
Cambiar solo batch-size no reduce la memoria del paso de entrenamiento por paciente.

Para evitar reposo por inactividad durante el entrenamiento se puede dejar
`caffeinate -i` activo en otra terminal integrada, con el Mac conectado y la tapa
abierta. La pantalla puede apagarse. Detener caffeinate con Ctrl+C al terminar.

## Ejecutar en el laboratorio

Después de sincronizar estos cambios del código con GitHub y actualizar la copia
del laboratorio, activar el entorno existente y mantener el PyTorch ROCm funcional:

```bash
conda activate cancer
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m src.training.entrenar --experimento v4 --perdida ambas --dispositivo cuda --batch-size 8
```

No se requieren nuevas dependencias. No reinstalar requirements.txt en el laboratorio.
Cada comando crea `models/<fecha_hora>_v4/` y empieza desde cero: no reanuda.

## Resultados y comparación

Se conservan las tres figuras principales: evolución de pérdida/AUC, ROC y
precisión–sensibilidad, y matrices de confusión. Al terminar normal se generan
con ella; al terminar ponderada se actualizan las mismas figuras con ambas.
La pérdida ahora se etiqueta **BCE por paciente**. No comparar directamente su
valor con la BCE por corte de v2/v3; la AUC por paciente sí usa la misma definición.

Los CSV contienen todas las épocas y las predicciones por corte y paciente.
El checkpoint contiene los pesos de BreastCNNV2; para cargarlo se instancia
esa clase. En configuracion.json figura `experimento=v4`, `modelo=v2`,
`unidad_entrenamiento=paciente`, el objetivo y la unidad del peso de clase.
La inferencia no requiere etiquetas: CNN por corte, sigmoide y media por paciente.

V4 reserva test. Repetir ajustes sobre fold 0 puede sobreajustar las decisiones
a esa validación. Tras fijar la configuración hay que evaluarla también en
los otros folds (`--fold 1`, ..., `--fold 4`) y resumir media y desviación;
ningún comando de este documento lanza automáticamente los cinco folds.
Un AUC superior a 0,70 en fold 0 no equivale a 0,70 en test.
El umbral 0,5 es todavía una referencia; su selección justificada en validación,
los intervalos de confianza y los análisis por cohorte siguen pendientes.

Para un control con la arquitectura y el protocolo por corte de v2, incluyendo
la evaluación completa de train y su lote original de 16 cortes:

```bash
python3 -m src.training.entrenar --experimento v2 --evaluar-train --perdida ambas --batch-size 16
```

No ejecutar ese control automáticamente: supone otro entrenamiento completo.
