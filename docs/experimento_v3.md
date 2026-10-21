# Experimento v3: cambios temporales y menor capacidad

## Hipótesis y punto de partida

V2 obtuvo AUC de validación por paciente de 0,6123 con BCE normal (época 24,
parada en 36) y 0,5662 con BCE ponderada (época 1, parada en 13). En la normal,
la pérdida de entrenamiento continuaba bajando mientras la de validación subía.
Este patrón motiva probar una CNN más pequeña, en vez de aumentar simplemente
su profundidad o las épocas. V3 es una hipótesis experimental: no se ha
demostrado que mejore v2 ni que alcance AUC 0,70.

## Arquitectura propia desde cero

La entrada externa sigue siendo **PRE, EARLY, LATE**, en ese orden, como tres
imágenes grises de 256 × 256 divididas entre 255. No se usan variables clínicas,
normalización ImageNet, modelos comerciales ni pesos preentrenados.

Dentro de la red, una transformación fija genera tres canales:

1. PRE: conserva la imagen inicial.
2. EARLY − PRE: cambio temprano de intensidad.
3. LATE − EARLY: cambio entre las fases posteriores.

No se divide entre PRE y no se recortan valores negativos. Así se evitan
divisiones por valores pequeños y se conserva el signo del cambio. Es un cambio
de representación que **no añade información nueva ni elimina las fases**:
EARLY y LATE pueden recuperarse sumando los canales. Una convolución podría
aprender estas diferencias por sí sola; aquí se hacen explícitas como hipótesis
para facilitar el aprendizaje. No son medidas físicas de concentración de contraste.

La transformación forma parte de `BreastCNNV3.forward`, por lo que también se
aplica al cargar los pesos para inferencia. La futura aplicación debe alimentar
las tres fases originales /255, sin calcular las diferencias una segunda vez.

| Etapa | Operaciones | Salida por corte |
|---|---|---|
| Representación temporal | PRE, EARLY−PRE, LATE−EARLY | 3 × 256 × 256 |
| Bloque 1 | Dos Conv 3×3 + BatchNorm + ReLU, MaxPool | 16 × 128 × 128 |
| Bloque 2 | Dos Conv 3×3 + BatchNorm + ReLU, MaxPool | 24 × 64 × 64 |
| Bloque 3 | Dos Conv 3×3 + BatchNorm + ReLU, MaxPool, Dropout2d 0,1 | 48 × 32 × 32 |
| Bloque 4 | Dos Conv 3×3 + BatchNorm + ReLU, MaxPool, Dropout2d 0,1 | 64 × 16 × 16 |
| Resumen espacial | Media adaptativa 2×2 | 64 × 2 × 2 |
| Cabeza | Flatten, Dropout 0,4, Linear 256→32, ReLU, Linear 32→1 | Un logit |

La media 2×2 conserva una distribución espacial aproximada, frente al resumen
global de v2. Se elimina el máximo global para probar un resumen menos sensible
a activaciones aisladas. BatchNorm usa estadísticas acumuladas exclusivamente
durante entrenamiento; en evaluación no las actualiza. Su dependencia del lote
es una limitación que hay que vigilar. La arquitectura es menor que v2 y mantiene
ocho convoluciones; no se añaden más bloques.

V3 tiene **115.857 parámetros entrenables**, frente a 310.513 de v2 (aproximadamente
un 63 % menos). Se conserva la inicialización predeterminada de PyTorch,
controlada por semilla. El número de parámetros se guarda en `configuracion.json`. Se mantienen
las clases originales para poder cargar los pesos de baseline y v2.

## Entrenamiento y comparación

Los valores predeterminados de v3 conservan los de v2: AdamW, lr=0,0001,
weight decay=0,001, batch=16, semilla=42, fold=0, aumentos alineados de las tres
fases, reducción de lr y selección por máxima AUC de validación por paciente.
Se agregan los cortes por **media**. Se entrenan pérdidas normal y ponderada
por separado, calculando el peso de clase solo con entrenamiento. Máximo de
40 épocas por pérdida; parada tras 12 épocas sin superar la mejor AUC.

Hay varios cambios de arquitectura a la vez. Una mejora no permitiría atribuir
el resultado a uno solo de ellos. Para comparar v2 y v3 con las mismas nuevas
mediciones, se puede ejecutar `--experimento v2 --evaluar-train`.

V3 evalúa todos los cortes de train al terminar cada época, sin aumentos ni
dropout, con los mismos pesos utilizados para validar. Se guarda:

- `train_loss`: pérdida durante la optimización, con regularización activa.
- `train_eval_loss`: pérdida de train en modo evaluación al final de la época.
- `train_auc_paciente`: AUC de train después de promediar cortes por paciente.
- `brecha_auc`: AUC de train menos AUC de validación.
- `predicciones_entrenamiento.csv`: predicciones de train del checkpoint elegido.
- En `metricas.json`: métricas de train y brecha correspondientes a la época
  elegida, además de las métricas de validación existentes.

La AUC de train es solo un diagnóstico: no interviene en selección, scheduler
ni parada. La figura de evolución muestra pérdidas comparables y ambas AUC;
se mantienen **tres figuras principales**. Las configuraciones antiguas siguen
funcionando. Esta pasada de evaluación añade tiempo a cada época; se puede
desactivar con `--no-evaluar-train`, aunque se perdería ese diagnóstico.

## Ejecutar en el laboratorio

Una vez sincronizados los cambios del código, desde la raíz del repositorio:

```bash
git pull --ff-only
conda activate cancer
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m src.training.entrenar --experimento v3 --perdida ambas --dispositivo cuda
```

No hay nuevas dependencias. Mantener la instalación funcional de PyTorch ROCm;
no reinstalar `requirements.txt`. `cuda` es el dispositivo que usa PyTorch
también para ROCm. Cada comando empieza con pesos nuevos y crea otra carpeta
`models/<fecha_hora>_v3/`; no reanuda ni sustituye v2.

Prueba de funcionamiento, sin valor como estimación de rendimiento:

```bash
python -m src.training.entrenar --experimento v3 --prueba --epocas 2 --batch-size 4 --dispositivo cuda
```

## Qué faltará para concluir

El primer experimento completo en fold 0 permite una comparación inicial. Una
AUC superior a 0,70 en la época elegida sería un resultado de validación interna,
no una garantía en test. Fijada la configuración, hacen falta los otros folds
internos (opción `--fold 1`, etc.) y el resumen de los cinco con media y desviación.
Este cambio no ejecuta automáticamente la validación cruzada.

El umbral 0,5 sigue siendo una referencia inicial, no un umbral clínico
justificado. La selección del umbral, análisis por cohortes, intervalos de
confianza y evaluación de errores son tareas pendientes. El test se reserva
para la evaluación final, con las decisiones ya fijadas. No se evalúa al ejecutar
este comando. Los resultados locales de `models/` siguen excluidos de Git.
