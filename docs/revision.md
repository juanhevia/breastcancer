# Revisión de la arquitectura y el entrenamiento

## Arquitectura: correcta como baseline

La arquitectura original se conserva en `src/models/modelo.py`. Cumple la sección 7
del PDF: CNN 2D creada desde cero con PyTorch y un único logit, sin sigmoid dentro
del modelo ni pesos preentrenados.

| Etapa | Forma por muestra |
|---|---|
| Entrada PRE, EARLY, LATE | 3 × 256 × 256 |
| Conv 3→32 + BN + ReLU + MaxPool | 32 × 128 × 128 |
| Conv 32→64 + BN + ReLU + MaxPool | 64 × 64 × 64 |
| Conv 64→128 + BN + ReLU + MaxPool | 128 × 32 × 32 |
| Media espacial adaptativa | 128 × 1 × 1 |
| Flatten + Dropout(0.4) + Linear | 1 logit |

Tiene **93.825 parámetros entrenables**. `squeeze(1)` mantiene la dimensión de
lote incluso con una sola muestra. La media espacial reduce los parámetros de
la cabeza; dropout regulariza; BatchNorm normaliza activaciones intermedias.
La inicialización es la predeterminada de PyTorch y se controla con semilla.

Que sea correcta no demuestra capacidad predictiva. Tres bloques y media global
son un punto de partida pequeño; las curvas y la validación indicarán si hay
sobreajuste o falta de capacidad. No es necesario añadir otra arquitectura antes
de obtener un baseline verificable.

## Entrenamiento original: buena base, incompleto para el experimento

Ya estaban bien: BCEWithLogitsLoss sobre logits, Adam, modo train/eval,
validación sin gradientes, pérdida media ponderada por tamaño real de lote,
shuffle=False en validación, parada anticipada, restauración de los mejores
pesos y evaluación final agregada por paciente. `particion` separa por paciente
y excluye test; `pos_weight` se calcula solo con el entrenamiento interno.

Se ha completado:

- Comandos para elegir épocas, semilla, fold, lote, pérdida y dispositivo.
- Comparación de BCE normal y ponderada con las mismas condiciones iniciales,
  obligatoria en el PDF. Antes solo se ejecutaba la ponderada.
- AUC por paciente en cada época; la AUC por corte queda como referencia secundaria.
- Guardado del mejor modelo en cada mejora y del historial después de cada época.
- Configuración, partición, tiempos, predicciones, métricas y curvas en carpetas separadas.
- Comprobación del orden de validación, datos vacíos e hiperparámetros inválidos.
- AUC por corte indefinida si solo hay una clase, sin abortar el entrenamiento.
- Rutas de datos independientes de la ubicación nueva del código y workers=0 por defecto.

La pérdida ponderada usa N0/N1 sobre los **cortes de entrenamiento**, que son las
muestras usadas por BCE; se guarda ese valor. Es un peso de clase, no un mecanismo
para igualar la contribución de pacientes con diferente número de cortes.

## Lo que queda después

Cinco épocas son una primera observación, no una duración óptima demostrada.
Revisar curvas, sensibilidad, especificidad, matriz de confusión y AUC por
paciente; estudiar umbral y calibración usando validación interna; fijar las
decisiones antes de evaluar test. Documentar la justificación de los parámetros,
limitaciones y resultados reales. Si se usa GPU, completar el registro de memoria
máxima cuando el dispositivo no sea CUDA, ya que esta implementación solo mide
ese pico con CUDA.

Aún no se ha construido ni desplegado la web ni preparado la presentación.
Estos entregables están fuera de esta primera fase de organización y entrenamiento.

## Verificación realizada

- Cuatro pruebas automáticas superadas: forma y gradientes con batch de una
  muestra, separación de pacientes en los cinco folds, guardado/restauración
  de los mejores pesos y rechazo de cero épocas.
- Comprobación estática con Ruff superada y configuración/lock de Poetry válidos.
- Una época de prueba por cada pérdida, con 36 cortes de 4 pacientes para
  entrenar y 40 cortes de otras 4 pacientes para validar. Se guardaron los pesos,
  historiales y curvas en `models/20261007_170718_323243_prueba/`.
- No se han ejecutado épocas con el conjunto completo. La persona usuaria ha
  elegido lanzar por su cuenta el comando documentado en el README.

Los resultados de este subconjunto solo verifican que el programa funciona;
no justifican ninguna conclusión sobre rendimiento predictivo.
