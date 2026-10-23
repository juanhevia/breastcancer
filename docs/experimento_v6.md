# V6: fases originales y una vista auxiliar con contraste conjunto

## Motivo

El usuario comunica un mejor AUC de aproximadamente 0,58 en el entrenamiento
v2_lr del laboratorio. No se han recibido aquí su historial completo ni sus
predicciones, por lo que no se atribuye ese resultado a una causa concreta.
La referencia v2 recibida conserva AUC 0,6123 en validación interna, fold 0.

V6 prueba la propuesta de ajustar intensidad/contraste. No se afirma que sea
mejor que v2 ni que vaya a llegar a 0,70 antes de entrenar y validarlo.

## Tratamiento de las imágenes

La entrada externa sigue siendo PRE, EARLY y LATE en escala de grises,
256×256, divididas por 255. No se modifican los PNG originales.

Dentro del modelo se crea una segunda vista de las tres fases:

1. Reunir los píxeles positivos de las tres fases de **un mismo corte**.
   Excluir ceros sirve para evitar el fondo negro; no identifica una lesión.
2. Obtener sus percentiles 5 y 95, comunes a PRE/EARLY/LATE.
3. Aplicar la misma transformación a las tres fases:
   `clip((x - p5) / max(p95 - p5, 1/3), 0, 1)`.
4. Si el corte es vacío o constante, conservarlo sin cambios.
5. Concatenar las fases originales y las fases ajustadas: seis canales internos.

La ganancia está limitada a 3. El desplazamiento reduce el nivel de los tonos
más bajos y el escalado amplía el contraste. El recorte puede perder detalle
en las colas de la vista auxiliar; por eso se conservan también los originales.
Antes del recorte, las diferencias entre fases se multiplican por el mismo
factor. No se promete preservar exactamente diferencias en píxeles saturados.
No se trata cada fase como un color RGB ni se utiliza saturación de color.

Una inspección de nueve cortes de entrenamiento (tres por cohorte, semilla 42)
mostró p99 positivo de aproximadamente 0,98–1,00: una ventana p1–p99 cambiaría
poco esas imágenes. Sus p95 fueron aproximadamente 0,55–0,69. Se eligió p5–p95
para que la vista auxiliar produzca un cambio medible. Son ejemplos de train,
no una estadística representativa de todo el conjunto ni evidencia de mejora.

El preprocesado está dentro de `BreastCNNV6`, también al cargar un `.pt` para
inferencia. No depende de otras imágenes del lote, no aprende estadísticas
sobre validación/test y no necesita etiquetas. Los percentiles y el límite de
ganancia están registrados como buffers en el `state_dict`.

## Arquitectura y protocolo

Se mantiene el cuerpo de v2: cuatro bloques convolucionales de 16/32/64/128
canales, GroupNorm, ReLU, pooling, dropout y combinación de media y máximo
global. Solo cambia la primera convolución, que recibe seis canales.
**310.945 parámetros entrenables**, 432 más que v2. Es una CNN propia desde
cero, sin pesos preentrenados ni variables clínicas.

- AdamW, lr **0,0001** (el original de v2), weight decay 0,001.
- Batch recomendado 16, semilla 42, fold 0, BCE por corte.
- Aumentos geométricos originales de v2, iguales en las tres fases.
- Sin cambios aleatorios adicionales de brillo ni giros de 90° en este ensayo.
- Media de probabilidades de todos los cortes para evaluar cada paciente.
- Normal y ponderada se entrenan por separado desde cero.
- Máximo 40 épocas por pérdida; paciencia 12, min_delta 0; selección por AUC
  de validación por paciente y el mismo planificador de v2.
- Train se evalúa sin aumentos/dropout para obtener curvas comparables; esa
  evaluación no decide qué pesos se seleccionan.

El ensayo combina una vista adicional y la ampliación de entrada necesaria
para usarla; no permite atribuir por sí solo una eventual mejora a uno de esos
componentes. La misma semilla no implica pesos iniciales idénticos a v2,
porque la primera capa tiene otra forma. No se cambia el objetivo del caso:
predicción de pCR por paciente, no localización o detección de lesiones.

## Entrenar en el laboratorio

Desde la raíz del repositorio, en la terminal integrada de VS Code:

```bash
git pull --ff-only
conda activate cancer
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Prueba corta para comprobar también la ejecución del preprocesado en ROCm:

```bash
python -m src.training.entrenar --experimento v6 --perdida ambas --dispositivo cuda --batch-size 16 --prueba --epocas 1
```

Si termina correctamente, entrenamiento completo:

```bash
python -m src.training.entrenar --experimento v6 --perdida ambas --dispositivo cuda --batch-size 16
```

No reinstalar requirements ni el PyTorch ROCm que ya funciona. `cuda` es también
el dispositivo de PyTorch para ROCm. Esta versión se verifica localmente en CPU;
la disponibilidad de GPU se comprueba en el laboratorio, no en el Mac.

Resultados en `models/<fecha_hora>_v6/`. Cada pérdida conserva
`epocas/epoca_001.pt`, etc., y `mejor_modelo.pt`. Las figuras de pérdida, AUC por
época, ROC, precisión–sensibilidad y confusión se exportan por separado en
PNG. Las épocas no guardadas de ejecuciones anteriores no se reconstruyen.

## Interpretación y siguiente decisión

La prueba corta solo comprueba funcionamiento, no rendimiento predictivo.
Comparar con v2 en el mismo fold, batch, criterio y protocolo. Revisar todas las
épocas, sensibilidad, especificidad, brecha train–validación y cohortes. No
parar por unas pocas épocas iniciales flojas ni prometer una mejora tardía.

No seleccionar una variante por su resultado en test: sigue reservado para el
final. El uso repetido de fold 0 para elegir modelos también puede adaptar las
decisiones a esa validación. Una variante prometedora debe contrastarse en los
cinco folds, con media y desviación, antes de fijar el protocolo final.

## Fundamento y límites

La literatura estudia la variación de intensidades de RM y la normalización,
pero **no valida esta configuración concreta ni garantiza mejorar pCR**:

- [Normalization of breast MRIs using Cycle-Consistent Generative Adversarial Networks](https://arxiv.org/abs/1912.08061): trata normalización entre equipos de RM mamaria con otro método.
- [On Sensitivity and Robustness of Normalization Schemes to Input Distribution Shifts in Automatic MR Image Diagnosis](https://proceedings.mlr.press/v227/madaan24a.html): estudia sensibilidad de esquemas de normalización ante cambios de distribución.

Aquí se usa una transformación fija explícita y una CNN desde cero; no se
entrena un generador ni se sintetizan lesiones. El aspecto visual mejorado no
implica por sí solo mayor AUC.

## Verificación local

Se superaron 50 pruebas automáticas en CPU; la prueba específica CUDA/ROCm se
omite cuando no hay GPU. Tras fijar p5–p95 se repitieron las cinco pruebas de CPU
específicas de v6. También terminó una prueba integrada de dos épocas por pérdida
con imágenes reales (cuatro pacientes de train y cuatro de validación): se
verificaron todos los `.pt` recargables y cinco PNG/PDF por pérdida.
Este subconjunto no estima AUC de generalización; no se ha ejecutado el
entrenamiento completo de v6.
