# V7: CNN residual compacta y paciencia 9

## Qué se prueba

El usuario comunica que v6 obtuvo un resultado peor que el ensayo anterior.
Su historial completo y sus predicciones no están disponibles en este checkout;
no se atribuye el resultado a una causa concreta ni se inventa una cifra de AUC.
La mejor referencia documentada sigue siendo v2: AUC de validación 0,6123.

V7 cambia la arquitectura: bloques residuales y una última etapa más estrecha.
Recupera las tres fases originales /255 de v2, sin la vista auxiliar de contraste
de v6. Se evalúa una nueva hipótesis, no un modelo ya demostrado como superior.

## Arquitectura propia desde cero

| Etapa | Operación | Salida por corte |
|---|---|---|
| Entrada | PRE, EARLY, LATE originales /255 | 3 × 256 × 256 |
| 1 | Bloque residual 3→16 + MaxPool2d(2) + Dropout2d(0,1) | 16 × 128 × 128 |
| 2 | Bloque residual 16→32 + MaxPool2d(2) + Dropout2d(0,1) | 32 × 64 × 64 |
| 3 | Bloque residual 32→64 + MaxPool2d(2) + Dropout2d(0,1) | 64 × 32 × 32 |
| 4 | Bloque residual 64→96 + MaxPool2d(2) + Dropout2d(0,1) | 96 × 16 × 16 |
| Resumen | Media y máximo global concatenados | 192 |
| Cabeza | Dropout(0,4), Linear(192,64), ReLU, Dropout(0,3), Linear(64,1) | 1 logit |

Cada bloque aplica Conv3×3 → GroupNorm(8) → ReLU → Conv3×3 → GroupNorm(8).
Suma a esa salida un atajo Conv1×1 → GroupNorm(8) para adaptar los canales y
aplica ReLU. El bloque también admite identidad cuando los canales coinciden.
No se usan modelos descargados, pesos preentrenados ni variables clínicas.

Los atajos buscan facilitar la transmisión de información y gradientes. La
última etapa de 96 canales reduce la capacidad respecto a los 128 de v2:
232.513 parámetros entrenables frente a 310.513 (aproximadamente 25 % menos).
Esto no garantiza menos sobreajuste ni mejor AUC. El fundamento de las conexiones
residuales procede de [He et al., CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html);
sus resultados no validan esta CNN concreta ni la predicción de pCR.

## Early stopping solicitado

**Paciencia 9**, contada desde la última mejora estricta de ROC-AUC por paciente
en validación interna. `min_delta=0`: un empate cuenta como época sin mejora.
Una mejora reinicia el contador. Por ejemplo, si el mejor AUC aparece en la
época 6 y no se supera, termina después de la 15. Nunca supera 40 épocas por
pérdida. Al finalizar se restaura el mejor modelo, no necesariamente el último.

Esta es la preferencia del usuario para los siguientes ensayos. Los presets
antiguos conservan sus valores para reproducirlos; v7 ya usa 9 sin argumentos
adicionales. Para aplicar 9 a otro preset, añadir `--paciencia 9`.
La espera menor ahorra tiempo, pero puede perder mejoras que llegasen más tarde.

## Protocolo

- Imágenes completas 256×256; canales correspondientes a PRE/EARLY/LATE, no colores RGB.
- AdamW, lr=0,0001, weight decay=0,001, batch 16, semilla 42.
- Aumentos geométricos de v2: reflejo horizontal y ±10°, comunes a las fases.
- BCE por corte, normal y ponderada en entrenamientos separados desde cero.
- Evaluación por paciente mediante media de sus probabilidades por corte.
- Train evaluado sin aumentos/dropout para curvas comparables.
- Scheduler ReduceLROnPlateau como v2; no cambia el criterio de selección.
- `.pt` de cada época, mejor `.pt`, métricas e imágenes separadas PNG.

Frente a v2 cambian arquitectura y paciencia solicitada; no es una ablación de
una única variable. Frente a v6 también se retira la vista auxiliar. Una mejora
no permitiría atribuir todo el efecto a los atajos. Mantener fold 0 para una
primera comparación y después comprobar los cinco folds si resulta prometedor.
La búsqueda repetida sobre fold 0 puede sobreajustar las decisiones a ese fold.
No consultar test para elegir arquitecturas, épocas o umbrales.

## Ejecutar en el laboratorio

Desde el repositorio, en la terminal integrada de Visual Studio Code:

```bash
git pull --ff-only
conda activate cancer
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Comprobación breve en la GPU (no mide rendimiento predictivo):

```bash
python -m src.training.entrenar --experimento v7 --perdida ambas --dispositivo cuda --batch-size 16 --prueba --epocas 1
```

Después, entrenamiento completo:

```bash
python -m src.training.entrenar --experimento v7 --perdida ambas --dispositivo cuda --batch-size 16
```

No reinstalar el entorno ROCm funcional. Resultados en `models/<fecha_hora>_v7/`.
No hay que añadir `--paciencia 9`: forma parte del preset. No se ha ejecutado
el entrenamiento completo ni se puede prometer que alcance AUC 0,70.

## Comprobaciones realizadas

- Suite: 55 pruebas descubiertas, 54 superadas y una de GPU omitida en el Mac.
- Se comprueba el atajo residual con la rama anulada, los gradientes finitos,
  la forma de salida y la recarga exacta de los pesos.
- Secuencia controlada de AUC: una mejora reinicia paciencia y nueve empates
  consecutivos detienen el entrenamiento, guardando también esas épocas.
- Prueba integrada en CPU: dos épocas por pérdida, cuatro pacientes de train y
  cuatro de validación; `.pt` recargables y cinco PNG/PDF por pérdida verificados.

El AUC de esa prueba pequeña llegó a 1,00 con solo cuatro pacientes de validación:
**no es evidencia de haber alcanzado el objetivo 0,70** y no se usa para elegir
la arquitectura. Solo confirma que el programa se ejecuta y exporta resultados.
