# Ensayo controlado de v7: tasa inicial de aprendizaje

Objetivo de desarrollo: comprobar si v7 normal con lr inicial 0,0003 mejora el AUC externo medio respecto a v7 normal con 0,0001 (0,5326373 en la comparación ya terminada). El objetivo orientativo es 0,57; no es un resultado garantizado ni un motivo para seleccionar únicamente folds favorables.

Solo cambia la tasa inicial. Se conservan arquitectura, inicialización/semilla 42, pérdida normal, batch 16, aumentos, AdamW, weight decay 0,001, máximo 40 épocas, paciencia 9, scheduler y selección por AUC interna. El scheduler puede producir una secuencia distinta de tasas/épocas como consecuencia del cambio inicial. Cinco entrenamientos: un v7 normal por fold. No se repiten v2 ni las pérdidas ponderadas.

Se mantienen las mismas particiones por paciente: tres folds para entrenamiento, el siguiente fold al externo para selección interna y el fold externo para evaluación. La comparación antigua conserva sus resultados. El test oficial no se utiliza.

## En el laboratorio

Desde la raíz del repositorio, con el entorno cancer activado, actualizar antes:

```bash
git pull --ff-only
```

Comprobación rápida, sin entrenar ni crear resultados:

```bash
python -m src.training.comparar_cv --modelos v7 --perdida normal --lr 0.0003 --salida models/cv_v7_lr3e4 --solo-plan
```

Entrenamiento completo en CUDA/ROCm:

```bash
python -m src.training.comparar_cv --modelos v7 --perdida normal --lr 0.0003 --dispositivo cuda --salida models/cv_v7_lr3e4
```

Debe anunciar cinco entrenamientos y lr inicial 0.0003. No usar `--prueba` en la ejecución completa. No instalar dependencias ni sustituir el PyTorch del laboratorio.

Si se interrumpe:

```bash
python -m src.training.comparar_cv --modelos v7 --perdida normal --lr 0.0003 --dispositivo cuda --salida models/cv_v7_lr3e4 --reanudar
```

Conserva los folds completados y archiva/reinicia el fold incompleto; no reanuda el optimizador a mitad de una época. El protocolo impide mezclar tasas, arquitecturas, datos o código distintos. La actualización cambia el hash de código: no utilizarla para reanudar una ejecución antigua sin terminar. La comparación original del laboratorio ya está completa.

## Resultados que conservar

Dentro de `models/cv_v7_lr3e4/` se generan `protocolo.json`, `resumen.json`, `resultados_por_fold.csv` y `COMPARACION.md`. Cada fold conserva selección interna, historial completo, predicciones, umbral congelado y resultado externo, además de los .pt por época. En este ensayo solo hay v7: no se genera un archivo de diferencias v7−v2 dentro de la nueva carpeta.

Comparar al terminar los cinco folds contra las filas v7/normal de la ejecución original, emparejando por fold. Revisar diferencias de AUC, F1 macro, sensibilidad, especificidad y separación entrenamiento/validación. No declarar mejora sólida por un único fold o por el AUC interno máximo.

Se han usado resultados de desarrollo para proponer este ensayo: la selección sigue teniendo sesgo por búsqueda. El test final seguirá siendo una evaluación separada después de congelar el candidato elegido. El umbral del v7 original no debe trasladarse a los nuevos pesos.

La referencia de tiempo observada en v7 anterior fue aproximadamente 91 segundos por época en el laboratorio. Si se mantiene, 5 × 40 épocas serían unas 5 horas más evaluación/escritura; la parada anticipada puede reducirlo. No es una garantía de duración.
