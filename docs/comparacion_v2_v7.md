# Comparación de desarrollo: v2 y v7

Objetivo: comprobar si la diferencia de AUC y sobreajuste se repite en cinco
grupos de pacientes. No se promete alcanzar 0,70 ni se elige el modelo por lo
juntas que estén sus curvas: también puede haber poco aprendizaje.

## Protocolo fijado antes de ejecutar

Se conservan los cinco folds oficiales de `split=train`. Para cada fold externo
`k`, se usa `(k+1)%5` como selección interna y los otros tres para entrenamiento.
Cada paciente se evalúa una vez en el exterior, con un modelo que no la ha usado
para entrenar, escoger la época ni elegir el umbral. Se comprueba la separación
por paciente y la presencia de ambas clases. El test oficial queda reservado:
no se cargan sus imágenes ni se calculan sus resultados.

Esto es CV externa con una partición interna fija por fold, no una búsqueda
anidada exhaustiva. Cada ajuste usa aproximadamente 60% de desarrollo para
entrenar, 20% para selección interna y 20% para evaluación externa. Por ello no
se deben comparar sus medias directamente con el máximo histórico 0,612 de v2,
que usaba 80% para entrenar y seleccionaba sobre la misma validación reportada.

Ambas arquitecturas parten de cero y usan el mismo protocolo: semilla 42,
AdamW, lr 0,0001, weight decay 0,001, aumentos geométricos sincronizados,
PRE/EARLY/LATE divididos por 255, máximo 40 épocas, paciencia 9, scheduler y
evaluación de train sin aumentos/dropout. La arquitectura v2 es la original;
su paciencia se iguala a la de v7 para comparar arquitecturas, de modo que no
es una repetición exacta del antiguo preset v2 (paciencia 12). No es v2 corto.

Cada modelo selecciona su checkpoint por máxima AUC **interna por paciente**.
El umbral se fija después usando máxima F1 macro interna; los empates favorecen
el valor más cercano a 0,5 y después el mayor. Es un criterio estadístico que
equilibra las dos clases, no una justificación clínica de costes. La agregación
se mantiene en media de probabilidades por corte. El exterior se evalúa una
sola vez con checkpoint/umbral congelados; también se reporta el umbral 0,5
y la referencia que siempre predice «sin pCR».

No cambiar arquitectura, hiperparámetros o umbrales después de mirar cada fold.
Esperar la comparación completa. Estos datos ya se han utilizado para diseñar
las variantes, por lo que persiste sesgo histórico de selección. Una semilla
no permite medir toda la variabilidad de inicialización. Los cinco ajustes
comparten parte del entrenamiento y no son cinco experimentos independientes.

## Ejecución en el laboratorio

Desde el repositorio, en la terminal integrada de VS Code:

```bash
git pull --ff-only
conda activate cancer
python -m src.training.comparar_cv --dispositivo cuda --perdida ambas --solo-plan
python -m src.training.comparar_cv --dispositivo cuda --perdida ambas
```

Son **20 entrenamientos secuenciales**: 2 arquitecturas × 5 folds × 2 pérdidas.
La ponderación se calcula solo con los cortes de entrenamiento de cada fold.
Para comparar primero únicamente las arquitecturas con pérdida normal, usar
`--perdida normal --salida models/comparacion_v2_v7_normal` (10 entrenamientos).
No mezclar después otra pérdida en la misma carpeta/protocolo.

No iniciar otro entrenamiento simultáneo en esa GPU. La duración depende de
las épocas hasta parada y del equipo; el tiempo de una ejecución previa no
es una garantía para las veinte. Se guardan todos los `.pt` por época: el disco
debe alojar hasta 800 checkpoints si los veinte trabajos llegan a 40 épocas.

Para continuar tras cerrar la sesión o interrumpir:

```bash
python -m src.training.comparar_cv --dispositivo cuda --perdida ambas --reanudar
```

Repetir la misma configuración, incluido `--salida` si se cambió. Se validan
protocolo, huella de metadatos y código. Los trabajos completos se omiten.
Un trabajo incompleto se archiva con sufijo `_incompleto_...` y se reinicia
desde cero: los `.pt` no contienen el estado del optimizador. Un fallo no se
interpreta como entrenamiento completado.

Prueba técnica opcional, separada del experimento real:

```bash
python -m src.training.comparar_cv --dispositivo cuda --perdida normal --prueba --epocas 1 --salida models/prueba_cv
```

Sus resultados se etiquetan como prueba reducida y no miden rendimiento.

## Resultados

En `models/comparacion_v2_v7/`:

- `protocolo.json`: configuración fijada y huellas de datos/código.
- `COMPARACION.md`: tabla compacta con AUC externa media ± desviación muestral.
- `resultados_por_fold.csv`: AUC, sensibilidad, especificidad y F1 macro externos,
  junto con AUC de train/interna, época y umbral seleccionados.
- `diferencias_pareadas.csv`: AUC v7 menos v2 sobre el mismo fold y pérdida.
- `resumen.json`: medias/desviaciones y número de folds completados; parciales
  se marcan expresamente incompletos.
- `fold_N/v2|v7/normal|ponderada/seleccion_interna/`: historial de todas las
  épocas, métricas internas y `epocas/epoca_NNN.pt`, además del mejor modelo.
- En la carpeta de cada trabajo: partición, selección congelada, predicciones
  externas y `resultado.json` con métricas externas, por clase y referencia.

Esta comparación genera tablas e historiales, sin multiplicar las gráficas.
Podremos escoger las figuras necesarias cuando se prepare la presentación.
No subir a GitHub los pesos ni los archivos con identificadores de pacientes;
para revisar basta con protocolo, tablas agregadas y resultados.json.

Se valoran AUC externa media, dispersión y consistencia de diferencias por fold,
además de sensibilidad/especificidad y F1. La desviación entre folds **no es un
intervalo de confianza**, y una diferencia pequeña no demuestra superioridad.
El umbral seleccionado internamente puede rendir peor fuera: eso forma parte
de la evaluación. No se retoca usando las etiquetas externas.

Si no hay un ganador claro, documentar la incertidumbre y considerar repetir
las dos arquitecturas con semillas prefijadas. Después puede plantearse una
mejora concreta y controlada. El test oficial se utiliza una vez, tras fijar el
modelo y el procedimiento final de entrenamiento y selección del umbral.

Referencia metodológica: [scikit-learn: selección y evaluación con CV anidada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
Usar el mismo conjunto para seleccionar y reportar rendimiento puede generar
estimaciones optimistas; la partición interna separa esas dos funciones.

## Verificación local (23 de octubre de 2026)

Siete pruebas automáticas pasan: separación por paciente y del test, selección
de umbral tras agregar cortes, congelación antes de evaluar fuera, estadísticas
pareadas y reanudación sin mezclar protocolos. También se completó el flujo
real en CPU con una época, dos pacientes por clase y grupo, cinco folds y ambas
arquitecturas con pérdida normal: diez ajustes y diez checkpoints por época.
Se verificaron los resúmenes completos y la separación externa. Es una prueba
técnica, no evidencia de rendimiento ni validación de la GPU del laboratorio.
Compilación Python y revisión de espacios correctas; Ruff no está instalado
en el entorno local y ese chequeo queda para la integración continua.
