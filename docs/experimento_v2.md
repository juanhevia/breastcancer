# Experimento v2: hipótesis para mejorar la AUC por paciente

## Punto de partida

En los historiales recibidos del laboratorio, el baseline obtuvo AUC de
validación interna por paciente de 0,561492 (BCE normal) y 0,561895 (ponderada).
Ambos seleccionaron la época 2 por mínima pérdida y pararon en la 9. La pérdida
de entrenamiento seguía bajando, pero la validación no mejoraba. Esto sugiere
sobreajuste; no demuestra que aumentar la profundidad vaya a solucionarlo.

El objetivo experimental es mejorar la discriminación por paciente. Cambiar el
umbral de decisión no mejora la ROC-AUC calculada sobre las puntuaciones medias.
No se garantiza AUC > 0,70 ni se usa test para buscarla.

## Cambios y justificación

| Componente | Baseline conservado | Experimento v2 |
|---|---|---|
| Entrada | PRE, EARLY, LATE / 255 | Igual: 3 × 256 × 256 |
| CNN | Tres convoluciones, 32/64/128 canales | Cuatro bloques, dos convoluciones 3×3 por bloque, 16/32/64/128 canales |
| Normalización intermedia | BatchNorm | GroupNorm de 8 grupos, sin estadísticas dependientes del lote |
| Reducción espacial | MaxPool tras cada convolución | MaxPool tras cada bloque; salida 128 × 16 × 16 |
| Resumen espacial | Media global | Concatenación de media y máximo globales |
| Cabeza | 128 → 1 | 256 → 64 → 1 |
| Parámetros entrenables | 93.825 | 310.513 |
| Regularización | Dropout 0,4 | Dropout2d 0,1 por bloque; dropout 0,4 y 0,3 en la cabeza; weight decay 0,001 |
| Aumentos | Ninguno | Reflejo horizontal con probabilidad 0,5 y giro uniforme ±10° |
| Optimizador | Adam | AdamW |
| lr inicial | 0,0001 | 0,0001 |
| Reducción de lr | Ninguna | Factor 0,5 tras cuatro épocas consecutivas sin mejora (patience=3 en PyTorch), mínimo 0,000001 |
| Selección y parada | Mínima BCE, paciencia 7 | Máxima AUC por paciente, paciencia 12 |
| Máximo de épocas | 5 por defecto, 40 en las ejecuciones recibidas | 40 |

Las convoluciones adicionales amplían el contexto espacial de cada activación.
La media y el máximo permiten resumir tanto señales distribuidas como
localizadas; el máximo también puede captar ruido, por lo que debe evaluarse.
GroupNorm evita depender de la composición del lote. La mayor capacidad se
acompaña de regularización y aumentos para intentar limitar el sobreajuste.
Todo es una CNN propia desde cero, sin arquitecturas ni pesos preentrenados.

Se mantiene la inicialización predeterminada de PyTorch, controlada por semilla
42. Batch=16, fold=0, agregación por media y umbral=0,5. AdamW usa betas y eps
predeterminados de PyTorch; el weight decay se aplica a todos los parámetros.
La configuración efectiva, incluyendo arquitectura y número de parámetros,
queda registrada en cada ejecución.

Los aumentos usan una única transformación para las tres fases de cada corte,
con interpolación bilineal y fondo cero. Se aplican solo a entrenamiento;
validación e inferencia mantienen las imágenes originales divididas entre 255.
La pérdida train se mide con aumentos y dropout activos; la pérdida val se mide
sin ellos. Por eso su diferencia no es una medida pura de sobreajuste.

## Ejecución y comparación

1. Ejecutar `--experimento v2 --perdida ambas` sobre la misma partición interna
   del baseline, sin `--prueba`. No sustituye ni reanuda sus pesos: crea dos
   modelos nuevos y una carpeta nueva.
2. Revisar todas las épocas, AUC por paciente, ROC, precisión–recall, sensibilidad
   y especificidad. Las gráficas se agrupan en tres figuras principales; los diagnósticos adicionales
   se generan con `--detalle`. La mejor época
   se decide por AUC; la pérdida también se guarda y se representa.
3. Si mejora, separar el efecto de los cambios antes de atribuirles el mérito.
   Una comparación útil es `--experimento v2 --modelo baseline`, que aplica las
   mismas opciones de entrenamiento a la arquitectura original. Otra es
   `--experimento v2 --no-aumentos`, que elimina solo los aumentos de la nueva CNN.
   Estas comparaciones son opcionales y deben documentarse, no ejecutarse como
   una búsqueda ilimitada hasta obtener un número deseado.
4. Con una configuración fijada, comprobar estabilidad en otras semillas y
   folds internos. Se puede cambiar `--semilla 43` o `--fold 1` sin editar código.
   Reportar también los resultados que no mejoren; no elegir solo el fold más
   favorable. Los modelos de folds distintos tienen conjuntos de entrenamiento
   solapados: no son cinco estudios independientes.
5. Fijar arquitectura, pérdida, agregación y cualquier umbral con validación
   interna. Reservar test para la evaluación final. La AUC máxima del mismo
   conjunto usado para elegir época y configuración puede ser optimista.

El baseline original seleccionaba por BCE, mientras v2 selecciona por AUC. Si
se quiere aislar el efecto de la arquitectura, hay que comparar bajo un mismo
criterio de selección, además de mantener el mismo fold y semilla. V2 combina
varios cambios: una mejora aislada no demostraría cuál de ellos la causó.

Se guarda `lr` en el historial y `seleccion`/`valor_seleccion` junto a las
métricas y el checkpoint. Los empates de AUC no reinician la paciencia. Los
pesos restaurados y las predicciones finales pertenecen a la mejor época,
mientras las gráficas temporales incluyen todas las épocas realizadas.

## Recursos y archivos

No hay dependencias nuevas. En el laboratorio se mantiene el entorno `cancer`
y su PyTorch ROCm; el dispositivo sigue llamándose `cuda` en la CLI. Si falta
memoria GPU, reducir `--batch-size 8` y registrar ese cambio para comparar ambas
pérdidas con las mismas condiciones.

Resultados: `models/<fecha_hora>_v2/{normal,ponderada}/`. Las ejecuciones cortas
añaden `_prueba` y no sirven para estimar rendimiento. `models/` está excluida
de Git: sincronizar el código no transfiere los pesos ni las gráficas del
laboratorio. Copiar la carpeta de resultados si se necesita revisarla aquí.
