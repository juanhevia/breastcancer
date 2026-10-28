# V8 — fases originales y diferencias temporales

## Estado y motivo

Ensayo preparado, sin resultado de rendimiento. No hay garantía de AUC 0,57 ni
0,70. La comparación anterior de cinco folds dio v7 normal 0,532637; el ensayo
con tasa 0,0003 terminó con media 0,521258 y desviación entre folds 0,039415
(cinco filas recibidas del laboratorio el 28 de octubre). No comparar esas
medias externas con la validación interna del checkpoint original (0,579335).

También se comprobó un promedio fijo 50/50 de las predicciones de v2 y v7
normales, seleccionando el umbral exclusivamente en el grupo interno de cada
fold. Dio AUC externa media 0,532117 en los cinco folds: no aporta mejora de AUC
frente a v7. El análisis es exploratorio posterior sobre desarrollo, no test.
Puede reproducirse con `python docs/analisis_complementariedad_v2_v7.py` cuando
se disponga de las predicciones importadas de la comparación inicial.

## Hipótesis

V8 recibe los mismos tres PNG de 256×256 divididos entre 255. Dentro de la red
concatena PRE, EARLY, LATE, EARLY−PRE y LATE−EARLY. Conserva las diferencias
negativas sin recortar, dividir por PRE, ecualizar ni ajustar colores.

Las diferencias son información redundante que una primera convolución ya
podría aprender. No crean información nueva ni añaden capacidad expresiva
fundamental: se prueba si representar explícitamente esos cambios facilita la
optimización. No se presenta esta variante como una solución demostrada al bajo
rendimiento. V3 ya utilizó diferencias con otra arquitectura y sin conservar
los tres canales originales; este ensayo mantiene los bloques de v7.

La entrada del primer bloque residual pasa de tres a cinco canales. El resto
conserva 16/32/64/96 canales, GroupNorm, media/máximo global y clasificador de
64 unidades. Hay 232.833 parámetros frente a 232.513 de v7. Las diferencias
se calculan en forward, idénticamente al entrenar y al recargar un checkpoint.

## Protocolo

Tasa inicial 0,0001, AdamW, weight decay 0,001, semilla 42, batch 16, máximo
40 épocas y paciencia 9. Mismos aumentos geométricos, BCE por corte y media de
probabilidades por paciente. El comparador selecciona época y umbral en el grupo
interno y evalúa el externo después. Primero se compara pérdida normal contra
v7 normal; las pérdidas ponderadas de v2/v7 ya tienen su comparación disponible.
Se mantienen los .pt por época y las figuras PNG del entrenamiento existente.

## Ejecución opcional en el laboratorio

El nuevo código cambia el hash del protocolo. Actualizar el checkout solo una
vez terminado el entrenamiento anterior. Si necesita reanudarse una ejecución
interrumpida, hacerlo con su versión anterior de código antes de actualizar.
Estos cambios no se han ejecutado ni instalado en el ordenador del laboratorio.

Desde la raíz del repositorio, tras sincronizar esta versión y activar cancer:

```bash
python -m src.training.comparar_cv --modelos v8 --perdida normal --lr 0.0001 --dispositivo cuda --salida models/cv_v8_realce --solo-plan
```

Debe indicar cinco entrenamientos. Para ejecutarlos, quitar `--solo-plan`:

```bash
python -m src.training.comparar_cv --modelos v8 --perdida normal --lr 0.0001 --dispositivo cuda --salida models/cv_v8_realce
```

No usa test ni sobrescribe resultados previos. Para continuar esa misma ejecución
con el mismo código y argumentos, añadir `--reanudar`; conserva los folds completos
y reinicia el incompleto. No es reanudación desde el último .pt del optimizador.

No ejecutar una búsqueda de arquitecturas o tasas adicional con estos folds sin
reconocer el sesgo acumulado. Comparar los cinco folds con v7 normal original y
reportar todas las diferencias, aunque sean negativas. La app actual admite v2
y v7; integrar v8 solo si se elige tras evaluar. No se ha cambiado su modelo activo.

## Comprobaciones técnicas

Pruebas de conservación de fases, diferencias negativas, gradientes finitos,
salida por corte, guardado/recarga exacta y compatibilidad de v7. El protocolo
hereda la paciencia 9 y los hiperparámetros de v7. Las pruebas técnicas y una
ejecución reducida no son mediciones de capacidad predictiva.
