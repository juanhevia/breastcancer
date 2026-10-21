# Gráficas para presentación

Validación interna · test reservado

pCR = respuesta patológica completa; sin pCR = ausencia de respuesta completa.

Cada pérdida tiene su carpeta: normal/ o ponderada/. Cada figura tiene PNG a 300 dpi y PDF vectorial.
Las explicaciones se conservan aquí para dejar las diapositivas despejadas.

- 01_perdida: todas las épocas; train_eval_loss si existe (sin aumentos/dropout), si no train_loss de optimización.
- 02_auc_por_epoca: AUC por paciente en todas las épocas; train solo cuando se registró. Línea vertical: época elegida.
- 03_roc y 04_precision_sensibilidad: validación del mejor modelo; recorren umbrales, no épocas. AP no es área trapezoidal.
- 05_matriz_confusion: pacientes; porcentaje dentro de cada clase real (fila).
Las pérdidas normal/ponderada no tienen la misma escala. No se suavizan las curvas.
Si faltan predicciones, solo se generan pérdida y AUC; no se inventan curvas ROC.

## Procedencia de las figuras

- BCE normal: época elegida 2; criterio mayor AUC por paciente; agregación mean; umbral 0.5. Sin predicciones disponibles.

Validación interna usada para elegir el modelo, no test. Pruebas reducidas no estiman rendimiento real.
Una sola clase impide estimar ROC/PR. Una sola época produce un punto.
Con voto, las curvas mantienen fijo el umbral de los cortes y varían el umbral sobre la fracción de votos.
--detalle exporta cuatro figuras adicionales separadas: calibración, umbral, puntuaciones y métricas por época.
Calibración agrupa puntuaciones, no épocas; no se recalibra el modelo. Las puntuaciones no garantizan probabilidades calibradas.
