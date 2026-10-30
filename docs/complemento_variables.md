# Complemento exploratorio de v8: imágenes y variables

Se mantiene la CNN v8 final (época 6) y su umbral 0,356243176. El añadido ajusta
dos regresiones logísticas L2 (C=1, lbfgs, semilla 42): edad + subtipo, y las mismas
variables más el logit de la media de probabilidades de los cortes de v8.

Se reutiliza la partición por paciente: 878 entrenamiento y 219 validación interna,
sin solapamientos. El ajuste usa predicciones ya guardadas; no ejecuta la CNN,
no modifica sus pesos y no consulta imágenes ni resultados del test.
Edad ausente: mediana de entrenamiento e indicador de ausencia. Subtipo:
HRposHER2neg, TripleNeg, HER2pos o No disponible, codificados en columnas binarias.
Mediana y escalado se ajustan exclusivamente en entrenamiento. El umbral de cada
regresión maximiza F1 macro en validación interna. No se han buscado hiperparámetros.

| Modelo | AUC interna | F1 macro | Sensibilidad | Especificidad | VP / VN / FP / FN |
|---|---:|---:|---:|---:|---|
| Imágenes v8 | 0,576 | 0,559 | 0,422 | 0,703 | 27 / 109 / 46 / 37 |
| Edad y subtipo | 0,689 | 0,658 | 0,812 | 0,613 | 52 / 95 / 60 / 12 |
| Imágenes + edad y subtipo | 0,717 | 0,665 | 0,750 | 0,658 | 48 / 102 / 53 / 16 |

Todas las filas corresponden a las mismas 219 pacientes (64 con pCR). Umbrales:
0,356243176; 0,32580391449182805; 0,3268511030247533, respectivamente.
El combinado tiene mayor AUC y especificidad que el tabular en esta validación,
pero menor sensibilidad al umbral elegido. No hay una mejora de todas las métricas.

## Alcance

Este añadido se planteó después de conocer el test de v8. La validación compartida
ya sirvió para seleccionar la CNN, y las puntuaciones de entrenamiento de la CNN
proceden de pacientes usados en su ajuste (no son predicciones fuera de fold).
Por tanto, esta comparación es exploratoria y no estima rendimiento independiente.
La diferencia de AUC de 0,028 frente a solo variables no demuestra por sí sola que
la imagen aporte una mejora generalizable. No se ha evaluado el complemento en test.
El resultado de test de la CNN permanece en AUC 0,504 y no se sustituye por 0,717.

En la app deben aportarse edad y subtipo de la misma paciente, conocidos antes del
tratamiento. No se usan pCR, cohorte, variables de adquisición ni datos posteriores
como predictores. `tum_vol` queda excluido hasta verificar unidad exacta y momento
de medición. «cm³ o mm³» no resuelve esa ambigüedad.

Paquete portable: `app/paquetes/complemento_v8.json` (coeficientes, transformaciones,
umbrales, métricas y SHA-256 de la CNN; sin identificadores de pacientes).
Resultados locales: `models/complemento_v8/`, con predicciones de validación y
partición para auditoría. No publicar esos CSV de pacientes.
