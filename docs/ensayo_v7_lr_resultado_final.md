# V7 con tasa 0,0003: resultado completo

Fuente: CSV pegado por el usuario desde el laboratorio el 28 de octubre de 2026,
`models/cv_v7_lr3e4/resultados_por_fold.csv`. Se recibieron los cinco folds;
no se recibieron en este mensaje los checkpoints ni historiales completos.

| Fold | Mejor época | AUC externo |
|---|---:|---:|
| 0 | 2 | 0,527520 |
| 1 | 10 | 0,502466 |
| 2 | 6 | 0,587325 |
| 3 | 7 | 0,491034 |
| 4 | 4 | 0,497945 |

Media ± desviación estándar muestral: **0,521258 ± 0,039415**.
V7 normal original, con tasa 0,0001: **0,532637 ± 0,043616**.
Diferencia de medias: **−0,011379**. Esta desviación no es un intervalo de
confianza y la comparación no demuestra significación estadística.

El aumento de la tasa no mejora el AUC medio observado; conservar v7 normal
original como referencia para el ensayo v8. Son resultados de validación externa
de desarrollo, no del test final reservado. Sustituye la conclusión provisional
de `ensayo_v7_lr_resultados_parciales.md`.
