# Resultados guardados del proyecto

Instantánea exportada el 21 de octubre de 2026 de los archivos disponibles en el Mac.
Los originales permanecen en `models/`. Esta carpeta sí se versiona en GitHub.

## Entrenamientos de casa

AUC de validación interna por paciente (media de probabilidades de sus cortes).
El máximo observado en un historial incompleto no es un resultado final ni prueba
que el modelo no pueda mejorar. No son resultados del conjunto de test.

| Ejecución | Pérdida | Épocas registradas | Mejor época por AUC | Máximo AUC | Estado |
|---|---|---:|---:|---:|---|
| [20261015_232942_594565_v3](entrenamientos_casa/20261015_232942_594565_v3/normal/) | normal | 0 | — | — | Sin cierre registrado |
| [20261016_000244_480633_v3](entrenamientos_casa/20261016_000244_480633_v3/normal/) | normal | 13 | 1 | 0.5496 | Finalizado (ejecucion.json) |
| [20261016_111022_260137_v4](entrenamientos_casa/20261016_111022_260137_v4/normal/) | normal | 37 | 25 | 0.5583 | Finalizado (ejecucion.json) |
| [20261016_111022_260137_v4](entrenamientos_casa/20261016_111022_260137_v4/ponderada/) | ponderada | 9 | 2 | 0.5633 | Sin cierre registrado |
| [20261017_133700_017574_v5](entrenamientos_casa/20261017_133700_017574_v5/normal/) | normal | 3 | 2 | 0.5047 | Sin cierre registrado |

## Organización

- `entrenamientos_casa/`: ejecuciones v3, v4 y v5, incluidos intentos sin épocas completas.
- `pruebas_funcionamiento/`: ejecuciones pequeñas con `--prueba`; sus AUC no evalúan el rendimiento real.
- `diagnosticos/`: auditoría agregada, comparación por cohorte, memorización y tiempos de CPU. La memorización no mide generalización.
- `referencias_laboratorio/`: resultados anteriores recibidos del laboratorio para comparación.
- [resumen.csv](resumen.csv): inventario de todas las ejecuciones y pérdidas con configuración disponible.

## Contenido y límites

Se conservan historiales completos disponibles, configuraciones, métricas agregadas,
informes de ejecución y las tres gráficas principales cuando existían.
Las rutas locales `datos` y `salida` de las configuraciones se sustituyeron por `.` y `models`;
los hiperparámetros y valores medidos se conservan.

No se incluyen pesos `.pt`, imágenes médicas, particiones ni predicciones individuales.
Este archivo sirve para revisar resultados; no permite reanudar un entrenamiento.
Las ejecuciones sin `ejecucion.json` se señalan como «sin cierre registrado», sin inferir
que acabaran correctamente. No se han reconstruido métricas ni curvas ROC ausentes.
V5 conserva tres épocas y su mejor validación, pero no llegó a exportar predicciones finales.

Se han regenerado las figuras disponibles de v3, v4, v5 y la referencia v2 en
`graficas/normal/` y `graficas/ponderada/`, una gráfica por archivo, en PNG.
Se retiraron los paneles duplicados del formato anterior. Se conservan los historiales, métricas y gráficas individuales de cada experimento.
Sin predicciones guardadas solo se pueden rehacer pérdida y AUC por época;
esto ocurre en v5, v4 ponderada y la copia local de la referencia v2.

Es una instantánea: los entrenamientos futuros no se publican automáticamente.
Para consultarla desde el laboratorio, actualizar el repositorio con `git pull --ff-only`.
