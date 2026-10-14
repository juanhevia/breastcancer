# BreastDCEDL: CNN para predecir pCR

Proyecto docente con una CNN 2D desde cero. Entrada: PRE, EARLY y LATE, en ese
orden, apiladas en un tensor de 3 × 256 × 256 y divididas entre 255.

## Organización

```text
.github/workflows/          Comprobaciones automáticas
app/                        Futura aplicación web
docs/                       PDF, guía y revisión del código
models/                     Pesos y resultados de cada ejecución
notebooks/                  Exploración y análisis
presentations/              Presentación final
src/
  data/                     utils_caso.py, descargar_datos.py, ver_muestras.py
  models/modelo.py          Tu arquitectura BreastCNN
  training/entrenar.py      Entrenamiento por épocas
tests/                      Comprobaciones del entrenamiento
.env.example                Explica la configuración del entorno
pyproject.toml              Dependencias y herramientas con Poetry
poetry.lock                 Versiones resueltas por Poetry
requirements.txt            Instalación alternativa con pip
dataset/                    Imágenes locales, excluidas de Git
metadata/                   Índices originales con rutas a dataset/
```

Los comandos siguientes se ejecutan desde la raíz de este proyecto. Se usa
`python3 -m` para que Python encuentre los paquetes de las subcarpetas.

## Empezar con las épocas

Con el entorno Python que tenga las dependencias instaladas:

```bash
python3 -m src.training.entrenar --epocas 5 --perdida ambas
```

Esto entrena **dos modelos independientes** con tu misma arquitectura: uno con
BCE normal y otro con BCE ponderada. Cada uno hace hasta 5 épocas. Ambos parten
de la misma semilla, inicialización y orden de lotes para poder comparar las
pérdidas. No se garantiza igualdad bit a bit entre dispositivos o versiones.

Una época significa recorrer una vez todos los cortes de entrenamiento y luego
validar. Con fold 0 son **8.759 cortes de 878 pacientes** para entrenar y
**2.186 cortes de 219 pacientes** para validar. Se usa exclusivamente el split
`train` original. `test` queda reservado para la evaluación final.

Para comenzar solo con la pérdida normal:

```bash
python3 -m src.training.entrenar --epocas 5 --perdida normal
```

Para una ejecución más larga después de comprobar las curvas:

```bash
python3 -m src.training.entrenar --epocas 40 --paciencia 7 --perdida ambas
```

Son máximos: la parada anticipada detiene el entrenamiento tras 7 épocas sin
mejorar la pérdida de validación y recupera los pesos de la mejor época. Cada
comando empieza desde cero; no continúa automáticamente una ejecución anterior.

Para verificar rápidamente el funcionamiento:

```bash
python3 -m src.training.entrenar --epocas 1 --prueba --batch-size 4
```

`--prueba` usa dos pacientes de cada clase en cada partición, conservando todos
sus cortes. Sus métricas **no son resultados del experimento completo**.

Opciones iniciales: Adam, lr=0.0001, batch=16, semilla=42, fold=0, umbral=0.5,
agregación por media y workers=0. Se detecta CUDA, MPS o CPU automáticamente;
puedes forzar `--dispositivo cpu`. `--hilos 4` limita los hilos de CPU. No se
usan aumentos de datos todavía: este es el baseline inicial.

## Resultados guardados

Cada ejecución crea `models/<fecha_hora>/normal/` y/o `ponderada/` con:

- `configuracion.json`: hiperparámetros, arquitectura, entorno y tamaños.
- `particion.csv`: asignación de muestras y pacientes a train/validación.
- `historial.csv`: pérdida train/validación, AUC por corte y paciente, accuracy,
  sensibilidad, especificidad, precisión y F1 de validación por paciente, tiempo por época.
- `mejor_modelo.pt`: pesos de la menor pérdida de validación, guardados durante el entrenamiento.
- `mejor_validacion.json`: época y métricas correspondientes a esos pesos.
- `metricas.json`: evaluación del mejor modelo por paciente, con matriz de confusión.
- `predicciones_validacion.csv`: probabilidades de cada corte del mejor modelo.
- `curvas.png`: pérdida de entrenamiento/validación y AUC por época.
- `metricas_epocas.png`: accuracy, sensibilidad, especificidad, precisión y F1 por época.
- `roc.png`: curva ROC y ROC-AUC del mejor modelo.
- `precision_recall.png`: curva precisión–recall, AP y prevalencia de referencia.
- `calibracion.png`: fracción observada frente a puntuación predicha y Brier.
- `matriz_confusion.png`: recuentos por paciente al umbral configurado.
- `umbrales.png`: sensibilidad, especificidad, precisión, F1 y accuracy frente al umbral.
- `distribucion_predicciones.png`: histograma de puntuaciones por clase real.
- `predicciones_pacientes.csv`: una fila por paciente, con puntuación, etiqueta y predicción.
- `metricas_umbrales.csv`: valores utilizados para dibujar las curvas de umbral.
- `ejecucion.json`: duración y pico de memoria CUDA si está disponible. No se mide el pico CPU/MPS.

`comparacion.json` reúne las métricas de ambos modelos. Compara AUC, sensibilidad
y especificidad **por paciente**; las pérdidas normal y ponderada tienen escalas
distintas. El umbral 0.5 es inicial, no está optimizado ni implica calibración.
Las métricas indefinidas aparecen como `null` en JSON.

Si entrenas ambas pérdidas, `comparacion_roc_pr.png` superpone sus curvas ROC y
precisión–recall en la carpeta de la ejecución. Usa los mejores modelos y las
mismas pacientes de validación, sin comparar directamente las escalas de BCE.

Las gráficas finales usan las predicciones del **mejor checkpoint** sobre la
validación interna, con el método de agregación y el umbral de esa ejecución.
No se evalúa test. En modo `--prueba` los títulos indican que es una prueba
reducida. ROC y precisión–recall muestran un aviso si falta alguna clase.
La curva de calibración es descriptiva: no modifica ni recalibra el modelo.
AP es average precision, no un área trapezoidal. Las métricas indefinidas se
representan como huecos, sin convertirlas artificialmente en cero.
Con agregación `voto`, se mantiene fijo el umbral que convierte los cortes en
votos y se varía el umbral sobre la fracción de votos por paciente.

Las imágenes se guardan automáticamente al terminar **cada pérdida**, sin abrir
ventanas. El comando de entrenamiento es el mismo y no requiere nuevas librerías.

`curvas.png` y `metricas_epocas.png` representan **todas las épocas realizadas**,
también las posteriores a la mejor época. El título indica el número de épocas
registradas y una línea vertical señala la mejor como referencia. Si solo se
entrena una época, aparecerá un punto por serie: no se pueden inventar épocas
adicionales. Si actúa la parada anticipada, se muestran las épocas completadas.
Las gráficas ROC, precisión–recall y calibración tienen otros ejes: describen
las predicciones del mejor modelo, no la evolución temporal del entrenamiento.

### Generar gráficas de un entrenamiento ya terminado

También puedes obtenerlas de los CSV guardados, **sin volver a entrenar**:

```bash
python -m src.training.graficas models/NOMBRE_DE_LA_EJECUCION
```

Sustituye el nombre por la carpeta real. También puedes pasar su subcarpeta
`normal` o `ponderada`. Conserva `configuracion.json`, `metricas.json`,
`historial.csv` y `predicciones_validacion.csv` juntos. Se regeneran las gráficas
y los CSV derivados; no se modifican los pesos ni los CSV originales. En
entrenamientos antiguos, `metricas_epocas.png` no se puede reconstruir porque
esas métricas no se guardaban por época; las demás gráficas sí.

Para el laboratorio con Conda y ROCm, activa `cancer` y conserva tu instalación
funcional de PyTorch. Este cambio de gráficas no necesita reinstalar
`requirements.txt` (que fija la versión 2.2.2 del entorno inicial del Mac).

## Otros comandos

```bash
python3 -m src.models.modelo
python3 -m src.data.ver_muestras muestra ISPY1_1001_z012 --salida models/muestra.png
python3 -m src.data.descargar_datos
python3 -m unittest discover -s tests -v
python3 -m src.training.entrenar --help
```

Si necesitas crear un entorno nuevo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

También puedes usar `poetry install` y anteponer `poetry run` a los comandos.
`poetry.lock` fija las dependencias; `requirements.txt` mantiene los mismos límites
para pip. Python compatible: 3.10–3.12. Se conserva PyTorch 2.2.2 del entorno
revisado y `numpy<2` por compatibilidad. Cada entrenamiento registra su entorno.

La estructura sigue la referencia proporcionada: `src/models/` contiene el
**código** de la red y `models/` contiene sus **pesos** y resultados. `dataset/`
y `metadata/` se conservan en la raíz para mantener las rutas del CSV original.
Las carpetas de aplicación, notebooks y presentación están preparadas, pero sus
entregables todavía no están desarrollados.

## Revisión y siguientes pasos

Lee [la revisión](docs/revision.md) y el
[enunciado oficial](docs/caso_breastdcedl.pdf). Después del entrenamiento
faltan estudiar overfitting y umbral, fijar el modelo, evaluar test, construir y
desplegar la aplicación y preparar las cinco diapositivas. La guía docente
original está en [docs/GUIA.md](docs/GUIA.md); sus ejemplos
conservan la estructura original del material proporcionado.

Datos BreastDCEDL; atribución y licencia CC BY-NC 4.0 conservadas en `LICENSE`
y en el enunciado. Uso educativo.
