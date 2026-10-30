# BreastDCEDL: CNN para predecir pCR

Proyecto docente con una CNN 2D desde cero. Entrada: PRE, EARLY y LATE, en ese
orden, apiladas en un tensor de 3 × 256 × 256 y divididas entre 255.

## Organización

```text
.github/workflows/          Comprobaciones automáticas
app/                        Interfaz HTML y servidor Python de inferencia
docs/                       PDF, guía y revisión del código
models/                     Pesos y resultados de cada ejecución
notebooks/                  Exploración y análisis
presentations/              Presentación final
src/
  data/                     utils_caso.py, descargar_datos.py, ver_muestras.py
  models/modelo.py          Arquitecturas propias: baseline, v2, v3 y v5
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
usan aumentos de datos en este modo: es el baseline inicial.

## Segundo experimento: mejorar la AUC por paciente

Las ejecuciones completas recibidas del laboratorio dieron una AUC interna de
aproximadamente **0,5615 (normal) y 0,5619 (ponderada)**, ambas en la época 2.
Después bajaba la pérdida de entrenamiento sin mejorar la validación. Añadir
épocas al mismo entrenamiento no ha mostrado una mejora.

`--experimento v2` activa una CNN propia de cuatro bloques, aumentos moderados
solo en entrenamiento, AdamW con regularización y reducción automática del lr.
Guarda la **máxima AUC por paciente** y para tras 12 épocas sin superarla, hasta
un máximo de 40. Si hay empate, conserva la primera época con esa AUC.
Se entrenan las pérdidas normal y ponderada de forma independiente.

En el laboratorio, después de subir los cambios a GitHub y actualizar la copia
local, ejecuta desde la raíz del repositorio:

```bash
git pull
conda activate cancer
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m src.training.entrenar --experimento v2 --perdida ambas --dispositivo cuda
```

`cuda` es también el nombre que PyTorch usa para tu GPU AMD con ROCm. Conserva
el PyTorch que ya funciona allí: **no reinstales requirements.txt**.

El experimento empieza con pesos nuevos porque cambia la arquitectura; no
carga el checkpoint anterior. Cada ejecución crea su propia carpeta
`models/<fecha_hora>_v2/` y conserva los resultados anteriores. No se promete
superar 0,70: hay que medirlo en el conjunto completo y comprobar que la mejora
se sostiene en otras particiones internas, manteniendo test reservado.

Para comprobar antes que funciona en la GPU (no mide rendimiento):

```bash
python -m src.training.entrenar --experimento v2 --epocas 1 --prueba --batch-size 4 --dispositivo cuda
```

Los argumentos explícitos prevalecen sobre el experimento. Por ejemplo,
`--epocas 60` cambia el máximo, `--modelo baseline` conserva la CNN original,
`--no-aumentos` desactiva los aumentos y `--seleccion loss` selecciona por BCE.
El comando original sin `--experimento v2` mantiene el comportamiento anterior.
Consulta la [justificación y el protocolo de comparación](docs/experimento_v2.md).

## Tercer experimento: v3 con diferencias entre fases

V3 prueba una CNN más pequeña que v2, con representación interna PRE,
EARLY−PRE y LATE−EARLY, y una media espacial 2×2. La entrada externa sigue
siendo las tres fases originales /255; no se añaden variables clínicas.
Los cambios buscan mejorar el aprendizaje y reducir el sobreajuste, pero
**no garantizan AUC 0,70**.

Después de sincronizar el código, con el entorno `cancer` activado y conservando
su instalación de PyTorch ROCm:

```bash
python -m src.training.entrenar --experimento v3 --perdida ambas --dispositivo cuda
```

Entrena desde cero hasta 40 épocas por pérdida, con paciencia 12. Mantiene los
hiperparámetros de v2 y añade evaluación completa de train al final de cada época,
sin aumentos ni dropout. Así se comparan AUC y pérdidas de train y validación con
los mismos pesos. Esta evaluación tarda un tiempo adicional y puede activarse
también para v2 con `--evaluar-train`.

Se exportan figuras individuales; la de AUC incorpora entrenamiento y validación. Los
resultados van a `models/<fecha_hora>_v3/`, sin sustituir los anteriores.
Consulta la [arquitectura, justificación y límites de v3](docs/experimento_v3.md).

## Cuarto experimento: v4 aprende por paciente

V3 normal obtuvo AUC de validación 0,550, por debajo del 0,612 de v2.
V4 conserva la arquitectura de **v2** y entrena pesos nuevos. El cambio está
en el aprendizaje: primero promedia las probabilidades de todos los cortes de
cada paciente y después calcula su pérdida. Así el objetivo coincide con la
unidad de evaluación, en lugar de exigir la etiqueta de la paciente a cada
corte de forma independiente. Todavía no se ha demostrado que mejore la AUC.

En la terminal integrada de VS Code en el Mac, desde la raíz del proyecto:

```bash
python3 -m src.training.entrenar --experimento v4 --perdida ambas --dispositivo cpu --batch-size 8
```

Entrena **normal y después ponderada**, hasta 40 épocas cada una y parada tras
12 sin mejorar. El entrenamiento usa 2 pacientes completas por paso; batch-size
solo controla el lote de evaluación en este modo. La ponderación se calcula
con las pacientes de train. Se guardan ambas AUC y pérdidas por paciente en
todas las épocas, además de las figuras individuales. Los resultados van
a `models/<fecha_hora>_v4/`. En el laboratorio, tras sincronizar el código,
usar `python` del entorno `cancer` y `--dispositivo cuda`.

Para una prueba corta, añadir `--prueba --epocas 2`; no mide rendimiento.
Los entrenamientos completos en CPU pueden tardar horas. No se promete AUC
0,70 ni menor tiempo que v3. Consulta la [hipótesis, comandos y comparación de v4](docs/experimento_v4.md).

## Quinto experimento: v5 a media resolución

V5 conserva las capas entrenables de v2, pero añade una media 2×2 dentro de
la red para trabajar a 128×128. También prueba giros de 90° conjuntos de las
tres fases en entrenamiento. Busca reducir el coste y la dependencia de la
orientación y el detalle fino; puede perder información útil y **todavía no
se ha demostrado que mejore el AUC**.

Vuelve a BCE por corte y mantiene evaluación por paciente. Se guardan las
curvas completas de train y validación. En la terminal integrada de VS Code,
con la ejecución anterior detenida y desde la raíz del proyecto:

```bash
python3 -m src.training.entrenar --experimento v5 --perdida ambas --dispositivo cpu --batch-size 8
```

Entrena normal y después ponderada, hasta 40 épocas cada una. V5 para tras
8 épocas sin superar en más de 0,002 la última referencia significativa.
Los avances pequeños se acumulan para esa comparación; cualquier récord de
AUC se guarda como mejor checkpoint, aunque no reinicie la paciencia.
El CSV incluye el contador `epocas_sin_mejora`: la parada ya no se deduce
solo de la época del mejor checkpoint. Los experimentos anteriores conservan
su criterio original por defecto.

Para probar el funcionamiento, añadir `--prueba --epocas 2`. No estima el AUC
del conjunto completo. La reducción de resolución busca acelerar el cálculo,
pero el entrenamiento completo puede seguir durando horas. Resultados en
`models/<fecha_hora>_v5/`. Consulta la [arquitectura, protocolo y comandos de v5](docs/experimento_v5.md).

## Sexto experimento: v6 con contraste conjunto

V6 conserva el cuerpo de v2 y recibe las tres fases originales junto con una
copia ajustada de intensidad/contraste: seis canales internos. La ventana
p5–p95 es común a PRE/EARLY/LATE, con ganancia máxima 3; no sustituye los
originales ni usa saturación RGB. Se aplica dentro del modelo también en
inferencia. Se recupera lr=0,0001 de v2, no el 0,0003 de v2_lr.

```bash
python -m src.training.entrenar --experimento v6 --perdida ambas --dispositivo cuda --batch-size 16
```

Hasta 40 épocas por pérdida, paciencia 12, con pesos por época y figuras
individuales PNG. Añadir `--prueba --epocas 1` para verificar primero la GPU.
El entrenamiento completo todavía no ha demostrado mejorar v2 ni alcanzar 0,70.
Consulta el [protocolo, preprocesado y límites de v6](docs/experimento_v6.md).

## Séptimo experimento: v7 residual, paciencia 9

Nueva CNN propia desde cero con cuatro bloques residuales de 16/32/64/96 canales.
Recibe las tres fases originales /255, con el cuerpo convolucional y la última
etapa más compactos: 232.513 parámetros entrenables, frente a 310.513 de v2.
Los atajos buscan facilitar el aprendizaje; su mejora de AUC está por comprobar.

```bash
python -m src.training.entrenar --experimento v7 --perdida ambas --dispositivo cuda --batch-size 16
```

Máximo 40 épocas por pérdida y **parada tras 9 épocas consecutivas sin mejorar
el AUC por paciente**. Los empates cuentan; una mejora reinicia el contador.
Conserva pesos por época y figuras individuales. Para la prueba corta, añadir
`--prueba --epocas 1`. La preferencia para próximos ensayos es paciencia 9;
los presets antiguos conservan sus valores para poder reproducirlos.
Consulta la [arquitectura y protocolo de v7](docs/experimento_v7.md).

## V2 original con entrenamiento corto

`v2_corto` conserva la arquitectura de v2 y limita el entrenamiento a **15
épocas**, con paciencia 9. El corte procede de la curva histórica de pérdida
normal, antes de su separación sostenida a partir de la época 16. No garantiza
un AUC mayor: el histórico tenía 0,568 en la 15 y 0,612 en la 24.

```bash
python -m src.training.entrenar --experimento v2_corto --perdida normal --dispositivo cuda --batch-size 16
```

Guarda todos los pesos de las épocas completadas y añade evaluación de train
sin aumentos/dropout para comparar mejor sus curvas con validación. Para usar
exactamente la época 15, cargar `normal/epocas/epoca_015.pt` si se ha completado;
`mejor_modelo.pt` puede corresponder a otra época. El historial antiguo queda
conservado y la ejecución nueva guarda sus propios resultados.
Consulta el [análisis del corte y el uso de los pesos](docs/experimento_v2_corto.md).

## Resultados guardados

Las pruebas realizadas en casa y las referencias recibidas del laboratorio se
pueden consultar en [resultados publicados](docs/resultados/README.md), con
historiales, métricas y gráficas disponibles. El índice distingue pruebas cortas,
entrenamientos finalizados e intentos sin cierre registrado.

Para revisar integridad, capacidad de aprendizaje y resultados por cohorte,
consulta el [diagnóstico y siguiente control de v2](docs/diagnostico_y_siguiente_control.md).
`python3 -m src.training.diagnostico` analiza metadatos y predicciones existentes;
`--memorizacion` añade una prueba pequeña sobre las mismas imágenes entrenadas,
sin valor como estimación de generalización. El nuevo control `--experimento v2_lr`
conserva v2 y cambia lr de 0,0001 a 0,0003; todavía no ha demostrado mejorar su AUC.

Cada ejecución crea `models/<fecha_hora>/normal/` y/o `ponderada/` con:

- `configuracion.json`: hiperparámetros, arquitectura, entorno y tamaños.
- `particion.csv`: asignación de muestras y pacientes a train/validación.
- `historial.csv`: pérdida train/validación, AUC por corte y paciente, accuracy,
  sensibilidad, especificidad, precisión y F1 de validación por paciente, tiempo
  y tasa de aprendizaje utilizada en cada época.
- `mejor_modelo.pt`: pesos según el criterio elegido: menor pérdida en baseline,
  mayor AUC por paciente en v2; guardados durante el entrenamiento.
- `mejor_validacion.json`: época y métricas correspondientes a esos pesos.
- `metricas.json`: evaluación del mejor modelo por paciente, con matriz de confusión.
- `predicciones_validacion.csv`: probabilidades de cada corte del mejor modelo.
- Con `--evaluar-train` (activado en v3, v4 y v5): `predicciones_entrenamiento.csv` del
  mejor modelo; `train_eval_loss`, `train_auc_paciente` y `brecha_auc` en el
  historial; métricas de train y brecha de la época elegida en `metricas.json`.
- `predicciones_pacientes.csv`: una fila por paciente, con puntuación, etiqueta y predicción.
- `metricas_umbrales.csv`: valores utilizados para dibujar las curvas de umbral.
- `ejecucion.json`: duración y pico de memoria CUDA si está disponible. No se mide el pico CPU/MPS.

Las predicciones de validación y metricas.json se guardan con cada nuevo mejor
checkpoint, sin esperar al final. Si se detiene con Ctrl+C tras una época completa,
se generan las gráficas disponibles y se escribe interrupcion.json; no se inicia
la siguiente pérdida. No se guarda un estado de optimizador para reanudar.

`comparacion.json` reúne las métricas de ambos modelos. Compara AUC, sensibilidad
y especificidad **por paciente**; las pérdidas normal y ponderada tienen escalas
distintas. El umbral 0.5 es inicial, no está optimizado ni implica calibración.
Las métricas indefinidas aparecen como `null` en JSON.

### Comparación v2/v7 con cinco folds

Para comparar ambas arquitecturas con selección interna de época/umbral y
evaluación externa separada, sin tocar el test oficial:

```bash
python -m src.training.comparar_cv --dispositivo cuda --perdida ambas --solo-plan
python -m src.training.comparar_cv --dispositivo cuda --perdida ambas
```

Son **20 entrenamientos**, con máximo 40 épocas y paciencia 9 para ambos
modelos. Se guardan los `.pt` de todas las épocas. `--reanudar` omite trabajos
completos y reinicia desde cero el incompleto. `--perdida normal` reduce el
protocolo a 10 entrenamientos; usar una carpeta `--salida` diferente.

Los resúmenes están en `models/comparacion_v2_v7/COMPARACION.md` y
`resultados_por_fold.csv`. Solo se generan tablas e historiales, sin nuevas
copias de gráficas. Consulta el [protocolo y sus límites](docs/comparacion_v2_v7.md):
se entrena con tres folds, se selecciona con uno y se evalúa con otro; la
desviación entre folds no es un intervalo de confianza ni sustituye al test final.

### Gráficas individuales para la presentación

Cada ejecución crea `models/<ejecucion>/graficas/normal/` y, cuando corresponde,
`graficas/ponderada/`. Cada archivo contiene **una sola gráfica**, con título,
ejes y leyenda breve. Se exporta únicamente en **PNG a 300 dpi**, sin copias en otros formatos.

| Archivo (PNG) | Contenido |
|---|---|
| `01_perdida` | Pérdida de entrenamiento y validación durante todas las épocas. |
| `02_auc_por_epoca` | ROC-AUC por paciente durante todas las épocas; train cuando se registró. |
| `03_roc` | ROC de validación del mejor modelo y su AUC. |
| `04_precision_sensibilidad` | Precisión–sensibilidad del mejor modelo y AP. |
| `05_matriz_confusion` | Recuentos de pacientes y porcentajes por fila, con umbral indicado. |

La línea vertical en las curvas por época señala el mejor modelo, sin ocultar
las otras épocas. Se usa `train_eval_loss` cuando existe para comparar pérdidas
sin aumentos ni dropout; en historiales antiguos se usa la pérdida de optimización.
No se suavizan los resultados. Las pérdidas normal y ponderada tienen escalas distintas.

Las explicaciones, la época elegida, el criterio, la agregación y el umbral se
guardan en `graficas/LEEME_GRAFICAS.md`. Las figuras indican validación interna;
las pruebas cortas llevan «Prueba reducida». No son resultados de test.

Se generan al terminar cada pérdida o al interrumpir con Ctrl+C tras una época
completa. Para regenerarlas sin entrenar:

```bash
python -m src.training.graficas models/NOMBRE_DE_LA_EJECUCION
```

También se admite una carpeta `normal/` o `ponderada/`. Con configuración e
historial se pueden rehacer pérdida y AUC. ROC, precisión–sensibilidad y matriz
requieren además las predicciones de validación y sus métricas. No se reconstruyen
predicciones ausentes. Los tres paneles antiguos, si existen en `graficas/`, se
conservan en `graficas/formato_anterior/`.

Para diagnósticos adicionales:

```bash
python -m src.training.graficas models/NOMBRE_DE_LA_EJECUCION --detalle
```

Añade cuatro gráficas separadas en la subcarpeta `detalle/` de cada pérdida:
calibración, efecto del umbral, distribución de puntuaciones y métricas por época.
La calibración agrupa puntuaciones, no épocas; no recalibra el modelo.

### Pesos de todas las épocas

Desde esta versión, cada época completada guarda sus pesos en:

```text
models/<ejecucion>/normal/epocas/epoca_001.pt
models/<ejecucion>/normal/epocas/epoca_002.pt
...
models/<ejecucion>/ponderada/epocas/epoca_001.pt
```

`mejor_modelo.pt` se conserva por separado. Cada `.pt` es un `state_dict` en CPU,
compatible con `modelo.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))`
tras crear la arquitectura indicada en `configuracion.json`. Las métricas de cada
época están en `historial.csv`. Estos archivos guardan pesos, no el estado completo
del optimizador para reanudar exactamente un entrenamiento.

Se aplica a **nuevos procesos de entrenamiento**. Actualizar el repositorio no
modifica uno que ya está ejecutándose. No es posible recuperar pesos de épocas
antiguas que nunca se guardaron. Los `.pt` siguen excluidos de GitHub y se conservan
en el ordenador donde se entrena. Para v2, 80 épocas guardadas (40 por pérdida)
ocupan aproximadamente 100 MB de pesos, además de los mejores modelos.

Para el laboratorio con Conda y ROCm, activa `cancer` y conserva tu instalación
funcional de PyTorch. Las gráficas y el experimento v2 no necesitan reinstalar
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
La [aplicación HTML](https://breastcancer-bn0a.onrender.com/) está publicada con
v8 congelado y el complemento exploratorio de edad y subtipo. Los pesos para
inferencia están en `app/paquetes/v8/`. [Instrucciones de despliegue](app/RENDER.md).

## Entrega

- [Aplicación pública](https://breastcancer-bn0a.onrender.com/).
- [Presentación visual de cinco diapositivas](presentations/entrega/BreastDCEDL_UAX_5_equilibrada.pptx).
- [Memoria](docs/memoria/Memoria_BreastDCEDL_Juan_Hevia.docx): pendiente de actualizar el enlace público y el complemento clínico.
- [Complemento de variables: método y limitaciones](docs/complemento_variables.md).

El modelo final, umbral y test ya están fijados. La AUC interna de la CNN es
0,576 y la del test disponible es 0,504. La combinación clínica alcanza 0,717
solo en validación interna reutilizada: no demuestra generalización independiente.

La guía docente
original está en [docs/GUIA.md](docs/GUIA.md); sus ejemplos
conservan la estructura original del material proporcionado.

Datos BreastDCEDL; atribución y licencia CC BY-NC 4.0 conservadas en `LICENSE`
y en el enunciado. Uso educativo.
