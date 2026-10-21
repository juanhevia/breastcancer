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

Se mantienen tres figuras principales; la primera incorpora ambas AUC. Los
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
todas las épocas, además de las tres figuras habituales. Los resultados van
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

### Tres figuras principales, sin duplicados por modelo

Cada ejecución crea una carpeta **`models/<ejecucion>/graficas/`** con tres
figuras. Si entrenas normal y ponderada, ambas aparecen en las mismas figuras:

| Archivo | Qué explica | Cómo se lee |
|---|---|---|
| `01_evolucion_del_aprendizaje.png` | Pérdida de entrenamiento/validación y AUC por paciente durante todas las épocas | Menor pérdida y mayor AUC son preferibles; la línea vertical indica la época elegida. Cada pérdida tiene su propia fila. |
| `02_capacidad_para_distinguir_pcr.png` | ROC y precisión–sensibilidad de los modelos elegidos | Compara normal y ponderada en las mismas pacientes. Estas curvas recorren umbrales, no épocas. |
| `03_aciertos_y_errores.png` | Pacientes bien y mal clasificadas con el umbral configurado | Filas: resultado real; columnas: predicción. Incluye nombres de los errores, recuentos y porcentajes dentro de cada clase real. |

En las figuras, **pCR significa respuesta patológica completa**. Una predicción
de pCR no significa una predicción de presencia de cáncer. Los títulos, ejes y
notas explican qué se mide y cómo interpretarlo. `LEEME_GRAFICAS.md` acompaña las
imágenes y registra qué época, criterio, agregación y umbral se usaron.

Las figuras se actualizan automáticamente al terminar **cada pérdida**. Mientras
solo ha terminado la normal, muestran ese modelo; cuando termina la ponderada,
las mismas tres figuras incluyen ambos. No se crea otra copia por cada métrica.

Se muestran **todas las épocas**, incluidas las posteriores a la elegida. Una
sola época produce un punto; los valores indefinidos quedan como huecos. La AUC
por corte y las demás métricas por época siguen en `historial.csv`; la figura
principal se centra en la AUC por paciente, que es la unidad de evaluación.
En v3, la figura principal también muestra la AUC de entrenamiento y utiliza
`train_eval_loss` para comparar las pérdidas sin aumentos ni dropout. La pérdida
durante la optimización sigue guardada en `train_loss`.
No se suavizan ni se alteran los resultados.

ROC y precisión–sensibilidad requieren ambas clases. La referencia de la segunda
es la proporción real de pCR. AP es average precision, no el área trapezoidal.
Las figuras usan validación interna, no test. En modo `--prueba` llevan un aviso
visible de que el subconjunto no estima el rendimiento real.

### Regenerar las figuras sin entrenar

Desde la raíz del proyecto, indica la carpeta que contiene `normal/` y
`ponderada/` para obtener el informe conjunto:

```bash
python -m src.training.graficas models/NOMBRE_DE_LA_EJECUCION
```

También admite una subcarpeta `normal` o `ponderada` para un informe de un único
modelo. Se necesitan `configuracion.json`, `metricas.json`, `historial.csv` y
`predicciones_validacion.csv`. Se actualizan las figuras y los CSV derivados;
los pesos y datos originales se conservan. Las imágenes de versiones anteriores
permanecen donde estaban; el informe nuevo está dentro de `graficas/`.

### Diagnósticos adicionales, solo si los necesitas

```bash
python -m src.training.graficas models/NOMBRE_DE_LA_EJECUCION --detalle
```

Añade **una hoja por modelo** dentro de `graficas/detalle/`, con cuatro paneles:
calibración, efecto del umbral, distribución de puntuaciones y otras métricas
por época. Son 3 figuras principales más 2 hojas adicionales si hay dos modelos.
Si un historial antiguo no contiene las métricas por época, ese panel lo indica.

La calibración usa intervalos de puntuaciones; pocos puntos significan pocos
intervalos ocupados, no pocas épocas. No recalibra el modelo. Las puntuaciones
no garantizan probabilidades calibradas, especialmente con BCE ponderada o
agregación por voto. Con `voto`, las curvas de umbral mantienen fijo el umbral que
convierte cada corte en voto y varían el umbral sobre la fracción de votos.

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
