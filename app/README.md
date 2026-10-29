# Aplicación HTML

Interfaz HTML/CSS/JavaScript en `app/static/`, servida por FastAPI. El `.pt` se
ejecuta en CPU en Python. No basta con abrir `index.html` directamente.

## Interfaz

Estética de imagen médica con contraste azul noche/blanco, acentos turquesa y ámbar,
y capas PRE/EARLY/LATE animadas,
arrastre de archivos, navegación entre cortes y ampliación de las cuatro vistas.
El botón «Pausar efectos» detiene las animaciones; se respeta la preferencia de
movimiento reducido del sistema. La animación del encabezado es decorativa.
El indicador de actividad solo aparece durante una petición real.

Incluye guía de uso, tratamiento de imágenes, arquitectura interactiva, curvas,
interpretación de resultados y preguntas frecuentes. La navegación sigue la
sección visible y se adapta al móvil.

El apartado de imágenes describe el código de v7/v8: PNG gris 256×256, float32
/255, reflejo horizontal al 50 % y giro uniforme entre −10° y +10° aplicado a las
tres fases juntas, solo durante entrenamiento. El esquema animado es sintético;
no es una imagen de paciente ni modifica las imágenes que se cargan al visor.

La evidencia muestra el v8 final: pérdida y AUC de las 15 épocas, comparación
CV completa, ROC, matriz, calibración y dos errores del test. Cada figura tiene
un único PNG en `app/static/evidencia/v8/`; el Word reutiliza esos mismos archivos.
Las métricas distinguen selección interna (AUC 0,576), CV (0,543 ± 0,048) y test
(AUC 0,504; IC 95 % 0,410–0,597). No se reajusta el modelo después del test.

`python -m app.exportar_v8 --origen models/importados_entrega_v8/models`
verifica particiones, pesos y métricas contra las predicciones guardadas y exporta
los resultados agregados. No entrena ni ejecuta inferencia. El exportador previo
`app.exportar_evidencia` conserva la evidencia histórica v7, sin cambiar v8.

La ficha y los resultados se rellenan con el paquete del modelo, sin editar el
HTML. Sin modelo conectado no se muestran predicciones ni métricas inventadas.

## Desarrollo local

Desde la raíz del repositorio, en el entorno `cancer`:

```bash
python -m pip install -r app/requirements.txt
python -m uvicorn app.servidor:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000`. Sin pesos permite inspeccionar imágenes, pero no
predice. No reinstalar los requirements generales ni cambiar PyTorch ROCm.

## Conectar el modelo elegido

Para la entrega, después de fijar modelo y umbral usando desarrollo/validación.
Admite v2/v7/v8;
necesita `mejor_modelo.pt`, `configuracion.json` y `metricas.json` de la misma
ejecución. En CV están en `seleccion_interna/`. No elegir un checkpoint por el
máximo del fold externo; fijar el procedimiento final tras la comparación.

`python -m app.preparar_modelo --help` describe cómo preparar el paquete local:
requiere `--origen`, `--umbral` y `--criterio`, y escribe en `models/app_modelo`
salvo que se indique `--destino`. No sobrescribe paquetes. No hay un umbral
preseleccionado ni envíos a internet. El manifiesto conserva la justificación y
el SHA-256 de los pesos. Los modelos de prueba se identifican como tales.

El paquete final es `models/app_modelo_v8`, época 6 y umbral 0,356243176.
Los pesos permanecen fuera de Git. La variable `BREAST_MODEL_DIR` debe apuntar
al paquete local; no al directorio de resultados de test.

```bash
BREAST_MODEL_DIR=models/app_modelo_v8 python -m uvicorn app.servidor:app --host 127.0.0.1 --port 8000
```

Los paquetes nuevos incluyen `metricas_internas_umbral.json`. Si el paquete se
preparó con la primera versión del recopilador, puede añadirse sin cambiar pesos:

```bash
cp models/final_train_v8/normal/metricas_internas_umbral.json models/app_modelo_v8/
```

La ficha muestra las métricas internas al umbral elegido cuando existe este
archivo; los paquetes históricos conservan sus métricas a 0,5, identificadas.

## Entrada y agregación

- Una paciente, entre 1 y 30 cortes, cada uno con tres fases.
- Nombres `PACIENTE_z012_PRE.png`, `_EARLY.png`, `_LATE.png`: conservar originales.
- PNG gris de 8 bits, 256×256. No se redimensiona, recolorea ni recontrasta.
- PRE/EARLY/LATE, float32 dividido por 255, igual que el Dataset.
- `eval()`, sin gradientes, lotes CPU de 8; media de probabilidades, no de logits.
- Un solo corte se identifica; no representa todos los cortes de la paciente.
- Fases y realce separados; realce es visualización, no explicación del modelo.
- Resultado con umbral, checksum, tiempo, dispositivo y puntuaciones por corte.
- Las métricas internas muestran su propio umbral, que puede diferir del desplegado.

Se rechazan fases ausentes/duplicadas, pacientes mezcladas, rutas, tamaños/tipos
incorrectos y archivos corruptos. Máximo 512 KiB/archivo y 16 MiB/petición, durante
la lectura. No se cargan pesos desde formularios ni se guardan imágenes subidas.
El JSON descargado contiene el identificador aportado: no publicarlo si procede
de la validación privada del profesor.

## Pendiente antes de entregar

La implementación local NO equivale al despliegue obligatorio por URL pública.
El paquete v8 está fijado. Falta desplegar en un servicio Python/PyTorch y verificar
la URL desde otro equipo. GitHub Pages por sí solo no ejecuta este servidor.
No se ha contratado alojamiento ni publicado un modelo provisional.

En el servidor: instalar PyTorch CPU compatible y las dependencias del proyecto
y `app/requirements.txt`, montar el paquete, definir `BREAST_MODEL_DIR` y arrancar
Uvicorn con `--host 0.0.0.0` y el puerto del proveedor. Un proceso por contenedor
inicialmente. Fijar y probar versiones en el destino antes de la defensa.

Pruebas: `python -m unittest tests.test_app -v` (requiere `httpx`). Verifican
equivalencia con el pipeline, API, errores y reproducibilidad; no rendimiento.
Referencia: [FastAPI](https://fastapi.tiangolo.com/).
