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

La sección de evidencia conserva resultados históricos de desarrollo,
independientes del modelo conectado: pérdida y AUC de todas las épocas de v7,
ROC interna por paciente y comparación CV de v2/v7. Cada gráfica se muestra una
vez, solo en PNG. Un selector permite consultar las métricas de cada época.
No presenta la validación interna ni la CV como test final. V8 figura pendiente.
`python -m app.exportar_evidencia` genera la evidencia desde las fuentes locales
documentadas en el script; exporta curvas y valores agregados, sin imágenes ni
identificadores de pacientes. Los PNG incluidos permiten servir la web sin
disponer de esas fuentes. Al cerrar la selección hay que actualizar explícitamente
esta evidencia, sus títulos y la arquitectura si cambia.

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

Los paquetes se identifican por defecto como «Candidato provisional». El campo
opcional `estado_modelo` del manifiesto permite documentar su estado definitivo.
En el Mac existe un paquete local `models/app_v7_candidato_20261027` para probar
el flujo completo con v7 entrenado. No está incluido en Git ni selecciona el
modelo final; la página lo etiqueta como provisional.

Con el paquete preparado:

```bash
BREAST_MODEL_DIR=models/app_modelo python -m uvicorn app.servidor:app --host 127.0.0.1 --port 8000
```

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
Falta elegir el paquete final, desplegar en un servicio Python/PyTorch y verificar
la URL desde otro equipo. GitHub Pages por sí solo no ejecuta este servidor.
No se ha contratado alojamiento ni publicado un modelo provisional.

En el servidor: instalar PyTorch CPU compatible y las dependencias del proyecto
y `app/requirements.txt`, montar el paquete, definir `BREAST_MODEL_DIR` y arrancar
Uvicorn con `--host 0.0.0.0` y el puerto del proveedor. Un proceso por contenedor
inicialmente. Fijar y probar versiones en el destino antes de la defensa.

Pruebas: `python -m unittest tests.test_app -v` (requiere `httpx`). Verifican
equivalencia con el pipeline, API, errores y reproducibilidad; no rendimiento.
Referencia: [FastAPI](https://fastapi.tiangolo.com/).
