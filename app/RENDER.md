# Despliegue gratuito en Render

Crear un **Web Service**, repositorio `juanhevia/breastcancer`, rama `main`.

| Campo | Valor |
|---|---|
| Language | Python 3 |
| Root Directory | Vacío |
| Build Command | `pip install -r app/requirements-render.txt` |
| Start Command | `python -m app.render` |
| Compute | Free ($0/month) |
| Environment: PYTHON_VERSION | `3.12.8` |

El paquete `app/paquetes/v8/` contiene los pesos congelados y métricas agregadas;
no incluye imágenes, identificadores ni filas de pacientes. El complemento
tabular está en `app/paquetes/complemento_v8.json`.

Se ejecuta en CPU, un proceso y un corte por lote, con la misma media por paciente,
pesos y umbral. No entrena ni accede al dataset. El inicio falla si falta el modelo.

Tras desplegar, comprobar `/api/modelo` y `/api/complemento` (disponible=true),
abrir la página y probar un conjunto completo PRE/EARLY/LATE de una sola paciente.
Validar el consumo real en Render: la comprobación local no garantiza que quepa
en sus 512 MB. El plan gratuito se duerme tras 15 minutos sin tráfico y puede
tardar aproximadamente un minuto en reactivarse.
