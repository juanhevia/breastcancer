# Memoria con los resultados finales de v8

`Memoria_BreastDCEDL_Juan_Hevia.docx` es el documento editable; `memoria.md`
conserva el texto fuente. Incluye comparación completa, entrenamiento final,
selección de época y umbral, test con IC, métricas por clase/cohorte, calibración
y dos casos erróneos. No se repite el test ni se ajusta el modelo con sus datos.

Las figuras finales se guardan solo en PNG, en `app/static/evidencia/v8/`;
la memoria utiliza esos mismos archivos. `figuras/` conserva la evidencia
histórica de v7, separada de la versión final.

Para regenerar el Word, con python-docx disponible:

```bash
python docs/memoria/generar_memoria.py
```

El generador usa las figuras existentes y los datos agregados de la web. No
entrena, no carga pesos ni evalúa imágenes. `fuentes.json` conserva los hashes.
Una edición manual del Word debe trasladarse al Markdown antes de regenerarlo.

Pendiente para la entrega completa: publicación y comprobación de la URL de la
app, presentación de cinco diapositivas y revisión de los datos académicos de
portada. El documento declara el alcance local de la aplicación; no inventa URL.
