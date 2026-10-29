'use strict';
// Esquema pedagógico de v8. No simula activaciones ni atribuciones del modelo.
const etapas = {
  entrada: ['01 / ENTRADA','Tres momentos del contraste','PRE, EARLY y LATE se dividen entre 255. V8 añade EARLY − PRE y LATE − EARLY, con signo: cinco canales para la primera convolución. Las fases no son colores RGB.','[cortes, 5, 256, 256]'],
  bloques: ['02 / EXTRACCIÓN','Patrones a distintas escalas','Cuatro bloques con dos convoluciones 3 × 3, GroupNorm y un atajo 1 × 1. Cada bloque termina con MaxPool y Dropout2d. Canales: 16, 32, 64 y 96; resolución: 128, 64, 32 y 16.','[cortes, 96, 16, 16]'],
  pooling: ['03 / RESUMEN','Media y máximo de cada canal','Se resume cada mapa de características mediante su media y su máximo espacial. Al concatenar ambos obtenemos 192 valores por corte, sin una gran capa densa sobre todos los píxeles.','[cortes, 192]'],
  salida: ['04 / CLASIFICACIÓN','Una puntuación; después, la paciente','El clasificador pasa de 192 a 64 unidades y a un logit. La función sigmoide lo convierte en una probabilidad por corte. La media de los cortes da la puntuación por paciente; el umbral fijado determina la clase.','sigmoide → media → umbral']
};
for (const boton of document.querySelectorAll('[data-stage]')) boton.addEventListener('click',()=>{
  for(const b of document.querySelectorAll('[data-stage]')){b.classList.toggle('selected',b===boton);b.setAttribute('aria-pressed',String(b===boton));}
  const e=etapas[boton.dataset.stage];
  ['stage-index','stage-title','stage-text','stage-shape'].forEach((id,i)=>document.getElementById(id).textContent=e[i]);
});
const graficas = {
  errores:{file:'errores_test.png',scope:'V8 FINAL · EJEMPLOS DESCRIPTIVOS DE TEST',title:'Dos errores y sus imágenes',alt:'Tres fases del falso positivo de mayor puntuación y del falso negativo de menor puntuación; un corte central de cada paciente.',caption:'Selección posterior al test: falso positivo de mayor puntuación y falso negativo de menor puntuación. Corte central disponible; escala original. Fuente: BreastDCEDL, adaptación docente, CC BY-NC 4.0.',reading:'Las puntuaciones 0,376 y 0,258 promedian todos los cortes de cada paciente. Las imágenes no explican causalmente el error y no se usaron para modificar el modelo.'},
  auc:{file:'auc.png',scope:'V8 FINAL · VALIDACIÓN INTERNA',title:'Aprendizaje a lo largo de las épocas',alt:'AUC de entrenamiento y validación interna de v8 en las 15 épocas; selección de la época 6.',caption:'Todas las épocas. La línea señala el checkpoint elegido por el máximo AUC interno por paciente.',reading:'La brecha de AUC crece de 0,025 en la época 6 a 0,123 en la 15. Se conserva la época 6; no se selecciona usando test.'},
  loss:{file:'perdida.png',scope:'V8 FINAL · ENTRENAMIENTO Y VALIDACIÓN',title:'Pérdida a lo largo de las épocas',alt:'Pérdida BCE por corte de entrenamiento y validación de v8.',caption:'Ambos conjuntos evaluados sin aumentos y en modo evaluación. Menor pérdida es mejor.',reading:'La pérdida valora las probabilidades; el AUC, su ordenación. El criterio de selección fue el AUC por paciente, no el mínimo de pérdida.'},
  roc:{file:'roc_test.png',scope:'V8 FINAL · 176 PACIENTES DE TEST',title:'Discriminación en pacientes reservadas',alt:'ROC de test con AUC 0,504 e intervalo del 95 % 0,410 a 0,597.',caption:'Una probabilidad media por paciente. IC 95 % por bootstrap estratificado: 2.000 remuestreos, semilla 42.',reading:'El intervalo incluye 0,5. Este test no demuestra discriminación útil. Cambiar el umbral no aumenta el AUC.'},
  confusion:{file:'matriz_confusion_test.png',scope:'V8 FINAL · UMBRAL FIJO 0,356243176',title:'Aciertos y errores en test',alt:'Matriz de test: 10 verdaderos positivos, 90 verdaderos negativos, 33 falsos positivos y 43 falsos negativos.',caption:'El umbral maximiza F1 macro exclusivamente en validación interna. No se reajustó con test.',reading:'Detecta 10 de las 53 pacientes con pCR (18,9 %), con 33 falsos positivos. La precisión de las predicciones positivas es 23,3 %.'},
  calibracion:{file:'calibracion_test.png',scope:'V8 FINAL · CALIBRACIÓN OBSERVADA EN TEST',title:'Puntuación frente a frecuencia observada',alt:'Diagrama de calibración de v8 en cinco grupos por cuantiles del test.',caption:'Cinco grupos de puntuaciones por cuantiles. Brier: 0,2124. No se ha ajustado un calibrador con test.',reading:'La diagonal representa coincidencia entre puntuación media y frecuencia observada. Con 176 pacientes hay incertidumbre; estas puntuaciones no son probabilidades clínicas validadas.'},
  cv:{file:'comparacion_cv.png',scope:'DESARROLLO · CINCO FOLDS EXTERNOS',title:'Comparación de arquitecturas y pérdidas',alt:'AUC por fold y media más menos desviación de v2, v7 y v8.',caption:'Cada punto es un fold; el rombo indica la media y la barra ± desviación estándar, no un intervalo de confianza.',reading:'V8 normal obtiene 0,543 ± 0,048 frente a 0,533 ± 0,044 de v7 normal. La diferencia es pequeña y no demuestra superioridad sólida.'}
};
for(const boton of document.querySelectorAll('[data-chart]')) boton.addEventListener('click',()=>{
  for(const b of document.querySelectorAll('[data-chart]')){b.classList.toggle('selected',b===boton);b.setAttribute('aria-pressed',String(b===boton));}
  const g=graficas[boton.dataset.chart], img=document.getElementById('research-chart');
  img.src='/static/evidencia/v8/'+g.file;img.alt=g.alt;
  document.getElementById('chart-download').href=img.src;
  for(const [id,value] of Object.entries({'chart-scope':g.scope,'chart-title':g.title,'chart-caption':g.caption,'chart-reading':g.reading}))document.getElementById(id).textContent=value;
});
(async()=>{
  try{
    const r=await fetch('/static/evidencia/v8/datos.json');if(!r.ok)return;const datos=await r.json();
    const slider=document.getElementById('epoch-slider'), historial=datos.historial;
    if(!Array.isArray(historial)||!historial.length)return;
    slider.max=historial.length;slider.value=datos.mejor_epoca;
    function actualizar(){const h=historial[Number(slider.value)-1],fmt=n=>n.toFixed(3).replace('.',',');
      document.getElementById('epoch-label').textContent=`Época ${h.epoca} / ${historial.length}${h.epoca===datos.mejor_epoca?' · elegida':''}`;
      document.getElementById('epoch-train').textContent=fmt(h.train_auc_paciente);
      document.getElementById('epoch-val').textContent=fmt(h.val_auc_paciente);
      document.getElementById('epoch-gap').textContent=fmt(h.train_auc_paciente-h.val_auc_paciente);
    }
    slider.addEventListener('input',actualizar);actualizar();document.getElementById('epoch-inspector').hidden=false;
  }catch{/* Las gráficas PNG siguen disponibles si falta el historial. */}
})();
for(const boton of document.querySelectorAll('[data-transform]'))boton.addEventListener('click',()=>{
  const activo=boton.getAttribute('aria-pressed')!=='true';boton.setAttribute('aria-pressed',String(activo));
  const property=boton.dataset.transform==='flip'?'--mirror':'--rotation';
  document.getElementById('phantoms').style.setProperty(property,boton.dataset.transform==='flip'?(activo?'-1':'1'):(activo?'10deg':'0deg'));
});
