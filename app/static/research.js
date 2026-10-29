'use strict';
// Esquema pedagógico de v7. No simula activaciones ni atribuciones del modelo.
const etapas = {
  entrada: ['01 / ENTRADA','Tres momentos del contraste','Las fases PRE, EARLY y LATE forman los tres canales de cada corte. Las intensidades se dividen entre 255; no se tratan como colores RGB.','[cortes, 3, 256, 256]'],
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
  auc:{file:'auc_v7.png',scope:'V7 ORIGINAL · VALIDACIÓN INTERNA',title:'Aprendizaje a lo largo de las épocas',alt:'AUC de entrenamiento y validación interna en las 15 épocas de v7; selección de la época 6.',caption:'Se muestran todas las épocas. La línea vertical marca la época 6, seleccionada por el mayor AUC interno, no por el resultado del test.',reading:'Después de la época 6, el AUC de entrenamiento sigue subiendo mientras el de validación baja. Esa separación apoya conservar el checkpoint anterior.'},
  loss:{file:'perdida_v7.png',scope:'V7 ORIGINAL · ENTRENAMIENTO Y VALIDACIÓN INTERNA',title:'Pérdida a lo largo de las épocas',alt:'Pérdida BCE de entrenamiento y validación interna de v7 por época.',caption:'Pérdida BCE evaluada sin aumentos y en modo evaluación en ambos conjuntos. Se representan todas las épocas registradas; menor es mejor.',reading:'La pérdida mide el error de las probabilidades y el AUC mide su ordenación. No tienen por qué mejorar a la vez; aquí el criterio de selección fue el AUC por paciente.'},
  roc:{file:'roc_v7.png',scope:'V7 ORIGINAL · 219 PACIENTES DE VALIDACIÓN INTERNA',title:'Sensibilidad y falsos positivos',alt:'Curva ROC de v7 en validación interna, AUC 0,579 y diagonal de referencia aleatoria.',caption:'Una puntuación por paciente, obtenida como media de sus cortes. Esta ROC es de validación interna y no del test final.',reading:'Cada punto corresponde a un umbral. Detectar más casos con pCR suele implicar más falsos positivos. Cambiar el umbral mueve el punto de decisión, pero no aumenta el AUC.'},
  cv:{file:'auc_cv.png',scope:'DESARROLLO · 5 FOLDS EXTERNOS',title:'Comparación entre modelos y pérdidas',alt:'Comparación del AUC externo de v2 y v7 con pérdida normal y ponderada, en cinco folds.',caption:'Cada combinación se evalúa en los mismos cinco folds externos, con selección interna separada. Media ± desviación estándar entre folds; no es un intervalo de confianza.',reading:'V7 con pérdida normal obtiene la mayor media entre estas cuatro combinaciones: 0,533. Las diferencias son pequeñas y los resultados no demuestran una alta capacidad predictiva.'}
};
for(const boton of document.querySelectorAll('[data-chart]')) boton.addEventListener('click',()=>{
  for(const b of document.querySelectorAll('[data-chart]')){b.classList.toggle('selected',b===boton);b.setAttribute('aria-pressed',String(b===boton));}
  const g=graficas[boton.dataset.chart], img=document.getElementById('research-chart');
  img.src='/static/evidencia/'+g.file;img.alt=g.alt;
  document.getElementById('chart-download').href=img.src;
  for(const [id,value] of Object.entries({'chart-scope':g.scope,'chart-title':g.title,'chart-caption':g.caption,'chart-reading':g.reading}))document.getElementById(id).textContent=value;
});
(async()=>{
  try{
    const r=await fetch('/static/evidencia/datos.json');if(!r.ok)return;const datos=await r.json();
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
