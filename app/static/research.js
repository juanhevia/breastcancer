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
  auc:{file:'auc.png?v=2',scope:'V8 FINAL · VALIDACIÓN INTERNA',title:'Aprendizaje a lo largo de las épocas',alt:'AUC de entrenamiento y validación interna de v8 en las 15 épocas; selección de la época 6.',caption:'Escala completa 0–1. Un AUC mayor es mejor. La línea vertical marca la época 6, elegida por el máximo AUC interno por paciente.',reading:'La brecha de AUC crece de 0,025 en la época 6 a 0,123 en la 15. Se conserva la época 6; no se selecciona usando test.'},
  loss:{file:'perdida.png?v=2',scope:'V8 FINAL · ENTRENAMIENTO Y VALIDACIÓN',title:'Pérdida a lo largo de las épocas',alt:'Pérdida BCE por corte de entrenamiento y validación de v8.',caption:'Escala de pérdida 0,50–0,70, distinta de la escala del AUC. Ambos conjuntos evaluados sin aumentos; menor pérdida es mejor.',reading:'La pérdida valora las probabilidades; el AUC, su ordenación. El criterio de selección fue el AUC por paciente, no el mínimo de pérdida.'},
  roc:{file:'roc_interna_comparada.png',scope:'DESARROLLO · 219 PACIENTES DE VALIDACIÓN INTERNA',title:'Imágenes y variables: discriminación interna',alt:'ROC interna de imágenes, variables y combinación, con AUC 0,576, 0,689 y 0,717.',caption:'Las tres curvas usan las mismas pacientes. La combinación es un análisis exploratorio posterior; no una evaluación independiente.',reading:'El subtipo y la edad mejoran la discriminación interna. La mejora debe comprobarse en pacientes independientes; la curva no mide calibración.'},
  confusion:{file:'matriz_interna.png',scope:'V8 · VALIDACIÓN INTERNA · UMBRAL 0,356243176',title:'Aciertos y errores de respuesta al tratamiento',alt:'Matriz interna: 27 verdaderos positivos, 109 verdaderos negativos, 46 falsos positivos y 37 falsos negativos.',caption:'Positivo significa pCR: respuesta patológica completa. No clasifica presencia o ausencia de cáncer.',reading:'Detecta 27 de las 64 respuestas completas. Un falso positivo predice respuesta completa cuando queda enfermedad residual; un falso negativo omite una respuesta completa real.'},
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

document.getElementById('reset-transform').addEventListener('click',()=>{
  for(const boton of document.querySelectorAll('[data-transform]'))boton.setAttribute('aria-pressed','false');
  document.getElementById('phantoms').style.setProperty('--mirror','1');
  document.getElementById('phantoms').style.setProperty('--rotation','0deg');
});

// Demostración geométrica: solo modifica la vista, nunca los archivos de entrada.
const giro = document.getElementById('rotation-angle');
function mostrarGiro(){
  document.getElementById('rotation-value').textContent = `${giro.value}°`;
  giro.setAttribute('aria-valuetext', `${giro.value} grados`);
  document.getElementById('phantoms').style.setProperty('--rotation',`${giro.value}deg`);
}
giro.addEventListener('input', mostrarGiro);
document.getElementById('reset-transform').addEventListener('click',()=>{giro.value=0;mostrarGiro();});
// Aparición única y decorativa; el contenido sigue visible sin JavaScript.
if ('IntersectionObserver' in window) {
  const entradas = new IntersectionObserver(items=>{
    for (const item of items) if (item.isIntersecting) {
      item.target.classList.add('section-arrived'); entradas.unobserve(item.target);
    }
  },{threshold:0.08});
  document.querySelectorAll('.content-section,.preprocessing-section,.architecture-section,.evidence-section').forEach(el=>entradas.observe(el));
}
