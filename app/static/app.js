'use strict';
const $ = id => document.getElementById(id);
const fases = ['pre', 'early', 'late', 'realce'];
let complemento = null, comparacion = null;
let modelo = null, archivos = [], vistas = [], resultado = null, revision = 0, ocupado = false, leyendo = false;
const porcentaje = x => (x * 100).toLocaleString('es-ES', {maximumFractionDigits: 1}) + ' %';
function lista(id, entradas) {
  const nodo = $(id); nodo.replaceChildren();
  for (const [nombre, valor] of entradas) {
    const dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = nombre; dd.textContent = valor ?? 'No disponible'; nodo.append(dt, dd);
  }
}
function error(mensaje) { $('error').textContent = mensaje; $('error').hidden = !mensaje; }
function progreso(paso) {
  ['paso-carga','paso-revision','paso-resultado'].forEach((id,i) => {
    $(id).classList.toggle('current',i===paso); $(id).classList.toggle('complete',i<paso);
    if(i===paso) $(id).setAttribute('aria-current','step'); else $(id).removeAttribute('aria-current');
  });
}
function controles() {
  document.body.classList.toggle('processing', ocupado);
  $('visor').setAttribute('aria-busy', String(ocupado));
  $('archivos').disabled = ocupado || leyendo;
  $('limpiar').disabled = ocupado || leyendo;
  $('inspeccionar').disabled = ocupado || leyendo || !archivos.length;
  $('predecir').disabled = ocupado || leyendo || !vistas.length || !modelo?.disponible;
  $('clinical-run').disabled = ocupado || leyendo || !vistas.length || !modelo?.disponible || !complemento?.disponible;
  $('clinical-age').disabled = ocupado || leyendo; $('clinical-subtype').disabled = ocupado || leyendo;
  $('corte').disabled = !vistas.length;
  const indice = Number($('corte').value);
  $('anterior').disabled = !vistas.length || indice === 0;
  $('siguiente').disabled = !vistas.length || indice >= vistas.length-1;
  $('ayuda-prediccion').textContent = !vistas.length ? 'Valida las imágenes para preparar el análisis.' :
    !modelo?.disponible ? 'Imágenes preparadas. La inferencia se activará al conectar el modelo seleccionado.' :
    `${vistas.length} ${vistas.length===1?'corte disponible: resultado limitado a ese corte.':'cortes disponibles: se promediarán sus probabilidades.'}`;
}
async function peticion(ruta, payload, extra = {}) {
  const controller = new AbortController(); const timeout = setTimeout(()=>controller.abort(),180000);
  try {
    const r = await fetch(ruta, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({files:payload, ...extra}),signal:controller.signal});
    let dato; try { dato = await r.json(); } catch { throw new Error('El servidor no devolvió una respuesta válida. Vuelve a intentarlo.'); }
    if (!r.ok) throw new Error(typeof dato.detail === 'string' ? dato.detail : 'No se pudo procesar la petición.'); return dato;
  } catch(e) {
    if(e.name==='AbortError') throw new Error('La petición ha tardado demasiado. Comprueba el servidor antes de volver a intentarlo.');
    if(e instanceof TypeError) throw new Error('No se pudo conectar con el servidor. Comprueba la conexión e inténtalo de nuevo.');
    throw e;
  } finally {clearTimeout(timeout);}
}
function leer(file) { return new Promise((resolve,reject) => {
  const reader = new FileReader(); reader.onerror = () => reject(new Error('No se pudo leer la imagen.'));
  reader.onload = () => resolve({name:file.name, content:reader.result.split(',')[1]}); reader.readAsDataURL(file);
}); }
function limpiarVista() {
  $('clinical-form').reset(); limpiarComparacion(); actualizarDemo(null);
  vistas=[]; resultado=null; $('resultado').hidden=true; $('corte').replaceChildren(new Option('— Corte —',''));
  $('titulo-paciente').textContent='Fases y realce'; $('detalle-paciente').textContent='Las tres fases se mantienen alineadas';
  $('visor-estado').textContent='Sin imágenes'; $('visor-estado').classList.remove('ready');
  for(const f of fases){$(f).hidden=true;$(f).removeAttribute('src');$(f).closest('.image-frame').classList.remove('loaded');$(f).parentElement.disabled=true;}
  progreso(0);
}
function limpiarTodo() {
  revision++; archivos=[]; $('archivos').value=''; limpiarVista(); error('');
  $('seleccion').textContent='Esperando imágenes'; $('contador').textContent='0 PNG';
  $('archivos-lista').replaceChildren(); $('archivos-lista').hidden=true; $('limpiar').hidden=true;controles();
}
async function seleccionar(files) {
  if(ocupado||leyendo)return;
  const actual=++revision; archivos=[]; limpiarVista();error('');leyendo=true;controles();
  $('contador').textContent=`${files.length} PNG`; $('limpiar').hidden=!files.length;
  $('seleccion').textContent=files.length?'Leyendo archivos…':'Esperando imágenes';
  $('archivos-lista').replaceChildren();$('archivos-lista').hidden=true;
  try {
    if(!files.length)return;
    if(files.length<3||files.length>90)throw new Error('Selecciona entre 3 y 90 imágenes, con las tres fases de cada corte.');
    if(files.some(f=>!f.name.toLowerCase().endsWith('.png')))throw new Error('Solo se admiten imágenes PNG. No cambies la extensión de una fotografía para convertirla.');
    if(files.some(f=>f.size>512*1024))throw new Error('Cada imagen debe ocupar como máximo 512 KiB.');
    if(files.reduce((n,f)=>n+f.size,0)>11*1024*1024)throw new Error('La selección supera el tamaño permitido. Elige menos cortes.');
    const preparados=await Promise.all(files.map(leer));
    if(actual===revision){
      archivos=preparados; $('seleccion').textContent=`${files.length} imágenes listas para validar`;
      const visibles=files.slice(0,6).map(f=>{const li=document.createElement('li');li.textContent=f.name;return li;});
      if(files.length>6){const li=document.createElement('li');li.textContent=`+ ${files.length-6} imágenes más`;visibles.push(li);}
      $('archivos-lista').replaceChildren(...visibles);$('archivos-lista').hidden=false;
    }
  }catch(e){if(actual===revision){error(e.message);$('seleccion').textContent='Revisa la selección';}}
  finally{leyendo=false;controles();}
}
$('archivos').addEventListener('change',()=>seleccionar([...$('archivos').files]));
$('limpiar').addEventListener('click',limpiarTodo);
const zona=$('dropzone');
let arrastres=0;
for(const tipo of ['dragenter','dragover','dragleave','drop'])zona.addEventListener(tipo,e=>{e.preventDefault();e.stopPropagation();});
zona.addEventListener('dragenter',()=>{arrastres++;if(!ocupado&&!leyendo)zona.classList.add('dragging');});
zona.addEventListener('dragleave',()=>{arrastres--;if(arrastres<=0)zona.classList.remove('dragging');});
zona.addEventListener('drop',e=>{arrastres=0;zona.classList.remove('dragging');seleccionar([...e.dataTransfer.files]);});
// Impide que soltar fuera del recuadro navegue accidentalmente a la imagen.
window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',e=>e.preventDefault());
$('inspeccionar').addEventListener('click',async()=>{
  ocupado=true;controles();error('');$('texto-inspeccionar').textContent='Validando…';$('seleccion').textContent='Comprobando fases y formato…';
  try{
    const r=await peticion('/api/inspeccionar',archivos);vistas=r.vistas;
    $('seleccion').textContent=`${r.numero_cortes} ${r.numero_cortes===1?'corte completo':'cortes completos'} · Validación correcta`;
    $('titulo-paciente').textContent=r.paciente;
    $('detalle-paciente').textContent=`${r.numero_cortes} ${r.numero_cortes===1?'corte':'cortes'} · PRE / EARLY / LATE`;
    $('visor-estado').textContent='Imágenes validadas';$('visor-estado').classList.add('ready');
    $('corte').replaceChildren(...vistas.map((v,i)=>new Option(`z${String(v.corte).padStart(3,'0')}`,i)));
    mostrarCorte();progreso(1);
  }catch(e){limpiarVista();error(e.message);$('seleccion').textContent='No se pudo validar la selección';}
  finally{ocupado=false;$('texto-inspeccionar').textContent='Validar imágenes';controles();}
});
function mostrarCorte(){
  const v=vistas[Number($('corte').value)];if(v)for(const f of fases){$(f).src=v[f];$(f).hidden=false;$(f).closest('.image-frame').classList.add('loaded');$(f).parentElement.disabled=false;}
  actualizarDemo(v); controles();
}
$('corte').addEventListener('change',mostrarCorte);
$('anterior').addEventListener('click',()=>{$('corte').value=Number($('corte').value)-1;mostrarCorte();});
$('siguiente').addEventListener('click',()=>{$('corte').value=Number($('corte').value)+1;mostrarCorte();});
for(const b of document.querySelectorAll('.image-open'))b.addEventListener('click',()=>{
  const f=b.dataset.fase, v=vistas[Number($('corte').value)];if(!v)return;
  $('imagen-ampliada').src=v[f];$('imagen-ampliada').alt=$(f).alt;
  $('ampliacion-titulo').textContent=`${f==='realce'?'Realce · EARLY − PRE':f.toUpperCase()} · Corte ${v.corte}`;$('ampliacion').showModal();
});
$('cerrar-ampliacion').addEventListener('click',()=>$('ampliacion').close());
$('ampliacion').addEventListener('click',e=>{if(e.target===$('ampliacion')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
$('predecir').addEventListener('click',async()=>{
  ocupado=true;controles();error('');$('texto-predecir').textContent='Analizando cortes…';$('resultado').hidden=true;resultado=null;
  try{
    resultado=await peticion('/api/predecir',archivos);
    $('probabilidad').textContent=porcentaje(resultado.probabilidad);
    $('clase').textContent=`Clase predicha: ${resultado.clase?'con pCR':'sin pCR'}`;
    $('alcance').textContent=resultado.alcance;
    $('score-fill').style.width=`${resultado.probabilidad*100}%`;
    $('threshold-marker').style.left=`${resultado.umbral*100}%`;
    $('threshold-label').textContent=`Umbral: ${porcentaje(resultado.umbral)}`;
    lista('datos-resultado',[['Cortes analizados',resultado.numero_cortes],['Tiempo de inferencia',resultado.segundos.toFixed(3)+' s'],['Dispositivo',resultado.dispositivo],['Modelo SHA-256',resultado.sha256]]);
    $('cortes-resultado').replaceChildren(...resultado.cortes.map(c=>{const tr=document.createElement('tr');for(const t of [`z${String(c.corte).padStart(3,'0')}`,porcentaje(c.probabilidad)]){const td=document.createElement('td');td.textContent=t;tr.append(td);}return tr;}));
    $('resultado').hidden=false;progreso(2);$('resultado').scrollIntoView({block:'start',behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
  }catch(e){resultado=null;error(e.message);$('error').scrollIntoView({block:'center'});}
  finally{ocupado=false;$('texto-predecir').textContent='Obtener estimación';controles();}
});
$('descargar').addEventListener('click',()=>{
  if(!resultado)return;const url=URL.createObjectURL(new Blob([JSON.stringify(resultado,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='resultado_paciente.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
const enlaces = [...document.querySelectorAll('.nav-link')];
function seccionActiva(enlace) {
  for (const n of enlaces) {
    n.classList.toggle('active', n === enlace);
    if (n === enlace) n.setAttribute('aria-current', 'location');
    else n.removeAttribute('aria-current');
  }
  $('seccion-actual').textContent = enlace.textContent.trim();
}
for (const a of enlaces) {a.title=a.textContent.trim();a.addEventListener('click', () => seccionActiva(a));}
// Sigue la sección visible sin modificar el historial ni interrumpir el scroll.
let scrollPendiente = false;
window.addEventListener('scroll', () => {
  if (scrollPendiente) return;
  scrollPendiente = true;
  requestAnimationFrame(() => {
    let actual = enlaces[0];
    for (const a of enlaces) {
      if (document.querySelector(a.getAttribute('href')).getBoundingClientRect().top <= 180) actual = a;
    }
    seccionActiva(actual); scrollPendiente = false;
  });
}, {passive:true});
seccionActiva(enlaces.find(a => a.getAttribute('href') === location.hash) || enlaces[0]);
(async()=>{try{
  const r=await fetch('/api/modelo');if(!r.ok)throw new Error('No se pudo consultar el modelo.');modelo=await r.json();
  $('estado').textContent=modelo.disponible ? (modelo.prueba?'Modelo de prueba reducida: sus resultados no estiman rendimiento predictivo.':`${modelo.estado_modelo || 'Candidato provisional'}: ${modelo.modelo.toUpperCase()} conectado para la demostración. El umbral procede de validación interna; la sección de evidencia documenta el v8 final.`) : 'V8 está seleccionado, pero sus pesos no están conectados a este servidor. Puedes explorar las imágenes.';
  if(modelo.disponible){
    $('modelo-badge').textContent=modelo.prueba?'Prueba técnica':(modelo.estado_modelo || 'Candidato provisional');$('modelo-badge').classList.add('ready');
    $('modelo-nombre').textContent=modelo.modelo.toUpperCase();$('modelo-umbral').textContent=modelo.umbral.toLocaleString('es-ES', {maximumFractionDigits:9});
    $('modelo-pendiente').hidden=true;
    lista('datos-modelo',[['Parámetros entrenables',modelo.parametros?.toLocaleString('es-ES')],['Estado',modelo.estado_modelo],['Dispositivo',modelo.dispositivo],['Criterio del umbral',modelo.criterio_umbral],['SHA-256',modelo.sha256]]);
    const e=modelo.entrenamiento,m=modelo.metricas_internas;
    $('modelo-auc').textContent=m.auc==null?'—':m.auc.toFixed(3).replace('.',',');
    $('modelo-auc-nota').textContent=modelo.prueba?'Prueba reducida; no mide rendimiento':'Por paciente · validación interna, no test';
    const dl=document.createElement('dl');dl.id='detalle-dl';dl.className='metadata';$('detalles-modelo').append(dl);
    lista('detalle-dl',[['Pérdida',e.perdida],['Máximo de épocas / paciencia',`${e.epocas??'—'} / ${e.paciencia??'—'}`],['Learning rate / batch / semilla',`${e.lr??'—'} / ${e.batch_size??'—'} / ${e.semilla??'—'}`],['Mejor época interna',m.mejor_epoca],['Pacientes de validación interna',m.pacientes],['Umbral de las métricas guardadas',m.umbral],['Sensibilidad / especificidad a ese umbral',`${m.sensibilidad==null?'—':porcentaje(m.sensibilidad)} / ${m.especificidad==null?'—':porcentaje(m.especificidad)}`]]);
  }
}catch(e){$('estado').textContent='No se pudo conectar con el servidor. Abre la app desde su URL y comprueba que está en marcha.';$('modelo-badge').textContent='Sin conexión';}finally{controles();}})();

// Los efectos son decorativos; nunca representan progreso ni actividad clínica.
const menosMovimiento = window.matchMedia('(prefers-reduced-motion: reduce)');
function movimiento(pausado) {
  document.body.classList.toggle('motion-paused', pausado);
  $('movimiento').setAttribute('aria-pressed', String(pausado));
  $('movimiento').textContent = pausado ? 'Activar efectos' : 'Pausar efectos';
  $('movimiento').title = pausado ? 'Activar efectos visuales' : 'Pausar efectos visuales';
}
movimiento(menosMovimiento.matches);
$('movimiento').addEventListener('click', () => movimiento(!document.body.classList.contains('motion-paused')));
menosMovimiento.addEventListener('change', e => movimiento(e.matches));

function limpiarComparacion() {
  comparacion=null; $('clinical-result').hidden=true; $('clinical-error').hidden=true;
}
for(const id of ['clinical-age','clinical-subtype']) $(id).addEventListener('input',limpiarComparacion);
$('clinical-form').addEventListener('submit',async e=>{
  e.preventDefault(); if($('clinical-run').disabled)return;
  const age=$('clinical-age').value;
  const variables={age:age===''?null:Number(age),HR_HER2_STATUS:$('clinical-subtype').value||null};
  limpiarComparacion(); ocupado=true; controles(); $('clinical-run').textContent='Comparando…';
  try {
    comparacion=await peticion('/api/predecir_combinado',archivos,{variables});
    const cards=[];
    for(const [key,title] of [['solo_imagenes','Solo imágenes'],['solo_variables','Edad + subtipo'],['combinado','Imágenes + variables']]) {
      const r=comparacion[key], card=document.createElement('article');
      const h=document.createElement('h4'), score=document.createElement('strong'), text=document.createElement('p');
      h.textContent=title;score.textContent=porcentaje(r.probabilidad);
      text.textContent=`${r.clase?'Con pCR':'Sin pCR'} · Umbral ${porcentaje(r.umbral)}`;
      card.append(h,score,text);cards.push(card);
    }
    $('clinical-scores').replaceChildren(...cards);
    $('clinical-result-note').textContent=`Paciente: ${comparacion.solo_imagenes.paciente}. ${comparacion.solo_imagenes.numero_cortes} cortes. ${comparacion.edad_imputada?'Edad ausente: se utiliza la mediana del entrenamiento. ':''}Puntuaciones de investigación; no equivalen a probabilidades clínicas validadas.`;
    $('clinical-result').hidden=false;
  }catch(e){comparacion=null;$('clinical-error').textContent=e.message;$('clinical-error').hidden=false;}
  finally {ocupado=false;$('clinical-run').textContent='Comparar estimaciones';controles();}
});
$('clinical-download').addEventListener('click',()=>{
  if(!comparacion)return;const url=URL.createObjectURL(new Blob([JSON.stringify(comparacion,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='comparacion_paciente.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
(async()=>{try{
  const r=await fetch('/api/complemento');if(!r.ok)throw new Error();complemento=await r.json();
  $('clinical-status').textContent=complemento.disponible?'Complemento conectado. Valida las imágenes y aporta edad o subtipo.':'Complemento sin conectar: la predicción de imágenes sigue disponible.';
  if(complemento.disponible && complemento.metricas) {
    const cards=[];
    for(const [key,label] of [['solo_imagenes','Solo imágenes'],['solo_variables','Edad + subtipo'],['combinado','Imágenes + variables']]) {
      const card=document.createElement('div'), name=document.createElement('span'), value=document.createElement('strong');
      name.textContent=label;value.textContent=complemento.metricas[key].auc.toFixed(3).replace('.',',');card.append(name,value);cards.push(card);
    }
    $('clinical-metrics').replaceChildren(...cards);
    const filas=[];
    for(const [key,label] of [['solo_imagenes','Imágenes'],['solo_variables','Edad + subtipo'],['combinado','Combinado']]) {
      const m=complemento.metricas[key], tr=document.createElement('tr');
      for(const value of [label,m.matriz_confusion.FN,m.matriz_confusion.FP,porcentaje(m.sensibilidad)]) {const td=document.createElement('td');td.textContent=value;tr.append(td);} filas.push(tr);
    }
    $('clinical-confusion-rows').replaceChildren(...filas);$('clinical-confusion').hidden=false;

  }
}catch(e){$('clinical-status').textContent='No se pudo consultar el complemento. Comprueba que el servidor está actualizado.';}finally{controles();}})();

function actualizarDemo(vista) {
  for(const fase of ['pre','early','late'])for(const tipo of ['original','transform']) {
    const img=$(`demo-${tipo}-${fase}`); img.hidden=!vista;
    if(vista)img.src=vista[fase];else img.removeAttribute('src');
    img.parentElement.classList.toggle('has-image',Boolean(vista));
  }
  $('phantoms').setAttribute('aria-label',vista?'Tres fases del corte seleccionado con una transformación geométrica de ejemplo':'Tres fases reales de la figura de ejemplo BreastDCEDL');
  $('demo-caption').textContent=vista?`Corte ${vista.corte} del visor. La transformación es una ilustración en pantalla: no cambia las imágenes enviadas al modelo ni reproduce la interpolación del entrenamiento.`:'Ejemplo real de la documentación BreastDCEDL (Fridman y colaboradores, CC BY-NC 4.0). Uso ilustrativo: no se envía al modelo. Al validar tus imágenes, verás el corte seleccionado.';
}
