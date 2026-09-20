'use strict';
(()=>{
 const source=window.HISTORICAL_STORY;
 const imageCache=new Map();
 function loadImage(path){if(imageCache.has(path))return imageCache.get(path);const promise=new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src=path});imageCache.set(path,promise);return promise}
 document.querySelectorAll('.case-viewer').forEach(viewer=>{
  const data=window.FRESH_STORY?.conditions?.[viewer.dataset.case] || source?.conditions?.[viewer.dataset.case];
  if(!data){viewer.querySelector('[data-status]').textContent='Saved-frame assets are unavailable.';return;}
  const slider=viewer.querySelector('[data-time-slider]'),scene=viewer.querySelector('[data-scene]'),play=viewer.querySelector('[data-play]');
  viewer.querySelector('[data-flow-reconstruction]').hidden=!data.fresh;
  if(data.fresh){
    const caption=viewer.querySelector('figcaption');caption.replaceChildren(document.createTextNode(data.caption+' '+data.selection+' '));
    const link=document.createElement('a');link.href=data.manifest;link.textContent='Frame identities and selection ↗';caption.append(link);
  }
  let timer=null,version=0;
  function stop(){if(timer)clearInterval(timer);timer=null;play.textContent='Play';play.setAttribute('aria-label','Play saved recorded-action forecasts')}
  async function render(){
   const v=++version,t=Number(slider.value),s=Number(scene.value),index=data.context-1+t;
   viewer.querySelector('[data-time]').textContent=t===0?'0':'+'+t;
   viewer.querySelector('[data-time-label]').textContent=t===0?'Observed':'Future';
   viewer.querySelector('.case-frame.jepa p').innerHTML='<span class="dot"></span>'+(t===0?'JEPA reconstruction':'JEPA imagination');
   viewer.querySelector('.case-frame.flow p').innerHTML='<span class="dot"></span>'+(t===0?'Flow reconstruction':'Flow sample');
   slider.setAttribute('aria-valuetext',t===0?'Last observed context frame':`Future step ${t} of ${data.horizon}`);
   viewer.querySelector('[data-status]').textContent=`${t===0?'Observed context and model reconstructions':`Forecast step +${t}`} · example ${s+1} of ${data.scenes} · ${data.fresh?'fixed window':'historical window'} (${data.selectedWindows[s]})${data.fresh?' · JEPA seed '+data.jepaModelSeed+', flow seed '+data.flowModelSeed:''}`;
   const pending=viewer.querySelector('[data-real-pending]');pending.hidden=t===0||data.fresh;
   const actions=viewer.querySelector('[data-actions]');actions.hidden=!data.fresh;
   if(data.fresh){actions.replaceChildren();data.actions[s].forEach((action,i)=>{const el=document.createElement('span');el.textContent=['←','↑','→','↓'][action];el.title=`Action ${i+1}: ${['left','up','right','down'][action]}`;if(i===t-1)el.className='active';actions.append(el)});}
   if(data.fresh&&t>0&&!data.validMasks[s][t-1]){viewer.querySelectorAll('canvas').forEach(canvas=>{canvas.hidden=true});viewer.querySelector('[data-status]').textContent=`Future step +${t} is outside the valid episode window; frames are hidden.`;return;}
   await Promise.all((data.fresh?['gt','jepa','flow','real_encoded','flow_encoded']:['gt','jepa','flow','real_encoded']).map(async key=>{
    const canvas=viewer.querySelector(`[data-view=${key}]`);
    if(key==='real_encoded'&&t>0&&!data.fresh){canvas.hidden=true;return;}
    try{
     const image=await loadImage(data.keys[key==='real_encoded'&&!data.fresh?'jepa':key]);if(v!==version)return;
     const context=canvas.getContext('2d');context.imageSmoothingEnabled=false;context.clearRect(0,0,63,63);context.drawImage(image,index*63,s*63,63,63,0,0,63,63);canvas.hidden=false;
     canvas.setAttribute('aria-label',`${key==='real_encoded'?'Decoded real JEPA latent':key} for ${viewer.dataset.case}, example ${s+1}, ${t===0?'last context frame':'future step '+t}`);
    }catch(error){if(v!==version)return;canvas.hidden=true;viewer.querySelector('[data-status]').textContent='A saved frame failed to load; consult the visual manifest.';}
   }));
  }
  slider.addEventListener('input',()=>{stop();render()});scene.addEventListener('change',()=>{stop();render()});viewer.querySelector('[data-observe]').addEventListener('click',()=>{stop();slider.value='0';render()});
  play.addEventListener('click',()=>{if(timer){stop();return;}if(Number(slider.value)>=data.horizon)slider.value='0';play.textContent='Pause';play.setAttribute('aria-label','Pause saved forecasts');render();timer=setInterval(()=>{if(Number(slider.value)>=data.horizon){stop();return;}slider.value=String(Number(slider.value)+1);render()},300)});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});render();
 });
 const demo=document.querySelector('.mean-demo');
 document.querySelectorAll('[data-score]').forEach(button=>button.addEventListener('click',()=>{
  const mode=button.dataset.score;demo.dataset.mode=mode;
  document.querySelectorAll('[data-score]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
  document.querySelector('[data-equation-a]').innerHTML=mode==='mean'?'0² = <strong>0</strong>':'½(−1)² + ½(+1)² = <strong>1</strong>';
  document.querySelector('[data-equation-b]').innerHTML='0.1² = <strong>0.01</strong>';
  document.querySelector('[data-choice]').innerHTML=mode==='mean'?'<strong>Scoring only the mean chooses A.</strong> Its predicted mean lies at the goal.':'<strong>Scoring the outcomes chooses B.</strong> Its expected squared cost is 0.01 instead of 1.';
 }));
})();

(()=>{
 const seed=document.getElementById('goal-model-seed'),episode=document.getElementById('goal-rgb-episode');
 if(!seed||!episode)return;
 function renderGoal(){
  const model=seed.value,ep=episode.value;
  const paths=document.getElementById('goal-path-image'),rgb=document.getElementById('goal-rgb-image');
  paths.src=`assets/goals/goal_paths_checkpoint${model}.svg`;
  paths.alt=`Actual trajectories for the three predeclared episodes zero, one and two, model seed ${model}. The JEPA H3 controller fails in these fixed examples; the privileged state reference reaches the goals.`;
  rgb.src=`assets/goals/goal_rgb_checkpoint${model}_episode${ep}.svg`;
  rgb.alt=`Unmodified RGB execution frames, model seed ${model}, fixed episode ${ep}, for random actions, JEPA H3 and the privileged state reference. Each row stops at first hit, death or action limit.`;
 }
 seed.addEventListener('change',renderGoal);episode.addEventListener('change',renderGoal);
})();

(()=>{
 const image=document.getElementById('range-main-image'),seed=document.getElementById('range-model-seed');
 if(!image||!seed)return;
 let view='performance';
 const figure=document.querySelector('.goal-range-figure'),caption=document.getElementById('range-caption');
 function renderRange(){
  figure.querySelector('[data-range-seed-control]').hidden=view!=='paths';
  if(view==='performance'){
   image.src='assets/goal-distance/goal_distance_response.svg';
   image.alt='Complete goal-control operating-range curve for all three model seeds and all six distances. H3 beats H1 at two and three moves, then success drops sharply to zero at twelve moves.';
   caption.textContent='The unchanged H3 controller reaches nearby goals reliably, then loses success with distance. Every model seed and distance is retained. Error bars are Wilson intervals for 128 starts separately in each cell; the same prefixes are reused across shells. The state reference is privileged. This diagnostic was designed after the original hard-goal failure.';
  }else{
   image.src=`assets/goal-distance/goal_distance_paths_m${seed.value}.svg`;
   image.alt=`Actual paths for fixed case zero at all six distance shells, model seed ${seed.value}. The same initial observation is used; short goals are reached while longer-goal paths fail.`;
   caption.textContent='Actual simulator paths for predeclared case 0 in every distance shell. Colored endpoint circles are goal hits; crosses mark other endings. Coordinates are diagnostic annotations on a faded initial frame, not JEPA inputs. The state reference is privileged. All six shells are retained.';
  }
 }
 figure.querySelectorAll('[data-range-view]').forEach(button=>button.addEventListener('click',()=>{view=button.dataset.rangeView;figure.querySelectorAll('[data-range-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));renderRange()}));
 seed.addEventListener('change',renderRange);
 const rgbSeed=document.getElementById('range-rgb-seed');rgbSeed.addEventListener('change',()=>{const rgb=document.getElementById('range-rgb-image');rgb.src=`assets/goal-distance/goal_distance_rgb_m${rgbSeed.value}.svg`;rgb.alt=`Unmodified JEPA H3 execution frames, fixed case zero across all six goal-distance shells, model seed ${rgbSeed.value}. Each row ends at first hit, death, or action limit.`});
})();
