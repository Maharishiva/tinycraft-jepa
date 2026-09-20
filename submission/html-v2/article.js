'use strict';
(() => {
  const stageCopy = {
    retain: '<strong>Retention.</strong> Decoding a real observation’s frozen latent can reveal information that an imagined rollout later loses. A successful readout establishes availability under that readout; a failed readout does not prove absence.',
    predict: '<strong>Prediction.</strong> The predictor is trained with squared error against future encodings. In the gate diagnostic, decoded predicted latents lose a wall that decoded real encodings retain. This observation does not identify which training mechanism causes the loss.',
    control: '<strong>Control.</strong> A policy trained in imagination depends on transitions, reward and termination heads, and the policy-learning procedure. Its real-environment performance tests this full chain. Comparing real and imagined training changes several links at once.'
  };
  document.querySelectorAll('[data-stage]').forEach(button => button.addEventListener('click', () => {
    const stage=button.dataset.stage;
    document.querySelectorAll('[data-stage]').forEach(b=>b.setAttribute('aria-pressed', String(b===button)));
    document.querySelectorAll('[data-step]').forEach(s=>s.classList.toggle('active',s.dataset.step===stage));
    document.getElementById('stage-explanation').innerHTML=stageCopy[stage];
  }));
  document.querySelector('[data-step="retain"]').classList.add('active');

  const data = window.ARTICLE_EVIDENCE;
  if (!data) return;
  const seed=document.getElementById('rollout-seed'), scene=document.getElementById('rollout-scene'), step=document.getElementById('rollout-step');
  for(let i=0;i<16;i++){const o=document.createElement('option');o.value=String(i);o.textContent=String(i+1).padStart(2,'0')+' / 16';scene.append(o);}
  const keys=['recorded','grid_recorded_future_encoded','grid_deterministic','flow_sample'];
  const images=new Map();
  let renderVersion=0;
  const getImage=(src)=>{if(images.has(src))return images.get(src);const p=new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src=src});images.set(src,p);return p};
  async function render(){
    const version=++renderVersion, s=seed.value, c=Number(scene.value), t=Number(step.value);
    document.getElementById('step-value').textContent='+'+(t+1);
    step.setAttribute('aria-valuetext','Future step '+(t+1)+' of 8');
    document.getElementById('viewer-status').textContent=`Model seed ${s} · stored context ${c+1} of 16 · future step +${t+1}`;
    await Promise.all(keys.map(async k=>{
      const cv=document.getElementById('frame-'+k), poster=document.getElementById('poster-'+k);
      const available=data.rollouts[s].keys.includes(k);
      if(k==='flow_sample')document.getElementById('flow-unavailable').hidden=available;
      if(!available){cv.hidden=true;poster.hidden=true;return;}
      try{
        const im=await getImage(`assets/rollout-s${s}-${k}.png`);
        if(version!==renderVersion)return;
        const ctx=cv.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.clearRect(0,0,63,63);ctx.drawImage(im,t*63,c*63,63,63,0,0,63,63);
        cv.hidden=false;poster.hidden=true;cv.setAttribute('aria-label',`${k.replaceAll('_',' ')}, JEPA seed ${s}, stored context ${c+1}, future step ${t+1}`);
      }catch(error){if(version!==renderVersion)return;cv.hidden=true;poster.hidden=true;document.getElementById('viewer-status').textContent='A saved image could not load. Unavailable frames are hidden; consult the frame manifest.';}
    }));
  }
  let timer=null;
  const play=document.getElementById('rollout-play');
  function pause(){if(timer){clearInterval(timer);timer=null;}play.textContent='Play';play.setAttribute('aria-label','Play saved rollout');}
  [seed,scene].forEach(el=>el.addEventListener('change',()=>{pause();render()}));
  step.addEventListener('input',()=>{pause();render()});
  play.addEventListener('click',()=>{if(timer){pause();return;}play.textContent='Pause';play.setAttribute('aria-label','Pause saved rollout');timer=setInterval(()=>{if(Number(step.value)>=7){pause();return;}step.value=String(Number(step.value)+1);render()},600);if(Number(step.value)>=7){step.value='0';render();}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause()});
  render();

  const group=document.getElementById('evidence-group'), model=document.getElementById('evidence-seed'), body=document.getElementById('endpoint-rows');
  const names={imagined_jepa_det:'JEPA · original',imagined_jepa_samp:'JEPA · residual sample',imagined_jepa_mean8:'JEPA · residual mean of 8',real_latent_mlp:'Real latents · MLP',real_decoded_cnn:'Real decoded latents · CNN',real_decoded_mlp:'Real decoded latents · MLP',real_pixels_cnn:'Real pixels · CNN',real_pixels_mlp:'Real pixels · MLP',imagined_flow_samp:'Flow · sample',imagined_flow_mean8:'Flow · mean of 8'};
  const order=Object.keys(names);
  const filtered=()=>data.runs.filter(r=>(group.value==='all'||r.main_paper_group.startsWith(group.value))&&(model.value==='all'||(!r.main_paper_group.includes('flow')&&!r.main_paper_group.includes('pixels')&&String(r.model_seed)===model.value))).sort((a,b)=>order.indexOf(a.main_paper_group)-order.indexOf(b.main_paper_group)||(a.model_seed??0)-(b.model_seed??0)||(a.policy_seed??0)-(b.policy_seed??0));
  function renderTable(){
    const rows=filtered();body.replaceChildren();
    rows.forEach(r=>{const tr=document.createElement('tr');const g=r.main_paper_group;const values=[names[g]||g,g.includes('pixels')?'—':String(r.model_seed??'—'),String(r.policy_seed??'—'),r.final.walls_passed.toFixed(3),(100*r.final.survival).toFixed(1)+'%',r.run_id.replace('gate/','')];
      values.forEach((v,i)=>{const el=document.createElement(i===0?'th':'td');if(i===0)el.scope='row';el.textContent=v;if(i===5)el.title=`${r.final_source}\nLedger SHA-256: ${r.ledger_sha256}`;tr.append(el)});body.append(tr)});
    document.getElementById('endpoint-count').textContent=rows.length+' endpoints';
  }
  [group,model].forEach(el=>el.addEventListener('change',renderTable));renderTable();
})();
