'use strict';
(()=>{
  const data=window.CRAFTAX_EVIDENCE;
  if(!data)return;
  const controls=[['noop','No-op','neutral'],['random','Uniform random','neutral'],['jepa_open_loop','JEPA · fixed six-action plan','jepa'],['jepa_receding','JEPA · receding-horizon plan','jepa'],['reference_replay','Generating-sequence replay','real']];
  const names={all:'All targets',displaced_goal:'Targets displaced by at least 1 grid cell',goal_displacement_at_least_2:'Targets displaced by at least 2 grid cells'};
  function renderSubset(key){
    const body=document.getElementById('craftax-result-rows');body.replaceChildren();
    controls.forEach(([id,label,color])=>{
      const d=data.policy_results[id][key],tr=document.createElement('tr');tr.className='method-'+color;
      const values=[label,`${d.successes} / ${d.n}`,`${(d.mean*100).toFixed(1)}%`,`${(d.wilson_95[0]*100).toFixed(1)}–${(d.wilson_95[1]*100).toFixed(1)}%`];
      values.forEach((value,i)=>{const cell=document.createElement(i===0?'th':'td');if(i===0)cell.scope='row';cell.textContent=value;tr.append(cell)});body.append(tr);
    });
    document.getElementById('craftax-subset-count').textContent=`${names[key]} · ${data.policy_results.noop[key].n} episodes · one trained checkpoint`;
  }
  document.querySelectorAll('[name=craftax-subset]').forEach(input=>input.addEventListener('change',()=>{if(input.checked)renderSubset(input.value)}));
  renderSubset('all');
  document.getElementById('craftax-scene').addEventListener('change',event=>{
    const id=event.target.value;
    for(const name of ['start','goal']){
      const image=document.getElementById('craftax-'+name);
      image.src=`assets/craftax/seed101_episode${id}_${name}.png`;
      image.alt=`Actual Craftax ${name==='start'?'planning-start':'target'} observation, evaluation seed 101, episode ${Number(id)}. Target displacement: 2 grid cells.`;
    }
  });
})();
