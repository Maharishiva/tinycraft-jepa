'use strict';
(()=>{
  const alphaInput=document.getElementById('gaussian-alpha');
  const rhoInput=document.getElementById('gaussian-rho');
  if(!alphaInput||!rhoInput)return;
  document.getElementById('analytical-illustration').classList.add('gaussian-interactive');
  const loss=(a,r)=>1-(1-a+a*r)**2;
  const x=a=>58+574*a;
  const y=v=>230-200*v;
  function render(){
    const a=Number(alphaInput.value)/100,r=Number(rhoInput.value)/100,mse=loss(a,r);
    document.getElementById('gaussian-alpha-value').textContent=a.toFixed(2);
    document.getElementById('gaussian-rho-value').textContent=r.toFixed(2);
    alphaInput.setAttribute('aria-valuetext',`State mixture alpha ${a.toFixed(2)}`);
    rhoInput.setAttribute('aria-valuetext',`State persistence rho ${r.toFixed(2)}`);
    document.getElementById('gaussian-mse').textContent=mse.toFixed(3);
    document.getElementById('gaussian-r2').textContent=a.toFixed(3);
    const path=Array.from({length:101},(_,i)=>{const v=i/100;return `${i===0?'M':'L'}${x(v).toFixed(2)} ${y(loss(v,r)).toFixed(2)}`}).join('');
    document.getElementById('gaussian-loss-path').setAttribute('d',path);
    const selected=document.getElementById('gaussian-selection-line');selected.setAttribute('x1',x(a));selected.setAttribute('x2',x(a));
    for(const [id,value] of [['gaussian-loss-point',mse],['gaussian-state-point',a]]){const point=document.getElementById(id);point.setAttribute('cx',x(a));point.setAttribute('cy',y(value));}
    document.getElementById('gaussian-chart-description').textContent=`Analytical illustration, not experimental data. State persistence rho ${r.toFixed(2)}. At selected state mixture alpha ${a.toFixed(2)}, optimal one-lag next-latent mean squared error is ${mse.toFixed(3)} and optimal current-state readout R squared is ${a.toFixed(3)}. Latent variance is one. The curves show alpha zero through one.`;
  }
  [alphaInput,rhoInput].forEach(input=>input.addEventListener('input',render));render();
})();
