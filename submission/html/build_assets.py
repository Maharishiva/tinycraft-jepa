"""Export original diagnostic frames and audited endpoints for the offline article."""
import numpy as np
from pathlib import Path
from PIL import Image
import json, hashlib
root=Path(__file__).resolve().parents[2]
out=Path(__file__).parent/'assets'
keys=['recorded','grid_recorded_future_encoded','grid_deterministic','flow_sample']
manifest={}
for seed in range(3):
 source=root/('runs/stoch/gate/latent_check'+('' if seed==0 else '_s'+str(seed))+'.npz')
 p=np.load(source)
 manifest[str(seed)]={'source':str(source.relative_to(root)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scenes':16,'steps':8,'keys':[]}
 for k in keys+['context']:
  if k not in p: continue
  x=p[k]
  sheet=np.concatenate([np.concatenate(scene,axis=1) for scene in x],axis=0)
  Image.fromarray(sheet).save(out/('rollout-s'+str(seed)+'-'+k+'.png'))
  manifest[str(seed)]['keys'].append(k)
p=np.load(root/'runs/stoch/gate/latent_check.npz')
for k in keys: Image.fromarray(p[k][0,1]).save(out/(k+'-poster.png'))
(out/'rollout-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
ledger=json.load(open(Path(__file__).parents[2]/'research/evidence-ledger.json'))
keep=['run_id','main_paper_group','model_seed','policy_seed','final','final_source','ledger_sha256']
runs=[{k:r[k] for k in keep if k in r} for r in ledger['all_policy_runs'] if r.get('main_paper_group')]
(out/'evidence.json').write_text(json.dumps({'runs':runs,'groups':ledger['main_gate_groups'],'rollouts':manifest},indent=2)+'\n')
(out/'evidence.js').write_text('window.ARTICLE_EVIDENCE = '+json.dumps({'runs':runs,'rollouts':manifest})+';\n')
print('Generated',len(list(out.iterdir())),'local assets; selected runs',len(runs))
print({k:round(v['all_runs_descriptive']['walls_passed']['mean'],4) for k,v in ledger['main_gate_groups'].items()})
