"""Copy exact frozen-checkpoint display frames into the standalone reader artifact."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'assets/fresh'
OUT.mkdir(parents=True,exist_ok=True)
conditions={}
for condition in ['gray_frame_still','gray_frame_wind','gray_episode_wind','gate_s1']:
 source=ROOT/'artifacts/revision/redcube_story'/condition
 if not all((source/name).exists() for name in ['browser_packet.json','case_manifest.json','frames.npz']):continue
 manifest=json.loads((source/'case_manifest.json').read_text())
 packet=json.loads((source/'browser_packet.json').read_text())
 z=np.load(source/'frames.npz')
 keys={}
 mapping={'gt':('context','real_future'),'real_encoded':('jepa_context','jepa_encoded_real'),'jepa':('jepa_context','jepa_future'),'flow':('flow_context','flow_future'),'flow_encoded':('flow_context','flow_encoded_real')}
 hashes={}
 for name,(context,future) in mapping.items():
  frames=np.concatenate([z[context][:,-1:],z[future]],axis=1)
  if frames.shape!=(3,17,63,63,3):raise ValueError((condition,name,frames.shape))
  sheet=np.concatenate([np.concatenate(scene,axis=1) for scene in frames],axis=0)
  dst=OUT/(condition+'-'+name+'.png');Image.fromarray(sheet).save(dst)
  if not np.array_equal(np.asarray(Image.open(dst)),sheet):raise ValueError('PNG mismatch')
  keys[name]=str(dst.relative_to(HERE));hashes[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
 provenance={'case_id':condition,'export_source':'artifacts/revision/redcube_story/'+condition,'raw_frames_npz_sha256':hashlib.sha256((source/'frames.npz').read_bytes()).hexdigest(),'selection_rule':manifest['selection_rule'],'selection_caveat':manifest['selection_caveat'],'scene_ids':manifest['scene_ids'],'checkpoint_and_data_sha256':{k:v['sha256'] for k,v in manifest['source_files'].items()},'array_sha256':manifest.get('array_sha256',{}),'display_sprite_sha256':hashes,'display_transform':'none; concatenation only; native uint8 RGB preserved','note':'Flow uses the corrected PPO recipe checkpoint, not the older redcube benchmark flow checkpoint.'}
 (OUT/(condition+'-manifest.json')).write_text(json.dumps(provenance,indent=2)+'\n')
 conditions[condition]={'fresh':True,'jepaModelSeed':packet['jepa_model_seed'],'flowModelSeed':packet['flow_model_seed'],'flowNFE':packet['flow_visual_nfe'],'context':1,'horizon':16,'scenes':3,'keys':keys,'selectedWindows':manifest['scene_ids'],'actions':z['future_actions'].tolist(),'validMasks':z['valid_mask'].tolist(),'manifest':str((OUT/(condition+'-manifest.json')).relative_to(HERE)),'caption':'At 0, model views reconstruct the last observation. From +1, the real-latent view decodes recorded futures; JEPA and flow forecast under the same actions. Flow shows one sample.','selection':'Three examples were fixed before model evaluation, requiring a reset-free context and future. The illustrations are therefore conditioned on survival.'}
(HERE/'assets/fresh-story.js').write_text('window.FRESH_STORY = '+json.dumps({'conditions':conditions})+';\n')
print('Fresh conditions:',list(conditions))
