import json, hashlib, importlib.util, re, statistics, math, argparse, os
from pathlib import Path

parser=argparse.ArgumentParser(description='Read-only rebuild of the historical stochastic study evidence ledger.')
parser.add_argument('--source-root',type=Path,default=Path('/Users/rldoeswork/Documents/rl/tinycraft-jepa'))
parser.add_argument('--output-dir',type=Path,default=Path('/private/tmp/tinycraft-jepa-iclr-20260919/research'))
args=parser.parse_args()
ROOT=args.source_root.resolve()
OUT=args.output_dir.resolve()
os.chdir(ROOT)
OUT.mkdir(exist_ok=True,parents=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

def rel(p):
    return str(p.relative_to(ROOT))

def aggregate(rows):
    out={'n_policy_runs':len(rows),'n_model_fits':len({r['model_id'] for r in rows if r['model_id'] is not None}),
         'policy_runs':[r['run_id'] for r in rows]}
    for metric in ['walls_passed','survival','length','return','deaths_per_100']:
        xs=[r['final'][metric] for r in rows if metric in r['final']]
        if xs:
            out[metric]={'mean':statistics.mean(xs),'sample_sd':statistics.stdev(xs) if len(xs)>1 else None,
                         'min':min(xs),'max':max(xs),'values':xs}
    out['discovered_runs']=sum(r['final']['survival']>.5 for r in rows)
    return out

def classify(name,c):
    model=None
    if c.get('wm'):
        model=re.sub(r'^.*?/(?:stoch|redcube)/','',c['wm'])
        model=re.sub(r'/model(?:_probe)?\.npz$','',model)
        # Remote binary hashes independently checked by gpu_inventory on 2026-09-19.
        if model=='gate/flow_model.npz':model='gate/flow'
    if 'amp' in name or re.search(r'_(?:d?wall|d?rand)4_',name):
        return 'excluded_environment_specific_feature_transform',model,False
    if c.get('arm')=='real':
        obs=c.get('obs','pixels')
        torso=c.get('torso','cnn')
        if '_mlp_' in name: torso='mlp'
        if obs=='latent':torso='mlp'
        return 'real_'+obs+'_'+torso,model,True
    if c.get('arm')=='jepa':
        mode='mean8' if c.get('sampler_mean')==8 else ('sample' if c.get('sampler') else 'deterministic')
        return 'imagined_jepa_'+mode,model,True
    if c.get('arm')=='dream':
        return 'imagined_flow_'+c.get('obs','pixels')+('_mean8' if c.get('sampler_mean')==8 else '_sample'),model,True
    return 'other',model,True

runs=[]
for directory in sorted((ROOT/'runs/stoch').glob('*/*')):
    if not directory.is_dir() or not (directory/'ledger.jsonl').exists():continue
    p=directory/'ledger.jsonl'; lines=[(i+1,json.loads(x)) for i,x in enumerate(p.read_text().splitlines()) if x.strip()]
    if not lines:continue
    cp=directory/'config.json';c=json.loads(cp.read_text()) if cp.exists() else {}
    group,model,allowed=classify(directory.name,c)
    training_seed=None
    if model and ('tile_factored' in model or 'lewm6' in model):
        m=re.search(r'_s(\d+)',model);training_seed=int(m.group(1)) if m else 0
    elif model and 'flow' in model:training_seed=0
    endpoint_line,endpoint=lines[-1]
    item={'run_id':str(directory.relative_to(ROOT/'runs/stoch')),'world':directory.parent.name,'group':group,
          'model_id':model,'model_seed':training_seed,'policy_seed':c.get('seed'),'config':c,
          'final':endpoint,'final_source':f'{rel(p)}:{endpoint_line}','ledger_sha256':sha(p),'config_sha256':sha(cp),
          'complete_marker_present':Path(str(directory)+'.done').exists(),
          'n_saved_evaluations':len(lines),'best_saved_survival':max(x['survival'] for _,x in lines),
          'world_model_recorded_path':c.get('wm'),'local_policy_checkpoint_present':(directory/'model.pkl').exists(),
          'eligible_under_no_environment_specific_feature_transform_rule':allowed}
    runs.append(item)

spec=importlib.util.spec_from_file_location('stoch_summarize',ROOT/'studies/stoch/summarize.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
raw=module.collect()
w=raw['gate']
main_selection={}
for ms in range(3):
    for mode in module.MODES:
        for ps in range(3):
            main_selection['gate/'+module.grid_run_name(ms,mode,ps)]='imagined_jepa_'+mode
for mode in ['samp','mean8']:
    for ps in range(8):
        if w['flow'].get(f'{mode}_p{ps}'):
            main_selection['gate/'+module.flow_run_name(mode,ps)]='imagined_flow_'+mode
for key,prefix in [('real_seeds','real_pixels_cnn'),('real_latent','real_latent_mlp'),('real_decoded','real_decoded_cnn'),('real_mlp','real_pixels_mlp'),('decoded_mlp','real_decoded_mlp')]:
    for ident,row in w[key].items():
        if not row:continue
        if key=='real_seeds':name='ppo_'+ident
        elif key=='real_latent':
            ms,ps=map(int,re.findall(r'\d+',ident));name=f'ppo_real_jepa_latent_p{ps}' if ms==0 else f'ppo_real_jepa_latent_m{ms}_p{ps}'
        elif key=='real_decoded':name='ppo_real_jepa_decoded_'+ident
        elif key=='real_mlp':name='ppo_real_mlp_'+ident
        else:name='ppo_real_jepa_decoded_mlp_'+ident
        main_selection['gate/'+name]=prefix
for r in runs:r['main_paper_group']=main_selection.get(r['run_id'])

grouped={}
for group in sorted(set(main_selection.values())):
    rows=[r for r in runs if r['main_paper_group']==group]
    grouped[group]={'all_runs_descriptive':aggregate(rows),'per_model':{}}
    for model in sorted({r['model_id'] or 'no_world_model' for r in rows}):
        mr=[r for r in rows if (r['model_id'] or 'no_world_model')==model]
        grouped[group]['per_model'][model]=aggregate(mr)

diagnostics=[]
for seed in range(3):
    tag='' if seed==0 else f'_s{seed}'
    lp=ROOT/f'runs/stoch/gate/latent_check{tag}.json';hp=ROOT/f'runs/stoch/gate/heads_audit_tile_factored{tag}.json'
    diagnostics.append({'model_seed':seed,'latent_source':rel(lp),'latent_sha256':sha(lp),'latent':json.loads(lp.read_text()),
                        'heads_source':rel(hp),'heads_sha256':sha(hp),'heads':json.loads(hp.read_text())})

issues=[
 {'id':'not_all_evaluations_paired','severity':'material','finding':'Final PPO evaluation keys are drawn from the training RNG. Real training uses an initial reset split and 183 iterations; imagined training uses 150 updates. Real-vs-imagined final episodes are not paired by a common fixed key.','sources':['studies/stoch/rl.py:460','studies/stoch/rl.py:467','studies/stoch/rl.py:569','studies/stoch/rl.py:572'],'fix':'Use unpaired descriptive comparisons for historical results, or reevaluate fixed final checkpoints with fixed shared evaluation keys.'},
 {'id':'head_holdout_not_wm_holdout','severity':'material','finding':'Posthoc heads and decoder use first 80000 frames of the same model-training dataset; head holdout uses the next 200 time rows. No different behavior-policy seed is supplied. This holdout is not held out from world-model fitting. Separate heads_audit uses eval.npz.','sources':['studies/stoch/train_arms.sh:37','studies/stoch/train_arms.sh:48','probe.py:230','probe.py:262'],'fix':'Correct appendix dataset provenance; use 25000-transition heads_audit for test metrics.'},
 {'id':'craftax_open_loop','severity':'material','finding':'plan.py closed_loop plans once and executes the fixed H-step action sequence. It is not receding-horizon closed-loop goal reaching.','sources':['plan.py:155','plan.py:195'],'fix':'Rename historical goal test short-horizon open-loop goal reaching; distinguish reward_loop MPC.'},
 {'id':'craftax_behaviour_reference','severity':'material','finding':'Craftax behavior-policy return is mean reward from a dataset multiplied by evaluation horizon, not a matched online rollout baseline.','sources':['plan.py:334'],'fix':'Label as dataset-rate reference or omit from matched control table.'},
 {'id':'hierarchical_replication','severity':'scope','finding':'27 imagined JEPA runs are three predictor variants x three policy seeds x three encoder fits, not 27 independent model seeds; flow has one fit and nine policy runs.','sources':['studies/stoch/summarize.py:161','studies/stoch/paper/sections/controllers.tex:13'],'fix':'Report per-model and per-policy endpoints, not family-level significance from pooled policy runs.'},
 {'id':'pipeline_confounds','severity':'scope','finding':'Real-vs-imagined controls alter heads, dynamics, data starts, rollout lengths, PPO organization and transition budget at once. Flow vs JEPA also alters reconstruction/reward supervision, architecture and history.','sources':['studies/stoch/rl.py:371','studies/stoch/rl.py:510','studies/stoch/paper/sections/worlds.tex:34'],'fix':'Describe complete pipeline comparison; do not causally attribute to squared loss or deterministic mean alone.'},
 {'id':'mean8_not_matched_compute','severity':'scope','finding':'Flow mean8 final 150-update runs use a 60-minute cap, whereas sampled runs use 12 minutes; separate earlier 12-minute runs are retained but excluded from main comparison. Transition budgets match; wall-clock does not.','sources':['studies/stoch/lane_f4.sh:4','studies/stoch/lane_f4.sh:17'],'fix':'State same updates/imagined transitions, unequal inference compute.'},
 {'id':'existing_amplification_runs','severity':'excluded','finding':'Repository contains 12 amp4/amp16 policy runs and 24 wall/random subspace gain-4 runs. They are not needed for any accepted core conclusion and are excluded from the proposed general-model evidence set.','sources':['studies/stoch/summarize.py:180','studies/stoch/rl.py:379'],'fix':'Do not present environment-specific amplification as a model improvement.'},
 {'id':'summary_copy_stale','severity':'reproducibility','finding':'runs/stoch/summarize.py is an older copy; studies/stoch/summarize.py supplies current round-2 aggregation. Current saved summary equals fresh collect(), but paired.tex manually omits simulator and center reference rows generated by current code.','sources':['runs/stoch/summarize.py','studies/stoch/summarize.py:251','studies/stoch/paper/tables/paired.tex'],'fix':'Use studies/stoch/summarize.py as canonical and new ledger for table generation.'},
 {'id':'checkpoint_hashes_missing','severity':'reproducibility','finding':'Raw config records absolute checkpoint paths, generally on VMs. Local directory has only a subset of policy checkpoint binaries and no comprehensive checkpoint SHA manifest.','sources':['runs/stoch/gate/ppo_jepa_tf_m1_det_p0/config.json'],'fix':'Generate remote artifact hash manifest before declaring independently reproducible checkpoint identity.'},
 {'id':'edge_still_shortcut','severity':'scope','finding':'Existing edge reset distribution does not eliminate state-blind survival: alternating left-right policy scores 100% in all three available still-world baselines_edge files. Starts are at least 3px from the boundary and action displacement is 3px.','sources':['runs/stoch/off_still/baselines_edge.json','runs/stoch/gray_frame_still/baselines_edge.json','runs/stoch/gray_episode_still/baselines_edge.json','studies/stoch/env.py:105'],'fix':'Do not use still-world edge survival to certify position feedback.'},
]

def finite_json(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_json(v) for v in value]
    return value

result={'schema_version':1,'generated_from':str(ROOT),'scope':'Read-only audit of existing files; no experiments rerun. Nonfinite source JSON numbers normalized to null.',
        'verified_model_aliases':[{'model_id':'gate/flow','sha256':'72b829ea8cfb5f4bf21f3b8c0ce8ffe53f34dd5aa67bad001bb8d4cd0bab27e9','paths':['/home/rldoeswork/personal/stoch/gate/flow/model.npz','/dev/shm/stoch/gate/flow_model.npz'],'verification':'Both remote binaries hashed by gpu_inventory agent on 2026-09-19.'}],
        'verified_dataset_aliases':[{'dataset':'gate training','archive_hashes':['c347aa2db1726bceadbc6a16fc2b4a7559343f988c4c2074e90c914b05354b55','ea1315a8c612afc7f5a6a80938d9ef982e3bd3af9ad04f6684ec9377df817cce'],'verification':'All decompressed NPY member SHA256 values, including meta, match across MG/e2e copies; gpu_inventory agent 2026-09-19. Archive-byte difference is packaging only.','sources':['/tmp/tinycraft-jepa-mg-gate-arrayhash.json','/tmp/tinycraft-jepa-e2e-gate-arrayhash.json']}],
        'selection_rule':'Final nonempty ledger row, completion marker recorded, never best checkpoint.',
        'main_paper_endpoint_runs':len(main_selection),'all_policy_runs':runs,'main_gate_groups':grouped,
        'gate_diagnostics_by_model_seed':diagnostics,'canonical_summary_matches_raw_collection':raw==json.loads((ROOT/'runs/stoch/summary.json').read_text()),
        'stoch_raw_collection':raw,'issues':issues}
for path in [Path('/tmp/tinycraft-jepa-evidence.json'),OUT/'evidence-ledger.json']:
    path.write_text(json.dumps(finite_json(result),indent=2,allow_nan=False)+'\n')
print(json.dumps({'runs':len(runs),'main_runs':len(main_selection),'groups':list(grouped),'output':str(OUT/'evidence-ledger.json')}))
