import ast
import hashlib
import inspect
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, '/private/tmp/tinycraft-jepa-iclr-20260919')
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from studies.revision import matched_ppo as O, matched_ppo_termination as N, policy as G
from studies.stoch import env as S, rl as P

class ToyModel(nnx.Module):
    latent_dim = 512
    def encode(self, frames):
        red = ((frames[..., 0] > 200) & (frames[..., 1] < 50) & (frames[..., 2] < 50)).astype(jnp.float32)
        xx = jnp.arange(S.W, dtype=jnp.float32)
        mean_x = jnp.sum(red * xx, axis=(-2,-1)) / jnp.maximum(jnp.sum(red, axis=(-2,-1)), 1)
        mean_y = jnp.sum(red * xx[:, None], axis=(-2,-1)) / jnp.maximum(jnp.sum(red, axis=(-2,-1)), 1)
        feature = jnp.stack([mean_x / 63, mean_y / 63], -1)
        return jnp.pad(feature, [(0,0)] * (feature.ndim - 1) + [(0,510)])

class ToyHeads(nnx.Module):
    def __call__(self, obs, act):
        r = 0.25 + obs[...,0] + 0.7 * act
        return jnp.stack([r - 0.1, r + 0.1], -1), jnp.full(act.shape, -3.0), jnp.zeros_like(r)

class FixedAC(nnx.Module):
    def step(self, obs, prev_a, g):
        logits = jnp.broadcast_to(jnp.array([0., -1e8, -1e8, -1e8]), (obs.shape[0],4))
        values = jnp.broadcast_to(jnp.where(jnp.arange(P.VBINS) == 36, 0., -1e8), (obs.shape[0],P.VBINS))
        return logits, values, g + 0.01

args = N.parser().parse_args(['--arm','oracle_reward','--model','unused','--data','unused','--out','unused','--model-seed','0','--batch','4','--hor','6','--chunk','6','--mb','2'])
assert N.ARMS == ('oracle_reward',)
assert O.sha256(O.__file__) == N.FROZEN_PARENT_SHA256
for name in ('sha256','episode_steps','sample_starts','validate_reconstruction','make_update'):
    assert ast.dump(ast.parse(inspect.getsource(getattr(O, name)))) == ast.dump(ast.parse(inspect.getsource(getattr(N, name))))
B,H = args.batch,args.hor
spec = S.CONFIGS['gate']
state = S.State(jnp.array([[S.SAFE_LO,30],[S.SAFE_LO+6,30],[35,30],[35,30]],jnp.int32), jnp.zeros((B,2),jnp.int32), jnp.zeros((B,2),jnp.uint32), jnp.array([0,0,0,126],jnp.int32), jnp.zeros(B,jnp.int32), jnp.full(B,S.NO_WALL,jnp.int32), jnp.zeros(B,jnp.int32))
frames = jax.vmap(lambda s,k: S.render(spec,s,k))(state,jax.random.split(jax.random.PRNGKey(4),B))
ctx = jnp.broadcast_to(frames[:,None],(B,P.CTX,*frames.shape[1:]))
ca = jnp.full((B,P.CTX),2,jnp.int32)
reset = jnp.ones(B,bool)
bundle = SimpleNamespace(model=ToyModel(), heads=ToyHeads())
key = jax.random.PRNGKey(107)
batch,diag = N.make_oracle_collector(bundle,spec,args)(FixedAC(),ctx,ca,reset,state,key)
batch = jax.tree.map(np.asarray,batch); diag=np.asarray(diag)
old_args = SimpleNamespace(**vars(args)); old_args.arm='oracle_heads'
old,old_diag=O.make_oracle_collector(bundle,spec,old_args)(FixedAC(),ctx,ca,reset,state,key)
true_args=SimpleNamespace(**vars(args));true_args.arm='oracle_true'
true,true_diag=O.make_oracle_collector(bundle,spec,true_args)(FixedAC(),ctx,ca,reset,state,key)
for k in ('obs','prev_a','g0','act','logp'):
    np.testing.assert_array_equal(batch[k],np.asarray(old[k])); np.testing.assert_array_equal(batch[k],np.asarray(true[k]))
for leaf in jax.tree.leaves((batch,diag)):
    assert np.all(np.isfinite(leaf))
# An independently driven simulator trace reconstructs the exact continuing and
# absorbing observations; only original environment step/render are reused.
current=np.asarray(frames); states=state; stopped=np.zeros(B,bool)
zs=[]; rs=[]; ds=[]; before=[]; true_rs=[]
for t,k in enumerate(jax.random.split(key,H)):
    zs.append(np.asarray(bundle.model.encode(jnp.asarray(current[:,None])))[:,0])
    act=batch['act'][t]
    # Incoming action is 2 at t=0; outgoing action is always 0.
    rs.append(0.25 + zs[-1][:,0] + 0.7*act)
    before.append(stopped.copy())
    _,ks=jax.random.split(k)
    nxt_frame,nxt,real_r,d=jax.vmap(S.make(spec)[1])(states,jnp.asarray(act),jax.random.split(ks,B))
    true_rs.append(np.where(stopped,0,np.asarray(real_r)))
    current=np.stack([current[b] if stopped[b] else np.asarray(nxt_frame[b]) for b in range(B)])
    states=jax.tree.map(lambda old,new:jnp.asarray(np.stack([np.asarray(old[b]) if stopped[b] else np.asarray(new[b]) for b in range(B)])),states,nxt)
    stopped=np.logical_or(stopped,np.asarray(d));ds.append(stopped.copy())
zs,rs,ds,before,true_rs=map(np.stack,(zs,rs,ds,before,true_rs))
np.testing.assert_array_equal(batch['obs'],zs)
alive=np.concatenate([np.ones((1,B)),np.cumprod(1-ds[:-1],axis=0)],axis=0)
np.testing.assert_array_equal(batch['alive'],alive)
assert np.flatnonzero(ds[:,0])[0]==0 and np.flatnonzero(ds[:,1])[0]==2 and np.flatnonzero(ds[:,3])[0]==1
assert not ds[:,2].any()  # true horizon truncation must bootstrap
v=float(np.asarray(P.hl_expect(FixedAC().step(jnp.zeros((1,512)),jnp.zeros(1,jnp.int32),jnp.zeros((1,P.PHID)))[1]))[0])
adv=np.zeros(B,np.float32); reference=[]
for t in reversed(range(H)):
    delta=rs[t]+args.gamma*(1-ds[t])*v-v
    adv=delta+args.gamma*args.lam*(1-ds[t])*adv
    reference.append(adv.copy())
reference=np.stack(reference[::-1])
np.testing.assert_allclose(batch['adv'],reference,rtol=2e-6,atol=2e-6)
np.testing.assert_allclose(batch['ret'],reference+v,rtol=2e-6,atol=2e-6)
np.testing.assert_allclose(diag[0],np.sum(rs*alive)/(alive.sum()+1e-6),rtol=1e-6)
np.testing.assert_allclose(diag[1],ds.mean(),rtol=1e-6)
np.testing.assert_allclose(diag[2],np.sum(rs*alive)/B,rtol=1e-6)
assert diag[5]==0 and diag[6]==0 and float(np.asarray(old_diag)[5])>0 and float(np.asarray(old_diag)[6])>0
assert np.all(rs[before]>0)  # zero effective reward is due solely to done masking
for b,t in ((0,0),(1,2),(3,1)):
    assert alive[t,b]==1 and rs[t,b]>0
    np.testing.assert_allclose(batch['ret'][t,b],rs[t,b],rtol=1e-6,atol=1e-6)
assert not np.allclose(batch['ret'][:,2],rs[:,2]) # surviving path includes future/bootstrap
# Real MLP and native-grid constructors, one finite update, and save/load parity.
torsos={}
for torso in ('mlp','native_grid'):
    ac1=G.make_ac(nnx.Rngs(2),512,torso); ac2=G.make_ac(nnx.Rngs(2),512,torso)
    real_batch,real_diag=N.make_oracle_collector(bundle,spec,args)(ac1,ctx,ca,reset,state,key)
    opt=lambda ac:nnx.Optimizer(ac,optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(args.lr,weight_decay=0.01)),wrt=nnx.Param)
    loss1=np.asarray(N.make_update(args)(ac1,opt(ac1),real_batch,jnp.asarray(0),jnp.asarray(0)))
    loss2=np.asarray(O.make_update(args)(ac2,opt(ac2),real_batch,jnp.asarray(0),jnp.asarray(0)))
    np.testing.assert_array_equal(loss1,loss2);assert np.isfinite(loss1).all()
    for p1,p2 in zip(jax.tree.leaves(nnx.state(ac1,nnx.Param)),jax.tree.leaves(nnx.state(ac2,nnx.Param))):
        np.testing.assert_array_equal(np.asarray(p1),np.asarray(p2));assert np.isfinite(p1).all()
    with tempfile.TemporaryDirectory(prefix='termination-cpu-') as d:
        dest=Path(d)/'model.pkl'; (P.save_ac if torso=='mlp' else G.save_ac)(ac1,dest,{'check':True})
        restored,meta=G.load_ac(dest);assert meta=={'check':True}
        for p1,p2 in zip(jax.tree.leaves(nnx.state(ac1,nnx.Param)),jax.tree.leaves(nnx.state(restored,nnx.Param))):
            np.testing.assert_array_equal(np.asarray(p1),np.asarray(p2))
    torsos[torso]={'finite_loss':loss1.tolist(),'update_parameter_parity':'exact','save_load_parameter_parity':'exact','params':P.n_params(ac1)}
assert O.sha256(O.__file__)==N.FROZEN_PARENT_SHA256
result={'status':'passed','platform':str(jax.devices()),'jax':jax.__version__,'parent_sha256':O.sha256(O.__file__),'sensitivity_sha256':O.sha256(N.__file__),'collector_checks':['finite','current_state_outgoing_action_reward','first_terminal_transition_keeps_reward','absorbing_reencoded_frames','true_death_and_cap_done','horizon_bootstrap','zero_postterminal_alive_mass','zero_postterminal_weighted_reward','positive_raw_postterminal_reward_unmodified','old_oracle_observation_action_parity'],'true_done_indices':[0,2,None,1],'new_diagnostics':diag.tolist(),'old_oracle_heads_diagnostics':np.asarray(old_diag).tolist(),'torsos':torsos}
Path('/private/tmp/termination_cpu_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
