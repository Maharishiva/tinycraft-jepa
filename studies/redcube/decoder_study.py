"""Frozen-latent decoder experiment inspired by UL's conditional diffusion decoder.

This is NOT full Unified Latents: encoder/dynamics are frozen; no latent prior is trained.
Train matched MSE and VP-diffusion decoders, evaluate true and predicted latents separately.
No test frames, positions or actions enter decoder training. Position probes are diagnostics.
"""
import argparse
import functools
import gc
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import eval as E
import probe as P
import train as T
from studies.redcube import flow as F
from studies.redcube import bench as B

MP = dict(dtype=jnp.bfloat16, param_dtype=jnp.float32)

def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))

def prepare(a):
    """Cache only latents and evaluation observations; read training pixels from their original file."""
    a.out.mkdir(parents=True, exist_ok=True)
    cache_path = a.out/'cache.npz'
    if cache_path.exists():
        print('[prepare] using existing cache', flush=True)
        return
    started = time.time()
    ckpt = a.root/a.arm/('model.npz' if a.arm == 'flow' else 'model_probe.npz')
    train = E.load_data(a.root/'train.npz')
    test = E.load_data(a.root/'eval.npz')
    fit_frames = train['frames'][:a.rows]
    indices = E.sample_indices(test, 2048)[:a.windows]
    frames, actions = E.window_batch(test, indices)
    times = indices[:, :1] + np.arange(24)[None]
    pos = test['pos'][times, indices[:, 1:]].astype(np.float32)
    if a.arm == 'flow':
        fm, _, model = F.load(a.flow_code, ckpt, 4)
        zfit = B.encode_flow(fm, model, fit_frames.transpose(1,0,2,3,4), chunk=2).transpose(1,0,2).reshape(-1,512)
        ztrue = B.encode_flow(fm, model, frames, chunk=8)
        oldtrue = np.concatenate([np.asarray(fm.decode(model, fm.encode(model,jnp.asarray(frames[i:i+8]))))
                                  for i in range(0,len(frames),8)])
        parts = [B.flow_rollout(fm,model,frames[i:i+32,:8],actions[i:i+32,:8],actions[i:i+32,8:],
                               jax.random.PRNGKey(710+i)) for i in range(0,len(frames),32)]
        meta = fm.load_meta(ckpt)
        tokens = 16
    else:
        bundle, meta = T.load_model(ckpt)
        zfit = B.encode_jepa(bundle,fit_frames).reshape(-1,bundle.model.latent_dim)
        ztrue = B.encode_jepa(bundle,frames)
        oldtrue = np.concatenate([np.asarray(B.jepa_reconstruct(bundle.model,bundle.decoder,jnp.asarray(frames[i:i+32])))
                                  for i in range(0,len(frames),32)])
        parts = [B.jepa_rollout(bundle,frames[i:i+32,:8],actions[i:i+32,:8],actions[i:i+32,8:])
                 for i in range(0,len(frames),32)]
        tokens = bundle.model.n_tokens
    oldpred,zpred = (np.concatenate([p[k] for p in parts]) for k in (0,1))
    # Covariance whitening fitted to TRAIN latents only, with the same 1%-std floor as the prior probe.
    mu = zfit.astype(np.float64).mean(0)
    cov = np.cov(zfit.astype(np.float64),rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    sd = np.sqrt(np.maximum(values,0))
    basis = vectors[:,::-1] / np.maximum(sd[::-1], .01*sd[-1])[None]
    basis = basis.astype(np.float32)
    stats = {'arm':a.arm,'config':a.root.name,'rows':a.rows,'windows':a.windows,'tokens':tokens,
             'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),'checkpoint_meta':meta,
             'dataset_sha256_recorded':meta['data_sha256'],'eval_sha256':sha(a.root/'eval.npz'),
             'source_sha256':sha(__file__),'elapsed_s':time.time()-started,
             'conditioning':'train-only full PCA whitening; std floor 1% of largest; no latent noise',
             'training_scope':'first rows of training split; separate eval split; fixed final checkpoint'}
    np.savez(cache_path,zfit=zfit,mu=mu.astype(np.float32),basis=basis,spectrum=sd[::-1],ztrue=ztrue,zpred=zpred,
             frames=frames,actions=actions,pos=pos,indices=indices,oldtrue=oldtrue,oldpred=oldpred,
             train_pos=train['pos'][:a.rows].reshape(-1,2).astype(np.float32))
    write_json(a.out/'provenance.json',stats)
    print('[prepare]',json.dumps({k:stats[k] for k in ('arm','config','elapsed_s','checkpoint_sha256')}),flush=True)

def vp(logsnr):
    return jnp.sqrt(jax.nn.sigmoid(logsnr)), jnp.sqrt(jax.nn.sigmoid(-logsnr))

def diffusion_weight(logsnr, bias=0.):
    """UL Fig.3 sigmoid(b-lambda) epsilon-MSE weighting, converted to v-MSE.

    epsilon_error^2 = alpha^2 * v_error^2. The linear log-SNR schedule derivative is constant.
    No claim of an ELBO/bits bound is made for this frozen, unregularized encoder experiment.
    """
    return jax.nn.sigmoid(logsnr) * jax.nn.sigmoid(bias-logsnr)

class Res(nnx.Module):
    def __init__(self,ch,rngs):
        self.n1=nnx.RMSNorm(ch,rngs=rngs)
        self.n2=nnx.RMSNorm(ch,rngs=rngs)
        self.c1=nnx.Conv(ch,ch,(3,3),padding='SAME',rngs=rngs,**MP)
        self.c2=nnx.Conv(ch,ch,(3,3),padding='SAME',rngs=rngs,**MP)
        self.time=nnx.Linear(192,2*ch,rngs=rngs)
    def __call__(self,x,t):
        scale,shift=jnp.split(self.time(nnx.silu(t)),2,-1)
        h=self.n1(x)*(1+scale[:,None,None])+shift[:,None,None]
        return x+self.c2(nnx.silu(self.n2(self.c1(nnx.silu(h)))))

class Denoiser(nnx.Module):
    def __init__(self,dim,seed=7):
        r=nnx.Rngs(seed)
        self.z1=nnx.Linear(dim,512,rngs=r)
        self.z2=nnx.Linear(512,8*8*32,rngs=r)
        self.t1=nnx.Linear(64,192,rngs=r)
        self.t2=nnx.Linear(192,192,rngs=r)
        self.stem=nnx.Conv(3,64,(3,3),strides=(2,2),padding='SAME',rngs=r,**MP)
        self.cond=nnx.Conv(32,64,(1,1),rngs=r,**MP)
        self.r32=Res(64,r)
        self.d16=nnx.Conv(64,128,(3,3),strides=(2,2),padding='SAME',rngs=r,**MP)
        self.r16=Res(128,r)
        self.d8=nnx.Conv(128,192,(3,3),strides=(2,2),padding='SAME',rngs=r,**MP)
        self.mid=nnx.List([Res(192,r),Res(192,r)])
        self.u16=nnx.Conv(192+128,128,(3,3),padding='SAME',rngs=r,**MP)
        self.o16=Res(128,r)
        self.u32=nnx.Conv(128+64,64,(3,3),padding='SAME',rngs=r,**MP)
        self.o32=Res(64,r)
        self.out=nnx.Conv(64,3,(3,3),padding='SAME',rngs=r,**MP)
    def __call__(self,x,logsnr,z):
        n=len(z)
        phase=logsnr[:,None]*jnp.exp(jnp.linspace(0,jnp.log(100.),32))[None]
        t=self.t2(nnx.silu(self.t1(jnp.concatenate([jnp.sin(phase),jnp.cos(phase)],-1))))
        cond=self.z2(nnx.gelu(self.z1(z))).reshape(n,8,8,32)
        resize=lambda h,s:jax.image.resize(h,(n,s,s,h.shape[-1]),method='nearest')
        h=self.stem(jnp.pad(x,((0,0),(0,1),(0,1),(0,0))))
        h32=self.r32(h+self.cond(resize(cond,32)),t)
        h16=self.r16(self.d16(h32),t)
        h=self.d8(h16)
        for block in self.mid:h=block(h,t)
        h=self.o16(self.u16(jnp.concatenate([resize(h,16),h16],-1)),t)
        h=self.o32(self.u32(jnp.concatenate([resize(h,32),h32],-1)),t)
        return self.out(resize(h,64))[:,:63,:63].astype(jnp.float32)

def normalize(z,cache):
    return (z-cache['mu'])@cache['basis']

def save_decoder(path,model,meta):
    state=nnx.state(model,nnx.Param)
    np.savez(path,meta=json.dumps(meta),**{f'p{i}':np.asarray(v) for i,v in enumerate(jax.tree.leaves(state))})

def load_decoder(path,dim):
    a=np.load(path);meta=json.loads(str(a['meta']))
    model=Denoiser(dim,meta['seed'])
    gd,params,rest=nnx.split(model,nnx.Param,...)
    leaves=[jnp.asarray(a[f'p{i}']) for i in range(len(jax.tree.leaves(params)))]
    return nnx.merge(gd,jax.tree.unflatten(jax.tree.structure(params),leaves),rest),meta

def fit(a):
    cache=np.load(a.out/'cache.npz')
    with np.load(a.root/'train.npz') as d:
        pixels=jnp.asarray(d['frames'][:a.rows].reshape(-1,63,63,3))
    for kind in a.kinds.split(','):
        if kind not in ('mse','mse_raw','diffusion','diffusion_x'):
            raise ValueError(kind)
        latents=jnp.asarray(cache['zfit'] if kind=='mse_raw' else normalize(cache['zfit'],cache),jnp.float32)
        path=a.out/f'{kind}.npz'
        if path.exists():
            print('[fit] exists',path,flush=True);continue
        is_diffusion=kind.startswith('diffusion')
        model=(load_decoder(a.out/'mse.npz',latents.shape[-1])[0] if kind=='diffusion_x'
               else Denoiser(latents.shape[-1],a.seed))
        schedule=optax.warmup_cosine_decay_schedule(0.,a.lr,min(200,a.steps//5),a.steps,end_value=a.lr*.05)
        opt=nnx.Optimizer(model,optax.chain(optax.clip_by_global_norm(1.),optax.adamw(schedule,weight_decay=1e-4)),wrt=nnx.Param)
        @nnx.jit
        def run(model,opt,pixels,latents,keys):
            @nnx.scan(in_axes=(nnx.Carry,0),out_axes=(nnx.Carry,0))
            def body(carry,key):
                m,o=carry
                ki,kt,kn=jax.random.split(key,3)
                idx=jax.random.randint(ki,(a.batch,),0,len(pixels))
                x=pixels[idx].astype(jnp.float32)/127.5-1
                z=latents[idx]
                lam=jax.random.uniform(kt,(a.batch,),minval=-12.,maxval=12.)
                alpha,sigma=vp(lam)
                al,si=alpha[:,None,None,None],sigma[:,None,None,None]
                eps=jax.random.normal(kn,x.shape)
                xt=al*x+si*eps if is_diffusion else jnp.zeros_like(x)
                target=al*eps-si*x if kind=='diffusion' else x
                logsnr=lam if is_diffusion else jnp.zeros_like(lam)
                weight=(diffusion_weight(lam)*12 if kind=='diffusion' else
                        jax.nn.sigmoid(lam)*12 if kind=='diffusion_x' else jnp.ones_like(lam))
                def loss(m):return jnp.mean(jnp.mean((m(xt,logsnr,z)-target)**2,(1,2,3))*weight)
                value,grads=nnx.value_and_grad(loss)(m)
                o.update(m,grads)
                return (m,o),value
            return body((model,opt),keys)[1]
        start=time.time();losses=[]
        keys=jax.random.split(jax.random.PRNGKey(a.seed+101),a.steps)
        for i in range(0,a.steps,100):
            block=np.asarray(run(model,opt,pixels,latents,keys[i:i+100]))
            if not np.isfinite(block).all():raise RuntimeError('Non-finite decoder loss')
            losses.extend(block.tolist())
            if i%500==0 or i+100>=a.steps:
                print(f'[fit {kind}] {min(i+100,a.steps)}/{a.steps} loss={np.mean(block):.6f} seconds={time.time()-start:.1f}',flush=True)
        meta={'kind':kind,'steps':a.steps,'seed':a.seed,'batch':a.batch,'lr':a.lr,'train_s':time.time()-start,
              'params':sum(v.size for v in jax.tree.leaves(nnx.state(model,nnx.Param))),
              'source_sha256':sha(__file__),'cache_provenance':json.loads((a.out/'provenance.json').read_text()),
              'loss':('UL-inspired sigmoid-weighted VP v-prediction' if kind=='diffusion' else
                      'UL-inspired sigmoid-weighted VP x-prediction; initialized from fitted MSE decoder' if kind=='diffusion_x' else 'pixel MSE'),
              'init_checkpoint_sha256':sha(a.out/'mse.npz') if kind=='diffusion_x' else None,
              'conditioning':'raw' if kind=='mse_raw' else 'whitened',
              'latent_noise':0,'logsnr_range':[-12,12],'last100_loss':float(np.mean(losses[-100:]))}
        save_decoder(path,model,meta)
        np.save(a.out/f'{kind}_losses.npy',losses)
        print('[saved]',path,flush=True)
        del model,opt,run;gc.collect();jax.clear_caches()

@functools.partial(nnx.jit,static_argnames=('kind','nfe'))
def decode(model,z,key,kind,nfe=32):
    if not kind.startswith('diffusion'):
        return jnp.clip(model(jnp.zeros((len(z),63,63,3)),jnp.zeros(len(z)),z),-1,1)
    initial=jax.random.normal(key,(len(z),63,63,3))
    levels=jnp.linspace(-12.,12.,nfe+1)
    def step(x,levels):
        now,nxt=levels
        al,si=vp(now);an,sn=vp(nxt)
        prediction=model(x,jnp.full((len(z),),now),z)
        clean=jnp.clip(prediction if kind=='diffusion_x' else al*x-si*prediction,-1,1)
        eps=(x-al*clean)/jnp.maximum(si,1e-5)
        return an*clean+sn*eps,clean
    _,clean=jax.lax.scan(step,initial,(levels[:-1],levels[1:]))
    return clean[-1]

def decode_all(model,z,kind,seed,nfe,batch=64):
    flat=z.reshape(-1,z.shape[-1]);parts=[]
    for i in range(0,len(flat),batch):
        y=np.asarray(decode(model,jnp.asarray(flat[i:i+batch]),jax.random.fold_in(jax.random.PRNGKey(seed),i),kind,nfe))
        parts.append(np.clip((y+1)*127.5+.5,0,255).astype(np.uint8))
    return np.concatenate(parts).reshape(z.shape[:-1]+(63,63,3))

def metric(images,pos,cnn):
    est=cnn(images).reshape(pos.shape)
    error=np.abs(est-pos).sum(-1)
    integer_pos=np.rint(pos).astype(np.int64)
    red=B.cube_stats(images if images.ndim==5 else images[:,None],integer_pos if pos.ndim==3 else integer_pos[:,None])
    return {'l1':float(error.mean()),'within3':float((error<=3).mean()),'per_step_l1':error.mean(0).tolist(),
            'red_mass_at_true':float(np.mean(red['mass_at_true']))},est

def evaluate(a):
    c=np.load(a.out/'cache.npz');prov=json.loads((a.out/'provenance.json').read_text())
    frames,pos=c['frames'],c['pos'];true,pred=c['ztrue'],c['zpred']
    with np.load(a.root/'train.npz') as d:fit_pixels=d['frames'][:a.rows].reshape(-1,63,63,3)
    print('[eval] fitting shared pixel and latent probes on train split',flush=True)
    cnn=B.cnn_fit(fit_pixels,c['train_pos'],steps=a.probe_steps)
    latent=B.tile_fit(c['zfit'],c['train_pos'],tokens=prov['tokens'],steps=6000)
    global_probe=B.tile_fit(c['zfit'],c['train_pos'],tokens=1,steps=6000) if prov['tokens']>1 else latent
    latent_true=latent(true.reshape(-1,true.shape[-1])).reshape(pos.shape)
    latent_pred=latent(pred.reshape(-1,pred.shape[-1])).reshape(pos[:,8:].shape)
    err=lambda guess,target:float(np.abs(guess-target).sum(-1).mean())
    report={'provenance':prov,'evaluation_source_sha256':sha(__file__),'windows':len(frames),'nfe':a.nfe,'decoder_seeds':list(range(a.samples)),
            'old':{},'decoders':{},'latent_probe':{'true':err(latent_true[:,8:],pos[:,8:]),
              'predicted':err(latent_pred,pos[:,8:]),'predicted_by_step':np.abs(latent_pred-pos[:,8:]).sum(-1).mean(0).tolist()},
            'global_probe_true':err(global_probe(true[:,8:].reshape(-1,true.shape[-1])).reshape(pos[:,8:].shape),pos[:,8:])}
    visuals={'gt':frames[:3],'pos':pos[:3],'latent_probe_true':latent_true[:3],'latent_probe_pred':latent_pred[:3]}
    report['real_pixel_probe'],gt_coords=metric(frames[:,8:],pos[:,8:],cnn)
    visuals['gt_pixel_probe']=gt_coords[:3]
    for label,img in [('true',c['oldtrue'][:,8:]),('predicted',c['oldpred'])]:
        report['old'][label],coords=metric(img,pos[:,8:],cnn)
        visuals[f'old_{label}']=img[:3];visuals[f'old_{label}_probe']=coords[:3]
    # Same latent targets; quantify changes in raw versus whitened coordinates independently of decoder.
    target=true[:,8:];white_true=normalize(target,c);white_pred=normalize(pred,c)
    report['latent_error']={'raw_mse_by_step':np.mean((pred-target)**2,(0,2)).tolist(),
        'white_mse_by_step':np.mean((white_pred-white_true)**2,(0,2)).tolist(),
        'copy_white_mse_by_step':np.mean((normalize(true[:,7:8],c)-white_true)**2,(0,2)).tolist(),
        'top48_white_mse':float(np.mean((white_pred[...,:48]-white_true[...,:48])**2)),
        'tail_white_mse':float(np.mean((white_pred[...,48:]-white_true[...,48:])**2))}
    for kind in a.kinds.split(','):
        model,meta=load_decoder(a.out/f'{kind}.npz',true.shape[-1])
        result={'meta':meta,'samples':[]}
        decoder_true=target if kind=='mse_raw' else white_true
        decoder_pred=pred if kind=='mse_raw' else white_pred
        for seed in range(a.samples if kind.startswith('diffusion') else 1):
            print(f'[eval] {kind} seed {seed}',flush=True)
            decoded_true=decode_all(model,decoder_true,kind,seed,a.nfe)
            decoded_pred=decode_all(model,decoder_pred,kind,seed,a.nfe)
            tr,tc=metric(decoded_true,pos[:,8:],cnn);pr,pc=metric(decoded_pred,pos[:,8:],cnn)
            tr['pixel_mse']=float(np.mean((decoded_true.astype(float)/255-frames[:,8:]/255)**2))
            result['samples'].append({'seed':seed,'true':tr,'predicted':pr})
            if seed==0:
                visuals[f'{kind}_true']=decoded_true[:3];visuals[f'{kind}_true_probe']=tc[:3]
                visuals[f'{kind}_predicted']=decoded_pred[:3];visuals[f'{kind}_predicted_probe']=pc[:3]
        # 512 true future frames, same noise seed with and without the correct latent.
        z=decoder_true.reshape(-1,true.shape[-1])[:512]
        p=pos[:,8:].reshape(-1,2)[:512]
        perm=np.random.default_rng(991).permutation(len(z))
        shuffled=decode_all(model,z[perm],kind,0,a.nfe)
        result['shuffled_latent'],_=metric(shuffled,p,cnn)
        report['decoders'][kind]=result
        write_json(a.out/f'{a.report_prefix}results.json',report)
        np.savez(a.out/f'{a.report_prefix}visuals.npz',**visuals)
        del model;gc.collect();jax.clear_caches()
    report['checkpoint_unchanged']=sha(prov['checkpoint'])==prov['checkpoint_sha256']
    assert report['checkpoint_unchanged']
    write_json(a.out/f'{a.report_prefix}results.json',report)
    print('[results]',json.dumps({k:[(r['true']['l1'],r['predicted']['l1']) for r in v['samples']]
                                for k,v in report['decoders'].items()}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','fit','eval','all','smoke'])
    p.add_argument('--root',type=Path,default=Path('/dev/shm/redcube/off_still'))
    p.add_argument('--flow-code',type=Path,default=Path.home()/'personal/codex-wm-push-20260916/recipes/model-51ffe8df')
    p.add_argument('--arm',default='lewm6')
    p.add_argument('--out',type=Path,default=Path('/tmp/decoder-study-smoke'))
    p.add_argument('--rows',type=int,default=800)
    p.add_argument('--windows',type=int,default=128)
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--nfe',type=int,default=32)
    p.add_argument('--samples',type=int,default=3)
    p.add_argument('--probe-steps',type=int,default=4000)
    p.add_argument('--kinds',default='mse,diffusion')
    p.add_argument('--report-prefix',default='')
    a=p.parse_args();E.N_ACTIONS=4
    if a.stage=='smoke':
        model=Denoiser(16);z=jnp.ones((2,16));x=jnp.zeros((2,63,63,3))
        y=model(x,jnp.zeros(2),z);assert y.shape==x.shape
        _,g=nnx.value_and_grad(lambda m:jnp.mean(m(x,jnp.zeros(2),z)**2))(model)
        assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(g))
        for kind in ['mse','diffusion']:
            y=decode(model,z,jax.random.PRNGKey(0),kind,4)
            assert y.shape==x.shape and np.isfinite(np.asarray(y)).all()
        # VP change of variables and exact denoising algebra.
        lam=jnp.array([-6.,0.,6.]);al,si=vp(lam)
        assert np.allclose(np.asarray(al**2+si**2),1,atol=1e-6)
        xt=al*.3+si*.7;v=al*.7-si*.3
        assert np.allclose(np.asarray(al*xt-si*v),.3,atol=1e-6)
        assert np.all(np.asarray(diffusion_weight(lam))>0)
        print('SMOKE PASSED: shape, finite gradients, both samplers, VP algebra',flush=True)
        return
    if a.stage in ('prepare','all'):prepare(a)
    if a.stage in ('fit','all'):fit(a)
    if a.stage in ('eval','all'):evaluate(a)

if __name__=='__main__':main()
