"""CPU-only execution of actual adapter AST with NumPy scatter/vmap shims.
This checks indexing and state math, not JAX/Pallas compilation or WM numerics.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent
class Scatter(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == 'set' and isinstance(f.value, ast.Subscript)
                and isinstance(f.value.value, ast.Attribute) and f.value.value.attr == 'at'):
            return ast.copy_location(ast.Call(func=ast.Name('scatter', ast.Load()),
                args=[f.value.value.value, f.value.slice, *node.args], keywords=[]), node)
        return node

def scatter(x, index, value):
    x = x.copy()
    x[index] = value
    return x

def vmap(fn):
    return lambda *xs: np.stack([fn(*(x[i] for x in xs)) for i in range(len(xs[0]))])

def scan(fn, carry, xs):
    ys=[]
    for i in range(len(xs[0])):
        carry, y=fn(carry, tuple(x[i] for x in xs))
        ys.append(y)
    return carry, np.stack(ys)

jax = SimpleNamespace(jit=lambda f:f, vmap=vmap,
    random=SimpleNamespace(split=lambda key,num=2:np.arange(num)+int(key)*100,
                           uniform=lambda key,shape:np.full(shape,.5),
                           randint=lambda key,shape,lo,hi:np.arange(np.prod(shape)).reshape(shape)%hi),
    lax=SimpleNamespace(cond=lambda p,yes,no,x:yes(x) if p else no(x), scan=scan))

class FakeWM:
    win=32
    ctx_signal=.875
    state=None
    def __init__(self, path=None): self.calls=[]
    def encode(self, frames):
        return np.repeat(frames[...,0,0,0,None].astype(float),2,-1)
    def decode(self,z): return z
    def flow(self,state,z,tau,actions,slot,key,nfe):
        self.calls.append((z.copy(),tau.copy(),actions.copy(),int(slot),int(key)))
        assert z.shape[:2]==(1,32) and np.prod(z.shape[2:])==2
        assert tau.shape==actions.shape==(1,32)
        assert np.shape(slot)==()
        assert tau[0,slot]==0 and (tau[0,slot+1:]==0).all()
        # Deterministic action/history dependent stand-in for native flow.
        # Future slots cannot contribute; two channels detect misalignment.
        live=tau>0
        return np.array([[np.sum(z.reshape(1,32,2)[...,0]*live)+actions[0,slot],
                          np.sum(actions*live)+slot]]).reshape((1,)+z.shape[2:])
    def read(self,state,z,actions,key):
        reward=np.broadcast_to(np.arange(32),(len(z),32))+np.arange(len(z))[:,None]*1000
        return reward, np.broadcast_to(np.arange(32)%2,(len(z),32)).astype(float)

def load_function(path,name,env):
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.decorator_list=[]
    mod=ast.fix_missing_locations(Scatter().visit(ast.Module([fn],type_ignores=[])))
    exec(compile(mod,str(path),'exec'),env)
    return env[name]

env=dict(np=np,jnp=np,jax=jax,scatter=scatter,WorldModel=FakeWM,hashlib=__import__('hashlib'))
transition=load_function(ROOT/'inventory_flow.py','transition',env)
make_env=load_function(ROOT/'inventory_flow.py','make_env',env)

class FlowASTTests(unittest.TestCase):
    def test_genuine_prefix_incoming_actions(self):
        values=np.array([np.arange(8),np.arange(8)+20],np.uint8)
        frames=np.broadcast_to(values[:,:,None,None,None],(2,8,63,63,3))
        actions=np.array([np.arange(7),np.arange(7)+8],np.int32)
        wm,reset,_=make_env('unused',reset_contexts_u8=frames,reset_actions=actions)
        obs,s=reset(np.array([3,4]),np.array([1,0]))
        np.testing.assert_allclose(obs[:,0,0,0],[27/255,7/255])
        np.testing.assert_array_equal(s['seq'][:,:8,0],values[[1,0]])
        np.testing.assert_array_equal(s['aseq'][:,:8],[[0,8,9,10,11,12,13,14],[0,0,1,2,3,4,5,6]])
        self.assertFalse(s['seq'][:,8:].any())
        self.assertFalse(s['aseq'][:,8:].any())
        np.testing.assert_array_equal(s['n'],[8,8]); np.testing.assert_array_equal(s['r'],[8,8])

    def test_async_fill_and_slide_and_head_slot(self):
        wm=FakeWM()
        z=np.broadcast_to(np.arange(32)[None,:,None],(4,32,2)).copy().astype(float)
        acts=np.broadcast_to(np.arange(32),(4,32)).copy()
        s=dict(seq=z,aseq=acts,r=np.array([8,8,8,0]),n=np.array([8,31,32,32]),t=np.array([0,0,1023,1]))
        _,out,rew,done,_=transition(wm,s,np.array([3,4,5,6]),7,8,9)
        np.testing.assert_array_equal(out['n'],[9,32,32,32]);np.testing.assert_array_equal(out['r'],[8,8,7,0])
        np.testing.assert_array_equal(rew,[8,1031,2031,3031]);np.testing.assert_array_equal(done,[0,1,1,1])
        np.testing.assert_array_equal(out['seq'][0,:8],z[0,:8])
        np.testing.assert_array_equal(out['seq'][1,:31],z[1,:31])
        np.testing.assert_array_equal(out['seq'][2,:31],z[2,1:])
        np.testing.assert_array_equal(out['aseq'][2,:31],acts[2,1:])
        self.assertEqual([x[3] for x in wm.calls],[8,31,31,31])
        self.assertEqual([int((x[1]==1).sum()) for x in wm.calls],[8,8,7,0])
        self.assertEqual([int((x[1]==.875).sum()) for x in wm.calls],[0,23,24,31])
        np.testing.assert_array_equal(s['seq'],z) # input remains immutable

    def test_matches_native_rollout_state_math_through_prefix_aging(self):
        wm=FakeWM()
        native_env=dict(env,CTX_SIGNAL=wm.ctx_signal,STEPS=4,WIN=32)
        native_env['_flow_one']=lambda apply,z,t,a,n,key,n_steps:wm.flow(None,z,t,a,n,key,n_steps)
        native=load_function(ROOT/'inventory_source/wm_generic.py','_rollout_latents',native_env)
        ctx=np.arange(16,dtype=float).reshape(1,8,1,1,2)/20
        ca=np.arange(8,dtype=np.int32)[None]
        actions=(np.arange(45,dtype=np.int32)%17)[:,None]
        expected=native(None,ctx,ca,actions,17)
        seq=np.zeros((1,32,1,1,2));seq[:,:8]=ctx
        aseq=np.zeros((1,32),np.int32);aseq[:,:8]=ca
        s=dict(seq=seq,aseq=aseq,n=np.array([8]),r=np.array([8]),t=np.array([0]))
        actual=[]
        for t,a in enumerate(actions):
            _,s,_,_,_=transition(wm,s,a,t,5,6)
            actual.append(s['seq'][:,s['n'][0]-1])
            if t==23: self.assertEqual((s['n'][0],s['r'][0]),(32,8))
            if t==24: self.assertEqual((s['n'][0],s['r'][0]),(32,7))
            if t==31: self.assertEqual((s['n'][0],s['r'][0]),(32,0))
        np.testing.assert_array_equal(np.stack(actual),expected)

if __name__=='__main__': unittest.main(verbosity=2)
