"""Clean historical Flow simulator; raw native heads, no task-specific reward hook.

Dynamics/heads use the frozen native model. Initialization uses genuine eight-frame
episode prefixes and the native expanding-window rollout; historical wm_gym
reset padding is not used.
The supplied shared reset bank fixes candidate contexts across model families.
"""
import os
import hashlib
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

SOURCE = Path(__file__).with_name("inventory_source")
sys.path.insert(0, str(SOURCE))
os.environ["TINYWM2_ROOT"] = str(SOURCE)
os.environ["TINYWM2_MODULE"] = "wm_generic"
from wm_env2 import WorldModel


def transition(wm, state, act, flow_key, head_key, done_key, nfe=4, max_steps=1024):
    """Native expanding-prefix rollout, vectorized over asynchronous episode ages.

    _flow_one receives its unchanged scalar slot for each environment. Causal
    attention excludes the unused future slots until the prefix fills the window.
    """
    N, win = act.shape[0], wm.win
    full = state["n"] == win
    def slide(v):
        candidate = jnp.concatenate([v[:, 1:], v[:, -1:]], 1)
        return jnp.where(full.reshape((N,) + (1,) * (v.ndim - 1)), candidate, v)
    seq, aseq = slide(state["seq"]), slide(state["aseq"])
    r = jnp.where(full, jnp.maximum(state["r"] - 1, 0), state["r"])
    n = jnp.minimum(state["n"], win - 1)
    aseq = aseq.at[jnp.arange(N), n].set(act)
    index = jnp.arange(win)[None]
    taus = jnp.where(index < n[:, None], jnp.where(index < r[:, None], 1.0, wm.ctx_signal), 0.0)
    # The source scalar-slot API is used unchanged, including its native cached attention.
    one = lambda z, tau, actions, slot, key: wm.flow(
        wm.state, z[None], tau[None], actions[None], slot, key, nfe)[0]
    z = jax.vmap(one)(seq, taus, aseq, n, jax.random.split(flow_key, N))
    seq = seq.at[jnp.arange(N), n].set(z)
    rew, pdone = wm.read(wm.state, seq, aseq, head_key)
    rew, pdone = rew[jnp.arange(N), n], pdone[jnp.arange(N), n]
    t = state["t"] + 1
    done = (jax.random.uniform(done_key, (N,)) < pdone) | (t >= max_steps)
    obs = wm.decode(z).astype(jnp.float32) / 255.0
    return obs, {"seq": seq, "aseq": aseq, "r": r, "n": n + 1, "t": t}, rew, done, {}


def make_env(wm_path, data_path=None, nfe=4, max_steps=1024, chunk=64, *,
             reset_contexts_u8, reset_actions):
    del data_path
    wm = WorldModel(wm_path)
    def parameter_sha256():
        digest = hashlib.sha256()
        for name, state in [('dynamics', wm.state), ('encoder', wm.estate), ('decoder', wm.dstate)]:
            digest.update(name.encode())
            for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
                value = np.asarray(value)
                digest.update(str(path).encode())
                digest.update(str(value.dtype).encode())
                digest.update(str(value.shape).encode())
                digest.update(value.tobytes())
        return digest.hexdigest()
    wm.parameter_sha256 = parameter_sha256
    win = wm.win
    contexts, outgoing = np.asarray(reset_contexts_u8), np.asarray(reset_actions)
    if contexts.dtype != np.uint8 or contexts.shape[1:] != (8, 63, 63, 3):
        raise ValueError("Expected genuine uint8 context bank [R,8,63,63,3]")
    if outgoing.shape != (len(contexts), 7) or not np.all((outgoing >= 0) & (outgoing < 17)):
        raise ValueError("Expected seven native outgoing prefix actions")
    start = jax.jit(wm.encode)
    prefix = jnp.concatenate([start(jnp.asarray(contexts[i:i + chunk])) for i in range(0, len(contexts), chunk)])
    rz = jnp.zeros((len(contexts), win) + prefix.shape[2:]).at[:, :8].set(prefix)
    incoming = np.concatenate([np.zeros((len(contexts), 1), np.int32), outgoing.astype(np.int32)], 1)
    ra = jnp.zeros((len(contexts), win), jnp.int32).at[:, :8].set(jnp.asarray(incoming))
    rf = jnp.asarray(contexts[:, -1])

    def vreset(keys, indices=None):
        N = keys.shape[0]
        ids = jax.random.randint(keys[0], (N,), 0, len(contexts)) if indices is None else indices
        state = {"seq": rz[ids], "aseq": ra[ids], "r": jnp.full((N,), 8, jnp.int32),
                 "n": jnp.full((N,), 8, jnp.int32), "t": jnp.zeros((N,), jnp.int32)}
        return rf[ids].astype(jnp.float32) / 255.0, state

    def vstep(keys, state, act):
        kf, kh, kd = jax.random.split(keys[0], 3)
        return transition(wm, state, act, kf, kh, kd, nfe=nfe, max_steps=max_steps)

    return wm, vreset, vstep
