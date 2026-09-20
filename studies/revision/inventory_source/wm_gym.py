"""The world model as a drop-in replacement for the Craftax vector environment.

`make_env` returns the `(vreset, vstep)` pair `craftax_env.make_env` returns, with the same signatures and the
same semantics, so `train_ppo_real.py` runs against the model with no other change: same PPO, same geometry, same
hyperparameters, same evaluator (which stays in real Craftax).

The env state is the model's own rollout state: a full 32-slot latent window whose last slot is the newest frame,
the actions that produced each slot, and `r`, the number of leading slots that are still real frames. A reset
fills every slot with the encoded reset frame (r = win) and hands back that frame as the observation; a step
slides the window one slot, writes the action, samples the next latent with `_flow_one` at 4 Euler steps, decodes
it to pixels, and reads reward and termination off the model's heads. That is exactly the regime
`_rollout_latents` runs in once its window is full, with `r` decaying one slot per step as the real context ages
out. Termination is sampled from the head's probability, so `done` is hard, as in the real env.

Reset frames come from the dataset's episode starts, encoded once into a lookup table at startup: the trainer
calls `vreset` on every step to auto-reset finished envs, so a reset has to be a gather, not a forward pass.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import wm_env2
from agent import A_N


def reset_frames(path):
    """Every episode-start frame in the dataset: the first row and every row after a done."""
    d = np.load(path)
    dn = np.asarray(d["dones"])
    T, E = dn.shape
    rt, re = np.nonzero(np.concatenate([np.ones((1, E), bool), dn[:T - 1] > 0.5], 0))
    return np.asarray(d["frames"])[rt, re]


CRAFT0 = 7


def make_env(wm_path, data_path, nfe=4, max_steps=1024, once=False, chunk=64):
    """`once`: pay a place/make action (actions 7-16) at most once per episode. Craftax pays every achievement
    once per episode; the head reads a 32-frame window and cannot know whether this furnace is the first, so it
    pays every one, and PPO learns to place furnaces on repeat. Collect/eat/drink events all go through DO and
    are left to the head: the ones worth farming leave an inventory count on screen that the head can read."""
    wm = wm_env2.WorldModel(wm_path)
    win = wm.win
    RF = jnp.asarray(reset_frames(data_path))
    R = RF.shape[0]
    start = jax.jit(lambda f: wm.encode(jnp.repeat(f[:, None], win, 1)))
    RZ = jnp.concatenate([start(RF[i:i + chunk]) for i in range(0, R, chunk)])
    slot = win - 1
    i_slot = jnp.arange(win)

    def vreset(keys):
        N = keys.shape[0]
        i = jax.random.randint(keys[0], (N,), 0, R)
        state = {"seq": RZ[i], "aseq": jnp.zeros((N, win), jnp.int32),
                 "r": jnp.full((N,), win, jnp.int32), "t": jnp.zeros((N,), jnp.int32),
                 "paid": jnp.zeros((N,), jnp.int32)}
        return RF[i].astype(jnp.float32) / 255.0, state

    def vstep(keys, state, act):
        N = act.shape[0]
        kf, kh, kd = jax.random.split(keys[0], 3)
        slide = lambda v: jnp.concatenate([v[:, 1:], v[:, -1:]], 1)
        seq, aseq = slide(state["seq"]), slide(state["aseq"])
        r = jnp.maximum(state["r"] - 1, 0)
        aseq = aseq.at[:, slot].set(act)
        taus = jnp.where(i_slot < slot, jnp.where(i_slot < r[:, None], 1.0, wm.ctx_signal), 0.0)
        z = wm.flow(wm.state, seq, taus, aseq, slot, kf, nfe)
        seq = seq.at[:, slot].set(z)
        rew, pdone = wm.read(wm.state, seq, aseq, kh)
        rew = rew[:, slot]
        paid = state["paid"]
        if once:
            craft = act >= CRAFT0
            bit = jnp.left_shift(1, jnp.clip(act - CRAFT0, 0, A_N - CRAFT0 - 1))
            seen = craft & (jnp.bitwise_and(paid, bit) > 0)
            paid = jnp.where(craft & (rew > 0.5), jnp.bitwise_or(paid, bit), paid)
            rew = jnp.where(seen, jnp.minimum(rew, 0.0), rew)
        t = state["t"] + 1
        done = (jax.random.uniform(kd, (N,)) < pdone[:, slot]) | (t >= max_steps)
        obs = wm.decode(z).astype(jnp.float32) / 255.0
        return obs, {"seq": seq, "aseq": aseq, "r": r, "t": t, "paid": paid}, rew, done, {}

    return wm, vreset, vstep
