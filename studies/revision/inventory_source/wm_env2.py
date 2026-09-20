"""The September world model (PR #1: temporal codec, shortcut targets, NorMuon) as an RL environment.

Same contract as wm_env.py, against the newer training module: the checkpoint carries its own reward and
termination heads, the sampler feeds real context at signal 1.0 and generated frames re-noised to CTX_SIGNAL,
and the two-frame tokenizer is fed the context window so its temporal input is the real one. Half of the starts
are true reset states (the reset frame repeated as context, noop actions, zero GRU), the rest uniform dataset
states with their real 8-frame context. The policy acts on decoded pixels, as in real Craftax.

One thing wm_env.py can do and this cannot: combine two checkpoints pessimistically. That needs both models to
read the same latents, and the September training script always trains its own tokenizer (the August one could be
handed an existing one with --tok), so a second seed's heads would be reading a foreign latent space."""
from __future__ import annotations

import os

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import tinywm_module
from agent import PHID, gae, hl_expect, logprob

CTX = 8


class WorldModel:
    def __init__(self, path, path2=None):
        if path2 is not None:
            raise ValueError("a second checkpoint cannot score these latents: every September model has its own tokenizer")
        self.m = tinywm_module.load(os.environ.get("TINYWM2_ROOT", os.path.expanduser("~/personal/wm_rl_eval/sota2")),
                                    os.environ.get("TINYWM2_MODULE", "train"))
        self.model = self.m.load_model(str(path))
        self.win = self.model.win
        self.ctx_signal = self.m.CTX_SIGNAL
        self.gdef, self.state = nnx.split(self.model.df)
        self.egdef, self.estate = nnx.split(self.model.enc)
        self.dgdef, self.dstate = nnx.split(self.model.dec)
        self.model2, self.state2 = None, None

    def fresh(self):
        """The model rebuilt at the current trace level: the module jits its codec, and flax refuses to reuse a
        module built outside the trace that calls it."""
        return self.m.ModelState(nnx.merge(self.egdef, self.estate), nnx.merge(self.dgdef, self.dstate),
                                 nnx.merge(self.gdef, self.state), self.win)

    def encode(self, u8):
        return self.m.encode(self.fresh(), u8)

    def decode(self, z):
        return self.m.decode(self.fresh(), z)

    def taus(self, B, n, real):
        """Signal level of every slot: real context clean, generated frames at CTX_SIGNAL, the future at zero."""
        i = jnp.arange(self.win)
        return jnp.broadcast_to(jnp.where(i < n, jnp.where(i < real, 1.0, self.ctx_signal), 0.0), (B, self.win))

    def flow(self, wstate, seq, taus, aseq, n, key, nfe):
        return self.m._flow_one(nnx.merge(self.gdef, wstate), seq, taus, aseq, n, key, n_steps=nfe)

    def read(self, wstate, seq, aseq, key):
        """Reward per frame and termination probability per frame, from the model's own heads."""
        return self.m._read_heads(nnx.merge(self.gdef, wstate), seq, aseq, key)


def make_dream(wm, info, B, HOR, nfe, p_reset, pen, cond_reset, gamma, lam):
    """dream(ac, ref, wstate, wstate2, F, A, C, R_t, R_e, ulog, key) -> (time-major PPO batch, diagnostics).

    Identical batch and diagnostics to wm_env.make_dream, so the trainer does not care which model it dreams in;
    `pen` weighs the two models' disagreement there and has nothing to weigh here.

    The horizon is not capped by the model's window: once the window is full it slides one slot per step, the real
    context ages out of it, and the model runs on its own frames alone - the regime the scoring rollouts use. The
    heads are then read per step on the current window rather than once at the end."""
    win, E = wm.win, info["E"]
    if CTX >= win:
        raise ValueError(f"context {CTX} must fit the window {win}")

    @nnx.jit
    def dream(ac, ref, wstate, wstate2, F, A, C, R_t, R_e, ulog, key):
        kp, kr, ku, ks = jax.random.split(key, 4)
        use_r = jax.random.bernoulli(kp, p_reset, (B,))
        ri = jax.random.randint(kr, (B,), 0, R_t.shape[0])
        uflat = jax.random.categorical(ku, ulog, shape=(B,))
        t = jnp.where(use_r, R_t[ri], uflat // E)
        e = jnp.where(use_r, R_e[ri], uflat % E)
        ti = jnp.where(use_r[:, None], t[:, None], t[:, None] + jnp.arange(CTX))
        ei = jnp.broadcast_to(e[:, None], ti.shape)
        ctx_f = F[ti, ei]
        ca = jnp.where(use_r[:, None], 0, A[jnp.maximum(ti - 1, 0), ei])
        ctx = wm.encode(ctx_f)
        cond = jnp.where(use_r[:, None], jnp.full((B, ac.cond_dim), cond_reset), C[ti[:, -1], e])

        def burn(g, inp):
            obs, pa = inp
            return ac.step(obs, pa, g, cond)[2], None
        g, _ = jax.lax.scan(burn, jnp.zeros((B, PHID)), (ctx_f[:, :-1].swapaxes(0, 1), ca[:, :-1].swapaxes(0, 1)))
        g = jnp.where(use_r[:, None], 0.0, g)
        rg = jnp.zeros((B, PHID))
        if ref is not None:
            def burn_ref(g, inp):
                obs, pa = inp
                return ref.step(obs, pa, g, cond)[2], None
            rg, _ = jax.lax.scan(burn_ref, rg, (ctx_f[:, :-1].swapaxes(0, 1), ca[:, :-1].swapaxes(0, 1)))
            rg = jnp.where(use_r[:, None], 0.0, rg)

        seq = jnp.zeros((B, win) + ctx.shape[-3:]).at[:, :CTX].set(ctx)
        aseq = jnp.zeros((B, win), jnp.int32).at[:, :CTX].set(ca)

        def step(carry, k):
            seq, aseq, n, r, g, rg, prev_a, obs = carry
            ka, kf, kd = jax.random.split(k, 3)
            logits, vlog, g2 = ac.step(obs, prev_a, g, cond)
            rlog, rg2 = logits, rg
            if ref is not None:
                rlog, _, rg2 = ref.step(obs, prev_a, rg, cond)
            act = jax.random.categorical(ka, logits).astype(jnp.int32)
            slide = lambda sa: tuple(jnp.concatenate([v[:, 1:], v[:, -1:]], 1) for v in sa)
            seq, aseq = jax.lax.cond(n == win, slide, lambda sa: sa, (seq, aseq))
            r = jnp.where(n == win, jnp.maximum(r - 1, 0), r)
            n = jnp.minimum(n, win - 1)
            aseq = aseq.at[:, n].set(act)
            z = wm.flow(wstate, seq, wm.taus(B, n, r), aseq, n, kf, nfe)
            seq = seq.at[:, n].set(z)
            rew, pdone = wm.read(wstate, seq, aseq, kd)
            out = (obs, prev_a, g, act, logprob(logits, act), hl_expect(vlog), rlog,
                   jnp.take(rew, n, 1), jnp.take(pdone, n, 1))
            return (seq, aseq, n + 1, r, g2, rg2, act, wm.decode(z)), out

        init = (seq, aseq, jnp.asarray(CTX), jnp.asarray(CTX), g, rg, ca[:, -1], ctx_f[:, -1])
        (seq, aseq, _, _, g, _, prev_a, obs), (obs_seq, pas, gs, acts, lps, vals, rlogs, rews, dn) = jax.lax.scan(
            step, init, jax.random.split(ks, HOR))
        _, vlog_last, _ = ac.step(obs, prev_a, g, cond)
        adv, ret = gae(vals, rews, dn, hl_expect(vlog_last), gamma, lam)
        alive = jnp.concatenate([jnp.ones((1, B)), jnp.cumprod(1.0 - dn[:-1], 0)], 0)
        na = jnp.sum(alive) + 1e-6
        diag = jnp.stack([jnp.sum(rews * alive) / na, jnp.mean(dn), jnp.sum(rews * alive) / B, alive.mean()])
        batch = {"obs": obs_seq, "prev_a": pas, "g0": gs, "act": acts, "logp": lps, "adv": adv, "ret": ret, "alive": alive,
                 "cond": cond, "ref_logits": rlogs}
        return jax.tree.map(jax.lax.stop_gradient, batch), diag
    return dream
