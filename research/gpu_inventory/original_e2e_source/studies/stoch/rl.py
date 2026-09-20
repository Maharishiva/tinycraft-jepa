"""Control with the flow-matching world model in the stochastic-control worlds (survival is the task).

  dream      PPO inside the flow-matching world model: the policy never touches the real environment while
             training, real episodes only grade it; rewards and terminations are the model's own heads.
             --obs pixels   the policy sees the model's decoded frames (and, when graded, real frames)
             --obs latent   the policy sees the model's 4x4x32 latent (and, when graded, the encoded real frame)
  jepa       the same PPO inside the JEPA: deterministic latent rollouts of the frozen predictor, rewards and
             terminations from the off-budget heads on its latents, the policy sees the latent (control cell:
             same policy class as `dream --obs latent`, mean predictor instead of a sampled one).
  real       PPO in the real environment (the upper reference, pixels).
  baselines  random actions and the centre controller on the true state.

    python -m studies.stoch.rl --config gate --arm baselines --out runs/stoch/gate
    python -m studies.stoch.rl --config gate --arm dream --obs pixels --wm ~/personal/stoch/gate/flow/model.npz \
        --wm-code studies/stoch/flow_recipe --data ~/personal/stoch/gate/train.npz --out runs/stoch/gate/ppo_dream_pixels
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import train as T
from studies.redcube import env as R
from studies.redcube import flow as F
from studies.stoch import env as S
from studies.stoch import sampler as SM

A_N, CTX = S.N_ACTIONS, 8
PCH, PEMB, PHID, VBINS, HLSIG = (32, 64, 64), 256, 128, 64, 0.18
LATENT = (4, 4, 32)
CDT, PDT = jnp.bfloat16, jnp.float32
MP = dict(dtype=CDT, param_dtype=PDT)
FP = dict(dtype=PDT, param_dtype=PDT)


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def hl_support(n):
    return symexp(jnp.linspace(-15.0, 15.0, n)) / 100.0


def hl_expect(logits):
    return jnp.sum(jax.nn.softmax(logits) * hl_support(logits.shape[-1]), -1)


def hl_loss(logits, target, sigma=HLSIG):
    c = jnp.linspace(-15.0, 15.0, logits.shape[-1])
    y = symlog(target * 100.0)[..., None]
    w = jnp.exp(-0.5 * ((c - y) / sigma) ** 2)
    w = w / jnp.maximum(w.sum(-1, keepdims=True), 1e-6)
    return -jnp.sum(w * jax.nn.log_softmax(logits), -1)


def logprob(logits, a):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), a[..., None], -1).squeeze(-1)


def entropy(logits):
    return -jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), -1)


def gae(values, rewards, dones, last_v, gamma, lam):
    """Time-major (T, B); dones in [0, 1] act as one minus the continuation probability."""
    def step(carry, x):
        adv, nv = carry
        v, r, d = x
        delta = r + gamma * (1.0 - d) * nv - v
        adv = delta + gamma * lam * (1.0 - d) * adv
        return (adv, v), adv
    _, adv = jax.lax.scan(step, (jnp.zeros_like(last_v), last_v), (values[::-1], rewards[::-1], dones[::-1]))
    adv = adv[::-1]
    return adv, adv + values


class ImpalaBlock(nnx.Module):
    def __init__(self, cin, ch, rngs):
        self.conv = nnx.Conv(cin, ch, (3, 3), padding="SAME", rngs=rngs, **MP)
        self.r1a = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)
        self.r1b = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)
        self.r2a = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)
        self.r2b = nnx.Conv(ch, ch, (3, 3), padding="SAME", rngs=rngs, **MP)

    def __call__(self, x):
        x = nnx.max_pool(self.conv(x), (3, 3), strides=(2, 2), padding="SAME")
        x = x + self.r1b(nnx.relu(self.r1a(nnx.relu(x))))
        return x + self.r2b(nnx.relu(self.r2a(nnx.relu(x))))


class GRU(nnx.Module):
    def __init__(self, din, dh, rngs):
        self.wz = nnx.Linear(din + dh, dh, rngs=rngs, **FP)
        self.wr = nnx.Linear(din + dh, dh, rngs=rngs, **FP)
        self.wh = nnx.Linear(din + dh, dh, rngs=rngs, **FP)

    def __call__(self, x, g):
        xg = jnp.concatenate([x, g], -1)
        z = jax.nn.sigmoid(self.wz(xg))
        r = jax.nn.sigmoid(self.wr(xg))
        return (1.0 - z) * g + z * jnp.tanh(self.wh(jnp.concatenate([x, r * g], -1)))


class AC(nnx.Module):
    """Actor-critic with a GRU over the observation embedding and the previous action (HL-Gauss critic).
    Pixels: Impala-CNN on 63x63 frames.  Latent: MLP on the world model's 4x4x32 latent."""

    def __init__(self, rngs, obs="pixels", dim=int(np.prod(LATENT))):
        self.obs, self.dim = obs, dim
        if obs == "pixels":
            self.b1 = ImpalaBlock(3, PCH[0], rngs)
            self.b2 = ImpalaBlock(PCH[0], PCH[1], rngs)
            self.b3 = ImpalaBlock(PCH[1], PCH[2], rngs)
            self.fc = nnx.Linear(8 * 8 * PCH[2], PEMB, rngs=rngs, **MP)
        else:
            self.fc0 = nnx.Linear(dim, 512, rngs=rngs, **MP)
            self.fc = nnx.Linear(512, PEMB, rngs=rngs, **MP)
        self.gru = GRU(PEMB + A_N, PHID, rngs)
        self.l1 = nnx.Linear(PEMB + PHID, 256, rngs=rngs, **FP)
        self.actor = nnx.Linear(256, A_N, kernel_init=nnx.initializers.normal(0.01), rngs=rngs, **FP)
        self.critic = nnx.Linear(256, VBINS, rngs=rngs, **FP)

    def embed(self, obs):
        if self.obs == "pixels":
            x = self.b3(self.b2(self.b1((obs.astype(jnp.float32) / 255.0).astype(CDT))))
            return nnx.relu(self.fc(nnx.relu(x).reshape((x.shape[0], -1)))).astype(jnp.float32)
        h = nnx.relu(self.fc0(obs.reshape((obs.shape[0], -1)).astype(CDT)))
        return nnx.relu(self.fc(h)).astype(jnp.float32)

    def step(self, obs, prev_a, g):
        e = self.embed(obs)
        g2 = self.gru(jnp.concatenate([e, jax.nn.one_hot(prev_a, A_N)], -1), g)
        h = nnx.relu(self.l1(jnp.concatenate([e, g2], -1)))
        return self.actor(h), self.critic(h), g2


def n_params(module):
    return int(sum(x.size for x in jax.tree.leaves(nnx.state(module, nnx.Param))))


def save_ac(ac, path, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as f:
        pickle.dump({"state": jax.device_get(nnx.split(ac)[1]), "meta": meta, "obs": ac.obs, "dim": ac.dim}, f)


def load_ac(path):
    with Path(path).open("rb") as f:
        d = pickle.load(f)
    ac = AC(nnx.Rngs(0), d.get("obs", "pixels"), d.get("dim", int(np.prod(LATENT))))
    nnx.update(ac, jax.tree.map(jnp.asarray, d["state"]))
    return ac, d["meta"]


def summarize(ret, length, walls):
    ret, length, walls = np.asarray(ret), np.asarray(length), np.asarray(walls)
    alive = ret > -0.5
    return {"return": float(ret.mean()), "survival": float(alive.mean()),
            "survival_sem": float(alive.std() / np.sqrt(len(ret))), "length": float(length.mean()),
            "deaths_per_100": float(100.0 * (~alive).sum() / length.sum()), "walls_passed": float(walls.mean()),
            "n_ep": int(len(ret))}


def make_eval(spec, episodes, observe=None, start="uniform"):
    """Sampled policy, one episode per slot, finished slots frozen -> (return, length, walls passed) per episode.
    `observe(frames_u8) -> policy observation` (identity for pixels, the world model's encoder for latents);
    `start` is the reset distribution (env.make)."""
    reset, step = S.make(spec, start)
    vreset, vstep = jax.vmap(lambda k: reset(k)), jax.vmap(step)
    N = episodes
    observe = observe or (lambda f: f)

    def loop(key, act_fn, init_carry):
        k0, key = jax.random.split(key)
        obs, es = vreset(jax.random.split(k0, N))

        def cond(c):
            return (~jnp.all(c[4])) & (jnp.max(c[6]) < S.MAX_STEPS)

        def body(c):
            es, obs, carry, pa, done, ret, length, walls, key = c
            key, ka, ks = jax.random.split(key, 3)
            a, carry2 = act_fn(carry, obs, es, pa, ka)
            nobs, nes, rew, d1 = vstep(es, a, jax.random.split(ks, N))
            active = ~done
            passed = active & (es.wall_x != S.NO_WALL) & (nes.wall_x == S.NO_WALL)
            es2 = jax.tree.map(lambda o, n: jnp.where(done.reshape((N,) + (1,) * (n.ndim - 1)), o, n), es, nes)
            obs2 = jnp.where(done[:, None, None, None], obs, nobs)
            return (es2, obs2, carry2, a, done | d1, ret + jnp.where(active, rew, 0.0),
                    length + active.astype(jnp.int32), walls + passed.astype(jnp.int32), key)

        init = (es, obs, init_carry, jnp.zeros((N,), jnp.int32), jnp.zeros((N,), bool), jnp.zeros((N,)),
                jnp.zeros((N,), jnp.int32), jnp.zeros((N,), jnp.int32), key)
        out = jax.lax.while_loop(cond, body, init)
        return out[5], out[6], out[7]

    @nnx.jit
    def eval_ac(ac, key):
        def act(g, obs, es, pa, ka):
            logits, _, g2 = ac.step(observe(obs), pa, g)
            return jax.random.categorical(ka, logits).astype(jnp.int32), g2
        return loop(key, act, jnp.zeros((N, PHID)))

    @jax.jit
    def eval_random(key):
        return loop(key, lambda c, obs, es, pa, ka: (jax.random.randint(ka, (N,), 0, A_N), c), 0)

    @jax.jit
    def eval_centre(key):
        return loop(key, lambda c, obs, es, pa, ka: (R.centre_policy(es), c), 0)

    @jax.jit
    def eval_alternate(key):
        """State-blind reference: left, right, left, right (the step counter is shared by all slots)."""
        def act(c, obs, es, pa, ka):
            return jnp.full((N,), jnp.where(c % 2 == 0, 0, 2), jnp.int32), c + 1
        return loop(key, act, jnp.asarray(0, jnp.int32))

    return eval_ac, eval_random, eval_centre, eval_alternate


def start_tables(dones, T, E, ctx):
    """Dream starts: every reset frame (t, e), and the uniform starts whose 8-frame context has no reset."""
    dn = dones.astype(np.float32)
    rt, re = np.nonzero(np.concatenate([np.ones((1, E), np.float32), dn[:T - ctx]], 0) > 0.5)
    ok = np.zeros((T, E), np.float32)
    for dt in range(-1, ctx - 1):
        ok[1:T - ctx + 1] += dn[1 + dt:T - ctx + 1 + dt]
    t = np.arange(T)[:, None]
    ok = (ok == 0) & (t >= 1) & (t <= T - ctx)
    ut, ue = np.nonzero(ok)
    return (rt.astype(np.int32), re.astype(np.int32)), (ut.astype(np.int32), ue.astype(np.int32))


def sample_starts(rng, frames, actions, resets, uniform, B, p_reset):
    """Host-side gather of the context frames and their incoming actions for B dream starts."""
    use_r = rng.random(B) < p_reset
    ri = rng.integers(0, len(resets[0]), B)
    ui = rng.integers(0, len(uniform[0]), B)
    t = np.where(use_r, resets[0][ri], uniform[0][ui])
    e = np.where(use_r, resets[1][ri], uniform[1][ui])
    ti = np.where(use_r[:, None], t[:, None], t[:, None] + np.arange(CTX))
    ei = np.broadcast_to(e[:, None], ti.shape)
    ctx_f = frames[ti, ei]
    ca = np.where(use_r[:, None], 0, actions[np.maximum(ti - 1, 0), ei]).astype(np.int32)
    return jnp.asarray(ctx_f), jnp.asarray(ca), jnp.asarray(use_r)


def make_dream(wm, B, HOR, gamma, lam, obs_mode, mean_of=0):
    """dream(ac, wstate, ctx_f, ca, use_r, key) -> (time-major PPO batch, diagnostics).

    Half the starts are reset frames (repeated as context, action 0), the rest uniform dataset states with their
    real 8-frame context.  The policy acts on decoded pixels or on the latent.  Once the window is full it slides,
    the real context ages out, and the model runs on its own frames; the heads are read per step on the window.
    With `mean_of` = K > 0 the next latent is the mean of K samples (the generative model used as a mean
    predictor: the control cell for the planning mechanism on a model whose latent holds the wall)."""
    win = wm.win
    see = (lambda z: wm.decode(z)) if obs_mode == "pixels" else (lambda z: z)

    def next_latent(wstate, seq, taus, aseq, n, kf):
        if mean_of:
            def acc(s, k):
                return s + wm.flow(wstate, seq, taus, aseq, n, k), None
            s, _ = jax.lax.scan(acc, jnp.zeros_like(seq[:, 0]), jax.random.split(kf, mean_of))
            return s / mean_of
        return wm.flow(wstate, seq, taus, aseq, n, kf)

    @nnx.jit
    def dream(ac, wstate, ctx_f, ca, use_r, key):
        ks = key
        ctx = wm.encode(ctx_f)
        ctx_obs = ctx_f if obs_mode == "pixels" else ctx

        def burn(g, inp):
            return ac.step(inp[0], inp[1], g)[2], None
        g, _ = jax.lax.scan(burn, jnp.zeros((B, PHID)), (ctx_obs[:, :-1].swapaxes(0, 1), ca[:, :-1].swapaxes(0, 1)))
        g = jnp.where(use_r[:, None], 0.0, g)
        seq = jnp.zeros((B, win) + ctx.shape[-3:]).at[:, :CTX].set(ctx)
        aseq = jnp.zeros((B, win), jnp.int32).at[:, :CTX].set(ca)

        def step(carry, k):
            seq, aseq, n, r, g, prev_a, obs = carry
            ka, kf, kd = jax.random.split(k, 3)
            logits, vlog, g2 = ac.step(obs, prev_a, g)
            act = jax.random.categorical(ka, logits).astype(jnp.int32)
            slide = lambda sa: tuple(jnp.concatenate([v[:, 1:], v[:, -1:]], 1) for v in sa)
            seq, aseq = jax.lax.cond(n == win, slide, lambda sa: sa, (seq, aseq))
            r = jnp.where(n == win, jnp.maximum(r - 1, 0), r)
            n = jnp.minimum(n, win - 1)
            aseq = aseq.at[:, n].set(act)
            z = next_latent(wstate, seq, wm.taus(B, n, r), aseq, n, kf)
            seq = seq.at[:, n].set(z)
            rew, pdone = wm.read(wstate, seq, aseq, kd)
            out = (obs, prev_a, g, act, logprob(logits, act), hl_expect(vlog), jnp.take(rew, n, 1), jnp.take(pdone, n, 1))
            return (seq, aseq, n + 1, r, g2, act, see(z)), out

        init = (seq, aseq, jnp.asarray(CTX), jnp.asarray(CTX), g, ca[:, -1], ctx_obs[:, -1])
        (seq, aseq, _, _, g, prev_a, obs), (obs_seq, pas, gs, acts, lps, vals, rews, dn) = jax.lax.scan(
            step, init, jax.random.split(ks, HOR))
        _, vlog_last, _ = ac.step(obs, prev_a, g)
        adv, ret = gae(vals, rews, dn, hl_expect(vlog_last), gamma, lam)
        alive = jnp.concatenate([jnp.ones((1, B)), jnp.cumprod(1.0 - dn[:-1], 0)], 0)
        na = jnp.sum(alive) + 1e-6
        diag = jnp.stack([jnp.sum(rews * alive) / na, jnp.mean(dn), jnp.sum(rews * alive) / B, alive.mean()])
        batch = {"obs": obs_seq, "prev_a": pas, "g0": gs, "act": acts, "logp": lps, "adv": adv, "ret": ret, "alive": alive}
        return jax.tree.map(jax.lax.stop_gradient, batch), diag
    return dream


def make_jepa_dream(bundle, B, HOR, gamma, lam, sampler=None):
    """dream(ac, ctx_f, ca, use_r, key) for the JEPA: the frozen predictor rolls its own latent forward
    deterministically from the encoded context (or samples it, with the off-budget flow head on the residual);
    reward and termination come from the off-budget heads read on the latent and the action; the policy sees
    the latent."""
    model, heads = bundle.model, bundle.heads
    hist = model.history

    @nnx.jit
    def dream(ac, ctx_f, ca, use_r, key):
        emb = model.encode(ctx_f)
        act_hist = jnp.concatenate([ca[:, CTX - hist + 1:], jnp.zeros((B, 1), jnp.int32)], 1)

        def burn(g, inp):
            return ac.step(inp[0], inp[1], g)[2], None
        g, _ = jax.lax.scan(burn, jnp.zeros((B, PHID)), (emb[:, :-1].swapaxes(0, 1), ca[:, :-1].swapaxes(0, 1)))
        g = jnp.where(use_r[:, None], 0.0, g)

        def step(carry, k):
            emb_hist, act_hist, g, prev_a, obs = carry
            k, k_pred = jax.random.split(k)
            logits, vlog, g2 = ac.step(obs, prev_a, g)
            act = jax.random.categorical(k, logits).astype(jnp.int32)
            acts = act_hist.at[:, -1].set(act)
            reward, logit_done, _ = heads(emb_hist[:, -1], act)
            if sampler is None:
                nxt = model.predict(emb_hist, model.act_embed(acts))[:, -1]
            else:
                nxt = sampler.sample(emb_hist, acts, k_pred)
            out = (obs, prev_a, g, act, logprob(logits, act), hl_expect(vlog), reward.mean(-1), jax.nn.sigmoid(logit_done))
            carry = (jnp.concatenate([emb_hist[:, 1:], nxt[:, None]], 1),
                     jnp.concatenate([acts[:, 1:], jnp.zeros((B, 1), jnp.int32)], 1), g2, act, nxt)
            return carry, out

        init = (emb[:, -hist:], act_hist, g, ca[:, -1], emb[:, -1])
        (_, _, g, prev_a, obs), (obs_seq, pas, gs, acts, lps, vals, rews, dn) = jax.lax.scan(
            step, init, jax.random.split(key, HOR))
        _, vlog_last, _ = ac.step(obs, prev_a, g)
        adv, ret = gae(vals, rews, dn, hl_expect(vlog_last), gamma, lam)
        alive = jnp.concatenate([jnp.ones((1, B)), jnp.cumprod(1.0 - dn[:-1], 0)], 0)
        na = jnp.sum(alive) + 1e-6
        diag = jnp.stack([jnp.sum(rews * alive) / na, jnp.mean(dn), jnp.sum(rews * alive) / B, alive.mean()])
        batch = {"obs": obs_seq, "prev_a": pas, "g0": gs, "act": acts, "logp": lps, "adv": adv, "ret": ret, "alive": alive}
        return jax.tree.map(jax.lax.stop_gradient, batch), diag
    return dream


def ppo_real(spec, a):
    ledger = a.out / "ledger.jsonl"
    E, T, LC = a.envs, a.collect, a.chunk
    NCH = T // LC
    NG = a.groups
    CPG = NCH // NG
    if T % LC or NCH % NG:
        raise ValueError("collect must split into chunk x groups")
    ac = AC(nnx.Rngs(a.seed + 2))
    opt = nnx.Optimizer(ac, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(a.lr, weight_decay=0.01)), wrt=nnx.Param)
    vreset, vstep = S.batched(spec)
    eval_ac, _, _, _ = make_eval(spec, a.eval_episodes)
    print(f"[init] params={n_params(ac)} devices={jax.devices()}", flush=True)

    @nnx.jit
    def collect(ac, es, obs_u8, pol, rng):
        def step(carry, key):
            es, obs, (g, prev_a) = carry
            ka, ks = jax.random.split(key)
            logits, vlog, g2 = ac.step(obs, prev_a, g)
            act = jax.random.categorical(ka, logits).astype(jnp.int32)
            _, nes, rew, done, nobs = vstep(es, act, ks)
            out = (obs, act, prev_a, rew, done.astype(jnp.float32), hl_expect(vlog), logprob(logits, act))
            return (nes, nobs, (jnp.where(done[:, None], 0.0, g2), jnp.where(done, 0, act))), out
        (es, obs_u8, pol), outs = jax.lax.scan(step, (es, obs_u8, pol), jax.random.split(rng, T))
        return es, obs_u8, pol, outs

    @nnx.jit
    def prep(ac, frames_full, acts, pas_full, rews, dones, g0):
        acts_full = jnp.concatenate([acts, jnp.zeros((1, E), jnp.int32)], 0)
        dones_full = jnp.concatenate([dones, jnp.zeros((1, E))], 0)

        def hstep(g, inp):
            obs, a_t, pa_t, d_t = inp
            logits, vlog, g2 = ac.step(obs, pa_t, g)
            return jnp.where(d_t[:, None] > 0.5, 0.0, g2), (logprob(logits, a_t), hl_expect(vlog), g)
        _, (lp, val, gs) = jax.lax.scan(hstep, g0, (frames_full, acts_full, pas_full, dones_full))
        adv, ret = gae(val[:T], rews, dones, val[T], a.gamma, a.lam)
        return lp[:T], adv, ret, gs[:T]

    @nnx.jit
    def ppo(ac, opt, frames, acts, pas, dones, gs, oldlp, adv, ret, gi):
        def rs(x):
            x = x.reshape((NG, CPG, LC) + x.shape[1:])[gi]
            return x.swapaxes(0, 1).reshape((LC, CPG * E) + x.shape[3:])
        advn = (adv - adv.mean()) * jax.lax.rsqrt(adv.var() + 1e-5)
        fr, ac_t, pa, d = rs(frames), rs(acts), rs(pas), rs(dones)
        olp, an, rt = rs(oldlp), rs(advn), rs(ret)
        g0 = jax.lax.stop_gradient(gs[::LC].reshape((NG, CPG * E, PHID))[gi])

        def loss_fn(ac):
            def sstep(g, inp):
                obs, pa_t, d_t = inp
                logits, vlog, g2 = ac.step(obs, pa_t, g)
                return jnp.where(d_t[:, None] > 0.5, 0.0, g2), (logits, vlog)
            _, (logits, vlog) = jax.lax.scan(sstep, g0, (fr, pa, d))
            ratio = jnp.exp(logprob(logits, ac_t) - olp)
            pg = -jnp.minimum(ratio * an, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * an).mean()
            vf = hl_loss(vlog, rt).mean()
            ent = entropy(logits).mean()
            return pg + a.vf * vf - a.ent * ent, (pg, vf, ent)
        (loss, aux), gr = nnx.value_and_grad(loss_fn, has_aux=True)(ac)
        opt.update(ac, gr)
        return jnp.stack([loss, aux[0], aux[1], aux[2]])

    rng = jax.random.PRNGKey(a.seed + 7)
    rng, r0 = jax.random.split(rng)
    obs_u8, es = vreset(jax.random.split(r0, E), jnp.zeros(E, jnp.int32))
    pol = (jnp.zeros((E, PHID)), jnp.zeros((E,), jnp.int32))
    env_steps, it, start, best = 0, 0, time.time(), None

    def evaluate():
        nonlocal rng, best
        rng, ke = jax.random.split(rng)
        row = {"phase": "ppo_real", "iter": it, "env_steps": env_steps, **summarize(*eval_ac(ac, ke)),
               "sec": round(time.time() - start, 1)}
        with ledger.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("[eval] " + json.dumps(row), flush=True)
        save_ac(ac, a.out / "model.pkl", row)
        if best is None or row["survival"] > best["survival"]:
            best = row
            save_ac(ac, a.out / "model_best.pkl", row)

    while env_steps + T * E <= a.budget:
        it += 1
        t0 = time.time()
        rng, kc = jax.random.split(rng)
        g_start = pol[0]
        es, obs_u8, pol, (frames, acts, pas, rews, dones, vals, lps) = collect(ac, es, obs_u8, pol, kc)
        env_steps += T * E
        oldlp, adv, ret_t, gs = prep(ac, jnp.concatenate([frames, obs_u8[None]], 0), acts,
                                     jnp.concatenate([pas, pol[1][None]], 0), rews, dones, g_start)
        last = np.zeros(4)
        for _ in range(a.epochs):
            for gi in range(NG):
                last = np.asarray(ppo(ac, opt, frames, acts, pas, dones, gs, oldlp, adv, ret_t, jnp.asarray(gi)))
        print("[iter] " + json.dumps({"iter": it, "env_steps": env_steps, "sec": round(time.time() - t0, 1),
                                      "deaths_per_step": round(float(jnp.mean(rews < -0.5)), 4),
                                      "pg": round(float(last[1]), 4), "vf": round(float(last[2]), 4),
                                      "ent": round(float(last[3]), 3)}), flush=True)
        out_of_time = time.time() - start > a.max_minutes * 60
        if it % a.eval_every == 0 or env_steps + T * E > a.budget or out_of_time:
            evaluate()
        if out_of_time:
            print(f"[stop] max_minutes {a.max_minutes} reached", flush=True)
            break
    (a.out / "final.json").write_text(json.dumps({"best": best, "last": best and None, "env_steps": env_steps, "iters": it}, indent=1))
    print(f"[done] best {json.dumps(best)}", flush=True)


def ppo_dream(spec, a):
    ledger = a.out / "ledger.jsonl"
    d = np.load(a.data)
    Fr, A = d["frames"], d["actions"].astype(np.int32)
    T_, E = A.shape
    resets, uniform = start_tables(np.asarray(d["dones"]), T_, E, CTX)
    host = np.random.default_rng(a.seed + 11)
    if a.arm == "jepa":
        bundle, _ = T.load_model(a.wm)
        if bundle.heads is None:
            raise RuntimeError("the JEPA checkpoint needs reward / done heads (probe.py --heads)")
        a.obs, dim = "latent", int(bundle.model.latent_dim)
        sampler = SM.load(bundle, a.sampler, a.nfe, a.sampler_mean)[0] if a.sampler else None
        dream = make_jepa_dream(bundle, a.batch, a.hor, a.gamma, a.lam, sampler)
        observe = lambda f: bundle.model.encode(f[:, None])[:, 0]
        note = f"jepa latent {dim} history {bundle.model.history} sampler {a.sampler} mean_of {a.sampler_mean}"
    else:
        train_mod, _, model = F.load(a.wm_code, a.wm, A_N)
        wm = F.WM(train_mod, model, a.nfe)
        dim = int(np.prod(LATENT))
        dream = make_dream(wm, a.batch, a.hor, a.gamma, a.lam, a.obs, a.sampler_mean)
        observe = None if a.obs == "pixels" else (lambda f: wm.encode(f[:, None])[:, 0])
        note = f"flow window {wm.win} nfe {a.nfe} mean_of {a.sampler_mean}"
    ac = AC(nnx.Rngs(a.seed + 2), a.obs, dim)
    opt = nnx.Optimizer(ac, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(a.lr, weight_decay=0.01)), wrt=nnx.Param)
    eval_ac, _, _, _ = make_eval(spec, a.eval_episodes, observe)
    B, HOR = a.batch, a.hor
    NCH, NB = HOR // a.chunk, B // a.mb
    if HOR % a.chunk or B % a.mb:
        raise ValueError("hor must split into chunks and batch into minibatches")
    print(f"[init] params={n_params(ac)} obs={a.obs} reset starts={len(resets[0])} {note}", flush=True)

    @nnx.jit
    def ppo_step(ac, opt, batch, ci, bi):
        m = batch["alive"]
        mu = jnp.sum(batch["adv"] * m) / (jnp.sum(m) + 1e-6)
        var = jnp.sum((batch["adv"] - mu) ** 2 * m) / (jnp.sum(m) + 1e-6)
        advn = (batch["adv"] - mu) * jax.lax.rsqrt(var + 1e-5)

        def sl(x):
            x = x.reshape((NCH, a.chunk) + x.shape[1:])[ci]
            return x.reshape((a.chunk, NB, a.mb) + x.shape[2:])[:, bi]
        o, pa, act = sl(batch["obs"]), sl(batch["prev_a"]), sl(batch["act"])
        olp, an, rt, mm = sl(batch["logp"]), sl(advn), sl(batch["ret"]), sl(m)
        g0 = jax.lax.stop_gradient(batch["g0"].reshape((NCH, a.chunk) + batch["g0"].shape[1:])[ci, 0]
                                   .reshape((NB, a.mb, PHID))[bi])

        def loss_fn(ac):
            def sstep(g, inp):
                logits, vlog, g2 = ac.step(inp[0], inp[1], g)
                return g2, (logits, vlog)
            _, (logits, vlog) = jax.lax.scan(sstep, g0, (o, pa))
            nm = jnp.sum(mm) + 1e-6
            ratio = jnp.exp(logprob(logits, act) - olp)
            pg = -jnp.sum(jnp.minimum(ratio * an, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * an) * mm) / nm
            vf = jnp.sum(hl_loss(vlog, rt) * mm) / nm
            ent = jnp.sum(entropy(logits) * mm) / nm
            return pg - a.ent * ent + a.vf * vf, (pg, vf, ent)
        (loss, aux), gr = nnx.value_and_grad(loss_fn, has_aux=True)(ac)
        opt.update(ac, gr)
        return jnp.stack([loss, aux[0], aux[1], aux[2]])

    rng = jax.random.PRNGKey(a.seed + 7)
    start, best = time.time(), None

    def evaluate(upd, extra):
        nonlocal rng, best
        rng, ke = jax.random.split(rng)
        row = {"phase": "ppo_" + a.arm, "obs": a.obs, "upd": upd, "imag_steps": upd * B * HOR, **summarize(*eval_ac(ac, ke)),
               "sec": round(time.time() - start, 1), **extra}
        with ledger.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("[eval] " + json.dumps(row), flush=True)
        save_ac(ac, a.out / "model.pkl", row)
        if best is None or row["survival"] > best["survival"]:
            best = row
            save_ac(ac, a.out / "model_best.pkl", row)

    for u in range(1, a.updates + 1):
        t0 = time.time()
        rng, kd = jax.random.split(rng)
        ctx_f, ca, use_r = sample_starts(host, Fr, A, resets, uniform, B, a.p_reset)
        batch, diag = dream(ac, ctx_f, ca, use_r, kd) if a.arm == "jepa" else dream(ac, wm.state, ctx_f, ca, use_r, kd)
        diag = np.asarray(diag)
        t_dream = time.time() - t0
        last = np.zeros(4)
        for _ in range(a.epochs):
            for ci in range(NCH):
                for bi in range(NB):
                    last = np.asarray(ppo_step(ac, opt, batch, jnp.asarray(ci), jnp.asarray(bi)))
        print("[upd] " + json.dumps({"upd": u, "dream_s": round(t_dream, 1), "sec": round(time.time() - t0, 1),
                                     "imag_r": round(float(diag[0]), 4), "imag_done": round(float(diag[1]), 4),
                                     "dream_ret": round(float(diag[2]), 3), "alive": round(float(diag[3]), 3),
                                     "pg": round(float(last[1]), 4), "vf": round(float(last[2]), 4),
                                     "ent": round(float(last[3]), 3)}), flush=True)
        if u % a.eval_every == 0 or u == a.updates:
            evaluate(u, {"imag_r": round(float(diag[0]), 4), "imag_done": round(float(diag[1]), 4),
                         "dream_ret": round(float(diag[2]), 3), "alive": round(float(diag[3]), 3)})
        if time.time() - start > a.max_minutes * 60:
            print(f"[stop] max_minutes {a.max_minutes} reached", flush=True)
            if u % a.eval_every:
                evaluate(u, {})
            break
    (a.out / "final.json").write_text(json.dumps({"best": best, "updates": u}, indent=1))
    print(f"[done] best {json.dumps(best)}", flush=True)


def baselines(spec, a):
    _, eval_random, eval_centre, eval_alternate = make_eval(spec, a.eval_episodes, start=a.eval_start)
    key = jax.random.PRNGKey(a.seed + 123)
    rows = {"random": summarize(*eval_random(key)), "centre": summarize(*eval_centre(key)),
            "alternate": summarize(*eval_alternate(key)), "start": a.eval_start}
    name = "baselines.json" if a.eval_start == "uniform" else f"baselines_{a.eval_start}.json"
    (a.out / name).write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows), flush=True)


def evaluate_policy(spec, a):
    """Grade a saved policy (model.pkl of a finished run) under another reset distribution, e.g. edge starts."""
    ac, meta = load_ac(a.policy)
    observe = None
    if ac.obs == "latent":
        if a.wm_code:
            train_mod, _, model = F.load(a.wm_code, a.wm, A_N)
            wm = F.WM(train_mod, model, a.nfe)
            observe = lambda f: wm.encode(f[:, None])[:, 0]
        else:
            bundle, _ = T.load_model(a.wm)
            observe = lambda f: bundle.model.encode(f[:, None])[:, 0]
    eval_ac, _, _, _ = make_eval(spec, a.eval_episodes, observe, a.eval_start)
    row = {"policy": str(a.policy), "start": a.eval_start, "trained": meta,
           **summarize(*eval_ac(ac, jax.random.PRNGKey(a.seed + 321)))}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"eval_{a.eval_start}.json").write_text(json.dumps(row, indent=1))
    print("[evaluate] " + json.dumps({k: v for k, v in row.items() if k != "trained"}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, choices=sorted(S.CONFIGS))
    p.add_argument("--arm", required=True, choices=["real", "dream", "jepa", "baselines", "evaluate"])
    p.add_argument("--obs", default="pixels", choices=["pixels", "latent"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=256)
    p.add_argument("--eval-start", default="uniform", choices=["uniform", "edge"], help="reset distribution when grading")
    p.add_argument("--policy", type=Path, help="evaluate: a saved model.pkl")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--max-minutes", type=float, default=12.0)
    # real
    p.add_argument("--budget", type=int, default=1_500_000)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--collect", type=int, default=64)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--groups", type=int, default=4)
    # dream
    p.add_argument("--wm", type=Path)
    p.add_argument("--wm-code", type=Path)
    p.add_argument("--sampler", type=Path, help="jepa arm: flow head on the residual (studies.stoch.sampler), sample instead of the mean")
    p.add_argument("--sampler-mean", type=int, default=0, help="K > 0: the next latent is the MEAN of K samples "
                   "(jepa: of the residual head; dream: of the flow model), the mean-versus-sample control")
    p.add_argument("--data", type=Path)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hor", type=int, default=16)
    p.add_argument("--mb", type=int, default=128)
    p.add_argument("--nfe", type=int, default=4)
    p.add_argument("--p-reset", type=float, default=0.5)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(vars(a), default=str, indent=1))
    spec = S.CONFIGS[a.config]
    if a.arm == "real":
        ppo_real(spec, a)
    elif a.arm in ("dream", "jepa"):
        a.chunk = min(a.chunk, a.hor)
        ppo_dream(spec, a)
    elif a.arm == "evaluate":
        evaluate_policy(spec, a)
    else:
        baselines(spec, a)


if __name__ == "__main__":
    main()
