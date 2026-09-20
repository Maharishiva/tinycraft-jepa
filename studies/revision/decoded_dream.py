"""Native JEPA/Flow imagination with the SAME external P.AC pixel policy.

Both families burn in on their own decoded real posterior context and then act
on their own decoded latent predictions. Starts must be eight actual consecutive
real frames within an episode (the driver's p_reset is zero). Native model heads,
returns, GAE, continuation weights, and uint8 quantization are preserved. This
module never instantiates, changes, or optimizes the actor or a world model.
"""
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from studies.stoch import rl as P


def _validate_context(ac, ctx_f, ca, batch):
    if ctx_f.shape != (batch, P.CTX, 63, 63, 3) or ctx_f.dtype != jnp.uint8:
        raise ValueError("dream starts require [B,8,63,63,3] real uint8 frames")
    if ca.shape != (batch, P.CTX):
        raise ValueError("context actions must be [B,8] incoming actions")
    if ac.obs != "pixels":
        raise ValueError("shared decoded protocol requires original P.AC(obs='pixels')")


def _burn(ac, decoded_context, incoming_actions, batch):
    def step(g, inp):
        return ac.step(inp[0], inp[1], g)[2], None
    return jax.lax.scan(step, jnp.zeros((batch, P.PHID)),
                        (decoded_context[:, :-1].swapaxes(0, 1),
                         incoming_actions[:, :-1].swapaxes(0, 1)))[0]


def _finish(ac, outputs, g, prev_a, obs, batch_size, gamma, lam):
    """The original P.make_dream batch, GAE, alive weights and four diagnostics."""
    obs_seq, pas, gs, acts, lps, values, rewards, dones = outputs
    _, vlog_last, _ = ac.step(obs, prev_a, g)
    adv, ret = P.gae(values, rewards, dones, P.hl_expect(vlog_last), gamma, lam)
    alive = jnp.concatenate([jnp.ones((1, batch_size)), jnp.cumprod(1.0 - dones[:-1], 0)], 0)
    na = jnp.sum(alive) + 1e-6
    diag = jnp.stack([jnp.sum(rewards * alive) / na, jnp.mean(dones),
                      jnp.sum(rewards * alive) / batch_size, alive.mean()])
    batch = dict(obs=obs_seq, prev_a=pas, g0=gs, act=acts, logp=lps, adv=adv, ret=ret, alive=alive)
    return jax.tree.map(jax.lax.stop_gradient, batch), diag


def make_jepa_dream(adapter, B, HOR, gamma, lam):
    hist = adapter.history
    if hist != P.CTX:
        raise ValueError("registered JEPA uses exactly eight history frames")

    @nnx.jit
    def dream(ac, ctx_f, ca, key):
        _validate_context(ac, ctx_f, ca, B)
        emb = adapter.encode(ctx_f)
        ctx_obs = adapter.decode(emb)
        # ca[j] drives frame[j-1] -> frame[j]; predict wants outgoing actions.
        act_hist = jnp.concatenate([ca[:, P.CTX - hist + 1:], jnp.zeros((B, 1), jnp.int32)], 1)
        g = _burn(ac, ctx_obs, ca, B)

        def step(carry, k):
            emb_hist, act_hist, g, prev_a, obs = carry
            # SAME actor key as Flow. JEPA is deterministic; remaining keys unused.
            ka, _, _ = jax.random.split(k, 3)
            logits, vlog, g2 = ac.step(obs, prev_a, g)
            act = jax.random.categorical(ka, logits).astype(jnp.int32)
            acts = act_hist.at[:, -1].set(act)
            reward, done = adapter.read_current(emb_hist[:, -1], act)
            nxt = adapter.predict(emb_hist, acts)
            out = (obs, prev_a, g, act, P.logprob(logits, act), P.hl_expect(vlog), reward, done)
            carry = (jnp.concatenate([emb_hist[:, 1:], nxt[:, None]], 1),
                     jnp.concatenate([acts[:, 1:], jnp.zeros((B, 1), jnp.int32)], 1),
                     g2, act, adapter.decode(nxt))
            return carry, out

        init = (emb[:, -hist:], act_hist, g, ca[:, -1], ctx_obs[:, -1])
        (_, _, g, prev_a, obs), outputs = jax.lax.scan(step, init, jax.random.split(key, HOR))
        return _finish(ac, outputs, g, prev_a, obs, B, gamma, lam)

    return dream


def make_flow_dream(adapter, B, HOR, gamma, lam):
    wm, win = adapter.wm, adapter.wm.win

    @nnx.jit
    def compiled_dream(ac, wstate, ctx_f, ca, key):
        _validate_context(ac, ctx_f, ca, B)
        ctx = adapter.encode(ctx_f)
        ctx_obs = adapter.decode(ctx)
        g = _burn(ac, ctx_obs, ca, B)
        seq = jnp.zeros((B, win) + ctx.shape[-3:]).at[:, :P.CTX].set(ctx)
        aseq = jnp.zeros((B, win), jnp.int32).at[:, :P.CTX].set(ca)

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
            z = wm.flow(wstate, seq, wm.taus(B, n, r), aseq, n, kf)
            seq = seq.at[:, n].set(z)
            reward, done = wm.read(wstate, seq, aseq, kd)
            out = (obs, prev_a, g, act, P.logprob(logits, act), P.hl_expect(vlog),
                   jnp.take(reward, n, 1), jnp.take(done, n, 1))
            return (seq, aseq, n + 1, r, g2, act, adapter.decode(z)), out

        init = (seq, aseq, jnp.asarray(P.CTX), jnp.asarray(P.CTX), g, ca[:, -1], ctx_obs[:, -1])
        (_, _, _, _, g, prev_a, obs), outputs = jax.lax.scan(step, init, jax.random.split(key, HOR))
        return _finish(ac, outputs, g, prev_a, obs, B, gamma, lam)

    def dream(ac, ctx_f, ca, key):
        # Keep the native explicit dynamics-state JIT argument: large frozen
        # weights need not become compiler constants. No state enters the actor.
        return compiled_dream(ac, wm.state, ctx_f, ca, key)

    return dream


def make_dream(adapter, B, HOR=48, gamma=0.99, lam=0.9):
    """Return dream(ac, real_context, incoming_actions, key) -> (PPO batch, diag).

    The driver owns the unchanged P.AC(obs='pixels'), optimizer and PPO updater;
    H48/B512/U50 with chunk16 is the scientific protocol. Smaller batches and
    horizons may be passed for explicitly labeled execution-only smoke tests.
    """
    if B < 1 or HOR < 1:
        raise ValueError("batch size and horizon must be positive")
    if adapter.family == "jepa":
        return make_jepa_dream(adapter, B, HOR, gamma, lam)
    if adapter.family == "flow":
        return make_flow_dream(adapter, B, HOR, gamma, lam)
    raise ValueError(f"unknown model family: {adapter.family}")
