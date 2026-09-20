"""A sampling predictor on the frozen JEPA latent: same encoder, same heads, only the one-step predictor changes.

The JEPA's deterministic predictor gives the mean m = predict(z_{t-h+1..t}, a_{t-h+1..t}) of the next latent.
A conditional flow-matching head learns the distribution of the residual z_{t+1} - m given (m, z_t, a_t) on the
same 500k transitions, off budget, with the encoder and the predictor frozen.  Rolling out with samples instead
of the mean turns the JEPA into a distributional world model with everything else held fixed: the control cell
that separates the planning mechanism (mean versus sample) from the representation.

    python -m studies.stoch.sampler --model /dev/shm/stoch/gate/tile_factored/model_probe.npz \
        --data /dev/shm/stoch/gate/train.npz --out /dev/shm/stoch/gate/tile_factored/sampler.npz
"""
import argparse
import json
import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import probe
import train as T


class Flow(nnx.Module):
    """Velocity field v(x_tau, tau, c) of a conditional flow on the standardised residual."""

    def __init__(self, dim, cond, hidden, depth, rngs):
        self.inp = nnx.Linear(dim + cond + 64, hidden, rngs=rngs)
        self.blocks = nnx.List([nnx.Linear(hidden, hidden, rngs=rngs) for _ in range(depth)])
        self.out = nnx.Linear(hidden, dim, rngs=rngs, kernel_init=nnx.initializers.zeros)

    def __call__(self, x, tau, c):
        freqs = jnp.exp(jnp.linspace(0.0, jnp.log(1000.0), 32))
        ang = tau[:, None] * freqs[None]
        h = nnx.gelu(self.inp(jnp.concatenate([x, c, jnp.sin(ang), jnp.cos(ang)], -1)))
        for block in self.blocks:
            h = h + nnx.gelu(block(h))
        return self.out(h)


class Sampler(nnx.Module):
    """The JEPA with its predictor's mean replaced by a sample: predict() is unchanged, sample() draws z_{t+1}."""

    def __init__(self, model, flow, mean, std, nfe, mean_of=0):
        self.model, self.flow = model, flow
        self.mean, self.std = nnx.Variable(mean), nnx.Variable(std)
        self.nfe, self.mean_of = nfe, mean_of
        self.history, self.latent_dim, self.n_actions = model.history, model.latent_dim, model.n_actions

    def encode(self, frames, train=False):
        return self.model.encode(frames)

    def act_embed(self, actions):
        return self.model.act_embed(actions)

    def predict(self, emb, act_emb, train=False):
        return self.model.predict(emb, act_emb)

    def cond(self, emb, acts):
        m = self.model.predict(emb, self.model.act_embed(acts))[:, -1]
        c = jnp.concatenate([m, emb[:, -1], jax.nn.one_hot(acts[:, -1], self.n_actions, dtype=m.dtype)], -1)
        return m, c

    def draw(self, m, c, key):
        x = jax.random.normal(key, m.shape, m.dtype)
        for i in range(self.nfe):
            tau = jnp.full((x.shape[0],), i / self.nfe, x.dtype)
            x = x + self.flow(x, tau, c) / self.nfe
        return m + self.mean[...] + self.std[...] * x

    def sample(self, emb, acts, key):
        """One sampled next latent per row from the history (N, h, D) and outgoing actions (N, h); with
        `mean_of` = K > 0 the head is used deterministically, as the mean of K of its samples (its own
        conditional mean, the control that separates the sample from the head's better accuracy)."""
        m, c = self.cond(emb, acts)
        if self.mean_of:
            return jnp.mean(jnp.stack([self.draw(m, c, k) for k in jax.random.split(key, self.mean_of)]), 0)
        return self.draw(m, c, key)


def load(bundle, path, nfe, mean_of=0):
    archive = np.load(path)
    meta = json.loads(str(archive["meta"]))
    flow = T._restore(Flow(meta["dim"], meta["cond"], meta["hidden"], meta["depth"], T.rngs()), archive, "model")
    return Sampler(bundle.model, flow, jnp.asarray(archive["res_mean"]), jnp.asarray(archive["res_std"]), nfe, mean_of), meta


def windows(model, frames, actions, dones, chunk=4096):
    """Per valid window: the predictor's mean, the last latent, the last action and the true next latent."""
    hist = model.history
    Z = probe.embed_all(model, frames)
    E = frames.shape[1]
    table = T.valid_starts(dones, hist + 1)
    acts = jnp.asarray(actions)

    @nnx.jit
    def means(model, start, env):
        idx = start[:, None] + jnp.arange(hist)[None]
        emb = Z[idx * E + env[:, None]]
        a = acts[idx, env[:, None]]
        return model.predict(emb, model.act_embed(a))[:, -1], emb[:, -1], a[:, -1], Z[(start + hist) * E + env]

    parts = [means(model, jnp.asarray(table[i:i + chunk, 0]), jnp.asarray(table[i:i + chunk, 1]))
             for i in range(0, table.shape[0], chunk)]
    m, z, a, target = (jnp.concatenate([p[k] for p in parts]) for k in range(4))
    return m, z, a, target, table


def fit(model, m, z, a, target, steps, batch, lr, hidden, depth, seed, held):
    n = m.shape[0] - held
    res = target[:n] - m[:n]
    mean, std = res.mean(0), res.std(0) + 1e-6
    flow = Flow(m.shape[1], 2 * m.shape[1] + model.n_actions, hidden, depth, T.rngs(seed + 3))
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, max(1, steps // 20), steps, end_value=0.0)
    opt = nnx.Optimizer(flow, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4)), wrt=nnx.Param)
    onehot = lambda x: jax.nn.one_hot(x, model.n_actions, dtype=m.dtype)

    @nnx.jit
    def run(flow, opt, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def loop(carry, key):
            flow, opt = carry
            k1, k2, k3 = jax.random.split(key, 3)
            ids = jax.random.randint(k1, (batch,), 0, n)
            x1 = (target[ids] - m[ids] - mean) / std
            c = jnp.concatenate([m[ids], z[ids], onehot(a[ids])], -1)
            x0 = jax.random.normal(k2, x1.shape, x1.dtype)
            tau = jax.random.uniform(k3, (batch,), x1.dtype)
            xt = (1.0 - tau[:, None]) * x0 + tau[:, None] * x1

            def loss_fn(flow):
                return jnp.mean(jnp.square(flow(xt, tau, c) - (x1 - x0)))

            loss, grads = nnx.value_and_grad(loss_fn)(flow)
            opt.update(flow, grads)
            return (flow, opt), loss

        return loop((flow, opt), keys)[1]

    keys = jax.random.split(jax.random.PRNGKey(seed + 5), steps)
    t0 = time.time()
    history = []
    for begin in range(0, steps, 500):
        history.append(np.asarray(jax.device_get(run(flow, opt, keys[begin:begin + 500]))))
        print(f"[flow] {begin + history[-1].shape[0]}/{steps} loss {history[-1][-100:].mean():.4f}  {time.time() - t0:.1f}s", flush=True)
    return flow, mean, std, float(np.concatenate(history)[-500:].mean())


def held_out_quality(sampler, m, z, a, target, held, samples, seed):
    """One-step latent error on held-out windows: the mean predictor, a single sample, and the mean of K samples."""
    hm, hz, ha, ht = m[-held:], z[-held:], a[-held:], target[-held:]
    key = jax.random.PRNGKey(seed + 9)

    @nnx.jit
    def draw(sampler, key):
        x = jax.random.normal(key, hm.shape, hm.dtype)
        c = jnp.concatenate([hm, hz, jax.nn.one_hot(ha, sampler.n_actions, dtype=hm.dtype)], -1)
        for i in range(sampler.nfe):
            tau = jnp.full((x.shape[0],), i / sampler.nfe, x.dtype)
            x = x + sampler.flow(x, tau, c) / sampler.nfe
        return hm + sampler.mean[...] + sampler.std[...] * x

    draws = jnp.stack([draw(sampler, k) for k in jax.random.split(key, samples)])
    mse = lambda p: float(jnp.mean(jnp.square(p - ht)))
    return {"mean_predictor_mse": mse(hm), "single_sample_mse": mse(draws[0]), "sample_mean_mse": mse(draws.mean(0)),
            "residual_var": float(jnp.mean(jnp.var(ht - hm, 0))), "sample_spread": float(jnp.mean(jnp.var(draws, 0))),
            "held_out_windows": int(held)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--nfe", type=int, default=8)
    p.add_argument("--held", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    bundle, _ = T.load_model(a.model)
    with np.load(a.data) as d:
        frames, actions, dones = d["frames"], d["actions"].astype(np.int32), d["dones"].astype(np.int64)
    m, z, act, target, table = windows(bundle.model, frames, actions, dones)
    print(f"[data] windows {m.shape[0]} latent {m.shape[1]}  {time.time() - t0:.1f}s", flush=True)
    flow, mean, std, final = fit(bundle.model, m, z, act, target, a.steps, a.batch, a.lr, a.hidden, a.depth, a.seed, a.held)
    sampler = Sampler(bundle.model, flow, mean, std, a.nfe)
    quality = held_out_quality(sampler, m, z, act, target, a.held, 8, a.seed)
    meta = {"dim": int(m.shape[1]), "cond": int(2 * m.shape[1] + bundle.model.n_actions), "hidden": a.hidden,
            "depth": a.depth, "steps": a.steps, "batch": a.batch, "lr": a.lr, "nfe": a.nfe, "final_loss": final,
            "quality": quality, "seconds": round(time.time() - t0, 1)}
    params, stats = T._weights(flow)
    arrays = {f"model.p.{i}": np.asarray(jax.device_get(leaf)) for i, leaf in enumerate(jax.tree.leaves(params))}
    arrays.update({f"model.b.{i}": np.asarray(jax.device_get(leaf)) for i, leaf in enumerate(jax.tree.leaves(stats))})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, meta=json.dumps(meta), res_mean=np.asarray(mean), res_std=np.asarray(std), **arrays)
    print("[sampler] " + json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
