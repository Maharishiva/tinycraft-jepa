"""Model-free recurrent PPO in real Craftax-Classic-Pixels-v1 with the shared policy: the pop-flow-wm recipe
(64 envs, 100-step collects, 4 chunked-BPTT epochs, constant learning rate) that reached 7.9 return at 1M steps.
One run covers several budgets: snapshots are written when the step count crosses each --save_at value.

    python train_ppo_real.py --out runs/ppo_5m --budget 5000000 --save_at 500000 1000000 2000000 --seed 0
    python train_ppo_real.py --out runs/ppo_20m --budget 20000000 --ac_from runs/ppo_5m/model.pkl \
        --start_steps 4998400 --save_at 10000000 15000000        # continue the 5M run (fresh optimizer state)

--ac_from warm-starts the policy weights; --start_steps offsets the step count so budgets, snapshots and the
ledger stay cumulative. The optimizer state is not restored (constant learning rate, so only Adam's moments differ).

--wm swaps the real environment for the world model (wm_gym) and changes nothing else: same PPO, same geometry,
same hyperparameters. The evaluator stays in real Craftax, so the ledger is directly comparable to a real run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import craftax_env
import wm_gym
from agent import AC, PHID, entropy, gae, hl_expect, hl_loss, load_ac, logprob, n_params, save_ac, to_u8


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=1_000_000)
    p.add_argument("--save_at", type=int, nargs="*", default=[])
    p.add_argument("--ac_from", type=Path, default=None)
    p.add_argument("--start_steps", type=int, default=0)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--collect", type=int, default=100)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.8)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=8)
    p.add_argument("--eval_episodes", type=int, default=128)
    p.add_argument("--wm", type=Path, default=None)
    p.add_argument("--wm_data", type=Path, default=Path("data/train.npz"))
    p.add_argument("--wm_nfe", type=int, default=4)
    p.add_argument("--wm_max_steps", type=int, default=1024)
    p.add_argument("--wm_once", action="store_true", help="pay each place/make action at most once per episode")
    p.add_argument("--no_gru", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(vars(a), default=str, indent=2))
    ledger = a.out / "ledger.jsonl"

    E, T, LC = a.envs, a.collect, a.chunk
    NCH = T // LC
    NG = a.groups
    CPG = NCH // NG
    if T % LC or NCH % NG:
        raise ValueError("collect must split into chunk x groups")
    zc = jnp.zeros((E, 0))

    if a.ac_from is not None:
        ac, src = load_ac(a.ac_from)
        print(f"[init] policy from {a.ac_from} "
              + json.dumps({k: v for k, v in src.items() if k != "rates"}), flush=True)
    else:
        ac = AC(nnx.Rngs(a.seed + 2), recurrent=not a.no_gru)
    print(f"[init] params={n_params(ac)} devices={jax.devices()}", flush=True)
    opt = nnx.Optimizer(ac, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(a.lr, weight_decay=0.01)),
                        wrt=nnx.Param)
    env, ep, vreset, vstep = craftax_env.make_env()
    eval_agent = craftax_env.make_eval(vreset, vstep, int(ep.max_timesteps) + 1, a.eval_episodes)
    if a.wm is not None:
        wm, vreset, vstep = wm_gym.make_env(a.wm, a.wm_data, a.wm_nfe, a.wm_max_steps, a.wm_once)
        print(f"[init] environment = world model {a.wm} ({a.wm_nfe} Euler steps, window {wm.win}, "
              f"cap {a.wm_max_steps}, once-per-episode crafting pay {a.wm_once}); evaluation stays in real Craftax",
              flush=True)

    @nnx.jit
    def collect(ac, es, obs_u8, pol, rng):
        def step(carry, key):
            es, obs, (g, prev_a), rng = carry
            ka, ks, kr = jax.random.split(key, 3)
            logits, vlog, g2 = ac.step(obs, prev_a, g, zc)
            act = jax.random.categorical(ka, logits).astype(jnp.int32)
            nobs, nes, rew, done, _ = vstep(jax.random.split(ks, E), es, act)
            robs, res = vreset(jax.random.split(kr, E))
            nes = jax.tree.map(lambda r, x: jnp.where(done.reshape((E,) + (1,) * (x.ndim - 1)), r, x), res, nes)
            nobs = jnp.where(done[:, None, None, None], robs, nobs)
            out = (obs, act, prev_a, rew, done.astype(jnp.float32), hl_expect(vlog), logprob(logits, act))
            g2 = jnp.where(done[:, None], 0.0, g2)
            return (nes, to_u8(nobs), (g2, jnp.where(done, 0, act)), rng), out

        (es, obs_u8, pol, rng), outs = jax.lax.scan(step, (es, obs_u8, pol, rng), jax.random.split(rng, T))
        return es, obs_u8, pol, outs

    @nnx.jit
    def prep(ac, frames_full, acts, pas_full, rews, dones, g0):
        acts_full = jnp.concatenate([acts, jnp.zeros((1, E), jnp.int32)], 0)
        dones_full = jnp.concatenate([dones, jnp.zeros((1, E))], 0)

        def hstep(g, inp):
            obs, a_t, pa_t, d_t = inp
            logits, vlog, g2 = ac.step(obs, pa_t, g, zc)
            out = (logprob(logits, a_t), hl_expect(vlog), g)
            return jnp.where(d_t[:, None] > 0.5, 0.0, g2), out

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
        zcg = jnp.zeros((CPG * E, 0))

        def loss_fn(ac):
            def sstep(g, inp):
                obs, pa_t, d_t = inp
                logits, vlog, g2 = ac.step(obs, pa_t, g, zcg)
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
    robs, es = vreset(jax.random.split(r0, E))
    obs_u8 = to_u8(robs)
    pol = (jnp.zeros((E, PHID)), jnp.zeros((E,), jnp.int32))
    pending = sorted(s for s in a.save_at if s > a.start_steps)
    env_steps, it, start = a.start_steps, 0, time.time()

    def evaluate(final):
        nonlocal rng
        rng, ke = jax.random.split(rng)
        ret, length, ach = eval_agent(ac, ke)
        row = {"phase": "ppo", "iter": it, "env_steps": env_steps, **craftax_env.summarize(ret, length, ach),
               "sec": round(time.time() - start, 1)}
        with ledger.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("[eval] " + json.dumps({k: v for k, v in row.items() if k != "rates"}), flush=True)
        return row

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
                                      "real_r": round(float(jnp.mean(rews)), 4), "pg": round(float(last[1]), 4),
                                      "vf": round(float(last[2]), 4), "ent": round(float(last[3]), 3)}), flush=True)
        final = env_steps + T * E > a.budget
        if it % a.eval_every == 0 or final:
            row = evaluate(final)
            save_ac(ac, a.out / "model.pkl", {"cond": "none", "arm": "ppo_real", **row})
        while pending and env_steps >= pending[0]:
            s = pending.pop(0)
            save_ac(ac, a.out / f"model_{s // 1000}k.pkl", {"cond": "none", "arm": "ppo_real", "iter": it, "env_steps": env_steps})
            print(f"[save] model_{s // 1000}k.pkl at {env_steps} steps", flush=True)
    print(f"[done] iters={it} env_steps={env_steps} elapsed={time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
