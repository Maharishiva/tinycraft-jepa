"""Closed-loop control with the JEPA in the stochastic-control worlds (survival is the task).

  goal     LeWM's own planner: squared latent distance to a goal image (the cube at the centre, no wall),
           replanned every step; the search is exhaustive over the 4^h action sequences of the horizon, which is
           at least as strong as the paper's CEM (300 samples x 10 iterations).
  plan     reward MPC through the off-budget reward / done / value heads on the same frozen latents and the same
           latent rollouts: exhaustive horizon-h search of the imagined discounted return, replanned every step.
  greedy   one imagined step plus the value head.
  random / centre   the matched references (same start states, same environment noise).

Every policy sees the same start states and the same per-step environment keys (wind, gaps), so the runs are
paired.  Episodes end on death (border or wall) or at the 128-step cap (survival = reaching the cap); an episode
that dies during the 7-step random warm-up is redrawn so every policy starts from a live 8-frame history, and
lengths count from the reset (the warm-up included), as in rl.py.

    python -m studies.stoch.control --config gate --model /dev/shm/stoch/gate/tile_factored/model_probe.npz \
        --out runs/stoch/gate/control_tile_factored.json
"""
import argparse
import functools
import json
import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import probe
import train as T
import viz
from studies.redcube import env as R
from studies.stoch import env as S
from studies.stoch import sampler as SM

import os

CTX = 8
ROWS = int(os.environ.get("ROWS", 4096))


def goal_frames(spec, states, key, where):
    """Goal image per episode on its own texture: the cube at the centre (or at the left safe zone) and no wall."""
    if where == "centre":
        target = jnp.full_like(states.pos, int(round(S.CENTRE)))
    else:
        target = jnp.stack([jnp.full_like(states.pos[:, 0], 8), jnp.full_like(states.pos[:, 1], int(round(S.CENTRE)))], 1)
    goal = S.State(target, states.wind, states.texture, states.step, states.episode,
                   jnp.full_like(states.wall_x, S.NO_WALL), states.gap)
    return jax.vmap(lambda s, k: S.render(spec, s, k))(goal, jax.random.split(key, states.pos.shape[0]))


def _children(model, emb, acts, key=None):
    n = emb.shape[0]
    emb4, acts4 = jnp.repeat(emb, S.N_ACTIONS, 0), jnp.repeat(acts, S.N_ACTIONS, 0)
    acts4 = acts4.at[:, -1].set(jnp.tile(jnp.arange(S.N_ACTIONS, dtype=jnp.int32), n))
    if key is None:
        nxt = model.predict(emb4, model.act_embed(acts4))[:, -1:]
    else:
        nxt = model.sample(emb4, acts4, key)[:, None]
    return (jnp.concatenate([emb4[:, 1:], nxt], 1),
            jnp.concatenate([acts4[:, 1:], jnp.zeros((n * S.N_ACTIONS, 1), jnp.int32)], 1), emb4[:, -1], acts4[:, -1])


@nnx.jit
def expand(model, emb, acts):
    """Every child of every node: (N, h, D) latents, (N, h) outgoing actions -> (N * 4, h, D), (N * 4, h)."""
    return _children(model, emb, acts)[:2]


@nnx.jit
def expand_heads(model, heads, emb, acts):
    """Children plus the reward and termination probability of the transition that made them."""
    emb, acts, parent, taken = _children(model, emb, acts)
    reward, logit_done, _ = heads(parent, taken)
    return emb, acts, reward.mean(-1), jax.nn.sigmoid(logit_done)


@nnx.jit
def expand_sampled(model, emb, acts, key):
    return _children(model, emb, acts, key)[:2]


@nnx.jit
def expand_heads_sampled(model, heads, emb, acts, key):
    emb, acts, parent, taken = _children(model, emb, acts, key)
    reward, logit_done, _ = heads(parent, taken)
    return emb, acts, reward.mean(-1), jax.nn.sigmoid(logit_done)


@nnx.jit
def value_of(heads, emb, action):
    return heads(emb[:, -1], action)[2].mean(-1)


def chunked(fn, *arrays):
    n = arrays[0].shape[0]
    outs = [fn(*[a[i:i + ROWS] for a in arrays]) for i in range(0, n, ROWS)]
    if not isinstance(outs[0], tuple):
        return jnp.concatenate(outs)
    return tuple(jnp.concatenate([o[k] for o in outs]) for k in range(len(outs[0])))


def tree_search(model, heads, emb_hist, act_hist, horizon, gamma, goal=None, key=None, samples=1):
    """Exhaustive search over the 4^horizon plans as a tree, so every prefix is predicted once.

    Returns the cost per (env, plan), plans in itertools.product order (first action slowest): the goal cost
    ||z_H - z_goal||^2, or minus the imagined discounted return through the heads (reward collected while the
    done head keeps the rollout alive, plus the bootstrapped value at the horizon).  With a key the model
    samples its next latent instead of returning the mean, `samples` independent rollouts per plan, and the
    cost is their average (the expected cost under the sampled model; within a rollout the samples share
    their prefixes like the plans do)."""
    if key is None:
        return _search(model, heads, emb_hist, act_hist, horizon, gamma, goal, None)
    costs = [_search(model, heads, emb_hist, act_hist, horizon, gamma, goal, k) for k in jax.random.split(key, samples)]
    return jnp.mean(jnp.stack(costs), 0)


def _search(model, heads, emb_hist, act_hist, horizon, gamma, goal, key):
    envs = emb_hist.shape[0]
    emb, acts = emb_hist, act_hist
    total, alive = jnp.zeros(envs), jnp.ones(envs)
    for j in range(horizon):
        if key is None:
            grow = (lambda e, a: expand(model, e, a)) if heads is None else (lambda e, a: expand_heads(model, heads, e, a))
        else:
            key, k = jax.random.split(key)
            grow = (lambda e, a: expand_sampled(model, e, a, k)) if heads is None else (lambda e, a: expand_heads_sampled(model, heads, e, a, k))
        if heads is None:
            emb, acts = chunked(grow, emb, acts)
        else:
            emb, acts, reward, p_done = chunked(grow, emb, acts)
            total = jnp.repeat(total, S.N_ACTIONS, 0) + (gamma ** j) * jnp.repeat(alive, S.N_ACTIONS, 0) * reward
            alive = jnp.repeat(alive, S.N_ACTIONS, 0) * (1.0 - p_done)
    rep = S.N_ACTIONS ** horizon
    if heads is None:
        cost = jnp.mean(jnp.square(emb[:, -1] - jnp.repeat(goal, rep, 0)), -1)
    else:
        value = chunked(lambda e, a: value_of(heads, e, a), emb, acts[:, -1])
        cost = -(total + (gamma ** horizon) * alive * value)
    return cost.reshape(envs, rep)


def plan_index_first_action(horizon):
    return (jnp.arange(S.N_ACTIONS ** horizon) // (S.N_ACTIONS ** (horizon - 1))).astype(jnp.int32)


def warm_up(spec, envs, key, hist, start="uniform"):
    """CTX random steps from fresh episodes; envs that die are redrawn (new episode, new key)."""
    reset, step = S.make(spec, start)
    vreset, vstep = jax.jit(jax.vmap(lambda k: reset(k))), jax.jit(jax.vmap(step))

    @jax.jit
    def roll(key):
        k0, key = jax.random.split(key)
        obs, state = vreset(jax.random.split(k0, envs))
        frames, acts, dead = [obs], [jnp.zeros(envs, jnp.int32)], jnp.zeros(envs, bool)
        for _ in range(CTX - 1):
            key, ka, ks = jax.random.split(key, 3)
            action = jax.random.randint(ka, (envs,), 0, S.N_ACTIONS)
            obs, state, _, done = vstep(state, action, jax.random.split(ks, envs))
            frames.append(obs)
            acts.append(action)
            dead = dead | done
        return jnp.stack(frames, 1), jnp.stack(acts, 1), state, dead

    frames, acts, state, dead = roll(key)
    for i in range(20):
        if not bool(dead.any()):
            break
        f2, a2, s2, d2 = roll(jax.random.fold_in(key, i + 1))
        pick = lambda new, old: jnp.where(dead.reshape((envs,) + (1,) * (new.ndim - 1)), new, old)
        frames, acts, state = pick(f2, frames), pick(a2, acts), jax.tree.map(pick, s2, state)
        dead = dead & d2
    return frames, acts, state, vstep


def run(bundle, spec, envs, steps, horizon, seed, gamma, policies, goal_where, sampler=None, samples=1, start="uniform"):
    model, heads = (sampler or bundle.model), bundle.heads
    hist = model.history
    key = jax.random.PRNGKey(seed)
    key, kw, kg, ks = jax.random.split(key, 4)
    frames, acts, state, vstep = warm_up(spec, envs, kw, hist, start)
    start_emb = viz.encode(model, frames[:, -hist:])
    start_act = jnp.concatenate([acts[:, -hist + 1:], jnp.zeros((envs, 1), jnp.int32)], 1)
    goal_emb = viz.encode(model, goal_frames(spec, state, kg, goal_where)[:, None])[:, 0]
    step_keys = jax.random.split(ks, steps * envs).reshape(steps, envs, -1)
    first = plan_index_first_action(horizon)
    single = plan_index_first_action(1)
    results = {}
    for name in policies:
        if name in ("plan", "greedy") and heads is None:
            continue
        t0 = time.time()
        emb_hist, act_hist, run_state = start_emb, start_act, state
        alive = jnp.ones(envs)
        running = jnp.ones(envs)
        length = jnp.full((envs,), CTX - 1, jnp.int32)
        walls = jnp.zeros(envs, jnp.int32)
        imagined = []
        pkey = jax.random.PRNGKey(seed + 17)
        for t in range(steps):
            pkey, sub, ksearch = jax.random.split(pkey, 3)
            draw = dict(key=ksearch, samples=samples) if sampler is not None else {}
            if name == "random":
                action = jax.random.randint(sub, (envs,), 0, S.N_ACTIONS)
            elif name == "centre":
                action = R.centre_policy(run_state)
            elif name == "goal":
                cost = tree_search(model, None, emb_hist, act_hist, horizon, gamma, goal_emb, **draw)
                action = first[jnp.argmin(cost, 1)]
            elif name == "greedy":
                cost = tree_search(model, heads, emb_hist, act_hist, 1, gamma, **draw)
                action = single[jnp.argmin(cost, 1)]
                imagined.append(-jnp.min(cost, 1))
            else:
                cost = tree_search(model, heads, emb_hist, act_hist, horizon, gamma, **draw)
                action = first[jnp.argmin(cost, 1)]
                imagined.append(-jnp.min(cost, 1))
            prev_wall = run_state.wall_x
            obs, run_state, reward, done = vstep(run_state, action, step_keys[t])
            length = length + running.astype(jnp.int32)
            walls = walls + (running * ((prev_wall != S.NO_WALL) & (run_state.wall_x == S.NO_WALL))).astype(jnp.int32)
            alive = alive * (1.0 - (reward < 0).astype(jnp.float32))
            running = running * (1.0 - done.astype(jnp.float32))
            emb_hist = jnp.concatenate([emb_hist[:, 1:], viz.encode(model, obs[:, None])], 1)
            act_hist = jnp.concatenate([act_hist[:, 1:-1], action[:, None], jnp.zeros((envs, 1), jnp.int32)], 1)
        length, walls, alive = np.asarray(length), np.asarray(walls), np.asarray(alive)
        results[name] = {"survival": float(alive.mean()), "survival_sem": float(alive.std() / np.sqrt(envs)),
                         "length": float(length.mean()), "deaths_per_100": float(100.0 * (1.0 - alive).sum() / length.sum()),
                         "walls_passed": float(walls.mean()), "seconds": round(time.time() - t0, 1),
                         "lengths": length.tolist()}
        if imagined:
            results[name]["imagined_return_mean"] = float(jnp.stack(imagined, 1).mean())
        print(f"[control] {name}: " + json.dumps({k: v for k, v in results[name].items() if k != "lengths"}), flush=True)
    results.update({"envs": envs, "steps": steps, "horizon": horizon, "gamma": gamma, "goal_image": goal_where,
                    "latent_dim": int(model.latent_dim), "history": int(hist), "protocol": "lengths from reset, cap is not a death",
                    "sampled": sampler is not None, "samples": samples, "start": start,
                    "sampler_mean_of": int(getattr(sampler, "mean_of", 0) or 0)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(S.CONFIGS))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=S.MAX_STEPS)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--goal", default="centre", choices=["centre", "left"])
    parser.add_argument("--policies", default="random,centre,goal,greedy,plan")
    parser.add_argument("--sampler", type=Path, help="flow head on the residual (studies.stoch.sampler): sample instead of the mean")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--sampler-mean", type=int, default=0, help="K > 0: the head's mean of K samples instead of a sample")
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--start", default="uniform", choices=["uniform", "edge"], help="reset distribution (env.make)")
    args = parser.parse_args()
    bundle, meta = T.load_model(args.model)
    sampler = SM.load(bundle, args.sampler, args.nfe, args.sampler_mean)[0] if args.sampler else None
    samples = 1 if (sampler is None or args.sampler_mean) else args.samples
    report = run(bundle, S.CONFIGS[args.config], args.envs, args.steps, args.horizon, args.seed, args.gamma,
                 args.policies.split(","), args.goal, sampler, samples, args.start)
    report["model"] = str(args.model)
    report["sampler"] = str(args.sampler) if args.sampler else None
    report["config"] = args.config
    if bundle.heads is not None:
        report["heads"] = meta["heads"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
