"""Planning in the JEPA latent inside the red-cube world.

  goal     LeWM's own planner: squared latent distance to a goal image (the cube at the centre), replanned
           every step over every action sequence of the horizon (4^h of them, so the search is exhaustive).
  myopic / greedy / plan   reward MPC through the off-budget reward / done / value heads (argmax r-hat,
           one imagined step plus V-hat, exhaustive horizon-h search of the imagined return).
  random / centre   the matched references (same start states, same environment noise).

    python -m studies.redcube.control --config color_frame_wind --model /dev/shm/redcube/color_frame_wind/grid/model_probe.npz \
        --out runs/redcube/color_frame_wind/control_grid.json
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import plan as P
import train as T
import viz
from studies.redcube import env as R

CTX = P.CTX


def all_plans(horizon):
    return jnp.asarray(list(itertools.product(range(R.N_ACTIONS), repeat=horizon)), jnp.int32)


def goal_frames(spec, states, key):
    """The cube at the centre, on each episode's own texture (a fresh draw where the noise is per frame)."""
    centre = jnp.full_like(states.pos, int(round(R.CENTRE)))
    goal_state = R.State(centre, states.wind, states.texture, states.step, states.episode)
    return jax.vmap(lambda s, k: R.render(spec, s, k))(goal_state, jax.random.split(key, states.pos.shape[0]))


def run(bundle, spec, envs, steps, horizon, seed, gamma):
    model, heads = bundle.model, bundle.heads
    reset, step = R.make(spec)
    vreset = jax.jit(jax.vmap(lambda k: reset(k)))
    vstep = jax.jit(jax.vmap(step))
    hist = model.history
    plans = all_plans(horizon)
    n_plans = plans.shape[0]

    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    obs, state = vreset(jax.random.split(sub, envs))
    frames, acts = [obs], [jnp.zeros(envs, jnp.int32)]
    for _ in range(CTX - 1):
        key, act_key, step_key = jax.random.split(key, 3)
        action = jax.random.randint(act_key, (envs,), 0, R.N_ACTIONS)
        obs, state, _, done = vstep(state, action, jax.random.split(step_key, envs))
        frames.append(obs)
        acts.append(action)
    warm_dead = float(jnp.mean(done))
    start_emb = viz.encode(model, jnp.stack(frames[-hist:], 1))
    start_act = jnp.concatenate([jnp.stack(acts[-hist + 1:], 1), jnp.zeros((envs, 1), jnp.int32)], 1)
    key, goal_key = jax.random.split(key)
    goal_emb = viz.encode(model, goal_frames(spec, state, goal_key)[:, None])[:, 0]
    step_keys = jax.random.split(key, steps * envs).reshape(steps, envs, -1)
    single = jnp.tile(jnp.arange(R.N_ACTIONS, dtype=jnp.int32)[None, :, None], (envs, 1, 1))
    tiled_plans = jnp.tile(plans[None], (envs, 1, 1)).reshape(-1, horizon)

    policies = ["random", "centre", "goal"] + (["myopic", "greedy", "plan"] if heads is not None else [])
    results = {}
    for name in policies:
        t0 = time.time()
        emb_hist, act_hist, run_state = start_emb, start_act, state
        alive = jnp.ones(envs)
        reward_log, imagined = [], []
        pkey = jax.random.PRNGKey(seed + 17)
        for t in range(steps):
            pkey, sub = jax.random.split(pkey)
            if name == "random":
                action = jax.random.randint(sub, (envs,), 0, R.N_ACTIONS)
            elif name == "centre":
                action = R.centre_policy(run_state)
            elif name == "goal":
                cost = P.batched_cost(model, jnp.repeat(emb_hist, n_plans, 0), jnp.repeat(act_hist, n_plans, 0),
                                      tiled_plans, jnp.repeat(goal_emb, n_plans, 0), horizon).reshape(envs, n_plans)
                action = plans[jnp.argmin(cost, 1), 0]
            elif name in ("myopic", "greedy"):
                cost = P.batched_reward_cost(model, heads, jnp.repeat(emb_hist, R.N_ACTIONS, 0),
                                             jnp.repeat(act_hist, R.N_ACTIONS, 0), single.reshape(-1, 1), 1,
                                             gamma if name == "greedy" else 0.0).reshape(envs, -1)
                action = jnp.argmin(cost, 1).astype(jnp.int32)
                imagined.append(-jnp.min(cost, 1))
            else:
                cost = P.batched_reward_cost(model, heads, jnp.repeat(emb_hist, n_plans, 0),
                                             jnp.repeat(act_hist, n_plans, 0), tiled_plans, horizon,
                                             gamma).reshape(envs, n_plans)
                action = plans[jnp.argmin(cost, 1), 0]
                imagined.append(-jnp.min(cost, 1))
            obs, run_state, reward, done = vstep(run_state, action, step_keys[t])
            reward_log.append(reward * alive)
            alive = alive * (1.0 - done.astype(jnp.float32))
            emb_hist = jnp.concatenate([emb_hist[:, 1:], viz.encode(model, obs[:, None])], 1)
            act_hist = jnp.concatenate([act_hist[:, 1:-1], action[:, None], jnp.zeros((envs, 1), jnp.int32)], 1)
        rewards = np.asarray(jnp.stack(reward_log, 1))
        alive_steps = np.asarray(jnp.stack([jnp.ones(envs)] + [1.0 - (rewards[:, :t + 1] < -0.5).any(1)
                                                            for t in range(steps - 1)], 1))
        results[name] = {"return": float(rewards.sum(1).mean()), "survival": float(np.asarray(alive).mean()),
                         "length": float(alive_steps.sum(1).mean()), "seconds": round(time.time() - t0, 1)}
        if imagined:
            results[name]["imagined_return_mean"] = float(jnp.stack(imagined, 1).mean())
        print(f"[control] {name}: {json.dumps(results[name])}", flush=True)
    results.update({"envs": envs, "steps": steps, "horizon": horizon, "gamma": gamma, "n_plans": int(n_plans),
                    "warmup_dead": warm_dead, "latent_dim": int(model.latent_dim)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(R.CONFIGS))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=R.MAX_STEPS)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    bundle, meta = T.load_model(args.model)
    report = run(bundle, R.CONFIGS[args.config], args.envs, args.steps, args.horizon, args.seed, args.gamma)
    report["model"] = str(args.model)
    if bundle.heads is not None:
        report["heads"] = meta["heads"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
