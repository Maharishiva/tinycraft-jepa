"""Fresh Craftax control evaluation for an existing frozen JEPA checkpoint.

The default runs goal control only. It preserves the native 17-action interface,
never transforms latents or rewards, and stores episode-level paired outcomes.
Goals come from random H-step trajectories. Their simulator noise is independent
of execution noise; the reference action sequence is therefore not an oracle.
This evaluates one complete model recipe, not a JEPA-versus-flow ablation.

Example (run from the repo's GPU Python environment):
  python studies/revision/craftax_eval.py --model runs/model_heads.npz \
    --out artifacts/revision/craftax.json --seeds 101 202 303 --reward
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scalar_summary(values):
    """Across-episode uncertainty only; does not imply training-seed replication."""
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "sem": None}
    mean = sum(float(x) for x in values) / n
    sd = math.sqrt(sum((float(x) - mean) ** 2 for x in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd, "sem": sd / math.sqrt(n) if sd is not None else None}


def binary_summary(values):
    out = scalar_summary(values)
    n = len(values)
    out["successes"] = sum(bool(x) for x in values)
    if not n:
        out["wilson_95"] = None
        return out
    p, z = out["mean"], 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    out["wilson_95"] = [max(0.0, center - half), min(1.0, center + half)]
    return out


def goal_summary(rows):
    result = {}
    # The full distribution is always reported. Strata expose easy stationary
    # goals without adapting or filtering the benchmark to model performance.
    for name, keep in (
        ("all", lambda r: True),
        ("initially_alive", lambda r: r["start_alive"]),
        ("start_and_target_alive", lambda r: r["start_alive"] and r["goal_alive"]),
        ("stationary_goal", lambda r: r["goal_displacement_l1"] == 0),
        ("displaced_goal", lambda r: r["goal_displacement_l1"] > 0),
        ("goal_displacement_at_least_2", lambda r: r["goal_displacement_l1"] >= 2),
    ):
        part = [r for r in rows if keep(r)]
        result[name] = {
            "reached_position": binary_summary([r["reached_position"] for r in part]),
            "reached_position_alive": binary_summary([r["reached_position_alive"] for r in part]),
            "position_l1": scalar_summary([r["position_l1"] for r in part]),
            "start_alive": binary_summary([r["start_alive"] for r in part]),
            "target_alive": binary_summary([r["goal_alive"] for r in part]),
        }
    return result


def runtime():
    # Keep --help and statistics checks usable without GPU dependencies.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    global np, jax, jnp, P, T, viz, ENV_FUNCS
    import numpy as np
    import jax
    import jax.numpy as jnp
    import plan as P
    import train as T
    import viz
    ENV_FUNCS = P.craftax()


def key(seed, stream, step=0):
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), stream), step)


def image_u8(obs):
    return jnp.clip(obs * 255.0, 0, 255).astype(jnp.uint8)


def append_context(emb, acts, observation, action, model):
    encoded = viz.encode(model, image_u8(observation)[:, None])
    emb = jnp.concatenate([emb[:, 1:], encoded], axis=1)
    # acts contains outgoing actions, with an unknown-current-action placeholder.
    acts = jnp.concatenate([acts[:, 1:-1], action[:, None], jnp.zeros_like(action[:, None])], axis=1)
    acts = acts[:, -model.history:]
    return emb, acts


def start_context(model, n, seed, warmup):
    reset, step = ENV_FUNCS
    obs, state = reset(jax.random.split(key(seed, 0), n))
    frames, actions = [image_u8(obs)], []
    alive = jnp.ones(n, dtype=bool)
    for t in range(warmup - 1):
        action = jax.random.randint(key(seed, 1, t), (n,), 0, model.n_actions)
        obs, state, _, done, _ = step(jax.random.split(key(seed, 2, t), n), state, action)
        alive = alive & ~done
        frames.append(image_u8(obs))
        actions.append(action)
    history = model.history
    emb = viz.encode(model, jnp.stack(frames[-history:], axis=1))
    past = jnp.stack(actions[-(history - 1):], axis=1) if history > 1 else jnp.zeros((n, 0), jnp.int32)
    acts = jnp.concatenate([past, jnp.zeros((n, 1), jnp.int32)], axis=1)
    return step, obs, state, alive, emb, acts


def evaluate_goals(model, args, seed, noop):
    n, horizon = args.envs, args.horizon
    step, start_obs, start_state, start_alive, initial_emb, initial_acts = start_context(
        model, n, seed, args.warmup)
    reference = jax.random.randint(key(seed, 3), (n, horizon), 0, model.n_actions)
    goal_obs, goal_state, goal_alive = start_obs, start_state, start_alive
    for t in range(horizon):
        goal_obs, goal_state, _, done, _ = step(
            jax.random.split(key(seed, 4, t), n), goal_state, reference[:, t])
        goal_alive = goal_alive & ~done
    target = np.asarray(goal_state.player_position)
    displacement = np.abs(target - np.asarray(start_state.player_position)).sum(-1)
    goal_emb = viz.encode(model, image_u8(goal_obs)[:, None])[:, 0]
    random_plan = jax.random.randint(key(seed, 6), (n, horizon), 0, model.n_actions)
    fixed, _ = P.cem(model, initial_emb, initial_acts, goal_emb, horizon, key(seed, 7),
                     pop=args.pop, elite=args.elite, iters=args.iters)
    all_rows, traces = {}, {}
    policies = ("noop", "random", "reference_replay", "jepa_open_loop", "jepa_receding")
    for policy in policies:
        state, obs, alive = start_state, start_obs, start_alive
        emb, acts = initial_emb, initial_acts
        action_trace, positions, survival = [], [], []
        for t in range(horizon):
            if policy == "jepa_receding":
                # Match the remaining real execution horizon, avoiding an
                # additional horizon budget that grows near the endpoint.
                if t == 0:
                    action = fixed[:, 0]
                else:
                    remaining, _ = P.cem(model, emb, acts, goal_emb, horizon - t, key(seed, 7, t),
                                         pop=args.pop, elite=args.elite, iters=args.iters)
                    action = remaining[:, 0]
            else:
                action = {"noop": jnp.full((n,), noop, jnp.int32),
                          "random": random_plan[:, t], "reference_replay": reference[:, t],
                          "jepa_open_loop": fixed[:, t]}[policy]
            obs, state, _, done, _ = step(jax.random.split(key(seed, 5, t), n), state, action)
            alive = alive & ~done
            action_trace.append(np.asarray(action))
            positions.append(np.asarray(state.player_position))
            survival.append(np.asarray(alive))
            if policy == "jepa_receding":
                emb, acts = append_context(emb, acts, obs, action, model)
        distance = np.abs(np.asarray(state.player_position) - target).sum(-1)
        rows = []
        for i in range(n):
            rows.append({"eval_seed": seed, "episode": i, "goal_displacement_l1": int(displacement[i]),
                         "start_alive": bool(start_alive[i]), "goal_alive": bool(goal_alive[i]),
                         "execution_alive": bool(alive[i]), "position_l1": int(distance[i]),
                         "reached_position": bool(distance[i] == 0),
                         "reached_position_alive": bool(distance[i] == 0 and alive[i])})
        all_rows[policy] = rows
        traces[policy] = {"actions": np.stack(action_trace, 1).tolist(),
                          "positions": np.stack(positions, 1).tolist(),
                          "alive": np.stack(survival, 1).tolist()}
        print(f"goal seed={seed} policy={policy}: {json.dumps(goal_summary(rows)['all'])}", flush=True)
    return {"rows": all_rows, "traces": traces, "target_positions": target.tolist(),
            "reference_actions": np.asarray(reference).tolist(),
            "start_positions": np.asarray(start_state.player_position).tolist()}


def evaluate_reward(model, heads, args, seed, noop):
    n = args.envs
    step, _, state0, start_alive, emb0, acts0 = start_context(model, n, seed, args.warmup)
    candidates = jnp.tile(jnp.arange(model.n_actions, dtype=jnp.int32), n)[:, None]
    result = {}
    for policy in ("noop", "random", "myopic", "greedy", "jepa_mpc"):
        state, emb, acts, alive = state0, emb0, acts0, start_alive
        rewards, actions, survival = [], [], []
        for t in range(args.steps):
            if policy == "noop":
                action = jnp.full((n,), noop, jnp.int32)
            elif policy == "random":
                action = jax.random.randint(key(seed, 20, t), (n,), 0, model.n_actions)
            elif policy in ("myopic", "greedy"):
                cost = P.batched_reward_cost(model, heads, jnp.repeat(emb, model.n_actions, 0),
                    jnp.repeat(acts, model.n_actions, 0), candidates, 1,
                    args.gamma if policy == "greedy" else 0.0).reshape(n, model.n_actions)
                action = jnp.argmin(cost, axis=1).astype(jnp.int32)
            else:
                sequence, _ = P.cem(model, emb, acts, None, args.horizon, key(seed, 21, t),
                    pop=args.pop, elite=args.elite, iters=args.iters, heads=heads, gamma=args.gamma)
                action = sequence[:, 0]
            obs, state, reward, done, _ = step(jax.random.split(key(seed, 22, t), n), state, action)
            rewards.append(np.asarray(reward * alive))
            alive = alive & ~done
            actions.append(np.asarray(action))
            survival.append(np.asarray(alive))
            emb, acts = append_context(emb, acts, obs, action, model)
        rewards = np.stack(rewards, axis=1)
        rows = [{"eval_seed": seed, "episode": i, "return": float(rewards[i].sum()),
                 "start_alive": bool(start_alive[i]), "alive_at_end": bool(alive[i])} for i in range(n)]
        result[policy] = {"rows": rows, "rewards": rewards.tolist(),
                          "actions": np.stack(actions, 1).tolist(),
                          "alive": np.stack(survival, 1).tolist()}
        print(f"reward seed={seed} policy={policy}: {json.dumps(scalar_summary([r['return'] for r in rows]))}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--pop", type=int, default=512)
    parser.add_argument("--elite", type=int, default=32)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--reward", action="store_true")
    parser.add_argument("--no-goal", action="store_true", help="evaluate reward control only (requires --reward)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    for name in ("envs", "horizon", "warmup", "pop", "elite", "iters", "steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.elite > args.pop or len(set(args.seeds)) != len(args.seeds):
        parser.error("elite must not exceed population; evaluation seeds must be unique")
    if args.no_goal and not args.reward:
        parser.error("--no-goal requires --reward")
    if args.out.exists():
        parser.error("output exists; use a fresh path to preserve earlier evidence")
    runtime()
    from craftax.craftax_classic.constants import Action
    noop = int(Action.NOOP.value)
    bundle, metadata = T.load_model(args.model)
    model = bundle.model
    if model.n_actions != 17:
        parser.error("checkpoint must use Craftax Classic's native 17-action interface")
    if args.warmup < model.history:
        parser.error("warmup must cover the checkpoint's history")
    if args.reward and bundle.heads is None:
        parser.error("--reward requires an existing checkpoint with frozen fitted heads")
    started = time.time()
    versions = {}
    for package in ("numpy", "jax", "flax", "craftax"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"schema": "craftax-revision-control-v1", "model": str(args.model.resolve()),
              "model_sha256": sha256(args.model), "model_metadata": metadata,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "versions": versions, "device": [str(d) for d in jax.devices()],
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                                (Path(__file__).resolve(), ROOT / "plan.py", ROOT / "train.py", ROOT / "viz.py", ROOT / "probe.py")},
              "protocol": {"goal": "random-action reachable-image target, independently sampled execution noise",
                           "goal_noise_stream": 4, "execution_noise_stream": 5,
                           "reference_replay_is_oracle": False, "rewards": "native unmodified environment rewards",
                           "random_and_noop_baselines": True, "latent_transform": None,
                           "uncertainty_unit": "environment episode, not model training seed",
                           "noop_action": noop}, "goal_runs": [], "reward_runs": []}
    def persist():
        report["elapsed_s"] = time.time() - started
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temp = args.out.with_suffix(args.out.suffix + ".tmp")
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(args.out)

    for seed in args.seeds:
        if not args.no_goal:
            report["goal_runs"].append(evaluate_goals(model, args, seed, noop))
            persist()
        if args.reward:
            report["reward_runs"].append(evaluate_reward(model, bundle.heads, args, seed, noop))
            persist()
    if not args.no_goal:
        policies = report["goal_runs"][0]["rows"]
        report["goal_summary"] = {p: goal_summary([r for run in report["goal_runs"] for r in run["rows"][p]]) for p in policies}
        for p in policies:
            for baseline in ("noop", "random"):
                differences = [int(x["reached_position_alive"]) - int(y["reached_position_alive"])
                               for run in report["goal_runs"]
                               for x, y in zip(run["rows"][p], run["rows"][baseline])]
                report["goal_summary"][p]["paired_success_difference_vs_" + baseline] = scalar_summary(differences)
    if args.reward:
        report["reward_summary"] = {p: scalar_summary([r["return"] for run in report["reward_runs"] for r in run[p]["rows"]]) for p in report["reward_runs"][0]}
        for p in report["reward_summary"]:
            differences = [x["return"] - y["return"] for run in report["reward_runs"]
                           for x, y in zip(run[p]["rows"], run["random"]["rows"])]
            report["reward_summary"][p]["paired_difference_vs_random"] = scalar_summary(differences)
    persist()
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
