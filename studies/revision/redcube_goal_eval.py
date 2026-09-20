"""Frozen JEPA goal-control positive control with fresh, paired valid histories.

The controller sees RGB history, outgoing actions and an RGB goal only. The
external task terminates at its first exact goal hit or native death. No model,
reward head, decoder, latent scale or environment dynamics is trained or changed.
The three-step planner uses a disclosed summed stage cost, unlike the historical
terminal-only planner. This is empty-arena navigation, not strategic planning.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(array.dtype).encode() + repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def goal_candidates(start, lo=2, hi=55, step=3, min_actions=6, max_actions=12):
    xy = np.asarray(list(itertools.product(range(lo, hi + 1), repeat=2)), np.int32)
    offset = xy - np.asarray(start)
    distance = np.abs(offset).sum(-1)
    keep = ((offset % step) == 0).all(-1) & (distance >= min_actions * step) & (distance <= max_actions * step)
    return xy[keep]


def binary(values):
    values = np.asarray(values, bool)
    n, k = len(values), int(values.sum())
    p, z = k / n, 1.959963984540054
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"count": k, "n": n, "fraction": p, "wilson_95": [max(0., c - half), min(1., c + half)]}


def initialize_runtime():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    global jax, jnp, nnx, R, T, viz
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import train as T
    import viz
    from studies.redcube import env as R


def cohort(spec, count, seed):
    """First surviving candidates, never selected using any policy outcome."""
    reset, step = R.make(spec)
    vreset, vstep = jax.jit(jax.vmap(reset)), jax.jit(jax.vmap(step))
    base = jax.random.PRNGKey(seed)
    pieces, start_id, rejected = [], 0, 0
    while sum(len(p[0]) for p in pieces) < count:
        ids = np.arange(start_id, start_id + 256, dtype=np.int32)
        reset_keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(base, 11), i))(jnp.asarray(ids))
        obs, state = vreset(reset_keys)
        frames, actions, alive = [obs], [], np.ones(len(ids), bool)
        for t in range(T.CTX - 1):
            action_keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(base, 21 + t), i))(jnp.asarray(ids))
            act = jax.vmap(lambda k: jax.random.randint(k, (), 0, R.N_ACTIONS))(action_keys)
            noise_keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(base, 41 + t), i))(jnp.asarray(ids))
            obs, state, _, done = vstep(state, act, noise_keys)
            alive &= ~np.asarray(done)
            frames.append(obs)
            actions.append(act)
        keep = np.nonzero(alive)[0]
        remaining = count - sum(len(p[0]) for p in pieces)
        keep = keep[:remaining]
        # Rejection count ends at the last admitted candidate, not unused draws.
        examined = int(keep[-1]) + 1 if len(keep) == remaining else len(ids)
        rejected += int((~alive[:examined]).sum())
        if len(keep):
            pieces.append((ids[keep], np.asarray(jnp.stack(frames, 1))[keep],
                           np.asarray(jnp.stack(actions, 1))[keep],
                           jax.tree.map(lambda a: np.asarray(a)[keep], state)))
        start_id += 256
    ids = np.concatenate([p[0] for p in pieces])
    frames = np.concatenate([p[1] for p in pieces])
    actions = np.concatenate([p[2] for p in pieces])
    states = jax.tree.map(lambda *xs: np.concatenate(xs), *[p[3] for p in pieces])
    rng = np.random.default_rng(seed + 10001)
    goals = np.stack([g[rng.integers(len(g))] for g in map(goal_candidates, states.pos)])
    goal_state = states._replace(pos=goals)
    goal_keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(base, 71), i))(jnp.asarray(ids))
    goal_rgb = np.asarray(jax.jit(jax.vmap(lambda s, k: R.render(spec, s, k)))(goal_state, goal_keys))
    return ids, frames, actions, states, goals, goal_rgb, rejected


def make_planner(model, horizon):
    plans = jnp.asarray(list(itertools.product(range(R.N_ACTIONS), repeat=horizon)), jnp.int32)
    nplans = len(plans)

    @nnx.jit
    def choose(model, frames, outgoing, goal_rgb):
        b = frames.shape[0]
        # Same encoder batch shape for goal and history; no synthetic history is
        # fed to the dynamics. The encoder itself is framewise in eval mode.
        emb = model.encode(frames)[:, -model.history:]
        goal = model.encode(jnp.repeat(goal_rgb[:, None], T.CTX, axis=1))[:, -1]
        emb = jnp.repeat(emb, nplans, axis=0)
        goal = jnp.repeat(goal, nplans, axis=0).astype(jnp.float32)
        tiled = jnp.tile(plans[None], (b, 1, 1)).reshape(-1, horizon)
        padded = jnp.concatenate([tiled, tiled[:, -1:]], 1)
        acts = jnp.repeat(outgoing[:, -model.history:], nplans, axis=0)
        acts = jnp.concatenate([acts[:, :-1], tiled[:, :1]], 1)

        def one(carry, t):
            history, past = carry
            nxt = model.predict(history, model.act_embed(past))[:, -1:]
            follow = jax.lax.dynamic_slice_in_dim(padded, t + 1, 1, axis=1)
            new = (jnp.concatenate([history[:, 1:], nxt], 1), jnp.concatenate([past[:, 1:], follow], 1))
            cost = jnp.mean(jnp.square(nxt[:, 0].astype(jnp.float32) - goal), -1)
            return new, (cost, nxt[:, 0])

        _, (costs, predicted) = jax.lax.scan(one, (emb, acts), jnp.arange(horizon))
        total = costs.sum(0).reshape(b, nplans)
        selected = jnp.argmin(total, axis=1)
        pred = predicted.transpose(1, 0, 2).reshape(b, nplans, horizon, model.latent_dim)
        return (plans[selected, 0], plans[selected], total[jnp.arange(b), selected],
                pred[jnp.arange(b), selected], jnp.isfinite(total).all())

    return functools.partial(choose, model)


def oracle_action(position, goals, horizon):
    """Privileged reference, same action search with physical distance cost."""
    plans = np.asarray(list(itertools.product(range(R.N_ACTIONS), repeat=horizon)), np.int32)
    moves = np.asarray(R.MOVES)
    future = position[:, None, None] + np.cumsum(moves[plans], axis=1)[None]
    dead = ((future < R.SAFE_LO) | (future > R.SAFE_HI)).any(axis=(2, 3))
    cost = np.square(future - goals[:, None, None]).sum(axis=(2, 3)).astype(np.float64)
    cost[dead] = np.inf
    if not np.isfinite(cost.min(1)).all():
        raise AssertionError("safe start should have at least one surviving 3-step plan")
    return plans[cost.argmin(1), 0]


def evaluate(bundle, spec, data, args, out):
    ids, starts, warm_actions, states, goals, goal_rgb, rejected = data
    b, n = args.batch, args.envs
    if n % b:
        raise ValueError("envs must be divisible by batch to preserve encoder shapes")
    if bundle.model.history > T.CTX:
        raise ValueError("model history exceeds the eight real prefix frames")
    model = bundle.model
    planners = {1: make_planner(model, 1), args.horizon: make_planner(model, args.horizon)}
    step = jax.jit(jax.vmap(R.make(spec)[1]))
    result, arrays = {}, {}
    visual = {"episode_indices": np.arange(3), "candidate_ids": ids[:3], "goals_rgb": goal_rgb[:3],
              "prefix_rgb": starts[:3], "goal_positions": goals[:3]}
    policy_names = ("random", "fixed_cycle", "oracle", "jepa_h1", f"jepa_h{args.horizon}")
    for name in policy_names:
        began = time.time()
        rows = []
        for lo in range(0, n, b):
            sl = slice(lo, lo + b)
            frames = jnp.asarray(starts[sl])
            outgoing = jnp.asarray(np.concatenate([warm_actions[sl], np.zeros((b, 1), np.int32)], 1))
            state = jax.tree.map(lambda x: jnp.asarray(x[sl]), states)
            active, alive, hits = np.ones(b, bool), np.ones(b, bool), np.zeros(b, bool)
            first = np.full(b, -1, np.int32)
            poslog, actlog, alivelog, hitlog, distlog, costlog = [np.asarray(state.pos)], [], [alive.copy()], [hits.copy()], [], []
            distances = np.abs(np.asarray(state.pos) - goals[sl]).sum(1)
            distlog.append(distances)
            min_dist = distances.copy()
            pixel_log = [np.asarray(frames[:3, -1])] if lo == 0 else None
            pred_log, plan_log = [], []
            for t in range(args.steps):
                if name == "random":
                    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(args.seed + 20001 + t), i))(jnp.asarray(ids[sl]))
                    action = np.asarray(jax.vmap(lambda k: jax.random.randint(k, (), 0, R.N_ACTIONS))(keys))
                elif name == "fixed_cycle":
                    action = np.full(b, (0, 2, 1, 3)[t % 4], np.int32)
                elif name == "oracle":
                    # Inactive terminal states may lie outside the arena. Replace
                    # only those for the discarded oracle computation.
                    p = np.asarray(state.pos).copy()
                    p[~active] = np.asarray(states.pos[sl])[~active]
                    action = oracle_action(p, goals[sl], args.horizon)
                else:
                    horizon = 1 if name == "jepa_h1" else args.horizon
                    action_jax, plans, cost, predicted, finite = planners[horizon](frames, outgoing, jnp.asarray(goal_rgb[sl]))
                    if not bool(finite):
                        raise FloatingPointError("nonfinite candidate plan cost; evaluation is incomplete")
                    action = np.asarray(action_jax)
                    costlog.append(np.asarray(cost))
                    if lo == 0:
                        plan_log.append(np.asarray(plans[:3]))
                        if bundle.decoder is not None:
                            pred_log.append(np.asarray(viz.decode(bundle.decoder, predicted[:3], horizon=horizon)))
                action = np.asarray(action, np.int32)
                executed = np.where(active, action, -1)
                noise = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(args.seed + 30001 + t), i))(jnp.asarray(ids[sl]))
                obs, new_state, reward, done = step(state, jnp.asarray(np.maximum(executed, 0)), noise)
                obs = jnp.where(jnp.asarray(active)[:, None, None, None], obs, frames[:, -1])
                new_state = jax.tree.map(lambda a, z: jnp.where(jnp.asarray(active).reshape((b,) + (1,) * (a.ndim - 1)), a, z), new_state, state)
                native_done = np.asarray(done) & active
                alive &= ~native_done
                distance = np.abs(np.asarray(new_state.pos) - goals[sl]).sum(1)
                new_hit = active & alive & (distance == 0)
                first[new_hit] = t + 1
                hits |= new_hit
                min_dist = np.minimum(min_dist, np.where(alive, distance, min_dist))
                active &= ~native_done & ~new_hit
                state = new_state
                frames = jnp.concatenate([frames[:, 1:], obs[:, None]], 1)
                outgoing = jnp.concatenate([outgoing[:, 1:-1], jnp.asarray(np.maximum(executed, 0))[:, None], jnp.zeros((b, 1), jnp.int32)], 1)
                poslog.append(np.asarray(state.pos))
                actlog.append(executed)
                alivelog.append(alive.copy())
                hitlog.append(hits.copy())
                distlog.append(distance)
                if lo == 0:
                    pixel_log.append(np.asarray(obs[:3]))
            rows.append({"positions": np.stack(poslog, 1), "actions": np.stack(actlog, 1), "alive": np.stack(alivelog, 1),
                         "hit": np.stack(hitlog, 1), "distance_l1": np.stack(distlog, 1), "first_hit": first,
                         "minimum_distance_l1": min_dist, "selected_cost": np.stack(costlog, 1) if costlog else np.zeros((b, 0))})
            if lo == 0:
                visual[name + "_rgb"] = np.stack(pixel_log, 1)
                visual[name + "_actions"] = np.stack(actlog, 1)[:3]
                visual[name + "_positions"] = np.stack(poslog, 1)[:3]
                visual[name + "_alive"] = np.stack(alivelog, 1)[:3]
                visual[name + "_hit"] = np.stack(hitlog, 1)[:3]
                # Selected forecasts after a terminal event are only padding;
                # visualizers must not present them as executed imagination.
                visual[name + "_forecast_valid"] = np.stack(actlog, 1)[:3] >= 0
                if pred_log:
                    visual[name + "_predicted_rgb"] = np.stack(pred_log, 1)
                if plan_log:
                    visual[name + "_selected_plan"] = np.stack(plan_log, 1)
            print(json.dumps({"policy": name, "episodes_done": lo + b, "total": n, "seconds": round(time.time() - began, 2)}), flush=True)
        joined = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        arrays.update({name + "_" + k: v for k, v in joined.items()})
        hit = joined["hit"][:, -1]
        result[name] = {"exact_goal_hit": binary(hit), "within_one_action_l1_3": binary(joined["minimum_distance_l1"] <= R.STEP),
                        "died_before_goal": binary(~joined["alive"][:, -1]),
                        "mean_minimum_distance_l1": float(joined["minimum_distance_l1"].mean()),
                        "mean_final_distance_l1": float(joined["distance_l1"][:, -1].mean()),
                        "mean_first_hit_steps_given_success": float(joined["first_hit"][hit].mean()) if hit.any() else None,
                        "seconds": time.time() - began}
        np.savez_compressed(out / "episodes.partial.npz", **arrays)
        (out / "summary.partial.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"policy_finished": name, "summary": result[name]}), flush=True)
    np.savez_compressed(out / "episodes.npz", **arrays)
    np.savez_compressed(out / "visual_packet.npz", **visual)
    return result


def self_test():
    # Task reachability and reference MPC are the two non-neural contracts.
    for start in ([2, 2], [28, 31], [55, 55]):
        candidates = goal_candidates(start)
        assert len(candidates)
        assert (((candidates - start) % 3) == 0).all()
        d = np.abs(candidates - start).sum(1) // 3
        assert ((6 <= d) & (d <= 12)).all()
    assert binary([False] * 128)["count"] == 0
    assert binary([True] * 128)["count"] == 128
    print("goal sampling, reachability, and binomial summaries: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", choices=("gray_frame_still", "off_still"), default="gray_frame_still")
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.model is None or args.out is None:
        parser.error("--model and --out are required")
    if args.horizon != 3 or args.steps != 24 or args.envs != 128 or args.batch != 16:
        parser.error("frozen protocol is horizon=3, steps=24, envs=128, batch=16")
    if args.out.exists():
        raise FileExistsError("refusing to overwrite an existing evaluation directory")
    args.out.mkdir(parents=True)
    initialize_runtime()
    started = time.time()
    bundle, meta = T.load_model(args.model)
    spec = R.CONFIGS[args.config]
    if spec.wind:
        raise AssertionError("this protocol is exclusively a deterministic-dynamics positive control")
    data = cohort(spec, args.envs, args.seed)
    ids, frames, actions, states, goals, goal_rgb, rejected = data
    np.savez_compressed(args.out / "cohort.npz", candidate_ids=ids, frames=frames, warmup_actions=actions,
                        goals=goals, goal_rgb=goal_rgb, **{"state_" + k: v for k, v in states._asdict().items()})
    manifest = {"protocol": "redcube_goal_control_v1", "config": args.config, "seed": args.seed, "n": args.envs,
                "control_steps": args.steps, "horizon": args.horizon, "encoder_batch": [args.batch, T.CTX],
                "model_sha256": sha256(args.model), "model_metadata": meta,
                "latent_dim": bundle.model.latent_dim, "history": bundle.model.history,
                "prefix_rejected_before_last_admission": rejected, "prefix_admitted": args.envs,
                "goal_distance_initial_l1_min": int(np.abs(states.pos - goals).sum(1).min()),
                "goal_distance_initial_l1_max": int(np.abs(states.pos - goals).sum(1).max()),
                "goal_distance_initial_l1_mean": float(np.abs(states.pos - goals).sum(1).mean()),
                "visual_indices_predeclared": [0, 1, 2], "cohort_sha256": sha256(args.out / "cohort.npz"),
                "cohort_array_sha256": {k: array_sha256(v) for k, v in
                                        {"candidate_ids": ids, "frames": frames, "warmup_actions": actions,
                                         "goals": goals, "goal_rgb": goal_rgb,
                                         **{"state_" + k: v for k, v in states._asdict().items()}}.items()},
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                                  (Path(__file__), ROOT / "train.py", ROOT / "viz.py", ROOT / "probe.py", ROOT / "studies/redcube/env.py")}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    result = evaluate(bundle, spec, data, args, args.out)
    result = {"manifest": manifest, "policies": result, "total_seconds": time.time() - started,
              "artifacts": {p.name: sha256(p) for p in (args.out / "episodes.npz", args.out / "visual_packet.npz", args.out / "cohort.npz")}}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"finished": True, "total_seconds": result["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
