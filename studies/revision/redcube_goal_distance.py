"""Post-outcome operating-range diagnostic; original hard-goal results unchanged.

Uses the exact frozen H1/H3 controller and evaluator implementation, all three
admitted model seeds, and paired 128-case goal shells at 1,2,3,6,9,12 actions.
Only the target distribution changes. No controller setting is retuned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from studies.revision import redcube_goal_eval as G

DISTANCES = (1, 2, 3, 6, 9, 12)
N = 128
PRIMARY_SOURCE_SHA = "364088b184d2368522d79276724ceaa3ff1eae1610012855344fee0162129683"


def make_distance_cohort(spec, evaluation_seed, goal_seed):
    original = G.cohort(spec, N, evaluation_seed)
    ids, frames, actions, states, _, _, rejected = original
    rng = np.random.default_rng(goal_seed)
    goals = []
    for distance in DISTANCES:
        for start in states.pos:
            candidates = G.goal_candidates(start, min_actions=distance, max_actions=distance)
            goals.append(candidates[rng.integers(len(candidates))])
    goals = np.asarray(goals, np.int32)
    copies = len(DISTANCES)
    ids = np.tile(ids, copies)
    frames = np.concatenate([frames] * copies)
    actions = np.concatenate([actions] * copies)
    states = G.jax.tree.map(lambda x: np.concatenate([x] * copies), states)
    # Same per-case background in all six goal images; only cube position varies.
    goal_keys = G.jax.vmap(lambda i: G.jax.random.fold_in(G.jax.random.fold_in(
        G.jax.random.PRNGKey(goal_seed), 71), i))(G.jnp.asarray(ids))
    goal_state = states._replace(pos=goals)
    goal_rgb = np.asarray(G.jax.jit(G.jax.vmap(lambda s, k: G.R.render(spec, s, k)))(goal_state, goal_keys))
    distances = np.repeat(np.asarray(DISTANCES, np.int32), N)
    np.testing.assert_array_equal(np.abs(states.pos - goals).sum(1), distances * G.R.STEP)
    return (ids, frames, actions, states, goals, goal_rgb, rejected), distances


def summarize_by_distance(episodes, distances):
    report = {}
    for distance in DISTANCES:
        take = distances == distance
        assert int(take.sum()) == N
        part = {}
        for name in ("random", "fixed_cycle", "oracle", "jepa_h1", "jepa_h3"):
            hit = episodes[name + "_hit"][take, -1]
            minimum = episodes[name + "_minimum_distance_l1"][take]
            part[name] = {
                "exact_goal_hit": G.binary(hit),
                "within_one_action_l1_3": G.binary(minimum <= G.R.STEP),
                "died_before_goal": G.binary(~episodes[name + "_alive"][take, -1]),
                "mean_minimum_distance_l1": float(minimum.mean()),
                "mean_final_distance_l1": float(episodes[name + "_distance_l1"][take, -1].mean()),
                "mean_first_hit_steps_given_success": float(episodes[name + "_first_hit"][take][hit].mean()) if hit.any() else None,
            }
        report[str(distance)] = part
    return report


def replay_preselected_rgb(spec, data, episodes, distances, evaluation_seed, out):
    """Exact keyed simulator re-rendering of recorded executions, not new control."""
    ids, frames, _, states, goals, goal_rgb, _ = data
    chosen = np.arange(len(DISTANCES), dtype=np.int32) * N
    assert np.array_equal(distances[chosen], DISTANCES)
    packet = {"row_indices": chosen, "within_shell_case_indices": np.zeros(len(DISTANCES), np.int32),
              "distance_actions": distances[chosen], "candidate_ids": ids[chosen],
              "goal_rgb": goal_rgb[chosen], "goal_positions": goals[chosen], "prefix_rgb": frames[chosen]}
    step = G.jax.jit(G.jax.vmap(G.R.make(spec)[1]))
    b = len(chosen)
    for name in ("random", "fixed_cycle", "oracle", "jepa_h1", "jepa_h3"):
        state = G.jax.tree.map(lambda x: G.jnp.asarray(x[chosen]), states)
        obs = G.jnp.asarray(frames[chosen, -1])
        rendered = [np.asarray(obs)]
        actions = episodes[name + "_actions"][chosen]
        for t in range(actions.shape[1]):
            active = actions[:, t] >= 0
            keys = G.jax.vmap(lambda i: G.jax.random.fold_in(
                G.jax.random.PRNGKey(evaluation_seed + 30001 + t), i))(G.jnp.asarray(ids[chosen]))
            new_obs, new_state, _, _ = step(state, G.jnp.asarray(np.maximum(actions[:, t], 0)), keys)
            obs = G.jnp.where(G.jnp.asarray(active)[:, None, None, None], new_obs, obs)
            state = G.jax.tree.map(lambda a, z: G.jnp.where(G.jnp.asarray(active).reshape((b,) + (1,) * (a.ndim - 1)), a, z), new_state, state)
            np.testing.assert_array_equal(np.asarray(state.pos), episodes[name + "_positions"][chosen, t + 1])
            rendered.append(np.asarray(obs))
        packet[name + "_rgb"] = np.stack(rendered, 1)
        for suffix in ("actions", "positions", "alive", "hit"):
            packet[name + "_" + suffix] = episodes[name + "_" + suffix][chosen]
        packet[name + "_active"] = actions >= 0
    np.savez_compressed(out / "range_visual_packet.npz", **packet)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluation-seed", type=int, default=20260920)
    parser.add_argument("--goal-seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.evaluation_seed != 20260920 or args.goal_seed != 20260921:
        parser.error("diagnostic seeds are frozen before launch")
    if args.out.exists():
        raise FileExistsError("refusing to overwrite an existing diagnostic")
    if G.sha256(Path(G.__file__)) != PRIMARY_SOURCE_SHA:
        raise ValueError("original controller/evaluator source changed; refuse unpaired follow-up")
    args.out.mkdir(parents=True)
    G.initialize_runtime()
    began = time.time()
    bundle, meta = G.T.load_model(args.model)
    spec = G.R.CONFIGS["gray_frame_still"]
    data, distances = make_distance_cohort(spec, args.evaluation_seed, args.goal_seed)
    ids, frames, actions, states, goals, goal_rgb, rejected = data
    model_seed = int(meta["cfg"]["seed"])
    if model_seed not in (0, 1, 2):
        raise ValueError("only the three prospectively admitted original model seeds are allowed")
    primary = ROOT / "runs/revision/redcube_goal/gray_frame_still" / f"m{model_seed}"
    primary_manifest = json.loads((primary / "manifest.json").read_text())
    if primary_manifest["model_sha256"] != G.sha256(args.model):
        raise ValueError("model differs from the original hard-goal evaluation")
    with np.load(primary / "cohort.npz", allow_pickle=False) as archive:
        for name, value in {"candidate_ids": ids[:N], "frames": frames[:N], "warmup_actions": actions[:N],
                            **{"state_" + k: v[:N] for k, v in states._asdict().items()}}.items():
            np.testing.assert_array_equal(archive[name], value)
    arrays = {"candidate_ids": ids, "frames": frames, "warmup_actions": actions, "goals": goals,
              "goal_rgb": goal_rgb, "distance_actions": distances,
              "within_shell_case_index": np.tile(np.arange(N, dtype=np.int32), len(DISTANCES)),
              "goal_offsets": goals - states.pos, **{"state_" + k: v for k, v in states._asdict().items()}}
    np.savez_compressed(args.out / "cohort.npz", **arrays)
    manifest = {
        "protocol": "redcube_goal_distance_v1", "classification": "post-outcome operating-range diagnostic",
        "primary_protocol_unchanged": "redcube_goal_control_v1", "config": spec.name,
        "prefix_seed": args.evaluation_seed, "evaluation_seed": args.evaluation_seed, "goal_seed": args.goal_seed,
        "n_unique_prefixes": N, "n_per_distance": N, "n_total_correlated_rows": len(ids), "distances_actions": list(DISTANCES),
        "layout": "shell-major; row = distance_index * 128 + within_shell_case_index",
        "control_steps": 24, "horizon": 3, "encoder_batch": [16, G.T.CTX],
        "model_sha256": G.sha256(args.model), "model_metadata": meta,
        "parent_primary_cohort_sha256": G.sha256(primary / "cohort.npz"),
        "parent_primary_manifest_sha256": G.sha256(primary / "manifest.json"),
        "latent_dim": bundle.model.latent_dim, "history": bundle.model.history,
        "prefix_rejected_before_last_admission": rejected, "prefix_admitted_unique": N,
        "visual_rows_predeclared": (np.arange(len(DISTANCES)) * N).tolist(), "visual_within_shell_case_index": 0,
        "visual_provenance": "exact keyed re-rendering of recorded actions/states, with first-terminal padding frozen",
        "cohort_sha256": G.sha256(args.out / "cohort.npz"),
        "cohort_array_sha256": {k: G.array_sha256(v) for k, v in arrays.items()},
        "source_sha256": {str(p.relative_to(ROOT)): G.sha256(p) for p in
                          (Path(__file__), Path(G.__file__), ROOT / "train.py", ROOT / "viz.py", ROOT / "probe.py", ROOT / "studies/redcube/env.py")},
        "inferential_scope": "Wilson intervals separately per model and distance; never an IID interval across all 768 correlated rows",
        "within_one_action_caution": "distance-one cases satisfy L1<=3 at the start; this is not evidence of control",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    eval_args = argparse.Namespace(batch=16, envs=len(ids), horizon=3, steps=24, seed=args.evaluation_seed)
    pooled_internal = G.evaluate(bundle, spec, data, eval_args, args.out)
    # Preserve original progress output, but explicitly label its pooled binomial
    # intervals as non-inferential bookkeeping over correlated shell cases.
    internal = {"warning": "Internal progress only; pooled confidence intervals across repeated prefixes are not inferential results.",
                "values": pooled_internal}
    (args.out / "summary.partial.json").rename(args.out / "internal_pooled_progress.original.json")
    (args.out / "internal_pooled_progress.json").write_text(json.dumps(internal, indent=2))
    with np.load(args.out / "episodes.npz", allow_pickle=False) as archive:
        episodes = {k: archive[k] for k in archive.files}
    report = summarize_by_distance(episodes, distances)
    replay_preselected_rgb(spec, data, episodes, distances, args.evaluation_seed, args.out)
    # The overlap with directly recorded first-shell visual frames must be exact.
    with np.load(args.out / "visual_packet.npz") as direct, np.load(args.out / "range_visual_packet.npz") as replay:
        for name in ("random", "fixed_cycle", "oracle", "jepa_h1", "jepa_h3"):
            np.testing.assert_array_equal(direct[name + "_rgb"][0], replay[name + "_rgb"][0])
    summary = {"manifest": manifest, "by_distance_actions": report,
               "policy_runtime_seconds": {k: v["seconds"] for k, v in pooled_internal.items()},
               "total_seconds": time.time() - began,
               "artifacts": {p.name: G.sha256(p) for p in
                             (args.out / "episodes.npz", args.out / "visual_packet.npz", args.out / "range_visual_packet.npz", args.out / "cohort.npz")}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"finished": True, "total_seconds": summary["total_seconds"],
                      "H3_exact_hits_per_128_by_distance": {d: r["jepa_h3"]["exact_goal_hit"]["count"] for d, r in report.items()}}), flush=True)


if __name__ == "__main__":
    main()
