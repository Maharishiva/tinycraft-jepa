"""Aggregate the predeclared three-model Craftax goal-control replication.

Report training-seed mean/SD separately from evaluation uncertainty. Episode
bootstrap resamples the same episode columns for every trained model and policy;
its interval is conditional on these fitted models, not a new-training-seed CI.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STRATA = {
    "all": lambda row: True,
    "displaced_goal": lambda row: row["goal_displacement_l1"] > 0,
    "goal_displacement_at_least_2": lambda row: row["goal_displacement_l1"] >= 2,
}
POLICIES = ("noop", "random", "reference_replay", "jepa_open_loop", "jepa_receding")
BASELINES = ("noop", "random", "reference_replay")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditional_paired_interval(delta, indices):
    """delta: model × episode; jointly resample shared episode columns."""
    by_episode = np.mean(delta, axis=0)
    boot = by_episode[indices].mean(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def model_summary(outcomes):
    rates = outcomes.mean(axis=1)
    return {"model_mean": float(rates.mean()),
            "model_sample_sd": float(rates.std(ddof=1)),
            "per_model_rates": rates.tolist(),
            "per_model_successes": outcomes.sum(axis=1).astype(int).tolist(),
            "episodes_per_model": int(outcomes.shape[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs=3, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; preserve it and use a fresh path")
    items = [(path, json.loads(path.read_text())) for path in args.reports]
    items.sort(key=lambda x: x[1]["model_metadata"]["cfg"]["seed"])
    paths, reports = zip(*items)
    assert [r["model_metadata"]["cfg"]["seed"] for r in reports] == [0, 1, 2]
    reference = reports[0]
    cfg0 = reference["model_metadata"]["cfg"]
    assert cfg0["steps"] == 8500 and cfg0["tokens"] * cfg0["token_dim"] == 512
    fixed_args = ("seeds", "envs", "horizon", "warmup", "pop", "elite", "iters", "reward", "no_goal")
    for seed, report in enumerate(reports):
        cfg = report["model_metadata"]["cfg"]
        assert cfg == {**cfg0, "seed": seed}
        assert report["model_metadata"]["data_sha256"] == reference["model_metadata"]["data_sha256"]
        assert report["model_metadata"]["source_sha256"] == reference["model_metadata"]["source_sha256"]
        assert report["model_metadata"]["params"] == reference["model_metadata"]["params"]
        assert report["source_sha256"] == reference["source_sha256"]
        assert report["versions"] == reference["versions"]
        assert all(report["arguments"][key] == reference["arguments"][key] for key in fixed_args)
        assert len(report["goal_runs"]) == 3
        for run, ref_run in zip(report["goal_runs"], reference["goal_runs"]):
            for key in ("start_positions", "target_positions", "reference_actions"):
                assert run[key] == ref_run[key]
            for baseline in BASELINES:
                assert run["rows"][baseline] == ref_run["rows"][baseline]
                assert run["traces"][baseline] == ref_run["traces"][baseline]
    rows = {policy: [[row for run in report["goal_runs"] for row in run["rows"][policy]]
                     for report in reports] for policy in POLICIES}
    expected_ids = [(seed, episode) for seed in [101, 202, 303] for episode in range(128)]
    for policy in POLICIES:
        for group in rows[policy]:
            assert [(r["eval_seed"], r["episode"]) for r in group] == expected_ids
    result = {"schema": "craftax-three-model-aggregate-v1", "analysis_source_sha256": digest(__file__), "model_seeds": [0, 1, 2],
              "eval_seeds": [101, 202, 303], "independent_episode_contexts": 384,
              "model_episode_outcomes": 1152, "same_episode_contexts_across_models": True,
              "training_seed_uncertainty": "mean and sample SD across 3 model success rates",
              "episode_interval_scope": "paired episode bootstrap conditional on the three fitted models; does not include training-seed uncertainty",
              "episode_bootstrap": {"replicates": 20000, "rng_seed": 20092026, "shared_columns_across_models_and_policies": True},
              "source_reports": [{"path": str(p), "sha256": digest(p), "model_sha256": r["model_sha256"],
                                  "model_seed": i, "train_s": r["model_metadata"]["train_s"],
                                  "params": r["model_metadata"]["params"]} for i, (p, r) in enumerate(items)],
              "source_sha256": reference["source_sha256"], "versions": reference["versions"],
              "training_data_sha256": reference["model_metadata"]["data_sha256"],
              "integrity_checks": "exact cfg except model seed, source/data/version equality, all 384 episode identifiers and baseline trajectories matched",
              "strata": {}}
    rng = np.random.default_rng(20092026)
    for name, keep in STRATA.items():
        ids = [i for i, row in enumerate(rows["noop"][0]) if keep(row)]
        bootstrap_indices = rng.integers(0, len(ids), size=(20000, len(ids)), dtype=np.int32)
        outcomes = {p: np.array([[int(group[i]["reached_position_alive"]) for i in ids] for group in rows[p]], dtype=float)
                    for p in POLICIES}
        section = {"n_episode_contexts": len(ids), "policies": {}, "paired_differences": {}}
        for policy in POLICIES:
            summary = model_summary(outcomes[policy])
            if policy in BASELINES:
                summary["model_sample_sd"] = None
                summary["note"] = "one shared nonlearned baseline; repeated rows are identical across model reports"
            summary["per_eval_seed"] = {}
            for eval_seed in [101, 202, 303]:
                local_ids = [j for j, i in enumerate(ids) if rows[policy][0][i]["eval_seed"] == eval_seed]
                summary["per_eval_seed"][str(eval_seed)] = {
                    "episodes_per_model": len(local_ids),
                    "per_model_successes": outcomes[policy][:, local_ids].sum(axis=1).astype(int).tolist()}
            section["policies"][policy] = summary
        for policy, baseline in (("jepa_open_loop", "noop"), ("jepa_open_loop", "random"),
                                 ("jepa_receding", "noop"), ("jepa_receding", "random"),
                                 ("jepa_receding", "jepa_open_loop")):
            delta = outcomes[policy] - outcomes[baseline]
            rates = delta.mean(axis=1)
            section["paired_differences"][f"{policy}_minus_{baseline}"] = {
                "mean": float(rates.mean()), "model_sample_sd": float(rates.std(ddof=1)),
                "per_model_differences": rates.tolist(),
                "conditional_episode_bootstrap_95": conditional_paired_interval(delta, bootstrap_indices)}
        result["strata"][name] = section
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for name, section in result["strata"].items():
        print(name, "n=", section["n_episode_contexts"])
        for policy, summary in section["policies"].items():
            print(policy, summary["per_model_successes"], summary["model_mean"], summary["model_sample_sd"])


if __name__ == "__main__":
    main()
