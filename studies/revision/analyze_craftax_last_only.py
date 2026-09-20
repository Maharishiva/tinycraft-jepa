"""Paired six-model Craftax objective comparison with frozen-context validation.

The primary endpoint is one-shot alive endpoint success on all 384 contexts.
Three matched model seeds provide descriptive mean/sample SD. Bootstrap intervals
resample the same episode columns for all six models and are conditional on the
fitted models; they are not confidence intervals over future training seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REFERENCE_REPORT_SHA256 = "44dc765a690037e8cc1b618be01f1348f4f8a54ad02c1d9424a230e02a13f867"
REFERENCE_MODEL_SHA256 = "59cc282c9bd7a49ab1c272342220bc8e0d8ca0f1aa77bbcb4c23ae31dfd6835d"
DATA_SHA256 = "09f2cb41a38b0f34937a89c4122f6b25bbb3094e34a077b3a572504269d0a1b2"
BASE_SOURCE_SHA256 = "72cd5c463b4ac2fe7e289f7032e91d88ffac2e59ebb15534524fafe4804e9d0b"
OBJECTIVE_SOURCE_SHA256 = "b5316f8997cbbae56772520b0232d4f05c35be271a29b8f694d53ad650da64e1"
EVAL_SOURCES = {
    "studies/revision/craftax_eval.py": "73db58b84e64b3405a347fe883d0c87cdad31a454b2a138ea4c73233ddb7e197",
    "plan.py": "3ef0d48cbe4ba026ac29ef44d189958cbf6e64d9e39f6fb00a1ce4a3804c6969",
    "train.py": BASE_SOURCE_SHA256,
    "viz.py": "f60f8586353e2a04ea89a5a1b272c30a155350d4be8790cd75b46c2c6e2fbee3",
    "probe.py": "110eff77f3dfb5fb3c7222caa58d85773b9bc8675cd95859d4f9705c171ea424",
}
MODEL_SEEDS = (0, 1, 2)
EVAL_SEEDS = (101, 202, 303)
POLICIES = ("noop", "random", "reference_replay", "jepa_open_loop", "jepa_receding")
BASELINES = POLICIES[:3]
FIXED_ARGS = {"seeds": list(EVAL_SEEDS), "envs": 128, "horizon": 6, "warmup": 8,
              "pop": 512, "elite": 32, "iters": 6, "reward": False, "no_goal": False,
              "steps": 100, "gamma": 0.99}
STRATA = {"all": lambda row: True,
          "displaced_goal": lambda row: row["goal_displacement_l1"] > 0,
          "goal_displacement_at_least_2": lambda row: row["goal_displacement_l1"] >= 2}
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20092026


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def check_metadata(meta, reference, seed, treatment, trainer_sha256):
    """Training-source inequality is intentional; all other recipe fields match."""
    ref = reference["model_metadata"]
    require(seed in MODEL_SEEDS, "unregistered model seed")
    require(meta["cfg"] == {**ref["cfg"], "seed": seed}, "training Config differs beyond model seed")
    require(meta["data_sha256"] == ref["data_sha256"] == DATA_SHA256, "training dataset differs")
    require(meta["params"] == ref["params"] == 11121556, "parameter count differs")
    expected_source = trainer_sha256 if treatment else BASE_SOURCE_SHA256
    require(meta["source_sha256"] == expected_source, "unexpected training implementation")
    if treatment:
        require(valid_sha(trainer_sha256) and trainer_sha256 != BASE_SOURCE_SHA256,
                "last-only requires a distinct, explicitly pinned wrapper")
        require(meta["base_train_source_sha256"] == BASE_SOURCE_SHA256, "base trainer differs")
        require(meta["objective_source_sha256"] == OBJECTIVE_SOURCE_SHA256, "last-only objective differs")
        require(meta["completed_steps"] == 8500 and meta["prediction_loss_mode"] == "last_only"
                and meta["target_gradient_mode"] == "joint", "wrong objective or incomplete training")
        require("decoder" not in meta and "heads" not in meta, "treatment has post-hoc auxiliaries")
        require(meta["sources"] == {**EVAL_SOURCES,
                    "studies/revision/craftax_last_only.py": trainer_sha256,
                    "studies/revision/last_only_objective.py": OBJECTIVE_SOURCE_SHA256},
                "last-only source inventory differs")
        require(meta["reference_report_sha256"] == REFERENCE_REPORT_SHA256
                and meta["reference_model_sha256"] == REFERENCE_MODEL_SHA256,
                "last-only reference provenance differs")
        for key in ("initial_weights_sha256", "window_table_sha256", "update_keys_sha256", "sampled_window_ids_sha256"):
            require(valid_sha(meta[key]), "missing training pairing fingerprint: " + key)


def check_report(report, reference, seed, treatment, trainer_sha256):
    check_metadata(report["model_metadata"], reference, seed, treatment, trainer_sha256)
    require(report["schema"] == reference["schema"] == "craftax-revision-control-v1", "unknown evaluator schema")
    require(report["source_sha256"] == reference["source_sha256"] == EVAL_SOURCES, "frozen evaluation source differs")
    require(report["versions"] == reference["versions"], "runtime package versions differ")
    require(report["protocol"] == reference["protocol"], "evaluation protocol differs")
    require(valid_sha(report["model_sha256"]), "invalid model identity")
    require(all(report["arguments"][key] == value for key, value in FIXED_ARGS.items()), "evaluation settings differ")
    require(not report["reward_runs"] and "goal_summary" in report, "goal evaluation incomplete or reward scope changed")
    require(len(report["goal_runs"]) == len(reference["goal_runs"]) == 3, "all three evaluation seeds required")
    for eval_seed, run, ref_run in zip(EVAL_SEEDS, report["goal_runs"], reference["goal_runs"]):
        for key in ("target_positions", "reference_actions", "start_positions"):
            require(run[key] == ref_run[key], "shared environment context differs: " + key)
        targets, starts = np.asarray(run["target_positions"]), np.asarray(run["start_positions"])
        require(targets.shape == starts.shape == (128, 2), "expected 128 position pairs per evaluation seed")
        displacement = np.abs(targets - starts).sum(axis=1)
        require(set(run["rows"]) == set(run["traces"]) == set(POLICIES), "controller set differs")
        for baseline in BASELINES:
            require(run["rows"][baseline] == ref_run["rows"][baseline]
                    and run["traces"][baseline] == ref_run["traces"][baseline],
                    "shared baseline outcomes or trajectories differ: " + baseline)
        for policy in POLICIES:
            rows, traces = run["rows"][policy], run["traces"][policy]
            require([(r["eval_seed"], r["episode"]) for r in rows] == [(eval_seed, i) for i in range(128)],
                    "duplicate, missing or reordered episode IDs")
            actions, positions, alive = (np.asarray(traces[key]) for key in ("actions", "positions", "alive"))
            require(actions.shape == alive.shape == (128, 6) and positions.shape == (128, 6, 2), "incomplete controller traces")
            require(np.issubdtype(actions.dtype, np.integer) and np.all((actions >= 0) & (actions < 17)), "non-native actions")
            require(np.issubdtype(positions.dtype, np.integer) and alive.dtype == np.bool_, "invalid position or survival traces")
            require(not np.any(alive[:, 1:] & ~alive[:, :-1]), "survival trajectory resurrects an episode")
            distances = np.abs(positions[:, -1] - targets).sum(axis=1)
            for i, row in enumerate(rows):
                for key in ("start_alive", "goal_alive", "goal_displacement_l1"):
                    require(row[key] == ref_run["rows"]["noop"][i][key], "episode stratum or start state differs")
                require(row["goal_displacement_l1"] == int(displacement[i]), "incorrect goal displacement")
                require(row["position_l1"] == int(distances[i]) and row["execution_alive"] == bool(alive[i, -1]),
                        "endpoint row disagrees with trajectory")
                require(row["reached_position"] == bool(distances[i] == 0)
                        and row["reached_position_alive"] == bool(distances[i] == 0 and alive[i, -1]),
                        "success label disagrees with endpoint")


def model_summary(values, binary=False):
    """Summarize matched model rows; never pool them as independent episodes."""
    values = np.asarray(values, dtype=float)
    require(values.ndim == 2 and values.shape[0] == 3 and values.shape[1] > 0, "expected three nonempty model rows")
    require(np.all(np.isfinite(values)), "nonfinite endpoint")
    means = values.mean(axis=1)
    result = {"model_mean": float(means.mean()), "model_sample_sd": float(means.std(ddof=1)),
              "per_model_means": means.tolist(), "episodes_per_model": int(values.shape[1])}
    if binary:
        require(np.all((values == 0) | (values == 1)), "success must be binary")
        result["per_model_successes"] = values.sum(axis=1).astype(int).tolist()
    return result


def paired_summary(treatment, original, indices):
    delta = np.asarray(treatment, dtype=float) - np.asarray(original, dtype=float)
    result = model_summary(delta)
    result["conditional_shared_episode_bootstrap_95"] = np.quantile(
        delta.mean(axis=0)[indices].mean(axis=1), [0.025, 0.975]).tolist()
    return result


def analyze(originals, treatments):
    """Pure aggregation; callers validate the six reports before entering."""
    rows = {name: {policy: [[row for run in report["goal_runs"] for row in run["rows"][policy]]
                           for report in reports] for policy in POLICIES}
            for name, reports in (("original", originals), ("last_only", treatments))}
    result = {"strata": {}}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for name, keep in STRATA.items():
        ids = [i for i, row in enumerate(rows["original"]["noop"][0]) if keep(row)]
        require(ids, "empty predeclared stratum: " + name)
        indices = rng.integers(0, len(ids), size=(BOOTSTRAP_REPLICATES, len(ids)), dtype=np.int32)
        section = {"n_shared_episode_contexts": len(ids), "policies": {}, "last_only_minus_original": {}}
        for policy in POLICIES:
            section["policies"][policy] = {}
            for endpoint, binary in (("reached_position_alive", True), ("position_l1", False)):
                outcomes = {objective: np.array([[group[i][endpoint] for i in ids] for group in rows[objective][policy]], dtype=float)
                            for objective in ("original", "last_only")}
                summary = {objective: model_summary(values, binary) for objective, values in outcomes.items()}
                for objective in summary:
                    summary[objective]["per_eval_seed"] = {}
                    for eval_seed in EVAL_SEEDS:
                        local_ids = [j for j, i in enumerate(ids) if rows[objective][policy][0][i]["eval_seed"] == eval_seed]
                        summary[objective]["per_eval_seed"][str(eval_seed)] = model_summary(outcomes[objective][:, local_ids], binary)
                    if policy in BASELINES:
                        summary[objective]["model_sample_sd"] = None
                        summary[objective]["note"] = "One shared nonlearned baseline; identical rows repeated for provenance only."
                        for entry in summary[objective]["per_eval_seed"].values():
                            entry["model_sample_sd"] = None
                            entry["note"] = summary[objective]["note"]
                section["policies"][policy][endpoint] = summary
                if policy not in BASELINES:
                    section["last_only_minus_original"].setdefault(policy, {})[endpoint] = paired_summary(
                        outcomes["last_only"], outcomes["original"], indices)
        result["strata"][name] = section
    result["primary"] = {"policy": "jepa_open_loop", "stratum": "all", "endpoint": "reached_position_alive",
                         "direction": "last_only minus original; positive favors last_only",
                         **result["strata"]["all"]["last_only_minus_original"]["jepa_open_loop"]["reached_position_alive"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-reports", type=Path, nargs=3, required=True)
    parser.add_argument("--last-only-reports", type=Path, nargs=3, required=True)
    parser.add_argument("--trainer-sha256", required=True, help="independently reviewed Craftax-only wrapper SHA256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "output exists; preserve it and use a fresh path")
    groups = {}
    for name, paths in (("original", args.original_reports), ("last_only", args.last_only_reports)):
        items = sorted([(path, json.loads(path.read_text())) for path in paths], key=lambda item: item[1]["model_metadata"]["cfg"]["seed"])
        require([report["model_metadata"]["cfg"]["seed"] for _, report in items] == list(MODEL_SEEDS), "exact seeds 0/1/2 required per objective")
        groups[name] = items
    reference_path, reference = groups["original"][0]
    require(sha256(reference_path) == REFERENCE_REPORT_SHA256, "original seed-0 report differs from frozen artifact")
    require(reference["model_sha256"] == REFERENCE_MODEL_SHA256, "original seed-0 model identity differs")
    for objective, items in groups.items():
        for seed, (_, report) in enumerate(items):
            check_report(report, reference, seed, objective == "last_only", args.trainer_sha256)
    originals = [report for _, report in groups["original"]]
    treatments = [report for _, report in groups["last_only"]]
    result = analyze(originals, treatments)
    result.update(schema="craftax-last-only-paired-six-model-v1", analysis_source_sha256=sha256(__file__),
        model_seeds=list(MODEL_SEEDS), eval_seeds=list(EVAL_SEEDS), independent_episode_contexts=384,
        model_episode_outcomes=2304, model_episode_outcomes_per_objective=1152,
        same_episode_contexts_across_models_and_objectives=True,
        training_seed_uncertainty="Descriptive mean and sample SD across three paired model-seed effects; no training-seed confidence interval.",
        episode_interval_scope="Paired shared-episode bootstrap conditional on these six fitted models; excludes training-seed uncertainty.",
        episode_bootstrap={"replicates": BOOTSTRAP_REPLICATES, "rng_seed": BOOTSTRAP_SEED,
                           "shared_columns_across_models_objectives_and_policies": True},
        secondary_endpoints="Displaced-goal and displacement>=2 success, receding-horizon success, and endpoint L1 (lower is better).",
        source_sha256=EVAL_SOURCES, versions=reference["versions"], training_data_sha256=DATA_SHA256,
        training_sources={"original": BASE_SOURCE_SHA256, "last_only_wrapper": args.trainer_sha256,
                          "last_only_objective": OBJECTIVE_SOURCE_SHA256,
                          "difference_expected": "Craftax-only wrapper selects last-target MSE; original base trainer and frozen evaluator unchanged."},
        integrity_checks="Exact matched-seed configs/data/parameter counts, expected training-source difference, frozen evaluator and package equality, all 384 contexts, baseline rows/traces, and learned-policy endpoint/trace consistency.",
        historical_pairing_scope="Original initialization/sampler pairing follows identical seed/base source; original runtime fingerprints were not recorded. Last-only fingerprints are measured, not evidence of a historical runtime comparison.",
        source_reports=[{"objective": objective, "model_seed": report["model_metadata"]["cfg"]["seed"],
                         "path": str(path.resolve()), "sha256": sha256(path), "model_sha256": report["model_sha256"],
                         "train_s": report["model_metadata"]["train_s"], "model_metadata": report["model_metadata"]}
                        for objective, items in groups.items() for path, report in items])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("stratum | policy | original success % (seeds 0/1/2) | last-only % | paired change pp [mean ± sample SD]")
    for name, section in result["strata"].items():
        for policy in POLICIES[3:]:
            success = section["policies"][policy]["reached_position_alive"]
            delta = section["last_only_minus_original"][policy]["reached_position_alive"]
            fmt = lambda values: "/".join(f"{100 * value:.2f}" for value in values)
            print(f"{name} (n={section['n_shared_episode_contexts']}) | {policy} | "
                  f"{fmt(success['original']['per_model_means'])} | {fmt(success['last_only']['per_model_means'])} | "
                  f"{100 * delta['model_mean']:+.2f} ± {100 * delta['model_sample_sd']:.2f}")
    print("384 shared contexts; 2304 model×episode outcomes. Bootstrap intervals in JSON are conditional on fitted models.")


if __name__ == "__main__":
    main()
