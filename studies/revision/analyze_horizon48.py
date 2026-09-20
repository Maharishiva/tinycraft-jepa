"""CPU-only artifact audit and analysis for the fixed, conditional H48 assay.

python -m studies.revision.analyze_horizon48 --self-test
python -m studies.revision.analyze_horizon48 --require-complete

Scientific analysis always uses all 3 x 3 x 4 planned runs at H48/U50.
``validate_run`` also accepts a shorter update count for separately authorized
timing fixtures; such fixtures can never enter ``analyze`` as scientific runs.
No training module is imported, and no checkpoint is unpickled.
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from studies.revision import analyze_matched as M


ARMS = ("learned", "oracle_heads", "oracle_reward", "oracle_true")
SEEDS = (0, 1, 2)
METRICS = ("walls_passed", "survival")
BASE_SOURCE = "studies/revision/matched_ppo.py"
TERMINATION_SOURCE = "studies/revision/matched_ppo_termination.py"
BASE_HASH = "6a3460b2db2e47713b7af604454a4b04066c96bfd7748b5c991cfbc433176269"
TERMINATION_HASH = "1c003dd4e019b2179981877310e131304d13a7150ac7da93e11d5164d011feb1"
SHARED_SOURCES = {
    BASE_SOURCE: BASE_HASH,
    "studies/stoch/rl.py": "76a163a3135bcc6a39bd4b3e9f7881123f72e2ad6e48bf5dd21c003eea9fd0fb",
    "studies/stoch/env.py": "bcceb9d5c660664ed2b0aab9ef30b9368b8a3db7a4fc9196bf5a5d6dc1d2bb69",
    "studies/redcube/env.py": "468212e57824968bbabb5363c897ea30492ce7dd53099add309af40ffe5e7b4e",
    "train.py": "72cd5c463b4ac2fe7e289f7032e91d88ffac2e59ebb15534524fafe4804e9d0b",
    "probe.py": "110eff77f3dfb5fb3c7222caa58d85773b9bc8675cd95859d4f9705c171ea424",
    "studies/revision/policy.py": "df66c5f29a359851b84cadfbc9c99d23797b4286fd0a61a2e47ac2eb15438dbd",
}
DATA_HASH = "c347aa2db1726bceadbc6a16fc2b4a7559343f988c4c2074e90c914b05354b55"
MODEL_HASHES = {
    0: "f81ef20ab6d6f032a85310fe9233e32749924839eb8bb55e4c0e6b1db2b880d2",
    1: "7190a8c231596043b62b6750f807e1c9fa542287dc161b7ca72182f2e68e625a",
    2: "8ab666aa5838fb81b11af99c5f700f8dd33373f757ef6109a07116df70f9f716",
}
FIXED_ARGS = dict(config="gate", torso="native_grid", batch=512, chunk=16,
                  mb=128, epochs=4, p_reset=0.5, gamma=0.99, lam=0.9,
                  lr=2.5e-4, clip=0.2, vf=0.5, ent=0.01,
                  eval_seed=20260919, eval_episodes=256, max_minutes=180.0)
ARTIFACTS = ("config.json", "final.json", "episodes.npz", "starts.npz", "ledger.jsonl", "model.pkl")
START_COLUMNS = ["context_start_t", "environment", "last_frame_t", "is_reset"]
TERMINAL_KEYS = ("true_terminal_fraction", "postterminal_alive_mass_fraction", "postterminal_reward_per_rollout")
CONTRASTS = {
    "exact_transitions_with_learned_heads": ("oracle_heads", "learned"),
    "true_termination_under_learned_reward": ("oracle_reward", "oracle_heads"),
    "true_reward_under_true_termination": ("oracle_true", "oracle_reward"),
    "complete_oracle_pipeline": ("oracle_true", "learned"),
}
LIMITATIONS = [
    "H48 was conditionally admitted after the audited H16 oracle_true discovery count was zero; this is an adaptive sensitivity, not independent confirmation or a preplanned primary result.",
    "H48 minus H16 changes a training package, not only GAE: start-context count, uninterrupted rollout length, collection-refresh delay and advantage normalization change while nominal positions and optimizer steps remain equal. Recurrent BPTT remains 16.",
    "Cross-horizon start prefixes and policy initialization constructors are matched; training action-noise keys and policy trajectories are not guaranteed identical.",
    "The exact-transition contrast includes absorbing terminal-support exposure. The termination contrast changes continuation, bootstrap and postterminal weighting together, and is not a pure reward intervention.",
    "Policy-seed intervals resample paired seeds within one fixed model and condition on the fixed evaluation set. With three policy seeds they are coarse descriptive intervals.",
    "Paired evaluation-slot intervals hold fitted policies fixed and resample shared slots jointly across policies; they do not measure training variability.",
    "All three fixed-model means, their equal-model descriptive mean and range are reported. No model-family confidence interval or pooled-nine-policy replication claim is justified.",
    "Intervals are unadjusted. No seed, arm, checkpoint, horizon or discovery-based subset is selected. A change does not identify a unique cause of H16 failure; failure of both packages bounds only these tested controller settings.",
    "The frozen learned collector does not record true-terminal/postterminal diagnostics; these remain unavailable, not zero. All arms record mean alive weight.",
]


def require(condition, message):
    M.require(condition, message)


def finite_tree(value, label):
    """Reject nonfinite numbers anywhere in recorded JSON, including metadata."""
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            finite_tree(child, f"{label}[{i}]")
    elif isinstance(value, (int, float)):
        require(np.isfinite(value), f"nonfinite {label}")


def read_json(path):
    value = json.loads(Path(path).read_text())
    finite_tree(value, str(path))
    return value


def normalized_sources(sources):
    """Only execution-root prefixes may differ; aliases/extra modules fail."""
    require(isinstance(sources, dict) and sources, "missing source hashes")
    allowed = set(SHARED_SOURCES) | {TERMINATION_SOURCE}
    result = {}
    for raw, digest in sources.items():
        name = str(raw).replace("\\", "/")
        matches = [key for key in allowed if name == key or name.endswith("/" + key)]
        require(len(matches) == 1, f"unapproved source entry: {raw}")
        key = matches[0]
        require(key not in result, f"duplicate normalized source entry: {key}")
        result[key] = digest
    return result


def scalar_metadata(archive, key, expected):
    value = np.asarray(archive[key])
    require(value.shape == () and np.issubdtype(value.dtype, np.integer)
            and value.item() == expected, f"episode {key} metadata differs")


def validate_run(directory, model_seed, policy_seed, arm, horizon=48, updates=50):
    """Strict per-run validator, also usable by the queue for timing fixtures.

    Returned raw arrays are for pairwise checks; remove ``episode_values`` and
    ``start_indices`` before JSON serialization. This validates saved policy
    bytes and metadata, not remote input files: the queue pins/rehashes those.
    """
    directory = Path(directory)
    require(model_seed in SEEDS and policy_seed in SEEDS and arm in ARMS, "unplanned model/policy/arm")
    require(type(updates) is int and updates > 0 and type(horizon) is int and horizon in (16, 48),
            "invalid validation budget")
    paths = {name: directory / name for name in ARTIFACTS}
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"missing or symlinked {name}")
    final, config = read_json(paths["final.json"]), read_json(paths["config.json"])
    args = config["args"]
    expected = {**FIXED_ARGS, "hor": horizon, "updates": updates,
                "model_seed": model_seed, "policy_seed": policy_seed, "arm": arm}
    require(set(args) == set(expected) | {"data", "model", "out"}, "unexpected or missing CLI arguments")
    for key, value in expected.items():
        require(type(args[key]) is not bool and args[key] == value, f"incorrect argument {key}: {args[key]}")
    for key, value in (("model_seed", model_seed), ("policy_seed", policy_seed)):
        require(final[key] == config[key] == value, f"{key} metadata differs")
    require(final["arm"] == arm and final["torso"] == "native_grid", "final arm/torso differs")
    require(final.get("budget_complete") is True, "incomplete requested budget")
    require(final["updates_completed"] == final["updates_requested"] == updates, "final update budget differs")
    positions = updates * 512 * horizon
    require(final["transitions"] == positions, "final nominal position budget differs")
    require(final["n_ep"] == 256 and final["eval_seed"] == 20260919, "final evaluation protocol differs")
    require(final["model_sha256"] == config["model_sha256"] == MODEL_HASHES[model_seed], "unapproved model hash")
    require(config["data_sha256"] == DATA_HASH, "unapproved data hash")
    cfg = config["model_cfg"]
    require(cfg["seed"] == model_seed and cfg["encoder"] == "tile" and cfg["tokens"] == 16
            and cfg["token_dim"] == 32 and cfg["history"] == 8, "model metadata/topology differs")
    require(config["latent_dim"] == 512 and config["policy_params"] == 583236,
            "latent dimension or native-grid policy size differs")
    require(isinstance(config["heads_cfg"], dict) and config["heads_cfg"], "missing frozen heads metadata")
    sources = normalized_sources(config["source_sha256"])
    expected_sources = dict(SHARED_SOURCES)
    if arm == "oracle_reward":
        expected_sources[TERMINATION_SOURCE] = TERMINATION_HASH
        require(config["parent_harness_sha256"] == BASE_HASH, "termination parent metadata differs")
        require(config["sensitivity_harness_sha256"] == TERMINATION_HASH, "termination harness metadata differs")
    else:
        require("parent_harness_sha256" not in config and "sensitivity_harness_sha256" not in config,
                "unexpected source-variant metadata in parent arm")
    require(sources == expected_sources, "source hashes/variant differ from exact approved whitelist")
    reconstruction = config["reconstruction"]
    require(reconstruction["frames_checked"] == reconstruction["exact_frames"] == 1024
            and reconstruction["seed"] == 173, "simulator reconstruction validation differs")
    require(final["train_seconds"] >= 0 and final["total_seconds"] >= final["train_seconds"], "invalid runtime metadata")
    with np.load(paths["episodes.npz"], allow_pickle=False) as archive:
        returns, lengths, walls = [np.asarray(archive[key], dtype=float) for key in ("returns", "lengths", "walls")]
        for key, value in (("model_seed", model_seed), ("policy_seed", policy_seed), ("eval_seed", 20260919)):
            scalar_metadata(archive, key, value)
    require(returns.shape == lengths.shape == walls.shape == (256,), "not 256 ordered evaluation slots")
    require(all(np.isfinite(a).all() for a in (returns, lengths, walls)), "nonfinite episode outcome")
    require(np.isin(returns, [-1., 0.]).all(), "unexpected gate return")
    require(((lengths >= 1) & (lengths <= 128) & (lengths == np.floor(lengths))).all(), "invalid episode lengths")
    require(((walls >= 0) & (walls == np.floor(walls))).all(), "invalid wall counts")
    survival = (returns > -.5).astype(float)
    require((lengths[survival == 1] == 128).all(), "survivor did not reach the episode cap")
    metrics = {"return": returns.mean(), "length": lengths.mean(), "walls_passed": walls.mean(),
               "survival": survival.mean(), "survival_sem": survival.std() / 16,
               "deaths_per_100": 100 * (1 - survival).sum() / lengths.sum()}
    for key, value in metrics.items():
        require(np.isclose(final[key], value, rtol=1e-6, atol=1e-7), f"final {key} contradicts raw episodes")
    with np.load(paths["starts.npz"], allow_pickle=False) as archive:
        starts, columns = np.asarray(archive["indices"]), archive["columns"].tolist()
    require(starts.shape == (updates, 512, 4) and np.issubdtype(starts.dtype, np.integer), "invalid start-index shape/type")
    require(np.isfinite(starts).all() and (starts >= 0).all(), "invalid start-index values")
    require(columns == START_COLUMNS, "invalid start-index columns")
    require(np.isin(starts[..., 3], [0, 1]).all(), "invalid reset flag")
    require(np.array_equal(starts[..., 2], starts[..., 0] + 7 * (1 - starts[..., 3])),
            "start history/reset semantics differ")
    ledger = [json.loads(line) for line in paths["ledger.jsonl"].read_text().splitlines() if line.strip()]
    finite_tree(ledger, "ledger")
    require([row["update"] for row in ledger] == list(range(1, updates + 1)), "incomplete or duplicated ledger updates")
    for row in ledger:
        require(row["transitions"] == row["update"] * 512 * horizon, "ledger position count differs")
        for key in ("seconds", "loss", "pg", "vf", "entropy", "reward", "done", "rollout_return", "alive"):
            require(key in row and isinstance(row[key], (int, float)) and np.isfinite(row[key]),
                    f"missing/nonfinite ledger {key}, update {row['update']}")
        require(row["seconds"] >= 0, "negative update time")
        for key in ("done", "alive"):
            require(-1e-6 <= row[key] <= 1 + 1e-6, f"invalid ledger {key}")
        if arm != "learned":
            for key in TERMINAL_KEYS:
                require(key in row and isinstance(row[key], (int, float)) and np.isfinite(row[key]),
                        f"missing/nonfinite terminal diagnostic {key}, update {row['update']}")
            for key in TERMINAL_KEYS[:2]:
                require(-1e-6 <= row[key] <= 1 + 1e-6, f"invalid terminal fraction {key}")
        if arm in ("oracle_reward", "oracle_true"):
            require(row["postterminal_alive_mass_fraction"] == row["postterminal_reward_per_rollout"] == 0.,
                    f"true termination has nonzero postterminal weighted mass/reward, update {row['update']}")
    require(M.file_hash(paths["model.pkl"]) == final["policy_sha256"], "saved final policy hash differs")
    diagnostics = {}
    for key in ("alive", *TERMINAL_KEYS):
        values = [row[key] for row in ledger if key in row]
        require(not values or len(values) == updates, f"partially recorded diagnostic {key}")
        diagnostics[key] = ({"mean_over_updates": float(np.mean(values)), "final_update": values[-1]}
                            if values else None)
    return {"directory": str(directory.resolve()), "model_seed": model_seed, "policy_seed": policy_seed,
            "arm": arm, "horizon": horizon, "updates": updates, "config": config, "final": final,
            "protocol_signature": {key: args[key] for key in (*FIXED_ARGS, "hor", "updates")},
            "source_identity": sources, "shared_source_identity": {key: sources[key] for key in SHARED_SOURCES},
            "start_indices_sha256": M.array_hash(starts), "start_prefix_sha256": M.array_hash(starts[:50]),
            "artifact_sha256": {name: M.file_hash(path) for name, path in paths.items()},
            "policy_checkpoint_locally_verified": True, "terminal_diagnostics": diagnostics,
            "optimizer_steps": updates * (horizon // 16) * 4 * 4,
            "episode_values": {"walls_passed": walls, "survival": survival}, "start_indices": starts}


def matching_errors(left, right, cross_horizon=False):
    """Initialization follows the identical pinned constructor and policy seed."""
    fields = {
        "model checkpoint": lambda r: r["final"]["model_sha256"],
        "data": lambda r: r["config"]["data_sha256"],
        "model config": lambda r: r["config"]["model_cfg"],
        "heads config": lambda r: r["config"]["heads_cfg"],
        "policy seed / initialization constructor": lambda r: (r["policy_seed"], r["config"]["policy_params"], r["shared_source_identity"]),
        "shared source hashes": lambda r: r["shared_source_identity"],
        "training/evaluation protocol": lambda r: {k: v for k, v in r["protocol_signature"].items()
                                                  if not cross_horizon or k not in ("hor", "updates")},
    }
    problems = [name + " differs" for name, getter in fields.items() if getter(left) != getter(right)]
    if cross_horizon:
        require(left["horizon"] == 48 and right["horizon"] == 16, "cross-horizon arguments reversed")
        if not np.array_equal(left["start_indices"], right["start_indices"][:50]):
            problems.append("H48 starts differ from the first 50 H16 batches")
    elif not np.array_equal(left["start_indices"], right["start_indices"]):
        problems.append("training starts differ across arms")
    return problems


def public_run(row):
    return {key: value for key, value in row.items() if key not in ("episode_values", "start_indices")}


def paired_contrast(pairs, label, model_seed, target, reference, draws=10000, seed=7341):
    result = {"target": target, "reference": reference, "n_paired_policy_seeds": len(pairs),
              "policy_seeds": [ps for ps, _, _ in pairs]}
    for metric in METRICS:
        delta = np.stack([a["episode_values"][metric] - b["episode_values"][metric] for _, a, b in pairs])
        policy_delta = delta.mean(axis=1)
        local_seed = M.stable_seed(seed, f"horizon48/{label}/{model_seed}/{metric}")
        stats = M.summarize_values(policy_delta)
        stats["paired_policy_bootstrap_95"] = M.bootstrap_mean(policy_delta, draws, local_seed)
        stats["fixed_policies_paired_episode_bootstrap_95"] = M.bootstrap_mean(delta.mean(axis=0), draws, local_seed + 1)
        stats["per_policy"] = [{"policy_seed": ps, "difference": float(policy_delta[j]),
                                "paired_episode_bootstrap_95": M.bootstrap_mean(delta[j], draws, local_seed + j + 2)}
                               for j, (ps, _, _) in enumerate(pairs)]
        result[metric] = stats
    return result


def descriptive(values):
    """Retain all three slots; never silently relabel a partial model average."""
    present = {str(ms): values.get(str(ms)) for ms in SEEDS}
    complete = all(v is not None for v in present.values())
    numbers = list(present.values()) if complete else []
    return {"per_model_mean": present, "n_fixed_models": sum(v is not None for v in present.values()),
            "equal_model_mean_descriptive": float(np.mean(numbers)) if complete else None,
            "model_mean_range_descriptive": [min(numbers), max(numbers)] if complete else None,
            "confidence_interval": None}


def load_matrix(root, arms, horizon, updates):
    root = Path(root)
    rows, inventory = {}, []
    expected_paths = set()
    for ms in SEEDS:
        for ps in SEEDS:
            for arm in arms:
                directory = root / f"m{ms}" / f"p{ps}" / arm
                expected_paths.add(directory / "final.json")
                entry = {"model_seed": ms, "policy_seed": ps, "arm": arm, "directory": str(directory.resolve())}
                try:
                    row = validate_run(directory, ms, ps, arm, horizon, updates)
                    rows[ms, ps, arm] = row
                    entry["status"] = "valid"
                except (ValueError, KeyError, OSError, TypeError, EOFError, zipfile.BadZipFile, OverflowError) as error:
                    entry.update(status="invalid" if (directory / "final.json").exists() else
                                 ("incomplete" if directory.exists() else "missing"), reason=str(error))
                inventory.append(entry)
    extra = [str(path.resolve()) for path in sorted(root.rglob("final.json")) if path not in expected_paths]
    return rows, inventory, extra


def analyze(runs_root, reference_root, termination_root, draws=10000, seed=7341):
    """The scientific matrix/budgets/seeds cannot be selected by CLI options."""
    high, inventory, extra = load_matrix(runs_root, ARMS, 48, 50)
    low, low_inventory, low_extra = load_matrix(reference_root, ("learned", "oracle_heads", "oracle_true"), 16, 150)
    termination, term_inventory, term_extra = load_matrix(termination_root, ("oracle_reward",), 16, 150)
    low.update(termination)
    mismatches, selected, models, blocks_by_model = [], [], {}, {}
    for ms in SEEDS:
        blocks = []
        for ps in SEEDS:
            block = {arm: high[ms, ps, arm] for arm in ARMS if (ms, ps, arm) in high}
            if len(block) != 4:
                continue
            problems = [f"{arm}: {problem}" for arm in ARMS[1:]
                        for problem in matching_errors(block["learned"], block[arm])]
            if problems:
                mismatches.append({"model_seed": ms, "policy_seed": ps, "reasons": problems})
            else:
                blocks.append((ps, block))
        # A model/head metadata change across policies invalidates this fixed-model stratum.
        if blocks:
            first = blocks[0][1]["learned"]["config"]
            problems = [key + " changes across policy seeds" for key in ("model_cfg", "heads_cfg")
                        if any(block["learned"]["config"][key] != first[key] for _, block in blocks)]
            if problems:
                mismatches.append({"model_seed": ms, "reasons": problems})
                blocks = []
        blocks_by_model[ms] = blocks
        model = {"status": "complete" if len(blocks) == 3 else "incomplete", "policy_seeds": [ps for ps, _ in blocks],
                 "model_sha256": MODEL_HASHES[ms], "arms": {}, "contrasts": {}}
        selected.extend(row for _, block in blocks for row in block.values())
        for arm in ARMS if blocks else ():
            rows = [block[arm] for _, block in blocks]
            summaries = {metric: M.summarize_values([row["final"][metric] for row in rows]) for metric in METRICS}
            summaries["per_policy"] = [{"policy_seed": row["policy_seed"], **{metric: row["final"][metric] for metric in METRICS},
                                        "discovery": row["final"]["survival"] > .5} for row in rows]
            summaries["discovery_count"] = sum(row["final"]["survival"] > .5 for row in rows)
            summaries["training_seconds"] = M.summarize_values([row["final"]["train_seconds"] for row in rows])
            summaries["terminal_diagnostics"] = {
                key: M.summarize_values([row["terminal_diagnostics"][key]["mean_over_updates"] for row in rows])
                if all(row["terminal_diagnostics"][key] is not None for row in rows) else None
                for key in ("alive", *TERMINAL_KEYS)}
            model["arms"][arm] = summaries
        for name, (target, reference) in CONTRASTS.items() if blocks else ():
            pairs = [(ps, block[target], block[reference]) for ps, block in blocks]
            model["contrasts"][name] = paired_contrast(pairs, name, ms, target, reference, draws, seed)
        models[str(ms)] = model
    cross_horizon, cross_errors = {}, []
    for arm in ARMS:
        arm_models = {}
        for ms in SEEDS:
            pairs = []
            for ps, block in blocks_by_model[ms]:
                if (ms, ps, arm) not in low:
                    continue
                problems = matching_errors(block[arm], low[ms, ps, arm], cross_horizon=True)
                if problems:
                    cross_errors.append({"model_seed": ms, "policy_seed": ps, "arm": arm, "reasons": problems})
                else:
                    pairs.append((ps, block[arm], low[ms, ps, arm]))
            arm_models[str(ms)] = {"status": "complete" if len(pairs) == 3 else "unresolved",
                                   "policy_seeds": [ps for ps, _, _ in pairs],
                                   "contrast": paired_contrast(pairs, "H48_minus_H16/" + arm, ms, "H48", "H16", draws, seed)
                                   if pairs else None}
        cross_horizon[arm] = {"status": "complete" if all(m["status"] == "complete" for m in arm_models.values()) else "unresolved",
                              "models": arm_models,
                              "cross_model_descriptive": {metric: descriptive({ms: model["contrast"][metric]["mean"]
                                  for ms, model in arm_models.items() if model["status"] == "complete"}) for metric in METRICS}}
    cross_model = {
        "arms": {arm: {metric: descriptive({ms: model["arms"][arm][metric]["mean"]
                   for ms, model in models.items() if model["status"] == "complete"}) for metric in METRICS} for arm in ARMS},
        "contrasts": {name: {metric: descriptive({ms: model["contrasts"][name][metric]["mean"]
                        for ms, model in models.items() if model["status"] == "complete"}) for metric in METRICS} for name in CONTRASTS},
    }
    complete = len(selected) == 36 and not mismatches and not extra
    cross_complete = all(arm["status"] == "complete" for arm in cross_horizon.values()) and not low_extra and not term_extra
    return {"schema_version": 1, "status": "complete" if complete else "incomplete", "publication_ready": complete and cross_complete,
            "cross_horizon_status": "complete" if cross_complete else "unresolved", "planned_runs": 36,
            "n_individually_valid_runs": len(high), "n_validated_runs_in_complete_blocks": len(selected),
            "planned_model_seeds": list(SEEDS), "planned_policy_seeds": list(SEEDS), "planned_arms": list(ARMS),
            "primary_endpoints": list(METRICS), "discovery_definition_secondary": "survival strictly greater than 0.5",
            "budget": {"H48": dict(horizon=48, updates=50, nominal_positions=1228800, optimizer_steps=2400, starts=25600),
                       "H16": dict(horizon=16, updates=150, nominal_positions=1228800, optimizer_steps=2400, starts=76800)},
            "bootstrap": {"draws": draws, "seed": seed, "interval": "95% percentile; unadjusted", "policy_unit": "within each fixed model"},
            "runs_root": str(Path(runs_root).resolve()), "reference_root": str(Path(reference_root).resolve()),
            "termination_reference_root": str(Path(termination_root).resolve()), "planned_run_inventory": inventory,
            "reference_run_inventory": low_inventory + term_inventory, "pairing_mismatches": mismatches,
            "cross_horizon_mismatches": cross_errors, "unplanned_runs_excluded": extra,
            "unplanned_references_excluded": low_extra + term_extra,
            "validated_runs": [public_run(row) for row in selected],
            "individually_valid_runs": [public_run(row) for row in high.values()],
            "validated_reference_runs": [public_run(row) for row in low.values()],
            "model_results": models, "cross_model_descriptive": cross_model, "cross_horizon": cross_horizon,
            "selection": "All 36 planned runs retained in inventory; only complete validated four-arm H48 blocks enter provisional contrasts. Final H48 reporting requires every block.",
            "limitations": LIMITATIONS}


def render(result):
    lines = ["# Conditional H48 training-package sensitivity", "", f"H48 status: **{result['status']}**; {result['n_validated_runs_in_complete_blocks']}/36 runs in validated four-arm blocks. Cross-H status: **{result['cross_horizon_status']}**.", "",
             "Walls/episode and survival are primary. Discovery (>50% survival) is secondary. Incomplete results are provisional.", "",
             "| Model | Arm | Policy seeds | Walls/episode | Survival (%) | Discovery |",
             "|---|---|---|---:|---:|---:|"]
    for ms, model in result["model_results"].items():
        for arm, values in model["arms"].items():
            lines.append(f"| {ms} | {arm} | {model['policy_seeds']} | {values['walls_passed']['mean']:.4f} | {100*values['survival']['mean']:.2f} | {values['discovery_count']}/{len(model['policy_seeds'])} |")
    lines += ["", "All differences below are target minus reference. Intervals resample paired policy seeds within a fixed model.", "",
              "| Comparison | Model | Paired seeds | Delta walls [95%] | Delta survival, pp [95%] |", "|---|---|---:|---:|---:|"]
    def add(label, ms, contrast):
        def cell(metric, scale):
            value = contrast[metric]; ci = value["paired_policy_bootstrap_95"]
            return f"{value['mean']*scale:+.4f}" + (f" [{ci[0]*scale:+.4f}, {ci[1]*scale:+.4f}]" if ci else " [unavailable]")
        lines.append(f"| {label} | {ms} | {contrast['n_paired_policy_seeds']} | {cell('walls_passed',1)} | {cell('survival',100)} |")
    for ms, model in result["model_results"].items():
        for contrast in model["contrasts"].values():
            add(contrast["target"] + " − " + contrast["reference"], ms, contrast)
    for arm, data in result["cross_horizon"].items():
        for ms, model in data["models"].items():
            if model["contrast"]:
                add(arm + ": H48 − H16", ms, model["contrast"])
    lines += ["", "Equal-model means, model ranges, individual policy outcomes, paired evaluation-slot intervals, terminal diagnostics and every missing/invalid run are retained in publication.json.", "", *["- " + text for text in LIMITATIONS], ""]
    return "\n".join(lines)


def self_test():
    """Real-shape synthetic artifacts with scientific and integrity failures."""
    def fixture(root, ms, ps, arm, horizon, updates):
        directory = root / f"m{ms}" / f"p{ps}" / arm
        directory.mkdir(parents=True)
        args = dict(FIXED_ARGS, arm=arm, model_seed=ms, policy_seed=ps, hor=horizon, updates=updates,
                    data="/frozen/train.npz", model="/frozen/model_probe.npz", out=str(directory))
        sources = {"/execution/root/" + key: value for key, value in SHARED_SOURCES.items()}
        config = dict(args=args, model_seed=ms, policy_seed=ps, data_sha256=DATA_HASH, model_sha256=MODEL_HASHES[ms],
                      source_sha256=sources, model_cfg=dict(seed=ms, encoder="tile", tokens=16, token_dim=32, history=8),
                      heads_cfg=dict(frozen=True), latent_dim=512, policy_params=583236,
                      reconstruction=dict(frames_checked=1024, exact_frames=1024, seed=173))
        if arm == "oracle_reward":
            sources["/execution/root/" + TERMINATION_SOURCE] = TERMINATION_HASH
            config.update(parent_harness_sha256=BASE_HASH, sensitivity_harness_sha256=TERMINATION_HASH)
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "model.pkl").write_bytes(f"synthetic-policy-{ms}-{ps}-{arm}-{horizon}".encode())
        # At the 128-step cap, the first 128 episodes are deaths, not survivors.
        returns = np.r_[np.full(128, -1.), np.zeros(128)]
        lengths = np.full(256, 128)
        wall = [0, 2, 4][ps] if arm == "learned" else 2 + ARMS.index(arm) - 1
        walls = np.full(256, wall + (1 if horizon == 48 else 0))
        final = dict(arm=arm, torso="native_grid", model_seed=ms, policy_seed=ps, eval_seed=20260919,
                     updates_completed=updates, updates_requested=updates, budget_complete=True,
                     transitions=updates*512*horizon, train_seconds=1., total_seconds=2., n_ep=256,
                     model_sha256=MODEL_HASHES[ms], policy_sha256=M.file_hash(directory / "model.pkl"),
                     length=128., walls_passed=float(walls.mean()), survival=.5, survival_sem=.5/16,
                     deaths_per_100=100*128/lengths.sum())
        final["return"] = -.5
        (directory / "final.json").write_text(json.dumps(final))
        np.savez(directory / "episodes.npz", returns=returns, lengths=lengths, walls=walls,
                 model_seed=ms, policy_seed=ps, eval_seed=20260919)
        starts = np.zeros((updates, 512, 4), np.int32)
        starts[..., 0] = np.arange(updates)[:, None] + ps
        starts[..., 1] = np.arange(512) % 4
        starts[..., 3] = np.arange(512) % 2
        starts[..., 2] = starts[..., 0] + 7 * (1 - starts[..., 3])
        np.savez(directory / "starts.npz", indices=starts, columns=START_COLUMNS)
        ledger = []
        for update in range(1, updates + 1):
            row = dict(update=update, transitions=update*512*horizon, seconds=1., loss=0., pg=0., vf=0.,
                       entropy=1., reward=0., done=0., rollout_return=0., alive=1.)
            if arm != "learned":
                row.update(true_terminal_fraction=.2, postterminal_alive_mass_fraction=0., postterminal_reward_per_rollout=0.)
            ledger.append(row)
        (directory / "ledger.jsonl").write_text("\n".join(map(json.dumps, ledger)) + "\n")
        return directory

    with tempfile.TemporaryDirectory(prefix="horizon48-cpu-test-") as tmp:
        root = Path(tmp); high, low, term = [root / name for name in ("h48", "h16", "term")]
        for ms in SEEDS:
            for ps in SEEDS:
                for arm in ARMS:
                    fixture(high, ms, ps, arm, 48, 50)
                    fixture(term if arm == "oracle_reward" else low, ms, ps, arm, 16, 150)
        evaluate = lambda: analyze(high, low, term, draws=200, seed=7341)
        result = evaluate()
        require(result["status"] == "complete" and result["publication_ready"], "valid 36-run design rejected")
        delta = result["model_results"]["0"]["contrasts"]["exact_transitions_with_learned_heads"]["walls_passed"]
        require(delta["values"] == [2., 0., -2.] and delta["mean"] == 0., "policy selection or unpaired effect")
        require(result["model_results"]["0"]["arms"]["learned"]["discovery_count"] == 0,
                "50% survival incorrectly called discovery")
        require(result["cross_horizon"]["learned"]["models"]["0"]["contrast"]["walls_passed"]["values"] == [1., 1., 1.],
                "cross-horizon sign/pairing wrong")
        require(result["cross_horizon"]["learned"]["models"]["0"]["contrast"]["walls_passed"]["paired_policy_bootstrap_95"] == [1., 1.],
                "constant cross-horizon effect interval wrong")
        require(result["model_results"]["0"]["arms"]["learned"]["survival"]["mean"] == .5, "death at cap counted as surviving")
        require(result["model_results"]["0"]["arms"]["learned"]["terminal_diagnostics"]["true_terminal_fraction"] is None,
                "unavailable learned diagnostic silently replaced")
        require(result["cross_model_descriptive"]["arms"]["learned"]["survival"]["confidence_interval"] is None,
                "model-family CI invented")
        directory = high / "m0/p0/oracle_reward"
        def corrupt_json(name, mutate, reason):
            path = directory / name; original = path.read_text(); value = json.loads(original); mutate(value)
            path.write_text(json.dumps(value))
            try:
                validate_run(directory, 0, 0, "oracle_reward")
            except (ValueError, KeyError) as error:
                require(reason in str(error), f"wrong rejection for {reason}: {error}")
            else:
                raise AssertionError("accepted corruption: " + reason)
            finally:
                path.write_text(original)
        corrupt_json("config.json", lambda c: c.update(parent_harness_sha256="0"*64), "parent metadata")
        corrupt_json("config.json", lambda c: c["source_sha256"].update({"/bad/extra.py": "0"*64}), "unapproved source")
        corrupt_json("config.json", lambda c: c["args"].update(chunk=48), "incorrect argument chunk")
        corrupt_json("config.json", lambda c: c.update(model_sha256="0"*64), "model hash")
        corrupt_json("final.json", lambda f: f.update(survival=1.), "contradicts raw episodes")
        corrupt_json("final.json", lambda f: f.update(budget_complete=False), "incomplete requested budget")
        corrupt_json("final.json", lambda f: f.update(updates_completed=49), "update budget")
        ledger_path = directory / "ledger.jsonl"; original = ledger_path.read_text()
        for key, value, reason in (("loss", float("nan"), "nonfinite"),
                                   ("transitions", 5, "position count"),
                                   ("postterminal_reward_per_rollout", .01, "nonzero postterminal")):
            rows = [json.loads(line) for line in original.splitlines()]; rows[2][key] = value
            ledger_path.write_text("\n".join(map(json.dumps, rows)))
            try:
                validate_run(directory, 0, 0, "oracle_reward")
            except ValueError as error:
                require(reason in str(error), "wrong ledger rejection")
            else:
                raise AssertionError("accepted ledger corruption")
        ledger_path.write_text(original)
        # Equal-shaped semantic starts mismatch: exclude the entire four-arm block.
        path = directory / "starts.npz"
        with np.load(path) as archive:
            starts = archive["indices"].copy()
        shifted = starts.copy(); shifted[..., (0, 2)] += 1
        np.savez(path, indices=shifted, columns=START_COLUMNS)
        bad = evaluate()
        require(bad["status"] == "incomplete" and bad["n_validated_runs_in_complete_blocks"] == 32
                and bad["model_results"]["0"]["policy_seeds"] == [1, 2], "mismatched block not excluded symmetrically")
        np.savez(path, indices=starts, columns=START_COLUMNS)
        # Later H16 batches may differ; only the 50-batch prefix is paired.
        path = low / "m0/p0/learned/starts.npz"
        with np.load(path) as archive:
            starts = archive["indices"].copy()
        starts[50:, :, (0, 2)] += 10
        np.savez(path, indices=starts, columns=START_COLUMNS)
        require(evaluate()["publication_ready"], "H16 tail incorrectly required identical")
        starts[0, :, (0, 2)] += 1
        np.savez(path, indices=starts, columns=START_COLUMNS)
        bad = evaluate()
        require(bad["status"] == "complete" and not bad["publication_ready"]
                and bad["cross_horizon"]["learned"]["status"] == "unresolved", "cross-H prefix mismatch accepted")
        # Saved policy must exist and match its hash, including for references.
        policy = directory / "model.pkl"; original = policy.read_bytes(); policy.write_bytes(b"changed")
        try:
            validate_run(directory, 0, 0, "oracle_reward")
        except ValueError as error:
            require("policy hash" in str(error), "wrong policy rejection")
        else:
            raise AssertionError("changed policy accepted")
        policy.write_bytes(original)
        # Timing fixtures are valid only under their explicit shorter budget.
        smoke = fixture(root / "timing", 0, 0, "oracle_true", 48, 2)
        require(validate_run(smoke, 0, 0, "oracle_true", updates=2)["optimizer_steps"] == 96, "smoke validator fails")
        try:
            validate_run(smoke, 0, 0, "oracle_true")
        except ValueError:
            pass
        else:
            raise AssertionError("timing fixture accepted as scientific run")
        # Shared evaluation slots must be resampled jointly, not as 512 iid rows.
        a, b = high / "m0/p0/learned", high / "m0/p1/learned"
        one, two = validate_run(a, 0, 0, "learned"), validate_run(b, 0, 1, "learned")
        zero = copy.deepcopy(one)
        for metric in METRICS:
            one["episode_values"][metric] = np.tile([1., -1.], 128)
            two["episode_values"][metric] = -one["episode_values"][metric]
            zero["episode_values"][metric] = np.zeros(256)
        contrast = paired_contrast([(0, one, zero), (1, two, zero)], "fixture", 0, "a", "b", 10000, 7341)
        require(contrast["walls_passed"]["fixed_policies_paired_episode_bootstrap_95"] == [0., 0.],
                "shared evaluation slots not resampled jointly")
        require(M.bootstrap_mean([5.], 100, 1) is None, "single policy falsely has seed uncertainty")
    print("PASS: 36-run matrix, all four contrasts, fixed-model summaries, paired slots, cap death, strict discovery, exact source/input lineage, ledger integrity, symmetric block exclusion, cross-H prefix-only matching, saved-policy hashes, and isolated smoke budget.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/revision/horizon48"))
    parser.add_argument("--reference-root", type=Path, default=Path("runs/revision/matched_grid"))
    parser.add_argument("--termination-root", type=Path, default=Path("runs/revision/termination-native-grid"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/revision/horizon48-analysis"))
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = analyze(args.runs_root, args.reference_root, args.termination_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "publication.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "table.md").write_text(render(result))
    print(json.dumps({key: result[key] for key in ("status", "publication_ready", "cross_horizon_status", "n_validated_runs_in_complete_blocks")}))
    if args.require_complete and not result["publication_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
