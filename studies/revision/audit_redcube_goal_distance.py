"""Independent NumPy-only audit of the secondary goal-distance response study.

Imports only independent audit helpers, never the experiment or environment.
The original primary evaluation and its auditor are left unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from studies.revision.audit_redcube_goal import (
    Check, MOVES, POLICIES, ROOT, array_digest, compare_summary, digest,
    infer_safe_positions, proportion_record, red_mask, rendered_mask,
)

SHELLS = (1, 2, 3, 6, 9, 12)


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def reconstruct_policy(check, name, episodes, starts, goals):
    n = len(starts)
    actions = episodes[name + "_actions"]
    stored_positions = episodes[name + "_positions"]
    check(name + " fixed trajectory shapes", actions.shape == (n, 24) and stored_positions.shape == (n, 25, 2))
    check(name + " initial positions", stored_positions[:, 0] == starts)
    p = starts.copy()
    alive, hit = np.ones(n, bool), np.zeros(n, bool)
    first = np.full(n, -1)
    initial_distance = np.abs(starts - goals).sum(1)
    minimum = initial_distance.copy()
    alive_history, hit_history, distances = [alive.copy()], [hit.copy()], [initial_distance.copy()]
    executed_transitions = 0
    for t in range(24):
        active = alive & ~hit
        check(name + f" native action support t={t}", ((actions[:, t] >= 0) & (actions[:, t] < 4)) == active)
        check(name + f" inactive action sentinel t={t}", actions[~active, t] == -1)
        new = p.copy()
        new[active] += MOVES[actions[active, t]]
        check(name + f" exact native transition t={t}", stored_positions[:, t + 1] == new)
        executed_transitions += int(active.sum())
        died = active & (((new < 2) | (new > 55)).any(1) | (8 + t >= 128))
        alive[died] = False
        distance = np.abs(new - goals).sum(1)
        new_hit = active & alive & (distance == 0)
        first[new_hit] = t + 1
        hit |= new_hit
        minimum = np.minimum(minimum, np.where(alive, distance, minimum))
        alive_history.append(alive.copy())
        hit_history.append(hit.copy())
        distances.append(distance)
        p = new
    alive_history, hit_history, distances = [np.stack(x, 1) for x in (alive_history, hit_history, distances)]
    for field, expected in (("alive", alive_history), ("hit", hit_history), ("distance_l1", distances),
                            ("first_hit", first), ("minimum_distance_l1", minimum)):
        check(name + " reconstructed " + field, episodes[name + "_" + field] == expected)
    if name.startswith("jepa"):
        costs = episodes[name + "_selected_cost"]
        check(name + " finite nonnegative selected costs", costs.shape == (n, 24) and np.all(np.isfinite(costs) & (costs >= 0)))
    if name == "fixed_cycle":
        cycle = np.broadcast_to(np.resize(np.array([0, 2, 1, 3]), 24), actions.shape)
        check("fixed blind cycle exactly preserved", actions[actions >= 0] == cycle[actions >= 0])
    if name == "oracle":
        check("oracle reaches every shell goal at exact shortest-path step", first == initial_distance // 3)
    return {"alive": alive_history, "hit": hit_history, "distances": distances, "first": first,
            "minimum": minimum, "executed_transitions": executed_transitions}


def summarize(replay, mask):
    hit = replay["hit"][mask, -1]
    first = replay["first"][mask]
    minimum = replay["minimum"][mask]
    return {"exact_goal_hit": proportion_record(hit), "within_one_action_l1_3": proportion_record(minimum <= 3),
            "died_before_goal": proportion_record(~replay["alive"][mask, -1]),
            "mean_minimum_distance_l1": float(minimum.mean()),
            "mean_final_distance_l1": float(replay["distances"][mask, -1].mean()),
            "mean_first_hit_steps_given_success": float(first[hit].mean()) if hit.any() else None}


def audit_prefixes(check, data):
    frames, actions, starts, goals = data["frames"], data["warmup_actions"], data["state_pos"], data["goals"]
    n = len(starts)
    check("six full shells of 128 cases", n == 768 and frames.shape == (768, 8, 63, 63, 3) and actions.shape == (768, 7))
    check("all warmup actions native", (actions >= 0) & (actions < 4))
    recovered = infer_safe_positions(frames[:128], check)
    check("all prefix states safe", (recovered >= 2) & (recovered <= 55))
    check("all prefix resets on native support", (recovered[:, 0] >= 16) & (recovered[:, 0] <= 42))
    check("all warmup RGB positions follow native actions", np.diff(recovered, axis=1) == MOVES[actions[:128]])
    check("prefix endpoint equals recorded start", recovered[:, -1] == starts[:128])
    for shell_index in range(1, 6):
        sl = slice(shell_index * 128, (shell_index + 1) * 128)
        for key in ("frames", "warmup_actions", "candidate_ids", "state_pos", "state_wind", "state_step", "state_episode", "state_texture"):
            check(f"identical starts/actions/IDs across shells {shell_index}: {key}", data[key][sl] == data[key][:128])
    check("deterministic zero wind", data["state_wind"] == 0)
    check("prefix clock seven", data["state_step"] == 7)
    check("prefix episode zero", data["state_episode"] == 0)
    check("safe goal support", (goals >= 2) & (goals <= 55))
    check("native goal lattice", (goals - starts) % 3 == 0)
    shells = np.repeat(np.array(SHELLS), 128)
    check("exact prespecified goal distances", np.abs(goals - starts).sum(1) == shells * 3)
    check("RGB goals exact square geometry", red_mask(data["goal_rgb"]) == rendered_mask(goals))
    # Compare the common, unoccluded background across all six goals per case.
    goals_rgb = data["goal_rgb"].reshape(6, 128, 63, 63, 3)
    for i in range(6):
        for j in range(i + 1, 6):
            left, right = goals_rgb[i], goals_rgb[j]
            visible = ~red_mask(left) & ~red_mask(right)
            check(f"same goal nuisance realization across shells {SHELLS[i]}/{SHELLS[j]}", left[visible] == right[visible])
    return shells


def validate_visual_packet(check, visual, data, episodes, indices):
    check("visual row selection was fixed", visual["episode_indices"] == indices)
    check("visual selected candidate IDs", visual["candidate_ids"] == data["candidate_ids"][indices])
    check("visual selected goal positions", visual["goal_positions"] == data["goals"][indices])
    check("visual selected goal RGB", visual["goals_rgb"] == data["goal_rgb"][indices])
    for name in POLICIES:
        for field in ("actions", "positions", "alive", "hit"):
            check(name + " visual " + field, visual[name + "_" + field] == episodes[name + "_" + field][indices])
        actions = episodes[name + "_actions"][indices]
        pixels = visual[name + "_rgb"]
        check(name + " first visual RGB equals real prefix", pixels[:, 0] == data["frames"][indices, -1])
        check(name + " rendered geometry exactly matches recorded native positions", red_mask(pixels) == rendered_mask(episodes[name + "_positions"][indices]))
        check(name + " visual forecast-valid mask", visual[name + "_forecast_valid"] == (actions >= 0))
        for row in range(len(indices)):
            stop = np.flatnonzero(actions[row] < 0)
            if len(stop):
                t = int(stop[0])
                check(name + f" terminal RGB frozen row={row}", pixels[row, t:] == pixels[row, t])
        if name + "_selected_plan" in visual:
            valid = actions >= 0
            check(name + " visual first planned action equals execution", visual[name + "_selected_plan"][..., 0][valid] == actions[valid])


def declared_array_digest(array):
    header = str(array.dtype).encode() + repr(array.shape).encode()
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()


def audit(run):
    check = Check()
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    data, episodes = archive(run / "cohort.npz"), archive(run / "episodes.npz")
    direct, rendered = archive(run / "visual_packet.npz"), archive(run / "range_visual_packet.npz")
    check("standalone and embedded manifests identical", manifest == summary["manifest"])
    check("post-outcome scope explicitly declared", manifest["classification"] == "post-outcome operating-range diagnostic")
    check("all six shells declared", manifest["distances_actions"] == list(SHELLS))
    check("separate frozen seeds", (manifest["prefix_seed"], manifest["evaluation_seed"], manifest["goal_seed"]) == (20260920, 20260920, 20260921))
    check("128 unique cases per shell", manifest["n_unique_prefixes"] == manifest["n_per_distance"] == 128)
    check("768 rows explicitly correlated", manifest["n_total_correlated_rows"] == 768)
    check("unchanged control and encoder configuration", manifest["control_steps"] == 24 and manifest["horizon"] == 3 and manifest["encoder_batch"] == [16, 8])
    check("unchanged native representation", manifest["latent_dim"] == 512 and manifest["history"] == 8)
    for filename, expected in summary["artifacts"].items():
        check("artifact SHA256 " + filename, digest(run / filename) == expected)
    check("cohort manifest SHA256", digest(run / "cohort.npz") == manifest["cohort_sha256"])
    for relative, expected in manifest["source_sha256"].items():
        check("launch source SHA256 " + relative, digest(ROOT / relative) == expected)
    for key, expected in manifest["cohort_array_sha256"].items():
        check("declared cohort array digest " + key, declared_array_digest(data[key]) == expected)

    model_seed = manifest["model_metadata"]["cfg"]["seed"]
    primary = ROOT / "runs/revision/redcube_goal/gray_frame_still" / f"m{model_seed}"
    primary_manifest = json.loads((primary / "manifest.json").read_text())
    primary_data = archive(primary / "cohort.npz")
    check("primary model identity unchanged", manifest["model_sha256"] == primary_manifest["model_sha256"])
    check("parent cohort identity", manifest["parent_primary_cohort_sha256"] == digest(primary / "cohort.npz"))
    check("parent manifest identity", manifest["parent_primary_manifest_sha256"] == digest(primary / "manifest.json"))
    for field in ("frames", "warmup_actions", "candidate_ids", "state_pos", "state_wind", "state_texture", "state_step", "state_episode"):
        check("original primary prefix exactly preserved: " + field, data[field][:128] == primary_data[field])
    check("native accepted-prefix rejection accounting", int(data["candidate_ids"][127]) + 1 == 128 + manifest["prefix_rejected_before_last_admission"])
    check("original unique candidate IDs sorted", np.diff(data["candidate_ids"][:128]) > 0)
    shells = audit_prefixes(check, data)
    check("explicit shell labels", data["distance_actions"] == shells)
    check("explicit within-shell case IDs", data["within_shell_case_index"] == np.tile(np.arange(128), 6))
    check("goal offsets exact", data["goal_offsets"] == data["goals"] - data["state_pos"])
    rng = np.random.default_rng(manifest["goal_seed"])
    independent_goals = []
    for position, distance in zip(data["state_pos"], shells):
        candidates = [(x, y) for x in range(2, 56) for y in range(2, 56)
                      if (x - int(position[0])) % 3 == 0 and (y - int(position[1])) % 3 == 0
                      and abs(x - int(position[0])) + abs(y - int(position[1])) == 3 * distance]
        independent_goals.append(candidates[int(rng.integers(len(candidates)))])
    check("all exact-shell goal draws independently reproduced", data["goals"] == np.array(independent_goals))

    metrics, total_transitions = {}, 0
    for name in POLICIES:
        replay = reconstruct_policy(check, name, episodes, data["state_pos"], data["goals"])
        total_transitions += replay["executed_transitions"]
        for distance in SHELLS:
            expected = summarize(replay, shells == distance)
            compare_summary(check, f"distance={distance} {name} summary", expected, summary["by_distance_actions"][str(distance)][name])
            metrics.setdefault(str(distance), {})[name] = expected
        if name in ("random", "fixed_cycle"):
            a = episodes[name + "_actions"].reshape(6, 128, 24)
            for i in range(6):
                for j in range(i + 1, 6):
                    both = (a[i] >= 0) & (a[j] >= 0)
                    check(f"goal-blind action stream paired across shells {name} {i}/{j}", a[i][both] == a[j][both])
    check("distance-one within-action secondary is explicitly trivial", all(metrics["1"][name]["within_one_action_l1_3"]["count"] == 128 for name in POLICIES))
    validate_visual_packet(check, direct, data, episodes, np.arange(3))
    chosen = np.arange(6) * 128
    check("range visuals prospectively fixed", manifest["visual_rows_predeclared"] == chosen.tolist())
    check("one fixed case zero per shell", rendered["within_shell_case_indices"] == 0)
    check("range visual shell labels", rendered["distance_actions"] == np.array(SHELLS))
    normalized = dict(rendered, episode_indices=rendered["row_indices"], goals_rgb=rendered["goal_rgb"])
    for name in POLICIES:
        normalized[name + "_forecast_valid"] = rendered[name + "_active"]
    validate_visual_packet(check, normalized, data, episodes, chosen)
    for name in POLICIES:
        check(name + " first-shell replay exactly equals directly recorded pixels", direct[name + "_rgb"][0] == rendered[name + "_rgb"][0])
    # All selected shell rows reuse the same original case ID. Check its visible
    # nuisance pixels across shells and policies when both trajectories execute.
    compared_frames = 0
    tracks = [(name, row) for name in POLICIES for row in range(6)]
    for i, (left, lr) in enumerate(tracks):
        for right, rr in tracks[i + 1:]:
            if left != right and lr != rr:
                continue
            jointly_executed = (rendered[left + "_actions"][lr] >= 0) & (rendered[right + "_actions"][rr] >= 0)
            for t in np.flatnonzero(jointly_executed):
                lframe, rframe = rendered[left + "_rgb"][lr, t + 1], rendered[right + "_rgb"][rr, t + 1]
                visible = ~red_mask(lframe) & ~red_mask(rframe)
                check(f"shared visible nuisance {left}/{lr} {right}/{rr} t={t}", lframe[visible] == rframe[visible])
                compared_frames += 1
    return {"status": "PASS", "run": str(run.relative_to(ROOT)), "checks_passed": len(check.entries),
            "model_training_seed": model_seed, "model_sha256": manifest["model_sha256"],
            "model_configuration": manifest["model_metadata"]["cfg"],
            "model_training_data_sha256": manifest["model_metadata"]["data_sha256"],
            "cohort_array_sha256": {k: array_digest(v) for k, v in data.items()},
            "executed_transitions_independently_reconstructed": total_transitions,
            "visible_paired_nuisance_frame_checks": compared_frames,
            "recomputed_by_distance_actions": metrics, "checks": check.entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "research/redcube_goal_distance_independent_audit")
    args = parser.parse_args()
    reports = [audit(run.resolve()) for run in args.runs]
    completed_seeds = sorted(r["model_training_seed"] for r in reports)
    coverage_complete = completed_seeds == [0, 1, 2]
    replication = {"status": "not_applicable_single_model"}
    if len(reports) > 1:
        if not all(r["cohort_array_sha256"] == reports[0]["cohort_array_sha256"] for r in reports):
            raise AssertionError("distance cohorts differ between model replications")
        reference = archive(args.runs[0] / "episodes.npz")
        baseline_keys = [k for k in reference if k.startswith(("random_", "fixed_cycle_", "oracle_"))]
        for run in args.runs[1:]:
            other = archive(run / "episodes.npz")
            if not all(np.array_equal(reference[k], other[k]) for k in baseline_keys):
                raise AssertionError("raw baseline episodes differ across models")
        configs = [{k: v for k, v in r["model_configuration"].items() if k != "seed"} for r in reports]
        if not all(c == configs[0] for c in configs):
            raise AssertionError("training configurations differ beyond seed")
        if len({r["model_sha256"] for r in reports}) != len(reports) or len({r["model_training_seed"] for r in reports}) != len(reports):
            raise AssertionError("model replication identities are not distinct")
        if len({r["model_training_data_sha256"] for r in reports}) != 1:
            raise AssertionError("training data provenance differs between models")
        replication = {"status": "PASS", "all_cohort_array_hashes_identical": True,
                       "all_raw_baseline_arrays_identical": True, "distinct_training_seeds_and_checkpoints": True,
                       "same_training_recipe_and_data": True}
    limits = ["This is the declared post-outcome distance-response diagnostic; it does not replace the failed hard-goal primary.",
              "No JAX PRNG or learned network is re-executed; geometry, transitions, observations and metrics are checked from raw artifacts.",
              "Runtime encoder shapes are established by hashed source and manifests, not independent GPU instrumentation.",
              "Confidence intervals remain separate for each model and distance; repeated prefixes are not pooled as independent trials."]
    result = {"status": "PASS", "auditor_source_sha256": digest(__file__),
              "independent_helper_sha256": digest(ROOT / "studies/revision/audit_redcube_goal.py"),
              "coverage": {"complete": coverage_complete, "audited_model_count": len(reports),
                           "admitted_model_count": 3, "audited_training_seeds": completed_seeds,
                           "admitted_training_seeds": [0, 1, 2], "final_curve_ready": coverage_complete},
              "replication_audit": replication, "runs": reports, "limits": limits}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    coverage_text = ("**Complete: 3 of 3 admitted models audited.**" if coverage_complete else
                     f"**Partial: {len(reports)} of 3 admitted models audited.** Remaining models have not been independently reviewed; do not render a final three-model curve yet.")
    lines = ["# Independent goal-distance diagnostic audit", "", coverage_text, "",
             "**Available-artifact audit: PASS.** Independent NumPy-only reconstruction; original primary report unchanged.", "",
             "| Training seed | Distance (actions) | JEPA H1 exact hits | JEPA H3 exact hits | Random exact hits | Fixed cycle exact hits | Oracle exact hits |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        for distance in SHELLS:
            cell = report["recomputed_by_distance_actions"][str(distance)]
            counts = [str(cell[name]["exact_goal_hit"]["count"]) + "/128" for name in ("jepa_h1", "jepa_h3", "random", "fixed_cycle", "oracle")]
            lines.append("| " + " | ".join([str(report["model_training_seed"]), str(distance)] + counts) + " |")
    lines += ["", f"Verified {sum(r['checks_passed'] for r in reports):,} checks and {sum(r['executed_transitions_independently_reconstructed'] for r in reports):,} executed native transitions.", "",
              "All original prefixes, exact-shell goal draws, warmup RGB geometry, first-hit/death masks, per-cell metrics and intervals, source/artifact hashes, selected visual rows, terminal padding and visible shared nuisance pixels were checked. When multiple models are included, their entire cohorts and raw baseline trajectories are identical.", "", "Limits:", ""]
    lines += ["- " + item for item in limits]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "coverage_complete": coverage_complete, "runs": len(reports), "checks": sum(r["checks_passed"] for r in reports), "out": str(args.out)}))


if __name__ == "__main__":
    main()
