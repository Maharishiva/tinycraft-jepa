"""Independent CPU audit of frozen red-cube goal-control artifacts.

This does not import the evaluator, environment, JAX, or learned model. Native
geometry and dynamics are checked directly from the documented task constants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MOVES = np.array([[-3, 0], [0, -3], [3, 0], [0, 3]], np.int32)
POLICIES = ("random", "fixed_cycle", "oracle", "jepa_h1", "jepa_h3")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    header = json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\n" + np.ascontiguousarray(array).tobytes()).hexdigest()


class Check:
    def __init__(self):
        self.entries = []

    def __call__(self, name, condition):
        success = bool(np.all(condition))
        self.entries.append({"check": name, "pass": success})
        if not success:
            raise AssertionError(name)


def red_mask(frames):
    return (frames[..., 0] == 255) & (frames[..., 1] == 0) & (frames[..., 2] == 0)


def rendered_mask(positions):
    y, x = np.mgrid[:63, :63]
    ring = (((x == 1) | (x == 61)) & (y >= 1) & (y <= 61)) | (((y == 1) | (y == 61)) & (x >= 1) & (x <= 61))
    p = np.clip(positions, 0, 57)
    cube = ((x >= p[..., 0, None, None]) & (x < p[..., 0, None, None] + 6)
            & (y >= p[..., 1, None, None]) & (y < p[..., 1, None, None] + 6))
    return cube | ring


def infer_safe_positions(frames, check):
    red = red_mask(frames)
    y, x = np.mgrid[:63, :63]
    cube = red & (x >= 2) & (x <= 60) & (y >= 2) & (y <= 60)
    count = cube.sum(axis=(-2, -1))
    check("every safe prefix RGB has exactly 36 red cube pixels outside the border", count == 36)
    xpos = np.where(cube, x, 63).min(axis=(-2, -1))
    ypos = np.where(cube, y, 63).min(axis=(-2, -1))
    positions = np.stack((xpos, ypos), -1)
    check("prefix RGB red mask exactly matches a six-pixel square and native border", red == rendered_mask(positions))
    return positions


def proportion_record(values):
    values = np.asarray(values, bool)
    n, k = values.size, int(values.sum())
    p = k / n
    z = 1.959963984540054
    denominator = n + z * z
    center = (k + z * z / 2) / denominator
    half = z * math.sqrt(n * p * (1 - p) + z * z / 4) / denominator
    return {"count": k, "n": n, "fraction": p,
            "wilson_95": [max(0., center - half), min(1., center + half)]}


def compare_summary(check, name, expected, actual):
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare_summary(check, name + "." + key, value, actual[key])
    elif expected is None:
        check(name, actual is None)
    else:
        check(name, np.allclose(expected, actual, atol=1e-12, rtol=1e-12))


def audit(run):
    check = Check()
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    with np.load(run / "cohort.npz", allow_pickle=False) as data:
        cohort = {k: data[k] for k in data.files}
    with np.load(run / "episodes.npz", allow_pickle=False) as data:
        episodes = {k: data[k] for k in data.files}
    with np.load(run / "visual_packet.npz", allow_pickle=False) as data:
        visual = {k: data[k] for k in data.files}
    check("standalone and embedded manifests identical", manifest == summary["manifest"])
    for filename, expected in summary["artifacts"].items():
        check("artifact SHA256 " + filename, digest(run / filename) == expected)
    check("cohort manifest SHA256", digest(run / "cohort.npz") == manifest["cohort_sha256"])
    for relative, expected in manifest["source_sha256"].items():
        check("local launch source SHA256 " + relative, digest(ROOT / relative) == expected)
    check("canonical encoder shape", manifest["encoder_batch"] == [16, 8])
    check("native 512-dimensional model", manifest["latent_dim"] == 512)
    check("frozen evaluation budget", (manifest["n"], manifest["control_steps"], manifest["horizon"]) == (128, 24, 3))
    check("deterministic control setup", manifest["config"] in ("gray_frame_still", "off_still"))
    check("model digest format", len(manifest["model_sha256"]) == 64)
    check("prespecified visual selection in manifest", manifest["visual_indices_predeclared"] == [0, 1, 2])

    frames, actions = cohort["frames"], cohort["warmup_actions"]
    starts, goals, ids = cohort["state_pos"], cohort["goals"], cohort["candidate_ids"]
    check("prefix shapes", frames.shape == (128, 8, 63, 63, 3) and actions.shape == (128, 7))
    check("native uint8 prefix and goal pixels", frames.dtype == np.uint8 and cohort["goal_rgb"].dtype == np.uint8)
    check("all admitted candidate IDs distinct and ascending", np.diff(ids) > 0)
    check("candidate rejection accounting", int(ids[-1]) + 1 == 128 + manifest["prefix_rejected_before_last_admission"])
    check("all prefix actions native", (actions >= 0) & (actions < 4))
    prefix_positions = infer_safe_positions(frames, check)
    check("all non-red prefix pixels are grayscale", red_mask(frames) | ((frames[..., 0] == frames[..., 1]) & (frames[..., 1] == frames[..., 2])))
    check("all prefix states safely inside native border", (prefix_positions >= 2) & (prefix_positions <= 55))
    check("prefix reset support", (prefix_positions[:, 0] >= 16) & (prefix_positions[:, 0] <= 42))
    check("every warmup RGB transition equals its recorded native action", np.diff(prefix_positions, axis=1) == MOVES[actions])
    check("last prefix RGB position equals stored state", prefix_positions[:, -1] == starts)
    check("zero wind throughout this deterministic setup", cohort["state_wind"] == 0)
    check("stored prefix clock is seven", cohort["state_step"] == 7)
    check("stored initial episode index is zero", cohort["state_episode"] == 0)
    check("goals inside native safe support", (goals >= 2) & (goals <= 55))
    check("goals share both start coordinate residues modulo three", (goals - starts) % 3 == 0)
    distance0 = np.abs(goals - starts).sum(1)
    check("all goals require six through twelve minimum actions", (distance0 >= 18) & (distance0 <= 36))
    check("RGB goals match exact target positions", red_mask(cohort["goal_rgb"]) == rendered_mask(goals))
    rng = np.random.default_rng(manifest["seed"] + 10001)
    reproduced_goals = []
    for position in starts:
        candidates = [(x, y) for x in range(2, 56) for y in range(2, 56)
                      if (x - int(position[0])) % 3 == 0 and (y - int(position[1])) % 3 == 0
                      and 18 <= abs(x - int(position[0])) + abs(y - int(position[1])) <= 36]
        reproduced_goals.append(candidates[int(rng.integers(len(candidates)))])
    check("goal draw reproduced independently from declared seed and uniform candidate support", goals == np.array(reproduced_goals))
    for label, expected in (("min", int(distance0.min())), ("max", int(distance0.max())), ("mean", float(distance0.mean()))):
        check("initial goal distance " + label, manifest["goal_distance_initial_l1_" + label] == expected)
    check("packet episode selection fixed before outcomes", visual["episode_indices"] == np.arange(3))
    check("packet candidate IDs", visual["candidate_ids"] == ids[:3])
    check("packet exact prefix RGB", visual["prefix_rgb"] == frames[:3])
    check("packet exact goal RGB", visual["goals_rgb"] == cohort["goal_rgb"][:3])
    check("packet exact goal positions", visual["goal_positions"] == goals[:3])

    recomputed = {}
    transitions = 0
    for policy in POLICIES:
        a = episodes[policy + "_actions"]
        positions = episodes[policy + "_positions"]
        check(policy + " trajectory shapes", a.shape == (128, 24) and positions.shape == (128, 25, 2))
        check(policy + " initial positions", positions[:, 0] == starts)
        p = starts.copy()
        alive, hit = np.ones(128, bool), np.zeros(128, bool)
        first = np.full(128, -1)
        minimum = distance0.copy()
        alive_history, hit_history, distances = [alive.copy()], [hit.copy()], [distance0.copy()]
        for t in range(24):
            active = alive & ~hit
            check(policy + f" action support t={t}", ((a[:, t] >= 0) & (a[:, t] < 4)) == active)
            check(policy + f" terminal sentinel t={t}", a[~active, t] == -1)
            nxt = p.copy()
            nxt[active] += MOVES[a[active, t]]
            check(policy + f" native transition t={t}", positions[:, t + 1] == nxt)
            transitions += int(active.sum())
            death = active & (((nxt < 2) | (nxt > 55)).any(1) | (8 + t >= 128))
            alive[death] = False
            distance = np.abs(nxt - goals).sum(1)
            new_hit = active & alive & (distance == 0)
            first[new_hit] = t + 1
            hit |= new_hit
            minimum = np.minimum(minimum, np.where(alive, distance, minimum))
            alive_history.append(alive.copy())
            hit_history.append(hit.copy())
            distances.append(distance)
            p = nxt
        alive_history, hit_history, distances = map(lambda x: np.stack(x, 1), (alive_history, hit_history, distances))
        check(policy + " exact native death mask", episodes[policy + "_alive"] == alive_history)
        check(policy + " first-hit mask without post-death successes", episodes[policy + "_hit"] == hit_history)
        check(policy + " first-hit steps", episodes[policy + "_first_hit"] == first)
        check(policy + " distances at every step", episodes[policy + "_distance_l1"] == distances)
        check(policy + " minimum valid distance", episodes[policy + "_minimum_distance_l1"] == minimum)
        expected = {"exact_goal_hit": proportion_record(hit), "within_one_action_l1_3": proportion_record(minimum <= 3),
                    "died_before_goal": proportion_record(~alive), "mean_minimum_distance_l1": float(minimum.mean()),
                    "mean_final_distance_l1": float(distances[:, -1].mean()),
                    "mean_first_hit_steps_given_success": float(first[hit].mean()) if hit.any() else None}
        compare_summary(check, policy + " summary", expected, summary["policies"][policy])
        recomputed[policy] = expected
        for field in ("actions", "positions", "alive", "hit"):
            check(policy + " packet " + field, visual[policy + "_" + field] == episodes[policy + "_" + field][:3])
        valid = alive_history[:3, :-1] & ~hit_history[:3, :-1]
        check(policy + " packet forecast-valid mask", visual[policy + "_forecast_valid"] == valid)
        pixels = visual[policy + "_rgb"]
        check(policy + " packet initial RGB exactly matches real prefix", pixels[:, 0] == frames[:3, -1])
        check(policy + " packet native red geometry at every displayed step", red_mask(pixels) == rendered_mask(positions[:3]))
        for row in range(3):
            terminal = np.flatnonzero(~valid[row])
            if terminal.size:
                t = int(terminal[0])
                check(policy + f" RGB terminal padding episode={row}", pixels[row, t:] == pixels[row, t])
        costs = episodes[policy + "_selected_cost"]
        if policy.startswith("jepa"):
            check(policy + " finite nonnegative costs", np.isfinite(costs) & (costs >= 0))
            plans = visual[policy + "_selected_plan"]
            check(policy + " selected first action matches execution", plans[..., 0][valid] == a[:3][valid])
        if policy == "fixed_cycle":
            cycle = np.resize(np.array([0, 2, 1, 3]), 24)[None]
            check("fixed-cycle sequence exactly predefined", a[a >= 0] == np.broadcast_to(cycle, a.shape)[a >= 0])
        if policy == "oracle":
            check("oracle reaches every goal", hit)
            check("oracle hits each goal in exactly the shortest-path number of native actions", first == distance0 // 3)
    paired_noise_frames = 0
    for left_index, left in enumerate(POLICIES):
        for right in POLICIES[left_index + 1:]:
            lp, rp = visual[left + "_rgb"], visual[right + "_rgb"]
            jointly_executed = (visual[left + "_actions"] >= 0) & (visual[right + "_actions"] >= 0)
            for row, t in np.argwhere(jointly_executed):
                lframe, rframe = lp[row, t + 1], rp[row, t + 1]
                shared_background = ~red_mask(lframe) & ~red_mask(rframe)
                check(f"shared observed nuisance noise {left}/{right} episode={row} t={t}", lframe[shared_background] == rframe[shared_background])
                paired_noise_frames += 1
    report = {"status": "PASS", "run": str(run.relative_to(ROOT)), "model_sha256": manifest["model_sha256"],
              "model_training_seed": manifest["model_metadata"]["cfg"]["seed"],
              "model_training_configuration": manifest["model_metadata"]["cfg"],
              "model_training_data_sha256": manifest["model_metadata"]["data_sha256"],
              "cohort_sha256": manifest["cohort_sha256"], "checks_passed": len(check.entries),
              "cohort_array_sha256": {k: array_digest(v) for k, v in cohort.items()},
              "independently_verified_executed_transitions": transitions, "prefix_rgb_frames_checked": 128 * 8,
              "visible_paired_noise_frame_comparisons": paired_noise_frames,
              "checks": check.entries, "recomputed_metrics": recomputed,
              "scope": "CPU audit of all recorded native transitions, RGB geometry, masks, goals, statistics, source and artifact digests; no learned-model/GPU re-execution.",
              "limits": ["Fresh reset and renderer RNG keys are verified from the reviewed source; JAX PRNG streams are not replayed by this NumPy-only audit.",
                         "The supplied model SHA identifies the launch checkpoint; this audit does not load checkpoint weights.",
                         "Encoder batch shape is established by the hashed source and manifest, rather than runtime instrumentation.",
                         "Decoded forecast pixels are illustrations; this audit verifies their policy selection and validity masks, not learned decoder fidelity."]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=ROOT / "research/redcube_goal_independent_audit")
    args = parser.parse_args()
    reports = [audit(p.resolve()) for p in args.runs]
    replication = {"status": "not_applicable_single_model"}
    if len(reports) > 1:
        with np.load(args.runs[0] / "cohort.npz", allow_pickle=False) as archive:
            reference = {k: archive[k] for k in archive.files}
        for run in args.runs[1:]:
            with np.load(run / "cohort.npz", allow_pickle=False) as archive:
                if set(archive.files) != set(reference) or not all(np.array_equal(reference[k], archive[k]) for k in reference):
                    raise AssertionError("model replications must use identical cohort arrays")
        with np.load(args.runs[0] / "episodes.npz", allow_pickle=False) as archive:
            baselines = {k: archive[k] for k in archive.files if k.startswith(("random_", "fixed_cycle_", "oracle_"))}
        for run in args.runs[1:]:
            with np.load(run / "episodes.npz", allow_pickle=False) as archive:
                if not all(np.array_equal(v, archive[k]) for k, v in baselines.items()):
                    raise AssertionError("all raw baseline trajectories must match exactly across model replications")
        configs = [{k: v for k, v in r["model_training_configuration"].items() if k != "seed"} for r in reports]
        if not all(c == configs[0] for c in configs):
            raise AssertionError("training configurations differ beyond model seed")
        if len({r["model_sha256"] for r in reports}) != len(reports) or len({r["model_training_seed"] for r in reports}) != len(reports):
            raise AssertionError("model replication identities must have distinct checkpoint hashes and training seeds")
        if len({r["model_training_data_sha256"] for r in reports}) != 1:
            raise AssertionError("training-data provenance must agree across model replications")
        replication = {"status": "PASS", "all_cohort_arrays_bit_identical": True,
                       "all_raw_baseline_arrays_bit_identical": True,
                       "checkpoint_hashes_and_training_seeds_distinct": True,
                       "training_configs_differ_only_in_seed": True, "same_training_data_digest": True,
                       "cohort_array_sha256": reports[0]["cohort_array_sha256"]}
    result = {"status": "PASS", "auditor_source_sha256": digest(__file__), "replication_audit": replication, "runs": reports}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Independent red-cube goal-control result audit", "", "**PASS.** The auditor imports no evaluator, environment, JAX, or learned-model code.", "",
             "Every native transition is reconstructed independently from the four three-pixel action vectors and the documented border. Prefix positions are recovered directly from the red square in raw RGB, excluding the red border.", "",
             "| Model | Checks | Native executed transitions | JEPA H1 exact hits | JEPA H3 exact hits | Random exact hits | Oracle exact hits |", "|---|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        metrics = report["recomputed_metrics"]
        counts = [str(metrics[p]["exact_goal_hit"]["count"]) + "/128" for p in ("jepa_h1", "jepa_h3", "random", "oracle")]
        lines.append("| " + " | ".join([report["run"].split("/")[-1], str(report["checks_passed"]), str(report["independently_verified_executed_transitions"])] + counts) + " |")
    lines += ["", "Verified all 1,024 prefix frames per model; every warmup action; safe start/goal geometry and 6–12-action lattice reachability; exact goal RNG draws; first-hit and first-death absorption; all success, death, distance and time summaries; Wilson intervals; visual episodes 0/1/2; terminal padding and forecast-valid masks; selected-plan first actions; observed shared nuisance pixels across jointly active visual trajectories; and recorded artifact/source hashes.", "",
              ("Across models, every cohort array and every raw random/fixed-cycle/oracle trajectory is bit-identical. Checkpoint hashes and training seeds are distinct; training configurations differ only in seed and name the same training-data digest." if len(reports) > 1 else "Only one model has been audited so far."), "",
              "The oracle reaches every goal in its exact shortest-path number of actions. This protocol measures empty-arena goal-directed control; an audited execution does not imply that JEPA achieved a successful positive control. The observed JEPA success counts must determine that interpretation. The task does not establish a requirement for deep planning.", "", "Limitations:", ""]
    lines.extend("- " + limit for limit in reports[0]["limits"])
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "runs": len(reports), "checks": sum(r["checks_passed"] for r in reports), "out": str(args.out)}))


if __name__ == "__main__":
    main()
