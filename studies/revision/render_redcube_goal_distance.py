"""Publication visuals for the post-outcome red-cube distance-response diagnostic.

Read only final results from all three admitted checkpoints, plus an independent
PASS audit. Preserve all six goal shells (1, 2, 3, 6, 9, 12 moves), all five
controllers, and separate n=128 Wilson intervals for every model x shell cell.
Never pool correlated rows across shells or world-model seeds. The old hard-goal
primary remains a separate result and is not replaced by this diagnostic.

Visual examples use predeclared within-shell case 0 at every shell. The packet
contains exact keyed simulator re-renderings of recorded executions, not newly
sampled trajectories. Display RGB unchanged and stop at the first goal hit or
native death. Position traces are labeled evaluation annotations; coordinates
are not JEPA inputs. No predicted images are invented or extrapolated.

Example::

    MPLCONFIGDIR=/private/tmp/tinycraft-mplconfig \\
      /private/tmp/tinycraft-figenv/bin/python studies/revision/render_redcube_goal_distance.py \\
      --run runs/revision/redcube_goal_distance/m0 \\
            runs/revision/redcube_goal_distance/m1 \\
            runs/revision/redcube_goal_distance/m2 \\
      --audit research/redcube_goal_distance_independent_audit.json \\
      --out output/revision/redcube-goal-distance-analysis

Use --contract to inspect the source/display contract without plotting. Output
directories are never overwritten. The renderer performs no model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from studies.revision import render_redcube_goals as B

DISTANCES = (1, 2, 3, 6, 9, 12)
POLICIES = ("random", "fixed_cycle", "jepa_h1", "jepa_h3", "oracle")
CONTRACT = {
    "protocol": "redcube_goal_distance_v1",
    "classification": "post-outcome operating-range diagnostic",
    "required_model_seeds": [0, 1, 2],
    "distance_actions": list(DISTANCES),
    "n_unique_prefixes": 128,
    "n_per_model_distance": 128,
    "correlated_rows_per_model": 768,
    "policies": list(POLICIES),
    "visual_rows": [0, 128, 256, 384, 512, 640],
    "visual_selection": "predeclared case 0 within every distance shell",
    "stopping_rule": "first exact goal hit, native death, or the 24-action limit",
    "uncertainty": "Wilson 95% intervals separately per checkpoint and distance; no pooled interval",
    "rgb_source": "exact keyed simulator re-rendering of recorded executions; unchanged uint8 RGB",
    "forecast_images": "none",
    "annotation_palette": B.CONTRACT["annotation_palette"],
    "prior_primary": "original 6-12-action hard-goal evaluation remains unchanged and separately reported",
    "independent_audit": "JSON report with top-level status, verdict, or overall equal to PASS",
}


def array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(array.dtype).encode() + repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def load_audit(path):
    data = json.loads(path.read_text())
    status = next((data[k] for k in ("status", "verdict", "overall") if k in data), None)
    if not isinstance(status, str) or status.upper() != "PASS":
        raise ValueError("independent audit does not report top-level PASS")
    coverage = data.get("coverage", {})
    if coverage.get("complete") is not True or coverage.get("final_curve_ready") is not True:
        raise ValueError("independent audit does not cover the complete three-model curve")
    if coverage.get("audited_training_seeds") != [0, 1, 2]:
        raise ValueError("independent audit does not cover all admitted model seeds")
    return {"path": str(path), "sha256": B.sha256(path), "status": status, "report": data}


def load_run(path):
    required = ("summary.json", "manifest.json", "episodes.npz", "cohort.npz",
                "visual_packet.npz", "range_visual_packet.npz")
    for name in required:
        if not (path / name).is_file():
            raise ValueError(f"final artifact missing: {path / name}")
    summary = json.loads((path / "summary.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    assert summary["manifest"] == manifest
    assert manifest["protocol"] == CONTRACT["protocol"]
    assert manifest["classification"] == CONTRACT["classification"]
    assert manifest["distances_actions"] == list(DISTANCES)
    assert manifest["n_unique_prefixes"] == manifest["n_per_distance"] == 128
    assert manifest["n_total_correlated_rows"] == 768
    assert manifest["control_steps"] == 24 and manifest["horizon"] == 3
    assert manifest["config"] == "gray_frame_still"
    assert manifest["visual_rows_predeclared"] == CONTRACT["visual_rows"]
    assert manifest["visual_within_shell_case_index"] == 0
    assert manifest["cohort_sha256"] == B.sha256(path / "cohort.npz")
    for name, digest in summary["artifacts"].items():
        if B.sha256(path / name) != digest:
            raise ValueError(f"artifact hash mismatch: {path / name}")
    cohort = archive(path / "cohort.npz")
    episodes = archive(path / "episodes.npz")
    visual = archive(path / "range_visual_packet.npz")
    direct = archive(path / "visual_packet.npz")
    for key, digest in manifest["cohort_array_sha256"].items():
        if array_sha256(cohort[key]) != digest:
            raise ValueError(f"cohort array hash mismatch: {key}")
    d = cohort["distance_actions"]
    rows = np.asarray(CONTRACT["visual_rows"])
    np.testing.assert_array_equal(d, np.repeat(DISTANCES, 128))
    np.testing.assert_array_equal(cohort["within_shell_case_index"], np.tile(np.arange(128), 6))
    np.testing.assert_array_equal(cohort["goals"] - cohort["state_pos"], cohort["goal_offsets"])
    np.testing.assert_array_equal(np.abs(cohort["goal_offsets"]).sum(1), 3 * d)
    for key in ("candidate_ids", "frames", "warmup_actions", "state_pos", "state_wind", "state_texture", "state_step", "state_episode"):
        np.testing.assert_array_equal(cohort[key], np.concatenate([cohort[key][:128]] * 6))
    np.testing.assert_array_equal(visual["row_indices"], rows)
    np.testing.assert_array_equal(visual["within_shell_case_indices"], np.zeros(6, int))
    np.testing.assert_array_equal(visual["distance_actions"], DISTANCES)
    np.testing.assert_array_equal(visual["candidate_ids"], cohort["candidate_ids"][rows])
    np.testing.assert_array_equal(visual["goal_rgb"], cohort["goal_rgb"][rows])
    np.testing.assert_array_equal(visual["goal_positions"], cohort["goals"][rows])
    np.testing.assert_array_equal(visual["prefix_rgb"], cohort["frames"][rows])
    for policy in POLICIES:
        actions, hit, alive = (episodes[policy + "_" + key] for key in ("actions", "hit", "alive"))
        assert actions.shape == (768, 24) and hit.shape == alive.shape == (768, 25)
        assert alive[:, 0].all() and not hit[:, 0].any()
        assert ((actions >= -1) & (actions <= 3)).all()
        assert (np.diff(hit.astype(int), axis=1) >= 0).all()
        assert (np.diff(alive.astype(int), axis=1) <= 0).all()
        active = alive[:, :-1] & ~hit[:, :-1]
        np.testing.assert_array_equal(actions >= 0, active)
        np.testing.assert_array_equal(visual[policy + "_active"], active[rows])
        for suffix in ("positions", "actions", "alive", "hit"):
            np.testing.assert_array_equal(visual[policy + "_" + suffix], episodes[policy + "_" + suffix][rows])
        rgb = visual[policy + "_rgb"]
        assert rgb.shape == (6, 25, 63, 63, 3) and rgb.dtype == np.uint8
        np.testing.assert_array_equal(rgb[:, 0], visual["prefix_rgb"][:, -1])
        np.testing.assert_array_equal(rgb[0], direct[policy + "_rgb"][0])
        for shell in range(6):
            stop, _ = B.terminal(visual, policy, shell)
            np.testing.assert_array_equal(rgb[shell, stop:], np.broadcast_to(rgb[shell, stop], (25 - stop, 63, 63, 3)))
            position = visual[policy + "_positions"][shell]
            np.testing.assert_array_equal(position[stop:], np.broadcast_to(position[stop], (25 - stop, 2)))
        for distance in DISTANCES:
            take = d == distance
            k = int(hit[take, -1].sum())
            result = summary["by_distance_actions"][str(distance)][policy]["exact_goal_hit"]
            assert result["count"] == k and result["n"] == 128
            np.testing.assert_allclose(result["fraction"], k / 128, atol=1e-14, rtol=0)
            np.testing.assert_allclose(result["wilson_95"], B.wilson(k, 128), atol=1e-14, rtol=0)
    model_seed = int(manifest["model_metadata"]["cfg"]["seed"])
    return {"path": path, "manifest": manifest, "summary": summary, "cohort": cohort,
            "episodes": episodes, "visual": visual, "model_seed": model_seed,
            "label": f"Model seed {model_seed}", "source_hashes": {name: B.sha256(path / name) for name in required}}


def distance_curves(runs, out, publication=False):
    plt = B.plt
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 3.4), sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.29, top=.75, wspace=.18)
    fig.text(.035, .96, "Visual-goal control versus goal distance", weight="bold", fontsize=10, va="top")
    subtitle = ("128 paired starts per distance · 24-action limit" if publication
                else "Post-outcome diagnostic · 128 starts per shell · 24-action limit")
    fig.text(.035, .88, subtitle, fontsize=7.5, color=B.GRAY, va="top")
    style = {
        "random": {"marker": "x", "linestyle": "--", "linewidth": .85, "markersize": 3},
        "fixed_cycle": {"marker": "v", "linestyle": ":", "linewidth": .85, "markersize": 3},
        "jepa_h1": {"marker": "^", "linestyle": "--", "linewidth": 1, "markersize": 3.2},
        "jepa_h3": {"marker": "o", "linestyle": "-", "linewidth": 1.5, "markersize": 3.5},
        "oracle": {"marker": "s", "linestyle": ":", "linewidth": .9, "markersize": 2.6},
    }
    rows, handles = [], []
    for run, ax in zip(runs, axes):
        for policy in ("fixed_cycle", "random", "jepa_h1", "oracle", "jepa_h3"):
            values = [run["summary"]["by_distance_actions"][str(d)][policy]["exact_goal_hit"] for d in DISTANCES]
            p = np.asarray([v["fraction"] for v in values]) * 100
            ci = np.asarray([v["wilson_95"] for v in values]) * 100
            ax.errorbar(DISTANCES, p, yerr=np.maximum(0, np.stack([p - ci[:, 0], ci[:, 1] - p])),
                        fmt="none", ecolor=B.COLORS[policy], elinewidth=.5,
                        alpha=.6 if policy != "jepa_h3" else .95, capsize=1.25, zorder=2)
            line, = ax.plot(DISTANCES, p, color=B.COLORS[policy], label=B.LABELS[policy],
                            markerfacecolor="white" if policy in ("jepa_h1", "fixed_cycle") else B.COLORS[policy],
                            markeredgewidth=.7, zorder=4 if policy == "jepa_h3" else 3, **style[policy])
            if run is runs[0]:
                handles.append(line)
            rows.extend({"model_seed": run["model_seed"], "distance_actions": d, "policy": policy, **stat}
                        for d, stat in zip(DISTANCES, values))
        ax.set_title(run["label"], fontsize=8, pad=5)
        ax.set_xlim(.5, 12.5)
        ax.set_ylim(-4, 106)
        ax.set_xticks(DISTANCES, [str(d) for d in DISTANCES], fontsize=6.7)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", length=2, width=.5, labelsize=7)
        ax.grid(axis="y", color=B.PALE, linewidth=.55)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Exact goal hit (%)", fontsize=8)
    fig.text(.55, .18, "Initial goal distance (cardinal moves)", ha="center", fontsize=8)
    handle_by_label = {h.get_label(): h for h in handles}
    if publication:
        handle_by_label[B.LABELS["jepa_h3"]].set_label("JEPA · horizon 3")
        handle_by_label[B.LABELS["jepa_h1"]].set_label("JEPA · horizon 1")
    order = ("jepa_h3", "jepa_h1", "oracle", "random", "fixed_cycle")
    fig.legend(handles=[handle_by_label[B.LABELS[p]] for p in order],
               loc="lower center", bbox_to_anchor=(.52, .005), ncol=3,
               frameon=False, fontsize=7, handlelength=2.3, columnspacing=1.6)
    caption = ("Post-outcome operating-range diagnostic of the unchanged frozen visual-goal controllers, conducted after "
               "the unsuccessful 6–12-action hard-goal primary. All six goal-distance shells and all three existing model "
               "seeds were admitted before the follow-up. Every point is exact goal hits out of 128 starts before native "
               "death or the 24-action limit; whiskers are Wilson 95% intervals for that model and shell only. The same "
               "128 valid prefixes are reused across shells, so neither shells nor model-seed results are pooled into "
               "an IID binomial interval. JEPA receives RGB history, outgoing actions and an RGB goal. The H1/H3 "
               "controllers retain exhaustive latent-distance search and the H3 summed stage cost; the state oracle "
               "uses privileged physical coordinates. The follow-up changes goal distance only and does not replace "
               "the original harder-goal result or establish long-horizon planning.")
    stem = "goal_distance_response_publication" if publication else "goal_distance_response"
    return B.export(fig, out, stem, caption, {"cells": rows, "publication_variant": publication})


def all_shell_paths(run, out):
    plt, v = B.plt, run["visual"]
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 4.3))
    fig.subplots_adjust(left=.035, right=.985, top=.81, bottom=.16, wspace=.11, hspace=.27)
    fig.text(.035, .965, f'Paired paths across goal distances · {run["label"]}', fontsize=9.5, weight="bold", va="top")
    fig.text(.035, .905, "Post-outcome diagnostic · case 0 in every shell · same initial observation", fontsize=7.3, color=B.GRAY, va="top")
    records = []
    for shell, (distance, ax) in enumerate(zip(DISTANCES, axes.flat)):
        ax.imshow(v["prefix_rgb"][shell, -1], interpolation="nearest", alpha=.24)
        goal = v["goal_positions"][shell]
        ax.add_patch(B.Rectangle(goal - .5, 6, 6, facecolor="none", edgecolor=B.INK, linewidth=.9, linestyle="--", zorder=8))
        for policy in ("random", "oracle", "jepa_h3"):
            stop, event = B.terminal(v, policy, shell)
            path = v[policy + "_positions"][shell, :stop + 1] + 2.5
            ax.plot(*path.T, color="white", linewidth=2.3, zorder=3)
            ax.plot(*path.T, color=B.COLORS[policy], linewidth=1.1, marker=".", markersize=2,
                    linestyle=":" if policy == "oracle" else "-", zorder=4)
            ax.scatter(*path[-1], color=B.COLORS[policy], marker="o" if event == "Goal hit" else "x", s=17, linewidth=.8, zorder=7)
            records.append({"distance_actions": distance, "policy": policy, "stop_step": stop, "event": event})
        start = v["jepa_h3_positions"][shell, 0] + 2.5
        ax.scatter(*start, facecolors="white", edgecolors=B.INK, s=17, linewidth=.8, zorder=9)
        ax.set_title(f'{distance} move' + ("" if distance == 1 else "s"), fontsize=8, pad=4)
        ax.set_xlim(-.5, 62.5)
        ax.set_ylim(62.5, -.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [B.Line2D([0], [0], color=B.COLORS[p], linewidth=1.5, label=B.LABELS[p],
                         linestyle=":" if p == "oracle" else "-") for p in ("random", "jepa_h3", "oracle")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .061), ncol=3, frameon=False, fontsize=7.3)
    fig.text(.035, .024, "Coordinate overlays are evaluation annotations, not JEPA inputs. Dashed square: goal.", fontsize=6.8, color=B.GRAY)
    return B.export(fig, out, f'goal_distance_paths_m{run["model_seed"]}',
                    "All six predeclared visual examples: within-shell case 0, with the same valid RGB prefix and a "
                    "different sampled goal at each initial distance. Paths are actual recorded positions, truncated "
                    "at first exact goal hit, native death, or the action limit. Coordinates are diagnostic overlays on "
                    "a faded initial RGB frame, not JEPA inputs. The state oracle is privileged. A dashed square marks "
                    "the goal; endpoint circle denotes a hit, cross another ending. This follow-up was designed after "
                    "the failed original hard-goal primary and is labeled post-outcome.",
                    {"model_seed": run["model_seed"], "records": records, "rgb_transform": "alpha .24 in labeled path diagrams only"})


def all_shell_rgb(run, out):
    plt, v, policy = B.plt, run["visual"], "jepa_h3"
    width, height = 5.5, 5.92
    fig = plt.figure(figsize=(width, height))
    fig.text(.035, .972, f'JEPA three-step control · {run["label"]}', fontsize=10, weight="bold", va="top")
    fig.text(.035, .93, "Post-outcome diagnostic · predeclared case 0 in all six goal shells", fontsize=7.5, color=B.GRAY, va="top")
    x0, tile, gap, row_top, row_gap = .87, .455, .035, .80, .79
    fig.text((x0 + tile / 2) / width, .887, "RGB goal", ha="center", fontsize=7)
    fig.text((x0 + tile + 2 * gap) / width, .887, "Observed step (preceding action)", fontsize=7)
    records = []
    for shell, distance in enumerate(DISTANCES):
        stop, event = B.terminal(v, policy, shell)
        times = B.strip_times(stop)
        y = 1 - (row_top + tile + shell * row_gap) / height
        fig.text(.145, y + tile / 2 / height, f'{distance} move' + ("" if distance == 1 else "s"),
                 ha="right", va="center", fontsize=7.4, weight="bold", color=B.JEPA)
        frames = [v["goal_rgb"][shell], *v[policy + "_rgb"][shell, times]]
        for column, frame in enumerate(frames):
            x = x0 + column * (tile + gap) + (gap if column else 0)
            ax = fig.add_axes([x / width, y, tile / width, tile / height])
            ax.imshow(frame, interpolation="nearest")
            ax.set_axis_off()
            if column:
                t = times[column - 1]
                action = "" if t == 0 else f" ({B.ARROWS[int(v[policy + '_actions'][shell, t - 1])]})"
                fig.text((x + tile / 2) / width, y - .011, f"{t}{action}", ha="center", va="top", fontsize=6.8)
        actions = v[policy + "_actions"][shell, :stop].tolist()
        sequence = " ".join(B.ARROWS[a] for a in actions)
        fig.text(x0 / width, y - .036, f"{event} at {stop}  ·  Actions: {sequence}", color=B.JEPA, fontsize=6.8)
        records.append({"distance_actions": distance, "within_shell_case_index": 0,
                        "global_row": int(v["row_indices"][shell]), "candidate_id": int(v["candidate_ids"][shell]),
                        "policy": policy, "frame_times": times, "stop_step": stop, "event": event,
                        "action_indices": actions, "action_names": [B.ACTION_NAMES[a] for a in actions]})
    fig.text(.035, .025, "Stored executions, exactly re-rendered from simulator keys. No frames after the terminal event.", fontsize=6.7, color=B.GRAY)
    return B.export(fig, out, f'goal_distance_rgb_m{run["model_seed"]}',
                    "JEPA H3 observed trajectories for every predeclared visual shell: the same within-shell case 0, "
                    "at goals one, two, three, six, nine and twelve moves away. Images are exact keyed simulator "
                    "re-renderings of the saved executions, with RGB values unchanged. The first-shell replay is "
                    "bit-for-bit checked against directly recorded RGB by the evaluator and renderer. Frames occur "
                    "every four actions plus the exact terminal frame; no padding after first goal hit or death is "
                    "displayed. Labels show the action entering each sampled frame and the full executed action "
                    "sequence. All six shells are retained independently of whether JEPA succeeds. This is a "
                    "post-outcome diagnostic, not a replacement for the original harder-goal primary.",
                    {"model_seed": run["model_seed"], "records": records, "rgb_transform": "none"})


def raw_assets(run, out):
    v = run["visual"]
    directory = out / "raw" / f'm{run["model_seed"]}'
    directory.mkdir(parents=True)
    result = []
    for shell, distance in enumerate(DISTANCES):
        goal = directory / f"distance{distance:02d}_case0_goal.png"
        B.Image.fromarray(v["goal_rgb"][shell]).save(goal)
        assert np.array_equal(np.asarray(B.Image.open(goal)), v["goal_rgb"][shell])
        for policy in POLICIES:
            stop, event = B.terminal(v, policy, shell)
            images = []
            for t, frame in enumerate(v[policy + "_rgb"][shell, :stop + 1]):
                path = directory / f"distance{distance:02d}_case0_{policy}_t{t:02d}.png"
                B.Image.fromarray(frame).save(path)
                assert np.array_equal(np.asarray(B.Image.open(path)), frame)
                images.append(str(path.relative_to(out)))
            result.append({"distance_actions": distance, "within_shell_case_index": 0,
                           "global_row": int(v["row_indices"][shell]), "policy": policy,
                           "goal": str(goal.relative_to(out)), "frames": images, "stop_step": stop,
                           "event": event, "actions": v[policy + "_actions"][shell, :stop].tolist()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output/revision/redcube-goal-distance-analysis"))
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--publication-only", action="store_true",
                        help="make a separate compact curve; keep post-outcome scope explicitly in its caption and provenance")
    args = parser.parse_args()
    if args.contract:
        print(json.dumps(CONTRACT, indent=2))
        return
    if not args.run or len(args.run) != 3 or args.audit is None:
        parser.error("all three final --run directories and a PASS --audit are required")
    audit = load_audit(args.audit)
    runs = sorted([load_run(path) for path in args.run], key=lambda r: r["model_seed"])
    if [r["model_seed"] for r in runs] != [0, 1, 2]:
        raise ValueError("the admitted model seeds must be exactly 0, 1, 2")
    audited = {r["model_training_seed"]: r for r in audit["report"]["runs"]}
    for run in runs:
        checked = audited[run["model_seed"]]
        assert checked["status"] == "PASS"
        assert checked["model_sha256"] == run["manifest"]["model_sha256"]
        # The independent auditor deliberately uses a JSON shape/dtype header,
        # while the run manifest uses the original repr-based header. Verify
        # each convention against the same loaded arrays, not against each other.
        for key, expected_digest in checked["cohort_array_sha256"].items():
            array = run["cohort"][key]
            header = json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode()
            actual_digest = hashlib.sha256(header + b"\n" + np.ascontiguousarray(array).tobytes()).hexdigest()
            assert actual_digest == expected_digest
        for distance in DISTANCES:
            for policy in POLICIES:
                expected = checked["recomputed_by_distance_actions"][str(distance)][policy]["exact_goal_hit"]
                actual = run["summary"]["by_distance_actions"][str(distance)][policy]["exact_goal_hit"]
                assert expected["count"] == actual["count"] and expected["n"] == actual["n"]
    if len({r["manifest"]["model_sha256"] for r in runs}) != 3:
        raise ValueError("duplicate model checkpoint")
    for run in runs[1:]:
        if run["manifest"]["cohort_array_sha256"] != runs[0]["manifest"]["cohort_array_sha256"]:
            raise ValueError("distance cohorts differ between checkpoints")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    B.setup_plotting()
    figures = [distance_curves(runs, args.out, publication=args.publication_only)]
    raw = {}
    if not args.publication_only:
        for run in runs:
            figures.extend([all_shell_paths(run, args.out), all_shell_rgb(run, args.out)])
            raw[f'm{run["model_seed"]}'] = raw_assets(run, args.out)
    report = {
        "contract": CONTRACT, "publication_only": args.publication_only, "independent_audit": {k: audit[k] for k in ("path", "sha256", "status")},
        "renderer_sha256": {str(Path(__file__).relative_to(ROOT)): B.sha256(__file__),
                            str(Path(B.__file__).relative_to(ROOT)): B.sha256(B.__file__)},
        "runs": [{"model_seed": r["model_seed"], "directory": str(r["path"]),
                  "model_sha256": r["manifest"]["model_sha256"], "source_hashes": r["source_hashes"]} for r in runs],
        "figures": figures, "raw_assets": raw,
    }
    (args.out / "figure_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "counts_by_distance.json").write_text(json.dumps(figures[0]["cells"], indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "figures": len(figures), "validation": "PASS",
                      "classification": CONTRACT["classification"]}))


if __name__ == "__main__":
    main()
