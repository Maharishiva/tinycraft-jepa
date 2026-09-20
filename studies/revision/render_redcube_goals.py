"""Render final, frozen red-cube goal-control results without model inference.

Source contract
---------------
Each --run directory must contain the FINAL ``summary.json``, ``manifest.json``,
``cohort.npz``, ``episodes.npz`` and ``visual_packet.npz`` produced by
``redcube_goal_eval.py``. Partial results are refused. Artifact SHA256s, counts,
Wilson intervals, paired starts, visual indices and terminal masks are checked
before rendering. No images or outcomes are synthesized. The predeclared visual
cohort is episodes [0, 1, 2], regardless of outcome. Every observed strip and path
ends at the first exact goal hit, native death, or the 24-action limit. Forecasts
are shown only when their forecast_valid mask is true, and are explicitly decoder
visualizations of the selected predicted latent sequence, not observed futures.

Raw RGB assets are lossless copies of uint8 arrays. Position paths use simulator
positions ONLY as evaluation annotations on a labeled, faded copy of t=0; these
positions are not inputs to JEPA. The privileged physical-state oracle is labeled.
All checkpoint panels retain their own n=128 Wilson intervals; there is no pooled
384-episode interval. ``figure_manifest.json`` records source hashes, captions,
selected frame times, action sequences, terminal events, and output hashes.

Example::

    /private/tmp/tinycraft-figenv/bin/python studies/revision/render_redcube_goals.py \\
        --run runs/revision/redcube_goals/m0 runs/revision/redcube_goals/m1 \\
        --model-labels 'Model seed 0' 'Model seed 1' \\
        --out output/revision/redcube-goal-analysis

Run ``--contract`` to print this contract without importing plotting libraries.
This module never trains, modifies, or evaluates a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CONTRACT = {
    "protocol": "redcube_goal_control_v1",
    "required_files": ["summary.json", "manifest.json", "cohort.npz", "episodes.npz", "visual_packet.npz"],
    "visual_episode_indices": [0, 1, 2],
    "display_policies": ["random", "jepa_h3", "oracle"],
    "success_policies": ["random", "fixed_cycle", "jepa_h1", "jepa_h3", "oracle"],
    "stopping_rule": "first exact goal hit or native death; otherwise 24 controlled actions",
    "rgb_transform": "none for strips and raw assets; faded initial frame only in explicitly annotated path diagrams",
    "forecast_rule": "first decision only, and only if forecast_valid is true; decoder visualization is not observation",
    "uncertainty": "Wilson 95% interval across 128 evaluation starts, separately per world-model checkpoint",
    "coordinate_annotation": "simulator cube upper-left position plus 2.5 pixels; diagnostic only",
    "selection_rule": "episodes 0, 1, 2 declared by evaluation protocol before policy outcomes",
}
INK, JEPA, JEPA_LIGHT, SIMULATOR, FLOW = "#17324D", "#CD663E", "#E3A389", "#087E83", "#7860A5"
GRAY, PALE = "#637583", "#dce4e8"
COLORS = {"random": GRAY, "fixed_cycle": "#b9bfc5", "jepa_h1": JEPA_LIGHT, "jepa_h3": JEPA, "oracle": SIMULATOR}
CONTRACT["annotation_palette"] = {"jepa": JEPA, "privileged_simulator": SIMULATOR, "flow_if_present": FLOW, "text": INK, "random": GRAY, "cycle": "#b9bfc5"}
LABELS = {"random": "Random", "fixed_cycle": "Fixed cycle", "jepa_h1": "JEPA · 1 step", "jepa_h3": "JEPA · 3 steps", "oracle": "State oracle"}
ARROWS = ("←", "↑", "→", "↓")
ACTION_NAMES = ("left", "up", "right", "down")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k, n):
    p, z = k / n, 1.959963984540054
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return np.asarray([max(0., c - h), min(1., c + h)])


def terminal(visual, policy, episode):
    hit = visual[policy + "_hit"][episode]
    alive = visual[policy + "_alive"][episode]
    end = np.flatnonzero(hit | ~alive)
    t = int(end[0]) if len(end) else len(hit) - 1
    event = "Goal hit" if hit[t] else "Died" if not alive[t] else "Time limit"
    return t, event


def load_run(directory, label):
    directory = Path(directory)
    for name in CONTRACT["required_files"]:
        if not (directory / name).is_file():
            raise ValueError(f"final artifact missing: {directory / name}")
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if summary["manifest"] != manifest:
        raise ValueError("manifest differs between final summary and manifest.json")
    assert manifest["protocol"] == CONTRACT["protocol"]
    assert manifest["n"] == 128 and manifest["control_steps"] == 24
    assert manifest["horizon"] == 3 and manifest["visual_indices_predeclared"] == [0, 1, 2]
    assert manifest["config"] in ("off_still", "gray_frame_still")
    for name, digest in summary["artifacts"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"artifact hash mismatch: {directory / name}")
    assert manifest["cohort_sha256"] == sha256(directory / "cohort.npz")
    with np.load(directory / "episodes.npz", allow_pickle=False) as f:
        episodes = {k: f[k] for k in f.files}
    with np.load(directory / "visual_packet.npz", allow_pickle=False) as f:
        visual = {k: f[k] for k in f.files}
    with np.load(directory / "cohort.npz", allow_pickle=False) as f:
        cohort = {k: f[k] for k in f.files}
    assert np.array_equal(visual["episode_indices"], [0, 1, 2])
    assert np.array_equal(visual["candidate_ids"], cohort["candidate_ids"][:3])
    assert np.array_equal(visual["goals_rgb"], cohort["goal_rgb"][:3])
    assert np.array_equal(visual["goal_positions"], cohort["goals"][:3])
    for policy in CONTRACT["success_policies"]:
        hit = episodes[policy + "_hit"]
        alive = episodes[policy + "_alive"]
        actions = episodes[policy + "_actions"]
        assert hit.shape == alive.shape == (128, 25) and actions.shape == (128, 24)
        assert not hit[:, 0].any() and alive[:, 0].all()
        assert (np.diff(hit.astype(int), axis=1) >= 0).all()
        assert (np.diff(alive.astype(int), axis=1) <= 0).all()
        expected_actions = alive[:, :-1] & ~hit[:, :-1]
        assert np.array_equal(actions >= 0, expected_actions)
        assert ((actions >= -1) & (actions <= 3)).all()
        for suffix in ("positions", "actions", "alive", "hit"):
            assert np.array_equal(visual[policy + "_" + suffix], episodes[policy + "_" + suffix][:3])
        assert np.array_equal(visual[policy + "_forecast_valid"], expected_actions[:3])
        assert np.array_equal(visual[policy + "_rgb"][:, 0], cohort["frames"][:3, -1])
        assert visual[policy + "_rgb"].dtype == np.uint8
        assert visual[policy + "_rgb"].shape == (3, 25, 63, 63, 3)
        k = int(hit[:, -1].sum())
        result = summary["policies"][policy]["exact_goal_hit"]
        assert result["count"] == k and result["n"] == 128
        np.testing.assert_allclose(result["fraction"], k / 128, atol=1e-14, rtol=0)
        np.testing.assert_allclose(result["wilson_95"], wilson(k, 128), atol=1e-14, rtol=0)
        for ep in range(3):
            stop, _ = terminal(visual, policy, ep)
            # Padding stays frozen; it will never be displayed.
            assert np.array_equal(visual[policy + "_rgb"][ep, stop:],
                                  np.broadcast_to(visual[policy + "_rgb"][ep, stop], (25 - stop, 63, 63, 3)))
            assert np.array_equal(visual[policy + "_positions"][ep, stop:],
                                  np.broadcast_to(visual[policy + "_positions"][ep, stop], (25 - stop, 2)))
    return {"directory": directory, "label": label, "summary": summary, "manifest": manifest,
            "episodes": episodes, "visual": visual, "cohort": cohort,
            "source_hashes": {n: sha256(directory / n) for n in CONTRACT["required_files"]}}


def setup_plotting():
    global plt, Rectangle, Line2D, Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from PIL import Image
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": PALE, "xtick.color": INK,
        "ytick.color": INK, "axes.titleweight": "bold", "axes.titlesize": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "axes.linewidth": .65,
    })


def export(fig, out, stem, caption, metadata):
    files = {}
    for ext in ("svg", "pdf", "png"):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, dpi=240, facecolor="white")
        files[ext] = {"path": path.name, "sha256": sha256(path)}
    size = list(map(float, fig.get_size_inches()))
    plt.close(fig)
    return {"caption": caption, "files": files, "size_inches": size, **metadata}


def success_figure(runs, out, stem="goal_success"):
    fig, axes = plt.subplots(1, len(runs), figsize=(max(5.5, 2.8 * len(runs)), 3.0),
                             squeeze=False, sharey=True)
    fig.subplots_adjust(left=.095, right=.98, bottom=.25, top=.77, wspace=.15)
    fig.text(.095, .95, "Visual-goal control in deterministic dynamics", weight="bold", fontsize=10)
    subtitle = ("Goals 6–12 moves away · 128 paired starts · 24-action limit" if len(runs) == 1
                else "Goals 6–12 moves away · exact hits within 24 actions · 128 paired starts per checkpoint")
    fig.text(.095, .88, subtitle, fontsize=8, color=GRAY)
    policies = CONTRACT["success_policies"]
    results = []
    for run, ax in zip(runs, axes[0]):
        for i, policy in enumerate(policies):
            stat = run["summary"]["policies"][policy]["exact_goal_hit"]
            p = stat["fraction"]
            lo, hi = stat["wilson_95"]
            ax.bar(i, p, width=.63, color=COLORS[policy], zorder=2)
            ax.errorbar(i, p, yerr=np.array([[p - lo], [hi - p]]), color=INK, linewidth=.75, capsize=2, zorder=3)
            ax.text(i, min(1.075, hi + .045), f'{stat["count"]}/128', ha="center", fontsize=6.7)
            results.append({"checkpoint": run["label"], "model_sha256": run["manifest"]["model_sha256"],
                            "policy": policy, **stat})
        ax.set_title(run["label"], pad=12)
        ax.set_ylim(0, 1.17)
        ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100"])
        ax.set_xticks(range(len(policies)), ["Random", "Cycle", "JEPA\n1 step", "JEPA\n3 steps", "State\noracle"], fontsize=7)
        ax.tick_params(length=0, pad=5)
        ax.grid(axis="y", color=PALE, linewidth=.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel("Exact goal hit (%)")
    caption = ("A frozen JEPA model chooses actions from RGB history and an RGB goal by exhaustive one- or three-step "
               "latent-distance search. The three-step cost sums all predicted stage distances. Random and fixed-cycle "
               "policies use the same 128 starts and goals; the state oracle is a privileged reference with physical-position "
               "access. Goals are 6–12 cardinal actions away. Bars show exact goal hits before native death or the 24-action "
               "limit; whiskers are Wilson 95% intervals across starts, separately for each checkpoint. Model-seed results "
               "are not pooled into a larger binomial sample. This is empty-arena navigation, not strategic planning.")
    return export(fig, out, stem, caption, {"results": results})


def compact_success_figure(runs, out, record):
    """Keep the complete three-checkpoint result legible at ICLR text width."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5, 3.0), squeeze=False, sharey=True)
    fig.subplots_adjust(left=.20, right=.975, bottom=.18, top=.73, wspace=.15)
    fig.text(.035, .955, "Visual-goal control in deterministic dynamics", fontsize=10, weight="bold", va="top")
    fig.text(.035, .87, "Goals 6–12 moves away · 128 paired starts · 24-action limit", fontsize=7.7, color=GRAY, va="top")
    policies = CONTRACT["success_policies"]
    for run, ax in zip(runs, axes[0]):
        for i, policy in enumerate(policies):
            y = len(policies) - 1 - i
            stat = run["summary"]["policies"][policy]["exact_goal_hit"]
            p = stat["fraction"]
            lo, hi = stat["wilson_95"]
            ax.barh(y, p, height=.61, color=COLORS[policy], zorder=2)
            ax.errorbar(p, y, xerr=np.array([[p - lo], [hi - p]]), color=INK,
                        linewidth=.75, capsize=2, zorder=3)
            inside = p > .75
            ax.text(p - .035 if inside else hi + .035, y, f'{stat["count"]}/128',
                    ha="right" if inside else "left", va="center", fontsize=6.9,
                    color="white" if inside else INK, weight="bold" if inside else "normal")
        ax.set_title(run["label"], fontsize=8, pad=7)
        ax.set_xlim(0, 1.08)
        ax.set_ylim(-.65, 4.65)
        ax.set_xticks([0, .5, 1], ["0", "50", "100"], fontsize=7)
        ax.set_yticks(range(4, -1, -1), ["Random", "Fixed cycle", "JEPA · 1 step", "JEPA · 3 steps", "State oracle"], fontsize=7.2)
        ax.tick_params(length=0, pad=5)
        ax.grid(axis="x", color=PALE, linewidth=.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.text(.59, .075, "Exact goal hit (%)", ha="center", fontsize=8)
    return export(fig, out, "goal_success", record["caption"],
                  {"results": record["results"], "layout": "fixed 5.5-inch publication width"})


def paths_figure(run, out, stem):
    v = run["visual"]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.75))
    fig.subplots_adjust(left=.035, right=.985, top=.74, bottom=.24, wspace=.10)
    fig.text(.035, .955, f'Actual paths · {run["label"]}', fontsize=10, weight="bold", va="top")
    fig.text(.035, .885, "Episodes 0, 1, 2, selected before outcomes · goal shown as dashed outline", fontsize=7.7, color=GRAY, va="top")
    records = []
    for ep, ax in enumerate(axes):
        ax.imshow(v["prefix_rgb"][ep, -1], interpolation="nearest", alpha=.24)
        goal = v["goal_positions"][ep]
        ax.add_patch(Rectangle(goal - .5, 6, 6, edgecolor=INK, facecolor="none", linewidth=1.0, linestyle="--", zorder=9))
        for policy in ("random", "oracle", "jepa_h3"):
            stop, event = terminal(v, policy, ep)
            path = v[policy + "_positions"][ep, :stop + 1] + 2.5
            # A white underlay makes overlapping colored traces distinguishable.
            ax.plot(*path.T, color="white", linewidth=2.4, alpha=.9, zorder=3)
            ax.plot(*path.T, color=COLORS[policy], linewidth=1.2, alpha=.95, zorder=4,
                    linestyle=":" if policy == "oracle" else "-", marker=".", markersize=2)
            ax.scatter(*path[-1], s=20, marker="o" if event == "Goal hit" else "x",
                       color=COLORS[policy], linewidth=.9, zorder=7)
            records.append({"episode": ep, "policy": policy, "stop_step": stop, "event": event})
        xy = v["jepa_h3_positions"][ep, 0] + 2.5
        ax.scatter(*xy, s=20, facecolors="white", edgecolors=INK, linewidth=.75, zorder=8)
        ax.set_title(f'Episode {ep}', fontsize=8.5, pad=5)
        ax.set_xlim(-.5, 62.5)
        ax.set_ylim(62.5, -.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    handles = [Line2D([0], [0], color=COLORS[p], linewidth=1.7, label=LABELS[p],
                      linestyle=":" if p == "oracle" else "-") for p in ("random", "jepa_h3", "oracle")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .09), ncol=3,
               frameon=False, fontsize=7.5, columnspacing=1.7)
    fig.text(.035, .055, "Position annotations on faded initial RGB; these coordinates are not JEPA inputs.", fontsize=7, color=GRAY)
    caption = ("Paired actual simulator paths for predeclared episodes 0, 1 and 2. The background is the initial RGB frame "
               "displayed at reduced opacity only to make the diagnostic coordinate overlays legible. Each path stops at its "
               "first exact goal hit, native death, or action limit. Open white circle marks the common start; a dashed square "
               "marks the target cube position. Colored endpoint circles denote hits and crosses denote other endings. "
               "Coordinates are evaluation annotations and are not provided to JEPA; the state oracle is privileged.")
    return export(fig, out, stem, caption, {"checkpoint": run["label"], "episodes": records, "rgb_transform": "alpha .24, path diagram only"})


def strip_times(stop):
    return sorted(set([t for t in (0, 4, 8, 12, 16, 20, 24) if t <= stop] + [stop]))


def raw_strip(run, out, stem, ep):
    v, policies = run["visual"], CONTRACT["display_policies"]
    fig = plt.figure(figsize=(5.5, 3.4))
    fig.text(.035, .955, f'Observed RGB · episode {ep} · {run["label"]}', fontsize=10, weight="bold", va="top")
    fig.text(.035, .89, "Unmodified frames; each row stops at the first hit, death or action limit", fontsize=7.7, color=GRAY, va="top")
    # Nine tiles: goal plus at most eight observed frames, all 63x63 source pixels.
    x0, tile, gap, row_top, row_gap = .87, .455, .035, .72, .82
    fig.text((x0 + tile / 2) / 5.5, .825, "RGB goal", ha="center", fontsize=7)
    fig.text((x0 + tile + 2 * gap) / 5.5, .825, "Observed step (preceding action)", fontsize=7)
    records = []
    for row, policy in enumerate(policies):
        stop, event = terminal(v, policy, ep)
        times = strip_times(stop)
        y = 1 - (row_top + tile + row * row_gap) / 3.4
        fig.text(.145, y + tile / 2 / 3.4, LABELS[policy], ha="right", va="center",
                 fontsize=7.1, weight="bold", color=COLORS[policy])
        frames = [v["goals_rgb"][ep], *v[policy + "_rgb"][ep, times]]
        for col, frame in enumerate(frames):
            x = x0 + col * (tile + gap) + (gap if col else 0)
            ax = fig.add_axes([x / 5.5, y, tile / 5.5, tile / 3.4])
            ax.imshow(frame, interpolation="nearest")
            ax.set_axis_off()
            if col:
                t = times[col - 1]
                action = "" if t == 0 else f" ({ARROWS[int(v[policy + '_actions'][ep, t - 1])]})"
                fig.text((x + tile / 2) / 5.5, y - .018, f"{t}{action}", ha="center", va="top", fontsize=6.8)
        actions = v[policy + "_actions"][ep, :stop].tolist()
        sequence = " ".join(ARROWS[a] for a in actions)
        fig.text(x0 / 5.5, y - .062, f"{event} at {stop}  ·  Actions: {sequence}", color=COLORS[policy], fontsize=6.8)
        records.append({"episode": ep, "candidate_id": int(v["candidate_ids"][ep]), "policy": policy,
                        "frame_times": times, "stop_step": stop, "event": event,
                        "action_indices": actions, "action_names": [ACTION_NAMES[a] for a in actions]})
    fig.text(.035, .038, "State oracle uses physical coordinates. JEPA observes only RGB history, actions and the RGB goal.", fontsize=6.7, color=GRAY)
    caption = (f"Unmodified RGB observations for predeclared episode {ep}; no success-based selection. The goal tile is "
               "the actual RGB image supplied to JEPA. Observed frames are sampled every four actions through each policy's "
               "terminal event, with the exact final frame included. Time labels show controlled actions since the common "
               "eight-frame prefix; parenthesized arrows show the action entering that frame. The complete executed action "
               "sequence is printed below each row. No padded frames after goal hit or death are displayed. Frame noise is "
               "the stored simulator RGB and is not smoothed or recolored. The state oracle is a privileged reference.")
    return export(fig, out, stem, caption, {"checkpoint": run["label"], "records": records, "rgb_transform": "none"})


def forecast_strip(run, out, stem, ep):
    v, policy, decision = run["visual"], "jepa_h3", 0
    key = policy + "_predicted_rgb"
    if key not in v or not v[policy + "_forecast_valid"][ep, decision]:
        return None
    forecast = v[key][ep, decision]
    if forecast.dtype != np.uint8 or forecast.shape != (3, 63, 63, 3):
        raise ValueError("unexpected stored three-step decoded forecast shape or dtype")
    plan = v[policy + "_selected_plan"][ep, decision].tolist()
    fig = plt.figure(figsize=(5.5, 1.85))
    fig.text(.035, .95, f'First selected forecast · episode {ep} · {run["label"]}', fontsize=9, weight="bold", va="top")
    fig.text(.035, .80, "Frozen decoder visualization of predicted latents; these frames are not observations", fontsize=7.2, color=GRAY, va="top")
    frames = [v[policy + "_rgb"][ep, 0], *forecast, v["goals_rgb"][ep]]
    names = ["Observed t=0", *[f"Predicted +{t + 1} ({ARROWS[plan[t]]})" for t in range(3)], "RGB goal"]
    for i, (frame, label) in enumerate(zip(frames, names)):
        ax = fig.add_axes([.05 + i * .19, .15, .155, .461])
        ax.imshow(frame, interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(label, fontsize=7, pad=3)
    fig.text(.035, .035, "Only the first selected action is executed; the controller replans after observing its result.", fontsize=7, color=GRAY)
    return export(fig, out, stem,
                  "The selected three-step JEPA prediction at the first valid decision of a predeclared episode, decoded "
                  "with the frozen auxiliary decoder. These are imagined latents under a candidate action sequence, not "
                  "recorded future observations. Only its first action is executed before replanning. No forecast from a "
                  "terminal/padded decision is displayed.",
                  {"checkpoint": run["label"], "episode": ep, "decision": 0, "forecast_valid": True,
                   "selected_plan": plan, "rgb_transform": "none"})


def save_raw_assets(run, out, slug):
    directory = out / "raw" / slug
    directory.mkdir(parents=True)
    v, records = run["visual"], []
    for ep in range(3):
        goal = directory / f"episode{ep}_goal.png"
        Image.fromarray(v["goals_rgb"][ep]).save(goal)
        for policy in CONTRACT["display_policies"]:
            stop, event = terminal(v, policy, ep)
            paths = []
            for t, frame in enumerate(v[policy + "_rgb"][ep, :stop + 1]):
                path = directory / f"episode{ep}_{policy}_t{t:02d}.png"
                Image.fromarray(frame).save(path)
                # Lossless asset identity, including uint8 shape and source RGB.
                assert np.array_equal(np.asarray(Image.open(path)), frame)
                paths.append(str(path.relative_to(out)))
            records.append({"episode": ep, "policy": policy, "goal": str(goal.relative_to(out)),
                            "frames": paths, "stop_step": stop, "event": event,
                            "actions": v[policy + "_actions"][ep, :stop].tolist()})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output/revision/redcube-goal-analysis"))
    parser.add_argument("--model-labels", nargs="+")
    parser.add_argument("--contract", action="store_true")
    args = parser.parse_args()
    if args.contract:
        print(json.dumps(CONTRACT, indent=2))
        return
    if not args.run:
        parser.error("at least one final --run directory is required")
    labels = args.model_labels or [f"Checkpoint {i + 1}" for i in range(len(args.run))]
    if len(labels) != len(args.run):
        parser.error("one --model-labels value is required per --run")
    runs = [load_run(path, label) for path, label in zip(args.run, labels)]
    if len({r["manifest"]["model_sha256"] for r in runs}) != len(runs):
        raise ValueError("duplicate model checkpoint: do not display it as model replication")
    for run in runs[1:]:
        if run["manifest"]["cohort_array_sha256"] != runs[0]["manifest"]["cohort_array_sha256"]:
            raise ValueError("checkpoint cohorts are not identical; this figure requires paired starts/goals")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    setup_plotting()
    if len(runs) > 1:
        wide = success_figure(runs, args.out, "goal_success_wide")
        figures = [compact_success_figure(runs, args.out, wide), wide]
    else:
        figures = [success_figure(runs, args.out)]
    assets = {}
    for i, run in enumerate(runs):
        slug = f"checkpoint{i}"
        figures.append(paths_figure(run, args.out, "goal_paths_" + slug))
        for ep in range(3):
            figures.append(raw_strip(run, args.out, f"goal_rgb_{slug}_episode{ep}", ep))
            forecast = forecast_strip(run, args.out, f"goal_forecast_{slug}_episode{ep}", ep)
            if forecast:
                figures.append(forecast)
        assets[slug] = save_raw_assets(run, args.out, slug)
    result = {"contract": CONTRACT, "renderer_sha256": sha256(__file__),
              "runs": [{"label": r["label"], "directory": str(r["directory"]),
                        "model_sha256": r["manifest"]["model_sha256"], "source_hashes": r["source_hashes"]} for r in runs],
              "figures": figures, "raw_assets": assets}
    (args.out / "figure_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "figures": len(figures), "validation": "PASS"}))


if __name__ == "__main__":
    main()
