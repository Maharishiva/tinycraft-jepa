"""Figure 1 of the paper: the gate decision, what each model imagines from the same pre-reveal context, and what
policies trained inside each model do in the real gate.  Reads runs/stoch/gate/latent_check.npz (studies.stoch.latent_check)
and runs/stoch/summary.json (studies.stoch.summarize); no JAX.

    /tmp/figenv/bin/python -m studies.stoch.hero --context 0 --out studies/stoch/paper/figures/hero.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42})

REF, JEPA, HEAD, FLOW, REAL = "#8a8a8a", "#d1373b", "#7b4fb2", "#2f6fc9", "#111111"
STEPS = 8
WALL_SPEED = 3


def strip(frames, gap=2):
    """Frames (T, 63, 63, 3) side by side with a light gutter."""
    T, H, W, _ = frames.shape
    out = np.full((H, T * W + (T - 1) * gap, 3), 235, np.uint8)
    for t in range(T):
        out[:, t * (W + gap):t * (W + gap) + W] = frames[t]
    return out


def panel_b(ax_rows, z, c):
    top = z["gap"][c] == 0
    rows = [("true future\n(gap %s)" % ("top" if top else "bottom"), z["recorded"][c], REAL),
            ("other future\n(gap %s)" % ("bottom" if top else "top"), z["counterfactual"][c], REAL),
            ("JEPA latent of the\ntrue future, decoded", z["grid_recorded_future_encoded"][c], REF),
            ("JEPA rollout,\noriginal predictor", z["grid_deterministic"][c], JEPA),
            ("JEPA rollout,\nhead, one sample", z["grid_mean_of_8"][c] * 0 + z["grid_sample"][c], HEAD),
            ("JEPA rollout,\nhead, mean of 8", z["grid_mean_of_8"][c], HEAD),
            ("flow rollout,\nsample", z["flow_sample"][c], FLOW),
            ("flow rollout,\nmean of 8", z["flow_mean_of_8"][c], FLOW)]
    for ax, (name, frames, colour) in zip(ax_rows, rows):
        ax.imshow(strip(frames), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=6.5, color=colour, labelpad=4)
    ax_rows[0].set_title("what each model imagines next, from the same context", fontsize=7, loc="left")
    for t in range(STEPS):
        ax_rows[-1].text(t * 65 + 31, 70, f"{t + 1}", ha="center", va="top", fontsize=6, color="#444")
    ax_rows[-1].text(-2, 70, "step", ha="right", va="top", fontsize=6, color="#444")


def panel_a(ax, z, c):
    """The context's last frame and the two true futures two steps later: the decision the agent faces."""
    ctx = z["context"][c][-1]
    a, b = z["recorded"][c][1], z["counterfactual"][c][1]
    ax.imshow(strip(np.stack([ctx, a, b]), gap=6), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for i, label in enumerate(["now: wall solid,\nno gap yet", "two steps later:\ngap at the top", "or: gap at\nthe bottom"]):
        ax.text(i * 69 + 31, 66, label, ha="center", va="top", fontsize=5.4, color="#444")
    ax.set_title("   the decision", fontsize=7, loc="left")


def finals(runs, prefix=None, contains=None):
    out = []
    for k, v in runs.items():
        if v is None:
            continue
        if (prefix is None or k.startswith(prefix)) and (contains is None or contains in k):
            out.append(v["final"])
    return out


def panel_c(ax, g):
    groups = [("JEPA\norig.", finals(g["grid"], contains="_det_"), JEPA),
              ("head\nmean 8", finals(g["grid"], contains="_mean8_"), HEAD),
              ("head\nsample", finals(g["grid"], contains="_samp_"), HEAD),
              ("flow\nmean 8", finals(g["flow"], prefix="mean8"), FLOW),
              ("flow\nsample", finals(g["flow"], prefix="samp"), FLOW),
              ("real\npixels\nCNN", [r["final"] for r in g["real_seeds"].values() if r], REAL),
              ("real\npixels\nMLP", [r["final"] for r in g.get("real_mlp", {}).values() if r], REAL),
              ("real\nJEPA\nlatent", [r["final"] for r in g.get("real_latent", {}).values() if r], JEPA),
              ("real\ndecoded\nlatent", [r["final"] for r in g.get("real_decoded", {}).values() if r], JEPA)]
    xs = np.arange(len(groups))
    for x, (name, rows, colour) in zip(xs, groups):
        if not rows:
            ax.text(x, 0.1, "pending", ha="center", va="bottom", fontsize=6, color=colour, rotation=90)
            continue
        walls = np.array([r["walls_passed"] for r in rows])
        ax.bar(x, walls.mean(), width=0.62, color=colour, alpha=0.25, edgecolor=colour, linewidth=0.6)
        jitter = np.linspace(-0.18, 0.18, len(walls)) if len(walls) > 1 else np.zeros(1)
        ax.scatter(x + jitter, walls, s=9, color=colour, zorder=3, linewidths=0)
        surv = np.array([r["survival"] for r in rows])
        ax.text(x, 5.0, f"{100 * surv.mean():.0f}%", ha="center", va="top", fontsize=6.5, color=colour)
        ax.text(x, 4.6, f"n={len(rows)}", ha="center", va="top", fontsize=5.5, color="#666")
    o = g["oracle"]
    for key, label in (("h5_k4", "simulator search, h=5"), ("h8_k4", "simulator search, h=8")):
        if o.get(key):
            y = o[key]["oracle"]["walls_passed"]
            ax.axhline(y, color=REF, lw=0.7, ls=(0, (3, 2)))
            ax.text(-0.7, y + 0.04, label, ha="left", va="bottom", fontsize=5.3, color=REF)
    rnd = g["baselines"]["random"]["walls_passed"]
    ax.axhline(rnd, color=REF, lw=0.7)
    ax.text(2.5, rnd + 0.04, "random", ha="center", va="bottom", fontsize=5.3, color=REF)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _, _ in groups], fontsize=5.2, linespacing=0.95)
    for label, (_, _, colour) in zip(ax.get_xticklabels(), groups):
        label.set_color(colour)
    ax.set_ylabel("walls passed per episode", fontsize=6.5)
    ax.set_ylim(-0.05, 5.5)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.text(-0.72, 5.45, "survival to the cap, runs", ha="left", va="top", fontsize=5.3, color="#444")
    ax.set_title("   PPO from each source, graded in the real gate", fontsize=7, loc="left")
    ax.set_xlim(-0.75, len(groups) - 0.25)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", type=int, default=0)
    p.add_argument("--npz", type=Path, default=Path("runs/stoch/gate/latent_check.npz"))
    p.add_argument("--summary", type=Path, default=Path("runs/stoch/summary.json"))
    p.add_argument("--out", type=Path, default=Path("studies/stoch/paper/figures/hero.pdf"))
    a = p.parse_args()
    z = np.load(a.npz)
    g = json.loads(a.summary.read_text())["gate"]
    fig = plt.figure(figsize=(7.0, 2.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.2, left=0.115, right=0.99, top=0.93, bottom=0.085)
    left = outer[0].subgridspec(8, 1, hspace=0.1)
    right = outer[1].subgridspec(2, 1, height_ratios=[0.95, 2.4], hspace=0.5)
    ax_rows = [fig.add_subplot(left[i]) for i in range(8)]
    panel_b(ax_rows, z, a.context)
    ax_a = fig.add_subplot(right[0])
    panel_a(ax_a, z, a.context)
    ax_c = fig.add_subplot(right[1])
    panel_c(ax_c, g)
    for ax, letter, dx in ((ax_rows[0], "A", -0.3), (ax_a, "B", -0.04), (ax_c, "C", -0.12)):
        ax.text(dx, 1.02, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(a.out.with_suffix(".png"), dpi=220)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
