#!/usr/bin/env python3
"""Exact analytical illustrations; no empirical measurements or model outputs.

Run from any directory. This writes reproducible SVG, PDF, PNG and caption JSON.
All randomized texture uses a fixed seed and is illustrative only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


INK = "#17324d"
BLUE = "#3b6f9a"
TEAL = "#087e83"
ORANGE = "#cd663e"
RED = "#c9403e"
GRAY = "#637583"
GRID = "#dce4e8"
PALE = "#f2f5f6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def txt(ax, x, y, s, *, size=9, color=INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", **kw)


def arrow(ax, a, b, *, color=GRAY, lw=1.2, style="-|>", zorder=1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8,
                                linewidth=lw, color=color, zorder=zorder,
                                shrinkA=0, shrinkB=0))


def dot(ax, x, y, *, color=ORANGE, size=52, hollow=False, zorder=3):
    ax.scatter([x], [y], s=size, facecolors="white" if hollow else color,
               edgecolors=color, linewidths=1.5, zorder=zorder)


def panel(ax, letter, title, subtitle):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    txt(ax, 0.015, 0.96, letter, size=9.7, weight="bold", color=TEAL)
    txt(ax, 0.105, 0.96, title, size=9.2, weight="bold")
    txt(ax, 0.015, 0.885, subtitle, size=8.2, color=GRAY)


def ranking(ax):
    panel(ax, "a", "Goal scoring", "Perfect one-step mean, wrong action")
    xx = lambda x: 0.555 + 0.31 * x
    y_a, y_b = 0.69, 0.54
    ax.plot([xx(0), xx(0)], [0.485, 0.79], ls=(0, (2, 2)), color=TEAL, lw=1)
    txt(ax, xx(0), 0.812, "goal 0", size=8.5, color=TEAL, ha="center")
    txt(ax, 0.035, y_a, "A", weight="bold", color=ORANGE)
    txt(ax, 0.035, y_b, "B", weight="bold", color=BLUE)
    ax.plot([xx(-1), xx(1)], [y_a, y_a], color=ORANGE, lw=1.2, alpha=.6)
    dot(ax, xx(-1), y_a)
    dot(ax, xx(1), y_a)
    dot(ax, xx(0), y_a, color=ORANGE, hollow=True)
    txt(ax, xx(-1), y_a + .055, "−1", size=8.5, ha="center")
    txt(ax, xx(1), y_a + .055, "+1", size=8.5, ha="center")
    txt(ax, xx(0), y_a - .05, "mean", size=8, ha="center", color=ORANGE)
    dot(ax, xx(.1), y_b, color=BLUE)
    txt(ax, xx(.1) + .055, y_b, "0.1", size=8.5, color=BLUE)
    # Equal-width cost columns distinguish the two decision rules.
    xs = [0.065, 0.425, 0.81]
    txt(ax, .5, .455, "Squared goal loss", size=8, color=GRAY, ha="center")
    txt(ax, xs[1], .397, "At mean", size=8.5, weight="bold", ha="center")
    txt(ax, xs[2], .397, "Expected", size=8.5, weight="bold", ha="center")
    ax.plot([.025, .98], [.351, .351], color=GRID, lw=.8)
    for row_y, name, col, mean_cost, true_cost in [
        (.292, "A", ORANGE, "0", "1"),
        (.19, "B", BLUE, "0.01", "0.01"),
    ]:
        txt(ax, xs[0], row_y, name, color=col, weight="bold")
        txt(ax, xs[1], row_y, mean_cost, size=10, ha="center")
        txt(ax, xs[2], row_y, true_cost, size=10, ha="center")
    for x, y, col in [(xs[1], .292, ORANGE), (xs[2], .19, TEAL)]:
        ax.add_patch(FancyBboxPatch((x-.145, y-.042), .29, .084,
            boxstyle="round,pad=0.004,rounding_size=0.013",
            edgecolor=col, facecolor=col, linewidth=.8, alpha=.095, zorder=0))
        arrow(ax, (x+.145, y), (x+.102, y), color=col, lw=1.05)
    txt(ax, .50, .051, "Mean score selects A.\nExpected loss favors B.",
        size=8.4, ha="center", color=ORANGE, linespacing=1.3)


def rollout(ax):
    panel(ax, "b", "Recursive rollout", r"$x_{t+1}=x_t^2+\xi_t,\quad\xi_t=\pm1$ equally")
    xpos = [.15, .50, .85]
    # The true distribution is represented by its support, not a decoded mean.
    yp = lambda v: .59 + .075 * v
    for x, label in zip(xpos, [r"$t=0$", r"$t=1$", r"$t=2$"]):
        txt(ax, x, .805, label, size=8.3, ha="center", color=GRAY)
    for a, b in [(0, -1), (0, 1)]:
        arrow(ax, (xpos[0]+.035, yp(a)), (xpos[1]-.035, yp(b)), color=TEAL, lw=1)
    for a in [-1, 1]:
        for b in [0, 2]:
            arrow(ax, (xpos[1]+.035, yp(a)), (xpos[2]-.035, yp(b)), color=TEAL, lw=.9)
    for x, values in zip(xpos, [[0], [-1, 1], [0, 2]]):
        for v in values:
            dot(ax, x, yp(v), color=TEAL, size=44)
            off = .042 if v != 2 else -.045
            txt(ax, x, yp(v)+off, str(v), size=8.5, ha="center", color=TEAL)
    txt(ax, .50, .425, r"True mean: $\mathbb{E}[x_2]=1$", size=8.8,
        color=TEAL, weight="bold", ha="center")
    ax.plot([.04, .96], [.36, .36], color=GRID, lw=.8)
    txt(ax, .5, .305, r"Mean predictor $f(x)=x^2$", size=8.5, ha="center")
    for i, x in enumerate(xpos):
        dot(ax, x, .205, color=ORANGE, size=43, hollow=True)
        txt(ax, x, .144, "0", size=9, ha="center", color=ORANGE)
        if i < 2:
            arrow(ax, (x+.045, .205), (xpos[i+1]-.045, .205), color=ORANGE, lw=1.3)
    txt(ax, .5, .073, r"$f(f(0))=0\ne\mathbb{E}[x_2]$", size=9,
        color=ORANGE, ha="center")


def texture(ax):
    panel(ax, "c", "Representation", "Position changes; texture B is fixed")
    rng = np.random.default_rng(20260920)
    background = .44 + .23 * rng.uniform(size=(13, 13))
    texture_rgb = np.repeat(background[:, :, None], 3, axis=2)
    lefts = [.035, .36, .685]
    y, width, height = .605, .28, .21
    for i, x in enumerate(lefts):
        frame = texture_rgb.copy()
        frame[4:7, 1+4*i:4+4*i, :] = matplotlib.colors.to_rgb(RED)
        ax.imshow(frame, extent=[x, x+width, y, y+height], interpolation="nearest", aspect="auto", zorder=1)
        ax.add_patch(Rectangle((x, y), width, height, facecolor="none", edgecolor=GRID, lw=.7))
        arrow(ax, (x+width/2, y-.025), (x+width/2, .498), color=GRAY, lw=1)
        txt(ax, x+width/2, .443, r"$z=B$", size=10, ha="center", color=BLUE)
    txt(ax, .5, .535, "encoder", size=8.0, ha="center", color=GRAY,
        bbox=dict(facecolor="white", edgecolor="none", pad=.7))
    ax.plot([.04, .96], [.36, .36], color=GRID, lw=.8)
    txt(ax, .07, .282, "Prediction error", size=8.5)
    txt(ax, .92, .282, "0", size=10, color=TEAL, weight="bold", ha="right")
    txt(ax, .07, .19, "Latent covariance", size=8.5)
    txt(ax, .92, .19, r"$I$", size=10, color=TEAL, weight="bold", ha="right")
    txt(ax, .5, .051, "Different positions become\nindistinguishable goals.",
        size=8.4, ha="center", color=ORANGE, linespacing=1.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "submission/shared-v2/figures")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.1))
    fig.subplots_adjust(left=.014, right=.988, top=.99, bottom=.035, wspace=.105)
    ranking(axs[0]); rollout(axs[1]); texture(axs[2])
    for x in [.336, .668]:
        fig.lines.append(plt.Line2D([x, x], [.13, .89], transform=fig.transFigure, color=GRID, linewidth=.75))
    stem = args.out_dir / "redcube_concepts"
    def export(figure, output):
        for suffix in ["svg", "pdf", "png"]:
            figure.savefig(output.with_suffix("."+suffix), dpi=240,
                           metadata={"Creator": "Reproducible analytical illustration"} if suffix == "pdf" else None)
    export(fig, stem)
    plt.close(fig)
    caption = {
        "title": "Three gaps between prediction and control",
        "caption": "Analytical counterexamples, not learned-model outputs. (a) With goal 0, action A yields −1 or +1 with equal probability, while B yields 0.1. Squared distance at the conditional mean ranks A first (0 versus 0.01), but expected squared distance ranks B first (1 versus 0.01). (b) For x_{t+1}=x_t^2+ξ_t with independent equiprobable ξ_t∈{−1,+1} and x_0=0, the exact one-step mean is f(x)=x^2. Its two-step rollout is 0; the true two-step distribution is uniform on {0,2}, with mean 1. (c) Let the observation contain controlled position and an independent episode-static texture parameter B∼N(0,I_d). An encoder retaining only B admits zero prediction error and identity marginal covariance while discarding position. The texture images illustrate this construction and are not frames from a trained decoder.",
        "scope": ["Panels a and b concern conditional-mean state prediction and squared latent-goal scoring or recursive state rollout; they do not apply to all deterministic value-equivalent models.", "Panel c is an existence counterexample, not evidence that a particular optimizer finds it. Static nuisance counterexamples have prior precedent, including Sobal et al. (2022).", "No experiment results, confidence intervals, probes, fitted decoders, model checkpoints, or selected rollout samples are used in this figure."],
        "provenance": "Exact finite-support calculations from the user-supplied motivating examples. Illustrative texture generated with numpy default_rng seed 20260920.",
        "source": "studies/revision/make_redcube_concepts.py",
        "alt_text": "Three analytical mechanisms. A mean-only goal score selects the risky action despite higher expected squared loss. Iterating an exact one-step conditional mean gives zero where the true two-step mean is one. A texture-only latent predicts perfectly and has identity covariance but cannot distinguish cube positions.",
    }
    stem.with_suffix(".caption.json").write_text(json.dumps(caption, ensure_ascii=False, indent=2)+"\n")
    # ICLR has a 5.5-inch text block. Keep native 8--10 pt typography by using
    # two panels here rather than shrinking the three-panel HTML composition.
    paper, paper_axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    paper.subplots_adjust(left=.012, right=.99, top=.99, bottom=.04, wspace=.14)
    ranking(paper_axes[0]); rollout(paper_axes[1])
    paper.lines.append(plt.Line2D([.504, .504], [.14, .86], transform=paper.transFigure,
                                  color=GRID, linewidth=.75))
    paper_stem = stem.with_name("redcube_concepts_paper")
    export(paper, paper_stem)
    plt.close(paper)
    paper_caption = dict(caption)
    paper_caption["title"] = "A perfect conditional mean is not a distribution"
    paper_caption["caption"] = caption["caption"].split(" (c) ")[0]
    paper_caption["scope"] = [caption["scope"][0], caption["scope"][2]]
    paper_caption["alt_text"] = "Two analytical counterexamples. A mean-only goal score selects action A, whose expected squared loss is 1 rather than action B's 0.01. Iterating an exact one-step mean gives zero where the true two-step mean is one."
    paper_caption["layout"] = "Native width 5.5 inches, height 2.8 inches; typography at least 8 pt. Include at the full ICLR text width without additional scaling."
    paper_stem.with_suffix(".caption.json").write_text(json.dumps(paper_caption, ensure_ascii=False, indent=2)+"\n")
    print(stem)
    print(paper_stem)


if __name__ == "__main__":
    main()
