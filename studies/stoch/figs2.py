"""The four figures of the short paper, from runs/stoch/gate/latent_check.npz and runs/stoch/summary.json (no JAX).

    /tmp/figenv/bin/python -m studies.stoch.figs2 --out studies/stoch/paper/figures

  hero.pdf      A: what the JEPA encodes and what it imagines from one pre-reveal context (true future, JEPA latent of
                the true future decoded, JEPA rollout, flow rollout); B: the decision the agent faces.
  contrast.pdf  walls passed per episode in the real gate for the MLP policy trained inside the JEPA, inside the flow
                model, in the real gate on the frozen JEPA latent and on true pixels (every final run, mean bar).
  discovery.pdf final survival of the convolutional policy on true pixels and on the frozen latent decoded to pixels,
                by encoder seed, against the 50 % discovery line.
  geometry.pdf  the wall's amplitude in the JEPA latent against the predictor's one-step residual, per rollout step.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42})
from studies.stoch.hero import FLOW, JEPA, REAL, REF, panel_a, strip  # noqa: E402

ENC = {0: "o", 1: "s", 2: "^"}


def clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6)


def hero(out, z, c):
    rows = [("true future", z["recorded"][c], REAL),
            ("JEPA latent of the\ntrue future, decoded", z["grid_recorded_future_encoded"][c], REF),
            ("JEPA rollout", z["grid_deterministic"][c], JEPA),
            ("flow rollout,\none sample", z["flow_sample"][c], FLOW)]
    fig = plt.figure(figsize=(7.0, 2.3))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.12, left=0.13, right=0.99, top=0.88, bottom=0.1)
    left = outer[0].subgridspec(len(rows), 1, hspace=0.12)
    axes = [fig.add_subplot(left[i]) for i in range(len(rows))]
    for ax, (name, frames, colour) in zip(axes, rows):
        ax.imshow(strip(frames), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=6.5, color=colour, labelpad=4)
    axes[0].set_title("eight steps from the same pre-reveal context", fontsize=7, loc="left")
    for t in range(8):
        axes[-1].text(t * 65 + 31, 70, f"{t + 1}", ha="center", va="top", fontsize=6, color="#444")
    axes[-1].text(-2, 70, "step", ha="right", va="top", fontsize=6, color="#444")
    ax_b = fig.add_subplot(outer[1])
    panel_a(ax_b, z, c)
    for ax, letter, dx in ((axes[0], "A", -0.32), (ax_b, "B", -0.02)):
        ax.text(dx, 1.05, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")
    fig.savefig(out / "hero.pdf"); fig.savefig(out / "hero.png", dpi=220)


def enc_of(key):
    return int(key[1]) if key.startswith("m") else 0


def dots(ax, x, items, colour, width=0.36):
    """items: list of (encoder seed, value); encoder seeds get their own sub-column and marker."""
    encs = sorted({e for e, _ in items})
    offs = np.linspace(-width, width, len(encs)) if len(encs) > 1 else np.zeros(1)
    for e, off in zip(encs, offs):
        vals = np.array([v for ee, v in items if ee == e])
        jit = np.linspace(-0.07, 0.07, len(vals)) if len(vals) > 1 else np.zeros(1)
        ax.scatter(x + off + jit, vals, s=13, marker=ENC[e], color=colour, linewidths=0, zorder=3, alpha=0.9)
    vals = np.array([v for _, v in items])
    ax.hlines(vals.mean(), x - width - 0.1, x + width + 0.1, color=colour, lw=1.4, zorder=2)
    return vals


def contrast(out, g):
    groups = [("trained inside\nthe JEPA", [(enc_of(k), r["final"]["walls_passed"]) for k, r in g["grid"].items() if r], JEPA),
              ("real gate,\nfrozen JEPA latent", [(enc_of(k), r["final"]["walls_passed"]) for k, r in g["real_latent"].items() if r], JEPA),
              ("real gate,\ntrue pixels", [(0, r["final"]["walls_passed"]) for r in g["real_mlp"].values() if r], REAL),
              ("trained inside\nthe flow model\n(one checkpoint)", [(0, r["final"]["walls_passed"]) for r in g["flow"].values() if r], FLOW)]
    fig, ax = plt.subplots(figsize=(7.0, 1.9), gridspec_kw=dict(left=0.07, right=0.99, top=0.9, bottom=0.27))
    for x, (name, items, colour) in enumerate(groups):
        vals = dots(ax, x, items, colour)
        ax.text(x, 4.35, f"mean {vals.mean():.2f}", ha="center", va="bottom", fontsize=6.5, color=colour)
        ax.text(x, 4.0, f"{len(vals)} runs", ha="center", va="bottom", fontsize=5.8, color="#666")
    rnd = g["baselines"]["random"]["walls_passed"]
    ax.axhline(rnd, color=REF, lw=0.6, ls=(0, (3, 2)))
    ax.text(3.45, rnd + 0.05, "random actions", ha="right", va="bottom", fontsize=5.5, color=REF)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([n for n, _, _ in groups], fontsize=6.5, linespacing=1.0)
    for label, (_, _, colour) in zip(ax.get_xticklabels(), groups):
        label.set_color(colour)
    ax.set_ylabel("walls passed per episode", fontsize=6.5)
    ax.set_ylim(-0.05, 4.9)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.tick_params(axis="x", length=0, pad=2)
    for e, m in ENC.items():
        ax.scatter([], [], marker=m, color="#444", s=13, linewidths=0, label=f"encoder seed {e}")
    ax.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=1, bbox_to_anchor=(1.0, 0.98), handletextpad=0.2, labelspacing=0.2)
    ax.set_title("the same MLP policy from four training sources, graded in the real gate", fontsize=7, loc="left")
    clean(ax)
    fig.savefig(out / "contrast.pdf"); fig.savefig(out / "contrast.png", dpi=220)


def discovery(out, g):
    groups = [("true pixels", [(0, r["final"]["survival"]) for r in g["real_seeds"].values() if r], REAL),
              ("frozen JEPA latent,\ndecoded to pixels", [(enc_of(k), r["final"]["survival"]) for k, r in g["real_decoded"].items() if r], JEPA)]
    fig, ax = plt.subplots(figsize=(7.0, 1.78), gridspec_kw=dict(left=0.08, right=0.99, top=0.88, bottom=0.24))
    for x, (name, items, colour) in enumerate(groups):
        vals = dots(ax, x, items, colour, width=0.28)
        ax.text(x, 1.06, f"{int((vals > 0.5).sum())} of {len(vals)} discover", ha="center", va="bottom", fontsize=6, color=colour)
    ax.axhline(0.5, color=REF, lw=0.6, ls=(0, (3, 2)))
    ax.text(1.5, 0.52, "discovery: final survival above 50%", ha="right", va="bottom", fontsize=5.5, color=REF)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([n for n, _, _ in groups], fontsize=6.5)
    for label, (_, _, colour) in zip(ax.get_xticklabels(), groups):
        label.set_color(colour)
    ax.set_ylabel("final survival, real gate", fontsize=6.5)
    ax.set_ylim(-0.03, 1.2)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(-0.55, 1.55)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_title("convolutional policy in the real gate; markers: encoder seed 0 (circle), 1 (square), 2 (triangle); frozen off-budget decoder", fontsize=7, loc="left")
    clean(ax)
    fig.savefig(out / "discovery.pdf"); fig.savefig(out / "discovery.png", dpi=220)


def geometry(out, g):
    gr = g["latent_check"]["grid"]
    steps = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(7.0, 1.78), gridspec_kw=dict(left=0.08, right=0.99, top=0.88, bottom=0.24))
    ax.plot(steps, gr["one_step_residual_sq"], color=REF, lw=1.1, marker="o", ms=2.5, label="teacher-forced one-step residual (total)")
    ax.plot(steps, gr["wall_amplitude_sq"], color=REAL, lw=1.1, marker="o", ms=2.5, label="wall present vs absent (the wall's amplitude)")
    ax.set_ylim(0, 0.30); ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("future step", fontsize=6.5)
    ax.set_ylabel("squared latent distance\n(sum over coordinates)", fontsize=6.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.set_title("grid JEPA, encoder 0, 64 pre-reveal contexts, recorded actions", fontsize=7, loc="left")
    clean(ax)
    fig.savefig(out / "geometry.pdf"); fig.savefig(out / "geometry.png", dpi=220)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", type=int, default=0)
    p.add_argument("--npz", type=Path, default=Path("runs/stoch/gate/latent_check.npz"))
    p.add_argument("--summary", type=Path, default=Path("runs/stoch/summary.json"))
    p.add_argument("--out", type=Path, default=Path("studies/stoch/paper/figures"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.npz)
    g = json.loads(a.summary.read_text())["gate"]
    hero(a.out, z, a.context)
    contrast(a.out, g)
    discovery(a.out, g)
    geometry(a.out, g)
    print("wrote hero, contrast, discovery, geometry ->", a.out)


if __name__ == "__main__":
    main()
