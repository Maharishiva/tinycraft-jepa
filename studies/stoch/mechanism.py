"""Figures 2 and 3 of the paper, from runs/stoch/summary.json and the PPO ledgers (no JAX):

  discovery  survival in the real gate against PPO updates for every run: PPO inside the JEPA in its three predictor
             modes, inside the flow model, and in the real environment (one line per run).
  geometry   (a) the wall's amplitude in the JEPA latent against the predictor's one-step residual, per rollout step,
             with the rollouts' distance to the recorded future; (b) the flow model's gap coordinate for a sample and
             for the mean of 8; (c) the heads' termination AUC on wall deaths against border deaths.

    /tmp/figenv/bin/python -m studies.stoch.mechanism discovery geometry --out studies/stoch/paper/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42})
from studies.stoch.hero import FLOW, HEAD, JEPA, REAL, REF  # noqa: E402
from studies.stoch.summarize import flow_run_name, grid_run_name  # noqa: E402

RUNS = Path("runs/stoch/gate")


def ledger(name):
    path = RUNS / name / "ledger.jsonl"
    if not path.exists() or not Path(str(path.parent) + ".done").exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    x = [r.get("upd", r.get("iter", i + 1)) for i, r in enumerate(rows)]
    return np.asarray(x, float), np.asarray([r["survival"] for r in rows])


def discovery(out):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.35), sharey=True, gridspec_kw=dict(left=0.07, right=0.99, top=0.85, bottom=0.22, wspace=0.12))
    panels = [("inside the grid JEPA", [(grid_run_name(ms, mode, ps), {"det": JEPA, "mean8": HEAD, "samp": HEAD}[mode], {"det": "-", "mean8": "--", "samp": ":"}[mode])
                                              for ms in range(3) for mode in ("det", "mean8", "samp") for ps in range(3)]),
              ("inside the flow model", [(flow_run_name(mode, ps), FLOW, {"samp": "-", "mean8": "--"}[mode]) for mode in ("samp", "mean8") for ps in range(8)]),
              ("real pixels: CNN 3, MLP 3 runs", [(f"ppo_{n}", REAL, "-") for n in ("real", "real_s1", "real_s2")] + [(f"ppo_real_mlp_p{ps}", REAL, ":") for ps in range(3)]),
              ("JEPA latent: raw 9, decoded 9 runs", [(f"ppo_real_jepa_latent_p{ps}" if ms == 0 else f"ppo_real_jepa_latent_m{ms}_p{ps}", JEPA, "-") for ms in range(3) for ps in range(3)]
                                        + [(f"ppo_real_jepa_decoded_m{ms}_p{ps}", HEAD, ":") for ms in range(3) for ps in range(3)])]
    for ax, (title, runs) in zip(axes, panels):
        n = 0
        for name, colour, ls in runs:
            r = ledger(name)
            if r is None:
                continue
            x, y = r
            x = x / x.max()
            ax.plot(x, y, color=colour, ls=ls, lw=0.9, alpha=0.85)
            n += 1
        ax.set_title(title if "runs" in title else f"{title} ({n} runs)", fontsize=6.5, loc="left")
        ax.axhline(0.5, color=REF, lw=0.5, ls=(0, (3, 2)))
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of the PPO budget", fontsize=6.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("survival, real gate", fontsize=6.5)
    axes[0].plot([], [], color=JEPA, ls="-", lw=0.9, label="original predictor")
    axes[0].plot([], [], color=HEAD, ls="--", lw=0.9, label="residual head, mean of 8")
    axes[0].plot([], [], color=HEAD, ls=":", lw=0.9, label="residual head, sample")
    axes[0].legend(fontsize=5.5, frameon=False, loc="upper left")
    axes[1].plot([], [], color=FLOW, ls="-", lw=0.9, label="sample")
    axes[1].plot([], [], color=FLOW, ls="--", lw=0.9, label="mean of 8")
    axes[1].legend(fontsize=5.5, frameon=False, loc="upper left")
    axes[2].plot([], [], color=REAL, ls="-", lw=0.9, label="convolutional torso")
    axes[2].plot([], [], color=REAL, ls=":", lw=0.9, label="MLP torso")
    axes[2].legend(fontsize=5.5, frameon=False, loc="upper left")
    axes[3].plot([], [], color=JEPA, ls="-", lw=0.9, label="frozen latent")
    axes[3].plot([], [], color=HEAD, ls=":", lw=0.9, label="latent decoded to pixels")
    axes[3].legend(fontsize=5.5, frameon=False, loc="upper left")
    fig.savefig(out / "discovery.pdf")
    fig.savefig(out / "discovery.png", dpi=220)


def geometry(out, g):
    c, h = g["latent_check"], g["heads_audit"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.5), gridspec_kw=dict(left=0.075, right=0.99, top=0.86, bottom=0.22, wspace=0.34, width_ratios=[1.45, 1, 0.75]))
    steps = np.arange(1, 9)
    ax = axes[0]
    gr = c["grid"]
    series = [(gr["deterministic"]["to_recorded_sq"], JEPA, "-", "original rollout,\ndistance to the true future"),
              (gr["mean_of_8"]["to_recorded_sq"], HEAD, "--", "residual head, mean of 8,\ndistance to the true future"),
              (gr["one_step_residual_sq"], REF, "-", "predictor's one-step error\n(total, teacher forced)"),
              (gr["wall_amplitude_sq"], REAL, "-", "the wall's amplitude:\nlatent of the true future,\nwall present vs absent")]
    for (ys, colour, ls, label), va, dy in zip(series, ("center", "center", "bottom", "top"), (1.0, 1.0, 1.15, 0.9)):
        ax.plot(steps, ys, color=colour, lw=1.0, ls=ls, marker="o", ms=2.3)
        ax.text(8.25, ys[-1] * dy, label, color=colour, fontsize=5.3, va=va, ha="left")
    ax.set_yscale("log")
    ax.set_xlim(0.7, 13.5)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("rollout step", fontsize=6.5)
    ax.set_ylabel("squared latent distance", fontsize=6.5)
    ax.set_title("grid JEPA: wall amplitude vs one-step error", fontsize=7, loc="left")
    ax = axes[1]
    fl = c["flow"].get("counterfactual_survives", c["flow"])
    ax.plot(steps, fl["sample"]["gap_coordinate_abs"], color=FLOW, lw=1.0, marker="o", ms=2.5, label="one sample")
    ax.plot(steps, fl["mean_of_8"]["gap_coordinate_abs"], color=FLOW, lw=1.0, ls="--", marker="o", ms=2.5, label="mean of 8")
    ax.axhline(1.0, color=REF, lw=0.6, ls=(0, (3, 2)))
    ax.axhline(0.0, color=REF, lw=0.6, ls=(0, (3, 2)))
    ax.text(8.1, 1.0, "a real gap", fontsize=5.5, color=REF, va="bottom", ha="right")
    ax.text(8.1, 0.0, "half-way between", fontsize=5.5, color=REF, va="bottom", ha="right")
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel("rollout step", fontsize=6.5)
    ax.set_ylabel("|gap coordinate| of the flow latent", fontsize=6.5)
    ax.set_title("flow model: the mean commits in 3 steps", fontsize=7, loc="left")
    ax.legend(fontsize=5.5, frameon=False, loc="center right")
    ax = axes[2]
    r = h["real"]
    vals = [r["done_auc_border_vs_alive"], r["done_auc_wall_vs_alive"]]
    ax.bar([0, 1], vals, width=0.6, color=[REF, JEPA], alpha=0.85)
    for x, v in zip([0, 1], vals):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0.5, color=REAL, lw=0.6, ls=(0, (3, 2)))
    ax.text(1.35, 0.5, "chance", fontsize=5.5, va="bottom", ha="right", color="#444")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"border deaths\n(n={r['border_deaths']})", f"wall deaths\n(n={r['wall_deaths']})"], fontsize=6)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("termination head AUC vs alive", fontsize=6.5)
    ax.set_title("heads: border vs wall deaths", fontsize=7, loc="left")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=6)
    fig.savefig(out / "geometry.pdf")
    fig.savefig(out / "geometry.png", dpi=220)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("which", nargs="+", choices=["discovery", "geometry"])
    p.add_argument("--summary", type=Path, default=Path("runs/stoch/summary.json"))
    p.add_argument("--out", type=Path, default=Path("studies/stoch/paper/figures"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    g = json.loads(a.summary.read_text())["gate"]
    if "discovery" in a.which:
        discovery(a.out)
    if "geometry" in a.which:
        geometry(a.out, g)
    print("wrote", a.which, "->", a.out)


if __name__ == "__main__":
    main()
