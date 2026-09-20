"""Charts for the Craftax half of the post, from the measured artifacts (nothing hand-typed)."""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                            "axes.spines.right": False, "figure.facecolor": "#faf9f6",
                            "axes.facecolor": "#faf9f6", "savefig.facecolor": "#faf9f6"})
ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
OUT = ROOT / "blog" / "assets"
INK, RED, GREEN, GREY, BLUE, ORANGE = "#1c1c1c", "#c42828", "#2a9d8f", "#9a9a9a", "#3a6ea5", "#e08a1e"


def ledger():
    rows = [json.loads(line) for line in (ROOT / "ledger.jsonl").read_text().splitlines() if line.strip()]
    return {r["tag"]: r for r in rows}


def curves_figure():
    curves = json.loads((ART / "compare" / "curves.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, name in zip(axes, ("daylight", "all")):
        c = curves[name]
        steps = np.arange(1, len(c["jepa"]) + 1)
        ax.plot(steps, c["persistence"], color=GREY, lw=2, ls=":", label="copy the last frame")
        ax.plot(steps, c["jepa"], color=RED, lw=2.6, label="LeWM (JEPA), decoded rollout")
        ax.plot(steps, c["flow"], color=GREEN, lw=2.6, label="flow-matching WM (ours)")
        ax.set_xlabel("imagined step")
        ax.set_title(f"{name} windows" if name == "daylight" else "all 512 windows", loc="left", fontsize=13, color=INK)
        ax.set_xticks([1, 4, 8, 12, 16])
    axes[0].set_ylabel("pixel MSE to the real future")
    axes[0].legend(frameon=False, fontsize=10.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "craftax_curves.png", dpi=170)
    plt.close(fig)


def benchmark_figure():
    led = ledger()
    rows = [("copy last frame", led["persistence"]["gFVD"], led["persistence"]["AF"], GREY),
            ("LeWM, class token\n(as in the paper)", led["lewm_s0_lr1e-3"]["gFVD"], led["lewm_s0_lr1e-3"]["AF"], RED),
            ("LeWM, 81x32 grid latent", led["lewm_s0_grid"]["gFVD"], led["lewm_s0_grid"]["AF"], RED),
            ("flow-matching WM (ours)", 0.001246, 0.920, GREEN)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    y = np.arange(len(rows))[::-1]
    for ax, key, title in ((a, 1, "gFVD — lower is better (log scale)"), (b, 2, "action following — higher is better")):
        vals = [r[key] for r in rows]
        ax.barh(y, vals, color=[r[3] for r in rows], height=0.6)
        ax.set_yticks(y, [r[0] for r in rows])
        ax.set_title(title, loc="left", fontsize=13, color=INK)
        for yy, v in zip(y, vals):
            ax.text(v * (1.06 if key == 1 else 1.0) + (0 if key == 1 else 0.01), yy,
                    f"{v:.5f}" if key == 1 else f"{v:.3f}", va="center", fontsize=11, color=INK)
        if key == 1:
            ax.set_xscale("log")
            ax.set_xlim(5e-4, 8e-2)
        else:
            ax.set_xlim(0, 1.12)
    fig.tight_layout()
    fig.savefig(OUT / "craftax_benchmark.png", dpi=170)
    plt.close(fig)


def planning_figure():
    goal_cls = json.loads((ART / "s0_lr1e-3_h4_plan.json").read_text())["craftax"]
    goal_grid = json.loads((ART / "s0_grid_h4_plan.json").read_text())["craftax"]
    reward = json.loads((ART / "s0_grid_reward.json").read_text())["reward_mpc"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    names = ["random actions", "LeWM class token", "LeWM grid latent"]
    vals = [goal_grid["random"]["reached"], goal_cls["planned"]["reached"], goal_grid["planned"]["reached"]]
    a.bar(names, vals, color=[GREY, RED, RED], width=0.55)
    for i, v in enumerate(vals):
        a.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=12, color=INK)
    a.set_ylim(0, 1.05)
    a.set_ylabel("episodes landing on the goal tile")
    a.set_title("Planning to a goal image (h = 4): works", loc="left", fontsize=13, color=INK)

    order = [("random\nactions", "random", GREY), ("myopic\nargmax r̂", "myopic", RED),
             ("greedy\n1 step + V̂", "greedy", RED), ("CEM\nhorizon 6", "plan", RED)]
    vals = [reward[k]["return"] for _, k, _ in order]
    errs = [reward[k]["return_std"] / np.sqrt(reward["envs"]) for _, k, _ in order]
    b.bar([o[0] for o in order], vals, yerr=errs, color=[o[2] for o in order], width=0.55, capsize=3)
    b.axhline(reward["behaviour_policy"], color=BLUE, lw=2, ls="--")
    b.text(3.35, reward["behaviour_policy"] - 0.25, "policy that collected the data", ha="right", va="top", fontsize=10.5, color=BLUE)
    for i, v in enumerate(vals):
        b.text(i, v + 0.25, f"{v:.2f}", ha="center", fontsize=12, color=INK)
    b.set_ylabel(f"real return over {reward['steps']} steps")
    b.set_title("Planning for reward in the same latent: does not", loc="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "craftax_planning.png", dpi=170)
    plt.close(fig)


def imagined_figure():
    reward = json.loads((ART / "s0_grid_reward.json").read_text())["reward_mpc"]
    order = [("random", "random actions"), ("myopic", "myopic"), ("greedy", "greedy"), ("plan", "CEM h=6")]
    imagined = [reward[k]["imagined_reward_mean"] for k, _ in order]
    realized = [reward[k]["reward_per_step"] * (reward["horizon"] if k == "plan" else 1) for k, _ in order]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(order))
    ax.bar(x - 0.18, imagined, 0.36, color=RED, label="reward the planner imagines")
    ax.bar(x + 0.18, realized, 0.36, color=GREEN, label="reward it actually collects")
    for i, (im, re) in enumerate(zip(imagined, realized)):
        ax.text(i - 0.18, im + 0.03, f"{im:.2f}", ha="center", fontsize=10.5, color=RED)
        ax.text(i + 0.18, re + 0.03, f"{re:.3f}", ha="center", fontsize=10.5, color=GREEN)
    ax.set_xticks(x, [n for _, n in order])
    ax.set_ylabel("reward per decision (CEM: per 6-step plan)")
    ax.legend(frameon=False, fontsize=10.5)
    ax.set_title("Imagined vs realised reward under each policy", loc="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "craftax_imagined.png", dpi=170)
    plt.close(fig)


def reward_trace_figure(scene=0):
    rec = json.loads((ART / "compare" / f"scene{scene}.json").read_text())
    steps = np.arange(1, len(rec["real_reward"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(steps, rec["real_reward"], color=GREY, width=0.6, label="real reward of the true actions")
    ax.plot(steps, rec["flow_reward"], color=GREEN, lw=2.4, marker="o", ms=4, label="flow-matching WM: its own reward head, on its rollout")
    if "jepa_reward" in rec:
        ax.plot(steps, rec["jepa_reward"], color=RED, lw=2.4, marker="o", ms=4, label="LeWM: off-budget reward head, on its rollout")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("imagined step")
    ax.set_ylabel("reward")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title(f"Reward along the imagined rollout, scene {scene}", loc="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"craftax_reward_scene{scene}.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    benchmark_figure()
    planning_figure()
    imagined_figure()
    if (ART / "compare" / "curves.json").exists():
        curves_figure()
        for s in range(4):
            if (ART / "compare" / f"scene{s}.json").exists():
                reward_trace_figure(s)
    print("ok", sorted(p.name for p in OUT.glob("craftax_*")))
