"""Figures for the toy examples: what a deterministic MSE predictor does to a stochastic world.

Everything here is closed form - no training - because the point is that the failure survives
perfect optimisation.  Outputs go to blog/assets.
"""
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                            "axes.spines.right": False, "figure.facecolor": "#faf9f6",
                            "axes.facecolor": "#faf9f6", "savefig.facecolor": "#faf9f6"})
OUT = Path(__file__).resolve().parent / "assets"
INK, RED, GREEN, GREY, BLUE = "#1c1c1c", "#c42828", "#2a9d8f", "#9a9a9a", "#3a6ea5"
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------- the fork
def fork_paths(n, steps=16, noise=0.12):
    """Action A: run at the wall, get thrown up or down with probability 1/2.  Ends at (10, +-2)."""
    x = np.linspace(0, 10, steps + 1)
    side = rng.choice([-1.0, 1.0], size=n)
    y = np.zeros((n, steps + 1))
    for t in range(1, steps + 1):
        target = side * 2.0 * np.clip((x[t] - 3.0) / 2.0, 0, 1)
        y[:, t] = target + noise * rng.standard_normal(n) * (x[t] > 3.0)
    return x, y


def fork_b(steps=16):
    """Action B: crawl along the safe lane, deterministically ending at (10, 0.6)."""
    x = np.linspace(0, 10, steps + 1)
    return x, 0.6 * np.clip(x / 4.0, 0, 1)


def draw_fork(ax, x, ys, xb, yb, upto, show_mean=True):
    ax.axvspan(3.0, 5.0, color="#ebe7dd", zorder=0)
    ax.text(4.0, 2.55, "gust: up or down, p = ½", ha="center", color=GREY, fontsize=10.5)
    ax.scatter([10], [0], marker="*", s=300, color=INK, zorder=7)
    ax.text(10.35, -0.08, "goal g", ha="left", va="center", color=INK, fontsize=11)
    for y in ys[:80]:
        ax.plot(x[:upto + 1], y[:upto + 1], color=GREEN, alpha=0.18, lw=1.2, zorder=2)
    ax.scatter(np.full(min(len(ys), 80), x[upto]), ys[:80, upto], s=14, color=GREEN, alpha=0.6, zorder=3)
    ax.plot(xb[:upto + 1], yb[:upto + 1], color=BLUE, lw=2.4, zorder=4)
    ax.scatter([xb[upto]], [yb[upto]], s=50, color=BLUE, zorder=5)
    if show_mean:
        mean = np.zeros_like(x)
        ax.plot(x[:upto + 1], mean[:upto + 1], color=RED, lw=2.6, ls="--", zorder=5)
        ax.scatter([x[upto]], [mean[upto]], s=60, color=RED, zorder=6)
    ax.set_xlim(-0.3, 11.6)
    ax.set_ylim(-3.3, 2.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def fork_figure():
    x, ys = fork_paths(400)
    xb, yb = fork_b()
    fig, (ax, bars) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.3, 1]})
    draw_fork(ax, x, ys, xb, yb, 16)
    ax.plot([], [], color=GREEN, lw=2, label="action A: real futures (Monte-Carlo)")
    ax.plot([], [], color=RED, lw=2.4, ls="--", label="action A: MSE-optimal rollout = the mean")
    ax.plot([], [], color=BLUE, lw=2.4, label="action B: safe lane, deterministic")
    ax.legend(loc="lower left", frameon=False, fontsize=10.5, bbox_to_anchor=(0.0, 0.02))
    ax.set_title("Two actions, one goal: A gets thrown to y = ±2, B crawls to y = 0.6", loc="left", fontsize=13, color=INK)

    model = [0.0, 0.36]
    truth = [4.0, 0.36]
    pos = np.arange(2)
    bars.bar(pos - 0.18, model, 0.36, color=RED, label="cost the planner sees\n(f(h,a) − g)²")
    bars.bar(pos + 0.18, truth, 0.36, color=GREEN, label="expected cost\nE[(z′ − g)²]")
    for p, (m, t) in enumerate(zip(model, truth)):
        bars.text(p - 0.18, m + 0.08, f"{m:.2f}", ha="center", color=RED, fontsize=11)
        bars.text(p + 0.18, t + 0.08, f"{t:.2f}", ha="center", color=GREEN, fontsize=11)
    bars.set_xticks(pos, ["action A", "action B"])
    bars.set_ylim(0, 4.8)
    bars.set_ylabel("cost to goal")
    bars.legend(frameon=False, fontsize=10, loc="upper right")
    bars.set_title("The planner picks A", loc="left", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "toy_fork.png", dpi=170)
    plt.close(fig)

    frames = []
    for upto in list(range(17)) + [16] * 6:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        draw_fork(ax, x, ys, xb, yb, upto)
        ax.set_title(f"step {upto}/16 — green: what really happens, red: the MSE-optimal rollout",
                     loc="left", fontsize=12, color=INK)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(OUT / "toy_fork.gif", frames, duration=1000 / 5, loop=0)


# ----------------------------------------------------------------------------- z^2 + xi
def square_figure():
    steps = 3
    states = {0: {0.0: 1.0}}
    for t in range(1, steps + 1):
        nxt = {}
        for z, p in states[t - 1].items():
            for xi in (-1.0, 1.0):
                nxt[z * z + xi] = nxt.get(z * z + xi, 0.0) + p / 2
        states[t] = nxt
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for t in range(steps):
        for z, p in states[t].items():
            for xi in (-1.0, 1.0):
                ax.plot([t, t + 1], [z, z * z + xi], color=GREEN, alpha=0.35, lw=1.4)
    for t in range(steps + 1):
        zs = np.array(list(states[t].keys()))
        ps = np.array(list(states[t].values()))
        ax.scatter(np.full(len(zs), t), zs, s=900 * ps, color=GREEN, zorder=5, alpha=0.9)
    mean = [sum(z * p for z, p in states[t].items()) for t in range(steps + 1)]
    ax.plot(range(steps + 1), mean, color=GREEN, lw=2.4, zorder=6, label=r"true mean  $\mathbb{E}[z_t]$ = 0, 0, 1, 2")
    ax.plot(range(steps + 1), [0] * (steps + 1), color=RED, lw=2.6, ls="--", zorder=7,
            label=r"rollout of the perfect one-step predictor  $f(z)=z^2$: 0, 0, 0, 0")
    for t in range(2, steps + 1):
        ax.annotate("", (t, mean[t]), (t, 0), arrowprops={"arrowstyle": "<->", "color": GREY, "lw": 1.2})
        ax.text(t + 0.05, mean[t] / 2, f"gap = {mean[t]:.0f}", color=GREY, fontsize=11)
    ax.set_xticks(range(steps + 1), [f"t={t}" for t in range(steps + 1)])
    ax.set_ylabel("z")
    ax.set_title(r"$z_{t+1} = z_t^2 + \xi_t$,   $\xi_t = \pm 1$ with p = ½   (dot area = probability)", loc="left", fontsize=13, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "toy_square.png", dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------- background
def observation(pos, texture, size=28, k=5):
    img = np.repeat(texture[..., None], 3, -1).copy()
    r, c = pos
    img[r:r + k, c:c + k] = (0.78, 0.16, 0.16)
    return img


def background_figure():
    texture = 0.55 + 0.18 * rng.standard_normal((28, 28))
    texture = np.clip(texture, 0, 1)
    obs = [observation((4, 5), texture), observation((17, 18), texture)]
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.9, 1.3])
    for i, o in enumerate(obs):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(o, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"goal image {i + 1}", fontsize=11, loc="left", color=INK)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("#d8d3c8")
    ax = fig.add_subplot(gs[:, 1])
    cloud = rng.standard_normal((500, 2))
    ax.scatter(cloud[:, 0], cloud[:, 1], s=10, color=GREY, alpha=0.45, label="other episodes: z = B ~ N(0, I)")
    ax.scatter([1.1], [0.6], s=260, color=RED, zorder=5, label="both goal images land HERE")
    ax.annotate("e(o₁) = e(o₂) = B", (1.1, 0.6), (1.5, 1.9), fontsize=12, color=RED,
                arrowprops={"arrowstyle": "->", "color": RED})
    ax.set_aspect("equal")
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-3.4, 3.4)
    ax.set_xlabel("latent dim 1")
    ax.set_ylabel("latent dim 2")
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    ax.set_title("Encoder that keeps only the texture", loc="left", fontsize=13, color=INK)
    ax = fig.add_subplot(gs[:, 2])
    feats = ["texture B\nconstant", "square x\nmoves"]
    var = [0.0, 1.0]
    ax.bar(feats, var, color=[RED, GREEN], width=0.55)
    ax.text(0, 0.06, "0\nprediction loss ✓\nGaussian marginal ✓\nanti-collapse ✓", ha="center", fontsize=10, color=RED)
    ax.text(1, 1.04, "> 0 — penalised", ha="center", fontsize=10, color=GREEN)
    ax.set_ylim(0, 1.35)
    ax.set_ylabel("E[ Var(z′ | h, a) ]")
    ax.set_title("The loss at the optimum", loc="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "toy_background.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fork_figure()
    square_figure()
    background_figure()
    print("ok", sorted(p.name for p in OUT.glob("toy_*")))
