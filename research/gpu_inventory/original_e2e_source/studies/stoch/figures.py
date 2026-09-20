"""Figures of the stochastic-control study that need no model (CPU): the worlds, a scripted gate episode, the
one-step theory witness, and the result charts from runs/stoch/summary.json.

    JAX_PLATFORMS=cpu python -m studies.stoch.figures worlds gate theory results --out studies/stoch/paper/figures
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from studies.stoch import env as S

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                            "axes.spines.right": False})
WORLDS = ["off_still", "off_wind", "gray_frame_still", "gray_frame_wind", "gray_episode_still", "gray_episode_wind", "gate"]
WORLD_NAMES = {"gate": "gate", "off_still": "black, still", "off_wind": "black, wind",
               "gray_frame_still": "noise / frame, still", "gray_frame_wind": "noise / frame, wind",
               "gray_episode_still": "noise / episode, still", "gray_episode_wind": "noise / episode, wind"}
OSC = [0, 2]


def scripted(spec, taken, pos=(19, 28), gap=0, seed=0):
    """Frames and states of one gate episode from a fixed start, under the given per-step actions, with the
    gap forced (the environment draws it at spawn; we overwrite it right after)."""
    _, step = S.make(spec)
    key = jax.random.PRNGKey(seed)
    state = S.State(jnp.asarray(pos, jnp.int32), jnp.zeros(2, jnp.int32), key, jnp.asarray(0, jnp.int32),
                    jnp.asarray(0, jnp.int32), jnp.asarray(S.NO_WALL, jnp.int32), jnp.asarray(gap, jnp.int32))
    frames, states, dead = [np.asarray(S.render(spec, state, key))], [state], False
    for t, a in enumerate(taken):
        key, ks = jax.random.split(key)
        obs, state, reward, done = step(state, jnp.asarray(a), ks)
        state = state._replace(gap=jnp.asarray(gap, jnp.int32))
        frames.append(np.asarray(obs))
        states.append(state)
        if bool(done):
            dead = True
            break
    return frames, states, dead


def upscale(img, k=3):
    return np.repeat(np.repeat(img, k, 0), k, 1)


def worlds(out):
    fig, axes = plt.subplots(1, 8, figsize=(14, 2.2))
    key = jax.random.PRNGKey(3)
    for ax, cfg in zip(axes, WORLDS[:-1]):
        spec = S.CONFIGS[cfg]
        reset, step = S.make(spec)
        obs, state = reset(key)
        for _ in range(3):
            key, ka, ks = jax.random.split(key, 3)
            obs, state, _, _ = step(state, jax.random.randint(ka, (), 0, 4), ks)
        ax.imshow(np.asarray(obs), interpolation="nearest")
        ax.set_title(WORLD_NAMES[cfg], fontsize=9)
        ax.axis("off")
    frames, _, _ = scripted(S.CONFIGS["gate"], OSC * 20, gap=0)
    axes[6].imshow(frames[26], interpolation="nearest")
    axes[6].set_title("gate: wall solid (t=26)", fontsize=9)
    axes[6].axis("off")
    axes[7].imshow(frames[30], interpolation="nearest")
    axes[7].set_title("gate: gap revealed (t=30)", fontsize=9)
    axes[7].axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_worlds.png", dpi=200)
    plt.close(fig)


def gate(out):
    """Three rows from the same pre-reveal state: oscillate (dies, gap top), react up after the reveal (survives,
    gap top), react down after the reveal (survives, gap bottom).  Columns t = 22..36 step 2."""
    spec = S.CONFIGS["gate"]
    osc = OSC * 20
    react_up = OSC * 14 + [1] * 5 + OSC * 6
    react_down = OSC * 14 + [3] * 5 + OSC * 6
    rows = [("hold the middle\ngap on top\n-> dies at t=34", osc, 0), ("react after the reveal\ngap on top\n-> lives", react_up, 0),
            ("react after the reveal\ngap at the bottom\n-> lives", react_down, 1)]
    ts = list(range(22, 37, 2))
    fig, axes = plt.subplots(3, len(ts) + 1, figsize=(1.25 * (len(ts) + 1), 1.35 * 3 + 0.5),
                             gridspec_kw={"width_ratios": [1.6] + [1] * len(ts), "wspace": 0.05, "hspace": 0.12})
    for r, (title, taken, g) in enumerate(rows):
        frames, states, dead = scripted(spec, taken, gap=g)
        axes[r, 0].axis("off")
        axes[r, 0].text(0.95, 0.5, title, ha="right", va="center", fontsize=8, transform=axes[r, 0].transAxes)
        for c, t in enumerate(ts):
            ax = axes[r, c + 1]
            if t < len(frames):
                ax.imshow(frames[t], interpolation="nearest")
            else:
                ax.imshow(np.zeros_like(frames[0]) + 40, interpolation="nearest")
                ax.text(31, 31, "dead", color="white", ha="center", va="center", fontsize=8)
            if r == 0:
                ax.set_title(f"t={t}", fontsize=8, pad=2)
            ax.axis("off")
    fig.suptitle("Gate world: the wall enters solid at t=22; the gap is drawn when the wall reaches x<=40 (t=29)", fontsize=9, y=0.995)
    fig.savefig(out / "fig_gate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def theory(out):
    """One-step witness: outcomes of action A are both safe, its mean is on the wall; outcomes of action B are
    both on the wall, its mean is in the gap."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, name, outcomes, mean, verdict in [
            (axes[0], "action A", [(0.5, 0.86), (0.5, 0.14)], (0.5, 0.5), "true: safe (both outcomes)\npredicted: dead (mean on the wall)"),
            (axes[1], "action B", [(0.5, 0.63), (0.5, 0.37)], (0.5, 0.5), "true: dead (both outcomes)\npredicted: safe (mean in the gap)")]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        gap = (0.42, 0.58) if name == "action B" else (0.46, 0.54)
        ax.add_patch(Rectangle((0.44, 0.0), 0.12, gap[0], color="0.75"))
        ax.add_patch(Rectangle((0.44, gap[1]), 0.12, 1 - gap[1], color="0.75"))
        if name == "action A":
            ax.add_patch(Rectangle((0.44, 0.80), 0.12, 0.12, color="white", ec="0.3", ls="--"))
            ax.add_patch(Rectangle((0.44, 0.08), 0.12, 0.12, color="white", ec="0.3", ls="--"))
            ax.add_patch(Rectangle((0.44, gap[0]), 0.12, gap[1] - gap[0], color="0.75"))
        for (x, y) in outcomes:
            ax.add_patch(Rectangle((x - 0.03, y - 0.03), 0.06, 0.06, color="#c0392b"))
            ax.add_patch(FancyArrowPatch((0.15, 0.5), (x - 0.04, y), arrowstyle="-|>", mutation_scale=12, color="#c0392b", lw=1.2))
        ax.add_patch(Rectangle((0.12, 0.47), 0.06, 0.06, color="#c0392b"))
        ax.add_patch(Rectangle((mean[0] - 0.03, mean[1] - 0.03), 0.06, 0.06, fill=False, ec="#2c3e50", lw=1.8, ls="--"))
        ax.add_patch(FancyArrowPatch((0.15, 0.5), (mean[0] - 0.04, mean[1]), arrowstyle="-|>", mutation_scale=12, color="#2c3e50", lw=1.2, ls="--"))
        ax.text(0.5, 1.04, name, ha="center", va="bottom", fontsize=11, weight="bold")
        ax.text(0.5, -0.02, verdict, ha="center", va="top", fontsize=9)
        ax.text(0.15, 0.58, "agent", ha="center", fontsize=8)
        ax.text(0.78, 0.5, "solid: outcomes (p = 1/2 each)\ndashed: MSE-optimal prediction", fontsize=8, va="center")
    fig.suptitle("Certainty-equivalent planning with the conditional mean picks B: death with probability 1 instead of 0", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_theory.png", dpi=200)
    plt.close(fig)


def results(out, summary):
    s = json.loads(Path(summary).read_text())

    def ctrl(w, arm, pol):
        c = w["control"].get(arm)
        return c.get(pol) if c else None

    def ppo(w, name):
        p = w["ppo"].get(name)
        return p["final"] if p else None

    cols = [("random", lambda w: w["baselines"] and w["baselines"]["random"]),
            ("centre\n(true state)", lambda w: w["baselines"] and w["baselines"]["centre"]),
            ("LeWM\ngoal MPC", lambda w: ctrl(w, "lewm6", "goal")),
            ("LeWM\nheads MPC", lambda w: ctrl(w, "lewm6", "plan")),
            ("LeWM\nPPO inside", lambda w: ppo(w, "jepa_lewm6")),
            ("grid JEPA\ngoal MPC", lambda w: ctrl(w, "tile_factored", "goal")),
            ("grid JEPA\nheads MPC", lambda w: ctrl(w, "tile_factored", "plan")),
            ("grid JEPA\nPPO inside", lambda w: ppo(w, "jepa_tile_factored")),
            ("grid+sampler\nheads MPC", lambda w: ctrl(w, "tile_factored_sampled", "plan")),
            ("grid+sampler\nPPO inside", lambda w: ppo(w, "jepa_tile_factored_sampled")),
            ("flow\nPPO pixels", lambda w: ppo(w, "dream_pixels")),
            ("flow\nPPO latent", lambda w: ppo(w, "dream_latent")),
            ("PPO\nreal env", lambda w: ppo(w, "real"))]
    colors = ["0.6", "0.4"] + ["#e67e22"] * 3 + ["#c0392b"] * 3 + ["#8e44ad"] * 2 + ["#2980b9"] * 2 + ["0.2"]
    # gate: walls passed and deaths per 100 steps (survival to 128 steps is rare for every controller)
    w = s["gate"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
    for ax, key, label in [(axes[0], "walls_passed", "walls passed per episode"), (axes[1], "deaths_per_100", "deaths per 100 steps")]:
        vals = [100 * 0 + (f(w) or {}).get(key, np.nan) if f(w) else np.nan for _, f in cols]
        ax.bar(range(len(cols)), vals, color=colors)
        for j, (_, f) in enumerate(cols):
            r = f(w)
            if r:
                ax.text(j, vals[j], f"{100 * r['survival']:.0f}%" if key == "walls_passed" else f"{r['length']:.0f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.replace("\n", " ") for c, _ in cols], fontsize=7, rotation=28, ha="right")
        ax.set_ylabel(label)
    axes[0].set_title("Gate world: walls passed (label: episodes surviving 128 steps, %)", fontsize=9)
    axes[1].set_title("Gate world: deaths per 100 steps (label: mean episode length)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_gate_results.png", dpi=200)
    plt.close(fig)
    # cube heatmap (the sampling-predictor cells are gate-only)
    cubes = ["off_still", "off_wind", "gray_frame_still", "gray_frame_wind", "gray_episode_still", "gray_episode_wind"]
    cols = [c for c in cols if "sampling" not in c[0]]
    grid = np.full((len(cubes), len(cols)), np.nan)
    for i, cfg in enumerate(cubes):
        for j, (_, f) in enumerate(cols):
            r = f(s[cfg])
            if r:
                grid[i, j] = 100 * r["survival"]
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c for c, _ in cols], fontsize=7)
    ax.set_yticks(range(len(cubes)))
    ax.set_yticklabels([WORLD_NAMES[c] for c in cubes], fontsize=9)
    for i in range(len(cubes)):
        for j in range(len(cols)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cube worlds: survival to 128 steps (%)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out / "fig_cube_results.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("which", nargs="+", choices=["worlds", "gate", "theory", "results"])
    p.add_argument("--out", type=Path, default=Path("studies/stoch/paper/figures"))
    p.add_argument("--summary", type=Path, default=Path("runs/stoch/summary.json"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for w in a.which:
        if w == "results":
            results(a.out, a.summary)
        else:
            globals()[w](a.out)
        print(f"[figures] {w} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
