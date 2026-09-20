"""Latents against cube position: t-SNE of real-frame latents coloured by x and by y,
the variance spectrum of the latent with how many principal directions the probe needs to read the position, and
rollouts read out step by step by the benchmark's MLP probe against the true position.  One row per
(configuration, model); flow matching rows use the tokenizer latent.

  python -m studies.redcube.latent_viz --root /dev/shm/redcube --flow-code <recipe dir> \
      --panels off_still:lewm6 off_still:tile_factored off_still:flow off_wind:lewm6 --out runs/redcube/latent_viz.png
"""
import argparse
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval as E
import train as T
from studies.redcube import env as R
from studies.redcube import flow as F
from studies.redcube.bench import ARMS, CTX, HOR, encode_flow, encode_jepa, flow_rollout, jepa_rollout, tile_fit
from studies.redcube.diag_cls import heatmap_probe

NAMES = {"off_still": "black, still", "off_wind": "black, wind", "gray_frame_still": "gray noise / frame, still",
         "gray_frame_wind": "gray noise / frame, wind", "gray_episode_still": "gray noise / episode, still",
         "gray_episode_wind": "gray noise / episode, wind", "color_frame_still": "colour noise / frame, still",
         "color_frame_wind": "colour noise / frame, wind", "color_episode_still": "colour noise / episode, still",
         "color_episode_wind": "colour noise / episode, wind"}
TOP_K = (4, 16, 48, 96, 192, 384)
LABELS = {"lewm6": "LeWM (class token, ViT-6)", "lewm": "LeWM (class token, ViT-12)", "tile_factored": "JEPA (grid latent 4x4x32)"}


def sq_dist(x):
    n2 = (x ** 2).sum(1)
    d = n2[:, None] + n2[None] - 2 * x @ x.T
    np.fill_diagonal(d, 0)
    return np.maximum(d, 0)


def tsne(x, perplexity=30.0, iters=750, lr=200.0, seed=0):
    """Exact t-SNE (van der Maaten 2008): perplexity-calibrated Gaussian affinities, Student-t embedding, early
    exaggeration 12 for the first 250 steps, momentum 0.5 then 0.8, adaptive gains; PCA initialisation."""
    n = len(x)
    d2 = sq_dist(x.astype(np.float64))
    beta, lo, hi = np.ones(n), np.full(n, -np.inf), np.full(n, np.inf)
    eye = np.eye(n, dtype=bool)
    for _ in range(60):
        p = np.exp(-d2 * beta[:, None])
        p[eye] = 0
        s = p.sum(1) + 1e-12
        entropy = np.log(s) + beta * (d2 * p).sum(1) / s
        up = entropy > np.log(perplexity)
        lo, hi = np.where(up, beta, lo), np.where(up, hi, beta)
        beta = np.where(up, np.where(np.isinf(hi), beta * 2, (beta + hi) / 2), np.where(np.isinf(lo), beta / 2, (beta + lo) / 2))
    p = p / s[:, None]
    p = (p + p.T) / (2 * n)
    p = np.maximum(p, 1e-12)
    xc = x - x.mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    y = xc @ vt[:2].T
    y = y / y.std(0) * 1e-4 + 1e-4 * np.random.default_rng(seed).standard_normal((n, 2))
    gains, step = np.ones((n, 2)), np.zeros((n, 2))
    for it in range(iters):
        q = 1 / (1 + sq_dist(y))
        q[eye] = 0
        pq = ((12.0 if it < 250 else 1.0) * p - q / q.sum()) * q
        grad = 4 * (pq.sum(1)[:, None] * y - pq @ y)
        gains = np.maximum(np.where(np.sign(grad) != np.sign(step), gains + 0.2, gains * 0.8), 0.01)
        step = (0.5 if it < 250 else 0.8) * step - lr * gains * grad
        y = y + step
        y = y - y.mean(0)
    return y


def l1(guess, pos):
    return float(np.abs(guess - pos).sum(-1).mean())


def spectrum_probe(z, pos, steps):
    """(std per principal direction, {k: held-out px of the MLP probe on the top-k whitened directions},
    number of distinct latent prototypes in the directions whose std is above 1% of the largest)."""
    split = int(0.9 * len(z))
    mu = z[:split].mean(0)
    _, s, vt = np.linalg.svd(z[:split] - mu, full_matrices=False)
    std = s / np.sqrt(split)
    white = (z - mu) @ vt.T / np.maximum(std, 1e-2 * std[0])
    ks = sorted({k for k in TOP_K if k < z.shape[1]} | {z.shape[1]})
    err = {k: l1(heatmap_probe(white[:split, :k], pos[:split], white[split:, :k], steps=steps), pos[split:]) for k in ks}
    big = int((std > 1e-2 * std[0]).sum())
    prototypes = len(np.unique(np.round(white[:, :big] * 2), axis=0))
    return std, err, big, prototypes


def load_panel(root, flow_code, cfg, model, n_windows, n_roll, seed):
    data = E.load_data(root / cfg / "eval.npz")
    sel = E.sample_indices(data, E.N_WIN)[:n_windows]
    frames, actions = E.window_batch(data, sel)
    times = sel[:, :1] + np.arange(frames.shape[1])[None]
    pos = data["pos"][times, sel[:, 1:]].astype(np.float32)
    if model == "flow":
        flow_train, _, flow_model = F.load(flow_code, root / cfg / "flow" / "model.npz", R.N_ACTIONS)
        z = encode_flow(flow_train, flow_model, frames)
        _, pred = flow_rollout(flow_train, flow_model, frames[:n_roll, :CTX], actions[:n_roll, :CTX], actions[:n_roll, CTX:],
                               jax.random.PRNGKey(seed))
        label, tokens = "FLOW MATCHING (tokenizer latent)", 16
    else:
        bundle = T.load_model(root / cfg / model / "model_probe.npz")[0]
        z = encode_jepa(bundle, frames)
        _, pred = jepa_rollout(bundle, frames[:n_roll, :CTX], actions[:n_roll, :CTX], actions[:n_roll, CTX:])
        label, tokens = LABELS.get(model, ARMS.get(model, model)), bundle.model.n_tokens
    return {"cfg": cfg, "label": label, "z": z, "pos": pos, "pred": np.asarray(pred), "tokens": tokens}


def draw_row(axes, panel, n_tsne, n_paths, probe_steps, seed):
    z, pos, pred = panel["z"], panel["pos"], panel["pred"]
    d = z.shape[-1]
    flat_z, flat_pos = z.reshape(-1, d), pos.reshape(-1, 2)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(flat_z), min(n_tsne, len(flat_z)), replace=False)
    guess = pred[:n_paths].reshape(-1, d)
    real = tsne(flat_z[idx], seed=seed)
    std, err, big, prototypes = spectrum_probe(flat_z, flat_pos, probe_steps)
    for ax, axis, name in ((axes[0], 0, "x"), (axes[1], 1, "y")):
        sc = ax.scatter(real[:, 0], real[:, 1], c=flat_pos[idx, axis], cmap="viridis", s=5, lw=0, vmin=0, vmax=R.W - 1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"t-SNE of real-frame latents, colour = cube {name}, px", fontsize=9.5, loc="left")
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    axes[0].set_ylabel(f"{NAMES[panel['cfg']].upper()}\n{panel['label']}\nlatent {d}-d", fontsize=12, fontweight="bold",
                       labelpad=12)
    axes[1].text(0.01, 0.01, f"{prototypes} distinct latents in the {big} directions above 1% of the top std ({len(flat_z)} frames)",
                 transform=axes[1].transAxes, fontsize=7.5, color="#444")

    ax = axes[2]
    ax.bar(np.arange(len(std)), std, width=1.0, color="#c9c9c9")
    ax.set_yscale("log")
    ax.set_ylim(max(std.min(), std.max() * 1e-5), std.max() * 2)
    ax.set_xlabel("principal direction", fontsize=9)
    ax.set_ylabel("std of the latent along it", fontsize=9)
    ax2 = ax.twinx()
    ks = sorted(err)
    ax2.plot(ks, [err[k] for k in ks], "o-", color="#7b4ea3", ms=4)
    for k in ks:
        ax2.annotate(f"{err[k]:.1f}", (k, err[k]), textcoords="offset points", xytext=(3, 4), fontsize=7.5, color="#7b4ea3")
    ax2.set_ylim(0, 26)
    ax2.set_ylabel("held-out px, MLP probe on the top-k whitened directions", fontsize=8, color="#7b4ea3")
    share = std ** 2 / (std ** 2).sum()
    rank = np.exp(-(share * np.log(share + 1e-12)).sum())
    ax.set_title(f"variance spectrum, effective rank {rank:.0f} / {d}; purple: probe error vs directions used", fontsize=9.5, loc="left")

    split = int(0.9 * len(flat_z))
    probe = tile_fit(flat_z[:split], flat_pos[:split], tokens=panel["tokens"])
    heldout = l1(probe(flat_z[split:]), flat_pos[split:])
    est = probe(guess).reshape(n_paths, HOR, 2)
    curve = np.abs(probe(pred.reshape(-1, d)).reshape(len(pred), HOR, 2) - pos[:len(pred), CTX:]).sum(-1).mean(0)
    ax = axes[3]
    steps = np.arange(1, HOR + 1)
    colours = plt.cm.tab10(np.arange(n_paths))
    for k in range(n_paths):
        ax.plot(steps, pos[k, CTX:, 0], color=colours[k], lw=1.6)
        ax.plot(steps, est[k, :, 0], color=colours[k], lw=1.6, ls="--")
    ax.set_ylim(-1, R.W)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("imagined step", fontsize=9)
    ax.set_ylabel("cube x, px", fontsize=9)
    ax.set_title(f"rollouts: solid = true x, dashed = benchmark probe on the predicted latent\n"
                 f"probe held-out {heldout:.2f} px on real frames; L1 error at step 1 / 8 / 16 = "
                 f"{curve[0]:.1f} / {curve[7]:.1f} / {curve[15]:.1f} px ({len(pred)} windows)", fontsize=8.5, loc="left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--flow-code", type=Path, required=True)
    parser.add_argument("--panels", nargs="+", required=True, help="<config>:<model>, model = arm name or flow")
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--tsne", type=int, default=1500)
    parser.add_argument("--probe-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    panels = [load_panel(args.root, args.flow_code, *p.split(":"), args.windows, args.rollouts, args.seed) for p in args.panels]
    fig, axes = plt.subplots(len(panels), 4, figsize=(22, 4.6 * len(panels)))
    axes = np.atleast_2d(axes)
    for row, panel in zip(axes, panels):
        draw_row(row, panel, args.tsne, args.paths, args.probe_steps, args.seed)
        print(f"[done] {panel['cfg']} {panel['label']}", flush=True)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
