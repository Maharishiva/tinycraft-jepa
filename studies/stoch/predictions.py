"""The half-open wall: what each model predicts from the same pre-reveal gate context.

Rows: the two real futures (gap on top / at the bottom), the LeWM class-token JEPA's mean prediction decoded
by its off-budget decoder, the grid JEPA's, and three samples of the flow model.  Columns: t = 29..36, the
frames after the reveal, under the same oscillating actions (the cube holds x = 19).

    XLA_PYTHON_CLIENT_MEM_FRACTION=.15 python -m studies.stoch.predictions --lewm /dev/shm/stoch/gate/lewm6/model_probe.npz \
        --grid /dev/shm/stoch/gate/tile_factored/model_probe.npz --flow /dev/shm/stoch/gate/flow_model.npz \
        --flow-code studies/stoch/flow_recipe --out studies/stoch/paper/figures/fig_predictions.png
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import train as T
import viz
from studies.redcube import flow as F
from studies.stoch import env as S
from studies.stoch import sampler as SM
from studies.stoch.figures import OSC, scripted
from studies.stoch.wallmass import sampled_rollout

CTX, HOR, T0 = 8, 8, 28


def jepa_rows(path, context, out_actions, real_future):
    """The decoded mean rollout, and the decoded *encoding* of the real future (what the latent can hold)."""
    bundle, _ = T.load_model(path)
    _, preds = T._rollout_latents(bundle.model, jnp.asarray(context), jnp.asarray(out_actions, jnp.int32), horizon=HOR)
    enc = viz.encode(bundle.model, jnp.asarray(real_future[None]))
    return np.asarray(viz.decode(bundle.decoder, preds, HOR))[0], np.asarray(viz.decode(bundle.decoder, enc, HOR))[0]


def flow_rows(code, path, context, incoming_ctx, incoming_future, n):
    train_mod, _, model = F.load(code, path, S.N_ACTIONS)
    wm = F.WM(train_mod, model, 4)
    rows = []
    for k in range(n):
        video, _, _ = wm.imagine(np.asarray(context), np.asarray(incoming_ctx), np.asarray(incoming_future), jax.random.PRNGKey(k))
        rows.append(video[0])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lewm", type=Path, required=True)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--flow", type=Path, required=True)
    p.add_argument("--flow-code", type=Path, required=True)
    p.add_argument("--grid-sampler", type=Path, help="flow head on the grid JEPA's residual: adds its sampled rollouts")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    spec = S.CONFIGS["gate"]
    taken = OSC * 20
    futures = {g: scripted(spec, taken, gap=g)[0] for g in (0, 1)}
    frames = futures[0]
    context = np.stack(frames[T0 - CTX + 1:T0 + 1])[None]
    out_actions = np.asarray(taken[T0 - CTX + 1:T0 + 1 + HOR], np.int32)[None]
    incoming_ctx = np.asarray(taken[T0 - CTX:T0], np.int32)[None]
    incoming_future = np.asarray(taken[T0:T0 + HOR], np.int32)[None]
    def real(frames):
        fut = frames[T0 + 1:T0 + 1 + HOR]
        dead = np.zeros_like(frames[0]) + 40
        return np.stack(fut + [dead] * (HOR - len(fut)))
    top = real(futures[0])
    lewm_pred, lewm_enc = jepa_rows(a.lewm, context, out_actions, top)
    grid_pred, grid_enc = jepa_rows(a.grid, context, out_actions, top)
    rows = [("real future, gap on top", top),
            ("real future, gap at the bottom", real(futures[1])),
            ("LeWM: real future encoded, decoded", lewm_enc),
            ("LeWM: mean rollout, decoded", lewm_pred),
            ("grid JEPA: real future encoded, decoded", grid_enc),
            ("grid JEPA: mean rollout, decoded", grid_pred)]
    if a.grid_sampler:
        bundle, _ = T.load_model(a.grid)
        sampler = SM.load(bundle, a.grid_sampler, 8)[0]
        for k in range(a.samples):
            latents = sampled_rollout(sampler, jnp.asarray(context), jnp.asarray(out_actions), HOR, jax.random.PRNGKey(k))
            rows.append((f"grid JEPA + sampling predictor: sample {k + 1}", np.asarray(viz.decode(bundle.decoder, latents, HOR))[0]))
    for k, video in enumerate(flow_rows(a.flow_code, a.flow, context, incoming_ctx, incoming_future, a.samples)):
        rows.append((f"flow model: sample {k + 1}", video[:HOR]))
    fig, axes = plt.subplots(len(rows), HOR + 2, figsize=(1.15 * (HOR + 2) + 0.8, 1.2 * len(rows) + 0.4),
                             gridspec_kw={"width_ratios": [2.2] + [1] * (HOR + 1), "wspace": 0.05, "hspace": 0.1})
    for r, (title, video) in enumerate(rows):
        axes[r, 0].axis("off")
        axes[r, 0].text(0.97, 0.5, title, ha="right", va="center", fontsize=7.5, transform=axes[r, 0].transAxes)
        axes[r, 1].imshow(context[0, -1], interpolation="nearest")
        axes[r, 1].axis("off")
        if r == 0:
            axes[r, 1].set_title(f"context t={T0}", fontsize=7.5, pad=2)
        for c in range(HOR):
            ax = axes[r, c + 2]
            ax.imshow(video[c], interpolation="nearest")
            ax.axis("off")
            if r == 0:
                ax.set_title(f"t={T0 + 1 + c}", fontsize=7.5, pad=2)
    fig.subplots_adjust(top=0.975, bottom=0.01)
    fig.suptitle("Same context, wall still solid: the two real futures, what each JEPA latent can hold, its mean rollout, "
                 "the sampled rollouts of the grid JEPA with the flow head on its residual, and the flow model's samples", fontsize=8.5, y=0.995)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    np.savez(a.out.with_suffix(".npz"), **{t: v for t, v in rows})
    print(f"[predictions] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
