"""Benchmarks of the three world models on one red-cube configuration.

  frozen benchmark  gFVD and action following of the two JEPA arms (this repository's eval.py) and of the
                    flow-matching model (the recipe's eval.py), on the configuration's eval set, same windows.
  cube metrics      from decoded rollouts: L1 error of the cube position (argmax of the 6x6-box-filtered
                    redness) and the redness mass sitting on the true cube, per imagined step.
  latent probes     ridge regression from the frozen latent to the cube position (R^2 on later time rows),
                    wind from two consecutive latents, and 1-NN episode identification against the same
                    statistic computed from the true position alone (does the latent carry the texture?).
  scenes            GT | LeWM 2-d | LeWM grid | flow gifs and filmstrips (context frames encoded and
                    decoded), plus scatter plots of the 2-d latent.

    python -m studies.redcube.bench --config color_frame_wind --root /dev/shm/redcube/color_frame_wind \
        --flow-code <recipe dir> --out runs/redcube/color_frame_wind
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image, ImageDraw, ImageFont

import eval as E
import probe
import train as T
from studies.compare import FONT, jepa_reconstruct, upscale
from studies.redcube import env as R
from studies.redcube import flow as F

CTX, HOR = E.CTX, E.HOR
E.N_ACTIONS = R.N_ACTIONS
INNER = np.zeros((R.H, R.W), bool)
INNER[R.BORDER + 1:R.H - 1 - R.BORDER, R.BORDER + 1:R.W - 1 - R.BORDER] = True
ARMS = {"jepa2": "LeWM, 2-d latent", "grid": "LeWM, grid latent",
        "tile_factored": "JEPA", "tile_ar": "LeWM, tile enc + AR pred",
        "tile_ar8": "LeWM, tile enc + AR pred h8 nodrop", "tile_ar_nodrop": "LeWM, tile enc + AR pred nodrop",
        "vit_factored": "LeWM, ViT enc + factored pred", "tile_cls_ar": "LeWM, tile enc CLS + AR pred",
        "tile_cls_ar8": "LeWM, tile enc CLS + AR pred h8 nodrop", "tile_cls_factored": "LeWM, tile enc CLS + factored pred",
        "tile_factored_b256": "JEPA, batch 256", "tile_factored_b512": "JEPA, batch 512",
        "tile_factored_s1": "JEPA, seed 1", "tile_factored_s2": "JEPA, seed 2",
        "lewm6": "LeWM, ViT-6 class token", "lewm": "LeWM, ViT-12 class token"}


def font(size):
    return ImageFont.truetype(FONT, size) if Path(FONT).exists() else ImageFont.load_default(size)


def redness(frames_u8):
    f = frames_u8.astype(np.float32)
    rho = np.clip(f[..., 0] - np.maximum(f[..., 1], f[..., 2]), 0, None) / 255.0
    return rho * INNER


def box_sum(rho, k=R.CUBE):
    s = np.pad(rho, [(0, 0)] * (rho.ndim - 2) + [(1, 0), (1, 0)]).cumsum(-2).cumsum(-1)
    return s[..., k:, k:] - s[..., :-k, k:] - s[..., k:, :-k] + s[..., :-k, :-k]


def cube_stats(frames_u8, pos):
    """frames (N, T, H, W, 3), pos (N, T, 2) top-left (x, y) -> per-step position error and masses."""
    box = box_sum(redness(frames_u8))
    n, t = box.shape[:2]
    flat = box.reshape(n, t, -1)
    peak = flat.argmax(-1)
    py, px = np.divmod(peak, box.shape[-1])
    x = np.clip(pos[..., 0], 0, box.shape[-1] - 1)
    y = np.clip(pos[..., 1], 0, box.shape[-2] - 1)
    at_true = np.take_along_axis(flat, (y * box.shape[-1] + x)[..., None], -1)[..., 0]
    area = float(R.CUBE * R.CUBE)
    return {"position_error": (np.abs(px - x) + np.abs(py - y)).mean(0).tolist(),
            "mass_at_true": (at_true / area).mean(0).tolist(),
            "peak_mass": (flat.max(-1) / area).mean(0).tolist(),
            "total_mass": (flat.sum(-1) / area).mean(0).tolist()}


def ridge_fit(x_train, y_train, lam=1e-2):
    """Standardised ridge regression; returns the predictor."""
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-6
    xt = (x_train - mu) / sd
    ym = y_train.mean(0)
    a = xt.T @ xt + lam * xt.shape[0] * np.eye(xt.shape[1])
    w = np.linalg.solve(a, xt.T @ (y_train - ym))
    return lambda x: ((x - mu) / sd) @ w + ym


def ridge_r2(x_train, y_train, x_test, y_test, lam=1e-2):
    pred = ridge_fit(x_train, y_train, lam)(x_test)
    ss = ((y_test - y_test.mean(0)) ** 2).sum(0)
    return [float(v) for v in 1.0 - ((y_test - pred) ** 2).sum(0) / np.maximum(ss, 1e-9)]


def probe_error(predict, latents, pos):
    """Mean L1 distance in pixels between the probe's read-out of latents (B, T, D) and the true position."""
    guess = predict(latents.reshape(-1, latents.shape[-1])).reshape(pos.shape)
    return np.abs(guess - pos).sum(-1).mean(0).tolist()


def tile_fit(x_train, y_train, tokens=16, hidden=128, steps=10000, batch=2048, lr=1e-3, seed=0):
    """Position probe as a 4x4 heat map: each cell gets a logit and an (x, y) offset, the position is the
    softmax-weighted sum of cell centre + offset.  Adam on the MSE.  For a grid latent (`tokens` > 1) one GELU MLP
    shared by the tokens maps each token to its own cell, on per-dimension standardized tokens; for a single-vector
    latent (class token, `tokens` == 1) the MLP maps the whole vector to all 16 cells, on per-dimension standardized
    or PCA-whitened inputs (singular values floored at 1% of the largest), whichever reads the last 10% of the fit
    rows better - on a black background the class token keeps the position in low-variance directions that
    standardization leaves buried (18.5 px held-out standardized, 0.7 px whitened, diag_cls.py), while on textured
    backgrounds the whitening amplifies noise directions (50 px against 20).  Estimates are clipped to the frame.
    (A plain MLP regressing (x, y) from the concatenated latent reads the JEPA latent but not the flow tokenizer's:
    16.7 px held-out against 1.3 px here.)"""
    n, d = x_train.shape
    c = d // tokens
    cells = tokens if tokens > 1 else 16
    side = int(round(cells ** 0.5))
    ii, jj = np.divmod(np.arange(cells), side)
    centres = jnp.asarray(np.stack([(jj + 0.5) / side, (ii + 0.5) / side], 1), jnp.float32)
    dims = (c, hidden, hidden, 3 * cells // tokens)
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)

    def standardized(x):
        flat = x.reshape(-1, c)
        return flat.mean(0), np.eye(c) / (flat.std(0) + 1e-6)

    def whitened(x):
        mu = x.mean(0)
        _, s, vt = np.linalg.svd(x - mu, full_matrices=False)
        return mu, vt.T / np.maximum(s, 1e-2 * s[0]) * np.sqrt(len(x))

    def apply(params, h):
        for i, (w, b) in enumerate(params):
            h = h @ w + b
            if i < 2:
                h = jax.nn.gelu(h)
        h = h.reshape(*h.shape[:-2], cells, 3)
        weights = jax.nn.softmax(h[..., 0], axis=-1)
        return (weights[..., None] * (centres + 0.25 * h[..., 1:])).sum(-2)

    tx = optax.adam(optax.cosine_decay_schedule(lr, steps))

    @jax.jit
    def step(params, opt, x, y, key):
        idx = jax.random.randint(key, (batch,), 0, x.shape[0])
        grads = jax.grad(lambda p: ((apply(p, x[idx]) - y[idx]) ** 2).mean())(params)
        updates, opt = tx.update(grads, opt, params)
        return optax.apply_updates(params, updates), opt

    def fit(x_rows, y_rows, mu, basis):
        params = [(jax.random.normal(keys[i], (dims[i], dims[i + 1])) / np.sqrt(dims[i]), jnp.zeros(dims[i + 1])) for i in range(3)]
        opt = tx.init(params)
        x, y = jnp.asarray((x_rows.reshape(-1, tokens, c) - mu) @ basis, jnp.float32), jnp.asarray(y_rows / R.W, jnp.float32)
        for key in jax.random.split(keys[3], steps):
            params, opt = step(params, opt, x, y, key)
        return lambda z: np.clip(np.asarray(apply(params, jnp.asarray((z.reshape(-1, tokens, c) - mu) @ basis, jnp.float32))) * R.W, 0, R.W - 1)

    if tokens > 1:
        return fit(x_train, y_train, *standardized(x_train))
    split = int(0.9 * n)
    heldout = {name: np.abs(fit(x_train[:split], y_train[:split], *stats(x_train[:split]))(x_train[split:]) - y_train[split:]).sum(-1).mean()
               for name, stats in (("standardized", standardized), ("whitened", whitened))}
    stats = whitened if heldout["whitened"] < heldout["standardized"] else standardized
    return fit(x_train, y_train, *stats(x_train))


class PixelProbe(nnx.Module):
    """CNN from a frame to the cube position: strided convolutions down to an 8x8 grid, a logit and an (x, y) offset
    per cell, position = softmax-weighted cell centre + offset."""

    def __init__(self, rngs, width=48):
        self.convs = nnx.List([nnx.Conv(3, width, (3, 3), padding="SAME", rngs=rngs)] +
                              [nnx.Conv(width, width, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs) for _ in range(3)])
        self.head = nnx.Conv(width, 3, (1, 1), rngs=rngs)

    def __call__(self, x01):
        h = x01
        for conv in self.convs:
            h = jax.nn.gelu(conv(h))
        out = self.head(h)
        g = out.shape[1]
        weights = jax.nn.softmax(out[..., 0].reshape(out.shape[0], -1), -1).reshape(out.shape[:3])
        ii, jj = jnp.meshgrid(jnp.arange(g), jnp.arange(g), indexing="ij")
        centres = jnp.stack([(jj + 0.5) / g, (ii + 0.5) / g], -1)
        return (weights[..., None] * (centres + 0.25 * out[..., 1:])).sum((1, 2))


def cnn_fit(frames_u8, pos, steps=4000, batch=256, lr=1e-3, seed=0):
    """Pixel position probe trained on real frames whose background outside the cube is randomly kept, replaced by a
    flat grey, or by grey plus noise, so it reads the cube whatever a decoder draws around it."""
    model = PixelProbe(T.rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(optax.cosine_decay_schedule(lr, steps)), wrt=nnx.Param)
    frames, pos = jnp.asarray(frames_u8), jnp.asarray(pos, jnp.float32)
    ys, xs = jnp.arange(R.H)[None, :, None], jnp.arange(R.W)[None, None, :]

    def augment(x, p, key):
        k_grey, k_noise, k_kind = jax.random.split(key, 3)
        px, py = p[:, 0, None, None], p[:, 1, None, None]
        cube = ((xs >= px) & (xs < px + R.CUBE) & (ys >= py) & (ys < py + R.CUBE))[..., None]
        grey = jax.random.uniform(k_grey, (x.shape[0], 1, 1, 1), minval=0.15, maxval=0.8)
        kind = jax.random.randint(k_kind, (x.shape[0], 1, 1, 1), 0, 3)
        background = jnp.where(kind == 0, x, jnp.where(kind == 1, grey, grey + 0.08 * jax.random.normal(k_noise, x.shape)))
        return jnp.where(cube, x, background)

    @nnx.jit
    def step(model, opt, frames, pos, key):
        k_idx, k_aug = jax.random.split(key)
        idx = jax.random.randint(k_idx, (batch,), 0, frames.shape[0])
        x, p = frames[idx].astype(jnp.float32) / 255.0, pos[idx]
        loss, grads = nnx.value_and_grad(lambda m: ((m(augment(x, p, k_aug)) - p / R.W) ** 2).mean())(model)
        opt.update(model, grads)
        return loss

    for key in jax.random.split(jax.random.PRNGKey(seed + 1), steps):
        step(model, opt, frames, pos, key)

    @nnx.jit
    def apply(model, x):
        return model(x.astype(jnp.float32) / 255.0)

    def predict(frames_u8):
        flat = frames_u8.reshape(-1, R.H, R.W, 3)
        return np.concatenate([np.asarray(apply(model, jnp.asarray(flat[i:i + 512]))) for i in range(0, len(flat), 512)]) * R.W

    return predict


def pixel_error(predict, frames, pos):
    """Mean L1 distance in pixels between the pixel probe's read-out of frames (B, T, H, W, 3) and the true position."""
    return np.abs(predict(frames).reshape(pos.shape) - pos).sum(-1).mean(0).tolist()


FIT_ROWS = 1001


def eval_latents(bundles, flow_train, flow_model, frames_te):
    """Latents (T, E, D) of a (T, E, H, W, 3) array; the flow tokenizer is temporal, so each env is encoded as a sequence."""
    z = {name: encode_jepa(bundle, frames_te) for name, bundle in bundles.items()}
    z["flow"] = encode_flow(flow_train, flow_model, frames_te.transpose(1, 0, 2, 3, 4), chunk=4).transpose(1, 0, 2)
    return z


def fit_probes(bundles, flow_train, flow_model, root):
    """Position probes fit on the first FIT_ROWS time rows of the training set (disjoint from eval): the per-tile
    latent probe of every model, and the pixel CNN applied to every decoded frame."""
    with np.load(root / "train.npz") as archive:
        frames, pos = archive["frames"][:FIT_ROWS], archive["pos"][:FIT_ROWS].reshape(-1, 2).astype(np.float32)
    tokens = {name: bundle.model.n_tokens for name, bundle in bundles.items()} | {"flow": 16}
    latent = {name: tile_fit(z.reshape(-1, z.shape[-1]), pos, tokens[name])
              for name, z in eval_latents(bundles, flow_train, flow_model, frames).items()}
    return latent, cnn_fit(frames.reshape(-1, R.H, R.W, 3), pos)


def jepa_rollout(bundle, ctx_frames, ctx_actions, actions):
    """Decoded rollout (B, HOR, H, W, 3) uint8 and the predicted latents (B, HOR, D)."""
    horizon = actions.shape[1]
    _, preds = T._rollout_latents(bundle.model, jnp.asarray(ctx_frames), T.outgoing_actions(ctx_actions, actions), horizon)
    frames = bundle.decoder(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape[0], horizon, R.H, R.W, 3)
    return np.asarray(jnp.clip(frames * 255.0 + 0.5, 0, 255).astype(jnp.uint8)), np.asarray(preds)


def flow_rollout(train_mod, model, ctx_frames, ctx_actions, actions, key):
    """The recipe's `imagine` (same keys, so the same video) also returning the predicted latents (B, HOR, D)."""
    ctx = train_mod.encode(model, jnp.asarray(ctx_frames))
    ca, fa = jnp.asarray(ctx_actions, jnp.int32), jnp.asarray(actions, jnp.int32)
    gdef, state = nnx.split(model.df)
    kr, _ = jax.random.split(key)
    zt = train_mod._rollout_latents(nnx.merge(gdef, state), ctx, ca, fa.T, kr, train_mod.STEPS, model.win)
    video = np.asarray(train_mod.decode(model, zt).transpose(1, 0, 2, 3, 4))
    return video, np.asarray(zt.transpose(1, 0, 2, 3, 4)).reshape(zt.shape[1], zt.shape[0], -1)


def knn_episode(z_train, ep_train, z_test, ep_test, chunk=1024):
    """1-NN episode identity (cosine) of held-out frames among the fitting frames."""
    zt = z_train / (np.linalg.norm(z_train, axis=1, keepdims=True) + 1e-9)
    hits = 0
    for i in range(0, len(z_test), chunk):
        q = z_test[i:i + chunk]
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        nn = (q @ zt.T).argmax(1)
        hits += int((ep_train[nn] == ep_test[i:i + chunk]).sum())
    return hits / len(z_test)


def probes(latents, data, rows_fit):
    """latents (T+1, E, D) of every eval frame; targets from the recorded true state."""
    t1, e = latents.shape[:2]
    pos = data["pos"].astype(np.float32)
    wind = data["wind"].astype(np.float32)
    episode = data["episode"].astype(np.int64) * 10_000 + np.arange(e)[None]
    fit, test = slice(0, rows_fit), slice(rows_fit, t1)
    flat = lambda a, s: a[s].reshape(-1, *a.shape[2:])
    z_fit, z_test = flat(latents, fit), flat(latents, test)
    r2_pos = ridge_r2(z_fit, flat(pos, fit), z_test, flat(pos, test))
    same = (data["episode"][1:] == data["episode"][:-1])
    pair = np.concatenate([latents[1:], latents[:-1]], -1)
    pair_fit, pair_test = pair[:rows_fit - 1][same[:rows_fit - 1]], pair[rows_fit:][same[rows_fit:]]
    w_fit, w_test = wind[1:rows_fit][same[:rows_fit - 1]], wind[rows_fit + 1:][same[rows_fit:]]
    r2_wind = ridge_r2(pair_fit, w_fit, pair_test, w_test) if len(pair_test) > 10 else [float("nan")] * 2
    rng = np.random.default_rng(0)
    sub_fit = rng.choice(len(z_fit), min(len(z_fit), 20_000), replace=False)
    sub_test = rng.choice(len(z_test), min(len(z_test), 4_000), replace=False)
    ep_fit, ep_test = flat(episode, fit)[sub_fit], flat(episode, test)[sub_test]
    acc_latent = knn_episode(z_fit[sub_fit], ep_fit, z_test[sub_test], ep_test)
    acc_pos = knn_episode(flat(pos, fit)[sub_fit], ep_fit, flat(pos, test)[sub_test], ep_test)
    return {"r2_pos_x": r2_pos[0], "r2_pos_y": r2_pos[1], "r2_wind_x": r2_wind[0], "r2_wind_y": r2_wind[1],
            "episode_1nn_latent": acc_latent, "episode_1nn_position_only": acc_pos,
            "chance_episode": float(1.0 / len(np.unique(ep_fit))), "dim": int(latents.shape[-1])}


def encode_jepa(bundle, frames):
    return np.asarray(probe.embed_all(bundle.model, jnp.asarray(frames))).reshape(*frames.shape[:2], -1)


def encode_flow(train_mod, model, frames, chunk=32):
    out = []
    for i in range(0, frames.shape[0], chunk):
        z = train_mod.encode(model, jnp.asarray(frames[i:i + chunk]))
        out.append(np.asarray(z).reshape(z.shape[0], z.shape[1], -1))
    return np.concatenate(out)


def latent_plots(z, data, spec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = data["pos"].reshape(-1, 2).astype(np.float32)
    ep = data["episode"].reshape(-1)
    zz = z.reshape(-1, z.shape[-1])
    if zz.shape[1] > 2:
        c = zz - zz.mean(0)
        _, _, vt = np.linalg.svd(c[np.random.default_rng(0).choice(len(c), min(len(c), 8000), replace=False)], full_matrices=False)
        zz = c @ vt[:2].T
        title = "first two principal components"
    else:
        title = "the 2-d latent itself"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, colour, name in zip(axes, (pos[:, 0], pos[:, 1], ep % 23), ("cube x", "cube y", "episode (texture)")):
        sc = ax.scatter(zz[:, 0], zz[:, 1], c=colour, s=2, alpha=0.5, cmap="viridis" if name != "episode (texture)" else "tab20")
        ax.set_title(f"coloured by {name}", loc="left", fontsize=12)
        ax.set_xticks([]), ax.set_yticks([])
        if name != "episode (texture)":
            fig.colorbar(sc, ax=ax, fraction=0.04)
    fig.suptitle(f"{spec.name}: {title}", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def compose(columns, labels, scale=4, gap=12, header=40, footer=30):
    big, small = font(18), font(14)
    cell = R.H * scale
    width = len(columns) * cell + (len(columns) + 1) * gap
    out = []
    for t in range(CTX + HOR):
        canvas = Image.new("RGB", (width, header + cell + footer), (250, 250, 248))
        draw = ImageDraw.Draw(canvas)
        imagined = t >= CTX
        for j, (label, video) in enumerate(zip(labels, columns)):
            x = gap + j * (cell + gap)
            canvas.paste(Image.fromarray(upscale(video[t], scale)), (x, header))
            colour = (196, 40, 40) if (imagined and j > 0) else (30, 30, 30)
            draw.text((x, 10), label, fill=colour, font=big)
            if imagined and j > 0:
                draw.rectangle([x - 3, header - 3, x + cell + 2, header + cell + 2], outline=(196, 40, 40), width=3)
        status = (f"context {t + 1}/{CTX}  (model columns: real frame through encoder and decoder)" if not imagined
                  else f"imagined step {t - CTX + 1}/{HOR}")
        draw.text((gap, header + cell + 6), status, fill=(90, 90, 90), font=small)
        out.append(np.asarray(canvas))
    return out


def filmstrip(columns, labels, steps, scale=3, gap=6, left=250):
    small = font(14)
    cell = R.H * scale
    canvas = Image.new("RGB", (left + len(steps) * (cell + gap), 26 + len(columns) * (cell + gap)), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    for c, t in enumerate(steps):
        draw.text((left + c * (cell + gap) + cell // 2 - 12, 5), f"t={t + 1}" if t < CTX else f"+{t - CTX + 1}",
                  fill=(90, 90, 90), font=small)
    for r, (label, video) in enumerate(zip(labels, columns)):
        y = 26 + r * (cell + gap)
        draw.text((6, y + cell // 2 - 8), label, fill=(30, 30, 30), font=small)
        for c, t in enumerate(steps):
            x = left + c * (cell + gap)
            canvas.paste(Image.fromarray(upscale(video[t], scale)), (x, y))
            if t >= CTX and r > 0:
                draw.rectangle([x - 2, y - 2, x + cell + 1, y + cell + 1], outline=(196, 40, 40), width=2)
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(R.CONFIGS))
    parser.add_argument("--root", type=Path, required=True, help="data and models of the configuration")
    parser.add_argument("--flow-code", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=E.N_WIN)
    parser.add_argument("--metric-windows", type=int, default=512)
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", default="all", help="comma-separated LeWM arms to benchmark, or `all` present")
    args = parser.parse_args()
    spec = R.CONFIGS[args.config]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = E.load_data(args.root / "eval.npz")
    indices = E.sample_indices(data, args.windows)
    wanted = ARMS if args.arms == "all" else args.arms.split(",")
    arms = [a for a in wanted if (args.root / a / "model_probe.npz").exists()]
    loaded = {a: T.load_model(args.root / a / "model_probe.npz") for a in arms}
    bundles = {a: loaded[a][0] for a in arms}
    metas = {a: loaded[a][1] for a in arms}
    labels = ["ground truth"] + [ARMS[a] for a in arms] + ["flow matching"]
    flow_train, flow_eval, flow_model = F.load(args.flow_code, args.root / "flow" / "model.npz", R.N_ACTIONS)
    flow_meta = flow_train.load_meta(str(args.root / "flow" / "model.npz"))
    report = {"config": spec.name, "spec": spec.__dict__, "benchmark": {}, "train_s": {}}

    # 1. frozen benchmark, identical windows for every model.
    report["benchmark"]["persistence"] = E.evaluate(T.Bundle(), data, indices)[0]
    for name, bundle in bundles.items():
        report["benchmark"][name] = E.evaluate(bundle, data, indices)[0]
        report["train_s"][name] = metas[name]["train_s"]
        report["benchmark"][name].update({f"heads_{k}": v for k, v in metas[name]["heads"].items()
                                          if k.startswith("heldout_")})
        print(f"[bench] {name}: {json.dumps(report['benchmark'][name])}", flush=True)
    gfvd, af, _ = flow_eval.evaluate(flow_model, data, indices)
    n_uniform = min(flow_eval.N_HEAD_WIN, len(indices))
    heads = flow_eval.evaluate_heads(flow_model, data, np.concatenate([indices[:n_uniform], flow_eval.death_indices(data)]),
                                     n_uniform)
    report["benchmark"]["flow"] = {"gFVD": gfvd, "AF": af, **heads}
    report["train_s"]["flow"] = flow_meta["train_s"]
    print(f"[bench] flow: {json.dumps(report['benchmark']['flow'])}  ({time.time() - t0:.0f}s)", flush=True)

    # 2. cube metrics and pixel error on the first windows, all models from the same context and actions.
    sel = indices[:args.metric_windows]
    frames, actions = E.window_batch(data, sel)
    times = sel[:, :1] + np.arange(CTX + HOR)[None]
    pos = data["pos"][times, sel[:, 1:]].astype(np.int64)
    real = frames[:, CTX:]
    rollouts = {"persistence": np.repeat(frames[:, CTX - 1:CTX], HOR, 1)}
    predicted = {}
    chunks = range(0, len(frames), 64)
    for name, bundle in bundles.items():
        parts = [jepa_rollout(bundle, frames[i:i + 64, :CTX], actions[i:i + 64, :CTX], actions[i:i + 64, CTX:]) for i in chunks]
        rollouts[name], predicted[name] = (np.concatenate([p[k] for p in parts]) for k in (0, 1))
    parts = [flow_rollout(flow_train, flow_model, frames[i:i + 64, :CTX], actions[i:i + 64, :CTX], actions[i:i + 64, CTX:],
                          jax.random.PRNGKey(args.seed + i)) for i in chunks]
    rollouts["flow"], predicted["flow"] = (np.concatenate([p[k] for p in parts]) for k in (0, 1))
    report["cube"] = {"ground_truth": cube_stats(real, pos[:, CTX:])}
    report["pixel_mse"] = {}
    for name, video in rollouts.items():
        report["cube"][name] = cube_stats(video, pos[:, CTX:])
        report["pixel_mse"][name] = (((video.astype(np.float32) - real.astype(np.float32)) / 255.0) ** 2).mean((0, 2, 3, 4)).tolist()
        print(f"[cube] {name}: position error by step {np.round(report['cube'][name]['position_error'], 2).tolist()}", flush=True)
    context = {a: np.asarray(jepa_reconstruct(bundles[a].model, bundles[a].decoder, jnp.asarray(frames[:, :CTX]))) for a in arms}
    context["flow"] = np.asarray(flow_train.decode(flow_model, flow_train.encode(flow_model, jnp.asarray(frames[:, :CTX]))))
    report["context_recon"] = {name: cube_stats(ctx, pos[:, :CTX]) for name, ctx in context.items()}
    report["context_recon"]["ground_truth"] = cube_stats(frames[:, :CTX], pos[:, :CTX])

    # 3. latent probes on every eval frame.
    all_frames = data["frames"]
    rows_fit = int(all_frames.shape[0] * 0.8)
    latents = eval_latents(bundles, flow_train, flow_model, all_frames)
    report["probes"] = {name: probes(z, data, rows_fit) for name, z in latents.items()}
    for name in latents:
        print(f"[probe] {name}: {json.dumps(report['probes'][name])}", flush=True)
        latent_plots(latents[name], data, spec, args.out / f"latent_{name}.png")
    # 3b. position probes fit on training frames.  Latent: the per-tile probe read off the PREDICTED latents of the
    #     rollouts, off the encoded true future frames (the floor set by the representation) and on every eval frame
    #     (held-out fit quality).  Pixel: the CNN read off the decoded rollouts, off the decoded context frames (the
    #     floor set by the decoder), off the real future frames and on every eval frame.
    window_latents = {name: encode_jepa(bundle, frames) for name, bundle in bundles.items()}
    window_latents["flow"] = encode_flow(flow_train, flow_model, frames)
    true_pos, ctx_pos = pos[:, CTX:].astype(np.float32), pos[:, :CTX].astype(np.float32)
    latent_probes, pixel_probe = fit_probes(bundles, flow_train, flow_model, args.root)
    report["probe_error"] = {}
    for name, predict in latent_probes.items():
        z = latents[name]
        heldout = np.abs(predict(z.reshape(-1, z.shape[-1])) - data["pos"].reshape(-1, 2)).sum(-1).mean()
        report["probe_error"][name] = {"predicted": probe_error(predict, predicted[name], true_pos),
                                       "true_latent": probe_error(predict, window_latents[name][:, CTX:], true_pos),
                                       "heldout": float(heldout)}
        print(f"[probe] {name}: tile probe held-out {heldout:.2f} px; error from the predicted latent by step "
              f"{np.round(report['probe_error'][name]['predicted'], 2).tolist()}", flush=True)
    heldout = np.abs(pixel_probe(all_frames) - data["pos"].reshape(-1, 2)).sum(-1).mean()
    report["pixel_probe"] = {"heldout": float(heldout), "real": pixel_error(pixel_probe, real, true_pos),
                             "persistence": pixel_error(pixel_probe, rollouts["persistence"], true_pos)}
    print(f"[probe] pixel probe held-out {heldout:.2f} px; on the real future frames by step "
          f"{np.round(report['pixel_probe']['real'], 2).tolist()}", flush=True)
    for name in list(bundles) + ["flow"]:
        report["pixel_probe"][name] = {"predicted": pixel_error(pixel_probe, rollouts[name], true_pos),
                                       "context": pixel_error(pixel_probe, context[name], ctx_pos)}
        print(f"[probe] {name}: pixel probe on the decoded rollout by step "
              f"{np.round(report['pixel_probe'][name]['predicted'], 2).tolist()}, on the decoded context "
              f"{np.round(report['pixel_probe'][name]['context'], 2).tolist()}", flush=True)

    # 4. scenes: the windows where the cube travels furthest.
    travel = np.abs(np.diff(pos[:, CTX:], axis=1)).sum((1, 2))
    chosen = np.argsort(-travel)[:args.scenes]
    import imageio.v2 as imageio
    for s, w in enumerate(chosen):
        columns = [frames[w]] + [np.concatenate([context[a][w], rollouts[a][w]]) for a in arms + ["flow"]]
        video = compose(columns, labels)
        imageio.mimsave(args.out / f"scene{s}.gif", video + video[-1:] * 4, duration=1000 / 4, loop=0)
        png_dir = args.out / f"scene{s}_frames"
        png_dir.mkdir(exist_ok=True)
        for t, im in enumerate(video):
            Image.fromarray(im).save(png_dir / f"{t:03d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-i", str(png_dir / "%03d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        str(args.out / f"scene{s}.mp4")], check=False)
        Image.fromarray(filmstrip(columns, labels, [CTX - 1, CTX, CTX + 1, CTX + 3, CTX + 7, CTX + 11, CTX + 15])).save(
            args.out / f"scene{s}_strip.png")
    report["scenes"] = [int(w) for w in chosen]
    report["seconds"] = round(time.time() - t0, 1)
    (args.out / "bench.json").write_text(json.dumps(report, indent=1))
    print(f"[done] {report['seconds']}s -> {args.out / 'bench.json'}", flush=True)


if __name__ == "__main__":
    main()
