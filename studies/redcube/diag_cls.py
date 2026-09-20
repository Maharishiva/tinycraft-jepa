"""Is a class-token latent unreadable, or just badly conditioned for the probe?  (CPU is enough.)

For each arm: the latent's per-dimension spread and effective rank on real frames, then the cube position read
out by three probes fit on the first --rows time rows and tested on the next --test rows: ridge on per-dimension
standardized latents, the benchmark's MLP heat-map probe on the same standardization, and the same MLP on
PCA-whitened latents with the singular values floored at 1% of the largest.

  JAX_PLATFORMS=cpu python -m studies.redcube.diag_cls --root /dev/shm/redcube/off_still --arms lewm6 tile_cls_ar
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probe as P
import train as T
from studies.redcube import env as R


def heatmap_probe(x_train, y_train, x_test, hidden=128, steps=6000, batch=2048, lr=1e-3, seed=0, cells=16):
    """bench.tile_fit for a single-vector latent, on already-normalized inputs: MLP -> 16 cells x (logit, dx, dy)."""
    y = jnp.asarray(y_train / R.W, jnp.float32)
    side = int(round(cells ** 0.5))
    ii, jj = np.divmod(np.arange(cells), side)
    centres = jnp.asarray(np.stack([(jj + 0.5) / side, (ii + 0.5) / side], 1), jnp.float32)
    dims = (x_train.shape[1], hidden, hidden, 3 * cells)
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    params = [(jax.random.normal(keys[i], (dims[i], dims[i + 1])) / np.sqrt(dims[i]), jnp.zeros(dims[i + 1])) for i in range(3)]

    def apply(params, h):
        for i, (w, b) in enumerate(params):
            h = h @ w + b
            if i < 2:
                h = jax.nn.gelu(h)
        h = h.reshape(*h.shape[:-1], cells, 3)
        weights = jax.nn.softmax(h[..., 0], axis=-1)
        return (weights[..., None] * (centres + 0.25 * h[..., 1:])).sum(-2)

    tx = optax.adam(optax.cosine_decay_schedule(lr, steps))
    opt = tx.init(params)
    x = jnp.asarray(x_train, jnp.float32)

    @jax.jit
    def step(params, opt, key):
        idx = jax.random.randint(key, (batch,), 0, x.shape[0])
        grads = jax.grad(lambda p: ((apply(p, x[idx]) - y[idx]) ** 2).mean())(params)
        updates, opt = tx.update(grads, opt, params)
        return optax.apply_updates(params, updates), opt

    for key in jax.random.split(keys[3], steps):
        params, opt = step(params, opt, key)
    return np.asarray(apply(params, jnp.asarray(x_test, jnp.float32))) * R.W


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--test", type=int, default=60)
    args = ap.parse_args()
    with np.load(args.root / "train.npz") as archive:
        frames = archive["frames"][:args.rows + args.test]
        pos = archive["pos"][:args.rows + args.test].reshape(-1, 2).astype(np.float32)
    n_fit = args.rows * frames.shape[1]
    for arm in args.arms:
        bundle, meta = T.load_model(args.root / arm / "model.npz")
        z = np.asarray(P.embed_all(bundle.model, jnp.asarray(frames))).reshape(len(pos), -1)
        sd = z.std(0)
        s = np.linalg.svd(z[:n_fit] - z[:n_fit].mean(0), compute_uv=False)
        p = s ** 2 / (s ** 2).sum()
        rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        print(f"[{arm}] latent {z.shape[1]}-d: dim std min {sd.min():.2e} median {np.median(sd):.3f} max {sd.max():.3f}, "
              f"{(sd < 1e-3 * sd.max()).sum()} dims below 1e-3 of the largest, effective rank {rank:.1f}, "
              f"top-4 variance share {p[:4].sum():.2f}", flush=True)
        mu = z[:n_fit].mean(0)
        plain = (z - mu) / (sd + 1e-6)
        u, s, vt = np.linalg.svd(z[:n_fit] - mu, full_matrices=False)
        white = (z - mu) @ vt.T / np.maximum(s / np.sqrt(n_fit), 1e-2 * s[0] / np.sqrt(n_fit))
        ym = pos[:n_fit].mean(0)
        a = plain[:n_fit]
        w = np.linalg.solve(a.T @ a + 1e-3 * len(a) * np.eye(a.shape[1]), a.T @ (pos[:n_fit] - ym))
        err = lambda guess: float(np.abs(guess - pos[n_fit:]).sum(-1).mean())
        print(f"[{arm}] held-out px: ridge {err(plain[n_fit:] @ w + ym):.2f}, "
              f"MLP on raw {err(heatmap_probe(z[:n_fit], pos[:n_fit], z[n_fit:])):.2f}, "
              f"MLP on standardized {err(heatmap_probe(plain[:n_fit], pos[:n_fit], plain[n_fit:])):.2f}, "
              f"MLP on whitened {err(heatmap_probe(white[:n_fit], pos[:n_fit], white[n_fit:])):.2f}, "
              f"mean-guess {err(np.broadcast_to(ym, pos[n_fit:].shape)):.2f}", flush=True)


if __name__ == "__main__":
    main()
