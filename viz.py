"""What is inside a JEPA rollout: latent diagnostics, decoded videos and latent heat maps.

Everything here is off-budget analysis.  The pixel columns come from the stop-gradient decoder
trained by probe.py; the latent numbers need no decoder at all.
"""
import argparse
import functools
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import eval as E
import train as T

CTX, HOR = E.CTX, E.HOR


def windows(data_path, n):
    data = E.load_data(data_path)
    frames, actions = E.window_batch(data, E.sample_indices(data, n))
    return frames, actions


@nnx.jit
def encode(model, frames):
    return model.encode(frames)


@nnx.jit
def predict_from(model, emb_hist, action_hist):
    return model.predict(emb_hist, model.act_embed(action_hist))[:, -1]


@functools.partial(nnx.jit, static_argnames="horizon")
def decode(decoder, latents, horizon):
    flat = decoder(latents.reshape(-1, latents.shape[-1]))
    return jnp.clip(flat * 255.0 + 0.5, 0, 255).astype(jnp.uint8).reshape(
        latents.shape[0], horizon, T.H, T.W, 3)


def rollouts(model, frames, actions, key):
    """True-action and random-action latent rollouts from the same 8-frame context."""
    context, future = actions[:, :CTX], actions[:, CTX:]
    random_future = np.asarray(jax.random.randint(key, future.shape, 0, model.n_actions))
    _, predicted = T._rollout_latents(model, jnp.asarray(frames[:, :CTX]),
                                      T.outgoing_actions(context, future), horizon=HOR)
    _, randomized = T._rollout_latents(model, jnp.asarray(frames[:, :CTX]),
                                       T.outgoing_actions(context, random_future), horizon=HOR)
    return encode(model, jnp.asarray(frames)), predicted, randomized


def action_identification(model, frames, actions, emb):
    """Rank the 17 actions by how well each explains the observed next latent."""
    hist = model.history
    emb_hist = emb[:, CTX - hist:CTX]
    past = jnp.asarray(actions[:, CTX - hist + 1:CTX], jnp.int32)
    batch = emb.shape[0]
    tiled = jnp.repeat(emb_hist, model.n_actions, axis=0)
    candidates = jnp.tile(jnp.arange(model.n_actions, dtype=jnp.int32), batch)[:, None]
    action_hist = jnp.concatenate([jnp.repeat(past, model.n_actions, axis=0), candidates], 1)
    pred = predict_from(model, tiled, action_hist).reshape(batch, model.n_actions, -1)
    cost = jnp.mean(jnp.square(pred - emb[:, CTX][:, None]), -1)
    truth = jnp.asarray(actions[:, CTX], jnp.int32)
    order = jnp.argsort(cost, axis=1)
    rank = jnp.argmin(jnp.where(order == truth[:, None], 0, 1), axis=1)
    spread = jnp.mean(jnp.std(pred, axis=1))
    return np.asarray(rank), float(spread)


def latent_stats(latents):
    """Collapse / Gaussianity diagnostics for the learned embedding space."""
    flat = latents.reshape(-1, latents.shape[-1])
    centred = flat - flat.mean(0)
    eig = np.linalg.eigvalsh(np.cov(centred, rowvar=False))
    eig = np.clip(eig, 0, None)
    return {"dim": int(flat.shape[-1]), "mean_abs": float(np.abs(flat.mean(0)).mean()),
            "std": float(flat.std()), "std_min": float(flat.std(0).min()),
            "std_max": float(flat.std(0).max()),
            "effective_rank": float(eig.sum() ** 2 / np.square(eig).sum()),
            "kurtosis": float(np.mean(((flat - flat.mean(0)) / (flat.std(0) + 1e-8)) ** 4))}


def measure(model, frames, actions, batch, seed):
    """Per-horizon latent errors against persistence, mean and random-action baselines."""
    keys = jax.random.split(jax.random.PRNGKey(seed), (len(frames) + batch - 1) // batch)
    parts, ranks, spreads = [], [], []
    for i, key in enumerate(keys):
        sl = slice(i * batch, (i + 1) * batch)
        emb, pred, rand = rollouts(model, frames[sl], actions[sl], key)
        emb, pred, rand = (np.asarray(jax.device_get(x)) for x in (emb, pred, rand))
        rank, spread = action_identification(model, frames[sl], actions[sl], jnp.asarray(emb))
        parts.append((emb, pred, rand))
        ranks.append(rank)
        spreads.append(spread)
    emb = np.concatenate([p[0] for p in parts])
    pred = np.concatenate([p[1] for p in parts])
    rand = np.concatenate([p[2] for p in parts])
    ranks = np.concatenate(ranks)

    future = emb[:, CTX:]
    persistence = np.repeat(emb[:, CTX - 1:CTX], HOR, 1)
    centre = emb.reshape(-1, emb.shape[-1]).mean(0)
    per_step = lambda a, b: np.mean(np.square(a - b), (0, 2))
    variance = future.var(0).mean(-1)
    curves = {
        "mse_model": per_step(pred, future),
        "mse_persistence": per_step(persistence, future),
        "mse_mean": per_step(centre[None, None], future),
        "variance_true": variance,
        "r2_model": 1 - per_step(pred, future) / variance,
        "r2_persistence": 1 - per_step(persistence, future) / variance,
        "std_ratio": pred.std(0).mean(-1) / future.std(0).mean(-1),
        "dist_to_centre": np.mean(np.square(pred - centre[None, None]), (0, 2)),
        "dist_to_centre_true": np.mean(np.square(future - centre[None, None]), (0, 2)),
        "action_effect": per_step(pred, rand),
        "step_change_true": per_step(future[:, 1:], future[:, :-1]),
    }
    stats = latent_stats(emb)
    stats.update({
        "windows": int(len(emb)),
        "action_top1": float(np.mean(ranks == 0)),
        "action_top3": float(np.mean(ranks < 3)),
        "action_mean_rank": float(ranks.mean()),
        "action_candidate_spread": float(np.mean(spreads)),
        "r2_step1": float(curves["r2_model"][0]),
        "r2_step16": float(curves["r2_model"][-1]),
        "r2_persistence_step1": float(curves["r2_persistence"][0]),
        "r2_persistence_step16": float(curves["r2_persistence"][-1]),
        "std_ratio_step16": float(curves["std_ratio"][-1]),
    })
    return curves, stats, (emb, pred)


def sharpness(video):
    """High-frequency energy per frame: mean |Laplacian| of the greyscale image."""
    grey = video.astype(np.float32).mean(-1) / 255.0
    lap = (4 * grey[:, :, 1:-1, 1:-1] - grey[:, :, :-2, 1:-1] - grey[:, :, 2:, 1:-1]
           - grey[:, :, 1:-1, :-2] - grey[:, :, 1:-1, 2:])
    return np.abs(lap).mean((2, 3))


def pixel_scores(model, decoder, data_path, n, batch, seed):
    """Frozen-benchmark pixel scores split into prediction error and decoder ceiling.

    Uses eval.py's own feature extractor, window sampler and metrics, so the rollout numbers
    reproduce the benchmark; the ceiling row decodes the *encoder's* latents of the real future
    frames, i.e. what the probe could show even with a perfect predictor.
    """
    data = E.load_data(data_path)
    indices = E.sample_indices(data, n)
    feat = E.feature_extractor()
    random_actions = np.random.default_rng(E.EVAL_SEED + 7).integers(
        0, E.N_ACTIONS, size=(len(indices), HOR)).astype(np.int32)
    features = {"real": [], "rollout": [], "ceiling": []}
    errors = {"rollout": [], "random": [], "ceiling": [], "persistence": []}
    sharp = {"real": [], "rollout": [], "ceiling": []}
    for start in range(0, len(indices), batch):
        frames, actions = E.window_batch(data, indices[start:start + batch])
        real, last = frames[:, CTX:], frames[:, CTX - 1].astype(np.float32) / 255
        emb = encode(model, jnp.asarray(frames))
        ceiling = np.asarray(decode(decoder, emb[:, CTX:], horizon=HOR))
        predicted = T.rollout((model, decoder), frames[:, :CTX], actions[:, :CTX], actions[:, CTX:], seed)
        randomized = T.rollout((model, decoder), frames[:, :CTX], actions[:, :CTX],
                               random_actions[start:start + batch], seed)
        persistence = np.repeat(frames[:, CTX - 1:CTX], HOR, 1)
        for key, video in (("real", real), ("rollout", predicted), ("ceiling", ceiling)):
            features[key].append(E.clip_feats(feat, video))
            sharp[key].append(sharpness(video))
        for key, video in (("rollout", predicted), ("random", randomized),
                           ("ceiling", ceiling), ("persistence", persistence)):
            errors[key].append(E.salient_mse(video, real, last))
    stack = {k: np.concatenate(v) for k, v in features.items()}
    errors = {k: np.concatenate(v) for k, v in errors.items()}
    sharp = {k: np.concatenate(v).mean(0) for k, v in sharp.items()}
    return {"windows": int(len(indices)),
            "sharpness_real": sharp["real"].tolist(),
            "sharpness_rollout": sharp["rollout"].tolist(),
            "sharpness_ceiling": sharp["ceiling"].tolist(),
            "gFVD_rollout": E.frechet(stack["real"], stack["rollout"]),
            "gFVD_ceiling": E.frechet(stack["real"], stack["ceiling"]),
            "AF": E.action_following(errors["rollout"], errors["random"]),
            "salient_mse_rollout": float(errors["rollout"].mean()),
            "salient_mse_ceiling": float(errors["ceiling"].mean()),
            "salient_mse_persistence": float(errors["persistence"].mean())}


def tile(rows, scale=3, gap=2):
    """Stack image rows into one frame with separators."""
    sep_v = np.full((rows[0][0].shape[0], gap, 3), (90, 90, 110), np.uint8)
    out = []
    for row in rows:
        line = [row[0]]
        for image in row[1:]:
            line += [sep_v, image]
        out.append(np.concatenate(line, 1))
    sep_h = np.full((gap, out[0].shape[1], 3), (90, 90, 110), np.uint8)
    grid = out[0]
    for row in out[1:]:
        grid = np.concatenate([grid, sep_h, row], 0)
    return np.repeat(np.repeat(grid, scale, 0), scale, 1)


def rollout_gif(model, decoder, frames, actions, path, k=6, seed=0):
    """real | decoded encoder latent (probe ceiling) | decoded JEPA rollout | error."""
    import imageio.v2 as imageio

    emb, pred, _ = rollouts(model, frames[:k], actions[:k], jax.random.PRNGKey(seed))
    recon = np.asarray(decode(decoder, emb, horizon=emb.shape[1]))
    imagined = np.asarray(decode(decoder, pred, horizon=HOR))
    real = frames[:k]
    images = []
    for t in range(CTX + HOR):
        rows = []
        for i in range(k):
            if t < CTX:
                dream = recon[i, t]
            else:
                dream = imagined[i, t - CTX]
            error = np.abs(dream.astype(np.int16) - real[i, t].astype(np.int16)).astype(np.uint8)
            rows.append([real[i, t], recon[i, t], dream, error])
        images.append(tile(rows))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, images, duration=1000 / 4, loop=0)
    return path


def demo_gif(model, decoder, frames, actions, path, k=4, seed=0, first=0):
    """Ground truth beside the imagined rollout: 8 real priming frames, then 16 imagined ones."""
    import matplotlib

    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    real = frames[first:first + k]
    _, pred, _ = rollouts(model, real, actions[first:first + k], jax.random.PRNGKey(seed))
    imagined = np.asarray(decode(decoder, pred, horizon=HOR))
    images = []
    for t in range(CTX + HOR):
        dreaming = t >= CTX
        figure, axes = plt.subplots(k, 2, figsize=(3.1, 1.52 * k), dpi=130, squeeze=False)
        for i in range(k):
            panels = (real[i, t], imagined[i, t - CTX] if dreaming else real[i, t])
            for j, (ax, image) in enumerate(zip(axes[i], panels)):
                ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#d1495b" if dreaming and j else "#cccccc")
                    spine.set_linewidth(2.0 if dreaming and j else 0.8)
                if i == 0:
                    ax.set_title(("ground truth", "LeWM imagined")[j], fontsize=8)
        figure.suptitle(f"imagined step {t - CTX + 1} / {HOR}" if dreaming
                        else f"priming frame {t + 1} / {CTX}", fontsize=9, y=0.995)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.canvas.draw()
        images.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
        plt.close(figure)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, images + images[-1:] * 3, duration=1000 / 3, loop=0)
    return path


def heatmap_gif(model, frames, actions, path, index=0, seed=0):
    """Latent heat maps: encoder latent, JEPA prediction and their difference, frame by frame."""
    import matplotlib

    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    emb, pred, _ = rollouts(model, frames[:4], actions[:4], jax.random.PRNGKey(seed))
    emb = np.asarray(jax.device_get(emb))[index]
    pred = np.asarray(jax.device_get(pred))[index]
    grid = (12, emb.shape[-1] // 12)
    limit = float(np.abs(emb).max())
    images = []
    for t in range(HOR):
        truth, guess = emb[CTX + t], pred[t]
        figure, axes = plt.subplots(1, 4, figsize=(11, 2.6), dpi=110)
        axes[0].imshow(frames[index, CTX + t])
        axes[0].set_title(f"frame t+{t + 1}")
        for ax, data, title in zip(axes[1:], (truth, guess, guess - truth),
                                   ("encoder latent", "JEPA rollout", "difference")):
            ax.imshow(data.reshape(grid), cmap="coolwarm", vmin=-limit, vmax=limit)
            ax.set_title(title)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        figure.tight_layout()
        figure.canvas.draw()
        images.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
        plt.close(figure)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, images, duration=1000 / 3, loop=0)
    return path


def drift_plot(curves, stats, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(1, HOR + 1)
    figure, axes = plt.subplots(1, 4, figsize=(17, 3.9), dpi=120)
    axes[0].plot(steps, curves["mse_model"], "o-", label="LeWM rollout")
    axes[0].plot(steps, curves["mse_persistence"], "s--", label="copy last latent")
    axes[0].plot(steps, curves["mse_mean"], "^:", label="dataset mean latent")
    axes[0].set_title("latent prediction error")
    axes[0].set_xlabel("imagined step")
    axes[0].set_ylabel("MSE per dimension")
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, curves["r2_model"], "o-", label="LeWM rollout")
    axes[1].plot(steps, curves["r2_persistence"], "s--", label="copy last latent")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("variance explained ($R^2$)")
    axes[1].set_xlabel("imagined step")
    axes[1].legend(fontsize=8)
    axes[2].plot(steps, curves["std_ratio"], "o-")
    axes[2].axhline(1, color="k", lw=0.8, ls=":")
    axes[2].set_ylim(0, 1.35)
    axes[2].set_title("predicted / true spread\n(< 1 means regression to the mean)")
    axes[2].set_xlabel("imagined step")
    axes[3].plot(steps, curves["action_effect"], "o-", label="true vs random actions")
    axes[3].plot(steps[:-1], curves["step_change_true"], "s--", label="true frame-to-frame change")
    axes[3].set_title("action sensitivity")
    axes[3].set_xlabel("imagined step")
    axes[3].legend(fontsize=8)
    figure.suptitle(f"{title}   |   action id top-1 {stats['action_top1']:.3f}"
                    f"   effective rank {stats['effective_rank']:.1f}/{stats['dim']}", y=1.04)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/tinywmcraft_500k/eval.npz"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--pixels", action="store_true", help="gFVD of the rollout and of the decoder ceiling")
    parser.add_argument("--demo", type=int, default=0, help="episodes in the side-by-side demo gif")
    args = parser.parse_args()

    bundle, meta = T.load_model(args.model)
    model, decoder = bundle.model, bundle.decoder
    tag = args.tag or args.model.parent.name
    frames, actions = windows(args.data, args.windows)
    curves, stats, _ = measure(model, frames, actions, args.batch, args.seed)
    if args.pixels:
        stats.update(pixel_scores(model, decoder, args.data, args.windows, 64, args.seed))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    payload = {"tag": tag, "model": str(args.model), "train_s": meta.get("train_s"),
               "cfg": meta.get("cfg"), "stats": stats,
               "curves": {k: [round(float(x), 6) for x in v] for k, v in curves.items()}}
    (out / f"{tag}_latent.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}, indent=1))
    drift_plot(curves, stats, out / f"{tag}_drift.png", tag)
    if args.gif:
        heatmap_gif(model, frames, actions, out / f"{tag}_latents.gif", seed=args.seed)
        if decoder is not None:
            rollout_gif(model, decoder, frames, actions, out / f"{tag}_rollout.gif", seed=args.seed)
    if args.demo:
        demo_gif(model, decoder, frames, actions, out / f"{tag}_demo.gif", k=args.demo, seed=args.seed)
    print(f"[viz] wrote {out}/{tag}_*", flush=True)


if __name__ == "__main__":
    main()
