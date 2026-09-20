"""Fixed, post-hoc position readouts for the last-only objective experiment.

Run only after freezing a world-model checkpoint, for every arm and training seed::

    python -m studies.revision.last_only_probe \
        --model run/model.npz --train data/train.npz --eval data/eval.npz --out run/readout

The primary endpoint is the final MLP's held-out coordinate-mean absolute error
(`mlp.mae_px`). Ridge is a separately reported diagnostic, never a readout selector.
Across matched model seeds, the primary contrast within each world is
    MAE_joint - MAE_last_only.
The wind-minus-still difference in these benefits is descriptive. The numeric
sampling, encoding, ridge and MLP routines exactly match the frozen
``target_gradient_probe.py``; only campaign documentation/metadata differ.
This file trains readouts only: no position label, reward, state feature, or
readout gradient enters the frozen world model. All frames are eligible,
including initial and post-reset frames. No reward/success filtering is used.

The fixed train/eval samples, raw latent ridge, and raw-input nonlinear probe
are deliberately shared across all arms. Linear failure is not interpreted as
absence of information. There is no feature whitening, latent-coordinate rescaling,
handcrafted image feature, readout hyperparameter sweep, or test-set selection.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

TRAIN_FRAMES = 50_000
EVAL_FRAMES = 10_000
SAMPLE_SEED = 19417
LATENT_DIM = 512
RIDGE_FACTOR = 1e-3
MLP_SEED = 0
MLP_STEPS = 4_000
MLP_BATCH = 1_024
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
PIXEL_UNITS = 63.0
REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sample_indices(shape, count, seed):
    """Uniformly select flat frame IDs without replacement; order is preserved."""
    if len(shape) != 2 or any(x <= 0 for x in shape):
        raise ValueError(f"Expected positive [time, environment] frame axes, got {shape}")
    total = int(np.prod(shape))
    if count > total:
        raise ValueError(f"Fixed sample requires {count} frames, archive has only {total}")
    flat = np.random.default_rng(seed).choice(total, count, replace=False).astype(np.int64)
    indices = np.stack(np.unravel_index(flat, shape), axis=-1).astype(np.int64)
    return flat, indices


def archive_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        if not {"meta", "frames", "pos"}.issubset(archive.files):
            raise ValueError(f"{path}: expected meta, frames, and pos arrays")
        meta = json.loads(str(archive["meta"]))
        pos = archive["pos"]
    if pos.ndim != 3 or pos.shape[-1] != 2 or not np.isfinite(pos).all():
        raise ValueError(f"{path}: expected finite positions shaped [time, environment, 2]")
    return meta, pos.shape[:2]


def validate_archives(train_meta, eval_meta):
    """Reject the split leakage that a later-time holdout would not detect."""
    if train_meta.get("split") != "train" or eval_meta.get("split") != "eval":
        raise ValueError("Expected explicit train and eval archive splits")
    seeds = (train_meta.get("seed"), eval_meta.get("seed"))
    if not all(isinstance(seed, int) for seed in seeds) or seeds[0] == seeds[1]:
        raise ValueError("Train and eval must have distinct, recorded integer collection seeds")
    for key in ("env", "config", "spec", "policy", "num_actions"):
        if key not in train_meta or key not in eval_meta or train_meta[key] != eval_meta[key]:
            raise ValueError(f"Train and eval must share the recorded {key}")
    if train_meta["env"] != "redcube" or train_meta["policy"] != "uniform random":
        raise ValueError("This registered experiment requires redcube uniform-random archives")


def fit_ridge(features, positions):
    """Minimize mean squared residual plus lambda ||W||² on raw centered latents.

    Covariance uses the population (1/N) convention. Lambda is fixed at 1e-3
    times its mean diagonal, with no per-coordinate scaling. Centering X and Y
    yields an unpenalized intercept. An exactly constant representation gives
    the training-mean predictor without a numerical denominator floor.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(positions, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x), 2) or not len(x):
        raise ValueError("Ridge requires nonempty [N, D] features and [N, 2] positions")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite ridge inputs")
    x_mean, y_mean = x.mean(0), y.mean(0)
    xc, yc = x - x_mean, y - y_mean
    covariance = xc.T @ xc / len(x)
    mean_feature_variance = float(np.trace(covariance) / x.shape[1])
    penalty = RIDGE_FACTOR * mean_feature_variance
    if mean_feature_variance == 0.0:
        weights = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    else:
        weights = np.linalg.solve(covariance + penalty * np.eye(x.shape[1]), xc.T @ yc / len(x))
    return {
        "weights": weights,
        "intercept": y_mean - x_mean @ weights,
        "feature_mean": x_mean,
        "position_mean": y_mean,
        "ridge_penalty": penalty,
        **covariance_statistics(covariance),
    }


def covariance_statistics(covariance):
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    positive = eigenvalues[eigenvalues > 0]
    probabilities = positive / positive.sum() if len(positive) else positive
    return {
        "covariance_eigenvalues": eigenvalues,
        "mean_feature_variance": float(np.trace(covariance) / covariance.shape[0]),
        "effective_rank": float(np.exp(-(probabilities * np.log(probabilities)).sum())) if len(positive) else 0.0,
        "participation_ratio": float(eigenvalues.sum() ** 2 / (eigenvalues @ eigenvalues)) if len(positive) else 0.0,
    }


def predict_ridge(readout, features):
    return np.asarray(features, dtype=np.float64) @ readout["weights"] + readout["intercept"]


def metrics(target, prediction):
    """Pixel errors; R² uses the held-out target mean, not the fitted intercept."""
    target, prediction = np.asarray(target, np.float64), np.asarray(prediction, np.float64)
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("Expected matching [N, 2] position targets and predictions")
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite readout predictions")
    error = prediction - target
    sse = np.square(error).sum(0)
    sst = np.square(target - target.mean(0)).sum(0)
    r2 = [float(1 - a / b) if b > 0 else None for a, b in zip(sse, sst)]
    return {
        "mae_px": float(np.abs(error).mean()),
        "l1_px": float(np.abs(error).sum(-1).mean()),
        "rmse_px": float(np.sqrt(np.square(error).mean())),
        "euclidean_rmse_px": float(np.sqrt(np.square(error).sum(-1).mean())),
        "mae_xy_px": np.abs(error).mean(0).tolist(),
        "rmse_xy_px": np.sqrt(np.square(error).mean(0)).tolist(),
        "r2": float(1 - sse.sum() / sst.sum()) if sst.sum() > 0 else None,
        "r2_xy": r2,
    }


def encode_sample(model, archive_path, shape, count, seed, chunk):
    """Read only pixels and post-hoc labels; encode frozen frames in inference mode."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    flat, indices = sample_indices(shape, count, seed)
    with np.load(archive_path, allow_pickle=False) as archive:
        frames = archive["frames"]
        positions = archive["pos"][indices[:, 0], indices[:, 1]].astype(np.float32)
    if frames.shape != (*shape, 63, 63, 3) or frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frames shaped {(*shape, 63, 63, 3)}, got {frames.shape} {frames.dtype}")

    @nnx.jit
    def encode(m, frame_batch):
        return m.encode(frame_batch[None], train=False)[0]

    features = np.empty((count, LATENT_DIM), dtype=np.float32)
    frame_hashes = np.empty(count, dtype="S64")
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        selected = frames[indices[start:end, 0], indices[start:end, 1]]
        frame_hashes[start:end] = [hashlib.sha256(memoryview(np.ascontiguousarray(frame)).cast("B")).hexdigest()
                                   for frame in selected]
        # Padding only avoids a second encoder compilation; padded outputs are discarded.
        batch = np.concatenate([selected, np.repeat(selected[-1:], chunk - len(selected), 0)]) if len(selected) < chunk else selected
        features[start:end] = np.asarray(jax.device_get(encode(model, jnp.asarray(batch))))[:end - start]
        if start == 0 or end == count or (start // chunk) % 25 == 0:
            print(f"[encode {archive_path.name}] {end}/{count}", flush=True)
    if not np.isfinite(features).all():
        raise ValueError("Frozen encoder returned nonfinite features")
    return {"features": features, "positions": positions, "flat_indices": flat,
            "indices": indices, "frame_sha256": frame_hashes}


def fit_mlp(features, positions):
    """One fixed raw-input FP32 MLP, with no access to held-out labels or pixels."""
    import jax
    import jax.numpy as jnp
    import optax

    key = jax.random.PRNGKey(MLP_SEED)
    key, init_key = jax.random.split(key)
    params = []
    for width_in, width_out, layer_key in zip((LATENT_DIM, 256, 256), (256, 256, 2), jax.random.split(init_key, 3)):
        kernel = jax.nn.initializers.glorot_uniform()(layer_key, (width_in, width_out), jnp.float32)
        params.append({"kernel": kernel, "bias": jnp.zeros(width_out, dtype=jnp.float32)})
    # Coordinate labels alone are converted from pixels to known image-width units.
    # Encoder outputs stay exactly in their original raw coordinates.
    x = jnp.asarray(features, dtype=jnp.float32)
    y = jnp.asarray(positions / PIXEL_UNITS, dtype=jnp.float32)
    optimizer = optax.adamw(MLP_LR, b1=0.9, b2=0.999, eps=1e-8, weight_decay=MLP_WEIGHT_DECAY)
    state = optimizer.init(params)

    def forward(parameters, batch):
        for layer in parameters[:-1]:
            batch = jax.nn.gelu(batch @ layer["kernel"] + layer["bias"])
        return batch @ parameters[-1]["kernel"] + parameters[-1]["bias"]

    @jax.jit
    def update(parameters, opt_state, rng, training_x, training_y):
        rng, batch_key = jax.random.split(rng)
        indices = jax.random.randint(batch_key, (MLP_BATCH,), 0, len(features))

        def loss(p):
            return jnp.square(forward(p, training_x[indices]) - training_y[indices]).mean()

        value, grads = jax.value_and_grad(loss)(parameters)
        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        return optax.apply_updates(parameters, updates), opt_state, rng, value

    losses = np.empty(MLP_STEPS, dtype=np.float32)
    for step in range(MLP_STEPS):
        params, state, key, loss = update(params, state, key, x, y)
        losses[step] = float(loss)
        if step == 0 or (step + 1) % 500 == 0:
            print(f"[MLP] {step + 1}/{MLP_STEPS} normalized MSE={losses[step]:.8g}", flush=True)
    if not np.isfinite(losses).all():
        raise ValueError("Nonfinite MLP loss; do not silently change the registered optimization")

    @jax.jit
    def predict(batch):
        return forward(params, batch) * PIXEL_UNITS

    def predict_numpy(values):
        return np.concatenate([np.asarray(jax.device_get(predict(jnp.asarray(values[i:i + 1024]))))
                               for i in range(0, len(values), 1024)])

    saved_params = {f"layer_{i}_{name}": np.asarray(value) for i, layer in enumerate(params) for name, value in layer.items()}
    return predict_numpy, saved_params, losses


def source_provenance():
    sources = [Path(__file__).resolve(), REPO / "train.py", REPO / "probe.py", REPO / "studies/redcube/env.py"]
    result = {"runtime_source_sha256": {str(path.relative_to(REPO)): sha256(path) for path in sources},
              "python": platform.python_version(), "packages": {}}
    for name in ("numpy", "jax", "jaxlib", "flax", "optax"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        result["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        result["git_status"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        result["git_head"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encode-batch", type=int, default=256, help="Computational chunk only; does not change the sample or readout")
    args = parser.parse_args()
    if args.encode_batch <= 0:
        parser.error("--encode-batch must be positive")
    # Refuse to mix a previous partial/completed run with a new checkpoint.
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    hashes = {name: sha256(getattr(args, name)) for name in ("model", "train", "eval")}
    train_meta, train_shape = archive_metadata(args.train)
    eval_meta, eval_shape = archive_metadata(args.eval)
    validate_archives(train_meta, eval_meta)
    if hashes["train"] == hashes["eval"]:
        raise ValueError("Train and eval archives have identical content")
    # Fail before importing JAX if a data archive cannot meet the registered count.
    sample_indices(train_shape, TRAIN_FRAMES, SAMPLE_SEED)
    sample_indices(eval_shape, EVAL_FRAMES, SAMPLE_SEED + 1)
    protocol = {
        "status": "running", "protocol": "last_only_readouts_v1", "primary_endpoint": "eval.mlp.mae_px",
        "primary_contrast": "joint_world-last_only_world, paired by model seed and reported separately for each world",
        "paths": {name: str(getattr(args, name).resolve()) for name in ("model", "train", "eval")},
        "sha256": hashes, "train_metadata": train_meta, "eval_metadata": eval_meta,
        "sampling": {"train_count": TRAIN_FRAMES, "eval_count": EVAL_FRAMES, "train_seed": SAMPLE_SEED,
                     "eval_seed": SAMPLE_SEED + 1, "replacement": False, "eligible": "every archived frame; no filtering",
                     "rng": "numpy.default_rng (PCG64)", "flat_index_order": "time-major then environment",
                     "frame_identity": "archive sha256 plus [time_index, environment_index]",
                     "frame_content_hash": "SHA-256 of contiguous uint8 HWC bytes"},
        "ridge": {"factor": RIDGE_FACTOR, "penalty": "1e-3 * trace(population feature covariance) / D",
                  "features": "raw centered latents, no per-coordinate scaling", "intercept": "unpenalized"},
        "mlp": {"widths": [LATENT_DIM, 256, 256, 2], "activation": "GELU (JAX default approximate)",
                "dtype": "float32", "seed": MLP_SEED, "steps": MLP_STEPS, "batch": MLP_BATCH,
                "optimizer": "AdamW", "lr": MLP_LR, "weight_decay": MLP_WEIGHT_DECAY,
                "b1": 0.9, "b2": 0.999, "epsilon": 1e-8, "initialization": "Glorot uniform kernels, zero biases",
                "batch_sampling": "with replacement from fixed training sample",
                "inputs": "raw 512-dimensional encoder outputs; no normalization",
                "target_scale_px": PIXEL_UNITS, "checkpoint": "final step; no selection"},
        "label_use": "position labels train frozen-model readouts only; no labels or readout gradients update the world model",
        "position_convention": "archive pos coordinates (top-left cube location), in pixels",
        "metric_conventions": {"mae_px": "mean absolute error over frames and both coordinates",
                               "l1_px": "mean summed-coordinate absolute error per frame",
                               "rmse_px": "root mean squared error over frames and coordinates",
                               "r2": "pooled coordinate SSE relative to test-set coordinate means",
                               "effective_rank": "exp(entropy(normalized nonnegative covariance eigenvalues))"},
        **source_provenance(),
    }
    write_json(args.out / "protocol.json", protocol)

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import train as T

    bundle, model_meta = T.load_model(args.model)
    model = bundle.model
    if model.latent_dim != LATENT_DIM:
        raise ValueError(f"Registered readout requires exactly {LATENT_DIM} latent dimensions, got {model.latent_dim}")
    if model_meta.get("data_sha256") != hashes["train"]:
        raise ValueError("Checkpoint data_sha256 does not match the exact training archive")
    protocol.update(model_metadata=model_meta, devices=[str(device) for device in jax.devices()],
                    model_training_labels="see preserved model metadata and training-source provenance")
    write_json(args.out / "protocol.json", protocol)

    samples = {
        "train": encode_sample(model, args.train, train_shape, TRAIN_FRAMES, SAMPLE_SEED, args.encode_batch),
        "eval": encode_sample(model, args.eval, eval_shape, EVAL_FRAMES, SAMPLE_SEED + 1, args.encode_batch),
    }
    train_sample = samples["train"]
    ridge = fit_ridge(train_sample["features"], train_sample["positions"])
    mlp_predict, mlp_parameters, mlp_losses = fit_mlp(train_sample["features"], train_sample["positions"])
    np.savez(args.out / "ridge_readout.npz", **ridge)
    np.savez(args.out / "mlp_readout.npz", **mlp_parameters, normalized_training_mse=mlp_losses)

    eval_centered = samples["eval"]["features"].astype(np.float64)
    eval_centered -= eval_centered.mean(0)
    eval_covariance_stats = covariance_statistics(eval_centered.T @ eval_centered / len(eval_centered))
    results = {"primary_endpoint": protocol["primary_endpoint"], "representation_train": {
        key: ridge[key] for key in ("mean_feature_variance", "effective_rank", "participation_ratio", "ridge_penalty")},
        "representation_eval": {key: eval_covariance_stats[key] for key in ("mean_feature_variance", "effective_rank", "participation_ratio")}}
    saved = {"eval_covariance_eigenvalues": eval_covariance_stats["covariance_eigenvalues"]}
    for split, sample in samples.items():
        y = sample["positions"]
        predictions = {"ridge": predict_ridge(ridge, sample["features"]), "mlp": mlp_predict(sample["features"]),
                       "train_mean": np.broadcast_to(ridge["position_mean"], y.shape)}
        results[split] = {name: metrics(y, prediction) for name, prediction in predictions.items()}
        protocol["sampling"][split + "_indices_sha256"] = array_sha256(sample["indices"])
        protocol["sampling"][split + "_positions_sha256"] = array_sha256(y)
        protocol["sampling"][split + "_raw_features_sha256"] = array_sha256(sample["features"])
        for name in ("positions", "flat_indices", "indices", "frame_sha256"):
            saved[split + "_" + name] = sample[name]
        for name, prediction in predictions.items():
            saved[f"{split}_{name}_prediction_px"] = prediction
            saved[f"{split}_{name}_error_px"] = prediction - y
    np.savez_compressed(args.out / "per_frame.npz", **saved)
    if sha256(args.model) != hashes["model"]:
        raise RuntimeError("Checkpoint changed during evaluation")
    results["elapsed_s"] = time.time() - started
    write_json(args.out / "results.json", results)
    protocol.update(status="complete", elapsed_s=results["elapsed_s"], artifacts={
        name: sha256(args.out / name) for name in ("results.json", "per_frame.npz", "ridge_readout.npz", "mlp_readout.npz")})
    write_json(args.out / "protocol.json", protocol)
    print(json.dumps({"out": str(args.out), "eval": results["eval"], "elapsed_s": results["elapsed_s"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
