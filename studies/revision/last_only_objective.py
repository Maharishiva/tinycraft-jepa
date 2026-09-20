"""Prospective last-only objective comparison with the frozen target-gradient grid.

This is a separate six-fit experiment: two worlds x three model seeds. It copies
``target_gradient.py`` and preserves its joint encoder/target gradient paths,
full nine-frame SIGReg, 512-coordinate architecture, initialization, sampled
windows, random projections, optimizer, 8,500 updates, and fixed checkpoints.
The only training-objective change is that prediction MSE averages the final
output/target pair instead of all eight causal-prefix output/target pairs.
The coordinate-mean normalization is unchanged; there is no latent rescaling.

This changes the supervised target positions and their count, as well as the
nominal prediction context. It is not an isolated conditional-history test.
BatchNorm also couples frames during training. No rewards or privileged state
enter world-model training; dones only exclude reset-crossing windows.

Example (run from the repository root):
 python -m studies.revision.last_only_objective --config gray_frame_still \
   --data /path/to/train.npz --seed 0 --mode last_only \
   --out /path/to/last_only/gray_frame_still/m0/last_only

No grid is launched on import. Production admission and GPU scheduling are
separate decisions. All six outcomes are retained, without truncation search.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import train as T


MODES = ("last_only",)
WORLDS = ("gray_frame_still", "gray_frame_wind")
CHECKPOINT_STEPS = (0, 500, 1000, 2000, 4000, 4250, 8500)
ROOT = Path(T.__file__).resolve().parent


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_sha256(value):
    """Hash the dtype, shape, and C-order content, independently of NPZ headers."""
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": str(value.dtype), "shape": value.shape},
                             sort_keys=True).encode())
    # Avoid a multi-GB bytes copy when hashing the frame array.
    digest.update(memoryview(np.ascontiguousarray(value).reshape(-1).view(np.uint8)))
    return digest.hexdigest()


def weights_sha256(model):
    """Hash checkpointed parameters and BatchNorm statistics, in T.save order."""
    digest = hashlib.sha256()
    for kind, state in zip(("p", "b"), T._weights(model)):
        for i, leaf in enumerate(jax.tree.leaves(state)):
            digest.update(f"{kind}.{i}:".encode())
            digest.update(array_sha256(jax.device_get(leaf)).encode())
    return digest.hexdigest()


def production_config(seed):
    """Exact Config corresponding to train_arms.sh's fixed-step tile_factored."""
    return T.Config(seed=seed, steps=8500, lr=1e-3, n_actions=4,
                    encoder="tile", emb_dim=128, enc_depth=6, enc_heads=8,
                    tokens=16, token_dim=32, token_hidden=512,
                    predictor="factored", pred_width=256, pred_depth=6,
                    pred_heads=8, history=8)


def prediction_mse(pred, target, mode):
    """Final causal output only; retain gradients through prediction and target.

    Both inputs remain [batch, history, latent]. The final mean is over batch
    and latent coordinates, with no extra history-length multiplier/divisor.
    """
    if mode not in MODES:
        raise ValueError(f"unknown prediction-loss mode: {mode}")
    return jnp.mean(jnp.square(pred[:, -1] - target[:, -1]))


def make_run(cfg, mode):
    """Copied T.train update; sampling, SIGReg, optimizer use are unchanged."""
    if mode not in MODES:
        raise ValueError(f"unknown prediction-loss mode: {mode}")
    span = cfg.history + cfg.num_preds

    @nnx.jit
    def run(model, opt, frames, actions, table, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def loop(carry, key):
            model, opt = carry
            k_idx, k_sig = jax.random.split(key)
            ids = jax.random.randint(k_idx, (cfg.batch,), 0, table.shape[0])
            start, env = table[ids, 0], table[ids, 1]
            index = start[:, None] + jnp.arange(span)[None]
            window = frames[index, env[:, None]]
            window_actions = actions[index[:, :-1], env[:, None]]

            def loss_fn(model):
                emb = model.encode(window, True)
                act_emb = model.act_embed(window_actions)
                pred = model.predict(emb[:, :cfg.history], act_emb[:, :cfg.history], True)
                pred_loss = prediction_mse(pred, emb[:, cfg.num_preds:], mode)
                reg = T.sigreg(emb.transpose(1, 0, 2), k_sig, cfg.sigreg_knots, cfg.sigreg_proj)
                return pred_loss + cfg.sigreg_weight * reg, (pred_loss, reg)

            (_, (pred_loss, reg)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            opt.update(model, grads)
            return (model, opt), jnp.stack([pred_loss, reg])

        return loop((model, opt), keys)[1]

    return run


def sample_ids(keys, batch, num_windows):
    """Record the exact minibatch table indices without altering the train loop."""
    def one(key):
        k_idx, _ = jax.random.split(key)
        return jax.random.randint(k_idx, (batch,), 0, num_windows)

    return np.asarray(jax.device_get(jax.jit(jax.vmap(one))(keys)))


def source_provenance():
    paths = (Path(__file__).resolve(), Path(T.__file__).resolve(),
             ROOT / "studies/stoch/train_arms.sh", ROOT / "pyproject.toml",
             ROOT / "studies/revision/target_gradient.py")
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "sha256": sources}


def validate_data_identity(meta, config, cfg):
    """Audit the file label before fitting; this metadata never enters a loss."""
    expected_spec = {"noise": "gray", "per_episode": False,
                     "wind": config == "gray_frame_wind", "gate": False}
    spec = {**meta.get("spec", {})}
    # The original redcube collector omits the later stoch collector's gate bit.
    spec.setdefault("gate", False)
    if (meta.get("env") not in ("redcube", "stoch") or meta.get("config") != config
            or meta.get("split") != "train" or meta.get("num_actions") != cfg.n_actions
            or spec != expected_spec):
        raise ValueError(f"dataset metadata does not identify the requested {config} training split")
    return {name: meta[name] for name in ("env", "config", "split", "num_actions", "spec", "seed")
            if name in meta}


def write_manifest(out, manifest):
    temporary = out / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(out / "manifest.json")


def train(data_npz, out, cfg, mode, config, checkpoint_steps=CHECKPOINT_STEPS):
    """Fit one fresh model. Only the CLI exposes the fixed production protocol.

    Tests may supply a small Config and fixed checkpoint list to compare this
    implementation directly with T.train, without a GPU or a full training run.
    """
    if mode not in MODES:
        raise ValueError(f"unknown prediction-loss mode: {mode}")
    if config not in WORLDS:
        raise ValueError(f"unknown world label: {config}")
    checkpoint_steps = tuple(checkpoint_steps)
    if (not checkpoint_steps or tuple(sorted(set(checkpoint_steps))) != checkpoint_steps
            or checkpoint_steps[-1] != cfg.steps or checkpoint_steps[0] != 0):
        raise ValueError("fixed checkpoint steps must start at zero, increase, and end at cfg.steps")
    if any(step % cfg.chunk for step in checkpoint_steps[:-1]):
        raise ValueError("intermediate checkpoints must lie on original chunk boundaries")
    out, data_npz = Path(out), Path(data_npz)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data_sha = sha256(data_npz)
    with np.load(data_npz) as archive:
        # Metadata validates file identity only. No rewards or hidden states are
        # loaded, and no metadata enters the model, objective, or window sampler.
        data_identity = validate_data_identity(json.loads(str(archive["meta"])), config, cfg)
        frames_np = archive["frames"]
        actions_np = archive["actions"].astype(np.int32)
        dones_np = archive["dones"].astype(np.int64)
    if actions_np.size != cfg.data_budget:
        raise ValueError(f"expected exactly {cfg.data_budget} training transitions; got {actions_np.size}")
    if actions_np.ndim != 2 or dones_np.shape != actions_np.shape:
        raise ValueError("actions and dones must be aligned [time, environment] arrays")
    expected_frames = (actions_np.shape[0] + 1, actions_np.shape[1], T.H, T.W, 3)
    if frames_np.shape != expected_frames or frames_np.dtype != np.uint8:
        raise ValueError(f"expected uint8 frames shaped {expected_frames}; got {frames_np.shape}")
    if np.any(actions_np < 0) or np.any(actions_np >= cfg.n_actions):
        raise ValueError("recorded action is outside the configured action vocabulary")
    span = cfg.history + cfg.num_preds
    table_np = T.valid_starts(dones_np, span)
    if len(table_np) == 0:
        raise ValueError("no reset-free training windows")
    data_arrays = {name: array_sha256(value) for name, value in
                   (("frames", frames_np), ("actions_int32", actions_np), ("dones_int64", dones_np))}
    table = jnp.asarray(table_np)
    frames = jnp.asarray(frames_np)
    actions = jnp.asarray(actions_np)
    data_shape = {"frames": list(frames_np.shape), "actions": list(actions_np.shape),
                  "windows": int(table_np.shape[0]), "span": span}
    del frames_np, actions_np, dones_np
    print(f"[data] frames {frames.shape} windows {table.shape[0]} in {time.time() - t0:.1f}s", flush=True)

    model = T.LeWM(cfg, T.rngs(cfg.seed))
    initial_weights_sha = weights_sha256(model)
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    warm = max(1, int(cfg.warmup * cfg.steps))
    schedule = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, warm, cfg.steps, end_value=0.0)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=cfg.weight_decay))
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
    print(f"[model] {n_params / 1e6:.2f}M parameters; prediction objective: {mode}", flush=True)
    run = make_run(cfg, mode)

    keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 7), cfg.steps)
    ids = sample_ids(keys, cfg.batch, len(table_np))
    keys_np = np.asarray(jax.device_get(keys))
    np.savez(out / "sampling.npz", window_table=table_np, update_keys=keys_np, sampled_window_ids=ids)
    sources = source_provenance()
    manifest = {
        "protocol": "last_only_objective_v1", "status": "running", "mode": mode,
        "config_label": config, "cfg": asdict(cfg), "params": n_params,
        "model_seed": cfg.seed, "parameter_rng_seed": cfg.seed,
        "dropout_rng_seed": cfg.seed + 1, "update_rng_seed": cfg.seed + 7,
        "rng_implementation": str(jax.config.jax_default_prng_impl),
        "data_path": str(data_npz.resolve()), "data_sha256": data_sha,
        "data_array_sha256": data_arrays, "data_shape": data_shape, "data_identity": data_identity,
        "initial_weights_sha256": initial_weights_sha,
        "window_table_sha256": array_sha256(table_np), "update_keys_sha256": array_sha256(keys_np),
        "sampled_window_ids_sha256": array_sha256(ids), "sampling_file_sha256": sha256(out / "sampling.npz"),
        "sources": sources, "fixed_checkpoint_steps": list(checkpoint_steps), "checkpoints": [],
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "optax", "numpy")},
        "devices": [str(device) for device in jax.devices()],
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "jax_enable_x64": jax.config.jax_enable_x64,
        "loss_columns": ["last_only_prediction_mse", "sigreg"],
        "prediction_loss_selection": "final output and target only; mean over batch and latent coordinates",
        "prediction_target_positions": [span - 1],
        "prediction_target_count_per_window": 1,
        "sigreg_frame_count_per_window": span,
        "training_fields": ["frames", "actions", "dones_for_window_boundaries_only"],
        "target_gradient_rule": "joint MSE gradients",
        "comparison_reference": "paired_target_gradient_v1 joint arm; frozen sources and completed fits unchanged",
        "sigreg_gradient_rule": "all embeddings, identical to train.py",
        "checkpoint_selection": "fixed final update; no selection by evaluation",
    }
    write_manifest(out, manifest)
    t1 = time.time()
    history = []

    def save_checkpoint(done):
        checkpoint = out / ("model.npz" if done == cfg.steps else f"model_step{done:06d}.npz")
        meta = {"cfg": asdict(cfg), "params": n_params, "train_s": round(time.time() - t0, 1),
                "loop_s": round(time.time() - t1, 1), "data_sha256": data_sha,
                "source_sha256": sha256(__file__), "sources": sources,
                "final_pred_loss": float(history[-1][-1, 0]) if history else None,
                "final_sigreg": float(history[-1][-1, 1]) if history else None,
                "target_gradient_mode": "joint", "prediction_loss_mode": mode,
                "experiment": "last_only_objective_v1",
                "config_label": config, "completed_steps": done,
                "initial_weights_sha256": initial_weights_sha,
                "sampled_window_ids_sha256": manifest["sampled_window_ids_sha256"],
                "fixed_checkpoint_steps": list(checkpoint_steps)}
        T.save(checkpoint, model, meta)
        manifest["checkpoints"].append({"step": done, "path": checkpoint.name,
                                         "sha256": sha256(checkpoint), "weights_sha256": weights_sha256(model)})
        np.savez(out / "losses.npz", history=np.concatenate(history) if history else np.empty((0, 2), np.float32))
        manifest.update(completed_steps=done, elapsed_s=round(time.time() - t0, 1))
        write_manifest(out, manifest)
        return meta

    save_checkpoint(0)
    for begin in range(0, cfg.steps, cfg.chunk):
        losses = run(model, opt, frames, actions, table, keys[begin:begin + cfg.chunk])
        history.append(np.asarray(jax.device_get(losses)))
        done = begin + history[-1].shape[0]
        print(f"[train] {done}/{cfg.steps} pred {history[-1][-1, 0]:.4f} sigreg {history[-1][-1, 1]:.4f}"
              f"  {time.time() - t1:.1f}s", flush=True)
        if not np.isfinite(history[-1]).all():
            np.savez(out / "losses.npz", history=np.concatenate(history))
            manifest.update(status="failed_nonfinite", completed_steps=done)
            write_manifest(out, manifest)
            raise FloatingPointError(f"nonfinite training loss by step {done}; run is not a completed fit")
        if done in checkpoint_steps:
            meta = save_checkpoint(done)
    manifest.update(status="complete", loss_file_sha256=sha256(out / "losses.npz"),
                    final_checkpoint_sha256=manifest["checkpoints"][-1]["sha256"],
                    final_weights_sha256=manifest["checkpoints"][-1]["weights_sha256"])
    write_manifest(out, manifest)
    print(f"[done] {time.time() - t0:.1f}s -> {out / 'model.npz'}", flush=True)
    return model, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new, empty output directory")
    parser.add_argument("--config", choices=WORLDS, required=True, help="provenance label; never input to training")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    train(args.data, args.out, production_config(args.seed), args.mode, args.config)


if __name__ == "__main__":
    main()
