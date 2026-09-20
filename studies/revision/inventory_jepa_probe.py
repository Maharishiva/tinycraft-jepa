"""Fit new pixel/reward/done readouts on all registered 500k training labels.

The JEPA encoder and predictor are frozen. No validation set, latent transform,
value head, reward transform, action filter, or model-selection rule is used.
All eligible training transitions are sampled uniformly with replacement.
Large frame/embedding caches are disk-backed; only each encode batch is moved
to the accelerator. The fitting arrays fit in approximately 7 GB of device RAM.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import probe as P
import train as T
from studies.revision.inventory_jepa import (
    EXPECTED_DATA_SHA256, EXPECTED_MODEL_SHA256, FORMAT, TransitionHeads,
    assert_frozen_archive, file_sha256, validate_model, weights_digest,
)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def materialize_frames(data_path, cache_dir, data_sha256):
    """Stream the .npy ZIP member; avoid decompressing the 6-GB bank into RAM."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"frames_{data_sha256}.npy"
    manifest = target.with_suffix(".json")
    with zipfile.ZipFile(data_path) as archive:
        info = archive.getinfo("frames.npy")
        expected = {"data_sha256": data_sha256, "member": "frames.npy",
                    "size": info.file_size, "zip_crc32": info.CRC}
        ready = target.exists() and manifest.exists() and json.loads(manifest.read_text()) == expected
        if not ready or target.stat().st_size != info.file_size:
            temporary = target.with_suffix(".partial.npy")
            with archive.open(info) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            temporary.replace(target)
            write_json(manifest, expected)
    frames = np.load(target, mmap_mode="r", allow_pickle=False)
    if frames.dtype != np.uint8 or frames.ndim != 5 or frames.shape[2:] != (63, 63, 3):
        raise ValueError(f"Unexpected training frames: {frames.shape}, {frames.dtype}")
    return frames


def current_frame_bank(frames, actions, rewards, dones, expected_count=500_000):
    """The archive stores T+1 frames and T outgoing transitions per environment."""
    if actions.ndim != 2 or actions.size != expected_count:
        raise ValueError(f"Require exactly {expected_count} time-by-environment transitions")
    if frames.shape[:2] != (actions.shape[0] + 1, actions.shape[1]):
        raise ValueError("Require T+1 observations aligned with T transition labels")
    if rewards.shape != actions.shape or dones.shape != actions.shape:
        raise ValueError("Native reward/done labels must match the action bank")
    if not np.isfinite(rewards).all() or not np.isin(dones, (0, 1)).all() or np.any((actions < 0) | (actions >= 17)):
        raise ValueError("Invalid native transition labels")
    return frames[:-1]


def embed_frames(model, frames, cache_dir, model_sha256, data_sha256, chunk=512):
    """Cache unmodified float32 embeddings in exact time-major archive order."""
    flat = frames.reshape(-1, 63, 63, 3)
    path = cache_dir / f"embeddings_{model_sha256}_{data_sha256}.npy"
    manifest = path.with_suffix(".json")
    identity = {"model_sha256": model_sha256, "data_sha256": data_sha256,
                "shape": [len(flat), 512], "dtype": "float32", "train": False,
                "transform": "none", "order": "time-major/environment-minor"}
    if path.exists() and manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved.get("identity") == identity and saved.get("array_sha256") == file_sha256(path):
            return np.load(path, mmap_mode="r", allow_pickle=False), saved
    temporary = path.with_suffix(".partial.npy")
    latents = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(flat), 512))

    @nnx.jit
    def encode(module, batch):
        return module.encode(batch[:, None], train=False)[:, 0]

    for begin in range(0, len(flat), chunk):
        end = min(begin + chunk, len(flat))
        batch = flat[begin:end]
        if len(batch) < chunk:
            batch = np.concatenate([batch, np.repeat(batch[-1:], chunk - len(batch), axis=0)])
        values = np.asarray(jax.device_get(encode(model, jnp.asarray(batch))))[:end - begin]
        if values.shape != (end - begin, 512) or not np.isfinite(values).all():
            raise ValueError("Encoder produced invalid raw embeddings")
        latents[begin:end] = values
        if begin == 0 or end == len(flat) or (begin // chunk) % 50 == 0:
            print(f"[embed] {end}/{len(flat)}", flush=True)
    latents.flush()
    del latents
    temporary.replace(path)
    saved = {"identity": identity, "array_sha256": file_sha256(path)}
    write_json(manifest, saved)
    return np.load(path, mmap_mode="r", allow_pickle=False), saved


def train_heads(latents, actions, rewards, dones, *, steps=6000, batch=1024,
                lr=1e-3, hidden=512, seed=0, block=500):
    """Original MSE reward + BCE done fitting recipe, with no value objective."""
    heads = TransitionHeads(latents.shape[-1], hidden, T.rngs(seed + 11))
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, max(1, steps // 20), steps, end_value=0.0)
    optimizer = nnx.Optimizer(heads, optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4)
    ), wrt=nnx.Param)
    z, a, r, d = latents, actions.reshape(-1), rewards.reshape(-1), dones.reshape(-1)
    if not (len(z) == len(a) == len(r) == len(d)):
        raise ValueError("Each current-frame embedding needs its aligned transition labels")

    @nnx.jit
    def run(module, opt, z, a, r, d, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def update(carry, key):
            module, opt = carry
            # Preserve the old probe's first split and index sampling convention.
            batch_key, _, _ = jax.random.split(key, 3)
            ids = jax.random.randint(batch_key, (batch,), 0, len(z))

            def loss_fn(candidate):
                pred_r, logit_d = candidate(z[ids], a[ids])
                mse = jnp.square(pred_r - r[ids]).mean()
                bce = optax.sigmoid_binary_cross_entropy(logit_d, d[ids]).mean()
                return mse + bce, jnp.stack([mse, bce])

            (_, losses), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(module)
            opt.update(module, gradients)
            return (module, opt), losses

        return update((module, opt), keys)[1]

    keys = jax.random.split(jax.random.PRNGKey(seed + 5), steps)
    history = []
    for begin in range(0, steps, block):
        values = np.asarray(jax.device_get(run(heads, optimizer, z, a, r, d, keys[begin:begin + block])))
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite head fitting losses")
        history.append(values)
        print(f"[heads] {begin + len(values)}/{steps} reward {values[-1, 0]:.6f} done {values[-1, 1]:.6f}", flush=True)
    return heads, np.concatenate(history)


def training_head_diagnostics(heads, latents, actions, rewards, dones, chunk=8192):
    """Training-set diagnostics only: never used to select the final checkpoint."""
    predicted_r, predicted_d = [], []
    apply = nnx.jit(lambda module, z, a: module(z, a))
    actions = actions.reshape(-1)
    for begin in range(0, len(latents), chunk):
        r, d = apply(heads, latents[begin:begin + chunk], actions[begin:begin + chunk])
        predicted_r.append(np.asarray(jax.device_get(r)))
        predicted_d.append(np.asarray(jax.device_get(jax.nn.sigmoid(d))))
    r, d = np.concatenate(predicted_r), np.concatenate(predicted_d)
    target_r, target_d = np.asarray(rewards).reshape(-1), np.asarray(dones).reshape(-1)
    nonzero = target_r != 0
    correlation = float(np.corrcoef(r, target_r)[0, 1]) if r.std() > 0 and target_r.std() > 0 else None
    return {"scope": "all training labels; descriptive only", "n": len(r),
            "reward_mse": float(np.square(r - target_r).mean()), "reward_corr": correlation,
            "reward_mean_predicted": float(r.mean()), "reward_mean_true": float(target_r.mean()),
            "reward_nonzero_count": int(nonzero.sum()),
            "reward_nonzero_mse": float(np.square(r[nonzero] - target_r[nonzero]).mean()) if nonzero.any() else None,
            "done_brier": float(np.square(d - target_d).mean()),
            "done_mean_predicted": float(d.mean()), "done_mean_true": float(target_d.mean())}


def save_bundle(path, model, decoder, heads, meta):
    """Explicit inventory_heads namespace prevents original Heads-tree confusion."""
    arrays = {}
    for prefix, module in (("model", model), ("decoder", decoder), ("inventory_heads", heads)):
        for kind, state in zip(("p", "b"), T._weights(module)):
            arrays.update({f"{prefix}.{kind}.{i}": np.asarray(jax.device_get(leaf))
                           for i, leaf in enumerate(jax.tree.leaves(state))})
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, meta=json.dumps(meta, allow_nan=False), **arrays)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New NPZ; refuses overwrite")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--encode-batch", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--head-steps", type=int, default=6000)
    parser.add_argument("--head-batch", type=int, default=1024)
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() or args.out.resolve() == args.model.resolve():
        parser.error("Output must be a new path distinct from the frozen checkpoint")
    if min(args.encode_batch, args.steps, args.batch, args.hidden, args.depth,
           args.head_steps, args.head_batch, args.head_hidden) <= 0:
        parser.error("All counts must be positive")
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hashes = {"model": file_sha256(args.model), "data": file_sha256(args.data)}
    if hashes != {"model": EXPECTED_MODEL_SHA256, "data": EXPECTED_DATA_SHA256}:
        raise ValueError(f"Registered model/data identity mismatch: {hashes}")
    bundle, original_meta = T.load_model(args.model)
    model = bundle.model
    validate_model(model, original_meta)
    frozen_digest = weights_digest(model)
    meta = copy.deepcopy(original_meta)
    meta.pop("heads", None)
    protocol = {"format": FORMAT, "status": "fitting", "source_sha256": hashes,
                "frozen_model_weights_sha256": frozen_digest,
                "original_metadata": original_meta,
                "data_count": 500_000, "seed": args.seed, "latent_transform": "none",
                "label_alignment": "frames[t,e], actions[t,e] -> rewards[t,e], dones[t,e]",
                "sampling": "uniform with replacement from every one of the 500000 transitions",
                "checkpoint_selection": "fixed final update; no validation selection",
                "model_gradients": False, "value_head": False,
                "runtime_source_sha256": {str(path.relative_to(REPO)): file_sha256(path) for path in (
                    REPO / "train.py", REPO / "probe.py", Path(__file__).resolve(),
                    REPO / "studies/revision/inventory_jepa.py")},
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "jax", "flax", "optax")},
                "devices": [str(device) for device in jax.devices()],
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    manifest_path = args.out.with_suffix(".manifest.json")
    write_json(manifest_path, protocol)
    frames = materialize_frames(args.data, args.cache, hashes["data"])
    with np.load(args.data, allow_pickle=False) as archive:
        actions = np.asarray(archive["actions"], np.int32)
        rewards = np.asarray(archive["rewards"], np.float32)
        dones = np.asarray(archive["dones"], np.float32)
    protocol["archive_shapes"] = {"frames": list(frames.shape), "actions": list(actions.shape),
                                  "rewards": list(rewards.shape), "dones": list(dones.shape)}
    frames = current_frame_bank(frames, actions, rewards, dones)
    protocol["fitting_frame_selection"] = "frames[:-1], the 500000 current observations; trailing next frame excluded"
    latents, cache_manifest = embed_frames(model, frames, args.cache, hashes["model"], hashes["data"], args.encode_batch)
    if weights_digest(model) != frozen_digest:
        raise RuntimeError("Encoding changed frozen model parameters/statistics")
    protocol["embedding_cache"] = cache_manifest
    write_json(manifest_path, protocol)
    z = jnp.asarray(latents)
    pixel_bank = jnp.asarray(frames.reshape(-1, 63, 63, 3))
    phase_started = time.time()
    decoder, decoder_losses = P.train_decoder(pixel_bank, z, args.steps, args.batch, args.lr,
                                              args.hidden, args.depth, args.seed, model.n_tokens)
    decoder_s = time.time() - phase_started
    if not np.isfinite(decoder_losses).all():
        raise ValueError("Nonfinite decoder fitting losses")
    del pixel_bank
    phase_started = time.time()
    heads, head_losses = train_heads(z, jnp.asarray(actions), jnp.asarray(rewards), jnp.asarray(dones),
        steps=args.head_steps, batch=args.head_batch, lr=args.head_lr, hidden=args.head_hidden, seed=args.seed)
    head_s = time.time() - phase_started
    diagnostics = training_head_diagnostics(heads, z, jnp.asarray(actions), rewards, dones)
    if weights_digest(model) != frozen_digest or file_sha256(args.model) != hashes["model"]:
        raise RuntimeError("Frozen checkpoint/model changed during fitting")
    meta["decoder"] = {"hidden": args.hidden, "depth": args.depth, "tokens": model.n_tokens,
                       "steps": args.steps, "batch": args.batch, "lr": args.lr, "frames": 500_000,
                       "decoder_s": decoder_s, "final_mse": float(decoder_losses[-1])}
    meta["inventory_heads"] = {"hidden": args.head_hidden, "steps": args.head_steps,
                               "batch": args.head_batch, "lr": args.head_lr, "transitions": 500_000,
                               "head_s": head_s, "final_losses": head_losses[-1].tolist(),
                               "architecture": f"reward 529-{args.head_hidden}-{args.head_hidden}-1; done 529-{args.head_hidden}-1; GELU", **diagnostics}
    protocol.update(status="complete", elapsed_s=time.time() - started)
    meta["inventory_probe"] = protocol
    save_bundle(args.out, model, decoder, heads, meta)
    assert_frozen_archive(args.model, args.out)
    protocol["frozen_array_verification"] = True
    from studies.revision.inventory_jepa import load_bundle
    restored, _, _, _ = load_bundle(args.out)
    if weights_digest(restored) != frozen_digest:
        raise RuntimeError("Saved bundle failed frozen model round-trip")
    protocol["saved_bundle_roundtrip_verified"] = True
    losses_path = args.out.with_suffix(".losses.npz")
    np.savez(losses_path, decoder_mse=decoder_losses, heads_reward_mse_done_bce=head_losses)
    protocol["artifacts_sha256"] = {args.out.name: file_sha256(args.out), losses_path.name: file_sha256(losses_path)}
    write_json(manifest_path, protocol)
    print(json.dumps({"out": str(args.out), "elapsed_s": protocol["elapsed_s"],
                      "sha256": protocol["artifacts_sha256"], "training_diagnostics": diagnostics}), flush=True)


if __name__ == "__main__":
    main()
