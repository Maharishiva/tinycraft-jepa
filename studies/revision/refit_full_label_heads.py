"""Refit the original JEPA heads on all 500,000 training transition labels.

    python -m studies.revision.refit_full_label_heads \
        --world gray_frame_still --model run/model_probe.npz \
        --train data/train.npz --out run/model_probe_fullheads.npz \
        --eval data/eval.npz

The original grid-512 encoder, predictor, and off-budget 80,000-frame decoder
remain frozen. Raw current-frame latents, actions, rewards and done labels enter
the unmodified probe.train_heads recipe. Every training transition is eligible
for uniform sampling; no reward clipping, weighting, or filtering is applied.
The original Monte-Carlo value objective and tail exclusion are retained, even
though the value head is unused by PPO. This matches available label budgets,
not native head architectures or end-to-end supervision, across model families.

All hyperparameters and the seed are fixed; --eval is an optional, post-save
diagnostic with no checkpoint/seed selection. Existing output files are refused.
Run the small CPU semantic checks with JAX_PLATFORMS=cpu ... --self-test.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.nnx as nnx
import jax
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
import jax.numpy as jnp
import numpy as np

import probe as P
import train as T
from studies.revision import decoded_models as D

REPO = Path(__file__).resolve().parents[2]
WORLDS = ("gray_frame_still", "gray_frame_wind", "gray_episode_wind", "gate")
TRANSITIONS, LATENT_DIM, ENCODE_BATCH = 500_000, 512, 512
STEPS, BATCH, HIDDEN, MEMBERS, LR, SEED, GAMMA = 6000, 1024, 512, 1, 1e-3, 0, 0.99


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
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def module_arrays(module, prefix):
    return {f"{prefix}.{kind}.{i}": np.asarray(jax.device_get(leaf))
            for kind, state in zip(("p", "b"), T._weights(module))
            for i, leaf in enumerate(jax.tree.leaves(state))}


def frozen_hashes(arrays):
    """Hash serialized parameters AND batch statistics, with shape and dtype."""
    return {key: array_sha256(value) for key, value in sorted(arrays.items())
            if key.startswith(("model.", "decoder."))}


def runtime_frozen_hashes(bundle):
    return frozen_hashes({**module_arrays(bundle.model, "model"),
                          **module_arrays(bundle.decoder, "decoder")})


def archive_identity(meta, world, split):
    spec = dict(meta.get("spec", {}))
    spec.setdefault("gate", False)
    expected = {"noise": "off" if world == "gate" else "gray",
                "per_episode": world == "gray_episode_wind",
                "wind": world in ("gray_frame_wind", "gray_episode_wind"),
                "gate": world == "gate"}
    config_names = ("gate", "gate_off_still") if world == "gate" else (world,)
    if (meta.get("env") not in ("redcube", "stoch")
            or (world == "gate" and meta.get("env") != "stoch")
            or meta.get("config") not in config_names or spec != expected
            or meta.get("split") != split or meta.get("num_actions") != 4
            or meta.get("policy") != "uniform random"
            or not isinstance(meta.get("seed"), int)):
        raise ValueError(f"Expected original {world} {split} archive, got {meta}")
    return {key: meta[key] for key in
            ("env", "config", "spec", "split", "seed", "num_actions", "policy")}


def load_data(path, world, split, expected_count=None):
    """Only pixels, recorded actions, reward/done labels, and metadata are read."""
    with np.load(path, allow_pickle=False) as archive:
        identity = archive_identity(json.loads(str(archive["meta"])), world, split)
        frames = archive["frames"]
        actions, rewards, dones = (archive[key] for key in ("actions", "rewards", "dones"))
    if (actions.ndim != 2 or min(actions.shape) <= 0
            or rewards.shape != actions.shape or dones.shape != actions.shape
            or frames.shape != (actions.shape[0] + 1, actions.shape[1], T.H, T.W, 3)
            or frames.dtype != np.uint8):
        raise ValueError("Expected aligned frames[T+1,E,63,63,3] and labels[T,E]")
    if expected_count is not None and actions.size != expected_count:
        raise ValueError(f"Expected exactly {expected_count} transition labels; got {actions.size}")
    if (not np.issubdtype(actions.dtype, np.integer)
            or np.any((actions < 0) | (actions >= 4))
            or not np.isfinite(rewards).all() or not np.isin(dones, (0, 1)).all()):
        raise ValueError("Invalid action, reward, or done labels")
    hashes = {key: array_sha256(value) for key, value in
              (("actions", actions), ("rewards", rewards), ("dones", dones))}
    # Preserve the original probe's conversion before return construction/fit.
    return (frames, actions.astype(np.int32), rewards.astype(np.float32),
            dones.astype(np.float32), {"identity": identity, "label_sha256": hashes,
                                      "frames_shape": list(frames.shape),
                                      "labels_shape": list(actions.shape)})


def make_cached_encoder(adapter):
    """The actual cache kernel, exposed for independent CPU/GPU parity checks.

    Close over the sealed adapter's graph/state exactly as the shared PPO path
    does. Its encode method owns fixed-512 zero padding and unpadding. Passing a
    live NNX module as a JIT argument can produce different BF16 kernels even at
    the same nominal batch size, so this must call the adapter itself.
    """
    if not isinstance(adapter, D.JepaAdapter) or D.CODEC_BATCH != ENCODE_BATCH:
        raise ValueError("Expected the registered fixed-512 JEPA adapter")

    @jax.jit
    def encode(batch):
        return adapter.encode(batch[:, None])[:, 0]

    return encode


def encode_current_frames(adapter, frames):
    """Cache raw latents using the shared adapter, with CPU pixel/latent staging."""
    encode = make_cached_encoder(adapter)
    flat = frames[:-1].reshape(-1, T.H, T.W, 3)
    latents = np.empty((len(flat), LATENT_DIM), dtype=np.float32)
    for begin in range(0, len(flat), ENCODE_BATCH):
        end = min(begin + ENCODE_BATCH, len(flat))
        latents[begin:end] = np.asarray(jax.device_get(encode(jnp.asarray(flat[begin:end]))))
        if begin == 0 or end == len(flat) or (begin // ENCODE_BATCH) % 25 == 0:
            print(f"[encode] {end}/{len(flat)} current frames", flush=True)
    if not np.isfinite(latents).all():
        raise ValueError("Nonfinite frozen encoder outputs")
    return latents.reshape(frames.shape[0] - 1, frames.shape[1], LATENT_DIM)


def value_tail(rows):
    return max(rows // 4, rows - int(np.ceil(4.0 / (1.0 - GAMMA))))


def fit_heads(latents, actions, rewards, dones, *, steps=STEPS, batch=BATCH, hidden=HIDDEN):
    """Original recipe; smaller budgets are exposed only for the CPU self-test."""
    targets = P.returns(rewards, dones, GAMMA)
    tail = value_tail(latents.shape[0])
    if tail <= 0:
        raise ValueError("Recorded window is too short for the original value-tail rule")
    heads, history = P.train_heads(
        jnp.asarray(latents), jnp.asarray(actions), jnp.asarray(rewards), jnp.asarray(dones),
        jnp.asarray(targets), steps, batch, LR, hidden, SEED, tail, MEMBERS, 4)
    return heads, history, tail


def save_refit(source_arrays, heads, meta, out):
    """Copy frozen array payloads directly, never round-trip them through JAX."""
    arrays = {key: value for key, value in source_arrays.items()
              if key != "meta" and not key.startswith("heads.")}
    arrays.update(module_arrays(heads, "heads"))
    with Path(out).open("xb") as handle:
        np.savez(handle, meta=json.dumps(meta, sort_keys=True, allow_nan=False), **arrays)
    with np.load(out, allow_pickle=False) as saved:
        if frozen_hashes(saved) != frozen_hashes(source_arrays):
            raise AssertionError("Saved model/decoder parameters or statistics changed")
        # Also retain any unexpected non-head payload without modification.
        for key in source_arrays:
            if key != "meta" and not key.startswith("heads."):
                if array_sha256(saved[key]) != array_sha256(source_arrays[key]):
                    raise AssertionError(f"Original non-head array changed: {key}")


def evaluate_saved_heads(bundle, adapter, path, world, train_identity):
    """Post-save diagnostics only; no evaluation data enters fitting or selection."""
    frames, actions, rewards, dones, provenance = load_data(path, world, "eval")
    identity = provenance["identity"]
    if identity["seed"] == train_identity["seed"]:
        raise ValueError("Train and eval collection seeds must differ")
    for key in ("env", "config", "spec", "num_actions", "policy"):
        if identity[key] != train_identity[key]:
            raise ValueError(f"Train and eval identity mismatch: {key}")
    z = encode_current_frames(adapter, frames).reshape(-1, LATENT_DIM)
    del frames
    a, r, d = actions.reshape(-1), rewards.reshape(-1), dones.reshape(-1)
    pred_r, logits_d, pred_v = [], [], []
    for begin in range(0, len(z), 8192):
        pr, pd, pv = bundle.heads(jnp.asarray(z[begin:begin + 8192]),
                                jnp.asarray(a[begin:begin + 8192]))
        pred_r.append(np.asarray(jax.device_get(pr[:, 0])))
        logits_d.append(np.asarray(jax.device_get(pd)))
        pred_v.append(np.asarray(jax.device_get(pv[:, 0])))
    pr, ld, pv = (np.concatenate(x) for x in (pred_r, logits_d, pred_v))
    target = P.returns(rewards, dones, GAMMA).reshape(-1)
    valid = value_tail(actions.shape[0]) * actions.shape[1]
    corr = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
    return {"path": str(path.resolve()), "sha256": sha256(path), **provenance,
            "transitions": int(len(z)), "reward_rmse": float(np.sqrt(np.mean((pr - r) ** 2))),
            "reward_correlation": corr(pr, r),
            "done_bce": float(np.mean(np.logaddexp(0.0, ld) - d * ld)),
            "value_eligible_transitions": valid,
            "value_rmse": float(np.sqrt(np.mean((pv[:valid] - target[:valid]) ** 2))) if valid else None,
            "used_for_fit_or_selection": False}


def run(args):
    started = time.monotonic()
    model_path, data_path, out = args.model.resolve(), args.train.resolve(), args.out.resolve()
    config_path = out.with_suffix(".config.json")
    history_path = out.with_suffix(".history.npz")
    result_path = out.with_suffix(".manifest.json")
    for path in (out, config_path, history_path, result_path):
        if path.exists() or path == model_path or path == data_path:
            raise FileExistsError(f"Refusing to overwrite {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    source_hash, data_hash = sha256(model_path), sha256(data_path)
    with np.load(model_path, allow_pickle=False) as archive:
        source_arrays = {key: archive[key] for key in archive.files}
    bundle, old_meta = T.load_model(model_path)
    if (bundle.decoder is None or bundle.heads is None or not bundle.model.grid
            or bundle.model.latent_dim != LATENT_DIM or bundle.model.n_actions != 4
            or old_meta.get("decoder", {}).get("frames") != 80_000
            or old_meta.get("heads", {}).get("hidden") != HIDDEN
            or old_meta.get("heads", {}).get("members") != MEMBERS):
        raise ValueError("Expected original grid512/4-action probe with 80k-frame decoder and 512/1 heads")
    for key, expected in (("steps", STEPS), ("batch", BATCH), ("lr", LR), ("gamma", GAMMA)):
        if old_meta["heads"].get(key) != expected:
            raise ValueError(f"Original head {key} does not match the fixed refit recipe")
    if old_meta.get("data_sha256") and old_meta["data_sha256"] != data_hash:
        raise ValueError("Training data hash differs from the original world-model training archive")
    frozen_before = frozen_hashes(source_arrays)
    if runtime_frozen_hashes(bundle) != frozen_before:
        raise AssertionError("Loaded model/decoder differs from original frozen payload")
    adapter = D.JepaAdapter(bundle, old_meta, {"model": model_path})
    adapter_before = adapter.assert_frozen()
    frames, actions, rewards, dones, provenance = load_data(data_path, args.world, "train", TRANSITIONS)
    recipe = {"hidden": HIDDEN, "members": MEMBERS, "steps": STEPS, "batch": BATCH,
              "lr": LR, "seed": SEED, "gamma": GAMMA, "transitions": TRANSITIONS,
              "reward_done_draws": STEPS * BATCH, "encode_batch": ENCODE_BATCH,
              "encode_kernel": "make_cached_encoder: jax.jit closure over decoded_models.JepaAdapter.encode",
              "encode_input_shape": [ENCODE_BATCH, 1, T.H, T.W, 3],
              "encode_tail_padding": "adapter zero pads final chunk to 512 and discards padded outputs",
              "latent_dim": LATENT_DIM, "sampling": "uniform with replacement",
              "loss": "MSE(reward) + BCE(done) + 0.1*MSE(MC return)",
              "optimizer": "original probe.train_heads warmup cosine AdamW, decay=1e-4, global clip=1",
              "value_targets": "Full training-window Monte-Carlo returns; original tail-start exclusion",
              "value_used_by_ppo": False, "reward_clipping": False,
              "reweighting": False, "latent_scaling": False, "decoder_refit": False}
    config = {"experiment": "frozen_jepa_full_training_label_heads", "world": args.world,
              "model": str(model_path), "model_sha256": source_hash,
              "train": str(data_path), "train_sha256": data_hash, "training_data": provenance,
              "out": str(out), "recipe": recipe, "frozen_before_sha256": frozen_before,
              "adapter_frozen_before_sha256": adapter_before,
              "source_sha256": {str(path.relative_to(REPO)): sha256(path) for path in
                                (Path(__file__), REPO / "probe.py", REPO / "train.py",
                                 Path(D.__file__).resolve())},
              "packages": {name: importlib.metadata.version(name) for name in
                           ("jax", "jaxlib", "flax", "optax", "numpy")},
              "devices": [str(device) for device in jax.devices()],
              "xla_flags": os.environ.get("XLA_FLAGS"),
              "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
              "comparison_scope": "Equal available training labels; native head architectures and supervision differ",
              "eval_after_save": str(args.eval.resolve()) if args.eval else None,
              "checkpoint_selection": "fixed final step, no evaluation-based selection"}
    write_json(config_path, config)
    t_encode = time.monotonic()
    latents = encode_current_frames(adapter, frames)
    encode_s = time.monotonic() - t_encode
    del frames
    latent_hash = array_sha256(latents)
    t_fit = time.monotonic()
    heads, history, tail = fit_heads(latents, actions, rewards, dones)
    fit_s = time.monotonic() - t_fit
    if not np.isfinite(history).all():
        raise ValueError("Nonfinite head training loss")
    frozen_after = runtime_frozen_hashes(bundle)
    adapter_after = adapter.assert_frozen()
    if frozen_after != frozen_before or sha256(model_path) != source_hash:
        raise AssertionError("Original frozen model/decoder or source file changed")
    meta = copy.deepcopy(old_meta)
    meta["heads"] = {**recipe, "value_tail": tail, "value_eligible_transitions": tail * actions.shape[1],
                     "heads_s": fit_s, "encode_s": encode_s, "final": history[-1].tolist(),
                     "training_data_sha256": data_hash}
    meta["heads_refit"] = {**config, "original_heads": old_meta["heads"],
                           "frozen_after_sha256": frozen_after, "latents_sha256": latent_hash,
                           "adapter_frozen_after_sha256": adapter_after,
                           "encode_s": encode_s, "fit_s": fit_s,
                           "seconds_before_save": time.monotonic() - started}
    save_refit(source_arrays, heads, meta, out)
    saved_bundle, saved_meta = T.load_model(out)
    if runtime_frozen_hashes(saved_bundle) != frozen_before:
        raise AssertionError("Reloaded output changed frozen parameters/statistics")
    if {key: array_sha256(value) for key, value in module_arrays(saved_bundle.heads, "heads").items()} != {
            key: array_sha256(value) for key, value in module_arrays(heads, "heads").items()}:
        raise AssertionError("Saved metadata failed to restore fitted heads exactly")
    with history_path.open("xb") as handle:
        np.savez(handle, history=history, columns=np.asarray(["reward_mse", "done_bce", "value_mse"]))
    result = {"config": config, "output_sha256": sha256(out), "frozen_after_sha256": frozen_after,
              "adapter_frozen_after_sha256": adapter_after,
              "history_sha256": sha256(history_path), "heads": meta["heads"],
              "latents_sha256": latent_hash, "seconds_through_save": time.monotonic() - started}
    del latents
    if args.eval:
        # Reload the saved artifact, rather than evaluating an unsaved in-memory model.
        saved_adapter = D.JepaAdapter(saved_bundle, saved_meta, {"model": out})
        result["eval"] = evaluate_saved_heads(saved_bundle, saved_adapter, args.eval,
                                              args.world, provenance["identity"])
        saved_adapter.assert_frozen()
        if runtime_frozen_hashes(saved_bundle) != frozen_before:
            raise AssertionError("Evaluation changed frozen parameters/statistics")
    result["seconds_total"] = time.monotonic() - started
    write_json(result_path, result)
    print(json.dumps({"saved": str(out), "manifest": str(result_path), "encode_s": encode_s,
                      "fit_s": fit_s, "seconds_total": result["seconds_total"]}), flush=True)
    return result


def self_test():
    """Small CPU check: causal labels, exact original optimizer recipe, frozen save."""
    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("Set JAX_PLATFORMS=cpu for --self-test")
    rng = np.random.default_rng(9)
    z = rng.normal(size=(16, 2, 8)).astype(np.float32)
    a = rng.integers(0, 4, (16, 2), dtype=np.int32)
    r = rng.normal(size=(16, 2)).astype(np.float32)
    d = np.zeros((16, 2), np.float32)
    d[[3, 9], :] = 1
    expected = np.zeros_like(r)
    for t in range(len(r)):
        for env in range(r.shape[1]):
            discount = 1.0
            for future in range(t, len(r)):
                expected[t, env] += discount * r[future, env]
                if d[future, env]:
                    break
                discount *= GAMMA
    np.testing.assert_allclose(P.returns(r, d, GAMMA), expected, atol=2e-6)
    heads, history, tail = fit_heads(z, a, r, d, steps=4, batch=16, hidden=8)
    reference, reference_history = P.train_heads(
        jnp.asarray(z), jnp.asarray(a), jnp.asarray(r), jnp.asarray(d),
        jnp.asarray(P.returns(r, d, GAMMA)), 4, 16, LR, 8, SEED, 4, 1, 4)
    assert tail == 4
    np.testing.assert_array_equal(history, reference_history)
    assert {k: array_sha256(v) for k, v in module_arrays(heads, "heads").items()} == {
        k: array_sha256(v) for k, v in module_arrays(reference, "heads").items()}
    cfg = T.Config(encoder="tile", emb_dim=8, enc_depth=0, enc_res=0, tokens=16,
                   token_dim=32, token_hidden=8, predictor="factored", pred_depth=0,
                   pred_width=8, pred_heads=1, n_actions=4, history=8)
    model = T.LeWM(cfg, T.rngs(0))
    decoder = P.Decoder(model.latent_dim, 8, 0, T.rngs(0), tokens=16)
    bundle = T.Bundle(model, decoder, P.Heads(LATENT_DIM, 8, T.rngs(0), 1, 4))
    adapter = D.JepaAdapter(bundle, {"cfg": asdict(cfg)}, {})
    original = runtime_frozen_hashes(bundle)
    frames = rng.integers(0, 256, (4, 2, T.H, T.W, 3), dtype=np.uint8)
    assert ENCODE_BATCH == 512
    embedded = encode_current_frames(adapter, frames)
    flat_frames = frames[:-1].reshape(-1, T.H, T.W, 3)
    canonical = np.asarray(jax.jit(adapter.encode)(jnp.asarray(flat_frames[:, None])))[:, 0].reshape(embedded.shape)
    np.testing.assert_array_equal(embedded, canonical)
    assert runtime_frozen_hashes(bundle) == original
    adapter.assert_frozen()
    new_heads, _, _ = fit_heads(embedded.repeat(2, axis=0), a[:6], r[:6], d[:6],
                                steps=2, batch=8, hidden=8)
    assert runtime_frozen_hashes(bundle) == original
    adapter.assert_frozen()
    with tempfile.TemporaryDirectory(prefix="full-label-heads-test-") as temp:
        source = Path(temp) / "source.npz"
        out = Path(temp) / "model_probe_fullheads.npz"
        meta = {"cfg": asdict(cfg), "decoder": {"hidden": 8, "depth": 0, "tokens": 16},
                "heads": {"hidden": 8, "members": 1}}
        T.save(source, model, meta, decoder=decoder, heads=new_heads)
        before_file = sha256(source)
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        save_refit(arrays, new_heads, meta, out)
        restored, _ = T.load_model(out)
        assert runtime_frozen_hashes(restored) == original
        assert sha256(source) == before_file
        try:
            save_refit(arrays, new_heads, meta, out)
        except FileExistsError:
            pass
        else:
            raise AssertionError("An existing output was overwritten")
    print(json.dumps({"self_test": "passed", "checks": ["MC episode boundaries",
          "original head recipe exact parameter/history parity", "sealed adapter cache encoder exact parity",
          "frozen runtime and saved parameter/statistics hashes", "source unchanged", "no overwrite"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--train", "--data", dest="train", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--world", choices=WORLDS)
    parser.add_argument("--eval", type=Path, help="Optional separate archive, evaluated only after saving")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if any(getattr(args, name) is None for name in ("model", "train", "out", "world")):
            parser.error("--model, --train, --out and --world are required")
        run(args)


if __name__ == "__main__":
    main()
