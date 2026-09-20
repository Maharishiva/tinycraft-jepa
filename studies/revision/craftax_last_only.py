"""Fixed Craftax last-target-only follow-up; no experiments run on import.

Production changes only the prediction target selection. Config, initialization,
window construction, keys, optimizer and chunks follow the original train.py.
The update itself comes from the immutable last_only_objective.make_run.

 python -m studies.revision.craftax_last_only --reference-report REPORT.json \
   --data train.npz --seed 0 --out NEW_DIRECTORY

Every admitted seed (0, 1, 2) is retained, with no checkpoint selection. Earlier
baseline initialization/sampling ledgers were not archived: pairing with those
fits is expected by identical source/seed construction, not artifact-proven.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import train as T
from studies.revision import last_only_objective as LO


ROOT = Path(T.__file__).resolve().parent
DATA_SHA256 = "09f2cb41a38b0f34937a89c4122f6b25bbb3094e34a077b3a572504269d0a1b2"
BASE_TRAIN_SHA256 = "72cd5c463b4ac2fe7e289f7032e91d88ffac2e59ebb15534524fafe4804e9d0b"
OBJECTIVE_SHA256 = "b5316f8997cbbae56772520b0232d4f05c35be271a29b8f694d53ad650da64e1"
EXPECTED_PARAMS = 11121556
PROTOCOL = "craftax-last-only-v1"
PAIRING = ("expected by identical initialization, source and seed construction; "
           "baseline initialization and sampling ledgers were not archived, "
           "so historical pairing is not artifact-proven")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_sha256(value):
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": str(value.dtype), "shape": value.shape},
                             sort_keys=True).encode())
    digest.update(memoryview(np.ascontiguousarray(value).reshape(-1).view(np.uint8)))
    return digest.hexdigest()


def weights_sha256(model):
    """Parameters and BatchNorm statistics in original checkpoint order."""
    digest = hashlib.sha256()
    for kind, state in zip(("p", "b"), T._weights(model)):
        for i, leaf in enumerate(jax.tree.leaves(state)):
            digest.update(f"{kind}.{i}:".encode())
            digest.update(array_sha256(jax.device_get(leaf)).encode())
    return digest.hexdigest()


def write_manifest(out, manifest):
    temporary = out / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(out / "manifest.json")


def config_from_reference(reference, seed):
    """Read every Config field from original metadata; replace only its seed."""
    if seed not in (0, 1, 2):
        raise ValueError("only predeclared model seeds 0, 1, 2 are admitted")
    values = reference["model_metadata"]["cfg"]
    if set(values) != {field.name for field in fields(T.Config)}:
        raise ValueError("reference must specify every original Config field exactly")
    cfg = T.Config(**values)
    original = T.Config(seed=0, steps=8500, lr=1e-3, n_actions=17,
                        encoder="tile", emb_dim=128, enc_depth=6, enc_heads=8,
                        tokens=16, token_dim=32, token_hidden=512,
                        predictor="factored", pred_width=256, pred_depth=6,
                        pred_heads=8, history=8)
    if asdict(cfg) != asdict(original):
        raise ValueError("reference Config differs from the frozen Craftax recipe")
    return replace(cfg, seed=seed)


def verify_sources(reference):
    """Fail closed on changed original sources or the copied objective."""
    meta = reference["model_metadata"]
    if (meta["source_sha256"] != BASE_TRAIN_SHA256
            or reference["source_sha256"].get("train.py") != BASE_TRAIN_SHA256
            or sha256(T.__file__) != BASE_TRAIN_SHA256):
        raise ValueError("original train.py identity differs from the frozen reference")
    if sha256(LO.__file__) != OBJECTIVE_SHA256:
        raise ValueError("immutable last_only_objective.py source changed")
    for relative, expected in reference["source_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise ValueError(f"reference source changed: {relative}")
    return {**reference["source_sha256"],
            "studies/revision/last_only_objective.py": OBJECTIVE_SHA256,
            "studies/revision/craftax_last_only.py": sha256(__file__)}


def sample_ids(keys, batch, num_windows):
    """Audit the original sampling formula without consuming model RNGs."""
    def one(key):
        k_idx, _ = jax.random.split(key)
        return jax.random.randint(k_idx, (batch,), 0, num_windows)
    return np.asarray(jax.device_get(jax.jit(jax.vmap(one))(keys)))


def load_training_arrays(path, cfg, expected_action_shape=None):
    """Load only observations, native actions, and reset-boundary flags."""
    with np.load(path, allow_pickle=False) as archive:
        frames = archive["frames"]
        raw_actions = archive["actions"]
        raw_dones = archive["dones"]
    if raw_actions.ndim != 2 or raw_dones.shape != raw_actions.shape:
        raise ValueError("actions/dones must be aligned [time, environment] arrays")
    if expected_action_shape is not None and raw_actions.shape != expected_action_shape:
        raise ValueError(f"expected native Craftax action shape {expected_action_shape}")
    if raw_actions.size != cfg.data_budget:
        raise ValueError(f"expected exactly {cfg.data_budget} transitions")
    if not np.issubdtype(raw_actions.dtype, np.integer):
        raise ValueError("native action IDs must be integers")
    if np.any(raw_actions < 0) or np.any(raw_actions >= cfg.n_actions):
        raise ValueError("native action ID outside the configured vocabulary")
    if not np.isin(raw_dones, (0, 1)).all():
        raise ValueError("reset-boundary flags must be binary")
    expected_frames = (raw_actions.shape[0] + 1, raw_actions.shape[1], T.H, T.W, 3)
    if frames.shape != expected_frames or frames.dtype != np.uint8:
        raise ValueError(f"expected uint8 frames shaped {expected_frames}")
    return frames, raw_actions.astype(np.int32), raw_dones.astype(np.int64)


def _fit(data_npz, out, cfg, *, run, provenance, expected_params=None,
         expected_action_shape=None, prediction_loss_mode="last_only"):
    """Common driver; a tiny CPU fixture supplies the original full-prefix run.

    This internal seam does not add a production CLI mode or change any frozen
    module globals. Production always supplies LO.make_run(cfg, 'last_only').
    """
    out, data_npz = Path(out), Path(data_npz)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    frames_np, actions_np, dones_np = load_training_arrays(data_npz, cfg, expected_action_shape)
    span = cfg.history + cfg.num_preds
    table_np = T.valid_starts(dones_np, span)
    if not len(table_np):
        raise ValueError("no reset-free training windows")
    data_arrays = {name: array_sha256(value) for name, value in
                   (("frames", frames_np), ("actions_int32", actions_np), ("dones_int64", dones_np))}
    frames, actions, table = jnp.asarray(frames_np), jnp.asarray(actions_np), jnp.asarray(table_np)
    data_shape = {"frames": list(frames_np.shape), "actions": list(actions_np.shape),
                  "dones": list(dones_np.shape), "windows": len(table_np), "span": span}
    del frames_np, actions_np, dones_np
    model = T.LeWM(cfg, T.rngs(cfg.seed))
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    if expected_params is not None and n_params != expected_params:
        raise ValueError(f"expected {expected_params} parameters; got {n_params}")
    initial_weights = weights_sha256(model)
    warm = max(1, int(cfg.warmup * cfg.steps))
    schedule = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, warm, cfg.steps, end_value=0.0)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=cfg.weight_decay))
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
    keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 7), cfg.steps)
    ids = sample_ids(keys, cfg.batch, len(table_np))
    keys_np = np.asarray(jax.device_get(keys))
    np.savez(out / "sampling.npz", window_table=table_np, update_keys=keys_np, sampled_window_ids=ids)
    ledger = {"initial_weights_sha256": initial_weights,
              "window_table_sha256": array_sha256(table_np),
              "update_keys_sha256": array_sha256(keys_np),
              "sampled_window_ids_sha256": array_sha256(ids)}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = {}
    for package in ("jax", "jaxlib", "flax", "optax", "numpy", "craftax"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        **provenance, **ledger, "protocol": PROTOCOL, "status": "running",
        "cfg": asdict(cfg), "params": n_params, "model_seed": cfg.seed,
        "source_sha256": sha256(__file__), "base_train_source_sha256": sha256(T.__file__),
        "objective_source_sha256": sha256(LO.__file__),
        "data_path": str(data_npz.resolve()), "data_array_sha256": data_arrays,
        "data_shape": data_shape, "sampling_file_sha256": sha256(out / "sampling.npz"),
        "parameter_rng_seed": cfg.seed, "dropout_rng_seed": cfg.seed + 1,
        "update_rng_seed": cfg.seed + 7, "rng_implementation": str(jax.config.jax_default_prng_impl),
        "python": platform.python_version(), "packages": packages, "git_commit": commit,
        "devices": [str(device) for device in jax.devices()],
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "jax_enable_x64": jax.config.jax_enable_x64,
        "environment_flags": {name: os.environ.get(name) for name in
                              ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS",
                               "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_DEFAULT_MATMUL_PRECISION")},
        "prediction_loss_mode": prediction_loss_mode, "target_gradient_mode": "joint",
        "prediction_loss_normalization": "mean over batch and latent coordinates; no history multiplier",
        "sigreg_frame_count_per_window": span, "sigreg_weight": cfg.sigreg_weight,
        "encoder_frame_count_per_window": span, "predictor_output_count_per_window": cfg.history,
        "training_fields": ["frames", "actions", "dones_for_window_boundaries_only"],
        "checkpoint_selection": "fixed final update; every admitted model retained",
        "fixed_checkpoint_steps": [cfg.steps], "historical_pairing": PAIRING,
        "loss_columns": [f"{prediction_loss_mode}_prediction_mse", "sigreg"],
        "completed_steps": 0, "preloop_s": time.monotonic() - t0,
    }
    write_manifest(out, manifest)
    print(f"[model] {n_params} parameters; {prediction_loss_mode}; {len(table_np)} windows", flush=True)
    t1 = time.monotonic()
    history = []
    try:
        for begin in range(0, cfg.steps, cfg.chunk):
            losses = run(model, opt, frames, actions, table, keys[begin:begin + cfg.chunk])
            history.append(np.asarray(jax.device_get(losses)))
            done = begin + history[-1].shape[0]
            manifest["completed_steps"] = done
            if not np.isfinite(history[-1]).all():
                raise FloatingPointError(f"nonfinite loss by step {done}")
            write_manifest(out, manifest)
            print(f"[train] {done}/{cfg.steps} pred {history[-1][-1, 0]:.6f} "
                  f"sigreg {history[-1][-1, 1]:.6f} {time.monotonic() - t1:.1f}s", flush=True)
        loop_s = time.monotonic() - t1
        train_s = time.monotonic() - t0
        history = np.concatenate(history)
        if history.shape != (cfg.steps, 2):
            raise ValueError("training did not produce the prescribed update count")
        np.savez(out / "losses.npz", history=history)
        meta = {"cfg": asdict(cfg), "params": n_params, "train_s": round(train_s, 1),
                "loop_s": round(loop_s, 1), "data_sha256": provenance["data_sha256"],
                "source_sha256": manifest["source_sha256"],
                "base_train_source_sha256": manifest["base_train_source_sha256"],
                "objective_source_sha256": manifest["objective_source_sha256"],
                "sources": provenance.get("sources", {}), **ledger,
                "final_pred_loss": float(history[-1, 0]), "final_sigreg": float(history[-1, 1]),
                "prediction_loss_mode": prediction_loss_mode, "target_gradient_mode": "joint",
                "experiment": PROTOCOL, "completed_steps": cfg.steps,
                "reference_report_sha256": provenance.get("reference_report_sha256"),
                "reference_model_sha256": provenance.get("reference_model_sha256"),
                "historical_pairing": PAIRING}
        T.save(out / "model.npz", model, meta)
        manifest.update(status="complete", train_s=train_s, loop_s=loop_s,
                        final_checkpoint_sha256=sha256(out / "model.npz"),
                        final_weights_sha256=weights_sha256(model),
                        loss_file_sha256=sha256(out / "losses.npz"))
        write_manifest(out, manifest)
    except Exception as error:
        if isinstance(history, list) and history:
            np.savez(out / "losses.npz", history=np.concatenate(history))
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_manifest(out, manifest)
        raise
    print(f"[done] {train_s:.1f}s -> {out / 'model.npz'}", flush=True)
    return model, meta


def train(reference_report, data_npz, seed, out):
    """The only production path; all recipe/source/data checks precede fitting."""
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"refusing to overwrite nonempty output: {out}")
    reference_report, data_npz = Path(reference_report), Path(data_npz)
    reference = json.loads(reference_report.read_text())
    cfg = config_from_reference(reference, seed)
    sources = verify_sources(reference)
    if reference["model_metadata"]["params"] != EXPECTED_PARAMS:
        raise ValueError("reference parameter count differs from the frozen model")
    if reference["model_metadata"]["data_sha256"] != DATA_SHA256 or sha256(data_npz) != DATA_SHA256:
        raise ValueError("training data SHA-256 differs from the original Craftax archive")
    for package, version in reference["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"reference package version changed: {package}")
    provenance = {"data_sha256": DATA_SHA256, "sources": sources,
                  "reference_report": str(reference_report.resolve()),
                  "reference_report_sha256": sha256(reference_report),
                  "reference_model": reference["model"],
                  "reference_model_sha256": reference["model_sha256"],
                  "reference_model_sha256_origin": "declared in reference report; not independently verified by this wrapper",
                  "historical_package_pairing": "reference report pins numpy/jax/flax/craftax; historical jaxlib/optax versions were not archived",
                  "config_changes_allowed": ["seed"]}
    return _fit(data_npz, out, cfg, run=LO.make_run(cfg, "last_only"),
                provenance=provenance, expected_params=EXPECTED_PARAMS,
                expected_action_shape=(2000, 250))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--out", type=Path, required=True, help="new, empty output directory")
    args = parser.parse_args()
    train(args.reference_report, args.data, args.seed, args.out)


if __name__ == "__main__":
    main()
