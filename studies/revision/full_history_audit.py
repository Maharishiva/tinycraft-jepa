"""Fixed full-history, one-step prediction audit of all 18 frozen final models.

This is a representation-relative diagnostic, not an information or quality
score. Lower latent MSE can reflect collapse. It never trains or chooses models.
See research/full_history_audit_protocol.md for the prospectively fixed design.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

if "--self-test" in sys.argv or "--pin-windows" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train as T
from studies.revision import target_gradient as G


ROOT = Path(T.__file__).resolve().parent
WORLDS = ("gray_frame_still", "gray_frame_wind")
MODES = ("joint", "stop_target", "last_only")
SEEDS = (0, 1, 2)
N_WINDOWS, SAMPLE_SEED, BATCH, HISTORY, SPAN, LATENT = 8192, 20260919, 128, 8, 9, 512
PROTOCOL = "frozen_full_history_audit_v1"


def write_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def state_sha256(model):
    """Hash every numerical NNX leaf, including RNGs, before/after inference."""
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(nnx.state(model))[0]:
        digest.update(str(path).encode())
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            digest.update(str(value.dtype).encode())
            value = jax.random.key_data(value)
        digest.update(G.array_sha256(jax.device_get(value)).encode())
    return digest.hexdigest()


def select_windows(dones, count=N_WINDOWS, seed=SAMPLE_SEED):
    """Unique table entries; overlapping time windows remain possible."""
    dones = np.asarray(dones)
    if dones.ndim != 2 or not np.all((dones == 0) | (dones == 1)):
        raise ValueError("dones must be a binary [time, environment] array")
    table = T.valid_starts(dones.astype(np.int64), SPAN)
    if len(table) < count:
        raise ValueError(f"need {count} distinct valid windows; only {len(table)} available")
    ids = np.random.Generator(np.random.PCG64(seed)).choice(len(table), count, replace=False)
    ids = np.asarray(ids, dtype=np.int64)
    return table, ids, table[ids]


def gather_windows(frames, actions, windows):
    indices = windows[:, :1] + np.arange(SPAN, dtype=np.int32)[None]
    env = windows[:, 1:2]
    return frames[indices, env], actions[indices[:, :-1], env]


def forward(model, frames, actions):
    """Same training forward indices, with frozen BN and deterministic dropout.

    The production caller always supplies [128,9,63,63,3] images. Encode the
    complete window in one call, preserving the 1152-frame flattened batch.
    """
    emb = model.encode(frames, False)
    act_emb = model.act_embed(actions)
    pred = model.predict(emb[:, :HISTORY], act_emb[:, :HISTORY], False)
    return pred, emb[:, 1:]


infer = nnx.jit(forward)


def prefix_errors(pred, target):
    """Coordinate means evaluated in float64 from unmodified float32 latents."""
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1] != HISTORY:
        raise ValueError("predictions/targets must be matching [batch,8,latent] arrays")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite frozen prediction or target")
    return np.mean(np.square(pred.astype(np.float64) - target.astype(np.float64)), axis=-1)


def target_geometry(target, endpoint_mse):
    """Population covariance diagnostics; no normalization changes inference."""
    x = np.asarray(target, dtype=np.float64)
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / len(x)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
    variance = np.diag(covariance).copy()
    total = float(eigenvalues.sum())
    positive = eigenvalues[eigenvalues > 0]
    probabilities = positive / total if total > 0 else positive
    erank = float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if total > 0 else None
    average_variance = float(variance.mean())
    summary = {
        "target_coordinate_variance_mean_ddof0": average_variance,
        "target_total_variance_ddof0": float(variance.sum()),
        "target_covariance_effective_rank": erank,
        "effective_rank_definition": "exp(entropy(normalized population covariance eigenvalues))",
        "target_mean_square_per_coordinate": float(np.mean(x * x)),
        "target_mean_vector_square_per_coordinate": float(np.mean(mean * mean)),
        "endpoint_mse_over_target_variance": float(endpoint_mse / average_variance)
        if average_variance > 0 else None,
        "normalization_definition": "endpoint coordinate-mean MSE / mean target coordinate population variance",
        "normalization_scope": "descriptive only; raw model inputs, outputs and saved latents are unchanged",
    }
    return summary, {"endpoint_target_mean": mean, "endpoint_target_coordinate_variance": variance,
                     "endpoint_target_covariance_eigenvalues": eigenvalues}


def check_source_map(sources):
    if "train.py" not in sources or "studies/revision/target_gradient.py" not in sources:
        raise ValueError("missing trainer/model source identities")
    for relative, expected in sources.items():
        path = ROOT / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or G.sha256(path) != expected:
            raise ValueError(f"checkpoint source does not match audit checkout: {relative}")


def load_case(root, world, seed, mode):
    directory = root / world / f"m{seed}" / mode
    manifest_path, checkpoint = directory / "manifest.json", directory / "model.npz"
    manifest = json.loads(manifest_path.read_text())
    with np.load(checkpoint, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        if any(name.startswith(("heads.", "decoder.")) for name in archive.files):
            raise ValueError("expected a raw final model, not a probe-augmented bundle")
    expected_cfg = asdict(G.production_config(seed))
    experiment = "last_only_objective_v1" if mode == "last_only" else "paired_target_gradient_v1"
    if (manifest.get("status") != "complete" or manifest.get("completed_steps") != 8500
            or manifest.get("protocol") != experiment or manifest.get("mode") != mode
            or manifest.get("config_label") != world or manifest.get("model_seed") != seed
            or manifest.get("cfg") != expected_cfg or meta.get("cfg") != expected_cfg
            or meta.get("completed_steps") != 8500 or meta.get("config_label") != world
            or meta.get("experiment") != experiment):
        raise ValueError(f"not a registered complete final checkpoint: {directory}")
    if mode == "last_only":
        if meta.get("prediction_loss_mode") != mode or meta.get("target_gradient_mode") != "joint":
            raise ValueError("last-only checkpoint objective identity mismatch")
    elif meta.get("target_gradient_mode") != mode:
        raise ValueError("target-gradient checkpoint objective identity mismatch")
    checkpoint_sha = G.sha256(checkpoint)
    final = manifest["checkpoints"][-1]
    if (final.get("step") != 8500 or final.get("path") != "model.npz"
            or final.get("sha256") != checkpoint_sha
            or manifest.get("final_checkpoint_sha256") != checkpoint_sha
            or final.get("weights_sha256") != manifest.get("final_weights_sha256")):
        raise ValueError("final checkpoint/manifest identity mismatch")
    for field in ("data_sha256", "initial_weights_sha256", "sampled_window_ids_sha256", "sources"):
        if meta.get(field) != manifest.get(field):
            raise ValueError(f"checkpoint/manifest mismatch in {field}")
    sources = manifest["sources"]["sha256"]
    check_source_map(sources)
    trainer = "last_only_objective.py" if mode == "last_only" else "target_gradient.py"
    if meta.get("source_sha256") != sources.get("studies/revision/" + trainer):
        raise ValueError("checkpoint trainer source identity mismatch")
    return {"world": world, "model_seed": seed, "mode": mode, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha, "manifest": str(manifest_path.resolve()),
            "manifest_sha256": G.sha256(manifest_path), "expected_weights_sha256": final["weights_sha256"],
            "training_data_sha256": manifest["data_sha256"], "training_sources": manifest["sources"],
            "initial_weights_sha256": manifest["initial_weights_sha256"],
            "sampled_window_ids_sha256": manifest["sampled_window_ids_sha256"],
            "update_keys_sha256": manifest["update_keys_sha256"],
            "training_window_table_sha256": manifest["window_table_sha256"]}


def registered_cases(target_root, last_only_root):
    cases = [load_case(last_only_root if mode == "last_only" else target_root, world, seed, mode)
             for world in WORLDS for seed in SEEDS for mode in MODES]
    # Require the prospective pairing rather than only matching architecture.
    for world in WORLDS:
        for seed in SEEDS:
            group = [case for case in cases if case["world"] == world and case["model_seed"] == seed]
            for field in ("training_data_sha256", "initial_weights_sha256", "sampled_window_ids_sha256",
                          "update_keys_sha256", "training_window_table_sha256"):
                if len({case[field] for case in group}) != 1:
                    raise ValueError(f"unpaired objective arms: {world}, seed {seed}, {field}")
    return cases


def eval_identity(meta, world):
    spec = dict(meta.get("spec", {}))
    spec.setdefault("gate", False)
    if (meta.get("env") not in ("redcube", "stoch") or meta.get("config") != world
            or meta.get("split") != "eval" or meta.get("num_actions") != 4
            or spec != {"noise": "gray", "per_episode": False, "wind": world == WORLDS[1], "gate": False}):
        raise ValueError(f"archive metadata does not identify original {world} evaluation data")
    return {name: meta[name] for name in ("env", "config", "split", "num_actions", "spec", "seed")
            if name in meta}


def load_data(path, world, expected_sha):
    before = G.sha256(path)
    if before != expected_sha:
        raise ValueError("evaluation archive does not match frozen target-gradient campaign")
    with np.load(path, allow_pickle=False) as archive:
        identity = eval_identity(json.loads(str(archive["meta"])), world)
        # The only loaded arrays are observations, recorded actions and boundaries.
        frames, actions, dones = archive["frames"], archive["actions"], archive["dones"]
    if actions.ndim != 2 or dones.shape != actions.shape or frames.dtype != np.uint8:
        raise ValueError("misaligned evaluation arrays")
    if frames.shape != (actions.shape[0] + 1, actions.shape[1], T.H, T.W, 3):
        raise ValueError("evaluation frame shape does not match recorded transitions")
    if not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions >= 4)):
        raise ValueError("invalid recorded action vocabulary")
    table, ids, windows = select_windows(dones)
    arrays = {"window_table_ids": ids, "windows_start_env": windows,
              "frame_time_indices": windows[:, :1] + np.arange(SPAN, dtype=np.int32)[None]}
    provenance = {"path": str(path.resolve()), "sha256": before, "identity": identity,
                  "frames_shape": list(frames.shape), "actions_shape": list(actions.shape),
                  "valid_window_count": len(table), "window_table_sha256": G.array_sha256(table),
                  "array_sha256": {name: G.array_sha256(value) for name, value in
                                   (("frames", frames), ("actions", actions), ("dones", dones))},
                  "sample_array_sha256": {key: G.array_sha256(value) for key, value in arrays.items()}}
    return frames, actions.astype(np.int32), arrays, provenance


def pin_windows(data_root, out):
    """Write exact common IDs before fitting; reads only metadata and dones."""
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite a nonempty window-pin directory")
    out.mkdir(parents=True, exist_ok=True)
    pin = {"protocol": PROTOCOL + "_window_pin", "status": "running",
           "created_utc": datetime.now(timezone.utc).isoformat(),
           "windows_per_world": N_WINDOWS, "sample_seed": SAMPLE_SEED,
           "sampler": "numpy.Generator(PCG64(seed)).choice(valid_windows,8192,replace=False)",
           "sources_sha256": {str(path.relative_to(ROOT)): G.sha256(path) for path in
                              (Path(__file__).resolve(), ROOT / "train.py", ROOT / "research/full_history_audit_protocol.md")},
           "numpy_version": np.__version__, "data": {}}
    write_json(out / "window_pin.json", pin)
    for world in WORLDS:
        path = data_root / world / "eval.npz"
        archive_sha = G.sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            identity = eval_identity(json.loads(str(archive["meta"])), world)
            dones = archive["dones"]
        table, ids, windows = select_windows(dones)
        arrays = {"window_table_ids": ids, "windows_start_env": windows,
                  "frame_time_indices": windows[:, :1] + np.arange(SPAN, dtype=np.int32)[None]}
        artifact = out / f"{world}_windows.npz"
        np.savez(artifact, **arrays)
        if G.sha256(path) != archive_sha:
            raise ValueError("evaluation archive changed during window pinning")
        pin["data"][world] = {"path": str(path.resolve()), "sha256": archive_sha, "identity": identity,
                              "valid_window_count": len(table), "window_table_sha256": G.array_sha256(table),
                              "artifact": artifact.name, "artifact_sha256": G.sha256(artifact),
                              "sample_array_sha256": {name: G.array_sha256(value) for name, value in arrays.items()}}
        write_json(out / "window_pin.json", pin)
    pin["status"] = "complete"
    write_json(out / "window_pin.json", pin)
    print(json.dumps({"window_pin": str((out / "window_pin.json").resolve()),
                      "sha256": G.sha256(out / "window_pin.json"), "sample_seed": SAMPLE_SEED,
                      "windows_per_world": N_WINDOWS}), flush=True)


def evaluate_case(case, frames, actions, sample, out):
    start = time.time()
    bundle, _ = T.load_model(case["checkpoint"])
    model = bundle.model
    if model.latent_dim != LATENT or model.history != HISTORY or model.n_actions != 4:
        raise ValueError("restored model violates latent/history/action contract")
    before_state, before_weights = state_sha256(model), G.weights_sha256(model)
    if before_weights != case["expected_weights_sha256"]:
        raise ValueError("restored model weights differ from final checkpoint manifest")
    errors = np.empty((N_WINDOWS, HISTORY), dtype=np.float64)
    predictions = np.empty((N_WINDOWS, LATENT), dtype=np.float32)
    targets = np.empty_like(predictions)
    frame_digest, action_digest = hashlib.sha256(), hashlib.sha256()
    for begin in range(0, N_WINDOWS, BATCH):
        end = begin + BATCH
        window, outgoing = gather_windows(frames, actions, sample["windows_start_env"][begin:end])
        if window.shape != (BATCH, SPAN, T.H, T.W, 3) or outgoing.shape != (BATCH, HISTORY):
            raise ValueError("noncanonical inference batch; padding/rebatching is forbidden")
        frame_digest.update(G.array_sha256(window).encode())
        action_digest.update(G.array_sha256(outgoing).encode())
        pred, target = jax.device_get(infer(model, jnp.asarray(window), jnp.asarray(outgoing)))
        if pred.shape != (BATCH, HISTORY, LATENT) or target.shape != pred.shape:
            raise ValueError("unexpected prediction/target shape")
        errors[begin:end] = prefix_errors(pred, target)
        predictions[begin:end], targets[begin:end] = pred[:, -1], target[:, -1]
    after_state, after_weights = state_sha256(model), G.weights_sha256(model)
    if before_state != after_state or before_weights != after_weights:
        raise ValueError("inference mutated frozen model state")
    endpoint = float(errors[:, -1].mean())
    geometry, geometry_arrays = target_geometry(targets, endpoint)
    saved = {**sample, "prefix_coordinate_mse": errors, "endpoint_predictions": predictions,
             "endpoint_targets": targets, **geometry_arrays}
    artifact = out / f"{case['world']}_m{case['model_seed']}_{case['mode']}.npz"
    np.savez(artifact, **saved)
    result = {**case, "status": "complete", "artifact": artifact.name,
              "artifact_sha256": G.sha256(artifact),
              "array_sha256": {name: G.array_sha256(value) for name, value in saved.items()},
              "batch_frame_hash_chain_sha256": frame_digest.hexdigest(),
              "batch_outgoing_action_hash_chain_sha256": action_digest.hexdigest(),
              "frozen_state_sha256_before": before_state, "frozen_state_sha256_after": after_state,
              "weights_sha256_before": before_weights, "weights_sha256_after": after_weights,
              "full_history_endpoint_coordinate_mse": endpoint,
              "prefix_coordinate_mse_means": errors.mean(axis=0).tolist(),
              "all_prefix_coordinate_mse_mean": float(errors.mean()),
              "endpoint_target_geometry": geometry, "elapsed_s": time.time() - start}
    del bundle, model, saved, predictions, targets, errors
    gc.collect()
    return result


def audit(target_root, last_only_root, data_root, window_pin, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite a nonempty output directory")
    # Resolve all 18 complete final checkpoints before encoding any data.
    cases = registered_cases(target_root, last_only_root)
    campaign_path = target_root / "campaign.json"
    campaign_sha = G.sha256(campaign_path)
    campaign = json.loads(campaign_path.read_text())
    if campaign.get("status") != "complete":
        raise ValueError("run the common audit only after the target-gradient campaign completes")
    pinned_eval = {}
    for world in WORLDS:
        matches = [digest for name, digest in campaign["data_sha256"].items()
                   if Path(name).parts[-2:] == (world, "eval.npz")]
        if len(matches) != 1:
            raise ValueError(f"no unique frozen campaign evaluation archive for {world}")
        pinned_eval[world] = matches[0]
    pin_path = window_pin / "window_pin.json"
    pin_sha = G.sha256(pin_path)
    pin = json.loads(pin_path.read_text())
    if (pin.get("protocol") != PROTOCOL + "_window_pin" or pin.get("status") != "complete"
            or pin.get("windows_per_world") != N_WINDOWS or pin.get("sample_seed") != SAMPLE_SEED
            or set(pin.get("data", {})) != set(WORLDS)):
        raise ValueError("missing or invalid prospective common window pin")
    for relative, expected in pin["sources_sha256"].items():
        if G.sha256(ROOT / relative) != expected:
            raise ValueError(f"audit source changed after prospective window pinning: {relative}")
    for world in WORLDS:
        entry = pin["data"][world]
        if (entry["sha256"] != pinned_eval[world]
                or G.sha256(window_pin / entry["artifact"]) != entry["artifact_sha256"]):
            raise ValueError("window pin archive/artifact differs from registered campaign")
    source_paths = [Path(__file__).resolve(), ROOT / "train.py", ROOT / "studies/revision/target_gradient.py",
                    ROOT / "studies/revision/last_only_objective.py", ROOT / "research/full_history_audit_protocol.md"]
    sources = {str(path.relative_to(ROOT)): G.sha256(path) for path in source_paths}
    out.mkdir(parents=True, exist_ok=True)
    report = {"protocol": PROTOCOL, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "expected_models": 18, "worlds": list(WORLDS), "model_seeds": list(SEEDS), "modes": list(MODES),
              "windows_per_world": N_WINDOWS, "sample_seed": SAMPLE_SEED,
              "sampler": "numpy.Generator(PCG64(seed)).choice(valid_windows,8192,replace=False)",
              "batch": BATCH, "encoded_frames_per_window": SPAN, "flattened_encoder_batch": BATCH * SPAN,
              "history": HISTORY, "num_preds": 1, "latent_coordinates": LATENT,
              "checkpoint_selection": "all registered final 8500-update models; no outcome selection",
              "metric": "coordinate-mean one-step latent MSE after eight observed frames and outgoing actions",
              "prefix_curve": "one-step targets after 1 through 8 observed frames; teacher-forced, not rollout horizon",
              "interpretation": "representation-relative diagnostic; neither raw nor normalized MSE establishes information, control, or quality",
              "evaluation_fields": ["frames", "actions", "dones_for_window_boundaries_only"],
              "sources_sha256": sources, "target_campaign": str(campaign_path.resolve()),
              "target_campaign_sha256": campaign_sha, "registered_cases": cases,
              "window_pin": str(pin_path.resolve()), "window_pin_sha256": pin_sha,
              "python": platform.python_version(),
              "packages": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "optax", "numpy")},
              "devices": [str(device) for device in jax.devices()],
              "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
              "jax_enable_x64": jax.config.jax_enable_x64, "data": {}, "results": []}
    report_path = out / "audit.json"
    write_json(report_path, report)
    try:
        for world in WORLDS:
            path = data_root / world / "eval.npz"
            frames, actions, sample, provenance = load_data(path, world, pinned_eval[world])
            entry = pin["data"][world]
            if (provenance["window_table_sha256"] != entry["window_table_sha256"]
                    or provenance["sample_array_sha256"] != entry["sample_array_sha256"]):
                raise ValueError("regenerated evaluation windows differ from prospective pin")
            with np.load(window_pin / entry["artifact"], allow_pickle=False) as pinned_arrays:
                for name, value in sample.items():
                    np.testing.assert_array_equal(value, pinned_arrays[name], err_msg="prospective window pin differs")
            report["data"][world] = provenance
            write_json(report_path, report)
            for case in [item for item in cases if item["world"] == world]:
                print(f"[audit] {world} m{case['model_seed']} {case['mode']}", flush=True)
                result = evaluate_case(case, frames, actions, sample, out)
                report["results"].append(result)
                write_json(report_path, report)
            del frames, actions, sample
            gc.collect()
        for relative, expected in sources.items():
            if G.sha256(ROOT / relative) != expected:
                raise ValueError(f"audit source changed during evaluation: {relative}")
        if G.sha256(campaign_path) != campaign_sha:
            raise ValueError("target campaign changed during audit; run after its completion")
        if G.sha256(pin_path) != pin_sha:
            raise ValueError("prospective window pin changed during audit")
        for case in cases:
            for name in ("checkpoint", "manifest"):
                if G.sha256(case[name]) != case[name + "_sha256"]:
                    raise ValueError(f"frozen {name} changed during evaluation")
        for world in WORLDS:
            entry = pin["data"][world]
            if G.sha256(window_pin / entry["artifact"]) != entry["artifact_sha256"]:
                raise ValueError("prospective window ID artifact changed during audit")
            if G.sha256(data_root / world / "eval.npz") != pinned_eval[world]:
                raise ValueError("evaluation archive changed during audit")
            rows = [row for row in report["results"] if row["world"] == world]
            for field in ("batch_frame_hash_chain_sha256", "batch_outgoing_action_hash_chain_sha256"):
                if len({row[field] for row in rows}) != 1:
                    raise ValueError("models received different evaluation inputs")
        if len(report["results"]) != 18:
            raise ValueError("incomplete fixed model matrix")
        report.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        report.update(status="failed_incomplete", failure={"type": type(error).__name__, "message": str(error)})
        write_json(report_path, report)
        raise
    write_json(report_path, report)
    print(f"[complete] all 18 models -> {report_path}", flush=True)


def timing_smoke(checkpoint, data_path, world, out):
    """Two canonical batches: compilation and warm runtime, without any scores."""
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite a nonempty output directory")
    with np.load(checkpoint, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
    steps = meta["cfg"]["steps"]
    if (steps not in (500, 8500) or meta.get("completed_steps") != steps
            or meta.get("cfg") != asdict(replace(G.production_config(meta["cfg"]["seed"]), steps=steps))
            or meta.get("config_label") != world):
        raise ValueError("timing fixture must have the exact production architecture and 500 or 8500 updates")
    checkpoint_sha = G.sha256(checkpoint)
    data_sha = G.sha256(data_path)
    frames, actions, sample, provenance = load_data(data_path, world, data_sha)
    model = T.load_model(checkpoint)[0].model
    state_before = state_sha256(model)
    seconds = []
    for index in range(2):
        window, outgoing = gather_windows(frames, actions, sample["windows_start_env"][index * BATCH:(index + 1) * BATCH])
        if window.shape != (BATCH, SPAN, T.H, T.W, 3) or outgoing.shape != (BATCH, HISTORY):
            raise ValueError("noncanonical timing fixture")
        start = time.perf_counter()
        pred, target = jax.device_get(infer(model, jnp.asarray(window), jnp.asarray(outgoing)))
        seconds.append(time.perf_counter() - start)
        if not np.isfinite(pred).all() or not np.isfinite(target).all():
            raise ValueError("nonfinite timing-fixture output; no timing admission")
    if state_sha256(model) != state_before or G.sha256(checkpoint) != checkpoint_sha or G.sha256(data_path) != data_sha:
        raise ValueError("timing smoke changed model state or input artifact identity")
    out.mkdir(parents=True, exist_ok=True)
    result = {"protocol": PROTOCOL + "_timing_only", "publication_eligible": False,
              "purpose": "runtime admission only; no metric computed or reported", "status": "complete",
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
              "data": provenance, "fixture_steps": steps, "batch": BATCH, "span": SPAN,
              "compiled_first_batch_seconds": seconds[0], "warmed_batch_seconds": seconds[1],
              "projected_64_batch_forward_seconds": seconds[0] + 63 * seconds[1],
              "projection_excludes": "checkpoint/archive loading, hashing, host window gathering, geometry and artifact writing",
              "frozen_state_sha256_before_after": state_before,
              "sources_sha256": {str(path.relative_to(ROOT)): G.sha256(path) for path in
                                 (Path(__file__).resolve(), ROOT / "train.py")},
              "packages": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "numpy")},
              "devices": [str(device) for device in jax.devices()],
              "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
              "jax_enable_x64": jax.config.jax_enable_x64}
    write_json(out / "timing_only.json", result)
    print(json.dumps({key: result[key] for key in ("publication_eligible", "compiled_first_batch_seconds", "warmed_batch_seconds", "projected_64_batch_forward_seconds")}))


def self_test():
    """CPU-only synthetic indexing, arithmetic, geometry and actual-model checks."""
    if jax.default_backend() != "cpu":
        raise RuntimeError("self-test must run on CPU")
    dones = np.zeros((20, 2), np.int64)
    dones[7, 0] = 1
    table, ids, windows = select_windows(dones, count=12, seed=SAMPLE_SEED)
    expected = np.array([(s, e) for s in range(13) for e in range(2)
                         if not dones[s:s + 8, e].any()], dtype=np.int32)
    np.testing.assert_array_equal(table, expected)
    np.testing.assert_array_equal(ids, select_windows(dones, 12, SAMPLE_SEED)[1])
    assert len(np.unique(ids)) == len(ids)
    try:
        select_windows(dones, count=len(table) + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("insufficient windows were silently accepted")
    frames = (100 * np.arange(21)[:, None] + np.arange(2)[None, :])[..., None, None, None]
    actions = 100 * np.arange(20)[:, None] + np.arange(2)[None, :]
    images, acts = gather_windows(frames, actions, windows)
    for row, (start, env) in enumerate(windows):
        np.testing.assert_array_equal(images[row, :, 0, 0, 0], frames[start:start + 9, env, 0, 0, 0])
        np.testing.assert_array_equal(acts[row], actions[start:start + 8, env])
    pred = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3) / 10
    target = pred / 3
    errors = prefix_errors(pred, target)
    from studies.revision import last_only_objective as L
    np.testing.assert_allclose(errors[:, -1].mean(), float(L.prediction_mse(jnp.asarray(pred), jnp.asarray(target), "last_only")), rtol=2e-7)
    np.testing.assert_allclose(errors.mean(), float(G.prediction_mse(jnp.asarray(pred), jnp.asarray(target), "joint")), rtol=2e-7)
    geometry, _ = target_geometry(np.zeros((10, 3)), 0.0)
    assert geometry["target_covariance_effective_rank"] is None
    assert geometry["endpoint_mse_over_target_variance"] is None
    cfg = replace(G.production_config(2), batch=2, emb_dim=8, enc_depth=0, enc_heads=1,
                  enc_res=0, token_dim=1, token_hidden=8, pred_width=8, pred_depth=0, pred_heads=1)
    model = T.LeWM(cfg, T.rngs(cfg.seed))
    images = jnp.asarray(np.random.default_rng(7).integers(0, 256, (2, 9, T.H, T.W, 3), dtype=np.uint8))
    acts = jnp.asarray(np.arange(16, dtype=np.int32).reshape(2, 8) % 4)
    before = state_sha256(model)
    actual, truth = infer(model, images, acts)
    # Independently spell out T.train's full-window forward and target slice,
    # changing only train=True to inference mode so BatchNorm cannot update.
    @nnx.jit
    def reference_forward(module, window, window_actions):
        emb = module.encode(window, False)
        act_emb = module.act_embed(window_actions)
        pred = module.predict(emb[:, :cfg.history], act_emb[:, :cfg.history], False)
        return pred, emb[:, cfg.num_preds:]

    expected_pred, expected_target = reference_forward(model, images, acts)
    # Comparing a fused JIT to eager BF16 operations would mix an execution-
    # fusion discrepancy into the indexing test. Both forwards are compiled.
    np.testing.assert_array_equal(actual, expected_pred)
    np.testing.assert_array_equal(truth, expected_target)
    check = prefix_errors(actual, truth)
    np.testing.assert_allclose(check[:, -1].mean(), float(jnp.mean(jnp.square(expected_pred[:, -1] - expected_target[:, -1]))), rtol=2e-6)
    assert state_sha256(model) == before
    assert asdict(G.production_config(0))["batch"] == BATCH
    assert N_WINDOWS % BATCH == 0
    print("PASS: CPU reset boundaries, fixed unique sampling, outgoing actions, MSE normalization, collapse geometry, actual-model full-history forward, immutable full NNX state", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--last-only-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/dev/shm/redcube"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pin-windows", action="store_true", help="CPU-only prospective common window indices; run before fitting")
    parser.add_argument("--window-pin", type=Path, help="required prospective pin directory for publication audit")
    parser.add_argument("--smoke", action="store_true", help="two fixed canonical batches; timing only, never publication")
    parser.add_argument("--model", type=Path, help="only for --smoke")
    parser.add_argument("--eval-data", type=Path, help="only for --smoke")
    parser.add_argument("--world", choices=WORLDS, help="only for --smoke")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.pin_windows:
        if not args.out:
            parser.error("--pin-windows requires --out")
        pin_windows(args.data_root, args.out)
        return
    if args.smoke:
        if not all((args.model, args.eval_data, args.world, args.out)):
            parser.error("--smoke requires --model, --eval-data, --world and --out")
        timing_smoke(args.model, args.eval_data, args.world, args.out)
        return
    if not all((args.target_root, args.last_only_root, args.window_pin, args.out)):
        parser.error("--target-root, --last-only-root, --window-pin and --out are required")
    audit(args.target_root, args.last_only_root, args.data_root, args.window_pin, args.out)


if __name__ == "__main__":
    main()
