"""Audit and analyze the fixed twelve-fit target-gradient experiment (NumPy only).

python -m studies.revision.analyze_target_gradient --self-test
python -m studies.revision.analyze_target_gradient --runs-root RUNS \
    --data-root DATA --output-dir OUTPUT --require-complete

No endpoint is selected by outcome. An incomplete matrix produces an explicit
status report and individual validated diagnostics, but no publication contrast.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np


WORLDS = ("gray_frame_still", "gray_frame_wind")
MODES = ("joint", "stop_target")
SEEDS = (0, 1, 2)
STEPS = 8500
CHECKPOINTS = (0, 500, 1000, 2000, 4000, 4250, 8500)
SOURCE_NAMES = (
    "studies/revision/run_target_gradient_queue.py", "studies/revision/target_gradient.py",
    "studies/revision/target_gradient_probe.py", "studies/revision/target_gradient_sensitivity.py",
    "studies/stoch/train_arms.sh", "studies/redcube/env.py", "train.py", "probe.py", "pyproject.toml",
)
TRAIN_SOURCES = ("studies/revision/target_gradient.py", "train.py", "studies/stoch/train_arms.sh", "pyproject.toml")
PROBE_SOURCES = ("studies/revision/target_gradient_probe.py", "train.py", "probe.py", "studies/redcube/env.py")
REPO = Path(__file__).resolve().parents[2]
LIMITATIONS = [
    "The replication units are three fixed model initialization/data-order seeds, conditional on one training and one evaluation archive per world.",
    "The 50,000 readout-training frames and 10,000 evaluation frames are not independent model fits; no frame-based model-family confidence interval or p-value is reported.",
    "Wind changes visitation, temporal complexity and episode lengths as well as innovation variance; the interaction is not unique causal identification of an innovation-variance mechanism.",
    "The MLP is the sole primary readout. Ridge, representation rank, training loss and state/nuisance sensitivity are separate descriptive diagnostics, without winner selection.",
    "Readout failure is not proof of information absence; readout success is not proof of successful control.",
    "All saved checkpoints belong to the 8,500-update schedule. The final checkpoint is fixed prospectively; intermediate losses or readouts never select it.",
    "A positive interaction is a relative effect. It can result from worse still-world errors without any wind-world improvement; both within-world benefits and all twelve raw errors must accompany it.",
    "This NumPy audit verifies saved predictions and their metrics, not a fresh encoder/readout execution. Raw-feature hashes cannot be independently recomputed because features are not saved; independent re-encoding is a separate check.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def training_array_hash(value):
    """Training digest; intentionally distinct from the probe digest."""
    value = np.asarray(value)
    digest = hashlib.sha256(json.dumps({"dtype": str(value.dtype), "shape": value.shape}, sort_keys=True).encode())
    digest.update(memoryview(np.ascontiguousarray(value).reshape(-1).view(np.uint8)))
    return digest.hexdigest()


def probe_array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_identity(sources):
    out = {}
    for path, digest in sources.items():
        normalized = str(path).replace("\\", "/")
        matches = [name for name in SOURCE_NAMES if normalized == name or normalized.endswith("/" + name)]
        require(bool(matches), f"unregistered source path: {path}")
        name = max(matches, key=len)
        require(re.fullmatch("[0-9a-f]{64}", digest) is not None, f"invalid source digest: {path}")
        require(name not in out, f"duplicate source identity: {name}")
        out[name] = digest
    return out


def expected_config(seed):
    return dict(seed=seed, steps=8500, batch=128, lr=.001, weight_decay=.001, warmup=.1,
                encoder="tile", predictor="factored", emb_dim=128, enc_depth=6, enc_heads=8,
                enc_mlp=768, enc_res=4, pred_depth=6, pred_heads=8, pred_dim_head=64,
                pred_mlp=2048, pred_width=256, mlp_hidden=2048, tokens=16, token_dim=32,
                token_hidden=512, latent_dim=0, history=8, num_preds=1, dropout=.1,
                sigreg_weight=.09, sigreg_knots=17, sigreg_proj=1024, action_bottleneck=10,
                n_actions=4, data_budget=500000, chunk=250)


def expected_samples(shape, count, seed):
    require(count <= int(np.prod(shape)), "archive smaller than fixed readout sample")
    flat = np.random.default_rng(seed).choice(int(np.prod(shape)), count, replace=False).astype(np.int64)
    return flat, np.stack(np.unravel_index(flat, shape), -1).astype(np.int64)


def npy_header(handle):
    version = np.lib.format.read_magic(handle)
    require(version in ((1, 0), (2, 0)), f"unsupported NPY format version {version}")
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    return reader(handle)


def inspect_archive(path, world, split):
    """Read labels/sampling inputs and stream exact selected pixel-byte hashes."""
    path = Path(path)
    digest = file_hash(path)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        positions = archive["pos"]
        require(positions.ndim == 3 and positions.shape[-1] == 2 and np.isfinite(positions).all(), "invalid position archive")
        require(meta.get("env") == "redcube" and meta.get("config") == world and meta.get("split") == split,
                f"incorrect data identity: {world}/{split}")
        require(meta.get("policy") == "uniform random" and meta.get("num_actions") == 4, "incorrect data policy/action vocabulary")
        spec = dict(meta.get("spec", {})); spec.setdefault("gate", False)
        require(spec == {"noise": "gray", "per_episode": False, "wind": world == WORLDS[1], "gate": False}, "incorrect world specification")
        require(isinstance(meta.get("seed"), int), "missing collection seed")
        shape = positions.shape[:2]
        count, seed = (50000, 19417) if split == "train" else (10000, 19418)
        flat, ids = expected_samples(shape, count, seed)
        labels = positions[ids[:, 0], ids[:, 1]].astype(np.float32)
        actions = archive["actions"].astype(np.int32) if split == "train" else None
        dones = archive["dones"].astype(np.int64) if split == "train" else None
    with zipfile.ZipFile(path) as archive, archive.open("frames.npy") as handle:
        frames_shape, fortran, dtype = npy_header(handle)
        require(frames_shape == (*shape, 63, 63, 3) and dtype == np.uint8 and not fortran, "invalid pixel array shape, dtype or order")
        hashed = hashlib.sha256(json.dumps({"dtype": str(dtype), "shape": frames_shape}, sort_keys=True).encode()) if split == "train" else None
        frame_bytes, read_bytes = 63 * 63 * 3, 0
        content_hashes = np.empty(count, dtype="S64")
        order = np.argsort(flat); ordered_flat = flat[order]; cursor = 0
        while block := handle.read(512 * frame_bytes):
            require(len(block) % frame_bytes == 0, "truncated frame-array payload")
            if hashed is not None:
                hashed.update(block)
            first_frame = read_bytes // frame_bytes
            last_frame = first_frame + len(block) // frame_bytes
            view = memoryview(block)
            while cursor < len(flat) and ordered_flat[cursor] < last_frame:
                offset = int(ordered_flat[cursor] - first_frame) * frame_bytes
                content_hashes[order[cursor]] = hashlib.sha256(view[offset:offset+frame_bytes]).hexdigest()
                cursor += 1
            read_bytes += len(block)
        require(read_bytes == int(np.prod(frames_shape)) and cursor == count, "truncated frame array")
        frame_digest = hashed.hexdigest() if hashed is not None else None
    result = {"path": str(path.resolve()), "sha256": digest, "metadata": meta, "shape": shape,
              "flat_indices": flat, "indices": ids, "positions": labels, "frame_sha256": content_hashes}
    if split == "train":
        require(actions.shape == dones.shape == (shape[0] - 1, shape[1]), "training action/done shape mismatch")
        require(actions.size == 500000 and ((actions >= 0) & (actions < 4)).all(), "training archive differs from fixed 500,000-transition budget")
        cumulative = np.concatenate([np.zeros((1, shape[1]), np.int64), np.cumsum(dones, axis=0)])
        starts = np.arange(actions.shape[0] - 9 + 2)
        rows, cols = np.nonzero(cumulative[starts + 8] == cumulative[starts])
        table = np.stack([starts[rows], cols], 1).astype(np.int32)
        result.update(window_table=table, data_shape={"frames": list(frames_shape), "actions": list(actions.shape),
                      "windows": len(table), "span": 9}, data_array_sha256={"frames": frame_digest,
                      "actions_int32": training_array_hash(actions), "dones_int64": training_array_hash(dones)})
    return result


def checkpoint_identity(path):
    digest = hashlib.sha256()
    n_params = 0
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        seen = []
        for kind in ("p", "b"):
            names = [name for name in archive.files if name.startswith(f"model.{kind}.")]
            indices = sorted(int(name.rsplit(".", 1)[1]) for name in names)
            require(indices == list(range(len(indices))), "noncontiguous checkpoint leaf indices")
            for index in indices:
                name = f"model.{kind}.{index}"; value = archive[name]; seen.append(name)
                require(np.isfinite(value).all(), f"nonfinite checkpoint array {name}")
                digest.update(f"{kind}.{index}:".encode()); digest.update(training_array_hash(value).encode())
                if kind == "p":
                    n_params += value.size
        require(n_params > 0 and set(archive.files) == {"meta", *seen}, "missing parameters or unexpected checkpoint arrays")
    return meta, digest.hexdigest(), n_params


def recompute_metrics(target, prediction):
    target, prediction = np.asarray(target, np.float64), np.asarray(prediction, np.float64)
    require(target.shape == prediction.shape and target.ndim == 2 and target.shape[1] == 2 and len(target), "invalid per-frame prediction shape")
    require(np.isfinite(target).all() and np.isfinite(prediction).all(), "nonfinite per-frame outcome")
    error = prediction - target
    mse_xy = np.square(error).mean(0)
    sse = np.square(error).sum(0); sst = np.square(target - target.mean(0)).sum(0)
    return {"mae_px": float(np.abs(error).mean()), "l1_px": float(np.abs(error).sum(-1).mean()),
            "rmse_px": float(np.sqrt(np.square(error).mean())), "euclidean_rmse_px": float(np.sqrt(mse_xy.sum())),
            "mae_xy_px": np.abs(error).mean(0).tolist(), "rmse_xy_px": np.sqrt(mse_xy).tolist(),
            "r2": float(1 - sse.sum() / sst.sum()) if sst.sum() > 0 else None,
            "r2_xy": [float(1 - a/b) if b > 0 else None for a, b in zip(sse, sst)]}


def close(actual, expected, label):
    if expected is None:
        require(actual is None, f"{label}: undefined metric should be null")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"{label}: invalid vector")
        for index, (left, right) in enumerate(zip(actual, expected)):
            close(left, right, f"{label}[{index}]")
    else:
        require(np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-6, atol=1e-7), f"{label}: saved value differs from independent recomputation")


def validate_probe_arrays(arrays, results, sampling, data):
    identity = {}
    means = data["train"]["positions"].astype(np.float64).mean(0)
    for split in ("train", "eval"):
        target = arrays[split + "_positions"]
        for suffix in ("positions", "flat_indices", "indices", "frame_sha256"):
            actual = arrays[split + "_" + suffix]
            require(np.array_equal(actual, data[split][suffix]), f"{split} {suffix} differs from exact registered archive sample")
        for suffix in ("positions", "indices"):
            require(probe_array_hash(arrays[split + "_" + suffix]) == sampling[split + "_" + suffix + "_sha256"], f"{split} {suffix} hash mismatch")
        hashes = arrays[split + "_frame_sha256"]
        require(hashes.shape == (len(target),) and hashes.dtype.kind == "S" and hashes.dtype.itemsize == 64, f"{split} invalid frame hash array")
        require(all(re.fullmatch(b"[0-9a-f]{64}", value) is not None for value in hashes), f"{split} invalid frame hash entry")
        identity[split] = {suffix: probe_array_hash(arrays[split + "_" + suffix]) for suffix in ("positions", "flat_indices", "indices", "frame_sha256")}
        for name in ("mlp", "ridge", "train_mean"):
            prediction = arrays[f"{split}_{name}_prediction_px"]
            error = arrays[f"{split}_{name}_error_px"]
            require(error.shape == target.shape and np.isfinite(error).all(), f"{split}/{name}: invalid raw error")
            # Saved FP32 MLP subtraction may round once; compare to the same
            # subtraction for the raw error, then compute metrics in FP64.
            require(np.array_equal(error, prediction - target), f"{split}/{name}: saved error differs from prediction minus label")
            if name == "train_mean":
                require(np.allclose(prediction, means[None], rtol=0, atol=1e-10), "training-mean baseline used different labels")
            expected = recompute_metrics(target, prediction)
            for metric, value in expected.items():
                close(results[split][name][metric], value, f"{split}/{name}/{metric}")
    return identity


def probe_signature(protocol):
    sampling = {k: v for k, v in protocol["sampling"].items() if not k.endswith("_sha256")}
    expected = {"train_count": 50000, "eval_count": 10000, "train_seed": 19417, "eval_seed": 19418, "replacement": False,
                "eligible": "every archived frame; no filtering", "rng": "numpy.default_rng (PCG64)",
                "flat_index_order": "time-major then environment", "frame_identity": "archive sha256 plus [time_index, environment_index]",
                "frame_content_hash": "SHA-256 of contiguous uint8 HWC bytes"}
    require(sampling == expected, "fixed readout frame-sampling protocol changed")
    mlp = protocol["mlp"]
    expected_mlp = dict(widths=[512, 256, 256, 2], activation="GELU (JAX default approximate)", dtype="float32", seed=0,
                        steps=4000, batch=1024, optimizer="AdamW", lr=.001, weight_decay=.0001, b1=.9, b2=.999,
                        epsilon=1e-8, initialization="Glorot uniform kernels, zero biases",
                        batch_sampling="with replacement from fixed training sample", inputs="raw 512-dimensional encoder outputs; no normalization",
                        target_scale_px=63., checkpoint="final step; no selection")
    require(mlp == expected_mlp, "fixed MLP protocol changed")
    ridge = protocol["ridge"]
    require(ridge == dict(factor=.001, penalty="1e-3 * trace(population feature covariance) / D",
                         features="raw centered latents, no per-coordinate scaling", intercept="unpenalized"), "fixed ridge protocol changed")
    return {"sampling": sampling, "mlp": mlp, "ridge": ridge, "metric_conventions": protocol["metric_conventions"],
            "position_convention": protocol["position_convention"], "label_use": protocol["label_use"]}


def load_probe(directory, checkpoint, meta, data, sources):
    protocol = read_json(directory / "protocol.json")
    require(protocol["status"] == "complete" and protocol["protocol"] == "target_gradient_readouts_v1", "incomplete or unregistered readout")
    require(protocol["primary_endpoint"] == "eval.mlp.mae_px", "primary readout endpoint changed")
    signature = probe_signature(protocol)
    require(source_identity(protocol["runtime_source_sha256"]) == {name: sources[name] for name in PROBE_SOURCES}, "probe source mismatch")
    require(protocol["sha256"] == {"model": file_hash(checkpoint), "train": data["train"]["sha256"], "eval": data["eval"]["sha256"]}, "readout checkpoint/data digest mismatch")
    require(protocol["model_metadata"] == meta, "readout checkpoint metadata mismatch")
    for split in ("train", "eval"):
        require(protocol[split + "_metadata"] == data[split]["metadata"], f"readout {split} metadata mismatch")
    for name in ("results.json", "per_frame.npz", "ridge_readout.npz", "mlp_readout.npz"):
        require(file_hash(directory / name) == protocol["artifacts"][name], f"readout artifact hash mismatch: {name}")
    result = read_json(directory / "results.json")
    require(result["primary_endpoint"] == "eval.mlp.mae_px", "results primary endpoint changed")
    with np.load(directory / "per_frame.npz", allow_pickle=False) as arrays:
        identity = validate_probe_arrays(arrays, result, protocol["sampling"], data)
        eval_eigenvalues = arrays["eval_covariance_eigenvalues"]
    with np.load(directory / "ridge_readout.npz", allow_pickle=False) as ridge:
        require(ridge["weights"].shape == (512, 2) and ridge["intercept"].shape == (2,), "ridge parameter shape mismatch")
        require(all(np.isfinite(ridge[name]).all() for name in ridge.files), "nonfinite ridge artifact")
        close(ridge["position_mean"].tolist(), data["train"]["positions"].astype(np.float64).mean(0).tolist(), "ridge position mean")
        eigenvalues = {"train": ridge["covariance_eigenvalues"], "eval": eval_eigenvalues}
        for split, values in eigenvalues.items():
            require(values.shape == (512,) and np.isfinite(values).all() and (values >= 0).all(), "invalid covariance eigenvalues")
            positive = values[values > 0]; p = positive / positive.sum() if len(positive) else positive
            diagnostics = dict(mean_feature_variance=float(values.sum() / 512),
                               effective_rank=float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.,
                               participation_ratio=float(values.sum()**2/(values@values)) if len(p) else 0.)
            for key, value in diagnostics.items():
                # Clipped eigenvalues and direct covariance trace can differ by
                # roundoff at a numerically degenerate constant representation.
                close(result["representation_" + split][key], value, f"representation {split}/{key}")
        close(result["representation_train"]["ridge_penalty"], .001 * result["representation_train"]["mean_feature_variance"], "ridge penalty")
        close(float(ridge["ridge_penalty"]), result["representation_train"]["ridge_penalty"], "ridge saved penalty")
    with np.load(directory / "mlp_readout.npz", allow_pickle=False) as mlp:
        require(mlp["normalized_training_mse"].shape == (4000,), "incomplete MLP optimizer trace")
        for name in mlp.files:
            require(np.isfinite(mlp[name]).all(), f"nonfinite MLP artifact: {name}")
        for index, shape in enumerate(((512, 256), (256, 256), (256, 2))):
            require(mlp[f"layer_{index}_kernel"].shape == shape and mlp[f"layer_{index}_bias"].shape == (shape[1],), "MLP parameter shape mismatch")
    return {"metrics": {split: result[split] for split in ("train", "eval")},
            "representation_train": result["representation_train"], "representation_eval": result["representation_eval"],
            "sample_identity": identity, "protocol_signature": signature,
            "artifacts": {**protocol["artifacts"], "protocol.json": file_hash(directory / "protocol.json")}}


def load_sensitivity(directory, checkpoint, data, cfg, sources):
    path = directory / "sensitivity.json"
    if not path.is_file():
        return None
    report = read_json(path)
    require(report["config"] == data["eval"]["metadata"]["config"] and report["count"] == 1024 and report["seed"] == 19418,
            "sensitivity sample protocol changed")
    require(report["position_displacement_pixels"] == 3, "sensitivity displacement changed")
    require(report["model_cfg"] == cfg and report["model_sha256"] == file_hash(checkpoint) and report["eval_data_sha256"] == data["eval"]["sha256"], "sensitivity checkpoint/data mismatch")
    required_sources = ("studies/revision/target_gradient_sensitivity.py", "train.py", "studies/redcube/env.py")
    require(source_identity(report["source_sha256"]) == {name: sources[name] for name in required_sources}, "sensitivity source mismatch")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        for name in arrays.files:
            require(np.isfinite(arrays[name]).all(), f"nonfinite sensitivity array: {name}")
        original, shifted, nuisance = (arrays[name] for name in ("latent_original", "latent_position_shift", "latent_nuisance_shift"))
        rng = np.random.default_rng(19418)
        flat = rng.choice(int(np.prod(data["eval"]["shape"])), 1024, replace=False)
        ids = np.stack(np.unravel_index(flat, data["eval"]["shape"]), axis=1)
        with np.load(data["eval"]["path"], allow_pickle=False) as archive:
            positions = archive["pos"][ids[:, 0], ids[:, 1]].astype(np.int32)
        axes = rng.integers(0, 2, 1024); steps = rng.choice(np.array([-3, 3], np.int32), 1024)
        moved = positions[np.arange(1024), axes] + steps
        steps = np.where((moved < 2) | (moved > 55), -steps, steps)
        displacements = np.zeros_like(positions); displacements[np.arange(1024), axes] = steps
        require(np.array_equal(arrays["indices"], ids) and np.array_equal(arrays["positions"], positions) and
                np.array_equal(arrays["displacements"], displacements), "sensitivity contexts differ from exact registered sample")
        require(((positions + displacements >= 2) & (positions + displacements <= 55)).all(), "sensitivity displacement leaves valid state support")
        require(arrays["background_keys"].shape == (2, 1024, 2) and arrays["background_keys"].dtype == np.uint32, "sensitivity background-key shape/dtype mismatch")
        require(original.shape == shifted.shape == nuisance.shape == (1024, 512), "sensitivity latent shape mismatch")
        numerator = np.square(shifted.astype(np.float64) - original).sum(1)
        denominator = np.square(nuisance.astype(np.float64) - original).sum(1)
        require(np.array_equal(arrays["state_distance_sq"], numerator) and np.array_equal(arrays["nuisance_distance_sq"], denominator), "sensitivity distances mismatch")
        close(report["mean_state_distance_sq"], float(numerator.mean()), "sensitivity numerator")
        close(report["mean_nuisance_distance_sq"], float(denominator.mean()), "sensitivity denominator")
        ratio = float(numerator.mean()/denominator.mean()) if denominator.mean() > 1e-20 else None
        close(report["state_to_nuisance_distance_ratio"], ratio, "sensitivity ratio")
        require(report["denominator_too_small"] == (denominator.mean() <= 1e-20), "sensitivity denominator flag mismatch")
        identity = {name: probe_array_hash(arrays[name]) for name in ("indices", "positions", "displacements", "background_keys")}
    return {"mean_state_distance_sq": report["mean_state_distance_sq"], "mean_nuisance_distance_sq": report["mean_nuisance_distance_sq"],
            "state_to_nuisance_distance_ratio": report["state_to_nuisance_distance_ratio"], "denominator_too_small": report["denominator_too_small"],
            "sample_identity": identity, "artifact_sha256": {path.name: file_hash(path), path.with_suffix(".npz").name: file_hash(path.with_suffix(".npz"))}}


def load_run(directory, world, seed, mode, data, sources, midpoint):
    manifest = read_json(directory / "manifest.json")
    require(manifest["protocol"] == "paired_target_gradient_v1" and manifest["status"] == "complete", "incomplete or unregistered model fit")
    require(manifest["completed_steps"] == STEPS and manifest["cfg"] == expected_config(seed), "fit budget/configuration differs from fixed 8,500-update protocol (500-step smoke is excluded)")
    require((manifest["config_label"], manifest["model_seed"], manifest["mode"]) == (world, seed, mode), "world/seed/mode mismatch")
    require((manifest["parameter_rng_seed"], manifest["dropout_rng_seed"], manifest["update_rng_seed"]) == (seed, seed+1, seed+7), "training RNG seed mismatch")
    require(manifest["fixed_checkpoint_steps"] == list(CHECKPOINTS), "fixed checkpoint schedule changed")
    require(manifest["training_fields"] == ["frames", "actions", "dones_for_window_boundaries_only"], "unexpected training inputs")
    require(manifest["target_gradient_rule"] == ("stop target embedding in MSE only" if mode == "stop_target" else "joint MSE gradients") and
            manifest["sigreg_gradient_rule"] == "all embeddings, identical to train.py" and
            manifest["checkpoint_selection"] == "fixed final update; no selection by evaluation", "objective/checkpoint-selection metadata changed")
    require(manifest["loss_columns"] == ["prediction_mse", "sigreg"], "loss column mismatch")
    require(source_identity(manifest["sources"]["sha256"]) == {name: sources[name] for name in TRAIN_SOURCES}, "training source mismatch")
    require(manifest["data_sha256"] == data["train"]["sha256"] and manifest["data_array_sha256"] == data["train"]["data_array_sha256"], "training archive/array digest mismatch")
    require(manifest["data_shape"] == data["train"]["data_shape"], "training data shape mismatch")
    identity = {name: data["train"]["metadata"][name] for name in ("env", "config", "split", "num_actions", "spec", "seed")}
    require(manifest["data_identity"] == identity, "training data metadata mismatch")
    require(file_hash(directory / "sampling.npz") == manifest["sampling_file_sha256"], "sampling artifact hash mismatch")
    with np.load(directory / "sampling.npz", allow_pickle=False) as sampling:
        table, keys, windows = (sampling[name] for name in ("window_table", "update_keys", "sampled_window_ids"))
        require(np.array_equal(table, data["train"]["window_table"]), "sample window table differs from independently reconstructed reset-free windows")
        require(keys.shape == (STEPS, 2) and keys.dtype == np.uint32, "update-key shape/dtype mismatch")
        require(windows.shape == (STEPS, 128) and np.issubdtype(windows.dtype, np.integer) and ((windows >= 0) & (windows < len(table))).all(), "sampled window IDs invalid")
        for name, values in (("window_table", table), ("update_keys", keys), ("sampled_window_ids", windows)):
            require(training_array_hash(values) == manifest[name + "_sha256"], f"{name} digest mismatch")
    records = manifest["checkpoints"]
    require([row["step"] for row in records] == list(CHECKPOINTS), "missing/duplicate/unplanned saved checkpoints")
    checkpoint_meta = {}
    for row in records:
        step = row["step"]
        name = "model.npz" if step == STEPS else f"model_step{step:06d}.npz"
        require(row["path"] == name and file_hash(directory / name) == row["sha256"], f"checkpoint file identity mismatch at {step}")
        meta, weights, params = checkpoint_identity(directory / name)
        require(weights == row["weights_sha256"] and params == manifest["params"], f"checkpoint weight/parameter digest mismatch at {step}")
        require(meta["params"] == params and meta["cfg"] == manifest["cfg"] and meta["completed_steps"] == step and meta["config_label"] == world and meta["target_gradient_mode"] == mode,
                f"checkpoint metadata mismatch at {step}")
        require(meta["data_sha256"] == manifest["data_sha256"] and meta["initial_weights_sha256"] == manifest["initial_weights_sha256"] and
                meta["sampled_window_ids_sha256"] == manifest["sampled_window_ids_sha256"] and meta["fixed_checkpoint_steps"] == list(CHECKPOINTS), f"checkpoint provenance mismatch at {step}")
        require(meta["experiment"] == "paired_target_gradient_v1" and meta["source_sha256"] == sources["studies/revision/target_gradient.py"] and
                source_identity(meta["sources"]["sha256"]) == {name: sources[name] for name in TRAIN_SOURCES}, "checkpoint source/experiment mismatch")
        checkpoint_meta[step] = meta
    require(records[0]["weights_sha256"] == manifest["initial_weights_sha256"], "initial checkpoint differs from declared initialization")
    require(records[-1]["sha256"] == manifest["final_checkpoint_sha256"] and records[-1]["weights_sha256"] == manifest["final_weights_sha256"], "final checkpoint digest mismatch")
    require(file_hash(directory / "losses.npz") == manifest["loss_file_sha256"], "training loss artifact hash mismatch")
    with np.load(directory / "losses.npz", allow_pickle=False) as arrays:
        losses = arrays["history"]
    require(losses.shape == (STEPS, 2) and np.isfinite(losses).all(), "incomplete or nonfinite training loss trace")
    for step, meta in checkpoint_meta.items():
        for column, field in enumerate(("final_pred_loss", "final_sigreg")):
            close(meta[field], float(losses[step-1, column]) if step else None, f"checkpoint {step} {field}")
    row = {"world": world, "model_seed": seed, "mode": mode, "directory": str(directory.resolve()), "cfg": manifest["cfg"],
           "params": manifest["params"], "data_sha256": manifest["data_sha256"], "initial_weights_sha256": manifest["initial_weights_sha256"],
           "window_table_sha256": manifest["window_table_sha256"], "update_keys_sha256": manifest["update_keys_sha256"],
           "sampled_window_ids_sha256": manifest["sampled_window_ids_sha256"], "checkpoints": records,
           "runtime_signature": {name: manifest[name] for name in ("rng_implementation", "packages", "jax_default_matmul_precision", "jax_enable_x64")},
           "artifact_sha256": {"manifest.json": file_hash(directory / "manifest.json"), "losses.npz": manifest["loss_file_sha256"], "sampling.npz": manifest["sampling_file_sha256"]},
           "loss_summary": {"prediction_mse_final": float(losses[-1, 0]), "sigreg_final": float(losses[-1, 1]),
                            "prediction_mse_final250_mean": float(losses[-250:, 0].mean()), "sigreg_final250_mean": float(losses[-250:, 1].mean())},
           "final": load_probe(directory / "readout_final", directory / "model.npz", checkpoint_meta[STEPS], data, sources),
           "_losses": losses}
    if midpoint:
        row["midpoint"] = load_probe(directory / "readout_midpoint", directory / "model_step004250.npz", checkpoint_meta[4250], data, sources)
    try:
        row["sensitivity"] = load_sensitivity(directory, directory / "model.npz", data, manifest["cfg"], sources)
        row["sensitivity_validation_error"] = None
    except (ValueError, KeyError, OSError, TypeError, EOFError, zipfile.BadZipFile) as error:
        # This optional diagnostic never changes eligibility or selection of
        # the fixed primary MLP result. Its invalid values are not reported.
        row["sensitivity"] = None
        row["sensitivity_validation_error"] = str(error)
    return row


def pairing_errors(rows, midpoint):
    errors = []
    by_key = {(row["world"], row["model_seed"], row["mode"]): row for row in rows}
    for world in WORLDS:
        world_rows = [row for row in rows if row["world"] == world]
        for seed in SEEDS:
            pair = [by_key[key] for mode in MODES if (key := (world, seed, mode)) in by_key]
            if len(pair) == 2:
                for field in ("cfg", "params", "data_sha256", "initial_weights_sha256", "window_table_sha256", "update_keys_sha256", "sampled_window_ids_sha256", "runtime_signature"):
                    if pair[0][field] != pair[1][field]:
                        errors.append(f"{world}/m{seed}: paired {field} differs")
        for label in (("final", "midpoint") if midpoint else ("final",)):
            for field in ("sample_identity", "protocol_signature"):
                signatures = [row[label][field] for row in world_rows]
                if signatures and any(value != signatures[0] for value in signatures[1:]):
                    errors.append(f"{world}/{label}: {field} differs across models")
        if midpoint and any(row["final"]["sample_identity"] != row["midpoint"]["sample_identity"] for row in world_rows):
            errors.append(f"{world}: final and midpoint readout samples differ")
    if rows:
        for row in rows[1:]:
            if row["final"]["protocol_signature"] != rows[0]["final"]["protocol_signature"]:
                errors.append("fixed readout protocol differs across worlds")
                break
        for seed in SEEDS:
            same_seed = [row for row in rows if row["model_seed"] == seed]
            for field in ("initial_weights_sha256", "update_keys_sha256", "runtime_signature"):
                if same_seed and any(row[field] != same_seed[0][field] for row in same_seed[1:]):
                    errors.append(f"m{seed}: {field} differs across worlds")
        # Different model seeds must represent actual distinct initial fits.
        init = {seed: {row["initial_weights_sha256"] for row in rows if row["model_seed"] == seed} for seed in SEEDS}
        for left in SEEDS:
            for right in SEEDS:
                if left < right and init[left] & init[right]:
                    errors.append(f"model seeds {left} and {right} reuse an identical initialization")
    return errors


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {"values": values.tolist(), "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max()), "confidence_interval": None, "p_value": None}


def endpoint_summary(rows, label):
    by_key = {(row["world"], row["model_seed"], row["mode"]): row for row in rows}
    require(len(rows) == 12 and len(by_key) == 12, "publication contrast requires all twelve fixed fits")
    score = lambda world, seed, mode, readout: by_key[world, seed, mode][label]["metrics"]["eval"][readout]["mae_px"]
    summaries = {}
    for readout in ("mlp", "ridge"):
        paired = []
        for seed in SEEDS:
            row = {"model_seed": seed}
            for world, name in zip(WORLDS, ("still", "wind")):
                for mode in MODES:
                    row[f"{name}_{mode}_mae_px"] = score(world, seed, mode, readout)
                row[name + "_benefit_px"] = row[f"{name}_joint_mae_px"] - row[f"{name}_stop_target_mae_px"]
            row["interaction_px"] = row["wind_benefit_px"] - row["still_benefit_px"]
            paired.append(row)
        summaries[readout] = {"role": "primary" if readout == "mlp" else "secondary diagnostic; no readout selection",
                             "paired_seed_rows": paired, "interaction": summarize([row["interaction_px"] for row in paired]),
                             "benefit": {name: summarize([row[name + "_benefit_px"] for row in paired]) for name in ("still", "wind")},
                             "arms": {world: {mode: summarize([score(world, seed, mode, readout) for seed in SEEDS]) for mode in MODES} for world in WORLDS}}
    return summaries


def validate_admission(campaign):
    admission = campaign["admission"]
    require(admission["admitted"] is True, "campaign was not admitted")
    projected = admission["projected_fit_seconds"]
    require(set(projected) == set(MODES) and all(np.isfinite(value) and 0 < value <= 1200 for value in projected.values()),
            "fit timing projections violate the fixed admission rule")
    probe, remaining = admission["readout_seconds"], admission["remaining_seconds"]
    require(np.isfinite(probe) and probe > 0 and np.isfinite(remaining) and remaining > 0, "invalid runtime projection inputs")
    require(isinstance(campaign["midpoint_skip_requested"], bool), "midpoint skip request must be boolean")
    midpoint_cost = 1.5 * 12 * (max(projected.values()) + 2 * probe + 60)
    expected_midpoint = probe <= 300 and not campaign["midpoint_skip_requested"] and midpoint_cost < remaining
    midpoint = admission["midpoint_all_models"]
    require(isinstance(midpoint, bool) and midpoint == expected_midpoint, "midpoint decision differs from the predeclared timing-only rule")
    expected_cost = 1.5 * 12 * (max(projected.values()) + (1 + int(midpoint)) * probe + 60)
    close(admission["projected_campaign_seconds_with_margin"], expected_cost, "campaign runtime cost")
    require(expected_cost < remaining, "campaign admission violates remaining deadline with 50% margin")
    return midpoint


def analyze(root, data_root=None, source_root=REPO):
    root = Path(root); rows, invalid, errors = [], [], []
    result = {"schema_version": 1, "status": "incomplete", "publication_eligible": False, "runs_root": str(root.resolve()),
              "fixed_matrix": {"worlds": list(WORLDS), "model_seeds": list(SEEDS), "modes": list(MODES), "steps": STEPS, "fits": 12},
              "primary_endpoint": "final.eval.mlp.mae_px", "primary_contrast": "(joint_wind-stop_target_wind)-(joint_still-stop_target_still), paired by model seed",
              "metric_units": "mean absolute pixel error per coordinate", "validated_runs": rows, "invalid_runs": invalid, "validation_errors": errors,
              "missing_runs": [], "excluded_smoke": [str(path) for path in sorted((root / "smoke").glob("*/manifest.json"))],
              "limitations": LIMITATIONS, "final_summary": None, "midpoint_summary": None, "midpoint_status": "not_known"}
    try:
        campaign = read_json(root / "campaign.json")
        result["campaign_sha256"] = file_hash(root / "campaign.json")
        result["campaign_status"] = campaign["status"]
        require(campaign["worlds"] == list(WORLDS) and campaign["model_seeds"] == list(SEEDS) and campaign["modes"] == list(MODES) and campaign["steps"] == STEPS and campaign["planned_fits"] == 12, "campaign matrix differs from fixed protocol")
        sources = source_identity(campaign["source_sha256"])
        require(set(sources) == set(SOURCE_NAMES), "campaign did not pin the required sources")
        for name, digest in sources.items():
            require(file_hash(Path(source_root) / name) == digest, f"current audit source differs from pinned campaign: {name}; supply matching --source-root")
        result["source_sha256"] = sources
        admission = campaign.get("admission", {})
        if admission.get("admitted") is not True:
            result["status"] = "not_admitted" if campaign["status"] == "not_admitted_runtime" else "incomplete"
            errors.append("Twelve-fit campaign was not admitted; smoke outputs are not scientific endpoints.")
            return result
        midpoint = validate_admission(campaign)
        result["midpoint_status"] = "pending_all_twelve" if midpoint else "omitted_prospectively_for_all_twelve"
        pinned_data = {}
        for path, digest in campaign["data_sha256"].items():
            parts = Path(path).parts
            require(len(parts) >= 2 and parts[-2] in WORLDS and parts[-1] in ("train.npz", "eval.npz"), "unregistered campaign dataset")
            key = (parts[-2], Path(parts[-1]).stem)
            require(key not in pinned_data, "duplicate pinned dataset")
            pinned_data[key] = (path, digest)
        require(len(pinned_data) == 4, "campaign did not pin all four archives")
        datasets = {}
        for world in WORLDS:
            datasets[world] = {}
            for split in ("train", "eval"):
                original, digest = pinned_data[world, split]
                path = Path(data_root) / world / (split + ".npz") if data_root else Path(original)
                item = inspect_archive(path, world, split)
                require(item["sha256"] == digest, f"archive differs from campaign pin: {world}/{split}")
                datasets[world][split] = item
            train_meta, eval_meta = (datasets[world][split]["metadata"] for split in ("train", "eval"))
            require(train_meta["seed"] != eval_meta["seed"], "train/eval collection seeds are identical")
            for key in ("env", "config", "spec", "policy", "num_actions"):
                require(train_meta[key] == eval_meta[key], f"train/eval {key} mismatch")
        result["data_sha256"] = {world: {split: datasets[world][split]["sha256"] for split in ("train", "eval")} for world in WORLDS}
        for world in WORLDS:
            for seed in SEEDS:
                for mode in MODES:
                    directory = root / world / f"m{seed}" / mode
                    key = {"world": world, "model_seed": seed, "mode": mode, "directory": str(directory)}
                    if not (directory / "manifest.json").is_file():
                        result["missing_runs"].append(key); continue
                    try:
                        rows.append(load_run(directory, world, seed, mode, datasets[world], sources, midpoint))
                    except (ValueError, KeyError, OSError, TypeError, EOFError, zipfile.BadZipFile) as error:
                        invalid.append({**key, "reason": str(error)})
        errors.extend(pairing_errors(rows, midpoint))
        if campaign["status"] != "complete":
            errors.append("Campaign status is not complete; final publication requires the full registered queue.")
        if len(rows) == 12 and not invalid and not errors and not result["missing_runs"]:
            result.update(status="complete", publication_eligible=True, final_summary=endpoint_summary(rows, "final"))
            if midpoint:
                result["midpoint_summary"] = endpoint_summary(rows, "midpoint")
                result["midpoint_status"] = "complete_all_twelve_from_8500_schedule"
        for world in WORLDS:
            world_rows = [row for row in rows if row["world"] == world and row["sensitivity"] is not None]
            identities = [row["sensitivity"]["sample_identity"] for row in world_rows]
            if identities and any(value != identities[0] for value in identities[1:]):
                for row in world_rows:
                    row["sensitivity"] = None
                    row["sensitivity_validation_error"] = "sensitivity contexts differ across models in this world; diagnostic omitted for all affected models"
        result["sensitivity_status"] = "complete_all_twelve" if len(rows) == 12 and all(row["sensitivity"] is not None for row in rows) else "missing_or_partial_descriptive_only"
    except (ValueError, KeyError, OSError, TypeError, EOFError, zipfile.BadZipFile) as error:
        errors.append(str(error))
    return result


def render_table(result):
    lines = ["# Target-gradient ablation", "", f"Status: **{result['status']}**; publication eligible: **{result['publication_eligible']}**.",
             "", "Each row is one fitted model. Errors are mean absolute error per coordinate in pixels; lower is better. Ridge is secondary.", "",
             "| World | Model seed | Mode | Final MLP MAE | Final ridge MAE | Train-mean MAE | Eval effective rank |", "|---|---:|---|---:|---:|---:|---:|"]
    for row in result["validated_runs"]:
        final = row["final"]; m = final["metrics"]["eval"]
        lines.append(f"| {row['world']} | {row['model_seed']} | {row['mode']} | {m['mlp']['mae_px']:.4f} | {m['ridge']['mae_px']:.4f} | {m['train_mean']['mae_px']:.4f} | {final['representation_eval']['effective_rank']:.2f} |")
    if result["final_summary"] is not None:
        lines += ["", "Primary paired MLP interaction. Benefit = joint MAE minus stop-target MAE; interaction = wind benefit minus still benefit.", "",
                  "| Model seed | Still benefit (px) | Wind benefit (px) | Interaction (px) |", "|---:|---:|---:|---:|"]
        for row in result["final_summary"]["mlp"]["paired_seed_rows"]:
            lines.append(f"| {row['model_seed']} | {row['still_benefit_px']:+.4f} | {row['wind_benefit_px']:+.4f} | {row['interaction_px']:+.4f} |")
        for name in ("mlp", "ridge"):
            s = result["final_summary"][name]["interaction"]
            lines += ["", f"{'Primary MLP' if name == 'mlp' else 'Secondary ridge'} interaction: {s['mean']:+.4f} px; seed SD {s['sample_sd']:.4f}; range [{s['min']:+.4f}, {s['max']:+.4f}]."]
    else:
        lines += ["", "The fixed twelve-fit comparison is incomplete or invalid. No publication contrast is calculated; any rows above are individually audited provisional diagnostics."]
    lines += ["", f"Midpoint status: {result['midpoint_status']}.", "", *["- " + limitation for limitation in LIMITATIONS], ""]
    return "\n".join(lines)


def render_conclusions(result):
    lines = [f"Analysis status: {result['status']}; {len(result['validated_runs'])}/12 individually validated fits."]
    if result["publication_eligible"]:
        stats = result["final_summary"]["mlp"]["interaction"]
        lines.append(f"The fixed final MLP interaction is {stats['mean']:+.6f} pixels (paired model-seed SD {stats['sample_sd']:.6f}; range {stats['min']:+.6f} to {stats['max']:+.6f}).")
        lines.append("Paired seed interactions: " + ", ".join(f"m{row['model_seed']}={row['interaction_px']:+.6f}" for row in result["final_summary"]["mlp"]["paired_seed_rows"]) + ".")
        primary = result["final_summary"]["mlp"]
        for world in ("still", "wind"):
            benefit = primary["benefit"][world]
            lines.append(f"{world.capitalize()} joint-minus-stop-target MAE: mean {benefit['mean']:+.6f} pixels; paired seed values " + ", ".join(f"{v:+.6f}" for v in benefit["values"]) + ".")
        wind_benefit = primary["benefit"]["wind"]["mean"]
        if stats["mean"] > 0 and wind_benefit <= 0:
            lines.append("The positive relative interaction does not show a wind-world improvement: stop-target has equal or higher mean wind MAE. Do not describe this as rescue.")
        elif wind_benefit > 0:
            lines.append("Stop-target lowers mean wind-world MLP error under this fixed protocol. Report the individual seeds and still-world effect; this is a readout improvement, without a control or unique-mechanism claim.")
        else:
            lines.append("Stop-target does not lower mean wind-world MLP error under this fixed protocol; null or harmful effects are retained as the result.")
        lines.append("The interaction is a relative effect, never an automatic rescue or control claim.")
    else:
        lines.append("Do not use this incomplete or invalid matrix as the registered publication endpoint. No partial-seed or favorable-readout aggregate substitutes for it.")
    lines.extend(LIMITATIONS)
    lines.extend("Validation issue: " + error for error in result["validation_errors"])
    lines.extend("Invalid run: " + row["directory"] + ": " + row["reason"] for row in result["invalid_runs"])
    lines.extend("Missing run: " + row["directory"] for row in result["missing_runs"])
    lines.extend("Optional sensitivity invalid and omitted: " + row["directory"] + ": " + row["sensitivity_validation_error"]
                 for row in result["validated_runs"] if row.get("sensitivity_validation_error"))
    return "\n".join(lines) + "\n"


def save_outputs(result, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    raw = {}; blocks = []
    for row in result["validated_runs"]:
        losses = row.pop("_losses")
        stem = f"{row['world']}_m{row['model_seed']}_{row['mode']}"
        raw[stem] = losses
        for start in range(0, STEPS, 250):
            block = losses[start:start+250]
            blocks.append({"world": row["world"], "model_seed": row["model_seed"], "mode": row["mode"],
                           "first_update": start+1, "last_update": start+len(block), "prediction_mse_mean": float(block[:, 0].mean()),
                           "sigreg_mean": float(block[:, 1].mean()), "total_loss_mean": float((block[:, 0] + .09*block[:, 1]).mean())})
    np.savez_compressed(output / "training_losses.npz", **raw)
    with (output / "training_loss_trajectories.csv").open("w", newline="") as handle:
        fields = ["world", "model_seed", "mode", "first_update", "last_update", "prediction_mse_mean", "sigreg_mean", "total_loss_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(blocks)
    paired = []
    for label in ("final", "midpoint"):
        if result[label + "_summary"] is not None:
            for name in ("mlp", "ridge"):
                paired += [{"checkpoint": label, "readout": name, **row} for row in result[label + "_summary"][name]["paired_seed_rows"]]
    with (output / "paired_seed_rows.csv").open("w", newline="") as handle:
        fields = ["checkpoint", "readout", "model_seed", "still_joint_mae_px", "still_stop_target_mae_px", "still_benefit_px",
                  "wind_joint_mae_px", "wind_stop_target_mae_px", "wind_benefit_px", "interaction_px"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(paired)
    result["analysis_outputs"] = {name: file_hash(output / name) for name in ("training_losses.npz", "training_loss_trajectories.csv", "paired_seed_rows.csv")}
    result["analysis_source_sha256"] = file_hash(__file__)
    write_json(output / "publication.json", result)
    (output / "table.md").write_text(render_table(result))
    (output / "conclusions.txt").write_text(render_conclusions(result))


def self_test():
    """Hand-computable fixtures; never relax production budgets for real runs."""
    y = np.array([[0, 2], [1, 2]], np.float32)
    p = np.array([[1, 4], [3, 6]], np.float32)
    metrics = recompute_metrics(y, p)
    require(metrics["mae_px"] == 2.25 and metrics["l1_px"] == 4.5, "per-coordinate MAE confused with per-frame L1")
    close(metrics["rmse_px"], 2.5, "known coordinate RMSE")
    require(metrics["r2_xy"][1] is None, "constant target coordinate must have undefined R2")
    array = np.arange(6, dtype=np.int32).reshape(3, 2)
    train_expected = hashlib.sha256(b'{"dtype": "int32", "shape": [3, 2]}' + array.tobytes()).hexdigest()
    probe_expected = hashlib.sha256(b'<i4[3, 2]' + array.tobytes()).hexdigest()
    require(training_array_hash(array) == train_expected and probe_array_hash(array) == probe_expected and train_expected != probe_expected, "training/probe hash schemes conflated")
    scalar = np.array(3., np.float32)
    require(training_array_hash(scalar) == hashlib.sha256(b'{"dtype": "float32", "shape": []}' + scalar.tobytes()).hexdigest(),
            "training scalar shape changed during hashing")
    campaign = {"midpoint_skip_requested": False, "admission": {"admitted": True, "projected_fit_seconds": {"joint": 600., "stop_target": 700.},
                "readout_seconds": 100., "remaining_seconds": 30000., "midpoint_all_models": True, "projected_campaign_seconds_with_margin": 17280.}}
    require(validate_admission(campaign) is True, "valid midpoint timing admission rejected")
    campaign["admission"]["midpoint_all_models"] = False
    try:
        validate_admission(campaign)
        raise AssertionError("inconsistent prospective midpoint omission accepted")
    except ValueError:
        pass
    ids = np.array([[0, 0], [1, 0]], np.int64)
    data = {split: {"positions": y, "indices": ids, "flat_indices": np.array([0, 1], np.int64),
                   "frame_sha256": np.array([b"a"*64, b"b"*64], dtype="S64")} for split in ("train", "eval")}
    arrays, summaries, sampling = {}, {}, {}
    for split in ("train", "eval"):
        summaries[split] = {}
        for suffix in ("positions", "indices", "flat_indices"):
            arrays[split + "_" + suffix] = data[split][suffix]
        arrays[split + "_frame_sha256"] = np.array([b"a"*64, b"b"*64], dtype="S64")
        for suffix in ("positions", "indices"):
            sampling[split + "_" + suffix + "_sha256"] = probe_array_hash(data[split][suffix])
        for name in ("mlp", "ridge", "train_mean"):
            prediction = np.broadcast_to(y.astype(np.float64).mean(0), y.shape) if name == "train_mean" else p
            arrays[f"{split}_{name}_prediction_px"] = prediction
            arrays[f"{split}_{name}_error_px"] = prediction-y
            summaries[split][name] = recompute_metrics(y, prediction)
    with tempfile.TemporaryDirectory(prefix="target-gradient-analysis-test-") as tmp:
        path = Path(tmp) / "per_frame.npz"; np.savez(path, **arrays)
        with np.load(path) as fixture:
            validate_probe_arrays(fixture, summaries, sampling, data)
            bad = json.loads(json.dumps(summaries)); bad["eval"]["mlp"]["mae_px"] *= 2
            try:
                validate_probe_arrays(fixture, bad, sampling, data)
                raise AssertionError("summed-coordinate MAE accepted as primary metric")
            except ValueError:
                pass
            wrong_data = {split: dict(value) for split, value in data.items()}
            wrong_data["eval"]["positions"] = y + 1
            try:
                validate_probe_arrays(fixture, summaries, sampling, wrong_data)
                raise AssertionError("wrong labels accepted")
            except ValueError:
                pass
        missing = analyze(Path(tmp) / "missing_campaign")
        require(missing["status"] == "incomplete" and not missing["publication_eligible"] and missing["final_summary"] is None, "missing campaign became a result")
        smoke = Path(tmp) / "smoke500"; smoke.mkdir()
        write_json(smoke / "manifest.json", {"protocol": "paired_target_gradient_v1", "status": "complete", "completed_steps": 500, "cfg": {"steps": 500}})
        try:
            load_run(smoke, WORLDS[0], 0, "joint", {}, {}, False)
            raise AssertionError("500-step runtime smoke admitted as an endpoint")
        except ValueError:
            pass
    rows = []
    for world in WORLDS:
        for seed in SEEDS:
            for mode in MODES:
                # Benefits: still 1, wind 2+seed. Interactions: 1,2,3.
                value = 5. if mode == "stop_target" else 6. if world == WORLDS[0] else 7.+seed
                final = {"metrics": {"eval": {name: {"mae_px": value} for name in ("mlp", "ridge")}}, "sample_identity": "same", "protocol_signature": "fixed"}
                rows.append(dict(world=world, model_seed=seed, mode=mode, cfg=expected_config(seed), params=100,
                                 data_sha256=world, initial_weights_sha256=str(seed), window_table_sha256=world,
                                 update_keys_sha256=str(seed), sampled_window_ids_sha256=world+str(seed), runtime_signature="fixed",
                                 final=final, sensitivity=None))
    require(not pairing_errors(rows, False), "valid pairing fixture rejected")
    summary = endpoint_summary(rows, "final")
    require(summary["mlp"]["interaction"]["values"] == [1., 2., 3.] and summary["mlp"]["interaction"]["mean"] == 2., "interaction sign or seed pairing wrong")
    try:
        endpoint_summary(rows[:-1], "final")
        raise AssertionError("eleven fits accepted as complete")
    except ValueError:
        pass
    rows[1]["sampled_window_ids_sha256"] = "mismatch"
    require(bool(pairing_errors(rows, False)), "mismatched paired training inputs accepted")
    relative_only = json.loads(json.dumps(summary))
    for row in relative_only["mlp"]["paired_seed_rows"]:
        row.update(still_benefit_px=-1., wind_benefit_px=0., interaction_px=1.)
    relative_only["mlp"]["interaction"] = summarize([1., 1., 1.])
    relative_only["mlp"]["benefit"] = {"still": summarize([-1., -1., -1.]), "wind": summarize([0., 0., 0.])}
    conclusion = render_conclusions({"status": "complete", "validated_runs": rows, "publication_eligible": True,
                                     "final_summary": relative_only, "validation_errors": [], "invalid_runs": [], "missing_runs": []})
    require("does not show a wind-world improvement" in conclusion and "Do not describe this as rescue" in conclusion,
            "positive interaction wrongly interpreted as improved wind performance")
    print("PASS: MAE units, raw metric/error recomputation, missing artifacts, smoke exclusion, all-twelve requirement, paired interaction sign, mismatched inputs, separate hash conventions, positive-relative-effect interpretation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/revision/target_gradient"))
    parser.add_argument("--data-root", type=Path, help="Optional local remapping containing WORLD/train.npz and WORLD/eval.npz")
    parser.add_argument("--source-root", type=Path, default=REPO, help="Exact frozen source checkout; content hashes must match campaign pins")
    parser.add_argument("--output-dir", type=Path, default=Path("output/revision/target-gradient-analysis"))
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    result = analyze(args.runs_root, args.data_root, args.source_root)
    save_outputs(result, args.output_dir)
    print(json.dumps({"status": result["status"], "publication_eligible": result["publication_eligible"], "validated_fits": len(result["validated_runs"]), "output_dir": str(args.output_dir)}))
    if args.require_complete and not result["publication_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
