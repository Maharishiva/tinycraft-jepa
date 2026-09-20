"""Fixed, read-only execution check after the complete target-gradient analysis.

For every registered model and every admitted checkpoint, independently encode
the first original 256-frame probe batch and check rows 0:32 of both train/eval
samples through serialized ridge and MLP readouts. No fitting or score selection.

python -m studies.revision.verify_target_gradient_execution --self-test
python -m studies.revision.verify_target_gradient_execution \
  --analysis ANALYSIS/publication.json --runs-root CAMPAIGN \
  --data-root /dev/shm/redcube --output execution_verification.json

The self-test uses NumPy only. The real check imports JAX and uses its selected
device; run it only after all twelve fits and admitted readouts have completed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys
import time

import numpy as np

from studies.revision import analyze_target_gradient as A


CHECKED_ROWS = 32
ENCODE_BATCH = 256
MLP_BATCH = 1024
TOLERANCES = {
    "ridge": {"atol_px": 1e-6, "rtol": 1e-7},
    "mlp": {"atol_px": 2e-4, "rtol": 2e-6},
}
SCOPE = [
    "Exactly rows 0:32 of each registered training/evaluation probe sample are checked for every admitted checkpoint; no selection uses errors or scores.",
    "The encoder sees the original first 256 sampled frames with the original [1,256,63,63,3] layout, preserving the BF16 compute batch shape.",
    "The independent MLP uses saved FP32 kernels/biases, approximate JAX GELU, a 1024-row padded batch and the fixed factor 63; the padding is discarded.",
    "Fixed subpixel tolerances allow FP32/FP64 reduction roundoff; they do not excuse arbitrary BF16 kernel/backend changes. No tolerance is widened after a failure.",
    "A pass checks execution on the fixed subset. It does not reproduce training, verify all saved feature bytes, certify every prediction or establish a control result.",
    "No position, reward, action, terminal or other simulator label is loaded or used by this execution check.",
]


def check_prediction(actual, saved, name):
    actual, saved = np.asarray(actual), np.asarray(saved)
    A.require(actual.shape == saved.shape == (CHECKED_ROWS, 2), f"{name}: expected exactly 32 two-coordinate predictions")
    A.require(np.isfinite(actual).all() and np.isfinite(saved).all(), f"{name}: nonfinite execution prediction")
    difference = actual.astype(np.float64) - saved.astype(np.float64)
    tolerance = TOLERANCES[name]
    allowed = tolerance["atol_px"] + tolerance["rtol"] * np.abs(saved.astype(np.float64))
    passing = np.abs(difference) <= allowed
    return {"passed": bool(passing.all()), "checked_frames": CHECKED_ROWS,
            "absolute_tolerance_px": tolerance["atol_px"], "relative_tolerance": tolerance["rtol"],
            "execution_max_abs_difference_px": float(np.abs(difference).max()),
            "execution_rms_difference_px": float(np.sqrt(np.square(difference).mean())),
            "max_tolerance_fraction": float((np.abs(difference) / allowed).max()),
            "coordinates_outside_tolerance": int((~passing).sum()), "signed_difference_px": difference.tolist(),
            "saved_prediction_sha256": A.probe_array_hash(saved), "reexecuted_prediction_sha256": A.probe_array_hash(actual)}


def independent_ridge(features, weights, intercept):
    A.require(features.shape == (CHECKED_ROWS, 512), "ridge feature shape changed")
    A.require(weights.shape == (512, 2) and intercept.shape == (2,), "saved ridge shape changed")
    A.require(weights.dtype == np.float64 and intercept.dtype == np.float64, "saved ridge dtype changed")
    return features.astype(np.float64) @ weights + intercept


def pad_mlp_features(features):
    A.require(features.shape == (CHECKED_ROWS, 512) and features.dtype == np.float32, "MLP features must be raw FP32 [32,512]")
    return np.concatenate([features, np.repeat(features[-1:], MLP_BATCH - CHECKED_ROWS, axis=0)])


def checkpoint_cases(analysis):
    A.require(analysis.get("status") == "complete" and analysis.get("publication_eligible") is True, "execution check requires a completed, validated twelve-fit analysis")
    A.require(analysis["fixed_matrix"] == {"worlds": list(A.WORLDS), "model_seeds": list(A.SEEDS), "modes": list(A.MODES), "steps": 8500, "fits": 12}, "analysis matrix differs from registered design")
    rows = analysis["validated_runs"]
    by_key = {(row["world"], row["model_seed"], row["mode"]): row for row in rows}
    expected = {(world, seed, mode) for world in A.WORLDS for seed in A.SEEDS for mode in A.MODES}
    A.require(len(rows) == len(by_key) == 12 and set(by_key) == expected, "analysis is missing/duplicating fixed model fits")
    midpoint = analysis["midpoint_status"] == "complete_all_twelve_from_8500_schedule"
    A.require(midpoint or analysis["midpoint_status"] == "omitted_prospectively_for_all_twelve", "midpoint analysis is incomplete")
    labels = (("final", 8500), ("midpoint", 4250)) if midpoint else (("final", 8500),)
    return [(by_key[world, seed, mode], label, step) for world in A.WORLDS for seed in A.SEEDS for mode in A.MODES for label, step in labels]


def complete_failure_records(report, cases, reason):
    """Keep every prescribed case visible even when shared input checks fail."""
    present = {(item["world"], item["model_seed"], item["mode"], item["checkpoint"]) for item in report["cases"]}
    for row, label, step in cases:
        key = (row["world"], row["model_seed"], row["mode"], label)
        if key not in present:
            report["cases"].append({"world": row["world"], "model_seed": row["model_seed"], "mode": row["mode"],
                                    "checkpoint": label, "completed_steps": step, "passed": False,
                                    "execution_performed": False, "error": reason})
            present.add(key)


def load_world_data(world, campaign, analysis, data_root):
    cached = {}
    for split in ("train", "eval"):
        pinned = [(path, digest) for path, digest in campaign["data_sha256"].items()
                  if Path(path).parts[-2:] == (world, split + ".npz")]
        A.require(len(pinned) == 1, f"ambiguous campaign archive: {world}/{split}")
        original, expected_hash = pinned[0]
        path = Path(data_root) / world / (split + ".npz") if data_root is not None else Path(original)
        before_stat = path.stat()
        digest = A.file_hash(path)
        A.require(digest == expected_hash == analysis["data_sha256"][world][split], f"archive identity changed: {world}/{split}")
        with np.load(path, allow_pickle=False) as archive:
            # Deliberately do not access pos/reward/action/done or hidden state.
            frames = archive["frames"]
            metadata = A.json.loads(str(archive["meta"]))
        A.require(frames.ndim == 5 and frames.shape[-3:] == (63, 63, 3) and frames.dtype == np.uint8, "invalid image archive")
        A.require(metadata["config"] == world and metadata["split"] == split, "archive world/split metadata changed")
        count, seed = (50000, 19417) if split == "train" else (10000, 19418)
        flat, indices = A.expected_samples(frames.shape[:2], count, seed)
        cached[split] = {"path": path, "frames": frames, "metadata": metadata, "sha256": digest,
                         "flat_indices": flat, "indices": indices, "stat": (before_stat.st_size, before_stat.st_mtime_ns)}
    return cached


def state_identity(model, nnx, jax):
    """Independently hash current Params/BatchStats in checkpoint leaf order."""
    digest = hashlib.sha256()
    counts = {}
    for kind, state_type in (("p", nnx.Param), ("b", nnx.BatchStat)):
        leaves = jax.tree.leaves(nnx.state(model, state_type))
        counts[kind] = len(leaves)
        for index, leaf in enumerate(leaves):
            array = np.asarray(jax.device_get(leaf))
            A.require(np.isfinite(array).all(), f"nonfinite restored model state {kind}.{index}")
            digest.update(f"{kind}.{index}:".encode())
            digest.update(A.training_array_hash(array).encode())
    return {"weights_sha256": digest.hexdigest(), "leaf_counts": counts}


def check_model_protocol(directory, row, label, step, sources, cached):
    checkpoint = directory / ("model.npz" if step == 8500 else "model_step004250.npz")
    readout = directory / ("readout_" + label)
    A.require(A.file_hash(directory / "manifest.json") == row["artifact_sha256"]["manifest.json"], "training manifest changed after analysis")
    records = [record for record in row["checkpoints"] if record["step"] == step]
    A.require(len(records) == 1 and checkpoint.name == records[0]["path"], "checkpoint identity is ambiguous")
    checkpoint_hash = A.file_hash(checkpoint)
    A.require(checkpoint_hash == records[0]["sha256"], "checkpoint file changed after analysis")
    artifact_hashes = row[label]["artifacts"]
    for name in ("protocol.json", "per_frame.npz", "ridge_readout.npz", "mlp_readout.npz", "results.json"):
        A.require(A.file_hash(readout / name) == artifact_hashes[name], f"{label}/{name} changed after analysis")
    protocol = A.read_json(readout / "protocol.json")
    A.require(protocol["status"] == "complete" and protocol["protocol"] == "target_gradient_readouts_v1", "readout protocol incomplete")
    A.require(A.probe_signature(protocol) == row[label]["protocol_signature"], "readout settings differ from audited protocol")
    A.require(A.source_identity(protocol["runtime_source_sha256"]) == {name: sources[name] for name in A.PROBE_SOURCES}, "probe source identity mismatch")
    A.require(protocol["sha256"]["model"] == checkpoint_hash, "probe checkpoint identity mismatch")
    for split in ("train", "eval"):
        A.require(protocol["sha256"][split] == cached[split]["sha256"] and protocol[split + "_metadata"] == cached[split]["metadata"], "probe dataset identity mismatch")
    return checkpoint, readout, protocol, records[0]


def verify_case(directory, row, label, step, cached, sources, T, nnx, jax, jnp, encode):
    started = time.time()
    checkpoint, readout, protocol, record = check_model_protocol(directory, row, label, step, sources, cached)
    bundle, metadata = T.load_model(checkpoint)
    model = bundle.model
    A.require(bundle.decoder is None and bundle.heads is None and model.latent_dim == 512, "unexpected model bundle or latent dimension")
    A.require(metadata == protocol["model_metadata"] and metadata["cfg"] == A.expected_config(row["model_seed"]) and
              metadata["completed_steps"] == step and metadata["config_label"] == row["world"] and metadata["target_gradient_mode"] == row["mode"], "restored checkpoint metadata differs from registered identity")
    before = state_identity(model, nnx, jax)
    A.require(before["weights_sha256"] == record["weights_sha256"], "restored parameter/stat shapes, dtypes or bytes differ from checkpoint identity")
    runtime = row["runtime_signature"]
    for name, value in runtime["packages"].items():
        A.require(importlib.metadata.version(name) == value == protocol["packages"][name], f"runtime package differs from fitted/readout artifacts: {name}")
    A.require(jax.config.jax_default_matmul_precision == runtime["jax_default_matmul_precision"] and
              jax.config.jax_enable_x64 == runtime["jax_enable_x64"], "JAX precision setting differs from audited training context")

    with np.load(readout / "ridge_readout.npz", allow_pickle=False) as archive:
        ridge_weights, ridge_intercept = archive["weights"], archive["intercept"]
    with np.load(readout / "mlp_readout.npz", allow_pickle=False) as archive:
        params = []
        for index, shape in enumerate(((512, 256), (256, 256), (256, 2))):
            kernel, bias = archive[f"layer_{index}_kernel"], archive[f"layer_{index}_bias"]
            A.require(kernel.shape == shape and bias.shape == (shape[1],) and kernel.dtype == bias.dtype == np.float32, "MLP serialized parameter shape/dtype mismatch")
            params.append((jnp.asarray(kernel), jnp.asarray(bias)))

    @jax.jit
    def independently_predict_mlp(features):
        value = features
        for kernel, bias in params[:-1]:
            value = jax.nn.gelu(value @ kernel + bias, approximate=True)
        kernel, bias = params[-1]
        return (value @ kernel + bias) * 63.0

    report = {"world": row["world"], "model_seed": row["model_seed"], "mode": row["mode"], "checkpoint": label,
              "completed_steps": step, "checkpoint_sha256": record["sha256"], "state_before": before, "splits": {},
              "probe_protocol_sha256": row[label]["artifacts"]["protocol.json"], "source_sha256": sources,
              "producer_devices": protocol["devices"], "verifier_devices": [str(device) for device in jax.devices()]}
    with np.load(readout / "per_frame.npz", allow_pickle=False) as archive:
        for split in ("train", "eval"):
            ids, flat = archive[split + "_indices"], archive[split + "_flat_indices"]
            A.require(np.array_equal(ids, cached[split]["indices"]) and np.array_equal(flat, cached[split]["flat_indices"]), "sample IDs differ from the complete registered draw")
            A.require(A.probe_array_hash(ids) == protocol["sampling"][split + "_indices_sha256"], "probe sample hash mismatch")
            sample_ids = ids[:ENCODE_BATCH]
            selected = cached[split]["frames"][sample_ids[:, 0], sample_ids[:, 1]]
            content_hashes = np.asarray([hashlib.sha256(memoryview(np.ascontiguousarray(frame)).cast("B")).hexdigest() for frame in selected], dtype="S64")
            A.require(np.array_equal(content_hashes, archive[split + "_frame_sha256"][:ENCODE_BATCH]), "original encoder-batch pixels differ from saved frame identities")
            raw_features = np.asarray(jax.device_get(encode(model, jnp.asarray(selected))))[:CHECKED_ROWS]
            A.require(raw_features.shape == (CHECKED_ROWS, 512) and raw_features.dtype == np.float32 and np.isfinite(raw_features).all(), "encoder returned invalid raw features")
            predictions = {"ridge": independent_ridge(raw_features, ridge_weights, ridge_intercept),
                           "mlp": np.asarray(jax.device_get(independently_predict_mlp(jnp.asarray(pad_mlp_features(raw_features)))))[:CHECKED_ROWS]}
            split_report = {"checked_probe_rows": list(range(CHECKED_ROWS)), "checked_frame_indices": ids[:CHECKED_ROWS].tolist(),
                            "dataset_sha256": cached[split]["sha256"], "encoded_batch_size": ENCODE_BATCH,
                            "raw_checked_features_sha256": A.probe_array_hash(raw_features),
                            "checked_frame_content_sha256": A.probe_array_hash(content_hashes[:CHECKED_ROWS]), "readouts": {}}
            for name in ("ridge", "mlp"):
                saved = archive[f"{split}_{name}_prediction_px"][:CHECKED_ROWS]
                split_report["readouts"][name] = check_prediction(predictions[name], saved, name)
            report["splits"][split] = split_report
    after = state_identity(model, nnx, jax)
    report.update(state_after=after, frozen_state_unchanged=before == after)
    A.require(A.file_hash(checkpoint) == record["sha256"], "checkpoint file changed during execution check")
    for name, digest in row[label]["artifacts"].items():
        A.require(A.file_hash(readout / name) == digest, f"probe artifact changed during execution check: {name}")
    report["passed"] = before == after and all(value["passed"] for split in report["splits"].values() for value in split["readouts"].values())
    report["seconds"] = round(time.time() - started, 3)
    del model, bundle, params
    return report


def verify(analysis_path, runs_root=None, data_root=None, source_root=A.REPO, output=None):
    started = time.time()
    cases = []
    report = {"schema_version": 1, "protocol": "target_gradient_execution_subset_v1", "status": "failed", "passed": False,
              "checked_rows_per_split": CHECKED_ROWS, "encode_batch": ENCODE_BATCH, "mlp_batch": MLP_BATCH,
              "tolerances": TOLERANCES, "scope": SCOPE, "cases": [], "errors": [], "verifier_source_sha256": A.file_hash(__file__)}
    try:
        analysis = A.read_json(analysis_path)
        report["analysis_sha256"] = A.file_hash(analysis_path)
        cases = checkpoint_cases(analysis)
        report["planned_cases"] = len(cases)
        root = Path(runs_root) if runs_root is not None else Path(analysis["runs_root"])
        campaign = A.read_json(root / "campaign.json")
        A.require(campaign["status"] == "complete" and A.file_hash(root / "campaign.json") == analysis["campaign_sha256"], "campaign changed or is incomplete")
        midpoint = A.validate_admission(campaign)
        A.require(len(cases) == (24 if midpoint else 12), "admitted checkpoint set differs from analyzed set")
        sources = A.source_identity(campaign["source_sha256"])
        A.require(sources == analysis["source_sha256"], "analysis/campaign source identities differ")
        for name, digest in sources.items():
            A.require(A.file_hash(Path(source_root) / name) == digest, f"frozen source changed: {name}")
        A.require(A.file_hash(A.__file__) == analysis["analysis_source_sha256"], "analyzer source differs from the completed audit")
        # Only import accelerator libraries after complete-matrix/source checks.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        sys.path.insert(0, str(Path(source_root).resolve()))
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        T = importlib.import_module("train")
        A.require(Path(T.__file__).resolve() == (Path(source_root) / "train.py").resolve(), "imported train module is not from the frozen source root")
        report["runtime_packages"] = {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "numpy", "optax")}
        report["runtime_matmul_precision"] = jax.config.jax_default_matmul_precision
        report["runtime_enable_x64"] = jax.config.jax_enable_x64

        @nnx.jit
        def encode(frozen_model, frame_batch):
            return frozen_model.encode(frame_batch[None], train=False)[0]

        for world in A.WORLDS:
            try:
                cached = load_world_data(world, campaign, analysis, data_root)
            except Exception as error:
                reason = f"{world}: {type(error).__name__}: {error}"
                report["errors"].append(reason)
                complete_failure_records(report, [case for case in cases if case[0]["world"] == world], reason)
                continue
            for row, label, step in cases:
                if row["world"] != world:
                    continue
                directory = root / world / f"m{row['model_seed']}" / row["mode"]
                try:
                    item = verify_case(directory, row, label, step, cached, sources, T, nnx, jax, jnp, encode)
                except Exception as error:
                    item = {"world": world, "model_seed": row["model_seed"], "mode": row["mode"], "checkpoint": label,
                            "passed": False, "error": f"{type(error).__name__}: {error}"}
                report["cases"].append(item)
                print(f"[execution] {world}/m{row['model_seed']}/{row['mode']}/{label}: {'PASS' if item['passed'] else 'FAIL'}", flush=True)
                if output is not None:
                    A.write_json(output, report)
                gc.collect()
            for split, values in cached.items():
                try:
                    after_stat = values["path"].stat()
                    A.require((after_stat.st_size, after_stat.st_mtime_ns) == values["stat"] and A.file_hash(values["path"]) == values["sha256"],
                              "dataset changed during execution check")
                except Exception as error:
                    reason = f"{world}/{split}: post-execution dataset verification failed: {type(error).__name__}: {error}"
                    report["errors"].append(reason)
                    for item in report["cases"]:
                        if item["world"] == world:
                            item["passed"] = False
                            item.setdefault("post_execution_identity_errors", []).append(reason)
            del cached
            gc.collect()
        for name, digest in sources.items():
            A.require(A.file_hash(Path(source_root) / name) == digest, f"frozen source changed during check: {name}")
        A.require(A.file_hash(root / "campaign.json") == analysis["campaign_sha256"] and A.file_hash(analysis_path) == report["analysis_sha256"], "campaign/analysis changed during check")
        report["passed"] = len(report["cases"]) == len(cases) and not report["errors"] and all(item["passed"] for item in report["cases"])
        report["status"] = "passed" if report["passed"] else "failed"
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    complete_failure_records(report, cases, "not executed because a shared validation/runtime prerequisite failed; see report errors")
    report["seconds"] = round(time.time() - started, 3)
    if output is not None:
        A.write_json(output, report)
    return report


def self_test():
    features = np.zeros((CHECKED_ROWS, 512), np.float32)
    features[:, 0] = np.arange(CHECKED_ROWS)
    weights = np.zeros((512, 2), np.float64); weights[0] = (2., -3.)
    bias = np.array([1., 5.], np.float64)
    actual = independent_ridge(features, weights, bias)
    expected = np.column_stack([2*np.arange(CHECKED_ROWS)+1, -3*np.arange(CHECKED_ROWS)+5]).astype(np.float64)
    A.require(np.array_equal(actual, expected), "independent ridge formula wrong")
    A.require(check_prediction(actual, expected, "ridge")["passed"], "identical predictions rejected")
    changed = actual.copy(); changed[17, 1] += .01
    A.require(not check_prediction(changed, expected, "ridge")["passed"], "prediction mismatch accepted")
    padded = pad_mlp_features(features)
    A.require(padded.shape == (1024, 512) and np.array_equal(padded[:CHECKED_ROWS], features) and np.array_equal(padded[-1], features[-1]), "MLP padding changes checked raw latents")
    flat, _ = A.expected_samples((1001, 100), 50000, 19417)
    wrong, _ = A.expected_samples((1001, 100), CHECKED_ROWS, 19417)
    A.require(not np.array_equal(flat[:CHECKED_ROWS], wrong), "sample fixture failed to distinguish drawing 32 from slicing registered draw")
    analysis = {"status": "complete", "publication_eligible": True, "fixed_matrix": {"worlds": list(A.WORLDS), "model_seeds": list(A.SEEDS),
                "modes": list(A.MODES), "steps": 8500, "fits": 12}, "midpoint_status": "omitted_prospectively_for_all_twelve",
                "validated_runs": [{"world": world, "model_seed": seed, "mode": mode} for world in A.WORLDS for seed in A.SEEDS for mode in A.MODES]}
    fixed_cases = checkpoint_cases(analysis)
    A.require(len(fixed_cases) == 12, "fixed final cases incomplete")
    failure_report = {"cases": []}
    complete_failure_records(failure_report, fixed_cases[:6], "first-world input unavailable")
    complete_failure_records(failure_report, fixed_cases, "remaining case unavailable")
    A.require(len(failure_report["cases"]) == 12 and all(not row["passed"] and not row["execution_performed"] for row in failure_report["cases"]),
              "shared input failure hides or duplicates prescribed cases")
    analysis["midpoint_status"] = "complete_all_twelve_from_8500_schedule"
    A.require(len(checkpoint_cases(analysis)) == 24, "admitted midpoint cases incomplete")
    analysis["validated_runs"].pop()
    try:
        checkpoint_cases(analysis)
        raise AssertionError("partial matrix admitted for execution verification")
    except ValueError:
        pass
    print("PASS: serialized ridge arithmetic, prediction mismatch detection, unchanged checked rows under padding, full registered sampling, fixed12/24 checkpoint coverage, incomplete-matrix exclusion; no JAX/GPU imported.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, help="Completed analyzer publication.json")
    parser.add_argument("--runs-root", type=Path, help="Optional remapping of the same audited campaign")
    parser.add_argument("--data-root", type=Path, help="Optional WORLD/train.npz and WORLD/eval.npz archive remapping")
    parser.add_argument("--source-root", type=Path, default=A.REPO, help="Frozen producer source checkout")
    parser.add_argument("--output", type=Path, default=Path("output/revision/target-gradient-analysis/execution_verification.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.analysis is None:
        parser.error("--analysis is required unless --self-test")
    A.require(not args.output.exists(), "refusing to overwrite a previous execution-verification report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = verify(args.analysis, args.runs_root, args.data_root, args.source_root, args.output)
    print(A.json.dumps({"status": report["status"], "passed": report["passed"], "completed_cases": len(report["cases"]), "output": str(args.output)}))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
