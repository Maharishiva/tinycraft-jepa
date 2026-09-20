"""CPU-only identity audit of six last-only fits versus their six joint pairs.

Reads saved manifests/protocols/checkpoints only. No fitting, metric selection,
GPU imports, rewriting of run artifacts, or claims about information/control.
This check is additional to the complete shared eighteen-model history audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

WORLDS = ("gray_frame_still", "gray_frame_wind")
SEEDS = (0, 1, 2)
PAIR_FIELDS = (
    "cfg", "params", "data_sha256", "data_array_sha256", "data_shape", "data_identity",
    "initial_weights_sha256", "window_table_sha256", "update_keys_sha256", "sampled_window_ids_sha256",
    "parameter_rng_seed", "dropout_rng_seed", "update_rng_seed", "rng_implementation",
    "python", "packages", "jax_default_matmul_precision", "jax_enable_x64")
CORE_SOURCES = ("train.py", "studies/stoch/train_arms.sh", "pyproject.toml", "studies/revision/target_gradient.py")
READOUT_FIELDS = ("train_metadata", "eval_metadata", "ridge", "mlp", "label_use", "position_convention", "metric_conventions", "packages", "python")
SAMPLE_FIELDS = (
    "train_count", "eval_count", "train_seed", "eval_seed", "replacement", "eligible", "rng", "flat_index_order",
    "frame_identity", "frame_content_hash", "train_indices_sha256", "eval_indices_sha256",
    "train_positions_sha256", "eval_positions_sha256")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare_pair(last_directory, joint_directory, world, seed):
    last, joint = (read_json(p / "manifest.json") for p in (last_directory, joint_directory))
    for manifest, mode, experiment in ((last, "last_only", "last_only_objective_v1"),
                                       (joint, "joint", "paired_target_gradient_v1")):
        require(manifest["status"] == "complete" and manifest["completed_steps"] == 8500,
                f"{mode}: not a completed 8500-update model")
        require(manifest["protocol"] == experiment and manifest["mode"] == mode
                and manifest["config_label"] == world and manifest["model_seed"] == seed,
                f"{mode}: wrong experiment/world/seed identity")
    for field in PAIR_FIELDS:
        require(last[field] == joint[field], "unpaired field: " + field)
    for name in CORE_SOURCES:
        require(last["sources"]["sha256"][name] == joint["sources"]["sha256"][name],
                "unpaired core source: " + name)
    require(last["fixed_checkpoint_steps"] == [0, 500, 1000, 2000, 4000, 4250, 8500],
            "last-only checkpoint schedule changed")
    require([entry["step"] for entry in last["checkpoints"]] == last["fixed_checkpoint_steps"],
            "last-only checkpoint set is incomplete")
    for directory, manifest in ((last_directory, last), (joint_directory, joint)):
        for checkpoint in manifest["checkpoints"]:
            require(sha256(directory / checkpoint["path"]) == checkpoint["sha256"], "checkpoint hash mismatch")
        require(sha256(directory / "sampling.npz") == manifest["sampling_file_sha256"], "sampling artifact hash mismatch")
        require(sha256(directory / "losses.npz") == manifest["loss_file_sha256"], "loss artifact hash mismatch")
        require(sha256(directory / "model.npz") == manifest["final_checkpoint_sha256"], "final checkpoint mismatch")
    last_probe, joint_probe = (read_json(p / "readout_final/protocol.json") for p in (last_directory, joint_directory))
    for probe, expected, manifest in ((last_probe, "last_only_readouts_v1", last),
                                       (joint_probe, "target_gradient_readouts_v1", joint)):
        require(probe["status"] == "complete" and probe["protocol"] == expected, "readout incomplete/wrong protocol")
        require(probe["sha256"]["model"] == manifest["final_checkpoint_sha256"], "readout checkpoint differs")
        require(probe["sha256"]["train"] == manifest["data_sha256"], "readout training archive differs")
        require(probe["mlp"]["steps"] == 4000 and probe["mlp"]["widths"] == [512, 256, 256, 2],
                "readout budget/width changed")
        require(probe["packages"] == manifest["packages"] and probe["python"] == manifest["python"],
                "readout runtime differs from training runtime")
    for field in READOUT_FIELDS:
        require(last_probe[field] == joint_probe[field], "unpaired readout field: " + field)
    for field in SAMPLE_FIELDS:
        require(last_probe["sampling"][field] == joint_probe["sampling"][field], "unpaired sample field: " + field)
    for split in ("train", "eval"):
        require(last_probe["sha256"][split] == joint_probe["sha256"][split], "unpaired readout archive: " + split)
    for directory, probe in ((last_directory, last_probe), (joint_directory, joint_probe)):
        for name, expected in probe["artifacts"].items():
            require(Path(name).name == name and sha256(directory / "readout_final" / name) == expected,
                    "readout artifact differs: " + name)
    return {"passed": True, "world": world, "model_seed": seed,
            "last_only_manifest_sha256": sha256(last_directory / "manifest.json"),
            "joint_manifest_sha256": sha256(joint_directory / "manifest.json"),
            "last_only_readout_protocol_sha256": sha256(last_directory / "readout_final/protocol.json"),
            "joint_readout_protocol_sha256": sha256(joint_directory / "readout_final/protocol.json"),
            "paired_fields": list(PAIR_FIELDS), "paired_core_sources": list(CORE_SOURCES),
            "paired_readout_fields": list(READOUT_FIELDS), "paired_sample_fields": list(SAMPLE_FIELDS),
            "packages": last["packages"], "cfg": last["cfg"]}


def audit(last_root, joint_root):
    report = {"protocol": "last_only_pairing_audit_v1", "status": "incomplete", "passed": False,
              "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha256(__file__),
              "scope": "all six last-only/joint manifest, runtime, checkpoint and fixed-readout identities; no outcome selection",
              "limitations": "does not reproduce training or every saved metric; common eighteen-model evaluation and independent score recomputation remain separate",
              "cases": []}
    for world in WORLDS:
        for seed in SEEDS:
            try:
                row = compare_pair(last_root / world / f"m{seed}" / "last_only",
                                   joint_root / world / f"m{seed}" / "joint", world, seed)
            except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
                row = {"passed": False, "world": world, "model_seed": seed,
                       "error_type": type(error).__name__, "error": str(error)}
            report["cases"].append(row)
    report["passed"] = all(row["passed"] for row in report["cases"])
    report["status"] = "complete" if report["passed"] else "incomplete_or_mismatched"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last-only-root", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "refusing to overwrite pairing audit")
    result = audit(args.last_only_root, args.joint_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "cases": len(result["cases"]), "out": str(args.out)}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
