"""Guarded three-fit Craftax last-target-only transfer queue, without retries.

Run only after the original two-replica queue reports complete. This launcher
changes neither the frozen evaluator nor original scientific source files.
Admission requires at least 90 minutes before the fixed campaign deadline;
partial outputs and failed seeds are preserved, never replaced or excluded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from studies.revision import analyze_craftax_last_only as A

DEADLINE = "2026-09-20T07:41:00+00:00"
RESERVE_PER_MODEL_SECONDS = 30 * 60
TRAINER_RELATIVE = "studies/revision/craftax_last_only.py"
OBJECTIVE_RELATIVE = "studies/revision/last_only_objective.py"
CHILD = None
ACTIVE = None


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def idle_gpu(timeout=30):
    processes = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=timeout).strip()
    A.require(not processes, "GPU0 already has compute processes: " + processes)


def terminate_child(force=False):
    """Stop and reap the whole isolated process group, including descendants.

    Deadline expiry uses immediate SIGKILL, without a grace period beyond the
    cutoff. On ordinary interrupts the leader gets at most two seconds to exit.
    """
    global CHILD
    child, CHILD = CHILD, None
    if child is None:
        return
    group = child.pid
    if not force:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            child.wait()
            return
        try:
            child.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def on_signal(number, _frame):
    terminate_child()
    raise RuntimeError(f"queue interrupted by signal {number}; partial artifacts preserved")


def load_model_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        has_auxiliaries = any(key.startswith(("decoder.", "heads.")) for key in archive.files)
    return meta, has_auxiliaries


def array_sha256(value):
    """The wrapper's dtype/shape/C-order digest, checked without importing JAX."""
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": str(value.dtype), "shape": value.shape}, sort_keys=True).encode())
    digest.update(memoryview(np.ascontiguousarray(value).reshape(-1).view(np.uint8)))
    return digest.hexdigest()


def verify_fit(run, meta, reference, seed, trainer_sha256):
    fit = json.loads((run / "manifest.json").read_text())
    A.require(fit["status"] == "complete" and fit["completed_steps"] == 8500
              and fit["protocol"] == "craftax-last-only-v1", "last-only fit is incomplete")
    for key in ("cfg", "params", "data_sha256", "source_sha256", "base_train_source_sha256",
                "objective_source_sha256", "sources", "reference_report_sha256", "reference_model_sha256",
                "prediction_loss_mode", "target_gradient_mode", "initial_weights_sha256",
                "window_table_sha256", "update_keys_sha256", "sampled_window_ids_sha256"):
        A.require(fit[key] == meta[key], "fit manifest/checkpoint provenance differs: " + key)
    A.check_metadata(fit, reference, seed, True, trainer_sha256)
    A.require(fit["final_checkpoint_sha256"] == A.sha256(run / "model.npz"), "fit checkpoint hash differs")
    A.require(fit["loss_file_sha256"] == A.sha256(run / "losses.npz"), "fit loss-file hash differs")
    A.require(fit["sampling_file_sha256"] == A.sha256(run / "sampling.npz"), "fit sampling-file hash differs")
    A.require(fit["fixed_checkpoint_steps"] == [8500] and fit["sigreg_frame_count_per_window"] == 9
              and fit["encoder_frame_count_per_window"] == 9 and fit["predictor_output_count_per_window"] == 8
              and fit["sigreg_weight"] == .09, "fit recipe or checkpoint selection differs")
    with np.load(run / "losses.npz", allow_pickle=False) as archive:
        A.require(archive["history"].shape == (8500, 2) and np.all(np.isfinite(archive["history"])),
                  "expected 8500 finite training updates")
    with np.load(run / "sampling.npz", allow_pickle=False) as archive:
        A.require(set(archive.files) == {"window_table", "update_keys", "sampled_window_ids"},
                  "training sampler ledger fields differ")
        for key in archive.files:
            A.require(array_sha256(archive[key]) == meta[key + "_sha256"], "training sampler fingerprint differs: " + key)
        A.require(archive["update_keys"].shape == (8500, 2)
                  and archive["sampled_window_ids"].shape == (8500, 128), "sampler ledger update count differs")
        A.require(np.all((archive["sampled_window_ids"] >= 0)
                         & (archive["sampled_window_ids"] < len(archive["window_table"]))), "invalid sampled window ID")
    digest = hashlib.sha256()
    with np.load(run / "model.npz", allow_pickle=False) as archive:
        A.require(sum(archive[key].size for key in archive.files if key.startswith("model.p.")) == meta["params"],
                  "checkpoint parameter count differs from its metadata")
        for kind in ("p", "b"):
            keys = sorted((key for key in archive.files if key.startswith("model." + kind + ".")),
                          key=lambda key: int(key.rsplit(".", 1)[1]))
            for i, key in enumerate(keys):
                A.require(key == f"model.{kind}.{i}", "noncontiguous checkpoint leaf order")
                digest.update(f"{kind}.{i}:".encode())
                digest.update(array_sha256(archive[key]).encode())
    A.require(digest.hexdigest() == fit["final_weights_sha256"], "fit weights fingerprint differs")
    return fit


def verify_original_queue(queue, reference, trainer_sha256):
    """Validate prior completion and every source/model/report link, without scores."""
    status_path, plan_path = queue / "status.json", queue / "plan.json"
    status = json.loads(status_path.read_text())
    plan = json.loads(plan_path.read_text())
    A.require(status["phase"] == "complete", "original replication queue must be complete before admission")
    A.require([item["model_seed"] for item in status["completed"]] == [1, 2], "original replication must retain both registered seeds")
    A.require(plan["schema"] == "craftax-exact-seed-replication-v1" and plan["training_seeds"] == [1, 2]
              and plan["reference_training_seed"] == 0, "wrong original replication matrix")
    A.require(plan["reference_report_sha256"] == A.REFERENCE_REPORT_SHA256
              and plan["reference_model_sha256"] == A.REFERENCE_MODEL_SHA256
              and plan["data_sha256"] == A.DATA_SHA256, "original replication input identity differs")
    A.require(plan["source_sha256"] == A.EVAL_SOURCES and plan["versions"] == reference["versions"],
              "original replication sources/runtime differ")
    A.require(plan["launcher_sha256"] == A.sha256(ROOT / "studies/revision/run_craftax_replication.py"),
              "original replication launcher changed")
    immutable = {status_path: A.sha256(status_path), plan_path: A.sha256(plan_path)}
    outputs = []
    for seed, completed in zip((1, 2), status["completed"]):
        report_path = queue / f"seed{seed}" / "goal_eval.json"
        model_path = queue / f"seed{seed}" / "model.npz"
        identity_path = queue / f"seed{seed}" / "identity.json"
        report = json.loads(report_path.read_text())
        identity = json.loads(identity_path.read_text())
        A.check_report(report, reference, seed, False, trainer_sha256)
        A.require(plan["configs"][str(seed)] == report["model_metadata"]["cfg"], "original config provenance mismatch")
        A.require(Path(completed["evaluation"]).resolve() == report_path.resolve()
                  and completed["evaluation_sha256"] == A.sha256(report_path), "original evaluation completion link differs")
        A.require(completed["model_sha256"] == report["model_sha256"] == identity["sha256"] == A.sha256(model_path),
                  "original checkpoint identity mismatch")
        A.require(completed["matched_environment_episode_check"] == "passed", "original episode pairing not recorded")
        meta, has_auxiliaries = load_model_metadata(model_path)
        A.require(meta == report["model_metadata"] == identity["metadata"] and not has_auxiliaries,
                  "original replication checkpoint metadata differs or has auxiliaries")
        with np.load(model_path.parent / "losses.npz", allow_pickle=False) as losses:
            A.require(losses["history"].shape == (8500, 2) and np.all(np.isfinite(losses["history"])),
                      "original replication did not finish 8500 finite updates")
        for path in (report_path, model_path, identity_path, model_path.parent / "losses.npz"):
            immutable[path] = A.sha256(path)
        outputs.append({"model_seed": seed, "report": str(report_path.resolve()), "report_sha256": A.sha256(report_path),
                        "model": str(model_path.resolve()), "model_sha256": A.sha256(model_path)})
    return immutable, outputs


def main():
    global ACTIVE, CHILD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original-queue", type=Path, default=ROOT / "runs/revision/craftax_seed_replication")
    parser.add_argument("--trainer-sha256", required=True, help="independently reviewed wrapper SHA256")
    parser.add_argument("--gpu", choices=("0",), default="0")
    parser.add_argument("--deadline-utc", choices=(DEADLINE,), default=DEADLINE)
    args = parser.parse_args()
    args.reference_report, args.data, args.out, args.original_queue = (
        path.resolve() for path in (args.reference_report, args.data, args.out, args.original_queue))
    deadline = datetime.fromisoformat(DEADLINE).timestamp()
    A.require(not args.out.exists(), "fresh output path required; no resume, overwrite or retry")
    A.require(deadline - time.time() >= 3 * RESERVE_PER_MODEL_SECONDS,
              "fewer than 90 minutes remain; do not launch the three-model campaign")
    A.require(A.valid_sha(args.trainer_sha256), "explicit reviewed trainer SHA256 required")
    A.require(A.sha256(args.reference_report) == A.REFERENCE_REPORT_SHA256, "reference report changed")
    reference = json.loads(args.reference_report.read_text())
    ref_model = Path(reference["model"])
    A.require(A.sha256(ref_model) == reference["model_sha256"] == A.REFERENCE_MODEL_SHA256, "seed-0 checkpoint changed")
    A.require(load_model_metadata(ref_model)[0] == reference["model_metadata"], "seed-0 model metadata differs")
    A.check_report(reference, reference, 0, False, args.trainer_sha256)
    A.require(A.sha256(args.data) == A.DATA_SHA256, "training archive changed")
    for package, version in reference["versions"].items():
        A.require(importlib.metadata.version(package) == version, "runtime package differs: " + package)
    sources = {**A.EVAL_SOURCES, TRAINER_RELATIVE: args.trainer_sha256,
               OBJECTIVE_RELATIVE: A.OBJECTIVE_SOURCE_SHA256,
               "studies/revision/run_craftax_last_only.py": A.sha256(__file__),
               "studies/revision/analyze_craftax_last_only.py": A.sha256(A.__file__),
               "studies/revision/run_craftax_replication.py": A.sha256(ROOT / "studies/revision/run_craftax_replication.py")}
    immutable = {ROOT / name: digest for name, digest in sources.items()}
    immutable.update({args.reference_report: A.REFERENCE_REPORT_SHA256, ref_model: A.REFERENCE_MODEL_SHA256})
    for path, digest in immutable.items():
        A.require(A.sha256(path) == digest, "source/input identity mismatch: " + str(path))
    original_immutable, original_outputs = verify_original_queue(args.original_queue, reference, args.trainer_sha256)
    immutable.update(original_immutable)
    # This lock is shared with the other revision campaign launchers. nvidia-smi
    # additionally rejects unrelated compute processes before every child.
    lock = open("/tmp/tinycraft-jepa-gpu0.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    idle_gpu()
    A.require(deadline - time.time() >= 3 * RESERVE_PER_MODEL_SECONDS, "input checks consumed admission reserve")
    args.out.mkdir(parents=True, exist_ok=False)
    logs = args.out / "logs"
    logs.mkdir()
    plan = {"schema": "craftax-last-only-queue-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "deadline_utc": DEADLINE, "gpu": "0", "training_seeds": list(A.MODEL_SEEDS),
            "reference_report": str(args.reference_report), "reference_report_sha256": A.REFERENCE_REPORT_SHA256,
            "reference_model": str(ref_model), "reference_model_sha256": A.REFERENCE_MODEL_SHA256,
            "data": str(args.data), "data_sha256": A.DATA_SHA256, "source_sha256": sources,
            "versions": reference["versions"], "original_queue": str(args.original_queue),
            "original_completed_outputs": original_outputs, "original_status_sha256": A.sha256(args.original_queue / "status.json"),
            "configs": {str(seed): {**reference["model_metadata"]["cfg"], "seed": seed} for seed in A.MODEL_SEEDS},
            "evaluation_arguments": A.FIXED_ARGS, "admission_reserve_seconds": 3 * RESERVE_PER_MODEL_SECONDS,
            "runtime_estimate_seconds": 70 * 60, "admission_basis": "correctness, original queue completion, exclusive GPU and time only; independent of outcome direction",
            "selection": "All three predeclared last-only seeds retained; no retries, replacement, early stopping or exclusion.",
            "sole_loss_change": "coordinate-mean MSE on the last prediction/target; full forward, joint target gradients, all-frame SIGReg and coefficient unchanged",
            "gpu_identity": subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid,name,driver_version", "--format=csv"], text=True, timeout=30)}
    save(args.out / "plan.json", plan)
    status = {"phase": "prepared", "completed": [], "jobs": [], "started_unix": time.time(),
              "plan_sha256": A.sha256(args.out / "plan.json"), "partial_artifacts": "preserve all on failure or interrupt"}
    ACTIVE = (args.out / "status.json", status)
    save(*ACTIVE)
    immutable[args.out / "plan.json"] = status["plan_sha256"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    env.pop("JAX_PLATFORMS", None)

    def guard(check_gpu=False, full_data=False):
        A.require(time.time() < deadline, "deadline reached; preserve partial campaign")
        for path, digest in immutable.items():
            A.require(A.sha256(path) == digest, "frozen source/artifact changed: " + str(path))
        if full_data:
            A.require(A.sha256(args.data) == A.DATA_SHA256, "training archive changed during campaign")
        A.require(time.time() < deadline, "deadline reached while checking immutable artifacts")
        if check_gpu:
            idle_gpu(timeout=min(30, max(.001, deadline - time.time())))
        A.require(time.time() < deadline, "deadline reached during GPU guard")

    def execute(command, phase, seed):
        global CHILD
        guard(check_gpu=True, full_data=True)
        log = logs / f"seed{seed}_{phase}.log"
        row = {"model_seed": seed, "phase": phase, "command": command, "log": str(log),
               "status": "running", "started_unix": time.time()}
        status["jobs"].append(row)
        status.update(phase=phase, model_seed=seed)
        save(*ACTIVE)
        print(f"[{phase}] model_seed={seed}; log={log}", flush=True)
        with log.open("x") as stream:
            A.require(time.time() < deadline, "deadline reached before child start")
            CHILD = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                code = CHILD.wait(timeout=max(.001, deadline - time.time()))
            except subprocess.TimeoutExpired:
                terminate_child(force=True)
                row.update(status="deadline_timeout", elapsed_s=time.time() - row["started_unix"])
                status.update(phase="deadline_incomplete")
                save(*ACTIVE)
                raise RuntimeError("hard deadline terminated " + phase + "; partial artifacts preserved") from None
            except BaseException:
                row.update(status="interrupted", elapsed_s=time.time() - row["started_unix"])
                save(*ACTIVE)
                raise
            finally:
                terminate_child()
        row.update(status="complete" if code == 0 else "failed", returncode=code,
                   elapsed_s=time.time() - row["started_unix"])
        save(*ACTIVE)
        A.require(code == 0, f"{phase} seed {seed} failed; no retry or seed replacement")
        guard(check_gpu=True, full_data=True)

    for seed in A.MODEL_SEEDS:
        guard(check_gpu=True)
        A.require(deadline - time.time() >= (3 - seed) * RESERVE_PER_MODEL_SECONDS,
                  "insufficient conservative time reserve for all remaining registered models; preserve partials")
        run = args.out / f"seed{seed}"
        model = run / "model.npz"
        command = [sys.executable, "-m", "studies.revision.craftax_last_only", "--reference-report", str(args.reference_report),
                   "--data", str(args.data), "--seed", str(seed), "--out", str(run)]
        execute(command, "training", seed)
        meta, has_auxiliaries = load_model_metadata(model)
        A.check_metadata(meta, reference, seed, True, args.trainer_sha256)
        A.require(not has_auxiliaries, "last-only checkpoint contains decoder or heads")
        verify_fit(run, meta, reference, seed, args.trainer_sha256)
        identity = {"model_seed": seed, "model": str(model), "model_sha256": A.sha256(model), "metadata": meta,
                    "manifest_sha256": A.sha256(run / "manifest.json"), "losses_sha256": A.sha256(run / "losses.npz"),
                    "sampling_sha256": A.sha256(run / "sampling.npz"), "pairing_fingerprints": {
                        key: meta[key] for key in ("initial_weights_sha256", "window_table_sha256", "update_keys_sha256", "sampled_window_ids_sha256")}}
        save(run / "identity.json", identity)
        for path in (model, run / "manifest.json", run / "identity.json", run / "losses.npz", run / "sampling.npz"):
            immutable[path] = A.sha256(path)
        evaluation = run / "goal_eval.json"
        command = [sys.executable, str(ROOT / "studies/revision/craftax_eval.py"), "--model", str(model), "--out", str(evaluation),
                   "--seeds", "101", "202", "303", "--envs", "128", "--horizon", "6", "--warmup", "8", "--pop", "512", "--elite", "32", "--iters", "6"]
        execute(command, "evaluation", seed)
        measured = json.loads(evaluation.read_text())
        A.check_report(measured, reference, seed, True, args.trainer_sha256)
        A.require(measured["model_sha256"] == identity["model_sha256"] and measured["model_metadata"] == meta,
                  "evaluation model identity mismatch")
        immutable[evaluation] = A.sha256(evaluation)
        completed = {"model_seed": seed, "model": str(model), "model_sha256": identity["model_sha256"],
                     "evaluation": str(evaluation), "evaluation_sha256": A.sha256(evaluation),
                     "manifest_sha256": identity["manifest_sha256"], "pairing_fingerprints": identity["pairing_fingerprints"],
                     "matched_environment_episode_check": "passed; all 384 contexts and noop/random/reference rows and traces"}
        status["completed"].append(completed)
        save(*ACTIVE)
        print("[seed_complete] " + json.dumps(completed), flush=True)
    guard(check_gpu=True, full_data=True)
    status.update(phase="complete", finished_unix=time.time())
    save(*ACTIVE)
    print("[complete] all three predeclared models trained and evaluated; no outcome selection", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        main()
    except BaseException as error:
        terminate_child()
        if ACTIVE is not None:
            path, status = ACTIVE
            if status["phase"] != "deadline_incomplete":
                status["phase"] = "failed_incomplete"
            status["failure"] = {"type": type(error).__name__, "message": str(error), "unix": time.time()}
            save(path, status)
        raise
