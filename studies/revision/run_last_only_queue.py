"""Guarded, timing-admitted six-fit last-only extension; no outcome selection.

The original twelve-fit campaign is not modified. Every production fit uses the
fixed 8,500-update CLI and its final checkpoint receives the fixed 4,000-update
MLP and ridge readouts. The 500-update smoke is a timing fixture only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

WORLDS = ("gray_frame_still", "gray_frame_wind")
SEEDS = (0, 1, 2)
STEPS = 8500
DEADLINE = "2026-09-20T07:41:00+00:00"
ROOT = Path(__file__).resolve().parents[2]
ACTIVE = None
CHILD = None


def write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def train_projection(log, manifest):
    points = [(int(step), float(seconds)) for step, seconds in re.findall(
        r"\[train\] (\d+)/500 .*?\s([\d.]+)s\s*$", log, re.MULTILINE)]
    require(len(points) == 2 and [p[0] for p in points] == [250, 500],
            "timing fixture must contain exactly updates 250 and 500")
    warm = (points[1][1] - points[0][1]) / 250
    require(warm > 0, "invalid warmed training time")
    return max(0.0, manifest["elapsed_s"] - points[1][1]) + points[0][1] + (STEPS - 250) * warm


def idle_gpu(gpu):
    lines = subprocess.check_output([
        "nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    require(not lines, "GPU has a compute process; exclusive ownership required: " + lines)


def terminate_child():
    """Reap the leader and stop its entire isolated group, including orphans."""
    global CHILD
    child, CHILD = CHILD, None
    if child is None:
        return
    group = child.pid
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        child.poll()
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            break
        time.sleep(.05)
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def on_signal(number, _frame):
    terminate_child()
    raise RuntimeError(f"queue interrupted by signal {number}; preserve partial artifacts")


def main():
    global ACTIVE, CHILD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--window-pin", type=Path, required=True)
    parser.add_argument("--gpu", choices=("0", "1"), required=True)
    parser.add_argument("--deadline-utc", choices=(DEADLINE,), default=DEADLINE)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc).timestamp()
    require(time.time() < deadline, "deadline has already passed")
    require(not args.out.exists(), "output path must not exist; no overwrite or resume")
    launch = json.loads(args.launch_manifest.read_text())
    require(launch["protocol"] == "last_only_objective_v1" and launch["planned_fits"] == 6,
            "wrong launch protocol/matrix")
    require(launch["worlds"] == list(WORLDS) and launch["model_seeds"] == list(SEEDS)
            and launch["modes"] == ["last_only"] and launch["steps"] == STEPS,
            "launch matrix differs from registered six fits")
    require(launch["deadline_utc"] == DEADLINE, "launch deadline mismatch")
    require(os.path.abspath(sys.executable) == launch["python_executable"],
            "queue interpreter differs from the frozen target-gradient interpreter")
    fixed_sources = {str(ROOT / name): digest for name, digest in launch["source_sha256"].items()}
    required_sources = {str(ROOT / name) for name in (
        "studies/revision/run_last_only_queue.py", "studies/revision/last_only_objective.py",
        "studies/revision/last_only_probe.py", "studies/revision/target_gradient.py", "train.py", "probe.py",
        "studies/redcube/env.py", "studies/stoch/train_arms.sh", "pyproject.toml")}
    require(set(fixed_sources) == required_sources, "required frozen source set differs")
    expected_data = {f"{world}/{split}.npz" for world in WORLDS for split in ("train", "eval")}
    require(set(launch["data_sha256"]) == expected_data, "four exact paired archives required")
    require(sha256(args.window_pin) == launch["window_pin_sha256"], "common evaluation window pin changed")
    pin = json.loads(args.window_pin.read_text())
    require(pin.get("protocol") == "frozen_full_history_audit_v1_window_pin"
            and pin.get("status") == "complete" and pin.get("windows_per_world") == 8192
            and pin.get("sample_seed") == 20260919 and set(pin.get("data", {})) == set(WORLDS),
            "common evaluation window pin is incomplete or uses another sampler")
    require(datetime.fromisoformat(pin["created_utc"]).timestamp() <= time.time(),
            "window pin creation time is in the future")
    require(set(pin["sources_sha256"]) == {"studies/revision/full_history_audit.py", "train.py",
            "research/full_history_audit_protocol.md"}, "window pin source set differs")
    require(pin.get("sampler") == "numpy.Generator(PCG64(seed)).choice(valid_windows,8192,replace=False)",
            "window pin sampler differs")
    for name, expected in pin["sources_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                "pin source must be inside frozen checkout")
        require(str(ROOT / name) not in fixed_sources or fixed_sources[str(ROOT / name)] == expected,
                "window pin source conflicts with frozen launch identity")
        fixed_sources[str(ROOT / name)] = expected
    for world in WORLDS:
        entry = pin["data"][world]
        require(entry["sha256"] == launch["data_sha256"][world + "/eval.npz"],
                "window pin uses another evaluation archive")
        require(Path(entry["artifact"]).name == entry["artifact"], "invalid pin artifact path")
        fixed_sources[str(args.window_pin.parent / entry["artifact"])] = entry["artifact_sha256"]
    fixed_sources[str(args.launch_manifest)] = sha256(args.launch_manifest)
    fixed_sources[str(args.window_pin)] = sha256(args.window_pin)
    fixed_sources[str(ROOT / "research/last_only_protocol.md")] = launch["protocol_sha256"]
    for path, expected in fixed_sources.items():
        require(sha256(path) == expected, "frozen source mismatch: " + path)
    # An advisory lock prevents cooperating launchers from sharing the GPU;
    # nvidia-smi also rejects any unrelated active compute process before jobs.
    lock_handle = open(f"/tmp/tinycraft-jepa-gpu{args.gpu}.lock", "a+")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    idle_gpu(args.gpu)
    data_hashes, data_stats = {}, {}
    for name, expected in launch["data_sha256"].items():
        path = args.data_root / name
        require(sha256(path) == expected, "archive differs from paired baseline: " + name)
        data_hashes[str(path)] = expected
        stat = path.stat()
        data_stats[str(path)] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    args.out.mkdir(parents=True, exist_ok=False)
    logs = args.out / "logs"
    logs.mkdir()
    manifest = {
        "protocol": "last_only_objective_v1", "status": "timing_smoke",
        "created_utc": datetime.now(timezone.utc).isoformat(), "gpu": args.gpu,
        "deadline_utc": DEADLINE, "worlds": list(WORLDS), "model_seeds": list(SEEDS),
        "modes": ["last_only"], "steps": STEPS, "planned_fits": 6,
        "fixed_checkpoint_steps": [0, 500, 1000, 2000, 4000, 4250, 8500],
        "source_sha256": fixed_sources, "data_sha256": data_hashes,
        "window_pin_sha256": launch["window_pin_sha256"], "window_pin_created_utc": pin["created_utc"],
        "launch_manifest": launch,
        "scientific_selection": "all six fixed seeds/final checkpoints; no outcome selection or retries",
        "runtime_pairing": "pending external post-campaign comparison to joint baselines; no identity inferred from labels",
        "completion_scope": "six last-only fits and readouts only; eighteen-model comparison requires external verification",
        "gpu_identity": subprocess.check_output([
            "nvidia-smi", "-i", args.gpu, "--query-gpu=uuid,name,driver_version", "--format=csv"], text=True),
        "jobs": []}
    ACTIVE = (args.out / "campaign.json", manifest)
    write_json(*ACTIVE)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
               XLA_PYTHON_CLIENT_MEM_FRACTION=".75", XLA_PYTHON_CLIENT_PREALLOCATE="false")

    def guard():
        require(time.time() < deadline, "deadline reached; preserve incomplete campaign")
        for path, expected in fixed_sources.items():
            require(sha256(path) == expected, "frozen source changed: " + path)
        for path, expected in data_stats.items():
            stat = Path(path).stat()
            require((stat.st_size, stat.st_mtime_ns, stat.st_ino) == expected,
                    "archive metadata changed during campaign: " + path)
        idle_gpu(args.gpu)

    def execute(name, command):
        global CHILD
        guard()
        log = logs / (name + ".log")
        start = time.time()
        row = {"name": name, "command": command, "status": "running",
               "started_utc": datetime.now(timezone.utc).isoformat(), "log": str(log)}
        manifest["jobs"].append(row)
        write_json(*ACTIVE)
        print(f"[start] {name}", flush=True)
        with log.open("x") as handle:
            CHILD = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                     cwd=ROOT, env=env, start_new_session=True)
            try:
                code = CHILD.wait(timeout=max(.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                terminate_child()
                row.update(status="deadline_timeout", seconds=time.time() - start)
                manifest["status"] = "deadline_incomplete"
                write_json(*ACTIVE)
                raise RuntimeError("deadline stopped " + name) from None
            finally:
                terminate_child()
        row.update(status="complete" if code == 0 else "failed", returncode=code,
                   seconds=round(time.time() - start, 3))
        write_json(*ACTIVE)
        require(code == 0, "job failed; no retries or seed replacement: " + name)
        guard()
        for path, expected in data_hashes.items():
            require(sha256(path) == expected, "archive changed during child: " + path)
        print(f"[finish] {name} seconds={row['seconds']}", flush=True)
        return row

    smoke_train = args.data_root / WORLDS[0] / "train.npz"
    smoke_out = args.out / "smoke" / "last_only"
    smoke_code = ("import sys; from dataclasses import replace; "
                  "from studies.revision import last_only_objective as G; "
                  "cfg=replace(G.production_config(0),steps=500); "
                  "G.train(sys.argv[1],sys.argv[2],cfg,'last_only','gray_frame_still',(0,500))")
    row = execute("smoke_train", [sys.executable, "-c", smoke_code, str(smoke_train), str(smoke_out)])
    smoke_meta = json.loads((smoke_out / "manifest.json").read_text())
    require(smoke_meta["status"] == "complete" and smoke_meta["completed_steps"] == 500
            and smoke_meta["data_sha256"] == data_hashes[str(smoke_train)], "invalid timing fixture")
    projected_fit = train_projection(Path(row["log"]).read_text(), smoke_meta)
    smoke_probe = args.out / "smoke" / "readout_final"
    row = execute("smoke_readout", [sys.executable, "-m", "studies.revision.last_only_probe",
        "--model", str(smoke_out / "model.npz"), "--train", str(smoke_train),
        "--eval", str(args.data_root / WORLDS[0] / "eval.npz"), "--out", str(smoke_probe)])
    probe_time = row["seconds"]
    # Budget an additional minute per model for the common 18-model audit and
    # ten minutes for transfer/reporting, then apply a 50% runtime margin.
    remaining = deadline - time.time()
    projected_total = 1.5 * (6 * (projected_fit + probe_time) + 18 * 60) + 600
    admitted = projected_total < remaining
    manifest["admission"] = {"projected_fit_seconds": projected_fit, "readout_seconds": probe_time,
        "audit_reserve_seconds": 18 * 60, "delivery_reserve_seconds": 600, "runtime_margin": 1.5,
        "projected_campaign_seconds_with_margin": projected_total, "remaining_seconds": remaining,
        "admitted": admitted, "basis": "runtime only; no evaluation values read or selected"}
    manifest["status"] = "running" if admitted else "not_admitted_runtime"
    write_json(*ACTIVE)
    print("[admission] " + json.dumps(manifest["admission"]), flush=True)
    if not admitted:
        return
    for world in WORLDS:
        train, evaluation = args.data_root / world / "train.npz", args.data_root / world / "eval.npz"
        for seed in SEEDS:
            out = args.out / world / f"m{seed}" / "last_only"
            stem = f"{world}_m{seed}_last_only"
            execute(stem + "_train", [sys.executable, "-m", "studies.revision.last_only_objective",
                "--config", world, "--data", str(train), "--seed", str(seed),
                "--mode", "last_only", "--out", str(out)])
            fit = json.loads((out / "manifest.json").read_text())
            require(fit["status"] == "complete" and fit["completed_steps"] == STEPS
                and fit["protocol"] == "last_only_objective_v1" and fit["mode"] == "last_only"
                and fit["data_sha256"] == data_hashes[str(train)]
                and fit["fixed_checkpoint_steps"] == manifest["fixed_checkpoint_steps"], "invalid production fit: " + stem)
            for checkpoint in fit["checkpoints"]:
                require(sha256(out / checkpoint["path"]) == checkpoint["sha256"], "checkpoint changed: " + stem)
            readout_dir = out / "readout_final"
            execute(stem + "_readout", [sys.executable, "-m", "studies.revision.last_only_probe",
                "--model", str(out / "model.npz"), "--train", str(train), "--eval", str(evaluation),
                "--out", str(readout_dir)])
            readout = json.loads((readout_dir / "protocol.json").read_text())
            require(readout["status"] == "complete" and readout["protocol"] == "last_only_readouts_v1"
                and readout["sha256"]["model"] == fit["final_checkpoint_sha256"]
                and readout["sha256"]["train"] == data_hashes[str(train)]
                and readout["sha256"]["eval"] == data_hashes[str(evaluation)]
                and readout["mlp"]["steps"] == 4000, "readout identity changed: " + stem)
    guard()
    for path, expected in data_hashes.items():
        require(sha256(path) == expected, "archive changed by completion: " + path)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(*ACTIVE)
    print("[complete] all six registered fits and final readouts", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        main()
    except BaseException as error:
        terminate_child()
        if ACTIVE is not None:
            path, manifest = ACTIVE
            if manifest["status"] not in ("complete", "not_admitted_runtime", "deadline_incomplete"):
                manifest["status"] = "failed_incomplete"
            manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
            write_json(path, manifest)
        raise
