"""Timing admission followed by the fixed twelve-fit target-gradient campaign.

Use one exclusive GPU only after the primary matched-PPO and grid campaigns.
Smoke fits are 500-update timing fixtures, excluded from scientific endpoints.
Full fits always use the production 8,500-update CLI and every registered seed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ACTIVE_CAMPAIGN = None


def write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def training_projection(log, manifest):
    points = [(int(step), float(seconds)) for step, seconds in
              re.findall(r"\[train\] (\d+)/500 .*?\s([\d.]+)s\s*$", log, flags=re.MULTILINE)]
    if len(points) != 2 or points[0][0] != 250 or points[1][0] != 500:
        raise ValueError("timing smoke must log exactly250 and500 updates")
    first, second = points
    warm_seconds_per_update = (second[1] - first[1]) / 250
    if warm_seconds_per_update <= 0:
        raise ValueError("invalid warmed training time")
    startup = manifest["elapsed_s"] - second[1]
    return startup + first[1] + (8500 - 250) * warm_seconds_per_update


def main():
    global ACTIVE_CAMPAIGN
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--deadline-utc", required=True, help="ISO8601 UTC deadline, e.g.2026-09-20T07:41:00+00:00")
    p.add_argument("--skip-midpoint", action="store_true", help="prospective resource decision only, before any fit")
    a = p.parse_args()
    deadline_value = datetime.fromisoformat(a.deadline_utc)
    if deadline_value.utcoffset() is None or deadline_value.utcoffset().total_seconds() != 0:
        raise ValueError("deadline must include an explicit UTC offset")
    deadline = deadline_value.timestamp()
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError("campaign output must be new and empty")
    a.out.mkdir(parents=True, exist_ok=True)
    log_root = a.out / "logs"
    log_root.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=".75")
    sources = [Path(__file__), Path(__file__).with_name("target_gradient.py"),
               Path(__file__).with_name("target_gradient_probe.py"),
               Path(__file__).with_name("target_gradient_sensitivity.py")]
    root = Path(__file__).resolve().parents[2]
    sources.extend(root / name for name in ("train.py", "probe.py", "studies/redcube/env.py",
                                            "studies/stoch/train_arms.sh", "pyproject.toml"))
    fixed_sources = {str(path): sha256(path) for path in sources}
    data_hashes = {str(a.data_root / world / (split + ".npz")): sha256(a.data_root / world / (split + ".npz"))
                   for world in ("gray_frame_still", "gray_frame_wind") for split in ("train", "eval")}
    manifest = {"status": "timing_smoke", "created_utc": datetime.now(timezone.utc).isoformat(),
                "gpu": a.gpu, "deadline_utc": a.deadline_utc, "source_sha256": fixed_sources,
                "data_sha256": data_hashes, "midpoint_skip_requested": a.skip_midpoint,
                "worlds": ["gray_frame_still", "gray_frame_wind"], "model_seeds": [0, 1, 2],
                "modes": ["joint", "stop_target"], "steps": 8500, "planned_fits": 12,
                "scientific_selection": "all fixed seeds and final checkpoints; no outcome-dependent selection",
                "jobs": []}
    write_json(a.out / "campaign.json", manifest)
    ACTIVE_CAMPAIGN = (a.out / "campaign.json", manifest)

    def execute(name, command):
        remaining = deadline - time.time()
        if remaining <= 0:
            manifest["status"] = "deadline_incomplete"
            write_json(a.out / "campaign.json", manifest)
            raise RuntimeError("deadline reached before the next job; preserve incomplete campaign")
        for path, expected in fixed_sources.items():
            if sha256(path) != expected:
                raise RuntimeError(f"frozen campaign source changed: {path}")
        start = time.time()
        log = log_root / (name + ".log")
        print(f"[start] {name}", flush=True)
        try:
            with log.open("w") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                           timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            manifest["jobs"].append({"name": name, "command": command, "status": "deadline_timeout",
                                      "seconds": round(time.time() - start, 3), "log": str(log)})
            manifest["status"] = "deadline_incomplete"
            write_json(a.out / "campaign.json", manifest)
            raise RuntimeError(f"deadline stopped {name}; retain its partial artifacts") from None
        row = {"name": name, "command": command, "returncode": completed.returncode,
               "seconds": round(time.time() - start, 3), "log": str(log)}
        manifest["jobs"].append(row)
        write_json(a.out / "campaign.json", manifest)
        print("[finish] " + json.dumps(row), flush=True)
        if completed.returncode:
            manifest["status"] = "failed"
            write_json(a.out / "campaign.json", manifest)
            raise RuntimeError(f"job failed; preserve its files and investigate: {name}")
        return row

    # Compile and time both backward graphs on full-size data/model/batches.
    # Only the smoke schedule is shortened; these outputs are never endpoints.
    smoke_train = a.data_root / "gray_frame_still" / "train.npz"
    projections = {}
    smoke_code = ("import sys; from dataclasses import replace; "
                  "from studies.revision import target_gradient as G; "
                  "cfg=replace(G.production_config(0),steps=500); "
                  "G.train(sys.argv[1],sys.argv[2],cfg,sys.argv[3],'gray_frame_still',(0,500))")
    for mode in manifest["modes"]:
        out = a.out / "smoke" / mode
        row = execute("smoke_" + mode, [sys.executable, "-c", smoke_code, str(smoke_train), str(out), mode])
        smoke_manifest = json.loads((out / "manifest.json").read_text())
        if smoke_manifest["data_sha256"] != data_hashes[str(smoke_train)]:
            raise RuntimeError("smoke training archive changed after the campaign was registered")
        projections[mode] = training_projection(Path(row["log"]).read_text(), smoke_manifest)
    smoke_probe = a.out / "smoke" / "readout"
    probe_command = [sys.executable, "-m", "studies.revision.target_gradient_probe",
                     "--model", str(a.out / "smoke" / "joint" / "model.npz"),
                     "--train", str(smoke_train), "--eval", str(a.data_root / "gray_frame_still" / "eval.npz"),
                     "--out", str(smoke_probe)]
    probe_row = execute("smoke_readout", probe_command)
    probe_seconds = probe_row["seconds"]
    smoke_probe_meta = json.loads((smoke_probe / "protocol.json").read_text())
    for split in ("train", "eval"):
        if smoke_probe_meta["sha256"][split] != data_hashes[str(a.data_root / "gray_frame_still" / (split + ".npz"))]:
            raise RuntimeError("smoke readout archive changed after the campaign was registered")
    remaining = deadline - time.time()
    midpoint_cost = 1.5 * 12 * (max(projections.values()) + 2 * probe_seconds + 60)
    midpoint = probe_seconds <= 300 and not a.skip_midpoint and midpoint_cost < remaining
    # Include a minute for the small sensitivity plus conservative50% margin.
    projected_seconds = 1.5 * 12 * (max(projections.values()) + (1 + int(midpoint)) * probe_seconds + 60)
    admitted = max(projections.values()) <= 1200 and projected_seconds < remaining
    manifest["admission"] = {"projected_fit_seconds": projections, "readout_seconds": probe_seconds,
                              "midpoint_all_models": midpoint, "projected_campaign_seconds_with_margin": projected_seconds,
                              "remaining_seconds": remaining, "admitted": admitted,
                              "basis": "runtime only; no readout outcomes inspected by launcher"}
    manifest["status"] = "running" if admitted else "not_admitted_runtime"
    write_json(a.out / "campaign.json", manifest)
    print("[admission] " + json.dumps(manifest["admission"]), flush=True)
    if not admitted:
        return
    for world in manifest["worlds"]:
        train, eval_data = a.data_root / world / "train.npz", a.data_root / world / "eval.npz"
        for seed in manifest["model_seeds"]:
            for mode in manifest["modes"]:
                stem = f"{world}_m{seed}_{mode}"
                out = a.out / world / f"m{seed}" / mode
                execute(stem + "_train", [sys.executable, "-m", "studies.revision.target_gradient",
                                          "--config", world, "--data", str(train), "--seed", str(seed),
                                          "--mode", mode, "--out", str(out)])
                fit = json.loads((out / "manifest.json").read_text())
                if fit.get("status") != "complete" or fit.get("completed_steps") != 8500:
                    raise RuntimeError(f"incomplete fixed-budget fit: {stem}")
                if fit["data_sha256"] != data_hashes[str(train)]:
                    raise RuntimeError(f"training archive differs from pinned campaign identity: {stem}")
                checkpoints = [("final", "model.npz")]
                if midpoint:
                    checkpoints.append(("midpoint", "model_step004250.npz"))
                for label, checkpoint in checkpoints:
                    execute(stem + "_" + label, [sys.executable, "-m", "studies.revision.target_gradient_probe",
                                                  "--model", str(out / checkpoint), "--train", str(train),
                                                  "--eval", str(eval_data), "--out", str(out / ("readout_" + label))])
                    readout = json.loads((out / ("readout_" + label) / "protocol.json").read_text())
                    if readout["sha256"]["train"] != data_hashes[str(train)] or readout["sha256"]["eval"] != data_hashes[str(eval_data)]:
                        raise RuntimeError(f"readout archives differ from pinned campaign identities: {stem}")
                execute(stem + "_sensitivity", [sys.executable, "-m", "studies.revision.target_gradient_sensitivity",
                                                  "--model", str(out / "model.npz"), "--eval-data", str(eval_data),
                                                  "--out", str(out / "sensitivity.json")])
                if json.loads((out / "sensitivity.json").read_text())["eval_data_sha256"] != data_hashes[str(eval_data)]:
                    raise RuntimeError(f"sensitivity archive differs from pinned campaign identity: {stem}")
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(a.out / "campaign.json", manifest)
    print("[complete] all12 registered fits and readouts", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ACTIVE_CAMPAIGN is not None:
            path, manifest = ACTIVE_CAMPAIGN
            if manifest["status"] not in ("complete", "not_admitted_runtime", "deadline_incomplete"):
                manifest["status"] = "failed_incomplete"
            manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
            write_json(path, manifest)
        raise
