"""Sequential three-model distance diagnostic, with a 20-minute block limit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/dev/shm/redcube/gray_frame_still"))
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "queue.json").exists() or any((args.out / f"m{s}").exists() for s in range(3)):
        raise FileExistsError("refusing to overwrite an existing distance queue")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    started = time.monotonic()
    record = {"started_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
              "maximum_block_seconds": 1200, "gpu": args.gpu,
              "wrapper_sha256": hashlib.sha256((root / "studies/revision/redcube_goal_distance.py").read_bytes()).hexdigest(),
              "prospective_model_order": [0, 1, 2], "runs": [], "complete": False}
    for seed in range(3):
        model = args.model_root / ("tile_factored" if seed == 0 else f"tile_factored_s{seed}") / "model_probe.npz"
        cmd = [sys.executable, "-u", "-m", "studies.revision.redcube_goal_distance", "--model", str(model), "--out", str(args.out / f"m{seed}")]
        row = {"model_seed": seed, "command": cmd, "started_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
        record["runs"].append(row)
        (args.out / "queue.json").write_text(json.dumps(record, indent=2))
        remaining = 1200 - (time.monotonic() - started)
        if remaining <= 0:
            row["status"] = "not started: resource time limit"
            break
        with (args.out / f"m{seed}.log").open("xb") as log:
            try:
                child = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
                row["returncode"] = child.returncode
                row["status"] = "complete" if child.returncode == 0 else "source or runtime failure"
            except subprocess.TimeoutExpired:
                row["returncode"] = 124
                row["status"] = "incomplete: resource time limit; partial artifacts preserved"
        row["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        print(json.dumps(row), flush=True)
        (args.out / "queue.json").write_text(json.dumps(record, indent=2))
        if row["returncode"] != 0:
            break
    record["seconds"] = time.monotonic() - started
    record["complete"] = len(record["runs"]) == 3 and all(r.get("returncode") == 0 for r in record["runs"])
    (args.out / "queue.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"queue_complete": record["complete"], "seconds": record["seconds"]}), flush=True)
    if not record["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
