"""Run a fixed, prospectively chosen matched PPO matrix on exclusive GPU lanes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--policy-seeds", type=int, required=True)
    p.add_argument("--torso", choices=("mlp", "native_grid"), default="mlp")
    p.add_argument("--decision", required=True, help="prospective seed-count/runtime decision")
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError("new campaign output must be empty")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "logs").mkdir()
    gpus = a.gpus.split(",")
    jobs = []
    for ms in range(3):
        model = a.model_root / ("tile_factored" if ms == 0 else f"tile_factored_s{ms}") / "model_probe.npz"
        for ps in range(a.policy_seeds):
            for arm in ("learned", "oracle_heads", "oracle_true"):
                out = a.out / f"m{ms}" / f"p{ps}" / arm
                command = [sys.executable, "-m", "studies.revision.matched_ppo", "--arm", arm,
                           "--model", str(model), "--data", str(a.data), "--out", str(out),
                           "--model-seed", str(ms), "--policy-seed", str(ps), "--torso", a.torso,
                           "--updates", "150", "--eval-episodes", "256", "--eval-seed", "20260919",
                           "--max-minutes", "180"]
                jobs.append({"model_seed": ms, "policy_seed": ps, "arm": arm,
                             "gpu": gpus[len(jobs) % len(gpus)], "out": str(out), "command": command})
    source = Path(__file__).with_name("matched_ppo.py")
    campaign = {"created_unix": time.time(), "decision": a.decision, "torso": a.torso,
                "policy_seeds": a.policy_seeds, "model_seeds": [0, 1, 2], "n_runs": len(jobs),
                "harness_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "jobs": jobs}
    (a.out / "campaign.json").write_text(json.dumps(campaign, indent=2))
    lock = threading.Lock()
    status = []

    def lane(gpu):
        for job in (j for j in jobs if j["gpu"] == gpu):
            name = f'm{job["model_seed"]}_p{job["policy_seed"]}_{job["arm"]}'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, XLA_PYTHON_CLIENT_MEM_FRACTION=".45")
            began = time.time()
            print(f"[start] GPU{gpu} {name}", flush=True)
            with (a.out / "logs" / f"{name}.log").open("w") as log:
                completed = subprocess.run(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT)
            row = {"name": name, "gpu": gpu, "returncode": completed.returncode,
                   "seconds": round(time.time() - began, 3)}
            final_path = Path(job["out"]) / "final.json"
            if final_path.exists():
                row["final"] = json.loads(final_path.read_text())
            with lock:
                status.append(row)
                temp = a.out / "campaign_status.tmp"
                temp.write_text(json.dumps(status, indent=2))
                temp.replace(a.out / "campaign_status.json")
            print("[finish] " + json.dumps(row), flush=True)
            if completed.returncode or not row.get("final", {}).get("budget_complete"):
                raise RuntimeError(f"lane {gpu} stopped after failed/incomplete job {name}; preserve its artifacts")
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(lane, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    print(f"[complete] {len(status)}/{len(jobs)} runs", flush=True)


if __name__ == "__main__":
    main()
