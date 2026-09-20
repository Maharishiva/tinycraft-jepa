"""Run the fixed 24-job true-termination sensitivity on one exclusive GPU.

The first 15 jobs are MLP (model seeds 0..2, policy seeds 0..4); the last
nine use native_grid (model seeds 0..2, policy seeds 0..2). Job artifacts go
to runs/revision/termination-{mlp,native-grid}/m#/p#/oracle_reward. Queue
provenance, status, and per-job logs live outside those job directories in
runs/revision/termination-campaign. Every job is registered before launch.

--dry-run validates hashes, paths, and the fixed plan without creating files,
importing JAX, inspecting a GPU, or launching a training process. A real run
requires an idle assigned GPU and refuses any existing campaign/output root.
Failures, deadline stops, and incomplete budgets are preserved; no retry,
resume, seed selection, outcome-based admission, or best-checkpoint selection
is supported. The deadline includes compilation and final evaluation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
HARNESS_SHA256 = "1c003dd4e019b2179981877310e131304d13a7150ac7da93e11d5164d011feb1"
PARENT_SHA256 = "6a3460b2db2e47713b7af604454a4b04066c96bfd7748b5c991cfbc433176269"
SOURCE_NAMES = (
    "studies/revision/run_termination_queue.py",
    "studies/revision/matched_ppo_termination.py",
    "studies/revision/matched_ppo.py",
    "studies/revision/policy.py", "train.py", "probe.py",
    "studies/stoch/rl.py", "studies/stoch/env.py", "studies/stoch/sampler.py",
    "studies/redcube/env.py", "studies/redcube/flow.py", "pyproject.toml",
    "research/termination_sensitivity_protocol.md",
)
FIXED_ARGS = {
    "config": "gate", "arm": "oracle_reward", "updates": 150, "batch": 512,
    "hor": 16, "chunk": 16, "mb": 128, "epochs": 4,
    "p-reset": 0.5, "gamma": 0.99, "lam": 0.9, "lr": 2.5e-4,
    "clip": 0.2, "vf": 0.5, "ent": 0.01,
    "eval-episodes": 256, "eval-seed": 20260919, "max-minutes": 180,
}
ACTIVE_CAMPAIGN = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_campaign(campaign_dir, manifest):
    write_json(campaign_dir / "campaign.json", manifest)
    write_json(campaign_dir / "campaign_status.json", {
        "status": manifest["status"], "updated_utc": utc_now(),
        "n_planned": 24,
        "n_complete": sum(job["status"] == "complete" for job in manifest["jobs"]),
        "jobs": [{key: job[key] for key in (
            "name", "torso", "model_seed", "policy_seed", "status", "out", "log",
            "started_utc", "finished_utc", "seconds", "returncode", "failure",
        ) if key in job} for job in manifest["jobs"]],
    })


def deadline_timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("--deadline-utc requires an explicit UTC offset")
    return parsed.timestamp()


def assert_hashes(expected):
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen campaign content changed: {path}")


def assert_new_paths(paths):
    for path in paths:
        # lexists also rejects broken symlinks; output directories are never reused.
        if os.path.lexists(path):
            raise ValueError(f"refusing to overwrite or resume an existing path: {path}")


def build_plan(args, root=ROOT):
    root = root.resolve()
    model_root, data = args.model_root.resolve(), args.data.resolve()
    output_roots = {"mlp": root / "runs/revision/termination-mlp",
                    "native_grid": root / "runs/revision/termination-native-grid"}
    campaign_dir = root / "runs/revision/termination-campaign"
    assert_new_paths([campaign_dir, *output_roots.values()])
    if not re.fullmatch(r"[0-9]+", args.gpu):
        raise ValueError("--gpu must name exactly one numeric GPU index")
    deadline = deadline_timestamp(args.deadline_utc)
    if deadline <= time.time():
        raise ValueError("deadline has already passed; no campaign created")
    if not math.isfinite(args.minimum_job_seconds) or args.minimum_job_seconds <= 0:
        raise ValueError("--minimum-job-seconds must be finite and positive")
    sources = {str(root / name): sha256(root / name) for name in SOURCE_NAMES}
    if sources[str(root / "studies/revision/matched_ppo_termination.py")] != HARNESS_SHA256:
        raise ValueError("termination harness hash does not match the admitted source")
    if sources[str(root / "studies/revision/matched_ppo.py")] != PARENT_SHA256:
        raise ValueError("original parent harness hash does not match the frozen source")
    models = {seed: model_root / ("tile_factored" if seed == 0 else f"tile_factored_s{seed}")
              / "model_probe.npz" for seed in range(3)}
    content = {str(path): sha256(path) for path in [data, *models.values()]}
    jobs = []
    for torso, policy_seeds in (("mlp", range(5)), ("native_grid", range(3))):
        for model_seed in range(3):
            for policy_seed in policy_seeds:
                name = f"{torso}_m{model_seed}_p{policy_seed}_oracle_reward"
                out = output_roots[torso] / f"m{model_seed}" / f"p{policy_seed}" / "oracle_reward"
                command = [sys.executable, "-u", "-m", "studies.revision.matched_ppo_termination",
                           "--model", str(models[model_seed]), "--data", str(data), "--out", str(out),
                           "--model-seed", str(model_seed), "--policy-seed", str(policy_seed),
                           "--torso", torso]
                for key, value in FIXED_ARGS.items():
                    command.extend(["--" + key, str(value)])
                jobs.append({"name": name, "torso": torso, "model_seed": model_seed,
                             "policy_seed": policy_seed, "arm": "oracle_reward", "gpu": args.gpu,
                             "model": str(models[model_seed]), "data": str(data), "out": str(out),
                             "log": str(campaign_dir / "logs" / (name + ".log")),
                             "command": command, "status": "not_started"})
    manifest = {
        "status": "registered", "created_utc": utc_now(), "deadline_utc": args.deadline_utc,
        "n_planned": 24, "gpu": args.gpu, "gpu_assignment": "one exclusive idle device",
        "harness_sha256": HARNESS_SHA256, "parent_harness_sha256": PARENT_SHA256,
        "source_sha256": sources, "content_sha256": content,
        "fixed_args": FIXED_ARGS, "output_roots": {key: str(value) for key, value in output_roots.items()},
        "runtime_admission": {"minimum_job_seconds": args.minimum_job_seconds,
                              "minimum_campaign_seconds": 24 * args.minimum_job_seconds,
                              "basis": "prospective runtime floor only; no outcomes inspected"},
        "scientific_selection": "all 24 registered final-budget outcomes; no retries or outcome-based selection",
        "incomplete_policy": "stop queue, preserve artifacts, retain every unstarted job in denominator",
        "jobs": jobs,
    }
    return campaign_dir, manifest, deadline


def assert_idle_gpu(gpu, deadline):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("deadline reached before GPU inspection")
    result = subprocess.run([
        "nvidia-smi", "-i", gpu, "--query-compute-apps=pid,process_name", "--format=csv,noheader",
    ], check=True, capture_output=True, text=True, timeout=min(20.0, remaining))
    if result.stdout.strip():
        raise RuntimeError(f"GPU {gpu} is not exclusive/idle: {result.stdout.strip()}")


def terminate_group(process):
    """Reap a deadline/interruption-stopped job, including any child processes."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def validate_output(job, manifest):
    out = Path(job["out"])
    required = ("config.json", "final.json", "episodes.npz", "starts.npz", "ledger.jsonl", "model.pkl")
    for name in required:
        if not (out / name).is_file():
            raise RuntimeError(f"missing required job artifact: {out / name}")
    config = json.loads((out / "config.json").read_text())
    final = json.loads((out / "final.json").read_text())
    if final.get("budget_complete") is not True or final.get("updates_completed") != 150:
        raise RuntimeError("job returned with an incomplete fixed training budget")
    expected_args = {key.replace("-", "_"): value for key, value in FIXED_ARGS.items()}
    expected_args.update(model_seed=job["model_seed"], policy_seed=job["policy_seed"], torso=job["torso"],
                         model=job["model"], data=job["data"], out=job["out"])
    for key, expected in expected_args.items():
        if config.get("args", {}).get(key) != expected:
            raise RuntimeError(f"job config differs from registered {key}={expected}")
    for key in ("arm", "torso", "model_seed", "policy_seed"):
        if final.get(key) != job[key]:
            raise RuntimeError(f"final identity mismatch: {key}")
    for key, expected in (("updates_requested", 150), ("transitions", 1228800),
                          ("eval_seed", 20260919), ("n_ep", 256)):
        if final.get(key) != expected:
            raise RuntimeError(f"final budget/evaluation mismatch: {key}")
    for field, expected in (("parent_harness_sha256", PARENT_SHA256),
                            ("sensitivity_harness_sha256", HARNESS_SHA256),
                            ("model_sha256", manifest["content_sha256"][job["model"]]),
                            ("data_sha256", manifest["content_sha256"][job["data"]])):
        if config.get(field) != expected:
            raise RuntimeError(f"job provenance mismatch: {field}")
    required_sources = {"studies/revision/matched_ppo_termination.py", "studies/revision/matched_ppo.py",
                        "studies/stoch/rl.py", "studies/stoch/env.py", "train.py", "probe.py",
                        "studies/redcube/env.py"}
    if job["torso"] == "native_grid":
        required_sources.add("studies/revision/policy.py")
    actual_sources = {str(Path(path).resolve()): digest for path, digest in config.get("source_sha256", {}).items()}
    expected_sources = {str(ROOT / name) for name in required_sources}
    if set(actual_sources) != expected_sources:
        raise RuntimeError("job source provenance has missing or unexpected files")
    for path, digest in actual_sources.items():
        if manifest["source_sha256"].get(str(Path(path).resolve())) != digest:
            raise RuntimeError(f"job source provenance differs from campaign: {path}")
    if final.get("model_sha256") != config["model_sha256"] or final.get("policy_sha256") != sha256(out / "model.pkl"):
        raise RuntimeError("final model or policy content hash mismatch")
    rows = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    if [row.get("update") for row in rows] != list(range(1, 151)):
        raise RuntimeError("ledger does not contain exactly the 150 scheduled updates")
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise RuntimeError(f"nonfinite ledger field: {key}")
        for key in ("postterminal_alive_mass_fraction", "postterminal_reward_per_rollout"):
            if row.get(key) != 0.0:
                raise RuntimeError(f"true-termination semantic invariant failed: {key}")
    return {name: sha256(out / name) for name in required}


def run_campaign(campaign_dir, manifest, deadline, root=ROOT):
    global ACTIVE_CAMPAIGN
    root = root.resolve()
    lock_path = Path(tempfile.gettempdir()) / f"tinycraft-termination-gpu{manifest['gpu']}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another termination queue owns this GPU lock") from None
        assert_idle_gpu(manifest["gpu"], deadline)
        assert_new_paths([campaign_dir, *map(Path, manifest["output_roots"].values())])
        campaign_dir.mkdir(parents=True, exist_ok=False)
        (campaign_dir / "logs").mkdir()
        ACTIVE_CAMPAIGN = (campaign_dir, manifest)
        save_campaign(campaign_dir, manifest)
        remaining = deadline - time.time()
        manifest["runtime_admission"].update(remaining_seconds=remaining,
            admitted=remaining >= manifest["runtime_admission"]["minimum_campaign_seconds"])
        if not manifest["runtime_admission"]["admitted"]:
            manifest["status"] = "not_admitted_deadline"
            save_campaign(campaign_dir, manifest)
            print("[not-admitted] insufficient time for the prospective 24-job runtime floor", flush=True)
            return
        for output_root in manifest["output_roots"].values():
            Path(output_root).mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=manifest["gpu"],
                   XLA_PYTHON_CLIENT_MEM_FRACTION=".75", PYTHONUNBUFFERED="1")
        manifest["status"] = "running"
        save_campaign(campaign_dir, manifest)
        for job in manifest["jobs"]:
            if deadline - time.time() < manifest["runtime_admission"]["minimum_job_seconds"]:
                manifest["status"] = "deadline_incomplete"
                raise TimeoutError("insufficient deadline margin to start the next registered job")
            assert_hashes(manifest["source_sha256"])
            assert_hashes({path: manifest["content_sha256"][path] for path in (job["model"], job["data"])})
            assert_new_paths([Path(job["out"]), Path(job["log"])])
            assert_idle_gpu(manifest["gpu"], deadline)
            remaining = deadline - time.time()
            if remaining < manifest["runtime_admission"]["minimum_job_seconds"]:
                manifest["status"] = "deadline_incomplete"
                raise TimeoutError("deadline reached during job preflight")
            began = time.monotonic()
            job.update(status="running", started_utc=utc_now())
            save_campaign(campaign_dir, manifest)
            print(f"[start] GPU{manifest['gpu']} {job['name']}", flush=True)
            process = None
            try:
                with Path(job["log"]).open("x") as log:
                    process = subprocess.Popen(job["command"], cwd=root, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        job["returncode"] = process.wait(timeout=max(0.001, deadline - time.time()))
                    except subprocess.TimeoutExpired:
                        terminate_group(process)
                        job["returncode"] = process.returncode
                        job["status"] = "deadline_timeout"
                        manifest["status"] = "deadline_incomplete"
                        raise TimeoutError(f"deadline stopped {job['name']}; retain partial artifacts") from None
                if job["returncode"] != 0:
                    raise RuntimeError(f"job failed with return code {job['returncode']}")
                assert_hashes(manifest["source_sha256"])
                assert_hashes({path: manifest["content_sha256"][path] for path in (job["model"], job["data"])})
                job["artifact_sha256"] = validate_output(job, manifest)
                job["status"] = "complete"
            except BaseException as error:
                if process is not None:
                    terminate_group(process)
                if job["status"] == "running":
                    job["status"] = "failed_incomplete"
                job["failure"] = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                job.update(finished_utc=utc_now(), seconds=round(time.monotonic() - began, 3))
                save_campaign(campaign_dir, manifest)
                print("[finish] " + json.dumps({key: job[key] for key in ("name", "status", "seconds")}), flush=True)
        manifest.update(status="complete", completed_utc=utc_now())
        save_campaign(campaign_dir, manifest)
        print("[complete] all 24 registered termination sensitivity jobs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True, help="existing gate directory containing tile_factored[_s#]")
    parser.add_argument("--data", type=Path, required=True, help="exact gate training archive used by matched controls")
    parser.add_argument("--gpu", default="1", help="one exclusive physical GPU index (default: 1)")
    parser.add_argument("--deadline-utc", required=True, help="absolute ISO8601 UTC deadline with +00:00 or Z offset")
    parser.add_argument("--minimum-job-seconds", type=float, default=300.0,
                        help="prospective deadline-admission floor per job, independent of outcomes (default: 300)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    campaign_dir, manifest, deadline = build_plan(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "campaign_dir": str(campaign_dir), **manifest}, indent=2, allow_nan=False))
        return
    try:
        run_campaign(campaign_dir, manifest, deadline)
    except BaseException as error:
        if ACTIVE_CAMPAIGN is not None:
            active_dir, active_manifest = ACTIVE_CAMPAIGN
            if active_manifest["status"] != "deadline_incomplete":
                active_manifest["status"] = "failed_incomplete"
            active_manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
            active_manifest["stopped_utc"] = utc_now()
            save_campaign(active_dir, active_manifest)
        raise


if __name__ == "__main__":
    main()
