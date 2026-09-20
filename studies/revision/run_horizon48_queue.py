"""Execute the admitted 36-run H48 package on exclusive MG physical GPU 1.

Scientific matrix: three frozen model checkpoints x three policy seeds x four
arms, all native_grid, 50 updates of 512 starts and 48 positions. No retries,
resumes, alternative horizons or outcome-dependent selection. Root admission
and the independent complete H16 audit are frozen inputs. Runtime execution
requires all 24 termination jobs and their queue process to have finished, an
idle GPU 1, and a separately saved two-update full-size H48 oracle_true smoke.

--dry-run validates immutable sources/inputs/admission and emits all 36 commands
without writes, JAX imports, GPU inspection or launching jobs. The smoke and
termination prerequisites are checked only when actually executing the queue.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
PARENT_SHA256 = "6a3460b2db2e47713b7af604454a4b04066c96bfd7748b5c991cfbc433176269"
TERMINATION_SHA256 = "1c003dd4e019b2179981877310e131304d13a7150ac7da93e11d5164d011feb1"
DATA_SHA256 = "c347aa2db1726bceadbc6a16fc2b4a7559343f988c4c2074e90c914b05354b55"
MODEL_SHA256 = {
    0: "f81ef20ab6d6f032a85310fe9233e32749924839eb8bb55e4c0e6b1db2b880d2",
    1: "7190a8c231596043b62b6750f807e1c9fa542287dc161b7ca72182f2e68e625a",
    2: "8ab666aa5838fb81b11af99c5f700f8dd33373f757ef6109a07116df70f9f716",
}
DEADLINE_UTC = "2026-09-20T07:41:00+00:00"
EXPECTED_HOST = "multigpu-vm.sas.yp-c.yandex.net"
EXPECTED_GPU_UUID = "GPU-7634d930-2a61-7013-764f-c10bd309a3d3"
ARMS = ("learned", "oracle_heads", "oracle_reward", "oracle_true")
CORE_SOURCES = (
    "studies/revision/matched_ppo.py", "studies/revision/policy.py",
    "train.py", "probe.py", "studies/stoch/rl.py", "studies/stoch/env.py",
    "studies/redcube/env.py",
)
SOURCE_NAMES = (
    "studies/revision/run_horizon48_queue.py", "studies/revision/analyze_horizon48.py",
    "studies/revision/analyze_matched.py", "studies/revision/analyze_termination.py",
    "studies/revision/matched_ppo_termination.py", *CORE_SOURCES,
    "studies/stoch/sampler.py", "studies/redcube/flow.py", "pyproject.toml",
    "research/horizon48_protocol.md", "research/horizon48_admission.json",
    "research/native_grid_final_audit.json", "output/revision/native-grid-analysis/publication.json",
)
FIXED_ARGS = {
    "config": "gate", "torso": "native_grid", "updates": 50, "batch": 512,
    "hor": 48, "chunk": 16, "mb": 128, "epochs": 4,
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
        "status": manifest["status"], "updated_utc": utc_now(), "n_planned": 36,
        "n_complete": sum(job["status"] == "complete" for job in manifest["jobs"]),
        "jobs": [{key: job[key] for key in (
            "name", "arm", "model_seed", "policy_seed", "status", "out", "log",
            "started_utc", "finished_utc", "seconds", "returncode", "failure",
        ) if key in job} for job in manifest["jobs"]],
    })


def deadline_timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("deadline requires an explicit UTC offset")
    if parsed != datetime.fromisoformat(DEADLINE_UTC):
        raise ValueError(f"this campaign's fixed deadline is {DEADLINE_UTC}")
    return parsed.timestamp()


def assert_hashes(expected):
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen campaign content changed: {path}")


def require_time(deadline, manifest=None, message="campaign deadline reached"):
    if time.time() >= deadline:
        if manifest is not None:
            manifest["status"] = "deadline_incomplete"
        raise TimeoutError(message)


def assert_new_paths(paths):
    for path in paths:
        if os.path.lexists(path):
            raise ValueError(f"refusing existing path, overwrite or resume: {path}")


def validate_admission(root):
    path = root / "research/horizon48_admission.json"
    admission = json.loads(path.read_text())
    if admission.get("status") != "scientifically_admitted_pending_execution_validation":
        raise ValueError("explicit root scientific admission is missing")
    audit_info, publication_info = admission["native_grid_audit"], admission["native_grid_publication"]
    audit_path, publication_path = root / audit_info["path"], root / publication_info["path"]
    assert_hashes({audit_path: audit_info["sha256"], publication_path: publication_info["sha256"]})
    audit, publication = json.loads(audit_path.read_text()), json.loads(publication_path.read_text())
    if audit.get("status") != "PASS" or audit.get("n_runs") != 27:
        raise ValueError("complete independent 27-run native-grid PASS is required")
    if publication.get("status") != "complete" or audit.get("publication_sha256") != sha256(publication_path):
        raise ValueError("native-grid publication does not match the complete independent audit")
    rows = publication.get("validated_runs", [])
    expected = {(m, p, a) for m in range(3) for p in range(3) for a in ("learned", "oracle_heads", "oracle_true")}
    identities = [(r.get("model_seed"), r.get("policy_seed"), r.get("arm")) for r in rows]
    if len(identities) != 27 or set(identities) != expected:
        raise ValueError("H16 admission report does not cover exactly all 27 identities")
    true_rows = [r for r in rows if r["arm"] == "oracle_true"]
    survival = [float(r["final"]["survival"]) for r in true_rows]
    if not all(math.isfinite(v) and 0 <= v <= 0.5 for v in survival):
        raise ValueError("conditional gate requires zero of nine oracle_true discoveries above 50% survival")
    if admission.get("matrix") != {"arms": list(ARMS), "model_seeds": [0, 1, 2],
            "policy_seeds": [0, 1, 2], "n_runs": 36, "torso": "native_grid", "horizon": 48, "updates": 50}:
        raise ValueError("root admission matrix differs from the fixed campaign")
    if admission.get("deadline_utc") != DEADLINE_UTC:
        raise ValueError("admission deadline differs from fixed deadline")
    return {"path": str(path), "sha256": sha256(path), "audit_sha256": sha256(audit_path),
            "publication_sha256": sha256(publication_path), "discovery_count": 0, "denominator": 9,
            "oracle_true_survival": survival, "root_authorization": admission["authorization"]}


def build_plan(args, root=ROOT):
    root = Path(root).resolve()
    if args.gpu != "1":
        raise ValueError("only the root-allocated MG physical GPU 1 is admitted")
    deadline = deadline_timestamp(args.deadline_utc)
    if deadline <= time.time():
        raise ValueError("deadline has passed; no campaign created")
    campaign_dir, output_root = root / "runs/revision/horizon48-campaign", root / "runs/revision/horizon48"
    assert_new_paths([campaign_dir, output_root])
    admission = validate_admission(root)
    sources = {str(root / name): sha256(root / name) for name in SOURCE_NAMES}
    if sources[str(root / "studies/revision/matched_ppo.py")] != PARENT_SHA256:
        raise ValueError("frozen parent harness hash mismatch")
    if sources[str(root / "studies/revision/matched_ppo_termination.py")] != TERMINATION_SHA256:
        raise ValueError("immutable termination harness hash mismatch")
    publication = json.loads((root / "output/revision/native-grid-analysis/publication.json").read_text())
    shared = [normalized_job_sources(row["config"], row["arm"]) for row in publication["validated_runs"]]
    current_shared = {name: sources[str(root / name)] for name in CORE_SOURCES}
    if any(identity != current_shared for identity in shared):
        raise ValueError("shared source content differs from the independently audited H16 campaign")
    model_root, data = args.model_root.resolve(), args.data.resolve()
    models = {m: model_root / ("tile_factored" if m == 0 else f"tile_factored_s{m}") / "model_probe.npz" for m in range(3)}
    content = {str(path): sha256(path) for path in [data, *models.values()]}
    if content[str(data)] != DATA_SHA256 or any(content[str(models[m])] != MODEL_SHA256[m] for m in range(3)):
        raise ValueError("checkpoint/data identity differs from the admitted original MG contents")
    require_time(deadline, message="deadline reached while validating immutable source/input content")
    jobs = []
    for model_seed in range(3):
        for policy_seed in range(3):
            for arm in ARMS:
                name = f"native_grid_m{model_seed}_p{policy_seed}_{arm}"
                out = output_root / f"m{model_seed}" / f"p{policy_seed}" / arm
                module = "studies.revision.matched_ppo_termination" if arm == "oracle_reward" else "studies.revision.matched_ppo"
                command = [sys.executable, "-u", "-m", module, "--arm", arm,
                           "--model", str(models[model_seed]), "--data", str(data), "--out", str(out),
                           "--model-seed", str(model_seed), "--policy-seed", str(policy_seed)]
                for key, value in FIXED_ARGS.items():
                    command.extend(["--" + key, str(value)])
                jobs.append({"name": name, "torso": "native_grid", "model_seed": model_seed,
                             "policy_seed": policy_seed, "arm": arm, "gpu": "1",
                             "model": str(models[model_seed]), "data": str(data), "out": str(out),
                             "log": str(campaign_dir / "logs" / (name + ".log")),
                             "command": command, "status": "not_started"})
    manifest = {
        "status": "registered", "created_utc": utc_now(), "deadline_utc": args.deadline_utc,
        "n_planned": 36, "gpu": "1", "host_allocation": "multigpu-vm physical GPU 1 only",
        "gpu_assignment": "exclusive idle GPU after all 24 termination jobs and queue process finish",
        "parent_harness_sha256": PARENT_SHA256, "termination_harness_sha256": TERMINATION_SHA256,
        "source_sha256": sources, "content_sha256": content, "root_admission": admission,
        "fixed_args": FIXED_ARGS, "output_root": str(output_root),
        "reference_roots": {"original": str(root / "runs/revision/matched_grid"),
                            "oracle_reward": str(root / "runs/revision/termination-native-grid")},
        "smoke_root": str(args.smoke_root.resolve()),
        "termination_campaign": str(root / "runs/revision/termination-campaign/campaign.json"),
        "runtime_admission": {"per_job_floor_seconds": 300.0, "projection_margin": 1.5,
                              "basis": "two-update timing only, never smoke control outcomes"},
        "scientific_selection": "all 36 final-budget outcomes; no retries, alternative horizons or outcome selection",
        "incomplete_policy": "stop queue, preserve artifacts, retain every unstarted job in denominator",
        "interpretation": "longer-horizon training package; fewer starts and collection updates; not a pure GAE effect",
        "jobs": jobs,
    }
    return campaign_dir, manifest, deadline


def assert_idle_gpu(gpu, deadline):
    if socket.gethostname() != EXPECTED_HOST:
        raise RuntimeError("H48 is allocated only to the verified multigpu-vm host")
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("deadline reached before GPU inspection")
    identity = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"],
                              check=True, capture_output=True, text=True, timeout=min(20.0, remaining))
    if identity.stdout.strip() != EXPECTED_GPU_UUID:
        raise RuntimeError("physical GPU1 identity differs from the approved allocation")
    result = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid,process_name",
        "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=min(20.0, remaining))
    if result.stdout.strip():
        raise RuntimeError(f"GPU {gpu} is not exclusive/idle: {result.stdout.strip()}")


def assert_termination_finished(manifest, deadline):
    path = Path(manifest["termination_campaign"])
    previous = json.loads(path.read_text())
    jobs = previous.get("jobs", [])
    expected = {(t, m, p, "oracle_reward") for t, n in (("mlp", 5), ("native_grid", 3))
                for m in range(3) for p in range(n)}
    identities = [(j.get("torso"), j.get("model_seed"), j.get("policy_seed"), j.get("arm")) for j in jobs]
    if previous.get("status") != "complete" or len(jobs) != 24 or set(identities) != expected:
        raise RuntimeError("the complete 24-job termination campaign has not finished")
    for job in jobs:
        if job.get("status") != "complete" or set(job.get("artifact_sha256", {})) != {
            "config.json", "final.json", "episodes.npz", "starts.npz", "ledger.jsonl", "model.pkl"}:
            raise RuntimeError("termination predecessor has incomplete or unverified artifacts")
        assert_hashes({Path(job["out"]) / name: digest for name, digest in job["artifact_sha256"].items()})
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("deadline reached during prerequisite validation")
    processes = subprocess.run(["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True,
                               timeout=min(20.0, remaining))
    for line in processes.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) < 2:
            continue
        try:
            tokens = shlex.split(fields[1])
        except ValueError:
            tokens = fields[1].split()
        if any(token == "studies.revision.run_termination_queue" or token.endswith("/run_termination_queue.py") for token in tokens):
            raise RuntimeError(f"termination queue process is still alive: PID {fields[0]}")
    return {"path": str(path), "sha256": sha256(path), "n_complete": 24, "process_gone": True}


def normalized_job_sources(config, arm):
    names = set(CORE_SOURCES)
    if arm == "oracle_reward":
        names.add("studies/revision/matched_ppo_termination.py")
    result = {}
    for path, digest in config.get("source_sha256", {}).items():
        matches = [name for name in names if path == name or path.endswith("/" + name)]
        if len(matches) != 1 or matches[0] in result:
            raise RuntimeError("job source provenance has unexpected or duplicate identities")
        result[matches[0]] = digest
    if set(result) != names:
        raise RuntimeError("job source provenance has missing identities")
    return result


def validate_against_manifest(record, job, manifest, updates=50):
    config, final = record["config"], record["final"]
    expected = {key.replace("-", "_"): value for key, value in FIXED_ARGS.items()}
    expected.update(updates=updates, arm=job["arm"], model_seed=job["model_seed"], policy_seed=job["policy_seed"],
                    model=job["model"], data=job["data"], out=job["out"])
    for key, value in expected.items():
        if config.get("args", {}).get(key) != value:
            raise RuntimeError(f"job config differs from registered {key}={value}")
    if config.get("model_sha256") != manifest["content_sha256"][job["model"]] or config.get("data_sha256") != DATA_SHA256:
        raise RuntimeError("job checkpoint/data provenance differs from the admitted content")
    source_by_name = {name: manifest["source_sha256"][str(ROOT / name)] for name in normalized_job_sources(config, job["arm"])}
    if normalized_job_sources(config, job["arm"]) != source_by_name:
        raise RuntimeError("job source provenance differs from campaign")
    if job["arm"] == "oracle_reward":
        if config.get("parent_harness_sha256") != PARENT_SHA256 or config.get("sensitivity_harness_sha256") != TERMINATION_SHA256:
            raise RuntimeError("termination variant lineage mismatch")
    if final.get("updates_requested") != updates or final.get("updates_completed") != updates or final.get("budget_complete") is not True:
        raise RuntimeError("job budget differs from the complete registered budget")


def validate_smoke(manifest):
    from studies.revision.analyze_horizon48 import validate_run
    out = Path(manifest["smoke_root"])
    record = validate_run(out, 0, 0, "oracle_true", horizon=48, updates=2)
    job = next(j.copy() for j in manifest["jobs"] if (j["model_seed"], j["policy_seed"], j["arm"]) == (0, 0, "oracle_true"))
    job["out"] = str(out)
    validate_against_manifest(record, job, manifest, updates=2)
    rows = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    final = record["final"]
    elapsed, step_seconds = float(final["total_seconds"]), float(rows[-1]["seconds"])
    if not math.isfinite(elapsed) or not math.isfinite(step_seconds) or elapsed <= 0 or step_seconds <= 0:
        raise RuntimeError("smoke timing evidence must be finite and positive")
    projected = elapsed + 48 * step_seconds
    runtime = manifest["runtime_admission"]
    allowance = max(runtime["per_job_floor_seconds"], runtime["projection_margin"] * projected)
    return {"root": str(out), "artifact_sha256": record["artifact_sha256"], "updates": 2,
            "measured_total_seconds": elapsed, "measured_steady_update_seconds": step_seconds,
            "projected_job_seconds": projected, "allowance_per_job_seconds": allowance,
            "minimum_campaign_seconds": 36 * allowance, "scientific_outcomes_used_for_admission": False}


def validate_output(job, manifest):
    from studies.revision.analyze_horizon48 import validate_run
    record = validate_run(Path(job["out"]), job["model_seed"], job["policy_seed"], job["arm"], horizon=48, updates=50)
    validate_against_manifest(record, job, manifest)
    # Pair all completed arms for this model/policy immediately; the independent
    # analyzer separately validates all complete four-arm blocks and H16 pairs.
    for prior in manifest["jobs"]:
        if prior.get("status") == "complete" and (prior["model_seed"], prior["policy_seed"]) == (job["model_seed"], job["policy_seed"]):
            if prior["start_indices_sha256"] != record["start_indices_sha256"]:
                raise RuntimeError("H48 arm start indices differ within the fixed seed block")
    job["start_indices_sha256"] = record["start_indices_sha256"]
    return record["artifact_sha256"]


def terminate_group(process):
    # This process was created with start_new_session=True, so its PID is the
    # owned process-group ID. Kill descendants even if the leader already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_campaign(campaign_dir, manifest, deadline, root=ROOT):
    global ACTIVE_CAMPAIGN
    root = Path(root).resolve()
    lock_path = Path(tempfile.gettempdir()) / "tinycraft-horizon48-gpu1.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another H48 queue owns GPU 1") from None
        assert_hashes(manifest["source_sha256"])
        assert_hashes(manifest["content_sha256"])
        predecessor = assert_termination_finished(manifest, deadline)
        smoke = validate_smoke(manifest)
        require_time(deadline, manifest, "deadline reached during predecessor/smoke validation")
        assert_idle_gpu("1", deadline)
        assert_new_paths([campaign_dir, Path(manifest["output_root"])])
        campaign_dir.mkdir(parents=True, exist_ok=False)
        (campaign_dir / "logs").mkdir()
        ACTIVE_CAMPAIGN = (campaign_dir, manifest)
        manifest.update(hostname=socket.gethostname(), gpu_uuid=EXPECTED_GPU_UUID,
                        termination_prerequisite=predecessor, smoke=smoke)
        remaining = deadline - time.time()
        manifest["runtime_admission"].update(remaining_seconds=remaining,
            minimum_campaign_seconds=smoke["minimum_campaign_seconds"],
            allowance_per_job_seconds=smoke["allowance_per_job_seconds"],
            admitted=remaining >= smoke["minimum_campaign_seconds"])
        # All 36 commands are saved before any scientific subprocess is started.
        save_campaign(campaign_dir, manifest)
        if not manifest["runtime_admission"]["admitted"]:
            manifest["status"] = "not_admitted_deadline"
            save_campaign(campaign_dir, manifest)
            print("[not-admitted] the full fixed 36-job runtime allowance does not fit", flush=True)
            return
        Path(manifest["output_root"]).mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=EXPECTED_GPU_UUID, XLA_PYTHON_CLIENT_MEM_FRACTION=".75", PYTHONUNBUFFERED="1")
        manifest["status"] = "running"
        save_campaign(campaign_dir, manifest)
        for job in manifest["jobs"]:
            allowance = manifest["runtime_admission"]["allowance_per_job_seconds"]
            if deadline - time.time() < allowance:
                manifest["status"] = "deadline_incomplete"
                raise TimeoutError("insufficient deadline allowance for the next fixed job")
            assert_hashes(manifest["source_sha256"])
            assert_hashes({path: manifest["content_sha256"][path] for path in (job["model"], job["data"])})
            assert_new_paths([Path(job["out"]), Path(job["log"])])
            assert_idle_gpu("1", deadline)
            if deadline - time.time() < allowance:
                manifest["status"] = "deadline_incomplete"
                raise TimeoutError("deadline reached during job preflight")
            began = time.monotonic()
            job.update(status="running", started_utc=utc_now())
            save_campaign(campaign_dir, manifest)
            print(f"[start] GPU1 {job['name']}", flush=True)
            process = None
            try:
                with Path(job["log"]).open("x") as log:
                    process = subprocess.Popen(job["command"], cwd=root, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        job["returncode"] = process.wait(timeout=max(0.001, deadline - time.time()))
                    except subprocess.TimeoutExpired:
                        terminate_group(process)
                        job.update(returncode=process.returncode, status="deadline_timeout")
                        manifest["status"] = "deadline_incomplete"
                        raise TimeoutError(f"deadline stopped {job['name']}; retain partial artifacts") from None
                if job["returncode"] != 0:
                    raise RuntimeError(f"job failed with return code {job['returncode']}")
                require_time(deadline, manifest, "deadline reached before job postflight")
                assert_hashes(manifest["source_sha256"])
                require_time(deadline, manifest, "deadline reached during source postflight")
                assert_hashes({path: manifest["content_sha256"][path] for path in (job["model"], job["data"])})
                require_time(deadline, manifest, "deadline reached during input postflight")
                job["artifact_sha256"] = validate_output(job, manifest)
                require_time(deadline, manifest, "deadline reached during artifact validation")
                job["status"] = "complete"
            except BaseException as error:
                if process is not None:
                    terminate_group(process)
                if job["status"] == "running":
                    job["status"] = "deadline_incomplete" if manifest["status"] == "deadline_incomplete" else "failed_incomplete"
                job["failure"] = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                job.update(finished_utc=utc_now(), seconds=round(time.monotonic() - began, 3))
                save_campaign(campaign_dir, manifest)
                print("[finish] " + json.dumps({key: job[key] for key in ("name", "status", "seconds")}), flush=True)
        require_time(deadline, manifest, "deadline reached before campaign completion")
        manifest.update(status="complete", completed_utc=utc_now())
        save_campaign(campaign_dir, manifest)
        print("[complete] all 36 registered H48 package jobs", flush=True)


def main():
    def interrupted(signum, frame):
        raise InterruptedError(f"queue interrupted by signal {signum}; stop its owned subprocess group")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True, help="separate fullsize m0/p0/oracle_true H48 two-update output")
    parser.add_argument("--gpu", default="1", help="fixed MG physical GPU 1; any other value is rejected")
    parser.add_argument("--deadline-utc", default=DEADLINE_UTC)
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
