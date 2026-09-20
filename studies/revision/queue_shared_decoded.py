"""Fixed thirty-policy scheduler; prepare first, then one worker per GPU.

Only stdlib is imported. No worker runs without a recorded runtime admission.
Claims are protected by flock; failures and deadline-incomplete runs are retained
and never automatically retried. A separately launched registered job can be
declared during preparation and adopted without duplicate execution.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORLDS = ("gray_frame_still", "gray_frame_wind", "gray_episode_wind", "gate")
DEADLINE = "2026-09-20T07:41:00+00:00"
SOURCES = ("studies/revision/queue_shared_decoded.py",
           "studies/revision/shared_decoded_ppo.py", "studies/revision/decoded_models.py",
           "studies/revision/decoded_dream.py", "studies/revision/warm_context_eval.py",
           "studies/revision/refit_full_label_heads.py", "studies/revision/matched_ppo.py",
           "studies/stoch/rl.py", "studies/stoch/env.py", "studies/redcube/env.py",
           "studies/redcube/flow.py", "train.py", "probe.py",
           "studies/stoch/flow_recipe/train.py", "studies/stoch/flow_recipe/eval.py",
           "studies/stoch/flow_recipe/kernels.py", "research/shared_decoded_cnn_protocol.md")
TERMINAL = {"complete", "failed", "deadline_timeout", "deadline_not_started",
            "external_failed", "worker_interrupted"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


@contextmanager
def locked(path, nonblock=False):
    with Path(path).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblock else 0))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def process_identity(pid):
    """Linux start time distinguishes a registered process from later PID reuse."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return dict(starttime=fields[19], ppid=int(fields[1]), state=fields[0])


def alive(pid, starttime=None):
    try:
        os.kill(int(pid), 0)
        info = process_identity(pid)
        return info["state"] != "Z" and (starttime is None or info["starttime"] == starttime)
    except (ProcessLookupError, FileNotFoundError):
        return False


def kill_external_tree(record):
    """Do not assume nohup gave the externally launched Python a private group."""
    if not alive(record["pid"], record["pid_starttime"]):
        return
    identities = {}
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            identities[int(entry.name)] = process_identity(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    ordered = [record["pid"]]
    for parent in ordered:
        ordered.extend(pid for pid, info in identities.items() if info["ppid"] == parent and pid not in ordered)
    for pid in reversed(ordered):
        if pid in identities and alive(pid, identities[pid]["starttime"]):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def matrix(a):
    jobs = []
    # The six decoded-real gate controls are ordered before all dreams.
    combinations = [("gate", family, "real", seed) for seed in range(3)
                    for family in ("jepa", "flow")]
    combinations += [(world, family, "dream", seed) for world in WORLDS
                     for seed in range(3) for family in ("jepa", "flow")]
    for world, family, source, seed in combinations:
        base = a.stoch_root / world
        data = (base if world == "gate" else a.gray_root / world) / "train.npz"
        model = ((base / "tile_factored_s1" / "model_probe.npz" if world == "gate"
                  else a.gray_root / world / "tile_factored" / "model_probe.npz")
                 if family == "jepa" else base / "flow" / "model.npz")
        heads = a.out / "heads" / world / "model_probe_fullheads.npz" if family == "jepa" and source == "dream" else None
        out = a.out / "policies" / world / family / source / f"p{seed}"
        packet = a.out / "packets" / f"{world}.npz"
        updates = 150 if world == "gate" else 50
        command = [str(a.python), "-m", "studies.revision.shared_decoded_ppo", "--config", world,
                   "--family", family, "--train-source", source, "--model", str(model),
                   "--data", str(data), "--eval-packet", str(packet), "--out", str(out),
                   "--policy-seed", str(seed), "--updates", str(updates), "--batch", "512", "--hor", "48"]
        if heads is not None:
            command += ["--heads", str(heads)]
        jobs.append(dict(id=f"{world}_{family}_{source}_p{seed}", world=world, family=family,
                         train_source=source, policy_seed=seed, model_seed=1 if world == "gate" and family == "jepa" else 0,
                         model=str(model), data=str(data), heads=str(heads) if heads else None,
                         packet=str(packet), out=str(out), updates=updates, batch=512, horizon=48,
                         nominal_positions=updates * 512 * 48, command=command))
    assert len(jobs) == 30 and len({j["id"] for j in jobs}) == 30
    return jobs


def validate_config(job, cfg, manifest):
    for name, value in cfg["source_sha256"].items():
        if manifest["source_sha256"].get(name) != value:
            raise ValueError(f"external job used different frozen source: {name}")
    if cfg["data_sha256"] != manifest["input_sha256"][job["data"]]:
        raise ValueError("external job used different training data")


def validate_external(job, argv, config, manifest):
    argv = [v.decode() for v in argv if v]
    if "--smoke" in argv:
        raise ValueError("cannot adopt a timing smoke as a scientific job")
    expected = {"--config": job["world"], "--family": job["family"],
                "--train-source": job["train_source"], "--policy-seed": str(job["policy_seed"]),
                "--updates": str(job["updates"]), "--batch": "512", "--hor": "48",
                "--model": job["model"], "--data": job["data"], "--out": job["out"],
                "--eval-packet": job["packet"]}
    defaults = {"--train-source": "dream", "--batch": "512", "--hor": "48"}
    paths = {"--model", "--data", "--out", "--eval-packet"}
    for flag, value in expected.items():
        actual = argv[argv.index(flag) + 1] if flag in argv else defaults.get(flag)
        equal = Path(actual).resolve() == Path(value).resolve() if flag in paths and actual else actual == value
        if not equal:
            raise ValueError(f"adopted process differs from registered {flag}: {actual}")
    if config.exists():
        validate_config(job, read(config), manifest)


def validate_completed(job, out, manifest):
    """A job finished by hand before registration is adopted only if its own records match the registered job."""
    good, result = final_result(job)
    if not good:
        raise FileExistsError(f"undeclared existing job output: {out}")
    cfg = read(out / "config.json")
    args = cfg["args"]
    if args["smoke"]:
        raise ValueError("cannot adopt a timing smoke as a scientific job")
    expected = {"config": job["world"], "family": job["family"], "train_source": job["train_source"],
                "policy_seed": job["policy_seed"], "updates": job["updates"], "batch": 512, "hor": 48}
    for key, value in expected.items():
        if args[key] != value:
            raise ValueError(f"completed job differs from registered {key}: {args[key]}")
    for key, value in {"model": job["model"], "data": job["data"], "out": job["out"], "eval_packet": job["packet"]}.items():
        if Path(args[key]).resolve() != Path(value).resolve():
            raise ValueError(f"completed job differs from registered {key}: {args[key]}")
    validate_config(job, cfg, manifest)
    return result


def prepare(a):
    q = a.out / "queue"
    q.mkdir(parents=True, exist_ok=True)
    with locked(q / "claims.lock"):
        if (q / "campaign.json").exists() or (q / "status.json").exists():
            raise FileExistsError("registered campaign already exists; preparation never overwrites")
        jobs = matrix(a)
        external = {}
        for item in a.external_running:
            name, gpu, pid = item.rsplit(":", 2)
            if name in external or name not in {j["id"] for j in jobs} or not alive(int(pid)):
                raise ValueError("invalid, duplicate or nonlive external registered job")
            identity = process_identity(int(pid))
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if b"studies.revision.shared_decoded_ppo" not in argv:
                raise ValueError("external PID is not the registered policy driver")
            external[name] = dict(gpu=gpu, pid=int(pid), pid_starttime=identity["starttime"],
                                  deadline_scope="verified external PID and descendants; its process group is not assumed private")
        fixed_sources = {name: sha(ROOT / name) for name in SOURCES}
        input_paths = sorted({j[k] for j in jobs for k in ("model", "data")})
        inputs = {path: sha(path) for path in input_paths}
        states = {}
        for j in jobs:
            out = Path(j["out"])
            if out.exists() and any(out.iterdir()) and j["id"] not in external:
                result = validate_completed(j, out, dict(source_sha256=fixed_sources, input_sha256=inputs))
                states[j["id"]] = dict(status="complete", adopted_completed_utc=now(), final=result,
                                        log=str(out.with_suffix(".log")), returncode=None,
                                        exit_observation="completed by hand before registration; adopted from its own final.json and config.json")
                continue
            if j["id"] in external:
                # Adoption must precede its final scientific outcome.
                if (out / "final.json").exists():
                    raise ValueError("external adoption must be registered before final outcome")
                cfg = out / "config.json"
                argv = Path(f"/proc/{external[j['id']]['pid']}/cmdline").read_bytes().split(b"\0")
                validate_external(j, argv, cfg, dict(source_sha256=fixed_sources, input_sha256=inputs))
                states[j["id"]] = dict(status="external_running", declared_utc=now(),
                                        log=str(out.with_suffix(".log")), **external[j["id"]])
            else:
                states[j["id"]] = dict(status="pending")
        manifest = dict(protocol="shared_decoded_cnn_v1", created_utc=now(), deadline_utc=DEADLINE,
                        n_jobs=30, jobs=jobs, source_sha256=fixed_sources, input_sha256=inputs,
                        external_declared=external, gate_jepa_selection="modelseed1 favorably selected using prior development; not an independent encoder replication",
                        ordering="six gate decoded-real controls first, then world/seed/family dreams; unavailable heads/packets are bypassed",
                        stopping="shared UTC deadline only; fixed budgets; failures retained; no retries or reward-based decisions")
        write(q / "campaign.json", manifest)
        write(q / "status.json", dict(updated_utc=now(), jobs=states, dependency_sha256={}))
    print(json.dumps(dict(prepared=str(q / "campaign.json"), jobs=30, external=list(external))), flush=True)


def check_sources(manifest):
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"frozen source changed: {name}")


def check_inputs(manifest):
    for path, expected in manifest["input_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError(f"frozen model or data changed: {path}")


def final_result(job):
    path = Path(job["out"]) / "final.json"
    result = read(path) if path.exists() else None
    good = bool(result and result.get("budget_complete") and not result.get("smoke")
                and result.get("updates") == job["updates"]
                and result.get("nominal_positions") == job["nominal_positions"]
                and result.get("family") == job["family"] and result.get("world") == job["world"]
                and result.get("train_source") == job["train_source"] and result.get("policy_seed") == job["policy_seed"])
    return good, result


def pin_dependency(path, state):
    path = str(path)
    digest = sha(path)
    old = state["dependency_sha256"].setdefault(path, digest)
    if old != digest:
        raise RuntimeError(f"registered dependency changed: {path}")
    return digest


def ready(job, state, manifest):
    packet = Path(job["packet"])
    if not packet.is_file():
        return False
    pin_dependency(packet, state)
    if job["heads"]:
        head = Path(job["heads"])
        report = head.with_suffix(".manifest.json")
        if not head.is_file() or not report.is_file():
            return False
        info = read(report)
        digest = pin_dependency(head, state)
        pin_dependency(report, state)
        if (digest != info["output_sha256"] or info["config"]["model_sha256"] != manifest["input_sha256"][job["model"]]
                or info["config"]["train_sha256"] != manifest["input_sha256"][job["data"]]
                or info["heads"]["transitions"] != 500_000 or info["heads"]["steps"] != 6000):
            raise RuntimeError("head dependency does not match the registered full-label refit")
    return True


def refresh_external(manifest, state, deadline):
    for job in manifest["jobs"]:
        record = state["jobs"][job["id"]]
        if record["status"] != "external_running":
            continue
        if alive(record["pid"], record["pid_starttime"]):
            if time.time() >= deadline:
                kill_external_tree(record)
                record.update(status="deadline_timeout", finished_utc=now(), external=True,
                              final=final_result(job)[1])
            continue
        good, result = final_result(job)
        record.update(status="complete" if good else "external_failed", finished_utc=now(),
                      final=result, returncode=None, exit_observation="external PID exited; exit code unavailable")


def refresh_orphans(manifest, state, deadline):
    for job in manifest["jobs"]:
        record = state["jobs"][job["id"]]
        if record["status"] != "running" or alive(record["worker_pid"], record.get("worker_starttime")):
            continue
        pid = record.get("child_pid")
        child_alive = pid is not None and alive(pid, record.get("child_starttime"))
        if child_alive and time.time() < deadline:
            continue  # adopt monitoring only; never rerun or overwrite this job
        if child_alive:
            if os.getpgid(pid) != pid:
                raise RuntimeError("owned child unexpectedly lost its private process group")
            os.killpg(pid, signal.SIGKILL)
        good, result = final_result(job)
        record.update(status="deadline_timeout" if child_alive else "complete" if good else "failed",
                      finished_utc=now(), final=result, returncode=None, orphan_reconciled=True)


def claim(a, manifest, deadline):
    q = a.out / "queue"
    with locked(q / "claims.lock"):
        state = read(q / "status.json")
        refresh_external(manifest, state, deadline)
        refresh_orphans(manifest, state, deadline)
        if time.time() >= deadline:
            for r in state["jobs"].values():
                if r["status"] == "pending":
                    r.update(status="deadline_not_started", finished_utc=now())
            selected = None
        elif any(r["status"] == "external_running" and r["gpu"] == a.gpu for r in state["jobs"].values()):
            selected = None
        else:
            selected = next((j for j in manifest["jobs"] if state["jobs"][j["id"]]["status"] == "pending"
                             and ready(j, state, manifest)), None)
            if selected:
                state["jobs"][selected["id"]] = dict(status="running", worker_pid=os.getpid(),
                    worker_starttime=process_identity(os.getpid())["starttime"], gpu=a.gpu, claimed_utc=now())
        state["updated_utc"] = now()
        write(q / "status.json", state)
        done = all(r["status"] in TERMINAL for r in state["jobs"].values())
        return selected, done


def run_job(a, job, deadline):
    q = a.out / "queue"
    logs = q / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / (job["id"] + ".log")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=".90", XLA_FLAGS="--xla_gpu_enable_triton_gemm=false")
    started = time.time()
    process = None
    record = dict(log=str(path), started_utc=now(), command=job["command"])
    try:
        with path.open("x") as log:
            process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with locked(q / "claims.lock"):
                state = read(q / "status.json")
                state["jobs"][job["id"]].update(child_pid=process.pid, process_group=process.pid,
                    child_starttime=process_identity(process.pid)["starttime"], **record)
                write(q / "status.json", state)
            try:
                code = process.wait(timeout=max(.01, deadline - time.time()))
                good, result = final_result(job)
                record.update(status="complete" if code == 0 and good else "failed", returncode=code, final=result)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
                record.update(status="deadline_timeout", returncode=code, final=final_result(job)[1])
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        record.update(status="worker_interrupted", error=f"{type(error).__name__}: {error}")
        if not isinstance(error, Exception):
            raise
    finally:
        record.update(seconds=round(time.time() - started, 3), finished_utc=now())
        with locked(q / "claims.lock"):
            state = read(q / "status.json")
            state["jobs"][job["id"]].update(record)
            state["updated_utc"] = now()
            write(q / "status.json", state)
        print("[finish] " + json.dumps(dict(id=job["id"], **record)), flush=True)


def worker(a):
    if a.gpu is None or a.admission_file is None:
        raise ValueError("worker requires --gpu and --admission-file")
    q = a.out / "queue"
    manifest = read(q / "campaign.json")
    admission = read(a.admission_file)
    if admission.get("admitted") is not True or admission.get("basis") != "runtime_only":
        raise ValueError("worker requires an explicit runtime-only admission of all30 jobs")
    deadline = datetime.fromisoformat(manifest["deadline_utc"]).timestamp()
    check_sources(manifest)
    check_inputs(manifest)
    with locked(q / "claims.lock"):
        pinned = q / "admission.json"
        if pinned.exists() and read(pinned) != admission:
            raise ValueError("admission cannot change after a worker starts")
        if not pinned.exists():
            write(pinned, admission)
    with locked(q / f"gpu{a.gpu}.lock", nonblock=True):
        while True:
            check_sources(manifest)
            job, done = claim(a, manifest, deadline)
            if done:
                print("[finished] all30 statuses terminal; inspect complete versus failed/incomplete", flush=True)
                return
            if job:
                print("[start] " + job["id"] + " GPU" + a.gpu, flush=True)
                run_job(a, job, deadline)
            else:
                time.sleep(min(30, max(.1, deadline - time.time())))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/revision/shared_decoded"))
    p.add_argument("--gray-root", type=Path, default=Path("/dev/shm/redcube"))
    p.add_argument("--stoch-root", type=Path, default=Path("/home/rldoeswork/personal/stoch"))
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--plan", action="store_true", help="print the matrix without filesystem mutations or GPU imports")
    p.add_argument("--external-running", action="append", default=[], metavar="JOB:GPU:PID")
    p.add_argument("--gpu", choices=("0", "1"))
    p.add_argument("--admission-file", type=Path)
    return p.parse_args()


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"worker interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    args = arguments()
    if args.plan:
        print(json.dumps(dict(deadline_utc=DEADLINE, jobs=matrix(args)), indent=2))
    elif args.prepare:
        prepare(args)
    else:
        worker(args)
