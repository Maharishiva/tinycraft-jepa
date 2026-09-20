"""Run two predeclared exact-recipe Craftax model-seed replications.

No source, data, objective, architecture, budget or evaluation settings change.
Only cfg.seed changes from the existing seed-0 checkpoint. Every completed
model is evaluated and retained regardless of its score; no heads are fitted.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.out.exists():
        parser.error("output exists; preserve it and use a fresh path")
    import numpy as np

    reference = json.loads(args.reference_report.read_text())
    ref_model = Path(reference["model"])
    ref_meta = reference["model_metadata"]
    cfg0 = ref_meta["cfg"]
    assert cfg0["seed"] == 0 and cfg0["steps"] == 8500 and cfg0["data_budget"] == 500000
    assert cfg0["tokens"] == 16 and cfg0["token_dim"] == 32 and cfg0["n_actions"] == 17
    assert cfg0["encoder"] == "tile" and cfg0["predictor"] == "factored"
    assert sha256(args.data) == ref_meta["data_sha256"]
    assert sha256(ref_model) == reference["model_sha256"]
    with np.load(ref_model) as archive:
        assert json.loads(str(archive["meta"])) == ref_meta
    for package, version in reference["versions"].items():
        assert importlib.metadata.version(package) == version, (package, version)

    def check_sources():
        for rel, expected in reference["source_sha256"].items():
            assert sha256(root / rel) == expected, f"source changed: {rel}"
        assert sha256(ref_model) == reference["model_sha256"], "seed-0 checkpoint changed"

    def gpu_idle():
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if processes:
            raise RuntimeError(f"GPU already has compute processes: {processes}")

    check_sources()
    gpu_idle()
    args.out.mkdir(parents=True)
    plan = {"schema": "craftax-exact-seed-replication-v1", "created_unix": time.time(),
            "reference_report": str(args.reference_report.resolve()),
            "reference_report_sha256": sha256(args.reference_report),
            "reference_model": str(ref_model), "reference_model_sha256": sha256(ref_model),
            "data": str(args.data.resolve()), "data_sha256": sha256(args.data),
            "source_sha256": reference["source_sha256"], "versions": reference["versions"],
            "launcher_sha256": sha256(__file__), "training_seeds": [1, 2],
            "reference_training_seed": 0, "config_changes_allowed": ["seed"],
            "no_decoder_or_heads": True, "selection": "all predeclared models retained regardless of scores",
            "eval_seeds": [101, 202, 303], "eval_envs_per_seed": 128, "eval_horizon": 6,
            "budget_rule": "exactly 8500 updates, measured elapsed time; no strict 600-second claim",
            "configs": {str(seed): {**cfg0, "seed": seed} for seed in [1, 2]}}
    save(args.out / "plan.json", plan)
    status = {"phase": "prepared", "completed": [], "started_unix": time.time()}
    save(args.out / "status.json", status)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    env.pop("JAX_PLATFORMS", None)

    def execute(command, log, phase, seed):
        check_sources()
        gpu_idle()
        status.update(phase=phase, model_seed=seed, phase_started_unix=time.time(), command=command)
        save(args.out / "status.json", status)
        print(f"[{phase}] model_seed={seed}; log={log}", flush=True)
        with log.open("w") as stream:
            process = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            status.update(phase="failed", failed_phase=phase, returncode=process.returncode)
            save(args.out / "status.json", status)
            raise RuntimeError(f"{phase} seed {seed} failed; see {log}")
        check_sources()

    for seed in [1, 2]:
        cfg = plan["configs"][str(seed)]
        assert {k for k in cfg if cfg[k] != cfg0[k]} == {"seed"}
        run = args.out / f"seed{seed}"
        run.mkdir()
        model = run / "model.npz"
        save(run / "predeclared_config.json", cfg)
        command = [sys.executable, str(root / "train.py"), "--data", str(args.data.resolve()),
                   "--out", str(model.resolve())]
        for name, value in cfg.items():
            command.extend(["--" + name.replace("_", "-"), str(value)])
        execute(command, run / "train.log", "training", seed)
        with np.load(model) as archive:
            meta = json.loads(str(archive["meta"]))
            assert not any(k.startswith(("decoder.", "heads.")) for k in archive.files)
        assert meta["cfg"] == cfg
        assert meta["params"] == ref_meta["params"]
        assert meta["data_sha256"] == ref_meta["data_sha256"]
        assert meta["source_sha256"] == ref_meta["source_sha256"]
        with np.load(run / "losses.npz") as archive:
            assert archive["history"].shape == (8500, 2)
        identity = {"model_seed": seed, "model": str(model.resolve()), "sha256": sha256(model),
                    "metadata": meta, "config_equality_except_seed": True}
        save(run / "identity.json", identity)
        evaluation = run / "goal_eval.json"
        command = [sys.executable, str(root / "studies/revision/craftax_eval.py"),
                   "--model", str(model.resolve()), "--out", str(evaluation.resolve()),
                   "--seeds", "101", "202", "303", "--envs", "128", "--horizon", "6",
                   "--warmup", "8", "--pop", "512", "--elite", "32", "--iters", "6"]
        execute(command, run / "goal_eval.log", "evaluation", seed)
        measured = json.loads(evaluation.read_text())
        assert measured["model_sha256"] == identity["sha256"]
        assert measured["source_sha256"] == reference["source_sha256"]
        assert measured["versions"] == reference["versions"]
        assert len(measured["goal_runs"]) == len(reference["goal_runs"]) == 3
        for new, old in zip(measured["goal_runs"], reference["goal_runs"]):
            for key in ("target_positions", "reference_actions", "start_positions"):
                assert new[key] == old[key], f"environment episode mismatch: {key}"
            for baseline in ("noop", "random", "reference_replay"):
                assert new["rows"][baseline] == old["rows"][baseline]
                assert new["traces"][baseline] == old["traces"][baseline]
        status["completed"].append({"model_seed": seed, "model_sha256": identity["sha256"],
                                    "evaluation": str(evaluation.resolve()),
                                    "evaluation_sha256": sha256(evaluation),
                                    "matched_environment_episode_check": "passed"})
        save(args.out / "status.json", status)
    status.update(phase="complete", finished_unix=time.time())
    save(args.out / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
