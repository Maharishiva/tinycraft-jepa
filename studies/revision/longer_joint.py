"""Fresh 34,000-update joint-JEPA budget sensitivity; preparation is CPU-only.

Production changes only Config.steps relative to the original tile_factored
8,500-update arm. This stretches both the warmup and cosine schedule by four.
The original 512-dimensional model, data, objective and optimizer are retained.
This is not checkpoint continuation: saved checkpoints omit optimizer state.

Launch requires an external, prospectively frozen whole-three-seed admission
record after the priority shared-CNN30 campaign. The parent queue must also
enforce the absolute 2026-09-20T04:30:00Z deadline on fits AND final readouts.
No GPU work is launched by --self-test.
"""
import argparse
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import train as T
from studies.revision import target_gradient as G


PROTOCOL = "longer_joint_budget_v1"
WORLD = "gray_frame_wind"
SEEDS = (0, 1, 2)
SHORT_STEPS, LONG_STEPS = 8500, 34000
CHECKPOINT_STEPS = (0, 8500, 17000, 25500, 34000)
DEADLINE_UTC = "2026-09-20T04:30:00+00:00"
FROZEN_TRAINER_SHA256 = "4dc9280ea077e954096bc7968f3ab19031f82369124b4de62acf5e5e6a14d488"
ROOT = Path(T.__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def production_config(seed):
    require(seed in SEEDS, "registered model seeds are 0,1,2")
    return replace(G.production_config(seed), steps=LONG_STEPS)


def learning_rate_schedule(cfg):
    # Exact train.py expression, including optimizer counter zero's zero LR.
    warm = max(1, int(cfg.warmup * cfg.steps))
    return optax.warmup_cosine_decay_schedule(0.0, cfg.lr, warm, cfg.steps, end_value=0.0)


def sources():
    require(G.sha256(G.__file__) == FROZEN_TRAINER_SHA256, "frozen target-gradient driver changed")
    paths = (Path(__file__).resolve(), Path(G.__file__).resolve(), Path(T.__file__).resolve(),
             ROOT / "studies/revision/target_gradient_probe.py", ROOT / "studies/stoch/train_arms.sh",
             ROOT / "pyproject.toml", ROOT / "research/longer_joint_protocol.md")
    return {str(path.relative_to(ROOT)): G.sha256(path) for path in paths}


def rng_audit(seed, steps, batch, num_windows):
    """Measure length-dependent split behavior; never splice or alter the keys."""
    key = jax.random.PRNGKey(seed + 7)
    keys = jax.random.split(key, steps)
    short_keys = jax.random.split(key, SHORT_STEPS)
    ids = G.sample_ids(keys, batch, num_windows)
    short_ids = G.sample_ids(short_keys, batch, num_windows)
    keys_np, short_np = np.asarray(keys), np.asarray(short_keys)
    count = min(steps, SHORT_STEPS)
    audit = {
        "jax_version": jax.__version__, "rng_implementation": str(jax.config.jax_default_prng_impl),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "seed_plus_seven": seed + 7, "comparison_steps": count,
        "long_split_length": steps, "short_split_length": SHORT_STEPS,
        "long_prefix_equals_regenerated_short": bool(np.array_equal(keys_np[:count], short_np[:count])),
        "long_sample_prefix_equals_regenerated_short": bool(np.array_equal(ids[:count], short_ids[:count])),
        "long_keys_sha256": G.array_sha256(keys_np), "short_keys_sha256": G.array_sha256(short_np),
        "long_prefix_keys_sha256": G.array_sha256(keys_np[:count]),
        "long_sample_ids_sha256": G.array_sha256(ids), "short_sample_ids_sha256": G.array_sha256(short_ids),
        "long_prefix_sample_ids_sha256": G.array_sha256(ids[:count]),
        "key_rule": "jax.random.split(jax.random.PRNGKey(seed+7), cfg.steps); no replacement or concatenation",
    }
    return keys, keys_np, ids, short_np, short_ids, audit


def check_reference(path, cfg):
    """Require the same-seed completed joint baseline, without reading a score."""
    path = Path(path)
    reference = json.loads(path.read_text())
    require(reference.get("protocol") == "paired_target_gradient_v1"
            and reference.get("status") == "complete" and reference.get("mode") == "joint"
            and reference.get("config_label") == WORLD and reference.get("completed_steps") == SHORT_STEPS,
            "reference must be the completed same-world 8,500-update joint fit")
    require(reference.get("cfg") == asdict(replace(cfg, steps=SHORT_STEPS)),
            "reference configuration differs beyond total update count")
    require(reference["sources"]["sha256"].get("studies/revision/target_gradient.py") == FROZEN_TRAINER_SHA256,
            "reference trainer source differs from the frozen baseline")
    require(reference["sources"]["sha256"].get("train.py") == G.sha256(T.__file__),
            "reference architecture/model source differs")
    sampling = path.parent / "sampling.npz"
    require(G.sha256(sampling) == reference["sampling_file_sha256"], "reference sampling file changed")
    with np.load(sampling, allow_pickle=False) as archive:
        recorded = {name: archive[name] for name in ("window_table", "update_keys", "sampled_window_ids")}
    for name, field in (("window_table", "window_table_sha256"), ("update_keys", "update_keys_sha256"),
                        ("sampled_window_ids", "sampled_window_ids_sha256")):
        require(G.array_sha256(recorded[name]) == reference[field], "reference array hash changed: " + name)
    return reference, recorded, {"path": str(path.resolve()), "sha256": G.sha256(path)}


def check_runtime_window(created, projected, deadline, seed, now):
    require(created.utcoffset() is not None and created.utcoffset().total_seconds() == 0,
            "admission creation time requires UTC")
    require(created.timestamp() <= now and created.timestamp() + projected < deadline.timestamp(),
            "whole block with margin does not finish before 04:30 UTC")
    require(now < deadline.timestamp(), "04:30 UTC deadline has passed")
    if seed == 0:
        # Admission may have been written well before the assigned GPU clears.
        require(now + projected < deadline.timestamp(),
                "stale admission: full block no longer fits at actual first launch")


def admission(path, seed, out):
    """Validate the parent's runtime-only, whole-block admission record.

    A separate launcher must verify priority completion and measured timing,
    populate this record before seed zero, and freeze it across all three jobs.
    The trainer validates the pinned evidence and plan, not scientific scores.
    """
    path = Path(path)
    record = json.loads(path.read_text())
    require(record.get("protocol") == PROTOCOL and record.get("admitted") is True,
            "the whole longer-joint block was not prospectively admitted")
    require(record.get("model_seeds") == list(SEEDS) and record.get("steps") == LONG_STEPS
            and record.get("world") == WORLD and record.get("readout_checkpoints") == [LONG_STEPS],
            "admission must register all three full fits and their final readouts")
    require(record.get("deadline_utc") == DEADLINE_UTC, "absolute deadline must be 04:30 UTC")
    require(record.get("selection_basis") == "runtime only; no model or readout outcomes", "invalid admission basis")
    require(record.get("exclusive_gpu_confirmed") is True, "exclusive idle allocation was not confirmed")
    priority = record.get("priority_shared_cnn30", {})
    require(priority.get("status") == "complete" and priority.get("completed") == 30
            and priority.get("expected") == 30, "priority shared-CNN30 campaign is incomplete")
    proof = Path(priority["evidence_path"])
    require(G.sha256(proof) == priority["evidence_sha256"], "priority completion evidence changed")
    # This evidence is a launcher-created audit of actual priority job artifacts.
    evidence = json.loads(proof.read_text())
    require(evidence.get("status") == "complete" and len(evidence.get("jobs", [])) == 30
            and len({item["id"] for item in evidence["jobs"]}) == 30
            and all(item.get("status") == "complete" for item in evidence["jobs"]),
            "priority evidence must enumerate 30 unique completed jobs")
    timing = record["timing"]
    for name in ("three_fits_seconds", "three_final_readouts_seconds", "artifact_checks_seconds"):
        require(np.isfinite(timing[name]) and timing[name] > 0, "invalid measured timing projection: " + name)
    require(np.isfinite(timing["margin_factor"]) and timing["margin_factor"] >= 1.25,
            "runtime admission needs at least 25 percent margin")
    projected = sum(timing[name] for name in
                    ("three_fits_seconds", "three_final_readouts_seconds", "artifact_checks_seconds")) * timing["margin_factor"]
    require(abs(projected - timing["projected_seconds_with_margin"]) <= 1e-6 * max(1, projected),
            "runtime admission arithmetic differs")
    require(timing.get("measurement_artifacts"), "runtime measurements must be pinned")
    for item in timing["measurement_artifacts"]:
        require(G.sha256(item["path"]) == item["sha256"], "runtime measurement artifact changed")
    created = datetime.fromisoformat(record["created_utc"])
    deadline = datetime.fromisoformat(DEADLINE_UTC)
    check_runtime_window(created, projected, deadline, seed, time.time())
    jobs = record["jobs"]
    require(len(jobs) == 3 and sorted(item["seed"] for item in jobs) == list(SEEDS), "all three fits must be registered")
    job = next(item for item in jobs if item["seed"] == seed)
    require(Path(job["out"]).resolve() == Path(out).resolve(), "unregistered output path")
    require(record.get("sources") == sources(), "admitted source or protocol bytes changed")
    return {"path": str(path.resolve()), "sha256": G.sha256(path), "record": record}, deadline.timestamp()


def fit(data_npz, out, cfg, checkpoint_steps, reference=None, admission_record=None, deadline=None):
    """Copied original loop using the independently tested, frozen joint update.

    The production CLI fixes cfg and every checkpoint. Small CPU fixtures may
    call this function with an explicit reduced cfg; those are not endpoints.
    """
    fixed_sources = sources()
    out, data_npz = Path(out), Path(data_npz)
    require(not out.is_symlink() and not out.exists(), "output must be a new directory; no resume or overwrite")
    require(tuple(checkpoint_steps) == tuple(sorted(set(checkpoint_steps)))
            and checkpoint_steps[0] == 0 and checkpoint_steps[-1] == cfg.steps,
            "checkpoint list must start at zero and end at the full update count")
    require(all(step % cfg.chunk == 0 for step in checkpoint_steps[:-1]), "checkpoint must be on a chunk boundary")
    t0 = time.time()
    data_sha = G.sha256(data_npz)
    with np.load(data_npz, allow_pickle=False) as archive:
        identity = G.validate_data_identity(json.loads(str(archive["meta"])), WORLD, cfg)
        frames_np = archive["frames"]
        actions_np = archive["actions"].astype(np.int32)
        dones_np = archive["dones"].astype(np.int64)
    require(actions_np.size == cfg.data_budget, "requires the entire fixed training transition budget")
    require(actions_np.ndim == 2 and dones_np.shape == actions_np.shape, "unaligned actions/dones")
    require(frames_np.dtype == np.uint8 and frames_np.shape == (actions_np.shape[0] + 1, actions_np.shape[1], T.H, T.W, 3),
            "invalid frame array")
    require(np.all((actions_np >= 0) & (actions_np < cfg.n_actions)), "invalid action vocabulary")
    table_np = T.valid_starts(dones_np, cfg.history + cfg.num_preds)
    require(len(table_np) > 0, "no reset-free windows")
    data_arrays = {name: G.array_sha256(value) for name, value in
                   (("frames", frames_np), ("actions_int32", actions_np), ("dones_int64", dones_np))}
    data_shape = {"frames": list(frames_np.shape), "actions": list(actions_np.shape),
                  "windows": len(table_np), "span": cfg.history + cfg.num_preds}
    frames, actions, table = jnp.asarray(frames_np), jnp.asarray(actions_np), jnp.asarray(table_np)
    del frames_np, actions_np, dones_np
    model = T.LeWM(cfg, T.rngs(cfg.seed))
    initial_sha = G.weights_sha256(model)
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    schedule = learning_rate_schedule(cfg)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=cfg.weight_decay))
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
    run = G.make_run(cfg, "joint")
    keys, keys_np, ids, short_keys, short_ids, rng = rng_audit(cfg.seed, cfg.steps, cfg.batch, len(table_np))
    packages = {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "optax", "numpy")}
    if reference is not None:
        baseline, recorded, reference_identity = reference
        for name, actual in (("data_sha256", data_sha), ("data_array_sha256", data_arrays),
                             ("initial_weights_sha256", initial_sha), ("window_table_sha256", G.array_sha256(table_np))):
            require(baseline[name] == actual, "long/short baseline identity differs: " + name)
        require(baseline.get("packages") == packages
                and baseline.get("jax_default_matmul_precision") == jax.config.jax_default_matmul_precision
                and baseline.get("jax_enable_x64") == jax.config.jax_enable_x64,
                "baseline package versions or JAX precision configuration differ")
        require(np.array_equal(recorded["update_keys"], short_keys)
                and np.array_equal(recorded["sampled_window_ids"], short_ids),
                "actual original keys/windows differ from regenerated short-schedule RNG")
        require(rng["long_prefix_equals_regenerated_short"] and rng["long_sample_prefix_equals_regenerated_short"],
                "runtime RNG does not preserve the registered matched prefix; do not splice keys or silently unpair")
        rng["long_prefix_equals_actual_baseline_keys"] = bool(np.array_equal(keys_np[:SHORT_STEPS], recorded["update_keys"]))
        rng["long_prefix_equals_actual_baseline_samples"] = bool(np.array_equal(ids[:SHORT_STEPS], recorded["sampled_window_ids"]))
    else:
        reference_identity = None
    out.mkdir(parents=True)
    rates = np.asarray(jax.vmap(schedule)(jnp.arange(cfg.steps)))
    np.savez(out / "sampling.npz", window_table=table_np, update_keys=keys_np, sampled_window_ids=ids,
             optimizer_learning_rates=rates)
    manifest = {
        "protocol": PROTOCOL, "status": "running", "mode": "joint", "config_label": WORLD,
        "cfg": asdict(cfg), "params": n_params, "model_seed": cfg.seed,
        "data_path": str(data_npz.resolve()), "data_sha256": data_sha, "data_identity": identity,
        "data_array_sha256": data_arrays, "data_shape": data_shape,
        "initial_weights_sha256": initial_sha, "window_table_sha256": G.array_sha256(table_np),
        "update_keys_sha256": G.array_sha256(keys_np), "sampled_window_ids_sha256": G.array_sha256(ids),
        "learning_rates_sha256": G.array_sha256(rates), "sampling_file_sha256": G.sha256(out / "sampling.npz"),
        "rng_audit": rng, "sources": fixed_sources, "reference": reference_identity, "admission": admission_record,
        "schedule": {"steps": cfg.steps, "warmup_steps": max(1, int(cfg.warmup * cfg.steps)),
                     "optimizer_counter": "zero-based; first update LR=0", "peak_lr": cfg.lr,
                     "warmup_fraction": cfg.warmup, "end_lr": 0.0,
                     "comparison": "fresh fit with stretched warmup and cosine; not a continuation"},
        "fixed_checkpoint_steps": list(checkpoint_steps), "checkpoints": [],
        "loss_columns": ["prediction_mse", "sigreg"], "checkpoint_selection": "final34000 only; no outcome selection",
        "training_fields": ["frames", "actions", "dones_for_window_boundaries_only"],
        "python": platform.python_version(),
        "packages": packages,
        "devices": [str(device) for device in jax.devices()],
        "execution_environment": {name: os.environ.get(name) for name in
                                  ("XLA_FLAGS", "JAX_DEFAULT_MATMUL_PRECISION", "JAX_ENABLE_X64", "CUDA_VISIBLE_DEVICES")},
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "jax_enable_x64": jax.config.jax_enable_x64,
    }
    G.write_manifest(out, manifest)
    history = []
    t1 = time.time()

    def checkpoint(done):
        path = out / ("model.npz" if done == cfg.steps else f"model_step{done:06d}.npz")
        meta = {"cfg": asdict(cfg), "params": n_params, "train_s": round(time.time() - t0, 1),
                "loop_s": round(time.time() - t1, 1), "data_sha256": data_sha,
                "source_sha256": fixed_sources["studies/revision/longer_joint.py"], "sources": fixed_sources,
                "experiment": PROTOCOL, "target_gradient_mode": "joint", "config_label": WORLD,
                "completed_steps": done, "initial_weights_sha256": initial_sha,
                "final_pred_loss": float(history[-1][-1, 0]) if history else None,
                "final_sigreg": float(history[-1][-1, 1]) if history else None,
                "fixed_checkpoint_steps": list(checkpoint_steps), "schedule": manifest["schedule"],
                "reference": reference_identity, "rng_audit": rng}
        T.save(path, model, meta)
        manifest["checkpoints"].append({"step": done, "path": path.name, "sha256": G.sha256(path),
                                         "weights_sha256": G.weights_sha256(model)})
        np.savez(out / "losses.npz", history=np.concatenate(history) if history else np.empty((0, 2), np.float32))
        manifest.update(completed_steps=done, elapsed_s=round(time.time() - t0, 1))
        G.write_manifest(out, manifest)
        return meta

    checkpoint(0)
    for begin in range(0, cfg.steps, cfg.chunk):
        if deadline is not None and time.time() >= deadline:
            manifest.update(status="deadline_incomplete", completed_steps=begin)
            np.savez(out / "losses.npz", history=np.concatenate(history) if history else np.empty((0, 2), np.float32))
            G.write_manifest(out, manifest)
            raise TimeoutError("04:30 UTC deadline; preserve incomplete run")
        losses = run(model, opt, frames, actions, table, keys[begin:begin + cfg.chunk])
        history.append(np.asarray(jax.device_get(losses)))
        done = begin + history[-1].shape[0]
        print(f"[train] {done}/{cfg.steps} pred {history[-1][-1, 0]:.4f} sigreg {history[-1][-1, 1]:.4f}"
              f"  {time.time() - t1:.1f}s", flush=True)
        if not np.isfinite(history[-1]).all():
            manifest.update(status="failed_nonfinite", completed_steps=done)
            np.savez(out / "losses.npz", history=np.concatenate(history))
            G.write_manifest(out, manifest)
            raise FloatingPointError("nonfinite loss; no outcome-dependent restart")
        if done in checkpoint_steps:
            meta = checkpoint(done)
    require(sources() == fixed_sources, "source or protocol changed during training")
    if deadline is not None and time.time() >= deadline:
        manifest.update(status="deadline_incomplete")
        G.write_manifest(out, manifest)
        raise TimeoutError("04:30 UTC deadline passed before fit completion checks")
    manifest.update(status="complete", loss_file_sha256=G.sha256(out / "losses.npz"),
                    final_checkpoint_sha256=manifest["checkpoints"][-1]["sha256"],
                    final_weights_sha256=manifest["checkpoints"][-1]["weights_sha256"])
    G.write_manifest(out, manifest)
    return model, meta


def self_test():
    require(all(device.platform == "cpu" for device in jax.devices()), "self-test requires JAX_PLATFORMS=cpu")
    cfg = production_config(0)
    differences = {name for name, value in asdict(cfg).items() if value != asdict(G.production_config(0))[name]}
    require(differences == {"steps"} and cfg.steps == 34000 and cfg.tokens * cfg.token_dim == 512,
            "production architecture/hyperparameters changed")
    long_schedule, short_schedule = learning_rate_schedule(cfg), learning_rate_schedule(G.production_config(0))
    require(float(long_schedule(0)) == 0.0 and float(long_schedule(1)) < float(short_schedule(1)), "early schedule did not change")
    np.testing.assert_allclose(float(long_schedule(3400)), cfg.lr, rtol=1e-6)
    require(float(long_schedule(34000)) == 0.0, "cosine does not end at34000")
    created = datetime.fromisoformat("2026-09-20T02:00:00+00:00")
    deadline = datetime.fromisoformat(DEADLINE_UTC)
    check_runtime_window(created, 7200, deadline, 0, created.timestamp())
    try:
        check_runtime_window(created, 7200, deadline, 0,
                             datetime.fromisoformat("2026-09-20T04:20:00+00:00").timestamp())
    except ValueError as error:
        require("stale admission" in str(error), "wrong stale-admission rejection")
    else:
        raise AssertionError("a delayed first launch with insufficient time was admitted")
    for seed in SEEDS:
        *_, audit = rng_audit(seed, LONG_STEPS, 3, 101)
        require(audit["long_prefix_equals_regenerated_short"] and audit["long_sample_prefix_equals_regenerated_short"],
                "CPU runtime does not support the registered prefix pairing")
        print(json.dumps({"seed": seed, "rng_audit": audit}), flush=True)
    # Actual T.train comparison includes one nonzero-learning-rate optimizer
    # update: the original cosine schedule assigns LR=0 to update counter zero.
    small = replace(cfg, steps=2, chunk=1, batch=2, data_budget=10,
                    emb_dim=16, enc_depth=1, enc_heads=2, enc_res=1, token_hidden=32,
                    pred_width=16, pred_depth=1, pred_heads=2, history=2, sigreg_knots=5, sigreg_proj=8)
    with tempfile.TemporaryDirectory(prefix="longer-joint-cpu-") as directory:
        root = Path(directory)
        rng = np.random.default_rng(31)
        data = root / "train.npz"
        np.savez(data, meta=json.dumps({"env": "redcube", "config": WORLD, "split": "train", "num_actions": 4,
                                       "seed": 31, "spec": {"noise": "gray", "per_episode": False, "wind": True}}),
                 frames=rng.integers(0, 256, (6, 2, T.H, T.W, 3), dtype=np.uint8),
                 actions=rng.integers(0, 4, (5, 2), dtype=np.int16), dones=np.zeros((5, 2), bool))
        original, _ = T.train(data, root / "original/model.npz", small)
        copied, _ = fit(data, root / "longer", small, (0, 1, 2))
        np.testing.assert_array_equal(np.load(root / "original/losses.npz")["history"],
                                      np.load(root / "longer/losses.npz")["history"])
        require(G.weights_sha256(original) == G.weights_sha256(copied), "joint driver differs from T.train")
        loaded, metadata = T.load_model(root / "longer/model.npz")
        require(G.weights_sha256(loaded.model) == G.weights_sha256(copied) and metadata["completed_steps"] == 2,
                "checkpoint roundtrip failed")
        try:
            fit(data, root / "expired", small, (0, 1, 2), deadline=0.0)
        except TimeoutError:
            incomplete = json.loads((root / "expired/manifest.json").read_text())
            require(incomplete["status"] == "deadline_incomplete" and incomplete["completed_steps"] == 0,
                    "expired job performed an update or reported completion")
        else:
            raise AssertionError("expired deadline was accepted")
    print("PASS production-only-steps-change;34k-schedule;all3-RNG-prefixes;two-update-exact-T.train-parity;checkpoint-roundtrip;expired-deadline;stale-admission", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--admission-json", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(all(value is not None for value in (args.data, args.out, args.seed, args.reference_manifest, args.admission_json)),
            "production requires data,out,seed,reference-manifest,admission-json")
    cfg = production_config(args.seed)
    record, deadline = admission(args.admission_json, args.seed, args.out)
    reference = check_reference(args.reference_manifest, cfg)
    job = next(job for job in record["record"]["jobs"] if job["seed"] == args.seed)
    require(str(args.reference_manifest.resolve()) == job["reference_manifest"]
            and G.sha256(args.reference_manifest) == job["reference_manifest_sha256"],
            "reference manifest differs from prospectively registered job")
    fit(args.data, args.out, cfg, CHECKPOINT_STEPS, reference, record, deadline)


if __name__ == "__main__":
    main()
