"""Prospectively matched, frozen-encoder value-aware predictor continuation.

This is PVE-inspired, not a proper-value-equivalence implementation.  Run from
the repository root with ``python -m studies.revision.pve --help``.  ``--smoke``
is a non-scientific execution check, and never establishes teacher adequacy.
Only observed pixels/actions/rewards/dones enter fitting. See research/pve_protocol.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def episode_targets(rewards, dones, gamma, seed=417):
    """Complete-episode MC targets; exclude every unfinished recording suffix.

    The dataset must begin at reset. Unique episode IDs include stream identity.
    Each entire episode belongs to exactly one of fit-A/fit-B/validation (40/40/20%).
    No privileged episode/state annotations are read.
    """
    rewards, dones = np.asarray(rewards), np.asarray(dones, bool)
    if rewards.ndim != 2 or rewards.shape != dones.shape:
        raise ValueError("rewards/dones must be equal [time, environment] arrays")
    targets = np.full(rewards.shape, np.nan, np.float32)
    ids = np.full(rewards.shape, -1, np.int64)
    folds = np.full(rewards.shape, -1, np.int8)
    rng = np.random.default_rng(seed)
    eid = 0
    for env in range(rewards.shape[1]):
        begin = 0
        for end in np.flatnonzero(dones[:, env]):
            acc = 0.0
            for t in range(int(end), begin - 1, -1):
                acc = float(rewards[t, env]) + gamma * acc
                targets[t, env] = acc
            ids[begin:end + 1, env] = eid
            folds[begin:end + 1, env] = rng.choice(3, p=[.4, .4, .2])
            eid += 1
            begin = int(end) + 1
    return targets, ids, folds


def window_table(dones, targets, folds, history, horizon, train=False):
    """Contexts and future targets all in the same non-terminal episode.

    A final target frame must itself have a complete MC suffix; terminal/reset
    transitions cannot be latent targets because the archive stores reset pixels.
    """
    span = history + horizon
    n, envs = dones.shape
    if span > n:
        return np.empty((0, 2), np.int32)
    starts = np.arange(n - span + 1)
    cs = np.concatenate([np.zeros((1, envs), np.int64), np.cumsum(dones, axis=0)])
    valid = cs[starts + span - 1] == cs[starts]
    valid &= np.isfinite(targets[starts + span - 1])
    if train:
        valid &= folds[starts] < 2
        valid &= folds[starts] >= 0
    rows, cols = np.nonzero(valid)
    return np.stack([starts[rows], cols], -1).astype(np.int32)


def json_write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def self_test():
    r = np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.], [5., 50.]])
    d = np.array([[0, 0], [1, 0], [0, 1], [1, 0], [0, 0]], bool)
    g, eid, folds = episode_targets(r, d, .5)
    np.testing.assert_allclose(g[:, 0], [2., 2., 5., 4., np.nan])
    np.testing.assert_allclose(g[:, 1], [27.5, 35., 30., np.nan, np.nan])
    assert eid[0, 0] == eid[1, 0] != eid[2, 0]
    assert eid[0, 0] != eid[0, 1]
    assert np.all(folds[:2, 0] == folds[0, 0])
    tab = window_table(d, g, folds, 1, 1)
    assert set(map(tuple, tab)) == {(0, 0), (2, 0), (0, 1), (1, 1)}
    # Every finite suffix ends at an observed done, with no recorded-tail bootstrap.
    assert np.all(eid[~np.isfinite(g)] == -1)
    print("self-test passed: exact MC alignment, episode split, terminal exclusion")


def runtime():
    global jax, jnp, nnx, optax, T
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax
    import train as T


def parameter_hash(module):
    h = hashlib.sha256()
    for group in T._weights(module):
        for x in jax.tree.leaves(group):
            a = np.asarray(jax.device_get(x))
            h.update(str(a.dtype).encode())
            h.update(np.asarray(a.shape, np.int64).tobytes())
            h.update(a.tobytes())
    return h.hexdigest()


def classes():
    class Value(nnx.Module):
        def __init__(self, dim, hidden, seed):
            self.l1 = nnx.Linear(dim, hidden, rngs=T.rngs(seed))
            self.l2 = nnx.Linear(hidden, hidden, rngs=T.rngs(seed + 1))
            self.out = nnx.Linear(hidden, 1, rngs=T.rngs(seed + 2))

        def __call__(self, z):
            return self.out(nnx.gelu(self.l2(nnx.gelu(self.l1(z)))))[..., 0]

    class PredictivePart(nnx.Module):
        """Only these three original modules may receive gradients."""
        def __init__(self, model):
            self.predictor = model.predictor
            self.pred_proj = model.pred_proj
            self.action_encoder = model.action_encoder
            self.n_actions = model.n_actions
            self.history = model.history

        def __call__(self, emb, actions, train=False):
            ae = self.action_encoder(jax.nn.one_hot(actions, self.n_actions, dtype=T.CDT)).astype(T.PDT)
            h = self.predictor(emb, ae, train)
            width = self.pred_proj.fc1.in_features
            return self.pred_proj(h.reshape(-1, width), train).reshape(emb.shape).astype(T.PDT)

    return Value, PredictivePart


def read_data(path, smoke):
    with np.load(path) as ar:
        meta = json.loads(str(ar["meta"]))
        if meta.get("policy") != "uniform random" or meta.get("env") != "stoch":
            raise ValueError("This prospective protocol requires stoch uniform-random data")
        keys = ("frames", "actions", "rewards", "dones")
        data = {k: ar[k] for k in keys}
    if smoke:
        data = {k: v[:501 if k == "frames" else 500, :8] for k, v in data.items()}
    if len(data["frames"]) != len(data["actions"]) + 1:
        raise ValueError("Expected T+1 frames for T outgoing actions")
    return data, meta


def encode(model, frames, chunk):
    @nnx.jit
    def enc(m, x):
        return m.encode(x[None], False)[0]
    flat = frames.reshape(-1, T.H, T.W, 3)
    parts = []
    for i in range(0, len(flat), chunk):
        parts.append(np.asarray(jax.device_get(enc(model, jnp.asarray(flat[i:i + chunk])))))
        if i % (chunk * 50) == 0:
            print(f"[encode] {i}/{len(flat)}", flush=True)
    return np.concatenate(parts).reshape(*frames.shape[:2], model.latent_dim)


def fit_value(Value, z, targets, folds, fold, args, mean, scale):
    t0 = time.time()
    critic = Value(z.shape[-1], args.head_hidden, args.seed + 110 + fold * 101)
    ids = np.flatnonzero((folds == fold).reshape(-1)).astype(np.int32)
    val_ids = np.flatnonzero((folds == 2).reshape(-1)).astype(np.int32)
    if len(ids) < args.batch or len(val_ids) < 100:
        raise ValueError("Too few complete train/validation samples")
    zflat = jnp.asarray(z[:-1].reshape(-1, z.shape[-1]))
    g = jnp.asarray(((targets.reshape(-1) - mean) / scale).astype(np.float32))
    idsj = jnp.asarray(ids)
    schedule = optax.warmup_cosine_decay_schedule(0., args.head_lr, max(1, args.head_steps // 20), args.head_steps)
    opt = nnx.Optimizer(critic, optax.chain(optax.clip_by_global_norm(1.), optax.adamw(schedule, weight_decay=1e-4)), wrt=nnx.Param)

    @nnx.jit
    def run(critic, opt, zflat, g, idsj, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def body(carry, key):
            critic, opt = carry
            take = idsj[jax.random.randint(key, (args.head_batch,), 0, len(ids))]
            loss, grad = nnx.value_and_grad(lambda c: jnp.mean((c(zflat[take]) - g[take]) ** 2))(critic)
            opt.update(critic, grad)
            return (critic, opt), loss
        return body((critic, opt), keys)[1]

    losses = []
    for start in range(0, args.head_steps, args.chunk):
        keys = jax.random.split(jax.random.PRNGKey(args.seed + 71 + fold), args.head_steps)[start:start + args.chunk]
        losses.extend(np.asarray(run(critic, opt, zflat, g, idsj, keys)).tolist())
        print(f"[value{fold}] {len(losses)}/{args.head_steps} loss={losses[-1]:.5f}", flush=True)
    p = np.concatenate([np.asarray(critic(zflat[v])) for v in np.array_split(val_ids, max(1, (len(val_ids) + 4095) // 4096))]) * scale + mean
    y = targets.reshape(-1)[val_ids]
    baseline = float(np.mean(targets.reshape(-1)[ids]))
    mse, base_mse = float(np.mean((p - y) ** 2)), float(np.mean((baseline - y) ** 2))
    corr = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 1e-10 and np.std(y) > 1e-10 else 0.
    quality = {"train_samples": len(ids), "validation_samples": len(val_ids), "mse": mse,
               "constant_mse": base_mse, "relative_mse_reduction": 1. - mse / max(base_mse, 1e-12),
               "correlation": corr, "prediction_std": float(p.std()), "target_std": float(y.std()),
               "passes_prespecified_gate": bool(mse < .9 * base_mse and corr > .2),
               "fit_and_validation_s": time.time() - t0}
    return critic, quality, np.asarray(losses, np.float32)


def rollout(part, zh, actions, horizon, train=False):
    predictions = []
    for h in range(horizon):
        nxt = part(zh, actions[:, h:h + part.history], train)[:, -1]
        predictions.append(nxt)
        zh = jnp.concatenate([zh[:, 1:], nxt[:, None]], 1)
    return jnp.stack(predictions, 1)


def continue_predictor(part, teacher, z, actions, targets, table, args, mean, scale, weight):
    z, actions, targets, table = map(jnp.asarray, (z, actions, targets, table))
    opt = nnx.Optimizer(part, optax.chain(optax.clip_by_global_norm(1.),
                        optax.adamw(optax.warmup_cosine_decay_schedule(0., args.lr,
                            max(1, args.steps // 20), args.steps), weight_decay=1e-3)), wrt=nnx.Param)
    hist, hor = part.history, args.horizon

    @nnx.jit
    def run(part, opt, teacher, z, actions, targets, table, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def body(carry, key):
            part, opt = carry
            idx = table[jax.random.randint(key, (args.batch,), 0, len(table))]
            t, e = idx[:, 0], idx[:, 1]
            ti = t[:, None] + jnp.arange(hist + hor)[None]
            zw = z[ti, e[:, None]]
            aw = actions[ti[:, :-1], e[:, None]]
            gt = (targets[ti[:, hist:], e[:, None]] - mean) / scale

            def loss_fn(p):
                pred = rollout(p, zw[:, :hist], aw, hor, True)
                lmse = jnp.mean((pred - zw[:, hist:]) ** 2)
                # Teacher parameters are not a differentiation argument; dV/dz remains.
                vpred = teacher(pred.reshape(-1, pred.shape[-1])).reshape(args.batch, hor)
                lvalue = jnp.mean((vpred - gt) ** 2)
                return lmse + weight * lvalue, jnp.stack([lmse, lvalue])

            (_, losses), grad = nnx.value_and_grad(loss_fn, has_aux=True)(part)
            opt.update(part, grad)
            return (part, opt), losses
        return body((part, opt), keys)[1]

    history = []
    keys = jax.random.split(jax.random.PRNGKey(args.seed + 901), args.steps)
    for start in range(0, args.steps, args.chunk):
        history.append(np.asarray(run(part, opt, teacher, z, actions, targets, table, keys[start:start + args.chunk])))
        print(f"[continue lambda={weight}] {start + len(history[-1])}/{args.steps} {history[-1][-1].tolist()}", flush=True)
    return np.concatenate(history)


def evaluate(part, teacher, auditor, z, actions, targets, ids, table, args, mean, scale):
    hist, hor = part.history, args.horizon
    selected = np.random.default_rng(args.seed + 4101).choice(len(table), min(args.eval_windows, len(table)), replace=False)
    table = table[selected]

    @nnx.jit
    def eval_batch(part, teacher, auditor, zh, aw, truth, target):
        pred = rollout(part, zh, aw, hor)
        read = lambda head, x: (head(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2]) * scale + mean)
        vt, va = read(teacher, pred), read(auditor, pred)
        rt, ra = read(teacher, truth), read(auditor, truth)
        # Model-only sensitivity, not a counterfactual accuracy score.
        repeated = jnp.repeat(zh, part.n_actions, 0)
        aa = jnp.repeat(aw[:, :hist], part.n_actions, 0)
        aa = aa.at[:, -1].set(jnp.tile(jnp.arange(part.n_actions), len(zh)))
        counter = part(repeated, aa)[:, -1].reshape(len(zh), part.n_actions, -1)
        cv = read(auditor, counter)
        return {"latent_mse": jnp.mean((pred - truth) ** 2, -1),
                "teacher_mc_mse": (vt - target) ** 2, "auditor_mc_mse": (va - target) ** 2,
                "teacher_target_value_mse": (vt - rt) ** 2, "auditor_target_value_mse": (va - ra) ** 2,
                "real_teacher_mc_mse": (rt - target) ** 2, "real_auditor_mc_mse": (ra - target) ** 2,
                "predicted_latent_rms": jnp.sqrt(jnp.mean(pred ** 2, -1)),
                "real_latent_rms": jnp.sqrt(jnp.mean(truth ** 2, -1)),
                "action_value_std": jnp.std(cv, -1),
                "action_latent_rms": jnp.sqrt(jnp.mean(jnp.var(counter, 1), -1))}

    pieces = []
    for start in range(0, len(table), args.eval_batch):
        block = table[start:start + args.eval_batch]
        ti = block[:, 0, None] + np.arange(hist + hor)[None]
        e = block[:, 1, None]
        zw = z[ti, e]
        aw = actions[ti[:, :-1], e]
        gt = targets[ti[:, hist:], e]
        result = eval_batch(part, teacher, auditor, *map(jnp.asarray, (zw[:, :hist], aw, zw[:, hist:], gt)))
        pieces.append(jax.tree.map(np.asarray, result))
    metrics = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    metrics["episode_id"] = ids[table[:, 0], table[:, 1]]
    metrics["start_env"] = table
    summary = {k: np.mean(v, axis=0).tolist() for k, v in metrics.items() if k not in ("episode_id", "start_env")}
    summary.update(windows=len(table), independent_episodes=int(len(np.unique(metrics["episode_id"]))))
    return summary, metrics


def paired_intervals(a, b, seed, draws=2000):
    if not np.array_equal(a["start_env"], b["start_env"]):
        raise ValueError("Evaluation windows must be paired")
    _, inv = np.unique(a["episode_id"], return_inverse=True)
    count = np.bincount(inv)
    rng = np.random.default_rng(seed)
    result = {}
    for key in ("latent_mse", "teacher_mc_mse", "auditor_mc_mse", "auditor_target_value_mse"):
        delta = b[key] - a[key]
        sums = np.stack([np.bincount(inv, weights=delta[:, h]) for h in range(delta.shape[1])], 1)
        samples = np.empty((draws, delta.shape[1]))
        for i in range(draws):
            ix = rng.integers(0, len(count), len(count))
            samples[i] = sums[ix].sum(0) / count[ix].sum()
        result[key] = {"value_aware_minus_mse": delta.mean(0).tolist(),
                       "episode_bootstrap_95_ci": np.quantile(samples, [.025, .975], axis=0).T.tolist()}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--model", type=Path)
    p.add_argument("--train", type=Path)
    p.add_argument("--eval", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--head-steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--head-batch", type=int, default=1024)
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--value-weight", type=float, default=.1)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--encode-chunk", type=int, default=1024)
    p.add_argument("--eval-windows", type=int, default=8192)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, k) is None for k in ("model", "train", "eval", "out")):
        p.error("--model, --train, --eval, --out are required")
    if args.smoke:
        args.out = args.out.with_name(args.out.name + "_smoke")
        args.steps, args.head_steps, args.batch, args.head_batch = 3, 3, 8, 16
        args.eval_windows, args.eval_batch, args.chunk, args.encode_chunk = 16, 8, 3, 128
    elif (args.value_weight, args.horizon) != (.1, 4):
        p.error("Prospective scientific protocol fixes lambda=.1 and horizon=4")
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    hashes = {k: sha256(getattr(args, k)) for k in ("model", "train", "eval")}
    if hashes["train"] == hashes["eval"]:
        raise ValueError("Training and held-out data must differ")
    protocol = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "sha256": hashes, "source_sha256": sha256(__file__), "status": "started",
                "scientific_run": not args.smoke, "name": "PVE-inspired value-aware continuation"}
    json_write(args.out / "manifest.json", protocol)
    runtime()
    Value, PredictivePart = classes()
    bundle, model_meta = T.load_model(args.model)
    protocol["original_model_metadata"] = model_meta
    json_write(args.out / "manifest.json", protocol)
    if model_meta.get("data_sha256") != hashes["train"]:
        raise ValueError("Source checkpoint training archive hash differs: independently verify archive-member equality before adapting protocol")
    model = bundle.model
    if model.latent_dim != 512:
        raise ValueError("Protocol fixes the existing 512-dimensional encoder")
    frozen_hash = {k: parameter_hash(getattr(model, k)) for k in ("encoder", "projector")}
    train, train_meta = read_data(args.train, args.smoke)
    test, test_meta = read_data(args.eval, args.smoke)
    if train_meta.get("seed") == test_meta.get("seed") or train_meta.get("spec") != test_meta.get("spec"):
        raise ValueError("Require different collection seeds and identical environment spec")
    if train_meta.get("split") != "train" or test_meta.get("split") != "eval":
        raise ValueError("Require the declared train/eval archives")
    protocol.update(train_metadata=train_meta, eval_metadata=test_meta,
                    runtime_versions={"jax": jax.__version__, "numpy": np.__version__})
    json_write(args.out / "manifest.json", protocol)
    encoding_start = time.time()
    ztrain = encode(model, train.pop("frames"), args.encode_chunk)
    protocol["train_encoding_s"] = time.time() - encoding_start
    gtrain, train_ids, folds = episode_targets(train["rewards"], train["dones"], args.gamma)
    fit = (folds >= 0) & (folds < 2)
    mean, scale = float(gtrain[fit].mean()), float(gtrain[fit].std())
    if scale < 1e-4:
        raise ValueError("Behavior returns are nearly constant; value objective uninformative")
    teacher, tq, tloss = fit_value(Value, ztrain, gtrain, folds, 0, args, mean, scale)
    auditor, aq, aloss = fit_value(Value, ztrain, gtrain, folds, 1, args, mean, scale)
    teacher_hashes = [parameter_hash(teacher), parameter_hash(auditor)]
    quality = {"teacher": tq, "independent_auditor": aq, "return_mean": mean, "return_std": scale,
               "normalization_source": "complete train episodes in folds A+B", "test_used_for_selection": False}
    json_write(args.out / "teacher_quality.json", quality)
    np.savez(args.out / "teacher_losses.npz", teacher=tloss, auditor=aloss)
    for name, head in (("teacher", teacher), ("auditor", auditor)):
        T.save(args.out / (name + ".npz"), head, {"dim": 512, "hidden": args.head_hidden, "mean": mean, "std": scale})
    if not args.smoke and not (tq["passes_prespecified_gate"] and aq["passes_prespecified_gate"]):
        protocol.update(status="teacher_quality_gate_failed", teacher_quality=quality, elapsed_s=time.time() - t0)
        json_write(args.out / "manifest.json", protocol)
        print("[stop] Frozen value probes fail prospective holdout skill requirement; no continuation launched.", flush=True)
        return
    table = window_table(train["dones"], gtrain, folds, model.history, args.horizon, True)
    if len(table) < args.batch:
        raise ValueError("Not enough complete-episode predictor training windows")
    # Held-out pixels are first encoded only after teacher adequacy is decided.
    ztest = encode(model, test.pop("frames"), args.encode_chunk)
    gtest, test_ids, test_folds = episode_targets(test["rewards"], test["dones"], args.gamma)
    test_table = window_table(test["dones"], gtest, test_folds, model.history, args.horizon)
    summaries, all_metrics, invariance = {}, {}, {}
    for name, weight in (("base", None), ("mse", 0.), ("value_aware", args.value_weight)):
        arm_start = time.time()
        arm_bundle, _ = T.load_model(args.model)
        arm = arm_bundle.model
        part = PredictivePart(arm)
        initial_predictor_hash = parameter_hash(part)
        training_s = 0.
        if weight is not None:
            training_start = time.time()
            losses = continue_predictor(part, teacher, ztrain, train["actions"], gtrain, table, args, mean, scale, weight)
            training_s = time.time() - training_start
            np.savez(args.out / (name + "_losses.npz"), losses=losses)
        arm.predictor, arm.pred_proj, arm.action_encoder = part.predictor, part.pred_proj, part.action_encoder
        checks = {k: parameter_hash(getattr(arm, k)) == frozen_hash[k] for k in frozen_hash}
        checks["teacher_unchanged"] = parameter_hash(teacher) == teacher_hashes[0]
        checks["auditor_unchanged"] = parameter_hash(auditor) == teacher_hashes[1]
        if not all(checks.values()):
            raise RuntimeError(f"Frozen parameter invariance failed: {checks}")
        invariance[name] = dict(checks, initial_predictor_sha256=initial_predictor_hash,
                                final_predictor_sha256=parameter_hash(part), frozen_sha256=frozen_hash,
                                trainable_parameters=sum(int(x.size) for x in jax.tree.leaves(nnx.state(part, nnx.Param))))
        evaluation_start = time.time()
        summaries[name], all_metrics[name] = evaluate(part, teacher, auditor, ztest, test["actions"],
                        gtest, test_ids, test_table, args, mean, scale)
        summaries[name]["elapsed_s"] = time.time() - arm_start
        summaries[name]["training_s"] = training_s
        summaries[name]["evaluation_s"] = time.time() - evaluation_start
        np.savez(args.out / (name + "_metrics.npz"), **all_metrics[name])
        if weight is not None:
            meta = dict(model_meta, pve_inspired={"source_checkpoint_sha256": hashes["model"],
                         "weight": weight, "steps": args.steps, "seed": args.seed,
                         "encoder_frozen": True, "manifest": protocol})
            T.save(args.out / (name + "_model.npz"), arm, meta, arm_bundle.decoder, arm_bundle.heads)
        print(f"[evaluate {name}] {json.dumps(summaries[name])}", flush=True)
    if len({v["initial_predictor_sha256"] for v in invariance.values()}) != 1:
        raise RuntimeError("Arms did not start from identical predictor parameters")
    summary = {"teacher_quality": quality, "arms": summaries,
               "paired_differences": paired_intervals(all_metrics["mse"], all_metrics["value_aware"], args.seed + 4001),
               "invariance": invariance, "scope": "Single training seed; realized behavior-return diagnostic, no control guarantee",
               "scientific_run": not args.smoke}
    json_write(args.out / "summary.json", summary)
    protocol.update(status="complete", elapsed_s=time.time() - t0, train_metadata=train_meta, eval_metadata=test_meta,
                    train_windows=len(table), eval_windows=len(test_table))
    json_write(args.out / "manifest.json", protocol)
    print(f"[done] {time.time() - t0:.1f}s {args.out}", flush=True)


if __name__ == "__main__":
    main()
