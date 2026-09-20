"""Final, post-result history-conditioned value-aware continuation protocol.

The preceding single-frame attempt failed its unchanged teacher-adequacy gate.
This separate attempt matches the predictor's eight observed frames, using a
GRU-128 value readout. It never changes the world model's 512-dimensional latent.
See research/pve_history_protocol.md. No further variants after a failed gate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
from studies.revision import pve as B

HISTORY, HIDDEN, HORIZON, WEIGHT = 8, 128, 4, .1


def incoming(actions, offset, xp=np):
    """Eight incoming slots: absent first action, then seven within-window actions.

    one_hot(-1) is the all-zero sentinel. No preceding-window or future outgoing
    action is exposed. ``offset`` is the first frame's index in the full window.
    """
    first = xp.full((actions.shape[0], 1), -1, dtype=xp.int32)
    return xp.concatenate([first, actions[:, offset:offset + HISTORY - 1]], axis=1)


def self_test():
    B.self_test()
    a = np.arange(HISTORY + HORIZON - 1, dtype=np.int32)[None]
    np.testing.assert_array_equal(incoming(a, 0), [[-1, 0, 1, 2, 3, 4, 5, 6]])
    np.testing.assert_array_equal(incoming(a, 1), [[-1, 1, 2, 3, 4, 5, 6, 7]])
    np.testing.assert_array_equal(incoming(a, 4), [[-1, 4, 5, 6, 7, 8, 9, 10]])
    history = np.arange(HISTORY)[None, :, None]
    for h in range(1, HORIZON + 1):
        history = np.concatenate([history[:, 1:], np.full((1, 1, 1), 7 + h)], 1)
        np.testing.assert_array_equal(history[0, :, 0], np.arange(h, 8 + h))
        assert incoming(a, h)[0, -1] == 6 + h  # last action ends at target frame 7+h
    enum = np.repeat(a[:, :HISTORY], 4, 0)
    enum[:, -1] = np.arange(4)
    np.testing.assert_array_equal(incoming(enum, 1)[:, -1], np.arange(4))
    r = np.ones((15, 1), np.float32)
    d = np.zeros_like(r, bool)
    d[-1] = True
    g, _, f = B.episode_targets(r, d, .99)
    table = B.window_table(d, g, f, HISTORY, 0)
    assert len(table) == 8 and table[0, 0] + HISTORY - 1 == 7
    print("history self-test passed: shifted contexts, incoming-action sentinel, target and counterfactual alignment")


def runtime():
    global jax, jnp, nnx, optax, T
    B.runtime()
    jax, jnp, nnx, optax, T = B.jax, B.jnp, B.nnx, B.optax, B.T


def value_class():
    class HistoryValue(nnx.Module):
        """Generic GRU readout, freshly initialized hidden state per window."""
        def __init__(self, dim, actions, seed):
            self.actions = actions
            r = T.rngs(seed)
            self.z = nnx.Linear(dim + actions + HIDDEN, HIDDEN, rngs=r)
            self.r = nnx.Linear(dim + actions + HIDDEN, HIDDEN, rngs=r)
            self.h = nnx.Linear(dim + actions + HIDDEN, HIDDEN, rngs=r)
            self.out = nnx.Linear(HIDDEN, 1, rngs=r)

        def __call__(self, history, incoming_actions):
            state = jnp.zeros((history.shape[0], HIDDEN), jnp.float32)
            onehot = jax.nn.one_hot(incoming_actions, self.actions, dtype=jnp.float32)
            for t in range(HISTORY):
                x = jnp.concatenate([history[:, t], onehot[:, t]], -1)
                joint = jnp.concatenate([x, state], -1)
                z = jax.nn.sigmoid(self.z(joint))
                reset = jax.nn.sigmoid(self.r(joint))
                proposal = jnp.tanh(self.h(jnp.concatenate([x, reset * state], -1)))
                state = (1. - z) * state + z * proposal
            return self.out(state)[..., 0]
    return HistoryValue


def fit_value(Value, z, actions, targets, folds, table, fold, args, mean, scale, n_actions):
    t0 = time.time()
    critic = Value(z.shape[-1], n_actions, args.seed + 110 + fold * 101)
    chosen = folds[table[:, 0], table[:, 1]]
    fit_table, val_table = table[chosen == fold], table[chosen == 2]
    if len(fit_table) < args.head_batch or len(val_table) < 100:
        raise ValueError("Too few complete-episode eight-frame training/validation windows")
    zj, aj, gj, tj = map(jnp.asarray, (z, actions, targets, fit_table))
    opt = nnx.Optimizer(critic, optax.chain(optax.clip_by_global_norm(1.),
        optax.adamw(optax.warmup_cosine_decay_schedule(0., args.head_lr,
                    max(1, args.head_steps // 20), args.head_steps), weight_decay=1e-4)), wrt=nnx.Param)

    @nnx.jit
    def run(critic, opt, z, actions, targets, table, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def body(carry, key):
            critic, opt = carry
            take = table[jax.random.randint(key, (args.head_batch,), 0, len(table))]
            ti = take[:, 0, None] + jnp.arange(HISTORY)[None]
            e = take[:, 1, None]
            zh = z[ti, e]
            ia = incoming(actions[ti[:, :-1], e], 0, jnp)
            g = (targets[ti[:, -1], take[:, 1]] - mean) / scale
            loss, grad = nnx.value_and_grad(lambda c: jnp.mean((c(zh, ia) - g) ** 2))(critic)
            opt.update(critic, grad)
            return (critic, opt), loss
        return body((critic, opt), keys)[1]

    keys = jax.random.split(jax.random.PRNGKey(args.seed + 71 + fold), args.head_steps)
    losses = []
    for start in range(0, args.head_steps, args.chunk):
        losses.extend(np.asarray(run(critic, opt, zj, aj, gj, tj, keys[start:start + args.chunk])).tolist())
        print(f"[history-value{fold}] {len(losses)}/{args.head_steps} loss={losses[-1]:.5f}", flush=True)

    @nnx.jit
    def pred(critic, zh, ia):
        return critic(zh, ia)

    predictions = []
    for start in range(0, len(val_table), 1024):
        block = val_table[start:start + 1024]
        ti = block[:, 0, None] + np.arange(HISTORY)[None]
        e = block[:, 1, None]
        predictions.append(np.asarray(pred(critic, jnp.asarray(z[ti, e]),
                            jnp.asarray(incoming(actions[ti[:, :-1], e], 0)))))
    p = np.concatenate(predictions) * scale + mean
    y = targets[val_table[:, 0] + HISTORY - 1, val_table[:, 1]]
    baseline = float(np.mean(targets[fit_table[:, 0] + HISTORY - 1, fit_table[:, 1]]))
    mse, base_mse = float(np.mean((p - y) ** 2)), float(np.mean((baseline - y) ** 2))
    corr = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 1e-10 and np.std(y) > 1e-10 else 0.
    quality = {"train_windows": len(fit_table), "validation_windows": len(val_table),
        "mse": mse, "constant_mse": base_mse, "relative_mse_reduction": 1. - mse / max(base_mse, 1e-12),
        "correlation": corr, "prediction_std": float(p.std()), "target_std": float(y.std()),
        "passes_prespecified_gate": bool(mse < .9 * base_mse and corr > .2),
        "fit_and_validation_s": time.time() - t0,
        "population": "complete-episode windows with eight observed frames; first seven episode frames excluded",
        "parameters": sum(int(x.size) for x in jax.tree.leaves(nnx.state(critic, nnx.Param)))}
    return critic, quality, np.asarray(losses, np.float32)


def rollout_contexts(part, history, actions, train=False):
    predictions, contexts = [], []
    for h in range(HORIZON):
        nxt = part(history, actions[:, h:h + HISTORY], train)[:, -1]
        history = jnp.concatenate([history[:, 1:], nxt[:, None]], 1)
        predictions.append(nxt)
        contexts.append(history)
    return jnp.stack(predictions, 1), jnp.stack(contexts, 1)


def future_incoming(actions):
    return jnp.stack([incoming(actions, h, jnp) for h in range(1, HORIZON + 1)], 1)


def read_histories(head, contexts, ia):
    b, hor, hist, dim = contexts.shape
    return head(contexts.reshape(b * hor, hist, dim), ia.reshape(b * hor, hist)).reshape(b, hor)


def continue_predictor(part, teacher, z, actions, targets, table, args, mean, scale, weight):
    z, actions, targets, table = map(jnp.asarray, (z, actions, targets, table))
    opt = nnx.Optimizer(part, optax.chain(optax.clip_by_global_norm(1.),
        optax.adamw(optax.warmup_cosine_decay_schedule(0., args.lr,
                    max(1, args.steps // 20), args.steps), weight_decay=1e-3)), wrt=nnx.Param)

    @nnx.jit
    def run(part, opt, teacher, z, actions, targets, table, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def body(carry, key):
            part, opt = carry
            take = table[jax.random.randint(key, (args.batch,), 0, len(table))]
            ti = take[:, 0, None] + jnp.arange(HISTORY + HORIZON)[None]
            e = take[:, 1, None]
            zw, aw = z[ti, e], actions[ti[:, :-1], e]
            target = (targets[ti[:, HISTORY:], e] - mean) / scale
            ia = future_incoming(aw)

            def loss_fn(p):
                predicted, contexts = rollout_contexts(p, zw[:, :HISTORY], aw, True)
                lmse = jnp.mean((predicted - zw[:, HISTORY:]) ** 2)
                value = read_histories(teacher, contexts, ia)
                lvalue = jnp.mean((value - target) ** 2)
                return lmse + weight * lvalue, jnp.stack([lmse, lvalue])

            (_, losses), grad = nnx.value_and_grad(loss_fn, has_aux=True)(part)
            opt.update(part, grad)
            return (part, opt), losses
        return body((part, opt), keys)[1]

    keys = jax.random.split(jax.random.PRNGKey(args.seed + 901), args.steps)
    history = []
    for start in range(0, args.steps, args.chunk):
        history.append(np.asarray(run(part, opt, teacher, z, actions, targets, table, keys[start:start + args.chunk])))
        print(f"[history-continue lambda={weight}] {start + len(history[-1])}/{args.steps} {history[-1][-1].tolist()}", flush=True)
    return np.concatenate(history)


def evaluate(part, teacher, auditor, z, actions, targets, ids, table, args, mean, scale):
    selected = np.random.default_rng(args.seed + 4101).choice(len(table), min(args.eval_windows, len(table)), replace=False)
    table = table[selected]

    @nnx.jit
    def batch(part, teacher, auditor, zw, aw, target):
        pred, contexts = rollout_contexts(part, zw[:, :HISTORY], aw)
        ia = future_incoming(aw)
        real_contexts = jnp.stack([zw[:, h:h + HISTORY] for h in range(1, HORIZON + 1)], 1)
        read = lambda head, ctx: read_histories(head, ctx, ia) * scale + mean
        vt, va = read(teacher, contexts), read(auditor, contexts)
        rt, ra = read(teacher, real_contexts), read(auditor, real_contexts)
        truth = zw[:, HISTORY:]
        repeated = jnp.repeat(zw[:, :HISTORY], part.n_actions, 0)
        aa = jnp.repeat(aw[:, :HISTORY], part.n_actions, 0)
        aa = aa.at[:, -1].set(jnp.tile(jnp.arange(part.n_actions, dtype=aa.dtype), len(zw)))
        counter = part(repeated, aa)[:, -1]
        counter_context = jnp.concatenate([repeated[:, 1:], counter[:, None]], 1)
        counter_value = auditor(counter_context, incoming(aa, 1, jnp)).reshape(len(zw), part.n_actions) * scale + mean
        counter = counter.reshape(len(zw), part.n_actions, -1)
        return {"latent_mse": jnp.mean((pred - truth) ** 2, -1),
            "teacher_mc_mse": (vt - target) ** 2, "auditor_mc_mse": (va - target) ** 2,
            "teacher_target_value_mse": (vt - rt) ** 2, "auditor_target_value_mse": (va - ra) ** 2,
            "real_teacher_mc_mse": (rt - target) ** 2, "real_auditor_mc_mse": (ra - target) ** 2,
            "predicted_latent_rms": jnp.sqrt(jnp.mean(pred ** 2, -1)),
            "real_latent_rms": jnp.sqrt(jnp.mean(truth ** 2, -1)),
            "action_value_std": jnp.std(counter_value, -1),
            "action_latent_rms": jnp.sqrt(jnp.mean(jnp.var(counter, 1), -1))}

    pieces = []
    for start in range(0, len(table), args.eval_batch):
        block = table[start:start + args.eval_batch]
        ti = block[:, 0, None] + np.arange(HISTORY + HORIZON)[None]
        e = block[:, 1, None]
        result = batch(part, teacher, auditor, *map(jnp.asarray,
                    (z[ti, e], actions[ti[:, :-1], e], targets[ti[:, HISTORY:], e])))
        pieces.append(jax.tree.map(np.asarray, result))
    metrics = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    metrics["episode_id"] = ids[table[:, 0], table[:, 1]]
    metrics["start_env"] = table
    summary = {k: v.mean(0).tolist() for k, v in metrics.items() if k not in ("episode_id", "start_env")}
    summary.update(windows=len(table), independent_episodes=int(len(np.unique(metrics["episode_id"]))))
    return summary, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    for name in ("model", "train", "eval", "out", "prior-attempt"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--head-steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--head-batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--encode-chunk", type=int, default=1024)
    p.add_argument("--eval-windows", type=int, default=8192)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, k) is None for k in ("model", "train", "eval", "out", "prior_attempt")):
        p.error("--model, --train, --eval, --out, --prior-attempt are required")
    prior = json.loads((args.prior_attempt / "manifest.json").read_text())
    if prior.get("status") != "teacher_quality_gate_failed" or not prior.get("scientific_run"):
        raise ValueError("This revision must explicitly reference the failed scientific single-frame attempt")
    if B.sha256(B.__file__) != prior["source_sha256"]:
        raise ValueError("The preceding attempt's helper source must remain exactly unchanged")
    if args.smoke:
        args.out = args.out.with_name(args.out.name + "_smoke")
        args.steps, args.head_steps, args.batch, args.head_batch = 3, 3, 8, 16
        args.eval_windows, args.eval_batch, args.chunk, args.encode_chunk = 16, 8, 3, 128
    elif (args.seed, args.steps, args.head_steps, args.batch, args.head_batch, args.lr, args.head_lr, args.gamma) != (0, 3000, 4000, 128, 1024, 1e-4, 1e-3, .99):
        p.error("The final scientific attempt fixes its training budget and optimizer settings")
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    hashes = {k: B.sha256(getattr(args, k)) for k in ("model", "train", "eval")}
    if hashes != prior["sha256"]:
        raise ValueError("This information-matching revision must use exactly the prior checkpoint/train/test archives")
    protocol = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sha256": hashes, "source_sha256": B.sha256(__file__), "helper_source_sha256": B.sha256(B.__file__),
        "prior_manifest_sha256": B.sha256(args.prior_attempt / "manifest.json"),
        "prior_teacher_quality": prior["teacher_quality"], "status": "started", "scientific_run": not args.smoke,
        "adaptation": "post-result history conditioning after failed single-frame teacher gate; validation reused",
        "name": "History-conditioned PVE-inspired value-aware continuation", "latent_dim": 512,
        "teacher_history": HISTORY, "teacher_gru_hidden": HIDDEN, "horizon": HORIZON, "value_weight": WEIGHT,
        "further_teacher_variants_permitted": False}
    B.json_write(args.out / "manifest.json", protocol)
    runtime()
    Value = value_class()
    _, PredictivePart = B.classes()
    bundle, model_meta = T.load_model(args.model)
    model = bundle.model
    protocol["original_model_metadata"] = model_meta
    if model_meta.get("data_sha256") != hashes["train"] or model.latent_dim != 512 or model.history != HISTORY:
        raise ValueError("Require matching source data and the original 512-dimensional, eight-frame model")
    frozen_hash = {k: B.parameter_hash(getattr(model, k)) for k in ("encoder", "projector")}
    train, train_meta = B.read_data(args.train, args.smoke)
    test, test_meta = B.read_data(args.eval, args.smoke)
    for data in (train, test):
        data["actions"] = data["actions"].astype(np.int32)
    if train_meta.get("seed") == test_meta.get("seed") or train_meta.get("spec") != test_meta.get("spec"):
        raise ValueError("Require separate collection seeds and identical environment specifications")
    if train_meta.get("split") != "train" or test_meta.get("split") != "eval":
        raise ValueError("Require declared train/eval splits")
    protocol.update(train_metadata=train_meta, eval_metadata=test_meta,
                    runtime_versions={"jax": jax.__version__, "numpy": np.__version__})
    B.json_write(args.out / "manifest.json", protocol)
    enc_start = time.time()
    ztrain = B.encode(model, train.pop("frames"), args.encode_chunk)
    protocol["train_encoding_s"] = time.time() - enc_start
    gtrain, train_ids, folds = B.episode_targets(train["rewards"], train["dones"], args.gamma)
    fit = (folds >= 0) & (folds < 2)
    mean, scale = float(gtrain[fit].mean()), float(gtrain[fit].std())
    if scale < 1e-4:
        raise ValueError("Behavior returns are nearly constant")
    probe_table = B.window_table(train["dones"], gtrain, folds, HISTORY, 0)
    teacher, tq, tloss = fit_value(Value, ztrain, train["actions"], gtrain, folds, probe_table, 0, args, mean, scale, model.n_actions)
    auditor, aq, aloss = fit_value(Value, ztrain, train["actions"], gtrain, folds, probe_table, 1, args, mean, scale, model.n_actions)
    teacher_hashes = [B.parameter_hash(teacher), B.parameter_hash(auditor)]
    quality = {"teacher": tq, "independent_auditor": aq, "return_mean": mean, "return_std": scale,
        "normalization_source": "all complete train episodes in folds A+B, identical to first attempt",
        "test_used_for_selection": False, "validation_reused_after_prior_attempt": True,
        "gate_population_differs_from_single_frame_attempt": True}
    B.json_write(args.out / "teacher_quality.json", quality)
    np.savez(args.out / "teacher_losses.npz", teacher=tloss, auditor=aloss)
    for name, head in (("teacher", teacher), ("auditor", auditor)):
        T.save(args.out / (name + ".npz"), head, {"architecture": "history_gru", "history": HISTORY,
            "dim": 512, "hidden": HIDDEN, "actions": model.n_actions, "mean": mean, "std": scale})
    if not args.smoke and not (tq["passes_prespecified_gate"] and aq["passes_prespecified_gate"]):
        protocol.update(status="teacher_quality_gate_failed_final_attempt", teacher_quality=quality, elapsed_s=time.time() - t0)
        B.json_write(args.out / "manifest.json", protocol)
        print("[stop] Final history-conditioned teacher attempt failed its unchanged adequacy gate; no further variants.", flush=True)
        return
    table = B.window_table(train["dones"], gtrain, folds, HISTORY, HORIZON, True)
    if len(table) < args.batch:
        raise ValueError("Not enough no-reset continuation windows")
    ztest = B.encode(model, test.pop("frames"), args.encode_chunk)
    gtest, test_ids, test_folds = B.episode_targets(test["rewards"], test["dones"], args.gamma)
    test_table = B.window_table(test["dones"], gtest, test_folds, HISTORY, HORIZON)
    summaries, all_metrics, invariance = {}, {}, {}
    for name, weight in (("base", None), ("mse", 0.), ("value_aware", WEIGHT)):
        arm_start = time.time()
        arm_bundle, _ = T.load_model(args.model)
        arm = arm_bundle.model
        part = PredictivePart(arm)
        initial_hash = B.parameter_hash(part)
        training_s = 0.
        if weight is not None:
            begin = time.time()
            losses = continue_predictor(part, teacher, ztrain, train["actions"], gtrain, table, args, mean, scale, weight)
            training_s = time.time() - begin
            np.savez(args.out / (name + "_losses.npz"), losses=losses)
        arm.predictor, arm.pred_proj, arm.action_encoder = part.predictor, part.pred_proj, part.action_encoder
        checks = {k: B.parameter_hash(getattr(arm, k)) == frozen_hash[k] for k in frozen_hash}
        checks.update(teacher_unchanged=B.parameter_hash(teacher) == teacher_hashes[0],
                      auditor_unchanged=B.parameter_hash(auditor) == teacher_hashes[1])
        if not all(checks.values()):
            raise RuntimeError(f"Frozen parameter invariance failed: {checks}")
        invariance[name] = dict(checks, initial_predictor_sha256=initial_hash,
            final_predictor_sha256=B.parameter_hash(part), frozen_sha256=frozen_hash,
            trainable_parameters=sum(int(x.size) for x in jax.tree.leaves(nnx.state(part, nnx.Param))))
        begin = time.time()
        summaries[name], all_metrics[name] = evaluate(part, teacher, auditor, ztest, test["actions"],
                            gtest, test_ids, test_table, args, mean, scale)
        summaries[name].update(elapsed_s=time.time() - arm_start, training_s=training_s, evaluation_s=time.time() - begin)
        np.savez(args.out / (name + "_metrics.npz"), **all_metrics[name])
        if weight is not None:
            meta = dict(model_meta, pve_history={"source_checkpoint_sha256": hashes["model"],
                "weight": weight, "steps": args.steps, "seed": args.seed, "encoder_frozen": True, "manifest": protocol})
            T.save(args.out / (name + "_model.npz"), arm, meta, arm_bundle.decoder, arm_bundle.heads)
        print(f"[history-evaluate {name}] {json.dumps(summaries[name])}", flush=True)
    if len({v["initial_predictor_sha256"] for v in invariance.values()}) != 1:
        raise RuntimeError("Predictor initializations differ")
    B.json_write(args.out / "summary.json", {"teacher_quality": quality, "arms": summaries,
        "paired_differences": B.paired_intervals(all_metrics["mse"], all_metrics["value_aware"], args.seed + 4001),
        "invariance": invariance, "scientific_run": not args.smoke,
        "scope": "One model/continuation seed, post-result revised protocol, survivor-conditioned behavior-return diagnostic"})
    protocol.update(status="complete", elapsed_s=time.time() - t0, teacher_quality=quality,
                    train_windows=len(table), eval_windows=len(test_table))
    B.json_write(args.out / "manifest.json", protocol)
    print(f"[done] {time.time() - t0:.1f}s {args.out}", flush=True)


if __name__ == "__main__":
    main()
