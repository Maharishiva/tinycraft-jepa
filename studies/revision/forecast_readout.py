"""Frozen-feature diagnostic: state readouts on real, forecast and copied latents.

This program never updates a world model or policy. Simulator labels supervise
only small offline scientific probes. See research/forecast_readout_protocol.md.
Example (run separately for each existing encoder seed):
  python -m studies.revision.forecast_readout --model MODEL.npz --model-seed 0 \
    --train train.npz --eval eval.npz --out runs/revision/readout/m0
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


HORIZONS = (1, 2, 4, 8)
DOMAINS = ("real", "forecast", "copy")
NO_WALL, REVEAL_X, WIDTH = -99, 40, 63


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def select_contexts(data, history, max_horizon, count, seed):
    """Past-only eligibility. No condition on future survival or wall labels."""
    dones = np.asarray(data["dones"], bool)
    total, envs = dones.shape
    if total < history + max_horizon - 1:
        raise ValueError("archive is too short")
    prefix = np.concatenate([np.zeros((1, envs), np.int32), np.cumsum(dones, axis=0)], axis=0)
    times = np.arange(history - 1, total - max_horizon + 1)
    valid = (prefix[times] - prefix[times - history + 1]) == 0
    ti, ei = np.nonzero(valid)
    ids = np.stack([times[ti], ei], axis=1).astype(np.int32)
    if not len(ids):
        raise ValueError("no contexts without a past reset")
    order = np.random.default_rng(seed).permutation(len(ids))[:min(count, len(ids))]
    ids = ids[order]
    return ids, {"eligible_contexts": int(valid.sum()), "selected_contexts": len(ids),
                 "seed": seed, "selected_ids_sha256": array_hash(ids)}


def gather_labels(data, ids, history):
    t, e = ids.T
    dones = np.asarray(data["dones"], bool)
    prefix = np.concatenate([np.zeros((1, dones.shape[1]), np.int32), np.cumsum(dones, axis=0)], axis=0)
    context_wall = np.asarray(data["wall_x"][t, e], np.int32)
    labels = {"ids": ids, "context_wall_x": context_wall,
              "context_pos": np.asarray(data["pos"][t, e], np.float32),
              "original_gap": np.asarray(data["gap"][t, e], np.float32),
              "hidden_gap_cohort": (context_wall != NO_WALL) & (context_wall > REVEAL_X)}
    for horizon in HORIZONS:
        target = t + horizon
        keep = (prefix[target, e] - prefix[t, e]) == 0
        wall = np.asarray(data["wall_x"][target, e], np.int32)
        pos = np.asarray(data["pos"][target, e], np.float32)
        labels[f"h{horizon}_continuing"] = keep
        labels[f"h{horizon}_wall_x"] = wall
        labels[f"h{horizon}_wall_present"] = wall != NO_WALL
        labels[f"h{horizon}_pos"] = pos
        # Each action is outgoing from its identically indexed frame. The final
        # dummy is consumed only when shifting history after the last forecast.
    return labels


def binary_metrics(target, score):
    target, score = np.asarray(target, bool), np.asarray(score, np.float64)
    n = len(target)
    if not n:
        return {"n": 0}
    pred = score >= 0.5
    positives, negatives = int(target.sum()), int((~target).sum())
    recalls = [float((pred[target]).mean())] if positives else []
    if negatives:
        recalls.append(float((~pred[~target]).mean()))
    result = {"n": n, "positive_count": positives, "prevalence": positives / n,
              "accuracy": float((target == pred).mean()),
              "balanced_accuracy": float(np.mean(recalls)),
              "brier_clipped_score": float(np.square(np.clip(score, 0, 1) - target).mean()),
              "threshold": 0.5}
    if positives and negatives:
        order = np.argsort(score, kind="stable")
        values = score[order]
        _, first, counts = np.unique(values, return_index=True, return_counts=True)
        average_ranks = first + (counts + 1) / 2.0
        ranks = np.repeat(average_ranks, counts)
        result["auroc"] = float((ranks[target[order]].sum() - positives * (positives + 1) / 2) /
                                 (positives * negatives))
        order = np.argsort(-score, kind="stable")
        values, truth = score[order], target[order]
        ends = np.r_[np.nonzero(values[:-1] != values[1:])[0], n - 1]
        tp = np.cumsum(truth)[ends]
        result["average_precision"] = float(np.sum(np.diff(np.r_[0, tp]) / positives * tp / (ends + 1)))
    return result


def position_metrics(target, pred):
    error = np.asarray(pred, np.float64) - np.asarray(target, np.float64)
    if not len(error):
        return {"n": 0}
    if error.ndim == 1:
        return {"n": len(error), "mae_px": float(np.abs(error).mean()),
                "rmse_px": float(np.sqrt(np.square(error).mean()))}
    return {"n": len(error), "mae_px_per_coordinate": float(np.abs(error).mean()),
            "rmse_px_per_coordinate": float(np.sqrt(np.square(error).mean())),
            "mean_euclidean_error_px": float(np.linalg.norm(error, axis=1).mean()),
            "x_mae_px": float(np.abs(error[:, 0]).mean()),
            "y_mae_px": float(np.abs(error[:, 1]).mean())}


def fit_ridge(x, y, alpha):
    """Mean squared loss plus alpha ||W||^2, with unpenalized intercept."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if y.ndim == 1:
        y = y[:, None]
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc / len(x)
    gram.flat[::len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, xc.T @ yc / len(x))
    intercept = ym - xm @ weights
    return weights, intercept


def apply_probe(probe, x):
    return np.asarray(x, np.float64) @ probe[0] + probe[1]


def load_archive(path):
    with np.load(path) as archive:
        names = ("frames", "actions", "dones", "episode", "pos", "wall_x", "gap")
        data = {name: archive[name] for name in names}
        meta = json.loads(str(archive["meta"]))
    spec = meta.get("spec", {})
    if not spec.get("gate") or spec.get("noise") != "off" or spec.get("wind"):
        raise ValueError("this preregistered diagnostic is restricted to the original gate_off_still")
    if meta.get("policy") != "uniform random":
        raise ValueError("the hidden-gap negative control requires uniformly random recorded actions")
    n, e = data["actions"].shape
    if data["frames"].shape[:2] != (n + 1, e):
        raise ValueError("archive must contain its final frame")
    if not np.array_equal(data["episode"][1:] != data["episode"][:-1], data["dones"]):
        raise ValueError("episode boundaries disagree with dones")
    return data, meta


def extract_features(model, data, ids, history, batch):
    """Cache each needed real frame once, then forecast from frozen encodings."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    t, e = ids.T
    envs = data["actions"].shape[1]
    ci = (t[:, None] - history + 1 + np.arange(history)) * envs + e[:, None]
    fi = (t[:, None] + np.asarray(HORIZONS)) * envs + e[:, None]
    unique = np.unique(np.concatenate([ci.ravel(), fi.ravel()]))
    flat_frames = data["frames"].reshape((-1,) + data["frames"].shape[2:])

    @nnx.jit
    def encode(model, frames):
        return model.encode(frames[:, None])[:, 0]

    # Match the public context encoder's flattened frame batch exactly. BF16
    # kernels can change rounding with batch shape even with frozen BatchNorm.
    encoder_batch = batch * history
    chunks = []
    start = time.time()
    for offset in range(0, len(unique), encoder_batch):
        indices = unique[offset:offset + encoder_batch]
        actual = len(indices)
        if actual < encoder_batch:
            indices = np.pad(indices, (0, encoder_batch - actual), mode="edge")
        z = np.asarray(jax.device_get(encode(model, jnp.asarray(flat_frames[indices]))))[:actual]
        chunks.append(z)
    cached = np.concatenate(chunks)
    context = cached[np.searchsorted(unique, ci)]
    real = cached[np.searchsorted(unique, fi)]
    action_idx = t[:, None] - history + 1 + np.arange(history + max(HORIZONS) - 1)
    actions = data["actions"][action_idx, e[:, None]].astype(np.int32)
    actions = np.concatenate([actions, np.zeros((len(actions), 1), np.int32)], axis=1)

    @nnx.jit
    def forecast(model, context, outgoing):
        acts = model.act_embed(outgoing)
        carry = (context, acts[:, :history])

        def step(carry, j):
            encoded, action_history = carry
            nxt = model.predict(encoded, action_history)[:, -1:]
            incoming = jax.lax.dynamic_slice_in_dim(acts, history + j, 1, axis=1)
            return (jnp.concatenate([encoded[:, 1:], nxt], axis=1),
                    jnp.concatenate([action_history[:, 1:], incoming], axis=1)), nxt[:, 0]

        _, predicted = jax.lax.scan(step, carry, jnp.arange(max(HORIZONS)))
        return predicted[jnp.asarray(HORIZONS) - 1].transpose(1, 0, 2)

    predictions = []
    for offset in range(0, len(ids), batch):
        current, taken = context[offset:offset + batch], actions[offset:offset + batch]
        actual = len(current)
        if actual < batch:
            current = np.pad(current, ((0, batch - actual), (0, 0), (0, 0)), mode="edge")
            taken = np.pad(taken, ((0, batch - actual), (0, 0)), mode="edge")
        pred = np.asarray(jax.device_get(forecast(model, jnp.asarray(current), jnp.asarray(taken))))[:actual]
        predictions.append(pred)
    predictions = np.concatenate(predictions)
    if not all(np.isfinite(x).all() for x in (real, predictions, context)):
        raise ValueError("nonfinite frozen features")
    # Check against the public implementation at the SAME full padded context
    # batch shape, rather than introducing a different BF16 kernel shape.
    import train as T
    check_n = min(batch, len(ids))
    check_frames = flat_frames[ci[:check_n]]
    check_actions = actions[:check_n]
    if check_n < batch:
        check_frames = np.pad(check_frames, ((0, batch - check_n), (0, 0), (0, 0), (0, 0), (0, 0)), mode="edge")
        check_actions = np.pad(check_actions, ((0, batch - check_n), (0, 0)), mode="edge")
    ref_context, ref_pred = T._rollout_latents(model, jnp.asarray(check_frames),
                                              jnp.asarray(check_actions), horizon=max(HORIZONS))
    ref_context, ref_pred = np.asarray(ref_context)[:check_n], np.asarray(ref_pred)[:check_n]
    context_diff = float(np.max(np.abs(np.asarray(ref_context) - context[:check_n])))
    forecast_diff = float(np.max(np.abs(np.asarray(ref_pred)[:, np.asarray(HORIZONS) - 1] - predictions[:check_n])))
    if not np.allclose(np.asarray(ref_context), context[:check_n], rtol=1e-4, atol=1e-5):
        raise ValueError(f"cached encoder differs from original: {context_diff}")
    if not np.allclose(np.asarray(ref_pred)[:, np.asarray(HORIZONS) - 1], predictions[:check_n], rtol=1e-3, atol=1e-4):
        raise ValueError(f"forecast implementation differs from original: {forecast_diff}")
    return {"real": real, "forecast": predictions, "copy": context[:, -1]}, {
        "unique_frames_encoded": len(unique), "seconds": round(time.time() - start, 3),
        "encoder_frame_batch": encoder_batch, "predictor_context_batch": batch,
        "parity_examples": check_n, "context_max_abs_difference": context_diff,
        "forecast_max_abs_difference": forecast_diff,
        "feature_sha256": {"real": array_hash(real), "forecast": array_hash(predictions),
                           "copy": array_hash(context[:, -1])}}


def metrics_for_task(task, y, pred):
    if task == "cube":
        return position_metrics(y * WIDTH, pred * WIDTH)
    if task == "wall_position":
        return position_metrics(y[:, 0] * WIDTH, pred[:, 0] * WIDTH)
    return binary_metrics(y[:, 0] > 0.5, pred[:, 0])


def task_data(labels, horizon, task):
    keep = labels[f"h{horizon}_continuing"]
    if task == "cube":
        return labels[f"h{horizon}_pos"] / WIDTH, keep
    if task == "wall_presence":
        return labels[f"h{horizon}_wall_present"][:, None].astype(np.float32), keep
    if task == "wall_position":
        present = labels[f"h{horizon}_wall_present"]
        # NO_WALL never becomes a target for a physical position regression.
        return np.where(present, labels[f"h{horizon}_wall_x"], 0)[:, None] / WIDTH, keep & present
    if task == "hidden_original_gap":
        # Keep ALL past-selected hidden-gap contexts, including later resets.
        # Conditioning this control on survival would reveal gap information.
        return labels["original_gap"][:, None], labels["hidden_gap_cohort"]
    raise ValueError(task)


def run(args):
    import flax.nnx as nnx
    import jax
    import train as T
    from studies.stoch import env as S

    if (S.NO_WALL, S.REVEAL_X, S.W) != (NO_WALL, REVEAL_X, WIDTH):
        raise ValueError("environment constants differ from the fixed diagnostic protocol")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("refusing to overwrite nonempty output")
    if args.alpha <= 0 or args.train_count < args.min_fit or args.eval_count < 1 or args.batch < 1:
        raise ValueError("invalid sample size or ridge penalty")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    train_hash, eval_hash = sha256(args.train), sha256(args.eval)
    if train_hash == eval_hash:
        raise ValueError("training and evaluation archives are identical")
    bundle, model_meta = T.load_model(args.model)
    model = bundle.model
    if model_meta.get("cfg", {}).get("seed") != args.model_seed:
        raise ValueError("model seed disagrees with checkpoint")
    history = int(model.history)
    initial_parameters = [np.asarray(x).copy() for x in jax.tree.leaves(nnx.state(model, nnx.Param, nnx.BatchStat))]
    all_features, all_labels, archive_info = {}, {}, {}
    for split, path, count, seed in (("train", args.train, args.train_count, args.selection_seed),
                                      ("eval", args.eval, args.eval_count, args.selection_seed + 1)):
        data, meta = load_archive(path)
        if meta.get("split") not in (None, split):
            raise ValueError(f"{split} archive metadata has unexpected split {meta.get('split')}")
        ids, selection = select_contexts(data, history, max(HORIZONS), count, seed)
        all_labels[split] = gather_labels(data, ids, history)
        all_features[split], extraction = extract_features(model, data, ids, history, args.batch)
        archive_info[split] = {"meta": meta, "selection": selection, "extraction": extraction}
        del data
        print("[features] " + json.dumps({"split": split, **selection, **extraction}), flush=True)
    if archive_info["train"]["meta"].get("seed") == archive_info["eval"]["meta"].get("seed"):
        raise ValueError("train and eval collection seeds must differ")

    # Two preprocessing analyses are declared before seeing outcomes. Neither
    # changes a model latent: both live solely inside the fitted readout.
    reference = all_features["train"]["copy"].astype(np.float64)
    mean, std = reference.mean(axis=0), reference.std(axis=0)
    scale = np.maximum(std, 1e-6)
    preprocess = {"mean": mean, "scale": scale}
    probe_arrays, prediction_arrays = dict(preprocess), {}
    for split, labels in all_labels.items():
        prediction_arrays.update({f"{split}_{name}": value for name, value in labels.items()})
    records, geometry = [], []
    for hi, horizon in enumerate(HORIZONS):
        raw = {}
        for split in ("train", "eval"):
            raw[split] = {domain: (values if domain == "copy" else values[:, hi]).astype(np.float64)
                          for domain, values in all_features[split].items()}
            reference_std = raw[split]["real"].std(axis=0)
            for domain in DOMAINS:
                values = raw[split][domain]
                feature_std = values.std(axis=0)
                nonconstant = reference_std > 1e-12
                ratio = feature_std[nonconstant] / reference_std[nonconstant]
                geometry.append({"horizon": horizon, "split": split, "domain": domain,
                    "raw_mean_l2": float(np.linalg.norm(values.mean(axis=0))),
                    "raw_std_rms": float(np.sqrt(np.square(feature_std).mean())),
                    "reference_nonconstant_features": int(nonconstant.sum()),
                    "std_ratio_to_real_median": float(np.median(ratio)) if len(ratio) else None,
                    "std_ratio_to_real_q10": float(np.quantile(ratio, 0.1)) if len(ratio) else None,
                    "std_ratio_to_real_q90": float(np.quantile(ratio, 0.9)) if len(ratio) else None,
                    "fraction_std_ratio_below_0_1": float((ratio < 0.1).mean()) if len(ratio) else None})
        for task in ("cube", "wall_presence", "wall_position", "hidden_original_gap"):
            ytrain, train_keep = task_data(all_labels["train"], horizon, task)
            yeval, eval_keep = task_data(all_labels["eval"], horizon, task)
            base = {"horizon": horizon, "task": task, "train_n": int(train_keep.sum()),
                    "eval_n": int(eval_keep.sum()), "train_fraction": float(train_keep.mean()),
                    "eval_fraction": float(eval_keep.mean())}
            if train_keep.sum() < args.min_fit or not eval_keep.any():
                records.append({**base, "status": "insufficient_data"})
                continue
            baseline = np.broadcast_to(ytrain[train_keep].mean(axis=0), (int(eval_keep.sum()), ytrain.shape[1]))
            records.append({**base, "normalization": "none", "trained_on": "train_mean", "evaluated_on": "constant",
                            "metrics": metrics_for_task(task, yeval[eval_keep], baseline)})
            for normalization in ("common", "per_source"):
                for source in DOMAINS:
                    if normalization == "common":
                        fitted_mean, fitted_scale = mean, scale
                    else:
                        fitted_mean = raw["train"][source].mean(axis=0)
                        fitted_scale = np.maximum(raw["train"][source].std(axis=0), 1e-6)
                    xtrain = (raw["train"][source] - fitted_mean) / fitted_scale
                    probe = fit_ridge(xtrain[train_keep], ytrain[train_keep], args.alpha)
                    name = f"{normalization}_h{horizon}_{task}_{source}"
                    probe_arrays[name + "_weights"], probe_arrays[name + "_intercept"] = probe
                    probe_arrays[name + "_mean"], probe_arrays[name + "_scale"] = fitted_mean, fitted_scale
                    for target in DOMAINS:
                        # Crucially, transfer retains the SOURCE training scaler.
                        xeval = (raw["eval"][target] - fitted_mean) / fitted_scale
                        pred = apply_probe(probe, xeval)
                        prediction_arrays[name + "_to_" + target] = pred.astype(np.float32)
                        records.append({**base, "normalization": normalization,
                            "trained_on": source, "evaluated_on": target,
                            "weight_l2": float(np.linalg.norm(probe[0])),
                            "raw_feature_weight_l2": float(np.linalg.norm(probe[0] / fitted_scale[:, None])),
                            "intercept_l2": float(np.linalg.norm(probe[1])),
                            "scale_floor_features": int((fitted_scale <= 1e-6).sum()),
                            "metrics": metrics_for_task(task, yeval[eval_keep], pred[eval_keep])})
        print("[readouts] " + json.dumps({"horizon": horizon, "records": len(records)}), flush=True)

    copy_parity = 0.0
    for name, value in prediction_arrays.items():
        if name.startswith("common_") and "_copy_to_" in name:
            counterpart = prediction_arrays[name.replace("common_", "per_source_", 1)]
            copy_parity = max(copy_parity, float(np.max(np.abs(value - counterpart))))
            if not np.allclose(value, counterpart, rtol=1e-5, atol=1e-6):
                raise ValueError("copy-domain predictions differ between identical preprocessing analyses")
    final_parameters = [np.asarray(x) for x in jax.tree.leaves(nnx.state(model, nnx.Param, nnx.BatchStat))]
    if not all(np.array_equal(a, b) for a, b in zip(initial_parameters, final_parameters)):
        raise ValueError("frozen model parameters/statistics changed")
    configuration = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_sha256": sha256(args.model), "train_sha256": train_hash, "eval_sha256": eval_hash,
        "source_sha256": {str(p): sha256(p) for p in
                          (Path(__file__), Path(T.__file__), Path(S.__file__), Path(S.R.__file__))},
        "model_cfg": model_meta.get("cfg"), "history": history, "latent_dim": int(model.latent_dim),
        "horizons": list(HORIZONS), "domains": list(DOMAINS), "archive_info": archive_info,
        "normalizations": {"common": "shared training current-latent mean/std for all horizons/sources",
                           "per_source": "source/horizon-specific training mean/std, fixed for every transfer"},
        "normalization_std_floor": 1e-6,
        "copy_normalization_parity_max_abs": copy_parity,
        "runtime_correction": {"previous_source_sha256": "f35fbb0726d87c761ae664034a9e89f0204e9ab4bcf51a816753e792c9ba05dc",
                               "reason": "match cached and public flattened encoder/predictor batch shapes; no dtype or tolerance change"},
        "ridge": "fixed alpha; mean squared loss; unpenalized intercept; no tuning or model selection",
        "physical_cohort": "past-only selected starts, censored separately at each future reset horizon",
        "hidden_gap_cohort": "all currently unrevealed walls, original wall gap, no future-survival censoring",
        "frozen_model_unchanged": True, "jax_version": jax.__version__,
        "devices": [str(x) for x in jax.devices()], "elapsed_seconds": round(time.time() - start, 3)}
    np.savez(args.out / "readouts.npz", **probe_arrays)
    np.savez_compressed(args.out / "predictions.npz", **prediction_arrays)
    if args.save_features:
        np.savez(args.out / "features.npz", **{f"{split}_{domain}": value for split, domains in all_features.items()
                                                for domain, value in domains.items()})
    (args.out / "config.json").write_text(json.dumps(configuration, indent=2))
    (args.out / "results.json").write_text(json.dumps({"model_seed": args.model_seed,
          "records": records, "geometry": geometry,
          "elapsed_seconds": configuration["elapsed_seconds"]}, indent=2, allow_nan=False))
    print("[done] " + json.dumps({"output": str(args.out), "records": len(records),
                                  "seconds": configuration["elapsed_seconds"]}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-seed", type=int, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-count", type=int, default=16384)
    p.add_argument("--eval-count", type=int, default=8192)
    p.add_argument("--selection-seed", type=int, default=20260919)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--min-fit", type=int, default=64)
    p.add_argument("--save-features", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
