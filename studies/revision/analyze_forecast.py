"""Prespecified descriptive slices of frozen forecast-readout predictions.

No model, probe, threshold or scaler is fitted by this program. Uncertainty is a
paired bootstrap of full evaluation environment streams, conditional on the
saved model/readouts. Ranking metrics are point estimates; errors, accuracy,
recall, specificity and Brier summaries receive stream-bootstrap intervals.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from studies.revision import forecast_readout as F


FORECAST_SHA256 = "37bbc37594579188613fd785670850d593caf62596f9d2f282d873c497f50b18"
NORMALIZATIONS = ("common", "per_source")
TRANSFERS = (("real", "real"), ("real", "forecast"), ("forecast", "forecast"), ("copy", "copy"))
TASKS = ("cube", "wall_presence", "wall_position", "hidden_original_gap")
SLICES = ("all_contexts", "current_wall_present", "current_wall_unrevealed")


def source_hashes(config):
    hashes = {}
    for name, value in config["source_sha256"].items():
        parts = Path(name).parts
        relative = "/".join(parts[parts.index("studies"):]) if "studies" in parts else Path(name).name
        if relative in hashes:
            raise ValueError(f"duplicate source identity {relative}")
        hashes[relative] = value
    return hashes


def load_run(path, expected_hash):
    config = json.loads((path / "config.json").read_text())
    results = json.loads((path / "results.json").read_text())
    if source_hashes(config).get("studies/revision/forecast_readout.py") != expected_hash:
        raise ValueError(f"unexpected forecast implementation hash in {path}")
    if not config.get("frozen_model_unchanged"):
        raise ValueError("missing frozen-model verification")
    if config.get("horizons") != list(F.HORIZONS) or set(config.get("normalizations", {})) != set(NORMALIZATIONS):
        raise ValueError("unexpected horizons or normalization protocol")
    if config.get("latent_dim") != 512 or config.get("history") != 8:
        raise ValueError("this analysis is scoped to the existing 512-dimensional eight-frame gate encoders")
    seed = config["args"]["model_seed"]
    if seed != results["model_seed"] or seed != config["model_cfg"]["seed"]:
        raise ValueError("model-seed metadata disagree")
    with np.load(path / "predictions.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    labels = {}
    for split in ("train", "eval"):
        labels[split] = {name.removeprefix(split + "_"): value for name, value in arrays.items()
                         if name.startswith(split + "_")}
        ids = labels[split]["ids"]
        selection = config["archive_info"][split]["selection"]
        if ids.ndim != 2 or ids.shape[1] != 2 or ids.dtype.kind not in "iu":
            raise ValueError("invalid context IDs")
        if len(np.unique(ids, axis=0)) != len(ids):
            raise ValueError("sampled contexts repeat")
        if F.array_hash(ids) != selection["selected_ids_sha256"] or len(ids) != selection["selected_contexts"]:
            raise ValueError("selected context identities disagree with provenance")
        if any(value.shape[0] != len(ids) for value in labels[split].values()):
            raise ValueError("label lengths disagree")
        meta = config["archive_info"][split]["meta"]
        if meta.get("split") != split or meta.get("policy") != "uniform random":
            raise ValueError("unexpected split or behavior-policy identity")
        spec = meta.get("spec", {})
        if not spec.get("gate") or spec.get("noise") != "off" or spec.get("wind"):
            raise ValueError("unexpected environment")
        context = labels[split]["context_wall_x"]
        expected_hidden = (context != F.NO_WALL) & (context > F.REVEAL_X)
        if not np.array_equal(labels[split]["hidden_gap_cohort"], expected_hidden):
            raise ValueError("hidden-gap cohort differs from its past-only definition")
        previous = np.ones(len(ids), bool)
        for h in F.HORIZONS:
            continuing = labels[split][f"h{h}_continuing"].astype(bool)
            if np.any(continuing & ~previous):
                raise ValueError("continuation masks are not nested across horizons")
            previous = continuing
            present = labels[split][f"h{h}_wall_x"] != F.NO_WALL
            if not np.array_equal(present, labels[split][f"h{h}_wall_present"]):
                raise ValueError("wall presence and wall position labels disagree")
    if config["train_sha256"] == config["eval_sha256"]:
        raise ValueError("train and evaluation data are identical")
    if config["archive_info"]["train"]["meta"]["seed"] == config["archive_info"]["eval"]["meta"]["seed"]:
        raise ValueError("train/evaluation collection seeds match")
    n = len(labels["eval"]["ids"])
    predictions = {name: value for name, value in arrays.items()
                   if name.startswith("common_") or name.startswith("per_source_")}
    if any(value.shape[0] != n or not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("prediction length mismatch or numerical failure")
    with np.load(path / "readouts.npz") as archive:
        if any(not np.isfinite(archive[name]).all() for name in archive.files):
            raise ValueError("nonfinite fitted readout/scaler")
    signature = {"train_sha256": config["train_sha256"], "eval_sha256": config["eval_sha256"],
                 "sources": source_hashes(config), "history": config["history"],
                 "latent_dim": config["latent_dim"], "normalizations": config["normalizations"],
                 "args": {key: config["args"][key] for key in
                          ("train_count", "eval_count", "selection_seed", "alpha", "min_fit")}}
    return {"path": path, "config": config, "labels": labels, "predictions": predictions,
            "results": results, "seed": seed, "signature": signature,
            "artifact_sha256": {name: F.sha256(path / name) for name in
                                ("config.json", "results.json", "readouts.npz", "predictions.npz")}}


def make_bootstrap(ids, reps, seed):
    envs, codes = np.unique(ids[:, 1], return_inverse=True)
    if reps < 1:
        raise ValueError("bootstrap repetitions must be positive")
    weights = np.random.default_rng(seed).multinomial(len(envs), np.full(len(envs), 1 / len(envs)), size=reps)
    return envs, codes, weights.astype(np.float64)


def summarize_bootstrap(point, draws, nonempty_streams):
    finite = np.isfinite(draws)
    usable = draws[finite]
    interval = None
    if nonempty_streams >= 2 and len(usable) >= 2 and point is not None:
        interval = [float(x) for x in np.quantile(usable, [0.025, 0.975])]
    return {"estimate": point, "ci95": interval, "valid_bootstrap_replicates": int(finite.sum())}


def stream_metrics(task, y, prediction, mask, codes, weights):
    """Pooled row statistics, with entire-stream multiplicities in every draw."""
    mask = np.asarray(mask, bool)
    ecount = weights.shape[1]
    counts = np.bincount(codes, weights=mask, minlength=ecount)
    nonempty = int((counts > 0).sum())
    metrics, draws = {}, {}

    def ratio(name, numerator, denominator=None, root=False):
        denominator = np.ones(len(mask)) if denominator is None else np.asarray(denominator)
        numerator = np.asarray(numerator)
        nums = np.bincount(codes, weights=np.where(mask, numerator, 0), minlength=ecount)
        dens = np.bincount(codes, weights=np.where(mask, denominator, 0), minlength=ecount)
        point = float(nums.sum() / dens.sum()) if dens.sum() > 0 else None
        totals, divisors = weights @ nums, weights @ dens
        sample = np.divide(totals, divisors, out=np.full(len(weights), np.nan), where=divisors > 0)
        if root:
            point = float(np.sqrt(max(point, 0))) if point is not None else None
            sample = np.sqrt(np.maximum(sample, 0))
        supporting = int((dens > 0).sum())
        metrics[name] = {**summarize_bootstrap(point, sample, supporting),
                         "supporting_eval_streams": supporting}
        draws[name] = sample

    if task == "cube":
        error = (prediction - y) * F.WIDTH
        ratio("mae_px_per_coordinate", np.abs(error).mean(axis=1))
        ratio("rmse_px_per_coordinate", np.square(error).mean(axis=1), root=True)
        ratio("mean_euclidean_error_px", np.linalg.norm(error, axis=1))
        ratio("x_mae_px", np.abs(error[:, 0]))
        ratio("y_mae_px", np.abs(error[:, 1]))
    elif task == "wall_position":
        error = (prediction[:, 0] - y[:, 0]) * F.WIDTH
        ratio("mae_px", np.abs(error))
        ratio("rmse_px", np.square(error), root=True)
    else:
        truth = y[:, 0] > 0.5
        score = prediction[:, 0]
        predicted = score >= 0.5
        ratio("accuracy", (predicted == truth).astype(float))
        ratio("positive_recall", (predicted & truth).astype(float), truth.astype(float))
        ratio("specificity", ((~predicted) & (~truth)).astype(float), (~truth).astype(float))
        ratio("brier_clipped_score", np.square(np.clip(score, 0, 1) - truth))
        ratio("mean_score", score)
        if truth[mask].any() and (~truth[mask]).any():
            point = (metrics["positive_recall"]["estimate"] + metrics["specificity"]["estimate"]) / 2
            sample = (draws["positive_recall"] + draws["specificity"]) / 2
            supporting = min(metrics["positive_recall"]["supporting_eval_streams"],
                             metrics["specificity"]["supporting_eval_streams"])
            metrics["balanced_accuracy"] = {**summarize_bootstrap(point, sample, supporting),
                                            "supporting_eval_streams": supporting}
            draws["balanced_accuracy"] = sample
            ranks = F.binary_metrics(truth[mask], score[mask])
            for name in ("auroc", "average_precision"):
                metrics[name] = {"estimate": ranks[name], "ci95": None,
                                 "uncertainty": "point-only ranking diagnostic"}
        else:
            for name in ("balanced_accuracy", "auroc", "average_precision"):
                metrics[name] = {"estimate": None, "ci95": None, "reason": "only one target class"}
    return metrics, draws, nonempty


def cohort_masks(labels, horizon, task):
    _, primary = F.task_data(labels, horizon, task)
    if task == "hidden_original_gap":
        # This negative control is intentionally NEVER survival-conditioned.
        return {"all_unrevealed_original_gap": primary}
    context_wall = labels["context_wall_x"]
    return {"all_contexts": primary,
            "current_wall_present": primary & (context_wall != F.NO_WALL),
            "current_wall_unrevealed": primary & (context_wall != F.NO_WALL) & (context_wall > F.REVEAL_X)}


def analyze(runs, reps, seed):
    first = runs[0]
    for run in runs[1:]:
        if run["signature"] != first["signature"]:
            raise ValueError("model runs do not share data, sources, readout settings and sampling protocol")
        for split in ("train", "eval"):
            if set(run["labels"][split]) != set(first["labels"][split]):
                raise ValueError("label schemas differ across models")
            if any(not np.array_equal(run["labels"][split][key], first["labels"][split][key])
                   for key in first["labels"][split]):
                raise ValueError("model runs use different selected labels or cohorts")
    seeds = [run["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate model seed")
    envs, codes, weights = make_bootstrap(first["labels"]["eval"]["ids"], reps, seed)
    rows, contrasts = [], []
    for run in runs:
        train, evaluation = run["labels"]["train"], run["labels"]["eval"]
        for horizon in F.HORIZONS:
            for task in TASKS:
                ytrain, train_mask = F.task_data(train, horizon, task)
                yeval, _ = F.task_data(evaluation, horizon, task)
                masks = cohort_masks(evaluation, horizon, task)
                if train_mask.sum() < run["config"]["args"]["min_fit"]:
                    rows.append({"model_seed": run["seed"], "horizon": horizon, "task": task,
                                 "status": "insufficient_training_data", "train_n": int(train_mask.sum())})
                    continue
                # One baseline per ORIGINAL task/horizon training cohort, reused
                # in every evaluation slice. Slices do not refit anything.
                baseline = np.broadcast_to(ytrain[train_mask].mean(axis=0), yeval.shape)
                for cohort, mask in masks.items():
                    base = {"model_seed": run["seed"], "horizon": horizon, "task": task, "cohort": cohort,
                            "train_n": int(train_mask.sum()), "eval_n": int(mask.sum()),
                            "fraction_of_all_contexts": float(mask.mean()), "eval_streams_total": len(envs)}
                    if not mask.any():
                        rows.append({**base, "status": "empty_evaluation_cohort"})
                        continue
                    if task in ("wall_presence", "hidden_original_gap"):
                        base.update(positive_count=int((yeval[:, 0][mask] > 0.5).sum()),
                                    negative_count=int((yeval[:, 0][mask] <= 0.5).sum()))
                    estimates, _, active = stream_metrics(task, yeval, baseline, mask, codes, weights)
                    rows.append({**base, "normalization": "none", "trained_on": "train_mean",
                                 "evaluated_on": "constant", "nonempty_eval_streams": active, "metrics": estimates})
                    for normalization in NORMALIZATIONS:
                        gathered = {}
                        for source, target in TRANSFERS:
                            name = f"{normalization}_h{horizon}_{task}_{source}_to_{target}"
                            if name not in run["predictions"]:
                                raise ValueError(f"missing required saved prediction {name} in {run['path']}")
                            pred = run["predictions"][name]
                            if pred.shape != yeval.shape:
                                raise ValueError(f"prediction shape differs from targets: {name}")
                            estimates, samples, active = stream_metrics(task, yeval, pred, mask, codes, weights)
                            rows.append({**base, "normalization": normalization, "trained_on": source,
                                         "evaluated_on": target, "nonempty_eval_streams": active, "metrics": estimates})
                            gathered[(source, target)] = estimates, samples
                        primary_metric = {"cube": "mae_px_per_coordinate", "wall_position": "mae_px",
                                          "wall_presence": "accuracy", "hidden_original_gap": "accuracy"}[task]
                        pairs = (("forecast_vs_real", ("forecast", "forecast"), ("real", "real")),
                                 ("transfer_vs_adapted", ("real", "forecast"), ("forecast", "forecast")),
                                 ("forecast_vs_copy", ("forecast", "forecast"), ("copy", "copy")))
                        for label, left, right in pairs:
                            lm, ld = gathered[left]
                            rm, rd = gathered[right]
                            difference = lm[primary_metric]["estimate"] - rm[primary_metric]["estimate"]
                            draws = ld[primary_metric] - rd[primary_metric]
                            contrasts.append({**base, "normalization": normalization, "contrast": label,
                                "metric": primary_metric, "left": list(left), "right": list(right),
                                "definition": "left minus right", "lower_metric_is_better": task in ("cube", "wall_position"),
                                "difference": summarize_bootstrap(difference, draws, active)})
    return rows, contrasts, envs, weights


def run(args):
    if F.sha256(Path(F.__file__)) != args.expected_forecast_sha256:
        raise ValueError("locally imported forecast definitions differ from the frozen analysis protocol")
    if not args.runs:
        raise ValueError("at least one result directory is required")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("refusing to overwrite nonempty analysis output")
    runs = [load_run(path, args.expected_forecast_sha256) for path in args.runs]
    rows, contrasts, envs, weights = analyze(runs, args.bootstrap_reps, args.bootstrap_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"analysis_source_sha256": F.sha256(Path(__file__)),
                "frozen_forecast_source_sha256": args.expected_forecast_sha256,
                "model_seeds": [run["seed"] for run in runs],
                "inputs": [{"path": str(run["path"]), "model_seed": run["seed"],
                            "model_sha256": run["config"]["model_sha256"],
                            "artifacts": run["artifact_sha256"]} for run in runs],
                "shared_protocol": runs[0]["signature"], "bootstrap_reps": args.bootstrap_reps,
                "bootstrap_seed": args.bootstrap_seed, "evaluation_stream_ids": envs.tolist(),
                "uncertainty_scope": "paired environment-stream resampling conditional on each fixed model and fitted probe; no training-seed uncertainty",
                "point_only_metrics": ["auroc", "average_precision"],
                "prediction_precision": "metrics recomputed from saved float32 predictions; no refitting",
                "interpretation": "descriptive retained-readout accessibility and transfer; failed probes do not establish information deletion"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    (args.out / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    (args.out / "paired_contrasts.json").write_text(json.dumps(contrasts, indent=2, allow_nan=False))
    np.savez_compressed(args.out / "bootstrap.npz", environment_ids=envs, multiplicities=weights.astype(np.int32))
    columns = ("model_seed", "horizon", "task", "cohort", "normalization", "trained_on", "evaluated_on",
               "eval_n", "nonempty_eval_streams", "metric", "estimate", "ci95_low", "ci95_high",
               "status", "reason", "valid_bootstrap_replicates", "supporting_eval_streams")
    with (args.out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            if "metrics" not in row:
                writer.writerow({**{key: row.get(key) for key in columns[:9]}, "status": row.get("status")})
            for name, value in row.get("metrics", {}).items():
                interval = value.get("ci95") or [None, None]
                writer.writerow({**{key: row.get(key) for key in columns[:9]}, "metric": name,
                                 "estimate": value.get("estimate"), "ci95_low": interval[0], "ci95_high": interval[1],
                                 "status": row.get("status", "available" if value.get("estimate") is not None else "undefined"),
                                 "reason": value.get("reason", value.get("uncertainty")),
                                 "valid_bootstrap_replicates": value.get("valid_bootstrap_replicates"),
                                 "supporting_eval_streams": value.get("supporting_eval_streams")})
    print(json.dumps({"output": str(args.out), "model_seeds": manifest["model_seeds"],
                      "rows": len(rows), "paired_contrasts": len(contrasts)}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20260921)
    p.add_argument("--expected-forecast-sha256", default=FORECAST_SHA256)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
