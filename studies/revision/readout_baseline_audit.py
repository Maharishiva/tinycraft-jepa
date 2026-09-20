"""Post-hoc constant-baseline check; does not fit or select feature readouts.

The original forecast audit used a full-cohort training mean for every slice.
This check asks whether a slice-conditioned training constant is already better.
All constants use training labels only. Original outputs remain unchanged.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    rows = []
    sources = {}
    for seed in range(3):
        path = root / f"runs/revision/forecast/m{seed}/predictions.npz"
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = np.load(path)
        for horizon in (1, 2, 4, 8):
            for cohort in ("all", "existing_wall", "unrevealed_wall"):
                masks = {}
                for split in ("train", "eval"):
                    mask = data[f"{split}_h{horizon}_continuing"] & data[f"{split}_h{horizon}_wall_present"]
                    if cohort != "all":
                        mask &= data[f"{split}_context_wall_x"] != -99
                    if cohort == "unrevealed_wall":
                        mask &= data[f"{split}_context_wall_x"] > 40
                    masks[split] = mask
                train_y = data[f"train_h{horizon}_wall_x"][masks["train"]].astype(float)
                eval_y = data[f"eval_h{horizon}_wall_x"][masks["eval"]].astype(float)
                row = dict(model_seed=seed, horizon=horizon, cohort=cohort,
                           train_n=len(train_y), eval_n=len(eval_y))
                if not len(train_y) or not len(eval_y):
                    row["status"] = "empty_cohort"
                    rows.append(row)
                    continue
                for name, constant in (("mean", float(train_y.mean())),
                                       ("median", float(np.median(train_y)))):
                    row[f"train_{name}"] = constant
                    row[f"constant_{name}_mae_px"] = float(np.abs(eval_y - constant).mean())
                    row[f"constant_{name}_rmse_px"] = float(np.sqrt(np.square(eval_y - constant).mean()))
                for normalization in ("common", "per_source"):
                    for source, target in (("real", "real"), ("real", "forecast"),
                                           ("forecast", "forecast"), ("copy", "copy")):
                        key = f"{normalization}_h{horizon}_wall_position_{source}_to_{target}"
                        prediction = data[key][masks["eval"], 0].astype(float) * 63
                        row[key] = {
                            "mae_px": float(np.abs(eval_y - prediction).mean()),
                            "rmse_px": float(np.sqrt(np.square(eval_y - prediction).mean())),
                            "correlation": float(np.corrcoef(eval_y, prediction)[0, 1])
                            if np.std(prediction) > 0 and np.std(eval_y) > 0 else None,
                        }
                rows.append(row)
    output = root / "research/forecast_constant_baseline_audit.json"
    output.write_text(json.dumps({
        "status": "complete", "analysis_status": "post_hoc_interpretation_check",
        "scope": "Training-cohort constants only; no feature-probe refit or hyperparameter selection",
        "source_sha256": sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
    }, indent=2, allow_nan=False))
    for row in rows:
        if row["model_seed"] == 2 and row["horizon"] == 8:
            print(json.dumps(row))
    print(output)


if __name__ == "__main__":
    main()
