"""Analyze final matched-PPO artifacts without importing training code or JAX.

python -m studies.revision.analyze_matched --self-test
python -m studies.revision.analyze_matched --require-complete

Only complete, validated three-arm policy-seed blocks enter contrasts. Bootstrap
intervals condition on one fixed encoder checkpoint; model seeds are never
treated as many independent fits by pooling their policy or evaluation seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np


ARMS = ("learned", "oracle_heads", "oracle_true")
METRICS = ("walls_passed", "survival")
CONTRASTS = {
    "exact_transitions_with_learned_heads": ("oracle_heads", "learned"),
    "true_signals_with_exact_transitions": ("oracle_true", "oracle_heads"),
    "complete_oracle_pipeline": ("oracle_true", "learned"),
}
MATCH_ARGS = ("config", "updates", "batch", "hor", "chunk", "mb", "epochs", "p_reset",
              "gamma", "lam", "lr", "clip", "vf", "ent", "torso")
LIMITATIONS = [
    "Policy-seed bootstrap intervals condition on each fixed model checkpoint and the fixed evaluation set.",
    "Episode bootstrap intervals condition on the fitted policies; they do not represent training-seed uncertainty.",
    "Only three independently fitted encoders are planned. Cross-model summaries are descriptive means and ranges, without model-family confidence intervals.",
    "Intervals are unadjusted descriptive 95% percentile intervals for two endpoints and three contrasts; no significance-selected claims or best-policy selection.",
    "oracle_heads freezes true terminal simulator states but retains learned reward and continuation signals. Its contrast with learned includes this terminal-support change.",
]


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_hash(array):
    """Hash semantic integer start indices, independent of NPZ packaging."""
    array = np.ascontiguousarray(array, dtype="<i8")
    digest = hashlib.sha256(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def source_identity(sources):
    """Source paths vary by host; preserve module suffixes and content hashes."""
    out = {}
    for name, digest in sources.items():
        name = str(name).replace("\\", "/")
        key = "studies/" + name.split("/studies/", 1)[1] if "/studies/" in name else Path(name).name
        if key in out and out[key] != digest:
            raise ValueError(f"ambiguous source identity: {key}")
        out[key] = digest
    return out


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_run(directory, expected_updates=150, expected_batch=512, expected_horizon=16):
    """Reject incomplete budgets and inconsistent metrics; never read best rows."""
    directory = Path(directory)
    paths = {name: directory / name for name in
             ("final.json", "config.json", "episodes.npz", "starts.npz", "ledger.jsonl")}
    for name, path in paths.items():
        require(path.is_file(), f"missing {name}")
    final = json.loads(paths["final.json"].read_text())
    config = json.loads(paths["config.json"].read_text())
    args = config["args"]
    require(args.get("config") == "gate", "this precommitted analysis is restricted to the gate task")
    require(args["batch"] == expected_batch and args["hor"] == expected_horizon,
            "rollout batch or horizon differs from the predeclared budget")
    for key, value in final.items():
        if isinstance(value, (int, float)):
            require(np.isfinite(value), f"nonfinite final value: {key}")
    require(final.get("budget_complete") is True, "budget_complete is not true")
    require(final["updates_completed"] == final["updates_requested"] == args["updates"] == expected_updates,
            "update count differs from the predeclared budget")
    arm, model_seed, policy_seed = final["arm"], int(final["model_seed"]), int(final["policy_seed"])
    require(arm in ARMS, f"unknown arm {arm}")
    require(args["arm"] == arm, "arm mismatch between final and config")
    for key, value in (("model_seed", model_seed), ("policy_seed", policy_seed)):
        require(config[key] == args[key] == value, f"{key} metadata mismatch")
    if config.get("model_cfg") is not None:
        require(config["model_cfg"].get("seed") == model_seed, "checkpoint training seed mismatch")
    if "torso" in final:
        require(final["torso"] == args.get("torso", "mlp"), "final policy torso mismatch")
    require(config["model_sha256"] == final["model_sha256"], "model hash metadata mismatch")
    require(bool(config.get("source_sha256")), "missing source hashes")
    rec = config["reconstruction"]
    require(rec["frames_checked"] > 0 and rec["frames_checked"] == rec["exact_frames"],
            "simulator reconstruction not fully verified")
    require(final["transitions"] == expected_updates * args["batch"] * args["hor"], "transition count mismatch")
    with np.load(paths["episodes.npz"], allow_pickle=False) as archive:
        returns = np.asarray(archive["returns"], dtype=float)
        lengths = np.asarray(archive["lengths"], dtype=float)
        walls = np.asarray(archive["walls"], dtype=float)
        for key, value in (("model_seed", model_seed), ("policy_seed", policy_seed), ("eval_seed", final["eval_seed"])):
            require(int(archive[key]) == value, f"episode {key} mismatch")
    require(returns.ndim == 1 and returns.size > 0, "episode returns must be a nonempty vector")
    n = returns.size
    require(lengths.shape == walls.shape == returns.shape, "episode vector shapes differ")
    require(n == final["n_ep"] == args["eval_episodes"], "evaluation episode count mismatch")
    require(final["eval_seed"] == args["eval_seed"], "evaluation key mismatch")
    require(all(np.isfinite(a).all() for a in (returns, lengths, walls)), "nonfinite episode outcome")
    require(np.isin(returns, [-1.0, 0.0]).all(), "unexpected gate return outside {-1,0}")
    require(((lengths >= 1) & (lengths <= 128) & (lengths == np.floor(lengths))).all(), "invalid episode length")
    require(((walls >= 0) & (walls == np.floor(walls))).all(), "invalid wall count")
    # Death at the time cap remains death; length==128 is not the survivor label.
    survive = (returns > -0.5).astype(float)
    metrics = {"return": returns.mean(), "length": lengths.mean(), "walls_passed": walls.mean(),
               "survival": survive.mean(), "survival_sem": survive.std() / np.sqrt(n),
               "deaths_per_100": 100 * (1 - survive).sum() / lengths.sum()}
    for key, value in metrics.items():
        require(np.isclose(final[key], value, rtol=1e-6, atol=1e-7), f"final {key} differs from episode outcomes")
    with np.load(paths["starts.npz"], allow_pickle=False) as archive:
        starts = np.asarray(archive["indices"])
        columns = archive["columns"].tolist()
    require(starts.shape == (expected_updates, args["batch"], 4), "start-index shape mismatch")
    require(np.issubdtype(starts.dtype, np.integer), "start indices must be integers")
    require(columns == ["context_start_t", "environment", "last_frame_t", "is_reset"], "start-index column mismatch")
    ledger = [json.loads(line) for line in paths["ledger.jsonl"].read_text().splitlines() if line.strip()]
    require([row["update"] for row in ledger] == list(range(1, expected_updates + 1)), "training ledger is incomplete or duplicated")
    require(ledger[-1]["transitions"] == final["transitions"], "final ledger transition mismatch")
    for row in ledger:
        for key in ("loss", "pg", "vf", "entropy", "reward", "done", "rollout_return", "alive"):
            require(key in row and np.isfinite(row[key]),
                    f"numerical training failure or missing {key} at update {row['update']}; planned seed remains unresolved")
    policy = directory / "model.pkl"
    if policy.is_file():
        require(file_hash(policy) == final["policy_sha256"], "saved final policy hash mismatch")
    signature = {key: args.get(key, "mlp" if key == "torso" else None) for key in MATCH_ARGS}
    terminal_diagnostics = {}
    for key in ("postterminal_alive_mass_fraction", "postterminal_reward_per_rollout", "true_terminal_fraction"):
        values = [row[key] for row in ledger if key in row]
        if values:
            terminal_diagnostics[key] = {"mean_over_updates": float(np.mean(values)), "final_update": values[-1]}
    return {"directory": str(directory.resolve()), "arm": arm, "model_seed": model_seed, "policy_seed": policy_seed,
            "final": final, "config": config, "protocol_signature": signature,
            "source_identity": source_identity(config["source_sha256"]), "start_indices_sha256": array_hash(starts),
            "artifact_sha256": {name: file_hash(path) for name, path in paths.items()},
            "policy_checkpoint_locally_verified": policy.is_file(), "terminal_diagnostics": terminal_diagnostics,
            "episode_values": {"walls_passed": walls, "survival": survive}}


def matching_errors(block):
    """A seed block must share all intended nuisance variables across arms."""
    first = block[ARMS[0]]
    checks = {
        "training starts": lambda r: r["start_indices_sha256"],
        "model checkpoint": lambda r: r["final"]["model_sha256"],
        "training dataset bytes": lambda r: r["config"]["data_sha256"],
        "training protocol": lambda r: r["protocol_signature"],
        "source version": lambda r: r["source_identity"],
        "policy architecture size": lambda r: r["config"]["policy_params"],
        "evaluation key": lambda r: r["final"]["eval_seed"],
        "evaluation slots": lambda r: r["final"]["n_ep"],
    }
    return [name + " differ across arms" for name, get in checks.items()
            if any(get(block[arm]) != get(first) for arm in ARMS[1:])]


def bootstrap_mean(values, samples, seed):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def stable_seed(base, label):
    return (base + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)) % (2**32)


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min": float(values.min()), "max": float(values.max()), "values": values.tolist()}


def analyze(root, model_seeds, policy_seeds, updates=150, bootstrap_samples=10000, bootstrap_seed=7341, label="primary",
            expected_batch=512, expected_horizon=16):
    root = Path(root)
    candidates, invalid, unplanned = {}, [], []
    for path in sorted(root.rglob("final.json")):
        try:
            row = load_run(path.parent, updates, expected_batch, expected_horizon)
            key = (row["model_seed"], row["policy_seed"], row["arm"])
            if key[0] not in model_seeds or key[1] not in policy_seeds:
                unplanned.append({"directory": row["directory"], "key": list(key)})
                continue
            candidates.setdefault(key, []).append(row)
        except (ValueError, KeyError, OSError, TypeError, EOFError, zipfile.BadZipFile) as error:
            invalid.append({"directory": str(path.parent.resolve()), "reason": str(error)})
    selected, missing, mismatches, model_results = [], [], [], {}
    for ms in model_seeds:
        blocks = []
        for ps in policy_seeds:
            block = {}
            for arm in ARMS:
                rows = candidates.get((ms, ps, arm), [])
                if len(rows) != 1:
                    missing.append({"model_seed": ms, "policy_seed": ps, "arm": arm,
                                    "reason": "absent or invalid" if not rows else "duplicate final artifacts"})
                else:
                    block[arm] = rows[0]
            if len(block) != len(ARMS):
                continue
            problems = matching_errors(block)
            if problems:
                mismatches.append({"model_seed": ms, "policy_seed": ps, "reasons": problems})
                continue
            blocks.append((ps, block))
        if not blocks:
            model_results[str(ms)] = {"status": "no_valid_complete_seed_blocks", "policy_seeds": []}
            continue
        # Within a purported fixed model seed, every policy must use one checkpoint and protocol.
        references = [b[ARMS[0]] for _, b in blocks]
        consistency = {"model checkpoint": lambda r:r["final"]["model_sha256"],
                       "training dataset bytes": lambda r:r["config"]["data_sha256"],
                       "evaluation key": lambda r:r["final"]["eval_seed"],
                       "evaluation slots": lambda r:r["final"]["n_ep"],
                       "training protocol": lambda r:r["protocol_signature"],
                       "source version": lambda r:r["source_identity"],
                       "policy architecture size": lambda r:r["config"]["policy_params"]}
        problems = [name + " changes across policy seeds" for name,get in consistency.items()
                    if any(get(r)!=get(references[0]) for r in references[1:])]
        if problems:
            mismatches.append({"model_seed":ms,"reasons":problems})
            blocks=[]
        if not blocks:
            model_results[str(ms)] = {"status": "metadata_mismatch", "policy_seeds": []}
            continue
        selected.extend(row for _, block in blocks for row in block.values())
        result = {"status": "complete" if len(blocks) == len(policy_seeds) else "partial",
                  "model_sha256": references[0]["final"]["model_sha256"], "policy_seeds": [ps for ps, _ in blocks],
                  "arms": {}, "contrasts": {}}
        for arm in ARMS:
            rows = [block[arm] for _, block in blocks]
            result["arms"][arm] = {metric: summarize_values([r["final"][metric] for r in rows]) for metric in METRICS}
            result["arms"][arm]["training_seconds"] = summarize_values([r["final"]["train_seconds"] for r in rows])
            result["arms"][arm]["discovery_count"] = sum(r["final"]["survival"] > .5 for r in rows)
        for name, (target, base) in CONTRASTS.items():
            contrast = {"target": target, "reference": base, "n_paired_policy_seeds": len(blocks)}
            for metric in METRICS:
                episode_delta = np.stack([block[target]["episode_values"][metric] - block[base]["episode_values"][metric]
                                          for _, block in blocks])
                policy_delta = episode_delta.mean(axis=1)
                seed = stable_seed(bootstrap_seed, f"{label}/{ms}/{name}/{metric}")
                stats = summarize_values(policy_delta)
                stats["paired_policy_bootstrap_95"] = bootstrap_mean(policy_delta, bootstrap_samples, seed)
                # The same evaluation slot is shared across policy seeds; resample slots jointly.
                stats["fixed_policies_paired_episode_bootstrap_95"] = bootstrap_mean(episode_delta.mean(axis=0), bootstrap_samples, seed + 1)
                stats["per_policy"] = [{"policy_seed": ps, "difference": float(policy_delta[j]),
                                        "paired_episode_bootstrap_95": bootstrap_mean(episode_delta[j], bootstrap_samples, seed + j + 2)}
                                       for j, (ps, _) in enumerate(blocks)]
                contrast[metric] = stats
            result["contrasts"][name] = contrast
        model_results[str(ms)] = result
    checkpoint_seeds = {}
    for ms, result in model_results.items():
        if result.get("model_sha256"):
            checkpoint_seeds.setdefault(result["model_sha256"], []).append(ms)
    for digest, seeds in checkpoint_seeds.items():
        if len(seeds) > 1:
            mismatches.append({"model_seeds": seeds, "reasons": ["identical checkpoint hash used for multiple model seeds"],
                               "model_sha256": digest})
    cross_model = {}
    for name in CONTRASTS:
        cross_model[name] = {}
        for metric in METRICS:
            values = {ms: result["contrasts"][name][metric]["mean"] for ms, result in model_results.items() if result.get("contrasts")}
            cross_model[name][metric] = {"per_model_mean": values, "n_fixed_models": len(values),
                                          "equal_model_mean_descriptive": float(np.mean(list(values.values()))) if values else None,
                                          "model_mean_range_descriptive": [min(values.values()), max(values.values())] if values else None,
                                          "confidence_interval": None}
    public_runs = [{key: value for key, value in row.items() if key != "episode_values"} for row in selected]
    complete = not missing and not mismatches and len(selected) == len(model_seeds) * len(policy_seeds) * len(ARMS)
    return {"schema_version": 1, "analysis_label": label, "runs_root": str(root.resolve()),
            "status": "complete" if complete else "incomplete", "primary_endpoints": list(METRICS),
            "planned_model_seeds": model_seeds, "planned_policy_seeds": policy_seeds, "requested_updates": updates,
            "requested_transitions": updates * expected_batch * expected_horizon,
            "selection": "final budget-complete artifacts only; complete validated three-arm seed blocks; no best-policy selection",
            "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed, "interval": "95% percentile"},
            "validated_runs": public_runs, "model_results": model_results, "cross_model_descriptive": cross_model,
            "missing_or_duplicate_runs": missing, "invalid_runs": invalid, "unplanned_runs_excluded": unplanned,
            "pairing_mismatches": mismatches, "limitations": LIMITATIONS}


def render_table(result):
    lines = [f"# {result['analysis_label']} matched-PPO final endpoints", "", f"Status: **{result['status']}**. Each row is one fixed checkpoint. Parentheses are policy-run sample SDs.", "",
             "| Model seed | Arm | Policy seeds | Walls/episode | Survival | Discovery |", "|---|---|---|---:|---:|---:|"]
    for ms, model in result["model_results"].items():
        for arm, values in model.get("arms", {}).items():
            def cell(metric, scale):
                v = values[metric]
                return f"{scale*v['mean']:.3f}" + (f" ({scale*v['sample_sd']:.3f})" if v["sample_sd"] is not None else "")
            lines.append(f"| {ms} | {arm} | {', '.join(map(str, model['policy_seeds']))} | {cell('walls_passed',1)} | {cell('survival',100)}% | {values['discovery_count']}/{len(model['policy_seeds'])} |")
    lines += ["", "Differences are target minus reference. Intervals resample paired policy seeds within one fixed checkpoint, conditional on the evaluation set.", "",
              "| Model seed | Contrast | Paired seeds | Δ walls [95% interval] | Δ survival percentage points [95% interval] |", "|---|---|---:|---:|---:|"]
    for ms, model in result["model_results"].items():
        for name, contrast in model.get("contrasts", {}).items():
            def cell(metric, scale):
                v = contrast[metric]; ci = v["paired_policy_bootstrap_95"]
                return f"{scale*v['mean']:+.3f}" + (f" [{scale*ci[0]:+.3f}, {scale*ci[1]:+.3f}]" if ci else " [not estimated]")
            lines.append(f"| {ms} | {contrast['target']} − {contrast['reference']} | {contrast['n_paired_policy_seeds']} | {cell('walls_passed',1)} | {cell('survival',100)} |")
    lines += ["", *["- " + item for item in result["limitations"]], ""]
    return "\n".join(lines)


def render_conclusions(result):
    lines = [f"Analysis status: {result['status']}. {len(result['validated_runs'])} validated final runs enter complete three-arm policy-seed blocks."]
    if result["status"] != "complete":
        lines.append("The planned comparison is incomplete; the following numbers are provisional and cannot establish the final endpoint.")
    for ms, model in result["model_results"].items():
        if not model.get("contrasts"):
            lines.append(f"Model seed {ms}: no eligible paired comparison.")
            continue
        lines.append(f"Model seed {ms}, {len(model['policy_seeds'])} paired policy seeds:")
        for contrast in model["contrasts"].values():
            lines.append(f"  {contrast['target']} minus {contrast['reference']}: {contrast['walls_passed']['mean']:+.4f} walls/episode and {100*contrast['survival']['mean']:+.2f} survival percentage points.")
    lines += ["These are effects within the saved checkpoints under this PPO protocol, not inference about a model family.", *result["limitations"]]
    if result["pairing_mismatches"]:
        lines.append(f"Excluded {len(result['pairing_mismatches'])} mismatched seed blocks; inspect the publication JSON before drawing conclusions.")
    return "\n".join(lines) + "\n"


def self_test():
    """Independent hand-calculable scientific checks, using synthetic artifact files."""
    def fixture(root, ps, arm, walls, returns, complete=True, shift=0):
        directory = root / "m0" / f"p{ps}" / arm; directory.mkdir(parents=True)
        n = len(walls); args = dict(arm=arm, model_seed=0, policy_seed=ps, updates=3, batch=2, hor=2,
                                   eval_seed=17, eval_episodes=n, config="gate", chunk=2, mb=2, epochs=1)
        config = dict(args=args, model_seed=0, policy_seed=ps, model_sha256="a"*64, data_sha256="b"*64,
                      source_sha256={"/remote/studies/revision/matched_ppo.py":"c"*64},
                      reconstruction=dict(frames_checked=2,exact_frames=2),policy_params=10)
        r=np.array(returns,float); l=np.full(n,128); w=np.array(walls,float); s=r>-.5
        final=dict(arm=arm,model_seed=0,policy_seed=ps,eval_seed=17,updates_completed=3,updates_requested=3,
                   budget_complete=complete,transitions=12,model_sha256="a"*64,policy_sha256="d"*64,n_ep=n,
                   train_seconds=1,return_=float(r.mean()),length=128.,walls_passed=float(w.mean()),survival=float(s.mean()),
                   survival_sem=float(s.std()/np.sqrt(n)),deaths_per_100=float(100*(~s).sum()/l.sum()))
        final["return"]=final.pop("return_")
        (directory/"final.json").write_text(json.dumps(final)); (directory/"config.json").write_text(json.dumps(config))
        np.savez(directory/"episodes.npz",returns=r,lengths=l,walls=w,eval_seed=17,model_seed=0,policy_seed=ps)
        np.savez(directory/"starts.npz",indices=np.arange(24).reshape(3,2,4)+shift,
                 columns=["context_start_t","environment","last_frame_t","is_reset"])
        (directory/"ledger.jsonl").write_text("\n".join(json.dumps(dict(update=u,transitions=u*4,loss=0.,pg=0.,vf=0.,
                                                                     entropy=1.,reward=0.,done=0.,rollout_return=0.,alive=1.)) for u in range(1,4)))
        return directory
    with tempfile.TemporaryDirectory(prefix="matched-analysis-test-") as tmp:
        root=Path(tmp)
        # Policy seed0: exact transitions +1 wall; seed1: -1 wall. Mean is0,
        # despite selecting the better policy alone giving a positive headline.
        for ps in (0,1):
            baseline=[0,0,0,0] if ps==0 else [2,2,2,2]
            fixture(root,ps,"learned",baseline,[-1,-1,0,0])
            fixture(root,ps,"oracle_heads",[1,1,1,1],[-1,-1,0,0])
            fixture(root,ps,"oracle_true",[2,2,2,2],[0,0,0,0])
        evaluate=lambda draws:analyze(root,[0],[0,1],updates=3,bootstrap_samples=draws,expected_batch=2,expected_horizon=2)
        result=evaluate(2000)
        require(result["status"]=="complete","synthetic valid design rejected")
        model=result["model_results"]["0"]
        delta=model["contrasts"]["exact_transitions_with_learned_heads"]["walls_passed"]
        require(delta["values"]==[1.,-1.] and delta["mean"]==0.,"paired policies were pooled or selected incorrectly")
        require(delta["paired_policy_bootstrap_95"]==[-1.,1.],"known two-policy bootstrap distribution incorrect")
        require(model["contrasts"]["true_signals_with_exact_transitions"]["survival"]["mean"]==.5,
                "death at length128 incorrectly counted as surviving")
        require(model["contrasts"]["true_signals_with_exact_transitions"]["walls_passed"]["paired_policy_bootstrap_95"]==[1.,1.],
                "constant paired effect not preserved")
        # Equal shape is insufficient: paired training starts must have exactly
        # the same semantic indices, otherwise the whole seed block is invalid.
        start_path=root/"m0/p0/oracle_heads/starts.npz"
        with np.load(start_path) as archive:
            original_indices=archive["indices"].copy();columns=archive["columns"].copy()
        np.savez(start_path,indices=original_indices+1,columns=columns)
        mismatched=evaluate(100)
        require(mismatched["model_results"]["0"]["policy_seeds"]==[1] and bool(mismatched["pairing_mismatches"]),
                "mismatched start keys were treated as paired")
        np.savez(start_path,indices=original_indices,columns=columns)
        # A numerically failed optimization remains an unresolved planned seed,
        # even if its saved sampled policy happens to produce finite returns.
        ledger_path=root/"m0/p0/oracle_heads/ledger.jsonl";original_ledger=ledger_path.read_text()
        broken_rows=[json.loads(line) for line in original_ledger.splitlines()]
        broken_rows[-1]["loss"]=float("nan")
        ledger_path.write_text("\n".join(map(json.dumps,broken_rows)))
        numerical=evaluate(100)
        require(numerical["status"]=="incomplete" and numerical["model_results"]["0"]["policy_seeds"]==[1]
                and "numerical training failure" in numerical["invalid_runs"][0]["reason"],"numerical failure counted as completed evidence")
        ledger_path.write_text(original_ledger)
        # An incomplete run must remove the complete seed block, not allow an
        # unpaired arm mean using a different seed population.
        path=root/"m0/p1/oracle_true/final.json";f=json.loads(path.read_text());f["budget_complete"]=False;path.write_text(json.dumps(f))
        partial=evaluate(100)
        require(partial["status"]=="incomplete" and len(partial["validated_runs"])==3,"incomplete budget leaked into contrast")
        require(partial["model_results"]["0"]["policy_seeds"]==[0],"asymmetric seed exclusion")
        # Deliberately corrupt saved final means: raw episode evidence wins.
        path=root/"m0/p0/learned/final.json";f=json.loads(path.read_text());f["walls_passed"]=99;path.write_text(json.dumps(f))
        broken=evaluate(100)
        require(not broken["validated_runs"],"mismatched summary accepted")
    require(array_hash(np.array([1,2],np.int32))==array_hash(np.array([1,2],np.int64)),"semantic start hash depends on integer dtype")
    require(bootstrap_mean([5.],100,1) is None,"one policy seed falsely yields uncertainty interval")
    print("PASS: paired policy effects, final-only selection, episode labels, incomplete/numerical-failure exclusion, start matching, summary consistency, semantic hashes, single-seed uncertainty.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root",type=Path,default=Path("runs/revision/matched"))
    parser.add_argument("--output-dir",type=Path,default=Path("output/revision/matched-analysis"))
    parser.add_argument("--model-seeds",default="0,1,2")
    parser.add_argument("--policy-seeds",default="0,1,2,3,4")
    parser.add_argument("--updates",type=int,default=150)
    parser.add_argument("--bootstrap-samples",type=int,default=10000)
    parser.add_argument("--bootstrap-seed",type=int,default=7341)
    parser.add_argument("--label",default="primary")
    parser.add_argument("--require-complete",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    seeds=lambda text:[int(value) for value in text.split(",")]
    model_seeds,policy_seeds=seeds(args.model_seeds),seeds(args.policy_seeds)
    require(len(set(model_seeds))==len(model_seeds) and len(set(policy_seeds))==len(policy_seeds),"duplicate planned seed")
    require(args.bootstrap_samples>=100,"use at least100 bootstrap draws")
    result=analyze(args.runs_root,model_seeds,policy_seeds,args.updates,args.bootstrap_samples,args.bootstrap_seed,args.label)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"publication.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (args.output_dir/"table.md").write_text(render_table(result))
    (args.output_dir/"conclusions.txt").write_text(render_conclusions(result))
    print(json.dumps({"status":result["status"],"validated_runs":len(result["validated_runs"]),"output_dir":str(args.output_dir)}))
    if args.require_complete and result["status"]!="complete":
        raise SystemExit(2)


if __name__=="__main__":
    main()
