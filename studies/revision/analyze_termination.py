"""Separate prospective analysis of the true-termination sensitivity.

Read-only: pairs new oracle_reward policies with final oracle_heads/oracle_true
policies from the frozen primary harness. Does not modify primary analysis.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from studies.revision import analyze_matched as M


BASE_HASH = "6a3460b2db2e47713b7af604454a4b04066c96bfd7748b5c991cfbc433176269"
SENSITIVITY_HASH = "1c003dd4e019b2179981877310e131304d13a7150ac7da93e11d5164d011feb1"
BASE_SOURCE = "studies/revision/matched_ppo.py"
NEW_SOURCE = "studies/revision/matched_ppo_termination.py"
NEW_ARM = "oracle_reward"
REFERENCE_ARMS = ("oracle_heads", "oracle_true")
CONTRASTS = {
    "true_termination_under_learned_reward": ("oracle_reward", "oracle_heads"),
    "true_reward_under_true_termination": ("oracle_true", "oracle_reward"),
}
LIMITATIONS = [
    "This is an explicitly triggered sensitivity after inspecting primary terminal diagnostics; it is not an independently preplanned primary experiment.",
    "All contrasts condition on each saved encoder/head checkpoint, policy torso and training protocol. Three model fits do not establish a model-family effect.",
    "The termination contrast changes continuation, termination weighting and postterminal support together; it is not a reward-only intervention.",
    "The reward contrast keeps exact dynamics and true termination fixed, but policies trained under different rewards subsequently visit different states.",
    "Unadjusted percentile intervals describe paired policy-seed variation conditional on the fixed evaluation set; episode intervals condition on the fitted policies.",
    "MLP and native-grid policy results are separate strata and are never pooled as model replication.",
]


def load_reference(directory):
    """Require complete finite terminal diagnostics in the frozen references."""
    row = M.load_run(directory)
    ledger = [json.loads(line) for line in (Path(directory) / "ledger.jsonl").read_text().splitlines() if line.strip()]
    for update in ledger:
        for key in ("postterminal_alive_mass_fraction", "postterminal_reward_per_rollout", "true_terminal_fraction"):
            M.require(key in update and np.isfinite(update[key]),
                      f"reference terminal diagnostic missing/nonfinite: {key}, update {update['update']}")
        if row["arm"] == "oracle_true":
            M.require(update["postterminal_alive_mass_fraction"] == update["postterminal_reward_per_rollout"] == 0.,
                      f"true-signal reference has postterminal weighted mass/reward, update {update['update']}")
    return row


def load_new_run(directory):
    """Independently validate the additional arm and its explicit lineage."""
    directory = Path(directory)
    names = ("final.json", "config.json", "episodes.npz", "starts.npz", "ledger.jsonl")
    for name in names:
        M.require((directory / name).is_file(), f"missing {name}")
    final = json.loads((directory / "final.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    args = config["args"]
    M.require(final["arm"] == args["arm"] == NEW_ARM, "incorrect sensitivity arm")
    M.require(args["config"] == "gate", "not the frozen gate task")
    M.require(final.get("budget_complete") is True, "incomplete requested budget")
    M.require(final["updates_completed"] == final["updates_requested"] == args["updates"] == 150,
              "not the predeclared 150-update budget")
    M.require(args["batch"] == 512 and args["hor"] == 16 and final["transitions"] == 1228800,
              "not the predeclared transition budget")
    for key in ("model_seed", "policy_seed"):
        M.require(final[key] == config[key] == args[key], f"{key} metadata differs")
    M.require(config["model_cfg"]["seed"] == final["model_seed"], "checkpoint training seed differs")
    M.require(final["torso"] == args["torso"] and args["torso"] in ("mlp", "native_grid"), "torso differs")
    M.require(final["model_sha256"] == config["model_sha256"], "model hash metadata differs")
    M.require(config["parent_harness_sha256"] == BASE_HASH, "sensitivity does not name the approved parent harness")
    sources = M.source_identity(config["source_sha256"])
    M.require(sources.get(BASE_SOURCE) == BASE_HASH, "parent harness source hash differs")
    M.require(sources.get(NEW_SOURCE) == config["sensitivity_harness_sha256"], "sensitivity source hash differs")
    M.require(config["sensitivity_harness_sha256"] == SENSITIVITY_HASH, "sensitivity is not the approved minimal-diff harness")
    reconstruction = config["reconstruction"]
    M.require(reconstruction["frames_checked"] > 0 and reconstruction["frames_checked"] == reconstruction["exact_frames"],
              "simulator reconstruction not fully verified")
    for key, value in final.items():
        if isinstance(value, (int, float)):
            M.require(np.isfinite(value), f"nonfinite final {key}")
    with np.load(directory / "episodes.npz", allow_pickle=False) as archive:
        returns, lengths, walls = [np.asarray(archive[key], dtype=float) for key in ("returns", "lengths", "walls")]
        for key in ("model_seed", "policy_seed", "eval_seed"):
            M.require(int(archive[key]) == final[key], f"episode {key} differs")
    M.require(returns.ndim == 1 and len(returns) == 256 and returns.shape == lengths.shape == walls.shape,
              "not the predeclared 256 ordered evaluation slots")
    M.require(final["n_ep"] == args["eval_episodes"] == 256 and final["eval_seed"] == args["eval_seed"] == 20260919,
              "not the predeclared evaluation key/count")
    M.require(all(np.isfinite(a).all() for a in (returns, lengths, walls)), "nonfinite episode outcome")
    M.require(np.isin(returns, [-1., 0.]).all(), "unexpected gate return")
    M.require(((lengths >= 1) & (lengths <= 128) & (lengths == np.floor(lengths))).all(), "invalid episode length")
    M.require(((walls >= 0) & (walls == np.floor(walls))).all(), "invalid wall count")
    survival = (returns > -.5).astype(float)
    metrics = {"return": returns.mean(), "length": lengths.mean(), "survival": survival.mean(),
               "walls_passed": walls.mean(), "survival_sem": survival.std() / 16,
               "deaths_per_100": 100 * (1 - survival).sum() / lengths.sum()}
    for key, value in metrics.items():
        M.require(np.isclose(final[key], value, rtol=1e-6, atol=1e-7), f"final {key} contradicts episodes")
    with np.load(directory / "starts.npz", allow_pickle=False) as archive:
        starts = np.asarray(archive["indices"])
        columns = archive["columns"].tolist()
    M.require(starts.shape == (150, 512, 4) and np.issubdtype(starts.dtype, np.integer), "invalid start indices")
    M.require(columns == ["context_start_t", "environment", "last_frame_t", "is_reset"], "invalid start columns")
    ledger = [json.loads(line) for line in (directory / "ledger.jsonl").read_text().splitlines() if line.strip()]
    M.require([row["update"] for row in ledger] == list(range(1, 151)), "incomplete or duplicated training ledger")
    M.require(ledger[-1]["transitions"] == final["transitions"], "final ledger transition count differs")
    for row in ledger:
        for key in ("loss", "pg", "vf", "entropy", "reward", "done", "rollout_return", "alive",
                    "postterminal_alive_mass_fraction", "postterminal_reward_per_rollout", "true_terminal_fraction"):
            M.require(key in row and np.isfinite(row[key]), f"numerical failure or missing {key}, update {row['update']}")
        M.require(row["postterminal_alive_mass_fraction"] == row["postterminal_reward_per_rollout"] == 0.,
                  f"nonzero continuation-weighted postterminal mass/reward at update {row['update']}")
    policy = directory / "model.pkl"
    if policy.exists():
        M.require(M.file_hash(policy) == final["policy_sha256"], "final policy binary hash differs")
    return {"directory": str(directory.resolve()), "arm": NEW_ARM, "model_seed": final["model_seed"],
            "policy_seed": final["policy_seed"], "final": final, "config": config,
            "protocol_signature": {key: args.get(key, "mlp" if key == "torso" else None) for key in M.MATCH_ARGS},
            "source_identity": sources, "start_indices_sha256": M.array_hash(starts),
            "artifact_sha256": {name: M.file_hash(directory / name) for name in names},
            "policy_checkpoint_locally_verified": policy.exists(),
            "terminal_diagnostics": {key: {"mean_over_updates": float(np.mean([r[key] for r in ledger])),
                                           "final_update": ledger[-1][key]} for key in
                                     ("postterminal_alive_mass_fraction", "postterminal_reward_per_rollout", "true_terminal_fraction")},
            "episode_values": {"walls_passed": walls, "survival": survival}}


def block_errors(block):
    new = block[NEW_ARM]
    errors = []
    new_dependencies = {key: value for key, value in new["source_identity"].items() if key != NEW_SOURCE}
    for arm in REFERENCE_ARMS:
        reference = block[arm]
        if reference["source_identity"].get(BASE_SOURCE) != BASE_HASH:
            errors.append(f"{arm}: parent harness is not the approved frozen version")
        if reference["source_identity"] != new_dependencies:
            errors.append(f"{arm}: dependency changes beyond the approved additional harness")
        checks = {
            "model checkpoint": (new["final"]["model_sha256"], reference["final"]["model_sha256"]),
            "training data": (new["config"]["data_sha256"], reference["config"]["data_sha256"]),
            "training starts": (new["start_indices_sha256"], reference["start_indices_sha256"]),
            "training protocol": (new["protocol_signature"], reference["protocol_signature"]),
            "policy architecture size": (new["config"]["policy_params"], reference["config"]["policy_params"]),
            "evaluation key": (new["final"]["eval_seed"], reference["final"]["eval_seed"]),
            "evaluation slots": (new["final"]["n_ep"], reference["final"]["n_ep"]),
        }
        errors += [f"{arm}: {key} differs" for key, (left, right) in checks.items() if left != right]
    return errors


def analyze_stratum(new_root, reference_root, torso, policy_seeds, model_seeds=(0, 1, 2), draws=10000, seed=7341):
    candidates, invalid, unplanned = {}, [], []
    for root, expected in ((Path(new_root), (NEW_ARM,)), (Path(reference_root), REFERENCE_ARMS)):
        for path in sorted(root.rglob("final.json")):
            try:
                header = json.loads(path.read_text())
                if header.get("arm") not in expected:
                    continue
                row = load_new_run(path.parent) if header["arm"] == NEW_ARM else load_reference(path.parent)
                key = (row["model_seed"], row["policy_seed"], row["arm"])
                M.require(row["config"]["args"].get("torso", "mlp") == torso, "wrong torso in stratum")
                if key[0] not in model_seeds or key[1] not in policy_seeds:
                    unplanned.append({"directory": row["directory"], "key": list(key)})
                    continue
                candidates.setdefault(key, []).append(row)
            except (ValueError, KeyError, OSError, TypeError, EOFError, zipfile.BadZipFile) as error:
                invalid.append({"directory": str(path.parent.resolve()), "reason": str(error)})
    missing, mismatches, selected, models = [], [], [], {}
    for ms in model_seeds:
        blocks = []
        for ps in policy_seeds:
            block = {}
            for arm in (NEW_ARM, *REFERENCE_ARMS):
                found = candidates.get((ms, ps, arm), [])
                if len(found) != 1:
                    missing.append({"model_seed": ms, "policy_seed": ps, "arm": arm,
                                    "reason": "missing or invalid" if not found else "duplicate artifacts"})
                else:
                    block[arm] = found[0]
            if len(block) != 3:
                continue
            problems = block_errors(block)
            if problems:
                mismatches.append({"model_seed": ms, "policy_seed": ps, "reasons": problems})
            else:
                blocks.append((ps, block))
        if blocks:
            first = blocks[0][1][NEW_ARM]
            fields = {"checkpoint": lambda r:r["final"]["model_sha256"],
                      "data": lambda r:r["config"]["data_sha256"],
                      "protocol": lambda r:r["protocol_signature"],
                      "source": lambda r:r["source_identity"],
                      "policy size": lambda r:r["config"]["policy_params"]}
            problems = [key + " varies across policy seeds" for key, value in fields.items()
                        if any(value(block[NEW_ARM]) != value(first) for _, block in blocks)]
            if problems:
                mismatches.append({"model_seed": ms, "reasons": problems})
                blocks = []
        result = {"status": "complete" if len(blocks) == len(policy_seeds) else "incomplete",
                  "policy_seeds": [ps for ps, _ in blocks], "arms": {}, "contrasts": {}}
        if blocks:
            result["model_sha256"] = blocks[0][1][NEW_ARM]["final"]["model_sha256"]
            selected.extend(row for _, block in blocks for row in block.values())
            for arm in (NEW_ARM, *REFERENCE_ARMS):
                rows = [block[arm] for _, block in blocks]
                result["arms"][arm] = {metric: M.summarize_values([row["final"][metric] for row in rows]) for metric in M.METRICS}
                result["arms"][arm]["postterminal_alive_mass_fraction"] = M.summarize_values([
                    row["terminal_diagnostics"]["postterminal_alive_mass_fraction"]["mean_over_updates"] for row in rows])
            for name, (target, reference) in CONTRASTS.items():
                contrast = {"target": target, "reference": reference, "n_paired_policy_seeds": len(blocks)}
                for metric in M.METRICS:
                    delta = np.stack([block[target]["episode_values"][metric] - block[reference]["episode_values"][metric]
                                      for _, block in blocks])
                    policy_delta = delta.mean(axis=1)
                    local_seed = M.stable_seed(seed, f"termination/{torso}/{ms}/{name}/{metric}")
                    values = M.summarize_values(policy_delta)
                    values["paired_policy_bootstrap_95"] = M.bootstrap_mean(policy_delta, draws, local_seed)
                    values["fixed_policies_paired_episode_bootstrap_95"] = M.bootstrap_mean(delta.mean(axis=0), draws, local_seed + 1)
                    values["per_policy"] = [{"policy_seed": ps, "difference": float(policy_delta[j]),
                                             "paired_episode_bootstrap_95": M.bootstrap_mean(delta[j], draws, local_seed + j + 2)}
                                            for j, (ps, _) in enumerate(blocks)]
                    contrast[metric] = values
                result["contrasts"][name] = contrast
        models[str(ms)] = result
    hashes = [model["model_sha256"] for model in models.values() if "model_sha256" in model]
    if len(hashes) != len(set(hashes)):
        mismatches.append({"reasons": ["identical checkpoint binary assigned to multiple model seeds"]})
    cross_model = {}
    for name in CONTRASTS:
        cross_model[name] = {}
        for metric in M.METRICS:
            effects = {ms: model["contrasts"][name][metric]["mean"] for ms, model in models.items() if model["contrasts"]}
            cross_model[name][metric] = {"per_model_mean": effects, "equal_model_mean_descriptive": float(np.mean(list(effects.values()))) if effects else None,
                                          "model_mean_range_descriptive": [min(effects.values()), max(effects.values())] if effects else None,
                                          "confidence_interval": None}
    new_count = sum(row["arm"] == NEW_ARM for row in selected)
    complete = not missing and not mismatches and new_count == len(model_seeds) * len(policy_seeds)
    return {"torso": torso, "status": "complete" if complete else "incomplete", "planned_model_seeds": list(model_seeds),
            "planned_policy_seeds": policy_seeds, "new_runs_root": str(Path(new_root).resolve()),
            "reference_runs_root": str(Path(reference_root).resolve()), "n_validated_new_runs": new_count,
            "model_results": models, "cross_model_descriptive": cross_model, "missing_or_duplicate_runs": missing,
            "pairing_mismatches": mismatches, "invalid_runs": invalid, "unplanned_runs_excluded": unplanned,
            "validated_runs": [{key: value for key, value in row.items() if key != "episode_values"} for row in selected]}


def render(result):
    lines = ["# True-termination sensitivity", "", f"Status: **{result['status']}**. All contrasts are within fixed checkpoints and policy classes.", "",
             "| Torso | Model seed | Paired seeds | Contrast | Δ walls [95% policy interval] | Δ survival, pp [95% policy interval] |",
             "|---|---|---:|---|---:|---:|"]
    conclusions = [f"Termination sensitivity status: {result['status']}; {result['n_validated_new_runs']}/24 planned additional policies validated."]
    if result["status"] != "complete":
        conclusions.append("Incomplete and provisional: do not replace the primary conclusions with a selected subset.")
    for torso, stratum in result["strata"].items():
        for ms, model in stratum["model_results"].items():
            for contrast in model["contrasts"].values():
                def cell(metric, scale):
                    row = contrast[metric]; interval = row["paired_policy_bootstrap_95"]
                    return f"{scale*row['mean']:+.3f}" + (f" [{scale*interval[0]:+.3f}, {scale*interval[1]:+.3f}]" if interval else " [not estimated]")
                label = f"{contrast['target']} − {contrast['reference']}"
                lines.append(f"| {torso} | {ms} | {contrast['n_paired_policy_seeds']} | {label} | {cell('walls_passed',1)} | {cell('survival',100)} |")
                conclusions.append(f"{torso}, model seed {ms}: {label} changes walls/episode by {contrast['walls_passed']['mean']:+.4f} and survival by {100*contrast['survival']['mean']:+.2f} percentage points.")
    lines += ["", *["- " + limitation for limitation in LIMITATIONS], ""]
    conclusions += LIMITATIONS
    return "\n".join(lines), "\n".join(conclusions) + "\n"


def self_test():
    """Hand-calculable counterexamples for new lineage and terminal checks."""
    def write(root, ps, arm, walls, survivors):
        path = root / "m0" / f"p{ps}" / arm; path.mkdir(parents=True)
        args = dict(arm=arm,config="gate",model_seed=0,policy_seed=ps,torso="mlp",updates=150,batch=512,hor=16,
                    eval_seed=20260919,eval_episodes=256,chunk=16,mb=128,epochs=4,p_reset=.5,gamma=.99,lam=.9,
                    lr=.00025,clip=.2,vf=.5,ent=.01)
        sources = {"/remote/"+BASE_SOURCE:BASE_HASH,"/remote/studies/stoch/rl.py":"c"*64}
        config = dict(args=args,model_seed=0,policy_seed=ps,model_cfg=dict(seed=0),model_sha256="a"*64,
                      data_sha256="b"*64,source_sha256=sources,reconstruction=dict(frames_checked=1,exact_frames=1),policy_params=10)
        if arm == NEW_ARM:
            sources["/remote/"+NEW_SOURCE]=SENSITIVITY_HASH
            config.update(parent_harness_sha256=BASE_HASH,sensitivity_harness_sha256=SENSITIVITY_HASH)
        returns = np.full(256,-1.); returns[:survivors]=0.;lengths=np.full(256,128);w=np.full(256,walls,dtype=float)
        survival=(returns==0).astype(float)
        final=dict(arm=arm,torso="mlp",model_seed=0,policy_seed=ps,eval_seed=20260919,updates_completed=150,
                   updates_requested=150,budget_complete=True,transitions=1228800,model_sha256="a"*64,
                   policy_sha256="e"*64,n_ep=256,train_seconds=1.,walls_passed=walls,length=128.,
                   survival=float(survival.mean()),survival_sem=float(survival.std()/16),
                   deaths_per_100=float(100*(1-survival).sum()/lengths.sum()))
        final["return"]=float(returns.mean())
        (path/"final.json").write_text(json.dumps(final));(path/"config.json").write_text(json.dumps(config))
        np.savez_compressed(path/"episodes.npz",returns=returns,lengths=lengths,walls=w,eval_seed=20260919,model_seed=0,policy_seed=ps)
        np.savez_compressed(path/"starts.npz",indices=np.zeros((150,512,4),np.int32),
                            columns=["context_start_t","environment","last_frame_t","is_reset"])
        rows=[dict(update=u,transitions=u*8192,loss=0.,pg=0.,vf=0.,entropy=1.,reward=0.,done=.1,
                   rollout_return=0.,alive=.9,postterminal_alive_mass_fraction=0.,postterminal_reward_per_rollout=0.,
                   true_terminal_fraction=.2) for u in range(1,151)]
        (path/"ledger.jsonl").write_text("\n".join(map(json.dumps,rows)))
        return path
    with tempfile.TemporaryDirectory(prefix="termination-analysis-test-") as temporary:
        base=Path(temporary)/"base";new=Path(temporary)/"new"
        for ps in (0,1):
            write(base,ps,"oracle_heads",0 if ps==0 else 2,64)
            write(base,ps,"oracle_true",3,256)
            write(new,ps,NEW_ARM,1,128)
        analyze=lambda:analyze_stratum(new,base,"mlp",[0,1],model_seeds=[0],draws=1000)
        result=analyze();M.require(result["status"]=="complete","valid lineage rejected")
        contrasts=result["model_results"]["0"]["contrasts"]
        M.require(contrasts["true_termination_under_learned_reward"]["walls_passed"]["values"]==[1.,-1.],"paired seed effects incorrect")
        M.require(contrasts["true_reward_under_true_termination"]["walls_passed"]["paired_policy_bootstrap_95"]==[2.,2.],"known constant reward effect incorrect")
        M.require(contrasts["true_reward_under_true_termination"]["survival"]["mean"]==.5,"return-based survivor contrast incorrect")
        config_path=new/"m0/p0/oracle_reward/config.json";original=config_path.read_text();config=json.loads(original)
        config["source_sha256"]["/remote/studies/stoch/rl.py"]="f"*64;config_path.write_text(json.dumps(config))
        bad=analyze();M.require(bad["status"]=="incomplete" and bad["model_results"]["0"]["policy_seeds"]==[1],"unapproved dependency change accepted")
        config_path.write_text(original)
        # The source fingerprint exception permits only the audited new file,
        # not an arbitrary new implementation claiming the same parent.
        config=json.loads(original);config["sensitivity_harness_sha256"]="f"*64
        config["source_sha256"]["/remote/"+NEW_SOURCE]="f"*64;config_path.write_text(json.dumps(config))
        bad=analyze();M.require(any("approved minimal-diff" in row["reason"] for row in bad["invalid_runs"]),"unapproved sensitivity implementation accepted")
        config_path.write_text(original)
        reference_ledger=base/"m0/p0/oracle_heads/ledger.jsonl";original_reference=reference_ledger.read_text()
        reference_rows=[json.loads(line) for line in original_reference.splitlines()]
        reference_rows[0]["postterminal_alive_mass_fraction"]=float("nan")
        reference_ledger.write_text("\n".join(map(json.dumps,reference_rows)))
        bad=analyze();M.require(bad["status"]=="incomplete" and any("reference terminal" in row["reason"] for row in bad["invalid_runs"]),"nonfinite reference terminal evidence accepted")
        reference_ledger.write_text(original_reference)
        ledger_path=new/"m0/p0/oracle_reward/ledger.jsonl";rows=[json.loads(line) for line in ledger_path.read_text().splitlines()]
        rows[0]["postterminal_alive_mass_fraction"]=.1;ledger_path.write_text("\n".join(map(json.dumps,rows)))
        bad=analyze();M.require(bad["status"]=="incomplete" and any("postterminal" in row["reason"] for row in bad["invalid_runs"]),"continued death-state mass accepted")
    print("PASS: approved harness lineage, unchanged dependencies, paired policy effects, true-terminal zero mass, survival labels.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-root",type=Path,default=Path("runs/revision/termination-mlp"))
    parser.add_argument("--grid-root",type=Path,default=Path("runs/revision/termination-native-grid"))
    parser.add_argument("--mlp-reference-root",type=Path,default=Path("runs/revision/matched"))
    parser.add_argument("--grid-reference-root",type=Path,default=Path("runs/revision/matched_grid"))
    parser.add_argument("--output-dir",type=Path,default=Path("output/revision/termination-analysis"))
    parser.add_argument("--bootstrap-samples",type=int,default=10000)
    parser.add_argument("--bootstrap-seed",type=int,default=7341)
    parser.add_argument("--require-complete",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    M.require(args.bootstrap_samples>=100,"use at least100 bootstrap draws")
    strata={"mlp":analyze_stratum(args.mlp_root,args.mlp_reference_root,"mlp",list(range(5)),draws=args.bootstrap_samples,seed=args.bootstrap_seed),
            "native_grid":analyze_stratum(args.grid_root,args.grid_reference_root,"native_grid",list(range(3)),draws=args.bootstrap_samples,seed=args.bootstrap_seed)}
    result={"schema_version":1,"status":"complete" if all(s["status"]=="complete" for s in strata.values()) else "incomplete",
            "approved_parent_harness_sha256":BASE_HASH,"primary_endpoints":list(M.METRICS),"planned_additional_runs":24,
            "approved_sensitivity_harness_sha256":SENSITIVITY_HASH,
            "n_validated_new_runs":sum(s["n_validated_new_runs"] for s in strata.values()),
            "bootstrap":{"draws":args.bootstrap_samples,"seed":args.bootstrap_seed,"interval":"95% percentile"},
            "selection":"all24 admitted additional final policies; no best-checkpoint selection; separately reported torso strata",
            "strata":strata,"limitations":LIMITATIONS}
    table,conclusions=render(result);args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"publication.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (args.output_dir/"table.md").write_text(table);(args.output_dir/"conclusions.txt").write_text(conclusions)
    print(json.dumps({"status":result["status"],"validated_additional_runs":result["n_validated_new_runs"],"output_dir":str(args.output_dir)}))
    if args.require_complete and result["status"]!="complete":
        raise SystemExit(2)


if __name__=="__main__":
    main()
