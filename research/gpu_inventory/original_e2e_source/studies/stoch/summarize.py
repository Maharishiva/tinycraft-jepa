"""Collect every result of the stochastic-control study into one JSON and the paper's tabulars.

    rsync -a multigpu-vm:~/personal/tinycraft-jepa/runs/stoch/ runs/stoch/
    rsync -a e2e:~/personal/tinycraft-jepa/runs/stoch/ runs/stoch/
    python -m studies.stoch.summarize            # -> runs/stoch/summary.json, studies/stoch/paper/tables/*.tex
"""
import json
import re
from pathlib import Path

import numpy as np

RUNS = Path("runs/stoch")
OLD = Path("runs/redcube")
TABLES = Path("studies/stoch/paper/tables")
CUBES = ["off_still", "off_wind", "gray_frame_still", "gray_frame_wind", "gray_episode_still", "gray_episode_wind"]
WORLDS = ["gate"] + CUBES
WORLD_NAMES = {"gate": "gate", "off_still": "black / still", "off_wind": "black / wind",
               "gray_frame_still": "noise per frame / still", "gray_frame_wind": "noise per frame / wind",
               "gray_episode_still": "noise per episode / still", "gray_episode_wind": "noise per episode / wind"}
ARMS = {"lewm6": "LeWM (class token)", "tile_factored": "grid JEPA"}


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def last_eval(run_dir):
    """Final and best evaluation rows of a PPO run."""
    ledger = Path(run_dir) / "ledger.jsonl"
    if not ledger.exists() or not Path(str(ledger.parent) + ".done").exists():
        return None
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    best = max(rows, key=lambda r: r["survival"])
    return {"final": rows[-1], "best": best, "n_eval": len(rows)}


WARM_UP = 7


def from_reset(row, steps):
    """Rows written before control.py counted from the reset: the warm-up is added back and reaching the cap
    (length == steps - warm-up, since the env's done also fires at the cap) is a survival, not a death."""
    if "lengths" not in row:
        out = dict(row)
        out["length"] = row["length"] + WARM_UP
        out["deaths_per_100"] = 100.0 * (1.0 - row["survival"]) / out["length"]
        return out
    lengths = np.asarray(row["lengths"]) + WARM_UP
    alive = (lengths >= steps).astype(np.float64)
    out = dict(row)
    out.update({"lengths": lengths.tolist(), "length": float(lengths.mean()), "survival": float(alive.mean()),
                "survival_sem": float(alive.std() / np.sqrt(len(alive))),
                "deaths_per_100": float(100.0 * (1.0 - alive).sum() / lengths.sum())})
    return out


def control_json(stem):
    """control_<arm>.json; the goal policy's row is recovered from the log where an early run overwrote it."""
    c = load_json(Path(str(stem) + ".json"))
    log = Path(str(stem) + ".log")
    if c is not None and not isinstance(c.get("goal"), dict) and log.exists():
        m = re.search(r"\[control\] goal: (\{.*\})", log.read_text())
        if m:
            c["goal"] = json.loads(m.group(1))
    if c is not None and "protocol" not in c:
        for k, v in list(c.items()):
            if isinstance(v, dict) and "length" in v:
                c[k] = from_reset(v, c.get("steps", 128))
    return c


def final_pred_loss(log):
    log = Path(log)
    if not log.exists():
        return None
    m = re.findall(r"\[train\] (\d+)/(\d+) pred ([0-9.]+) sigreg ([0-9.]+)\s+([0-9.]+)s", log.read_text())
    if not m:
        return None
    steps, total, pred, sig, sec = m[-1]
    return {"steps": int(steps), "pred": float(pred), "sigreg": float(sig), "train_s": float(sec)}


def probe_metrics(log):
    log = Path(log)
    if not log.exists():
        return None
    m = re.search(r"\[probe\] heads (\{.*\})", log.read_text())
    return json.loads(m.group(1)) if m else None


def flow_eval(cfg):
    """The recipe's eval.py prints its JSON row to stdout (eval_flow.log) as well as appending it to --out."""
    for path in (RUNS / cfg / "eval_flow.jsonl", RUNS / cfg / "eval_flow.log"):
        if path.exists():
            rows = [l for l in path.read_text().splitlines() if l.startswith("{") and '"gFVD"' in l]
            if rows:
                return json.loads(rows[-1])
    return None


def collect():
    out = {}
    for cfg in WORLDS:
        w = {"baselines": load_json(RUNS / cfg / "baselines.json"), "control": {}, "ppo": {}, "models": {}}
        for arm in ARMS:
            w["control"][arm] = control_json(RUNS / cfg / f"control_{arm}")
            w["control"][f"{arm}_sampled"] = control_json(RUNS / cfg / f"control_{arm}_sampled")
            w["ppo"][f"jepa_{arm}"] = last_eval(RUNS / cfg / f"ppo_jepa_{arm}")
            w["ppo"][f"jepa_{arm}_sampled"] = last_eval(RUNS / cfg / f"ppo_jepa_{arm}_sampled")
            src = RUNS / cfg if cfg == "gate" else OLD / cfg
            w["models"][arm] = {"train": final_pred_loss(src / f"train_{arm}.log"),
                                "probe": probe_metrics(src / f"probe_{arm}.log")}
        for name in ("dream_pixels", "dream_latent", "real", "real_e2e", "real_long", "real_s1", "real_s2", "real_long_s1", "real_long_s2"):
            w["ppo"][name] = last_eval(RUNS / cfg / f"ppo_{name}")
        w["models"]["flow"] = flow_eval(cfg)
        out[cfg] = w
    out["craftax"] = load_json(RUNS / "craftax" / "tile_factored_plan.json")
    out["wallmass"] = load_json(RUNS / "gate" / "wallmass.json")
    return out


def pct(x, sem=None):
    if x is None:
        return "--"
    return f"{100 * x:.0f}" + (f"$\\pm${100 * sem:.0f}" if sem is not None else "")


def num(x, digits=1):
    return "--" if x is None else f"{x:.{digits}f}"


def control_row(w, arm, policy):
    c = w["control"].get(arm)
    r = c.get(policy) if c else None
    return r if isinstance(r, dict) else None


def ppo_row(w, name):
    p = w["ppo"].get(name)
    return p["final"] if p else None


def gate_table(w):
    rows = [("random", w["baselines"] and w["baselines"]["random"]),
            ("centre controller (true state)", w["baselines"] and w["baselines"]["centre"])]
    for arm, label in ARMS.items():
        rows += [(f"{label}: goal-image MPC", control_row(w, arm, "goal")),
                 (f"{label}: heads MPC", control_row(w, arm, "plan")),
                 (f"{label}: PPO inside the JEPA", ppo_row(w, f"jepa_{arm}"))]
        if w["control"].get(f"{arm}_sampled") or w["ppo"].get(f"jepa_{arm}_sampled"):
            rows += [(f"{label} + sampling predictor: heads MPC (4 samples)", control_row(w, f"{arm}_sampled", "plan")),
                     (f"{label} + sampling predictor: PPO inside", ppo_row(w, f"jepa_{arm}_sampled"))]
    rows += [("flow model: PPO inside, pixels", ppo_row(w, "dream_pixels")),
             ("flow model: PPO inside, latent", ppo_row(w, "dream_latent")),
             ("PPO in the real environment, 1.5M steps", ppo_row(w, "real")),
             ("PPO in the real environment, 30 min (6.9M steps)", ppo_row(w, "real_long"))]
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "controller & survival (\\%) & length & walls passed & deaths / 100 steps \\\\", "\\midrule"]
    for name, r in rows:
        if r is None:
            lines.append(f"{name} & -- & -- & -- & -- \\\\")
        else:
            lines.append(f"{name} & {pct(r['survival'], r.get('survival_sem'))} & {num(r['length'])} & "
                         f"{num(r.get('walls_passed'), 2)} & {num(r['deaths_per_100'], 2)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def cube_table(out):
    cols = [("random", lambda w: w["baselines"] and w["baselines"]["random"]),
            ("centre", lambda w: w["baselines"] and w["baselines"]["centre"]),
            ("LeWM goal", lambda w: control_row(w, "lewm6", "goal")),
            ("LeWM heads", lambda w: control_row(w, "lewm6", "plan")),
            ("grid goal", lambda w: control_row(w, "tile_factored", "goal")),
            ("grid heads", lambda w: control_row(w, "tile_factored", "plan")),
            ("grid PPO", lambda w: ppo_row(w, "jepa_tile_factored")),
            ("flow PPO px", lambda w: ppo_row(w, "dream_pixels")),
            ("flow PPO z", lambda w: ppo_row(w, "dream_latent")),
            ("PPO real", lambda w: ppo_row(w, "real"))]
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "world & " + " & ".join(c for c, _ in cols) + " \\\\", "\\midrule"]
    for cfg in CUBES:
        w = out[cfg]
        cells = []
        for _, f in cols:
            r = f(w)
            cells.append("--" if r is None else pct(r["survival"]))
        lines.append(f"{WORLD_NAMES[cfg]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def models_table(out):
    lines = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{LeWM (class token)} & \\multicolumn{2}{c}{grid JEPA} & \\multicolumn{4}{c}{flow model} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}",
             "world & pred. loss & reward $r$ & pred. loss & reward $r$ & gFVD & AF & done AUC & im. done AUC \\\\", "\\midrule"]
    for cfg in WORLDS:
        m = out[cfg]["models"]
        cells = []
        for arm in ARMS:
            t, p = m[arm]["train"], m[arm]["probe"]
            cells += [num(t and t["pred"], 4), num(p and p.get("heldout_reward_corr"), 2)]
        f = m["flow"]
        cells += [num(f and f.get("gFVD"), 4), num(f and f.get("AF"), 3), num(f and f.get("done_auc"), 3),
                  num(f and f.get("imagined_done_auc"), 3)]
        lines.append(f"{WORLD_NAMES[cfg]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def craftax_table(c):
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "grid JEPA on Craftax-Classic & planned & random & oracle \\\\", "\\midrule"]
    if c and "craftax" in c:
        k = c["craftax"]
        lines.append("closed loop: goal reached (fraction) & " + " & ".join(num(k[n]["reached"], 3) for n in ("planned", "random", "oracle")) + " \\\\")
        lines.append("closed loop: position error ($L_1$) & " + " & ".join(num(k[n]["position_l1"], 2) for n in ("planned", "random", "oracle")) + " \\\\")
    else:
        lines += ["closed loop: goal reached (fraction) & -- & -- & -- \\\\", "closed loop: position error ($L_1$) & -- & -- & -- \\\\"]
    if c and "reward_mpc" in c:
        r = c["reward_mpc"]
        keys = [k for k in r if isinstance(r[k], dict) and "return" in r[k]]
        lines.append("\\midrule")
        for k in keys:
            lines.append(f"reward MPC: {k} return & {num(r[k]['return'], 2)} & & \\\\")
        if "behaviour_policy" in r:
            lines.append(f"behaviour policy return & {num(r['behaviour_policy'], 2)} & & \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def wallmass_table(r):
    """Wall retention of each model's 8-step rollout from pre-reveal contexts (runs/stoch/gate/wallmass.json)."""
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "model & wall mass kept, step 1 & step 4 & step 8 & gap in the last frame \\\\", "\\midrule"]
    if r is None:
        lines += ["-- & -- & -- & -- & -- \\\\"]
    else:
        def row(name, v, gap):
            m = v["rollout_white_mass_ratio"]
            lines.append(f"{name} & {100 * m[0]:.0f}\\% & {100 * m[3]:.0f}\\% & {100 * m[7]:.0f}\\% & {gap} \\\\")
        row("LeWM (class token): latent of the real future, decoded", {"rollout_white_mass_ratio": r["lewm"]["encoded_white_mass_ratio"]}, "--")
        row("LeWM (class token): mean rollout", r["lewm"], f"undecided in {100 * r['lewm']['rollout_gap_side_undecided']:.0f}\\%")
        row("grid JEPA: latent of the real future, decoded", {"rollout_white_mass_ratio": r["grid"]["encoded_white_mass_ratio"]}, "--")
        row("grid JEPA: mean rollout", r["grid"], f"undecided in {100 * r['grid']['rollout_gap_side_undecided']:.0f}\\%")
        if "grid_sampled" in r:
            g = r["grid_sampled"]
            row("grid JEPA + sampling predictor: sampled rollout (4 samples)", g, f"top in {100 * g['sample_gap_top_fraction']:.0f}\\% of samples; both sides seen in {100 * g['contexts_with_both_gaps_across_samples']:.0f}\\% of contexts")
        f = r["flow"]
        row("flow model: sampled rollout (4 samples)", f, f"top in {100 * f['sample_gap_top_fraction']:.0f}\\% of samples; both sides seen in {100 * f['contexts_with_both_gaps_across_samples']:.0f}\\% of contexts")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def markdown(out):
    lines = ["| world | random | centre | LeWM goal | LeWM heads | LeWM PPO | grid goal | grid heads | grid PPO | grid+samp MPC | grid+samp PPO | flow PPO px | flow PPO z | PPO real |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in WORLDS:
        w = out[cfg]
        r = [w["baselines"] and w["baselines"]["random"], w["baselines"] and w["baselines"]["centre"],
             control_row(w, "lewm6", "goal"), control_row(w, "lewm6", "plan"), ppo_row(w, "jepa_lewm6"),
             control_row(w, "tile_factored", "goal"), control_row(w, "tile_factored", "plan"), ppo_row(w, "jepa_tile_factored"),
             control_row(w, "tile_factored_sampled", "plan"), ppo_row(w, "jepa_tile_factored_sampled"),
             ppo_row(w, "dream_pixels"), ppo_row(w, "dream_latent"), ppo_row(w, "real")]
        lines.append(f"| {WORLD_NAMES[cfg]} | " + " | ".join("--" if x is None else f"{100 * x['survival']:.0f}" for x in r) + " |")
    return "\n".join(lines)


def main():
    out = collect()
    (RUNS / "summary.json").write_text(json.dumps(out, indent=1))
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "gate.tex").write_text(gate_table(out["gate"]))
    (TABLES / "cube.tex").write_text(cube_table(out))
    (TABLES / "models.tex").write_text(models_table(out))
    (TABLES / "craftax.tex").write_text(craftax_table(out["craftax"]))
    (TABLES / "wallmass.tex").write_text(wallmass_table(out["wallmass"]))
    print(markdown(out))


if __name__ == "__main__":
    main()
