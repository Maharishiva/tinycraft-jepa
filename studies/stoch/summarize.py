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
        w["edge"] = collect_edge(cfg)
        out[cfg] = w
    out["craftax"] = load_json(RUNS / "craftax" / "tile_factored_plan.json")
    out["wallmass"] = load_json(RUNS / "gate" / "wallmass.json")
    out["gate"].update(collect_gate_round2())
    out["gray_episode_still"]["sigreg"] = collect_sigreg()
    return out


MODES = ("det", "mean8", "samp")
MODE_NAMES = {"det": "original predictor", "mean8": "residual head, mean of 8", "samp": "residual head, one sample"}
LEGACY = {"det": "ppo_jepa_tile_factored", "mean8": "ppo_jepa_tile_factored_mean8", "samp": "ppo_jepa_tile_factored_sampled"}
DISCOVERED = 0.5


def grid_run_name(ms, mode, ps):
    return LEGACY[mode] if (ms, ps) == (0, 0) else f"ppo_jepa_tf_m{ms}_{mode}_p{ps}"


def flow_run_name(mode, ps):
    if (mode, ps) == ("samp", 0):
        return "ppo_dream_latent"
    return f"ppo_dream_latent_{mode}_p{ps}"


def probe_full(log):
    """Heads quality plus the decoder's pixel MSE and the latent's spread from a probe log."""
    log = Path(log)
    if not log.exists():
        return None
    text = log.read_text()
    out = probe_metrics(log) or {}
    m = re.search(r"\[probe\] decoder mse ([0-9.]+)", text)
    if m:
        out["decoder_mse"] = float(m.group(1))
    m = re.search(r"\[probe\] latents \((\d+), (\d+)\) in [0-9.]+s mean ([-0-9.]+) std ([0-9.]+)", text)
    if m:
        out.update({"latent_dim": int(m.group(2)), "latent_mean": float(m.group(3)), "latent_std": float(m.group(4))})
    return out or None


def collect_gate_round2():
    g = RUNS / "gate"
    grid = {}
    for ms in range(3):
        for mode in MODES:
            for ps in range(3):
                r = last_eval(g / grid_run_name(ms, mode, ps))
                if r is not None:
                    grid[f"m{ms}_{mode}_p{ps}"] = r
    flow = {}
    for mode in ("samp", "mean8"):
        for ps in range(8):
            r = last_eval(g / flow_run_name(mode, ps))
            if r is not None:
                flow[f"{mode}_p{ps}"] = r
    real = {name: last_eval(g / f"ppo_{name}") for name in ("real", "real_s1", "real_s2")}
    real_latent = {f"m{ms}_p{ps}": last_eval(g / (f"ppo_real_jepa_latent_p{ps}" if ms == 0 else f"ppo_real_jepa_latent_m{ms}_p{ps}"))
                   for ms in range(3) for ps in range(3)}
    real_decoded = {f"m{ms}_p{ps}": last_eval(g / f"ppo_real_jepa_decoded_m{ms}_p{ps}") for ms in range(3) for ps in range(3)}
    real_amp = {f"g{gain}_m{ms}_p{ps}": last_eval(g / f"ppo_real_jepa_latent_amp{gain}_m{ms}_p{ps}") for gain in (4, 16) for ms in range(2) for ps in range(3)}
    real_probe = {f"{arm}4_m{ms}_p{ps}": last_eval(g / f"ppo_real_jepa_latent_{arm}4_m{ms}_p{ps}") for arm in ("wall", "rand", "dwall", "drand") for ms in range(2) for ps in range(3)}
    real_mlp = {f"p{ps}": last_eval(g / f"ppo_real_mlp_p{ps}") for ps in range(3)}
    decoded_mlp = {f"m{ms}_p{ps}": last_eval(g / f"ppo_real_jepa_decoded_mlp_m{ms}_p{ps}") for ms in range(2) for ps in range(3)}
    budget = {}
    for mult in (1, 2, 4):
        tag = "" if mult == 1 else f"_x{mult}"
        row = {"train": final_pred_loss(g / f"train_tile_factored{tag}.log"),
               "probe": probe_full(g / f"probe_tile_factored{tag}.log"), "ppo": {}}
        for mode in MODES:
            row["ppo"][mode] = last_eval(g / (grid_run_name(0, mode, 0) if mult == 1 else f"ppo_jepa_tf_x{mult}_{mode}_p0"))
        budget[f"x{mult}"] = row
    seeds = {f"s{ms}": {"train": final_pred_loss(g / f"train_tile_factored{'' if ms == 0 else f'_s{ms}'}.log"),
                        "probe": probe_full(g / f"probe_tile_factored{'' if ms == 0 else f'_s{ms}'}.log")} for ms in range(3)}
    oracle = {name: load_json(g / f"oracle_{name}.json") for name in ("h5_k4", "h5_k16", "h8_k4", "h5_leak")}
    tag = lambda ms: "" if ms == 0 else f"_s{ms}"
    return {"grid": grid, "flow": flow, "real_seeds": real, "real_latent": real_latent, "real_decoded": real_decoded, "real_amp": real_amp, "real_probe": real_probe, "real_mlp": real_mlp, "decoded_mlp": decoded_mlp, "budget": budget, "model_seeds": seeds,
            "oracle": oracle, "heads_audit": load_json(g / "heads_audit_tile_factored.json"),
            "latent_check": load_json(g / "latent_check.json"),
            "latent_checks": {f"s{ms}": load_json(g / f"latent_check{tag(ms)}.json") for ms in range(3)},
            "heads_audits": {f"s{ms}": load_json(g / f"heads_audit_tile_factored{tag(ms)}.json") for ms in range(3)}}


def collect_edge(cfg):
    d = RUNS / cfg
    e = {"baselines": load_json(d / "baselines_edge.json"), "control": {}, "ppo": {}}
    for arm in ARMS:
        e["control"][arm] = control_json(d / f"control_{arm}_edge")
        e["ppo"][f"jepa_{arm}"] = load_json(d / f"ppo_jepa_{arm}" / "eval_edge.json")
    for name in ("dream_pixels", "dream_latent", "real"):
        e["ppo"][name] = load_json(d / f"ppo_{name}" / "eval_edge.json")
    return e


SIGREG_ARMS = {"lewm6_lam0": 0.0, "lewm6_lam0.009": 0.009, "lewm6": 0.09, "lewm6_lam0.9": 0.9, "lewm6_d32": 0.09}


def collect_sigreg():
    d = RUNS / "gray_episode_still"
    out = {}
    for arm, lam in SIGREG_ARMS.items():
        src = OLD / "gray_episode_still" if arm == "lewm6" else d
        out[arm] = {"sigreg_weight": lam, "latent_dim": 32 if arm.endswith("d32") else 192,
                    "train": final_pred_loss(src / f"train_{arm}.log"), "probe": probe_full(src / f"probe_{arm}.log"),
                    "control": control_json(d / f"control_{arm}")}
    return out


def stats(rows, key):
    xs = [r[key] for r in rows if r is not None and key in r]
    if not xs:
        return None
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, "n": len(xs),
            "min": float(np.min(xs)), "max": float(np.max(xs)), "values": xs}


def pm(s, scale=1.0, digits=2):
    if s is None:
        return "--"
    if s["n"] == 1:
        return f"{scale * s['mean']:.{digits}f}"
    return f"{scale * s['mean']:.{digits}f}$\\pm${scale * s['std']:.{digits}f}"


def discovered(rows):
    rows = [r for r in rows if r is not None]
    if not rows:
        return "--"
    return f"{sum(r['survival'] >= DISCOVERED for r in rows)}/{len(rows)}"


def paired_table(w):
    """The round-2 main table: each model used with a point prediction and with a sample, PPO inside, graded in the
    real gate; the simulator planner, the state-blind baselines and PPO in the real environment as references."""
    grid, flow = w["grid"], w["flow"]
    rows = []
    for mode in MODES:
        finals = [grid[k]["final"] for k in grid if f"_{mode}_" in k]
        ms = sorted({k.split("_")[0] for k in grid if f"_{mode}_" in k})
        rows.append((f"grid JEPA, {MODE_NAMES[mode]}", len(ms), len(finals), finals))
    for mode, name in (("mean8", "flow model, mean of 8 samples"), ("samp", "flow model, one sample")):
        finals = [flow[k]["final"] for k in flow if k.startswith(mode)]
        rows.append((name, 1 if finals else 0, len(finals), finals))
    o = w["oracle"]
    rows += [("simulator, exhaustive search $h=5$ (no terminal value)", "--", "--", [o["h5_k4"] and o["h5_k4"]["oracle"]]),
             ("simulator, exhaustive search $h=8$ (no terminal value)", "--", "--", [o["h8_k4"] and o["h8_k4"]["oracle"]]),
             ("random", "--", "--", [w["baselines"] and w["baselines"]["random"]]),
             ("state-blind alternating", "--", "--", [w["baselines"] and w["baselines"].get("alternate")]),
             ("centre controller (true state)", "--", "--", [w["baselines"] and w["baselines"]["centre"]])]
    reals = [r["final"] for r in w["real_seeds"].values() if r]
    rows.append(("PPO in the real gate, pixels, CNN torso", "--", len(reals), reals))
    rm = [r["final"] for r in w.get("real_mlp", {}).values() if r]
    rows.append(("PPO in the real gate, pixels, MLP torso (latent policy class)", "--", len(rm), rm))
    rl = [r["final"] for r in w.get("real_latent", {}).values() if r]
    rows.append(("PPO in the real gate, frozen JEPA latent, MLP torso", len({k[:2] for k, r in w.get("real_latent", {}).items() if r}), len(rl), rl))
    rd = [r["final"] for r in w.get("real_decoded", {}).values() if r]
    rows.append(("PPO in the real gate, decoded JEPA latent, CNN torso", len({k[:2] for k, r in w.get("real_decoded", {}).items() if r}), len(rd), rd))
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "controller & model seeds & policies & walls passed & survival (\\%) & discovered \\\\", "\\midrule"]
    for name, nm, npol, finals in rows:
        finals = [f for f in finals if f is not None]
        if not finals:
            lines.append(f"{name} & {nm} & {npol} & -- & -- & -- \\\\")
            continue
        lines.append(f"{name} & {nm} & {npol} & {pm(stats(finals, 'walls_passed'))} & "
                     f"{pm(stats(finals, 'survival'), 100, 0)} & {discovered(finals) if isinstance(npol, int) and npol > 1 else '--'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def seeds_table(w):
    grid = w["grid"]
    lines = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
             " & & \\multicolumn{3}{c}{survival (\\%)} & \\multicolumn{3}{c}{walls passed} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             "predictor & model seed & PPO 0 & PPO 1 & PPO 2 & PPO 0 & PPO 1 & PPO 2 \\\\", "\\midrule"]
    for mode in MODES:
        for ms in range(3):
            cells = [grid.get(f"m{ms}_{mode}_p{ps}") for ps in range(3)]
            s = [pct(c["final"]["survival"]) if c else "--" for c in cells]
            wp = [num(c["final"]["walls_passed"], 2) if c else "--" for c in cells]
            lines.append(f"{MODE_NAMES[mode] if ms == 0 else ''} & {ms} & " + " & ".join(s + wp) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def latentctl_table(w):
    """Appendix: every control on the latent policy class in the real gate (torso, decoded input, amplified wall signal)."""
    def sel(d, pre):
        return [r["final"] for k, r in d.items() if r and k.startswith(pre)]
    rows = [("pixels, convolutional torso", [r["final"] for r in w["real_seeds"].values() if r]),
            ("pixels, MLP torso", [r["final"] for r in w.get("real_mlp", {}).values() if r]),
            ("frozen JEPA latent, MLP torso", [r["final"] for r in w.get("real_latent", {}).values() if r]),
            ("decoded JEPA latent, convolutional torso", [r["final"] for r in w.get("real_decoded", {}).values() if r]),
            ("decoded JEPA latent, MLP torso", [r["final"] for r in w.get("decoded_mlp", {}).values() if r])]
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "observation of the real-gate PPO & runs & walls passed & survival (\\%) & discovered \\\\", "\\midrule"]
    for name, finals in rows:
        if not finals:
            continue
        lines.append(f"{name} & {len(finals)} & {pm(stats(finals, 'walls_passed'))} & {pm(stats(finals, 'survival'), 100, 0)} & {discovered(finals)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def budget_table(w):
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "budget & steps & pred. loss & held-out reward $r$ & PPO inside: original & mean of 8 & one sample \\\\", "\\midrule"]
    for mult in (1, 2, 4):
        b = w["budget"][f"x{mult}"]
        t, p = b["train"], b["probe"]
        cells = [num(t and t["steps"], 0), num(t and t["pred"], 4), num(p and p.get("heldout_reward_corr"), 2)]
        for mode in MODES:
            r = b["ppo"][mode]
            cells.append("--" if r is None else f"{pct(r['final']['survival'])} / {num(r['final']['walls_passed'], 2)}")
        lines.append(f"{mult}$\\times$ & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def heads_table(h):
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "read-out on real latents & wall deaths & border deaths & alive \\\\", "\\midrule"]
    if h is None:
        lines.append("-- & -- & -- & -- \\\\")
    else:
        r = h["real"]
        lines += [f"transitions & {r['wall_deaths']} & {r['border_deaths']} & {r['transitions'] - r['deaths']} \\\\",
                  f"reward head, mean prediction & {num(r['reward_pred_on_wall_deaths'], 2)} & {num(r['reward_pred_on_border_deaths'], 2)} & {num(r['reward_pred_on_alive'], 2)} \\\\",
                  f"termination head, mean $p(\\text{{done}})$ & {num(r['p_done_wall_deaths'], 2)} & {num(r['p_done_border_deaths'], 2)} & {num(r['p_done_alive'], 2)} \\\\",
                  f"termination head, AUC against alive & {num(r['done_auc_wall_vs_alive'], 2)} & {num(r['done_auc_border_vs_alive'], 2)} & \\\\"]
        lines += ["\\midrule", "imagined return, horizon 8 & mean & fraction $>0$ & \\\\"]
        for k, name in (("deterministic", "original predictor"), ("sample", "residual head, one sample"), ("mean_of_8", "residual head, mean of 8")):
            if k in h["imagined"]:
                v = h["imagined"][k]
                lines.append(f"{name} & {num(v['imagined_return_mean'], 2)} & {pct(v['imagined_return_positive_fraction'])}\\% & \\\\")
        if "planner" in h:
            p = h["planner"]
            lines += ["\\midrule", f"planner's view, horizon {p['horizon']}, {p['plans']} plans & mean over plans {num(p['mean_over_plans'], 2)} & "
                      f"best plan {num(p['max_over_plans'], 2)} & best plan $>0$ in {pct(p['max_over_plans_positive_fraction'])}\\% \\\\"]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def latent_table(c):
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "squared latent distance, rollout step & 1 & 3 & 5 & 8 \\\\", "\\midrule"]
    if c is None:
        lines.append("-- & -- & -- & -- & -- \\\\")
    else:
        idx = [0, 2, 4, 7]
        g, f = c["grid"], c["flow"]
        def row(name, xs, digits=3):
            lines.append(f"{name} & " + " & ".join(num(xs[i], digits) for i in idx) + " \\\\")
        lines.append("\\multicolumn{5}{l}{grid JEPA (latent dimension %d)} \\\\" % g["latent_dim"])
        row("wall present vs absent (encoded true futures)", g["wall_amplitude_sq"])
        if "counterfactual_survives" in g:
            row("gap top vs bottom (encoded true futures, both branches survive)", g["counterfactual_survives"]["gap_amplitude_sq"])
        row("predictor's one-step residual (teacher forced, total)", g["one_step_residual_sq"])
        if "one_step_residual_along_wall_sq" in g:
            row("one-step residual along the wall direction", g["one_step_residual_along_wall_sq"])
            row("one-step residual orthogonal to it", g["one_step_residual_orthogonal_sq"])
            row("one-step prediction: mean wall coordinate ($0$ = no wall, $1$ = true latent)", g["one_step_wall_coordinate"], 2)
            row("same, contexts whose wall is still solid at the first predicted frame", g["one_step_wall_coordinate_unrevealed"], 2)
        row("original rollout to the recorded future", g["deterministic"]["to_recorded_sq"], 2)
        row("original rollout to the no-wall future", g["deterministic"]["to_nowall_sq"], 2)
        row("residual-head sample to the recorded future", g["sample"]["to_recorded_sq"], 2)
        row("residual-head mean of 8 to the recorded future", g["mean_of_8"]["to_recorded_sq"], 2)
        lines.append("\\midrule")
        lines.append("\\multicolumn{5}{l}{flow model (latent dimension %d)} \\\\" % f["latent_dim"])
        row("wall present vs absent (encoded true futures)", f["wall_amplitude_sq"], 0)
        fs = f.get("counterfactual_survives", f)
        row("gap top vs bottom (encoded true futures, both branches survive)", fs["gap_amplitude_sq"], 0)
        row("one sample: gap coordinate, $|\\cdot|$ ($1$ = a real gap), both survive", fs["sample"]["gap_coordinate_abs"], 2)
        row("mean of 8: gap coordinate, $|\\cdot|$, both survive", fs["mean_of_8"]["gap_coordinate_abs"], 2)
        pic = c["grid"]["pictures"]
        lines.append("\\midrule")
        lines.append("\\multicolumn{5}{l}{decoded wall intensity on the shared wall pixels} \\\\")
        row("grid JEPA, encoded true future", pic["recorded_future_encoded"]["shared_wall"], 2)
        row("grid JEPA, original rollout", pic["deterministic"]["shared_wall"], 2)
        row("flow model, one sample", c["flow"]["pictures"]["sample"]["shared_wall"], 2)
        row("flow model, mean of 8: recorded gap band", c["flow"]["pictures"]["mean_of_8"]["recorded_gap_band"], 2)
        row("flow model, mean of 8: other gap band", c["flow"]["pictures"]["mean_of_8"]["other_gap_band"], 2)
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def seeddiag_table(w):
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "model seed & wall amplitude 1 / 8 & one-step residual 1 / 8 & wall coordinate, step 1 & decoded wall 1 / 2 & AUC border & AUC wall & PPO inside, walls \\\\",
             "\\midrule"]
    for ms in range(3):
        c, h = w["latent_checks"].get(f"s{ms}"), w["heads_audits"].get(f"s{ms}")
        g = c and c["grid"]
        cells = [f"{num(g['wall_amplitude_sq'][0], 3)} / {num(g['wall_amplitude_sq'][7], 3)}" if g else "--",
                 f"{num(g['one_step_residual_sq'][0], 3)} / {num(g['one_step_residual_sq'][7], 3)}" if g else "--",
                 num(g["one_step_wall_coordinate"][0], 2) if g and "one_step_wall_coordinate" in g else "--",
                 f"{num(g['pictures']['deterministic']['shared_wall'][0], 2)} / {num(g['pictures']['deterministic']['shared_wall'][1], 2)}" if g else "--",
                 num(h["real"]["done_auc_border_vs_alive"], 2) if h else "--",
                 num(h["real"]["done_auc_wall_vs_alive"], 2) if h else "--"]
        finals = [w["grid"][k]["final"] for k in w["grid"] if k.startswith(f"m{ms}_")]
        cells.append(pm(stats(finals, "walls_passed")) if finals else "--")
        lines.append(f"{ms} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def sigreg_table(s):
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "SIGReg weight & latent & pred. loss & SIGReg & decoder MSE & held-out reward $r$ & goal MPC & heads MPC \\\\", "\\midrule"]
    for arm, v in s.items():
        t, p, c = v["train"], v["probe"], v["control"]
        goal = c and isinstance(c.get("goal"), dict) and c["goal"]
        plan = c and isinstance(c.get("plan"), dict) and c["plan"]
        lines.append(f"{v['sigreg_weight']} & {v['latent_dim']} & {num(t and t['pred'], 4)} & {num(t and t['sigreg'], 2)} & "
                     f"{num(p and p.get('decoder_mse'), 5)} & {num(p and p.get('heldout_reward_corr'), 2)} & "
                     f"{pct(goal['survival']) if goal else '--'} & {pct(plan['survival']) if plan else '--'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def edge_table(out):
    cols = [("random", lambda e: e["baselines"] and e["baselines"]["random"]),
            ("alternate", lambda e: e["baselines"] and e["baselines"].get("alternate")),
            ("centre", lambda e: e["baselines"] and e["baselines"]["centre"]),
            ("LeWM goal", lambda e: e["control"]["lewm6"] and e["control"]["lewm6"].get("goal")),
            ("LeWM heads", lambda e: e["control"]["lewm6"] and e["control"]["lewm6"].get("plan")),
            ("LeWM PPO", lambda e: e["ppo"]["jepa_lewm6"]),
            ("grid goal", lambda e: e["control"]["tile_factored"] and e["control"]["tile_factored"].get("goal")),
            ("grid heads", lambda e: e["control"]["tile_factored"] and e["control"]["tile_factored"].get("plan")),
            ("grid PPO", lambda e: e["ppo"]["jepa_tile_factored"]),
            ("flow PPO px", lambda e: e["ppo"]["dream_pixels"]),
            ("flow PPO z", lambda e: e["ppo"]["dream_latent"]),
            ("PPO real", lambda e: e["ppo"]["real"])]
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "world (edge starts) & " + " & ".join(c for c, _ in cols) + " \\\\", "\\midrule"]
    for cfg in CUBES:
        e = out[cfg]["edge"]
        cells = []
        for _, f in cols:
            r = f(e)
            cells.append("--" if not isinstance(r, dict) else pct(r["survival"]))
        lines.append(f"{WORLD_NAMES[cfg]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def markdown_round2(w):
    lines = ["", "gate round 2:"]
    for mode in MODES:
        for ms in range(3):
            cells = [w["grid"].get(f"m{ms}_{mode}_p{ps}") for ps in range(3)]
            lines.append(f"  grid {mode} m{ms}: " + " ".join("--" if c is None else f"{100 * c['final']['survival']:.0f}%/{c['final']['walls_passed']:.2f}" for c in cells))
    for mode in ("samp", "mean8"):
        cells = [w["flow"].get(f"{mode}_p{ps}") for ps in range(8)]
        lines.append(f"  flow {mode}: " + " ".join("--" if c is None else f"p{i} {100 * c['final']['survival']:.0f}%/{c['final']['walls_passed']:.2f}" for i, c in enumerate(cells) if c))
    for mult in (1, 2, 4):
        b = w["budget"][f"x{mult}"]
        lines.append(f"  budget x{mult}: train {b['train'] and b['train']['pred']} " + " ".join(f"{m} " + ("--" if b["ppo"][m] is None else f"{100 * b['ppo'][m]['final']['survival']:.0f}%/{b['ppo'][m]['final']['walls_passed']:.2f}") for m in MODES))
    return "\n".join(lines)


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
            ("LeWM PPO", lambda w: ppo_row(w, "jepa_lewm6")),
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
    (TABLES / "paired.tex").write_text(paired_table(out["gate"]))
    (TABLES / "seeds.tex").write_text(seeds_table(out["gate"]))
    (TABLES / "budget.tex").write_text(budget_table(out["gate"]))
    (TABLES / "heads.tex").write_text(heads_table(out["gate"]["heads_audit"]))
    (TABLES / "latent.tex").write_text(latent_table(out["gate"]["latent_check"]))
    (TABLES / "seeddiag.tex").write_text(seeddiag_table(out["gate"]))
    (TABLES / "latentctl.tex").write_text(latentctl_table(out["gate"]))
    (TABLES / "sigreg.tex").write_text(sigreg_table(out["gray_episode_still"]["sigreg"]))
    (TABLES / "edge.tex").write_text(edge_table(out))
    print(markdown(out))
    print(markdown_round2(out["gate"]))


if __name__ == "__main__":
    main()
