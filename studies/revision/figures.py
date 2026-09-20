"""Publication figures from frozen results; no training, relabeling, or image editing.

Run from the worktree root::

    /private/tmp/tinycraft-figenv/bin/python studies/revision/figures.py

All numeric panels retain model seeds. Within-model policy means are descriptive;
they are never treated as independent world-model replications. Raw image arrays
are rendered with nearest-neighbor interpolation and their original RGB values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

INK = "#17324d"
JEPA = "#cd663e"
SIM = "#087e83"
FLOW = "#7860a5"
GRAY = "#637583"
GRID = "#dce4e8"
PAPER = "#ffffff"
PALE = "#f2f5f6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": GRID, "xtick.color": INK, "ytick.color": INK,
    "axes.titleweight": "bold", "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": PAPER,
})


def clean(ax, grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=2.5, width=.6, pad=3)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=.55, zorder=0)
    ax.set_axisbelow(True)


def heading(fig, title, subtitle=None):
    fig.text(.035, .965, title, fontsize=10, fontweight="bold", va="top")
    if subtitle:
        fig.text(.035, .91, subtitle, fontsize=7.5, va="top", color=GRAY)


def export(fig, out, stem, record):
    # Fixed physical size makes all text reviewable at the 5.5-inch ICLR width.
    for suffix in ("svg", "pdf", "png"):
        options = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(out / f"{stem}.{suffix}", **options)
    record["width_inches"], record["height_inches"] = map(float, fig.get_size_inches())
    record["files"] = {x: f"{stem}.{x}" for x in ("svg", "pdf", "png")}
    plt.close(fig)
    return record


def rollouts(z, context, out, full=False):
    rows = [
        ("Simulator", "recorded future", "recorded", SIM),
        ("Encoded future", "decoded JEPA latent", "grid_recorded_future_encoded", GRAY),
        ("JEPA rollout", "original predictor", "grid_deterministic", JEPA),
        ("Flow rollout", "one sample", "flow_sample", FLOW),
    ]
    if full:
        rows = [rows[0],
                ("Counterfactual", "opposite gap", "counterfactual", SIM),
                rows[1], rows[2],
                ("JEPA + head", "one sample", "grid_sample", JEPA),
                ("JEPA + head", "mean of 8", "grid_mean_of_8", JEPA),
                rows[3],
                ("Flow rollout", "mean of 8", "flow_mean_of_8", FLOW)]
    n = len(rows)
    fig = plt.figure(figsize=(5.5, .715 + n * .46))
    # Row labels are drawn in figure coordinates to avoid horizontal clipping.
    x0, width, gap, cgap = 1.21, .425, .03, .07
    top, rowgap = .66, .035
    for j in range(9):
        x = x0 + j * (width + gap) + (cgap if j else 0)
        label = "Context" if j == 0 else str(j)
        fig.text((x + width/2)/5.5, 1-.34/fig.get_figheight(), label,
                 ha="center", va="center", fontsize=7.5,
                 fontweight="bold" if j == 0 else "normal")
    fig.text((x0+width/2)/5.5, 1-.49/fig.get_figheight(), "pre-reveal",
             ha="center", va="center", fontsize=6.7, color=GRAY)
    fig.text((x0+width+cgap+gap)/5.5, 1-.13/fig.get_figheight(),
             "Recorded action sequence; future step", fontsize=8, color=INK)
    for i, (label, detail, key, color) in enumerate(rows):
        y_top = top + i*(width+rowgap)
        y = 1-(y_top+width)/fig.get_figheight()
        fig.text(1.12/5.5, y+.62*width/fig.get_figheight(), label,
                 ha="right", va="center", fontsize=8, color=color, fontweight="bold")
        fig.text(1.12/5.5, y+.27*width/fig.get_figheight(), detail,
                 ha="right", va="center", fontsize=6.7, color=GRAY)
        frames = [z["context"][context, -1], *z[key][context]]
        for j, frame in enumerate(frames):
            x = x0+j*(width+gap)+(cgap if j else 0)
            ax = fig.add_axes([x/5.5, y, width/5.5, width/fig.get_figheight()])
            ax.imshow(frame, interpolation="nearest")
            ax.set_axis_off()
    stem = "rollouts_full" if full else "rollouts"
    return export(fig, out, stem, {
        "title": "Observation retention and open-loop prediction are distinct tests",
        "caption": (
            f"The same observed pre-reveal context (saved context {context}, JEPA model seed 0) "
            "and recorded eight-action sequence are used in every row. The context column repeats "
            "the last observed RGB frame. Encoded future decodes the JEPA encoding of each true "
            "future observation; it is not a prediction. JEPA rollout feeds predicted latents back "
            "to the original predictor. Flow shows one stochastic pixel-model rollout. Frames are "
            "unmodified RGB arrays with nearest-neighbor display. The frozen decoder is an auxiliary "
            "diagnostic and its compute is outside world-model training budgets. This metadata-selected "
            "context is an illustration; aggregate evidence uses 64 survival-selected contexts "
            "per model seed, with no terminal event in the observed and recorded-future window."
            + (" Counterfactual changes only the gap side; head-sample and mean-of-eight rows are "
               "distinct residual-head sampling operators." if full else "")
        ),
        "source": "runs/stoch/gate/latent_check.npz",
        "context_index": context,
        "array_keys": ["context"] + [r[2] for r in rows],
        "rgb_transform": "none", "interpolation": "nearest",
        "selection_rule": ("first stored context with wall_x=45, selected by environment metadata "
                           "before inspecting rollout appearance; first predicted frame remains "
                           "unrevealed") if not full else "first stored context, index 0; no outcome-based selection",
    })


def policy_dots(ax, x, rows, color):
    """One cell = one world model and one operator; policies stay as visible dots."""
    values = np.array([r["value"] for r in rows])
    offset = np.linspace(-.15, .15, len(values))
    ax.scatter(x+offset, values, s=23, color=color, edgecolor="white",
               linewidth=.5, zorder=4)
    ax.hlines(values.mean(), x-.27, x+.27, color=color, linewidth=1.5, zorder=3)


def control_by_seed(g, out):
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.85), sharey=True,
                             gridspec_kw={"left": .09, "right": .985, "top": .76,
                                           "bottom": .24, "wspace": .24})
    heading(fig, "Control varies across policies and world models",
            "Final real-gate outcomes after training each MLP policy in a frozen model")
    records = []
    for seed, ax in enumerate(axes[:3]):
        for x, op in enumerate(("det", "samp", "mean8")):
            rows = [{"key": k, "value": r["final"]["walls_passed"]}
                    for k,r in sorted(g["grid"].items())
                    if r and k.startswith(f"m{seed}_{op}_")]
            assert len(rows) == 3
            policy_dots(ax, x, rows, JEPA)
            records.append({"model": "grid_jepa", "model_seed": seed,
                            "operator": op, "runs": rows})
        ax.set_title(f"JEPA seed {seed}", color=JEPA, pad=8)
        ax.set_xticks([0,1,2], ["Orig.", "Draw", "Mean8"], fontsize=7)
        ax.set_xlim(-.48, 2.48)
    ax = axes[3]
    for x, op in enumerate(("samp", "mean8")):
        rows = [{"key":k, "value":r["final"]["walls_passed"]}
                for k,r in sorted(g["flow"].items()) if r and k.startswith(op+"_")]
        policy_dots(ax, x, rows, FLOW)
        records.append({"model": "flow", "model_seed": 0, "operator": op, "runs": rows})
    ax.set_title("Flow: 1 model", color=FLOW, pad=8)
    ax.set_xticks([0,1], ["Draw", "Mean8"], fontsize=7)
    ax.set_xlim(-.55, 1.55)
    for ax in axes:
        ax.set_ylim(-.12,4.25)
        ax.set_yticks([0,1,2,3,4])
        clean(ax)
        ax.axhline(g["baselines"]["random"]["walls_passed"], color=GRAY,
                   linestyle=(0,(2,2)), linewidth=.7, zorder=1)
    axes[0].set_ylabel("Walls passed / episode")
    fig.text(.09,.125,"Dots: policy seeds. Bars: means within one model and operator.",
             fontsize=7.2, color=GRAY)
    fig.text(.09,.065,"JEPA: 3 policies per cell. Flow: 6 draws, 3 mean8 policies. Dashed: random actions.",
             fontsize=7, color=GRAY)
    return export(fig,out,"control_by_seed",{
        "title":"Policy outcomes by world-model seed and rollout operator",
        "caption":"Final walls passed per episode in the real gate. Each orange panel is a separately "
                  "trained JEPA world-model seed; three policy seeds are shown for each operator. "
                  "Orig. uses the original deterministic predictor; Draw samples its separately trained "
                  "residual head; Mean8 averages eight head draws. Flow uses one world-model checkpoint, "
                  "with six sampled-rollout policies and three mean-of-eight policies. Each dot is a "
                  "policy's final evaluation over 256 episodes. Horizontal bars summarize policies "
                  "within a model/operator cell; no pooled standard error or inference over world-model "
                  "seeds is shown. Dashed line is random actions. All trained-model policies receive "
                  "1,228,800 imagined transitions. These comparisons do not isolate dynamics from "
                  "reward heads, observation interfaces, or optimization.",
        "source":"runs/stoch/summary.json", "json_paths":["gate.grid", "gate.flow", "gate.baselines.random"],
        "metric":"final.walls_passed", "records":records,
    })


def representation_by_seed(g,out):
    fig,axes=plt.subplots(1,2,figsize=(5.5,3.0),sharey=True,
                         gridspec_kw={"left":.10,"right":.98,"top":.75,"bottom":.25,"wspace":.22})
    heading(fig,"A decodable observation can still be hard to control from",
            "Policies trained in the real simulator; frozen JEPA observation interfaces")
    records=[]
    for ax,group,reference,title in zip(axes,
        ["real_latent","real_decoded"],["real_mlp","real_seeds"],
        ["MLP on latent vs. pixels","CNN on decoded latent vs. pixels"]):
        for seed in range(3):
            rows=[{"key":k,"value":r["final"]["survival"]*100}
                  for k,r in sorted(g[group].items()) if r and k.startswith(f"m{seed}_")]
            policy_dots(ax,seed,rows,JEPA)
            records.append({"group":group,"model_seed":seed,"runs":rows})
        rows=[{"key":k,"value":r["final"]["survival"]*100}
              for k,r in sorted(g[reference].items()) if r]
        policy_dots(ax,3.2,rows,SIM)
        records.append({"group":reference,"model_seed":None,"runs":rows})
        ax.axvline(2.6,color=GRID,linewidth=.8)
        ax.axhline(50,color=GRAY,linewidth=.7,linestyle=(0,(3,3)))
        ax.set_xlim(-.5,3.7); ax.set_ylim(-3,105)
        ax.set_yticks([0,25,50,75,100])
        ax.set_xticks([0,1,2,3.2],["Seed 0","Seed 1","Seed 2","Pixels"],fontsize=7.2)
        ax.set_title(title,pad=10,fontsize=8)
        for tick in ax.get_xticklabels()[:3]:tick.set_color(JEPA)
        ax.get_xticklabels()[-1].set_color(SIM)
        clean(ax)
    axes[0].set_ylabel("Final survival (%)")
    fig.text(.1,.115,"Each cell: 3 policy seeds. Horizontal bars: within-cell means.",fontsize=7.2,color=GRAY)
    fig.text(.1,.055,"Dashed: descriptive 50% discovery threshold. Decoder trained outside the model budget.",fontsize=7,color=GRAY)
    return export(fig,out,"representation_by_seed",{
        "title":"Direct and decoded latent interfaces under real dynamics",
        "caption":"Final survival at the 128-step cap, evaluated on 256 episodes per policy after "
                  "1,499,136 real environment transitions. Left: MLP policies consume frozen JEPA latents, "
                  "with a true-pixel MLP reference. Right: CNN policies consume a frozen auxiliary "
                  "decoder's images, with a true-pixel CNN reference. The three encoder seeds remain "
                  "separate, with three policy seeds each. Decoder training is outside the world-model "
                  "budget; decoding tests accessibility through a changed interface, not a compute-matched "
                  "repair. The dashed 50% threshold is descriptive. No confidence interval is inferred "
                  "from the policy runs, and the MLP and CNN panels are not an architecture-matched comparison.",
        "source":"runs/stoch/summary.json", "json_paths":["gate."+x for x in
        ["real_latent","real_mlp","real_decoded","real_seeds"]],
        "metric":"100 * final.survival", "records":records,
    })


def geometry_by_seed(g,out):
    fig,axes=plt.subplots(1,3,figsize=(5.5,2.65),sharey=True,
        gridspec_kw={"left":.105,"right":.985,"top":.68,"bottom":.24,"wspace":.20})
    heading(fig,"Prediction error exceeds the wall-removal latent contrast",
            "Teacher-forced error and wall-removal contrast, 64 survival-selected contexts per seed")
    records=[]
    for seed,ax in enumerate(axes):
        data=g["latent_checks"][f"s{seed}"]["grid"]
        steps=np.arange(1,9)
        for key,color,label,marker in [("one_step_residual_sq",JEPA,"Prediction error","o"),
                                      ("wall_amplitude_sq",SIM,"Wall-removal contrast","s")]:
            ax.plot(steps,data[key],color=color,linewidth=1.5,marker=marker,markersize=3.0,
                    markeredgecolor="white",markeredgewidth=.35,label=label,zorder=3)
        ax.set_title(f"JEPA seed {seed}",pad=8,color=JEPA)
        ax.set_xlim(.8,8.2);ax.set_xticks([1,2,4,6,8]);ax.set_ylim(0,.4)
        ax.set_yticks([0,.1,.2,.3,.4]);ax.set_xlabel("Future step")
        clean(ax)
        records.append({"model_seed":seed,**{k:data[k] for k in ["one_step_residual_sq","wall_amplitude_sq"]}})
    axes[0].set_ylabel("Squared latent distance\n(sum over coordinates)",fontsize=7.7)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.56,.835),
               frameon=False,ncol=2,fontsize=7.5,handlelength=2.0,columnspacing=2.0)
    fig.text(.105,.055,"Lines connect recorded steps. This scale comparison does not establish a causal mechanism.",
             fontsize=7,color=GRAY)
    return export(fig,out,"geometry_by_seed",{
        "title":"One-step prediction residual versus encoded obstacle contrast",
        "caption":"For each of three independently trained JEPA model seeds, means across 64 "
                  "reconstructed pre-reveal, survival-selected contexts compare total teacher-forced one-step prediction "
                  "error with the squared latent distance between the recorded future and a matched "
                  "future with the wall removed. Distances sum over latent coordinates. Each later "
                  "one-step prediction is supplied the true context history, so these curves are not "
                  "open-loop rollout errors. The wall-removal contrast is not a causal attribution of "
                  "the total residual. Only 40 of the 64 contexts remain unrevealed at the first future "
                  "step; selection requires no terminal in the observed and recorded-future window. "
                  "Context-level errors were not retained in the aggregate summary, "
                  "so no uncertainty band is fabricated. All eight discrete steps are shown; lines "
                  "connect measurements without smoothing.",
        "source":"runs/stoch/summary.json", "json_paths":[f"gate.latent_checks.s{s}.grid" for s in range(3)],
        "records":records,
    })


def evidence_staircase(out):
    fig=plt.figure(figsize=(5.5,2.70));ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
    ax.text(.035,.955,"Reduce complexity; sharpen the diagnosis",fontsize=10,fontweight="bold",va="top")
    ax.text(.035,.875,"Three complementary tests, from a rich environment to controlled nuisance factors",
            fontsize=7.3,color=GRAY)
    boxes=[(.035,.415,"01", "Craftax", "Goal reaching / reward control",
            "Does goal matching support\nreward-based control?",FLOW),
           (.365,.29,"02", "Gate", "Retain / predict / control",
            "Is a visible wall retained\nbefore its gap is revealed?",JEPA),
           (.695,.165,"03", "Nuisance cube", "3 backgrounds x 2 dynamics",
            "How do noise and dynamics\naffect position readout?",SIM)]
    for x,y,num,title,subtitle,question,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),.27,.36,boxstyle="round,pad=0.012,rounding_size=.015",
                                    facecolor=PALE,edgecolor=GRID,linewidth=.65))
        ax.text(x+.015,y+.305,num,fontsize=7,fontweight="bold",color=color)
        ax.text(x+.015,y+.225,title,fontsize=10,fontweight="bold",color=color)
        ax.text(x+.015,y+.158,subtitle,fontsize=6.5,color=INK)
        ax.text(x+.015,y+.062,question,fontsize=7,linespacing=1.45,color=GRAY,va="center")
    for x,y in [(.315,.55),(.645,.425)]:
        ax.add_patch(FancyArrowPatch((x,y),(x+.035,y-.05),arrowstyle="-|>",
                                     mutation_scale=10,linewidth=1,color=GRAY))
    ax.text(.035,.053,"Protocol schematic. Ordering denotes experimental scope, not measured difficulty.",
            fontsize=7,color=GRAY)
    return export(fig,out,"evidence_staircase",{
        "title":"Experimental scope: rich environment, diagnostic obstacle, controlled nuisances",
        "caption":"Protocol schematic, not a measured performance plot or difficulty ranking. "
                  "Craftax separates visual goal reaching from reward-based planning; the gate "
                  "separates information retention, prediction and control around a late-revealed "
                  "gap; the cube crosses absent, frame-wise and episode-fixed visual noise with "
                  "still or wind dynamics under survival reward. The current gate uses a white "
                  "wall, black background and no wind. The six nuisance-cube conditions have no wall. "
                  "The diagram does not imply that the full nuisance grid was evaluated in the gate.",
        "source":["studies/stoch/env.py", "studies/redcube/env.py", "runs/stoch/summary.json"],
        "semantic_schematic":True,
    })


def craftax_summary(c,out):
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.85),
        gridspec_kw={"left":.16,"right":.97,"top":.71,"bottom":.31,"wspace":.74})
    heading(fig,"Craftax: goal reaching and reward control give different answers",
            "Six-step open-loop goal matching and 100-step reward control; one model checkpoint")
    groups=[(["Random", "JEPA plan", "Ref. replay"],
             [100*c["craftax"][k]["reached"] for k in ["random","planned","oracle"]],
             [GRAY,JEPA,SIM],"Position matched (%)","6-step open-loop goal",100),
            (["Random","Myopic","Greedy","MPC"],
             [c["reward_mpc"][k]["return"] for k in ["random","myopic","greedy","plan"]],
             [GRAY,JEPA,JEPA,JEPA],"Mean real return","Reward planning",1.65)]
    for ax,(labels,values,colors,xlabel,title,lim) in zip(axes,groups):
        yy=np.arange(len(labels))
        ax.barh(yy,values,color=colors,alpha=.85,height=.57,zorder=3)
        ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlim(0,lim*1.18)
        ax.set_xlabel(xlabel);ax.set_title(title,pad=12,fontsize=8.4)
        for y,v in zip(yy,values):
            ax.text(v+lim*.035,y,f"{v:.1f}" if lim==100 else f"{v:.3f}",va="center",fontsize=7.4)
        clean(ax,False);ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
        ax.set_xticks([0,50,100] if lim==100 else [0,.5,1,1.5])
        ax.grid(axis="x",color=GRID,linewidth=.5,zorder=0)
    fig.text(.16,.115,"128 saved start states per task. Reward control runs for 100 real steps.",fontsize=7.2,color=GRAY)
    fig.text(.16,.055,"Descriptive means from one checkpoint; tasks and reward scales are distinct.",fontsize=7.2,color=GRAY)
    return export(fig,out,"craftax_summary",{
        "title":"Craftax visual goal reaching versus reward planning",
        "caption":"Left: fraction matching the goal position in a single six-step open-loop action "
                  "plan across 128 saved states, with goals generated by random six-action sequences; "
                  "the simulator reference executes the known recorded action sequence. Right: mean "
                  "100-step real-environment return across 128 states with a six-step planning horizon. "
                  "Random, myopic, greedy and MPC are distinct action-selection procedures. Both panels "
                  "use one JEPA model checkpoint and are descriptive, without model-seed confidence "
                  "intervals. They test different objectives and should not be interpreted as a single "
                  "success metric. The goal result is local recorded-goal reaching, not full-game success.",
        "source":"runs/stoch/summary.json", "json_paths":["craftax.craftax","craftax.reward_mpc"],
        "data": {"goal_reached_percent":dict(zip(groups[0][0],groups[0][1])),
                 "real_return":dict(zip(groups[1][0],groups[1][1]))},
    })


def nuisance_cube(s,out):
    worlds=["off_still","off_wind","gray_frame_still","gray_frame_wind","gray_episode_still","gray_episode_wind"]
    names=["No noise / still","No noise / wind","Frame noise / still","Frame noise / wind",
           "Episode noise / still","Episode noise / wind"]
    cols=["Global\nJEPA","Grid\nJEPA","Flow\npixels","Flow\nlatent","Real\nCNN","Center\nrule"]
    keys=["jepa_lewm6","jepa_tile_factored","dream_pixels","dream_latent","real"]
    data=np.array([[100*s[w]["ppo"][k]["final"]["survival"] for k in keys]+
                   [100*s[w]["baselines"]["centre"]["survival"]] for w in worlds])
    fig=plt.figure(figsize=(5.5,3.35));ax=fig.add_axes([.29,.17,.68,.60])
    cmap=LinearSegmentedColormap.from_list("survival",["#f7e2d9","#f6f7f4","#c0e2de"])
    ax.imshow(data,cmap=cmap,vmin=0,vmax=100,aspect="auto",interpolation="nearest")
    for r in range(len(worlds)):
        for c in range(len(cols)):
            ax.text(c,r,f"{data[r,c]:.1f}",ha="center",va="center",fontsize=8,
                    fontweight="bold" if data[r,c]>=90 else "normal",color=INK)
    ax.set_xticks(range(len(cols)),cols,fontsize=7.5);ax.xaxis.tick_top()
    ax.set_yticks(range(len(names)),names,fontsize=7.5)
    ax.tick_params(axis="both",length=0,pad=6)
    for line in [1.5,3.5]:ax.axhline(line,color="white",linewidth=3)
    ax.axvline(4.5,color="white",linewidth=3)
    for spine in ax.spines.values():spine.set_visible(False)
    heading(fig,"Single-run survival audit under visual noise and wind",
            "Final survival (%) at the 128-step cap; one training run per condition")
    fig.text(.035,.092,"State-blind alternation also survives all still worlds in the edge-start check.",fontsize=7.2,color=GRAY)
    fig.text(.035,.043,"Survival alone does not certify feedback. Each cell: 256 episodes, one training run.",fontsize=7.2,color=GRAY)
    return export(fig,out,"nuisance_cube",{
        "title":"Single-run survival across the six nuisance-cube conditions",
        "caption":"Descriptive final survival percentages over 256 evaluation episodes per cell, "
                  "with one learned-model/policy run per condition. Global and Grid JEPA policies "
                  "train in their latent models; Flow pixels and Flow latent denote different "
                  "policy observation interfaces inside the pixel-generative model. Real CNN "
                  "trains in the simulator. Center rule is a nonlearned position-feedback baseline, "
                  "not a learned world model. Imagined-policy training uses 1,228,800 transitions; "
                  "the real CNN uses 1,499,136 real transitions. The no-wall cube crosses no noise, "
                  "frame-wise gray noise and episode-fixed gray noise with still or wind dynamics. "
                  "A state-blind alternating policy achieves 100% survival in the archived edge-start "
                  "checks of all three still conditions (runs/stoch/*still/baselines_edge.json). "
                  "Thus survival alone cannot certify use of visual feedback. These single-run cells "
                  "support a sensitivity audit, not model-family rankings "
                  "with training-seed uncertainty. The color scale is fixed from 0% to 100%.",
        "source":"runs/stoch/summary.json", "worlds":worlds,"columns":cols,"values_percent":data.tolist(),
        "json_paths":[w+".ppo" for w in worlds]+[w+".baselines.centre" for w in worlds],
    })


def readout_by_seed(root,out):
    worlds=["gray_frame_still","gray_frame_wind","gray_episode_still","gray_episode_wind"]
    arms=["tile_factored","tile_factored_s1","tile_factored_s2"]
    paths=[root/"runs"/"redcube"/w/"allarms"/"bench.json" for w in worlds]
    benches=[json.loads(p.read_text()) for p in paths]
    data=np.array([[b["probe_error"][a]["heldout"] for a in arms] for b in benches])
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.9),sharey=True,
        gridspec_kw={"left":.105,"right":.98,"top":.71,"bottom":.25,"wspace":.22})
    heading(fig,"Position readout error rises with frame noise plus wind",
            "Frozen 512-dimensional grid JEPA; identical schedules, three model seeds per condition")
    markers=["o","s","^"]
    for j,ax in enumerate(axes):
        for x in range(2):
            vals=data[j*2+x]
            for seed,(v,marker) in enumerate(zip(vals,markers)):
                ax.scatter(x+(seed-1)*.11,v,s=30,marker=marker,color=JEPA,
                           edgecolor="white",linewidth=.5,zorder=4)
            ax.hlines(vals.mean(),x-.26,x+.26,linewidth=1.5,color=JEPA,zorder=3)
            ax.text(x,vals.mean()+1.45,f"{vals.mean():.2f}",ha="center",va="bottom",fontsize=8,color=INK)
        ax.set_title(["Per-frame gray noise","Episode-fixed gray noise"][j],fontsize=8.3,pad=9)
        ax.set_xlim(-.5,1.5);ax.set_ylim(-.6,25.5)
        ax.set_xticks([0,1],["Still dynamics","Wind dynamics"],fontsize=7.8)
        ax.set_yticks([0,5,10,15,20,25]);clean(ax)
    axes[0].set_ylabel("Position readout error (L1 pixels)",fontsize=7.8)
    handles=[Line2D([0],[0],marker=m,color="none",markerfacecolor=JEPA,markeredgecolor="white",
                    markersize=5,label=f"Model seed {i}") for i,m in enumerate(markers)]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.56,.87),frameon=False,
               ncol=3,fontsize=7.1,handletextpad=.35,columnspacing=1.4)
    fig.text(.105,.12,"Markers: individual models. Bars and labels: three-model means. Lower is better.",fontsize=7.1,color=GRAY)
    fig.text(.105,.058,"A failed tested readout does not prove that position information is absent.",fontsize=7.1,color=GRAY)
    return export(fig,out,"readout_by_seed",{
        "title":"Frozen-latent position readout under gray visual noise and wind",
        "caption":"Held-out L1 position error of the trained probe on true-observation latents "
                  "for three independently trained 512-dimensional grid JEPA seeds per condition. "
                  "Every model uses the same 500,000-transition data-size and 8,500-update training "
                  "schedule. Gray noise is redrawn per frame or fixed within each episode, crossed "
                  "with still or hidden-wind dynamics. Each point is a model-seed result; horizontal "
                  "bars and labels show the three-model means. Results are taken only from the "
                  "allarms/bench.json benchmark family. This is a failure of the tested readout, "
                  "not proof of information-theoretic loss. The failing frame-noise plus wind "
                  "condition does not have near-zero prediction loss, so it must not be presented "
                  "as an empirical zero-loss optimum.",
        "source":[str(p) for p in paths],
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "json_paths":["probe_error."+a+".heldout" for a in arms],
        "worlds":worlds,"model_arms":arms,"values_l1_pixels":data.tolist(),
    })


def craftax_goal_v2(root, out):
    """Fresh V2 artifact; never rewrites any frozen V1 figure."""
    from PIL import Image
    table_path = root / "research/craftax_table_v2.json"
    table = json.loads(table_path.read_text())
    raw_path = root / table["source"]
    raw = json.loads(raw_path.read_text())
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == table["source_sha256"]
    frame_dir = root / "artifacts/revision/craftax_frames_v2"
    meta_path = frame_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    assert meta["eval_report_sha256"] == table["source_sha256"]
    # This scene was chosen by displacement metadata, without consulting outcomes.
    scene = meta["scenes"][0]
    image_paths = [frame_dir / scene["frames"][k]["path"] for k in ("start", "goal")]
    for k, p in zip(("start", "goal"), image_paths):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == scene["frames"][k]["sha256"]
    policies = ["noop", "random", "jepa_open_loop", "jepa_receding"]
    labels = ["No-op", "Random", "Open loop", "Receding"]
    strata = ["all", "displaced_goal"]
    # Confirm the table endpoints against raw paired episode records.
    for policy in policies:
        all_rows = [row for run in raw["goal_runs"] for row in run["rows"][policy]]
        for stratum in strata:
            selected = [r for r in all_rows if stratum == "all" or r["goal_displacement_l1"] > 0]
            successes = sum(r["reached_position_alive"] for r in selected)
            summary = table["policy_results"][policy][stratum]
            assert summary["n"] == len(selected)
            assert summary["successes"] == successes
            assert np.isclose(summary["mean"], successes / len(selected))
    fig = plt.figure(figsize=(5.5, 2.4))
    fig.text(.035, .958, "JEPA plans local Craftax goals beyond no-op", fontsize=10,
             fontweight="bold", va="top")
    fig.text(.035, .868, "Six real steps; one frozen checkpoint; independently sampled execution noise",
             fontsize=7.5, color=GRAY, va="top")
    for k, p, bottom in zip(("Start", "Goal"), image_paths, (.515, .245)):
        ax = fig.add_axes([.044, bottom, .094, .216])
        ax.imshow(np.asarray(Image.open(p)), interpolation="nearest")
        ax.set_axis_off()
        ax.text(.5, 1.03, k, transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
    fig.text(.091, .187, "Example pair", fontsize=6.4, color=GRAY, ha="center")
    axes = [fig.add_axes([.315, .28, .272, .435]),
            fig.add_axes([.697, .28, .272, .435])]
    colors = [GRAY, GRAY, JEPA, JEPA]
    for ax, stratum, title in zip(axes, strata,
            ["All starts (n = 384)", "Displaced goals (n = 247)"]):
        for y, (policy, color) in enumerate(zip(policies, colors)):
            datum = table["policy_results"][policy][stratum]
            mean = 100 * datum["mean"]
            lo, hi = np.asarray(datum["wilson_95"]) * 100
            ax.barh(y, mean, color=color, height=.54, alpha=.82, zorder=3)
            ax.errorbar(mean, y, xerr=[[mean-lo],[hi-mean]], fmt="none",
                        ecolor=INK, elinewidth=.8, capsize=2, capthick=.8, zorder=4)
            ax.text(hi+3.1, y, f"{mean:.1f}", fontsize=7.2, va="center", ha="left")
        ax.set_ylim(3.55, -.55); ax.set_xlim(-1.8, 121)
        ax.set_xticks([0, 50, 100]); ax.set_xlabel("Goal success (%)", fontsize=7.3, labelpad=3)
        ax.set_yticks(range(4), labels, fontsize=7.3)
        for tick,color in zip(ax.get_yticklabels(),colors):tick.set_color(color)
        ax.set_title(title, fontsize=8, pad=9)
        clean(ax, grid=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.grid(axis="x", color=GRID, linewidth=.6, zorder=0)
    fig.text(.035, .097, "Orange: JEPA goal planning. Whiskers: 95% Wilson intervals over evaluation episodes.",
             fontsize=7, color=GRAY)
    fig.text(.035, .040, "The displaced subset excludes goals at the initial position. No training-seed replication.",
             fontsize=7, color=GRAY)
    sources = [table_path, raw_path, meta_path, *image_paths]
    record = export(fig, out, "craftax_goal_v2", {
        "title": "Six-step local Craftax goal planning beyond no-op",
        "caption": "Six-step local goal planning with one frozen 512-dimensional JEPA checkpoint. "
                   "Goals are generated by six random actions; execution uses fresh environment "
                   "randomness independent of target generation, paired across policies. Goal "
                   "success requires the exact target position after six actions and survival "
                   "throughout warmup and execution. Left: all 384 starts from three 128-episode "
                   "evaluation batches. Right: "
                   "the 247 starts whose goal position differs from the initial position. Open "
                   "loop plans once; receding-horizon control replans after each observation. "
                   "Whiskers are 95% Wilson episode-level intervals for one trained model, not "
                   "uncertainty over model training seeds. The exact native-color start and target "
                   "images show evaluation seed 101, episode 5: the first stored scene with L1 "
                   "goal displacement at least two, chosen without inspecting model outcomes. "
                   "These results establish local goal-position control for this checkpoint, "
                   "not general Craftax achievement or long-horizon reward competence.",
        "source_sha256": {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "model_sha256": raw["model_sha256"],
        "metric": "reached_position_alive",
        "policies": policies, "strata": strata,
        "results": {p:{s:table["policy_results"][p][s] for s in strata} for p in policies},
        "scene": {"eval_seed":meta["seed"],"episode":scene["episode"],
                  "selection":meta["selection"],"displacement_l1":scene["displacement_l1"],
                  "rgb_transform":"none", "interpolation":"nearest"},
        "uncertainty_scope": table["uncertainty_scope"],
    })
    (out/"craftax_goal_v2.caption.json").write_text(json.dumps(record, indent=2)+"\n")
    print("Wrote fresh V2 Craftax figure and source-hashed caption ->", out)


def target_gradient_method(root, out):
    """Protocol schematic: the intervention changes backward routing, not the loss value."""
    fig = plt.figure(figsize=(5.5, 2.15))
    ax = fig.add_axes([0,0,1,1]); ax.set_axis_off()
    ax.text(.025, .957, "One intervention: stop the direct target gradient", fontsize=10,
            fontweight="bold", va="top")
    ax.text(.025, .854, "Same forward loss: prediction MSE + 0.09 SIGReg; fixed 512-dimensional latent",
            fontsize=7.8, color=GRAY)
    def box(x,y,w,h,text,color=INK,face=PALE,size=8.8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.006,rounding_size=.015",
                                   facecolor=face,edgecolor=GRID,linewidth=.7))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",color=color,fontsize=size,
                linespacing=1.25)
    def arrow(start,end,color=INK,style="-|>",ls="solid",lw=.9,connectionstyle="arc3"):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=9,
                                     color=color,linewidth=lw,linestyle=ls,
                                     connectionstyle=connectionstyle))
    box(.025,.615,.125,.15,"History\nframes")
    box(.025,.385,.125,.15,"Next\nframe")
    box(.215,.36,.155,.385,"Shared\nencoder\n"+r"$f_\theta$",color=JEPA,size=9)
    arrow((.152,.69),(.212,.69))
    arrow((.152,.46),(.212,.46))
    arrow((.372,.69),(.425,.69))
    arrow((.372,.46),(.425,.46))
    ax.text(.451,.69,r"$z_{\leq t}$",fontsize=10,ha="center",va="center")
    ax.text(.451,.46,r"$z_{t+1}$",fontsize=10,ha="center",va="center")
    box(.525,.615,.174,.15,"Predictor\n+ actions",size=8.8)
    arrow((.477,.69),(.522,.69))
    arrow((.701,.69),(.751,.69))
    ax.text(.782,.69,r"$\hat z_{t+1}$",fontsize=10,ha="center",va="center")
    box(.866,.495,.109,.155,"MSE",size=9)
    arrow((.810,.69),(.87,.646))
    arrow((.48,.46),(.86,.542))
    # The return arrow is exclusively the direct target-side MSE gradient.
    arrow((.92,.487),(.92,.274),color=JEPA,ls=(0,(3,2)),lw=1.25,style="-")
    arrow((.92,.274),(.45,.274),color=JEPA,ls=(0,(3,2)),lw=1.25,style="-")
    arrow((.45,.274),(.45,.416),color=JEPA,ls=(0,(3,2)),lw=1.25)
    ax.text(.704,.368,"Direct target-side gradient",fontsize=8.3,color=JEPA,ha="center")
    ax.text(.704,.307,"Joint: on   |   Stop-target: off",fontsize=8.3,color=INK,ha="center")
    box(.185,.12,.223,.13,"SIGReg (all latents)",color=SIM,size=8.3)
    arrow((.292,.351),(.292,.257),color=SIM,style="<->",lw=1.05)
    ax.text(.173,.303,"active in\nboth modes",fontsize=7.5,color=SIM,ha="right",va="center")
    ax.text(.485,.168,"Forward target values are unchanged.",fontsize=7.8,color=GRAY)
    ax.text(.025,.037,"Input-path, overlapping-history and BatchNorm gradient couplings remain.",
            fontsize=8,color=GRAY)
    sources=[root/"research/target_gradient_protocol.md",root/"studies/revision/target_gradient.py"]
    record=export(fig,out,"target_gradient_method",{
        "title":"Stopping only the direct target-side prediction-loss gradient",
        "caption":"Prospective paired intervention: joint and stop-target training have the "
                  "same forward squared latent prediction loss, SIGReg coefficient, architecture, "
                  "512-dimensional latent size, initialization and data/update schedule. Stop-target "
                  "blocks the direct target-embedding branch of the MSE derivative; target values "
                  "in the forward pass are unchanged. The shared encoder continues to receive "
                  "predictor-input gradients and gradients of SIGReg over all embeddings. Frames "
                  "appearing in overlapping predictor histories retain their input-path gradients, "
                  "and batch normalization may couple frame contributions. Solid dark arrows show "
                  "the prediction forward pass; the dashed orange return arrow marks the single "
                  "gradient path changed by the intervention. The teal connection indicates that "
                  "regularization remains differentiable in both modes. This schematic specifies "
                  "the experimental comparison and makes no claim about its empirical benefit.",
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "semantic_schematic":True,"reports_outcomes":False,
        "changed_path":"direct target-side MSE gradient",
        "unchanged_paths":["MSE predictor/input branch","SIGReg all embeddings",
                           "overlapping-history input paths","BatchNorm couplings"],
    })
    (out/"target_gradient_method.caption.json").write_text(json.dumps(record,indent=2)+"\n")
    print("Wrote prospective target-gradient method schematic ->",out)


def matched_control_v2(root, out):
    path = root / "output/revision/matched-analysis/publication.json"
    audit_path = root / "research/matched_final_audit.json"
    data = json.loads(path.read_text())
    audit = json.loads(audit_path.read_text())
    assert data["status"] == "complete"
    assert len(data["validated_runs"]) == 45
    assert audit["publication_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    arms = ["learned", "oracle_heads", "oracle_true"]
    colors = [JEPA, GRAY, SIM]
    labels = ["Learned", "Exact +\nheads*", "Exact +\ntrue"]
    model_seeds = data["planned_model_seeds"]
    policy_seeds = data["planned_policy_seeds"]
    runs = {(r["model_seed"], r["policy_seed"], r["arm"]): r for r in data["validated_runs"]}
    assert model_seeds == [0,1,2] and policy_seeds == [0,1,2,3,4]
    assert len(runs) == 45
    max_survival = max(100*r["final"]["survival"] for r in runs.values())
    assert max_survival < 4
    fig = plt.figure(figsize=(5.5,2.6))
    fig.text(.035,.965,"Gate survival stays low with exact transitions and signals",fontsize=10,
             fontweight="bold",va="top")
    fig.text(.035,.883,"3 frozen encoders x 5 paired policy seeds x 3 arms; final checkpoints only",
             fontsize=7.5,color=GRAY)
    markers=["o","s","^","D","v"]
    records=[]
    for col,seed in enumerate(model_seeds):
        left=.105+col*.295
        for row,(metric,factor,bottom,ylim,ticks) in enumerate([
            ("walls_passed",1,.55,(-.035,1.0),[0,.5,1.0]),
            ("survival",100,.205,(-.15,4.0),[0,2,4]),
        ]):
            ax=fig.add_axes([left,bottom,.265,.23])
            values=np.array([[factor*runs[(seed,p,a)]["final"][metric] for a in arms]
                             for p in policy_seeds])
            for i,(p,marker) in enumerate(zip(policy_seeds,markers)):
                x=np.arange(3)+(i-2)*.056
                ax.plot(x,values[i],color="#b5c0c7",linewidth=.6,alpha=.8,zorder=2)
                for j,color in enumerate(colors):
                    ax.scatter(x[j],values[i,j],s=15,marker=marker,color=color,
                               edgecolor="white",linewidth=.35,zorder=4)
            for j,color in enumerate(colors):
                ax.hlines(values[:,j].mean(),j-.23,j+.23,color=color,linewidth=1.35,zorder=3)
            ax.set_xlim(-.40,2.4);ax.set_ylim(*ylim);ax.set_yticks(ticks)
            if row==0:
                ax.set_title(f"Encoder seed {seed}",fontsize=8.3,pad=6)
                ax.set_xticks([])
            else:
                ax.set_xticks([0,1,2],labels,fontsize=7,linespacing=1.0)
                for tick,color in zip(ax.get_xticklabels(),colors):tick.set_color(color)
            if col==0:
                ax.set_ylabel("Walls / episode" if row==0 else "Survival (%)",fontsize=7.3,labelpad=3)
            else:
                ax.set_yticklabels([])
            clean(ax)
            ax.tick_params(axis="x",length=0,pad=2)
            if row==1:
                ax.text(.98,.87,"0-4% detail",transform=ax.transAxes,fontsize=6.5,color=GRAY,
                        ha="right",va="top")
        records.append({"model_seed":seed,"model_sha256":data["model_results"][str(seed)]["model_sha256"],
                        "policy_seeds":policy_seeds,
                        "arms":{a:{m:data["model_results"][str(seed)]["arms"][a][m]
                                   for m in ["walls_passed","survival"]} for a in arms}})
    fig.text(.105,.058,"Lines pair policy seeds; bars show within-encoder means. All 45 runs remain below 4% survival.",
             fontsize=6.8,color=GRAY)
    fig.text(.105,.012,"Exact transitions + true signals: 0/15 policies exceed 50% survival.",fontsize=7,color=INK)
    record=export(fig,out,"matched_control_v2",{
        "title":"Matched latent-policy training with learned and exact gate simulation",
        "caption":"Final real-gate outcomes for all 45 budget-complete runs: three independently "
                  "trained frozen encoders, five paired policy seeds per encoder, and three training "
                  "arms. Learned uses learned latent dynamics and learned reward/continuation heads; "
                  "Exact + heads uses true simulator transitions with frozen-encoder observations "
                  "and learned reward/continuation heads; Exact + true uses true transitions and "
                  "true reward/termination signals with the same frozen-latent MLP interface. "
                  "Every policy receives 1,228,800 nominal training-transition slots and is evaluated on 256 "
                  "paired episodes. Lines connect the same policy seed within each fixed encoder; "
                  "short horizontal bars are within-encoder means. Top: walls passed per episode, "
                  "the primary endpoint. Bottom: survival at the 128-step cap on an explicitly "
                  "expanded 0-4% scale. No arm approaches the descriptive 50% discovery threshold; "
                  "the exact-transition/true-signal arm has zero discoveries among 15 policies. "
                  "Raw points are shown without confidence bars. In the accompanying analysis, "
                  "contrast intervals resample the five paired policy seeds within each checkpoint, "
                  "conditional on the fixed evaluation set; they are not model-family intervals. "
                  "*The Exact + heads arm freezes true terminal states but retains learned "
                  "continuation/reward on that absorbing extension, so its comparison is not a "
                  "clean dynamics-only intervention. The result concerns this fixed latent MLP "
                  "training pipeline and does not rank model families.",
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,audit_path]},
        "artifact_selection":data["selection"], "max_survival_percent":max_survival,
        "records":records, "arms":dict(zip(arms,[s.replace("\n"," ") for s in labels])),
        "axis_limits":{"walls_passed":[0,1],"survival_percent":[0,4]},
        "pairing":"policy seed within fixed world-model checkpoint",
        "uncertainty_scope":"paired-policy within-checkpoint contrasts in source; no confidence interval drawn",
    })
    (out/"matched_control_v2.caption.json").write_text(json.dumps(record,indent=2)+"\n")
    print("Wrote complete matched-control endpoint figure ->",out)


def forecast_readout_v2(root, out):
    base = root / "runs/revision/forecast/analysis"
    metrics_path, contrasts_path = base/"metrics.json", base/"paired_contrasts.json"
    manifest_path = base/"manifest.json"
    constant_path = root/"research/forecast_constant_baseline_audit.json"
    constant_audit_path = root/"research/forecast_constant_independent_audit.json"
    constant_audit = json.loads(constant_audit_path.read_text())
    assert constant_audit["status"] == "PASS"
    assert constant_audit["source_audit_sha256"] == hashlib.sha256(constant_path.read_bytes()).hexdigest()
    constant_rows = constant_audit["h8_unrevealed_rows"]
    constants = {r["model_seed"]:r for r in constant_rows}
    assert set(constants) == {0,1,2}
    assert all(r["train_n"] == 1509 and r["eval_n"] == 716 for r in constant_rows)
    selected = lambda r: (r["task"] == "wall_position" and r["horizon"] == 8
                          and r["cohort"] == "current_wall_unrevealed")
    rows = [r for r in json.loads(metrics_path.read_text()) if selected(r)]
    contrasts = [r for r in json.loads(contrasts_path.read_text()) if selected(r)]
    assert len(rows) == 27 and len(contrasts) == 18
    assert {r["eval_n"] for r in rows} == {716}
    by_key = {(r["model_seed"],r["normalization"],r["trained_on"],r["evaluated_on"]):r for r in rows}
    pairs=[("real","real"),("real","forecast"),("forecast","forecast"),("copy","copy")]
    colors=[SIM,JEPA,JEPA,GRAY]
    fig,axes=plt.subplots(1,3,figsize=(5.5,2.4),sharey=True,
        gridspec_kw={"left":.105,"right":.98,"top":.595,"bottom":.255,"wspace":.23})
    fig.text(.035,.965,"Refitting changes forecast readout error",fontsize=10,
             fontweight="bold",va="top")
    fig.text(.035,.88,"Eight-step forecasts; 716 continuing contexts; visible wall with hidden gap at the start",
             fontsize=7.4,color=GRAY)
    for seed,ax in enumerate(axes):
        baseline=by_key[(seed,"none","train_mean","constant")]["metrics"]["mae_px"]["estimate"]
        assert np.isclose(baseline,14.237523580193418)
        ax.axhline(baseline,color=GRAY,linewidth=.8,linestyle=(0,(3,2)),zorder=1)
        median_baseline=constants[seed]["constant_median_mae"]
        assert np.isclose(median_baseline,4.64245810055866)
        ax.axhline(median_baseline,color=INK,linewidth=1.0,linestyle=(0,(5,2,1,2)),zorder=1)
        for j,(pair,color) in enumerate(zip(pairs,colors)):
            for norm,offset,filled in [("common",-.10,False),("per_source",.10,True)]:
                row=by_key[(seed,norm,*pair)]
                d=row["metrics"]["mae_px"]
                v=d["estimate"];lo,hi=d["ci95"]
                ax.errorbar(j+offset,v,yerr=[[v-lo],[hi-v]],fmt="o",markersize=3.9,
                            markerfacecolor=color if filled else "white",markeredgecolor=color,
                            markeredgewidth=.9,ecolor=color,elinewidth=.75,capsize=1.7,
                            capthick=.65,zorder=3)
        ax.set_title(f"Encoder seed {seed}",fontsize=8.4,pad=7)
        ax.set_xlim(-.47,3.47);ax.set_ylim(0,18)
        ax.set_yticks([0,5,10,15]);ax.set_xticks(range(4),["R→R","R→F","F→F","Copy"],fontsize=7.2)
        clean(ax);ax.tick_params(axis="x",length=0,pad=4)
    axes[0].set_ylabel("Wall-position MAE (px)",fontsize=7.7)
    handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor="white",markeredgecolor=INK,
                    markersize=4,label="Common normalization"),
             Line2D([0],[0],marker="o",color="none",markerfacecolor=INK,markeredgecolor=INK,
                    markersize=4,label="Per-source normalization"),
             Line2D([0],[0],color=GRAY,linestyle=(0,(3,2)),linewidth=.8,label="Global train mean: 14.24 px"),
             Line2D([0],[0],color=INK,linestyle=(0,(5,2,1,2)),linewidth=1.0,
                    label="Post-hoc train-cohort median: 4.64 px")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.535,.828),frameon=False,
               ncol=2,fontsize=6.8,handlelength=2.0,columnspacing=1.8,handletextpad=.4,
               labelspacing=.45)
    fig.text(.105,.125,"R/F: real/forecast latents. Readout fit → test. Copy uses the current latent.",
             fontsize=7,color=GRAY)
    fig.text(.105,.058,"95% evaluation-stream intervals. Post-hoc constants use training labels; no model/probe refit.",
             fontsize=7,color=GRAY)
    audit_path=base/"correlation_audit.json"
    sources=[metrics_path,contrasts_path,manifest_path,audit_path,constant_path,constant_audit_path,
             root/"research/forecast_readout_protocol.md"]
    record=export(fig,out,"forecast_readout_v2",{
        "title":"Refitting changes forecast readout error across fitted encoders",
        "caption":"Wall-position MAE at horizon eight in the prespecified current-wall-unrevealed "
                  "evaluation cohort: 716 contexts (8.7% of 8,192 sampled starts) have a wall already "
                  "present but its gap unrevealed, and continue without reset to a wall-present future. "
                  "This is a survival-conditioned descriptive subset, not the full-cohort primary "
                  "analysis. All three encoder seeds and "
                  "both predefined train-only normalizations are shown. R→R fits/evaluates a linear "
                  "readout on real future encodings; R→F transfers that real-feature readout to "
                  "forecast features; F→F fits/evaluates the same readout class on forecast features. "
                  "Copy fits/evaluates on the current real latent without future actions. Hollow "
                  "markers use the common current-training-latent scaler; filled markers use "
                  "source/horizon-specific training scalers, retained unchanged during transfer. "
                  "The dashed reference is the original training-mean wall-position predictor "
                  "evaluated on this cohort, not a mean refit to the subset. The dash-dot line "
                  "adds a separately labeled post-hoc interpretation check: the constant median "
                  "wall position computed from the corresponding 1,509 TRAINING-cohort labels, "
                  "with no model or feature-probe refit. Its evaluation MAE is 4.642 px; the "
                  "training-cohort mean also achieves 4.769 px. Both outperform the seed-2 adapted "
                  "forecast readout (6.164 px) and real-feature positive control (6.531 px). "
                  "This independently verified check supplements rather than replaces the "
                  "originally declared global-mean baseline. Whiskers use 2,000 "
                  "evaluation-stream bootstrap draws, conditional on each fitted model and readout; "
                  "they do not quantify model-training uncertainty. In seed 2, adaptation reduces "
                  "MAE by 3.85 px [3.48,4.22] under per-source normalization and 3.89 [3.52,4.25] "
                  "under common normalization. Its F→F-minus-Copy interval includes zero in both "
                  "analyses (-0.63 [-1.41,0.13] and -0.64 [-1.42,0.12], respectively). This reduction "
                  "in MAE does not by itself establish wall tracking or information recovery: "
                  "calibration to the selected subset can reduce error relative to the global "
                  "training-mean baseline without tracking variation within the subset. It also "
                  "does not establish superior forecasting relative to Copy. A separately labeled "
                  "post-hoc interpretation audit finds seed-2 F→F position correlation -0.175 in "
                  "this slice; in the full wall-present cohort its correlation is +0.246 but "
                  "RMSE is 18.65 px versus 13.49 px for the fixed training-mean baseline. Thus "
                  "the displayed MAE reduction cannot certify reliable information recovery. "
                  "Seeds 0 and 1 have "
                  "much weaker real-feature positive controls. All other horizons and cohorts remain available "
                  "in the source analysis. Supervised labels are used only for this frozen-model "
                  "diagnostic and never enter model training, control inputs or reward.",
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "selection":{"task":"wall_position","horizon":8,"cohort":"current_wall_unrevealed",
                     "model_seeds":[0,1,2],"normalizations":["common","per_source"]},
        "records":rows,"paired_contrasts":contrasts,
        "post_hoc_tracking_audit_source":str(audit_path),
        "post_hoc_training_cohort_constants":constant_rows,
        "uncertainty_scope":"evaluation streams conditional on each fitted model and readout",
    })
    (out/"forecast_readout_v2.caption.json").write_text(json.dumps(record,indent=2)+"\n")
    print("Wrote complete forecast readout transfer figure ->",out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,default=Path("runs/stoch/summary.json"))
    p.add_argument("--npz",type=Path,default=Path("runs/stoch/gate/latent_check.npz"))
    p.add_argument("--out",type=Path,default=Path("submission/shared/figures"))
    p.add_argument("--context",type=int,default=None,
                   help="Default: first stored wall_x=45 context, selected using metadata only")
    p.add_argument("--craftax-v2",action="store_true",help="Generate only the fresh V2 Craftax goal figure")
    p.add_argument("--target-gradient-method",action="store_true",help="Generate only the V2 target-gradient method schematic")
    p.add_argument("--matched-control-v2",action="store_true",help="Generate only the complete primary V2 matched-control figure")
    p.add_argument("--forecast-v2",action="store_true",help="Generate only the complete V2 forecast readout transfer figure")
    args=p.parse_args()
    if args.craftax_v2 or args.target_gradient_method or args.matched_control_v2 or args.forecast_v2:
        if args.out == Path("submission/shared/figures"):
            args.out = Path("submission/shared-v2/figures")
        if args.out.resolve() == Path("submission/shared/figures").resolve():
            raise ValueError("V2 generation cannot overwrite frozen V1 figures")
        args.out.mkdir(parents=True,exist_ok=True)
        if args.craftax_v2:
            craftax_goal_v2(Path("."),args.out)
        if args.target_gradient_method:
            target_gradient_method(Path("."),args.out)
        if args.matched_control_v2:
            matched_control_v2(Path("."),args.out)
        if args.forecast_v2:
            forecast_readout_v2(Path("."),args.out)
        return
    args.out.mkdir(parents=True,exist_ok=True)
    s=json.loads(args.summary.read_text());z=np.load(args.npz,allow_pickle=False)
    context = int(np.flatnonzero(z["wall_x"]==45)[0]) if args.context is None else args.context
    records={"rollouts":rollouts(z,context,args.out),
             "rollouts_full":rollouts(z,0,args.out,full=True),
             "control_by_seed":control_by_seed(s["gate"],args.out),
             "representation_by_seed":representation_by_seed(s["gate"],args.out),
             "geometry_by_seed":geometry_by_seed(s["gate"],args.out),
             "evidence_staircase":evidence_staircase(args.out),
             "craftax_summary":craftax_summary(s["craftax"],args.out),
             "nuisance_cube":nuisance_cube(s,args.out),
             "readout_by_seed":readout_by_seed(Path("."),args.out)}
    manifest={"generator":"studies/revision/figures.py",
              "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.summary,args.npz]},
              "palette":{"ink":INK,"jepa":JEPA,"simulator":SIM,"flow":FLOW},
              "figures":records}
    (args.out/"captions.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Wrote {len(records)} figures in SVG, PDF and PNG to {args.out}")


if __name__=="__main__":main()
