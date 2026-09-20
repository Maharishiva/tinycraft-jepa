# Post-outcome distance-response diagnostic, version 1

The failed 6–12-action hard-goal primary is complete and remains unchanged.
This follow-up was proposed after seeing those outcomes and is labeled as such.
Its purpose is to characterize the operating range of the exact same controller,
not to establish deep planning or replace the primary with easier successes.

The parent approved all six exact goal-distance shells **1, 2, 3, 6, 9, 12 native
actions**, all **128 original fresh valid prefixes**, all **three existing model
seeds**, and the same H3 planner, H1 comparison, random/fixed-cycle references,
and privileged physical-distance oracle. No outcome-dependent selection,
planner change, representation scaling, or learned reward modification occurs.

The evaluator is imported unchanged from `redcube_goal_eval.py`, frozen SHA256
`364088b184d2368522d79276724ceaa3ff1eae1610012855344fee0162129683`.
The wrapper refuses to run if this identity changes. It uses one 768-row call per
model, arranged by distance then case, so the exact `[16,8]` encoder and predictor
batch shapes and compiled planner are reused. All 24 actions and every originally
declared controller setting remain fixed. Every case is kept in its denominator.

Prefix and execution seeds remain **20260920**. The newly fixed goal-sampling
seed is **20260921**. Goals are uniform among safe reachable positions at exactly
the shell's Manhattan distance, without model outputs. The goal RGB background
is identical across the six goals for a given prefix; only cube position changes.
Execution noise and random-policy actions are paired across shells/models by
reusing original candidate IDs. Terminal absorption may cause executed action
counts to differ. Goal offsets are stored to expose boundary-truncated direction
distributions.

Exact goal hit before native death remains primary. Report each **model ×
distance count/128** with its own Wilson interval, plus distance and death
secondary outcomes. Do not pool the 768 correlated rows into one inferential
interval. The inherited evaluator's pooled progress summaries are explicitly
internal bookkeeping; final `summary.json` contains only stratified result cells.
The ≤3-pixel secondary is trivially met initially at distance one and must not be
presented as successful control there.

Visual case **0 at every shell** is predeclared: global rows
`[0,128,256,384,512,640]`. Save goal images, prefixes, full recorded positions and
actions, terminal masks, and RGB frames exactly re-rendered using the original
recorded actions and execution keys. Freeze padding at first success/death. The
first-shell replay is checked bit-for-bit against directly recorded images.
No model forecast is invented or extrapolated for these visualizations.

Estimated total for three models: 12–15 GPU minutes with shared compilation;
allocated MG1 slot is **20 minutes**, after the oracle study and brief gate-seed-1
visual export. Do not launch before explicit release from the coordinator.
All six shells and all three models are prospectively admitted regardless of
outcome. Stop only for source/runtime/resource failure and preserve partial data.

```sh
python -m studies.revision.redcube_goal_distance \
  --model /dev/shm/redcube/gray_frame_still/tile_factored/model_probe.npz \
  --out runs/revision/redcube_goal_distance/m0
```

A conventional H12 CEM alternative was assessed separately and not launched;
its computational and search-adequacy limitations remain in
`research/redcube_goal_followup_assessment.md`.
