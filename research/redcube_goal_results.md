# Red-cube goal-control results and scope

The unchanged three-model hard-goal primary failed to provide a positive control:
H3 reached 5, 5, and 9 of 128 goals; H1 reached zero for all three; paired random
actions reached 3 and the privileged physical-distance oracle 128. These results
remain in full at `runs/revision/redcube_goal/gray_frame_still/`.

The separately declared post-outcome operating-range diagnostic retained the
exact same controller and 128 prefixes, with all six exact target distances and
all three model seeds admitted before follow-up outcomes. Every entry below is an
exact goal-hit count out of 128, before native death, within 24 actions.

| Controller |1 action|2 actions|3 actions|6 actions|9 actions|12 actions|
|---|---:|---:|---:|---:|---:|---:|
|jepa_h1, model 0|128|39|22|4|0|0|
|jepa_h3, model 0|128|128|121|25|3|0|
|jepa_h1, model 1|128|36|16|4|0|0|
|jepa_h3, model 1|128|128|123|32|2|0|
|jepa_h1, model 2|128|46|21|1|0|0|
|jepa_h3, model 2|128|128|124|34|1|0|
|random (identical across models)|65|46|19|5|1|0|
|fixed_cycle (identical across models)|67|0|0|0|0|0|
|oracle (identical across models)|128|128|128|128|128|128|

This supports nearby visual-goal control with a frozen learned dynamics model.
The H3 versus H1 comparison shows the benefit of the tested multi-step search
within its operating range. It does not establish deep strategic planning or a
causal explanation of the far-goal failures. Latent goal-distance geometry,
predictor errors, and search horizon all remain possible contributors.

The original failed primary is not replaced by nearby successes. The distance
sweep is explicitly post-outcome, contains all six distances and all three models,
and holds the planner, model weights, latent width, scoring, action budget, and
native dynamics fixed. No reward training, latent amplification, privileged
controller feature, or result-dependent trajectory selection was used.

Intervals are separate Wilson intervals for each model and distance. The 768 rows
per model reuse 128 prefixes and must not be pooled into an IID confidence interval.
The one-action-distance secondary of being within 3 pixels is already satisfied at
the start and is not evidence of control.

The block completed normally in 753.12 seconds and GPU1 was released immediately
after completion. Independent final verification passed all 8,179 checks and
reconstructed 176,334 native executed transitions. The audit and figures are in
`research/redcube_goal_distance_independent_audit.{md,json}` and
`output/revision/redcube-goal-distance-analysis/`.
