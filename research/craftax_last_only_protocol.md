# Craftax last-target-only transfer: admitted, pending correctness checks

Purpose: test whether the same prespecified loss-target selection studied in gray cube worlds changes actual nearby pixel-goal control in Craftax. This is a separate prospective follow-up, admitted independently of the cube outcome direction. It does not alter the original three-model replication.

## Frozen matrix

- Baseline: original joint full-prefix objective, model seeds 0/1/2. Seed 0 is already complete; seeds 1/2 are in the independently reviewed exact-recipe queue.
- Treatment: last-target-only joint objective, new model seeds 0/1/2, each trained from scratch for all 8,500 updates. Retain and evaluate every model regardless of direction or size of effect.
- Exact original Craftax Config in the seed-0 checkpoint; 512 coordinates (16×32), tile encoder/factored predictor, full history 8, 17 native actions, batch 128, AdamW lr .001, 500k transitions. Config differs only by model seed; the loss-selection label is a separate provenance field.
- Keep the entire nine-frame encoder forward, all eight predictor outputs, SIGReg over all nine frames, encoder gradients through both target and prediction, identical RNG construction/windows/optimizer/schedule, no latent transformations or extra history-length multiplier.
- The sole loss change is prediction MSE: from mean((pred − emb[:,1:])²) to mean((pred[:,-1] − emb[:,-1])²), both using a coordinate mean. The coefficient on SIGReg stays .09.
- No decoder, reward/value head, auxiliary labels, reward shaping or downstream fine-tuning.

## Evaluation and estimand

Use the unchanged Craftax evaluator and its existing 384 contexts (evaluation seeds 101/202/303 × 128 starts), independent goal/execution noise, H6, CEM population512/elites32/iterations6, all five controllers. Verify exact equality of reset/start/goal coordinates, reference action sequences and no-op/random/reference trajectories against the original report. The evaluator's input model changes; no task definition or action interface changes.

Primary endpoint: alive endpoint success of the one-shot six-action JEPA plan on all384 goals. Primary contrast: last-only minus original success, paired by training seed and by evaluation episode. Report each of the three model contrasts plus mean and sample SD across matched model seeds. Do not treat 2304 model×episode outcomes as independent new environmental contexts. Secondary endpoints, fixed now: displaced-goal and displacement≥2 success, endpoint L1, and receding-horizon success (which uses additional planning compute).

This changes the supervised target positions and their number. It is not a pure conditional-history ablation: inference history remains8, and training BatchNorm still couples encoded frames. Positive results would support this specific loss-target selection for local goal control; negative or mixed results would establish a task-dependent tradeoff. Neither result establishes general Craftax skill learning.

## Implementation boundary and admission checks

The pinned gray-only CLI intentionally rejects Craftax and must not be edited or repurposed. A new Craftax-only wrapper can load the original checkpoint Config and import the unchanged hash-pinned `last_only_objective.make_run(cfg, "last_only")`, which is action-count agnostic. Its data validation must require the known Craftax archive SHA, native17 actions and exact frame/action shapes. No edits to the six pinned cube files or original train/evaluator.

Before production launch: independently review the new wrapper; show a 17-action numerical fixture where the full-prefix update/driver matches original `T.train`; verify only the last prediction residual contributes, both target and prediction gradients remain, and the all-nine-frame SIGReg term/normalization is unchanged. Record config, all training/evaluator source hashes, dataset hash, RNG construction, initial model hash and sampled-window/key hashes, train/final checkpoint hashes and measured times. Earlier baseline initialization/sampling were determined by the identical source/seed construction but not all archived at runtime; distinguish reconstructed pairing audits from measured historical logs.

Runtime estimate: about10min fit +13min frozen goal evaluation per model from measured existing runs; three fits/evaluations about70min, with 10–15min allowance for wrapper validation and compilation. Start only after the original two-replica queue releases e2eGPU0 and after root reviews the concrete code/test evidence. Root admitted this fixed matrix before any last-only outcomes; production launch is conditioned only on correctness/runtime checks, exclusive GPU release, and the fixed deadline. The deadline remains2026-09-20 07:41UTC.
