# V2 working outline — evidence decides the thesis

V1 is frozen. This document guides the separate V2 manuscript; it is not a results section or a claim that pending experiments succeeded.

## Decision from the independent reviews

The independent V1 assessments were 4/10 and 5/10. Both credited numerical accuracy and claim discipline but identified the same missing contribution: the gate result does not locate a reproducible failure mechanism, and the nuisance theory does not explain the observed high-loss gray-frame/wind failure. Rewording those limitations cannot substitute for an intervention.

The V2 main paper will therefore contain at most **two or three empirical findings**, selected by evidential strength. The Gaussian nuisance construction and measurable small-amplitude proof move to a compact boundary discussion and appendix; neither will be advertised as an empirical explanation. The sensitive downstream object in the Lipschitz paragraph becomes a generic score/action-value read-out, not the fresh-bit example's constant optimal state value.

## Completed new evidence, usable now

Source: `research/craftax_table_v2.json`, derived from `artifacts/revision/craftax_goal_v2.json` (SHA-256 recorded in the table). These are fresh evaluations of one existing 512-dimensional checkpoint, not new model fits.

| Nearby Craftax goal protocol | All 384 starts | Displaced goals, n=247 | Displacement >=2, n=99 |
| --- | ---: | ---: | ---: |
| No-op | 137/384 = 35.68% | 0/247 | 0/99 |
| Random | 65/384 = 16.93% | 19/247 = 7.69% | 1/99 = 1.01% |
| JEPA, six-action open loop | 358/384 = 93.23% | 222/247 = 89.88% | 80/99 = 80.81% |
| JEPA, actual receding-horizon MPC | 369/384 = 96.09% | 232/247 = 93.93% | 87/99 = 87.88% |
| Generating-sequence replay | 382/384 = 99.48% | 245/247 = 99.19% | 97/99 = 97.98% |

- Three evaluation seeds (101, 202, 303), 128 starts each, one trained model seed.
- Goal-generation noise is independent of execution noise. All tested controllers share execution keys and starts.
- Receding-horizon MPC solves horizons 6, 5, 4, 3, 2, 1. Each CEM solve has population 512, 32 elites, and six iterations. It consumes six times the candidate-sequence searches and 3.5 times the latent transition evaluations of the one-shot plan, so the difference is not a compute-matched feedback ablation.
- Position matching is local and six steps long. It is not the full Craftax achievement benchmark.
- Uncertainty intervals concern evaluation episodes for this one checkpoint. They do not estimate training-seed uncertainty.
- This supports a **positive boundary**: the model can guide local action selection beyond no-op, including displaced goals, under fresh execution noise. It does not validate the old reward-head pipeline or explain gate behavior.

## Pending result gates: no numerical conclusions yet

### A. Fixed-protocol gate interventions

Read only final completed artifacts whose reconstruction and protocol checks pass. Compare source of transition and reward/termination under the same policy family, rollout starts, burn-in, horizon, optimizer, updates, and evaluation keys. Report separate model and policy replication; paired contrasts are within the declared model/policy pair and evaluation design.

The oracle-head arm uses true simulator dynamics that become absorbing after true termination while learned continuation can remain nonzero. Report post-termination weights/return diagnostics; do not describe this as an additive decomposition of independent dynamics and head effects.

**Narrative switch:**
- If the simulator/true-signal condition succeeds and learned transitions fail with fixed heads, center the located transition limitation conditional on those heads.
- If simulator/true signals succeed but simulator/learned signals fail, center the fitted-head bottleneck.
- If the common policy fails even with simulator transitions and true signals, explicitly reject the attempted model-specific attribution. Natural-grid CNN sensitivity can test the policy interface; do not pool it into the primary controller family or silently replace failed runs.
- If all successful controllers perform similarly, state that the earlier pipeline gap was not robust to the matched recipe.

### B. Read-out transfer into forecasts

Use the prospective real-to-real, real-to-forecast, forecast-to-forecast, and forecast-to-real probe matrix, with copied-current baselines and train-only normalization. Report horizons and selected-context scope.

**Narrative switch:**
- A strong forecast-trained probe with a weak real-trained probe on forecasts identifies read-out distribution mismatch and weakens the visual information-erasure interpretation.
- Poor forecast-trained decoding is a limitation of the tested probe and forecast, not an information-theoretic proof of absence. Check copied-current and action-dependent baselines before attributing predictive content.

### C. Target-gradient × stochasticity intervention

Prospective factorial: joint target gradients versus detached prediction targets; deterministic versus fresh-wind gray-frame data; three matched seeds; unchanged 512-dimensional representation, architecture, data sizes, update budget, SIGReg, initialization and random-key schedules. Stop-gradient is the only objective-gradient intervention; it is not an EMA teacher or a frozen encoder.

Primary object is the within-seed interaction in accessible position and its training trajectory. For lower-is-better read-out error E, record

`[E(joint, wind) - E(detached, wind)] - [E(joint, still) - E(detached, still)]`.

The interaction's sign is unknown until results exist. Preserve per-seed points and final checkpoints. Intermediate checkpoints explain trajectories and do not select the winning model. Data distributions differ between the two dynamics conditions; the experiment measures this stated stochasticity interaction within the recipe, not all stochastic environments.

**Narrative switch:**
- A consistent wind-selective benefit of detachment, with matched learning curves and diagnostics, can become the primary contribution.
- An equally large deterministic benefit suggests a broader optimization effect instead.
- A null or adverse result rules out this proposed repair at the declared budget; report it prominently.

### D. PVE-inspired value-aware predictor fine-tuning

Secondary and task-supervised. Retain the original model dimensions and rewards. First establish teacher quality on held-out episodes versus the constant-return baseline. Compare one predeclared positive weight against a zero-weight fine-tuning control with all other settings matched. This is not Proper Value Equivalence and cannot establish policy-value preservation beyond the fitted teacher's scope. If teacher validation is weak, that weakness is the result; do not make policy claims from the fine-tuned loss.

## Candidate nine-page structures

**If C provides the strongest distinctive mechanism:**
1. Motivation and bounded finding (1 page).
2. Objective, unchanged model and prospective comparison (1 page).
3. Target-gradient × stochasticity result and learning curves (2 pages).
4. What forecast transfer and fixed-protocol control do/do not localize (2 pages).
5. Craftax positive boundary (0.6 page).
6. Closest related work, theoretical boundary and limitations (1.4 pages).
7. Reproducibility/AI statements (remaining space).

**If A and B provide the strongest localization:**
1. Motivation: diagnosing the point at which an abstraction ceases to serve control.
2. Shared-model intervention design and controller qualification.
3. Matched gate results, with failed controls prominent.
4. Forecast read-out transfer as an explanatory test.
5. Nuisance intervention as a boundary, positive or null.
6. Craftax positive boundary, prior work and precise limits.

**If all new interventions are null or optimizer-dominated:**
Write a careful case study about the instability of model-failure attribution under read-out and training protocol changes. Do not retain the old model-deficiency title merely because it was memorable. State that the evidence does not support a JEPA-specific failure mechanism. This can be useful science without forcing a strong-accept claim.

## Presentation changes

- V2 figures use only `../shared-v2/figures`; never mutate the V1 shared assets.
- Main tables show causal conditions, counts and primary outcomes, not dozens of historical recipe rows.
- Keep negative controls and failed variants adjacent to the headline contrast.
- Move historical unmatched gate rows, all broad recipe comparisons and full population proofs to the appendix.
- State the main conclusion once in the abstract/introduction and establish it through the result sections; avoid repeating long defensive lists after every measurement.
- Use current primary literature to identify the exact contribution relative to Sobal 2022, LeWM, PSG-JEPA, Temporal-Distance JEPA, physically grounded JEPA, and value-aware modeling. No first-discovery claim for the retention/prediction/control distinction.

## Forecast interpretation correction (during independent audit)

The completed read-out arrays show a valid reduction of transferred-probe MAE after refitting, but the model-2 h8 pre-reveal slice has negative within-slice forecast-probe correlation in an independent diagnostic. The 14.24-pixel baseline is a global training-label mean evaluated on a selected slice; an MAE gain over it can be calibration rather than position tracking. Suspend the earlier proposed claim of recovered useful wall-position information until the full correlation/RMSE/constant-baseline audit resolves its exact scope. Keep all models and both normalizations. The original decoder's failure still does not prove information deletion.
