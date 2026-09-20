# Independent review of the H48 analyzer

Status: protocol and reference-code review complete; new analyzer review pending its author's ready signal. This is not a scientific campaign audit or an approval to execute GPU work.

## Scope and prospective acceptance requirements

The analyzer must preserve the frozen 36-run H48 matrix (three fixed model checkpoints, three policy seeds, four arms), the 50-update/48-step package, 1,228,800 nominal positions, and final-only evaluation over 256 ordered slots at key 20260919. H16 references are a separate 150-update/16-step stratum. Missing H16 references may leave cross-horizon contrasts unresolved without invalidating a complete H48 campaign.

Artifact validation must enforce the exact checkpoint/data hashes, native-grid policy and 512-coordinate encoder dimensions, all fixed CLI hyperparameters, and the complete normalized shared source map. The approved termination harness is the only extra source entry. Merely agreeing across artifacts is insufficient if all artifacts share an unapproved value. All six required files, including `model.pkl`, must exist; the saved binary must match its recorded hash. The older analyzers allow a missing policy binary, so this behavior must not be inherited.

The ledger must contain the full ordered update sequence, correct position counts at every update, finite recorded diagnostics, and exact zero postterminal weighted mass and reward for `oracle_reward` and `oracle_true`. The learned-transition arm does not record true-terminal diagnostics in the immutable harness; their absence must remain explicit, rather than being replaced by invented zeros. Raw episode outcomes must reconstruct all scalar summaries. Survival uses return zero; a death at step 128 remains a death. A return-zero survivor must have reached step 128.

Within H48, one missing or invalid arm excludes the full policy-seed block from provisional four-arm contrasts. All 36 intended paths remain represented, including paths without `final.json`. Across horizons, matching must check the first 50 semantic H16 start batches against all 50 H48 batches. Within-horizon full start arrays must agree across arms. Matching shapes or compressed NPZ hashes alone is insufficient. Start reset flags and the relation between first and last context frame should also be valid.

## Statistical checks

For each fixed model and contrast, form `D[policy, slot] = target - reference`. The policy bootstrap resamples `mean_slot(D)` over the three paired policy seeds. The fixed-policy evaluation bootstrap resamples `mean_policy(D)` over the shared evaluation slots. Pooling the 768 outcomes, independently bootstrapping the arms, or independently resampling slots for each policy changes the estimand. Cross-horizon contrasts use H48 minus H16 and preserve `(model, policy, arm, slot)` pairing.

Useful hand-calculable fixtures, independently confirmed for 10,000 draws, are:

| Episode-difference matrix | Mean | Policy interval | Joint-slot interval |
|---|---:|---|---|
| `[[0,0,0,0],[2,2,2,2],[4,4,4,4]]` | 2 | [0,4] | [2,2] |
| `[[0,0,6,6],[6,6,0,0],[3,3,3,3]]` | 3 | [3,3] | [3,3] |

These expose conflation of training-seed and evaluation-slot variation. Report all arms, policies and fixed-model means, equal-model descriptive averages and ranges, four within-H48 contrasts, and all four cross-horizon contrasts. Keep the adaptive admission and package-change qualification. No pooled policy count is a count of independent models.

## Read-only reference checks completed

The eight current immutable source hashes match the frozen protocol and audited native-grid reference config. All 27 `matched_grid` reference runs have the six required local artifact files. At the time of this check, none of the nine `termination-native-grid` reference directories had been copied locally; their cross-horizon arm must remain unresolved until valid artifacts arrive. These checks did not change any training source, run artifact or GPU process.

## New-code review and CPU failure tests

Pending the new analyzer's ready signal.
