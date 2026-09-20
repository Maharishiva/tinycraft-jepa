# Analysis protocol for the true-termination sensitivity

Declared on 2026-09-19 before training the additional arm. This is a sensitivity triggered by the prospectively reserved terminal-support diagnostic: existing `oracle_heads` runs placed continuation-weighted mass after true death and sometimes received positive fitted reward there. It is reported separately from the immutable primary three-arm experiment.

## Fixed design and contrasts

The added `oracle_reward` arm uses exact simulator transitions and true termination, with the original frozen learned reward head evaluated on the current encoded observation and outgoing action. Rewards are not clipped, reshaped, replaced, or manually zeroed. True termination cuts GAE and survival weighting, so continuation-weighted postterminal mass and reward must be exactly zero. Hidden simulator state remains absent from the actor.

Include all three model seeds and all previously admitted policy seeds: 15 additional MLP policies (five per model) and nine additional native-grid policies (three per model), for 24 new runs. All request 150 updates, batch 512, horizon 16, and the fixed 256-episode evaluation key 20260919. Never pool the two policy classes or omit unfavorable seeds.

Predeclared primary endpoints are walls per episode and survival probability. Compare:

1. `oracle_reward − oracle_heads`: true termination versus learned continuation under learned rewards and exact absorbing transitions. This contrast includes termination weighting and the after-death extension; it is not reward-only.
2. `oracle_true − oracle_reward`: true reward versus fitted reward under true termination and exact transitions. This is the matched reward-signal contrast. Training policies may subsequently visit different states, so it does not quantify a statewise reward-model error alone.

Positive differences favor the more exact signal. Preserve all endpoint and contrast values. The new sensitivity cannot replace the original estimand or retroactively turn the primary experiment into a four-arm preplanned study.

## Artifact validation and the permitted source difference

`analyze_termination.py` reads new `oracle_reward` final artifacts plus existing `oracle_heads` and `oracle_true` artifacts. A valid seed block contains exactly one full-budget run of each, with equal model SHA256, dataset SHA256, semantic training-start indices, matching hyperparameters, policy parameter count, ordered evaluation slots and evaluation key. Config, final metrics, raw episodes, checkpoint metadata, reconstruction evidence and every training ledger step are checked independently. Nonfinite training diagnostics leave the planned seed unresolved rather than silently converting the report into a survivor-only comparison.

The original harness must have SHA256 `6a3460b2db2e47713b7af604454a4b04066c96bfd7748b5c991cfbc433176269`. The sensitivity harness is independently audited and pinned to SHA256 `1c003dd4e019b2179981877310e131304d13a7150ac7da93e11d5164d011feb1`. Its config must record that `parent_harness_sha256`, the approved `sensitivity_harness_sha256`, and both harnesses in `source_sha256`. After removing the new harness entry, the entire normalized dependency-hash dictionary must exactly match the reference run. Thus the approved new harness file is the only permitted source difference; policy, encoder loader, environment, heads and PPO implementation hashes remain equal. The separate implementation audit establishes the minimal semantic change in that new harness.

The analyzer additionally requires exactly zero recorded continuation-weighted postterminal mass and reward in every new-arm update. It reports the corresponding baseline diagnostics, even if the additional arm changes performance only weakly. A missing, duplicated, invalid or mismatched arm excludes the entire seed block from provisional numerical comparisons and prevents final complete status. The final publication report requires all 24 new policies and their paired references.

## Statistics and scope

Report every fixed model checkpoint separately within each torso. Each policy-seed contrast is the mean of its paired episode differences; then average policy differences with equal seed weights. Use the same deterministic 10,000-draw, 95% percentile bootstrap as the primary analysis: resample paired policy seeds within one fixed model, conditional on the evaluation set. Also report conditional evaluation intervals from paired episode resampling with fitted policies held fixed. Shared evaluation slots are resampled jointly across policy seeds.

The three model means, their equal-model descriptive average and their range may be shown. No model-family confidence interval is computed. Two policy classes are distinct experimental strata, not extra model replicates. All intervals are unadjusted descriptive intervals; no threshold or significant subset determines what is reported. This sensitivity was triggered by primary diagnostics and must be identified as such in the paper.

## Reproduction

```bash
python -m studies.revision.analyze_termination --self-test
python -m studies.revision.analyze_termination --require-complete
```

Defaults read `runs/revision/termination-mlp` and `runs/revision/termination-native-grid`, paired respectively with `runs/revision/matched` and `runs/revision/matched_grid`. Explicit CLI root arguments support the finalized scheduler paths without changing the estimand. Outputs are `output/revision/termination-analysis/publication.json`, `table.md`, and `conclusions.txt`. No run files, original primary tables, policies or plots are modified. An incomplete report is written with explicit missing/invalid blocks; `--require-complete` then exits with code 2.

Synthetic checks use known paired differences to verify that opposed policy effects average correctly, a constant reward effect has the expected bootstrap interval, deaths at the cap remain deaths, unauthorized dependency changes break pairing, and positive postterminal weighted mass invalidates the true-termination arm.
