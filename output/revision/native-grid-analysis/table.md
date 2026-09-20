# native-grid matched-PPO final endpoints

Status: **complete**. Each row is one fixed checkpoint. Parentheses are policy-run sample SDs.

| Model seed | Arm | Policy seeds | Walls/episode | Survival | Discovery |
|---|---|---|---:|---:|---:|
| 0 | learned | 0, 1, 2 | 0.000 (0.000) | 0.000 (0.000)% | 0/3 |
| 0 | oracle_heads | 0, 1, 2 | 0.431 (0.436) | 2.083 (1.846)% | 0/3 |
| 0 | oracle_true | 0, 1, 2 | 0.616 (0.090) | 1.172 (0.391)% | 0/3 |
| 1 | learned | 0, 1, 2 | 0.284 (0.246) | 0.651 (0.597)% | 0/3 |
| 1 | oracle_heads | 0, 1, 2 | 0.738 (0.067) | 2.344 (0.677)% | 0/3 |
| 1 | oracle_true | 0, 1, 2 | 0.367 (0.238) | 0.521 (0.451)% | 0/3 |
| 2 | learned | 0, 1, 2 | 0.202 (0.158) | 0.000 (0.000)% | 0/3 |
| 2 | oracle_heads | 0, 1, 2 | 0.229 (0.370) | 0.391 (0.677)% | 0/3 |
| 2 | oracle_true | 0, 1, 2 | 0.452 (0.256) | 0.651 (0.813)% | 0/3 |

Differences are target minus reference. Intervals resample paired policy seeds within one fixed checkpoint, conditional on the evaluation set.

| Model seed | Contrast | Paired seeds | Δ walls [95% interval] | Δ survival percentage points [95% interval] |
|---|---|---:|---:|---:|
| 0 | oracle_heads − learned | 3 | +0.431 [+0.000, +0.871] | +2.083 [+0.000, +3.516] |
| 0 | oracle_true − oracle_heads | 3 | +0.185 [-0.180, +0.516] | -0.911 [-2.734, +1.172] |
| 0 | oracle_true − learned | 3 | +0.616 [+0.516, +0.691] | +1.172 [+0.781, +1.562] |
| 1 | oracle_heads − learned | 3 | +0.454 [+0.242, +0.758] | +1.693 [+1.172, +1.953] |
| 1 | oracle_true − oracle_heads | 3 | -0.371 [-0.676, -0.074] | -1.823 [-3.125, -1.172] |
| 1 | oracle_true − learned | 3 | +0.083 [-0.312, +0.395] | -0.130 [-1.172, +0.781] |
| 2 | oracle_heads − learned | 3 | +0.027 [-0.230, +0.312] | +0.391 [+0.000, +1.172] |
| 2 | oracle_true − oracle_heads | 3 | +0.223 [-0.461, +0.707] | +0.260 [-0.781, +1.562] |
| 2 | oracle_true − learned | 3 | +0.250 [-0.148, +0.477] | +0.651 [+0.000, +1.562] |

- Policy-seed bootstrap intervals condition on each fixed model checkpoint and the fixed evaluation set.
- Episode bootstrap intervals condition on the fitted policies; they do not represent training-seed uncertainty.
- Only three independently fitted encoders are planned. Cross-model summaries are descriptive means and ranges, without model-family confidence intervals.
- Intervals are unadjusted descriptive 95% percentile intervals for two endpoints and three contrasts; no significance-selected claims or best-policy selection.
- oracle_heads freezes true terminal simulator states but retains learned reward and continuation signals. Its contrast with learned includes this terminal-support change.
