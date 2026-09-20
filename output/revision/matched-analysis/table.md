# primary matched-PPO final endpoints

Status: **complete**. Each row is one fixed checkpoint. Parentheses are policy-run sample SDs.

| Model seed | Arm | Policy seeds | Walls/episode | Survival | Discovery |
|---|---|---|---:|---:|---:|
| 0 | learned | 0, 1, 2, 3, 4 | 0.285 (0.265) | 0.938 (1.284)% | 0/5 |
| 0 | oracle_heads | 0, 1, 2, 3, 4 | 0.230 (0.366) | 0.625 (0.980)% | 0/5 |
| 0 | oracle_true | 0, 1, 2, 3, 4 | 0.452 (0.099) | 0.469 (0.327)% | 0/5 |
| 1 | learned | 0, 1, 2, 3, 4 | 0.217 (0.356) | 0.781 (1.353)% | 0/5 |
| 1 | oracle_heads | 0, 1, 2, 3, 4 | 0.068 (0.152) | 0.156 (0.349)% | 0/5 |
| 1 | oracle_true | 0, 1, 2, 3, 4 | 0.341 (0.210) | 0.391 (0.391)% | 0/5 |
| 2 | learned | 0, 1, 2, 3, 4 | 0.320 (0.196) | 0.391 (0.276)% | 0/5 |
| 2 | oracle_heads | 0, 1, 2, 3, 4 | 0.096 (0.119) | 0.000 (0.000)% | 0/5 |
| 2 | oracle_true | 0, 1, 2, 3, 4 | 0.543 (0.119) | 1.406 (0.445)% | 0/5 |

Differences are target minus reference. Intervals resample paired policy seeds within one fixed checkpoint, conditional on the evaluation set.

| Model seed | Contrast | Paired seeds | Δ walls [95% interval] | Δ survival percentage points [95% interval] |
|---|---|---:|---:|---:|
| 0 | oracle_heads − learned | 5 | -0.055 [-0.368, +0.259] | -0.312 [-1.328, +0.312] |
| 0 | oracle_true − oracle_heads | 5 | +0.222 [-0.109, +0.452] | -0.156 [-0.859, +0.312] |
| 0 | oracle_true − learned | 5 | +0.166 [+0.015, +0.348] | -0.469 [-1.406, +0.469] |
| 1 | oracle_heads − learned | 5 | -0.149 [-0.492, +0.045] | -0.625 [-1.875, +0.000] |
| 1 | oracle_true − oracle_heads | 5 | +0.273 [-0.027, +0.498] | +0.234 [-0.312, +0.703] |
| 1 | oracle_true − learned | 5 | +0.123 [-0.195, +0.437] | -0.391 [-1.406, +0.469] |
| 2 | oracle_heads − learned | 5 | -0.223 [-0.429, +0.046] | -0.391 [-0.625, -0.156] |
| 2 | oracle_true − oracle_heads | 5 | +0.447 [+0.329, +0.598] | +1.406 [+1.094, +1.719] |
| 2 | oracle_true − learned | 5 | +0.223 [+0.034, +0.420] | +1.016 [+0.625, +1.406] |

- Policy-seed bootstrap intervals condition on each fixed model checkpoint and the fixed evaluation set.
- Episode bootstrap intervals condition on the fitted policies; they do not represent training-seed uncertainty.
- Only three independently fitted encoders are planned. Cross-model summaries are descriptive means and ranges, without model-family confidence intervals.
- Intervals are unadjusted descriptive 95% percentile intervals for two endpoints and three contrasts; no significance-selected claims or best-policy selection.
- oracle_heads freezes true terminal simulator states but retains learned reward and continuation signals. Its contrast with learned includes this terminal-support change.
