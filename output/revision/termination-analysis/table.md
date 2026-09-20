# True-termination sensitivity

Status: **complete**. All contrasts are within fixed checkpoints and policy classes.

| Torso | Model seed | Paired seeds | Contrast | Δ walls [95% policy interval] | Δ survival, pp [95% policy interval] |
|---|---|---:|---|---:|---:|
| mlp | 0 | 5 | oracle_reward − oracle_heads | -0.066 [-0.486, +0.261] | +0.000 [-1.172, +1.094] |
| mlp | 0 | 5 | oracle_true − oracle_reward | +0.287 [+0.003, +0.512] | -0.156 [-0.938, +0.625] |
| mlp | 1 | 5 | oracle_reward − oracle_heads | +0.148 [-0.083, +0.380] | +0.703 [-0.156, +1.719] |
| mlp | 1 | 5 | oracle_true − oracle_reward | +0.124 [-0.075, +0.391] | -0.469 [-1.328, +0.391] |
| mlp | 2 | 5 | oracle_reward − oracle_heads | -0.018 [-0.162, +0.094] | +0.156 [+0.000, +0.312] |
| mlp | 2 | 5 | oracle_true − oracle_reward | +0.465 [+0.312, +0.619] | +1.250 [+0.938, +1.562] |
| native_grid | 0 | 3 | oracle_reward − oracle_heads | -0.329 [-0.785, +0.000] | -1.432 [-3.516, +0.000] |
| native_grid | 0 | 3 | oracle_true − oracle_reward | +0.514 [+0.422, +0.605] | +0.521 [-0.391, +1.172] |
| native_grid | 1 | 3 | oracle_reward − oracle_heads | -0.738 [-0.793, -0.664] | -2.344 [-3.125, -1.953] |
| native_grid | 1 | 3 | oracle_true − oracle_reward | +0.367 [+0.117, +0.590] | +0.521 [+0.000, +0.781] |
| native_grid | 2 | 3 | oracle_reward − oracle_heads | -0.048 [-0.656, +0.543] | +0.260 [-1.172, +1.953] |
| native_grid | 2 | 3 | oracle_true − oracle_reward | +0.271 [+0.164, +0.453] | +0.000 [-0.391, +0.391] |

- This is an explicitly triggered sensitivity after inspecting primary terminal diagnostics; it is not an independently preplanned primary experiment.
- All contrasts condition on each saved encoder/head checkpoint, policy torso and training protocol. Three model fits do not establish a model-family effect.
- The termination contrast changes continuation, termination weighting and postterminal support together; it is not a reward-only intervention.
- The reward contrast keeps exact dynamics and true termination fixed, but policies trained under different rewards subsequently visit different states.
- Unadjusted percentile intervals describe paired policy-seed variation conditional on the fixed evaluation set; episode intervals condition on the fitted policies.
- MLP and native-grid policy results are separate strata and are never pooled as model replication.
