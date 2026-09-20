# Independent goal-distance diagnostic audit

**Complete: 3 of 3 admitted models audited.**

**Available-artifact audit: PASS.** Independent NumPy-only reconstruction; original primary report unchanged.

| Training seed | Distance (actions) | JEPA H1 exact hits | JEPA H3 exact hits | Random exact hits | Fixed cycle exact hits | Oracle exact hits |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 128/128 | 128/128 | 65/128 | 67/128 | 128/128 |
| 0 | 2 | 39/128 | 128/128 | 46/128 | 0/128 | 128/128 |
| 0 | 3 | 22/128 | 121/128 | 19/128 | 0/128 | 128/128 |
| 0 | 6 | 4/128 | 25/128 | 5/128 | 0/128 | 128/128 |
| 0 | 9 | 0/128 | 3/128 | 1/128 | 0/128 | 128/128 |
| 0 | 12 | 0/128 | 0/128 | 0/128 | 0/128 | 128/128 |
| 1 | 1 | 128/128 | 128/128 | 65/128 | 67/128 | 128/128 |
| 1 | 2 | 36/128 | 128/128 | 46/128 | 0/128 | 128/128 |
| 1 | 3 | 16/128 | 123/128 | 19/128 | 0/128 | 128/128 |
| 1 | 6 | 4/128 | 32/128 | 5/128 | 0/128 | 128/128 |
| 1 | 9 | 0/128 | 2/128 | 1/128 | 0/128 | 128/128 |
| 1 | 12 | 0/128 | 0/128 | 0/128 | 0/128 | 128/128 |
| 2 | 1 | 128/128 | 128/128 | 65/128 | 67/128 | 128/128 |
| 2 | 2 | 46/128 | 128/128 | 46/128 | 0/128 | 128/128 |
| 2 | 3 | 21/128 | 124/128 | 19/128 | 0/128 | 128/128 |
| 2 | 6 | 1/128 | 34/128 | 5/128 | 0/128 | 128/128 |
| 2 | 9 | 0/128 | 1/128 | 1/128 | 0/128 | 128/128 |
| 2 | 12 | 0/128 | 0/128 | 0/128 | 0/128 | 128/128 |

Verified 8,179 checks and 176,334 executed native transitions.

All original prefixes, exact-shell goal draws, warmup RGB geometry, first-hit/death masks, per-cell metrics and intervals, source/artifact hashes, selected visual rows, terminal padding and visible shared nuisance pixels were checked. When multiple models are included, their entire cohorts and raw baseline trajectories are identical.

Limits:

- This is the declared post-outcome distance-response diagnostic; it does not replace the failed hard-goal primary.
- No JAX PRNG or learned network is re-executed; geometry, transitions, observations and metrics are checked from raw artifacts.
- Runtime encoder shapes are established by hashed source and manifests, not independent GPU instrumentation.
- Confidence intervals remain separate for each model and distance; repeated prefixes are not pooled as independent trials.
