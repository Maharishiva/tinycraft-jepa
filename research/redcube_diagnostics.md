# Redcube representation diagnostics audit

Read-only audit of current local source and saved artifacts on 2026-09-19. No model/GPU execution. Paths below are relative to `/Users/rldoeswork/Documents/rl/tinycraft-jepa`.

## Verdict

Saved evidence supports **poor conditioning of controlled-state information in black-background class-token encoders**, and **a large real-latent vs predicted-latent probe gap**. It does not justify saying that failed probes prove position information is absent, nor that the predictor necessarily deletes all position information. The spectral story currently has one training seed, no machine-readable spectral artifact, and diagnostic-source/log drift. Keep it as a scoped mechanism case study, not a multi-seed universal claim.

## 1. Directly saved conditioning results

`studies/redcube/diag_cls.py:64-93` reads the first 300 time rows of `train.npz` for probe fitting and the next 60 for testing (defaults). It compares a standardized ridge, a fixed MLP on raw/standardized/PCA-whitened latents, and the mean-position guess. The MLP is two hidden layers of width 128, GELU, 6000 Adam steps, batch 2048, seed 0 (`:25-57`). PCA directions are floored at 1% of the largest singular standard deviation (`:83-84`). Errors are mean Manhattan pixel error (`:88`).

| Saved setting and arm | Effective rank | Standardized MLP L1 | Whitened MLP L1 | Mean guess L1 | Evidence |
|---|---:|---:|---:|---:|---|
| off_still, lewm6 | 47.7 / 192 | 18.53 | 0.71 | 19.56 | `runs/redcube/diag_cls_off_still.log:1-2` |
| off_still, tile_cls_ar | 47.8 / 512 | 19.56 | 0.68 | 19.56 | same `:3-4` |
| off_still, tile_ar grid | 248.6 / 512 | 0.05 | 0.11 | 19.56 | same `:5-6` |
| off_wind, lewm6 | 43.8 / 192 | 0.10 | 0.01 | 19.84 | `runs/redcube/diag_cls_off_wind.log:1-2` |
| gray_episode_still, lewm6 | 110.6 / 192 | 17.93 | 30.00 | 19.56 | `runs/redcube/diag_cls_gray_episode_still.log:1-2` |
| gray_episode_still, tile_cls_ar | 90.0 / 512 | 18.10 | 41.36 | 19.56 | same `:3-4` |
| gray_episode_wind, lewm6 | 122.7 / 192 | 20.84 | 24.17 | 19.84 | `runs/redcube/diag_cls_gray_episode_wind.log:1-2` |
| gray_frame_still, lewm6 | 166.4 / 192 | 32.35 | 32.85 | 19.56 | `runs/redcube/diag_cls_gray_frame_still.log:1-2` |
| color_episode_wind, lewm6 | 143.4 / 192 | 21.30 | 23.89 | 19.84 | `runs/redcube/diag_cls_color_episode_wind.log:1-2` |
| color_episode_wind, tile_cls_ar | 99.5 / 512 | 21.13 | 43.62 | 19.84 | same `:3-4` |

Interpretation: whitening recovers near-exact position on black still scenes, so the relevant information is present in the saved encoders. It does not rescue the tested textured cases. All logs say zero coordinate dimensions below 1e-3 of the largest std, and their coordinate std are order one. Therefore "tiny directions" refers to principal/covariance directions, not individually dead coordinates. Effective rank alone cannot diagnose the failure: off_wind has even lower rank than off_still but much better positional decoding.

Caveats:

- Standardization uses `sd = z.std(0)` over fit **and held-out** rows (`diag_cls.py:74,82`), although mean and PCA basis use fit rows only (`:81,83`). This unsupervised test-statistic leakage is avoidable in a clean rerun.
- These are temporally held-out rows within the training archive, not a separate evaluation archive and not guaranteed held-out episodes (`:67-70`).
- Current code emits `MLP on raw` (`:90`), but all six saved logs omit that field. This proves source/log drift; current source cannot be attributed exactly to the old runs without the historical revision/command.
- Saved diagnostic logs contain no exact checkpoint hash, invocation, rows/test settings or training seed. Source training script defaults to seed 0, but exact artifact identity is not independently recoverable from these logs.

## 2. Spectral artifact supports a narrow geometric interpretation

I visually inspected `runs/redcube/latent_viz_black.png`. Its first row, off_still LeWM, shows a sharp spectrum drop at about 48 principal directions, effective rank about 47/192, no useful improvement in the top-k whitened probe through k=48 (about 17.5 px), then improvement at k=96 (about 9.6 px) and k=192 (about 7.9 px). The off_wind LeWM row has controlled position visible in early components and a small probe error.

The image does **not** replicate the 0.71 px result: its procedure uses eval windows, a 90/10 flattened window-frame split, and default 3000 rather than 6000 MLP steps (`studies/redcube/latent_viz.py:81-93,96-103,124,177-182`). The label "193 distinct latents in the 48 directions above 1%" is a **rounded prototype count**: `np.unique(np.round(white[:, :big] * 2), axis=0)` (`:91-92`), not an exact count of distinct encoder outputs. Do not write "the latent collapses to ~200 values" as a literal claim.

The spectral plot itself is visually overcrowded and its top-right numbers are clipped in the saved PNG. Rebuild as a compact spectrum plus probe-vs-k plot with exact numeric export, train-only fit, a separate eval split, and provenance. Fit/test windows in the old diagnostic can overlap because `eval.py:69-76` samples windows with replacement and no disjointness enforcement; this further limits a claim of strict generalization from the image.

## 3. True-latent versus predicted-latent behavior is independently logged by bench

The benchmark fits probes on training frames and reads them on evaluation real/predicted latents (`studies/redcube/bench.py:472-488`). Class-token preprocessing is selected between standardization and whitening on the last 10% of fit rows, then the selected probe is refit on all training rows (`:104-165`). Outputs are clipped to the frame (`:157`).

For black still scenes:

- `runs/redcube/off_still/allarms/bench.log:83`: lewm6 real-frame held-out probe error 0.09 px; predicted-latent error 44.30 at step 1, 45.27 at step 8, 46.46 at step 16.
- Same file `:80`: tile_cls_ar real-frame held-out 0.29 px; predicted 27.18 / 28.32 / 29.68 at steps 1 / 8 / 16.
- Same file `:76`: tile_ar grid real-frame held-out 0.12 px; predicted 0.24 / 0.27 / 0.31.
- Same file `:94`: lewm6 decoded-context pixel probe errors 16.39--17.65 px, decoded predicted error 17.76--19.71 px. Thus its trained visualization decoder also fails to recover the position that whitening can recover from true latents.

For black wind scenes:

- `runs/redcube/off_wind/allarms/bench.log:83`: lewm6 real-frame held-out 0.05 px; predicted 0.73 / 8.96 / 21.33.
- Same file `:80`: tile_cls_ar real-frame held-out 0.04 px; predicted 0.74 / 9.12 / 21.95.

These support a mismatch between precise encoder information and the learned predictor/decoder in the still setting. They do **not** prove that predicted latents contain no position information: a probe trained only on real latents may extrapolate badly on predicted latents. Use the precise phrase "the real-latent position probe fails on predictions" unless a separate predicted-latent probe or controlled intervention establishes stronger suppression.

## 4. diag_jepa source exists, numeric evidence does not

`studies/redcube/diag_jepa.py` would measure a shared linear cube-token detector on true vs predicted latents (`:102-114,150-160`), one-step latent MSE against copy-last and mean-latent baselines (`:127-149`), action sensitivity (`:145-149`), and decoded true/predicted latent cube recovery (`:162-174`). I found no saved corresponding output anywhere under local `runs/` or `studies/` by exact distinctive log strings (`[predictor] one-step`, `[encoder] shared`, `[decoder] decode`). Do not cite its prospective metrics as observed.

Method cautions if revived: it uses `train.npz` (`:69`) and first 40 time rows x 40 envs by default (`:65-73`); most predictor metrics mix rows that trained the detector with held-out rows (`:127-155`), and history is hardcoded to 3 regardless of model recipe (`:129-130`). It should be adapted to the selected model history and a held-out archive. The decoder red-mass heuristic is environment-specific (`:47-59`) and should stay a diagnostic, not a general objective.

## 5. Seed evidence

The spectral/whitening evidence has no independent training-seed repeats. `train_arms.sh:61-67` defines seeds 1/2 only for `tile_factored_s1/s2`; lewm6 and tile_cls_ar use common seed 0. Empty `runs/redcube/seed0.out` and `seed1.out` contain no usable evidence.

There is distinct three-training-seed grid evidence:

- Gray frame still: real-latent probe 0.11 / 0.13 / 0.10 px for base/s1/s2; predicted step-8 4.27 / 6.15 / 4.11 (`runs/redcube/gray_frame_still/allarms/bench.log:95,103-104`).
- Gray frame wind: real-latent probe 21.35 / 21.37 / 21.26; predicted step-8 19.70 / 20.03 / 21.44 (`runs/redcube/gray_frame_wind/allarms/bench.log:89,97-98`).

This is a repeatable failure under the tested probe on that grid recipe/environment pair, not a replication of the black-background class-token spectral mechanism. Multiple evaluation windows and probe RNG seeds are not training seeds.

## 6. Minimal worthwhile follow-up

If needed for the main paper, cache true and predicted latents from the exact same held-out windows, with model/data/source hashes. Fit PCA and all normalization on training episodes only. Compare same-capacity/same-budget position probes fit separately to true and predicted latents, report both transfer directions and copy-last baseline. Save spectrum, probe-vs-k, and normalized predictor error per principal direction as JSON/NPZ. Repeat the key off_still/off_wind class-token pair across training seeds if making a general causal claim. This would distinguish (a) information in low-variance encoder directions, (b) off-manifold prediction, and (c) genuinely unrecoverable predicted state, without modifying latent size or reward behavior.
