# Forecast readout results: three fixed encoder seeds

## Verdict

**The diagnostic does not support a general claim that predicted latents erase the wall.** In encoder seed 2, fitting the same small readout on predicted features substantially improves wall-position MAE relative to transferring a real-feature readout. This improvement also occurs in the prespecified slice where a wall is already present and unrevealed in context. However, large squared errors and weak within-slice tracking prevent a claim of reliable wall-state recovery. Seeds 0 and 1 have weak linear readouts even on real future encodings, so their negative forecast-probe results do not locate an information loss.

The warranted distinction is between **a wall disappearing through the original real-latent decoder** and **wall information being unavailable in the predicted representation**. The first was observed; the second is not established. Seed 2 has a coarse predictive association and a strong readout-transfer effect, with substantial limitations described below.

## Completed and verified protocol

All three existing model checkpoints completed the fixed protocol: 16,384 train contexts, 8,192 contexts from an independently collected evaluation archive, horizons 1/2/4/8, both declared train-only normalization analyses, full 3×3 readout transfer, and unchanged mean/copy baselines. Every run verified unchanged model parameters/BatchNorm statistics, identical sampled IDs/labels across seeds, and exact-zero cached/public inference parity at the production batch shape. All common/copy and per-source/copy prediction arrays are identical. No world-model or policy parameter was trained or modified.

Per-model elapsed times were 138.2, 132.9 and 136.4 seconds. Seed 2 emitted a CUDA timing-accuracy warning and subsequently completed with exit code 0 and all invariants satisfied. The earlier batching audit and computational-only correction are documented separately in `research/forecast_numerics.md`; smoke fits are excluded from these results.

Final analysis: `runs/revision/forecast/analysis`, containing 1,080 metric rows, 720 paired contrasts, data/source/artifact identity checks and 2,000 paired evaluation-stream bootstrap draws. These intervals are conditional on each fitted model and readout; they do not quantify model-training uncertainty.

## Primary full-cohort wall-position results

Pixel MAE on horizon-specific continuing episodes whose future frame contains a wall. The table uses the declared source-specific training scaler; the common-scaler analysis is retained in full and gives the same qualitative interpretation. R→F applies a readout trained on real future encodings to forecast features; F→F fits and evaluates on forecast features. Copy→copy uses the current real latent without future actions.

| Encoder | Horizon | Mean baseline | R→R | R→F | F→F | Copy→copy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11.15 | 11.82 | 13.07 | 11.93 | 11.68 |
| 1 | 1 | 11.15 | 11.54 | 12.37 | 11.74 | 11.40 |
| 2 | 1 | 11.15 | 7.61 | 9.64 | 9.51 | 9.27 |
| 0 | 8 | 10.77 | 10.87 | 12.62 | 11.34 | 11.02 |
| 1 | 8 | 10.77 | 11.38 | 12.30 | 11.66 | 11.40 |
| 2 | 8 | 10.77 | 7.40 | 16.47 | 9.11 | 8.94 |

There are 3,038 evaluated wall-present examples at horizon 1 and 3,173 at horizon 8. The original all-context sample has 8,192 starts; continuation counts are 7,846, 7,552, 6,863 and 5,555 at horizons 1/2/4/8. These are horizon-specific conditional forecasts, not prediction through resets.

For seed 2 at horizon 8, adapting the readout reduces MAE by 7.36 pixels relative to R→F, with paired evaluation-stream interval [6.77, 7.96]. F→F remains worse than R→R by 1.72 pixels [1.25, 2.14]; its difference from copy→copy is 0.17 [−0.60, 0.93], which is not an equivalence test. Adaptation improves this measured readout error; it does not establish an accurate or improved general-purpose dynamics model.

The predeclared squared-error metric materially limits this interpretation: seed-2 F→F wall-position RMSE is 18.65 pixels at horizon 8, worse than the fixed mean baseline's 13.49; R→R RMSE is also high at 19.17. MAE improvement must therefore be reported alongside large errors in the tails: maximum F→F absolute error is 675.54 pixels. A post-hoc interpretation audit of saved predictions finds full-cohort F→F position correlation +0.246, with evaluation-stream interval [0.142, 0.453], indicating a coarse statistical association rather than reliable position recovery. This additional statistic was not used to select, calibrate or refit a probe, and observational correlations do not identify which physical state information the model uses.

Seeds 0 and 1 do not beat the constant wall-position baseline at the displayed endpoints even in the R→R positive control. Their F→F performance cannot be interpreted as evidence of wall erasure. The probe class, regularization, covariance conditioning and finite data remain possible limitations.

## Prespecified existing-wall slices

Seed 2, horizon 8, pixel MAE; all rows retain the same originally fitted probes and original training-mean baseline.

| Evaluation cohort | n | Normalization | Mean | R→R | R→F | F→F | Copy |
|---|---:|---|---:|---:|---:|---:|---:|
| Any continuing future with wall present | 3,173 | common | 10.77 | 7.42 | 16.13 | 9.12 | 8.94 |
| Any continuing future with wall present | 3,173 | per-source | 10.77 | 7.40 | 16.47 | 9.11 | 8.94 |
| Wall present in context and future | 1,013 | common | 16.64 | 9.80 | 14.80 | 10.32 | 11.70 |
| Wall present in context and future | 1,013 | per-source | 16.64 | 9.71 | 14.81 | 10.32 | 11.70 |
| Unrevealed wall in context, continuing | 716 | common | 14.24 | 6.62 | 10.05 | 6.16 | 6.79 |
| Unrevealed wall in context, continuing | 716 | per-source | 14.24 | 6.53 | 10.02 | 6.16 | 6.79 |

In the unrevealed-wall slice, the source-normalized adaptation gain R→F minus F→F is 3.85 pixels [3.48, 4.22]. Its difference from copy→copy is −0.63 [−1.41, 0.13]; no equivalence is established. The lower MAE is a literal transfer-error improvement, not proof of tracking position variations within this slice. The post-hoc correlation is −0.175 [−0.255, −0.094] there: true positions average 29.52 pixels and forecasts average 33.10. Moving a biased readout toward the slice's typical position can improve its MAE against the unchanged global baseline without recovering detailed position variation. The original decoder and reward/termination heads were not modified or reevaluated by this probe.

The unrevealed-wall slice contains no absent walls through horizon 8, so its wall-presence AUROC, AP and balanced accuracy are unavailable. High positive recall there cannot by itself establish discrimination; the analysis retains this explicitly rather than treating a one-class AP as success.

## Other diagnostics

Wall-presence discrimination is modest in seeds 0/1. At horizon 8, R→R AUROC is 0.574/0.545 and F→F 0.523/0.518. Seed 2 has R→R AUROC 0.734, R→F 0.406 and F→F 0.615, again showing a transfer mismatch with incomplete adapted recovery. These are point-only ranking diagnostics.

Cube-position readouts also remain coarse. At horizon 8, source-normalized coordinate MAE for R→R versus F→F is 7.82 versus 8.49 pixels (seed 0), 7.97 versus 8.50 (seed 1), and 8.26 versus 8.23 (seed 2). Forecast-minus-real paired differences are 0.67 [0.52, 0.81], 0.52 [0.35, 0.70] and −0.04 [−0.20, 0.11]. These support a readout-specific forecast deficit in two fitted encoders, not a family-level control conclusion.

The hidden-original-gap control retains all 1,947 currently unrevealed-wall evaluation contexts, regardless of later termination/reset. Forecast-readout accuracy ranges from 48.43% to 51.10% over the three seeds, four horizons and two normalization analyses; every evaluation-stream interval includes 50%. It supplies no evidence that the predictor can infer the unavailable original gap. Real-future linear probes are also weak on this control, so this is a pipeline sanity check rather than a positive demonstration of optimal uncertainty modeling.

## Manuscript repair

Recommended wording:

> A decoder trained on real encodings loses the wall when applied to predicted latents. This visual disappearance does not certify loss of wall information: in one of three fitted encoders, a matched readout trained on forecast features substantially reduces wall-position MAE relative to transferring a real-feature readout. Large squared errors and weak within-slice tracking limit that recovery, while linear positive controls are weak in the other two encoders. The control deficit therefore remains a failure of the tested prediction, readout and policy-learning pipeline, with no identified information-erasure mechanism.

Avoid “the predictor erases the wall,” “the latent contains no wall,” “distributional prediction is necessary,” or treating the adapted probe as a deployable policy. It uses supervised simulator labels solely as an offline diagnostic, and no such labels enter the model, controller or reward.

All horizons, normalizations, model seeds and descriptive slices remain available in the machine-readable analysis. The highlighted seed-2 result establishes a readout-transfer effect; it does not replace the full three-seed record, demonstrate reliable position tracking, or imply every checkpoint behaves similarly. The extra correlation audit is explicitly labeled post hoc in `runs/revision/forecast/analysis/correlation_audit.json`, with its calculation script and input hashes retained.
