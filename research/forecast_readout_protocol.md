# Frozen forecast-readout diagnostic

Protocol fixed before reading its outcomes. Implementation: `studies/revision/forecast_readout.py`. No GPU launch is authorized by this document; allocation is coordinated after the primary matched-PPO/PVE queues.

## Question and scope

Does the disappearance of a wall through a decoder trained on real encodings reflect poor transfer of that decoder to predicted latents, or reduced accessibility of wall information in those predicted latents?

This is an offline scientific diagnostic. The three existing gate encoders/predictors remain frozen. Simulator state labels supervise only small readouts; they never enter a world model or policy, change its objective, choose actions, modify rewards, or rescale model latents. A failed probe cannot establish information-theoretic deletion.

## Fixed design

- Checkpoints: the existing gate `tile_factored`, `tile_factored_s1`, and `tile_factored_s2` model-probe archives. Each seed is analyzed separately.
- Data: existing `gate/train.npz` for fitting and existing separately collected `gate/eval.npz` for final evaluation. Paths, hashes, collection seeds and split metadata are checked and saved. No evaluation-based choice of hyperparameters or checkpoint is made.
- Horizons: 1, 2, 4 and 8 steps from a real encoded context. Original outgoing recorded actions drive the frozen autoregressive predictor.
- Sampling: up to 16,384 training contexts and 8,192 evaluation contexts, uniformly sampled without replacement using fixed seeds 20260919 and 20260920. Only the **past context** must not cross an episode reset. The same context IDs are used for all encoder seeds and readout domains. Availability of eight future archive frames is required, but their survival is not.
- Domains: real encoded future frame; autoregressive predicted future latent; copied current latent. The copy readout is trained to predict the future labels with the same capacity and is action independent. Each domain has the same latent width.
- Readout: one affine ridge regressor per horizon/task/domain. The objective is mean squared error plus `1e-3 * ||W||²`; the intercept is not penalized. No hyperparameter search, best-epoch selection or evaluation-dependent threshold is used.
- Preprocessing has **two analyses declared before results**: (a) a common mean/standard deviation fitted on current training latents and shared by every domain/horizon; (b) a mean/standard deviation fitted independently on each source domain's training features at each horizon. Both use a fixed `1e-6` standard-deviation floor. Source-specific normalization is fitted on all selected training features, before task-specific censoring. This is ordinary readout preprocessing only: model features, policy inputs and model outputs remain untouched. Every transfer uses exactly its source-trained scaler; there is no evaluation or target-domain recentering.

The full 3×3 matrix fits on each domain and evaluates on every domain **under both preprocessing analyses**. Neither analysis is selected according to its result. The primary comparison is real→real, forecast→forecast, real→forecast and copy→copy. A constant training-label-mean baseline is included for every task/horizon.

A shared scaler does not remove feature-scale differences. Under fixed ridge regularization, a uniformly shrunken but invertible forecast can be harder to decode without losing information. The source-specific analysis controls coordinatewise scale using only training features. Per-domain feature-standard-deviation ratios and fitted weight norms (both standardized and raw-feature coordinates) are saved. Neither analysis removes all conditioning problems: useful directions can have low covariance eigenvalues even when individual coordinates are standardized. No principal-component whitening, hand-chosen gains or test fitting is added. Any negative result remains specific to these regularized readouts and can reflect scale, conditioning, sample support or capacity.

## Labels and metrics

1. **Cube position:** two simulator top-left pixel coordinates; targets divided by 63 for fitting. Report coordinate MAE/RMSE and mean Euclidean position error in pixels.
2. **Wall presence:** binary target. Report prevalence/counts, threshold-0.5 accuracy and balanced accuracy, AUROC and average precision. The affine score is not claimed calibrated; only the optional Brier summary clips it to [0,1].
3. **Wall position:** fit and evaluate only on truth-present rows. Never regress the `NO_WALL=-99` sentinel as a physical position. Report MAE/RMSE in pixels. This metric is conditioned on true presence and must be shown alongside wall-presence metrics.
4. **Hidden original-wall gap:** a separate negative control. Select all contexts whose current wall exists but remains visually unrevealed. The label is that original wall's stored fair gap bit. Retain all such contexts regardless of later death or reset. Fit/evaluate this label separately from the physical-state probes. Under uniform-random recorded actions it is unavailable to the predictor; chance performance is expected. The real-future readout can acquire information once the gap is revealed, and reset outcomes can also be informative. Do not treat this as a requirement that the predictor forecast an unknowable future bit.

All metrics retain the number of eligible training/evaluation examples. Binary AUROC/AP are omitted when only one class is present. Tasks with fewer than 64 fit examples or no evaluation examples are explicitly marked insufficient; no alternative split or hyperparameter is selected to replace them.

## Terminal and reset handling

The archive stores the reset frame/state as the successor after a `done`; the physical terminal frame was discarded by collection. Thus a wall absent after reset cannot be interpreted as disappearance from an imagined physical continuation.

Primary cube/wall probes use a **horizon-specific continuing-episode mask**. The h=1 cohort is not conditioned on surviving eight steps. Selection fractions/counts are reported at every horizon. This is a conditional physical-forecast diagnostic, not an unconditional episodic dynamics score. The same mask and targets are applied to real, forecast, copy and mean baselines.

The hidden-gap negative control deliberately does not use this future-survival mask: selecting survivors can reveal the hidden gap through the recorded action sequence. Its past-only cohort preserves the interpretation of an unavailable hidden bit.

## Verification and retained artifacts

- Local CPU checks passed for past-only indexing, horizon-specific reset masks, the smallest valid archive, AUROC/AP tie handling, affine ridge recovery and a synthetic invertible-shrinkage control showing why the two declared normalizations are needed. A second independent static review found no indexing or hidden-gap-censoring defect. GPU extraction/reference parity remains a required smoke check.
- Unique required frames are encoded once in inference mode to reduce compute, using a fixed padded frame batch of `B * history`, where the forecast context batch is `B`. The reference check uses the same full padded context batch `B`, so its flattened encoder frame count and predictor batch shapes match. Fixed tolerances are enforced before fitting.
- Parameters and BatchNorm statistics are compared before/after extraction; any mutation aborts the run.
- The output retains model/data/source hashes, selected IDs, labels/masks, every fitted weight/intercept and its scaler, per-example evaluation predictions for both analyses/all transfer directions, and all aggregate metrics. Raw features can optionally be retained with `--save-features`.
- Start IDs and future actions are checkpoint independent. No head receives elapsed episode time, wall/gap labels, actions or any other extra input; labels are targets only. Forecasts receive the original actions through the unchanged world-model interface.
- Readout uncertainty, if summarized, must be clustered by evaluation environment stream or episode rather than treating overlapping windows as independent. Compare paired per-example errors with the saved IDs; encoder seeds remain the model replication level.

## Interpretation fixed in advance

| Result | Supported interpretation |
|---|---|
| forecast→forecast strong, real→forecast weak | Information is accessible to a readout adapted to the forecast distribution; transfer/coordinate shift explains at least part of the original decoder result. |
| forecast→forecast weak with common normalization but strong with source-specific training normalization | Coordinatewise scale and its interaction with regularization explain the weak common-normalization readout; the model was unchanged. |
| forecast→forecast weak, real→real and copy→copy strong | Forecasting makes information less accessible to this fixed regularized readout procedure. Scale, conditioning, limited data or information loss can explain this; it does not prove irreversible information loss. |
| all domain-specific probes weak | Probe capacity, sample support or encoder information may be limiting; the diagnostic does not locate the problem. |
| forecasts fail only on the hidden-gap negative control | Consistent with correctly uncertain prediction; not evidence of failing to track the predictable wall. |
| wall position strong but wall presence weak | Report both; a position score conditional on true presence does not establish that the feature identifies when a wall exists. |

A generic future-pixel decoder trained on predicted latents would be a useful follow-up if the state readouts recover wall state while the existing real-latent decoder does not. It is not needed to interpret this first matched-capacity diagnostic and is not included in its budget.

## Planned invocation

Remote Python: `/home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python`.
Remote checkout: `/home/rldoeswork/personal/tinycraft-jepa-iclr-20260919`.
Data/model root: `/home/rldoeswork/personal/stoch/gate`.

Run one process per allocated free GPU, one existing encoder seed at a time. Use `tile_factored/model_probe.npz` for seed 0 and `tile_factored_s1/model_probe.npz`, `tile_factored_s2/model_probe.npz` for seeds 1 and 2. A small smoke invocation may reduce counts/batch while retaining the same fixed protocol; its outputs are excluded from final evidence. Full runs use the stated defaults and distinct output directories.

## Prespecified descriptive context slices (declared before forecast results)

`studies/revision/analyze_forecast.py` analyzes the already saved predictions and labels. It never refits a probe, scaler, threshold or baseline; the frozen extraction/fitting script remains unchanged.

The original full cohort remains primary. Two additional physical-forecast slices address the existing-wall question directly: (1) a wall is present in the last real context frame, `context_wall_x != NO_WALL`; (2) that wall is present but still unrevealed, also `context_wall_x > REVEAL_X`. Each physical task retains its existing horizon-specific no-reset mask. Wall-position regression additionally requires the future wall to be present. These are evaluation subsets only; every probe was fitted on its original declared training examples. The constant baseline is computed once from that original training task cohort and reused unchanged in every evaluation slice.

For every encoder seed, horizon, preprocessing analysis and slice, report real→real, real→forecast, forecast→forecast and copy→copy alongside the training-mean baseline. The all-context score mixes ongoing walls and new spawns whose episode phase need not be fully observable from an eight-frame window; the slices isolate forecasts of a wall already visible in context. They do not by themselves identify a control failure.

The unrevealed-wall physical slice has a future-present wall throughout horizons 1–8 under this environment. Thus its wall-presence evaluation reports positive recall, threshold accuracy, raw score and Brier error; AUROC, AP and balanced accuracy are unavailable when there are no absent examples. A one-class AP of 1 must not be presented as successful discrimination.

The hidden-original-gap negative control remains exactly its **all-unrevealed, no-future-censoring** cohort. No physical continuation mask or extra slice is imposed on it.

Uncertainty uses 2,000 paired bootstrap draws of full evaluation environment streams, with fixed seed 20260921 and identical stream multiplicities across readouts, slices, normalizations and model seeds. This retains overlap/dependence within each stream. Intervals apply to MAE/RMSE, Euclidean error, accuracy, recall, specificity, Brier error and mean score; AUROC/AP remain explicitly point-only ranking diagnostics. Intervals are conditional on the saved model and fitted readout, not model/probe fitting uncertainty. Fewer than two nonempty streams yields no interval. Undefined statistics and valid replicate counts remain explicit.

Prespecified paired contrasts are forecast→forecast minus real→real, real→forecast minus forecast→forecast, and forecast→forecast minus copy→copy. The primary contrast metric is coordinate MAE for cube position, pixel MAE for wall position, and accuracy for wall presence/hidden gap. Difference signs are literal left-minus-right; lower errors and higher accuracies are preferable. No combined cross-model confidence interval is inferred from these three fixed encoder fits.

Analysis validates the frozen forecast-source hash, data hashes, collection-policy identity, train/evaluation separation, selected-ID hashes, matching labels/cohorts across seeds, prediction lengths/finite values, and both declared normalizations. It retains input artifact hashes, stream bootstrap multiplicities, machine-readable metrics and paired differences. It recomputes statistics from saved float32 predictions; tiny differences from the fitter's in-memory float64 summaries can occur. Every failed-probe interpretation continues to allow limited readout capacity, feature scale, covariance conditioning and sample support rather than claiming information erasure.

## Pre-fit runtime correction: fixed inference batch shapes

The first seed-0 smoke with source `f35fbb0726d87c761ae664034a9e89f0204e9ab4bcf51a816753e792c9ba05dc` aborted at its strict extraction-parity check before fitting any probe. It compared cached frame batches of 32 with a public encoder call flattened from 16 contexts × 8 frames. The encoder's BF16 arithmetic changed with batch shape despite inference mode and unchanged parameters/BatchNorm statistics. A separate numerical audit found bit-exact encoder and rollout parity at matched shapes. Original source, traceback and per-frame numerical measurements are retained under `runs/revision/forecast/encoder_parity`.

The approved correction changes only computational grouping: cache real frames in fixed padded batches `B * history` and run the public reference on `B` padded contexts, with predictor batch `B` in both paths. Real, copied-current and future features share this canonical encoder shape. No dtype, parity tolerance, data selection, objective, readout normalization or trained-model parameter changes. This correction precedes every successful probe fit.

The numerical audit is a separate inference limitation. A ratio of total rounding-error norm to wall-removal norm is descriptive and can reflect error orthogonal to the wall direction. It does not establish that rounding removes wall information or causes control failure. Task-aligned projections, when reported, are also diagnostics rather than causal evidence.
