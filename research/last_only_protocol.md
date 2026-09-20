# Last-only prediction objective: prospective six-fit extension

Status: design and CPU validation only; no GPU launch is authorized by this document. The existing twelve-fit `paired_target_gradient_v1` study, its sources, matrix, endpoint and analysis remain frozen. This document registers a separate `last_only_objective_v1` experiment before any of its six fitted-model outcomes. Admission and resource scheduling are decided by the parent campaign after independent review.

## Question and limits

Does scoring only the final prediction of a fixed eight-frame sequence change the position information accessible from the learned representation, relative to the original joint-gradient objective that scores all eight causal prefix predictions?

The original objective includes a first prediction with only one frame in its nominal causal attention context. That frame does not explicitly reveal wind. For an ordinary reset-free transition in the cube world, two position observations and the intervening action identify the current wind through `wind_t = position_t − position_(t−1) − move(action_(t−1))`. The next wind still contains fresh randomness: the increment is −1, 0 or +1 with probabilities 0.1, 0.8 and 0.1, clipped to [−2,2]. Its conditional variance is 0.09 at a clipping boundary and 0.2 elsewhere. Additional context therefore does not make the next transition deterministic.

This motivation does not turn last-only into an isolated test of conditional-history causality. Last-only removes seven directly supervised targets, changes the supervised target-position distribution, changes their averaging and stochastic-gradient variance, and gives eight times fewer scored residuals per fixed number of updates. No factor of eight is applied to compensate. Moreover, the encoder projector's training BatchNorm pools the nine frames and tokens, while the predictor projector pools all eight output positions. Thus even the first nominal causal output is not strictly a function of only one frame during training. Retaining the entire encoding and prediction calls preserves those BatchNorm computations in this comparison.

A benefit, null result or harm supports a statement about this complete objective package at the registered budget. It cannot establish that short histories alone cause failure, that stochastic innovations alone cause failure, or that the change is a general remedy for JEPA. Latent MSE remains descriptive because representations themselves change.

## Frozen matrix and intervention

Run exactly six fresh fits, with no warm start:

| World | Model/data-order seeds | Mode | Updates |
| --- | --- | --- | --- |
| `gray_frame_still` | 0, 1, 2 | `last_only` | 8,500 each |
| `gray_frame_wind` | 0, 1, 2 | `last_only` | 8,500 each |

For every world and seed, use the identical existing 500,000-transition training archive used by its frozen `joint` and `stop_target` pair. Archive SHA-256 must match the paired baseline manifests exactly, not only the filename or world label. No recollection, filtering by outcome, dataset substitution or seed replacement is permitted. The driver rejects any archive with a transition count different from `cfg.data_budget`; the production CLI fixes this to 500,000.

`studies/revision/last_only_objective.py` is a separate full copy of `target_gradient.py`, with no imported mutable globals, monkeypatching or modifications to the frozen driver. `production_config(seed)` is identical: tile encoder width 128/depth 6/heads 8, sixteen 32-channel tokens (512 latent coordinates), factored predictor width 256/depth 6/heads 8, history 8 and one-step targets; batch 128; AdamW learning rate 0.001, weight decay 0.001, 10% warm-up followed by cosine decay to zero; global-gradient clipping 1.0; and 8,500 updates. All projector defaults and precision settings remain the same. SIGReg keeps weight 0.09, 17 knots and 1,024 directions.

For a nine-frame window, the unchanged forward calls and new loss are:

```python
emb = model.encode(window, True)                 # all nine frames
act_emb = model.act_embed(window_actions)
pred = model.predict(emb[:, :8], act_emb[:, :8], True)  # all eight outputs
prediction_loss = mean((pred[:, -1] - emb[:, -1]) ** 2)
regularizer = sigreg(emb.transpose(1, 0, 2), k_sig, 17, 1024)
loss = prediction_loss + 0.09 * regularizer
```

The prediction mean remains a mean over examples and all 512 coordinates, with no coordinate amplification, feature replication or history-length scale factor. Both prediction-side and target-side gradients remain active. Prior frames can still receive prediction gradients through the input path, attention and BatchNorm; removing their direct residuals does not freeze their encodings. SIGReg differentiates all nine embeddings. There is one shared encoder, with no EMA teacher, latent enlargement, reward-dependent term, privileged-state loss or environment-specific training scaffold.

Initialization seed, dropout seed, update keys, sampled reset-free windows, SIGReg projection keys, optimizer schedule and update count must match each frozen joint pair. After objective-dependent updates, weights and BatchNorm statistics may diverge. Store archive and array hashes; configuration; parameter and BatchNorm initialization hash; sampled window table and indices; update keys; source hashes including the reference driver; package versions; device/precision settings; complete loss history; and checkpoint hashes. Save updates 0, 500, 1000, 2000, 4000, 4250 and 8500. Update 8500 is the only primary checkpoint. Earlier checkpoints are descriptive, never candidates for favorable selection.

The run manifest has `protocol="last_only_objective_v1"` and `mode="last_only"`. Checkpoints record `experiment="last_only_objective_v1"`, `prediction_loss_mode="last_only"` and `target_gradient_mode="joint"`. Distinct names prevent this intervention from being mistaken for another member of the twelve-fit target-gradient experiment. The logged loss column is `last_only_prediction_mse`, not the baseline all-prefix MSE.

## Common evaluation of all eighteen models

Evaluate every final checkpoint: the existing twelve `joint`/`stop_target` fits and all six new `last_only` fits. Retain all outcomes, including failed fits or readouts. Do not compare their logged training prediction losses as evidence of improvement: those losses score different target sets.

1. **Primary accessibility endpoint:** use the already fixed nonlinear position readout from `target_gradient_protocol.md` for each frozen encoder: identical 50,000 train and 10,000 evaluation frames, raw 512-dimensional latents, 512→256→256→2 GELU MLP, 4,000 updates and its fixed initialization/minibatch seed. Report coordinate-mean MAE in pixels. Preserve the fixed ridge readout, its covariance-scaled global penalty, RMSE/R², per-coordinate errors, baseline, variance and effective-rank diagnostics. Do not rescale or augment latents. Existing saved readouts may be reused only when checkpoint/archive/source/sample/config identities satisfy the same fixed protocol.
2. **Common predictive diagnostics:** pin one shared set of 8,192 held-out reset-free nine-frame windows from the independent evaluation archive for each world before examining last-only results. Use identical window IDs and recorded actions for all nine models of that world, with canonical batches of 128 nine-frame windows. Encode all nine frames and predict all eight outputs in inference mode (`train=False`), without updating parameters or BatchNorm. Report per-prefix coordinate-mean MSE, the mean over all eight prefixes, and the final-position MSE conditioned on the full nominal eight-frame history. Report SIGReg separately if collected. The same calculations apply to every mode. Hash archive, window IDs, source and checkpoint, and verify parameter/BatchNorm hashes remain unchanged by evaluation. These are comparable evaluation definitions, but still representation-dependent descriptive errors, not a common physical scale or proof of information preservation.

Use the separate `studies/revision/last_only_probe.py` for the six new checkpoints. It is a full copy of the frozen `target_gradient_probe.py`, with only its module documentation and the two protocol-label/primary-contrast strings changed. All numerical and source-provenance functions are AST-identical; its main function is AST-identical after normalizing those two metadata strings. It preserves the raw 512-dimensional and exact `data_sha256` contract and records `protocol="last_only_readouts_v1"` with the correct within-world joint-minus-last-only contrast. No original probe metadata is relabeled or overwritten. The frozen `analyze_target_gradient.py` is not a last-only analyzer and must not silently ingest the new experiment.

For each seed and world, the primary paired contrast is:

`last_only_benefit(world, seed) = MAE(joint, world, seed) − MAE(last_only, world, seed)`.

Report all six new raw endpoint values with the twelve reference values; all three within-world paired benefits; their mean and spread; and the descriptive wind-minus-still difference of benefits. A positive interaction does not imply an absolute improvement in wind, since it can arise from harm in still. Show absolute within-world results. The comparison with `stop_target` is secondary and displayed for all seeds, not selected after viewing outcomes. Training seeds are the replication units; thousands of sampled frames do not create more independent trained models. Three seeds limit inferential strength. Wind changes state visitation and episode-length distribution in addition to transition uncertainty, so cross-world differences are not a unique causal isolation.

## Admission, reproducibility and reporting

Before admitting the six fits, require independent source review and CPU numerical checks:

- The copied full forward/update loop, configuration, sampling and optimizer computations match the joint reference; only `prediction_mse` selects the final output and target. All original reference source hashes remain unchanged.
- Independent-array derivatives are zero at the first seven direct residual positions and equal/opposite `2 * residual / (batch * latent_dim)` at the final prediction/target positions. The final target derivative is nonzero. The all-nine-frame SIGReg gradient is unchanged and nonzero on every frame in the fixture.
- A tiny two-update tile/factored CPU pair has identical initialization/data/window/update/sample hashes and equal SIGReg values through the first nonzero update. The learning rate at the first step is zero; its complete parameter/BatchNorm hash remains paired. The second step changes weights, demonstrating an active intervention. Both runs keep history 8 and512 latent coordinates; only other computational dimensions and the fixture data budget are reduced.
- Checkpoint 0/intermediate/final files reload to their saved complete weight hashes and preserve the 512-dimensional/data-SHA readout contract. The new driver rejects undersized data and nonempty outputs.

The six-fit decision must use available GPU time and timing estimates only, before their fitted-model outcomes. Budget for all six 8,500-step fits and common evaluation, with a deadline margin. If the complete matrix is not admitted, report it as not run. If an admitted job later fails or the deadline interrupts it, retain and report that outcome and partial artifacts as incomplete; do not substitute an earlier checkpoint or replace its seed. An optional timing fixture must be separated from registered fits and never reported as a scientific endpoint.

There is no extra last-k, history-truncation, loss-weight, latent-size or target-selection search under this protocol. No downstream outcome may trigger a revised matrix or selective reporting. Any later intervention requires a new named protocol and is separate evidence.

Validation record (2026-09-19): independent CPU objective/config/SIGReg checks pass. A separate end-to-end two-update CPU comparison passes all checks above, using history 8 and 512 latents. Paired SIGReg values are exactly `[1.7310367822647095, 1.1183913946151733]`; paired initialization, archive/array, window table, update-key and sampled-window hashes match; complete step 1 weights/BatchNorm match; step 2 weights differ; and every saved checkpoint reloads exactly. Tests used `JAX_PLATFORMS=cpu /private/tmp/tinywm-cpu-checks-20260916/bin/python`. No production fit has been launched by the implementation agent.

Reviewed implementation hashes: training driver `b5316f8997cbbae56772520b0232d4f05c35be271a29b8f694d53ad650da64e1`; reference driver `4dc9280ea077e954096bc7968f3ab19031f82369124b4de62acf5e5e6a14d488`; separate last-only probe `868396f01a8b38c792ee3b13693313ba0b58ceb6e98039765f7b6b4d908f78b6`. The training driver received an independent GREEN source/objective review. Final independent review of this protocol and the probe copy is GREEN. The reviewer verified whole-module probe AST identity after removing documentation and normalizing exactly two metadata constants, plus numerical sampling/ridge/covariance/prediction/metric/hash/split-guard parity. Before any new fitted results are inspected, the common evaluator must additionally pin its exact sampling RNG/seed/replacement rule and the resulting 8,192 window IDs; this is an execution prerequisite, not an outcome-dependent choice.
