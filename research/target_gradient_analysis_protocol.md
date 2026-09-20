# Prospective analysis of the target-gradient experiment

This analysis was specified before the twelve full fits. The scientific design is [target_gradient_protocol.md](target_gradient_protocol.md). `studies/revision/analyze_target_gradient.py` imports only NumPy and the Python standard library and never trains a model, changes a run artifact, or selects a checkpoint.

The fixed matrix is two worlds (`gray_frame_still`, `gray_frame_wind`), model seeds 0–2 and modes `joint`, `stop_target`. Publication requires all twelve completed 8,500-update fits and their final MLP and ridge readouts. A missing, invalid, numerically failed or partial fit leaves the matrix incomplete; no subset becomes the registered comparison. The 500-update runtime fixtures under `smoke/` never enter any endpoint. The analyzer independently recomputes the campaign's timing formula and prospective midpoint decision. If runtime admission included the 4,250 checkpoint, all twelve corresponding readouts must pass too; each checkpoint must retain the original 8,500-update optimizer schedule.

The analyzer verifies the campaign source and archive pins against actual files, all seven checkpoint files and numerical parameter/state digests, fixed configurations and completed steps, initialization identity, reset-free window tables, sampled-window and update-key hashes, and complete finite training-loss traces. Paired modes must share initialization and every sampled update within a world. Same-seed initializations and update keys must match across worlds. Distinct model seeds must have distinct initializations. The two array-hash conventions used by training and probing are deliberately implemented separately. Source paths can be remapped to an identical checkout; source-content mismatches fail verification.

Readout settings must match the fixed raw 512-coordinate MLP and centered ridge protocol. The analyzer independently regenerates the exact 50,000 training and 10,000 evaluation frame IDs, reads their labels from the pinned archives, and streams the corresponding image bytes to verify frame-content hashes. Train/eval collection seeds must differ. Indices, labels and image identities must match every model and any midpoint within each world. The analyzer recomputes MAE, summed-coordinate L1, RMSE and R² from saved labels and predictions in float64, verifies native-precision saved residuals, and reconstructs the training-position-mean baseline. **Primary MAE averages over frames and both coordinates; summed L1 is twice that value.** Saved spectra support independent effective-rank and variance checks. MLP optimization must contain all 4,000 finite losses.

The primary readout is always the final MLP. Ridge is reported separately. For each seed report both within-world benefits `joint MAE − stop-target MAE`, then the relative interaction `wind benefit − still benefit`. Show all twelve raw endpoint values, all three paired values for each contrast, their equal-seed mean, sample standard deviation and range. Do not report p-values or model-family confidence intervals from three fits; the 10,000 frames are not model replications. The estimates condition on one fixed training and one fixed evaluation archive per world and on the fixed readout seed.

A positive interaction can occur because stop-target harms the still world without improving wind. The generated conclusions explicitly distinguish that case from a positive wind benefit and always report both within-world effects. A wind readout improvement does not establish control success. Wind also changes visitation and temporal complexity, so the interaction cannot uniquely identify innovation variance as the causal mechanism. Null and harmful findings remain part of the final result.

Sensitivity remains descriptive. When present, verify its exact sampled states and action-sized valid displacement, checkpoint and data identities, and source hashes. Recompute state and nuisance squared distances from the saved latents and report both means with the within-model ratio. Check matching state/background-key identities across models. A negligible denominator yields a null ratio rather than an arbitrary floor. A missing or invalid optional sensitivity is explicitly reported and its numbers omitted; it does not remove an otherwise valid primary model fit or alter the fixed comparison.

The audit does not re-execute the encoder or readout and cannot verify saved raw-feature hashes because the features themselves are not saved. A separate independent check can re-encode a fixed sample from the pinned checkpoint and verify inference/readout execution; neither this check nor its outcome may change which seeds or checkpoints are reported. Source hashes pin executable code, not the separate Markdown protocol.

Run on the machine holding the original archives and complete checkpoints to avoid transferring every model:

```bash
python -m studies.revision.analyze_target_gradient --self-test
python -m studies.revision.analyze_target_gradient \
  --runs-root /path/to/target_gradient_campaign \
  --data-root /path/to/redcube/data \
  --source-root /path/to/frozen/checkout \
  --output-dir /path/to/target_gradient_analysis \
  --require-complete
```

`--data-root` is optional if the pinned archive paths remain accessible. Outputs are `publication.json`, `table.md`, `conclusions.txt`, `paired_seed_rows.csv`, all per-update losses in `training_losses.npz`, and fixed 250-update means in `training_loss_trajectories.csv` for plotting. JSON includes every validated model's train/eval means, coordinate errors, rank/variance, checkpoint identities and any sensitivity diagnostics. There is no favorable-window or best-checkpoint smoothing/selection. An incomplete run still writes transparent status and individual validated diagnostics, leaves publication contrasts empty, and exits 2 with `--require-complete`.

The embedded synthetic checks cover MAE-versus-L1 units, raw metric consistency, wrong labels, missing artifacts, the all-twelve requirement, the interaction sign, paired-input mismatch and the two hashing conventions. Independent review should also test one real completed artifact bundle and one deliberately corrupted copy before publication.
