# Frozen full-history next-target audit

This prospective diagnostic evaluates every final checkpoint from the twelve-model target-gradient study and the six-model last-only objective study. The matrix is `gray_frame_still` and `gray_frame_wind`, model seeds 0–2, and objectives `joint`, `stop_target`, and `last_only`: all 18 fixed 8,500-update checkpoints. The existing training queue and its frozen files remain untouched. No new fitting, checkpoint selection, task features, rewards, position labels or model mutation enter this audit. The position MLP remains the primary readout; ridge is reported separately.

The common endpoint is the **coordinate-mean one-step target MSE after eight observed frames**. For a window beginning at time `s`, encode observations `s` through `s+8` together in frozen inference mode. Feed embeddings `s` through `s+7` and their eight outgoing actions `a[s:s+8]` to the causal predictor. Compare its final output `pred[:,7]` with `emb[:,8]`. The other seven outputs yield a descriptive curve for one-step prediction after one through eight observed frames. This is a teacher-forced prefix-context curve, not an eight-step autoregressive forecast.

Each world uses exactly 8,192 distinct valid windows from its original `/dev/shm/redcube/<world>/eval.npz`. Regenerate the complete `train.valid_starts(dones,9)` table: each window must contain no reset in its eight intervening transitions. Draw table indices once with `numpy.Generator(numpy.PCG64(20260919)).choice(len(table),8192,replace=False)`, preserving draw order. Every model and objective in that world receives these identical indices. Fewer than 8,192 valid starts is an explicit failure; there is no smaller sample, replacement sampling or model-specific fallback. Different windows may overlap and are not independent experimental replications. Seeds refer to trained models, not windows.

Before the first last-only fit or inspection of its outcomes, execute the CPU-only `--pin-windows` command against the original archives. It reads metadata and episode boundaries, stores exact indices for both worlds, and pins the sampler implementation/protocol, archive and index-array hashes. This complete pin is a required input to the final audit. Regenerated samples must equal every stored index array and all recorded hashes; changing the pin, source or archive fails evaluation. The launch manifest must retain the prospective pin identity and creation time. No model or score is loaded during pinning.

Inference uses exactly 64 batches of 128 complete nine-frame windows. The encoder sees `[128,9,63,63,3]`, corresponding to 1,152 flattened frames, matching the production training input shape. The full window is encoded on every batch. No alternative cache batch shape, padding, separate target encoding, latent tiling, amplification or representation scaling is permitted. Encoder and predictor both run with `train=False`; stored BatchNorm statistics are used and dropout is deterministic. The forward pass retains the trained precision. Compute each window's squared error mean over the 512 unmodified latent coordinates in NumPy float64 after retrieving the float32 prediction and target. Average windows equally for the endpoint and each prefix position. Training losses used float32 reductions, so tiny accumulation-roundoff differences are possible.

The script saves all 8,192 × 8 per-window prefix errors, sampled table indices, start/environment pairs and frame-time indices. It also saves every 512-coordinate endpoint prediction and target as float32: approximately 32 MiB per model, or 576 MiB for the 18 models before ancillary arrays. Earlier-prefix latent vectors are not saved. Endpoint errors and geometry can therefore be independently reconstructed from complete endpoint latent arrays, while earlier-prefix results are retained as per-window errors. Each output archive has a SHA-256 identity and array-level hashes covering dtype, shape and bytes. Full window image/action batch hashes demonstrate identical executed inputs across the nine models within each world.

Descriptive endpoint target geometry includes the coordinate population variances (`ddof=0`), their mean and sum, the target mean and squared energy, and the complete population covariance eigenvalue spectrum. Effective rank is `exp(entropy(eigenvalues / sum(eigenvalues)))`; it is null for a zero-variance target. A variance-normalized MSE divides the endpoint coordinate-mean MSE by the mean target coordinate variance, with null for a zero denominator. This normalization changes no model output, evaluation error, saved feature or primary readout. Both raw and normalized errors are representation-relative diagnostics. **Lower MSE can result from losing information or collapsing the target. Neither measure is a model-quality, information-preservation or control-success test, and neither licenses ranking independently learned representations by quality.** The curve compares different target positions as well as different context lengths; it is not an isolated intervention on history length.

Before any model inference, require the complete 18-model matrix, exact production configurations (including latent dimension 512, history 8, `num_preds=1`, four actions, batch 128, and 8,500 updates), completed fit manifests, final checkpoint file hashes, recorded numerical weight/state digests, objective identities and training source identities. Within each world and seed, all three modes must share the training archive, initialization, sampled training windows, update keys and valid-start table identities. Pin the original evaluation archive hashes from the target-gradient campaign manifest. Record the audit code/protocol, trainer/model source hashes, every actual input archive hash, package versions, precision configuration and device identities. The audit must run after the target-gradient campaign completes so its manifest stays unchanged.

Check restored parameter/BatchNorm hashes against the training manifests. Hash the full NNX numerical state, including typed RNG leaves, before and after every model's inference; any mutation fails the audit. Recheck source, checkpoint, fit manifest, campaign manifest and evaluation archive identities at the end. No successful report is produced unless all 18 models complete and every prescribed identity check passes. A runtime failure preserves partial diagnostic artifacts with `failed_incomplete`; partial results do not constitute the registered comparison, and the script never silently retries or excludes a failed case. Existing nonempty output directories are rejected.

CPU validation exercises explicit reset-boundary enumeration, insufficient-window rejection, deterministic sampling without replacement, outgoing-action indexing, eight-prefix and last-only MSE normalization, zero-variance geometry, full-state immutability and a small instance of the actual tile/factored model. The model test independently repeats the `train.py` forward indices with inference flags and compares the final output to frame eight. It cannot establish GPU backend equivalence; production batch shapes are fixed to avoid a known source of BF16 numerical discrepancies.

```bash
JAX_PLATFORMS=cpu /private/tmp/tinywm-cpu-checks-20260916/bin/python \
  -m studies.revision.full_history_audit --self-test

# Run on the archive host before starting any last-only fit.
JAX_PLATFORMS=cpu python -m studies.revision.full_history_audit --pin-windows \
  --data-root /dev/shm/redcube --out /path/to/new_full_history_window_pin

# Run only after independent source review and all 18 final fits are complete.
python -m studies.revision.full_history_audit \
  --target-root /path/to/target_gradient_campaign \
  --last-only-root /path/to/last_only_campaign \
  --data-root /dev/shm/redcube \
  --window-pin /path/to/new_full_history_window_pin \
  --out /path/to/new_full_history_audit

# Optional runtime admission, using a fixed 500-update timing fixture or final model.
# This computes no score and cannot generate publication results.
python -m studies.revision.full_history_audit --smoke \
  --model /path/to/smoke/joint/model.npz \
  --eval-data /dev/shm/redcube/gray_frame_still/eval.npz \
  --world gray_frame_still --out /path/to/new_full_history_timing
```

The optional timing smoke uses the first two complete 128-window batches of the same 8,192-window draw and exact production architecture. It records compilation and warm forward time, finite-output checks and frozen-state identities. It neither computes nor reports prediction errors. Its 64-batch runtime projection explicitly excludes loading, hashing, host window gathering, geometry and artifact writing; admission must budget those separately. Its report is always marked `publication_eligible=false` and cannot enter the common 18-model result matrix.

The publication implementation exposes no window-count, batch-shape, metric, seed, checkpoint or tolerance overrides. It launches no training and never modifies model/checkpoint artifacts. No GPU execution was needed to design or run the synthetic CPU checks.
