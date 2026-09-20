# Fixed execution verification of target-gradient readouts

This check supplements the independent artifact/metric audit in `analyze_target_gradient.py`. It may run only after that audit admits the complete twelve-fit matrix. It covers every final checkpoint and, if prospectively admitted and fully analyzed, every 4,250-update checkpoint. No result, seed, checkpoint or readout is selected by its score. No fit is repeated or updated.

For every checkpoint, verify rows 0–31 of the fixed training sample and rows 0–31 of the fixed evaluation sample. Regenerate the original full 50,000/10,000 index draws before slicing; drawing only 32 would change the sample. Load the original first 256 sampled images and apply frozen inference with the original `[1,256,63,63,3]` encoder input shape. Compare the first 32 returned raw 512-coordinate latents through both serialized readouts with the saved predictions. The host image archives are loaded once per world and reused across models, requiring approximately 6.3 GB for the existing training/evaluation pair. No position, reward, action, done or privileged-state array is accessed.

The independent ridge forward pass is NumPy float64 matrix multiplication plus the saved intercept. The independent MLP forward pass uses the saved float32 kernels/biases, two JAX approximate GELUs and the final factor 63. Its first 32 raw inputs are padded to the original 1,024-row prediction batch shape, then padding is discarded. Padding does not normalize, amplify, repeat coordinates or change any checked latent. The rowwise readout has no batch-dependent operations.

Predeclared acceptance is coordinatewise `abs(reexecuted − saved) <= atol + rtol * abs(saved)`. Ridge uses `atol=1e-6` pixels and `rtol=1e-7`; MLP uses `atol=2e-4` pixels and `rtol=2e-6`. These tight subpixel thresholds accommodate floating-point reductions, while preserving the original computational shapes avoids granting a broad BF16 discrepancy allowance. They are acceptance thresholds, not proven backend-independent error bounds. Record device identities, exact package versions and JAX precision configuration. A failed comparison remains failed and is investigated; the script exposes no tolerance override or retry selection.

Before execution, require the unchanged completed campaign and publication JSON, exact frozen producer/analyzer source hashes, exact checkpoint and all probe-artifact hashes, and actual input archive hashes. The imported `train.py` must come from the specified frozen checkout. The restored Params and BatchStats digest must equal the audited checkpoint digest, including array shapes, dtypes, numerical leaf order and bytes. Hash them again after both splits and require exact equality. Recheck checkpoint/readout files and source/archive identities after execution.

The pass/fail JSON contains all prescribed cases, checked frame IDs, subset feature/prediction digests, signed prediction differences, maximum/RMS numerical discrepancies, every fixed tolerance, and before/after frozen-state identities. It contains no recomputed task score and never refits either readout. Individual failures do not remove subsequent registered cases. Missing/changed artifacts or an incomplete case set prevent a pass. The command exits 2 on failure and refuses to overwrite an existing report.

```bash
python -m studies.revision.verify_target_gradient_execution --self-test
python -m studies.revision.verify_target_gradient_execution \
  --analysis /path/to/target_gradient_analysis/publication.json \
  --runs-root /path/to/target_gradient_campaign \
  --data-root /dev/shm/redcube \
  --source-root /path/to/frozen/checkout \
  --output /path/to/target_gradient_analysis/execution_verification.json
```

The self-test imports NumPy only and checks ridge arithmetic, deliberate prediction mismatches, padding invariance, the full-draw sampling rule, fixed 12/24-checkpoint coverage and incomplete-matrix rejection. It does not launch GPU work.

A pass independently checks serialized encoder/readout execution on the fixed subset. It does not recreate readout training or prove all unsaved feature bytes and all prediction rows correct. The full-archive metric audit and this subset execution check have distinct scopes; report both without claiming universal execution verification.
