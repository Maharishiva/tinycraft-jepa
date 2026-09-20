# Fresh longer-budget joint-JEPA sensitivity

This is a prospective training-budget sensitivity for the existing RedCube representation study. It asks whether the original joint objective's position readout improves under a longer training schedule. It is not a new objective, a reward intervention, a continuation of a saved model, or evidence by itself about control or general intelligence. This preparation does not authorize immediate GPU execution.

The complete block is **three fresh fits**, model seeds **0, 1, 2**, on the original `gray_frame_wind` training archive. Each fit uses **34,000 updates**, exactly four times 8,500. `longer_joint.production_config(seed)` equals the frozen `target_gradient.production_config(seed)` in every field except `steps`. Thus the model remains the original 4×4×32 spatial representation, 512 latent coordinates, tile encoder with width 128, depth 6 and eight heads, factored predictor with width 256, depth 6 and eight heads, history 8, one-step shift, batch 128, LR 0.001, AdamW weight decay 0.001, global-gradient clipping 1.0, original SIGReg weight 0.09/17 knots/1,024 projections, and original precision. All 500,000 archived transitions remain eligible subject to the same reset-free nine-frame window rule. The trainer reads only frames, actions and episode boundaries for computation. Archive metadata validates file identity; positions, rewards and simulator state do not enter training.

The update is the independently tested joint kernel from frozen `studies/revision/target_gradient.py`, SHA256 `4dc9280ea077e954096bc7968f3ab19031f82369124b4de62acf5e5e6a14d488`. The longer driver refuses different source bytes. Prediction MSE and SIGReg remain numerically the same scalar objectives; both differentiate the shared encoder exactly as before. No latent scaling, duplication, teacher/EMA, auxiliary decoder, position loss, reward model or task-specific action selection is added.

## Fresh optimization and the changed schedule

Each seed initializes a fresh model and fresh AdamW state. Existing model checkpoints omit optimizer state and are unsuitable for exact continuation. Training uses the original schedule expression `warmup_cosine_decay_schedule(0,lr,int(warmup*steps),steps,end_value=0)`. The warmup therefore becomes **3,400 updates**, compared with 850, and cosine decay ends at 34,000 rather than 8,500. Optimizer counter 0 has LR 0 in both schedules; counter 1 is already different. The longer run's checkpoint at 8,500 is **not the original 8,500-step baseline**, even when initialization and all first 8,500 sampled updates match.

The registered comparison changes both total updates and the schedule duration together. It cannot isolate update count from the learning-rate schedule. A beneficial result is consistent with optimization-budget sensitivity; a null result under this one fixed schedule is not a universal convergence guarantee. This distinction must accompany the figure/table.

## Actual RNG pairing, not assumed prefix stability

The longer run uses the unmodified original rule `jax.random.split(jax.random.PRNGKey(seed+7),34000)`. It never concatenates streams, replaces prefixes or changes keys to obtain a desired outcome. The shorter rule is independently regenerated at length 8,500, and both actual key arrays and the resulting minibatch table indices are compared. Production also compares against each same-seed original joint fit's saved `sampling.npz`, not merely a newly regenerated short stream.

On the preparation CPU runtime, **JAX 0.10.1**, `threefry2x32`, `jax_threefry_partitionable=True`, all three seeds have exact prefix equality. The first 8,500 update-key hashes, using the training driver's shape/dtype/content hash, were:

| Model seed | Original 8,500 and longer first 8,500 key SHA256 |
|---|---|
|0|`1d18116a8bb81d7addb67c33c76d27aa6d04aa4cbfc973c642685d973fc6b0be`|
|1|`3dfaf32b98216f240b47ea3110967268dc464abb53b11f4d819945653a181d6c`|
|2|`d13c1e9123e28ce6a7c73ef585ea6095be952fc7f3ee4010a7c1619b4ad54981`|

These are CPU observations, not a claim about an untested GPU runtime or every JAX configuration. Before any production update, verify the original data-file/array hashes, window table, full initial Params/BatchStats digest, short keys and samples, and longer-prefix equality. If the execution runtime fails those checks, stop before training and report the mismatch. Do not silently switch RNG rules or describe unmatched streams as paired. The manifest records actual full/prefix hashes, JAX/PRNG configuration and comparison results. `sampling.npz` retains the entire 34,000-step key array, every sampled table index, the valid-start table, and all 34,000 optimizer learning rates.

## Fixed artifacts and readouts

Save checkpoints at **0, 8,500, 17,000, 25,500, 34,000**, with full per-update prediction-MSE and SIGReg history. The final checkpoint is `model.npz`; other files are `model_step000000.npz`, `model_step008500.npz`, `model_step017000.npz`, and `model_step025500.npz`. They remain compatible with `train.load_model`. The zero checkpoint must match the original same-seed initialization exactly. No checkpoint is chosen by any result.

Only the **final 34,000 checkpoint of every seed** receives the registered position MLP and ridge readout. Earlier checkpoints/losses are descriptive trajectory artifacts; no additional readout or selection is planned. Use the unchanged numerical procedure from `target_gradient_probe.py`, with `--encode-batch 256` specified explicitly. Existing metadata checks accept steps 34,000, require 512 coordinates and exact training-data SHA. Fixed training/evaluation samples are 50,000/10,000 frames, selected without replacement by NumPy PCG64 seeds 19417/19418. The MLP is raw 512→256→256→2, GELU, seed 0, 4,000 updates, batch 1,024, AdamW LR 0.001/weight decay 0.0001. Ridge uses raw centered features, an unpenalized intercept, and penalty 0.001 times mean feature variance. No readout is chosen by evaluation accuracy.

The existing numerical fitter's `protocol.json` contains the old target-gradient campaign label and interaction description. Preserve that raw artifact as provenance, but require a new companion `longer_joint_readout_context.json` for each result. It must state this protocol ID, seed, completed step 34,000, checkpoint SHA, raw probe `protocol.json` SHA, explicit encode batch 256, same-seed 8,500 baseline checkpoint/readout hashes, and the **budget contrast**, not the old stop-target interaction. Its use of the shared numerical fitter is explicit. No old scientific result file is edited.

Primary endpoint: held-out coordinate-mean position MAE, `results.json["eval"]["mlp"]["mae_px"]`. Ridge MAE is reported separately. For each seed show original 8,500 joint MAE and longer 34,000 joint MAE, then the paired difference `MAE8500−MAE34000` (positive means lower error with the longer schedule). Report all three paired values, their equal-seed mean, sample SD and range. No p-values or model-population confidence intervals from three seeds; evaluation frames are not model replications. Report a null or harmful result identically. This conditions on the one fixed wind train/evaluation archive and one fixed readout seed.

Existing all-twelve/8,500-step analyzers and execution verifiers **do not validate this new three-fit study**. Before publication, a separate audit must verify all three completed 34,000 fits, exact source/data/checkpoint/config identities, unchanged initialization and sampled prefixes, finite 34,000×2 loss arrays, all five checkpoint hashes, fixed readout sample/frame/label identities and all 4,000 MLP losses. Independently recompute MAE from saved predictions and labels. Verify the frozen encoder/readout execution on the same fixed 32-frame subsets and original 256-frame computational shapes; compare numerical state hashes before/after inference. The original verifier may inform a separate copy but must not be weakened in place. An incomplete or invalid block is not the registered comparison.

## Priority, runtime admission and deadline

The **entire three-fit block plus all three final readouts and artifact checks** may launch only after the priority shared-CNN **30-job** block is complete and an exclusive GPU is available. No existing queue is interrupted. Admission is runtime-only, before starting any of these fits or inspecting their results. Use measured same-host/device training and readout times, project all work, and include at least 25% margin. The whole block with margin must finish before **2026-09-20 04:30:00 UTC**. If it does not fit, start none. Do not shorten runs, omit seeds/readouts, use early endpoints, inspect outcomes to decide completion, or change the schedule to meet the deadline.

An external launcher must write and freeze an admission JSON before seed 0, register all three output paths/reference manifests, and enforce a hard subprocess timeout at 04:30 for training **and readouts**. Immediately before seed 0, the trainer rechecks that the entire projected block still fits from the actual launch time, so an admission written before a delayed GPU release cannot license a late start. The launcher must also recheck remaining-work projections before later jobs. The trainer checks the same deadline between its 250-update chunks, which does not replace the launcher's hard timeout. Unexpected failure or deadline overrun preserves partial artifacts as incomplete; no outcome-dependent retry or favorable-subset publication is permitted.

The admission schema consumed by the new driver is:

```json
{
  "protocol": "longer_joint_budget_v1",
  "admitted": true,
  "world": "gray_frame_wind",
  "model_seeds": [0,1,2],
  "steps": 34000,
  "readout_checkpoints": [34000],
  "deadline_utc": "2026-09-20T04:30:00+00:00",
  "created_utc": "ACTUAL_PROSPECTIVE_UTC_TIME",
  "selection_basis": "runtime only; no model or readout outcomes",
  "exclusive_gpu_confirmed": true,
  "priority_shared_cnn30": {
    "status": "complete", "completed": 30, "expected": 30,
    "evidence_path": "/absolute/priority_completion_audit.json",
    "evidence_sha256": "ACTUAL_SHA256"
  },
  "timing": {
    "three_fits_seconds": "MEASURED_PROJECTION_NUMBER",
    "three_final_readouts_seconds": "MEASURED_PROJECTION_NUMBER",
    "artifact_checks_seconds": "MEASURED_PROJECTION_NUMBER",
    "margin_factor": 1.25,
    "projected_seconds_with_margin": "SUM_OF_THREE_COSTS_TIMES_MARGIN",
    "measurement_artifacts": [{"path": "/absolute/timing_artifact", "sha256": "ACTUAL_SHA256"}]
  },
  "sources": "EXACT_RESULT_OF_longer_joint.sources()",
  "jobs": [
    {"seed":0,"out":"/absolute/new/m0","reference_manifest":"/absolute/original/gray_frame_wind/m0/joint/manifest.json","reference_manifest_sha256":"ACTUAL_SHA256"},
    {"seed":1,"out":"/absolute/new/m1","reference_manifest":"/absolute/original/gray_frame_wind/m1/joint/manifest.json","reference_manifest_sha256":"ACTUAL_SHA256"},
    {"seed":2,"out":"/absolute/new/m2","reference_manifest":"/absolute/original/gray_frame_wind/m2/joint/manifest.json","reference_manifest_sha256":"ACTUAL_SHA256"}
  ]
}
```

Placeholders deliberately are not a usable permit. The priority completion audit must have `status:"complete"` and exactly 30 unique `jobs`, each with an `id`, `status:"complete"`, and recorded underlying artifact identities. The launcher is responsible for verifying those actual priority artifacts and the timing projections before signing this factual record; the trainer verifies the pinned report, registered paths, source pins, arithmetic and absolute deadline. It does not infer measured runtime or priority completion from model outcomes.

Commands, to be used only after that admission:

```bash
# Preparation test: CPU only, no original file/model is modified.
JAX_PLATFORMS=cpu /private/tmp/tinywm-cpu-checks-20260916/bin/python \
  -m studies.revision.longer_joint --self-test

# For each seed 0,1,2; parent queue sets the already-exclusive GPU and timeout.
python -m studies.revision.longer_joint \
  --data /dev/shm/redcube/gray_frame_wind/train.npz \
  --seed 0 --out /absolute/new/m0 \
  --reference-manifest /absolute/original/gray_frame_wind/m0/joint/manifest.json \
  --admission-json /absolute/new_campaign/admission.json

python -m studies.revision.target_gradient_probe \
  --model /absolute/new/m0/model.npz \
  --train /dev/shm/redcube/gray_frame_wind/train.npz \
  --eval /dev/shm/redcube/gray_frame_wind/eval.npz \
  --out /absolute/new/m0/readout --encode-batch 256
```

The CPU self-test checks that only production `steps` changes, validates the 34,000-step warmup/cosine and early-LR difference, compares all three actual-length RNG prefixes and sampled-ID prefixes, and compares a small tile/factored fit against actual `T.train` over two updates. Two updates include a nonzero-LR step because update 0 uses LR 0. Loss arrays and complete parameter/BatchNorm digests must match exactly; saved checkpoints must reload. This tests trainer semantics and configuration, not full-size GPU numerics or convergence. Independent source review and launch admission remain separate from these checks.
