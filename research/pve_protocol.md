# Prospectively fixed value-aware continuation diagnostic

**Status:** the scientific attempt stopped at its predeclared teacher-adequacy gate. There are no scientific continuation or downstream-control results. The probe-validation result below is retained; smoke continuation outputs must not enter the paper as evidence.

The question is whether a small amount of task-value supervision repairs a deterministic predictor while holding its representation and capacity fixed. This experiment cannot establish general intelligence, task-independent model quality, or proper value equivalence.

Source: Grimm et al., [*Proper Value Equivalence* (NeurIPS 2021)](https://papers.nips.cc/paper_files/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html), especially Eq. 9, and the preceding [*The Value Equivalence Principle for Model-Based Reinforcement Learning* (NeurIPS 2020)](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html). The rigorous PVE objective compares actual policy values under repeated model Bellman operators and depends on policy coverage. Here we use one uniform-random behavior policy and approximate, frozen value probes. The required label is **PVE-inspired value-aware predictor continuation**.

The implemented term more directly resembles the value-regression term in the paper's MuZero analysis (Eq. 11). It does not implement the jointly learned reward term or the exact Bellman-operator objective in Eq. 9.

## Exact intervention

Start from the existing gate `tile_factored/model_probe.npz`. Keep the existing encoder and projector exactly fixed at 512 output coordinates, with no latent rescaling, expansion, extra state input, new data collection, reward shaping, or privileged-state loss. Update only the original action encoder, predictor, and prediction projector. Preserve architecture, parameter count, decoder, and pre-existing control heads.

The two continuation arms start from identical checkpoint parameters, with fresh identical optimizers, sampling keys, training windows, learning rates, update counts, and dropout initialization:

* MSE continuation: average coordinate MSE over autoregressively predicted horizons 1, 2, 3, and 4.
* Value-aware continuation: identical MSE plus `0.1 * mean(((V_A(z_pred[t+h]) - G[t+h]) / sigma_G)^2)` over the same horizons. `G[t+h]` is the realized, unmodified discounted return starting at the **future target frame**, ending at its observed episode termination. It is not the return starting at the current context endpoint.

The continuation uses 3,000 updates, batch 128, AdamW, learning rate 1e-4, weight decay 1e-3, gradient clipping at 1, a 5% warmup and cosine decay. Gamma is .99. Lambda .1 and four horizons were chosen prospectively; there is no test-set hyperparameter selection. Additional training is explicit and both continuation arms receive it. Both calculate the teacher loss, including the zero-weight arm, to make the compute paths similar; report actual wall time separately.

Value normalization is a fixed scalar loss normalization using the train-return standard deviation. It does not transform latents or rewards delivered to the environment/controller. Targets have irreducible Monte Carlo noise; this is squared expected-value regression, not exact expected-value supervision. A uniform-random behavior policy makes the recorded action continuation policy-consistent. No claim extends this protocol unchanged to a state-dependent logged behavior policy.

## Train-only probes and adequacy gate

Only archive `frames`, `actions`, `rewards`, `dones`, and collection metadata are read. The archive begins at reset. Derive episode boundaries from observed terminations and exclude every unfinished recording suffix. Split complete training episodes 40/40/20 into disjoint A, B, and validation folds using seed 417. Episode identity includes the environment stream. The continuation sees A+B only; validation episodes enter neither probe fitting nor continuation.

Train two generic two-layer GELU MLP value probes, width 256, directly on the fixed 512-dimensional latents. A is the frozen training teacher; B is a separately initialized, independently fitted auditor never used for predictor gradients. Each fits normalized Monte Carlo returns for 4,000 steps, batch 1,024, learning rate 1e-3, AdamW decay 1e-4, clipping 1, 5% warmup and cosine decay. Both estimate returns conditional on a single encoded observation, while the predictor sees an eight-frame history. They do not represent exact full-history values; partial observability and unobserved elapsed time can introduce irreducible error.

Each probe must reduce validation MSE by more than 10% against its own training-fold mean-return constant and achieve validation correlation above .2. The gate is checked before continuation or held-out evaluation. If either fails, stop and report that the proposed value teacher is inadequate. Do not lower the threshold or tune architecture/lambda against final test data. Report validation target variance, prediction variance, MSE, baseline MSE, and correlation, including failed gates.

Teacher parameters remain frozen while their derivative with respect to predicted latents is retained. This can still produce off-manifold probe exploitation; agreement with an independently fitted auditor, latent MSE, and latent norms are diagnostics, not proofs against it.

## Final evaluation and interpretation

`eval.npz` must have a different collection seed, an identical environment specification, and declared `eval` split; reject an identical archive hash. It is never used for fitting, normalization, early stopping, checkpoint selection, or hyperparameter choice. The original encoder may have used all original training frames before this experiment; the new probe and continuation split does not retroactively change that fact.

Select up to 8,192 no-reset windows from complete test episodes with one fixed random seed, identically for the original model and both continuation arms. For each horizon report coordinate latent MSE, squared Monte Carlo return errors from both frozen probes on predicted latents, discrepancies between probe values of predicted and actual future latents, and corresponding probe errors on actual future latents. Also report predicted/real latent RMS.

These windows are conditioned on surviving all transitions before the final target frame. This follows the original predictor's valid-window contract, but it is a restricted distribution, not unconditional next-death prediction accuracy. Value targets still include subsequent observed deaths. Report this conditioning in any result table or caption.

Enumerate all actions at a context's final timestep and report the standard deviation of their predicted auditor values and RMS latent variation. This is **model-only action sensitivity**, not a counterfactual correctness or control-improvement score. An increase alone is not success. The primary diagnostic is the matched change in the independent auditor's held-out MC-value error; its own real-latent error defines a critical limitation.

Report paired value-aware-minus-MSE differences and 95% episode-cluster bootstrap intervals (2,000 replicates), because overlapping windows from one episode are not independent. This is one continuation seed from one existing training checkpoint. Episode intervals do not quantify training-seed uncertainty. A successful diagnostic supports only a local improvement in this fixed representation's behavior-value prediction.

Before seeing any scientific result, root also predeclared downstream control if both probes pass the adequacy gate: load both saved raw-model-compatible continuation checkpoints into `matched_ppo`'s learned-model arm with the shared original reward/done/value heads, the original MLP policy, policy seeds 0–4, 150 updates, batch 512, rollout horizon 16, and 256 real-evaluation episodes with the same evaluation key. The corresponding five-policy-seed original-checkpoint arm is the base comparator. No heads are refitted per continuation, no control budget changes by arm, and offline regression does not justify dropping an arm. This downstream test is required before claiming control improvement. Additional spatial-policy variants are not part of this precommitment.

## Provenance and execution

The script records SHA-256 for the checkpoint, train archive, test archive, and source; original model metadata (also if the adequacy gate fails); all hyperparameters; collection metadata; exact initial/final predictor hashes; and bitwise frozen encoder, projector, teacher, and auditor hashes. It requires the source checkpoint's recorded training-archive hash to match the supplied train archive. Hashes include parameters and BatchNorm statistics; these probe classes have no other normalization buffers. It saves per-window paired metrics and episode IDs as well as summary JSON. Full outputs are separate from original artifacts. `--smoke` is forced into a suffixed output directory, bypasses the skill gate only for execution testing, and marks all outputs non-scientific.

Before launch: independent source review; `python -m studies.revision.pve --self-test`; short real-checkpoint `--smoke`; review alignment, freeze invariance, checkpoint compatibility, and resource use. Root assigns GPU after the ongoing Craftax evaluation.

Execution checks completed on 2026-09-19: independent source review found no blocking alignment, leakage, or checkpoint defect; local and e2e CPU self-tests passed. The real-checkpoint smoke completed in 340.4 seconds. All arms shared the same initial predictor hash, 8,440,562 trainable parameters, and exactly unchanged encoder/projector/teacher/auditor hashes. An additional NumPy comparison verified all 58 original decoder/control-head arrays and the original model configuration were identical after checkpoint saving. Smoke outputs are explicitly non-scientific at `runs/revision/pve_seed0_smoke/`. The fixed full run was then launched on e2e GPU0 into `runs/revision/pve_seed0/`; interpret its status only from its manifest.

## Recorded outcome of the first scientific attempt

The full attempt finished after 69.2 seconds with manifest status `teacher_quality_gate_failed`. On 102,122 frames from complete validation episodes, the teacher achieved MSE 0.0075795 versus constant-baseline MSE 0.0083052: an 8.74% reduction, correlation .3004. The independently fitted auditor achieved MSE 0.0075592 versus 0.0083052: an 8.98% reduction, correlation .3060. Their prediction standard deviations were .0323 and .0335; the realized-return standard deviation was .0911. Fitting used 198,840 and 197,451 frames respectively, in disjoint complete-episode folds.

Both probes passed the correlation criterion but missed the prospectively fixed 10% MSE-reduction criterion. They are modestly informative; this result does not prove that a better value teacher is impossible or that value-aware model learning would fail. The protocol stopped before predictor continuation and before any final-test encoding/evaluation. The predeclared downstream control condition was therefore not met. No threshold was lowered, and no smoke model may substitute for a scientific model.

Evidence: `runs/revision/pve_seed0/teacher_quality.json` and `manifest.json`, plus `teacher_losses.npz`. Source SHA-256 `495bd9ecf869df0800ebde1bda88cc98561c630a74cf83dfa49c2147ec5ace5b`; existing checkpoint SHA-256 `f81ef20ab6d6f032a85310fe9233e32749924839eb8bb55e4c0e6b1db2b880d2`. A history-conditioned probe would be a distinct revised protocol motivated by the observation/history mismatch, and would require preserving this failed attempt and fixing its design before final-test access.

```sh
CUDA_VISIBLE_DEVICES=0 /home/rldoeswork/personal/tinywmcraft-e2e/.venv/bin/python \
  -m studies.revision.pve \
  --model /dev/shm/stoch/gate/tile_factored/model_probe.npz \
  --train /dev/shm/stoch/gate/train.npz \
  --eval /dev/shm/stoch/gate/eval.npz \
  --out runs/revision/pve_seed0
```
