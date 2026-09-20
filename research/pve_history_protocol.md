# Final history-conditioned value-aware attempt

**Final status: stopped at the unchanged teacher-adequacy gate. No scientific predictor continuation, final-test evaluation, or downstream control was run. No further teacher variants are permitted.**

**Protocol fixed before this attempt's teacher fitting or scientific test evaluation. This is a post-result adaptation, not a preregistration of the entire research program.** The earlier single-frame teacher and auditor reduced validation MSE by 8.74% and 8.98%, missing the unchanged 10% gate. Its code, protocol, and artifacts remain intact under `pve.py`, `research/pve_protocol.md`, and `runs/revision/pve_seed0/`. Its scientific attempt stopped before continuation/test evaluation. The validation episodes are reused here, so a passing validation gate does not itself remove adaptation or selection bias.

This final attempt addresses a specific information mismatch: the predictor conditions on eight encoded observations and their actions, while the failed teacher saw one observation. Fit two generic GRU value readouts with hidden size 128 over the existing eight 512-dimensional latents and their observed incoming actions. The GRU state starts at zero for every window. The readout is one linear scalar head. There is no new world-model representation, expanded latent, privileged-state input, latent rescaling, reward shaping, environment modification, or control-policy change.

The interpretation remains **PVE-inspired value-aware continuation**, not proper value equivalence. [Grimm et al., Proper Value Equivalence, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/400e5e6a7ce0c754f281525fae75a873-Paper.pdf) requires actual values/policy coverage for its guarantees; the implemented auxiliary term resembles the value-regression component of Eq. 11 and does not implement Eq. 9 or its jointly learned reward term. These are approximate behavior-history values fitted to noisy realized returns from one uniform-random behavior policy.

## Exactly matched information and targets

For a context with frames `z[s:s+8]`, the teacher input is those eight latents and incoming-action slots `[zero sentinel, a[s], ..., a[s+6]]`. The first incoming action is intentionally absent: its predecessor lies outside the window. The target is the complete real discounted return `G[s+7]`, gamma .99, including the native reward on the outgoing transition from that final frame. The final frame's outgoing action is never a teacher input. Use an all-zero one-hot sentinel, implemented by index -1, instead of confusing a real action with missing history.

At rollout horizon `h=1..4`, shift the context to frames `s+h,...,s+7+h`, appending the predictor's previous future latents as needed. The frozen teacher receives that mixed observed/predicted history and `[zero sentinel, a[s+h], ..., a[s+6+h]]`; its target is `G[s+7+h]`. No future outgoing action is included. At h=1, the last incoming action is `a[s+7]`, which caused the first predicted frame; at h=4 it is `a[s+10]`. In action-sensitivity evaluation, the enumerated action must be this last incoming action of the one-step shifted history.

## Fitting, unchanged adequacy gate, and permanent stop

Reuse the existing exact checkpoint and train/eval archives from the first attempt and verify all three hashes against its manifest. Derive complete episodes solely from rewards/dones; no simulator-state fields are read. Reuse the same deterministic complete-episode A/B/validation split (40/40/20%, split seed 417). A trains the frozen teacher, B trains the independent auditor, and validation fits neither. Exclude incomplete recording suffixes and reset-crossing histories.

Teacher examples require eight observed frames. Therefore the first seven frames of each episode are excluded and the validation population differs from the single-frame attempt. Do not present the two validation percentages as a matched comparison or evidence that history alone improved value prediction. Return normalization uses all complete A+B episode frames exactly as in the first attempt; it changes only scalar loss units, never model features or environment rewards.

Both critics use GRU hidden128, 4,000 updates, batch1,024, AdamW learning rate1e-3/decay1e-4, clipping1, 5% warmup, and cosine decay. Seeds are fixed in code. No best checkpoint is selected. Each must reduce validation MSE by **more than 10%** against its own fitting-fold mean-return constant and achieve correlation **above .2**. Both must pass. Report all errors, target/prediction variance, sample counts, parameter counts, and fitting times.

**If either fails, stop permanently: no more critic architectures, threshold changes, training-budget extensions, or teacher variants.** The model-comparison and control steps below are conditional on both passing. A gate failure is a failure of this proposed supervision source under this budget, not a proof that value-aware model learning cannot work.

## Conditional matched continuation and final test

If the gate passes, freeze both critics. Differentiate the teacher with respect to its input histories, never its weights. Compare the same two 3,000-step continuations from the identical original checkpoint: four-horizon latent MSE, and that MSE plus `.1 * mean(((V_A(predicted_history) - G_future)/sigma_G)^2)`. Both update only the original action encoder, predictor, and prediction projector. Encoder and projector remain bitwise fixed at 512 dimensions. Architectures, initialization, batches, seed, optimizer, schedule, and training budget are identical. Keep batch128, learning rate1e-4, weight decay1e-3, clipping1, 5% warmup, and cosine decay.

Final `eval.npz` is not used for fitting, normalization, early stopping, architecture choice, lambda choice, or teacher selection. The first attempt's non-scientific smoke used a tiny fixed test slice for execution checking; no scientific model result was obtained from it and no design choice here is based on those smoke metrics. After the history gate, evaluate the base and both continuations on identical fixed up-to8,192 no-reset windows from complete test episodes. Report horizon1–4 latent MSE, teacher and independent-auditor MC-return error on predicted histories, their disagreement with values of actual future histories, real-history probe error, latent RMS, and model-only action sensitivity.

These continuation/evaluation windows condition on surviving all transitions before the final target frame. They do not measure unconditional next-death prediction accuracy. Action sensitivity alone is not accuracy. In this recurrent readout, action-value spread can reflect both changed predicted latents and the teacher's direct incoming-action conditioning; it is not solely predictor sensitivity. Report latent spread separately. Auditor agreement cannot rule out shared approximation error or off-manifold exploitation. Use paired episode-cluster bootstrap intervals (2,000 replicates) and disclose that they do not include training-seed uncertainty. There is one model and continuation seed, fixed to 0.

Both resulting checkpoints then receive the predeclared downstream `matched_ppo` learned-model comparison, regardless of offline improvement/regression: original MLP actor, unchanged shared original reward/done/value heads, policy seeds0–4,150 updates,batch512,rollout16,chunk16,minibatch128,4epochs,256 final real episodes,evaluation seed20260919. The existing five-policy-seed original-checkpoint result is the base comparator. No per-arm head refitting, best-checkpoint selection, or policy-feature scaling. Do not claim control improvement without these real evaluations.

## Review and provenance

New source is `studies/revision/pve_history.py`; the earlier `pve.py` remains unmodified and provides the audited episode, hash, encoding, and bootstrap helpers. Record both source hashes, the failed prior manifest hash and probe results, exact input hashes, parameters, versions, elapsed times, and initial/final predictor/frozen-module hashes. Save raw-model-compatible continuation checkpoints preserving decoder and heads. Require independent design/source approval and a separate `--smoke` before the scientific run; smoke outputs are forced into a separate suffixed directory and marked non-scientific.

Independent design/source review found no blocking defect. It included 120 randomized complete-episode/reset window checks, index-coded shifted-history/action tests, and a second review of gradient/optimizer/checkpoint handling. The separate e2e smoke completed in 351.3 seconds with finite losses/metrics, unchanged encoder/projector/critic hashes, and identical initial predictor hashes across all arms. Independent NumPy round-trip checks confirmed both saved continuations preserved all 58 original decoder/control-head arrays and original configuration. Smoke source SHA-256 and the subsequent fixed scientific source are `374df2db0c52200f5a93ed3ef5408bc19cb41b5185588468154be72195a13300`; unchanged helper SHA is `495bd9ecf869df0800ebde1bda88cc98561c630a74cf83dfa49c2147ec5ace5b`. Smoke artifacts remain separate at `runs/revision/pve_history_seed0_smoke/` and are not scientific evidence.

## Recorded final outcome

The full attempt terminated after 77.3 seconds with manifest status `teacher_quality_gate_failed_final_attempt`. Neither critic passed the fixed >10% MSE-reduction requirement, although both passed the correlation threshold.

| Validation metric | Training teacher A | Independent auditor B |
|---|---:|---:|
| Fitting windows | 153,502 | 152,322 |
| Validation windows | 78,793 | 78,793 |
| Critic parameters | 247,809 | 247,809 |
| MSE | 0.0077666 | 0.0075200 |
| Own training-mean constant MSE | 0.0065243 | 0.0065243 |
| Relative MSE reduction | **−19.04%** | **−15.26%** |
| Correlation | 0.2913 | 0.3182 |
| Prediction standard deviation | 0.06573 | 0.06622 |
| Realized-return standard deviation | 0.08077 | 0.08077 |

Negative reductions mean that these fitted critics were worse than their respective constant predictors on the episode-disjoint validation windows. Their low fitting losses did not transfer to this validation population. This is a failure of these supervision sources under the specified architecture/budget; it does not establish that the representation lacks value-relevant information or that PVE/value-aware learning cannot help. The history and single-frame gates use different populations and architectures and are not a controlled comparison of the effect of history.

The stopped manifest and teacher metrics are `runs/revision/pve_history_seed0/manifest.json` and `teacher_quality.json`; fitting curves are in `teacher_losses.npz`. The first failed attempt remains separately recorded at `runs/revision/pve_seed0/`. No threshold was lowered, no calibration or extra training was added, and no scientific continuation checkpoint exists. The conditional downstream control experiment was not triggered. e2e GPU0 was released to the separately authorized forecast experiment after confirming this process had exited.

```sh
CUDA_VISIBLE_DEVICES=0 /home/rldoeswork/personal/tinywmcraft-e2e/.venv/bin/python \
  -m studies.revision.pve_history \
  --model /dev/shm/stoch/gate/tile_factored/model_probe.npz \
  --train /dev/shm/stoch/gate/train.npz \
  --eval /dev/shm/stoch/gate/eval.npz \
  --prior-attempt runs/revision/pve_seed0 \
  --out runs/revision/pve_history_seed0
```
