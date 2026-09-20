# Prospective Craftax inventory control comparison

Protocol: `native-inventory-v1`. Registered 2026-09-20 before inspecting outcomes from this matrix. The scientific matrix is approved; source review, numerical checks, runtime smoke tests, and execution receipts remain required. The native environment contract is documented in [craftax_native_inventory_protocol.md](craftax_native_inventory_protocol.md).

**Question:** With a fixed pixel-policy learning algorithm and budget, can policies trained from scratch inside each frozen world-model system obtain native tools in real Craftax-Classic? This is a comparison of two complete frozen-model recipes, not a causal isolation of the JEPA objective.

## Fixed matrix and model identities

| Frozen simulator | WM training seed | Independent policy seeds | Completed imagined transitions per policy |
|---|---:|---|---:|
| Flow | 0 | 0, 1, 2 | 4,998,400 |
| Original JEPA with newly fitted frozen readouts | 0 | 0, 1, 2 | 4,998,400 |

All six policies start from scratch. For each policy seed, both families receive the same freshly initialized actor parameters. There is no behavior-cloned initializer, BC anchor, KL prior, real-environment policy-training data, or policy warm start. The original JEPA is used; last-target-only models and earlier walking-goal experiments are outside this matrix.

- Shared training archive SHA-256: `09f2cb41a38b0f34937a89c4122f6b25bbb3094e34a077b3a572504269d0a1b2` (500,000 transitions).
- Flow checkpoint SHA-256: `b4d7b82ba698f1b7aeaab7383c324523bf27bde1aba2fe817745fce38efa0aef`. Its source SHA is `0cee51a6007fe8b3e732e9cfedf9c241693e2caf5df194da91b78c2e9822f9f9`. Preserve its native 32-frame history, 4×4×32 latent, codec, reward/done heads, and four-step Euler sampler.
- Original JEPA source bundle SHA-256: `59cc282c9bd7a49ab1c272342220bc8e0d8ca0f1aa77bbcb4c23ae31dfd6835d`; original training-source SHA `72cd5c463b4ac2fe7e289f7032e91d88ffac2e59ebb15534524fafe4804e9d0b`. Preserve every model parameter and BatchNorm array exactly, native history 8, 16×32=512 latent coordinates, and 17 actions. The historical decoder/heads are replaced by the separately fitted readouts below; no WM parameter is updated.

Fit one JEPA readout bundle, then freeze and hash it before any policy run. Fit the decoder on all 500,000 pre-action training images and the transition heads on all 500,000 aligned `(z_t,a_t,r_t,done_t)` training labels, sampled uniformly with replacement. No evaluation labels, inventory/achievement labels, split selection, or policy outcomes enter fitting. Fixed seed 0; raw float32 latent cache without transformation. Decoder: original spatial decoder, 16 tokens, width 256, depth 6, 8,000 updates × batch 256. Reward: 529→512→512→1 GELU MLP with native-scale MSE; done: 529→512→1 GELU MLP with binary cross-entropy on the done logit; 6,000 updates × batch 1,024. Both use learning rate .001, global gradient clip 1, AdamW weight decay .0001, 5% linear warmup and cosine decay to zero; BF16 computation/FP32 parameters. No value head is fitted. Record the readout source/config/data/model hashes, losses and elapsed time, and verify exact original-model array preservation.

## Fixed policy learner and simulator interface

Use the recovered clean pixel PPO with **no GRU**, no conditioning scalar, and exactly **6,092,177 parameters**: RGB Impala channels 64/128/128, embedding 512, previous 17-way action input, common policy/value heads. Inputs are only 63×63 RGB uint8 pixels and previous native action; the policy never reads simulator state, inventory, achievement flags, or WM latents. Actor initialization uses `nnx.Rngs(policy_seed + 2)`; training key starts at `PRNGKey(policy_seed + 7)`.

The exact budget is **64 environments × 100 collected transitions × 781 iterations = 4,998,400 imagined transitions** per policy. Preserve collection, GAE and PPO update semantics from the pinned original source: chunks 25, four groups, four PPO epochs, 16 minibatch updates per iteration (12,496 optimizer updates total), gamma .99, GAE lambda .8, ratio clip .2, value coefficient .5, entropy coefficient .01, normalized advantages, AdamW learning rate .00025/weight decay .01, global gradient clip 1. No schedule or policy hyperparameter search. Pin the common runtime, dtype and matmul-precision settings for both families. Omitted historical intermediate evaluations still consume the original single RNG split every eight iterations and at the final iteration, preserving the training-key sequence.

Prospective initialization amendment, made before policy outcomes: build one hashed bank of **genuine eight-frame prefixes beginning at recorded episode starts**, with their **seven actual outgoing actions**, from the shared archive. Enumerate starts in `np.nonzero([ones(1,E); dones[:-1]>.5])` order; exclude only prefixes lacking all eight frames/seven actions or containing an intervening reset. Do not filter inventory, achievement, reward, map or subsequent policy outcomes. Both simulators receive the same prefix bank and candidate draws at every absolute training step. Flow starts with exactly eight real occupied context slots, corresponding real action alignment, and native prefix/future masking; it fills then slides its native 32-slot history, without fake clean left padding. JEPA uses the eight genuine context embeddings and seven real outgoing actions plus the current-action placeholder. The policy initially observes the last prefix frame; its previous-action reset convention remains the original learner's native NOOP in both arms. **Accepted reset times can differ** because model-predicted termination differs; do not claim identical realized training trajectories. These are offline prefixes used only to initialize imagination, not extra real-environment policy-training steps. Native evaluation still begins at fresh ordinary resets with no behavior-policy prefix.

The learner receives raw single-model reward predictions. Predicted termination is sampled as Bernoulli, and both imagination environments additionally end episodes at **1,024 transitions**. This is a disclosed training time limit; evaluation uses the full native limit. Preserve the transition's reward before reset. No action-dependent paid bitmask, once-per-crafting suppression, reward clipping/rescaling/bonus, symbolic rule, hand-coded controller, privileged input, pessimistic ensemble, latent scaling/replication, or extra latent coordinates are allowed. Flow uses its own reward-on-arrival convention. JEPA reads reward/done from current `z_t` and outgoing `a_t`, then predicts/decodes `z_(t+1)` using the unchanged outgoing-action history. This indexing must be verified against each model's existing rollout interface.

## Fixed native evaluation and endpoints

Evaluate only each prescribed final checkpoint on **256 ordinary native reset episodes**, shared across all six policies, evaluation seed **123**. Use `Craftax-Classic-Pixels-v1`, native default parameters, `auto_reset=False`, all 17 actions, and native termination with **10,000 maximum actions**. No extra evaluation cap, starting inventory, favorable-map filter, removed enemies, pre-placed resources, or achieved-state starts. Policy actions are categorical samples from unchanged logits. State/inventory are evaluator-only observations.

For episode `i=0..255` and step `t=0..9999`, use independent fixed streams:

```text
E_i = fold_in(PRNGKey(123), i)
reset_i = fold_in(E_i, 0)
environment_step_i,t = fold_in(fold_in(E_i, 1), t)
policy_action_i,t = fold_in(fold_in(E_i, 2), t)
```

Save these key arrays or their exact reproducible construction and hashes. Identical keys define paired native contexts, not identical action trajectories. For every step, gate metrics by the pre-step active mask, inspect the post-action native state, include terminal-action rewards and tool acquisition, then freeze a finished episode at its first native done. A player who crafts a tool and dies on the same step succeeds at that tool endpoint; survival is reported separately.

**Primary endpoint:** fraction of admitted native episodes with `Achievement.MAKE_WOOD_PICKAXE` true at the first terminal state, denominated by all 256 resets. No alive condition is added. Report the successful-episode count and rate for every policy, including zero.

Prespecified accompanying outcomes: all four tool flags separately (wood/stone pickaxe and wood/stone sword), all 22 native achievement rates, all-four-tools rate, mean number of the four tools, native undiscounted return, executed steps, first-acquisition steps, terminal inventory, health, death/lava/time-limit indicators, and survival to the native time limit. Do not substitute a tool union or walking success for the primary endpoint. Preserve raw per-episode flags and outcomes; any geometric achievement-rate aggregate must state its formula and remain distinct from native per-episode `info['score']`.

Evaluator acceptance checks: zero reset inventory/achievements, valid native action IDs, monotone flags, correct tool action-ID versus achievement-index mapping, inclusion of terminal-action acquisitions, no post-terminal metrics, agreement between native tool inventory and achievement indicators, and the native reward identity `sum(r)=sum(A_T)-sum(A_0)+.1*(health_T-health_0)`. The exact installed environment source/version and parameters are hashed in provenance.

## Analysis and interpretation

Report all six fixed-final outcomes. Within each family, give the mean and sample SD of the three policy-seed success rates. Report the three paired policy-seed differences **JEPA minus Flow**, their mean and sample SD. These are three policy-training replications conditional on **one frozen WM seed per family**, not three independent WM fits. There are 256 shared native episode contexts, 768 policy×episode outcomes per family and 1,536 across both; none of these counts replaces the three policy-seed replication units.

Episode intervals are conditional on the fitted policies. If paired-bootstrap intervals are shown, resample the same episode columns jointly across both families and all three policy seeds, label the interval conditional on these six policies, and do not imply that it includes WM-training uncertainty. Treat three-seed variation descriptively; report secondary outcomes without selecting favorable tool types, policy seeds, or checkpoints.

The systems differ in native history, codecs, objectives and head training, including JEPA's separately supervised frozen readouts versus Flow's jointly trained heads. A result supports a statement about these declared control recipes. It does not prove that the JEPA objective alone causes a difference, that all JEPA models fail/succeed at tools, or that a single favorable seed establishes general physical intelligence.

## Admission, integrity and deadline

Root approval covers the complete fixed matrix. Production still requires independent source review and small correctness/runtime smokes; their outcomes are not scientific samples. Pin all adapter, trainer, evaluator, queue, dependency, source-snapshot and protocol hashes before launch; do not edit pinned files during the campaign. Check paired initial actor hashes and training-key schedules, common reset-bank identity, exact WM/readout identities and unchanged frozen arrays/state. Preserve every completed or failed run and its reason. A partial run is incomplete, never a substitute final checkpoint; no outcome-based early stopping, rerun, seed replacement, or retrospective matrix reduction.

Scheduling target: queue admitted before **2026-09-20 00:15 UTC**, approximately five GPU-hours as the prospective upper scheduling estimate, with delivery before the user's **2026-09-20 07:41 UTC** deadline. Admission and scheduling may use only correctness/resource/runtime evidence. The estimate is not a measured result or a shortened scientific budget. Queue controls must enforce exclusive GPU use, fresh output paths, bounded child process groups and the hard deadline; any interruption or insufficient time is recorded transparently. No approval question is needed for this already approved matrix.

## Numerical-admission amendment before policy training

The original strict Flow comparison failed and remains `passed=false` in the preserved [failure receipt](../artifacts/revision/inventory_flow_parity_failed/result.json). Root authorized a separate, specifically identified numerical admission after the [independent review](craftax_flow_numerical_approval.json), SHA-256 `d65f4e1fe7af9c2d301c014730ca64e1b241bc3695a9ee3dbd03f9c0d22ee990`. This does not change the original tolerance or relabel the failed test.

The archived executed [diagnostic source](../artifacts/revision/inventory_flow_diagnose_source_d91.py) and [canonical results](../artifacts/revision/inventory_flow_diagnose_canonical/result.json) establish exact prepared states, actions, masks, slots, real-context counts, keys and Gaussian arrays for the mixed-age cases. The injected-noise full-batch helper exactly reproduces the unchanged native helper; `lax.map` exactly reproduces scalar execution. With identical Gaussian arrays, canonical native batch-versus-scalar maximum errors are `[.015625, .0625, .015625, .015625]`; native batch-versus-vmap maxima are `[.015625, .056640625, .015625, .015625]`. Vmap MAE and RMSE are also no greater in every row, with exact values bound in the independent approval. Native BF16 execution itself is therefore batching-sensitive in this controlled comparison; no indexing, masking or RNG defect was found.

Admission recomputes these inequalities and exact checks from the hash-pinned raw arrays and requires the independent approval, original failed receipt, executed diagnostic source, unchanged production Flow/dependencies and registered checkpoint. It is specific to that evidence, not a universal error bound or bitwise trajectory-equivalence claim. No model weight, latent width, reward, precision or production transition is changed. Both policy families record and assert the reviewed common JAX configuration: `tensorfloat32` matmul precision, `threefry2x32`, x64 disabled and partitionable Threefry enabled. Current-source GPU training smokes of at least two updates per family and same-policy/same-key native evaluator chunk-invariance checks remain mandatory before production.
