# Revision experiment plan (prospective)

Written before full matched runs on 2026-09-19. Original source/results remain unchanged; all revision code and outputs use a separate worktree and remote directory `~/personal/tinycraft-jepa-iclr-20260919`.

**Admission decision, before full runs:** all three five-update MLP smokes completed with finite losses and intact artifacts. Learned dynamics required 37.9 seconds for first-update compilation then approximately 0.39 seconds per update; exact/fitted-head dynamics 24.5 seconds then 0.27–0.30 seconds; exact/true signals 23.3 seconds then 0.26 seconds. Full 150-update training is projected below two minutes per run, excluding initialization and final evaluation, comfortably below the ten-minute admission threshold. The fixed **45-run MLP campaign is admitted**. Native-grid exact/true smoke also completed (27.6 seconds first update; approximately 0.29 seconds thereafter), so the separately declared **27-run native-grid campaign is admitted after the MLP campaign**. Seed counts are now fixed and will not depend on scientific outcomes.

## Decision and budget

The highest-value new experiment replaces the transition source inside one shared PPO protocol. The existing real-latent versus imagined-latent comparison changes collection length, starts, update organization and total transitions; it therefore cannot attribute its performance gap to the world model alone. A new objective or another visually elaborate environment would not resolve that uncertainty.

Run all three existing gate encoders with **five policy seeds (0–4) and three arms, 45 runs**, conditional only on a five-update, full-batch smoke measurement projecting at most ten minutes per run. If this runtime threshold is exceeded, use three policy seeds (0–2), 27 runs. This runtime decision is made before full-run outcomes and does not depend on reward, survival or wall counts. Keep 150 updates in every admitted run. Two multigpu-vm A100 GPUs run one process each. At historical timings this requires approximately 2–5 GPU hours plus compilation/validation. The third GPU remains available for orthogonal diagnostics or a later justified objective pilot.

## Primary matched experiment

| Arm | Next policy observation | Reward and continuation |
|---|---|---|
| `learned` | Original frozen JEPA prediction | Original frozen fitted heads |
| `oracle_heads` | Exact simulator next frame through the same frozen encoder | The same fitted heads on the current encoded frame/action |
| `oracle_true` | Exact simulator next frame through the same frozen encoder | True environment reward and termination |

Shared protocol: 512 rollout starts, horizon 16, 150 updates (1,228,800 nominal transitions), half dataset resets and half valid eight-frame dataset histories, identical recurrent burn-in, MLP policy plus GRU, latent dimension 512, 16-step backpropagation chunks, minibatch 128, four PPO epochs, learning rate 0.00025, gamma 0.99, GAE lambda 0.9, entropy coefficient 0.01. Original optimizer, clipping and value loss are retained. No reward clipping/shaping, latent rescaling, privileged action choice, or checkpoint selection is introduced.

For each policy seed, all arms share initialization and start-selection seeds. Training randomness is independent of evaluation randomness. Evaluate only the final checkpoint using the same fixed key 20260919 and 256 episodes from the original reset distribution. Save per-episode returns, lengths and walls, exact start indices for every update, complete configuration, source/data/model/policy SHA256 hashes, and training ledger. Historical runs are not claimed to have the same training trajectories: their intermediate evaluations advanced their training RNG.

Primary endpoint: walls passed per real evaluation episode. Secondary: survival through 128 steps, length and the existing descriptive discovery criterion (survival strictly above 50%). Report all admitted runs, including null/negative outcomes and incomplete budgets. Average policies within each encoder before assessing replication across encoders; show the three encoder-level paired differences. A policy seed is not an independent dynamics fit. Per-episode uncertainty is conditional on a fixed policy and does not replace model replication.

## Exact-state and terminal semantics

The gate archive stores position, wind, episode identifier, wall position and hidden gap. Within-episode time is reconstructed from recorded episode boundaries. The actor receives only pixels passed through the same frozen encoder; hidden gap, wall position and elapsed time never enter its observation or action choice. State reconstruction is restricted to the original texture-free gate for which the missing texture key is irrelevant.

Both exact-transition arms freeze the true terminal state and frame after death/time limit, implementing an absorbing extension. `oracle_heads` still reads its original learned reward and continuation on those terminal observations. Those observations are outside the heads' original pretransition training support. Therefore:

- `oracle_heads – learned` is an intervention replacing learned transitions with exact absorbing transitions under this stated convention. It is not a pure measure of prediction error on ordinary live states.
- `oracle_true – oracle_heads` measures the consequence of replacing the learned reward/continuation signals, including their off-support terminal behavior.
- `oracle_true – learned` compares the entire learned imagination pipeline with exact transitions/signals under a genuinely shared PPO protocol.

The ledger records true-terminal fraction, continuation-weighted mass after true termination, and reward accumulated there. These quantify whether the terminal extension materially affects interpretation. A fourth termination-safeguard arm is reserved as a sensitivity only if interpretation requires it; it cannot silently replace the prospectively defined arms.

## Acceptance checks before the full queue

1. Reconstruct 1,024 archived current frames exactly; assert archive episode boundaries agree with dones and initial frames are resets.
2. Compare new and original start samplers at identical seeds. Independent host audit covered 38,550 synthetic starts across reset probabilities 0, 0.5 and 1, with exact parity and correct final-state indices.
3. Replay recorded non-spawn deterministic transitions and match next pixels, reward, termination and next state. Terminal archive frames are resets, so compare terminal reward/done but only continuing next pixels/states.
4. Compare one shared PPO update numerically against the original nested updater on the exact same collected batch and initialization.
5. Complete five-update full-size smoke for each arm, check finite losses, saved artifacts, fixed-key evaluation and runtime. Then record the policy-seed-count decision before starting the full queue.

The independent code review found no blocking indexing, hidden-state leakage, action-key, head timing, burn-in, updater or absorbing-state bug. Runtime checks remain mandatory. Dataset container hashes differ between the original two VM copies, but inventory verification found byte-identical decompressed contents for every archive member; the model seeds therefore use the same actual dataset.

Runtime verification subsequently passed: 1,024/1,024 reconstructed frames; 1,002 deterministic reward/termination transitions and 981 continuing successor frames/states; five actual 32-start batches identical to the original sampler; one copied PPO update produced exactly identical parameters to the original updater (maximum absolute difference 0; loss rounding difference below 5e-7). Evidence is saved in `runs/revision/check_matched.json`.

## Prospectively declared native-grid sensitivity

Before primary full-run outcomes, a second, separately reported campaign was declared: three existing encoders × three policy seeds × all three transition/signal arms, 27 runs with the same budget and evaluation. It changes only the policy torso: reshape the actual 512-coordinate tile latent into its existing 4×4×32 topology, apply two 3×3 SAME convolutions, and feed the unchanged embedding/GRU/actor/critic interface. It does not decode, resize, replicate, amplify or append any feature. The torso has 317,824 parameters versus 393,984 in the original MLP (total actor-critic 583,236 versus 659,396), so this is an architecture sensitivity rather than an equal-parameter architecture comparison. Every within-torso comparison is matched. The experiment only runs if its smoke timing fits the remaining deadline, independently of outcome.

Implementation is `studies/revision/policy.py`, enabled by `matched_ppo.py --torso native_grid`; the default branch directly constructs the unchanged original MLP. Encoder metadata must certify the native tile layout. Independent CPU tests verified original MLP initialization parity, native-grid JIT forward shapes, finite nonzero gradients, exact save/load output parity, and legacy checkpoint compatibility. A full collector/updater GPU smoke precedes admission. Primary and grid campaigns use distinct output roots and are never pooled as one policy protocol.

## Interpreting possible outcomes

- Exact transitions help while fitted signals remain: evidence that learned transition substitution materially limits this matched pipeline; terminal diagnostics qualify its extent.
- Only exact signals help: the fitted reward/continuation interface is a major bottleneck even with exact transitions.
- Neither exact arm learns more: the short-rollout optimizer/controller is itself a serious limitation; the old real-versus-imagined result cannot support a strong dynamics claim.
- Exact arms improve walls without reliable strategy discovery: report that narrower finding. Do not reinterpret partial progress as task mastery.

The existing decoder/CNN result can establish useful information in particular frozen encoders. It does not remove a controller confound or establish sufficiency in every seed. The fitted residual sampler tests one learned distribution with unchanged heads; its failure does not prove that stochastic modeling cannot help.

## Conditional additional work

First complete V1 from currently verified artifacts. Incorporate the matched experiment into V2 only after its entire fixed seed set is complete and audited. Re-evaluate any existing policy used in a newly paired comparison with the independent fixed evaluation seed and save raw episodes.

A PVE-inspired objective is lower priority and exploratory until the requested PVE paper is unambiguously identified. If it means Proper Value Equivalence (Grimm et al., 2021), it is task/value informed rather than a general reward-free objective. A defensible small pilot would freeze the existing encoder and policy/latent capacities, fit an independently validated value teacher on complete, episode-disjoint trajectories, and fine-tune only the predictor with ordinary latent MSE plus a value-aware term. Compare lambda=0 and one predeclared nonzero lambda from the same checkpoint, batches and compute. Do not call one behavior-policy value penalty exact PVE or claim its planning guarantees. Existing fitted value heads were not independently validated adequately for that teacher role. No reward shaping, altered success criterion, or environment-specific geometry enters the pilot.

A reward-free alternative should preserve the same latent shape and data and use generic observations/actions only (for example, a masked future-feature objective with a fixed target and held-out multi-step evaluation). It should be run only if the primary diagnosis identifies a mechanism it can test within the remaining time. Adding objectives merely to obtain a positive result is not justified.

Noise, red walls and new wall layouts are not part of the primary revision. Changing them would require new datasets, matched retraining and an explanation of the scientific variable being changed. Existing static/per-frame nuisance worlds already supply controlled simpler cases; Craftax can supply broader context only with exact dataset/checkpoint/protocol provenance and clearly limited claims.

## Artifacts

- `studies/revision/matched_ppo.py`: matched training and final fixed-key evaluation.
- `studies/revision/check_matched.py`: simulator replay, start sampler and original-updater parity checks.
- `runs/revision/check_matched.json`: runtime validation evidence (remote first, copied locally later).
- `runs/revision/matched/m{encoder}/p{policy}/{arm}/`: full-run artifacts.
