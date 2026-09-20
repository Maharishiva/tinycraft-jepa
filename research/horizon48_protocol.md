# Conditional horizon-adequacy assay: one fixed 48-step package

Prepared on 2026-09-19 at approximately 20:55 UTC, before inspection of the complete native-grid campaign outcomes. **Scientific admission was subsequently granted explicitly by root after the complete native-grid audit; execution remains pending implementation review, CPU checks, a computational smoke and GPU release.** The original prospective design is retained below, with the admission record appended. No H48 scientific run was launched while preparing this protocol. All existing 16-step experiments and their estimands remain unchanged.

## Question and conditional admission

The short-rollout control procedure may fail to discover a strategy even with exact transitions and task signals. One longer-rollout sensitivity can test whether the negative result depends on that training package. The selected alternative is **48 rollout steps and 50 collection updates**, once, with no horizon search, follow-up length selection, or additional seed selection.

The environment supplies a concrete reason to inspect temporal coverage: reset states contain no wall and the first wall spawns at simulator step 22 (`studies/stoch/env.py:32,125`). A 16-step reset-start rollout cannot reach that spawn; a surviving 48-step rollout can. The half of starts sampled from archive histories can already encounter walls at H16. This is a limitation of temporal coverage for one component of the start mixture, not a claim that H16 training never observes walls. No environment geometry enters the policy or changes its objective.

Admission requires all of the following:

1. An independent audit accepts all 27 intended native-grid runs: three frozen model checkpoints × three policy seeds × the three original arms, at their original complete 16-step/150-update budgets. Missing, invalid or incomplete runs do not count as failures or satisfy this condition.
2. Under the original, already declared discovery definition, **none of the nine final `oracle_true` policies has survival strictly above 50%** across its 256 fixed-key evaluation episodes. This is a count of policies, not a zero-survival or zero-wall criterion. A final rate of exactly 50% is not discovery. The primary endpoints remain walls per episode and survival; discovery remains descriptive.
3. Root explicitly admits this fixed 36-run sensitivity after reviewing the complete audit, the prospective protocol, and a deadline-feasible GPU allocation. Record the audit artifact and its hash, the nine discovery decisions, admission time, and root's decision in a new campaign manifest before launch.
4. The full campaign has an operational runtime allowance through the conservative execution deadline **2026-09-20 07:41:00 UTC**, including initialization, compilation, training, final evaluation and artifact validation. The user deadline is 07:41:45 UTC. Admission uses runtime evidence and resource availability; no partial H48 scientific outcome may determine the matrix or whether remaining arms run.

If any condition fails, record the assay as not admitted and retain the original results. The conditional design is explicitly adaptive to the audited H16 result. It cannot be described as a primary experiment planned before that result or as independent confirmation of the selected failure. If the condition holds, there is one test at H48; an unhelpful outcome does not admit H96 or another optimizer/policy variant.

## Fixed matrix, artifacts and immutable sources

Use only the existing **native-grid policy** on the original texture-free `gate` environment. Use model seeds 0, 1, 2 and policy seeds 0, 1, 2 for every arm: **3 × 3 × 4 = 36 runs**. There are no MLP runs in this assay.

| Arm | Training transition source | Reward | Termination/continuation | Executable |
|---|---|---|---|---|
| `learned` | Original frozen latent predictor | Original learned head | Original learned head | `studies.revision.matched_ppo` |
| `oracle_heads` | Exact absorbing simulator, re-encoded | Original learned head | Original learned head | `studies.revision.matched_ppo` |
| `oracle_reward` | Exact absorbing simulator, re-encoded | Original learned head | True simulator done | `studies.revision.matched_ppo_termination` |
| `oracle_true` | Exact absorbing simulator, re-encoded | True environment reward | True simulator done | `studies.revision.matched_ppo` |

The parent harness must retain SHA256 `6a3460b2db2e47713b7af604454a4b04066c96bfd7748b5c991cfbc433176269`. The separate termination harness must retain SHA256 `1c003dd4e019b2179981877310e131304d13a7150ac7da93e11d5164d011feb1`. The termination executable checks its parent hash before training. It admits only `oracle_reward`. Neither source is changed for this assay; the changed horizon and update count are existing CLI parameters. The original three arms use the canonical parent `matched_ppo.py`; `oracle_reward` uses the immutable termination module listed in the table.

Use the MG archive `/home/rldoeswork/personal/stoch/gate/train.npz`, SHA256 `c347aa2db1726bceadbc6a16fc2b4a7559343f988c4c2074e90c914b05354b55`, for all 36 runs. Use these exact original checkpoint contents, rooted at `/home/rldoeswork/personal/stoch/gate`:

| Model seed | Checkpoint | SHA256 |
|---:|---|---|
| 0 | `tile_factored/model_probe.npz` | `f81ef20ab6d6f032a85310fe9233e32749924839eb8bb55e4c0e6b1db2b880d2` |
| 1 | `tile_factored_s1/model_probe.npz` | `7190a8c231596043b62b6750f807e1c9fa542287dc161b7ca72182f2e68e625a` |
| 2 | `tile_factored_s2/model_probe.npz` | `8ab666aa5838fb81b11af99c5f700f8dd33373f757ef6109a07116df70f9f716` |

These identities come from the existing artifact inventory and must be rehashed on the execution host before admission. A transferred copy is allowed only with identical archive/checkpoint hashes. Reuse no newly fitted model, decoder, reward head or value head. The encoder and policy input remain the original 16 × 32 = 512 coordinates. There is no feature amplification, replication, privileged observation, altered action space, reward shaping, clipping or reward repair. The native 4 × 4 × 32 torso is the already declared architecture, unchanged.

Register the complete matrix in model-seed, policy-seed, then table-arm order before any H48 launch. New outputs belong under `runs/revision/horizon48/m{model}/p{policy}/{arm}`. Campaign metadata and logs belong under `runs/revision/horizon48-campaign`, outside job directories. Refuse every existing output/campaign path, including symlinks; do not overwrite, resume or retry in those roots. Existing `matched`, `matched_grid` and `termination-*` artifacts are read-only references.

## Budget algebra and what this intervention changes

All parameters except rollout horizon and collection-update count remain fixed: batch 512, recurrent chunk 16, minibatch 128, four epochs, reset-start probability 0.5, gamma 0.99, GAE lambda 0.9, learning rate 2.5e-4, policy clip 0.2, value coefficient 0.5, entropy coefficient 0.01, AdamW weight decay 0.01, gradient-norm clip 1.0 and the inherited 180-minute per-job safety cap. The campaign deadline is the tighter stop. Evaluation is final-checkpoint only, 256 episodes, key 20260919.

| Quantity per fitted policy | Original H16 | Conditional H48 |
|---|---:|---:|
| Rollout horizon | 16 | 48 |
| Collection updates | 150 | 50 |
| Starts per update | 512 | 512 |
| Scheduled rollout positions | 150 × 512 × 16 = 1,228,800 | 50 × 512 × 48 = 1,228,800 |
| Sampled start-context instances | 76,800 | 25,600 |
| Recurrent chunks per rollout | 1 | 3 |
| Minibatches per chunk | 4 | 4 |
| PPO epochs | 4 | 4 |
| Optimizer steps | 150 × 1 × 4 × 4 = 2,400 | 50 × 3 × 4 × 4 = 2,400 |
| Optimizer steps between fresh collections | 16 | 48 |
| Nominal position appearances across epochs | 4,915,200 | 4,915,200 |

The divisibility conditions are satisfied: 48 is divisible by chunk 16 and 512 by minibatch 128. **This tests a longer-horizon training package, not a pure GAE effect.** GAE recurs through up to 48 positions, but backpropagation through the recurrent policy still uses 16-step chunks, each initialized from the stopped-gradient rollout hidden state. H48 draws one third as many start contexts, collects longer uninterrupted trajectories from each start, changes the delay between data collection and policy refresh, and presents three consecutive chunks under each epoch. Advantage normalization uses the whole collected batch, so its support also changes. The number of AdamW steps and nominal positions are equal; wall time, effective alive-weighted positions, state distribution and optimization trajectory are not equal.

Longer rollouts also change exposure to true death and the learned-continuation absorbing extension. Thus report true-terminal fraction, mean alive weight, postterminal alive-mass fraction and postterminal learned reward from every available ledger, alongside control endpoints. `oracle_reward` and `oracle_true` must assign exactly zero effective weight and weighted reward after true termination for finite signals. The terminal transition itself retains the arm's prescribed reward. Raw learned rewards are never manually zeroed.

## Matching and permissible comparisons

**Within H48**, every four-arm seed block shares the same checkpoint, canonical data, policy initialization, native-grid source, start sampler, number/order of draws, burn-in, optimizer organization and evaluation key. Validate semantic start arrays, not just their shape: all four `starts.npz` indices must be identical with shape `(50, 512, 4)`. Policy and rollout trajectories may diverge after the interventions change updates. For the three exact arms, the same true simulator process is used with the same conditional random keys; actor inputs remain latent observations, never hidden simulator state.

**Across H16 and H48**, match checkpoint, policy seed, initialization constructor, architecture, evaluation source/key and all unchanged hyperparameters. The first 50 sampled start batches should equal the first 50 H16 batches because start sampling has its own identically seeded host RNG; verify that prefix explicitly. Do not claim identical training action noise or trajectories. The collector calls `jax.random.split(rollout_key, H)`, whose cross-H key-prefix identity is not a source guarantee. The two runs also use different numbers of collections and different policies after the first update. Shared initialization and paired evaluation do not turn this into a one-variable causal GAE comparison.

Require exact equality of the normalized shared dependency-hash maps both within H48 and across H16/H48: parent harness, `studies/stoch/rl.py`, `studies/stoch/env.py`, `studies/redcube/env.py`, `train.py`, `probe.py` and `studies/revision/policy.py`. For `oracle_reward`, the approved termination harness is the sole additional source entry; require its admitted hash and the parent-hash metadata. Normalize execution-root prefixes only, never file contents or digest values. Other campaign dependencies should also be pinned for no-drift execution even when the per-job source map does not include them.

The H16 comparator is the existing **native-grid** stratum only. Use its original nine seed pairs per arm; `oracle_reward` comes from the separately admitted termination sensitivity after independent completion/provenance audit. Never pool H16 MLP seeds with native-grid seeds, reuse H16 outcomes as H48 replications, or replace the H16 table with whichever horizon looks better. The H48 campaign may be completed even if a termination reference is unavailable, but any corresponding cross-H contrast remains explicitly unresolved until that reference is independently valid.

## Prospectively fixed reporting and analysis

Retain all 36 planned outcomes, including negative, null, failed, invalid and incomplete jobs. Use final-budget endpoints; do not select checkpoints, seeds, horizons or a favorable subset of arms. A valid H48 seed block requires all four arms at 50 updates and 1,228,800 scheduled positions, consistent raw episodes and scalar summaries, finite recorded diagnostics, equal source/input identity apart from the admitted termination harness, exact start matching and valid saved-policy hashes. An incomplete block remains in the intended denominator and is excluded as a whole from provisional four-arm numerical contrasts; final complete reporting requires all nine blocks.

The endpoints are **walls passed per real episode** and **survival to the 128-step cap**. Survival is determined by unchanged episode reward/return, not length alone: death at step 128 is death. Secondary discovery is survival strictly above 50%. Report individual policies, each fixed model's three-policy mean, all three model means, their equal-model descriptive average and their range. Every arm is shown even if oracle performance remains poor.

Report these H48 within-block contrasts with target minus reference signs:

1. `oracle_heads − learned`: replacing learned transitions with exact absorbing transitions while retaining fitted signals; terminal-support exposure qualifies this contrast.
2. `oracle_reward − oracle_heads`: replacing learned continuation with true termination under exact transitions and fitted reward; this combines preterminal continuation, termination/bootstrap and postterminal weighting changes.
3. `oracle_true − oracle_reward`: replacing fitted reward with true reward under exact transitions and true termination.
4. `oracle_true − learned`: replacing the full imagination pipeline with exact transitions and signals under the shared H48 package.

Separately report `H48 − H16` for **all four arms**, model by model and matched policy seed by policy seed. The motivation centers on `oracle_true`, but outcomes for the remaining arms cannot be dropped. Cross-H comparisons quantify this fixed package change. A higher exact-arm result supports sensitivity of this controller to the package; it neither proves H48 is optimal nor establishes that the world model alone caused the original failure. Failure of both packages bounds only the tested optimizer/controller settings, not representational sufficiency or the impossibility of learning a policy.

Use the established deterministic 10,000-draw, 95% percentile bootstrap with base analysis seed 7341, resampling paired policy seeds **within each fixed model**. With three policy seeds these are coarse descriptive intervals. Also report paired evaluation-slot intervals with policies fixed and shared slots resampled jointly across policies. They quantify a different source of variability. No model-family interval, pooled 9-policy model-replication claim, discovery-based subset, multiplicity-adjusted significance claim or causal mediation claim is warranted. The intervals are unadjusted and the admission is explicitly adaptive. The current fixed-H16 analyzers must not be silently reused with their budget constants: a separate H48 analysis entry point or externally reviewed parameterization must enforce this protocol before accepting the artifacts.

## Execution and validation contract

The new standalone launcher is `studies/revision/run_horizon48_queue.py`; it must be independently reviewed before launch. It enumerates all 36 jobs before execution, pins the protocol/source/checkpoint/data hashes, and records explicit root admission and the complete H16 audit. It refuses pre-existing paths, rechecks immutable content before and after each job, verifies an exclusively allocated idle GPU immediately before launch, and stops the subprocess group at the absolute deadline. Root has allocated **MG physical GPU 1 only**, after forecast extraction and the complete 24-run termination queue release it. The launcher requires all 24 predecessor jobs and their saved artifacts to be complete and requires the predecessor queue process to be gone. MG GPU 0 and e2e are outside this allocation.

The runtime gate projects the **whole** 36-run matrix plus validation within the remaining allocation. Root requested one separately saved full-size computational smoke: model 0, policy 0, `oracle_true`, H48, two collection updates, all other settings including the 256-episode evaluation unchanged. This fixture is never a scientific endpoint, never replaces a registered policy, and cannot choose parameters. The queue checks complete finite artifacts and source/config/input identity, then uses timing alone. The per-job allowance is the larger of 300 seconds and 1.5 times `(smoke total_seconds + 48 × second-update seconds)`; all 36 allowances must fit before a scientific launch. This is conservative operational extrapolation, not a guarantee that arms cost the same. Every scientific subprocess also has the absolute campaign deadline. No H48, update, batch or seed count is shortened to fit. If the full matrix is infeasible, record not admitted. Unexpected deadline stops or failures preserve all partial files and all unstarted jobs in the manifest; they do not license outcome-dependent retries.

Required per-run artifacts remain `config.json`, `final.json`, `episodes.npz`, `starts.npz`, `ledger.jsonl` and `model.pkl`. Every complete ledger must contain updates 1 through 50 in order, final transitions must equal 1,228,800, episode count 256, eval seed 20260919, and the final saved policy/checkpoint hashes must agree. Check true-termination zero-postterminal invariants for both exact-true-done arms. Record the original learned-head arm's corresponding values, including zeros, without editing them. No active source, original result, frozen V1 file or current GPU process is modified by preparing this protocol.

## Source review record

Both harness hashes were verified locally during protocol preparation. Independent read-only review confirmed the CLI accepts the proposed values, the dimensions divide exactly, the 2,400-step/1,228,800-position algebra is correct, recurrent BPTT remains 16, and cross-horizon action-key identity is not guaranteed. In `studies/revision/matched_ppo.py`, relevant anchors are full-sequence GAE at line 152, collection-wide advantage normalization at line 170, stopped-gradient chunk initialization at line 181, divisibility checks at line 208, native topology at line 230, optimizer construction at line 240, independent start RNG at line 264, nested update loops at line 277, and final-only evaluation at line 295. The termination source checks its frozen parent at `studies/revision/matched_ppo_termination.py:200`; learned reward with true done is at line 137. This is source-compatibility evidence, not an H48 GPU smoke, independent full-campaign audit, admission or scientific result.

## Root scientific admission record

On 2026-09-19, after the prospective design above was completed, root explicitly admitted the single 36-run matrix through the experiment coordinator. The machine-readable record is `research/horizon48_admission.json`. Independent `research/native_grid_final_audit.json` has status PASS for all 27 native-grid runs, SHA256 `7001e8670a9b6c51ca7bb742184b8d4f61ca5841ec101a42cf1879f910e19633`. It audits the complete publication report SHA256 `5fe798798b64fbdbf2ea735bb6a8ae1613d8669ca8fdf2592dd43869f5c42809`. The nine `oracle_true` survival rates in `(model, policy)` order are `(0,0)=0.0078125`, `(0,1)=0.01171875`, `(0,2)=0.015625`, `(1,0)=0.0078125`, `(1,1)=0`, `(1,2)=0.0078125`, `(2,0)=0.00390625`, `(2,1)=0`, `(2,2)=0.015625`; therefore discovery is 0/9 under the original strictly-above-50% rule.

The admission fixes all four arms and all nine seed pairs regardless of subsequent signs or effect sizes. Execution conditions remain: independent implementation review and CPU tests; the separately authorized two-update computational smoke; complete release from the 24-run termination queue; idle MG GPU 1; and the full deadline allowance. This admission does not permit another horizon, retries, endpoint selection, or changes to either immutable harness. The separate analyzer is `studies/revision/analyze_horizon48.py`; it validates raw artifacts and reports all within-H48 and cross-H package contrasts without altering original reports.
