# Revision progress and handoff

Started 2026-09-19 19:41:45 UTC. User deadline: 2026-09-20 07:41:45 UTC.

## User-directed rewrite at 23:02 UTC

### Concrete progress by 23:27 UTC

### Update at 23:49 UTC

The complete post-outcome distance curve now passes 8,179 independent checks and 176,334 native transition replays. H3 exact hits out of128 at distances1,2,3,6,9,12 are seed0 `[128,128,121,25,3,0]`, seed1 `[128,128,123,32,2,0]`, seed2 `[128,128,124,34,1,0]`. At distance3, H1 gets22/16/21 versus H3's121/123/124 and random19. This supports nearby multi-step visual-goal control and exposes the short planner's operating range; it does not replace the failed hard-goal primary. The final full curve can return case1 to a qualified positive result. Files: `research/redcube_goal_distance_table.json`, `redcube_goal_results.md`. All goal GPU work ended; MG1 released.

The exact frozen-encoder variance audit is verified:843 paired safe contexts by25 goals. Mean-only versus exact expected latent-cost action disagreement is2.652% for black wind and1.077% for persistent-gray wind. Latent regrets are small; physical-cost confidence intervals include zero. Use one concise main sentence and full appendix evidence, not a new main percentage figure. `research/oracle_variance_results.md` and raw artifacts preserve adverse physical examples and exact-zero controls.

All original visual packets plus the gate-seed1 control-aligned export pass independent source/array/PNG checks. The omitted imported flow kernel has a separate posthoc receipt: initial source snapshot, local and VM bytes agree at SHA7358061ebaae220616196e8905a912270247a517c5315034e40d538781a511ef. Selection starts at lexicographic t>=1 to make incoming actions available. Gate same-flow rerun has four RGB channels differing by one level; this numerical variation is retained and documented, not silently overwritten. Global annotation palette is JEPAorangeCD663E, flowpurple7860A5, realteal087E83, navy17324D.

Craftax admission numerical issue is resolved in principle, with final runtime smokes pending. Strict scalar-vmap bit-parity failed and its receipt is preserved. Inputs/masks/slots/actions/keys/noise are exactly equal. The native full-batch helper itself differs from scalar execution by max0.0625 latent units; vmap differs from native full-batch by max0.056640625 and is no worse in every tested row by max/MAE/RMSE. Independent review found no indexing/RNG defect. Root approved evidence-based numerical admission without changing model weights, precision, outputs, reward rules, or the old failed check. JEPA policy smoke projects about26 minutes per full4,998,400-step policy. Full six-policy production queue is not yet admitted as of this note.

Verified supervision difference in exact Flow checkpoints: reward/done weights0.1 each; full task-head gradient into dynamics for first8,000 steps, then0.1 strength for last8,000; tokenizer9,000-step reconstruction stage is frozen before dynamics. JEPA fits MSE+SIGReg without rewards, then frozen heads. Equal available500k labels does not isolate objective or sampling. Both comparison protocols now say this.

Anonymous packager now supports html-redcube, explicit scientific-source dependency allowlist, dynamic inventory source, exact JS/frame asset validation and a portable fresh-run command planner. Read-only preflight passes406 payload files/278 links;37 fixture tests pass. No final bundle built. `portable_inventory.py` verifies user-supplied data/model hashes and emits a dry-run14-stage fresh plan; historical private-attestation replay remains a distinct boundary. Recheck after final drafts and result reports freeze.

- The old audit-centered V2 is archived at `submission/archive/20260919T230951Z-audit-v2` with hashes. The replacement manuscript compiles with six main pages before final empirical figures. New title is an internal working choice; the public submission is untouched. Cold story review rates current completed novelty about 4/10 and clarity 7/10, not strong accept. The key potential contribution is the within-recipe noise-persistence by wind interaction; Sobal 2022 already includes moving-dot/noise/probe/generative-reference experiments and the nuisance construction.
- All four fixed visual exports are complete (three predetermined scenes each, five genuine frame streams, actual actions). All 1,056 native PNGs and packet hashes pass exact checks. Root inspected the new HTML in a browser and clicked its exact discrete-cost toggle; also viewed the concept figure and fresh frame strips. `html-redcube` now uses fresh exports. Gate seed0 illustrations are explicitly distinguished from the new gate control's seed1.
- The goal-MPC positive-control attempt failed across all three model seeds: H3 exact hits 5/128, 5/128, 9/128; H1 zero; random 3/128; privileged oracle 128/128. All unchanged outcomes and preselected failed trajectories are retained. A separately labeled post-outcome distance-response diagnostic is admitted for all distances 1,2,3,6,9,12 and all three models, same H3 planner, to expose its operating range. No long-CEM run is admitted.
- H48 old-control queue stopped safely at 24/36 completed policies solely for the user's scope change; all checkpoints retained and twelve unstarted jobs documented. No additional head-only/last-only variants will launch. Target-gradient twelve-model queue continues on MG0.
- Shared-CNN red-cube protocol now fixes gate H48/U150 (3,686,400 positions) for every dream and decoded-real arm; other worlds use H48/U50 (1,228,800). Four worlds and three paired policy seeds give 24 dream policies plus six gate real-interface controls. JEPA gate seed1 is prospectively selected from favorable prior decoded-real control evidence, explicitly disclosed; no new policy outcomes guide selection. Canonical framewise encoder/decoder chunking will avoid shape-dependent BF16 differences. Implementation and independent smoke review are underway.
- Meaningful Craftax source recovered: clean from-scratch flow-WM PPO obtained wood pickaxes in 94.5% of 256 native episodes in historical evidence. Repeated-crafting reward-masked runs are excluded. New matched experiment is two recipes by three paired policy seeds, exactly 4,998,400 imagined transitions each, same original no-GRU pixel actor and PPO core, raw native RGB final evaluation, all 22 achievements reported. To eliminate fabricated reset histories, both use all 2,533 eligible genuine eight-frame prefixes from recorded episode starts (five incomplete exclusions), not repeated frames. No outcome filtering; mean prefix return0.960 is descriptive and not an inventory estimate. Readout refit finished in152.5s; original JEPA arrays are byte-identical. E2E runs source/parity/runtime smokes before the full queue.
- Exact oracle variance audit completed both declared conditions after strict renderer/encoder/identity controls; independent raw-cost verification is pending. MG1 proceeds to the short gate-seed1 visual export and goal-distance curve, then shared-CNN training. E2E is exclusively reserved for meaningful Craftax.
- Longer JEPA wind training is assessed but NOT admitted: three fresh34,000-step fits would take about two GPU hours. Existing checkpoints omit optimizer state, so this would be a compute/schedule sensitivity rather than exact continuation. Admission requires completing all three after primary work before04:30UTC.

The user rejected the audit-centered V2 and nearby Craftax walking task as the main story. The new main paper is organized around minimal red-cube environments: (1) useful representation and successful control, (2) loss of control-relevant representation, and (3) retained state information but failed stochastic prediction/planning. The third empirical case remains a hypothesis, not an established result. The user's exact discrete action-ranking and nonlinear-recursion examples explain the mechanisms; they do not prove that every deterministic world model fails.

Preserve the previous V2 and both existing HTML versions. The author is rewriting the main manuscript; the new companion is `submission/html-redcube`. Heads, probes, PPO qualification and the complete negative experimental record remain available in appendices and internal reports, but no longer determine the main narrative. The historical JEPA latent-MLP versus flow pixel-CNN comparison is not a fair matched controller comparison. A shared stock-CNN decoded-RGB comparison is being prepared with unchanged native rewards, no latent amplification, valid histories, matched policy budgets, and exact checkpoint provenance. Case 3 will not be manufactured if this comparison does not support it.

Resource reprioritization follows the user's scope change, independently of experimental outcomes. The twelve-model target-gradient study is preserved. Other queued objective/head variants should stop at safe boundaries unless needed for the new comparison. E2E Craftax seed replication finished before the interruption: both additional final checkpoints are preserved and the GPU was verified idle at 23:02 UTC. No last-only Craftax production run launched. Craftax now requires an actual inventory-acquisition WM-trained policy and fair JEPA comparison to return to the main paper; the local-goal results are archived.

New independent work: exact one-step oracle variance evaluator for fixed red-cube encoders; matched raw/reconstructed-real/predicted-future visual export with preselected windows; independent challenge of whether the three proposed cases support the desired causal story. Original deadline remains 2026-09-20 07:41:45 UTC. The goal tool currently reports `paused` despite the user's explicit continuation; assistant cannot resume goal status through the goal tool. Authorized task work is continuing in this user turn.

## Resume at 22:02 UTC

An assistant-usage interruption stopped some agent turns, but remote processes survived. The user explicitly said `i aprove all copying and running experiments`; pending source transfers are now authorized. No further permission is required for the scoped experiment files on the two existing user VMs. The goal tool still reports `usageLimited`; its status cannot be resumed by the assistant tool, but this user-triggered turn and collaborator tools are working.

Live verification: MG0 runs the fixed twelve-model target-gradient queue (PID264170); MG1 runs the 24-policy termination queue (PID276827). Craftax seed replication has now launched on e2e using its existing dataset. Avoid duplicate launches. The six last-only fits retain the safer original-host data plan and can follow the H48 queue on a free MG device; no gray-archive transfer is necessary.

Completed since earlier notes: all45 primary and27 native-grid policies are fully audited, with zero discoveries even in all24 exact-transition/true-signal policies. Allthree forecast evaluations are complete with exact canonical-batch parity. Forecast results establish readout-transfer errors, not reliable wall-state recovery. Root's posthoc training-cohort constant audit is independently verified: in the h8 unrevealed slice, mean/median constants achieve4.769/4.642px MAE versus6.164 for the forecast-fitted readout. Both PVE teacher attempts failed their unchanged adequacy gate; no scientific predictor continuation or PVE control experiment exists.

The single H48 follow-up is admitted:36 native-grid policies, four arms, H48×50updates×512 =1,228,800 nominal positions and2,400 optimizer minibatches, after termination24. The separately reviewed last-only secondary study has six fits with final full-history loss, unchanged dimensions/data/budget/SIGReg; all original12 remain unchanged. See the corresponding protocols and launchers. Two additional Craftax training seeds1/2 use only the original recipe seed changed and the same384-start evaluator.

V2 is a working eight-main-page draft, not final. V1 remains frozen. Source/protocol files, experiment queues, analysis/audit scripts and the anonymous-bundle builder are implemented. Final V2 must wait for all admitted results, honest independent review, visual QA and packaging.

## Preserved source

Original checkout: `/Users/rldoeswork/Documents/rl/tinycraft-jepa`.
Separate worktree: `/private/tmp/tinycraft-jepa-iclr-20260919`.
Branch: `codex/iclr-strong-paper`; base commit `1fa2853b5f365a8bdd6d732eebfa851e0197400b`.
Source snapshot includes the original dirty tracked files and untracked scientific artifacts; hashes are in `source_snapshot.json`.

## Completed work

- Full evidence audit: 146 policy runs, 66 selected gate endpoints, 109 cube architecture/condition rows, separate Craftax and spectral diagnostics.
- Source/checkpoint/data inventory on both VMs. Gate archive containers differ, every decompressed member is identical. Flow aliases are one identical checkpoint.
- Ten ICLR 2026 award/oral primary papers read and visually inspected; reading ledger and official PDFs retained internally under `research/reading/`.
- Current JEPA/PVE literature review, including close overlapping work and explicit novelty limits.
- V1 frozen in `submission/v1`: nine main pages, 21 total; official style restored. PDF SHA256 `7aeb87b8ce80836f598bc51a3b91fd703fb9071fb29bcf03ca5a62c6887dea79`. `v1_freeze.json` covers 111 manuscript/HTML/figure files; zero drift verified after V2 setup.
- HTML companion in `submission/html`, offline assets, real saved-frame viewer, 66-endpoint explorer, desktop/mobile QA.
- Nine scientific figure sets (SVG/PDF/PNG), fixed palette and data provenance, in `submission/shared/figures`.
- Independent V1 reviews: empirical 5/10, novelty 4/10. Preserve actual reviews; do not relabel them as strong accept.

## Running experiments

1. **Matched gate PPO:** multigpu-vm GPUs 0 and 1; experiment_design/campaign_monitor own queue. 3 existing encoder checkpoints × 5 policy seeds × 3 arms = 45 runs. Original learned transitions+heads; exact absorbing transitions+same heads; exact transitions+true signals. All share starts, policy, updates, horizon and fixed final evaluation keys. Raw per-episode endpoints retained. All smokes/parity checks passed. Started around 20:00 UTC. Frozen remote snapshot is `~/personal/tinycraft-jepa-iclr-20260919`.
2. **Native-grid policy sensitivity:** 3 encoders × 3 policy seeds × 3 arms = 27 runs, queued only after all primary runs complete. Native 4×4×32 convolutional torso, fewer parameters than original MLP, no decoder, no latent scaling, no privileged inputs.
3. **Craftax fresh evaluation COMPLETE:** one frozen model, 384 starts; no-op 137/384, random 65/384, fixed plan 358/384, receding plan 369/384. Displaced goals: 247, fixed plan 222/247, receding 232/247. Reference replay 382/384 under independent noise. Receding uses 3.5 times latent-transition compute, so do not isolate feedback as the gain's cause. `research/craftax_table_v2.json` and `artifacts/revision/craftax_goal_v2.json` contain exact evidence. Actual fixed-selection CPU-rendered frames are in `artifacts/revision/craftax_frames_v2`.
4. **PVE first attempt STOPPED at teacher gate:** teacher and independent auditor improve validation MSE by 8.74% and 8.98%, below the fixed 10% requirement. No scientific continuation or final-test evaluation; smoke artifacts are not results. `runs/revision/pve_seed0/manifest.json` records failure. Preserve it. A separately documented, adaptive second attempt uses an eight-frame GRU value teacher and the same strict gate, with no further retries if it fails. `pve_history.py` is independently reviewed and its smoke is running on e2e GPU0. If the gate passes, both continuation arms and five downstream policy seeds per arm are admitted regardless of result direction. Proper Value Equivalence remains the explicit inferred acronym; user has not clarified.

## Prepared follow-ups

- **Forecast readout READY:** `forecast_readout.py` and protocol independently reviewed. Both common and per-source train-only normalization analyses were declared before results; full real/predicted/copy transfer matrix, no future-survivor selection for hidden-gap negative control. MG1 after primary/grid, coordinated by experiment_design. Failed regularized probes do not establish erasure.
- **Target-gradient mechanism READY:** twelve fresh fits: gray-frame still/wind × joint/stop-target gradients × three seeds; same architecture/data/8,500 updates, both modes rerun. CPU full-gradient parity against original trainer is exact. Launcher reviewed and staged, starts MG0 only after primary+grid complete, with runtime-only admission and hard deadline. Fixed generic MLP coordinate MAE is primary, ridge secondary; all final checkpoints, all midpoint checkpoints if timing gate passes. Positive interaction may reflect harm in still rather than improvement in wind; report all raw within-world effects. A checked linear example shows target detachment need not eliminate nuisance preference.
- **Termination safeguard ADMITTED for review:** completed oracle-head runs show material postterminal effective mass (roughly 8–17% in several runs). Separate new immutable harness will add exact transitions + true termination + original learned rewards. Fifteen MLP and nine native-grid runs, same seeds/protocol as their counterparts. This tests the reserved sensitivity; old runs and source hashes remain unchanged. Admission is motivated by measured terminal-support exposure, not reward outcomes.
- Analysis script `studies/revision/analyze_matched.py` independently tested; campaign monitor invokes it as complete blocks arrive. Main inference must show model seeds separately, no best-checkpoint selection or family-level confidence claim from three fits.

## Important interpretation corrections

- Historical Craftax goal routine is open loop; goal success is nearby endpoint position, not achievements.
- Historical gate real-vs-dream evaluations are not paired, and PPO protocols differ.
- Head holdout rows overlap encoder training; separate eval.npz head audit is genuine held-out-data evaluation.
- A real-latent decoder failing on predicted latents does not prove information erasure.
- Per-frame gray noise plus wind gives high prediction loss, not an empirically low-loss nuisance optimum.
- Static nuisance shortcut is prior work (Sobal 2022); exact-Gaussian theorem is a narrow specialization, not a new mechanism.
- Legacy heads_audit imagined/planner scores misalign incoming actions and successor states. Real-latent head statistics used in V1 remain valid.
- Legacy residual quality MSE is measured with 8 NFE, policy rollouts use 4 NFE; timers distinguish fit-only and total cost.
- Do not use archived latent-amplification/subspace-gain experiments as proposed models. No new such runs.

## Remaining deliverables

- Freeze V1 with hashes and preserve it independently of V2.
- Complete/audit all admitted experiments, write V2 around what they actually establish, and generate updated figures/HTML in separate copies. `submission/v2`, `submission/shared-v2`, and `submission/html-v2` already exist; V2 is not frozen. New Craftax results are incorporated, all pending outcomes remain absent. Author notes are in `research/v2_outline.md`.
- Independent cold reviews of final V2 and numerical/format/anonymity audit. Report earned ratings honestly.
- Export anonymized submission bundle separately from internal research reports (which contain paths/VM identities). Builder `studies/revision/package_submission.py` and protocol are complete, tested, and preflight currently only lacks final V2 PDF. Add new finalized result reports explicitly to its allowlist/arguments; do not bundle third-party paper caches, models or datasets.
- Personal visual review of final PDF and HTML, including mobile and print.
- Deliver durable copy under original checkout `output/` while preserving original source; keep separate worktree and branch. Commit only appropriate source/artifacts, not copyrighted reading caches, datasets or checkpoints. No publishing/submission/push has been requested.
