# Existing evidence audit for the ICLR revision

Audit date: 2026-09-19. Original checkout: `/Users/rldoeswork/Documents/rl/tinycraft-jepa`. This audit reads source, raw JSON/JSONL, training logs and fresh remote metadata; it does not rerun experiments or alter original files. Canonical machine ledger: `research/evidence-ledger.json`. Companion reports: `craftax_audit.md`, `redcube_audit.md`, `redcube_diagnostics.md` and their raw extraction JSON files.

## Verdict and strongest verified storyline

The evidence supports a paper about **distinct gaps between preserving observations, predicting representations, and using predictions for control**. It does not support an objective-only causal claim that all JEPAs fail under stochasticity, or that distributional dynamics alone solves control. The cleanest empirical result is the three-model-seed gray-noise interaction in a fixed 512-dimensional spatial JEPA, accompanied by the three-model-seed gate wall-loss diagnostic. Craftax adds a useful harder-environment motivation with sharply bounded scope.

An effective hard-to-easy progression is:

1. **Craftax:** a model reaches nearby goal positions in a short open-loop test, yet its learned-head reward-MPC pipeline performs below uniform random. This motivates asking why prediction and control disagree, rather than asserting an explanation from the Craftax result.
2. **Delayed-reveal gate:** the real encoder retains a moving wall under a fitted decoder, predicted latents lose its decoded image, and imagined PPO is weaker than real-transition PPO on the same frozen encoders. A CNN acting on decoded real latents sometimes solves the gate. Heads and PPO optimization are measured competing explanations.
3. **Noise-factorized cube:** fixed architecture, data size and update budget expose a three-seed interaction between independent per-frame noise and stochastic wind. Probe error is already high on true encoded frames in the failing cell. This separates representation accessibility from the gate's forecast/readout issue.
4. **Theory:** non-collapse is not a sufficient guarantee of control-relevant information or its scale. The empirical study exhibits compatible failure modes, without claiming the abstract construction explains every trained model.

## Main gate endpoints, recomputed from final ledger rows

Every endpoint below is 256 fresh real episodes from reset, cap 128. Final checkpoint means the last saved ledger row, not the best row in `final.json`. The saved `runs/stoch/summary.json` is exactly equal to a fresh non-mutating call to `studies/stoch/summarize.py:collect()`.

| Pipeline | Model fits | Policy runs | Walls, mean ± policy-run SD | Survival, mean | Discovery count (>50%) |
|---|---:|---:|---:|---:|---:|
| JEPA original predictor | 3 | 9 | 0.27257 ± 0.21535 | 0.4340% | 0/9 |
| JEPA residual mean of 8 | 3 | 9 | 0.26215 ± 0.29705 | 0.6944% | 0/9 |
| JEPA residual sample | 3 | 9 | 0.26128 ± 0.32239 | 0.8247% | 0/9 |
| Flow sample | 1 | 6 | 1.45443 ± 1.22912 | 19.0104% | 1/6 |
| Flow mean of 8 | 1 | 3 | 0.88281 ± 0.13871 | 2.7344% | 0/3 |
| Real pixels, CNN | — | 3 | 2.91016 ± 1.65098 | 65.2344% | 2/3 |
| Real pixels, MLP | — | 3 | 0.91276 ± 0.08028 | 3.7760% | 0/3 |
| Real frozen JEPA latent, MLP | 3 | 9 | 0.93533 ± 0.03655 | 3.8628% | 0/9 |
| Real decoded JEPA latent, CNN | 3 | 9 | 1.56510 ± 1.26197 | 21.9618% | 2/9 |
| Real decoded JEPA latent, MLP | 2 | 6 | 0.91667 ± 0.08256 | 3.4505% | 0/6 |

Use machine-ledger exact floating point values as authoritative; prose rounds to two decimals. The three JEPA predictor variants pooled have 0.26533565 walls per episode (27 policies), versus 0.93532986 for real frozen latent (nine policies), a descriptive deficit of 0.66999421. This pooling is useful as a visual summary but does not turn 27 policy runs into 27 independent models.

Per-model gate wall means make replication structure explicit:

| Model seed | Original | Residual mean8 | Residual sample | Real latent MLP | Real decoded CNN |
|---|---:|---:|---:|---:|---:|
| 0 | 0.29818 | 0.30859 | 0.27734 | 0.91406 | 0.86068 |
| 1 | 0.28255 | 0.22396 | 0.015625 | 0.93750 | 2.77604 |
| 2 | 0.23698 | 0.25391 | 0.49089 | 0.95443 | 1.05859 |

Both decoded-CNN discoveries occur on encoder seed 1. Flow's single successful policy is seed 0 (98.05% survival, 3.94531 walls); the other five sampled policies and all three mean8 policies are much weaker. All flow policies share the exact same binary checkpoint across VM path aliases, SHA256 `72b829ea8cfb5f4bf21f3b8c0ce8ffe53f34dd5aa67bad001bb8d4cd0bab27e9`. Do not imply a family-level generative advantage based on one fitted checkpoint.

## Main claim inventory and interpretation

**Encoder information exists in the gate.** `runs/stoch/gate/latent_check{,_s1,_s2}.json` each uses the same 64 pre-reveal contexts; re-simulated recorded futures match pixels exactly in all contexts. Shared-wall intensity decoded from real future encodings is above .99 at all eight steps for all seeds. CNN control from decoded latents adds evidence beyond reconstruction: two successful seed-1 policies. This shows useful information remains in at least one encoder under a powerful supervised readout; it does not show every encoder preserves all task state.

**The original gate prediction does not preserve the decoded wall.** At prediction step 2, shared-wall decoded intensity is 0, 0 and .0172204 for seeds 0,1,2. Seed 2 retains .6823 at step1 then loses it. The 64-context diagnostic conditions on the recorded branch surviving the full horizon; 40 counterfactual branches also survive. Forty of the 64 contexts remain unrevealed at their first future step. Report this as a forecast/readout failure, not proof predicted latent contains literally no wall information. A decoder trained on real latents can behave unpredictably off manifold.

**A pure gap-mixture cartoon does not explain the pixel observation.** Both gap futures share solid wall pixels, so a pixelwise conditional mean would preserve those shared pixels. This observation does not rule out latent averaging effects, a poorly fitted predictor, or a decoder extrapolation effect. The current residual sampling experiment is not a certified accurate conditional distribution and therefore cannot rule out distributional objectives generally.

**Scale comparisons are descriptive.** Seed0 first-step squared wall-present/absent latent distance .035279 versus total teacher-forced residual .144774 comes from `latent_check.json`. The residual contains all latent coordinates; comparing it with a single feature's effect cannot prove that amplitude causes prediction or control failure. Latent scales across separately learned models are not comparable.

**Heads are a substantial competing explanation.** On 25,000 truly held-out gate transitions, seed0 termination AUC is .648874 for wall death versus alive, .963277 for border death versus alive. Mean reward predictions are −.054056 and −.890718 respectively, versus true −1. Seed1 is similarly weak for walls (.645614 AUC, −.049834 reward); seed2 improves to .759248 AUC and −.233525 reward. Source: `heads_audit_tile_factored{,_s1,_s2}.json`, each 610 wall deaths and159 border deaths. A failure in learned task heads already exists before long rollouts.

**Extra JEPA optimization does not rescue the tested policy.** The x2/x4 runs increase model fitting from8500 to17000/34000 steps for one encoder seed, then evaluate one policy seed per predictor variant. They bound the tested budget explanation but are not an independent three-model replication. Exact endpoints are in the ledger/raw collection.

**Craftax demonstrates a dissociation, not a solved task.** New512d tile model:121/128 reached six-action nearby goals, versus24/128 random. Reward CEM-MPC averages .47344 return over100steps versus1.44531 random, with128starts and one model seed. The goal routine plans once then executes the entire sequence: name it short-horizon **open-loop** goal reaching. Targets are reachable random-rollout snapshots; success is player endpoint position, not entire scene or achievements. Heads generalize weakly, so this cannot isolate a dynamics defect. Full exact table and model identity are in `craftax_audit.md`.

**Cube evidence is specific and replicated.** Real-latent position L1 for gray frame+wind21.3295±.0571px across three model seeds; frame+still .1151±.0115; episode+wind .6676±.0486; episode+still .7637±.0680. The failed cell's prediction losses remain near .733, unlike successful cells near .001, so calling it an optimized low-loss nuisance shortcut is false. The probe is supervised and nonlinear; large error establishes failure of that readout, not information-theoretic erasure. Some architecture variants and flow itself have adverse cells; full109-arm extraction prevents selecting only favorable rows.

## Material corrections to existing prose and reproducibility

1. **Final real-vs-imagined gate evaluations are not all paired.** `rl.py` draws evaluation keys from the evolving training RNG. Real training has an extra initial-reset split and ends at183iterations; dream training ends at150updates. The appendix claim of shared keys across all arms is unsupported. Matching seeds among identical dream protocols can pair keys; real-vs-dream historical final endpoints do not. A fixed-key reevaluation should save episode arrays for proper paired uncertainty.
2. **Readout training is not a separate behavior-policy seed.** `train_arms.sh` passes the same training NPZ to WM fitting and `probe.py`. The head/decoder use the first80k frames; head holdout uses the next200time rows. Those rows are held out from heads, not encoder fitting. The separate `heads_audit` on eval.npz is the proper held-out-dataset metric. Craftax's80k frames means320time rows×250streams, whereas gate means800×100.
3. **Uniform ten-minute/data-budget language is inaccurate.** Gate grid log reports573s at last update and586.5s total; flow563.6s. Craftax grid604.5s total. Original black/color cube datasets have100k transitions, gray cube datasets500k. Some original flow fits take710s. State actual budgets per family and do not call all models strict≤600s.
4. **PPO protocol is not fully matched.** Main real policies use1,499,136real steps, imagined policies1,228,800imagined steps. Starts, collection length, GAE context and update organization differ. Model reward/done probabilities replace true outcomes. Thus the0.67-wall gap is a pipeline comparison.
5. **Mean8 flow consumes more inference compute.** Its completed150-update runs were rerun with60-minute caps, versus12-minute caps for samples. The old12-minute endpoints remain separately archived. Match transitions if that is the intended estimand, but show compute costs.
6. **Still-world survival is an invalid representation certificate, including current edge starts.** Alternating state-blind actions survive100% in all three still-world `baselines_edge.json` files. Edge distances3–8px and3px moves permit safe oscillation. The source comment claiming edge starts require position feedback is contradicted by the measured baseline.
7. **Custom gFVD is not standard video FVD.** It uses an untrained fixed random CNN over short clips. It cannot alone substantiate control or physical-state quality. Name it a random-feature Fréchet proxy or define it explicitly.
8. **Historical large-grid Craftax models must stay separate.** Old `s0_grid` uses2592 latent scalars, new `tile_factored`512. Mixing their goal, reward, or prediction results as one model would be false. Avoid old inflated-latent variants in the core matched comparison.
9. **Existing task-specific amplification runs are excluded from the proposed model evidence.** Twelve amp4/amp16 and24wall/random-subspace gain4 gate policy runs are preserved in the internal ledger with explicit exclusion flags. They are not a general objective improvement and do not support the paper's core conclusions.
10. **There are two summarizer copies.** `runs/stoch/summarize.py` lacks current round2 support. Use `studies/stoch/summarize.py`. Current `paired.tex` manually drops simulator/center rows from its generator; regenerate the new paper's tables from the new ledger rather than assuming checked-in TeX is a complete generated artifact.
11. **PPO final.json is not the final endpoint.** It records a best row. Fresh VM pull found local mean8_p1/final.json stale at79updates while remote says150; the local and remote ledger.jsonl are byte-identical and already contain the correct150 endpoint. Use last ledger row. All fresh remote JSONL endpoints match originals; remote-only final.json files do not create new runs.
12. **Hashes need artifact-level tracking.** Ledger/config/result hashes are in this audit. GPU inventory has independently collected remote checkpoint metadata and hashes, including flow alias confirmation. Source/data hashes embedded in checkpoints are evidence of their stated provenance, not proof the current dirty source equals training code. Keep train-source snapshots and exact checkpoints with the submission artifact.

## Efficient next evidence, if additional runs are available

The highest-value steps are fixed-key final-policy evaluation with per-episode outcomes; direct forecast diagnostics with a probe trained on predicted latents; objective ablations with the same512-dimensional grid and matched update budget; and Craftax goal tests stratified by actual displacement, with no-op and independent-execution-noise controls. Several independent model fits matter more than many additional policy seeds on one checkpoint. No result should be rewritten as stronger because its plot becomes more attractive.

## Artifacts produced

- `evidence-ledger.json`:146 policy runs,66 main-paper endpoint runs, full raw configs and final metrics, explicit policy/model seeds, source hashes, per-model aggregates, three-seed gate diagnostics,11 interpretation/reproducibility issues, canonical flow alias.
- `craftax_evidence.json` and `craftax_audit.md`: scalar results, protocol, source hashes, old/new model distinction and bounded inclusion wording.
- `redcube_evidence.json`, `redcube_table.md`, `redcube_audit.md`:109 raw architecture-condition rows and all main caveats; `redcube_diagnostics.md` audits older probe diagnostics.

Memory was consulted only to locate prior claim-scoping cautions, not as current empirical evidence. All numerical findings above are from the current artifacts or freshly checked remote copies. Relevant memory registry lines75–81 identify the earlier danger of presenting planned Craftax outcomes as measured; the current audit supersedes those stale empirical-status notes.
