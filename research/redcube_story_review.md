# Independent challenge of the red-cube storyline

Review date: 2026-09-20. Scope: existing source and saved evidence only. No jobs were launched, no model or environment was changed, and no new outcome is claimed.

## Recommendation

**Use the three-case narrative, but separate the observed failures from the exact stochastic-planning argument.** The current gray-frame-still / gray-frame-wind / gate selection can provide a clear progression of behavior. It cannot presently establish the stronger causal progression “successful planning / state information erased / correct representation but stochastic averaging causes control failure.” A shared CNN and decoded RGB remove a major actor mismatch, but do not by themselves close those gaps.

The smallest useful additions are:

1. A frozen-checkpoint, multi-goal MPC test in gray-frame-still, reporting actual cube trajectories and goal distance. Existing survival does not establish planning.
2. The proposed matched decoded-RGB PPO comparison, with **real-environment decoded-RGB positive controls** before attributing a gate failure to learned dynamics. Raw-pixel PPO is not a substitute for these controls.
3. The exact frozen-encoder variance audit already specified in `oracle_variance_design.md`. This is the clean causal experiment about discarded variance. It must retain its stated latent-goal-cost scope, including a possible null result.

Do not spend the remaining window adding wall colors, background noise to gate, or a new environment expressly chosen to make one method fail. Those would require newly matched training and turn a source-grounded revision into another benchmark-development cycle.

## What the three selected worlds actually establish

| Case | Existing evidence | Warranted statement now | Additional evidence needed for the intended stronger statement |
|---|---|---|---|
| Gray frame, still | Three grid JEPA seeds have real-latent position L1 0.1151 ± 0.0115 px. Historical goal-MPC survival is about 92%, and imagined PPO survival is 100%. | Current position is accessible and the tested control pipeline survives. | A goal-conditioned behavior test: state-blind oscillation also survives 100%, including the existing edge-start distribution. |
| Gray frame, wind | Three grid JEPA seeds have real-latent position L1 21.3295 ± 0.0571 px; historical JEPA policies fail and flow-pixel PPO survives about 97%. | The tested JEPA recipe has a reproducible position-readout failure and poor control under combined dynamic nuisance and wind. | Shared-actor control and reconstruction evidence can make the practical failure compelling. They still cannot prove information-theoretic erasure or an optimized nuisance-only solution. Final prediction loss near 0.733 is high. |
| Gate | A wall is accurately reconstructed from real future encodings; the same decoder largely loses the common wall from predicted latents. Historical decoded-real CNN succeeds in 2/9 policies, both on encoder seed 1. Historical flow-pixel PPO succeeds once at about 56%; the often quoted 98% is a different latent-policy run and is unstable across policy seeds. | There is a forecast/readout failure in a task requiring feedback to newly revealed gaps. Some real encoded observations support successful control. | Reliable same-actor control from real decoded observations, then a learned-model deficit under the same training package. A separate intervention is needed to attribute that deficit specifically to stochastic averaging. |

Sources: `research/redcube_audit.md`; `research/evidence-audit.md:15–70`; `runs/stoch/summary.json`. These summaries distinguish model seeds from policy seeds and should remain the source of historical qualifications.

### Case 1: surviving is weaker than moving toward a requested goal

The base environment has four 3-pixel moves and no motion when wind is disabled (`studies/redcube/env.py:24–32,99–107`). Alternating left/right keeps a cube in a two-position orbit. Starts are far enough from the boundary that this succeeds without reading a single pixel. The existing “edge” reset still leaves enough clearance for this orbit; its source comment claiming that position feedback is necessary is contradicted by the measured baseline (`studies/stoch/env.py:96–107`; `research/evidence-audit.md:70`).

A survival bar labeled “JEPA plans” would therefore repeat the same problem the user identified in the Craftax local-goal narrative. A trajectory to several independent requested targets is the inexpensive repair. This remains a deliberately elementary planning test; do not dress it up as long-horizon reasoning.

### Case 2: dynamic gray noise is not the static-nuisance theorem

The most reproducible empirical failure is **per-frame** gray noise plus stochastic wind. The Telegram counterexample uses **episode-static** nuisance, and produces a low-loss nuisance-only optimum. The two cannot be described as the same mechanism. In the empirical grid JEPA, making gray noise persistent actually restores subpixel position readout under wind. This is a useful and surprising result, not a detail to omit.

The clean causal description of the four gray cells is an interaction between nuisance persistence and stochastic motion under a fixed training recipe. It is not proof that the model has globally minimized its objective by rejecting unpredictable state. Likewise, a failed readout does not establish absence of information from every possible decoder.

The paper can show one honest, direct picture: true frame; reconstruction from the current observation; predicted future. If the cube is already missing from the current reconstruction, say that the fitted observation interface fails to expose it. Add one behavior bar and put the full probes in the appendix. A post-hoc decoder cannot certify information erasure; its role is to make the measured failure visible.

### Case 3: the gate is stochastic, but the observed missing wall is not its conditional pixel mean

At spawn, the gap is independently top or bottom. The wall moves deterministically and reveals the gap once its left edge reaches x ≤ 40 (`studies/stoch/env.py:32–35,68–93,121–139`). Before reveal, histories do not identify the random gap. After reveal, ordinary feedback can use it. A successful policy can wait farther left to gain reaction time; it need not predict the unrevealed gap.

Both future gap outcomes contain the same solid wall pixels outside the two possible openings. Their exact pixel mean preserves those common pixels. Therefore the observed disappearance of the common wall through a decoder is **not the cartoon of a correct predictor averaging the two openings**. Latent averaging followed by a nonlinear decoder, underfitting, distribution shift and fitted reward/termination heads remain possible explanations. Current reward heads already underpredict wall death on held-out real latents (`research/evidence-audit.md:55`).

The latest forecast readout audit also does not prove wall erasure: some real-latent linear positive controls fail, and the seed-2 adapted forecast readout has large tails and weak tracking. Keep the observed statement concrete: the learned rollout does not preserve a control-relevant wall through the tested observation interface. Do not rename it a demonstrated Bayes-mean failure.

## Minimum additional evaluation

### A. Small goal-conditioned positive control, without fitting anything

Use the existing three gray-frame-still grid JEPA checkpoints. Freeze all encoder, predictor and decoder parameters. Reuse the existing exhaustive latent-goal MPC operator from `studies/stoch/control.py`, correcting only evaluation bookkeeping in a new standalone script. Do not use the old `studies/redcube/control.py` loop unchanged: its warm-up tracks only the last done flag and it continues stepping after termination (`studies/redcube/control.py:50–65,99–108`).

Suggested fixed initial protocol:

- 32 fresh independent eight-frame histories per checkpoint, generated by seven random actions, admitted using past survival only. Share physical histories and all following environment keys across policies and checkpoints.
- Four fixed image goals with top-left coordinates `(8,8), (8,48), (48,8), (48,48)` for **every** history. Render the goal with a fixed independently drawn gray background. No future outcomes or model costs determine goal choice.
- Receding-horizon latent-goal MPC with horizon 3, all 64 action sequences, 32 controlled steps. Use the original 512 latent coordinates without whitening, gains, replication or state heads.
- Paired references: uniform random actions, fixed left/right alternation, and an explicitly privileged physical-state greedy target controller as a ceiling. A control with the requested goal randomly permuted across the four targets checks that movement follows the supplied goal rather than a fixed attractor.
- Primary endpoint: physical distance to the **requested** goal at each step, with deaths shown separately. Supplementary success is entry within one action length, fixed at 3 px in max-coordinate distance, before death. This tolerance reflects the 3-pixel action lattice and must be fixed before results. Report initial distances and use all targets, including initially easy cases.
- Preserve the actual executed trajectories, frames, requested target, actions and terminal masks. Select illustration by a fixed context ID; display all four target continuations from that same context. No best-success montage.

The physical target is used for evaluation and the privileged ceiling only; the JEPA planner receives an image goal. This is an image-goal evaluation, not a change to the native survival reward or training. A successful result supports “the representation and predictor can support elementary goal-directed control.” If it fails, report it rather than changing horizon, goal bank or checkpoints until it succeeds.

This requires no training and roughly 3 × 32 × 4 × 32 × 64 short candidate evaluations before tree-prefix sharing. A small timing fixture should determine the batch size; do not promise a runtime without measuring it. If necessary reduce the fixed number of independent histories before any scientific outcomes are inspected, not the hard targets after observing failures.

### B. The shared-CNN comparison needs a real decoded-observation ceiling

For each family and world, the same actor must consume the same type of input during its real evaluation and its training: posterior-decoded true RGB when interacting with the simulator, and decoded predicted RGB in imagination. Keep the stock CNN/GRU, native reward, action space, starts, optimizer package, final evaluation keys and total optimization/transition budget fixed. Include all decoder/head fitting and flow inference costs in the compute table. This compares complete model recipes; it does not isolate the training objective.

The critical two-axis comparison is:

| Training transitions | JEPA observation interface | Flow observation interface |
|---|---|---|
| Exact simulator and native reward/done | Frozen JEPA encode/decode → same CNN | Frozen flow posterior encode/decode → same CNN |
| Learned model and its fitted reward/done | JEPA predicted latent → frozen decoder → same CNN | Sampled flow future → frozen decoder → same CNN |

For the gate, run the exact-simulator / decoded-JEPA positive control before adding more learned-model policy replications. Use three policy seeds, not a selected successful seed. Existing raw-pixel PPO is a useful task/optimizer reference, but it does not show that the JEPA decoded interface is sufficient for the selected CNN and budget. If decoded-real PPO remains unreliable, the strong case-3 attribution must stay unresolved.

A large true-vs-imagined gap with the same decoded interface locates the failure in the **learned imagination pipeline**, which still includes transitions and reward/done predictions. It does not by itself isolate stochastic transition prediction. The internal head audits can remain in the appendix while the main figure says exactly this.

For gray-frame-wind, if both real-decoded and imagined JEPA control fail while their flow counterparts succeed, the observational interface is already a practical bottleneck. This supports a useful representation failure under the selected learned interface, without inventing a theorem that no policy could recover state.

Use survival probability and the actual physical trajectories as the visible endpoints. Gate wall counts can supplement survival, but do not claim a wall was safely passed when the counter increments on the same terminal transition. The native reward is unchanged and death at the cap remains failure.

### C. Run the frozen-encoder exact variance audit; do not overextend it

`research/oracle_variance_design.md` is the smallest strong causal test already designed. It compares the exact next-latent mean against the exact finite conditional distribution, under one fixed encoder and one fixed squared latent-goal objective. No fitting, decoder, reward head or PPO optimizer can explain its action-ranking discrepancy.

Its negative control matters: in the actual red-cube wind dynamics, conditional **position** covariance is independent of action. Thus exact mean-position and expected squared physical-goal costs give the same one-step ranking. If a nonlinear learned encoder produces action-dependent latent variance, a latent ranking discrepancy is possible; it is still not automatically a physical control improvement. Show both, including a null or adverse physical effect.

Keep the user's two exact mathematical examples immediately adjacent to this audit. They establish the general limitation under ideal training; the learned-representation audit connects the omission to an actual checkpoint if it is positive. The gate recipe comparison then demonstrates practical consequences of a learned imagination pipeline without claiming the oracle audit causally explains every gate failure.

## Decision rules for the rewritten main text

- **If shared-CNN JEPA improves:** retain the improvement prominently. The previous comparison had an interface problem. A fair paper should make JEPA's strongest ordinary implementation the headline comparator.
- **If flow remains unstable across policy seeds:** show the full range and weaken “flow succeeds” to the observed rate; one successful seed is not reliable success.
- **If decoded-real JEPA fails:** case 3 cannot claim that a useful-for-control observation representation has been demonstrated for that interface and budget. Reconstruction is weaker evidence.
- **If oracle mean/full-distribution choices agree:** the analytical counterexample still stands, but this particular learned encoder/world did not reproduce its one-step ranking consequence. Do not search a new goal bank until one does.
- **If goal MPC succeeds:** use a single four-goal trajectory panel to establish elementary planning competence. Do not use survival alone.
- **If none of the additional tests support the requested empirical causal chain:** write the stronger true paper: exact limitations of point-prediction planning, a replicated noise–dynamics interaction, and a carefully bounded practical control failure. Do not imply a completed disproof of all JEPA world models.

## A compact main-paper structure

1. Start with the ideal-training action-ranking counterexample, drawn in red-cube language and labeled analytical. Its point is concrete: predicting the mean correctly need not choose the better action.
2. Show one red-cube positive-control trajectory panel: the same observation, four requested target images, four actual paths.
3. Show the minimal nuisance/dynamics intervention: the four gray conditions as a small 2×2 figure, with one representation picture and one physical behavior endpoint per cell. This preserves the essential fact that persistent noise can help this particular recipe.
4. Show gate observations, predicted futures and actual paths for both model families. State the evidence level as a learned-imagination failure unless the stronger controls warrant more.
5. State the frozen-encoder variance audit and the recursive-mean identity as the reason the failure is not fixed merely by obtaining a better conditional-mean estimate. Keep this logical statement separate from causal attribution of the empirical gate result.

A few charts are enough. The proof, source hashes, raw trials, task-head diagnostics, historical adverse flow cells and complete seed tables belong in the appendix/artifact. Their details should protect the story, not become its topic.
