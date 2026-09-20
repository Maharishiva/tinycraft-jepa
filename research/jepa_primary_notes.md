# JEPA primary-source reading notes (19 September 2026)

Read status: downloaded full arXiv HTML; close-read methods, objective equations, planning setup, conclusions, and the relevant theory/implementation appendices. Inspected experimental findings and limitations. This is a targeted full-text review, not an assertion that every proof, benchmark table, or appendix paragraph was exhaustively audited. Latest versions were checked against arXiv submission histories. No model was run or original research file edited.

## LeWorldModel — Maes et al., arXiv:2603.19312v3 (3 June 2026)

Sources: [§3.1](https://arxiv.org/html/2603.19312v3#S3.SS1), [§3.2](https://arxiv.org/html/2603.19312v3#S3.SS2), [§4.2](https://arxiv.org/html/2603.19312v3#S4.SS2), [Appendix D](https://arxiv.org/html/2603.19312v3#A4), [§6](https://arxiv.org/html/2603.19312v3#S6).

- Directly relevant recipe: jointly learned encoder/predictor, teacher-forced next-embedding squared loss, no stop-gradient, EMA, rewards, reconstruction, or pretrained encoder. Eq. (3): `L = ||pred(z_t,a_t)-z_(t+1)||² + λ SIGReg(Z)`. Defaults: λ=0.1; 1,024 random projections. SIGReg tests projected embeddings against a standard Gaussian. Encoder uses one compact CLS embedding (default dimension 192).
- Planning minimizes terminal squared latent distance to an encoded image goal, via CEM. Appendix D uses horizon 5 with action-block frame skip 5, hence 25 environment steps; it executes the complete planned sequence before replanning.
- Tests cover TwoRoom, PushT, Reacher and single-cube OGBench. TwoRoom underperformance is already acknowledged, with Gaussian-prior/low-intrinsic-dimension mismatch offered as a hypothesis, not established cause. Appendix F reports strong TwoRoom state probes despite weaker planning.
- Inference for our scope: this is the closest objective family. Gaussian occupancy alone should not be described as guaranteeing state sufficiency. Its existing TwoRoom probe/planning dissociation makes “decodable but not usable for control” an existing phenomenon; our evidence must identify what additional mechanism is established.

## LeJEPA — Balestriero and LeCun, arXiv:2511.08544v3 (14 November 2025)

Sources: [§3.1](https://arxiv.org/html/2511.08544v3#S3.SS1), [§3.2](https://arxiv.org/html/2511.08544v3#S3.SS2), [§5.1](https://arxiv.org/html/2511.08544v3#S5.SS1), [Appendix A](https://arxiv.org/html/2511.08544v3#A1).

- This is primarily representation pretraining across image views, not an action-conditioned stochastic-control result. Its implementation aligns each view embedding to the mean of global-view embeddings and adds per-view SIGReg. No teacher/stop-gradient. Code weights are `(1−λ) similarity + λ SIGReg`; Eq. (9) uses a different relative-weight parametrization.
- Linear-probe theory compares matrices with identical column spans and equal total energy. Thus the comparison holds captured information fixed and changes its geometry.
- Nonlinear-probe analysis uses smooth embedding densities and smooth target functions of the embedding. Radius-kNN analysis additionally invokes a mean-zero isotropic gradient prior; kernel analysis bounds integrated bias through Fisher information under covariance constraints. These are substantive assumptions, not task-free guarantees of preserving every possible label in the encoder.
- Inference for our scope: an exactly Gaussian nuisance-only representation does not contradict SIGReg's distribution-matching property. It shows that good marginal geometry and retained control information are separate requirements. State this distinction explicitly; do not frame a stochastic-temporal result as disproving the image-pretraining experiments.

## V-JEPA 2 — Assran et al., arXiv:2506.09985v1 (11 June 2025)

Sources: [§2.1](https://arxiv.org/html/2506.09985v1#S2.SS1), [§3.1](https://arxiv.org/html/2506.09985v1#S3.SS1), [§3.2](https://arxiv.org/html/2506.09985v1#S3.SS2), [§4.3](https://arxiv.org/html/2506.09985v1#S4.SS3).

- Eq. (1) pretraining uses L1 masked-feature prediction with a stop-gradient EMA target. It is not jointly differentiated squared-loss prediction.
- V-JEPA 2-AC freezes that visual encoder, then learns an action-conditioned predictor using end-effector state and inferred end-effector actions from under 62 hours of DROID video. Eqs. (2)–(4) combine L1 teacher-forcing with a two-step recurrent rollout loss. Planning uses L1 terminal feature distance and CEM.
- Robot experiments concern image-goal reaching/grasping and pick-and-place; pick-and-place uses two intermediate image subgoals. Authors report camera-position sensitivity and manually selected a working camera placement, limited long-horizon prediction/search, and dependence on visual goal specification.
- Inference for our scope: motivate the broader appeal of latent prediction using this work, but exclude it from a theorem about jointly trained squared-error encoders. L1 targets medians rather than conditional means; frozen features also remove the studied training-time incentive to erase information. Neither fact guarantees success under stochastic dynamics, but the present squared-loss proof does not establish its failure.

## What Drives Success in Physical Planning with Joint-Embedding Predictive World Models? — Terver et al., arXiv:2512.24497v4 (2 September 2026)

Sources: [§3](https://arxiv.org/html/2512.24497v4#S3), [§5.1](https://arxiv.org/html/2512.24497v4#S5.SS1), [§6](https://arxiv.org/html/2512.24497v4#S6), [Appendix D](https://arxiv.org/html/2512.24497v4#A4). ArXiv history records TMLR acceptance in the v3 update.

- Study holds pretrained visual encoders frozen while training action/proprioceptive embeddings and predictor with MSE. Ablates planning, encoder, conditioning, context and rollout loss; therefore it does not test joint visual-encoder information selection.
- Critical novelty boundary: §6 already identifies deterministic MSE prediction of multimodal futures as a conditional-mean limitation and proposes stochastic latent/diffusion extensions. Do not sell this observation alone as new.
- Appendix D analyzes propagation of one-step error δ through a Λ-Lipschitz predictor: upper bound `δ(Λ^H−1)/(Λ−1)` (or `Hδ` at Λ=1). This is different from constructing a Bayes-optimal one-step mean whose iteration is not the multi-step conditional mean.
- Scope of evidence: DROID score is offline planned-action agreement on recorded real-robot videos, not real-robot execution success. Simulation tasks supply success rates. The final models use three training seeds; uncertainty accounting distinguishes seeds from averaging evaluation epochs.
- Inference: cite this as direct recent prior work; sharpen contribution around feature selection despite anti-collapse and controlled, quantified planning failures, not a blanket critique of all JEPAs.

## Concrete manuscript corrections and experimental implications

1. `studies/stoch/paper/sections/related.tex` currently groups PLDM with frozen-feature systems. Correct this: LeWM explicitly identifies PLDM as end-to-end learned representation/predictor training with a multi-term VICReg-derived objective.
2. Separate three claims throughout: retained state information; learned transition statistics; compatibility of the planner's cost/rollout with the task. A successful probe demonstrates decodability and cannot by itself diagnose why a planner fails.
3. Treat the nuisance-only Gaussian construction as contextual groundwork until its novelty is checked. A subsequent independent theory audit reported that Sobal (2022), §2.1, already gives a static-Gaussian-nuisance/identity-predictor construction; merely specializing it to SIGReg is not a new mechanism. Distinguish any new result from that prior construction, and distinguish empirical representation omission from usable-but-poorly-scaled features. Avoid claiming all runs erase state when decoded-state control succeeds.
4. Before any further training, freeze the evaluation contract: identical data, observation preprocessing, native latent size, training budget, goals, planning compute and task labels. Treat a stronger generative model as a feasibility reference unless the comparison isolates the intended objective change.
5. A mean-vs-distribution experiment should compare transition objectives with the same informative encoder and downstream cost, then separately test joint learning. Otherwise improved control cannot be assigned specifically to stochastic prediction.
6. A new intervention needs a clean paired design. Adding static visual nuisance and dynamic uncertainty together would make interpretation harder; vary them separately before their combination. Color improves figure legibility but is not an experimental contribution.

Local full-text HTML copies: `/tmp/arxiv-{2603.19312v3,2511.08544v3,2506.09985v1,2512.24497v4}.html`. Plain-text extracts exist beside them; equations should be checked against HTML because the first extraction pass can strip angle-bracket content from TeX.

Memory used only to recover intended claim scope; all literature claims above were checked against current primary sources. Memory registry anchor: MEMORY.md:75-76; rollout ID 01a0b6fb-e660-7053-a7c9-49280c7e2f93.
