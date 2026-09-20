# Literature and objective review — 19 September 2026

This review uses primary papers and current arXiv version histories. It is a targeted full-text review of the relevant objectives, assumptions, experiments and limitations; it is not an exhaustive audit of every cited appendix. Original repository files and GPU workloads were not modified. Companion notes: `/tmp/tinycraft-jepa-2026-primary.md`; latest-overlap addendum is being prepared separately.

## Decisions for the manuscript

The paper should ask **which control-relevant information survives a reward-free predictive learning and imagination pipeline**, with distinct tests for encoding, prediction, readout and policy learning. It should not claim that all JEPA models fail, that Gaussian regularization promises control sufficiency, or that conditional-mean averaging is a new discovery.

The strongest source-supported comparison is to the **jointly trained, squared-error, Gaussian-regularized LeWorldModel recipe**. Frozen visual foundations, L1 objectives, stochastic predictors, and task-supervised dynamics are different settings. A small generative reference can establish feasibility on a task, but not a family ranking when supervision, context, architecture and head fitting differ.

Two novelty threats require explicit attribution: LeWM already reports good TwoRoom probes with weaker planning; Terver et al. already discuss conditional means under multimodal dynamics. The theory-review agent independently reports that Sobal et al. (2022) already gives a Gaussian static-nuisance/identity-predictor shortcut. Therefore the contribution must rest on the new controlled experiment, quantitatively located pipeline failure, or a genuinely new theorem beyond that construction. These points do not make the experimental question uninteresting; they constrain the novelty language.

## Current primary-source map

| Source | Verified distinction and relevance | Read depth |
|---|---|---|
| [LeWorldModel, Maes et al., 2603.19312v3](https://arxiv.org/html/2603.19312v3) | Joint encoder/predictor training with next-latent squared loss plus SIGReg; no reward/reconstruction/EMA/stop-gradient. Goal-image CEM planning. This is the direct objective comparator. Appendix F already distinguishes state decodability from TwoRoom planning performance. | Methods, main findings, relevant appendices closely read by subagent. |
| [LeJEPA, Balestriero and LeCun, 2511.08544v3](https://arxiv.org/html/2511.08544v3) | Distribution/geometry arguments are conditional on retained information and target assumptions. Linear result fixes feature span and energy; nonlinear results require smoothness. The theorem does not state that every reward-relevant variable will be retained by arbitrary temporal learning. | Theory assumptions, objective, relevant appendix closely read by subagent. |
| [V-JEPA 2, Assran et al., 2506.09985v1](https://arxiv.org/html/2506.09985v1) | L1 masked prediction with EMA/stop-gradient pretraining; frozen encoder for action-conditioned L1 teacher forcing plus two-step rollout loss. Its model is outside the present joint squared-loss theorem. Robot results include intermediate image subgoals and reported camera sensitivity. | Methods, planning and limitations closely read by subagent. |
| [What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?, Terver et al., 2512.24497v4](https://arxiv.org/html/2512.24497v4) | Frozen pretrained encoders; MSE predictor; controlled architecture/planner study. Section 6 explicitly discusses conditional means of multimodal futures and stochastic extensions. Appendix D gives ordinary Lipschitz rollout-error accumulation. DROID results are offline action agreement, not real execution success. | Methods, limitations, rollout theory closely read by subagent. |
| [PLDM, Sobal et al., 2502.14819v4](https://arxiv.org/html/2502.14819v4#S3.SS3) | Encoder and predictor are learned using multi-step latent similarity and VICReg-derived/inverse-dynamics terms, then frozen for goal planning. Current `related.tex` incorrectly groups it among frozen-feature training systems. Section 3.1 explicitly limits its transition model setting to deterministic dynamics. | Sections 3.1–3.3 and Appendix D inspected directly. |
| [Hallucination in World Models is Predictable and Preventable, Hansen and Wang, 2606.27326v1](https://arxiv.org/html/2606.27326v1) | Separates tokenizer, action-insensitive dynamics, and scene-diverging rollout failures in a 350M generative model. Uses action shuffling and reconstruction/rollout diagnostics; relates errors to coverage. Relevant precedent for locating failure within a pipeline rather than interpreting a single aggregate score. | Methods, diagnostics, evaluation contract, limitations read; not every benchmark result. |
| [Var-JEPA, Gögl and Yau, 2603.20111v2](https://arxiv.org/html/2603.20111v2) | A coupled latent-variable ELBO includes context and target reconstruction terms, fixed-prior KL terms and a conditional target-prior KL. Thus it is not a decoder-free stochastic replacement of the existing two-term LeWM objective. Evaluation is tabular/synthetic rather than TinyCraft control. | Generative factorization, ELBO, synthetic ablation inspected. |
| [VJEPA, Huang, 2601.14354v1](https://arxiv.org/html/2601.14354v1) | Proposes latent predictive NLL plus target-posterior KL and a probabilistic control interpretation. Section 6.3 explicitly requires control-relevant predictive sufficiency; its broad abstract is not an unconditional guarantee from the objective. The empirical example is a small linear noisy-signal system. | Sections 3–6, relevant theorem and toy setup inspected; no complete proof audit. |

Latest versions were verified: LeWM v3 (3 June 2026), LeJEPA v3 (14 November 2025), PLDM v4 (29 October 2025), Terver et al. v4 (2 September 2026; arXiv history records TMLR acceptance in v3). Preserve publication dates rather than relabeling all current work as 2026 papers.

## What the JEPA interest actually amounts to

The appealing hypothesis is that visual prediction can occur in a compact space that omits nuisance detail, avoiding the cost of reproducing every pixel. The stronger 2025–2026 results combine this with useful visual pretraining, action conditioning, temporal losses, and explicit goal/planner design. These are empirical engineering successes and a research program, not proof that low self-prediction error selects every feature required by unknown future tasks. This synthesis is supported by the four JEPA sources above.

For this manuscript, a defensible motivation is: **abstraction needs a criterion for what may be ignored**. Marginal non-collapse, one-step prediction accuracy, decodability, and usefulness under recursive control are different criteria. The experiment should expose their disagreement under a stated data and compute regime. Stochasticity is one reason they can disagree, but observed wall deletion should not automatically be attributed to conditional-mean averaging.

The generative comparison must receive the same scrutiny. Hansen and Wang's 2026 work is evidence that visually plausible stochastic models also suffer perceptual, action-conditioning and rollout errors. It does not prove that our observed failures have the same cause. Our proposed extra evaluation is to stratify errors by the available transition support, compare factual versus shuffled actions, and inspect encoded-real versus recursively-predicted latents before changing architectures.

## PVE identification and exact scope

“PVE” alone is ambiguous. Exact PVE+JEPA/ICLR-2026 searches did not identify a current JEPA paper with that acronym. Relevant older expansions include Proper Value Equivalence and Position Velocity Encoders. A recent near-match is [PAVE](https://arxiv.org/abs/2608.30378), which combines a VLA's multi-horizon predictive alignment with value-guided policy relabeling; it is neither named PVE nor a drop-in world-model loss. The adjacent local Muesli-Craftax project's PVE audit makes **Proper Value Equivalence** the best contextual interpretation, pending user clarification.

[Proper Value Equivalence, Grimm et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/400e5e6a7ce0c754f281525fae75a873-Paper.pdf), Sections 3–4, defines

\[
\mathcal M_\infty(\Pi)=\{\widetilde m: \widetilde v^\pi=v^\pi,\ \forall\pi\in\Pi\},
\qquad
\ell^k_{\Pi,\infty}(m,\widetilde m)
=\sum_{\pi\in\Pi}\|v^\pi-\widetilde{\mathcal T}_\pi^k v^\pi\|.
\]

Any positive integer k identifies the same exact class when true policy values are used. The planning guarantee concerns sufficiently rich policy coverage, including all deterministic policies; one fitted behavior critic does not establish it. Section 4 connects MuZero's reward/value losses to an upper bound under explicit approximations. The main paper, including experiments and related work, was read; the separately linked supplement was not exhaustively reviewed.

## Proposed small objective (our adaptation, not a PVE reproduction)

Freeze the existing encoder f and retain native D=512. Train a small independent scalar teacher Vbar from actual complete training episodes, using the unchanged environment rewards, existing discount gamma, and the environment's correct terminal/truncation convention:

\[
G_t=\sum_{j=0}^{T-t-1}\gamma^j r_{t+j},\qquad
\bar V=\arg\min_V\mathbb E[(V(f(x_t))-G_t)^2].
\]

These are exact **realized** discounted returns, not exact expected policy values. Split by complete episodes before generating overlapping windows. Do not replace the teacher's validation with its training fit. A constant-return predictor is a necessary reference: nearly constant random-policy returns give a nearly useless teacher even with low MSE.

Starting from the same existing predictor checkpoint, compare lambda=0 with one predeclared lambda>0, both optimizing

\[
\mathcal L_\lambda=
\frac1K\sum_{k=1}^K\mathbb E\left[
\frac1D\|\widehat z_{t+k}-f(x_{t+k})\|^2
+\lambda\,\big(\bar V(\widehat z_{t+k})-G_{t+k}\big)^2\right].
\]

Use K=1 initially; a small K such as 4 is only useful if both arms use the same recursive rollout and episode masks. Freeze teacher parameters while retaining gradients through its input. A lower-variance alternative replaces G with stop-gradient Vbar(f(x)); then teacher bias must be explicit. Neither form is exact PVE: it is a **PVE-inspired, task-supervised value-aware predictive loss**. It does not fit a complete policy Bellman operator or establish equivalence for all policies.

Match initial checkpoint, predictor architecture, encoder, latent dimension, data order, batch size, optimizer, update count, model seeds, and downstream heads/planning protocol. Report both added fitting time and total time. Choose lambda from a predeclared scale convention, not downstream evaluation reward. No reward reshaping, privileged environment state, feature-specific loss weights, latent expansion, action restrictions, or outcome filtering are needed.

The teacher should pass held-out episode RMSE/correlation/calibration checks before expensive policy training. Then compare held-out latent error, realized-return prediction through the fixed teacher, decoded rollout fidelity, action sensitivity, and real closed-loop reward. A positive result means this particular task signal improves the tested predictor; a null result is compatible with teacher weakness, limited behavior coverage or downstream head errors. It is not a general solution to reward-free physical intelligence.

Factual action replay is policy-consistent for the gate's uniform random behavior policy. For a state-dependent Craftax behavior policy, replayed actions in predicted states are not the same as running that policy in the model. This further limits any PVE claim; call the result supervised prediction on logged action sequences.

### Code anchors for the experiment agent

- `train.py:456–465`: current encoder/predictor jointly optimized; mean squared error is averaged over latent coordinates. Predictor-only fine-tuning needs an explicit frozen-encoder parameter filter.
- `probe.py:87–93`: Monte Carlo return recursion initializes the end of the stored array with zero. Exclude incomplete trajectory suffixes or handle known truncation appropriately.
- `probe.py:96–125`: existing reward/done/value head training; do not reuse the old teacher without a held-out test.
- `probe.py:142–158`: head-quality function explicitly scores fitting frames, so it does not establish generalization.
- `studies/stoch/sampler.py:1–7`: the existing residual distribution experiment changes prediction after freezing the JEPA and shares heads. Its null result cannot rule out every distributional model.

## Minimal citation additions

```bibtex
@inproceedings{grimm2021proper,
  title={Proper Value Equivalence},
  author={Grimm, Christopher and Barreto, Andr{\'e} and Farquhar, Gregory and Silver, David and Singh, Satinder},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34}, pages={7773--7786}, year={2021},
  url={https://proceedings.neurips.cc/paper/2021/file/400e5e6a7ce0c754f281525fae75a873-Paper.pdf}
}
@article{terver2026planning,
  title={What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?},
  author={Terver, Basile and Yang, Tsung-Yen and Ponce, Jean and Bardes, Adrien and LeCun, Yann},
  journal={Transactions on Machine Learning Research}, year={2026},
  eprint={2512.24497}, archivePrefix={arXiv},
  url={https://arxiv.org/abs/2512.24497}
}
@misc{hansen2026hallucination,
  title={Hallucination in World Models is Predictable and Preventable},
  author={Hansen, Nicklas and Wang, Xiaolong}, year={2026},
  eprint={2606.27326}, archivePrefix={arXiv}, primaryClass={cs.LG},
  url={https://arxiv.org/abs/2606.27326}
}
```

Existing LeWM, LeJEPA, V-JEPA 2 and PLDM entries already have verified titles/authors/arXiv IDs in `studies/stoch/paper/references.bib`; correct the prose classification rather than duplicating entries. Probabilistic JEPA references are optional context and should not displace the closer empirical/theory predecessors.

Memory was used only to recover intended claim scope and the earlier warning against inventing Craftax results. All literature facts above were checked in current primary sources. Relevant registry: `MEMORY.md:75–81`; rollout `01a0b6fb-e660-7053-a7c9-49280c7e2f93`.
