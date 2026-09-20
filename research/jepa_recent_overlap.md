# Recent JEPA overlap addendum — checked 19 September 2026

Verdict: the broad claims that forward latent prediction need not preserve control information, and that Euclidean latent planning can be misaligned with goal progress, are already explicit themes in 2026 papers. Our contribution must be a specifically established phenomenon, diagnostic, or remedy beyond these themes. None of the three papers below establishes the particular stochastic-pressure result sought in our draft; this absence is a reading conclusion, not a proof of novelty across all literature.

Reading depth: close-read Temporal-Distance-JEPA's full main text and training/evaluation appendix, plus relevant official code; read PSG-JEPA's complete methods and planning protocol, skimmed remaining experiments; read Physically Grounded JEPA's methods, experiment protocol, and conclusion. No experiments run.

## Temporal-Distance-JEPA: Plan-Aware Representation Learning for Latent World Model Predictive Control

Bai and Xiong, [arXiv:2607.25337v2](https://arxiv.org/abs/2607.25337v2), 29 July 2026. Relevant anchors: [§3](https://arxiv.org/html/2607.25337v2#S3), [§3.5–3.6](https://arxiv.org/html/2607.25337v2#S3.SS5), [§4.3](https://arxiv.org/html/2607.25337v2#S4.SS3), [Appendix C](https://arxiv.org/html/2607.25337v2#A3).

- Learns encoder, predictor and directed cost head. Eq. (9): `Lpred + 0.5 Lroll + 1.0 Ltd + 0.09 SIGReg`. `Ltd` fits ordered same-trajectory pairs to temporal gaps with Smooth-L1 and pushes cross-trajectory pairs above a margin. Demonstration lengths are acknowledged proxies, not shortest-path truth; negatives can be false.
- Dynamics are deterministic MSE with stop-gradient targets (Eqs. 1, 8), unlike original LeWM's fully differentiated objective. Five-step rollout loss regularizes latent predictions; temporal supervision also shapes representation geometry.
- Deploys learned directed cost on TwoRoom/Reacher, but latent L2 on PushT/Cube using the same temporally trained checkpoint. Thus it separately studies changing the planner cost and changing learned representations.
- Table 2: TwoRoom 100.0 versus LeWM 97.4; Cube 82.2 versus 68.0. These uncertainties are ten planning seeds with a fixed checkpoint and episode manifest, not ten independently trained models.
- Novelty threat: directly anticipates a generic train–plan mismatch narrative and cost-vs-representation ablations. Distinction: no demonstrated information deletion specifically induced by stochastic transition variance; no stochastic transition objective.

Official implementation confirms joint learning and target detachment: [losses.py](https://github.com/HKBU-KnowComp/Temporal-Distance-JEPA/blob/main/losses.py#L324) has `mse_alignment_loss` with detached targets; recurrent rollout also detaches its targets. [Temporal loss](https://github.com/HKBU-KnowComp/Temporal-Distance-JEPA/blob/main/losses.py#L463) evaluates the cost head on un-detached encoder outputs. [train.py](https://github.com/HKBU-KnowComp/Temporal-Distance-JEPA/blob/main/train.py) optimizes the whole model. This supports “jointly learned with semi-gradient dynamics,” not “frozen encoder” or “identical gradient objective to LeWM.” These are public `main` files inspected today, not checkpoint provenance verification.

## Naming collision: Bagatella et al.'s TD-JEPA is different

[TD-JEPA: Latent-Predictive Representations for Zero-Shot Reinforcement Learning](https://arxiv.org/abs/2510.00739) is by Bagatella, Pirotta, Touati, Lazaric and Tirinzoni. The [published ICLR 2026 paper](https://openreview.net/pdf/ff9a6a776994a760cb3d8cc5b8786f4c02065023.pdf) explicitly expands TD as temporal difference. It learns state/task encoders, a policy-conditioned long-term predictor and a policy family from offline reward-free transitions; the predictor is related to successor features. It concerns zero-shot reward optimization, not Bai and Xiong's temporal-distance cost for CEM. Use separate bibkeys and the full “Temporal-Distance-JEPA” name for the July paper. This disambiguation was checked against primary abstracts and the accepted-paper front matter; no fresh full proof audit of Bagatella et al. was performed here.

## PSG-JEPA: Is Forward Prediction Enough? Physical State Grounding for JEPA World Models

Yan et al., [arXiv:2608.06799v1](https://arxiv.org/abs/2608.06799), 7 August 2026. Anchors: [§3](https://arxiv.org/html/2608.06799#S3), [§4.1–4.3](https://arxiv.org/html/2608.06799#S4.SS1), [§4.4](https://arxiv.org/html/2608.06799#S4.SS4).

- Jointly learns LeWM-style image encoder/predictor. Preserves teacher-forced MSE + SIGReg and adds single-latent proprioceptive-state regression plus multi-horizon endpoint-pair regression of joint-angle changes. Eq. (7): `LJEPA + 0.1(Lstatic + Ldynamic)`; SIGReg coefficient 0.09. Training windows contain four frames; grounding horizons 1–3. Physical measurements supervise training but are not predictor/encoder inputs.
- Explicitly diagnoses imperfect recoverability of physical state/change from forward-trained LeWM representations and evaluates probes, goal reaching and downstream policy learning. This directly overlaps a generic “prediction does not ensure control-relevant information” claim.
- Important protocol boundary: §4.4 discards auxiliary heads, freezes the encoder, and trains a separate GC-IDM goal-conditioned inverse-dynamics policy. Its Cube/Scene control does not use the learned world-model predictor. Recursive predictor errors are evaluated separately. Do not present these planning results as CEM with the original learned transition model.
- Distinction: privileged physical-state supervision addresses accessibility; it does not isolate aleatoric uncertainty as the cause or preserve a wholly state-label-free training contract.

## Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning

Liu, Huang, Liang and Gao, [arXiv:2609.03565v1](https://arxiv.org/abs/2609.03565), 3 September 2026; arXiv lists acceptance at an IROS 2026 workshop. Anchors: [§II-B](https://arxiv.org/html/2609.03565#S2.SS2), [§II-D](https://arxiv.org/html/2609.03565#S2.SS4), [§III-A/B](https://arxiv.org/html/2609.03565#S3.SS1).

- All components jointly optimized: `Ltotal = Lpred + α Lstate-alignment + β Linverse-dynamics`. Pair heads regress the executed action and measured physical state; no SIGReg term is listed. Uses randomly initialized ViT-Tiny, native 192-dimensional latent, six-layer predictor. State labels are training-only; deployed CEM minimizes terminal squared latent distance.
- Tests TwoRoom/Reacher/PushT/Cube; state-alignment targets use full available environment state. Reports improvements over inverse-dynamics-only ablation and relates temporal straightening to effective transition dimension.
- Baseline means are imported from LeWM; its own results use three training seeds and 50 fixed planning problems. Therefore its table is not a fresh matched retraining comparison of every method.
- Distinction: a supervised physical-grounding remedy with ordinary deterministic dynamics, not a reward-free observation/action-only stochastic-learning diagnosis. Still cite it when discussing state retention or probe/geometry limitations.

## Consequences for our rewrite

1. Remove first-discovery language for latent prediction/control mismatch, Euclidean goal-cost inadequacy, or state-probe insufficiency. A useful paper can instead separate these mechanisms with stronger evidence.
2. Do not substitute a known Gaussian nuisance construction for a novel result. The team's separate theory audit reports a directly prior Sobal (2022), §2.1 construction; this supersedes my earlier suggestion to lead with that construction as the strongest possible novelty.
3. Identify the actually measured deficit before selecting the headline. If state-decoded control succeeds, “the encoder erased the controlled state” is not the supported empirical conclusion. A statement about the direct latent planning pipeline needs a cost/dynamics/representation decomposition.
4. Cite privileged-state remedies as complementary, while declaring their extra information. The user's general-model constraint makes a state-labelled repair a diagnostic oracle or separately scoped baseline rather than an unobtrusive change to the main reward-free recipe.
5. Every further comparison should hold native dimension, dataset, planner budget and target/task interface fixed. A cost-only probe, frozen-encoder stochastic-dynamics comparison, and separately joint-trained objective address different causal questions; their conclusions should not be pooled.

Downloaded HTML and text extracts are under `/tmp/arxiv-2607.25337v2.*`, `/tmp/arxiv-2608.06799.*`, and `/tmp/arxiv-2609.03565.*`.
