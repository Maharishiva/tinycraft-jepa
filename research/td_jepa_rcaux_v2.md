# TD-JEPA and RC-aux: V2 positioning

Read status: TD-JEPA main methods/theory/experiments and implementation/evaluation appendix were read from full arXiv HTML; objectives and venue checked against the published ICLR PDF. RC-aux notes follow in the companion review. This is a focused methodological reading, not a full audit of every proof.

## TD-JEPA (Bagatella et al., ICLR 2026)

[Published paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/3d158f054ff0cb83397367234899db07-Paper-Conference.pdf), pp. 3–6, Eq. 9 and Algorithm 1; [readable full text](https://arxiv.org/html/2510.00739v1#S3.SS2).

Its asymmetric TD loss is

\[
\mathbb E\left\|T_\phi(\phi(s),a,z)
-\operatorname{sg}\left[\psi(s')+\gamma T_\phi(\phi(s'),a',z)\right]\right\|^2,
\qquad a'\sim\pi_z(\cdot\mid s').
\]

The reciprocal loss trains the other encoder. Algorithm 1 uses EMA targets and orthonormality regularization. The prediction represents policy-conditioned successor features; actions optimize their inner product with a task vector. It does not recursively simulate next-state latents for CEM. Its MDP formalism permits stochastic transitions. The non-collapse result uses idealized linear predictors equilibrated before continuous-time encoder updates.

Experiments compare zero-shot control across ExoRL/OGBench, plus one-step versus multistep/policy-conditioned targets and separate versus shared encoders. They do not manipulate environment stochasticity crossed with target detachment. Mean successor features support linear reward evaluation; this should not be conflated with evaluating a nonlinear downstream function on an iterated mean state.

## RC-aux (Li et al., arXiv 2026)

[Primary full text](https://arxiv.org/html/2605.07278v1), Sections 3–4 and Appendix B.1; [official objective code](https://github.com/Guang000/RC-aux/blob/main/rcaux.py#L70).

RC means Reachability-Correction, not reconstruction. Its objective combines multi-horizon open-loop MSE, SIGReg, and budget-conditioned reachability classification with temporal negatives. The multi-horizon term replaces one-step prediction. Official code uses live encoded targets without detachment in both one-step and open-loop MSE. Thus target gradients are present, as in original LeWM.

It evaluates continuation training and matched training from scratch, including a shared Euclidean-CEM comparison and an optional reachability planning term. Its analysis assumes deterministic controlled dynamics (Appendix B.1). The intervention changes temporal/reachability supervision and planner costs; it does not cross transition stochasticity with the target-gradient path. It already establishes closely related prediction/planning mismatch and correction experiments.

The code inspection concerns public `main` on 19 September 2026, not provenance verification of the paper's checkpoints.

## Positioning recommendation (our inference)

A successful matched intervention would contribute evidence about how gradients through future encoding interact with exogenous transition uncertainty in the tested jointly learned next-state model. It should acknowledge semi-gradient predictive learning as established and should not claim that prior latent prediction ignores stochastic MDPs. If the new result only shows stop-gradient improves one score, it establishes neither the claimed interaction nor a general control mechanism.

## Verified TD-JEPA bibliography

```bibtex
@inproceedings{bagatella2026tdjepa,
  title={{TD-JEPA}: Latent-predictive Representations for Zero-Shot Reinforcement Learning},
  author={Bagatella, Marco and Pirotta, Matteo and Touati, Ahmed and Lazaric, Alessandro and Tirinzoni, Andrea},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://proceedings.iclr.cc/paper_files/paper/2026/file/3d158f054ff0cb83397367234899db07-Paper-Conference.pdf}
}
```

This is temporal-**difference** TD-JEPA, distinct from Bai and Xiong's temporal-**distance** JEPA (arXiv:2607.25337).

```bibtex
@misc{li2026rcaux,
  title={Predictive but Not Plannable: {RC-aux} for Latent World Models},
  author={Li, Wenyuan and Li, Guang and Maeda, Keisuke and Ogawa, Takahiro and Haseyama, Miki},
  year={2026}, eprint={2605.07278}, archivePrefix={arXiv}, primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.07278}
}
```
