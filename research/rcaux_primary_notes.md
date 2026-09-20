# RC-aux comparison for V2

Read on 19 September 2026: main objectives and evaluation, Appendix B planning analysis, Appendix C protocol, and official public loss functions. No experiments or manuscript edits.

Verified identity: **Predictive but Not Plannable: RC-aux for Latent World Models**, Wenyuan Li, Guang Li, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama; arXiv:2605.07278v1, 8 May 2026. [Primary metadata](https://arxiv.org/abs/2605.07278).

Three distinctions:

1. **Objective/intervention.** RC means Reachability-Correction, not reconstruction. There is **no reconstruction loss**. Eq. (18) is `Lmulti-horizon + α SIGReg + β Lreachability`; multi-horizon open-loop MSE replaces one-step MSE. Reachability BCE uses trajectory-derived finite-budget labels. This changes supervisory content and rollout training; our proposed intervention instead holds those fixed and varies prediction-target gradient flow and transition stochasticity. [§3.1–3.3, Eqs. 6/17/18](https://arxiv.org/html/2605.07278v1#S3).

2. **Gradient object.** Eq. (12) stops gradients on predicted latents supplied to the reachability classifier; encoded-pair reachability still shapes the encoder. This is distinct from stopping the encoded **prediction target** in a squared dynamics loss. No factorial target-gradient ablation is reported. [§3.2](https://arxiv.org/html/2605.07278v1#S3.SS2), [Appendix B.9](https://arxiv.org/html/2605.07278v1#A2.SS9).

3. **Mechanism/causality.** Appendix B.1 assumes deterministic controlled dynamics. Its analysis concerns rollout-cost distortion and empirical reachability. The planner ablation separates changed training from reachability-aware scoring; it does not identify a gradient-by-transition-noise interaction. Thus generic prediction/planning mismatch is prior work, while a properly matched stochastic-gradient intervention asks a different causal question. [Appendix B.1](https://arxiv.org/html/2605.07278v1#A2.SS1), [§4.3](https://arxiv.org/html/2605.07278v1#S4.SS3).

Implementation verification, separate from manuscript formulation: current official [rcaux.py, lines 70–102](https://github.com/Guang000/RC-aux/blob/main/rcaux.py#L70) uses un-detached `tgt_emb` for both one-step and open-loop prediction. [train.py, line 50](https://github.com/Guang000/RC-aux/blob/main/train.py#L50) does the same for LeWM. `JEPA.encode` returns live encoder/projector outputs, and `rcaux_forward` passes them directly to the losses. Thus the public prediction paths retain target gradients when the encoder is trainable. This is code verification, not reconstruction of every reported checkpoint/configuration. An optional reachability-only configuration can freeze the backbone.

Narrow implementation caveat: current `rcaux.py` also passes `pred_emb` directly into predicted-pair reachability (`reachability_loss`, around lines 697–715); I did not find the predicted-input detachment described by Eq. (12) on that path. Do not make a causal claim from this code/paper mismatch or use it to infer the reported checkpoint's exact execution.

Protocol boundary: local LeWM-family results use five fixed groups of 50 episodes and group-level standard deviations, **not five training seeds**. TwoRoom/Reacher/PushT/Cube compare continuation training; Wall uses local from-scratch LeWM. LIBERO trains an OFT-style action head rather than running latent MPC. [Official repository protocol](https://github.com/Guang000/RC-aux#main-reported-results), [paper Appendix C](https://arxiv.org/html/2605.07278v1#A3).

BibTeX, using the official repository's author/title spelling:

```bibtex
@article{li2026rcaux,
  title={Predictive but Not Plannable: {RC-aux} for Latent World Models},
  author={Li, Wenyuan and Li, Guang and Maeda, Keisuke and Ogawa, Takahiro and Haseyama, Miki},
  journal={arXiv preprint arXiv:2605.07278},
  year={2026},
  url={https://arxiv.org/abs/2605.07278}
}
```

Suggested manuscript positioning (inference): “Prior work improves latent planning through rollout and reachability supervision. We instead isolate whether stochastic transitions interact with gradients through the learned prediction target, using otherwise matched objectives and models.” This statement becomes an empirical contribution only when the factorial runs and uncertainty support it; a detached-target rescue alone is insufficient to establish an interaction.

Local evidence: `/tmp/arxiv-2605.07278v1.html`, `/tmp/arxiv-2605.07278v1.txt`, `/tmp/rcaux-primary-rcaux.py`, `/tmp/rcaux-primary-train.py`.
