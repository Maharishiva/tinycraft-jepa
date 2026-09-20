# FlowWM: related-work note

Checked 20 September 2026. Read methods and main results (pp. 3–9), regression analysis (Appendix A), and limitations (Appendices J–K) in the [primary v2 PDF](https://arxiv.org/pdf/2606.29059v2). No experiments reproduced.

## Verified scope

FlowWM forecasts feature sequences with flow matching on frozen DINOv3 ViT-S tokens (384 channels), adding temporal consistency through a one-step endpoint projection. Detector supervision is used on FuturePerception, omitted on synthetic bouncing shapes. Synthetic F1 error is 4.31 versus 17.1 for its deterministic comparator. Real evaluation uses best-of-N large-object detection AP and depth. The model conditions on video, not actions; planning and end-to-end encoder learning are future work. No control or policy-learning tests are reported. Sources: Sections 3–5, Tables 1–2, Appendices J–K.

## Ready-to-use related-work paragraph

FlowWM models stochastic futures through flow matching in frozen pretrained features and evaluates mode coverage and future perception (Porcher et al., 2026). Our study instead examines jointly learned representations and the consequences of using mean predictions for control. The two settings are complementary: distributional prediction cannot recover information already omitted by an encoder, while representation sufficiency alone does not ensure correct nonlinear rollout evaluation.

## Citation

```bibtex
@misc{porcher2026flowwm,
  title={Flow Matching in Feature Space for Stochastic World Modeling},
  author={Porcher, Fran{\c{c}}ois and Carion, Nicolas and Alahari, Karteek and Chen, Shizhe},
  year={2026}, eprint={2606.29059}, archivePrefix={arXiv}, primaryClass={cs.CV},
  url={https://arxiv.org/pdf/2606.29059v2}
}
```

## Scope guards for our argument

These are our mathematical checks, not new results attributed to FlowWM. Avoid copying Appendix A's broad L1 language: for a fair two-point distribution, every point between the modes minimizes absolute error, including the endpoints. A conditional-mean statement belongs specifically to squared error. Also, a generative model may train a velocity field with squared error without reducing its final sampled distribution to one conditional mean.

Do not characterize all deterministic or nongenerative models as incapable of stochastic control. The examples below concern evaluating nonlinear costs at means and recursively applying nonlinear conditional-mean dynamics. Deterministic value-equivalent models, successor features for suitable reward classes, or exact Bellman expectations can avoid these mistakes.

## User's exact action-ranking example

For squared distance to goal g=0, action A gives X=-1 or +1 equiprobably and action B gives X=0.1 deterministically.

| Quantity | A | B |
|---|---:|---:|
| Mean E[X] | 0 | 0.1 |
| Cost of mean (E[X]-g)^2 | 0 | 0.01 |
| Correct expected cost E[(X-g)^2] | 1 | 0.01 |

The mean-only goal score selects A; correct expected cost selects B. The missing term is action-dependent variance:

\[
\mathbb E[(X-g)^2]=(\mathbb E[X]-g)^2+\operatorname{Var}(X).
\]

A deterministic predictor of expected cost can output 1 and 0.01 directly; sampling is not logically required for this decision. This example therefore supports a restricted claim about mean-state goal evaluation.

## User's exact recursive-mean example

Let

\[
X_{t+1}=X_t^2+\xi_t,\qquad X_0=0,\qquad
\Pr(\xi_t=-1)=\Pr(\xi_t=+1)=\tfrac12,
\]

with independent noise across steps. The exact one-step conditional mean is m(x)=x^2, and its recursive prediction from zero remains zero at every horizon. In the real process, X1 is ±1; X2 is 0 or 2; X3 is -1,1,3,5. The respective probabilities are 1/2, 1/2, and 1/4 for each listed value.

| Horizon | True mean | Recursively predicted mean | True squared goal cost E[X_t^2] |
|---|---:|---:|---:|
| 1 | 0 | 0 | 1 |
| 2 | 1 | 0 | 2 |
| 3 | 2 | 0 | 9 |

The mismatch is already present with a perfect one-step predictor:

\[
\mathbb E[X_{t+1}]=(\mathbb E[X_t])^2+\operatorname{Var}(X_t).
\]

Every finite-horizon distribution here has finite support, so no Gaussian substitution or extra moment assumption is needed. Keep this as an analytical illustration; it does not identify the cause of any trained red-cube model's failure.
