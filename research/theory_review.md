# Independent theory and code-semantics review

Reviewed 2026-09-19 from the current `studies/stoch/paper` manuscript, all its sections and proofs, and relevant training/control/diagnostic implementations. No original files or GPU jobs were changed. A second reviewer independently checked the three propositions/proofs and agreed with the validity verdict below.

## Verdict

The current proofs are mathematically valid under their stated population and measurable-function assumptions. They establish **missing guarantees**, not the cause of the empirical gate failure. The current paper has already repaired many earlier overclaims, but its abstract still connects theory to the wall failure too strongly. The texture-trap result is also much closer to existing work than its contribution wording admits: Sobal et al. (2022), §2.1, already give the Gaussian static-nuisance plus identity-predictor construction. Extending that observation to SIGReg and making deployment/control assumptions precise is useful, but not a new failure mechanism.

The strongest defensible story is: **non-collapse, encoded information, predicted information, and usable control information are different requirements. The experiments distinguish some of them; neither a low latent loss nor a successful decoder certifies an imagination pipeline.** A protocol-matched dynamics/head intervention would strengthen this story more than another abstract counterexample or cosmetic task variant.

## Exact theorem audit

### 1. Gaussian nuisance-only optimum

`studies/stoch/paper/sections/theory.tex:22–50` is correct. A recoverable episode-static nuisance with a measurable Gaussianization supplies an exactly Gaussian encoding. Copying its last value gives zero prediction error; nonnegative population terms imply global optimality. Independence under the training distribution gives the mutual-information statement. The deployment coupling correctly allows the nuisance to influence later states through controller actions; it does not incorrectly assume deployment state/nuisance independence after time zero.

Keep these restrictions explicit:

- This is the defined **population** SIGReg functional, not a global-optimum theorem for the finite-batch implementation.
- A finite-valued rendered texture cannot be deterministically mapped to an exact continuous Gaussian. The unrestricted function-class assumption matters.
- The conclusion covers a controller given latents, time, and its own actions. A controller additionally given rewards or other environment feedback is not covered by the stated coupling unless those signals are explicitly allowed in the comparison class.
- A state-blind policy can solve some tasks. Zero state information alone is not a universal quantitative control-performance bound.

**Novelty repair:** label this an extension of the known slow-feature shortcut to the SIGReg population objective and state its control consequence, rather than presenting the Gaussian nuisance construction as a new conceptual finding. [Sobal et al., 2022, §2.1](https://arxiv.org/html/2211.10831v1).

### 2. Innovation lower bound

`theory.tex:53–66` and `appendix.tex:8–35` are correct. The sigma-field inclusion, conditional-variance decomposition, conditionally independent copies, and the fixed-nuisance co-Lipschitz inequality give the claimed bound. No independence assumption on persistent wind is needed.

The assumption is much stronger than injectivity, task sufficiency, or empirical decodability: it requires every state distance to be preserved at a uniform scale. It therefore does **not** establish that every useful encoder must have high prediction loss. Its conditioning removes whatever wind is inferable from history; persistent noise is not automatically fresh innovation.

The implementation uses coordinate-mean MSE (`train.py:460`). For that convention the bound is

\[
\mathbb E[D^{-1}\|g-Z\|^2]\ge (\kappa^2/D)\mathbb E\operatorname{tr}\Sigma_t.
\]

The sum-of-squares theorem is valid as written, but the manuscript must not equate its constants with logged per-coordinate losses.

### 3. Conditional mean

`theory.tex:73–81` and `appendix.tex:37–48` are standard conditional least squares and correct for a **fixed encoder/data law**, unrestricted square-integrable predictors, and the specified conditioning information. Jointly optimizing an encoder does not invalidate this conditional statement, but it does not show that the trained neural predictor attains the conditional mean.

An empirical mean of eight learned samples is neither the true conditional mean nor deterministic. If samples are conditionally iid under the fitted sampler, its conditional covariance is the fitted sampling covariance divided by eight. Its conditioning also differs from the original full history, as described below.

### 4. Exactly Gaussian, decodable, small-amplitude state

`theory.tex:86–109` and `appendix.tex:73–106` are correct. The truncated-Gaussian quantile construction recovers both the fair bit and nuisance almost surely, has the exact Gaussian marginal, and places the two states within one interval of width `2 epsilon`. Its conditional-mean prediction error is their squared separation divided by four. For logged coordinate-mean MSE the bound is `epsilon² / D`.

The result is a useful separation between exact information and prediction-loss scale. It uses infinite-precision, non-robust measurable coding and gives no finite-network realization or optimization trajectory. More importantly, the next bit is intrinsically unpredictable: averaging it is Bayes optimal. The theorem demonstrates neither avoidable predictive error nor a policy/planning deficit from using a fitted predictor. It cannot explain deletion of a deterministically advancing wall before gap revelation.

The current final sentence of the theory section acknowledges this accurately and should survive the rewrite. Move the construction and proof to the appendix if space is needed; keep its one-sentence implication in the main text.

### 5. Finite-sample SIGReg identity

`appendix.tex:57–68` is correct for iid samples. The additional `(1-|phi|²)/n` term depends on the data law, so its value at a Gaussian is not a universal optimization floor. Finite directions/knots do not establish Gaussianity. Temporal dependence adds covariance terms. Retain these distinctions.

There is an additional normalization difference in the implemented regularizer: it multiplies the batch statistic by `B`. Thus after this scaling the Gaussian finite-batch bias does not shrink as `1/B`; dividing the implementation by `B` is the normalization that converges to the corresponding population discrepancy. This does not invalidate the stated population result, but it makes a direct finite-objective interpretation especially inappropriate.

## Important implementation alignment repairs

### Exact objective normalization

The main objective `worlds.tex:8–18` is a schematic version of the code but is presented as exact. `train.py:90–100` and `456–462` implement:

\[
L_{pred}=\frac1{BHD}\sum_{b,h}\|\hat z_{b,h+1}-z_{b,h+1}\|^2,
\]

and, for encoded windows with `T` frames,

\[
R=\frac{B}{TP}\sum_{t,p,k}q_k w_k\left[\left(\frac1B\sum_b\cos(t_k u_p^Tz_{b,t})-w_k\right)^2+\left(\frac1B\sum_b\sin(t_k u_p^Tz_{b,t})\right)^2\right].
\]

Here `q_k=dt` at the endpoints and `2dt` in the interior, `dt=3/(K-1)`. The regularizer first averages characteristic functions over the batch separately at each temporal position, then averages temporal positions and projections. It is not one pooled temporal marginal. The doubled trapezoid weights can be absorbed into a stated normalization, but the factor `B` cannot silently disappear while retaining the reported coefficient `lambda=0.09`.

Training supervises one-step predictions at every temporal position of a teacher-forced window, with a shorter causal prefix at early positions. It does not only supervise the final prediction with a full eight-frame history (`train.py:459–460`). Describe this in the appendix.

### Batch normalization and causal-map idealization

`train.py:260–269` uses `Linear–BatchNorm–GELU–Linear` in both projection heads. Encoder projection flattens batch, time and spatial tokens (`393–398`); prediction projection also flattens them (`404–408`). During training, BatchNorm statistics couple different temporal positions and batch examples. At inference, running statistics give a fixed pointwise encoder and the full predictor is causal. The attention trunk itself is causal.

This is not an established bug; it mirrors the reference architecture. However, the mathematical `f(x_t)` and strictly causal `g(history)` are an inference/population idealization rather than the literal full training graph. Disclose BatchNorm in the objective/architecture description and do not claim a train-time no-leakage theorem from the existing test: `tests/test_port.py:67–75` only checks the attention predictor at inference, not the full train-mode predictor/projection path. No claim that this causes any empirical deficit is warranted without a controlled test.

### Sampling intervention is narrower than its comments suggest

The residual flow sees `[original predicted mean, last latent, current action]`, not the full latent/action history (`studies/stoch/sampler.py:62–65`). It models standardized residuals and adds an unconstrained correction (`67–72`); it need not preserve the original predictor's mean. Sample averaging is still random (`74–81`), despite the docstring calling it deterministic and its own conditional mean. The paper's cautious language is substantially better than these comments.

Recommended wording: “We compare one sample with an eight-sample average from the same fitted residual model.” Do not call this an exact mean-versus-distribution intervention. The observed lower one-sample MSE than the original deterministic predictor is evidence that the original network is not the exact population least-squares solution under this evaluation law; it does not contradict the conditional-mean lemma.

### Decoder and geometry diagnostics

The wall-removal contrast is carefully scoped in the current results. `latent_check.py:86–107` measures within-encoder sums of squared coordinate distances and projections; `177–183` also decomposes one-step error into along-wall and orthogonal components. Compare wall-aligned error with wall amplitude if discussing predictive wall precision. Comparing one feature contrast with total 512-dimensional residual is descriptive and can be dominated by unrelated errors.

The selected recorded futures survive the diagnostic horizon, while counterfactual gap futures need not (`156–162`). These remain valid rendered counterfactuals, but they do not identify the true conditional next-latent distribution in every state. “Two future encodings” or their midpoint must not become “the exact conditional mean.”

A small support check would be more diagnostic than another raw-distance plot: show the decoded real latent, the fitted one-step prediction, and the midpoint of correctly weighted counterfactual encodings. If even the exact two-branch midpoint decodes a wall, fitted wall deletion is not explained by this particular two-branch averaging account. If it does not, decoder extension away from the real-encoding support remains an explanation. Neither outcome alone establishes policy causality.

### Gate wording

The gap is sampled **at wall spawn** (`env.py:129–131`) and rendered only once `wall_x <= 40` (`68–76`), rather than sampled at reveal. Replace ambiguous “drawn only once the wall is past x=40” with “sampled at spawn and visually revealed after the wall crosses x=40.” The present gap location is latent state before reveal, so prediction from pixels is genuinely partially observed even though later wall motion is deterministic. The wall itself before reveal is predictable.

## Novelty and classical control

The following distinctions should appear explicitly, not as buried caveats:

1. **Slow nuisance shortcut is prior art.** Sobal et al. (2022) already study moving dots with per-frame/per-sequence distractors and give a Gaussian nuisance construction. The contribution here can be a SIGReg extension plus control and imagination diagnostics, not discovering that predictable backgrounds distract self-prediction. [Primary paper](https://arxiv.org/html/2211.10831v1).
2. **A directly relevant 2026 paper must be discussed.** Gulati and Nemenman study slow trajectory noise and show that changing contrastive negative sampling from across trajectories to within trajectories changes what is learned. This concerns contrastive objectives, not a proven fix for SIGReg; it reinforces the need to distinguish objective families. [Primary paper](https://arxiv.org/abs/2606.07770).
3. **Self-predictive representation dynamics are already studied.** Tang et al. analyze semi-gradient/predictor dynamics and spectral structure; their setting differs from an exact marginal-constrained global optimum. Ni et al. connect state/history abstraction and self-predictive sufficiency. Cite them to locate the question rather than claim all self-prediction fails. [Tang et al.](https://proceedings.mlr.press/v202/tang23d.html), [Ni et al.](https://arxiv.org/abs/2401.08898).
4. **Mean prediction is not inherently unsuitable for stochastic control.** Expectation models can give exact Bellman policy-evaluation updates with values linear in the modeled features. Full distribution learning is sufficient under appropriate state information, but not necessary. [Wan et al., 2019](https://www.ijcai.org/proceedings/2019/506).
5. **A generic Jensen/misranking example is classical, not a JEPA theorem.** Let action A lead to terminal `Z=±1` equiprobably, action B to `Z=1/2`, and reward `r(Z)=Z²`. True expected rewards are `1` versus `1/4`; applying reward to each mean gives `0` versus `1/4`. This is a risk-neutral expectation error, not an argument that the controller is risk seeking. LQG certainty-equivalence conditions cannot be used to excuse arbitrary action-dependent covariance and nonlinear downstream functions. A second elementary example is `Z1=±1`, `Z2=Z1²`: recursively substituting a mean predicts zero instead of `E Z2=1`. Use either only as explanatory background.
6. **Task-aware objectives have an established foundation.** Value-aware model learning and value equivalence constrain model errors according to their effect on Bellman computations rather than raw latent distances. Proper value equivalence concerns preservation of policy values; adding one scalar auxiliary reward/value loss is only a restricted PVE-inspired experiment, not a PVE guarantee. [VAML](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf), [PVE](https://papers.nips.cc/paper/2021/file/400e5e6a7ce0c754f281525fae75a873-Paper.pdf).

## A rigorous, simpler storyline

Suggested organizing question: **What must an abstract world model preserve for imagination to support control?**

Use three empirical questions and one theoretical boundary:

1. **Does the encoder retain observable decision-relevant information?** Evaluate on real encoded observations with read-outs and downstream policies; distinguish direct latent policies from decoder-supervised CNN policies.
2. **Does prediction preserve that information through imagined transitions?** Compare matched real encodings and teacher-forced/open-loop predictions. Report event-specific errors and fitted reward/termination quality, not only global MSE.
3. **Can the complete imagination pipeline train useful control?** State the exact PPO, heads, rollout starts, model fits and policy replication. Use protocol-matched oracle transitions to identify dynamics/head effects if possible.
4. **What does the objective certify?** A compact theorem box says Gaussian non-collapse alone certifies neither state sufficiency nor robust geometric scale. It does not turn the empirical observations into a proven optimization mechanism.

Craftax can motivate the problem first, followed by controlled gate and nuisance worlds that progressively remove confounders. Do not imply the minimal-world mechanisms have been identified as the cause of Craftax performance. A narrative from hard to easier tasks is credible if it is explicitly a diagnostic progression, not a claim that all environments have the same failure.

Proposed main-text theoretical paragraph:

> Marginal non-collapse and decision sufficiency are different properties. Extending the static-nuisance construction of Sobal et al. to the population SIGReg objective, we show that an exactly Gaussian latent can be a global optimum while containing no controlled-state information. Even when a latent is exactly decodable, Gaussianity alone imposes no positive lower bound on the Euclidean scale of a relevant feature. These are limits of the objective's guarantees; our experiments separately test what the fitted encoders and predictors retain.

Replace abstract “We prove that the objective permits this” (`main.tex:82–84`) with:

> Separately, population counterexamples show that Gaussian non-collapse guarantees neither state sufficiency nor a fixed prediction-loss scale for decodable features.

Also change “keeping the latent marginal Gaussian” (`main.tex:74`) to “regularizing the latent marginal toward a Gaussian.” The finite empirical objective does not guarantee an exactly Gaussian learned law.

The present results warrant “a measured pipeline deficit” and “useful information in at least one encoder.” They do not yet warrant “the predictor causes the control deficit,” “conditional means erase the wall,” “distributional models solve the problem,” “JEPA cannot support physical intelligence,” or “all useful state is preserved by the tested encoders.”

## Best additional experiment for causal clarity

Use the same imagination PPO collection/update protocol and the same frozen encoder, rollout starts, fitted heads, horizon and step budget in three arms:

| Arm | Transition | Reward / termination | What it tests |
|---|---|---|---|
| Current JEPA | Fitted latent transition | Same fitted heads | Existing pipeline |
| Oracle transition | True simulator transition, freshly encoded | Same fitted heads | Transition substitution under fixed training and heads |
| Oracle transition and signals | True simulator transition, freshly encoded | True reward / termination | Remaining fitted-head contribution |

The simulator state must be restored exactly for both reset and dataset starts; preserve policy-history burn-in. A useful true-transition arm needs state, wall/gap, episode time and nuisance/wind restoration as applicable. The gate has no background/wind, simplifying this reconstruction. Use unmodified reward semantics. This is a diagnostic control, not a proposed agent or environment-specific scaffold.

With at least three encoder seeds and matched policy seeds, the first substitution can identify a transition-related deficit conditional on the heads and PPO protocol; the second can expose fitted-head limitations. Interactions remain possible, so avoid additive causal decompositions not measured by the design. A result where oracle transitions still fail is also informative and should change the storyline toward heads/optimization rather than be hidden.

This intervention is scientifically more valuable within the deadline than manually enlarging selected latent directions, recoloring walls, or treating extra noisy environments as a causal test. Additional environment variants can establish scope later after the core comparison is interpretable.

## Optional main-text intuition: a standard sensitivity bound

A short sensitivity statement connects the theory to the empirical audit more naturally than leading with the fresh-bit construction. For a fixed `L`-Lipschitz readout `V` and any coupling of true/predicted next latents,

\[
|\mathbb E V(Z')-\mathbb E V(\widehat Z')|
\le L\mathbb E\|Z'-\widehat Z'\|
\le L\sqrt{\mathbb E\|Z'-\widehat Z'\|^2}.
\]

For the deterministic predictor, use `Zhat'=g(c,a)`. Thus a uniform per-action reward error `delta_r` and latent root-sum-square prediction error `epsilon` bound one-step score error by `delta_r + gamma L epsilon`. An optimal action gap exceeding twice this quantity preserves the action ranking. If using logged coordinate-mean MSE, `epsilon=sqrt(D*MSE)`. This is a standard Lipschitz/Cauchy–Schwarz consequence, not a new contribution. No empirical guarantee follows without a valid bound on `L` and the relevant prediction error.

The useful point is that exact decodability alone supplies no bound on downstream sensitivity. Low latent prediction error is meaningful for a task only relative to that task's readout sensitivity. A conditional mean can be perfectly fitted while this bound remains positive because conditional innovation remains. This paragraph complements, rather than validates, the empirical scale diagnostic.

## Memory provenance

A quick memory pass was used only to retain the previously requested distinction between representation loss and mean-prediction effects, then all claims above were checked against the live paper/code or linked primary sources. Relevant memory entry: `/Users/rldoeswork/.codex/memories/MEMORY.md:75–76`, rollout `01a0b6fb-e660-7053-a7c9-49280c7e2f93`.
