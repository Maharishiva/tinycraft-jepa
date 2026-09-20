# Independent round-2 critique

## Superseding update after NEW_RESULTS.md

E1 is now measured: original / mean-of-eight / sample pass 0.00 / 0.91 / 0.66 walls. Improved conditional prediction, not sampling, carries the JEPA gain at model and PPO seed 0. There is no two-family recovery claim. E2 and E3 are pending. E9 measures a border-versus-wall deficit in the fitted supervised read-outs, without proving information-theoretic absence or explaining the entire residual control gap. E4 is horizon- and score-limited; leaking the gap barely changes h=5. The +0.29 statistic is a maximum over candidate plans, not evidence of an implementation bug. The prospective recommendations below document the pre-result review; the manuscript follows this update.

## Verdict

Round 1 establishes useful counterexamples and reports an interesting failure. It does not identify the cause of that failure. The approved interventions can change that, but their outcomes remain unknown.

The strongest paper is a controlled study of replacing samples with point predictions inside two fixed learned models. The texture result supplies an objective-level warning, with substantial prior art. Do not present these as a universal impossibility theorem for JEPAs.

I recommend **The Half-Open Wall: When Mean Predictions Fail**. The proposed “Why a JEPA Cannot Plan Where the Future Branches” claims too much. Branching does not imply failure. Proposition 3 supplies a counterexample to that title, and exact expected-value predictors can support control without sampling.

Keep the wall metaphor only if E6 actually shows the decoded midpoint. A latent midpoint need not decode to a half-open wall. The current grid rollout largely deletes the wall, including before the reveal.

## Reviewer 1

The reviewer is right about the sampling confound, seed uncertainty, failed-probe overclaims, goal mismatch, reward-head audit, and the unresolved residual gap. These are central defects, not presentation preferences. The finite-sample SIGReg correction and exogenous-seed argument are also necessary.

The state-blind alternating baseline invalidates still-world survival as evidence that state feedback is necessary. It does not invalidate the representation counterexample. Conversely, successful goal control does not prove that the encoder represents position: an adequate state-blind strategy could also succeed.

The proposed oracle conditional-mean example is useful if its off-support scoring rule is explicit. It would establish a mechanism under that rule, not explain the learned model automatically. E2 is more directly matched to the empirical claim, but still inserts potentially unsupported latent states.

The exact-value-head objection is correct and consequential. Calling it vacuous would confuse an available escape from a theorem with a practical baseline requirement. A terminal value can reward reaching a state where future feedback helps.

The reviewer need not demand a large oracle-value experiment. The manuscript must instead restrict Remark 1 to the information available to both rollout and score.

## Reviewer 2

The reviewer is right that a vanished deterministic wall cannot be explained by averaging the unrevealed gap alone. E6 should begin before the reveal. The reviewer also correctly identifies preliminary-checkpoint provenance, unsupported mechanism attribution, and the missing slow-feature literature.

The claimed 57% underfit is a useful arithmetic diagnostic, not a population estimate with uncertainty. The held-out comparison demonstrates a better tested predictor. Establishing the exact excess-risk fraction requires matched targets, evaluation details, and sampling uncertainty.

The Gaussian-regularizer explanation is plausible but overstated. Covariance rank is not intrinsic dimension. Even one continuous variable can generate many linearly independent nonlinear features. Arbitrary measurable maps can transform an atomless scalar into a Gaussian vector. Smooth low-dimensional encoders and finite discrete observations impose different restrictions.

Thus, “two state degrees of freedom cannot fill 192 dimensions” needs a regularity or finite-support assumption. E7 measures an optimization response; it does not establish that dimensional argument.

A reproduced published LeWM result would strengthen external validity. It is not necessary for an objective-level counterexample or a carefully scoped within-model intervention. A budget sweep cannot establish convergence either.

Proposition 3 needs no wind experiment. The survival task violates its quadratic-cost assumption. Retain the theorem as a boundary, remove claims that those worlds test it.

The review's strongest requests are concrete. Its prospective “solid accept” is not evidence that E1-E10 will resolve every attribution issue.

## Weaknesses neither review resolves

**Finite sample averages are stochastic transitions.** Mean-of-eight resamples a residual with conditional covariance Sigma/8. Calling it deterministic is false unless its randomness is fixed. Fixing randomness makes an approximation deterministic, but does not make it the exact expectation. Report K sensitivity and conditional Monte Carlo error before attributing a gap to mean replacement.

**Averages can leave model support.** E2 changes inputs seen by the predictor, policy, and reward heads. Failure can reflect their extrapolation at latent midpoints. This is a meaningful failure of mean substitution, but not proof that lost uncertainty is the sole cause.

**Marginally plausible samples need temporal consistency.** Resampling a hidden gap independently at every step changes the environment. Preserve each sampled gap within a rollout until the next wall. Check that revealed gaps remain fixed in learned rollouts.

**Oracle MPC is not a ceiling.** Receding-horizon open-loop search can exploit later observations through replanning. Exact short-horizon reward optimization is not an upper bound on long-horizon survival. E4 has no value head, whereas the learned heads-MPC does. Label E4 a simulator reference with a different score, and report that difference.

**The oracle must share the agent's information.** Forking a simulator must not reveal its stored gap before the reveal. Branch draws must respect the posterior and remain coupled along each candidate trajectory. Random tie-breaking and branch integration need their own uncertainty estimates.

**Edge starts do not automatically require feedback.** Starts near different borders may still admit a common safe first action. Show incompatible safe actions or compute the best state-blind value for the specified reset distribution. A failed alternating baseline alone proves little about the best state-blind policy.

**The seed hierarchy matters.** Three policies trained on one world model are not three independent model fits. Report model-seed means, policy variation within each model, and paired within-model mode differences. Separate the fixed-flow-checkpoint claim from model-training reproducibility if E3 does not retrain flow.

**Evaluation protocols are not identical.** Planning rejects starts that die during warm-up; policy evaluation begins at reset. Paired random keys do not remove that selection. Compare each method with its own harness reference, or rerun a common protocol.

**Wall mass is a thresholded probe output.** Low mass can mean subthreshold intensity, decoder extrapolation, wrong location, or true omission. E6 needs soft intensity, latent distances, and the encoded true midpoint together.

**The remedies have different guarantees.** A second moment suffices for quadratic scoring, not arbitrary collision costs. Reconstruction, reward learning, and inverse dynamics can each miss task-relevant information. None is a general cure under finite capacity or partial observation.

**The plan cannot promise that scaling fails.** E5 tests two larger training budgets, not capacity or asymptotic training. Report the measured range. Do not write that more steps or more capacity cannot help.

## The amplitude proposition

The literal candidate is false without qualifications. If a small feature is predictable from context, the unrestricted MSE optimum predicts it exactly. Dropping it incurs excess risk; it is not an alternative optimum. If the feature is a fresh symmetric innovation, its conditional mean vanishes because it is unpredictable.

Appending an epsilon-scaled coordinate to a Gaussian nuisance also fails the strongest proposed claim. Its variance is epsilon squared, so its full marginal is not a standard Gaussian. Adding Gaussian noise directly to the feature usually destroys exact decodability.

A clean exact-Gaussian construction uses quantile cells. Partition the real line into intervals of width 2 epsilon. Draw a cell with its standard-normal probability mass and a uniform nuisance coordinate within it. Encode a fair bit into the lower or upper conditional-quantile half of that cell. Add independent Gaussian nuisance coordinates for the remaining dimensions.

The resulting vector is exactly standard Gaussian. The bit is recovered by checking which quantile half contains the first coordinate. Paired encodings with the same nuisance differ by less than 2 epsilon. Their midpoint therefore incurs positive MSE at most epsilon squared.

The latent reveals the nuisance as well as the bit. With independent fair bits across time, the optimal next-latent predictor is precisely that midpoint. It contains no information about the realized next bit. A current-bit classification reward makes the feature control-relevant, with value one from the encoded observation and one-half without it.

This proves an arbitrarily small upper bound, not a universal equality to epsilon squared. It requires the stated measurable encoder and predictor classes. The decoder becomes arbitrarily sensitive as epsilon decreases. The proof belongs in the paper with those limitations visible.

The proposition connects two concerns through latent geometry: exact marginal non-collapse and perfect decodability need not give a feature appreciable prediction-loss weight. It does not unify information deletion and certainty-equivalent planning into one causal mechanism. Nor does it explain deletion of a predictable wall by an optimal regressor.

A hostile reviewer can correctly call the quantile encoding artificial and the fresh-bit averaging classical. The response is its narrow guarantee: even exact Gaussianity supplies no lower bound on task-feature amplitude. E6 must independently test whether a related geometry occurs in the learned encoder.

## Claims E1-E10 can carry

If paired seed results support both interventions, claim that sample rollouts outperform finite-sample mean rollouts in two fixed learned-model families on the gate task. Give effect sizes and uncertainty. Scope mean convergence using K sensitivity.

If E6 shows small wall amplitude, report the amplitude relative to predictor residuals and its relation to control. Do not call that relationship causal without an intervention that changes amplitude while preserving other factors.

If E7 changes recoverability with regularizer strength, report a dose response for this architecture and dataset. It neither proves optimizer selection of Proposition 1 nor validates every assumption of the innovation bound.

E8 exposes weak task specifications. E10 supports a stronger representation test only after establishing that its reset distribution requires feedback. E9 localizes scoring failures but does not automatically remove them.

Craftax remains exploratory. Its 94.5% goal success and poor held-out reward head demonstrate task dependence, not a replicated branching mechanism.

If either matched mean arm recovers equally well, abandon “recovers, twice.” The paper then studies underfitting, model support, or one successful intervention. Red placeholders must never conceal an unmeasured directional claim.

## Three sentences to remember

The mean of two possible futures need not be a possible future.

A squared-error JEPA targets that mean, while marginal non-collapse does not guarantee that its latent geometry preserves what matters for control.

The decisive experiment uses each learned model twice, with samples and their averages, and measures the resulting policies in the real world.

After E1-E3 establish the result, the last sentence may become: “Replacing averages with samples improves control within the same learned model, in both model families.”

## Literature checked

Sobal et al. already show the fixed-background slow-feature problem; our delta must be explicit. [Primary source](https://arxiv.org/abs/2211.10831).

Tang et al. study self-predictive learning dynamics, collapse avoidance, and spectral structure. [Primary source](https://proceedings.mlr.press/v202/tang23d.html).

Ni et al. connect state and history abstractions through self-prediction and analyze learning objectives. [Primary source](https://openreview.net/forum?id=ms0VgzSGF2).
