# Cropped-prefix supervision: a population counterexample

Verified 2026-09-19. This is a mathematical construction and a prospective
experimental motivation, not a result from the planned gray-cube fits.

**Verdict.** The proposed `1/H` versus `0` comparison is correct. It provides a
cleaner main intuition for a context-length intervention than the existing
small-amplitude measurable-bit example: the encoders are ordinary linear
projections, the predictors can be action-gated linear maps, and the uncertainty
disappears when one preceding observation is supplied. It is a counterexample
to a guarantee, not an account of what gradient training must learn.

## Process, population, and information sets

Draw independently at the start of each episode

\[
U\sim\operatorname{Unif}\{-1,+1\},\qquad P_0,B\sim N(0,1).
\]

The unobserved mode `U` and nuisance `B` remain fixed. During training draw actions
\(A_t\sim\operatorname{Unif}\{-1,+1\}\) independently of these variables and of
one another, and set

\[
P_{t+1}=UA_tP_t,\qquad X_t=(P_t,B).
\]

A training window starts at a fixed index `s`, or at an index selected independently
of the trajectory, and contains `H+1` consecutive observations from one episode.
There is no recurrent state carried in from before the crop. Define

\[
Z_t=f(X_t),\qquad
C_k=(Z_s,\ldots,Z_{s+k-1},A_s,\ldots,A_{s+k-1}),\quad 1\le k\le H.
\]

Thus target `k` is the next embedding `Z_{s+k}`, its current action is included,
and all preceding embeddings within the crop are teacher-forced. In particular,
`C_1` contains only `Z_s,A_s`; it does not contain `Z_{s-1}`. Define the optimal
population prediction losses

\[
L_{\rm dense}^*(f)=\inf_g\frac1H\sum_{k=1}^H
\mathbb E\bigl[(g_k(C_k)-Z_{s+k})^2\bigr],\qquad
L_{\rm last}^*(f)=\inf_g\mathbb E\bigl[(g_H(C_H)-Z_{s+H})^2\bigr].
\]

The same causal predictor class is used in both objectives. It contains copy,
zero, and a two-step lag multiplied by the product of its two associated action
signs. These are finite action-gated linear maps. A shared predictor can implement
the first-position boundary with a positional mask and the same two-step rule at
all later positions. An ordinary linear map on concatenated raw latent/action
inputs, without action gating or interaction features, is not enough for the
claimed exact predictor.

`L_last` is a one-step target with the longest available teacher-forced context.
It is not an `H`-step open-loop prediction from the initial frame.

## Proposition and proof

For the two scalar linear encoders

\[
f_{\rm state}(p,b)=p,\qquad f_{\rm nuisance}(p,b)=b,
\]

both latent marginals are exactly `N(0,1)` at every training time and hence have
zero population SIGReg. For `H >= 2`,

| Encoder | Dense prefixes | Last target only |
| --- | ---: | ---: |
| State `P_t` | `1/H` | `0` |
| Nuisance `B` | `0` | `0` |

**Proof.** Let \(V_s=\prod_{j=0}^{s-1}A_j\). Then
\(P_s=U^sV_sP_0\). Conditional on `U` and the whole action sequence, this is a
sign change of an independent centered standard Gaussian. Consequently `P_s` is
standard Gaussian and independent of `U` and of the action sequence. It is also
independent of `B`. This proves both marginal claims, and remains valid when the
window start is sampled independently.

For the state encoder, the first target satisfies

\[
\mathbb E[P_{s+1}\mid P_s,A_s]=0,\qquad
\operatorname{Var}(P_{s+1}\mid P_s,A_s)=P_s^2.
\]

The minimum first-target MSE is therefore `E[P_s^2]=1`, attained by zero. For
every subsequent target, two consecutive transitions cancel the mode:

\[
P_{s+k}=A_{s+k-1}A_{s+k-2}P_{s+k-2},\qquad k\ge2.
\]

The right-hand side belongs to `C_k`; the stipulated gated-linear predictor
therefore attains zero MSE at all these targets. Nonnegativity and the first
target's conditional-variance lower bound establish `L_dense*=1/H` and
`L_last*=0`. For the nuisance encoder every target is `B`, so copying the most
recent latent gives zero for both objectives. ∎

Adding any nonnegative multiple of population SIGReg leaves this table unchanged.
Under dense supervision the nuisance has a strictly smaller objective than this
useful state encoder. Last-only supervision removes that strict preference but
continues to admit the nuisance as a zero-loss solution.

The Gaussian claims describe the ensemble across independent episodes. Within
one episode `B` and `|P_t|` are constant; an episode's empirical latent distribution
is not Gaussian. Per-time batches sampled from independent episodes have the
specified Gaussian population law. Finite-batch SIGReg need not be zero, and this
proposition does not equate its gradient behavior across the two encoders.

## Control separation

Consider deployment from a fresh episode and reward
\(\mathbf1\{P_{t+1}>0\}\) at a fixed terminal decision after at least one previous
transition. The controller knows its actions and can retain its latent history.
A state-encoder controller can use

\[
A_t=A_{t-1}\operatorname{sign}(P_{t-1})
\quad\Longrightarrow\quad P_{t+1}=|P_{t-1}|>0
\quad\text{almost surely}.
\]

Equivalently, observing `P_{t-1},P_t,A_{t-1}` identifies
\(U=P_t/(A_{t-1}P_{t-1})\) almost surely, and then the controller chooses
\(A_t=\operatorname{sign}(UP_t)\). Two observations suffice; this construction
does **not** establish that both are necessary, because the preceding formula
ignores the most recent observation.

A nuisance-only controller sees only `B`, time, its own action history, and
independent private randomness. It receives no rewards, state-dependent
termination signals, or earlier state-informed actions that could leak the sign.
Couple each episode with an episode having `-P_0` and the same `U,B` and private
randomness. The controller takes the same actions inductively, and every position
in the paired episode is negated. The two terminal signs are opposite almost
surely, so success is exactly `1/2`. This remains true for a finite stopping time
based only on the permitted state-blind information. State-dependent feedback
would require a separate argument and is excluded here.

## A robust encoder lower bound

Let a square-integrable vector encoder `f` satisfy the sign-separation condition

\[
\|f(p,b)-f(-p,b)\|\ge2c|p|\qquad\text{almost surely},\quad c>0.
\]

For any first-prefix predictor, condition on `P_s,B,A_s`. The hidden mode remains
fair, so the target has the two equally likely values
\(y_+=f(A_sP_s,B)\) and \(y_-=f(-A_sP_s,B)\). The identity

\[
\tfrac12\|v-y_+\|^2+\tfrac12\|v-y_-\|^2
=\left\|v-\tfrac12(y_++y_-)\right\|^2
+\tfrac14\|y_+-y_-\|^2
\]

gives first-target loss at least \(c^2\mathbb E P_s^2=c^2\). Therefore
\(L_{\rm dense}^*(f)\ge c^2/H\) for squared Euclidean loss. With the manuscript's
coordinate-mean convention in dimension `D`, the bound is `c^2/(HD)`.
A statewise co-Lipschitz condition
\(\|f(p,b)-f(q,b)\|\ge c|p-q|\) implies this sign-separation assumption.
Gaussianity is not needed for the lower bound; if such an encoder also has a
standard Gaussian marginal, its population SIGReg remains zero.

Sign separation alone does **not** guarantee zero last-target loss. A sufficient
extra condition is that reflection factors through the latent: there exists a
measurable `T` such that `f(-p,b)=T(f(p,b))`. The predictor then copies the two-step
lag when the two-action product is positive and applies `T` when it is negative.
Full joint Borel injectivity of `f(p,b)` is another sufficient condition, because
the predictor can decode `(p,b)`, reflect `p`, and re-encode. Injectivity only in
`p` for each fixed, unrecoverable `b` is insufficient. These extensions permit
general measurable predictors; exact gated-linear realization requires the
reflection map itself to be linear or affine in latent coordinates.

## Optional strengthening within a transparent linear encoder class

The two-encoder comparison does not require claims about all learned encoders.
If useful, a narrow stronger statement is available without measurable coding.
Restrict the encoder class to
\(f_{a,b}(p,B)=ap+bB\), \(a^2+b^2=1\). Every encoder in this class has the same
standard Gaussian marginal. Its first target has conditional mean `b^2 Z_s` and
minimum MSE

\[
1-b^4=a^2(2-a^2).
\]

Hence `L_dense* >= a^2(2-a^2)/H > 0` whenever `a != 0`, whereas `a=0,b=+-1` has
zero loss. Thus every global minimizer **within this normalized linear class**
ignores the state under dense supervision. Last-only supervision includes the
pure state and pure nuisance encoders among its zero-loss optima. This says
nothing about gradient convergence, local minima, nonlinear encoder classes, or
the choice made by finite networks. In particular, mixture encoders need not have
zero loss on longer prefixes, so do not claim their total dense risk equals the
displayed first-target contribution.

To check the conditional mean, `U` is independent of `(P_s,B,Z_s,A_s)` and has zero
mean, while the Gaussian regression identity gives `E[B|Z_s]=b Z_s`. The target
has second moment one, and the conditional mean has second moment `b^4`; the
Bayes MSE is their difference.

## Interpretation, attribution, and empirical boundary

- The uncertainty is due to a hidden episode-static mode and the omission of
  available preceding context. Given the mode, dynamics are deterministic. This
  is not a lower bound caused by fresh process noise.
- The first-prefix penalty is irreducible under that prefix's information set;
  it is avoidable by supplying one preceding context frame. It is not avoidable
  by fitting a better predictor to the same first prefix.
- The dense objective adds this penalty to the state encoder but not to the
  nuisance. Removing the penalty does not itself select useful state or prove
  improved downstream control after optimization.
- Increasing `H` dilutes this construction's sole costly prefix to weight `1/H`.
  A general weighted-prefix objective instead pays exactly the weight assigned
  to its first prefix for the state encoder. A burn-in frame followed by the
  same number of supervised targets also removes this construction's penalty;
  last-target-only is one intervention, not the only one.
- The nuisance/identity-predictor construction is established prior work:
  [Sobal et al. (2022), Section 2.1](https://arxiv.org/html/2211.10831#S2.SS1).
  The conditional-variance calculation is classical. Present the explicit
  context-dependent comparison as an explanatory proposition, without claiming
  a new general JEPA impossibility theorem or a novelty priority beyond a proper
  literature search.
- The current gray-cube wind code includes fresh random wind increments
  (`studies/redcube/env.py:84-87,100-102`). It is not this static-mode process.
  In particular, longer context need not eliminate all forecast uncertainty in
  that environment. The planned six matched dense-versus-last fits can test
  whether the intervention helps; this construction cannot predict their result
  or establish the cause of previously observed failures.
- Last-only changes the number and positions of supervised losses as well as
  their context length. Same architecture, samples, updates, and Gaussian
  regularization do not on their own isolate context from supervision density
  or gradient variance. A matched burn-in or full-context target control would
  make a stronger causal test if the initial result warrants it.

## Suggested compact manuscript wording

> **A cost of cropped context.** Draw an unobserved episode-static sign `U`,
> independent standard Gaussians `P_0,B`, and independent fair training actions
> `A_t`, with `P_{t+1}=U A_t P_t` and observations `(P_t,B)`. Both scalar linear
> encoders `f_state=P_t` and `f_nuisance=B` have an exactly standard Gaussian
> marginal, hence zero population SIGReg. On a teacher-forced window containing
> `H+1` frames, the first state prediction has conditional mean zero and expected
> variance one. Every later prediction is exact from
> `P_{t+1}=A_t A_{t-1}P_{t-1}`. Consequently, mean dense-prefix prediction loss is
> `1/H` for the state encoder and zero for the nuisance. Supervising only the last
> target, with `H>=2`, gives zero for both. The state history permits perfect
> control of the next sign after one transition, while a nuisance-only controller
> without state-dependent feedback succeeds with probability one half. Thus
> short-context targets can strictly penalize useful state even when the same
> dynamics are predictable from longer context. This context comparison builds
> on the established static-nuisance shortcut. It neither ensures that last-target
> training selects state nor establishes the cause of our finite-network results.
