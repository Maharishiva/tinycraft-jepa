# Exact one-step variance audit in a frozen redcube encoder

## Decision

**Worth a small, fixed-budget run.** It cleanly tests whether replacing a stochastic next-latent distribution by its exact mean changes an action ranking under the stated expected squared latent-goal cost. Both alternatives use the same fixed encoder, goals and exact transition branches. There is no learned-predictor error, fitted head, value probe, policy optimizer, reward change or latent rescaling in the comparison.

It is **not** a test of survival, physical goal-reaching performance, representation sufficiency, or the accuracy of the fitted JEPA predictor. In the proposed safe cohort all one-step environment rewards are identical. It also does not test recursive-mean error; the user's exact finite-support analytical recursion is the cleaner demonstration of that separate issue.

No GPU work has been launched for this design.

## Fixed comparison

Let `s=(p,w,b,t)` be a sampled simulator state: cube position, current wind, episode texture, and time. The encoder `f` is fixed and evaluated with frozen running statistics. For each action `a` and wind innovation `delta`:

\[
w'_\delta=\operatorname{clip}(w+\delta,-2,2),\qquad
p'_{a\delta}=p+u_a+w'_\delta,\qquad
z'_{a\delta}=f(R(p'_{a\delta},b)).
\]

There are nine innovation labels, `delta in {-1,0,1}²`, with independent coordinate probabilities `(0.1,0.8,0.1)` in that order. The four actions are the original 3-pixel cardinal translations. This is the transition at `studies/redcube/env.py:84–107`.

Keep all nine labels with their original probabilities even when clipping produces duplicate wind states. Never average uniformly over the unique clipped outcomes.

For a goal image on the same background, let `z_g=f(R(g,b))`. In coordinate-mean squared units, with `D=512`, compute:

\[
\begin{aligned}
\mu_a &= \sum_\delta q_\delta z'_{a\delta},\\
M_a(g)&=\|\mu_a-z_g\|^2/D,\\
V_a&=\sum_\delta q_\delta\|z'_{a\delta}-\mu_a\|^2/D,\\
J_a(g)&=\sum_\delta q_\delta\|z'_{a\delta}-z_g\|^2/D
       =M_a(g)+V_a.
\end{aligned}
\]

Compare minimizing `M` with minimizing `J`. The second is the exact expectation over the environment's finite branch law, up to renderer/encoder/arithmetic precision. Do not sample a flow model or call the learned predictor. A discrepancy then comes from the omitted variance term under this fixed feature-space objective, rather than a fitted mean failing to be the mean.

## What the conditioning means

Call the quantity **the oracle next-latent mean conditioned on physical state and episode texture**. It is not automatically the conditional mean available to the learned predictor from its latent history.

For a valid no-reset history, the actual current wind is recoverable from positions and the previous action:

\[
w_t=p_t-p_{t-1}-u_{a_{t-1}}.
\]

Verify this equality exactly for every admitted context. On the black-background primary condition, raw images expose the cube's position unambiguously in the safe interior; a raw observation/action history thus determines the relevant transition state. A successful empirical position readout does not prove that the latent history is exactly sufficient, so retain the full-state-oracle qualification.

In the episode-static gray condition, stored texture is additional oracle information: the eight frames may not expose every background pixel hidden under the cube. This is another reason to use the black background as primary and to label the textured extension carefully. No simulator state is supplied to a fitted model or controller; it is used only to enumerate the evaluation transition law and render counterfactual observations.

## Checkpoints and ordering, fixed before costs are observed

1. **Primary:** existing `off_wind/tile_factored/model_probe.npz`, model seed 0, 16×32=512 latent coordinates. Its already audited nonlinear position readout is 0.126-pixel L1, which establishes that the tested readout can recover current position. The inventory records `/dev/shm/stoch/redcube/off_wind/tile_factored/model_probe.npz` on e2e; recheck identity before use.
2. **Secondary:** existing `gray_episode_wind/tile_factored/model_probe.npz`, seed 0, same latent size. Its corresponding audited readout is 0.717-pixel L1. The inventory records `/dev/shm/stoch/redcube/gray_episode_wind/tile_factored/model_probe.npz` on e2e.

Both conditions are declared now. Report both whether positive or null. Do not replace a checkpoint, goal bank or context cohort after examining variance/rankings. Additional saved encoder seeds can be listed and admitted before any objective outcomes are observed; otherwise this is an encoder-specific mechanism check, not a model-family estimate.

The position-readout numbers only motivate choosing an encoder that retains usable state. No position probe is loaded or used in the new calculation. Record checkpoint bytes/hash, configuration and all source hashes before collection.

## Fresh contexts and goals

- Generate a fresh uniform-random-action collection with 128 independent environment streams and 512 transitions per stream. Use fixed environment seed `20260923`; use the same physics/action key schedule across the two background conditions. No world-model training uses these data.
- Require an eight-frame context contained in one episode. Eligibility uses past resets only, not the recorded next outcome. Uniformly choose 1,024 eligible context IDs without replacement with selection seed `20260924`. If fewer exist, use all and report the shortage. Save the chosen IDs before evaluating the encoder.
- Reconstruct/retain the exact current simulator state including the texture key. Verify rendered current/context frames against collection and the wind-inference identity above.
- The primary evaluation keeps only sampled contexts for which **all 36 action/innovation branches** remain within `SAFE_LO=2` and `SAFE_HI=55` in both coordinates, and `step+1 < MAX_STEPS=128`. This excludes every possible terminal/reset branch, including the time cap. Do not merely filter on the one recorded continuation. Report the sampled count, admitted count, separate border/cap exclusions, positions and all 25 wind-state frequencies. Do not refill excluded contexts according to encoder outcomes.
- Use every goal in the fixed 5×5 coordinate grid `{8,18,28,38,48}²`, with equal weight. Each goal is a rendered observation with the cube at that position and the **same episode texture** as its context. Goals do not depend on predictions, costs, variance, success, or the future random branch. Keep goals equal to or near the current position; do not selectively remove ties or inconvenient cases.
- Save all context–goal pairs and their generation seeds/hashes before scoring. Do not search for a visually dramatic goal or context to become the main result. An illustrative example, if displayed, should be the first fixed-index robust-disagreement case, with the complete distribution alongside it.

All-branch safety is an explicitly selected interior cohort. It prevents terminal/autoreset effects from masquerading as a variance mechanism; it also limits the claim to one-step safe-interior decisions. Contexts/goals remain evaluation specifications, not modifications of the original environment or reward.

## Primary and secondary endpoints

For each context and each of its 25 goals, retain all four `M`, `V` and `J` values.

Use a tolerance fixed before results:

`tau = 1e-10 * max(1, max(abs(M)), max(abs(J)))`

with float64 host arithmetic. Define `A_M={a: M[a] <= min(M)+tau}` and `A_J={a: J[a] <= min(J)+tau}`.

**Primary:** fraction of pairs with disjoint tolerance-optimal action sets, and the minimum possible exact-cost regret among mean-objective minimizers:

\[
r_{\min}=\min_{a\in A_M}J_a-\min_aJ_a.
\]

This prevents arbitrary tie-breaking from becoming the result. Also report the maximum regret across `A_M`, and regret under a uniform tie-break within `A_M`. Report the unfiltered distribution, including zeros; do not summarize only disagreements. An ordinary lowest-index argmin disagreement rate may be shown as a labeled supplementary statistic, not as the principal finding.

Report per-context action-variance spread `max(V)-min(V)`, best/second-best mean-cost gap, expected-cost gap, and direct decomposition residual. Provide raw coordinate-mean squared units for each checkpoint separately. Do not compare the magnitude of raw latent regret across independently trained encoders as though their metrics were physically calibrated. Optional cost-range normalization must retain the raw values and explicitly exclude/report near-flat ranges.

**Secondary physical diagnostic:** compute exact `E||p'−g||²` in pixel-squared units from the same branches. For each objective's selected action set, report this physical cost and its regret relative to the best physical-cost action. The difference between mean-latent and full-latent choices in physical cost can have either sign; it is not a nonnegative latent-cost regret and must not be called a demonstrated control benefit.

## Essential negative controls

1. **Identity-position encoding.** Set `z=p` only inside the analytic test harness. Because actions translate position while the wind law is fixed,
   `Cov(p'|s,a)=Cov(w'|w)` is action independent. Therefore mean-position and expected-position squared-goal objectives must have identical tolerance-optimal action sets. Wind variance per coordinate is 0.20 for current wind −1/0/+1 and 0.09 for current wind ±2; the 2D trace is 0.40, 0.29 or 0.18. Any robust ranking disagreement is an implementation error.
2. **Affine-position encoding.** A fixed predetermined matrix and offset applied to position must likewise yield action-independent feature covariance and no robust ranking difference. No learned weights or goal-dependent transform are used.
3. **Deterministic branch law.** Concentrate probability on the zero-innovation branch of the same enumerated images. Then `V=0` and both objectives coincide. This is a software identity check, not a new environment result.
4. **Exact user examples.** For goal zero, action A produces ±1 fairly and B produces 0.1: mean-only costs 0/0.01 versus exact costs 1/0.01. Separately, `X'=X²+ξ`, iid fair `ξ∈{−1,+1}`, `X0=0`, has `E X2=1` but recursively applying `g(x)=x²` gives zero. Do not substitute Gaussian noise or different constants in the displayed examples.

The first two controls are scientifically important: stochastic motion alone does not force one-step squared-goal misranking. A nonlinear encoder can make latent variance action dependent even when position variance is constant. Conversely, deterministic expectation/value-equivalent models can preserve the necessary expected costs without generating samples; the test is not a proof that stochastic generative models are required.

## Code blueprint

The numerical core needs no learned heads or predictor:

```python
delta = np.array(list(itertools.product([-1, 0, 1], repeat=2)))
q1 = np.array([0.1, 0.8, 0.1], dtype=np.float64)
q = np.outer(q1, q1).reshape(9)
next_wind = np.clip(wind[:, None, :] + delta[None], -2, 2)
next_pos = (pos[:, None, None, :]
            + moves[None, :, None, :]
            + next_wind[:, None, :, :])              # N,A,K,2
safe = ((next_pos >= 2) & (next_pos <= 55)).all((1, 2, 3))
safe &= episode_step + 1 < 128

# Render admitted next_pos using the stored same-episode texture; encode every
# branch and goal under the same fixed, padded inference shape.
z = encoded_branches.astype(np.float64)              # N,A,K,D
goal = encoded_goals.astype(np.float64)                # N,G,D
mu = np.einsum('k,nakd->nad', q, z)
var = np.einsum('k,nak->na', q, ((z-mu[:, :, None])**2).mean(-1))
mean_cost = ((mu[:, :, None]-goal[:, None])**2).mean(-1)  # N,A,G

# In small context blocks, compute the direct expectation independently from
# branch-to-goal differences. Avoid a giant N*A*G*K*D temporary.
exact_cost = np.einsum(
    'k,nagk->nag', q,
    ((z[:, :, None]-goal[:, None, :, None])**2).mean(-1))
assert_close(exact_cost, mean_cost + var[:, :, None])
```

Production implementation should use 32-context blocks for the direct cost calculation, preserve the canonical branch order and retain all duplicate clipped outcomes. The independent direct-versus-decomposed calculation is a verification, not the sole source of reported exact cost.

## Numerical and provenance requirements

- Freeze parameters and BatchNorm statistics; hash before/after. Assert checkpoint latent width 512, four native actions, and the expected architecture/configuration.
- Use the checkpoint's original compute dtype. The previous forecast audit showed batch-shape-dependent BF16 differences, so use one fixed padded encoder frame batch (e.g. 2,048) for branches and goals, and enforce same-shape reference/permutation parity before accepting results. No relaxed tolerances or precision changes after inspecting action rankings.
- Compute weighted means, centered variances, costs and regret in float64 after encoding. Save the maximum absolute/relative identity residual. Require it to be below the declared tolerance before computing robust mismatches; a failure blocks interpretation rather than becoming a finding.
- Validate branch transition positions and clipping against the simulator's `wind_walk`/step semantics on fixed test states, including wind extrema. The nine probabilities must sum to one. Verify zero-probability/duplicate handling with the identity controls.
- Save fresh context IDs/frames or exact reconstruction state, all texture keys, action history, goals, branch weights/states, encoded branch/goal arrays, every action cost, minimizer sets, regret, safety exclusions, seed/config/source/checkpoint hashes and runtime. No best checkpoint/example selection.
- For uncertainty, average uniformly over goals within each context, then bootstrap full collection streams using shared resamples across objectives/conditions. Do not treat 25 goals per context or overlapping histories as independent model replications. If several encoder seeds are admitted, report them separately.

## Compute estimate and interpretation gate

For 1,024 admitted contexts, direct enumeration requires 36,864 branch encodings. The black-background goal bank can reuse 25 identical goal images; the textured condition adds at most 25,600 goal encodings. The existing 512-dimensional encoder handled larger extraction jobs in roughly a minute after compilation, so **a few minutes per checkpoint** is a reasonable planning estimate, to be measured by a small shape/identity smoke. Collection/rendering and host cost arithmetic add modest overhead. No training is needed.

A one-step positive result with nontrivial, tie-robust expected-latent-cost regret is a clean demonstration of the variance-omission mechanism in a real trained representation. It does not attribute the prior forecast/control results to that mechanism. If disagreements are absent or confined to near ties, report that; do not expand the goal bank or select another checkpoint until a new follow-up protocol is explicitly declared.

The separate recursion example should remain analytical. Recursing an oracle conditional latent mean is not straightforward: a mean can lie off the support of actual encodings, where an oracle conditional model is not uniquely defined. Forcing a decoder/inverse map or learned predictor into that second step would reintroduce the confound this exact one-step test removes.

**Recommended paper role:** one compact operator diagram and a table showing exact conditional mean, omitted variance, corrected expected cost, robust disagreement rate and latent regret. Keep the three redcube observation/dynamics cases as the empirical storyline; do not replace them with another large policy-head ablation table.

## Implemented evaluator, tests and ready commands

Implementation is `studies/revision/oracle_variance.py`; independent CPU controls are in `studies/revision/test_oracle_variance.py`. All 15 tests pass, including the exact user examples, every clipped wind state, identity/affine encoders, a realizable nonlinear-encoder positive control, stable centered variance, ties and action permutations. Independent source and statistical reviews found no blocking defect. Actual renderer/encoder execution remains subject to the strict GPU smoke.

The suite also passes in the actual MG virtual environment with GPU visibility disabled and an assertion that JAX was not imported. A separate actual-simulator CPU-only check (`JAX_PLATFORMS=cpu`) collected four streams × 24 transitions in both conditions, verified wind inference, admitted seven of eight sampled contexts, and checked 28 simulator action transitions against the exact enumerated support in each condition. The complete collected state/action/done arrays were bit-identical across the two backgrounds. This verifies helpers only and supplies no model-mechanism result.

Source SHA256: `37b5bac0ad36830dbd4e1f4e4d0c3694670d168cff2991025c81d511e766a4b9`.
Test SHA256: `6e0e801b24a7af3fdb018c1e76a558c4995d847c0eb171d4bee43a82cc0a5f1e`.

The actual float64 numerical core was timed locally at **0.63 seconds** for 1,024 contexts × 4 actions × 9 branches × 512 features and 25 goals, with maximum decomposition residual `1.33e-15` on random finite test arrays. That is a CPU arithmetic benchmark, not an experimental outcome. Model encoding and compilation dominate the estimated few-minute checkpoint evaluation.

Fresh collection retains original simulator states/actions/dones rather than every rendered frame, reducing memory substantially. For the two admitted conditions rendering is deterministic given the saved position and episode texture; images are generated through the original renderer only when required. Complete collection states, selected histories, texture keys, branch/goal encodings, costs and minimizer sets are retained. The driver writes `contexts_fixed_before_encoding` and its artifact hashes before evaluating any encoder features. No existing data or checkpoint is modified.

MG paths and checkpoint hashes were verified by the experiment owner:

- `/dev/shm/redcube/off_wind/tile_factored/model_probe.npz`: `c7efc430a0d37db232aea702d79120ac69027f6c531b4497ca5486ef34111b1d`.
- `/dev/shm/redcube/gray_episode_wind/tile_factored/model_probe.npz`: `55dbaa39693fbcfc5af0fe95f200949de954084c070496fc83f12e6b16a07798`.

Run from `/home/rldoeswork/personal/tinycraft-jepa-iclr-20260919`, **only after the owner hands over MG GPU 1**. Current order is visual export → goal evaluation → this diagnostic. The other GPU remains allocated to the target-gradient campaign.

First, a runtime-only smoke; its outcomes are excluded from the final mechanism results:

```sh
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.45 OPENBLAS_NUM_THREADS=4 \
  /home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python -m studies.revision.oracle_variance \
  --condition off_wind \
  --model /dev/shm/redcube/off_wind/tile_factored/model_probe.npz \
  --expected-model-sha256 c7efc430a0d37db232aea702d79120ac69027f6c531b4497ca5486ef34111b1d \
  --contexts 32 --envs 16 --steps 64 --encoder-batch 2048 --bootstrap-reps 100 \
  --out runs/revision/oracle_variance/smoke_off_wind
```

Only after status `complete`, exact same-shape/permutation parity, unchanged state hashes, simulator support checks and all three negative controls pass, run the fixed production protocol:

```sh
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.45 OPENBLAS_NUM_THREADS=4 \
  /home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python -m studies.revision.oracle_variance \
  --condition off_wind \
  --model /dev/shm/redcube/off_wind/tile_factored/model_probe.npz \
  --expected-model-sha256 c7efc430a0d37db232aea702d79120ac69027f6c531b4497ca5486ef34111b1d \
  --out runs/revision/oracle_variance/off_wind

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.45 OPENBLAS_NUM_THREADS=4 \
  /home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python -m studies.revision.oracle_variance \
  --condition gray_episode_wind \
  --model /dev/shm/redcube/gray_episode_wind/tile_factored/model_probe.npz \
  --expected-model-sha256 55dbaa39693fbcfc5af0fe95f200949de954084c070496fc83f12e6b16a07798 \
  --out runs/revision/oracle_variance/gray_episode_wind
```

Defaults are the fixed 1,024 sampled contexts, 128×512 fresh collection, 25 goals, canonical 2,048-frame encoding, float64 costs and 2,000 collection-stream bootstrap repetitions. Do not adapt those settings according to a positive or null result. A failed parity/control check or empty safe cohort is a diagnostic failure, not a completed positive/negative mechanism result.
