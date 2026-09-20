# Exact variance omission: completed frozen-encoder diagnostic

## Verdict

**The exact conditional mean changes the optimal action under the stated expected latent-goal cost in a small, nonzero fraction of cases.** This occurs without any fitted predictor, reward head, value probe or policy-learning error. The measured effects are modest, and the experiment does not establish better physical goal-reaching or survival from correcting the latent cost.

The two predeclared conditions completed unchanged. Both use the same 843 all-branch-safe contexts and all 25 fixed goals: **21,075 context–goal pairs per checkpoint**. Of 1,024 past-valid sampled contexts, 181 were excluded by the prospectively defined border-safety rule; none was excluded by the episode cap. No goals, checkpoints or filters were changed after outcomes were observed.

## Main result

For each action, all nine wind innovations were enumerated with their original probabilities, including duplicated clipped outcomes. The two scores are

\[
M_a=\|\mu_a-z_g\|^2/512,\qquad
J_a=\mathbb E[\|Z'_a-z_g\|^2]/512=M_a+V_a.
\]

Disagreement means the two tolerance-optimal action sets are disjoint; regret is the exact expected latent cost of the mean-score action minus the minimum exact expected latent cost. Reported regret averages **all pairs, including zeros**. Confidence intervals resample entire collection streams, with goals averaged within contexts; they are conditional on the single fitted encoder in each condition.

| Fixed encoder condition | Disagreeing pairs | Robust action disagreement | Mean expected latent-cost regret |
|---|---:|---:|---:|
| Black background + wind | 559 / 21,075 | **2.65% [2.19, 3.15]** | **7.50e−5 [5.77e−5, 9.63e−5]** |
| Episode-static gray noise + wind | 227 / 21,075 | **1.08% [0.83, 1.36]** | **3.80e−6 [2.02e−6, 6.86e−6]** |

There were no consequential mean-cost ties in these results: minimum, maximum and uniform-tie regrets agree. Raw lowest-index argmin disagreement rates also equal the robust rates. Regret units are coordinate-mean squared distances **within each encoder**; their magnitudes must not be treated as calibrated physical units or compared across encoders as a model-family ranking.

The mean action-variance spread is 0.00655 in the black condition and 0.000589 in the gray condition, versus mean best/second-best mean-cost gaps of 0.0965 and 0.0135, respectively. This helps contextualize why the exact omission produces occasional rather than pervasive ranking changes in this cohort.

## Physical secondary result

The exact same branches also give `E||p'−g||²` in raw pixel-squared units. The difference below is physical cost of the mean-latent choice minus physical cost of the expected-latent choice; a positive value would favor the latter.

| Condition | Mean physical-cost difference | 95% collection-stream interval |
|---|---:|---:|
| Black background + wind | −0.169 px² | [−1.107, 0.666] |
| Episode-static gray noise + wind | +0.200 px² | [−0.433, 0.961] |

**These results do not establish a physical-cost improvement.** A better decision under expected latent squared distance need not improve physical distance when the learned latent metric is not calibrated to it. Every enumerated branch was safe and received the same original one-step environment reward, so no survival or reward advantage was measured.

## Fixed illustrative cases

These are the **first** robust disagreements in the fixed admitted-context/goal order, as specified before the run. They are not selected for maximum regret, visual appeal or physical benefit. Complete per-action values and state indices are in `runs/revision/oracle_variance/summary.json`.

- Black condition: current position `(29,32)`, wind `(0,1)`, goal `(8,38)`. Mean-only scoring selects down: mean cost 1.820421 versus right's 1.820614. The variance terms are 0.057573 versus 0.055685, so exact scoring selects right: 1.876299 versus down's 1.877994. Latent regret is 0.001695. In this example the exact-latent choice has **higher** physical squared-goal cost, 601.4 versus 445.4 px².
- Gray condition: current position `(20,29)`, wind `(−1,0)`, goal `(8,48)`. Mean-only scoring selects up; exact scoring selects left. Latent regret is 0.0001585. Here the exact-latent choice has lower physical cost, 425.4 versus 605.4 px².

The opposing physical outcomes illustrate the stated scope rather than support a general physical-control claim.

## Verification and provenance

- Existing 512-dimensional `tile_factored` checkpoints were unchanged. Checkpoint SHA256 values are `c7efc430a0d37db232aea702d79120ac69027f6c531b4497ca5486ef34111b1d` (black) and `55dbaa39693fbcfc5af0fe95f200949de954084c070496fc83f12e6b16a07798` (gray).
- Evaluator source SHA256: `37b5bac0ad36830dbd4e1f4e4d0c3694670d168cff2991025c81d511e766a4b9`.
- Strict same-shape encoder/context and encoder/permutation parity differences were exactly zero. Parameters and BatchNorm statistics had identical before/after hashes.
- Identity-position, affine-position and deterministic-innovation controls all produced zero robust disagreements. Raw position variance is action independent under this transition law; nonlinear latent variance need not be.
- Direct expected costs matched mean-cost-plus-variance to `1.33e−15` maximum absolute residual in black and `1.67e−16` in gray.
- An independent reviewer recomputed 109 numerical checks per condition. Cost differences were at most `8.88e−16`, physical-regret differences `1.36e−12`, and bootstrap endpoint differences `5.68e−13`; decision sets and indicators matched exactly.
- Full fresh collections, sampled/admitted context IDs, branch geometry, goal grid and bootstrap multiplicities matched across the conditions. The same physical contexts are paired; the encoders and backgrounds differ, so this is not an objective-only model comparison.
- Measured complete runtimes were 35.2 seconds per condition. The runtime-only smoke is excluded from these results. GPU 1 was released after both production runs completed.

The oracle conditions on physical state and episode texture. In the black world raw position/action history recovers wind; in the gray world hidden texture pixels can remain unavailable from an eight-frame history. These quantities therefore must not be labeled the fitted predictor's exact conditional mean given its latent history.

## Suggested paper and HTML use

One small plot of robust disagreement percentages with collection-stream intervals is interpretable. Show the two encoders separately, identify one model fit per condition, and state the small latent regrets and absence of established physical improvement in the caption. Avoid portraying the result as a large practical failure, a head-to-head victory for flow models, or evidence that all deterministic world models must fail.

Suggested sentence:

> With the learned encoder fixed, we enumerate the exact next-state distribution and replace it by its exact latent mean. Omitting the variance term changes the minimum expected latent-goal-cost action in 2.65% of safe black-background cases and 1.08% of matched static-noise cases. The regrets are small and do not establish improved physical control; the comparison isolates the mean-only cost approximation from predictor fitting error.

For this squared-goal objective, the mean plus a scalar covariance trace is sufficient; full generative sampling is not necessary. This experiment supports the analytical variance decomposition, while the separate user-specified `X'=X²+ξ`, fair `ξ∈{−1,+1}`, example illustrates recursive-mean error. Neither should be attributed as the proven cause of the earlier learned-rollout failures.

## Artifacts

- `studies/revision/oracle_variance.py` and `test_oracle_variance.py`: evaluator and 15 passing CPU controls.
- `research/oracle_variance_design.md`: predeclared protocol and executable commands.
- `runs/revision/oracle_variance/{off_wind,gray_episode_wind}/`: manifests, results, full collections, selected histories, encoded branch/goal features, costs, action sets, bootstrap draws and logs.
- `runs/revision/oracle_variance/summary.json`: compact plot/report payload and fixed illustrative cases.
