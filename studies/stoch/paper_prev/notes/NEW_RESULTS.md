# Round-2 results that have arrived (2026-09-19, 10:10 UTC)

All on the gate world, grid JEPA `tile_factored` (model seed 0) unless stated. Numbers are final for the cells named;
cells marked pending are still running.

## E1: same JEPA, residual head as mean-of-8 versus one sample (PPO inside, PPO seed 0)

| predictor | survival | length | walls passed | deaths / 100 |
|---|---|---|---|---|
| original deterministic predictor | 0.0 % | 20.4 | 0.0 | 4.9 |
| residual head, one sample | 2.0 % | 50.5 | 0.66 | 1.93 |
| residual head, mean of 8 samples | 2.7 % | 54.4 | 0.91 | 1.79 |

The mean-of-8 arm matches or exceeds the sampled arm. The round-1 gain of the sampling arm over the deterministic
JEPA was the residual head's better conditional mean, not the sample. "Recovers, twice" is dead for the JEPA:
inside the JEPA, mean versus sample makes no difference. Model seeds 1, 2 and PPO seeds 1, 2 of all three modes
are pending (24 runs, ~10:50 UTC).

## E2: flow model, latent PPO with the next latent = mean of 8 samples versus one sample

Pending (the mean-of-8 arm needs ~40 min per seed; first cell ~10:50 UTC). Round 1: one sample, PPO seed 0 =
98 % survival. PPO seeds 1, 2 of the sampled arm are running now.

## E4: oracle open-loop planner on the true simulator (exhaustive 4^h search, replanned every step, -1 at death,
no terminal value, unrevealed gap resampled per copy, 128 envs x 128 steps, paired with the model controllers'
random row bit-for-bit)

| planner | survival | length | walls passed | deaths / 100 |
|---|---|---|---|---|
| random | 0.0 % | 32.2 | 0.09 | 3.11 |
| simulator, h = 5, K = 4 gap copies | 0.8 % | 54.8 | 0.69 | 1.81 |
| simulator, h = 5, K = 16 | 0.8 % | 54.8 | 0.69 | 1.81 |
| simulator, h = 5, true gap leaked | 0.8 % | 55.0 | 0.70 | 1.80 |
| simulator, h = 8, K = 4 | 0.8 % | 62.4 | 1.03 | 1.59 |

Reading: exhaustive short-horizon search on the TRUE simulator doubles the lifetime and passes 7-11x more walls
than random, but essentially never reaches the 128-step cap; leaking the gap changes nothing at h = 5; the number
of gap copies changes nothing; horizon does (h 5 -> 8: +14 % length, +50 % walls). Over 20,412 states with a wall at
x in 42..60 the hidden gap flips the zero-cost plan set in 31.4 % of states but the set of safe FIRST actions in
only 1.84 %. So the planner harness is bounded by its horizon and its terminal score, not by the gap: it is a
reference with a different score, not a ceiling (Codex's own caveat, confirmed). The JEPA heads-MPC planners of
round 1 (0.04-0.16 walls at h = 3 and h = 5) sit far below this reference; PPO inside the flow model (98 %)
sits far above it because it carries a learned value.

## E9: heads audit on the grid JEPA (heads trained off-budget on 80k real latents; audit on the 25k-transition
evaluation set: 769 deaths, 610 at the wall, 159 at the border)

Real latents:
- reward head: corr with -1{death} 0.346; mean prediction on wall deaths -0.054, on border deaths -0.891, alive -0.027
- termination head: AUC all 0.714; wall-death vs alive 0.649; border-death vs alive 0.963; mean p(done) on wall
  deaths 0.057, border deaths 0.673, alive 0.028
- value head: mean -0.856, never positive (0 % > 0.01)

So the supervised heads, given the death labels, read border deaths off the latent almost perfectly and wall deaths
barely at all: the latent (plus the action) does not carry where the wall is relative to the cube. This is the
representation deficit measured with labels rather than with a planner.

Imagined rollouts from 512 real 8-frame contexts, the dataset's own actions, horizon 8, gamma 0.99:

| rollout | mean imagined return | fraction > 0 | reward part | value part |
|---|---|---|---|---|
| deterministic predictor | -0.78 | 1.4 % | -0.28 | -0.50 |
| one sample | -0.86 | 0.2 % | -0.27 | -0.59 |
| mean of 8 samples | -0.86 | 0.0 % | -0.27 | -0.60 |

The imagined return of a typical rollout is never positive. The +0.29 that the round-1 heads-MPC planner reported
is the MAX over the 4^h candidate plans of a noisy score (control.py records -min(cost) per step): selection of the
most optimistic of 64-1024 head read-outs on latents the heads never saw. A direct measurement of mean-over-plans
versus max-over-plans on the same contexts is queued. The deterministic rollout's p(done) climbs from 0.03 to 0.11
over 8 steps while the sampled rollouts stay at 0.03-0.05: the deterministic latent drifts to where the heads
fire; the sampled one does not.

## Codex's mean-of-K caveat is now an observation

Mean of 8 is not the exact conditional mean; it is a resample with covariance Sigma/8. E1 shows it behaves like the
sample and unlike the original predictor, so what separates the residual head from the original predictor is the
accuracy of its centre, not its spread.

## Timing facts for the budget table

PPO inside the JEPA on the gate: ~4 min per run (mean-of-8 ~2 min longer); PPO inside the flow model's latent: ~5
min at NFE 4, mean-of-8 ~40 min; grid JEPA training 8500 steps = 575 s; probe ~3 min; residual sampler ~75 s;
oracle planner h = 8: 15 s.
