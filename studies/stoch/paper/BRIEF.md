# Paper brief: "Learning to Ignore What Matters: JEPAs in Stochastic Worlds" (ICLR 2027 submission)

Deadline: ICLR 2027 full paper, 25 Sep 2026 AoE. Double blind: no author names, no affiliations, no company
names (never "Yandex"), no identifying repository URLs ("code: anonymous repository, released on acceptance").
Format: ICLR conference style (try to fetch `iclr2026_conference.sty` + `.bst` from
https://raw.githubusercontent.com/ICLR/iclr2026/main/... or the 2025 version; if the network fetch fails, use
`article` with `\usepackage{times,geometry,natbib}` mimicking ICLR's 10pt / 5.5in text width). Build with
`tectonic main.tex` from `studies/stoch/paper/`. Target length 9 pages main text + references + appendix.

## Thesis (organising principle: two independent mechanisms)

JEPA-style world models (LeWorldModel, arXiv 2603.19312: joint-embedding encoder + deterministic latent
predictor trained on next-embedding MSE + SIGReg, no decoder, planning by CEM on a goal-latent cost) fail as
world models for control in stochastic worlds through two independent mechanisms:

1. **Representation.** The joint predictability objective is minimised by encoders that discard
   control-relevant state while staying non-collapsed. Two witnesses: (a) *texture trap*: with a per-episode
   random background, encoding only the (perfectly predictable) background reaches near-zero prediction loss
   (0.0004) and satisfies SIGReg, and the class-token models actually converge there (latent = texture, cube
   position unreadable); (b) *noise basin*: with per-frame noise the predictor cannot beat the noise floor and
   under hidden wind 0/3 seeds of the grid model escape the basin (prediction loss stuck at .73), while 3/3
   escape without wind. Not collapse: latent spread stays ~1.0 of the true through 16 steps (established in
   the prior study; numbers below).
2. **Planning.** A deterministic MSE predictor is the conditional mean. Latent goal-cost planning with it
   drops the variance term: E||z'-g||^2 = ||m-g||^2 + tr Cov(z'|h,a). It is exactly valid only when the trace
   variance is action-independent (the hidden-wind cube world: additive, action-independent noise, affine
   dynamics) - so that world can never show it. With action- or state-dependent multimodal outcomes the mean
   is not a state of the world ("a half-open wall on both sides") and certainty-equivalent planning ranks
   actions wrongly; recursion of means is not the multi-step expectation. One-step witness (theory figure):
   action A lands the agent in one of two safe pads with probability 1/2 each, its mean lands on the wall;
   action B lands on the wall in both outcomes, its mean lands in a pad. The MSE-optimal predictor prefers
   B: death with probability 1 versus 0, the maximal regret (with a terminal -1 the regret is bounded by 1,
   say "maximal", not "arbitrarily large"). A second, policy-class point from the review: even an exact
   stochastic model with open-loop action sequences cannot value waiting for information; a closed-loop
   policy trained inside a sampled model can. We therefore include PPO inside the JEPA (same policy class,
   mean predictor) as the control cell that separates the model from the policy class.

## Worlds (all 63x63 RGB, red 6x6 cube, four 3-px moves, 128-step cap, death = -1 on touching the border ring)

Data: uniform-random behaviour policy, 500k transitions (100 envs x 5000 steps) per world for training,
25k for evaluation. Six cube worlds = {background: black / gray noise per frame / gray noise per episode} x
{still / hidden wind (a +-1 px random walk added to every move, never observed)}. Names:
`off_still, off_wind, gray_frame_still, gray_frame_wind, gray_episode_still, gray_episode_wind`.
**Gate world** (`gate`, black background, still): a white wall 3 px wide, full height, enters at the right
edge every 22 steps and scrolls left 3 px/step; it carries a 16-px gap at the top or the bottom (p = 1/2)
which is *drawn only once the wall reaches x <= 40* (before that the wall is solid). Touching the wall is
death. A cube at the centre (x = 28) sees the gap when the wall is at x = 39 and is hit two steps later at
x = 33; reaching either gap needs five vertical moves. A cube on the left third (x <= 16) sees the gap six
or more steps before contact and can always make it. There is no stationary action: holding a position means
oscillating. Random play dies at the first wall almost always (mean episode 30.6 steps, no timeouts).
Allegory: perceptual noise (per-frame), nuisance appearance (per-episode), control noise (wind), a safety
margin (the border), and a multimodal event (the gap) that the agent must wait to observe.

## Models (each trained within a 10-minute A100 budget, 500k transitions, seed 0)

- `lewm6` = LeWM as published, shrunk: ViT (192 wide, 6 layers), class-token read-out (192-d latent), the
  paper's AdaLN autoregressive predictor (6 blocks, 16 heads x 64, MLP 2048), history 3, dropout 0.1, objective
  next-embedding MSE + 0.09 SIGReg, no stop-gradient / EMA / decoder. 15.3M parameters.
- `tile_factored` = "grid JEPA": the tile transformer + strided convolution trunk of our flow model's tokenizer
  (2.6M) producing a 4x4x32 spatial latent, plus a factored predictor (attention over the frame's tokens then
  causal attention over frames, 8.3M), history 8, same JEPA objective. Our best JEPA variant.
- `flow` = the tinywmcraft recipe: tokenizer (reconstruction-trained) + flow-matching dynamics transformer over
  the 4x4x32 latent with a 32-frame window, diffusion forcing, 4 sampling steps at inference, plus symlog
  two-hot reward head and termination head trained jointly (weight 0.1). We fixed the recipe's reward head to
  the transition form (s_t, a_{t+1}) -> r_{t+1} so the terminal -1 is learned (the original masked it out).
Off-budget read-outs fitted afterwards on frozen JEPA latents (stop-gradient, 80k frames, 200 held-out rows):
a pixel decoder (analysis only) and reward / termination / behaviour-value heads (2-layer MLPs on latent +
one-hot action) - these are what a planner needs on top of a reward-free JEPA.

## Controllers (all graded in the real environment, 128 or 256 paired episodes: same starts, same noise keys)

- `goal` (LeWM's own): squared latent distance to a goal image = the cube at the centre with no wall (its
  own texture), replanned every step; exhaustive search over the 4^h sequences (h = 5 gate, 3 cube worlds),
  at least as strong as the paper's CEM (300 samples x 10 iterations, top-30 elites).
- `plan` (heads-MPC): exhaustive horizon-h search of the imagined discounted return through the reward /
  done / value heads on the same latent rollouts (gamma .99).
- `ppo_jepa`: PPO (Impala/MLP + GRU actor-critic, HL-Gauss critic) trained entirely inside the JEPA's
  deterministic latent rollouts with the heads as reward and termination; the policy sees the latent.
- `ppo_dream pixels|latent`: the same PPO inside the flow model's sampled rollouts, reward and termination
  from its heads; the policy sees decoded frames or the 4x4x32 latent. 150 updates x 512 x 16 imagined steps
  (12-minute cap), starts half from reset frames, half from dataset states with real 8-frame context.
- `ppo_real`: PPO in the real environment (1.5M steps, 12-minute cap) - the reference.
- `random`, `centre` (true-state controller stepping against the centre offset; in the gate it dies at the
  first wall: the goal reached perfectly).
Metrics: survival to 128 steps (%), mean episode length, walls passed (gate), deaths per 100 steps.

## Results tables (generated by studies/stoch/summarize.py into studies/stoch/paper/tables/*.tex; \input them)

- `tables/gate.tex`   the gate world: every controller x metrics.
- `tables/cube.tex`   six cube worlds x controllers: survival %.
- `tables/models.tex` per world: JEPA final prediction loss and probe read-outs; flow gFVD / AF / done AUC.
- `tables/craftax.tex` Craftax-Classic grid-JEPA: goal reaching (offline latent cost, closed-loop position
  error vs random / oracle) and reward-MPC return vs behaviour policy.
Until the runs finish the files contain placeholder rows; write the prose so numbers can be dropped in and
guard every \input and \includegraphics with \IfFileExists.

## Prior-study numbers you may quote (from runs/redcube and blog/redcube, established)

- Per-episode colour/gray noise: class-token JEPAs reach prediction loss ~0.0004 with the latent encoding only
  the texture (decoding the true latent cannot place the cube); the grid JEPA keeps the cube but its predictor
  accumulates error. Per-frame noise + wind: tile_factored 0/3 seeds escape the noise basin (loss .73), 3/3
  without wind. Black background: JEPA planning works (goal reaching .85 vs random .25 in Craftax for the grid
  latent, .375 for the class token). SIGReg gives Gaussian marginals but effective rank ~51/192.
  Read `blog/PLAN-2026-09-19.md`, `blog/research-note.ru.html` (Russian), `runs/redcube/*/bench.json`,
  `runs/redcube/*/oracle.json` for exact values; cite them as "our preliminary study".

## Figures (I will produce PNG/PDF files under studies/stoch/paper/figures/; reference these names)

fig_worlds (frames of the worlds), fig_gate (a gate episode strip: solid wall, reveal, the mean prediction of
each model vs the two real futures), fig_theory (the one-step witness), fig_gate_results (bar chart of survival
by controller), fig_cube_results (survival across the six worlds), fig_mechanisms (schematic of the two
mechanisms). Write captions now; the files come later.

## Related work to cover (cite properly with bibtex entries; arXiv ids where known)

LeWorldModel (2603.19312), JEPA / I-JEPA / V-JEPA / V-JEPA 2, DINO-WM, PLDM, SIGReg / LeJEPA, Dreamer v1-v4,
Diffusion forcing, flow matching, world models for stochastic environments (stochastic video prediction,
SV2P, Genie), certainty-equivalence control, distributional RL, model-based RL with learned models
(PlaNet, MuZero), information-gathering / value of information, JEPA critiques (collapse, predictability
objectives), "predictable features" literature (bisimulation, information bottleneck).

## Tone

Sober, precise, no hype. Every claim tied to a measurement or a proposition. Report cells where JEPA is fine
(black background, wind-only) as prominently as the failures. Limitations: one seed for most cells,
10-minute budgets, toy worlds, LeWM shrunk to our budget, no reproduction of a published LeWM number.
