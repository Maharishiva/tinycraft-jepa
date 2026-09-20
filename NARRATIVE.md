# Paper narrative anchor

Author decision, 2026-09-20. This file fixes the claim and the voice. Every draft is written to
serve it. Anything that does not serve it goes to the appendix or is cut.

## Title

**Learning to Ignore What Matters: JEPAs in Stochastic Worlds**

## Claim

A joint-embedding predictive objective can favor predictability at the expense of the information
a decision needs. Two *independent* mechanisms produce this, and they fail in different places:

1. **Representation.** The objective chooses *what* to encode. A predictable distractor can
   displace controlled state, so a non-collapsed representation with low prediction error can be
   useless for control.
2. **Planning.** The objective chooses *how* to predict. Squared error gives the conditional mean.
   Scoring a plan at the mean can misrank actions, and recursive mean prediction need not recover
   a multi-step expectation.

Punchline: avoiding collapse and predicting latents accurately are *insufficient* for stochastic
control. Distributional dynamics fix the second mechanism only when the representation has
survived the first.

## Voice

The abstract from the earlier draft sets the register: direct claims, mechanisms named, no
hedging-by-default. Hedge exactly where the evidence is thin, and say why in one clause, not in
every sentence. Do not write "this does not prove" more than the evidence requires.

## What earns a place in the main text

- The two exact analytical examples (A vs B action ranking; x' = x^2 + xi recursion). These are
  correct, short, and carry mechanism 2.
- The Gaussian nuisance construction for mechanism 1, with Sobal et al. 2022 credit.
- The noise-persistence x dynamics reversal: 0.115 / 21.33 / 0.764 / 0.668 px. This is the
  distinctive empirical result and it is a *matched* comparison.
- The gate world: an interface-qualified JEPA-versus-flow control contrast.
- Craftax under matched data and budget, as the consequence outside a toy.
- The sampling-predictor control cell: a distributional head on the *same* latent. This is what
  separates "the mean was the problem" from "the representation was the problem".
- The imagined-reward diagnostic (added 2026-09-20 after the shared-CNN and Craftax campaigns):
  in worlds whose true reward is never positive, PPO inside the JEPA drives the imagined reward per
  step to +0.06..+0.21 (cube worlds) and to +0.90 in Craftax, while inside the flow model it stays at
  or below the data rate. The JEPA's imagination is exploitable: the policy finds latent
  trajectories the heads mis-score. Report it next to every imagination result; it is the visible
  form of mechanism 2 in a learned system.

## What stays out of the main text

Audit ledgers, the 45/27/24 null policy sweeps, forecast-readout transfer matrices, termination
safeguards, PVE attempts, the historical gate tables, the 66-endpoint explorer, latent
amplification and subspace controls, nearby-Craftax goal reaching. They bound interpretation and
live in appendices and the evidence manifest.

## Honesty constraints that must survive the rewrite

These are load-bearing. Removing them would make the paper wrong, not just shorter.

- The empirical per-frame-noise + wind failure has **high prediction loss** (~0.733/coordinate).
  It is not an instance of the zero-loss static-nuisance optimum. Say so once, plainly.
- The gate JEPA encoder is **model seed 1, favorably preselected** from prior development
  evidence. Disclose it where the gate result appears.
- Flow and JEPA differ in **supervision stage**, not only in deterministic versus distributional
  prediction: flow's task heads send gradients into its dynamics trunk, JEPA is reward-free with
  post-hoc heads. Equal label counts do not equalize this. Any JEPA-versus-flow gap is a
  *recipe-level* gap.
- Decoded-real controls qualify the interface before any imagination result is interpreted.
- Goal-MPC succeeds only at short distances; the original 6-to-12-action test failed and stays in
  the record.
