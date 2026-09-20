# Independent empirical review of v1

**Overall: 5/10, marginal reject. Confidence: 4/5 for the empirical assessment.**

The paper reports its existing evidence carefully, and I found no discrepancy in the principal numerical results. Its most convincing new empirical result is the replicated gray-noise/dynamics interaction. My reservation is scientific rather than editorial: the study stops before an intervention that distinguishes a limitation of the jointly learned squared-loss representation/predictor from weaknesses of the fitted heads, policy interface, or optimization protocol. The manuscript now explicitly admits these alternatives, which improves its correctness but does not supply the missing experiment. I would not raise the score for additional polish or for experiments that are merely planned.

Reviewed on 2026-09-19 during concurrent copyediting. I read the main TeX and sections, raw result files, and relevant evaluator/probe source. I did not run GPUs or retrain/evaluate checkpoints. The source hashes at the end of the review pass are saved in `v1_empirical_snapshot.json`. This is an empirical review; I did not conduct an exhaustive novelty search or independently certify every theorem.

## Summary and strengths

The paper separates three properties that are too easily conflated: state recoverable from real encodings, useful state accessible in predicted encodings, and successful real-environment control after training with a world model. A Craftax example motivates the distinction; a delayed-reveal wall task supplies decoder, head, and policy diagnostics; a cube factorial study varies nuisance persistence and stochastic motion. Two population constructions show that Gaussian non-collapse alone does not guarantee state retention or an appropriate geometric scale.

1. **The evidence is substantially more credible than a collection of rollout pictures.** There are three independently fitted gate encoders, explicit policy-seed replication, held-out head diagnostics, real-pixel and frozen-latent policy controls, and final-checkpoint reporting. The paper correctly separates nine policies from nine independent world models.
2. **The matched gray-noise cube comparison is strong within its stated scope.** Architecture, latent dimension, dataset size, and updates are held fixed across a four-cell comparison with three model initializations. Per-frame noise plus wind yields a large and reproducible readout failure while the other three cells retain subpixel position readout.
3. **Useful information in at least one gate encoder is supported behaviorally.** Two successful decoded-latent CNN policies, both on encoder seed 1, make a stronger case than reconstruction alone. The paper appropriately discloses the decoder's additional reconstruction supervision.
4. **Limitations are stated where the results appear.** The flow reference is identified as one fitted model with different architecture, supervision, and context. Craftax is identified as a nearby-goal protocol and a separate reward-control protocol. The paper does not mistake the high-loss cube cell for an optimized low-loss nuisance solution, or disappearance under a real-latent decoder for proof that a predicted latent contains no wall information.

## Main concerns

**1. The central gate control comparison has not validated a common successful controller or isolated the changed component.** Real-latent MLP PPO reaches 0.935 walls per episode but has zero discoveries in nine runs; true-pixel MLP PPO also has zero discoveries. The successful comparison uses a supervised decoder and a CNN instead. Meanwhile real versus imagined training changes transition source, reward and termination source, start distribution, collection organization, and total transitions (1,499,136 versus 1,228,800). The manuscript is honest about this (`sections/controllers.tex`, comparison paragraph; `sections/results.tex`, first two subsections), but the resulting 0/27 imagined failures still cannot identify a JEPA-specific or dynamics-specific limitation. The 27 policies are also three predictor treatments on three encoders, not 27 independent replications of a model-family claim. The evidence supports a failing recipe, with a decomposition of plausible failure points; that is weaker than an identified failure mechanism.

**2. Fitted heads provide a directly measured sufficient competing concern, and residual sampling does not resolve it.** On independently held-out real gate transitions, encoder 0 predicts wall-death reward approximately -0.054 instead of -1, with termination AUC 0.649; border-death reward is -0.891 and AUC 0.963. These are not merely hypothetical confounds. The real-encoding decoder can preserve the wall while the learned reward signal largely ignores its collision consequence. Sampling a residual model while retaining these heads and the same policy protocol is therefore a narrow intervention. Its negative result does not establish that stochastic prediction is unhelpful, or that low geometric scale is the practical bottleneck. A diagnostic paper should take the next step and measure how much of the control gap these heads explain.

**3. The strongest empirical failure and the objective constructions are not connected by a demonstrated mechanism.** The replicated failing gray-frame/wind cell has prediction loss 0.733–0.735 per coordinate. The persistent-gray cells, which are closer to the nuisance-optimum story, retain position with less than one pixel error. Thus the strongest replicated finding is a difficult optimization/representation-accessibility interaction, not evidence that training found the bad low-loss Gaussian optimum. The exact-Gaussian small-scale construction uses infinite-precision measurable coding and a fresh, intrinsically unpredictable bit; as the authors state, it does not explain loss of a predictable wall. These are useful logical counterexamples, but without a matched objective intervention the empirical and theoretical parts remain adjacent rather than mutually explanatory. The practical conclusion that probes, prediction, and control need separate evaluation is sensible but, by itself, a modest contribution for ICLR.

**4. Both the prediction and task-level demonstrations remain narrow.** The gate wall diagnostic selects 64 contexts whose recorded futures survive the horizon and uses a decoder fitted on real encodings. Disappearance is a valid forecast/readout failure, but whether task information remains in the forecast is untested. Craftax uses one checkpoint, a six-action reachable-goal distribution, position-only success, and shared goal-generation/execution randomness. A no-op baseline and displacement strata are absent. Its reward head already generalizes poorly, and even myopic reward selection is below random. These results are useful motivation, but their contrast is not a transfer/generalization experiment with a common task, nor a causal test of imagination. They cannot carry the external-validity burden for the toy studies.

## Numerical reproduction

I loaded each of the 66 main gate runs from its raw `ledger.jsonl`, selected the final nonempty row, and verified exact equality to the manifest's saved final row. These aggregates reproduce the main table; the ledger's group labels were used only for membership, not for aggregate values.

| Gate condition | Policy runs | Walls, mean ± sample SD | Discoveries |
|---|---:|---:|---:|
| JEPA original | 9 | 0.272569 ± 0.215347 | 0 |
| JEPA residual sample | 9 | 0.261285 ± 0.322387 | 0 |
| JEPA residual mean8 | 9 | 0.262153 ± 0.297051 | 0 |
| Real frozen latent, MLP | 9 | 0.935330 ± 0.036545 | 0 |
| Real pixels, MLP | 3 | 0.912760 ± 0.080276 | 0 |
| Real decoded latent, CNN | 9 | 1.565104 ± 1.261970 | 2 |
| Real pixels, CNN | 3 | 2.910156 ± 1.650976 | 2 |
| Flow sample | 6 | 1.454427 ± 1.229119 | 1 |
| Flow mean8 | 3 | 0.882813 ± 0.138713 | 0 |

The additional six decoded-latent MLP controls reproduce 0.916667 ± 0.082557 walls and zero discoveries.

I independently read the four `runs/redcube/gray_*/allarms/bench.json` files and aggregated `probe_error` for `tile_factored`, `_s1`, and `_s2`:

| Gray condition | Real-encoding position L1 | Eight-step prediction L1 |
|---|---:|---:|
| Frame / deterministic | 0.115134 ± 0.011520 | 4.844280 ± 1.134164 |
| Frame / wind | 21.329479 ± 0.057090 | 20.387848 ± 0.924825 |
| Episode / deterministic | 0.763683 ± 0.068044 | 1.829285 ± 0.219610 |
| Episode / wind | 0.667641 ± 0.048645 | 10.951576 ± 0.534569 |

The three failing-cell logs end at 8,500/8,500 updates with prediction losses 0.7345, 0.7336, and 0.7331. The black-background class-token example is 0.086991 pixels on real encodings and 45.267525 at prediction step eight. The persistent-color deterministic flow example is 18.562727 pixels on real encodings. These also agree with the prose.

`runs/stoch/craftax/tile_factored_plan.json` directly confirms 121/128 versus 24/128 goals and mean returns 0.473438 for CEM versus 1.445313 random. `plan.py:179–204` confirms common simulator keys, one-shot plan execution, and endpoint-position-only success. These protocol restrictions are therefore substantive, not inferred from the numbers.

## Minor issues and inexpensive corrections

- **Correct the residual-sampler evaluation protocol.** The exact reported held-out MSE values (0.0003071555 original, 0.0002046269 sample, 0.0001319237 mean8) occur in seed-0 sampler metadata with `nfe=8`, identically in both `research/gpu_inventory/tinycraft-jepa-{mg,e2e}-metadata.json` snapshots. The sampled-policy config instead records `nfe=4`. The appendix currently places those quality values beside “four function evaluations”; distinguish quality evaluation at eight from policy rollouts at four. Thus the demonstrated held-out improvement is not at precisely the inference setting used for the policy failure. The same checkpoint metadata reports 275.2 seconds total; the quoted 74 seconds needs a documented loop-only label or correction. This does not change the raw policy endpoints.
- Fully specify the nonlinear cube probe: the source uses a shared token MLP with two hidden GELU layers of width 128, 10,000 Adam updates, batch 2,048, learning rate 0.001, and the first 1,001 training time rows. Class-token probes choose standardization versus PCA whitening on the last 10% of the fit data; grid probes use standardization. These choices matter to the interpretation of readout failure (`studies/redcube/bench.py:104–166,234–251`). The expanded appendix describes the split but still omits most fitting details as of this review.
- Add no-op and displacement-stratified Craftax goal results before emphasizing 94.5% as strong planning performance. Prefer independent execution randomness as an explicitly additional protocol. Preserve per-episode outcomes so uncertainty can be computed without inventing paired covariance from aggregate SDs.
- Describe discoveries as an operational threshold, rather than direct evidence of a particular learned strategy, unless successful-policy trajectories verify retreat, reveal, and gap selection. The raw continuous survival/wall outcomes should remain primary.
- Give a state-blind position baseline for the cube probe. A 21-pixel error is clearly much worse than the successful cells, but its relation to trivial prediction and position-distribution changes should be explicit.
- Keep confidence intervals, if added, at the correct level: policy seeds conditional on a model do not estimate model-family uncertainty. More evaluation episodes cannot compensate for the single flow or Craftax fit.

## Three most valuable feasible additions

1. **A matched gate intervention that can actually succeed.** Run the same PPO implementation, observation interface, starts, rollout lengths, update count, and evaluation keys with (a) simulator transitions and simulator outcomes; (b) simulator transitions and fitted reward/done heads; and (c) learned transitions and the same fitted heads. Use the frozen encoder in all three. First require the common policy interface to learn the task; if the MLP fails even in (a), use the already demonstrated decoder/CNN interface consistently rather than interpreting another failed MLP run. Retain per-episode evaluations and several independent encoder fits. This separates the cost of the learned heads from the remaining cost of model transitions under a fixed training protocol. Do not assign “oracle rewards” to arbitrary imagined latents without defining their physical semantics.
2. **Test information in forecasts directly.** Freeze the existing encoders/predictors and fit the same-capacity task-state readout separately on real latents and on predicted latents, with disjoint training/evaluation contexts and horizons. Measure predictable wall position/presence and cube position under recorded and counterfactual actions, including terminal-adjacent contexts rather than only surviving windows. If a readout trained on forecasts recovers the wall, the prominent visual failure is substantially a readout-distribution issue; if it cannot, the evidence for a prediction-information failure becomes stronger. Keep the original real-trained decoder result alongside this intervention.
3. **A matched objective/optimization control on the replicated cube interaction.** Use the same 512-dimensional architecture, gray datasets, three model seeds, and budget to compare the joint squared-loss recipe with a frozen reconstruction-trained encoder plus the same squared-loss predictor; a deterministic L1 variant is another useful small arm. Report representation readout, teacher-forced loss, and open-loop task-state error together, and extend optimization on the failing cell rather than only on the gate. A frozen informative encoder localizes representation learning versus transition fitting; the loss intervention tests whether squared loss is special. A positive or negative outcome is valuable provided budgets and selection are fixed before reading results.

With a clean successful oracle-transition control and one convincing localization of the failure, I could move to marginal accept. At present I see a careful, informative diagnostic report whose principal causal and JEPA-specific questions remain unresolved.
