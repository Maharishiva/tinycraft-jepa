# Proposed nine-page main text

This is the target layout after E1-E10 arrive. Page allocations include figures and tables; references and appendices are excluded. The edited scratch manuscript is not yet that final layout.

The title is **The Half-Open Wall: When Mean Predictions Fail**. CODEX_CRITIQUE.md explains why the stronger proposed title exceeds the theorem and evidence.

| Order | Section | Pages | Text target | Content and evidence |
|---|---|---:|---:|---|
| 0 | Abstract | 0.25 | 170 words | Question, two matched interventions, measured effects, scoped theory. |
| 1 | Introduction | 0.75 | 430 words | Wall example, squared-loss scope, same-model intervention, precise contribution. |
| 2 | The gate and its decision | 0.60 | 250 words | Two versus five steps; retreat, reveal, response; hidden-gap information contract. |
| 3 | What the objective and predictor guarantee | 2.60 | 1,200 words plus equations | Texture optimum and short proof; innovation bound; exact-mean lemma; regret witness and proof; positive boundary; amplitude proposition and proof. |
| 4 | Models, interventions, and evaluation | 0.55 | 200 words | Table 1, within-model controls, seed hierarchy, finite-K caveat, evaluation contract. |
| 5 | Results | 2.85 | 1,050 words | Hero figure; paired effects; E4/E9 controls; E6 geometry; E5 budgets; E7/E10 representation; E8 caveat; one Craftax paragraph. |
| 6 | Related work | 0.55 | 280 words | Slow-feature priority, self-predictive RL, exact delta, classical certainty equivalence. |
| 7 | Discussion and limitations | 0.85 | 420 words | What each remedy addresses, scope, seeds, finite means, off-support scoring, budget, task limitations. |
| | **Total** | **9.00** | **About 4,000 words plus equations** | |

Float Figure 1 onto pages 1-2 by placing its source near the introduction. Charge its space to the results allocation above, not twice. Do not compress proofs with negative spacing to hit the page budget. Trim architectural catalogues and repeated caveats first.

## Results order (superseded by NEW_RESULTS.md)

1. E1 original / mean-of-eight / sample: 0.00 / 0.91 / 0.66 walls, with 0.0 / 2.7 / 2.0 percent survival. Accuracy moved the number at one model and PPO seed; sampling did not.
2. E9 labelled border-versus-wall audit, scoped to the tested heads. Explain +0.29 as the maximum over candidate plans and keep the paired selection audit pending.
3. E4 simulator reference: 0.69 versus 1.03 walls at h=5 versus h=8, negligible gap-leak effect at h=5; different terminal score, no ceiling.
4. Explicit pending cells and the questions they decide: E2 flow pair, E3 seeds, E5 fitting budget, E6 geometry, E7 regularization, E8 baseline scores, E10 feedback requirement.
5. One exploratory Craftax paragraph.

The current source implements this order. The nine-page layout and hero below remain a future production target, not completed experimental evidence. The current draft uses one compact main-text E1 table and retains historical generated tables in the appendix.

## Hero Figure 1

Use one full-width figure with three panels. Keep labels readable at final column width. References are gray; class-token LeWM is orange; grid JEPA is red; its residual head is purple; flow is blue; real-environment PPO is black.

**Panel A: the decision.** Show one shared pre-reveal frame and both counterfactual real futures under identical actions. Add a small timing diagram for centre contact, retreat, reveal, and gap reachability. Label the gap as unobserved before reveal. Do not reveal simulator state to the policy.

**Panel B: what one context becomes.** Align frames by rollout time. Show a flow top sample, flow bottom sample, flow mean-of-K, JEPA residual-head sample, JEPA mean-of-eight, and original deterministic JEPA. This adds the matched residual mean missing from the plan's initial five-row sketch. Include a narrow reference row decoding the encoded real future. Label decoded latent mean separately from mean decoded pixels; they are different quantities. Use soft intensities rather than a white-pixel threshold. Choose the context before comparing method outcomes and publish the selection rule.

**Panel C: real control.** Show grouped point estimates for original JEPA, residual mean, residual samples, flow mean, and flow samples. The vertical axis is walls passed; survival appears directly beneath each group. Show each model-seed mean and smaller policy-seed points. State whether intervals or standard deviations are shown. Include random and real-environment references. Put h=5 and h=8 oracle-search results as dashed references labeled “simulator MPC, no terminal value.” They are not upper bounds.

**One-sentence caption:** “From the same hidden-gap context, sampled and averaged rollouts yield different imagined futures; paired policy evaluations test whether those differences change real survival.”

After supportive results arrive, replace “test whether” with “show how,” retaining the measured scope. Do not pre-label averages as half-open walls unless Panel B actually shows that geometry.

## Prospective main-text table list (not the current implementation)

Keep exactly two tables in the final main text. Neither is an existing generated round-1 table pasted unchanged.

**Table 1: Models and matched interventions.** Rows: class-token JEPA; grid JEPA original; grid JEPA residual predictor in averaged and sampled modes; flow in averaged and sampled modes. Columns: encoder training signal; latent and history; transition mode; reward/termination/value source; quantities held fixed; model/head/policy compute separately. Use human-readable model names. Merge cells for shared components within each pair.

**Table 2: Paired gate control results.** Rows: original grid predictor; residual mean-of-eight; residual sample; flow mean-of-K; flow sample; simulator MPC h=5; simulator MPC h=8; random; real-environment PPO. Columns: model-seed count; policies per model; walls passed; survival; paired sample-minus-average difference where defined. The caption separates model-seed variation from episode uncertainty. State finite-K approximation status and the oracle's different terminal score.

The original class-token gate rows remain in the appendix unless replicated evidence makes them central. E5-E7 use figures, avoiding a third main table. E9 receives a compact main-text summary and a full appendix table.

All five existing generated files remain unchanged. Their tables belong in the appendix: gate.tex, cube.tex, models.tex, wallmass.tex, and craftax.tex. Table 2 requires a future generator update outside this editing task.

## Appendix order

Keep the innovation-bound and conditional-mean proofs, then Proposition 3's complete proof. Main-text proofs remain accessible where their statements appear.

Follow with hyperparameters and separate model, head, policy, and inference costs. Specify model seeds, policy seeds, evaluation episode pairing, checkpoint hashes, finite-K settings, and hidden-gap handling.

Next present full round-1 gate and cube tables, the alternating baseline, edge-start state-blind analysis, head audit, all budget and regularizer sweeps, and all counterfactual diagnostics. Label preliminary-study checkpoints distinctly.

Finish with Craftax, failure cases, and the reward-head correction. A zero-reward model has zero return regardless of termination; preserve that corrected explanation.
