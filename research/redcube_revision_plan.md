# Revised empirical story and resource allocation

Decision time: 2026-09-19 23:10 UTC. Deadline: 2026-09-20 07:41:45 UTC.

This replaces the audit-centered main-paper plan after explicit user feedback. It does not erase completed or stopped experiments.

## Main story

1. **JEPA can support goal-directed control.** Existing deterministic gray-frame-noise model, fresh visual goals, fixed-horizon MPC, actual trajectories. Still-world survival alone is insufficient because blind oscillation solves it.
2. **Prediction training can fail to expose useful state.** Gray-frame noise with stochastic wind versus the matched deterministic and persistent-noise conditions. The empirical failure has high prediction loss and is not an observed globally optimized nuisance-only minimum. Show reconstructions of real observations separately from forecasts.
3. **Correct state means do not suffice for stochastic planning.** Use the exact discrete action-ranking and recursive-mean counterexamples. Evaluate the trained frozen-encoder variance identity separately. A learned-model comparison can establish a recipe-level performance gap; it does not automatically establish that omitted variance caused the gap.

The hoped-for third learned-model behavioral case remains unconfirmed. Do not relabel off-wind as a failure: existing JEPA policies perform well there. Do not call a disappearing common gate wall the exact average of the two gap futures. Include unfavorable flow results in the complete evidence table.

## Fixed evaluations

- **Goal positive control:** 128 paired valid eight-frame histories; gray-frame-still; goals 6–12 native actions away on the reachable lattice; H3 receding MPC with summed latent squared-distance stage costs, 24 real steps. H1, random, blind cycle, and privileged simulator references are separately labeled. Exact hit primary; minimum/final L1 distance and within-three-pixel hit secondary. Trace episodes 0–2 regardless of outcome.
- **Visual evidence:** first three eligible distinct-episode windows per condition, selected before model inference. GT, decoded real JEPA observations, JEPA forecast, decoded real flow observations, and one flow sample. No best-of sampling or replacing real-policy execution with recorded-action forecasts.
- **Oracle variance:** exact nine wind branches per four actions, fixed goals, frozen encoder. Identity-position negative control. Primary inference is expected latent-goal cost, not survival or physical goal improvement.
- **Shared-CNN red-cube comparison:** four worlds: gray-frame-still, gray-frame-wind, gray-episode-wind, gate. Two existing model recipes, three paired policy seeds: 24 dream policies. Six gate decoded-real policies qualify the control interface. All use the stock pixel CNN+GRU, own posterior decoded observations in real evaluation, valid histories, native rewards, matched optimization budget, and fixed final endpoints. Admit the full set only after smoke runtime review; retain all admitted outcomes.
- **Meaningful Craftax comparison:** clean from-scratch five-million-imagined-step PPO recipe, JEPA versus flow, three paired policy seeds, same pixel policy and native final evaluation. Wooden-pickaxe acquisition primary; all achievements, return, sword and stone-tool rates reported. Exclude paid-action reward masking and BC initialization from this primary comparison. Exact source/protocol and smoke must pass before launching.

## Resource assignment

- MG GPU0: finish all 12 admitted target-gradient model fits/readouts, then shared-CNN red-cube queue.
- MG GPU1: H48 active job finishes; remaining unstarted jobs stop for user-requested scope change. Short visual, goal and oracle evaluations, then shared-CNN red-cube queue.
- E2E GPU0: reserve for the meaningful Craftax adapter, smoke and paired five-million-step runs. Original nearby-goal replication already finished and is archived. No last-only Craftax runs.

No additional latent amplification, reward modifications, best-policy endpoint selection, or environment-specific reward suppression. New objective variants are deferred. Native evaluation, source/checkpoint/data identity, paired seeds, valid histories and visual inspection remain required. Reports distinguish model seeds from policy seeds and recipes from isolated objective comparisons.
