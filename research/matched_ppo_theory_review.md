# Independent review of matched PPO intervention

Reviewed `studies/revision/matched_ppo.py` on 2026-09-19 against `studies/stoch/rl.py`, `studies/stoch/env.py`, `studies/redcube/env.py`, `probe.py`, and `train.py`. Review is read-only; the implementation remains owned by experiment_design.

## Verdict

No blocking action-alignment, hidden-state leakage, start-distribution, PPO-update or evaluation-pairing defect was found in the current implementation. The design is suitable for the **effect of replacing learned transitions by exact absorbing simulator transitions under fixed encoders, heads, starts and PPO protocol**. It does not isolate preterminal prediction accuracy from the terminal-state extension.

Full-run interpretation should be conditional on complete fixed update budgets and the verification listed below. It must retain the postterminal-support caveat and diagnostics. These diagnostics are scientifically useful even if the oracle arms do not solve the task.

## Verified semantics

- Reset starts repeat the dataset reset frame eight times and zero incoming actions; uniform starts use the same valid eight-frame histories as the original collector. The final simulator state is the state at `t` for a reset or `t+7` for a uniform history.
- Within-episode time is reconstructed from actual episode-boundary/done metadata, with shape and consistency assertions. Current latent generation uses exactly the recorded final frame; future frames use the simulator.
- The hidden gap belongs only to simulator state and affects policy observations through the original renderer. Before reveal the renderer is independent of the gap. Actor, critic and fitted heads receive no simulator state, coordinates, gap labels or privileged action choice.
- Future simulator randomness is fresh and separate from the categorical action key. The first split of each rollout-step key matches the original JEPA collector's action key.
- The actor burn-in and incoming-action history match the original collector. Fitted reward and done predictions are read from current latent and outgoing action, exactly as in the JEPA arm.
- Both oracle arms have the same absorbing exact transition map. At a first true terminal step the true arm includes the actual terminal reward and masks later steps; it does not drop that reward. Its state/frame freeze after termination and later true rewards are zero.
- GAE bootstraps the final observation for surviving horizon-truncated trajectories. True environment cap events use the environment's terminal convention. Predicted done probabilities enter GAE and effective weights continuously, as in the original dream collector.
- Policy initialization, optimizer, burn-in, minibatch order, truncation chunk, update count, start RNG and scheduled rollout count are shared. There is no best-checkpoint selection. The explicit `budget_complete` field identifies time-capped runs that must not be pooled as equal-budget results.
- Evaluation uses a fresh fixed key independent of training and the shared original real-environment harness. Per-episode arrays permit paired differences on the same evaluation randomness. Pairing reduces evaluation variance but does not make episodes independent model replications.
- Actual effective sample weights differ between arms because continuation predictions differ. Report equal scheduled rollout positions / PPO updates, not equal effective alive transitions or equal real-environment data.

## Terminal-support limitation

The ordinary dataset replaces a terminal successor with a reset observation. The oracle collector instead exposes the physical terminal image and freezes it, because learned done can remain below one. Such terminal images and their repeated self-transitions may lie outside the fitted head's training support. Fitted heads can consequently assign rewards and continuation to a physically ended episode.

This choice is explicit and mathematically defines a valid hybrid diagnostic. However, learned versus oracle-head comparisons include the absorbing terminal extension, and oracle-head versus oracle-true differences include learned continuation effects after actual termination. They are not a clean additive decomposition into one-step dynamics error plus head error.

Required reporting: fraction of rollouts that truly terminate, effective alive-weight mass after true termination, and accumulated head reward after true termination. The implementation agent is adding these fields. If mass is substantial, foreground it in the interpretation. Forcing true done in only one arm would remove this artifact while introducing a different intervention, so do not silently change semantics after seeing results.

## Independent verification performed

Host-side start construction was executed independently through extracted functions with a NumPy shim for array constructors. Synthetic arrays contained 600 time rows, 12 environments, and varying episode lengths. Across 50 seeds, reset-mixture probabilities 0, 0.5 and 1, and batches of 257, all 38,550 sampled starts matched the original collector exactly in frames, incoming actions and reset flags. Reconstructed state indices, positions, wall positions, hidden gaps and elapsed episode times matched their source rows. All uniform contexts stayed inside a single episode; all reset states had elapsed time zero.

This test verifies host indexing and RNG consumption, not JAX/model execution.

## Requested GPU smoke evidence before full-run interpretation

1. Replay recorded actions from reconstructed states for deterministic non-spawn transitions and verify successor pixels/state, reward and termination. Current-frame re-rendering alone cannot expose a one-step time/action-index error. For a spawn transition, use the recorded next gap only inside the validation fixture or compare state components whose evolution does not depend on the new random draw.
2. On the same checkpoint, policy initialization, batch and key, compare the learned collector numerically with the original collector and compare one shared PPO update with its original counterpart. Small floating-point tolerances are acceptable if explained; semantic parity should be exact.
3. Confirm end-to-end smoke outputs contain checkpoint/data/source hashes, no nonfinite values, fixed evaluation identity, start indices and explicit complete/incomplete update status.

There is no request to alter rewards, scale latents, change actor observations or select successful checkpoints.
