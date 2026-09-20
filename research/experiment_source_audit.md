# Source findings relevant to the revision

These findings concern the current source snapshot. They do not retroactively establish the exact launch provenance of every historical artifact.

## Historical real-versus-imagined PPO is a pipeline contrast

`studies/stoch/rl.py:ppo_real` uses persistent real episodes, 128 environments, 64-step collection and grouped 16-step chunks. `ppo_dream` restarts 512 trajectories from a reset/data-context mixture every 16 steps. Their nominal budgets also differ (1.5M versus 1.2288M). The new `studies/revision/matched_ppo.py` fixes these protocol dimensions and changes transition/signal sources only, subject to its explicit absorbing-state convention.

Historical intermediate evaluation advances the same JAX key used for training. Removing those evaluations therefore changes subsequent training randomness even when seed numbers match. The new campaigns compare contemporaneously rerun arms, not numerically reproduced historical trajectories, and use independent fixed evaluation keys.

## Imagined-head audit pairs the wrong state and action

`studies/stoch/heads_audit.py:87` obtains successor predictions from `_rollout_latents`; its first returned latent is the successor of the final context frame. Lines 93–97 pair these latents with actions beginning at `CTX - 1`, the actions that generated them. The heads were trained on current-state/outgoing-action pairs (`probe.py:train_heads`). The same offset appears in `planner_view` at lines 129–131. Consequently those historical diagnostic numbers do not equal the actual planner's scored returns.

The actual controller is correctly indexed: `studies/stoch/control.py:73–76` reads heads on `parent, taken`, before the child latent. Both original in-model PPO and the new matched collector likewise read heads on the current latent and selected outgoing action. This is a diagnostic defect, not evidence that PPO or the controller used the same defective indexing.

The real-latent head audit (`heads_audit.py:real_latents`) pairs `Z[:T]` with `actions[:T]` correctly. Its observed wall-versus-border reward/termination statistics remain usable with their own data/checkpoint scope. Avoid using the imagined/planner portion without recomputation and a timing-correct bootstrap.

## Head-training data description requires provenance correction

The manuscript says posthoc heads used 80,000 frames from a different behavior-policy seed. Default launch scripts do not support that statement: `studies/stoch/train_arms.sh:46–47` and `lane_s.sh:19` pass the same `$DATA` variable used to train the world model. `probe.py` fits on the first `--frames` frames and evaluates its heads on the *subsequent* `--held-out` time rows; these are not the last 200 rows of the fitting frames. Actual historical command/log evidence should settle provenance before rewriting that paragraph. Do not infer an independent head-training dataset from an independent final evaluation archive.

## Wall-count endpoint uses pretransition activity

`studies/stoch/rl.py:make_eval` counts `active & old_wall_present & new_wall_absent`, where `active` is the mask before the current transition. It does not require the current transition's `done` flag to be false. A wall that leaves the frame on the same transition as an agent death can therefore be counted. This is shared across all arms and remains frozen for the primary matched campaign, but the manuscript should not define it more strictly as passing while still alive after the transition. A stricter counter, if examined, is a separately reported sensitivity.

