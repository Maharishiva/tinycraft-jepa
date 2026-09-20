# True-termination sensitivity of the frozen matched-PPO diagnostic

This is an adaptively admitted, fully enumerated sensitivity analysis. Admission followed the original `oracle_heads` collector's logged postterminal effective weight: approximately 7.7–17.1% of weighted rollout positions in the inspected runs occurred after true termination, and at least one inspected run assigned a positive accumulated learned reward there. This admission criterion concerns the collector's terminal-support semantics, not which control arm won. The sensitivity was specified before any `oracle_reward` control outcome. It does not replace, extend the replication count of, or retrospectively change the original 45-run MLP estimand.

## Intervention and immutable lineage

The original harness is `studies/revision/matched_ppo.py`, SHA-256 `6a3460b2db2e47713b7af604454a4b04066c96bfd7748b5c991cfbc433176269`. It remains unchanged. The new standalone full-copy harness is `studies/revision/matched_ppo_termination.py`, SHA-256 `1c003dd4e019b2179981877310e131304d13a7150ac7da93e11d5164d011feb1`. Its CLI admits only `oracle_reward`. The parent hash is checked before training; each run saves both parent and new harness content hashes, all original dependency hashes, model/data hashes and original model/head metadata. No import-time monkeypatch or mutation of the frozen parent is used.

`oracle_reward` uses exactly the original absorbing simulator transition map and re-encodes each next image with the same frozen encoder. It reads the original fitted reward head from the **current encoded observation and outgoing sampled action**, then uses the simulator's **true done** in GAE and the alive mask. The learned reward is neither clipped, shifted nor zeroed. In particular, the first terminal transition retains its original predicted reward. The collector may compute learned rewards and actor states on later absorbing positions, but true done gives all such positions zero optimization weight and prevents their reward/value contributions from propagating through GAE into the live episode. The environment's 128-step cap retains the original terminal convention.

Only the done value differs from original `oracle_heads`. The full-copy collector retains true-signal branches for source comparability, but they are not admitted arms of this new executable. Hidden simulator state never reaches the actor, reward head or critic. The actor receives the original latent width and interface; there is no latent gain, privileged action choice, reward shaping, reward repair or objective tuning.

## Fixed matrix and budget

Every intended outcome is reported, including failures and incomplete jobs:

| Separate policy stratum | Model seeds | Policy seeds | New arm | Intended runs |
|---|---|---|---|---:|
| Original MLP | 0, 1, 2 | 0, 1, 2, 3, 4 | `oracle_reward` | 15 |
| Original native-grid sensitivity | 0, 1, 2 | 0, 1, 2 | `oracle_reward` | 9 |

The checkpoints are the existing `gate/tile_factored/model_probe.npz`, `gate/tile_factored_s1/model_probe.npz` and `gate/tile_factored_s2/model_probe.npz`, paired to the exact checkpoint/data copies used by each model's archived controls. Models, predictors, heads and encoders are frozen. MLP and native-grid results are always separate; they are different policy classes, not pooled replications.

Each run uses 150 PPO updates × 512 sampled starts × 16 rollout positions = 1,228,800 scheduled positions, chunk length 16, minibatch 128, four epochs, reset-start probability 0.5, gamma 0.99, GAE lambda 0.9, learning rate 2.5e-4, policy clip 0.2, value weight 0.5, entropy weight 0.01, AdamW weight decay 0.01 and gradient norm clip 1.0. Policy initialization, start RNG (`policy_seed + 11`), rollout RNG (`policy_seed + 7`), burn-in, minibatch order and final-only evaluation are inherited verbatim. A 180-minute safety limit matches the existing campaigns; interrupted budgets are ineligible as completed outcomes and remain explicit missing/incomplete entries in the 24-run denominator. No budget or checkpoint is selected from performance.

Evaluation uses 256 real-environment episodes and fixed key seed 20260919. Saved `starts.npz` arrays must match the corresponding archived `oracle_heads` and `oracle_true` controls exactly after canonical integer/shape hashing. Same policy seed gives the same initialization and scheduled start contexts; rollout actions can differ after policies diverge. The evaluation seed, evaluator sources, model/data identities, architectures and all training hyperparameters must match. Equal scheduled positions and updates do not imply equal effective alive weight.

## Contrasts and interpretation fixed before outcomes

The endpoints remain walls passed per real episode and survival probability to the 128-step cap. Survival is return 0 rather than merely length 128; death at the cap remains death. No best policy, seed or endpoint is selected.

1. **Reward replacement under true termination:** `oracle_true − oracle_reward`. Both use the same exact absorbing transitions, encoder and true termination; their sole task-signal difference is true versus original learned reward. This contrast tests the effect of replacing reward under this training protocol. It cannot establish that any particular reward prediction error alone causes a policy's behavior.
2. **Continuation and terminal-extension correction:** `oracle_reward − oracle_heads`. Both use the same learned reward and exact absorbing transitions; true done replaces predicted continuation. This changes preterminal continuation values, masking/bootstrap at death and subsequent terminal-support weight together. It is not exclusively a postterminal artifact test.

The same contrasts are evaluated separately in the MLP and native-grid strata, within each fixed model and policy seed. All 24 new outcomes remain in the report, irrespective of sign, magnitude, numerical failure or incomplete budget. Comparisons require complete matched three-arm blocks; incomplete/invalid blocks remain unresolved, not zero-valued or silently discarded. Intervals are descriptive and the analysis remains explicitly adaptive in admission. Existing original-arm endpoints from `runs/revision/matched` (MLP) and `runs/revision/matched_grid` (native-grid) are reused only after provenance and pairing checks, never retrained or chosen according to the new outcomes. Follow the separate termination analyzer's declared uncertainty procedure; do not treat repeated episodes as independent model replications.

Report the original and new terminal diagnostics: fraction of rollouts that truly terminate, postterminal effective alive-mass fraction and postterminal learned reward per rollout. In `oracle_reward`, the latter two must be exactly zero for finite signals. Nonzero mass, nonzero weighted postterminal reward, nonfinite losses or inconsistent provenance invalidate that run pending investigation. Zero terminal mass is a semantic correctness check, not evidence of improved control. Original learned reward may be positive on a terminal transition; that value is retained without adjustment.

## Verification and output contract

Before launch, require an independent minimal-diff review and CPU toy collector checks for current-state/outgoing-action learned reward, absorbing next observations, true done including the terminal transition, horizon bootstrap, finite batches, and zero postterminal effective weight/reward. Verify the original harness hash again after testing. Policy constructors/savers for both MLP and native-grid are unchanged. A full-checkpoint GPU smoke remains useful after device allocation, but toy checks do not count as control evidence.

Verification completed on CPU with JAX 0.10.1: a four-start/six-step fixture ended at rollout indices 0 and 2 by death, at index 1 by the real environment cap, and left one trajectory alive through the horizon. Incoming action 2 and outgoing action 0 were deliberately different. A separately stepped simulator trace matched every emitted current observation and absorbing frame; a NumPy GAE recurrence matched advantages/returns, including the final surviving bootstrap. Positive learned rewards on first terminal transitions were retained. Positive raw rewards after death remained unchanged, while effective postterminal mass/reward were exactly 0/0 (the original learned-done fixture gave 0.477470/1.805481). All batches and update losses were finite. Actual MLP and native-grid policies each passed one exact-parameter/loss-parity update against the unchanged parent updater and an exact saved-policy reload. Parent and new source hashes remained unchanged. An independent source/protocol review found no blockers and verified the single collector-expression difference; full archived per-block provenance pairing remains an analysis prerequisite.

The output schema is the original harness schema. `config.json` additionally records `parent_harness_sha256` and `sensitivity_harness_sha256`; `source_sha256` contains both harness files plus unchanged dependencies. `final.json`, `episodes.npz`, `starts.npz`, `ledger.jsonl` and `model.pkl` are preserved with their original fields. Every final/config arm is `oracle_reward`. The only changed descriptive field is `terminal_semantics`. The original 45-run report is untouched.

MLP outputs: `runs/revision/termination-mlp/m{0,1,2}/p{0,1,2,3,4}/oracle_reward`. Native-grid outputs: `runs/revision/termination-native-grid/m{0,1,2}/p{0,1,2}/oracle_reward`. Queue admission and GPU allocation are controlled by the experiment coordinator. This protocol itself launches no GPU process; the active target-gradient lane is unaffected.

Example invocation for MLP seed pair (0, 0), from the remote revision checkout:

```bash
/home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python \
  -m studies.revision.matched_ppo_termination --arm oracle_reward \
  --model /home/rldoeswork/personal/stoch/gate/tile_factored/model_probe.npz \
  --data /home/rldoeswork/personal/stoch/gate/train.npz \
  --out runs/revision/termination-mlp/m0/p0/oracle_reward \
  --model-seed 0 --policy-seed 0 --torso mlp \
  --updates 150 --batch 512 --hor 16 --chunk 16 --mb 128 --epochs 4 \
  --eval-episodes 256 --eval-seed 20260919 --max-minutes 180
```

For native-grid, use `--torso native_grid` and the corresponding separate output root. Enumerate every seed listed above prospectively; do not change seed count in response to observed returns.

## Fixed queue execution

`studies/revision/run_termination_queue.py` enumerates all 24 jobs before launch: MLP model seeds 0–2 with policy seeds 0–4, followed by native-grid model seeds 0–2 with policy seeds 0–2. It records the matrix, all source/input identities, logs and completion/failure states under `runs/revision/termination-campaign`; logs remain outside each training output directory. It requires an explicitly assigned idle GPU, defaults to physical GPU 1, and uses a queue lock. The coordinator must first release that lane; the target-gradient experiment on GPU 0 is unaffected.

The queue requires an explicit absolute UTC deadline and records its minimum-remaining-time admission guard. Both are operational budget rules fixed before launch, never functions of control outcomes. A subprocess timeout covers compilation, training and final evaluation. On failure, incomplete budget or deadline, the queue stops without retrying, preserves all partial artifacts, and retains every unstarted job in the planned denominator. It refuses existing campaign/output roots and rechecks immutable source, checkpoint and data hashes before and after each job. Dry run validates and prints the complete plan without importing JAX, querying GPUs, creating files or launching training.

```bash
/home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python \
  -m studies.revision.run_termination_queue \
  --model-root /home/rldoeswork/personal/stoch/gate \
  --data /home/rldoeswork/personal/stoch/gate/train.npz \
  --gpu 1 --deadline-utc DEADLINE_IN_UTC --dry-run
```

The coordinator supplies the deadline and inspects this concrete plan before invoking the same command without `--dry-run`. This operational check does not select policies or alter the scientific matrix. The complete analysis is specified separately in `research/termination_analysis_protocol.md`.
