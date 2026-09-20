# Native Craftax-Classic inventory and achievement protocol

Verified on 2026-09-20 from the installed e2e package, using source reads only. No environment was imported or stepped, and no GPU work was performed. This report concerns **Craftax-Classic-Pixels**, the existing 17-action environment, not the larger Craftax environment. Earlier local-goal and last-only work is superseded and is not part of this protocol.

**Recommendation:** evaluate whether the policy actually crafts native tools during ordinary episodes. Use persistent achievement flags as the primary success observations, record inventory as a consistency check, and report native return separately. Keep the four requested tool outcomes individually visible. These are evaluation measurements; they do not add rewards or provide privileged state to the policy.

## Exact installed identity and source references

- Host: `e2e`; interpreter: `/home/rldoeswork/personal/tinywmcraft-e2e/.venv/bin/python`.
- Versions read with `importlib.metadata`: Craftax **1.5.0**, Gymnax **0.0.9**, JAX/JAXLIB **0.10.1**, Flax **0.12.7**, NumPy **2.4.6**.
- Installed Craftax source root: `/home/rldoeswork/personal/tinywmcraft-e2e/.venv/lib/python3.11/site-packages/craftax`.
- Local read-only source snapshot: `/private/tmp/craftax-native-source-20260920/`; the Craftax source text was copied via a read-only remote tar stream. `gymnax_environment.py` is the installed `gymnax/environments/environment.py`.
- Source references below are relative to that snapshot; they identify inspected installed code, not an assumed upstream revision.

| Source file | SHA-256 |
|---|---|
| `craftax_classic/constants.py` | `c1e78c1c1dc9e6d24f70ea80da194aba1471bac49df9d673c620dddfe77cd090` |
| `craftax_classic/game_logic.py` | `3d1d61066d71fa6730d93fa4c874ca9b862de0e404102ff1fe5b23ccec60da1b` |
| `craftax_classic/envs/craftax_state.py` | `cee0dfe2d0be17803125e366180967dac9485e02d8fdbf10a10856e3cc6e082c` |
| `craftax_classic/envs/common.py` | `51f89ae914892e5b316695d93dec86319ad39c528ca3c1c40f7c80ebd40f8f20` |
| `craftax_classic/envs/craftax_pixels_env.py` | `6fa6a620748326aaeda8415e16aa18e7e0151096a839959b58264d0d7f465cd1` |
| `craftax_classic/world_gen.py` | `cf0b42999d675ef722f74c2e356e8625515948ed35f8e0392d5ce5285c583f4f` |
| `environment_base/environment_bases.py` | `c96af3e0690c0546fef0b6f7b4d605b4c22f6599f2299e1cd71ac9a36ba0ca53` |
| `craftax_env.py` | `f74d828dbc9802984e026ace29293527dfd3901cc945d3c7792d199dc92affa3` |
| `gymnax_environment.py` | `c1be89a1e1cb282ff7435c137e5d31097fcc61277689c0731fada788a4bd59db` |

## Actions, inventory, and achievements

Action IDs and achievement indices are different namespaces. Read enum names in implementation rather than using an action ID to index `state.achievements`.

| Target label | Native action ID | Achievement index | Inventory field | Resource cost |
|---|---:|---:|---|---|
| `MAKE_WOOD_PICKAXE` | 11 | 5 | `state.inventory.wood_pickaxe` | 1 wood |
| `MAKE_STONE_PICKAXE` | 12 | 13 | `state.inventory.stone_pickaxe` | 1 wood + 1 stone |
| `MAKE_WOOD_SWORD` | 14 | 6 | `state.inventory.wood_sword` | 1 wood |
| `MAKE_STONE_SWORD` | 15 | 14 | `state.inventory.stone_sword` | 1 wood + 1 stone |

All four require a crafting table in any of the **eight surrounding cells**, including diagonals. `is_near_block` checks bounds and the `CLOSE_BLOCKS` offsets. The action must succeed under these native prerequisites; merely selecting its ID is not success. A sleeping player has the requested action replaced by NOOP. Sources: `constants.py:45–62,74–86,107–129`; `game_logic.py:385–440,475–511,1644–1648`.

The inventory contains twelve separate integer counters, initially zero: `wood`, `stone`, `coal`, `iron`, `diamond`, `sapling`, `wood_pickaxe`, `stone_pickaxe`, `iron_pickaxe`, `wood_sword`, `stone_sword`, `iron_sword`. It is not a single highest-tool-tier field. Successful crafting subtracts ingredients and increments the corresponding tool counter. All inventory entries are capped at 9 at the end of the transition. The inspected Classic logic increments these tool counters and has no durability/decrement path for them. Sources: `envs/craftax_state.py:9–22`; `game_logic.py:397–546,1632–1635`.

Each crafting achievement is updated as `old_flag OR successful_craft`. It therefore remains true for the rest of that episode. Crafting a second copy can increase inventory but does not earn the same achievement again. A native reset creates zero inventory, all 22 achievements false, health 9, and timestep 0. Sources: `game_logic.py:411–420,431–441,482–511`; `world_gen.py:230–256`.

The mechanics create a meaningful progression without extra shaping: mining a tree with native `DO` yields wood; placing a table costs 2 wood; a wood pickaxe allows mining stone; stone tools require the collected stone. In particular, the native Classic stone-mining check explicitly reads `inventory.wood_pickaxe`. Sources: `game_logic.py:189–228,375–380,586–612`. This description explains the task and must not become a scripted controller or action mask for one comparison arm.

## Native reward, return, score, and episode end

Let `A_t` be the 22 native achievement flags and `h_t` player health after transition `t`. The exact native reward is

`r_t = sum(A_t) - sum(A_(t-1)) + 0.1 * (h_t - h_(t-1))`.

Thus the undiscounted sum through any correctly recorded episode prefix must satisfy

`R_T = sum(A_T) - sum(A_0) + 0.1 * (h_T - h_0)`.

For an ordinary fresh reset, `A_0=0` and `h_0=9`. This telescoping identity is a useful evaluator audit. It is not a replacement reward: retain the actual returned native rewards and check their sum against the identity up to floating-point error. Do not subtract the health component, add a tool bonus, reward action selection, multiply rare achievements, or substitute learned-model reward when reporting real-environment return. Source: `game_logic.py:1641–1642,1676–1682`.

The native `info` fields use a different computation. `Achievements/<lowercase_name>` is `100 * achievement_flag * done`. The native scalar is

`info['score'] = exp(mean_j(log(1 + 100 * A_T[j] * done))) - 1`, over all **22** achievements.

On nonterminal steps those fields are zero even if tools have already been crafted. At a terminal state with `k` achievements, this scalar is `101**(k/22) - 1`. It is neither native return, number of achievements, nor tool success. Averaging this per-episode scalar is also different from first estimating population achievement rates `p_j` and then computing `exp(mean_j(log(1+100*p_j)))-1`. If the latter is reported, name it an **achievement-rate geometric aggregate**, state the equation, and retain the 22 individual rates. Do not silently interchange the two. Source: `envs/common.py:5–12`.

Native termination occurs after the transition if **any** of these are true: `timestep >= params.max_timesteps`, player stands on lava, or health is nonpositive. The installed default is **10,000 steps**, not 1,000; `day_length=300`, `always_diamond=True`, and the default map is 64×64. The transition increments `timestep` before the environment checks terminal status. Sources: `envs/craftax_state.py:77–100`; `game_logic.py:7–15,1686–1689`; `envs/craftax_pixels_env.py:35–50`.

## Terminal-state handling

Use `make_craftax_env_from_name('Craftax-Classic-Pixels-v1', auto_reset=False)` or the equivalent `CraftaxClassicPixelsEnvNoAutoReset`. It returns the real post-action state. It does **not** make dead/finished states absorbing, so the evaluator must stop each episode at its first native `done` and freeze its state and accumulated metrics thereafter. Source: `craftax_env.py:48–53`; `environment_base/environment_bases.py:17–30`.

The auto-reset class instead inherits installed Gymnax's `Environment.step`: it splits the supplied key into step/reset keys, computes native reward/done/info, computes a reset, and selects the reset observation/state when done. Its returned terminal reward and info describe the ending episode while its returned state describes the next one. Reading terminal inventory from that returned reset state loses the achievement. Moreover, the auto-reset and no-auto-reset wrappers do not consume the same supplied key identically; all comparison arms must use the same wrapper and key-construction rule. Sources: `envs/craftax_pixels_env.py:91–123`; `gymnax_environment.py:47–60`.

Include the terminal transition in success and return. Crafting happens before mob updates and health updates, so a tool can be earned on the same step the player dies. Native achievement success counts this event. Record survival separately if surviving acquisition is of interest; adding an alive requirement changes the endpoint.

## Proposed evaluation contract

1. **Freeze the evaluated episode distribution before observing outcomes.** Use ordinary native resets, unchanged default environment/static parameters, all 17 native actions, and the same policy observation/preprocessing interface across arms. The four requested achievements are evaluation targets; do not inject resources, pre-place tables, filter for favorable maps, remove enemies, restrict actions with symbolic rules, or initialize from achieved states. If a policy has a genuine goal-conditioning interface, declare and match that interface across arms; otherwise measure all four achievements on the same generic-policy episodes without implying goal-conditioned control.
2. **Choose and name the horizon.** Full native episodes use the first native done, at most 10,000 actions. A cheaper prespecified evaluator cap `H < 10000` is permissible as an explicitly named “success by H actions” measurement; keep native `max_timesteps` unchanged and identify live episodes stopped by that cap as truncated. Do not report capped prefixes as complete native episodes. Use the same cap for every arm, and do not choose it after inspecting tool success.
3. **One episode means one native reset.** For every active step, accumulate the native reward; inspect the post-action achievement vector and inventory; record first acquisitions; then apply native done/evaluator-cap stopping. For a vectorized fixed-length scan, gate every metric update by the pre-step active mask and preserve the first terminal state. Do not sum rewards or achievement changes from post-terminal transitions or subsequent resets.
4. **Primary outcomes are four individual episode success rates.** For each requested target `g`, use `success_g = bool(A_T[g])` and report successful episodes / all admitted reset episodes. Also report the mean number of these four tools obtained per episode and the all-four rate as prespecified summaries. Keep all 22 native flags for broader skill context. Do not use a union that conceals failure to obtain stone tools as the sole result.
5. **Acquisition timing and survival are separate.** `first_step_g` is the first post-action step in `1..T` with the achievement true; otherwise `null`. A death/lava failure before acquisition is a competing failure, not ordinary right-censoring. A live episode stopped at the evaluator cap is censored for acquisition timing but remains unsuccessful in the fixed-H success denominator. Show unconditional success-by-step curves or explicitly modeled competing-risk estimates; a mean time among successes alone is not overall acquisition speed.
6. **Record native return without relabeling it.** Save undiscounted native `sum(r_t)`, the full achievement count, final health, survival, executed steps, and each termination condition. Store native terminal `info['score']` only when a native terminal transition occurred. If computing the same formula from a live capped endpoint, label that quantity an endpoint recomputation. State whether any geometric summary aggregates per-episode scores or achievement rates.
7. **Pair environmental contexts and preserve independent model replication.** Reuse a prespecified reset-seed list across policies. Derive environment step keys solely from episode identity and step, separately from policy sampling keys; save the construction. Retain every model/policy training seed and report its four success rates. Paired episode contrasts within a seed and variability across independent trained policies answer different questions; pooled model×episode rows do not create additional independent training replicas.

For ordinary native resets and this pinned Classic source, `inventory.<tool> > 0` should agree with the matching achievement flag. Record both and flag any disagreement for investigation. The achievement flag remains the direct native success definition; no visual classifier or predicted reward head is needed to grade it. State is used by the evaluator only and is not an extra policy/world-model input.

Recommended per-episode record:

```text
policy_checkpoint_sha256, world_model_checkpoint_sha256_if_used
policy_training_seed, world_model_training_seed_if_used
reset_seed, episode_id, environment_key_rule, policy_key_rule
environment_source_hashes, environment_params, static_params, evaluator_horizon
executed_steps, native_done, health_nonpositive, in_lava, native_time_limit, evaluator_truncated
native_return, initial_health, final_health, all_22_final_achievement_flags
four_tool_success_flags, four_first_acquisition_steps_or_null, final_inventory
native_terminal_info_score_or_null, optional_named_endpoint_score
```

Minimum acceptance checks for a future evaluator: native action IDs are in `0..16`; every reset starts with empty inventory/achievements; flags are monotone within each episode; terminal-action rewards and acquisitions are included; no post-terminal updates enter metrics; inventory/achievement indicators agree; and native return satisfies the telescoping identity. These are proposed implementation checks, not claims that a new evaluator or policy has already been run.

## Scope of the verification

The action/achievement mapping, prerequisites, persistence, reward, score, reset, termination and auto-reset semantics above are verified from the copied live installed source. A second read-only reviewer independently checked the native reward/score and no-auto-reset boundary reasoning. No policy provenance, learning effectiveness, acquisition success rate, numerical rollout, or runtime is inferred by this source audit.
