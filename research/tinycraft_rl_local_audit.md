# TinyCraft WM-policy audit — local source and artifacts

Audit date: 2026-09-20. Read-only inspection of adjacent repositories and saved result files; no remote calls, policy training, GPU calls, or modification of those repositories. This report is the only new artifact. Checkpoint binaries and metrics are distinguished below.

**Verdict:** There is an executable, general pixel-policy pipeline that learns inside a frozen TinyWMCraft model and obtains native wood tools in real Craftax. The most directly reusable version is `train_ppo_real.py --wm ...` with `--wm_once` absent. The filename and saved `arm=ppo_real` label are misleading: its `--wm` branch replaces training resets/steps with the learned simulator, while evaluation remains in real Craftax. A separate short-dream trainer supports identical BC initialization/anchors across WMs. High BC-initialized scores are not evidence that WM refinement learned those tools from scratch; the stored final and selected-best results differ substantially.

Path abbreviations used only in this report:

- `B = /Users/rldoeswork/Documents/rl/tinywmcraft-oss/.claude/worktrees/rl-eval-plan/studies/rl_eval`
- `O = /Users/rldoeswork/Documents/rl/tinywmcraft-oss`
- `P = /Users/rldoeswork/Documents/rl/tinywmcraft-pop-flow-wm`
- `J = /private/tmp/tinycraft-jepa-iclr-20260919`

## 1. Actual code and immutable WM identity

The ordinary `O` checkout (HEAD `22d057faae061388ab96a9c13462e2c834c748da`) contains WM training, evaluation, and keyboard play, but no policy trainer. The RL work is untracked under the hidden worktree above (worktree HEAD `d10d9d68c77bb8910221488603d5e4619cf6c4b9`). HEAD alone therefore does not identify the RL source.

The September Flow checkpoint exists locally at `/Users/rldoeswork/Documents/rl/wm_ckpt/normuon_head_bridge_s0.npz`:

- File SHA-256: `b4d7b82ba698f1b7aeaab7383c324523bf27bde1aba2fe817745fce38efa0aef` (55,515,862 bytes), independently rehashed in this audit.
- This matches `O/validation/normuon_seed023.json:80`.
- Embedded data SHA: `09f2cb41a38b0f34937a89c4122f6b25bbb3094e34a077b3a572504269d0a1b2`, the same 500k archive used by the present JEPA experiment.
- Embedded training-source SHA: `0cee51a6007fe8b3e732e9cfedf9c241693e2caf5df194da91b78c2e9822f9f9`; the current bytes of `O/train.py` match exactly.
- Seed 0; 9,000 tokenizer and 16,000 WM updates; 32-frame rollout; native 17 actions; `train_s=566.6`.
- `B/chain_e2e6.sh:15` names its remote experiment copy as `wm/sota2_headbridge_s0.npz`. The local equality of that remote filename cannot be inferred; its bytes still need a remote hash.

There are no policy `.pkl` or WM `.npz` files under local `B`. It contains code, configs, logs, JSONL, and visual artifacts. The policy binaries named in those records must be recovered and hashed remotely before evaluation or reuse.

## 2. Policy trained inside the WM, versus data collection

**Continuous PPO in the frozen WM:** `B/train_ppo_real.py:82-88` constructs the real evaluator first, then replaces only training `vreset/vstep` with `wm_gym.make_env` when `--wm` is present. The policy alone is optimized (`:80-81,148-150`). Its settings are 64 environments, 100-step collection, recurrent chunks of 25, four groups, four PPO epochs, AdamW learning rate 2.5e-4/weight decay .01, global gradient clip 1, gamma .99, GAE lambda .8, ratio clip .2, value coefficient .5, entropy .01 (`:41-58`). Every iteration consumes 6,400 imagined transitions and performs 16 PPO minibatch updates. A nominal 5M budget ends at 4,998,400 transitions. World-model and policy recurrence continue between collections and reset on predicted termination.

`B/wm_gym.py:28-34` obtains starting images only from recorded episode resets. `:45-50` loads the frozen WM and caches encoded reset images. `:62-84` slides the model window, predicts a next latent, decodes pixels, and returns the model's reward and termination prediction. There are no real-environment training transitions in this branch. `B/train_ppo_real.py:190,193` still saves `arm="ppo_real"`; the config and `[init] environment = world model ...` log are the authoritative classification evidence.

**Short-dream recurrent PPO:** `B/train_dream.py:88-111` loads an optional actor checkpoint and a frozen WM. `:175-201` alternates fresh imagined rollouts, actor-critic updates, and optional BC updates. The September interface is `B/wm_env2.py`: eight real context frames, 50% reset starts and 50% eligible dataset starts, native latent propagation, pixel decoding, single-model predicted rewards/termination, and stop-gradient rollout batches. Default matrix: batch 1,024, horizon 24, 400 updates = 9,830,400 imagined transitions; chunks 8, minibatches 256, one PPO epoch (12 minibatch updates per dream), gamma .99/lambda .9. `nfe=4` is the historical Flow sampler setting, not an instruction to alter JEPA latents.

**Data collector is different:** the 500k dataset comes from an independently trained stochastic discoRL behavior policy. Its checkpoint (for example `tinywmcraft-scratch-claude/runs/disco_policy_50m/policy_params_final.npz`) is not a policy learned in this WM. Likewise, `train_ppo_real.py` without `--wm` is real-environment PPO and is a model-free control, not frozen-WM evidence. The older `P/train_dream2.py` genuinely optimizes its actor inside a frozen WM, but uses a different 218,848-transition expert-plus-random dataset and different WM/head recipe; do not pool it with this 500k study.

## 3. Shared actor and BC initialization

`B/agent.py:113-143` defines the common actor: RGB Impala CNN with channels 64/128/128, 512-wide embedding, GRU-256, 512-wide final hidden layer, 17 action logits, and a 128-bin HL-Gauss critic. The unconditioned recurrent net has 6,818,193 parameters. Its input is one `(B,63,63,3)` uint8 frame, previous native action `(B,)`, recurrent state `(B,256)`, and a `(B,cond_dim)` conditioning array. It never consumes the WM's latent directly. At evaluation it acts on real pixels, with no WM call. `B/agent.py:162-171` restores this same architecture and actor weights from both current and old policy checkpoint formats.

The shared BC initializer is `runs/bc_rtg/model_best.pkl`, selected at BC update 4,500 according to `B/runs/ledger_final.jsonl:2`. It is return-to-go conditioned: one scalar `(target-return_so_far)/10`, target 13. The training data is the same 500k archive, using all columns; `B/data.py:23-41` computes return-to-go from the recorded native rewards/dones, without temperature metadata or inventory labels. Recurrent BC uses batch 96, burn-in 8 plus 24 scored positions, learning rate 3e-4, and a nominal 6,000-step fit (`B/runs/bc_rtg/config.json`). That initializer is already competent at obtaining tools.

Two refinement recipes must remain separate:

- Anchored PPO: same initializer, lr 5e-5, ratio clip .1, four recurrent BC steps after each dream update, 15 initial updates with policy-gradient terms disabled. Historical August run stopped at update 150; it is not a completed 400-update baseline.
- KL PPO: same initializer, lr 2.5e-4, ratio clip .2, no BC anchor steps, KL coefficient .01 toward a frozen copy of the initializer, 15-update delay. September runs use a single WM without pessimistic reward modification.

The comments describing a literally frozen actor during `pg_delay` are too strong: `B/train_dream.py:146-149` retains critic loss and updates the shared CNN/GRU, so the actor's outputs can change even while its direct PPO/entropy/KL terms are disabled. This behavior must be held fixed in a comparison or changed explicitly in every arm.

## 4. Native tool achievements, not walking or a visual parser

`B/craftax_env.py:17-22` uses native `Craftax-Classic-Pixels-v1`, `auto_reset=False`, and default environment parameters. `:29-57` runs sampled actions until each episode ends, freezes completed slots, accumulates the native reward, and returns the terminal `(N,22)` `state.achievements` matrix. `:67-72` reports per-achievement rates. The score is `exp(mean(log(1 + 100 * achievement_rate))) - 1` (`:61-64`). The main evaluation is `B/evaluate.py`, ordinarily 256 episodes and seed 123. These rates are native simulator flags, not actions attempted, latent goals, or parsed imagined frames.

For reference, local vendored Craftax source maps wood pickaxe/sword and stone pickaxe/sword to achievement indices 5/6/13/14, and crafting action IDs 11/14/12/15. The corresponding inventory fields are `wood_pickaxe`, `wood_sword`, `stone_pickaxe`, `stone_sword`. A new evaluation should retain both the native achievement matrix and terminal inventory, keyed by episode, rather than only rounded aggregate rates. Existing evaluator logs only the achievement rates; inventory vectors and raw per-episode outcomes are not archived in the inspected JSONL.

Exact records (rates below are the stored three-decimal fractions converted to percentages; do not invent more precision):

| Policy/checkpoint | Episodes | Return ± SEM | Wood pickaxe | Wood sword | Stone pickaxe | Stone sword |
|---|---:|---:|---:|---:|---:|---:|
| BC RTG initializer, selected update 4500 | 256 | 9.810 ± .140 | 79.7% | 89.5% | 1.2% | 0.0% |
| September WM PPO, scratch, final update 400 | 256 | 3.918 ± .074 | 29.3% | 0.4% | 0.0% | 0.0% |
| September WM PPO, scratch, selected update 325 | 256 | 4.089 ± .086 | 30.1% | 1.2% | 0.0% | 0.0% |
| September WM PPO, BC+KL, final update 400 | 256 | 8.924 ± .072 | 98.8% | 1.2% | 0.0% | 0.0% |
| September WM PPO, BC+KL, selected update 275 | 256 | 10.128 ± .090 | 98.8% | 84.4% | 0.4% | 0.0% |
| Continuous WM PPO, recurrent, final 4,998,400 steps | 128 | 5.633 ± .184 | 73.4% | 57.0% | 0.0% | 0.0% |
| Continuous WM PPO, no GRU, final 4,998,400 steps | 128 | 6.199 ± .148 | 92.2% | 3.1% | 0.0% | 0.0% |

Sources: BC is `B/runs/ledger_final.jsonl:2`; four September rows are `B/runs/ledger_sota2.jsonl:1-4`; continuous rows are the last records of `B/viz/wmgym/wmgym_5m/ledger.jsonl` and `.../wmgym_5m_nogru/ledger.jsonl`. The continuous rows are training-time evaluations with their own episode seeds, not the separate 256-episode final grades quoted in the report. `B/REPORT.md:329,332` incorrectly names the selected September updates as 375/250; the raw checkpoint metadata says 325/275.

The incremental September BC+KL result is therefore mixed: the selected checkpoint increases wood-pickaxe frequency but does not improve wood swords or stone tools versus BC; the fixed final checkpoint largely loses wood-sword behavior. This is not evidence for robust broad tool improvement attributable to WM PPO.

## 5. Reward/termination hooks and exclusions

- **Exclude `--wm_once`/`wmonce_*`.** `B/wm_gym.py:75-80` tracks crafting/placing action IDs in a bitmask and suppresses later positive rewards. This is a hand-coded action reward rule, not a native predicted reward. The clean branch (`once=False`) leaves `rew[:,slot]` unchanged.
- `B/wm_env2.py:124-134` uses the single WM's reward and termination predictions directly. Its `pen` parameter is unused; there is no added bonus or action-dependent reward clipping. Standard PPO advantage normalization and likelihood-ratio clipping occur in the learner, not in the environment reward.
- August `B/wm_env.py:115-124` and old `P/train_dream2.py:158-170` use pessimistic ensemble reward combinations. Keep them separate from the clean single-WM condition. September `*_pessimistic_bug` records must not be reused: the report identifies an invalid cross-tokenizer head combination; current `wm_env2.WorldModel` rejects a second checkpoint.
- Short dreams use soft predicted continuation in GAE and the alive mask. Continuous `wm_gym` samples Bernoulli termination and additionally terminates at 1,024 imagined steps (`:82`). That is an explicit training time limit. The local vendored native environment has a 10,000-step limit; verify the actual installed version's `ep.max_timesteps` remotely and disclose any difference. It is not correct to claim identical termination semantics without this qualification.
- Historical best-checkpoint selection uses real evaluation returns (`B/train_dream.py:165-167`); a new fixed-budget experiment should retain all outcomes and use the prescribed final checkpoint, without reusing historical selection as if it were prospective.

## 6. Viable JEPA comparison within the remaining budget

**Technical feasibility: yes**, using the same RGB actor and a separate JEPA simulator adapter. No latent enlargement or new actor is needed. `J/train.py:531-550` already loads a `Bundle(model,decoder,heads)`; `J/probe.py:22-48` decodes native 512-dimensional JEPA latents to 63×63 RGB; `J/probe.py:75-78` returns native-scale reward and done-logit predictions from the current latent plus outgoing action.

Required adapter transition:

1. Cache reset frames from the identical archive and initialize the JEPA's native eight-frame context. Preserve its raw 512 coordinates, frozen parameters, and BatchNorm statistics.
2. Select the actor action on the current decoded/real pixel frame. Read JEPA reward/done from `(current_latent, outgoing_action)` **before** advancing. The Flow adapter's reward-on-arrival index cannot simply be copied.
3. Replace the last outgoing-action placeholder with that action; use `model.predict(history, model.act_embed(actions))[:, -1]`; shift native history eight by one; decode the prediction to uint8 pixels. `J/train.py:555-576` documents the outgoing/incoming alignment.
4. Return raw single-head reward and sigmoid done probability; no behavior-value head, action tracker, reward correction, bonus, or latent transform. Use the same sampled/soft termination rule and declared time-limit treatment as the chosen Flow arm.
5. Verify that a fixed recorded-action rollout matches `J/train.py`'s existing rollout path, that reward alignment is correct, and that frozen parameter/BatchNorm hashes are unchanged.

For a BC-refinement comparison, restore the **same exact BC checkpoint** in Flow and JEPA, and hold its RTG scalar, optimizer, actor architecture, start indices, action RNG, rollout budget, PPO epochs, anchor/KL schedule and final checkpoint fixed. Include both the untouched BC initializer and a matched extra-BC-only control when using repeated BC anchor updates; otherwise a gain could come from the extra BC steps. The unconditioned continuous-PPO trainer cannot directly reuse the RTG actor: it supplies a `(B,0)` condition array (`train_ppo_real.py:71,95`); use the condition-aware dream trainer or deliberately implement the same conditioning in both arms.

For clean from-scratch WM control, the most direct recovered recipe is recurrent continuous PPO at 4,998,400 imagined transitions in each frozen simulator with `once=False`, a fresh common actor initialization, and native full-episode achievement evaluation. It already obtains wood pickaxes/swords without BC, so it directly addresses the user's request. Keep no-GRU as separate evidence, not a favorable architecture substitution.

The comparison remains a **frozen-system comparison** unless decoder/reward/done supervision is also matched. The existing JEPA probe used only 80,000 frames/transitions, whereas Flow's heads trained with its full 500k WM dataset. Existing JEPA held-out reward-head weakness is a known confound; poor policy transfer would not isolate the JEPA dynamics objective. Fitting identical frozen-latent readouts across JEPA objectives is possible, but must be separately named/budgeted and must not silently update the WMs.

Historical observed timing: continuous recurrent WM PPO at 5M took 2,799.2 s (~47 min); no-GRU 2,736.4 s; BC RTG 6,000 updates took 961.1 s. The September 400-update short-dream report quotes ~30–32 min; the older August raw ledgers show 4,980–5,043 s. These are past runs, not a current timing guarantee. An 8-hour budget plausibly accommodates adapter checks, a fixed paired experiment and evaluation, conditional on recovered binaries/runtime and a measured timing check. Do not launch an unbounded `max_hours` loop or select seeds/checkpoints by outcome.

## 7. Receipts and remaining evidence gaps

Current local source SHA-256:

| Source under B | SHA-256 |
|---|---|
| `train_dream.py` | `3bff1e517b869003f81c6e7a81a22d14cd6927be2dbb65da860b79db4b88ccfa` |
| `train_ppo_real.py` | `fe11deb02d014a69b8f5a31b6ac025266ed03ac6ff37a44dc568ad50decea9f6` |
| `wm_env2.py` | `ee369385583d7cb3eab49e0899fc791632608c51a709c83df45844ba9017bc63` |
| `wm_gym.py` | `a9c787d1ca02463e8c7df7ec35cc55e1a7a31e5ce93eedf2184b8bda58623dcd` |
| `agent.py` | `1bfeb69f0a6098959a409c4601e8a92219b1967f297b1e35beb96440edceb132` |
| `craftax_env.py` | `38e333992872625a5a1883116ca741e9b2e6cba7761dd0b2c0f50675d2af9e43` |
| `runs/ledger_sota2.jsonl` | `f8740411a3f24d789ca245585bb1be31eabc8c1a6912894d0d8f646a706a1337` |
| `runs/ledger_final.jsonl` | `26117f5c0f08393e9e1eabdb54115ed1653f8d411ccffba21e993d9e8e9c0976` |

Still required: exact remote actor checkpoint hashes and architectures; confirmation that remote source bytes correspond to each run rather than later edits; clean `wm_once=False` launch configs; installed package/runtime identities; native termination limit; fresh per-episode achievements/inventory and returns under one fixed evaluator. Local aggregate JSONL is useful historical evidence, but does not supply immutable actor identity or raw paired-episode uncertainty.
