# Existing TinyCraft WM-trained policies on multigpu-vm

Read-only live inventory, 2026-09-20. No models were trained, evaluated, or modified in this audit. Checkpoint metadata was deserialized on CPU only. SSH identity was verified as `rldoeswork` on `multigpu-vm.sas.yp-c.yandex.net`; the inspected files belong to that user.

**Finding:** retained policies were actually optimized inside a learned TinyCraft world model and subsequently made tools in native Craftax. The strongest fully traced existing example in this inventory is `wmgym_5m_nogru`: 94.5% wooden-pickaxe success on its recorded 256-episode real-environment evaluation. Stone tools are rare. This is a diffusion/flow WM with learned reward/termination heads, not the JEPA local-goal model. The evidence does not establish the same result for JEPA.

## Precisely traced candidate

All relative paths below are under `/home/rldoeswork/personal/wm_rl_eval2` on multigpu-vm.

- Trained policy: `runs/wmgym_5m_nogru/model.pkl`; retained copy: `pol/wmgym_5m_nogru.pkl`. Their live SHA256 hashes match exactly: `6570703f8b3b244631d28bfd9410f95b56ce3625e39378d5bb0d321431960836`.
- Training configuration: `runs/wmgym_5m_nogru/config.json`, SHA256 `4ed59ff1e2908805b57946f5b978cf0f497c29d9102cd0fb95538f1ae6ba04c2`.
- Log: `runs/wmgym_5m_nogru.log:4` explicitly records the training environment as `/home/rldoeswork/personal/codex-wm-push-20260916/checkpoints/normuon_head_bridge_s0/model.npz`, four Euler steps, window 32, cap 1024, with evaluation remaining in real Craftax.
- Configuration: seed 0; `ac_from=null`; `no_gru=true`; budget 5,000,000; actual final checkpoint 4,998,400 imagined transitions, iteration 781; 64 environments, 100-step collection, four PPO epochs, learning rate .00025, gamma .99, lambda .8, clip .2, value coefficient .5, entropy coefficient .01. The policy has 6,092,177 parameters and consumes decoded pixels and the previous native action.
- `train_ppo_real.py:73-85` initializes the policy and creates the native Craftax evaluator first, then replaces training `vreset/vstep` with `wm_gym.make_env` when `--wm` is supplied. Collection at lines 99-110 consumes that replacement's observations/rewards; PPO gradients are computed at lines 137-156. This is genuine WM-policy optimization despite the script name.
- The final training log lines `runs/wmgym_5m_nogru.log:887-888` finish at iteration 781 / 4,998,400 steps. `train_ppo_real.py:187-193` saves this final `model.pkl` without best-return selection.
- Native evaluation ledger: `runs/ledger_wmgym.jsonl`, SHA256 `5286ad642d17d4cf873a5bbebaac2504b0cfff5794da0ccbc9ce7533d6b417f1`. Its final row names the exact retained run checkpoint and agrees with its metadata about iteration, steps, and the separate final in-training evaluation.

The CLI corresponding to the saved configuration is:

```sh
python train_ppo_real.py --out runs/wmgym_5m_nogru --seed 0 \
  --budget 5000000 --save_at 500000 1000000 2000000 \
  --wm /home/rldoeswork/personal/codex-wm-push-20260916/checkpoints/normuon_head_bridge_s0/model.npz \
  --wm_data /home/rldoeswork/personal/tinywmcraft-research/data/train.npz \
  --wm_nfe 4 --wm_max_steps 1024 --no_gru
```

This command is reconstructed from the saved config and current CLI, not an archived shell command. Runtime module selection is additionally controlled by `TINYWM2_ROOT`/`TINYWM2_MODULE` (`wm_env2.py:30-31`); their historical values are not recorded in this run config. The current source files were hashed, but the policy checkpoint does not bind them cryptographically to training time.

## Existing native achievement evidence

All rates in the first table come from the saved independent evaluation, seed 123, 256 episodes, sampled actions. Rates were rounded to three decimal places when logged; raw per-episode achievement matrices were not recovered here. Return uncertainty is the evaluator's across-episode SEM, not model-seed uncertainty.

| Policy/checkpoint | Training interaction | Native return ± SEM | Wood pickaxe | Wood sword | Stone pickaxe | Stone sword |
|---|---|---:|---:|---:|---:|---:|
| `wmgym_5m_nogru/model_2000k.pkl` | 2,003,200 imagined transitions | 4.551 ± .113 | 27.7% | 30.5% | 0% | 0% |
| `wmgym_5m_nogru/model.pkl` | 4,998,400 imagined transitions | 6.220 ± .103 | 94.5% | 3.1% | 0.4% | 0% |
| `ppo_5m_nogru/model.pkl` | 4,998,400 real transitions | 10.150 ± .115 | 94.5% | 76.2% | 0.4% | 0% |

The 2M policy SHA256 is `dd3549e76d0e7d4825f986f43c2ce7ffd5e4312d4434afb2f428b957be0227db`. The real-PPO final checkpoint and retained `pol/ppo_5m_nogru.pkl` match at `7320bd519c980bbc70fdad05fd7fd446bd162edf38093ba374ad911acc3ca6b8`.

This table exposes progression and tradeoffs in the same WM-policy run: wooden-pickaxe success rises while wooden-sword success declines. The intermediate checkpoint is not a replacement for the final checkpoint. There is one policy training seed here. No new evaluation or uncertainty estimate was manufactured.

The evaluator uses `Craftax-Classic-Pixels-v1` with default parameters and no autoreset (`craftax_env.py:17-22`). Finished slots are frozen (`:48-57`); achievement rates come directly from `es.achievements` (`:57`, `:67-72`), not parsed videos, action frequencies, predicted rewards, or model-rendered inventory. They certify an achievement during the real episode, not a final inventory count.

## WM identity and training data

- Frozen WM path: `/home/rldoeswork/personal/codex-wm-push-20260916/checkpoints/normuon_head_bridge_s0/model.npz`.
- Live WM SHA256: `b4d7b82ba698f1b7aeaab7383c324523bf27bde1aba2fe817745fce38efa0aef`.
- WM metadata `source_sha256`: `0cee51a6007fe8b3e732e9cfedf9c241693e2caf5df194da91b78c2e9822f9f9`, matching the live `/home/rldoeswork/personal/codex-wm-push-20260916/recipes/model-51ffe8df/train.py`.
- WM metadata data SHA256: `09f2cb41a38b0f34937a89c4122f6b25bbb3094e34a077b3a572504269d0a1b2`; 500,000-transition budget, 17 actions, seed 0, tokenizer 9,000 updates, WM 16,000 updates, rollout 32, learned reward/done coefficients .1/.1. Recorded training time is 566.6 seconds including tokenizer time 169.8 seconds. This audit did not rehash the full training archive.
- `wm_env2.py:57-62` calls the WM's learned transition sampler and its own reward/termination heads. `wm_gym.py` decodes predicted latents to pixels for the policy, samples termination, and resets from the offline dataset's episode-start frames. No real simulator transitions enter the policy training update in this path; real interactions were nevertheless used to collect the WM dataset and to evaluate policies.

## Other retained candidates, with weaker or different provenance

| Retained policy | Existing metadata evidence | Limitation |
|---|---|---|
| `pol/wmgym_5m.pkl` | Matches `runs/wmgym_5m/model.pkl`; scratch recurrent WM-PPO, 4,998,400 imagined steps. Final in-training real evaluation: 128 episodes, return 5.633 ± .184, wood pickaxe 73.4%, wood sword 57.0%, stone tools 0%. | This is the final in-training evaluator result, not the 256-episode ledger above. |
| `pol/wmgym_20m.pkl` | Recurrent policy metadata records 20,000,000 steps; final recorded native evaluation 128 episodes, return 5.849 ± .127, wood pickaxe 93.0%, wood sword 5.5%, stone tools 0%. | Original corresponding run/config was not recovered in the bounded MG search; do not assert exact continuation provenance solely from its name. |
| `pol/dream2_scratch.pkl` | Metadata `phase=dream`, update 325, 7,987,200 imagined steps; native evaluation 128 episodes, return 4.074 ± .107, wood pickaxe 27.3%, wood sword .8%, stone tools 0%. | Copied policy only; original run/config and final-versus-selected-checkpoint identity are not established here. The name does not prove scratch initialization. |
| `pol/dream2_kl.pkl` | Metadata `phase=dream`, update 275, 6,758,400 imagined steps; return-conditioned architecture; native evaluation 128 episodes, return 10.161 ± .089, wood pickaxe 100%, wood sword 85.9%, stone pickaxe/sword .8%/.8%. | Original run/config not recovered; source supports BC initialization plus a reference-policy penalty. Do not call this learning from scratch or a pure WM effect. Real-evaluation checkpoint selection may apply. |
| `pol/dream_kl_oldwm.pkl` | Dream metadata and return conditioning; recorded native evaluation 128 episodes, return 8.905 ± .133, wood pickaxe 99.2%, wood sword 78.9%, stone sword .8%. | This is the older-WM family. Later scoring inside the newer WM does not change which WM trained the policy. |

Hashes:

```text
058619aad0d5972ac7327295bbde9e21707442e8fe48adeb6e81b48d79a2c512  pol/wmgym_5m.pkl
61c07782a8cfe4c07e7859d3fa7dd6d122d2a14c0da902e84c653292d0ab1a12  pol/wmgym_20m.pkl
1b3b0b5a0b2ff78ed3d03246758da1ab2cc1128c990913df363a69d2c4128249  pol/dream2_scratch.pkl
4a23acd5e568d24b42328c20f6b1823a5a276387675d3bb9098c7427077ee1f8  pol/dream2_kl.pkl
c9d48abc35994e398f09dde0af0501478e1b20445183c478d315c7db08ebfe3e  pol/dream_kl_oldwm.pkl
```

## Exclusions and material limits

1. Exclude `wmonce_*` from an unmodified-reward comparison: its saved configs set `wm_once=true`; `wm_gym.py:75-80` tracks place/make actions in an explicit bitmask and suppresses later positive rewards for those actions. This is a task-specific reward intervention. The earlier `wmgym` configs/logs predate that option; its field is absent there, whereas the modified runs record it explicitly. The historical initialization log also lacks the once-status text now emitted by the current trainer. The exact historical source snapshot was not retained with those policy checkpoints.
2. Checkpoint metadata `arm=ppo_real` is hardcoded for both real and WM runs (`train_ppo_real.py:190,193`). It is not a reliable provenance classifier. Config, initialization log, retained run checkpoint, and native evaluation ledger together support the primary candidate.
3. `train_dream.py` can save `model_best.pkl` by real-evaluation return. Copied dream policies need original-hash matching before claiming fixed-final-checkpoint results. Their metadata `kl` is an observed diagnostic that overwrites the configured coefficient; it cannot recover the regularization coefficient.
4. Existing rulebook/video diagnostics are different endpoints. For example, `viz/score512_wmgym_5m_nogru_rules.json` parses real and generated pixels over 128 worlds ×512 steps. Native `es.achievements` is stronger evidence for toolmaking than those parsed events. No results here are inferred from a video.
5. An existing 128-world ×512-step diagnostic, `viz/long_wmgym_5m_nogru.json`, reports native return 6.323 versus model-head return 32.365 (alive-weighted model-head return 30.876). This demonstrates a substantial learned-reward/transfer gap in that diagnostic; it is not native environment success. It should not be merged with the 256-episode evaluation protocol.
6. The discovered `tinywmcraft-slim` directory is an extensive WM architecture sweep without a `.git` repository. Its presence does not bind every policy to a slim checkpoint. The traced policy instead names the September `normuon_head_bridge_s0` checkpoint above.

Current source hashes (snapshot provenance, not proof of historical training-source immutability):

```text
fe11deb02d014a69b8f5a31b6ac025266ed03ac6ff37a44dc568ad50decea9f6  train_ppo_real.py
a9c787d1ca02463e8c7df7ec35cc55e1a7a31e5ce93eedf2184b8bda58623dcd  wm_gym.py
ee369385583d7cb3eab49e0899fc791632608c51a709c83df45844ba9017bc63  wm_env2.py
38e333992872625a5a1883116ca741e9b2e6cba7761dd0b2c0f50675d2af9e43  craftax_env.py
bcf331b94bea433cea2d287fd7ab47fc5f4b29a957530cc408c245bc9a3ad47b  evaluate.py
```
