# Craftax evidence and protocol audit

Audit date: 2026-09-19. Read-only audit of the current checkout at `/Users/rldoeswork/Documents/rl/tinycraft-jepa`; source git HEAD `1fa2853b5f365a8bdd6d732eebfa851e0197400b`, with modified source files and untracked experiment artifacts. No models trained in this audit. A fresh evaluation was subsequently authorized; its measured results are tracked separately below. Full raw JSON and SHA-256 manifest: `craftax_evidence.json`.

## Verdict

Craftax can responsibly appear as **exploratory motivation or an appendix transfer study**: one fitted joint-embedding model supports short-horizon goal reaching but the tested post-hoc-head reward-MPC pipeline performs poorly. It does **not** isolate a causal effect of stochastic latent prediction, establish representation insufficiency, or demonstrate general Craftax performance. The goal test is open loop, despite its function/flag names. New results should not be merged with the old grid results under one unnamed model.

## Fresh frozen-model follow-up: completed after this audit

The independent implementation `studies/revision/craftax_eval.py` completed 384 new starts (evaluation seeds 101, 202, 303; 128 each) on the verified native 512-dimensional seed-0 checkpoint. H=6, CEM population 512, elite count 32, six iterations. Goal generation and execution used independent simulator randomness; compared policies shared starting states and execution keys. Total evaluation time recorded by the script was 796.6 seconds. No weights, latents, actions or rewards were modified.

| Policy | All nearby goals, n=384 | Displaced goals only, n=247 | Displacement ≥2, n=99 | Mean endpoint L1 |
|---|---:|---:|---:|---:|
| No-op | 137/384 = 35.7% | 0/247 = 0.0% | 0/99 = 0.0% | 0.940 |
| Uniform random | 65/384 = 16.9% | 19/247 = 7.7% | 1/99 = 1.0% | 1.560 |
| JEPA, one fixed plan | 358/384 = 93.2% | 222/247 = 89.9% | 80/99 = 80.8% | 0.107 |
| JEPA, replan after each observation | 369/384 = 96.1% | 232/247 = 93.9% | 87/99 = 87.9% | 0.078 |
| Goal-generating sequence replayed | 382/384 = 99.5% | 245/247 = 99.2% | 97/99 = 98.0% | 0.005 |

Success requires being alive at the target position. All warmup starts and generated targets were alive. Goal displacement counts were 137 at distance 0, 148 at distance 1, 84 at distance 2 and 15 at distance 3; mean 0.940 tiles. This is explicitly **nearby goal control**, not long-range Craftax skill acquisition. The reference sequence is not an oracle under independent execution noise. Receding control uses additional planner calls, so the one-shot/receding comparison is not compute matched.

The all-goal 95% Wilson intervals are 90.26–95.34% for one-shot JEPA and 93.66–97.62% for receding JEPA. These describe independent episode evaluation uncertainty for **one trained model**, not training-seed robustness. Per-evaluation-seed success counts were 119/119/120 for one-shot and 123/121/125 for receding control, each out of 128.

Raw complete evidence: `artifacts/revision/craftax_goal_v2.json`, SHA-256 `44dc765a690037e8cc1b618be01f1348f4f8a54ad02c1d9424a230e02a13f867`; execution log: `artifacts/revision/craftax_goal_v2.log`. The report contains full checkpoint metadata, code/package hashes, all arguments, per-episode outcomes and raw executed actions/positions. The two-episode smoke and `.partial.json` copy are validation artifacts, not extra scientific samples. No fresh reward evaluation was run; reward-control numbers below remain the original seed-0 pipeline measurements.

Recommended current paper sentence: “On nearby Craftax-Classic goals, the same frozen 512-dimensional joint-embedding model achieved 93.2% success with a fixed six-action plan and 96.1% with replanning, versus 35.7% for no-op and 16.9% for random actions (384 fresh starts, one trained model). On the 247 goals requiring displacement, success remained 89.9% and 93.9%. This positive control limits the claim: the tested representation supports local goal reaching, while the earlier reward-control failure remains confounded by post-hoc head generalization.”

## New 512-dimensional model: exact measured results

Source: `runs/stoch/craftax/tile_factored_plan.json`; training and probe logs in the same directory.

| Goal reaching, H=6, 128 fresh starts | Reach | Endpoint position L1 | Pixel MSE (0–255 scale) |
|---|---:|---:|---:|
| Planned | 121/128 = 0.9453125 | 0.078125 | 104.001724 |
| Random | 24/128 = 0.1875 | 1.53125 | 1477.915039 |
| Generating action sequence, same randomness | 128/128 = 1.0 | 0 | 0 |

| Reward policy, 128 starts × 100 steps | Mean return | Return SD across starts | Alive at end |
|---|---:|---:|---:|
| Uniform random | 1.4453125 | 1.0897146 | 0.6328125 |
| Myopic reward argmax | 0.78125 | 1.1108661 | 0.7265625 |
| One predicted step + value | 0.9437500 | 1.2013502 | 0.6328125 |
| CEM, H=6 + terminal value | 0.4734375 | 1.1038629 | 0.671875 |

The logged data-policy number 4.0740397 is **mean reward in the stored dataset multiplied by 100**, not a matched online policy return. Exclude it from an online performance comparison.

Heads fit the first 80,000 frame/action transitions (320 time rows × 250 streams). Reward/value train correlations are 0.921984/0.994872; held-out-time correlations are 0.352734/0.266465. Held-out reward RMSE is 0.222052. Held-out value prediction mean 2.356160 exceeds empirical discounted-return mean 1.347605. These held-out frames are excluded from head fitting, but not from encoder training. Value fitting uses only the first 80 time rows because of truncated-return handling. The strong fit/generalization gap is a substantial alternative explanation for failed reward MPC.

In offline action recovery on 256 evaluation windows, first-action accuracy is 0.203125; the quantity labeled sequence accuracy is 0.220703125. The latter is **mean token/action accuracy**, not probability that the whole action sequence is correct. Mean CEM / recorded-action / random costs are 0.119346 / 0.278756 / 0.881070. About 0.169060 of random candidate sequences score better than the recorded sequence, but alternative actions can reach the same goal, so this is not by itself an action-ranking failure.

## Model identity and budget

The executed recipe is documented in `studies/stoch/lane_b.sh:12`, `studies/stoch/train_arms.sh:22`, and `studies/stoch/lane_b2.sh:22`:

- Training seed 0; tile encoder; 16 tokens × 32 dimensions = **512-dimensional latent**; factored predictor, width 256, depth 6, history 8; 17 actions; 8500 updates; batch 128; AdamW learning rate 0.001; default MSE + 0.09 SIGReg.
- Training log: 11.12M parameters, frames `(2001,250,63,63,3)`, 479970 valid short training windows. Data path `~/personal/tinywmcraft-e2e/starter/data/tinywmcraft_500k/train.npz`.
- Logged training elapsed time is **604.5 s**, including loading and compilation. This does not satisfy a strict ≤600 s claim. Final update's loop time is 578.5 s. Save/hash overhead follows the main timer definition (`train.py:470-492`).
- Checkpoint paths are `/dev/shm/stoch/craftax/tile_factored/model.npz` and `model_probe.npz`. Live read-only verification on e2e confirmed `model_probe.npz` SHA-256 `59cc282c9bd7a49ab1c272342220bc8e0d8ca0f1aa77bbcb4c23ae31dfd6835d`. Its stored metadata confirms seed 0, the architecture above, 11,121,556 parameters, `train_s=604.5`, `loop_s=579.7`, training-data SHA-256 `09f2cb41a38b0f34937a89c4122f6b25bbb3094e34a077b3a572504269d0a1b2`, and training-source SHA-256 `72cd5c463b4ac2fe7e289f7032e91d88ffac2e59ebb15534524fafe4804e9d0b`. Current training-data bytes were independently rehashed and match the checkpoint. The current evaluation archive SHA-256 is `caec5e6a0e73b432740b1a1d40ce6de19ebf83a4bfc3540436d38025386661e7`.
- Pixel decoder and reward/done/value heads are post-hoc, encoder-frozen, outside the world-model budget. Probe log: decoder plus embedding 113.4 s; reward-head phase 62.0 s; total probe phase 175.6 s.
- Main evaluation script uses default environment/planner seed 0, 128 starts, horizon 6, CEM population 512, elites 32, six iterations. Only one trained-model seed exists in the supplied Craftax tile results.

## Protocol corrections that must reach the paper

1. **Goal reaching is open loop.** `plan.py:155-207` computes one action sequence at line 190 and executes all H actions at lines 197–198. There is no replanning/observation feedback during those H actions. Label it “short-horizon open-loop goal reaching”; reserve closed loop / MPC for `reward_loop`.
2. **The goal distribution is intentionally easy and reachable.** Reset, seven random warmup actions, then H random actions produce a goal image. Success checks final player position only (`plan.py:179-204`). Goals are not arbitrary distant objectives. No-op/stay, displacement-stratified performance and an independent execution-randomness variant would sharpen the claim. No privileged simulator coordinates or target action sequence enter the learned planner's cost.
3. **Goal execution uses common simulator random keys.** The goal-generating rollout, model plan, random baseline and replayed target actions share exact future noise (`plan.py:181,184,195-198`). The replay “oracle” is therefore a protocol check, not a demanding planning baseline. Use independent execution randomness as an additional robustness result, not silently as a replacement.
4. **Reward MPC is genuinely closed loop.** It executes the first selected action then re-encodes the actual frame (`plan.py:245-276`). Reward is accumulated only while alive, no autoreset, common initial states and random keys. This is a short 100/150-step evaluation, not standard full-episode Craftax achievement score.
5. **Post-hoc head failure prevents a clean representation/dynamics attribution.** Reward argmax without dynamics already underperforms uniform random, and held-out head generalization is weak. Claim “the tested reward-MPC pipeline fails,” not “the latent cannot encode reward” or “mean dynamics cause this return gap.”
6. **Imagined-versus-realized reward is not fixed-plan calibration.** Predicted H-step CEM reward at time t is compared with actual later MPC rewards while the controller replans every step (`plan.py:267-291`). Actual later actions can differ from the originally scored plan. Any calibration figure must state this or rerun fixed plans on cloned states.
7. **Logged behaviour policy is not an online control baseline.** `plan.py:334-335` simply takes archive rewards.mean × steps.
8. **Offline action labels are not unique optimal controls.** Action recovery is a diagnostic; multiple actions are inert or lead to the same position. `sequence_accuracy` is misnamed per-action accuracy (`plan.py:148`).
9. **One-model uncertainty must remain visible.** The 128/256 fresh starts quantify evaluation variation, not training-seed uncertainty. Only aggregate return SDs are saved, so paired confidence intervals cannot be reconstructed even though common starts/noise were used. Do not fabricate paired statistical significance.
10. **gFVD is a custom proxy.** `eval.py:88-128` applies a fixed randomly initialized CNN to overlapping four-frame clips and computes a Fréchet moment distance. It is not standard pretrained-video FVD and is not pixel MSE. The action-following score compares salient pixel error under recorded versus random actions (`eval.py:131-140,169-193`), not a control metric.

## Older Craftax results: retain separate identities

Prediction scores in `ledger.jsonl`: one training seed 0; evaluator seed 0; 2048 held-out windows; eight context frames and 16 forecast steps.

| Model | Latent | Train seconds | Custom gFVD ↓ | Action following ↑ |
|---|---|---:|---:|---:|
| LeWM-like ViT/AR, `s0_lr1e-3` | 192-d class token | 574.2 | 0.037073782 | 0.67285156 |
| LeWM spatial ViT/AR, `s0_grid` | 81×32 = 2592 | 588.6 | 0.030976080 | 0.84594727 |
| Last-frame repetition | n/a | 0 | 0.013330484 | 0.5 |
| Flow, `normuon_head_bridge_s0` | architecture differs | 566.6 | 0.001245920 | 0.92016602 |

The old 2592-d grid is **not** the new 512-d `tile_factored` checkpoint. Comparing these as the same “grid JEPA” would be misleading. The user's current restriction against inflated latents argues for moving old 2592-d results to clearly marked historical/exploratory context, if retained at all.

Flow provenance is preserved in `output/research-review/evidence.json:craftax_flow_ledger`: recipe commit `51ffe8dfc1eaa1c659cb33793ec4644852220ab4`, train source SHA `0cee51a6007fe8b3e732e9cfedf9c241693e2caf5df194da91b78c2e9822f9f9`, seeds 0/2/3. Custom gFVD values are 0.001245920, 0.001271970, 0.001163737; mean 0.001227209, sample SD 0.000056491. AF mean 0.928466797, sample SD 0.007389036. These are archived ledger measurements, not re-evaluated in this audit. The flow architecture, reconstruction objective, reward/done supervision and training procedure differ, so this is a system comparison, not an isolated objective ablation. The seed-0 checkpoint was selected through a substantial earlier recipe campaign; report seed-confirmation versus recipe selection honestly.

Older goal results (256 starts): class token reaches 96/256 at H4 and 56/256 at H8; old grid reaches 218/256 at H4 and 172/256 at H8; common random baselines 65/256 and 33/256. All are the same open-loop protocol.

Older grid reward MPC uses 128 starts × **150 steps**, not 100: random 1.485938, myopic 0.919531, one-step/value 1.423438, CEM 1.203906. Five-member ensemble (λ=0) returns 1.131250 / 1.232813 / 1.318750 for the three learned policies; λ=1 returns 1.149219 / 1.229688 / 1.071094. These alternatives do not show an improvement over the random mean; no paired significance can be inferred from available summaries. The old single-head fit used 250k transitions versus 80k in the new run.

## Recommended manuscript wording

“An exploratory Craftax-Classic experiment separates local reachability from reward control. A 512-dimensional joint-embedding model reached 121 of 128 nearby goal positions in a six-action open-loop test, compared with 24 for random actions. However, a controller using separately fitted reward and value heads obtained mean 100-step return 0.47, versus 1.45 for uniform random actions. The reward head's correlation fell from 0.92 on its fitting data to 0.35 on held-out time rows. This experiment establishes a limitation of the tested control pipeline; it does not identify whether representation learning, head generalization, dynamics or planning is the dominant cause.”

The most useful additions are exact checkpoint/data/source manifests, several independent evaluation-seed batches preserving per-episode outcomes, no-op and displacement-stratified goal results, independent-noise execution, and direct fixed-plan rollout calibration. A matched flow-control baseline with equal head training data is needed before any comparative reward-control conclusion.

## Fresh evaluation implementation for the revision

`studies/revision/craftax_eval.py` evaluates the frozen native checkpoint without training or latent/reward transformations. This is an additional, separately identified protocol. It deliberately does not overwrite or reinterpret the legacy matched-randomness experiment.

The fresh goal test retains random reachable-image targets, then executes under **independent simulator randomness**. It compares no-op, uniform random, the goal-generating action sequence replayed under fresh noise, one-shot JEPA planning, and JEPA replanning after every true observation. The replay is called a reference sequence, not an oracle. Receding control uses H−t remaining steps. All controllers receive common initial states and common execution randomness, and use the same native action set. CEM population, elites and iteration count are fixed; repeated replanning has greater total planning cost and must be described as such.

The primary success field requires reaching the target position while alive. Position-only success is also preserved for legacy comparability. Report all goals, stationary goals, displaced goals and displacement at least two separately; these strata are defined before seeing results. No episodes are adaptively selected. The code records deaths during warmup, goal generation and execution.

Optional reward evaluation compares no-op, uniform random, direct reward argmax, one-step/value and H-step MPC. Rewards come directly from the environment and are masked after first termination. Each policy starts from the same warmup state and receives identical per-step simulator keys. No dataset-derived behaviour-policy number or unmatched fixed-plan calibration statistic is emitted.

Every output contains checkpoint SHA-256, checkpoint metadata, source SHA-256, package versions, device identity, all arguments, evaluation seed, per-episode returns/successes, and action/outcome trajectories. It reports episode-level Wilson intervals and paired mean differences; these quantify evaluation uncertainty of a single fitted checkpoint, **not training-seed uncertainty**. Fresh seeds are 101, 202 and 303 by default, independent of legacy seed 0. The script writes completed results atomically and refuses an existing output path.

Suggested smoke check in the existing GPU environment:

```sh
python studies/revision/craftax_eval.py \
  --model /dev/shm/stoch/craftax/tile_factored/model_probe.npz \
  --out artifacts/revision/craftax_smoke.json \
  --seeds 999 --envs 2 --horizon 2 --pop 8 --elite 2 --iters 1
```

Then freeze parameters and evaluate the goal protocol:

```sh
python studies/revision/craftax_eval.py \
  --model /dev/shm/stoch/craftax/tile_factored/model_probe.npz \
  --out artifacts/revision/craftax_goal_v2.json \
  --seeds 101 202 303 --envs 128 --horizon 6
```

Only if the control-pipeline distinction will remain central, add the longer reward run:

```sh
python studies/revision/craftax_eval.py \
  --model /dev/shm/stoch/craftax/tile_factored/model_probe.npz \
  --out artifacts/revision/craftax_reward_v2.json \
  --seeds 101 202 303 --envs 128 --horizon 6 --steps 100 --reward --no-goal
```

Validation completed locally: Python compilation, command-line help without JAX imports, exact success aggregation, empty strata, Wilson interval containment and fixed displacement strata. The goal-evaluation smoke and the complete three-seed goal run subsequently passed on e2e GPU0. The reward branch has not been GPU-tested by this subaudit. The harness currently supports the existing JEPA checkpoint API; it does not claim a matched flow baseline.

## Placement and visual recommendation

Lead the paper with the controlled mechanism and its measured control consequences. Craftax works as a one-paragraph motivating observation or an appendix result, with the task-confound sentence beside the numbers. Opening with a visually rich Craftax scene is defensible only when its caption says **exploratory, one fitted model, distinct tasks**. It should not become the first quantitative proof of the claimed mechanism.

Use one compact two-panel figure/table: blue for goal success (JEPA .945 vs random .188) and ochre for reward return (JEPA MPC .473 vs random 1.445), with clear H=6 versus 100-step labels. The head generalization gap should sit directly under the reward panel. Do not put a custom-gFVD ratio beside returns under a single quality axis; they measure different properties. Avoid the inflated historical 2592-dimensional spatial latent in the main result because the current 512-dimensional model is the relevant native recipe.

## Publication data and real environment illustrations

`research/craftax_table_v2.json` is the compact figure/table input. It includes all/displaced/≥2 success counts, episode-level Wilson 95% intervals, per-evaluation-seed counts, and paired bootstrap percentile intervals (10,000 resamples; RNG seed 19092026). For example, the one-shot JEPA minus no-op success difference is 57.55 percentage points (95% paired bootstrap 52.60–62.50); receding minus one-shot is 2.86 points (0.78–4.95), with the extra planning compute explicitly acknowledged.

Actual environment PNGs are in `artifacts/revision/craftax_frames_v2/`. CPU-only regeneration selected the first three seed-101 episode indices with target displacement at least two (episodes 5, 9, 10), without consulting model outcomes. The script verified all regenerated start/goal coordinates and reference actions against the completed quantitative evaluation. `metadata.json` records selection, episode states, native 63×63 shape, CPU device and every PNG hash. All three start/goal pairs were visually inspected. They are genuine egocentric Craftax renders; use nearest-neighbor display scaling and label them as start/goal illustrations, not model predictions or additional evaluation samples. V1 source/results have not been rewritten by this subtask.
