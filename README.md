# tinycraft-jepa

Two things live here.

1. **A JAX port of LeWorldModel** (JEPA world model: next-embedding MSE + SIGReg, no decoder in the
   loss) trained on Craftax-Classic pixels under a 10-minute / 500k-transition budget, plus the
   off-budget analysis tools. This is the original project; its results are kept in
   [Part 2](#part-2-the-lewm-port) below.
2. **The paper "Learning to Ignore What Matters: JEPAs in Stochastic Worlds"** and every experiment
   behind it: minimal red-cube worlds with controllable nuisance and stochastic dynamics, a gate
   world with a late-revealed opening, a flow-matching reference model trained on the same data and
   budget, and Craftax under matched budgets. Everything for the paper is under `studies/` and
   `submission/`.

If you are here to check whether the numbers can be trusted, start with [AUDIT.md](AUDIT.md). If
you are here to rerun something, start with [Part 1](#part-1-reproducing-the-paper).

## Map

| path | what |
|---|---|
| `train.py`, `probe.py`, `viz.py`, `plan.py`, `eval.py` | the LeWM port: model, objective, budgeted training, decoder probe, diagnostics, CEM planning, frozen benchmark |
| `studies/redcube/` | the six cube worlds (`env.py`), flow + JEPA training per world (`train_config.sh`, `train_arms.sh`), position read-outs (`diag_jepa.py`, `bench.py`) |
| `studies/stoch/` | the gate world and the stochastic study: `env.py`, `rl.py` (PPO real / dream / inside-JEPA), `control.py`, `sampler.py` (residual sampling head), `summarize.py`, `flow_recipe/` (the flow model) |
| `studies/revision/` | the revision harness behind every number in the paper's main text; see the table in Part 1 |
| `research/` | protocols, audits, decision logs and evidence ledgers written by the coding agents during the revision; supporting material, not the paper |
| `submission/v1`, `submission/v2` | frozen earlier drafts; `submission/v3` is the current manuscript (`bash build.sh`, needs Tectonic) |
| `runs/`, `artifacts/`, `output/` | results: run ledgers and `final.json` files, exported figures and packets, analysis outputs with hashes |
| `NARRATIVE.md` | the claim the paper is written to serve |
| `AUDIT.md` | harness audit: what each safeguard is, where it lives, what it does not cover |

## Part 1: reproducing the paper

### Environment

Python 3.11, JAX 0.10.1, Flax 0.12.7, Optax 0.2.8, NumPy 2.4, Craftax 1.5.0, one A100-40GB.

```bash
uv venv --python 3.11 && uv pip install -r pyproject.toml --extra cuda && source .venv/bin/activate
```

The shell scripts take the interpreter from `PY` (default `~/personal/tinywmcraft-slim/.venv/bin/python`,
the VM the study ran on) and the GPU from `GPU`. Python entry points take paths as flags. The remaining
machine-specific defaults are listed at the end of this section.

### Data

| world | how to make it | size used in the paper |
|---|---|---|
| cube worlds | `python -m studies.redcube.env --config <world> --out <dir> --train-steps 5000` | 100 envs x 5000 steps = 500k transitions (gray worlds); 100k for black/color |
| gate | `python -m studies.stoch.env --config gate --out <dir> --train-steps 5000` | 500k transitions, seed 0, uniform random policy |
| Craftax | `python eval.py --download` (tinywmcraft_500k from Hugging Face) | 500k transitions |

Worlds: `gray_frame_still`, `gray_frame_wind`, `gray_episode_still`, `gray_episode_wind`, plus the
`black_*` and `color_*` variants; `gate`. `studies/redcube/env.py --preview` renders a strip.

Checkpoints used in the paper are identified by SHA-256 in `research/gpu_inventory.md`; the VM
copies have no git history, so the hash is the identity.

### Models

```bash
# JEPA (grid latent 4x4x32, factored predictor, 8500 updates) + off-budget decoder and heads
ROOT=<dir with train.npz> GPU=0 FIXED_STEPS=8500 bash studies/stoch/train_arms.sh gate tile_factored
# flow reference (same data, same budget)
(cd studies/stoch/flow_recipe && python train.py --data <dir>/train.npz --out <dir>/flow/model.npz --seed 0)
# both for one cube world, data included
GPU=1 TRAIN_STEPS=5000 bash studies/redcube/train_config.sh gray_frame_wind
```

### Experiments in the main text

| section | script(s) | notes |
|---|---|---|
| cube position read-outs (Table 1) | `studies/redcube/diag_jepa.py`, `studies/redcube/bench.py`; table `research/redcube_table.md` | three model seeds per gray world |
| detached-target intervention | `studies/revision/target_gradient.py`, `run_target_gradient_queue.py`, `analyze_target_gradient.py` | protocol `research/target_gradient_protocol.md`; result `output/revision/target_gradient_strict_analysis_20260920/` |
| filmstrips (real, decoded-real, predicted, flow) | `studies/revision/export_redcube_story.py` | packets and manifests in `artifacts/revision/redcube_story/<world>/` |
| matched latent-MLP PPO (appendix) | `studies/revision/matched_ppo.py`, `run_matched_queue.py`, `analyze_matched.py` | exact-simulator arms require byte-exact re-rendering first |
| shared-CNN comparison (gate + cube worlds) | `studies/revision/prep_shared_decoded.sh` -> `queue_shared_decoded.py` -> `analyze_shared_decoded.py` | protocol `research/shared_decoded_cnn_protocol.md`; budgets are enforced by `shared_decoded_ppo.py` |
| Craftax inventory | `studies/revision/run_inventory_queue.py` (wraps `inventory_ppo.py` and `inventory_eval.py`) | protocol `research/craftax_native_inventory_protocol.md`; evaluation contract frozen at 256 episodes, seed 123 |
| historical gate study (appendix) | `studies/stoch/rl.py`, tables from `studies/stoch/summarize.py` | use the last ledger row, not `final.json` (see AUDIT.md) |

Every training script in `studies/revision/` refuses budgets other than the registered ones unless
`--smoke` is passed, writes `config.json` with model, data and source hashes, and evaluates only the
final checkpoint.

### Tests

CPU only, no data needed (commands and coverage in `AUDIT.md`, section 7):

```bash
JAX_PLATFORMS=cpu python -m unittest studies.revision.test_inventory_eval studies.revision.test_warm_context_eval studies.revision.test_inventory_jepa studies.revision.test_inventory_queue
JAX_PLATFORMS=cpu python -m pytest tests/ studies/revision/test_target_gradient.py
```

### Machine-specific defaults

These are defaults, all overridable:

- `studies/revision/queue_shared_decoded.py`: `--gray-root` (`/dev/shm/redcube`), `--stoch-root`
  (`~/personal/stoch`), `--python`.
- `studies/revision/prep_shared_decoded.sh`: `PY`, `STOCH`, `GRAY`, `OUT` environment variables.
- `studies/redcube/train_config.sh`, `studies/stoch/train_arms.sh`: `ROOT`, `DATA`, `PY`, `GPU`,
  `FLOW`, and a `cd ~/personal/tinycraft-jepa` that assumes the VM layout.
- `studies/stoch/lane_*.sh`, `studies/redcube/lane.sh`: one-off overnight queues from the original
  study, kept as a record of what was run; not maintained entry points.

## Part 2: the LeWM port

**LeWorldModel (LeWM) on Craftax-Classic pixels, in JAX.** A joint-embedding predictive world
model trained under exactly the budget our generative pixel world model uses, 500k transitions and
10 minutes on one A100, and then taken apart off-budget: what does a JEPA rollout actually contain,
does it blur, mean-regress or go noisy, and can you plan in it?

Port of [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) (Maes\*, Le Lidec\*, Scieur,
LeCun, Balestriero, *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture
from Pixels*, arXiv:2603.19312) from PyTorch to JAX / Flax NNX. **The objective is unchanged**:

```
loss = mean((predict(z_t, a_t) - z_{t+1})^2) + 0.09 * SIGReg(z)
```

No stop-gradient on the target, no EMA teacher, no pretrained encoder, **no pixel decoder in the
loss**, no auxiliary supervision. Collapse is held off by SIGReg alone, which is checked numerically
against a transcription of the reference implementation in [tests/test_port.py](tests/test_port.py).

![what the latent keeps and how the rollout drifts](artifacts/comparison.png)

### Quickstart

```bash
python eval.py --download
python train.py --out runs/lewm/model.npz --lr 1e-3 --steps 12250     # the 10-minute budget
python viz.py  --model runs/lewm/model.npz --tag lewm --windows 2048  # latent diagnostics
python probe.py --model runs/lewm/model.npz                           # off-budget pixel decoder
python eval.py --model runs/lewm/model_decoder.npz --tag lewm --gif   # frozen benchmark
python plan.py --model runs/lewm/model.npz --tag lewm --closed-loop   # CEM planning in Craftax
```

`run_all.sh` chains all of it. Only `train.py` is inside the budget; everything after it is
analysis on a frozen model.

### Model

| part | what it is |
|---|---|
| encoder | ViT-tiny (192 wide, 12 layers), 7x7 patches over the 63x63 frame = 9x9 = 81 patches |
| read-out | class token (reference) or all 81 patch tokens (`--tokens 81`, "grid latent") |
| projector | Linear-BatchNorm-GELU-Linear, as in the reference |
| action encoder | one-hot(17) through the reference `Embedder` (bottleneck 10, SiLU MLP) |
| predictor | 6 causal AdaLN-zero blocks over the last 3 latents, conditioned on the action |
| regularizer | SIGReg: 17 knots, 1024 random projections, weight 0.09 |

17.9M parameters with the class-token read-out, 36.8M with the grid latent. The paper's grid JEPA
(`--encoder tile`, 4x4x32 latent, factored predictor) is a smaller variant defined in
`studies/stoch/train_arms.sh`.

### Results of the port

Single seed per configuration; 2048 held-out windows (8 context frames, 16 imagined steps).
Wall-clock includes loading the 500k dataset and XLA compilation.

| run | latent | lr | wall | R² step 1 | R² step 16 | R² step 16, copy-last | spread ratio step 16 | action id top-1 | effective rank |
|---|---|---|---|---|---|---|---|---|---|
| `s0_lr5e-5` (paper lr) | 192 (cls) | 5e-5 | 574.9 s | 0.973 | 0.393 | 0.043 | 0.96 | 0.195 | 36 / 192 |
| `s0_lr3e-4` | 192 (cls) | 3e-4 | 576.0 s | 0.987 | 0.550 | 0.074 | 1.02 | 0.238 | 48 / 192 |
| `s0_lr1e-3` | 192 (cls) | 1e-3 | 574.2 s | 0.988 | **0.552** | 0.123 | 1.01 | 0.251 | 51 / 192 |
| `s0_grid` | 81x32 = 2592 | 1e-3 | 588.6 s | 0.987 | 0.141 | -0.276 | 0.94 | **0.559** | 64 / 2592 |

R² is measured in each model's own latent space against the variance of the true future latents,
so it compares a model to *itself*, not across rows; "copy-last" is the same number for a rollout
that repeats the last context latent. Action id top-1 is 17-way action recovery from one observed
transition (chance 0.059).

**Does it collapse?** No. Class-token model: per-dimension std 1.03, kurtosis 2.99, standard-normal
marginals as claimed, but effective rank 51 of 192. The 2592-d grid latent is further off: kurtosis
16.1, effective rank 64.

**Does the rollout blur, mean-regress or go noisy?** None of the three.

| | gFVD ↓ | AF ↑ | salient MSE ↓ | decoder MSE |
|---|---|---|---|---|
| copy the last frame | 0.01333 | 0.500 | 0.345 | – |
| `s0_lr1e-3` rollout | 0.03707 | 0.673 | 0.192 | 0.00695 |
| `s0_lr1e-3` decoder ceiling | 0.02590 | – | 0.139 | |
| `s0_grid` rollout | 0.03098 | 0.845 | 0.173 | 0.00043 |
| `s0_grid` decoder ceiling | **0.00160** | – | 0.016 | |
| tinywmcraft generative pixel WM | 0.00123 | 0.928 | – | – |

The *ceiling* rows decode the encoder's latents of the **real** future frames. For the class-token
model the ceiling alone costs gFVD 0.0259, i.e. 70% of its pixel error is the 192-d bottleneck, not
the prediction. A spatial latent drops the ceiling 16x. Per-step diagnostics
([drift plot](artifacts/s0_lr1e-3_drift.png), [grid version](artifacts/s0_grid_drift.png)): batch
spread stays at 1.01 / 0.94 of the true spread through 16 steps, high-frequency energy barely
decays, rollouts stay in-distribution.

![ground truth beside the imagined rollout](artifacts/s0_grid_demo.gif)

**Can we plan in it?** With the grid latent, yes at short horizons: CEM over LeWM's own latent-goal
criterion reaches the exact goal tile in 85% of 256 closed-loop Craftax episodes at h=4 (random
25%); the class-token model barely beats random (37.5%). Both share one defect: the CEM plan always
reaches a lower model cost than the true action sequence. Per-run detail in `artifacts/*_reward.json`
and `ledger.jsonl`.

### Deviations from the reference implementation

1. 63x63 frames with patch size 7 (81 patches) instead of 224x224 with patch size 14 (256).
2. Discrete actions: one-hot(17) into the reference `Embedder`.
3. One warmup+cosine cycle over the step budget instead of 100 epochs at 5e-5; the learning rate is
   swept. The SIGReg weight stays at 0.09.
4. Attention-matrix dropout is dropped (`jax.nn.dot_product_attention` has no dropout argument).
5. The grid read-out uses the input/condition/output projections the reference `Transformer`
   already provides for a latent wider than the predictor.

### Citation

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2603.19312},
  year={2026}
}
```

Data: [tinywmcraft_500k](https://huggingface.co/datasets/maharishiva/tinywmcraft_500k).
Scaffold and frozen evaluator: [tinywmcraft-task-starter](https://github.com/Maharishiva/tinywmcraft-task-starter).
Generative baseline: [tinywmcraft](https://github.com/Maharishiva/tinywmcraft).
