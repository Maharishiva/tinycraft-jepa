# studies/stoch: JEPA vs flow-matching world model in stochastic-control worlds

Survival is the task (-1 on touching the border or the wall, nothing else). Three 10-minute models per world
(`lewm6` = LeWM as published, shrunk; `tile_factored` = grid JEPA; `flow` = the tinywmcraft recipe with the
reward head fixed to the transition form), two paradigms (MPC on the JEPA latent; PPO inside the flow model),
plus the control cell PPO inside the JEPA. The gate world adds a multimodal event the agent must wait to observe.

| file | what |
|---|---|
| `env.py` | the worlds: the six cube worlds of `studies/redcube` plus the gate (`--config gate`) |
| `flow_recipe/` | the flow model (`train.py` patched: reward head predicts r_{t+1} from (s_t, a_{t+1})) |
| `train_arms.sh` | JEPA arms (`lewm6`, `tile_factored`) + off-budget decoder and heads |
| `control.py` | JEPA controllers: goal-image MPC, heads MPC, greedy; random and centre references; tree search over 4^h plans |
| `rl.py` | PPO inside the flow model (`--arm dream --obs pixels|latent`), inside the JEPA (`--arm jepa`), in the real env, baselines |
| `sampler.py` | off-budget flow-matching head on the JEPA predictor's residual: the same encoder and heads with a sampled next latent (`--sampler` in `control.py`, `rl.py`, `wallmass.py`, `predictions.py`) |
| `predictions.py` | the half-open-wall figure: real futures vs each model's rollout from the same pre-reveal context |
| `figures.py`, `summarize.py` | CPU figures and the result tables (`runs/stoch/summary.json`, `paper/tables/*.tex`) |
| `lane_*.sh` | the overnight queues (A = multigpu-vm: flow + PPO; B = e2e: JEPA arms, controls, Craftax; C = e2e: the sampling-predictor cell) |
| `paper/`, `article/` | the ICLR 2027 paper (LaTeX, `tectonic main.tex`) and the Russian blog article |

Data and checkpoints: `multigpu-vm:~/personal/stoch/<world>/` (gate data, all flow models),
`e2e:/dev/shm/stoch/{gate,craftax,redcube/<world>}/<arm>/model_probe.npz` (JEPA arms); the six cube worlds'
data and original JEPA arms stay read-only under `multigpu-vm:/dev/shm/redcube/`.
