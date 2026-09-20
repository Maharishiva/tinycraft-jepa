#!/bin/bash
# Round 2, first control cells (multigpu-vm, GPU 0): the state-blind baseline row, then PPO inside the grid JEPA with
# the residual head in MEAN-OF-8 mode (the head's own conditional mean: same accuracy gain as the sampler, no sample).
#   (setsid nohup bash studies/stoch/lane_e1.sh > runs/stoch/lane_e1.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.45
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
run $LOG/baselines_r2 $PY -m studies.stoch.rl --config gate --arm baselines --out $LOG
run $LOG/ppo_jepa_tile_factored_mean8 $PY -m studies.stoch.rl --config gate --arm jepa --wm $G/tile_factored/model_probe.npz \
    --sampler $G/tile_factored/sampler.npz --sampler-mean 8 --data $G/train.npz --out $LOG/ppo_jepa_tile_factored_mean8
echo "LANE_E1_DONE $(date -u +%H:%M:%S)"
