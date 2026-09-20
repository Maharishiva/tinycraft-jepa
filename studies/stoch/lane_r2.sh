#!/bin/bash
# Round 2, lane R2 (multigpu GPU 1): the frozen-latent control on model seeds 1 and 2 (three policy seeds each).
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.45
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for MS in 1 2; do
for PS in 0 1 2; do
    NAME=$LOG/ppo_real_jepa_latent_m${MS}_p$PS
    run $NAME $PY -m studies.stoch.rl --config gate --arm real --obs latent --wm $G/tile_factored_s$MS/model_probe.npz --seed $PS --max-minutes 30 --out $NAME
    echo "DONE real latent m$MS p$PS $(date -u +%H:%M:%S)"
done
done
echo LANE_R2_DONE
