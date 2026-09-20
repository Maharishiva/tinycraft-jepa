#!/bin/bash
# Lane B, part 3 (e2e, after lane_b2.sh): PPO inside the JEPA (control cell: same policy class as the flow
# arm's latent PPO, mean predictor + off-budget heads instead of a sampled model).
#   nohup bash studies/stoch/lane_b3.sh > runs/stoch/lane_b3.log 2>&1 &
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-e2e/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=runs/stoch
until grep -q LANE_B2_DONE $LOG/lane_b2.log; do sleep 60; done
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
jepa() { run $LOG/$1/ppo_jepa_$2 $PY -m studies.stoch.rl --config $1 --arm jepa --wm $3/$2/model_probe.npz \
             --data $4 --out $LOG/$1/ppo_jepa_$2; }
jepa gate tile_factored /dev/shm/stoch/gate /dev/shm/stoch/gate/train.npz
jepa gate lewm6 /dev/shm/stoch/gate /dev/shm/stoch/gate/train.npz
echo "GATE_JEPA_PPO_DONE $(date -u +%H:%M:%S)"
echo LANE_B3_DONE
