#!/bin/bash
# Lane A (multigpu-vm, GPU 0): the flow-matching world model with the corrected death reward, on the gate world
# and re-trained on the six red-cube worlds (their data stays read-only under /dev/shm/redcube), each followed by
# the recipe's own evaluation (gFVD, action faithfulness, reward / done head metrics).
#   nohup bash studies/stoch/lane_a.sh > runs/stoch/lane_a.log 2>&1 &
set -e
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
OUT=$HOME/personal/stoch
LOG=runs/stoch
FLOW=$PWD/studies/stoch/flow_recipe
train_flow() {
    CFG=$1
    DATA=$2
    mkdir -p $OUT/$CFG/flow $LOG/$CFG
    [ -f $OUT/$CFG/flow/model.npz ] || (cd $FLOW && $PY train.py --data $DATA/train.npz --out $OUT/$CFG/flow/model.npz --seed 0) > $LOG/$CFG/train_flow.log 2>&1
    [ -f $LOG/$CFG/eval_flow.jsonl ] || (cd $FLOW && $PY eval.py --model $OUT/$CFG/flow/model.npz --data $DATA/eval.npz --out $LOG/$CFG/eval_flow.jsonl --tag $CFG) > $LOG/$CFG/eval_flow.log 2>&1
    echo "FLOW_DONE $CFG $(date -u +%H:%M:%S)"
}
train_flow gate $OUT/gate
for CFG in off_still off_wind gray_frame_still gray_frame_wind gray_episode_still gray_episode_wind; do
    train_flow $CFG /dev/shm/redcube/$CFG
done
echo LANE_A_TRAIN_DONE
