#!/bin/bash
# Data + the two 10-minute models for one red-cube configuration: the flow-matching recipe and the JEPA with the
# same architecture (`tile_factored` in train_arms.sh, which also fits its off-budget probes).
#   GPU=1 bash studies/redcube/train_config.sh color_frame_wind
#   TRAIN_STEPS=5000 -> 500k transitions (100 envs x 5000 steps) instead of the default 100k; eval stays 100 x 250.
set -e
CFG=$1
ROOT=/dev/shm/redcube/$CFG
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
FLOW=${FLOW:-$HOME/personal/codex-wm-push-20260916/recipes/model-51ffe8df}
cd ~/personal/tinycraft-jepa
export CUDA_VISIBLE_DEVICES=${GPU:-1}
mkdir -p $ROOT runs/redcube/$CFG
LOG=runs/redcube/$CFG
[ -f $ROOT/train.npz ] || $PY -m studies.redcube.env --config $CFG --out $ROOT --train-steps ${TRAIN_STEPS:-1000} > $LOG/data.log 2>&1
[ -f $ROOT/flow/model.npz ] || (cd $FLOW && $PY train.py --data $ROOT/train.npz --out $ROOT/flow/model.npz --seed 0) > $LOG/train_flow.log 2>&1
GPU=$CUDA_VISIBLE_DEVICES FIXED_STEPS=8500 bash studies/redcube/train_arms.sh $CFG tile_factored
echo TRAIN_DONE $CFG
