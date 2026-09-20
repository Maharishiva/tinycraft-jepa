#!/bin/bash
# Benchmark of EVERY trained LeWM arm of a configuration (the page's benchmark keeps only tile_factored), written to
# runs/redcube/<cfg>/allarms.  Re-runnable: delete allarms/bench.done to redo it after new arms land.
#   GPU=1 bash studies/redcube/eval_arms.sh gray_frame_still
CFG=$1
ROOT=/dev/shm/redcube/$CFG
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
FLOW=${FLOW:-$HOME/personal/codex-wm-push-20260916/recipes/model-51ffe8df}
cd ~/personal/tinycraft-jepa
export CUDA_VISIBLE_DEVICES=${GPU:-1}
OUT=runs/redcube/$CFG/allarms
mkdir -p $OUT
[ -f $OUT/bench.done ] && exit 0
$PY -m studies.redcube.bench --config $CFG --root $ROOT --flow-code $FLOW --out $OUT > $OUT/bench.log 2>&1 && touch $OUT/bench.done || echo "FAILED $CFG allarms"
echo ARMS_EVAL_DONE $CFG
