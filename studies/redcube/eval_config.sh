#!/bin/bash
# After training for one red-cube configuration: the frozen benchmark, cube metrics, latent probes and scenes.
# (Planners and PPO arms live in eval_config_full.sh, to be run later.)
#   GPU=1 bash studies/redcube/eval_config.sh color_frame_wind
CFG=$1
ROOT=/dev/shm/redcube/$CFG
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
FLOW=${FLOW:-$HOME/personal/codex-wm-push-20260916/recipes/model-51ffe8df}
cd ~/personal/tinycraft-jepa
export CUDA_VISIBLE_DEVICES=${GPU:-1}
OUT=runs/redcube/$CFG
mkdir -p $OUT
run() { NAME=$1; shift; [ -f $OUT/$NAME.done ] && return; "$@" > $OUT/$NAME.log 2>&1 && touch $OUT/$NAME.done || echo "FAILED $CFG $NAME"; }
run bench $PY -m studies.redcube.bench --config $CFG --root $ROOT --flow-code $FLOW --out $OUT --arms tile_factored
run oracle env JAX_PLATFORMS=cpu $PY -m studies.redcube.oracle --root $ROOT --out $OUT
run probe_scenes $PY -m studies.redcube.probe_scenes --config $CFG --root $ROOT --flow-code $FLOW --out $OUT
for ARM in tile_factored flow; do cp $ROOT/$ARM/losses.npz $OUT/losses_$ARM.npz 2>/dev/null; done
echo EVAL_DONE $CFG
