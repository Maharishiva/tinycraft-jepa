#!/bin/bash
# Everything after training for one red-cube configuration: benchmark, JEPA planners, reference policies,
# PPO in the real environment, PPO inside the flow-matching world model.
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
run bench $PY -m studies.redcube.bench --config $CFG --root $ROOT --flow-code $FLOW --out $OUT
for ARM in jepa2 grid; do
  run control_$ARM $PY -m studies.redcube.control --config $CFG --model $ROOT/$ARM/model_probe.npz --out $OUT/control_$ARM.json
done
run baselines $PY -m studies.redcube.rl --config $CFG --arm baselines --out $OUT
run ppo_real $PY -m studies.redcube.rl --config $CFG --arm real --out $OUT/ppo_real
run ppo_dream $PY -m studies.redcube.rl --config $CFG --arm dream --wm $ROOT/flow/model.npz --wm-code $FLOW --data $ROOT/train.npz --out $OUT/ppo_dream
for ARM in jepa2 grid flow; do cp $ROOT/$ARM/losses.npz $OUT/losses_$ARM.npz 2>/dev/null; done
echo EVAL_DONE $CFG
