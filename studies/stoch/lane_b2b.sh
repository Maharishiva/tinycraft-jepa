#!/bin/bash
# Lane B, part 2b (e2e): the six cube worlds' JEPA controls at horizon 3 (lane_b2 skipped them: the per-world
# log directories did not exist on this machine).  Runs beside whatever else holds the GPU, at a 20 % pool.
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-e2e/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
export ROWS=1024
LOG=runs/stoch
run() { NAME=$1; shift; mkdir -p $(dirname $NAME); [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
control() { run $LOG/$1/control_$2 $PY -m studies.stoch.control --config $1 --model $3/$2/model_probe.npz \
                --out $LOG/$1/control_$2.json --envs 128 --horizon $4; }
for CFG in off_still off_wind gray_frame_still gray_frame_wind gray_episode_still gray_episode_wind; do
    for ARM in lewm6 tile_factored; do control $CFG $ARM /dev/shm/stoch/redcube/$CFG 3; done
done
echo LANE_B2B_DONE
