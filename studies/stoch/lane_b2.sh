#!/bin/bash
# Lane B, part 2 (e2e, after lane_b.sh): closed-loop control with the JEPA arms.
# Gate world at horizon 5 (both arms), the six cube worlds at horizon 3, then Craftax through plan.py (goal-image
# CEM offline + closed loop, reward MPC through the heads, matched baselines).
#   nohup bash studies/stoch/lane_b2.sh > runs/stoch/lane_b2.log 2>&1 &
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-e2e/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=runs/stoch
until grep -q LANE_B_TRAIN_DONE $LOG/lane_b.log; do sleep 60; done
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
control() { run $LOG/$1/control_$2 $PY -m studies.stoch.control --config $1 --model $3/$2/model_probe.npz \
                --out $LOG/$1/control_$2.json --envs 128 --horizon $4; }
control gate lewm6 /dev/shm/stoch/gate 5
control gate tile_factored /dev/shm/stoch/gate 5
echo "GATE_CONTROL_DONE $(date -u +%H:%M:%S)"
for CFG in off_still off_wind gray_frame_still gray_frame_wind gray_episode_still gray_episode_wind; do
    for ARM in lewm6 tile_factored; do control $CFG $ARM /dev/shm/stoch/redcube/$CFG 3; done
done
echo "CUBE_CONTROL_DONE $(date -u +%H:%M:%S)"
CD=$HOME/personal/tinywmcraft-e2e/starter/data/tinywmcraft_500k
run $LOG/craftax/plan_tile_factored $PY plan.py --model /dev/shm/stoch/craftax/tile_factored/model_probe.npz \
    --data $CD/eval.npz --train-data $CD/train.npz --closed-loop --reward --out $LOG/craftax --tag tile_factored --envs 128
echo LANE_B2_DONE
