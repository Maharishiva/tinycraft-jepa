#!/bin/bash
# Lane C, part 2 (e2e): rerun of the sampled heads-MPC on the gate with a larger pool (the .3 pool died on a
# 4.75 GiB allocation in the goal search), once the sampled PPO and the cube controls have released the GPU.
#   (setsid nohup bash studies/stoch/lane_c2.sh > runs/stoch/lane_c2.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-e2e/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export ROWS=2048
LOG=runs/stoch
G=/dev/shm/stoch/gate
until grep -q SAMPLED_PPO_DONE $LOG/lane_c.log && grep -q LANE_B2B_DONE $LOG/lane_b2b.log; do sleep 30; done
rm -f $LOG/gate/control_tile_factored_sampled.done
run() { NAME=$1; shift; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
run $LOG/gate/control_tile_factored_sampled $PY -m studies.stoch.control --config gate --model $G/tile_factored/model_probe.npz \
    --sampler $G/tile_factored/sampler.npz --out $LOG/gate/control_tile_factored_sampled.json
echo "SAMPLED_CONTROL_DONE $(date -u +%H:%M:%S)"
echo LANE_C2_DONE
