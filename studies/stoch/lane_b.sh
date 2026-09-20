#!/bin/bash
# Lane B (e2e, A100): the two JEPA arms on the gate world, then the grid JEPA on Craftax (same 10-minute budget,
# off-budget decoder and heads), each with probes.
#   nohup bash studies/stoch/lane_b.sh > runs/stoch/lane_b.log 2>&1 &
set -e
cd ~/personal/tinycraft-jepa
export PY=$HOME/personal/tinywmcraft-e2e/.venv/bin/python
export GPU=0
mkdir -p runs/stoch
ROOT=/dev/shm/stoch/gate FIXED_STEPS=8500 bash studies/stoch/train_arms.sh gate tile_factored
ROOT=/dev/shm/stoch/gate bash studies/stoch/train_arms.sh gate lewm6
ROOT=/dev/shm/stoch/craftax DATA=$HOME/personal/tinywmcraft-e2e/starter/data/tinywmcraft_500k/train.npz NACT=17 \
    FIXED_STEPS=8500 bash studies/stoch/train_arms.sh craftax tile_factored
echo LANE_B_TRAIN_DONE
