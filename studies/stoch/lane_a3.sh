#!/bin/bash
# Round 2, heads audit with the planner view, third attempt (multigpu-vm, GPU 1, after both lane_p streams).
#   (setsid nohup bash studies/stoch/lane_a3.sh > runs/stoch/lane_a3.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.45
G=$HOME/personal/stoch/gate
until grep -q LANE_P_DONE runs/stoch/lane_p0.log && grep -q LANE_P_DONE runs/stoch/lane_p1.log; do sleep 30; done
$PY -m studies.stoch.heads_audit --config gate --data $G/eval.npz --model $G/tile_factored/model_probe.npz \
    --sampler $G/tile_factored/sampler.npz --out runs/stoch/gate/heads_audit_tile_factored.json > runs/stoch/gate/heads_audit_tile_factored.log 2>&1 \
    && echo AUDIT_DONE || echo AUDIT_FAILED
