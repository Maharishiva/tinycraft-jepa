#!/bin/bash
# Round 2, re-run of the flow mean-of-8 cells after the sequential-mean fix (multigpu-vm, GPU 0, after lane_f.sh).
#   (setsid nohup bash studies/stoch/lane_f2.sh > runs/stoch/lane_f2.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.42
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
until grep -q LANE_F_DONE runs/stoch/lane_f.log; do sleep 30; done
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for PS in 0 1 2; do
    NAME=$LOG/ppo_dream_latent_mean8_p$PS
    run $NAME $PY -m studies.stoch.rl --config gate --arm dream --obs latent --wm $G/flow/model.npz --wm-code studies/stoch/flow_recipe \
        --data $G/train.npz --seed $PS --sampler-mean 8 --out $NAME
    echo "DONE flow mean8 p$PS $(date -u +%H:%M:%S)"
done
echo LANE_F2_DONE
