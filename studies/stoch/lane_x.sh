#!/bin/bash
# Round 2, budget sweep controls (e2e, after lane_d.sh): PPO inside the 2x and 4x grid JEPAs on the gate, in the three
# predictor modes (det / mean8 / samp), PPO seed 0.
#   (setsid nohup bash studies/stoch/lane_x.sh > runs/stoch/lane_x.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-e2e/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.42
G=/dev/shm/stoch/gate
LOG=runs/stoch/gate
until grep -q LANE_D_DONE runs/stoch/lane_d.log; do sleep 60; done
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for MULT in 2 4; do
    M=$G/tile_factored_x$MULT
    for MODE in det mean8 samp; do
        NAME=$LOG/ppo_jepa_tf_x${MULT}_${MODE}_p0
        case $MODE in det) EXTRA="" ;; mean8) EXTRA="--sampler $M/sampler.npz --sampler-mean 8" ;; samp) EXTRA="--sampler $M/sampler.npz" ;; esac
        run $NAME $PY -m studies.stoch.rl --config gate --arm jepa --wm $M/model_probe.npz $EXTRA --data $G/train.npz --seed 0 --out $NAME
        echo "DONE x$MULT $MODE $(date -u +%H:%M:%S)"
    done
done
echo LANE_X_DONE
