#!/bin/bash
# Round 2, budget sweep (e2e, A100): the grid JEPA on the gate at 2x and 4x the 8500-step budget (seed 0), each with
# the off-budget decoder + heads and the residual sampling head.  Same flags as train_arms.sh.
#   (setsid nohup bash studies/stoch/lane_d.sh > runs/stoch/lane_d.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-e2e/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
G=${G:-/dev/shm/stoch/gate}
DATA=$G/train.npz
LOG=runs/stoch/gate
TILE="--encoder tile --emb-dim 128 --enc-depth 6 --enc-heads 8 --tokens 16 --token-dim 32 --token-hidden 512"
FACTORED="--predictor factored --pred-width 256 --pred-depth 6 --pred-heads 8 --history 8"
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for MULT in ${MULTS:-2 4}; do
    OUT=$G/tile_factored_x$MULT
    mkdir -p $OUT
    run $LOG/train_tile_factored_x$MULT $PY train.py --data $DATA --seed 0 --lr 1e-3 --n-actions 4 $TILE $FACTORED \
        --out $OUT/model.npz --steps $((8500 * MULT))
    run $LOG/probe_tile_factored_x$MULT $PY probe.py --model $OUT/model.npz --data $DATA --heads --frames 80000 --held-out 200 \
        --out $OUT/model_probe.npz
    run $LOG/sampler_tile_factored_x$MULT $PY -m studies.stoch.sampler --model $OUT/model_probe.npz --data $DATA --out $OUT/sampler.npz
    echo "BUDGET_DONE x$MULT $(date -u +%H:%M:%S)"
done
echo LANE_D_DONE
