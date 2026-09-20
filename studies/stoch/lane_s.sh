#!/bin/bash
# Round 2, seeds (multigpu-vm, GPU 1): grid JEPA model seeds 1 and 2 on the gate world, each with the off-budget
# decoder + heads (probe.py) and the residual sampling head (sampler.py).  Same flags as train_arms.sh.
#   (setsid nohup bash studies/stoch/lane_s.sh > runs/stoch/lane_s.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
G=${G:-$HOME/personal/stoch/gate}
DATA=$G/train.npz
LOG=runs/stoch/gate
TILE="--encoder tile --emb-dim 128 --enc-depth 6 --enc-heads 8 --tokens 16 --token-dim 32 --token-hidden 512"
FACTORED="--predictor factored --pred-width 256 --pred-depth 6 --pred-heads 8 --history 8"
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for SEED in ${SEEDS:-1 2}; do
    OUT=$G/tile_factored_s$SEED
    mkdir -p $OUT
    run $LOG/train_tile_factored_s$SEED $PY train.py --data $DATA --seed $SEED --lr 1e-3 --n-actions 4 $TILE $FACTORED \
        --out $OUT/model.npz --steps 8500
    run $LOG/probe_tile_factored_s$SEED $PY probe.py --model $OUT/model.npz --data $DATA --heads --frames 80000 --held-out 200 \
        --out $OUT/model_probe.npz
    run $LOG/sampler_tile_factored_s$SEED $PY -m studies.stoch.sampler --model $OUT/model_probe.npz --data $DATA --out $OUT/sampler.npz
    echo "SEED_DONE $SEED $(date -u +%H:%M:%S)"
done
echo LANE_S_DONE
