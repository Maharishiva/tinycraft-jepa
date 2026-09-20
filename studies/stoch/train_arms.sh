#!/bin/bash
# JEPA arms for the stochastic-control study (studies/redcube/train_arms.sh with the paths parametrised).
# Each arm first runs 500 calibration steps to measure its step time, then trains for the number of steps that
# fits 575 s (or FIXED_STEPS when set; tile_factored measured 8500 on the red-cube data), then gets the off-budget
# pixel decoder and the reward / done / value heads (probe.py).
#   lewm6          LeWM as published, ViT (192 wide, 6 layers) read out through its class token (192-d latent) +
#                  the paper's AdaLN AR predictor (6 blocks, 16 heads x 64, MLP 2048, history 3, dropout 0.1)
#   tile_factored  tile encoder (tinywmcraft tokenizer trunk, 4x4x32 grid latent) + factored predictor, history 8
#   ROOT=/dev/shm/stoch/gate GPU=0 FIXED_STEPS=8500 bash studies/stoch/train_arms.sh gate tile_factored
#   ROOT=/dev/shm/stoch/craftax DATA=.../tinywmcraft_500k/train.npz NACT=17 bash studies/stoch/train_arms.sh craftax tile_factored
set -e
CFG=$1
shift
ROOT=${ROOT:-/dev/shm/stoch/$CFG}
DATA=${DATA:-$ROOT/train.npz}
NACT=${NACT:-4}
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
cd ~/personal/tinycraft-jepa
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=runs/stoch/$CFG
mkdir -p $LOG $ROOT
TILE="--encoder tile --emb-dim 128 --enc-depth 6 --enc-heads 8 --tokens 16 --token-dim 32 --token-hidden 512"
FACTORED="--predictor factored --pred-width 256 --pred-depth 6 --pred-heads 8 --history 8"
VITCLS="--encoder vit --emb-dim 192 --enc-heads 3 --enc-mlp 768 --tokens 1"
LEWM="--predictor ar --pred-width 0 --pred-mlp 2048 --pred-depth 6 --pred-heads 16 --pred-dim-head 64 --mlp-hidden 2048 --history 3 --dropout 0.1"
budget() {
    awk '/^\[train\] 250\//{t1=$7} /^\[train\] 500\//{t2=$7} END{s=(t2-t1)/250; o=t1-250*s; n=int((575-o)/s/250)*250; if(n<250)n=250; print n}' $1
}
for ARM in "$@"; do
    case $ARM in
        tile_factored) FLAGS="$TILE $FACTORED" ;;
        lewm6) FLAGS="$VITCLS --enc-depth 6 $LEWM" ;;
        *) echo "unknown arm $ARM"; exit 1 ;;
    esac
    COMMON="--data $DATA --seed 0 --lr 1e-3 --n-actions $NACT"
    if [ ! -f $ROOT/$ARM/model.npz ]; then
        if [ -n "$FIXED_STEPS" ]; then
            STEPS=$FIXED_STEPS
        else
            $PY train.py $COMMON $FLAGS --out /tmp/stoch_calib_$ARM.npz --steps 500 > $LOG/calib_$ARM.log 2>&1
            STEPS=$(budget $LOG/calib_$ARM.log)
            echo "calibration $ARM -> $STEPS steps" >> $LOG/calib_$ARM.log
        fi
        $PY train.py $COMMON $FLAGS --out $ROOT/$ARM/model.npz --steps $STEPS > $LOG/train_$ARM.log 2>&1
    fi
    [ -f $ROOT/$ARM/model_probe.npz ] || $PY probe.py --model $ROOT/$ARM/model.npz --data $DATA --heads \
        --frames 80000 --held-out 200 --out $ROOT/$ARM/model_probe.npz > $LOG/probe_$ARM.log 2>&1
done
echo ARMS_DONE $CFG "$@"
