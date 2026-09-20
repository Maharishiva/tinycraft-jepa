#!/bin/bash
# Extra LeWM arms on a configuration whose data already exists.  Each arm first runs 500 calibration
# steps to measure its step time, then trains for the number of steps that fits 575 s (or FIXED_STEPS when set,
# tile_factored measured 8500 on config 1), then gets the off-budget probes.  Arms:
# Encoders: `vit` = LeWM's ViT with the 81x32 patch-grid read-out; `tile` = the tile transformer + strided
# convolution trunk of the tinywmcraft tokenizer (2.6M, 4x4x32 latent).  Predictors: `ar` = LeWM's AdaLN
# autoregressive transformer over the flattened latent; `factored` = the tinywmcraft dynamics transformer
# (attention over the frame's tokens, then causal attention over frames, 8.3M).  Objective is LeWM's in every arm.
#   tile_factored    tile encoder + factored predictor (history 8)
#   tile_ar          tile encoder + AR predictor 9.0M (history 3, dropout 0.1)
#   tile_ar8         tile encoder + AR predictor, history 8, no dropout (isolates history / dropout)
#   tile_ar_nodrop   tile encoder + AR predictor, history 3, no dropout
#   vit_factored     ViT encoder, 81x32 grid latent + factored predictor (history 8).  Trained on the 100k
#                    configurations; on the 500k ones its train step dies allocating a 21.81 GiB buffer next to the
#                    5.95 GB of frames on device, at the default 75% pool and at a 92% pool alike (2026-09-18)
#   tile_cls_ar      tile encoder with a class-token read-out (512-d latent) + AR predictor (history 3, dropout 0.1)
#   tile_cls_ar8     same encoder + AR predictor, history 8, no dropout
#   tile_cls_factored same encoder + factored predictor over the single 512-d token (history 8)
#   tile_factored_b256 / _b512   tile_factored at batch 256 / 512 (default 128), steps calibrated to the same 575 s.
#                    On the 500k configurations (5.95 GB of frames on device) batch 128 uses 18 GB; batch 256 needs a
#                    14.25 GiB train-step arena, over JAX's default 75% pool of the 40 GB A100, and runs with a 92%
#                    pool (3750 steps in 572 s); batch 512 asks for 28.50 GiB and dies at the 92% pool too
#   tile_factored_s1 / _s2       tile_factored with seeds 1 / 2 (seed 0 stuck at pred loss .73 on gray_frame_wind)
#   lewm6            LeWM as published, encoder shrunk to match ours: ViT (192 wide, 6 layers) read out through its
#                    class token (192-d latent) + the paper's AdaLN AR predictor (6 blocks, 16 heads x 64, MLP 2048,
#                    history 3, dropout 0.1); 15.3M parameters (2.7M encoder + 10.8M predictor)
#   lewm             the full LeWM: same with the 12-layer ViT-tiny; 17.9M parameters
#   GPU=1 bash studies/redcube/train_arms.sh color_frame_wind tile_ar8 vit_factored
set -e
CFG=$1
shift
ROOT=/dev/shm/redcube/$CFG
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
cd ~/personal/tinycraft-jepa
export CUDA_VISIBLE_DEVICES=${GPU:-1}
LOG=runs/redcube/$CFG
mkdir -p $LOG
TILE="--encoder tile --emb-dim 128 --enc-depth 6 --enc-heads 8 --tokens 16 --token-dim 32 --token-hidden 512"
VIT="--encoder vit --emb-dim 192 --enc-depth 12 --enc-heads 3 --tokens 81 --token-dim 32 --token-hidden 512"
TILECLS="--encoder tile_cls --emb-dim 128 --enc-depth 6 --enc-heads 8 --tokens 1 --latent-dim 512"
FACTORED="--predictor factored --pred-width 256 --pred-depth 6 --pred-heads 8 --history 8"
AR="--predictor ar --pred-width 256 --pred-mlp 1536 --pred-depth 6 --pred-heads 4"
VITCLS="--encoder vit --emb-dim 192 --enc-heads 3 --enc-mlp 768 --tokens 1"
LEWM="--predictor ar --pred-width 0 --pred-mlp 2048 --pred-depth 6 --pred-heads 16 --pred-dim-head 64 --mlp-hidden 2048 --history 3 --dropout 0.1"
budget() {
    awk '/^\[train\] 250\//{t1=$7} /^\[train\] 500\//{t2=$7} END{s=(t2-t1)/250; o=t1-250*s; n=int((575-o)/s/250)*250; if(n<250)n=250; print n}' $1
}
for ARM in "$@"; do
    ENV=""
    case $ARM in
        tile_factored) FLAGS="$TILE $FACTORED" ;;
        tile_ar) FLAGS="$TILE $AR --history 3" ;;
        tile_ar8) FLAGS="$TILE $AR --history 8 --dropout 0.0" ;;
        tile_ar_nodrop) FLAGS="$TILE $AR --history 3 --dropout 0.0" ;;
        vit_factored) FLAGS="$VIT $FACTORED" ;;
        tile_cls_ar) FLAGS="$TILECLS $AR --history 3" ;;
        tile_cls_ar8) FLAGS="$TILECLS $AR --history 8 --dropout 0.0" ;;
        tile_cls_factored) FLAGS="$TILECLS --token-dim 512 $FACTORED" ;;
        tile_factored_b256) FLAGS="$TILE $FACTORED --batch 256"; ENV="XLA_PYTHON_CLIENT_MEM_FRACTION=.92" ;;
        tile_factored_b512) FLAGS="$TILE $FACTORED --batch 512" ;;
        tile_factored_s1) FLAGS="$TILE $FACTORED --seed 1" ;;
        tile_factored_s2) FLAGS="$TILE $FACTORED --seed 2" ;;
        lewm6) FLAGS="$VITCLS --enc-depth 6 $LEWM" ;;
        lewm) FLAGS="$VITCLS --enc-depth 12 $LEWM" ;;
        *) echo "unknown arm $ARM"; exit 1 ;;
    esac
    COMMON="--data $ROOT/train.npz --seed 0 --lr 1e-3 --n-actions 4"
    if [ ! -f $ROOT/$ARM/model.npz ]; then
        if [ -n "$FIXED_STEPS" ]; then
            STEPS=$FIXED_STEPS
        else
            env $ENV $PY train.py $COMMON $FLAGS --out /tmp/redcube_calib_$ARM.npz --steps 500 > $LOG/calib_$ARM.log 2>&1
            STEPS=$(budget $LOG/calib_$ARM.log)
            echo "calibration $ARM -> $STEPS steps" >> $LOG/calib_$ARM.log
        fi
        env $ENV $PY train.py $COMMON $FLAGS --out $ROOT/$ARM/model.npz --steps $STEPS > $LOG/train_$ARM.log 2>&1
    fi
    [ -f $ROOT/$ARM/model_probe.npz ] || $PY probe.py --model $ROOT/$ARM/model.npz --data $ROOT/train.npz --heads \
        --frames 80000 --held-out 200 --out $ROOT/$ARM/model_probe.npz > $LOG/probe_$ARM.log 2>&1
done
echo ARMS_DONE $CFG "$@"
