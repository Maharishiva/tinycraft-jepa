#!/bin/bash
# Round 2, SIGReg dose-response (e2e, GPU 0): LeWM (class token) on the per-episode-noise still cube world at
# SIGReg weights 0 / 0.009 / 0.9 (0.09 = the published weight = the existing lewm6 arm) and at a 32-d latent
# (weight 0.09, predictor kept 192 wide), each at the arm's calibrated 24000 steps, then the off-budget probe
# and the horizon-3 controllers.
#   (setsid nohup bash studies/stoch/lane_l.sh > runs/stoch/lane_l.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-e2e/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
CFG=gray_episode_still
ROOT=/dev/shm/stoch/$CFG
LOG=runs/stoch/$CFG
mkdir -p $LOG
VITCLS="--encoder vit --emb-dim 192 --enc-heads 3 --enc-mlp 768 --tokens 1 --enc-depth 6"
LEWM="--predictor ar --pred-width 0 --pred-mlp 2048 --pred-depth 6 --pred-heads 16 --pred-dim-head 64 --mlp-hidden 2048 --history 3 --dropout 0.1"
COMMON="--data $ROOT/train.npz --seed 0 --lr 1e-3 --n-actions 4 --steps 24000"
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
arm() {
    ARM=$1; shift
    M=$ROOT/$ARM
    run $LOG/train_$ARM $PY train.py $COMMON $VITCLS $LEWM "$@" --out $M/model.npz
    run $LOG/probe_$ARM $PY probe.py --model $M/model.npz --data $ROOT/train.npz --heads --frames 80000 --held-out 200 --out $M/model_probe.npz
    run $LOG/control_$ARM $PY -m studies.stoch.control --config $CFG --model $M/model_probe.npz --out $LOG/control_$ARM.json \
        --envs 128 --horizon 3 --policies random,centre,goal,plan
    echo "ARM_DONE $ARM $(date -u +%H:%M:%S)"
}
arm lewm6_lam0.009 --sigreg-weight 0.009
arm lewm6_lam0.9 --sigreg-weight 0.9
arm lewm6_d32 --latent-dim 32 --pred-width 192
arm lewm6_lam0 --sigreg-weight 0
echo LANE_L_DONE
