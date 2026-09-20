#!/bin/bash
# Round 2, the PPO-inside-the-JEPA grid on the gate (multigpu-vm, GPU 1, after lane_s.sh): model seeds 0, 1, 2 x
# predictor modes {det: the JEPA's own mean predictor, mean8: the residual head as the mean of 8 samples, samp: one
# sample} x PPO seeds 0, 1, 2.  The model-seed-0 / PPO-seed-0 cells already exist under their round-1 names
# (ppo_jepa_tile_factored, ppo_jepa_tile_factored_mean8, ppo_jepa_tile_factored_sampled).  Two streams share the GPU:
#   (STREAM=0 setsid nohup bash studies/stoch/lane_p.sh > runs/stoch/lane_p0.log 2>&1 < /dev/null &)
#   (STREAM=1 setsid nohup bash studies/stoch/lane_p.sh > runs/stoch/lane_p1.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.42
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
STREAM=${STREAM:-0}
until grep -q LANE_S_DONE runs/stoch/lane_s.log; do sleep 60; done
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
jepa() {
    MS=$1; MODE=$2; PS=$3
    M=$G/tile_factored; [ $MS = 0 ] || M=$G/tile_factored_s$MS
    NAME=$LOG/ppo_jepa_tf_m${MS}_${MODE}_p$PS
    case $MODE in det) EXTRA="" ;; mean8) EXTRA="--sampler $M/sampler.npz --sampler-mean 8" ;; samp) EXTRA="--sampler $M/sampler.npz" ;; esac
    run $NAME $PY -m studies.stoch.rl --config gate --arm jepa --wm $M/model_probe.npz $EXTRA --data $G/train.npz --seed $PS --out $NAME
}
I=0
for MS in 1 2 0; do for MODE in det mean8 samp; do for PS in 0 1 2; do
    if [ $MS = 0 ] && [ $PS = 0 ]; then continue; fi
    if [ $((I % 2)) = $STREAM ]; then jepa $MS $MODE $PS; echo "DONE m$MS $MODE p$PS $(date -u +%H:%M:%S)"; fi
    I=$((I + 1))
done; done; done
echo LANE_P_DONE
