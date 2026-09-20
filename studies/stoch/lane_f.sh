#!/bin/bash
# Round 2, the flow model as a mean predictor (multigpu-vm, GPU 0): PPO inside the flow model's latent rollouts with
# the next latent replaced by the MEAN of 8 samples (the generative model forced to plan through the mean), against
# the sampled arm (ppo_dream_latent = PPO seed 0 exists from round 1), PPO seeds 0, 1, 2.
#   (setsid nohup bash studies/stoch/lane_f.sh > runs/stoch/lane_f.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.42
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
dream() {
    MODE=$1; PS=$2
    NAME=$LOG/ppo_dream_latent_${MODE}_p$PS
    EXTRA=""; [ $MODE = mean8 ] && EXTRA="--sampler-mean 8"
    run $NAME $PY -m studies.stoch.rl --config gate --arm dream --obs latent --wm $G/flow/model.npz --wm-code studies/stoch/flow_recipe \
        --data $G/train.npz --seed $PS $EXTRA --out $NAME
    echo "DONE flow $MODE p$PS $(date -u +%H:%M:%S)"
}
dream mean8 0
dream samp 1
dream samp 2
dream mean8 1
dream mean8 2
echo LANE_F_DONE
