#!/bin/bash
# Round 2, more PPO seeds inside the flow model's sampled latent rollouts (e2e, GPU 0, after lane_x.sh): PPO seeds
# 3, 4, 5, to estimate how often PPO inside the flow model discovers the retreat-and-wait strategy (seed 0 did,
# seed 1 did not; PPO in the real environment discovers it in 2 of 3 seeds).
#   (setsid nohup bash studies/stoch/lane_f3.sh > runs/stoch/lane_f3.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-e2e/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.42
G=/dev/shm/stoch/gate
LOG=runs/stoch/gate
until grep -q LANE_X_DONE runs/stoch/lane_x.log; do sleep 60; done
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for PS in 3 4 5; do
    NAME=$LOG/ppo_dream_latent_samp_p$PS
    run $NAME $PY -m studies.stoch.rl --config gate --arm dream --obs latent --wm $G/flow_model.npz --wm-code studies/stoch/flow_recipe \
        --data $G/train.npz --seed $PS --out $NAME
    echo "DONE flow samp p$PS $(date -u +%H:%M:%S)"
done
echo LANE_F3_DONE
