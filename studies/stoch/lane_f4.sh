#!/bin/bash
# Round 2, the flow mean-of-8 cells without the 12-minute PPO cap (multigpu-vm, GPU 0, after lane_f.sh): the mean
# of 8 samples costs 8 flow calls per imagined step, so the default --max-minutes 12 truncated the first runs at
# ~80 of 150 updates; the sampled arm finishes its 150 updates in ~5 min.  Same updates, same seeds, 60-min cap.
#   (setsid nohup bash studies/stoch/lane_f4.sh > runs/stoch/lane_f4.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.42
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
until grep -q LANE_F_DONE runs/stoch/lane_f.log; do sleep 30; done
for PS in 0 1 2; do
    NAME=$LOG/ppo_dream_latent_mean8_p$PS
    [ -d $NAME ] && mv $NAME ${NAME}_12min && rm -f $NAME.done
    $PY -m studies.stoch.rl --config gate --arm dream --obs latent --wm $G/flow/model.npz --wm-code studies/stoch/flow_recipe \
        --data $G/train.npz --seed $PS --sampler-mean 8 --max-minutes 60 --out $NAME > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"
    echo "DONE flow mean8 p$PS $(date -u +%H:%M:%S)"
done
echo LANE_F4_DONE
