#!/bin/bash
# Lane B, part 4 (e2e, after lane_b3.sh): PPO in the real gate world, more seeds (the reference is seed-sensitive:
# seed 0 gave 3 % survival at 1.5M steps in one run and 75 % at 1M steps in another launch of the same config).
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-e2e/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=runs/stoch
until grep -q LANE_B3_DONE $LOG/lane_b3.log; do sleep 60; done
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for SEED in 1 2; do
    run $LOG/gate/ppo_real_s$SEED $PY -m studies.stoch.rl --config gate --arm real --seed $SEED --out $LOG/gate/ppo_real_s$SEED
    run $LOG/gate/ppo_real_long_s$SEED $PY -m studies.stoch.rl --config gate --arm real --seed $SEED --budget 8000000 --max-minutes 30 --eval-every 20 --out $LOG/gate/ppo_real_long_s$SEED
done
echo LANE_B4_DONE
