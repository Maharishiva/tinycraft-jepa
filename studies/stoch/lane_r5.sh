#!/bin/bash
# Round 2, lane R5: the amplitude intervention (wall-difference subspace). PPO in the real gate on the frozen JEPA latent with the wall subspace
# (ridge-probe span) scaled by GAIN, against a random subspace of the same rank. GPU 0: model seed 0, GPU 1: model seed 1.
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.45
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
GAIN=${GAIN:-4}
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for MS in ${MSS:-0}; do
    if [ $MS = 0 ]; then WM=$G/tile_factored/model_probe.npz; else WM=$G/tile_factored_s$MS/model_probe.npz; fi
    run $LOG/dwallspace_m$MS $PY -m studies.stoch.wallspace2 --wm $WM --data $G/train.npz --out $LOG/dwallspace_m$MS.npz
    for ARM in dwall drand; do
    for PS in 0 1 2; do
        NAME=$LOG/ppo_real_jepa_latent_${ARM}${GAIN}_m${MS}_p$PS
        run $NAME $PY -m studies.stoch.rl --config gate --arm real --obs latent --wm $WM --scale-basis $LOG/${ARM}space_m$MS.npz --gain $GAIN --seed $PS --max-minutes 30 --out $NAME
        echo "DONE $ARM$GAIN m$MS p$PS $(date -u +%H:%M:%S) $(tail -n 1 $NAME.log | cut -c1-120)"
    done
    done
done
echo LANE_R5_DONE
