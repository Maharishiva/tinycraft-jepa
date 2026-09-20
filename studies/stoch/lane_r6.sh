#!/bin/bash
# Round 2, lane R6: the amplitude intervention proper. PPO in the real gate on the frozen JEPA latent with the wall's own
# latent signal amplified: z + (g - 1)(z - z_nowall), z_nowall = the encoder on the same frame with the wall pixels erased.
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for GAIN in ${GAINS:-4 16}; do
for MS in ${MSS:-0}; do
    if [ $MS = 0 ]; then WM=$G/tile_factored/model_probe.npz; else WM=$G/tile_factored_s$MS/model_probe.npz; fi
    for PS in 0 1 2; do
        NAME=$LOG/ppo_real_jepa_latent_amp${GAIN}_m${MS}_p$PS
        run $NAME $PY -m studies.stoch.rl --config gate --arm real --obs latent --wm $WM --amplify-wall $GAIN --seed $PS --max-minutes 30 --out $NAME
        echo "DONE amp$GAIN m$MS p$PS $(date -u +%H:%M:%S) $(tail -n 1 $NAME.log | cut -c1-160)"
    done
done
done
echo LANE_R6_DONE
