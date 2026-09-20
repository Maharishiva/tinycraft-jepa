#!/bin/bash
# Round 2, lane R3 (multigpu GPU 0): PPO in the REAL gate on the JEPA latent DECODED back to pixels (fixed encoder, pixel torso), three model seeds x three policy seeds.
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.45
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for MS in 0 1 2; do
D=$G/tile_factored; [ $MS != 0 ] && D=$G/tile_factored_s$MS
for PS in 0 1 2; do
    NAME=$LOG/ppo_real_jepa_decoded_m${MS}_p$PS
    run $NAME $PY -m studies.stoch.rl --config gate --arm real --obs decoded --wm $D/model_probe.npz --seed $PS --max-minutes 30 --out $NAME
    echo "DONE real decoded m$MS p$PS $(date -u +%H:%M:%S)"
done
done
echo LANE_R3_DONE
