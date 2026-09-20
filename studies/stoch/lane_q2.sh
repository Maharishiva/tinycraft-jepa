#!/bin/bash
# Round 2, lane Q2 (multigpu GPU 1): the latent diagnostic on the HELD-OUT evaluation set for all three grid-JEPA
# model seeds (the training-set versions are kept as *_train).
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
until grep -q LANE_Q_DONE runs/stoch/lane_q.log; do sleep 20; done
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for MS in 0 1 2; do
    if [ $MS = 0 ]; then M=tile_factored; N=latent_check; FLOW="--flow $G/flow/model.npz --flow-code studies/stoch/flow_recipe"; else M=tile_factored_s$MS; N=latent_check_s$MS; FLOW=""; fi
    for ext in json npz log done; do [ -f $LOG/$N.$ext ] && mv $LOG/$N.$ext $LOG/${N}_train.$ext; done
    run $LOG/$N $PY -m studies.stoch.latent_check --data $G/eval.npz --grid $G/$M/model_probe.npz --grid-sampler $G/$M/sampler.npz $FLOW --out $LOG/$N.json
    echo "DONE $N eval $(date -u +%H:%M:%S)"
done
echo LANE_Q2_DONE
