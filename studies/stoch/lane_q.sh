#!/bin/bash
# Round 2, lane Q (multigpu GPU 1, small pool): latent diagnostic with the wall-direction decomposition for all three
# grid-JEPA model seeds, and the heads audit for model seeds 1 and 2.
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.35
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
rm -f $LOG/latent_check.done
run $LOG/latent_check $PY -m studies.stoch.latent_check --data $G/train.npz --grid $G/tile_factored/model_probe.npz \
    --grid-sampler $G/tile_factored/sampler.npz --flow $G/flow/model.npz --flow-code studies/stoch/flow_recipe --out $LOG/latent_check.json
echo "DONE latent_check s0 $(date -u +%H:%M:%S)"
for MS in 1 2; do
    M=tile_factored_s$MS
    run $LOG/latent_check_s$MS $PY -m studies.stoch.latent_check --data $G/train.npz --grid $G/$M/model_probe.npz \
        --grid-sampler $G/$M/sampler.npz --out $LOG/latent_check_s$MS.json
    echo "DONE latent_check s$MS $(date -u +%H:%M:%S)"
    run $LOG/heads_audit_$M $PY -m studies.stoch.heads_audit --config gate --data $G/eval.npz --model $G/$M/model_probe.npz \
        --sampler $G/$M/sampler.npz --out $LOG/heads_audit_$M.json
    echo "DONE heads_audit s$MS $(date -u +%H:%M:%S)"
done
echo LANE_Q_DONE
