#!/bin/bash
# Lane C (e2e): the sampling-predictor control cell on the gate: the grid JEPA with a flow head on its
# predictor's residual (studies.stoch.sampler), same encoder and heads, samples instead of the mean.
#   (setsid nohup bash studies/stoch/lane_c.sh > runs/stoch/lane_c.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-e2e/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
LOG=runs/stoch
G=/dev/shm/stoch/gate
until [ -f $G/tile_factored/sampler.npz ]; do sleep 30; done
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
run $LOG/gate/control_tile_factored_sampled $PY -m studies.stoch.control --config gate --model $G/tile_factored/model_probe.npz \
    --sampler $G/tile_factored/sampler.npz --out $LOG/gate/control_tile_factored_sampled.json
echo "SAMPLED_CONTROL_DONE $(date -u +%H:%M:%S)"
run $LOG/gate/ppo_jepa_tile_factored_sampled $PY -m studies.stoch.rl --config gate --arm jepa --wm $G/tile_factored/model_probe.npz \
    --sampler $G/tile_factored/sampler.npz --data $G/train.npz --out $LOG/gate/ppo_jepa_tile_factored_sampled
echo "SAMPLED_PPO_DONE $(date -u +%H:%M:%S)"
run $LOG/gate/wallmass_sampled $PY -m studies.stoch.wallmass --data $G/eval.npz --lewm $G/lewm6/model_probe.npz \
    --grid $G/tile_factored/model_probe.npz --grid-sampler $G/tile_factored/sampler.npz --flow $G/flow_model.npz \
    --flow-code studies/stoch/flow_recipe --out $LOG/gate/wallmass.json
run $LOG/gate/predictions_sampled $PY -m studies.stoch.predictions --lewm $G/lewm6/model_probe.npz \
    --grid $G/tile_factored/model_probe.npz --grid-sampler $G/tile_factored/sampler.npz --flow $G/flow_model.npz \
    --flow-code studies/stoch/flow_recipe --out studies/stoch/paper/figures/fig_predictions.png
echo LANE_C_DONE
