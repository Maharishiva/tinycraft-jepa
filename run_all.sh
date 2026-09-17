#!/bin/bash
# Full pipeline: train inside the budget, then everything else off-budget.
set -e
PY=${PY:-python}
TAG=${TAG:-lewm_s0}
DATA=${DATA:-data/tinywmcraft_500k}

# 1. LeWM, one 10-minute A100 run (data loading + compilation + training).
$PY train.py --data $DATA/train.npz --out runs/$TAG/model.npz --seed ${SEED:-0} --steps ${STEPS:-12250}

# 2. Latent diagnostics: no decoder, no pixels (model selection lives here).
$PY viz.py --model runs/$TAG/model.npz --tag $TAG --windows 2048

# 3. Off-budget pixel probe: decoder on frozen embeddings (stop-gradient).
$PY probe.py --model runs/$TAG/model.npz --data $DATA/train.npz

# 4. Frozen benchmark over the decoded rollouts, plus videos and heat maps.
$PY eval.py --model runs/$TAG/model_decoder.npz --tag $TAG --note "LeWM + stop-grad decoder" --gif
$PY viz.py --model runs/$TAG/model_decoder.npz --tag $TAG --windows 2048 --gif

# 5. Planning in the latent space: offline action recovery and closed loop in Craftax.
$PY plan.py --model runs/$TAG/model.npz --tag $TAG --closed-loop
