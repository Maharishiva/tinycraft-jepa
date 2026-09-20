#!/bin/bash
# Round 2, lane R7: the torso control. PPO in the real gate with the latent policy's MLP torso on the flattened frame,
# on true pixels (ARMS=pix) and on the decoded JEPA latent (ARMS=dec, model seeds MSS). Waits for lane R5 to free the GPU.
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.45
G=$HOME/personal/stoch/gate
LOG=runs/stoch/gate
while pgrep -f "[l]ane_r5.sh" > /dev/null; do sleep 20; done
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for ARM in ${ARMS:-pix}; do
  if [ $ARM = pix ]; then
    for PS in 0 1 2; do
        NAME=$LOG/ppo_real_mlp_p$PS
        run $NAME $PY -m studies.stoch.rl --config gate --arm real --obs pixels --torso mlp --seed $PS --max-minutes 30 --out $NAME
        echo "DONE pixels mlp p$PS $(date -u +%H:%M:%S) $(tail -n 1 $NAME.log | cut -c1-160)"
    done
  else
    for MS in ${MSS:-0}; do
      if [ $MS = 0 ]; then WM=$G/tile_factored/model_probe.npz; else WM=$G/tile_factored_s$MS/model_probe.npz; fi
      for PS in 0 1 2; do
        NAME=$LOG/ppo_real_jepa_decoded_mlp_m${MS}_p$PS
        run $NAME $PY -m studies.stoch.rl --config gate --arm real --obs decoded --torso mlp --wm $WM --seed $PS --max-minutes 30 --out $NAME
        echo "DONE decoded mlp m$MS p$PS $(date -u +%H:%M:%S) $(tail -n 1 $NAME.log | cut -c1-160)"
      done
    done
  fi
done
echo LANE_R7_DONE
