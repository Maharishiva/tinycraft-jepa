#!/bin/bash
# Round 2, edge starts (multigpu-vm, GPU 0, after the latent check): the six cube worlds graded with the cube reset
# 3 to 8 px from a border, where a state-blind policy dies: baselines (random, centre, alternate), the two JEPA
# planners at horizon 3, and every saved PPO policy re-graded (rl.py --arm evaluate).
#   (setsid nohup bash studies/stoch/lane_g.sh > runs/stoch/lane_g.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
PY=${PY:-$HOME/personal/tinywmcraft-slim/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=.25
R=/dev/shm/redcube
LOG=runs/stoch
until grep -q LATENT_CHECK_EXIT runs/stoch/latent_check_lane.log; do sleep 30; done
run() { NAME=$1; shift; [ -f $NAME.done ] && return; "$@" > $NAME.log 2>&1 && touch $NAME.done || echo "FAILED $NAME"; }
for CFG in off_still off_wind gray_frame_still gray_frame_wind gray_episode_still gray_episode_wind; do
    run $LOG/$CFG/baselines_edge $PY -m studies.stoch.rl --config $CFG --arm baselines --eval-start edge --out $LOG/$CFG
    for ARM in lewm6 tile_factored; do
        run $LOG/$CFG/control_${ARM}_edge $PY -m studies.stoch.control --config $CFG --model $R/$CFG/$ARM/model_probe.npz \
            --out $LOG/$CFG/control_${ARM}_edge.json --envs 128 --horizon 3 --start edge --policies random,centre,goal,plan
        run $LOG/$CFG/ppo_jepa_${ARM}_edge $PY -m studies.stoch.rl --config $CFG --arm evaluate --policy $LOG/$CFG/ppo_jepa_$ARM/model.pkl \
            --wm $R/$CFG/$ARM/model_probe.npz --eval-start edge --out $LOG/$CFG/ppo_jepa_$ARM
    done
    for OBS in pixels latent; do
        run $LOG/$CFG/ppo_dream_${OBS}_edge $PY -m studies.stoch.rl --config $CFG --arm evaluate --policy $LOG/$CFG/ppo_dream_$OBS/model.pkl \
            --wm $R/$CFG/flow/model.npz --wm-code studies/stoch/flow_recipe --eval-start edge --out $LOG/$CFG/ppo_dream_$OBS
    done
    run $LOG/$CFG/ppo_real_edge $PY -m studies.stoch.rl --config $CFG --arm evaluate --policy $LOG/$CFG/ppo_real/model.pkl \
        --eval-start edge --out $LOG/$CFG/ppo_real
    echo "EDGE_DONE $CFG $(date -u +%H:%M:%S)"
done
echo LANE_G_DONE
