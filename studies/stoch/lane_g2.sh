#!/bin/bash
# Round 2, second pass of the edge-start lane with a larger pool for the cells that ran out of memory while GPU 0
# was shared four ways (multigpu-vm, GPU 0, after lane_g.sh; done cells are skipped).
#   (setsid nohup bash studies/stoch/lane_g2.sh > runs/stoch/lane_g2.log 2>&1 < /dev/null &)
cd ~/personal/tinycraft-jepa
until grep -q LANE_G_DONE runs/stoch/lane_g.log; do sleep 60; done
sed -e 's/^until grep -q LATENT_CHECK_EXIT.*$//' -e 's/MEM_FRACTION=.25/MEM_FRACTION=.4/' -e 's/LANE_G_DONE/LANE_G2_DONE/' studies/stoch/lane_g.sh > /tmp/lane_g2_body.sh
bash /tmp/lane_g2_body.sh
