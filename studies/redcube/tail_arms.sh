#!/bin/bash
# After a lane is done: the batch arms on the per-frame gray configurations (they OOMed there at the default pool
# before the 92% pool went in; train_arms.sh skips arms whose model exists), then the all-arms benchmark redone.
#   bash studies/redcube/tail_arms.sh 0 "gray_frame_wind"
GPU=$1
CFGS=$2
cd ~/personal/tinycraft-jepa
Q=runs/redcube/queue.log
until grep -q "LANE $GPU done" $Q; do sleep 300; done
echo "TAIL $GPU started $(date)" >> $Q
for CFG in $CFGS; do
    for ARM in tile_factored_b256 tile_factored_b512; do
        GPU=$GPU bash studies/redcube/train_arms.sh $CFG $ARM >> $Q 2>&1 || echo "ARMS_FAILED $CFG $ARM" >> $Q
    done
    rm -f runs/redcube/$CFG/allarms/bench.done
    GPU=$GPU bash studies/redcube/eval_arms.sh $CFG >> $Q 2>&1
done
echo "TAIL $GPU done $(date)" >> $Q
