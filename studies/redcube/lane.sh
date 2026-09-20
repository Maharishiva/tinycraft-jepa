#!/bin/bash
# One GPU lane of the overnight work (Fred, 03:00 MSK 2026-09-18; re-ordered at 04:00 after his LeWM message), one
# job per GPU so the 575 s budgets stay honest.  Stages:
#   1. wait for whatever the previous driver left running on this lane's configurations to finish
#   2. MAIN configurations: data + flow + JEPA + page benchmark (idempotent), then the extra arms (seven + lewm6, and
#      the full lewm on FULL) with the all-arms benchmark
#   3. MAIN: JEPA seeds 1, 2 (was the gray_frame_wind collapse luck?) and batch 256 / 512, benchmark redone
#   4. REBENCH configurations lacking the pixel probe: page benchmark redone
#   5. OTHER configurations: the extra arms
#   bash studies/redcube/lane.sh 1 "gray_frame_still gray_episode_still" gray_frame_still "off_wind off_still" "off_wind off_still"
GPU=$1
MAIN=$2
FULL=$3
REBENCH=$4
OTHER=$5
cd ~/personal/tinycraft-jepa
Q=runs/redcube/queue.log
ARMS="tile_ar tile_ar8 tile_ar_nodrop vit_factored tile_cls_ar tile_cls_ar8 tile_cls_factored lewm6"
train() { GPU=$GPU FIXED_STEPS=$3 bash studies/redcube/train_arms.sh $1 $2 >> $Q 2>&1 || echo "ARMS_FAILED $1 $2" >> $Q; }
evalarms() { rm -f runs/redcube/$1/allarms/bench.done; GPU=$GPU bash studies/redcube/eval_arms.sh $1 >> $Q 2>&1; }
while pgrep -f "studies/redcube/(train|eval)_(config|arms)\.sh ($(echo $MAIN | tr ' ' '|'))" > /dev/null; do sleep 60; done
echo "LANE $GPU started $(date)" >> $Q
for CFG in $MAIN; do
    GPU=$GPU TRAIN_STEPS=5000 bash studies/redcube/train_config.sh $CFG >> $Q 2>&1 || echo "TRAIN_FAILED $CFG" >> $Q
    GPU=$GPU bash studies/redcube/eval_config.sh $CFG >> $Q 2>&1
done
for CFG in $MAIN; do
    [ -d /dev/shm/redcube/$CFG/_tile_ar ] && mv /dev/shm/redcube/$CFG/_tile_ar /dev/shm/redcube/$CFG/tile_ar
    for ARM in $ARMS; do train $CFG $ARM; done
    [ "$CFG" = "$FULL" ] && train $CFG lewm
    evalarms $CFG
done
for CFG in $MAIN; do
    for ARM in tile_factored_s1 tile_factored_s2; do train $CFG $ARM 8500; done
    for ARM in tile_factored_b256 tile_factored_b512; do train $CFG $ARM; done
    evalarms $CFG
done
for CFG in $REBENCH; do
    grep -q "pixel probe" runs/redcube/$CFG/bench.log 2>/dev/null && continue
    echo "REBENCH $CFG pixel probe $(date)" >> $Q
    rm -f runs/redcube/$CFG/bench.done runs/redcube/$CFG/probe_scenes.done
    GPU=$GPU bash studies/redcube/eval_config.sh $CFG >> $Q 2>&1
done
for CFG in $OTHER; do
    for ARM in $ARMS; do train $CFG $ARM; done
    evalarms $CFG
done
echo "LANE $GPU done $(date)" >> $Q
