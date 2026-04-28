#!/bin/bash
# Master script — distribute 5 groups × 2 tasks across 8 GPUs in 2 rounds.
# Each (group, task) takes ~64 min on one GPU at B=4.
# Round 1: 8 jobs on GPU 0..7, Round 2: 2 jobs on GPU 0..1.

set -e
PY="/home/nvidia/miniconda/envs/pixelflow/bin/python"
SCRIPT="/home/nvidia/Zach/MSFlow/PixelFlow/debug_IP4/run_validation_eval.py"
LOG="/home/nvidia/Zach/MSFlow/PixelFlow/debug_IP4/eval_logs"
mkdir -p "$LOG"

# Round 1: groups 0-3 box on GPU 0-3, groups 0-3 random on GPU 4-7
for g in 0 1 2 3; do
    $PY $SCRIPT --group $g --task box    --gpu $g          > $LOG/g${g}_box.log    2>&1 &
    $PY $SCRIPT --group $g --task random --gpu $((g + 4))  > $LOG/g${g}_random.log 2>&1 &
done
echo "Round 1 launched (8 jobs)"
wait
echo "Round 1 complete"

# Round 2: group 4 box on GPU 0, group 4 random on GPU 1
$PY $SCRIPT --group 4 --task box    --gpu 0  > $LOG/g4_box.log    2>&1 &
$PY $SCRIPT --group 4 --task random --gpu 1  > $LOG/g4_random.log 2>&1 &
echo "Round 2 launched (2 jobs)"
wait
echo "All 10 jobs done"
