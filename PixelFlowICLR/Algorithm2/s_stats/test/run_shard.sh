#!/bin/bash
cd /CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/s_stats/test
for i in $(seq 1 40); do
  SHARD="$1" CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
    ~/anaconda3/envs/pixelflow/bin/python run_s4_test.py >> "s4_$2.log" 2>&1
  grep -q "S4 SHARD DONE" "s4_$2.log" && exit 0
  echo "[wrapper] attempt $i failed, retry 30s" >> "s4_$2.log"; sleep 30
done
echo "S4 SHARD GAVE UP" >> "s4_$2.log"
