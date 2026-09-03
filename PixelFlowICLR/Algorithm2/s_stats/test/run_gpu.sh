#!/bin/bash
# usage: run_gpu.sh <gpu>
cd /CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/s_stats/test
for i in $(seq 1 60); do
  SHARD="pooled_all,spectral_all,pooled_class,spectral_class" \
    CUDA_VISIBLE_DEVICES=$1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
    ~/anaconda3/envs/pixelflow/bin/python run_s4_test.py >> "s4_gpu$1.log" 2>&1
  grep -q "S4 SHARD DONE" "s4_gpu$1.log" && exit 0
  echo "[wrapper] attempt $i failed, retry 30s" >> "s4_gpu$1.log"; sleep 30
done
