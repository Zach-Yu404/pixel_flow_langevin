#!/bin/bash
cd /CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/s_stats/test
for i in $(seq 1 60); do
  SHARD="pooled_all,spectral_all,pooled_class,spectral_class" \
    CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
    ~/anaconda3/envs/pixelflow/bin/python run_s4_test.py >> s4_single.log 2>&1
  grep -q "S4 SHARD DONE" s4_single.log && exit 0
  echo "[wrapper] attempt $i failed, retry 30s" >> s4_single.log; sleep 30
done
echo "S4 GAVE UP" >> s4_single.log
