#!/bin/bash
cd /CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/s_stats
for i in $(seq 1 20); do
  CUDA_VISIBLE_DEVICES=2 ~/anaconda3/envs/pixelflow/bin/python compute_s_stats.py >> sstats.log 2>&1
  grep -q "S STATS DONE" sstats.log && exit 0
  echo "[wrapper] attempt $i failed, retrying in 30s" >> sstats.log; sleep 30
done
echo "S STATS GAVE UP" >> sstats.log
