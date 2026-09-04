#!/bin/bash
# usage: gamma2_stats/run_gpu.sh <gpu>   — one shard per GPU, relaunched on ceph EIO until it prints DONE
cd "$(dirname "$0")/.." || exit 1
LOG="gamma2_stats/g2_gpu$1.log"
for i in $(seq 1 2000); do
  PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=$1 ~/anaconda3/envs/pixelflow/bin/python gamma2_stats/compute_gamma2_stats.py >> "$LOG" 2>&1 && exit 0
  echo "[wrapper] attempt $i failed, retry 60s" >> "$LOG"; sleep 60
done
echo "[wrapper] GAVE UP" >> "$LOG"
