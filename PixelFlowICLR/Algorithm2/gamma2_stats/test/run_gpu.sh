#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
LOG="gamma2_stats/test/g2test_gpu$1.log"
for i in $(seq 1 300); do
  PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=$1 ~/anaconda3/envs/pixelflow/bin/python gamma2_stats/test/run_gamma2_test.py >> "$LOG" 2>&1 && exit 0
  echo "[wrapper] attempt $i failed, retry 60s" >> "$LOG"; sleep 60
done
echo "[wrapper] GAVE UP" >> "$LOG"
