#!/usr/bin/env bash
# 36 configs across 8 GPUs (1..6 only — leave 0 and 7 free), serial within GPU.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/home/nvidia/miniconda/envs/pixelflow/bin/python"
GPUS=(0 1 2 3 4 5 6 7)
NGPU=${#GPUS[@]}
TOTAL=$("$PY" -c "from fv_configs import CONFIGS; print(len(CONFIGS))")
CHUNK=$(( (TOTAL + NGPU - 1) / NGPU ))   # 36/8 = 5
echo "[launch] $TOTAL configs / ${NGPU} GPUs, chunk=$CHUNK"
mkdir -p "$HERE/runs"
pids=()
for ((i=0; i<NGPU; i++)); do
  g=${GPUS[$i]}
  start=$(( i * CHUNK )); end=$(( start + CHUNK ))
  if [ "$end" -gt "$TOTAL" ]; then end=$TOTAL; fi
  if [ "$start" -ge "$TOTAL" ]; then continue; fi
  log="$HERE/runs/_chunk_gpu${g}.log"
  echo "[launch] gpu=$g  CONFIGS[${start}:${end}]  log=$log"
  cd "$HERE"
  nohup "$PY" run_chunk.py --start "$start" --end "$end" --gpu "$g" \
      > "$log" 2>&1 &
  pids+=($!)
done
echo "[launch] ${#pids[@]} jobs running"
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=$(( fail + 1 )); fi
done
echo "[launch] done. failures=$fail"
