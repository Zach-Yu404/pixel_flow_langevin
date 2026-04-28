#!/usr/bin/env bash
# 24 configs across 6 GPUs (1..6), 4 configs each, serial within a GPU.
# GPU 0 left alone (sweep_denoise.py running there). GPU 7 reserved for headroom.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/home/nvidia/miniconda/envs/pixelflow/bin/python"

GPUS=(1 2 3 4 5 6)
NGPU=${#GPUS[@]}
TOTAL=$("$PY" -c "from sweep2_configs import CONFIGS; print(len(CONFIGS))")
CHUNK=$(( (TOTAL + NGPU - 1) / NGPU ))
echo "[launch] $TOTAL configs / ${NGPU} GPUs, chunk=$CHUNK"

mkdir -p "$HERE/runs"
pids=()
for ((i=0; i<NGPU; i++)); do
  g=${GPUS[$i]}
  start=$(( i * CHUNK ))
  end=$(( start + CHUNK ))
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
