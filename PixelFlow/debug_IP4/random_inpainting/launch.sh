#!/usr/bin/env bash
# 18 configs × 8 GPUs, ~3 chunks per GPU (last chunks may be partial).
# NOTE: $GROUPS is a bash builtin readonly array, do not use as variable name.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/home/nvidia/miniconda/envs/pixelflow/bin/python"
GPUS=(0 1 2 3 4 5 6 7)
NGPU=${#GPUS[@]}
TOTAL=$("$PY" -c "from oat_configs import CONFIGS; print(len(CONFIGS))")
CHUNK=$(( (TOTAL + NGPU - 1) / NGPU ))   # ceil(18/8) = 3
echo "[launch] $TOTAL configs / ${NGPU} GPUs, chunk=$CHUNK"

mkdir -p "$HERE/runs/_logs"
pids=()
for ((i=0; i<NGPU; i++)); do
  g=${GPUS[$i]}
  start=$(( i * CHUNK )); end=$(( start + CHUNK ))
  if [ "$end" -gt "$TOTAL" ]; then end=$TOTAL; fi
  if [ "$start" -ge "$TOTAL" ]; then continue; fi
  log="$HERE/runs/_logs/_chunk_gpu${g}.log"
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
