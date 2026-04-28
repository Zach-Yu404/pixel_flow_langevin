#!/usr/bin/env bash
# Launch larger_box sweep: 32 configs split into 8 chunks of 4 (one chunk per GPU).
# Each GPU loads model once, then runs its 4 configs serially.
#
# Logs: results/_chunk_gpu<i>.log
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/home/nvidia/miniconda/envs/pixelflow/bin/python"
NGPU=8

TOTAL=$("$PY" -c "from sweep_configs import CONFIGS; print(len(CONFIGS))")
CHUNK=$(( (TOTAL + NGPU - 1) / NGPU ))   # ceil(total/8)
echo "[launch] $TOTAL configs, $NGPU GPUs, chunk size = $CHUNK (last chunk may be smaller)"

mkdir -p "$HERE/results"
pids=()
for ((g=0; g<NGPU; g++)); do
  start=$(( g * CHUNK ))
  end=$(( start + CHUNK ))
  if [ "$end" -gt "$TOTAL" ]; then end=$TOTAL; fi
  if [ "$start" -ge "$TOTAL" ]; then continue; fi
  log="$HERE/results/_chunk_gpu${g}.log"
  echo "[launch] gpu=$g  configs[${start}:${end}]  log=$log"
  cd "$HERE"
  nohup "$PY" run_chunk.py --start "$start" --end "$end" --gpu "$g" \
      > "$log" 2>&1 &
  pids+=($!)
done

echo "[launch] ${#pids[@]} chunk jobs running; tail $HERE/results/_chunk_gpu*.log to follow"
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=$(( fail + 1 )); fi
done
echo "[launch] done. failures=$fail"

if [ "$fail" -eq 0 ]; then
  echo "[launch] running aggregate.py"
  "$PY" "$HERE/aggregate.py"
fi
