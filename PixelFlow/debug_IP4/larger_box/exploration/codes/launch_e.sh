#!/usr/bin/env bash
# Launch exploration sweep across NGPU GPUs.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # exploration/scripts
EXP="$(dirname "$HERE")"                                  # exploration
PY="/home/nvidia/miniconda/envs/pixelflow/bin/python"
NGPU="${NGPU:-8}"

TOTAL=$("$PY" -c "import sys; sys.path.insert(0, '$HERE'); from sweep_configs_e import CONFIGS; print(len(CONFIGS))")
CHUNK=$(( (TOTAL + NGPU - 1) / NGPU ))   # ceil(total/NGPU)
echo "[launch_e] $TOTAL configs, $NGPU GPUs, chunk size = $CHUNK"

mkdir -p "$EXP/runs"
pids=()
for ((g=0; g<NGPU; g++)); do
  start=$(( g * CHUNK ))
  end=$(( start + CHUNK ))
  if [ "$end" -gt "$TOTAL" ]; then end=$TOTAL; fi
  if [ "$start" -ge "$TOTAL" ]; then continue; fi
  log="$EXP/runs/_chunk_gpu${g}.log"
  echo "[launch_e] gpu=$g  configs[${start}:${end}]  log=$log"
  cd "$HERE"
  nohup "$PY" run_chunk_e.py --start "$start" --end "$end" --gpu "$g" \
      > "$log" 2>&1 &
  pids+=($!)
done

echo "[launch_e] ${#pids[@]} chunk jobs running; tail $EXP/runs/_chunk_gpu*.log to follow"
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=$(( fail + 1 )); fi
done
echo "[launch_e] done. failures=$fail"
