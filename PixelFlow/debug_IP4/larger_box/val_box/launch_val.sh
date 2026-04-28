#!/usr/bin/env bash
# Launch 15 (group, config) experiments across 8 GPUs.
# Each GPU runs at most 2 jobs sequentially.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/home/nvidia/miniconda/envs/pixelflow/bin/python"

CONFIGS=(
  "pareto_dual_king_srs1e-4__fd1_tr1"
  "LPIPS_king_srs1e-4__fd1_tr1"
  "balanced_perceptual_srs1e-4__fd1_tr1"
)
# NOTE: $GROUPS is a bash built-in array (user's Unix groups). Use a different name.
VAL_GROUPS=(0 1 2 3 4)

# build flat job list: 15 (group, config) tuples
JOBS=()
for g in "${VAL_GROUPS[@]}"; do
  for c in "${CONFIGS[@]}"; do
    JOBS+=("$g|$c")
  done
done
echo "[launch] ${#JOBS[@]} jobs to dispatch"

mkdir -p "$HERE/runs/_logs"
pids=()

# Assign job idx i to GPU (i % 8). Each GPU runs its jobs sequentially in a single bash subshell.
NGPU=8
for ((gpu=0; gpu<NGPU; gpu++)); do
  (
    for ((i=gpu; i<${#JOBS[@]}; i+=NGPU)); do
      job="${JOBS[$i]}"
      g="${job%%|*}"
      c="${job##*|}"
      log="$HERE/runs/_logs/job${i}_g${g}_${c}.log"
      echo "[gpu=$gpu] -> job $i: group=$g config=$c (log=$log)"
      "$PY" "$HERE/run_one.py" --group "$g" --config "$c" --gpu "$gpu" \
        > "$log" 2>&1
    done
  ) &
  pids+=($!)
done

echo "[launch] launched ${#pids[@]} GPU subshells, waiting..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=$(( fail + 1 )); fi
done
echo "[launch] done. failures=$fail"
