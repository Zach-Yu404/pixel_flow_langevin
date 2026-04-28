#!/usr/bin/env bash
# Launch jobs as a per-GPU serial queue so each GPU runs one job at a time.
# Total jobs = N_configs * 5 groups; jobs are split round-robin into 8 queues.
#
# Usage:
#   ./launch.sh                                    # all 25 configs (base + CFG sweep)
#   ./launch.sh A0_reference_WINNER3 E1_W1+lam20   # subset
#   CFGS_FILTER="_cfg" ./launch.sh                 # only CFG variants (skip base)
#
# Logs: <OUT_BASE>/<cfg>/group_<g>/run.log
# Per-GPU queue progress: <OUT_BASE>/_queue_gpu<i>.log
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_BASE="$HERE/output"
PY="/home/nvidia/miniconda/envs/pixelflow/bin/python"
NGPU=8

ALL_CFGS=(
  "A0_reference_WINNER3"
  "E1_W1+lam20"
  "C1_W1+hx_multistage"
  "H1_W1+X4+noise0.3"
  "S3_stage3_only"
  "A0_reference_WINNER3_cfg1" "A0_reference_WINNER3_cfg2" "A0_reference_WINNER3_cfg3" "A0_reference_WINNER3_cfg4"
  "E1_W1+lam20_cfg1"          "E1_W1+lam20_cfg2"          "E1_W1+lam20_cfg3"          "E1_W1+lam20_cfg4"
  "C1_W1+hx_multistage_cfg1"  "C1_W1+hx_multistage_cfg2"  "C1_W1+hx_multistage_cfg3"  "C1_W1+hx_multistage_cfg4"
  "H1_W1+X4+noise0.3_cfg1"    "H1_W1+X4+noise0.3_cfg2"    "H1_W1+X4+noise0.3_cfg3"    "H1_W1+X4+noise0.3_cfg4"
  "S3_stage3_only_cfg1"       "S3_stage3_only_cfg2"       "S3_stage3_only_cfg3"       "S3_stage3_only_cfg4"
)
if [ "$#" -gt 0 ]; then CFGS=("$@"); else CFGS=("${ALL_CFGS[@]}"); fi

# Optional substring filter (e.g. CFGS_FILTER="_cfg" skips base configs)
CFGS_FILTER="${CFGS_FILTER:-}"
if [ -n "$CFGS_FILTER" ]; then
  KEEP=()
  for c in "${CFGS[@]}"; do
    [[ "$c" == *"$CFGS_FILTER"* ]] && KEEP+=("$c")
  done
  CFGS=("${KEEP[@]}")
fi

mkdir -p "$OUT_BASE"

# Build per-GPU job queues (round-robin assignment).
declare -a QUEUE
for ((g=0; g<NGPU; g++)); do QUEUE[g]=""; done
i=0
total_jobs=0
for cfg in "${CFGS[@]}"; do
  for grp in 0 1 2 3 4; do
    gpu=$(( i % NGPU ))
    QUEUE[gpu]+="$cfg|$grp "
    i=$(( i + 1 ))
    total_jobs=$(( total_jobs + 1 ))
  done
done

echo "[launch] ${#CFGS[@]} configs x 5 groups = $total_jobs jobs, distributed across $NGPU GPUs"
for ((g=0; g<NGPU; g++)); do
  n=$(echo "${QUEUE[g]}" | wc -w)
  echo "[launch] gpu $g queue: $n jobs"
done

# Spawn one bash subshell per GPU to run its queue serially.
queue_pids=()
for ((g=0; g<NGPU; g++)); do
  qlog="$OUT_BASE/_queue_gpu${g}.log"
  (
    for item in ${QUEUE[g]}; do
      cfg="${item%|*}"; grp="${item#*|}"
      log_dir="$OUT_BASE/$cfg/group_$grp"
      mkdir -p "$log_dir"
      log="$log_dir/run.log"
      echo "[gpu $g] $(date +%H:%M:%S) START cfg=$cfg group=$grp"
      "$PY" "$HERE/run_eval.py" --config "$cfg" --group "$grp" --gpu "$g" \
          > "$log" 2>&1
      rc=$?
      echo "[gpu $g] $(date +%H:%M:%S) DONE  cfg=$cfg group=$grp rc=$rc"
    done
  ) > "$qlog" 2>&1 &
  queue_pids+=($!)
done

echo "[launch] all queues started; tail $OUT_BASE/_queue_gpu*.log to follow"
fail=0
for pid in "${queue_pids[@]}"; do
  if ! wait "$pid"; then fail=$(( fail + 1 )); fi
done
echo "[launch] done. queue-failures=$fail"

if [ "$fail" -eq 0 ]; then
  echo "[launch] running aggregate.py"
  "$PY" "$HERE/aggregate.py"
fi
