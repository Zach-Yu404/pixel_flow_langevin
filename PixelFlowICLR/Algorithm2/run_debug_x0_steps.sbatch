#!/bin/bash
#SBATCH -J dbg-x0steps
#SBATCH -A cbig-ece
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /sfs/ceph/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/logs/%x-%j.out
set -euo pipefail
export PYTHONHASHSEED=0
PY=/home/rqx6rs/anaconda3/envs/pixelflow/bin/python
cd /sfs/ceph/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2
echo "[job] $(date) node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
srun "$PY" debug_box_h0.py --x0_steps 1 3 5 --gamma2_scales 1.0 0.0 \
  --out results/debug_box_x0steps
echo "[job] done $(date)"
