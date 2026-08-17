#!/bin/bash
#SBATCH -J full-ip-compare
#SBATCH -A cbig-ece
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /sfs/ceph/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/logs/%x-%j.out

set -euo pipefail
export PYTHONHASHSEED=0
PY=/home/rqx6rs/anaconda3/envs/pixelflow/bin/python
cd /sfs/ceph/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2
echo "[job] $(date)  node=$(hostname)  gpu=$(nvidia-smi -L | head -1)"
echo "[smoke] 1 task / 1 image / S=2 first"
srun --export=ALL "$PY" full_ip_compare.py --smoke --out results/full_ip_smoke
echo "[smoke] passed; full run"
srun --export=ALL "$PY" full_ip_compare.py --out results/full_ip
echo "[job] done $(date)"
