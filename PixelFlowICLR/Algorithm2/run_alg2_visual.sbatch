#!/bin/bash
#SBATCH -J alg2-visual
#SBATCH -A cbig-ece
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /sfs/ceph/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/logs/%x-%j.out

set -euo pipefail
export PYTHONHASHSEED=0   # same mask-seed pinning as run_alg2.sbatch
PY=/home/rqx6rs/anaconda3/envs/pixelflow/bin/python
cd /sfs/ceph/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2
echo "[job] $(date)  node=$(hostname)  gpu=$(nvidia-smi -L | head -1)"
srun --export=ALL "$PY" alg2_visual.py --out results --chunk 4
echo "[job] done $(date)"
