#!/bin/bash
#SBATCH -J onestep-visual
#SBATCH -A cbig-ece
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /sfs/ceph/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/logs/%x-%j.out

set -euo pipefail
PY=/home/rqx6rs/anaconda3/envs/pixelflow/bin/python
cd /sfs/ceph/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR
echo "[job] $(date)  node=$(hostname)  gpu=$(nvidia-smi -L | head -1)"
srun "$PY" onestep_visual.py --out results/onestep_mse --chunk 4
echo "[job] done $(date)"
