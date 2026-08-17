#!/bin/bash
#SBATCH -J dbg-box-confirm
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
for IMG in breastplate_armor crane_structure ibex_horns junco lakeside_beach sea_anemone shetland_sheepdog; do
  echo "===== $IMG ====="
  srun "$PY" debug_box_h0.py --image "$IMG" --anchors 0.0 25.0 \
    --out "results/debug_box_confirm/$IMG"
done
echo "[job] done $(date)"
