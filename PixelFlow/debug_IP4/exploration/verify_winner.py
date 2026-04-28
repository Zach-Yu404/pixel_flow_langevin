"""Verify IP4 X4 winner profile works end-to-end."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
from omegaconf import OmegaConf
from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "debug_IP4"))
from ms_sampler_v5 import run_ip4, hf_energy

DEVICE = "cuda:3"

pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
d = torch.load(pt, map_location="cpu", weights_only=False)
gt = d["gt"][:2].to(DEVICE)

config = OmegaConf.load("pretrained_models/c2img/config.yaml")
model = config_utils.instantiate_from_config(config.model).to(DEVICE)
ckpt = torch.load("pretrained_models/c2img/model.pt",
                  map_location="cpu", weights_only=False)
model.load_state_dict(ckpt, strict=True); model.eval()
print("Model loaded.", flush=True)

sigma_n = 0.05
op = get_operator("inpainting", resolution=256, device=DEVICE, sigma=sigma_n,
                  mask_type="box", mask_len_range=(80, 160), mask_prob_range=None)
y = op(gt).detach()
mask = op.get_mask(x=gt).float().to(DEVICE)
gt_hf = hf_energy(gt * (1 - mask[0:1]))
print(f"GT_HF={gt_hf:.3f}", flush=True)

print("\n=== X4 winner profile ===", flush=True)
xf, po, pa, res, t, hf = run_ip4(
    model, config, gt, y, op, sigma_n, DEVICE,
    h_x=[0.1, 0.1, 0.1, 0.7],
    terminal_replace_weight=1.0,
)
print(f"X4  : PSNR_all={pa:.2f}  PSNR_obs={po:.2f}  HF={hf:.3f}  |Δ|={abs(hf-gt_hf):.3f}  res={res:.1f}  t={t:.0f}s",
      flush=True)

print("\n=== Baseline for reference ===", flush=True)
xf2, po2, pa2, res2, t2, hf2 = run_ip4(
    model, config, gt, y, op, sigma_n, DEVICE)
print(f"base: PSNR_all={pa2:.2f}  PSNR_obs={po2:.2f}  HF={hf2:.3f}  |Δ|={abs(hf2-gt_hf):.3f}  res={res2:.1f}  t={t2:.0f}s",
      flush=True)

print("\n=== Summary ===", flush=True)
print(f"X4 gain: PSNR {pa-pa2:+.2f} dB, HF |Δ| {abs(hf-gt_hf)-abs(hf2-gt_hf):+.3f}, res {res-res2:+.1f}", flush=True)
