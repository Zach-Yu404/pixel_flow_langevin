"""Smoke test: run IP4 baseline alone, confirm equality to IP3 baseline_L10."""
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

DEVICE = "cuda:0"

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

print("Running IP4 baseline config…", flush=True)
xf, po, pa, res, t, hf = run_ip4(
    model, config, gt, y, op, sigma_n, DEVICE)
print(f"res={res:.1f}  psnr_obs={po:.2f}  psnr_all={pa:.2f}  hf={hf:.3f}  t={t:.0f}s",
      flush=True)
print("IP3 baseline_L10 reference: res=37  psnr_all=13.25  hf=0.584  t=155s",
      flush=True)
