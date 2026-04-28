#!/usr/bin/env python
"""Sweep CG iterations and Langevin steps — test if fewer = sharper."""
import sys, os, math, copy, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch, torch.nn.functional as F, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor
from inpaintingStart import get_operator
from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, compute_sigma_tau, direct_estimate_x1,
    make_Ak_fns, make_velocity_fn, sample_block_noise,
    principle_langevin_sample,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results_cg")
os.makedirs(RESULT_DIR, exist_ok=True)

def hf_energy(x):
    fft = torch.fft.fft2(x); mag = fft.abs()
    H, W = x.shape[-2:]
    total = mag.pow(2).sum().item()
    lf = mag[:, :, :H//4, :W//4].pow(2).sum().item()
    return 1 - lf / max(total, 1e-12)

def run(model, config, gt, y, operator, sigma_n, num_langevin=10, cg_max_iter=50, label=""):
    B = gt.shape[0]; seed_everything(20000120)
    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=config.scheduler.num_stages, gamma=-1/3)
    prompt_embeds = torch.tensor([10]*B, dtype=torch.int32, device=DEVICE)
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    h, w = 32, 32
    x1_k = torch.randn(B, 3, h, w, device=DEVICE)
    eps_k = randn_tensor((B, 3, h, w), device=DEVICE); lat = eps_k.clone()
    t0 = time.time()
    for si in range(4):
        sc = copy.deepcopy(scheduler)
        sc.set_timesteps(10, si, device=DEVICE, shift=1.0)
        sk = float(sc.start_t[si]); ek = float(sc.end_t[si])
        if si > 0:
            h *= 2; w *= 2
            lat = F.interpolate(lat, size=(h, w), mode="nearest")
            ost = sc.original_start_t[si]; gam = sc.gamma
            al = 1/(math.sqrt(1-(1/gam))*(1-ost)+ost)
            be = al*(1-ost)/math.sqrt(-gam)
            noise = sample_block_noise(sc, B, 3, h, w).to(device=DEVICE, dtype=lat.dtype)
            lat = al*lat + be*noise
            x1_k = F.interpolate(x1_k, size=(h, w), mode="nearest")
            eps_k = (lat - sk*apply_G(x1_k, stage_idx=si))/max(1-sk, 1e-8)
        Ak, ATk = make_Ak_fns(operator, y, (B, 3, h, w), DEVICE)
        st = torch.tensor([h//model.patch_size], dtype=torch.int32, device=DEVICE)
        pe = get_2d_rotary_pos_embed(embed_dim=model.attention_head_dim,
            crops_coords=((0, 0), (h//model.patch_size, w//model.patch_size)),
            grid_size=(h//model.patch_size, w//model.patch_size), device=DEVICE, output_type="pt")
        rope = torch.stack(pe, -1)
        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx].to(DEVICE))
            sig = compute_sigma_tau(tau, sk, ek)
            xtau = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k
            vfn = make_velocity_fn(model, T, prompt_embeds, st, rope, False, 0.0, si)
            with torch.no_grad(): mu = vfn(xtau)
            x1_k = direct_estimate_x1(xtau - tau*mu, xtau + (1-tau)*mu, sk, ek).detach()
            if sig > 1e-8: eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig
            if sig >= 0.01:
                x1_k, eps_k, _, _ = principle_langevin_sample(
                    x1_init=x1_k, eps_init=eps_k, tau=tau, s_k=sk, e_k=ek,
                    velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                    y=y, sigma_n=sigma_n, h_x=0.1, h_epsilon=0.01,
                    lambda_x=0.01, lambda_reg=50.0, rho_s=1.0, rho_e=1.0,
                    cg_tol=1e-5, cg_max_iter=cg_max_iter,
                    num_Langevin=num_langevin,
                    device=DEVICE, stage_idx=si, x1_init_mode="model",
                    noise_scale=0.0, return_traj=True, record_every=99)
            lat = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k
    elapsed = time.time() - t0
    xf = x1_k.detach()
    psnr_all = -10*torch.log10(((xf - gt)**2).mean()).item()
    psnr_obs = -10*torch.log10(((mask[0:1]*xf - mask[0:1]*gt)**2).mean()).item()
    res = (y - Ak(xf)).pow(2).sum().item()
    hf_u = hf_energy(xf * (1-mask[0:1]))
    return xf.cpu(), psnr_obs, psnr_all, res, elapsed, hf_u

def main():
    print("Loading...", flush=True)
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    if os.path.exists(pt):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        gt = d["gt"][:2].to(DEVICE)
    else:
        from torchvision import transforms, datasets
        from ms_posterior_sampling_article_version_final_utils import center_crop_arr
        data_dir = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/train/"
        tf = transforms.Compose([transforms.Lambda(lambda p: center_crop_arr(p,256)),
              transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3,inplace=True)])
        ds = datasets.ImageFolder(data_dir, tf); torch.manual_seed(20000120)
        ci = [i for i,(_, y) in enumerate(ds.samples) if int(y)==10]
        perm = torch.randperm(len(ci))[:2]
        gt = torch.stack([ds[ci[i]][0] for i in perm.tolist()]).to(DEVICE)
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()
    print("Model loaded.\n", flush=True)

    sigma_n = 0.05
    results = {}
    for mask_type, op_kw in [
        ("box", dict(mask_type="box", mask_len_range=(80,160), mask_prob_range=None)),
    ]:
        operator = get_operator("inpainting", resolution=256, device=DEVICE, sigma=sigma_n, **op_kw)
        y = operator(gt).detach()
        mask = operator.get_mask(x=gt).float().to(DEVICE)
        gt_hf = hf_energy(gt * (1-mask[0:1]))
        print(f"  {mask_type.upper()} (GT HF_unobs={gt_hf:.3f})", flush=True)

        configs = [
            # baseline
            ("L10_CG50", 10, 50),
            # reduce CG iterations
            ("L10_CG1",  10, 1),
            ("L10_CG2",  10, 2),
            ("L10_CG3",  10, 3),
            ("L10_CG5",  10, 5),
            ("L10_CG10", 10, 10),
            ("L10_CG20", 10, 20),
            # reduce Langevin steps
            ("L1_CG50",  1,  50),
            ("L2_CG50",  2,  50),
            ("L3_CG50",  3,  50),
            ("L5_CG50",  5,  50),
            # reduce both
            ("L3_CG3",   3,  3),
            ("L3_CG5",   3,  5),
            ("L5_CG5",   5,  5),
            ("L5_CG10",  5,  10),
            ("L2_CG3",   2,  3),
            ("L1_CG5",   1,  5),
            ("L1_CG10",  1,  10),
            ("L1_CG1",   1,  1),
        ]

        for name, nl, cg in configs:
            print(f"  {name}...", end=" ", flush=True)
            xf, po, pa, res, t, hf = run(model, config, gt, y, operator, sigma_n,
                                          num_langevin=nl, cg_max_iter=cg)
            results[f"{mask_type}_{name}"] = (xf, po, pa, res, t, hf)
            print(f"res={res:.0f}  PSNR_obs={po:.2f}  PSNR_all={pa:.2f}  "
                  f"HF={hf:.3f}  t={t:.0f}s", flush=True)

    # Summary
    print(f"\n{'='*75}", flush=True)
    print(f"{'Config':<20} {'Langevin':>8} {'CG':>4} {'Res':>7} {'PSNR_obs':>9} "
          f"{'PSNR_all':>9} {'HF_unobs':>9} {'Time':>6}", flush=True)
    print("-"*75, flush=True)
    for k in sorted(results.keys()):
        xf, po, pa, res, t, hf = results[k]
        parts = k.split("_", 1)[1]  # remove mask prefix
        nl = parts.split("_CG")[0].replace("L","")
        cg = parts.split("_CG")[1]
        print(f"{parts:<20} {nl:>8} {cg:>4} {res:>7.0f} {po:>9.2f} "
              f"{pa:>9.2f} {hf:>9.3f} {t:>5.0f}s", flush=True)

    # Visualization
    keys = sorted(results.keys())
    n = len(keys)
    cols = min(7, n+1); rows = (n+1+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2))
    if rows == 1: axes = axes[np.newaxis, :]
    axes = axes.flatten()
    img_gt = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    axes[0].imshow(img_gt); axes[0].set_title("GT", fontsize=7); axes[0].axis("off")
    for i, k in enumerate(keys):
        xf, po, pa, res, t, hf = results[k]
        img = (xf[0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        short = k.replace("box_","")
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"{short}\n{pa:.1f}dB hf={hf:.2f}", fontsize=5)
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle("CG iterations & Langevin steps sweep", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/cg_sweep.png", dpi=150)
    plt.close()
    print(f"Saved cg_sweep.png", flush=True)
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
