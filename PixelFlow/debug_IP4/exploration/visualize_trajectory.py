"""Trajectory visualizer — dumps x_1 at every (stage, ODE-step) checkpoint
for 4 key configs, plus eps-entropy curves. Shows exactly WHERE content
emerges or collapses across the sampling chain."""
import os, sys, time, math, copy, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch, torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from inpaintingStart import get_operator
from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, compute_sigma_tau, direct_estimate_x1,
    make_Ak_fns, make_velocity_fn, sample_block_noise,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "debug_IP4"))
from langevin_v5 import (
    principle_langevin_v5, dps_gradient_kick, terminal_replacement,
)

DEVICE = "cuda:1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_trajectory")
os.makedirs(OUT, exist_ok=True)


def _ps(val, k, default):
    if val is None: return default
    if isinstance(val, (list, tuple)): return val[k]
    return val


def denorm(x):
    return (x*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()


@torch.no_grad()
def run_with_traj(model, config, gt, y, operator, sigma_n, device,
                  h_x=0.1, h_epsilon=0.01, lambda_prox=0.0,
                  terminal_replace_weight=1.0, num_langevin=10,
                  lambda_reg=50.0, lambda_x=0.01,
                  seed=20000120, **_):
    """Minimal run_ip4 variant that records x1_after and eps_norm per ODE step."""
    B = gt.shape[0]
    seed_everything(seed); torch.manual_seed(seed)
    num_stages = int(config.scheduler.num_stages)
    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                    num_stages=num_stages, gamma=-1/3)
    pe_labels = torch.tensor([10]*B, dtype=torch.int32, device=device)
    mask_full = operator.get_mask(x=gt).float().to(device)
    if mask_full.shape[0] == 1 and B > 1:
        mask_full = mask_full.expand(B, -1, -1, -1)

    h = w = 256 // (2 ** (num_stages-1))
    shape = (B, 3, h, w)
    x1_k  = torch.randn(shape, device=device, dtype=torch.float32)
    eps_k = randn_tensor(shape, device=device, dtype=torch.float32)
    lat   = eps_k.clone()

    # trajectory records (upsample samples to 256 for consistent display)
    traj_x1 = []    # list of dicts {stage, step, x1_256 tensor, eps_norm float, tau}
    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        ode_steps = 10
        sc.set_timesteps(ode_steps, si, device=device, shift=1.0)
        sk = float(sc.start_t[si]); ek = float(sc.end_t[si])
        eff_si = si

        if si > 0:
            h *= 2; w *= 2
            lat = F.interpolate(lat, size=(h, w), mode="nearest")
            ost = sc.original_start_t[si]; gam = sc.gamma
            al = 1/(math.sqrt(1-(1/gam))*(1-ost)+ost); be = al*(1-ost)/math.sqrt(-gam)
            nz = sample_block_noise(sc, B, 3, h, w).to(device=device, dtype=lat.dtype)
            lat = al*lat + be*nz
            x1_k = F.interpolate(x1_k, size=(h, w), mode="nearest")
            eps_k = (lat - sk*apply_G(x1_k, stage_idx=eff_si))/max(1-sk, 1e-8)

        Ak, ATk = make_Ak_fns(operator, y, (B, 3, h, w), device)
        mask_k = mask_full if h == 256 else F.interpolate(mask_full, size=(h, w), mode="nearest")
        st = torch.tensor([h//model.patch_size], dtype=torch.int32, device=device)
        pe = get_2d_rotary_pos_embed(embed_dim=model.attention_head_dim,
              crops_coords=((0, 0), (h//model.patch_size, w//model.patch_size)),
              grid_size=(h//model.patch_size, w//model.patch_size), device=device, output_type="pt")
        rope = torch.stack(pe, -1)

        n_lang  = int(_ps(num_langevin, si, 10))
        hx_k    = float(_ps(h_x, si, 0.1))
        he_k    = float(_ps(h_epsilon, si, 0.01))
        lam_k   = float(_ps(lambda_reg, si, 50.0))

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx].to(device))
            sig = compute_sigma_tau(tau, sk, ek)
            xtau = apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si) + sig*eps_k
            vfn = make_velocity_fn(model, T, pe_labels, st, rope, False, 0.0, si)

            # warm_restart
            mu = vfn(xtau)
            x1_k = direct_estimate_x1(xtau - tau*mu, xtau + (1-tau)*mu, sk, ek).detach().clone()
            if sig > 1e-8:
                eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si))/sig

            if sig >= 0.01 and n_lang > 0:
                x1_k, eps_k = principle_langevin_v5(
                    x1_init=x1_k, eps_init=eps_k,
                    tau=tau, s_k=sk, e_k=ek,
                    velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                    y=y, sigma_n=sigma_n, h_x=hx_k, h_epsilon=he_k,
                    lambda_x=lambda_x, lambda_reg=lam_k,
                    rho_s=1.0, rho_e=1.0, cg_tol=1e-5, cg_max_iter=50,
                    num_Langevin=n_lang, device=device,
                    stage_idx=eff_si, x1_init_mode="model",
                    noise_scale=0.0, mask_k=mask_k, h_x_obs_ratio=1.0,
                    lambda_prox=lambda_prox)

            # Upsample x1_k snapshot to 256 for visualization
            x1_vis = x1_k if h == 256 else F.interpolate(x1_k, size=(256, 256), mode="nearest")
            traj_x1.append(dict(
                stage=si, step=step_idx, tau=tau,
                x1_img=x1_vis[0].detach().cpu(),
                eps_norm_per_px=float(eps_k.pow(2).mean().sqrt()),
                eps_L2=float(eps_k.norm()),
                eps_rel_L2=float(eps_k.norm() / math.sqrt(eps_k.numel())),
                res_box_mse=float(((1-mask_k) * (x1_k - gt if h == 256 else
                    F.interpolate(gt, size=(h, w)))).pow(2).mean()),
            ))
            lat = apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si) + sig*eps_k

    if terminal_replace_weight > 0:
        x1_k = terminal_replacement(x1_k, y, mask_full, terminal_replace_weight)
    # Final snapshot after terminal replace
    traj_x1.append(dict(stage=3, step=10, tau=1.0,
        x1_img=x1_k[0].detach().cpu(),
        eps_norm_per_px=float(eps_k.pow(2).mean().sqrt()),
        eps_L2=float(eps_k.norm()),
        eps_rel_L2=float(eps_k.norm() / math.sqrt(eps_k.numel())),
        res_box_mse=float(((1-mask_full) * (x1_k - gt)).pow(2).mean())))
    return traj_x1, x1_k.detach()


def main():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    gt = torch.load(pt, map_location="cpu", weights_only=False)["gt"][:2].to(DEVICE)

    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    sigma_n = 0.05
    operator = get_operator("inpainting", resolution=256, device=DEVICE,
                             sigma=sigma_n, mask_type="box",
                             mask_len_range=(80, 160), mask_prob_range=None)
    y = operator(gt).detach()
    mask = operator.get_mask(x=gt).float().to(DEVICE)

    configs = [
        ("baseline",    dict(h_epsilon=0.01,  h_x=0.1)),
        ("W1_heps0.001",dict(h_epsilon=0.001, h_x=0.1)),
        ("WINNER3",     dict(h_epsilon=0.001, h_x=[0.1,0.1,0.1,0.7])),
        ("S0_only_LO",  dict(h_epsilon=[0.001, 0.01, 0.01, 0.01], h_x=0.1)),
    ]

    all_traj = {}
    for name, kw in configs:
        print(f"running {name}...", flush=True)
        traj, _ = run_with_traj(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
        all_traj[name] = traj

    # === Plot 1: trajectory grid for each config (key checkpoints) ===
    # pick 10 representative checkpoints per config: stage 0 step 0,5,9; stg 1 0,5,9; stg 2 0,5,9; stg 3 final
    def pick(traj):
        picks = []
        for (si, st) in [(0,0),(0,5),(0,9),(1,0),(1,5),(1,9),(2,0),(2,5),(2,9),(3,9)]:
            for e in traj:
                if e["stage"] == si and e["step"] == st:
                    picks.append(e); break
        return picks

    rows = len(configs); cols = 11  # 10 checkpoints + GT
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2))
    for r, (name, _) in enumerate(configs):
        traj = all_traj[name]
        picks = pick(traj)
        axes[r,0].imshow(denorm(gt[0])); axes[r,0].set_title("GT" if r==0 else "",fontsize=7); axes[r,0].set_ylabel(name, fontsize=8); axes[r,0].set_xticks([]); axes[r,0].set_yticks([])
        for c, e in enumerate(picks):
            ax = axes[r, c+1]
            ax.imshow(denorm(e["x1_img"]))
            title = f"stg{e['stage']} st{e['step']}\n|ε|={e['eps_rel_L2']:.2f}"
            ax.set_title(title if r==0 else "", fontsize=6)
            ax.axis("off")
    plt.suptitle("x_1 estimate at checkpoints across stages/steps  (GT = left col)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/trajectory_grid.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory_grid.png", flush=True)

    # === Plot 2: eps norm curves ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for name, traj in all_traj.items():
        xs = list(range(len(traj)))
        ys = [e["eps_rel_L2"] for e in traj]
        ax.plot(xs, ys, marker="o", markersize=3, label=name)
    # stage boundaries
    for s in [10, 20, 30]:
        ax.axvline(s - 0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("global step (stage*10 + step)")
    ax.set_ylabel("||eps||₂ / √N   (1.0 = standard Gaussian)")
    ax.set_title("eps Gaussian norm across sampling chain — collapse diagnosis")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/eps_norm_curves.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved eps_norm_curves.png", flush=True)

    # === Plot 3: box-region MSE curves ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for name, traj in all_traj.items():
        xs = list(range(len(traj)))
        ys = [e["res_box_mse"] for e in traj]
        ax.plot(xs, ys, marker="o", markersize=3, label=name)
    for s in [10, 20, 30]:
        ax.axvline(s - 0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("global step")
    ax.set_ylabel("box-region MSE vs GT (unobserved)")
    ax.set_title("reconstruction MSE in box region over time")
    ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/box_mse_curves.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved box_mse_curves.png", flush=True)

    print(f"\nAll outputs → {OUT}/")


if __name__ == "__main__":
    main()
