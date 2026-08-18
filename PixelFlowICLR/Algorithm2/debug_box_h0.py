#!/usr/bin/env python
"""Debug: box_inpainting Alg2 hole filling — h0 sweep (task debug-box-alg2-hole).

Hypothesis (see .research/tasks/debug-box-alg2-hole.md): the hole never fills
because the ONLY image-prior channel in Algorithm 2 is the Block-2 x0-Langevin
whose pull on x_tau is ~ sigma*h0/2 per inner step; h0=0.1 was never swept.

Minimal test: box_inpainting, ONE image (junco), h0 in {0.1, 0.5, 1.0},
everything else identical (same seed, same LPIPS_king-best kw, S=10).
Records per-tau OBSERVED vs HOLE MSE separately + end-of-stage x1 snapshots.
"""

import argparse
import json
import os

ORIG_CWD = os.getcwd()

import algorithm2 as alg  # noqa: E402  (chdir side effect via base import chain)
import onestep_mse_vs_t as base  # noqa: E402
import full_ip_compare as full_ip  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pixelflow.utils import config as config_utils  # noqa: E402
from demo_runner import build_setup_and_measurement, load_demo_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(alg.HERE, "results", "debug_box_h0"))
    ap.add_argument("--image", default="junco")
    ap.add_argument("--h0", nargs="*", type=float, default=[0.1, 0.5, 1.0])
    ap.add_argument("--anchors", nargs="*", type=float, default=None,
                    help="sweep Tweedie anchor at h0=0.1 (NOTE: anchor path currently "
                         "disabled in the sampler by user edit — runs are paper-Alg2)")
    ap.add_argument("--x0_steps", nargs="*", type=int, default=None,
                    help="sweep x0-Langevin steps K (l.16-17 loop), e.g. 1 3 5")
    ap.add_argument("--gamma2_scales", nargs="*", type=float, default=[1.0],
                    help="gamma2 table multiplier per variant (0 => gamma2=0)")
    args = ap.parse_args()
    if not os.path.isabs(args.out):
        args.out = os.path.join(ORIG_CWD, args.out)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda:0"
    task = "box_inpainting"

    demos_all = load_demo_images(resolution=256,
                                 demo_dir=os.path.join(base.IP_PACKAGE, "demo"))
    demo = {d["short_name"]: d for d in demos_all}[args.image]
    gt = demo["gt"].unsqueeze(0).to(device)

    model_dir = os.path.join(base.IP_PACKAGE, "pretrained_models", "c2img")
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print("[setup] model loaded", flush=True)

    cfg, op_cfg = full_ip.best_cfg(task)
    kw = dict(cfg["kw"])
    kw["class_label"] = int(demo["class_idx"])
    sigma_n = float(cfg.get("sigma_n", 0.05))
    op, mask, y, _, _, make_Ak_fns_fn, _ = build_setup_and_measurement(
        task, op_cfg, demo, sigma_n, 256, device)
    gamma2_tab = json.load(open(os.path.join(alg.HERE, "gamma2_meas.json")))["table"]
    num_stages = int(config.scheduler.num_stages)
    pyr = base.gt_stage_pyramid(gt, num_stages)
    hole = (1.0 - mask).to(device)          # [1,1,256,256]; 1 inside the box

    if args.x0_steps is not None:
        variants = [(f"K={k},g2x{gs:g}", 0.1, 0.0, k, gs)
                    for k in args.x0_steps for gs in args.gamma2_scales]
    elif args.anchors is not None:
        variants = [(f"anchor={a}", 0.1, a, 1, 1.0) for a in args.anchors]
    else:
        variants = [(f"h0={h}", h, 0.0, 1, 1.0) for h in args.h0]
    all_rows = []
    finals = {}
    for label, h0, anchor_val, x0_steps, g2_scale in variants:
        x1, rows, traj = full_ip.run_posterior_sampling_alg2(
            model, config, gt, y, op, sigma_n, device,
            gamma2_tab=gamma2_tab, make_Ak_fns_fn=make_Ak_fns_fn,
            seed=42, record_trajectory=True, h0=h0, anchor=anchor_val,
            x0_langevin_steps=x0_steps, gamma2_scale=g2_scale,
            **{k: v for k, v in kw.items() if k not in ("class_label", "seed")},
            class_label=int(demo["class_idx"]))
        # traj entries follow the same (stage, step) order as rows
        for r, (x_tau_rec, x1_rec, _) in zip(rows, traj):
            k = r["stage"]
            x1k = x1_rec.unsqueeze(0).to(device)
            h, w = x1k.shape[-2:]
            m_k = F.interpolate(hole, size=(h, w), mode="nearest")
            gt_k = pyr[k]
            err = (x1k - gt_k) ** 2
            r2 = dict(r)
            r2["variant"] = label
            r2["h0"] = h0
            r2["anchor"] = anchor_val
            r2["x0_steps"] = x0_steps
            r2["gamma2_scale"] = g2_scale
            r2["mse_hole"] = float((err * m_k).sum() / (m_k.sum() * 3))
            r2["mse_obs"] = float((err * (1 - m_k)).sum() / ((1 - m_k).sum() * 3))
            all_rows.append(r2)
        finals[label] = x1.detach().cpu()
        hm = [r for r in all_rows if r["variant"] == label]
        print(f"[{label}] final-stage last-step hole MSE = {hm[-1]['mse_hole']:.4f} "
              f"obs = {hm[-1]['mse_obs']:.4f}", flush=True)

    # ── raw CSV ──
    import csv
    with open(os.path.join(args.out, "h0_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    # ── hole/obs MSE curves (global step axis, one line per h0) ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for label, *_ in variants:
        rs = [r for r in all_rows if r["variant"] == label]
        xs = list(range(len(rs)))
        axes[0].semilogy(xs, [r["mse_hole"] for r in rs], label=label)
        axes[1].semilogy(xs, [r["mse_obs"] for r in rs], label=label)
    for ax, ttl in zip(axes, ["HOLE region MSE", "OBSERVED region MSE"]):
        for b in (10, 20, 30):
            ax.axvline(b, color="gray", lw=0.6, alpha=0.5)
        ax.set_title(f"box·{args.image} · {ttl}"); ax.set_xlabel("global step (4 stages x 10)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "h0_sweep_mse.png"), dpi=150)
    plt.close(fig)

    # ── final x1 comparison panel ──
    n = len(variants)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.4))
    axes[0].imshow(((gt[0].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1))
    axes[0].set_title("GT"); axes[0].axis("off")
    for i, (label, *_) in enumerate(variants):
        axes[i + 1].imshow(((finals[label][0].permute(1, 2, 0) + 1) / 2).clamp(0, 1))
        last = [r for r in all_rows if r["variant"] == label][-1]
        axes[i + 1].set_title(f"{label}\nhole MSE {last['mse_hole']:.3f}", fontsize=9)
        axes[i + 1].axis("off")
    fig.suptitle(f"Alg2 final x1 — box_inpainting · {args.image} · sweep", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "h0_sweep_final_x1.png"), dpi=150)
    plt.close(fig)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
