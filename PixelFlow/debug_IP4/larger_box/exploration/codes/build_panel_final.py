"""Top comparison panel showing the same-batch F sweep winner alongside anchors.

Produces:
  exploration/top_comparison_panel.png
  exploration/best_results/WINNER_*.{png,json,pt}
"""
import os, sys, json, shutil
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP  = os.path.dirname(HERE)
LB   = os.path.dirname(EXP)
DBG  = os.path.dirname(LB)
REPO = os.path.dirname(DBG)
sys.path.insert(0, HERE); sys.path.insert(0, LB); sys.path.insert(0, DBG); sys.path.insert(0, REPO)
os.chdir(REPO)

BEST = os.path.join(EXP, "best_results")
os.makedirs(BEST, exist_ok=True)


def to_img(t): return (t.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()


def load_xf(pt_path):
    if not os.path.exists(pt_path): return None
    return torch.load(pt_path, map_location="cpu", weights_only=False)["xf"]


def find_pt(name):
    for d in [os.path.join(EXP, "runs_f"),
              os.path.join(EXP, "runs_p"),
              os.path.join(EXP, "runs"),
              os.path.join(LB,  "results")]:
        p = os.path.join(d, f"{name}.pt")
        if os.path.exists(p): return p
    return None


def load_metrics(name):
    for d in [os.path.join(EXP, "runs_f"),
              os.path.join(EXP, "runs_p"),
              os.path.join(EXP, "runs"),
              os.path.join(LB,  "results")]:
        p = os.path.join(d, f"{name}.json")
        if os.path.exists(p):
            return json.load(open(p))
    return None


# Panel composition: GT, Measurement, then 8 selected configs in pareto-meaningful order.
# Each pair of anchors comes from runs_f (same batch). Then add cross-batch best for context.
PANEL = [
    ("F_anchor_C_L5",        "anchor CFG=0 (same-batch)"),
    ("F_anchor_N_cfg2",      "anchor CFG=2 (same-batch)"),
    ("F_cfg25_lr200_L7",     "CFG=2.5 + lr=200 + L=7"),
    ("F_cfg20_lr150_hxu02",  "★ WINNER  CFG=2 + lr=150 + h_x=0.2u"),
    ("I_ref_F1",             "ref_baseline_F1 (paper)"),
    ("C_L10",                "C_L10 (existing best LPIPS)"),
    ("N_cfg2",               "N_cfg2 (cross-batch)"),
    ("F_cfg20_lr300",        "CFG=2 + lr=300 (alt combo)"),
]


def main():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    gt_path = os.path.abspath(os.path.join(REPO, pt))
    gt = torch.load(gt_path, map_location="cpu", weights_only=False)["gt"][:2]
    from inpaintingStart import get_operator
    torch.manual_seed(7919)
    op = get_operator("inpainting", resolution=256, device="cpu", sigma=0.05,
                     mask_type="box", mask_len_range=(128, 129), mask_prob_range=None)
    mask = op.get_mask(x=gt).float()
    if mask.shape[0] == 1: mask = mask.expand(gt.shape[0], -1, -1, -1)
    inv = (1.0 - mask[0, 0]).numpy()
    rs = np.where(np.any(inv > 0.5, axis=1))[0]
    cs = np.where(np.any(inv > 0.5, axis=0))[0]
    rmin, rmax = int(rs[0]) - 5, int(rs[-1]) + 5
    cmin, cmax = int(cs[0]) - 5, int(cs[-1]) + 5
    rmin = max(0, rmin); rmax = min(255, rmax)
    cmin = max(0, cmin); cmax = min(255, cmax)
    meas = mask * gt + (1 - mask) * (-1.0)

    n_cfg = len(PANEL)
    ncol = n_cfg + 2
    fig, axes = plt.subplots(4, ncol, figsize=(ncol * 2.4, 9.5))

    # Col 0: GT
    axes[0, 0].imshow(to_img(gt[0])); axes[0, 0].set_title("GT img1", fontsize=9); axes[0, 0].axis("off")
    axes[1, 0].imshow(to_img(gt[0])[rmin:rmax+1, cmin:cmax+1]); axes[1, 0].set_title("GT box1", fontsize=8); axes[1, 0].axis("off")
    axes[2, 0].imshow(to_img(gt[1])); axes[2, 0].set_title("GT img2", fontsize=9); axes[2, 0].axis("off")
    axes[3, 0].imshow(to_img(gt[1])[rmin:rmax+1, cmin:cmax+1]); axes[3, 0].set_title("GT box2", fontsize=8); axes[3, 0].axis("off")

    # Col 1: Measurement
    axes[0, 1].imshow(to_img(meas[0])); axes[0, 1].set_title("Measurement1", fontsize=9); axes[0, 1].axis("off")
    axes[1, 1].imshow(to_img(meas[0])[rmin:rmax+1, cmin:cmax+1]); axes[1, 1].set_title("(black box1)", fontsize=8); axes[1, 1].axis("off")
    axes[2, 1].imshow(to_img(meas[1])); axes[2, 1].set_title("Measurement2", fontsize=9); axes[2, 1].axis("off")
    axes[3, 1].imshow(to_img(meas[1])[rmin:rmax+1, cmin:cmax+1]); axes[3, 1].set_title("(black box2)", fontsize=8); axes[3, 1].axis("off")

    for j, (name, label) in enumerate(PANEL):
        col = j + 2
        m = load_metrics(name)
        ptp = find_pt(name)
        xf = load_xf(ptp) if ptp else None
        is_winner = name == "F_cfg20_lr150_hxu02"
        if xf is None or m is None:
            for k in range(4):
                axes[k, col].axis("off")
                axes[k, col].set_title(f"{label}\n(missing pt)", fontsize=6)
            continue
        full0 = to_img(xf[0]); full1 = to_img(xf[1])
        box0  = full0[rmin:rmax+1, cmin:cmax+1]
        box1  = full1[rmin:rmax+1, cmin:cmax+1]
        col_color = "red" if is_winner else "black"
        axes[0, col].imshow(full0)
        axes[0, col].set_title(f"{label}\nP={m['psnr']:.2f} L={m['lpips']:.3f}",
                                fontsize=7, color=col_color)
        axes[0, col].axis("off")
        axes[1, col].imshow(box0); axes[1, col].set_title(f"box1   |dHF|={m.get('dhf',0):.3f}", fontsize=7); axes[1, col].axis("off")
        axes[2, col].imshow(full1); axes[2, col].set_title(f"img2", fontsize=7); axes[2, col].axis("off")
        axes[3, col].imshow(box1); axes[3, col].set_title(f"box2", fontsize=7); axes[3, col].axis("off")

    plt.suptitle("PRINCIPLE 128x128 box, res=256  —  Top comparison panel  (★ WINNER = F_cfg20_lr150_hxu02)\n"
                 "left-to-right: GT, Measurement, then anchors (same-batch F sweep), the WINNER, and reference candidates",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(EXP, "top_comparison_panel.png")
    plt.savefig(out, dpi=180); plt.close()
    print(f"wrote {out}")

    # Copy winner files to best_results/
    winner = "F_cfg20_lr150_hxu02"
    for ext in ["png", "json", "pt"]:
        src = os.path.join(EXP, "runs_f", f"{winner}.{ext}")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(BEST, f"WINNER_{winner}.{ext}"))
            print(f"copied -> best_results/WINNER_{winner}.{ext}")

    # also copy same-batch anchors and runner-up
    for n in ["F_anchor_N_cfg2", "F_anchor_C_L5", "F_cfg25_lr200_L7"]:
        for ext in ["png", "json", "pt"]:
            src = os.path.join(EXP, "runs_f", f"{n}.{ext}")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(BEST, f"{n}.{ext}"))


if __name__ == "__main__":
    main()
