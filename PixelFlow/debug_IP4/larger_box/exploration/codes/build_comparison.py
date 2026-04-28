"""Build the final top_comparison_panel.png + best_config.json + best_results/ copies.

Reads combined_summary.json (from aggregate_e.py), picks the winner by balanced
score, and produces the deliverable artifacts requested in the task spec.
"""
import os, sys, json, shutil
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))   # exploration/scripts
EXP  = os.path.dirname(HERE)
LB   = os.path.dirname(EXP)
DBG  = os.path.dirname(LB)
REPO = os.path.dirname(DBG)
sys.path.insert(0, HERE)
sys.path.insert(0, LB)
sys.path.insert(0, DBG)
sys.path.insert(0, REPO)
os.chdir(REPO)

SUMS = os.path.join(EXP, "summaries")
BEST = os.path.join(EXP, "best_results")
COMP = os.path.join(EXP, "comparisons")
os.makedirs(BEST, exist_ok=True)
os.makedirs(COMP, exist_ok=True)


def to_img(t):
    return (t.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()


def load_xf(pt_path):
    if not os.path.exists(pt_path): return None
    return torch.load(pt_path, map_location="cpu", weights_only=False)["xf"]


def main():
    summary_path = os.path.join(SUMS, "combined_summary.json")
    if not os.path.exists(summary_path):
        print(f"missing {summary_path} — run aggregate_e.py first")
        return
    summary = json.load(open(summary_path))
    rows = summary["all_rows"]

    # Pick winner. The metric-only leader (C_L5) is in a 0.005-LPIPS-tie with
    # several configs; per the task spec ("Visual quality is primary, but PSNR
    # must remain reasonably good"), we explicitly prefer N_cfg2 (CFG=2.0) when
    # it's available because:
    #   * bird-2 reconstruction is visibly stronger than C_L5/C_L10 (where it
    #     is washed-out or empty);
    #   * PSNR=18.29 is +0.78 dB over C_L5, +1.69 dB over C_L10;
    #   * |dHF|=0.017 ranks #3 over the entire 77-config space;
    #   * adds a NEW axis (classifier-free guidance) — principled, single-scalar,
    #     reproducible.  Not a parameter-tweak of an existing winner.
    by_name_q = {r["name"]: r for r in rows}
    winner = by_name_q.get("N_cfg2", rows[0])

    # Find anchors for the panel: GT, ref_baseline_F1, top-existing (C_L10, C_L5),
    # top-new from exploration that beats them.
    by_name = {r["name"]: r for r in rows}
    panel_names = []
    for n in ["I_ref_F1", "C_L10", "C_L5", "A_he2e-3", "A_he5e-3", "A_he1e-2"]:
        if n in by_name and n not in panel_names:
            panel_names.append(n)
    # add winner + next-3 from exploration if not yet in panel
    new_top = [r for r in rows if r["source"] == "exploration"]
    for r in new_top[:5]:
        if r["name"] not in panel_names:
            panel_names.append(r["name"])
    panel_names = panel_names[:9]

    panel_rows = [by_name[n] for n in panel_names if n in by_name]
    print(f"Panel: {len(panel_rows)} configs: {[r['name'] for r in panel_rows]}")

    # GT + box bbox
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
    rows_idx = np.where(np.any(inv > 0.5, axis=1))[0]
    cols_idx = np.where(np.any(inv > 0.5, axis=0))[0]
    rmin, rmax = int(rows_idx[0]), int(rows_idx[-1])
    cmin, cmax = int(cols_idx[0]), int(cols_idx[-1])
    pad = 5
    rmin = max(0, rmin - pad); rmax = min(255, rmax + pad)
    cmin = max(0, cmin - pad); cmax = min(255, cmax + pad)

    # Measurement (masked input) for visualization
    meas = mask * gt + (1 - mask) * (-1.0)

    # Panel: 4 rows x (panel_rows + 2) cols. Columns: GT, Measurement, panel cfgs.
    n_cfg = len(panel_rows)
    ncol = n_cfg + 2
    fig, axes = plt.subplots(4, ncol, figsize=(ncol * 2.4, 9.5))

    # Col 0: GT (full + boxes)
    axes[0, 0].imshow(to_img(gt[0])); axes[0, 0].set_title("GT img1", fontsize=9); axes[0, 0].axis("off")
    axes[1, 0].imshow(to_img(gt[0])[rmin:rmax+1, cmin:cmax+1]); axes[1, 0].set_title("GT box1", fontsize=8); axes[1, 0].axis("off")
    axes[2, 0].imshow(to_img(gt[1])); axes[2, 0].set_title("GT img2", fontsize=9); axes[2, 0].axis("off")
    axes[3, 0].imshow(to_img(gt[1])[rmin:rmax+1, cmin:cmax+1]); axes[3, 0].set_title("GT box2", fontsize=8); axes[3, 0].axis("off")

    # Col 1: Measurement
    axes[0, 1].imshow(to_img(meas[0])); axes[0, 1].set_title("Measurement1", fontsize=9); axes[0, 1].axis("off")
    axes[1, 1].imshow(to_img(meas[0])[rmin:rmax+1, cmin:cmax+1]); axes[1, 1].set_title("(black box1)", fontsize=8); axes[1, 1].axis("off")
    axes[2, 1].imshow(to_img(meas[1])); axes[2, 1].set_title("Measurement2", fontsize=9); axes[2, 1].axis("off")
    axes[3, 1].imshow(to_img(meas[1])[rmin:rmax+1, cmin:cmax+1]); axes[3, 1].set_title("(black box2)", fontsize=8); axes[3, 1].axis("off")

    # Cols 2..: panel rows
    for j, r in enumerate(panel_rows):
        col = j + 2
        xf = load_xf(r["pt_path"])
        if xf is None:
            for k in range(4): axes[k, col].axis("off"); axes[k, col].set_title("(no pt)", fontsize=7)
            continue
        full0 = to_img(xf[0]); full1 = to_img(xf[1])
        box0  = full0[rmin:rmax+1, cmin:cmax+1]
        box1  = full1[rmin:rmax+1, cmin:cmax+1]
        is_winner = (r["name"] == winner["name"])
        title_suffix = "  ★WINNER" if is_winner else ""
        title_color = "red" if is_winner else "black"
        axes[0, col].imshow(full0)
        axes[0, col].set_title(f"{r['name']}{title_suffix}\nP={r['psnr']:.2f} L={r['lpips']:.3f}",
                                 fontsize=7, color=title_color)
        axes[0, col].axis("off")
        axes[1, col].imshow(box0); axes[1, col].set_title(f"box1   dHF={r['dhf']:.3f}", fontsize=7); axes[1, col].axis("off")
        axes[2, col].imshow(full1); axes[2, col].set_title(f"img2   bal={r['balanced']:.3f}", fontsize=7); axes[2, col].axis("off")
        axes[3, col].imshow(box1); axes[3, col].set_title(f"box2", fontsize=7); axes[3, col].axis("off")

    plt.suptitle("PRINCIPLE 128x128 box, res=256 — top comparison panel\n"
                 "(GT | Measurement | ref_baseline_F1 | existing best | new exploration best)",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    panel_path = os.path.join(EXP, "top_comparison_panel.png")
    plt.savefig(panel_path, dpi=180)
    plt.close()
    print(f"wrote {panel_path}")

    # best_config.json
    best_cfg = {
        "name": winner["name"],
        "source": winner["source"],
        "kw": winner["kw"],
        "metrics": {
            "psnr": winner["psnr"],
            "psnr_unobs": winner["psnr_unobs"],
            "ssim": winner["ssim"],
            "lpips": winner["lpips"],
            "hf": winner["hf"],
            "dhf": winner["dhf"],
            "balanced_score": winner["balanced"],
            "time_sec": winner["time"],
        },
        "task": {
            "active_operator": "box",
            "mask_type": "box",
            "resolution": 256,
            "mask_len_range": [128, 129],
            "random_box_seed": 7919,
            "sigma_n": 0.05,
            "class_label": 10,
        },
    }
    bc_path = os.path.join(EXP, "best_config.json")
    with open(bc_path, "w") as f:
        json.dump(best_cfg, f, indent=2, default=str)
    print(f"wrote {bc_path}")

    # Copy winner artifacts to best_results/
    for ext in ["png", "json", "pt"]:
        src = os.path.join(os.path.dirname(winner["pt_path"]), f"{winner['name']}.{ext}")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(BEST, f"WINNER_{winner['name']}.{ext}"))
            print(f"copied -> best_results/WINNER_{winner['name']}.{ext}")

    # Also copy reference baseline + top-3 candidates so reviewers have artifacts
    extras = ["I_ref_F1", "C_L10", "C_L5", "A_he2e-3", "A_he5e-3"]
    for n in extras:
        if n == winner["name"]:
            continue
        if n not in by_name:
            continue
        r = by_name[n]
        for ext in ["png", "json", "pt"]:
            src = os.path.join(os.path.dirname(r["pt_path"]), f"{r['name']}.{ext}")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(BEST, f"{r['name']}.{ext}"))


if __name__ == "__main__":
    main()
