"""Aggregate larger_box sweep results.

Outputs:
  results/sweep_summary.json  -- all configs sorted by LPIPS (lower=better perceptually)
  results/grid_all.png        -- 4xN grid of all configs' box-region crops (visual scan)
  results/grid_top.png        -- box crops of the LPIPS top-8 winners (large)
  results/group_<X>_curve.png -- per-group axis curves (LPIPS, PSNR, |dHF|)
  results/WINNERS.md          -- markdown report
"""
import os, sys, json, glob
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")

from sweep_configs import CONFIGS, GROUP_A, GROUP_B, GROUP_C, GROUP_D


def load_metrics(name):
    p = f"{RES}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def load_xf(name):
    p = f"{RES}/{name}.pt"
    if not os.path.exists(p):
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["xf"]


def to_img(x_neg11):
    return (x_neg11.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()


def main():
    rows = []
    for name, kw in CONFIGS:
        m = load_metrics(name)
        if m is None:
            print(f"[skip] {name}: no metrics")
            continue
        rows.append(dict(name=name, kw=kw,
                         psnr=m["psnr"], ssim=m["ssim"], lpips=m["lpips"],
                         psnr_unobs=m["psnr_unobs"], hf=m["hf"], dhf=m["dhf"],
                         time=m["time"]))
    if not rows:
        print("No results found. Run launch.sh first.")
        return

    # Sort by LPIPS (perceptual winner)
    rows_lpips = sorted(rows, key=lambda r: r["lpips"])
    rows_psnr  = sorted(rows, key=lambda r: -r["psnr"])
    rows_dhf   = sorted(rows, key=lambda r: r["dhf"])

    # ---------------- table print
    print("="*100)
    print(f"{'CONFIG':<20s}  {'h_epsilon':<32s}  {'L':<18s}  {'PSNR':>6}  {'SSIM':>6}  {'LPIPS':>7}  {'PSNRu':>6}  {'|dHF|':>6}  {'t':>5}")
    print("-"*100)
    for r in rows_lpips:
        kw = r["kw"]
        he = str(kw["h_epsilon"])
        L = str(kw["num_langevin"])
        print(f"{r['name']:<20s}  {he:<32s}  {L:<18s}  {r['psnr']:6.2f}  {r['ssim']:6.4f}  {r['lpips']:7.4f}  {r['psnr_unobs']:6.2f}  {r['dhf']:6.3f}  {r['time']:5.0f}")
    print("="*100)

    # ---------------- save sweep summary json
    summary = dict(
        all=rows_lpips,
        winners_by_lpips=[r["name"] for r in rows_lpips[:8]],
        winners_by_psnr =[r["name"] for r in rows_psnr[:8]],
        winners_by_dhf  =[r["name"] for r in rows_dhf[:8]],
    )
    with open(f"{RES}/sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"-> saved {RES}/sweep_summary.json")

    # ---------------- per-group axis curves (only meaningful for scalar groups A and C)
    def plot_scalar_axis(group, axis_name, axis_key, fname):
        xs, lp, ps, dhfs = [], [], [], []
        for name, kw in group:
            m = load_metrics(name)
            if m is None: continue
            v = kw[axis_key]
            if isinstance(v, list): continue
            xs.append(v); lp.append(m["lpips"]); ps.append(m["psnr"]); dhfs.append(m["dhf"])
        if not xs: return
        order = np.argsort(xs)
        xs = np.array(xs)[order]; lp = np.array(lp)[order]; ps = np.array(ps)[order]; dhfs = np.array(dhfs)[order]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(xs, lp, "o-"); ax[0].set_xscale("log"); ax[0].set_xlabel(axis_name); ax[0].set_ylabel("LPIPS (lower=better)"); ax[0].grid(True, alpha=0.3)
        ax[1].plot(xs, ps, "s-"); ax[1].set_xscale("log"); ax[1].set_xlabel(axis_name); ax[1].set_ylabel("PSNR"); ax[1].grid(True, alpha=0.3)
        ax[2].plot(xs, dhfs,"^-"); ax[2].set_xscale("log"); ax[2].set_xlabel(axis_name); ax[2].set_ylabel("|dHF|"); ax[2].grid(True, alpha=0.3)
        plt.suptitle(f"larger_box sweep — {axis_name} scalar curve (128x128 box)", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{RES}/{fname}", dpi=140)
        plt.close()
        print(f"-> saved {RES}/{fname}")

    plot_scalar_axis(GROUP_A, "h_epsilon", "h_epsilon", "group_A_h_epsilon_curve.png")
    plot_scalar_axis(GROUP_C, "num_langevin", "num_langevin", "group_C_num_langevin_curve.png")

    # ---------------- big grid of all configs, box-crop view
    # Load the GT once (any tensor file has gt indirectly via xf alignment)
    pt = "../../trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    gt_path = os.path.abspath(os.path.join(HERE, pt))
    gt = torch.load(gt_path, map_location="cpu", weights_only=False)["gt"][:2]
    # We need the box mask used for visualization — recompute the same one (deterministic seed=7919 in run_chunk)
    # Easiest: use the actual operator (same code path)
    REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "debug_IP4"))
    os.chdir(REPO)
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
    rmin = max(0, rmin-pad); rmax = min(255, rmax+pad)
    cmin = max(0, cmin-pad); cmax = min(255, cmax+pad)

    # ---------------- grid_all: 3 rows (GT crop, recon img1 crop, recon img2 crop) x N configs (+1 GT col)
    N = len(rows_lpips)
    fig, axes = plt.subplots(3, N+1, figsize=((N+1)*1.6, 4.8))
    gt_crop_0 = to_img(gt[0])[rmin:rmax+1, cmin:cmax+1]
    gt_crop_1 = to_img(gt[1])[rmin:rmax+1, cmin:cmax+1]
    axes[0,0].imshow(to_img(gt[0])); axes[0,0].set_title("GT", fontsize=4); axes[0,0].axis("off")
    axes[1,0].imshow(gt_crop_0); axes[1,0].set_title("GT box1", fontsize=4); axes[1,0].axis("off")
    axes[2,0].imshow(gt_crop_1); axes[2,0].set_title("GT box2", fontsize=4); axes[2,0].axis("off")
    for j, r in enumerate(rows_lpips):
        xf = load_xf(r["name"])
        if xf is None:
            for k in range(3): axes[k, j+1].axis("off")
            continue
        full = to_img(xf[0])
        c0 = to_img(xf[0])[rmin:rmax+1, cmin:cmax+1]
        c1 = to_img(xf[1])[rmin:rmax+1, cmin:cmax+1]
        axes[0, j+1].imshow(full)
        axes[0, j+1].set_title(f"{r['name']}\nP={r['psnr']:.2f}", fontsize=3); axes[0, j+1].axis("off")
        axes[1, j+1].imshow(c0)
        axes[1, j+1].set_title(f"L={r['lpips']:.3f} dHF={r['dhf']:.2f}", fontsize=3); axes[1, j+1].axis("off")
        axes[2, j+1].imshow(c1)
        axes[2, j+1].set_title(f"t={r['time']:.0f}s", fontsize=3); axes[2, j+1].axis("off")
    plt.suptitle(f"larger_box sweep ({N} configs)  --  sorted by LPIPS (left = best perceptual)", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{RES}/grid_all.png", dpi=200)
    plt.close()
    print(f"-> saved {RES}/grid_all.png")

    # ---------------- grid_top: top-8 LPIPS, big crops
    top = rows_lpips[:8]
    fig, axes = plt.subplots(3, len(top)+1, figsize=((len(top)+1)*2.2, 6.5))
    axes[0,0].imshow(to_img(gt[0])); axes[0,0].set_title("GT", fontsize=8); axes[0,0].axis("off")
    axes[1,0].imshow(gt_crop_0); axes[1,0].set_title("GT box1", fontsize=8); axes[1,0].axis("off")
    axes[2,0].imshow(gt_crop_1); axes[2,0].set_title("GT box2", fontsize=8); axes[2,0].axis("off")
    for j, r in enumerate(top):
        xf = load_xf(r["name"])
        if xf is None: continue
        full = to_img(xf[0])
        c0 = to_img(xf[0])[rmin:rmax+1, cmin:cmax+1]
        c1 = to_img(xf[1])[rmin:rmax+1, cmin:cmax+1]
        axes[0, j+1].imshow(full); axes[0, j+1].set_title(f"#{j+1} {r['name']}\nP={r['psnr']:.2f} L={r['lpips']:.3f}", fontsize=7); axes[0, j+1].axis("off")
        axes[1, j+1].imshow(c0); axes[1, j+1].set_title(f"S={r['ssim']:.3f} dHF={r['dhf']:.2f}", fontsize=7); axes[1, j+1].axis("off")
        axes[2, j+1].imshow(c1); axes[2, j+1].set_title("box2", fontsize=7); axes[2, j+1].axis("off")
    plt.suptitle("Top-8 perceptual winners (sorted by LPIPS, lower=better)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{RES}/grid_top.png", dpi=180)
    plt.close()
    print(f"-> saved {RES}/grid_top.png")

    # ---------------- WINNERS.md
    md = []
    md.append("# larger_box sweep — perceptual winners (128x128 box)\n")
    md.append(f"Sweep covers {len(rows)} configs over h_epsilon and num_langevin axes.\n")
    md.append(f"Base setup: WINNER3 (h_x=[0.1,0.1,0.1,0.7], lambda_reg=50, terminal_replace=1).\n")
    md.append("Metrics computed on the 2-image debug GT, box=128x128 random-position.\n\n")
    md.append("## Top-8 by LPIPS (lower = better perceptual)\n")
    md.append("| Rank | Config | h_epsilon | L | PSNR | SSIM | LPIPS | PSNRu | |dHF| | time |")
    md.append("|------|--------|-----------|---|------|------|-------|-------|-------|------|")
    for i, r in enumerate(rows_lpips[:8]):
        kw = r["kw"]
        md.append(f"| {i+1} | `{r['name']}` | `{kw['h_epsilon']}` | `{kw['num_langevin']}` | "
                  f"{r['psnr']:.2f} | {r['ssim']:.3f} | **{r['lpips']:.4f}** | {r['psnr_unobs']:.2f} | {r['dhf']:.3f} | {r['time']:.0f}s |")
    md.append("\n## Top-8 by PSNR\n")
    md.append("| Rank | Config | h_epsilon | L | PSNR | SSIM | LPIPS | |dHF| |")
    md.append("|------|--------|-----------|---|------|------|-------|-------|")
    for i, r in enumerate(rows_psnr[:8]):
        kw = r["kw"]
        md.append(f"| {i+1} | `{r['name']}` | `{kw['h_epsilon']}` | `{kw['num_langevin']}` | "
                  f"**{r['psnr']:.2f}** | {r['ssim']:.3f} | {r['lpips']:.4f} | {r['dhf']:.3f} |")
    md.append("\n## Top-8 by |dHF| (closest to GT high-frequency energy)\n")
    md.append("| Rank | Config | h_epsilon | L | PSNR | LPIPS | |dHF| |")
    md.append("|------|--------|-----------|---|------|-------|-------|")
    for i, r in enumerate(rows_dhf[:8]):
        kw = r["kw"]
        md.append(f"| {i+1} | `{r['name']}` | `{kw['h_epsilon']}` | `{kw['num_langevin']}` | "
                  f"{r['psnr']:.2f} | {r['lpips']:.4f} | **{r['dhf']:.3f}** |")
    md.append("\n## All configs (sorted by LPIPS)\n")
    md.append("| Config | h_epsilon | L | PSNR | SSIM | LPIPS | PSNRu | |dHF| |")
    md.append("|--------|-----------|---|------|------|-------|-------|-------|")
    for r in rows_lpips:
        kw = r["kw"]
        md.append(f"| `{r['name']}` | `{kw['h_epsilon']}` | `{kw['num_langevin']}` | "
                  f"{r['psnr']:.2f} | {r['ssim']:.3f} | {r['lpips']:.4f} | {r['psnr_unobs']:.2f} | {r['dhf']:.3f} |")
    md.append("\nSee `grid_top.png` for visual comparison of top winners; `grid_all.png` for full sweep.")
    md_text = "\n".join(md)
    with open(f"{RES}/WINNERS.md", "w") as f:
        f.write(md_text)
    print(f"-> saved {RES}/WINNERS.md")
    print(f"\n*** WINNERS BY LPIPS:")
    for i, r in enumerate(rows_lpips[:5]):
        print(f"  #{i+1}  {r['name']}  he={r['kw']['h_epsilon']}  L={r['kw']['num_langevin']}  LPIPS={r['lpips']:.4f}  PSNR={r['psnr']:.2f}")


if __name__ == "__main__":
    main()
