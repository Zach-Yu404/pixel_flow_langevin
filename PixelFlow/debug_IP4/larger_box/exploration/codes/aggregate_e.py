"""Aggregate exploration sweep + the larger_box/results sweep into a unified ranking.

Outputs (under exploration/summaries/):
  combined_summary.json           : all configs + per-config metrics
  ranked_results.csv              : machine-readable ranking
  EXPLORATION_RANKINGS.md         : top-N tables (LPIPS / PSNR / balanced score)
  exploration_grid_top.png        : visual grid of top-N by balanced score
"""
import os, sys, json, glob, csv
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))   # exploration/scripts
EXP  = os.path.dirname(HERE)                        # exploration
LB   = os.path.dirname(EXP)                         # larger_box
DBG  = os.path.dirname(LB)                          # debug_IP4
REPO = os.path.dirname(DBG)                         # PixelFlow
sys.path.insert(0, HERE)
sys.path.insert(0, LB)
sys.path.insert(0, DBG)
sys.path.insert(0, REPO)
os.chdir(REPO)

SUMS = os.path.join(EXP, "summaries")
os.makedirs(SUMS, exist_ok=True)
EXP_RUNS  = os.path.join(EXP, "runs")
LB_RES    = os.path.join(LB, "results")


def load_one(path_json):
    with open(path_json) as f:
        m = json.load(f)
    return m


def gather(dirpath, source_tag):
    rows = []
    for jp in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        if os.path.basename(jp).startswith("_"):
            continue
        if "summary" in os.path.basename(jp):
            continue
        try:
            m = load_one(jp)
        except Exception:
            continue
        if "name" not in m or "psnr" not in m:
            continue
        rows.append(dict(
            name=m["name"], kw=m.get("kw", {}),
            psnr=m["psnr"], ssim=m["ssim"], lpips=m["lpips"],
            psnr_unobs=m.get("psnr_unobs", float("nan")),
            hf=m.get("hf", float("nan")), dhf=m.get("dhf", float("nan")),
            time=m.get("time", float("nan")),
            source=source_tag, json_path=jp,
            pt_path=jp[:-5] + ".pt", png_path=jp[:-5] + ".png",
        ))
    return rows


def balanced_score(r):
    """Single-number tradeoff for ranking the BALANCED corner.

    Lower is better.  Combines LPIPS (perceptual) + soft penalty for low PSNR
    and large |dHF|.  Tuned so:
      LPIPS pulls strongest (perceptual is primary).
      PSNR penalty kicks in below 17 dB ("acceptable PSNR").
      |dHF| penalty kicks in above 0.05 (HF not too far from GT).
    """
    lpips = r["lpips"]
    psnr  = r["psnr"]
    dhf   = r.get("dhf", 0.0)
    psnr_pen = max(0.0, 17.0 - psnr) * 0.02   # 0.02 LPIPS-units per dB below 17
    dhf_pen  = max(0.0, dhf - 0.05) * 0.5     # 0.5 LPIPS-units per +0.1 |dHF|
    return lpips + psnr_pen + dhf_pen


def to_img_neg11(t):
    return (t.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()


def main():
    rows  = gather(EXP_RUNS, "exploration")
    rows += gather(LB_RES,    "larger_box_initial")
    if not rows:
        print("no results found"); return

    for r in rows:
        r["balanced"] = balanced_score(r)

    rows_lpips    = sorted(rows, key=lambda r: r["lpips"])
    rows_psnr     = sorted(rows, key=lambda r: -r["psnr"])
    rows_balanced = sorted(rows, key=lambda r: r["balanced"])

    summary = dict(
        total=len(rows),
        winners_by_lpips    =[r["name"] for r in rows_lpips[:10]],
        winners_by_psnr     =[r["name"] for r in rows_psnr[:10]],
        winners_by_balanced =[r["name"] for r in rows_balanced[:10]],
        all_rows=rows_balanced,
    )
    with open(os.path.join(SUMS, "combined_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # CSV
    csv_path = os.path.join(SUMS, "ranked_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank_balanced", "name", "source", "psnr", "psnr_unobs", "ssim", "lpips", "hf", "dhf",
                    "balanced_score", "time_sec", "kw"])
        for i, r in enumerate(rows_balanced):
            w.writerow([i + 1, r["name"], r["source"],
                        f"{r['psnr']:.3f}", f"{r['psnr_unobs']:.3f}", f"{r['ssim']:.4f}",
                        f"{r['lpips']:.4f}", f"{r['hf']:.4f}", f"{r['dhf']:.4f}",
                        f"{r['balanced']:.4f}", f"{r['time']:.0f}",
                        json.dumps(r["kw"], default=str)])
    print("wrote", csv_path)

    # Markdown tables
    md = ["# Exploration rankings — 128x128 box, res=256\n",
          f"Combined sweep size = {len(rows)} (existing larger_box + new exploration).\n",
          "**Balanced score** = LPIPS + 0.02·max(0, 17-PSNR) + 0.5·max(0, |dHF|-0.05). Lower = better.\n\n",
          "## Top-15 by balanced score\n",
          "| Rank | Config | Source | PSNR | PSNRu | SSIM | LPIPS | |dHF| | Balanced |",
          "|------|--------|--------|------|-------|------|-------|-------|----------|"]
    for i, r in enumerate(rows_balanced[:15]):
        md.append(f"| {i+1} | `{r['name']}` | {r['source']} | "
                  f"{r['psnr']:.2f} | {r['psnr_unobs']:.2f} | {r['ssim']:.3f} | "
                  f"**{r['lpips']:.4f}** | {r['dhf']:.3f} | **{r['balanced']:.4f}** |")
    md.append("\n## Top-10 by LPIPS (perceptual only)\n")
    md.append("| Rank | Config | Source | PSNR | LPIPS | |dHF| |")
    md.append("|------|--------|--------|------|-------|-------|")
    for i, r in enumerate(rows_lpips[:10]):
        md.append(f"| {i+1} | `{r['name']}` | {r['source']} | "
                  f"{r['psnr']:.2f} | **{r['lpips']:.4f}** | {r['dhf']:.3f} |")
    md.append("\n## Top-10 by PSNR\n")
    md.append("| Rank | Config | Source | PSNR | LPIPS | |dHF| |")
    md.append("|------|--------|--------|------|-------|-------|")
    for i, r in enumerate(rows_psnr[:10]):
        md.append(f"| {i+1} | `{r['name']}` | {r['source']} | "
                  f"**{r['psnr']:.2f}** | {r['lpips']:.4f} | {r['dhf']:.3f} |")
    md.append("\n## All configs (sorted by balanced score)\n")
    md.append("| Config | Source | PSNR | PSNRu | SSIM | LPIPS | |dHF| | Balanced |")
    md.append("|--------|--------|------|-------|------|-------|-------|----------|")
    for r in rows_balanced:
        md.append(f"| `{r['name']}` | {r['source']} | "
                  f"{r['psnr']:.2f} | {r['psnr_unobs']:.2f} | {r['ssim']:.3f} | "
                  f"{r['lpips']:.4f} | {r['dhf']:.3f} | {r['balanced']:.4f} |")
    md_path = os.path.join(SUMS, "EXPLORATION_RANKINGS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print("wrote", md_path)

    # Visual grid: top-10 by balanced score
    # Need GT + box bbox
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    gt_path = os.path.abspath(os.path.join(REPO, pt))
    gt = torch.load(gt_path, map_location="cpu", weights_only=False)["gt"][:2]
    sys.path.insert(0, REPO); sys.path.insert(0, DBG)
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

    top_n = 10
    top = rows_balanced[:top_n]
    fig, axes = plt.subplots(3, top_n + 1, figsize=((top_n + 1) * 2.0, 6.5))
    axes[0, 0].imshow(to_img_neg11(gt[0])); axes[0, 0].set_title("GT", fontsize=8); axes[0, 0].axis("off")
    axes[1, 0].imshow(to_img_neg11(gt[0])[rmin:rmax+1, cmin:cmax+1]); axes[1, 0].set_title("GT box1", fontsize=8); axes[1, 0].axis("off")
    axes[2, 0].imshow(to_img_neg11(gt[1])[rmin:rmax+1, cmin:cmax+1]); axes[2, 0].set_title("GT box2", fontsize=8); axes[2, 0].axis("off")
    for j, r in enumerate(top):
        if not os.path.exists(r["pt_path"]):
            for k in range(3): axes[k, j+1].axis("off")
            continue
        d = torch.load(r["pt_path"], map_location="cpu", weights_only=False)
        xf = d["xf"]
        full = to_img_neg11(xf[0])
        c0   = to_img_neg11(xf[0])[rmin:rmax+1, cmin:cmax+1]
        c1   = to_img_neg11(xf[1])[rmin:rmax+1, cmin:cmax+1]
        axes[0, j+1].imshow(full)
        axes[0, j+1].set_title(f"#{j+1} {r['name']}\nP={r['psnr']:.2f} L={r['lpips']:.3f}", fontsize=6)
        axes[0, j+1].axis("off")
        axes[1, j+1].imshow(c0); axes[1, j+1].set_title(f"S={r['ssim']:.3f} dHF={r['dhf']:.2f}", fontsize=6); axes[1, j+1].axis("off")
        axes[2, j+1].imshow(c1); axes[2, j+1].set_title(f"bal={r['balanced']:.3f}", fontsize=6); axes[2, j+1].axis("off")
    plt.suptitle(f"Top-{top_n} by balanced score (lower=better)  •  PRINCIPLE 128x128 box, res=256", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    grid_path = os.path.join(SUMS, "exploration_grid_top.png")
    plt.savefig(grid_path, dpi=180)
    plt.close()
    print("wrote", grid_path)

    # Best winner summary
    w = rows_balanced[0]
    print("\n*** BEST BY BALANCED SCORE ***")
    print(f"  name = {w['name']}  source = {w['source']}")
    print(f"  PSNR = {w['psnr']:.2f}  PSNRu = {w['psnr_unobs']:.2f}  LPIPS = {w['lpips']:.4f}  |dHF| = {w['dhf']:.3f}")
    print(f"  balanced = {w['balanced']:.4f}")
    print(f"  kw = {w['kw']}")


if __name__ == "__main__":
    main()
