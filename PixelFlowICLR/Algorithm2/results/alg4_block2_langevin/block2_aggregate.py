"""Aggregate results/alg4_block2_langevin: summary.csv/md (mean±std over seeds per arm), trajectory montage
(arms x frames, seed 42), and hole-MSE-vs-frame plot."""
import os, json, glob, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
OUT = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/results/alg4_block2_langevin"
ARMS = ["baseline", "h1.0", "h0.5", "h0.1", "h5e-2", "h1e-2", "h5e-3", "h1e-5"]
rows = [json.load(open(f)) for f in glob.glob(f"{OUT}/*/junco_s*/final.json")]
by = {a: [r for r in rows if r["arm"] == a] for a in ARMS}
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["arm", "h0", "n_seeds", "mse_hole_mean", "mse_hole_std", "mse_full_mean", "mse_full_std", "psnr_range2_mean", "psnr_range2_std", "x0_rms_last_mean", "cg_bad_total", "secs_mean"])
    for a in ARMS:
        rs = by[a]
        if not rs: continue
        g = lambda k: np.array([r[k] for r in rs], float)
        w.writerow([a, rs[0]["h0"], len(rs), f"{g('mse_hole').mean():.4f}", f"{g('mse_hole').std():.4f}", f"{g('mse_full').mean():.4f}", f"{g('mse_full').std():.4f}", f"{g('psnr_range2').mean():.2f}", f"{g('psnr_range2').std():.2f}", f"{g('x0_rms_last').mean():.3f}", int(g('cg_bad').sum()), f"{g('secs').mean():.0f}"])
md = ["# Block-2 Langevin probe — junco box, [2,2,1,1], default S (spectral_class), seeds 42-44\n",
      "| arm | h0 | seeds | hole MSE | full MSE | PSNR(range2) | x0 rms (last frame) | per-seed hole MSE |", "|---|---|---|---|---|---|---|---|"]
for a in ARMS:
    rs = sorted(by[a], key=lambda r: r["seed"])
    if not rs: continue
    g = lambda k: np.array([r[k] for r in rs], float)
    md.append(f"| {a} | {rs[0]['h0']} | {len(rs)} | {g('mse_hole').mean():.4f}±{g('mse_hole').std():.4f} | {g('mse_full').mean():.4f}±{g('mse_full').std():.4f} | {g('psnr_range2').mean():.2f}±{g('psnr_range2').std():.2f} | {g('x0_rms_last').mean():.3f} | {', '.join(f'{r['mse_hole']:.4f}' for r in rs)} |")
open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n")
# montage: rows = arms (seed 42), cols = GT, masked, frames 0,5,10,15,20,25,30,35,last
FR = [0, 5, 10, 15, 20, 25, 30, 35, "last"]
gt = np.array(Image.open(f"{OUT}/junco_gt.png")); mk = np.array(Image.open(f"{OUT}/junco_masked.png"))
arms_present = [a for a in ARMS if os.path.exists(f"{OUT}/{a}/junco_s42/traj_x1.npy")]
fig, axes = plt.subplots(len(arms_present), len(FR) + 2, figsize=(2.0 * (len(FR) + 2), 2.1 * len(arms_present)))
axes = np.atleast_2d(axes)
for i, a in enumerate(arms_present):
    tr = np.load(f"{OUT}/{a}/junco_s42/traj_x1.npy"); fin = json.load(open(f"{OUT}/{a}/junco_s42/final.json"))
    for j, im in enumerate([gt, mk] + [tr[k] for k in range(len(FR))]):
        ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(["GT", "y (masked)"][j] if j < 2 else f"f{FR[j-2]}", fontsize=9)
    axes[i, 0].set_ylabel(f"{a}\nhole {fin['mse_hole']:.4f}", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/trajectory.png", dpi=110); plt.close()
# hole MSE vs frame
plt.figure(figsize=(8, 4))
for a in arms_present:
    rr = list(csv.DictReader(open(f"{OUT}/{a}/junco_s42/trajectory_metrics.csv")))
    if "mse_hole" in rr[0]:
        plt.plot([int(r["frame"]) for r in rr], [float(r["mse_hole"]) for r in rr], label=a, lw=1.4)
plt.yscale("log"); plt.xlabel("frame"); plt.ylabel("hole MSE (x1 vs GT pyramid)"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.title("junco box, seed 42: Block-2 exact draw vs Langevin probe"); plt.tight_layout(); plt.savefig(f"{OUT}/mse_hole_vs_frame.png", dpi=110)
print("\n".join(md))
