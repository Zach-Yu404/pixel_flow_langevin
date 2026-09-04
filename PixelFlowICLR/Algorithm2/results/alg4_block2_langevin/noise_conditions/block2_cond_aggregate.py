"""Aggregate noise_conditions/: per-condition mean±std over seeds, a combined h0 x condition matrix (allnoise from the
parent dir), per-condition montage (seed 42) and hole-MSE-vs-frame curves."""
import os, json, glob, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
ROOT = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/results/alg4_block2_langevin"; OUT = f"{ROOT}/noise_conditions"
ARMS = ["baseline", "h1.0", "h0.5", "h0.1", "h5e-2", "h1e-2", "h5e-3", "h1e-5"]
CONDS = ["allnoise", "no_noise", "xi0_zero", "all_zero"]
LABEL = {"allnoise": "all noise (ref)", "no_noise": "no_noise (ξy=ξh=ξs=0)", "xi0_zero": "ξ0=0", "all_zero": "all four ξ=0"}
def load(cond):
    files = glob.glob(f"{ROOT}/*/junco_s*/final.json") if cond == "allnoise" else glob.glob(f"{OUT}/{cond}/*/junco_s*/final.json")
    return [json.load(open(f)) for f in files]
R = {c: load(c) for c in CONDS}
def stat(rs, k): a = np.array([r[k] for r in rs], float); return a.mean(), a.std(), len(a)
rows = []; md = ["# Block-2 Langevin probe × noise conditions — junco box, [2,2,1,1], default S, seeds 42-44\n",
      "hole MSE mean±std (n seeds). Columns: noise condition; rows: Block-2 variant. allnoise = the earlier run (reference).\n",
      "| arm | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---|" * len(CONDS)]
for a in ARMS:
    cells = []
    for c in CONDS:
        rs = [r for r in R[c] if r["arm"] == a]
        if rs:
            m, s, n = stat(rs, "mse_hole"); cells.append(f"{m:.4f}±{s:.4f}" + ("" if n == 3 else f" (n={n})"))
            pm, ps, _ = stat(rs, "psnr_range2"); fm, fs, _ = stat(rs, "mse_full"); xr = stat(rs, "x0_rms_last")[0]
            rows.append(dict(cond=c, arm=a, h0=rs[0]["h0"], n=n, mse_hole_mean=f"{m:.4f}", mse_hole_std=f"{s:.4f}", mse_full_mean=f"{fm:.4f}", mse_full_std=f"{fs:.4f}", psnr_mean=f"{pm:.2f}", psnr_std=f"{ps:.2f}", x0_rms_last=f"{xr:.3f}", cg_bad=int(sum(r["cg_bad"] for r in rs))))
        else: cells.append("–")
    md.append(f"| {a} | " + " | ".join(cells) + " |")
md += ["\nPSNR(range2) mean:\n", "| arm | " + " | ".join(LABEL[c] for c in CONDS) + " |", "|---|" + "---|" * len(CONDS)]
for a in ARMS:
    md.append(f"| {a} | " + " | ".join(next((r["psnr_mean"] for r in rows if r["cond"] == c and r["arm"] == a), "–") for c in CONDS) + " |")
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
FR = [0, 5, 10, 15, 20, 25, 30, 35, "last"]
gt = np.array(Image.open(f"{ROOT}/junco_gt.png")); mk = np.array(Image.open(f"{ROOT}/junco_masked.png"))
for c in CONDS[1:]:
    arms = [a for a in ARMS if os.path.exists(f"{OUT}/{c}/{a}/junco_s42/traj_x1.npy")]
    if not arms: continue
    fig, axes = plt.subplots(len(arms), len(FR) + 2, figsize=(2.0 * (len(FR) + 2), 2.1 * len(arms))); axes = np.atleast_2d(axes)
    for i, a in enumerate(arms):
        tr = np.load(f"{OUT}/{c}/{a}/junco_s42/traj_x1.npy"); fin = json.load(open(f"{OUT}/{c}/{a}/junco_s42/final.json"))
        for j, im in enumerate([gt, mk] + [tr[k] for k in range(len(FR))]):
            ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(["GT", "y (masked)"][j] if j < 2 else f"f{FR[j-2]}", fontsize=9)
        axes[i, 0].set_ylabel(f"{a}\nhole {fin['mse_hole']:.4f}", fontsize=9)
    fig.suptitle(f"junco box seed 42 — {LABEL[c]}", fontsize=11); plt.tight_layout(); plt.savefig(f"{OUT}/trajectory_{c}.png", dpi=110); plt.close()
    plt.figure(figsize=(8, 4))
    for a in arms:
        rr = list(csv.DictReader(open(f"{OUT}/{c}/{a}/junco_s42/trajectory_metrics.csv")))
        plt.plot([int(r["frame"]) for r in rr], [float(r["mse_hole"]) for r in rr], label=a, lw=1.4)
    plt.yscale("log"); plt.xlabel("frame"); plt.ylabel("hole MSE"); plt.legend(fontsize=8); plt.grid(alpha=.3); plt.title(f"junco box seed 42 — {LABEL[c]}")
    plt.tight_layout(); plt.savefig(f"{OUT}/mse_hole_vs_frame_{c}.png", dpi=110); plt.close()
# combined: hole MSE vs h0 per condition
plt.figure(figsize=(7, 4)); hs = {"h1.0": 1.0, "h0.5": 0.5, "h0.1": 0.1, "h5e-2": 5e-2, "h1e-2": 1e-2, "h5e-3": 5e-3, "h1e-5": 1e-5}
for c in CONDS:
    pts = [(hs[a], float(r["mse_hole_mean"]), float(r["mse_hole_std"])) for a in hs for r in rows if r["cond"] == c and r["arm"] == a]
    if pts:
        pts.sort(); plt.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o", ms=4, lw=1.3, capsize=2, label=LABEL[c])
        b = next((float(r["mse_hole_mean"]) for r in rows if r["cond"] == c and r["arm"] == "baseline"), None)
        if b is not None: plt.axhline(b, ls="--", lw=0.9, color=plt.gca().lines[-1].get_color(), alpha=.6)
plt.xscale("log"); plt.xlabel("h0 (dashed = that condition's exact-draw baseline)"); plt.ylabel("hole MSE (3 seeds)"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.title("junco box: Block-2 Langevin step vs h0 under noise conditions"); plt.tight_layout(); plt.savefig(f"{OUT}/hole_mse_vs_h0.png", dpi=120)
