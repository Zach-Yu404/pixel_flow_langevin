"""Stitch per-config summary PNGs (which exist for every run) into a top panel.

Each per-config PNG already contains [GT | Measurement | Recon | box-zoom] for
both birds, so we just stack them vertically with a coloured banner highlighting
the winner.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
EXP  = os.path.dirname(HERE)
LB   = os.path.dirname(EXP)


def find_png(name):
    for d in [os.path.join(EXP, "runs_f"),
              os.path.join(EXP, "runs_p"),
              os.path.join(EXP, "runs"),
              os.path.join(LB,  "results")]:
        p = os.path.join(d, f"{name}.png")
        if os.path.exists(p): return p
    return None


def find_json(name):
    for d in [os.path.join(EXP, "runs_f"),
              os.path.join(EXP, "runs_p"),
              os.path.join(EXP, "runs"),
              os.path.join(LB,  "results")]:
        p = os.path.join(d, f"{name}.json")
        if os.path.exists(p):
            return json.load(open(p))
    return None


PANEL = [
    ("F_cfg20_lr150_hxu02",   "★ WINNER  CFG=2 + λ_reg=150 + h_x=0.2 uniform   (same-batch F)"),
    ("F_cfg25_lr200_L7",      "Runner-up  CFG=2.5 + λ_reg=200 + L=7   (same-batch F)"),
    ("F_anchor_N_cfg2",       "Anchor  N_cfg2  (CFG=2, defaults)   (same-batch F)"),
    ("F_anchor_C_L5",         "Anchor  C_L5    (CFG=0, defaults)   (same-batch F)"),
    ("I_ref_F1",              "Reference  ref_baseline_F1 (paper IP4 default)"),
    ("C_L10",                 "Existing best LPIPS  C_L10"),
]


def main():
    rows = []
    for name, label in PANEL:
        p = find_png(name)
        m = find_json(name)
        if p is None or m is None:
            print(f"[skip] {name}: missing png or json")
            continue
        rows.append((name, label, p, m))

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.0 * n))
    if n == 1: axes = [axes]
    for ax, (name, label, p, m) in zip(axes, rows):
        img = mpimg.imread(p)
        ax.imshow(img); ax.axis("off")
        is_winner = name.startswith("F_cfg20_lr150_hxu02")
        bg = "#fde8e8" if is_winner else "#f6f7fb"
        ec = "#cf2222" if is_winner else "#3b3f55"
        ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                         facecolor=bg, alpha=0.0, zorder=-1))
        title = (f"{label}\nPSNR={m['psnr']:.2f} dB   LPIPS={m['lpips']:.4f}   "
                 f"|dHF|={m.get('dhf',0):.4f}   t={m.get('time',0):.0f}s")
        ax.set_title(title, fontsize=10, color=ec, weight=("bold" if is_winner else "normal"))

    plt.suptitle("PRINCIPLE 128×128 box, res=256 — Top Comparison Panel\n"
                 "Each row is one config: [GT img1 | Measurement1 | Recon1 | box1 zoom]  +  [GT img2 | Measurement2 | Recon2 | box2 zoom]",
                 fontsize=12, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(EXP, "top_comparison_panel.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
