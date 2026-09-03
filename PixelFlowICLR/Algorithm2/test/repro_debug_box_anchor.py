#!/usr/bin/env python
"""Reproduce results/debug_box_anchor — the run that produced the deprecated
anchor's numbers (74a0c39, job 18635053).

Only `h0_sweep.csv` was ever committed; the two figures were not. The MSE-curve
figure is redrawn losslessly from that CSV. The final-x1 panel needs the
reconstructions, so this re-runs the verbatim deprecated sampler
(`_deprecated_anchor_ref.sample_alg2`) on the ORIGINAL measurement path:

    demo_runner.build_setup_and_measurement("box_inpainting",
        {"mask_len_range": [128, 129]}, demo, 0.05, 256, device)

i.e. NOT the current measurement.py wrapper, and with no "center": true -- that
option, and measurement_mode, were added later when the measurement was aligned
with the 8 baselines. The committed CSV is the check: anchor 0/1/5/25 must come
back as hole 0.969124 / 0.886168 / 0.451631 / 0.086083.

    PYTHONHASHSEED=0 python repro_debug_box_anchor.py
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ALG2 = os.path.dirname(HERE)
if ALG2 not in sys.path:
    sys.path.insert(0, ALG2)

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from omegaconf import OmegaConf                                # noqa: E402

import utils                                                   # noqa: E402
import main as alg2_main                                       # noqa: E402
import _deprecated_anchor_ref as ref                           # noqa: E402
from demo_runner import build_setup_and_measurement, load_demo_images  # noqa: E402

OUT = os.path.join(ALG2, "results", "debug_box_anchor")
ANCHORS = [0.0, 1.0, 5.0, 25.0]
IMAGE = "junco"
# the committed CSV's final hole MSE, per anchor -- the reproduction check
EXPECT = {0.0: 0.969124, 1.0: 0.886168, 5.0: 0.451631, 25.0: 0.086083}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", nargs="*", type=float, default=None,
                    help="anchor values to run (default: the historical 0/1/5/25)")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output files, so an extension sweep "
                         "does not overwrite the reproduction")
    a = ap.parse_args()
    anchors = ANCHORS if a.anchors is None else a.anchors
    tag = a.tag or ""
    device = "cuda:0"
    demos = load_demo_images(resolution=256,
                             demo_dir=os.path.join(utils.base.IP_PACKAGE, "demo"))
    demo = {d["short_name"]: d for d in demos}[IMAGE]
    gt = demo["gt"].unsqueeze(0).to(device)
    model_dir = os.path.join(utils.base.IP_PACKAGE, "pretrained_models", "c2img")
    mcfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    alg2_main.PATHS = {"model_dir": model_dir}
    model = alg2_main._load_model(mcfg, device)
    print("[setup] model loaded", flush=True)

    # the ORIGINAL measurement: raw demo_runner, box_len 128, no "center"
    sigma_n = 0.05
    op, mask, y, _, _, mkA, _ = build_setup_and_measurement(
        "box_inpainting", {"mask_len_range": [128, 129]}, demo, sigma_n, 256, device)
    setup = dict(op=op, y=y, mkA=mkA)
    kw = dict(num_langevin=10, ode_steps_per_stage=10, shift=1.0,
              guidance_scale=2.0, g_bypass_stage3=True, cg_tol=1e-5,
              cg_max_iter=50, class_label=int(demo["class_idx"]))
    gamma2_tab = json.load(open(os.path.join(ALG2, "gamma2_meas.json")))["table"]
    hole = (1.0 - mask).to(device)

    os.makedirs(OUT, exist_ok=True)
    finals, check = {}, []
    for anc in anchors:
        x1, rows, _ = ref.sample_alg2(model, mcfg, gt, y, setup, kw, sigma_n,
                                      gamma2_tab, device, seed=42, record=False,
                                      h0=0.1, anchor=anc)
        err = (x1 - gt) ** 2
        h = float((err * hole).sum() / (hole.sum() * 3))
        o = float((err * (1 - hole)).sum() / ((1 - hole).sum() * 3))
        finals[anc] = x1.detach().cpu()
        exp = EXPECT.get(anc)
        d = abs(h - exp) / exp if exp else ""
        check.append(dict(anchor=anc, hole=h, obs=o, committed_hole=exp if exp else "",
                          rel_diff=d, reproduced=(d < 0.02) if exp else ""))
        if exp:
            print(f"[anchor={anc:g}] hole={h:.6f} (committed {exp:.6f}, "
                  f"rel {d:.2%})  obs={o:.7f}", flush=True)
        else:
            print(f"[anchor={anc:g}] hole={h:.6f}  obs={o:.7f}  (new point)",
                  flush=True)

    with open(os.path.join(OUT, f"repro_check{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(check[0].keys()))
        w.writeheader(); w.writerows(check)

    def img(t):
        return ((t[0].detach().cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1)
    n = len(anchors)
    fig, ax = plt.subplots(1, n + 2, figsize=(3.2 * (n + 2), 3.6))
    ax[0].imshow(img(gt)); ax[0].set_title("GT")
    ax[1].imshow(img(y)); ax[1].set_title("y (observed)")
    for i, anc in enumerate(anchors):
        c = [r for r in check if r["anchor"] == anc][0]
        ax[i + 2].imshow(img(finals[anc]))
        ax[i + 2].set_title(f"anchor={anc:g}\nhole {c['hole']:.4f}  obs {c['obs']:.5f}",
                            fontsize=9)
    for a in ax:
        a.axis("off")
    fig.suptitle("Alg2 final x1 — box_inpainting / junco — Tweedie anchor sweep "
                 "(reproduction of results/debug_box_anchor)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"h0_sweep_final_x1{tag}.png"), dpi=150)
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
