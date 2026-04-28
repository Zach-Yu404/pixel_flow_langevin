#!/usr/bin/env python
"""
Probe: sweep h_x / lambda_reg to see if large step size is what article version
needs. Also probes disabling eps Langevin ("frozen eps") and using x1_hat directly
instead of Langevin-refined x1.
"""
import os, sys, time, json, copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from debug_IP2.run_harness import run_pipeline, save_figures

OUT = "debug_IP2/probe_h_x"
os.makedirs(OUT, exist_ok=True)

BASE = {
    "seed": 20000120, "num_stages": 4, "resolution": 256, "num_examples": 2,
    "sigma_n": 0.05, "class_label": 10, "shift": 1.0, "stage_steps": 5,
    "num_langevin": 10, "measurement_mode": "measure", "active_operator": "box",
    "device": "cuda:0", "data_dir": "/data/Zach_dataset/imagenet256/train/",
    "model_dir": "./pretrained_models/c2img",
    "box_operator": {"mask_type": "box", "mask_len_range": [80, 160], "mask_prob_range": None},
    "random_operator": {"mask_type": "random", "mask_len_range": None, "mask_prob_range": [0.8, 0.8]},
    "use_old_langevin": False, "g_identity_stage_last": True,
    "warm_x1_from_vpred": True,  # always re-init x1 from v-pred
    "joint_eps": True,
    "sigma_floor": 0.02, "skip_sigma": 0.02,
    "rho_s": [0.1, 0.3, 0.6, 1.0], "rho_e": 1.0,
    "lambda_x": 0.01,
    "h_eps": 1e-3,
}


def mk(over):
    d = dict(BASE); d.update(over); return d


# Sweep: h_x × lambda_reg combinations
EXPERIMENTS = {
    # Small h_x (current default)
    "P0_hx1e-3_lr1e-2":   mk({"h_x": 1e-3, "lambda_reg": 1e-2}),
    # Larger h_x
    "P1_hx1e-2_lr1e-1":   mk({"h_x": 1e-2, "lambda_reg": 1e-1}),
    "P2_hx5e-2_lr5e-1":   mk({"h_x": 5e-2, "lambda_reg": 5e-1}),
    "P3_hx1e-1_lr1e0":    mk({"h_x": 1e-1, "lambda_reg": 1e0}),
    # h_x very large
    "P4_hx5e-1_lr1e0":    mk({"h_x": 5e-1, "lambda_reg": 1e0}),
    # Try dropping lambda_reg
    "P5_hx1e-2_lr1e-3":   mk({"h_x": 1e-2, "lambda_reg": 1e-3}),
    # Frozen eps (sanity): only x1 Langevin
    "P6_frozen_eps":      mk({"h_x": 5e-2, "lambda_reg": 5e-1, "joint_eps": False}),
    # No warm restart from v-pred: see if cold is enough with bigger h
    "P7_cold_big_hx":     mk({"h_x": 1e-1, "lambda_reg": 1e0, "warm_x1_from_vpred": False}),
    # Warm from WLS also
    "P8_hx1e-1_wls":      mk({"h_x": 1e-1, "lambda_reg": 1e0, "warm_x1_from_wls": True}),
}


def main():
    results = {}
    for name, flags in EXPERIMENTS.items():
        print(f"\n--- {name} ---")
        t0 = time.time()
        try:
            r = run_pipeline(flags)
            print(f"  {name} final={r['residuals'][-1]:.2f} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            continue
        results[name] = r
        torch.save(r, os.path.join(OUT, f"{name}.pt"))

    save_figures(results, OUT, "probe_h_x")
    summary = {n: {"final_res": r['residuals'][-1] if r['residuals'] else None,
                   "residuals": r['residuals']} for n, r in results.items()}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
