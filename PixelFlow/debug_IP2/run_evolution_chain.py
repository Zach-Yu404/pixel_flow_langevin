#!/usr/bin/env python
"""
Runs the full evolution chain in one process (loads the model once).

Steps (each isolates ONE change):
  E0 : OLD baseline (autograd ULA, works)
  E1 : ARTICLE as-shipped (G always low-pass, h_x=1e-3, joint eps)
  E2 : E1 + g_identity_stage_last=True (G=I at stage 3, match old DownUp)
  E3 : E2 + warm_x1_from_vpred=True (re-init x1_k from model velocity each outer step)
  E4 : E3 + warm_x1_from_wls=True (also re-init x1_k with WLS estimate at each inner step)
  E5 : E4 + h_x=5e-3, lambda_reg=0.1 (tuned step size)
  E6 : E5 + rho_s per-stage + skip_sigma=0.02 (avoid blow-up at tau≈1)
  E7 : Minimal article-like, no joint eps (eps frozen, debug whether eps-Langevin hurts)

Writes everything to debug_IP2/evolution/.
"""
import os, sys, json, time, copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from debug_IP2.run_harness import run_pipeline, save_figures

OUT = "debug_IP2/evolution"
os.makedirs(OUT, exist_ok=True)

BASE = {
    "seed": 20000120,
    "num_stages": 4,
    "resolution": 256,
    "num_examples": 2,
    "sigma_n": 0.05,
    "class_label": 10,
    "shift": 1.0,
    "stage_steps": 5,
    "num_langevin": 10,
    "measurement_mode": "measure",
    "active_operator": "box",
    "device": "cuda:0",
    "data_dir": "/data/Zach_dataset/imagenet256/train/",
    "model_dir": "./pretrained_models/c2img",
    "box_operator": {"mask_type": "box", "mask_len_range": [80, 160], "mask_prob_range": None},
    "random_operator": {"mask_type": "random", "mask_len_range": None, "mask_prob_range": [0.8, 0.8]},
}


def mk(overrides):
    d = dict(BASE)
    d.update(overrides)
    return d


EXPERIMENTS = {
    "E0_old":          mk({"use_old_langevin": True, "lr_base": 2e-2, "num_langevin": 10}),

    "E1_article_asis": mk({
        "use_old_langevin": False, "g_identity_stage_last": False,
        "h_x": 1e-3, "h_eps": 1e-3, "lambda_x": 0.01, "lambda_reg": 0.01,
        "rho_s": 1.0, "rho_e": 1.0,
        "warm_x1_from_vpred": False, "warm_x1_from_wls": False, "joint_eps": True,
        "sigma_floor": 0.01, "skip_sigma": 0.01,
    }),

    "E2_G_ident_last": mk({
        "use_old_langevin": False, "g_identity_stage_last": True,
        "h_x": 1e-3, "h_eps": 1e-3, "lambda_x": 0.01, "lambda_reg": 0.01,
        "rho_s": 1.0, "rho_e": 1.0,
        "warm_x1_from_vpred": False, "warm_x1_from_wls": False, "joint_eps": True,
        "sigma_floor": 0.01, "skip_sigma": 0.01,
    }),

    "E3_warm_vpred": mk({
        "use_old_langevin": False, "g_identity_stage_last": True,
        "h_x": 1e-3, "h_eps": 1e-3, "lambda_x": 0.01, "lambda_reg": 0.01,
        "rho_s": 1.0, "rho_e": 1.0,
        "warm_x1_from_vpred": True, "warm_x1_from_wls": False, "joint_eps": True,
        "sigma_floor": 0.01, "skip_sigma": 0.01,
    }),

    "E4_warm_wls": mk({
        "use_old_langevin": False, "g_identity_stage_last": True,
        "h_x": 1e-3, "h_eps": 1e-3, "lambda_x": 0.01, "lambda_reg": 0.01,
        "rho_s": 1.0, "rho_e": 1.0,
        "warm_x1_from_vpred": True, "warm_x1_from_wls": True, "joint_eps": True,
        "sigma_floor": 0.01, "skip_sigma": 0.01,
    }),

    "E5_tuned_hx": mk({
        "use_old_langevin": False, "g_identity_stage_last": True,
        "h_x": 5e-3, "h_eps": 5e-3, "lambda_x": 0.1, "lambda_reg": 0.1,
        "rho_s": 1.0, "rho_e": 1.0,
        "warm_x1_from_vpred": True, "warm_x1_from_wls": True, "joint_eps": True,
        "sigma_floor": 0.05, "skip_sigma": 0.05,
    }),

    "E6_rho_per_stage": mk({
        "use_old_langevin": False, "g_identity_stage_last": True,
        "h_x": 5e-3, "h_eps": 5e-3, "lambda_x": 0.1, "lambda_reg": 0.1,
        "rho_s": [0.1, 0.3, 0.6, 1.0], "rho_e": 1.0,
        "warm_x1_from_vpred": True, "warm_x1_from_wls": True, "joint_eps": True,
        "sigma_floor": 0.05, "skip_sigma": 0.05,
    }),

    "E7_frozen_eps": mk({
        "use_old_langevin": False, "g_identity_stage_last": True,
        "h_x": 5e-3, "h_eps": 5e-3, "lambda_x": 0.1, "lambda_reg": 0.1,
        "rho_s": [0.1, 0.3, 0.6, 1.0], "rho_e": 1.0,
        "warm_x1_from_vpred": True, "warm_x1_from_wls": True, "joint_eps": False,
        "sigma_floor": 0.05, "skip_sigma": 0.05,
    }),
}


def main():
    results = {}
    for name, flags in EXPERIMENTS.items():
        print(f"\n===== {name} =====")
        t0 = time.time()
        try:
            r = run_pipeline(flags)
            print(f"  {name} final residual = {r['residuals'][-1]:.2f}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            continue
        results[name] = r
        torch.save(r, os.path.join(OUT, f"{name}.pt"))

    save_figures(results, OUT, "evolution_chain")

    # Write JSON summary of per-step residuals
    summary = {name: {
        "final_residual": r["residuals"][-1] if r["residuals"] else None,
        "residuals": r["residuals"],
    } for name, r in results.items()}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {os.path.join(OUT, 'summary.json')}")


if __name__ == "__main__":
    main()
