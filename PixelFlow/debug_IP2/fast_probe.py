#!/usr/bin/env python
"""Fast, focused probe — smaller problem + more configs at once."""
import os, sys, time, json, copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from debug_IP2.run_harness import run_pipeline, save_figures

OUT = "debug_IP2/fast_probe"
os.makedirs(OUT, exist_ok=True)

BASE = {
    "seed": 20000120, "num_stages": 4, "resolution": 256, "num_examples": 2,
    "sigma_n": 0.05, "class_label": 10, "shift": 1.0,
    "stage_steps": 3,               # reduced to speed up
    "num_langevin": 10,
    "measurement_mode": "measure", "active_operator": "box",
    "device": "cuda:0", "data_dir": "/data/Zach_dataset/imagenet256/train/",
    "model_dir": "./pretrained_models/c2img",
    "box_operator": {"mask_type": "box", "mask_len_range": [80, 160], "mask_prob_range": None},
    "random_operator": {"mask_type": "random", "mask_len_range": None, "mask_prob_range": [0.8, 0.8]},
    "use_old_langevin": False, "g_identity_stage_last": False,
    "warm_x1_from_vpred": True,
    "joint_eps": True,
    "sigma_floor": 0.02, "skip_sigma": 0.02,
    "rho_s": [0.1, 0.3, 0.6, 1.0], "rho_e": 1.0,
    "lambda_x": 0.1, "h_eps": 1e-3,
}
def mk(over):
    d = dict(BASE); d.update(over); return d

EXPERIMENTS = {
    # Baseline: article defaults
    "F0_baseline":         mk({"h_x": 1e-3, "lambda_reg": 1e-2}),
    # Larger h_x
    "F1_hx5e-3":           mk({"h_x": 5e-3, "lambda_reg": 5e-2}),
    "F2_hx5e-2":           mk({"h_x": 5e-2, "lambda_reg": 5e-1}),
    # Freeze eps (no joint Langevin)
    "F3_frozen_eps_hx5e-2": mk({"h_x": 5e-2, "lambda_reg": 5e-1, "joint_eps": False}),
    # Reset eps each outer step
    "F4_reset_eps_hx5e-2":  mk({"h_x": 5e-2, "lambda_reg": 5e-1, "reset_eps_each_step": True}),
    # WLS warm start (instead of x1_k carry-over), no joint eps
    "F5_wls_warm_frozen":   mk({"h_x": 5e-2, "lambda_reg": 5e-1, "joint_eps": False,
                                "warm_x1_from_wls": True}),
}


def main():
    results = {}
    for name, flags in EXPERIMENTS.items():
        print(f"\n--- {name} ---", flush=True)
        t0 = time.time()
        try:
            r = run_pipeline(flags)
            resids = r['residuals']
            stages = [resids[s*3:(s+1)*3] for s in range(4)]
            print(f"  {name} final={resids[-1]:.1f} min={min(resids):.1f} ({time.time()-t0:.1f}s)", flush=True)
            for s, sub in enumerate(stages):
                print(f"    stage{s}: {[f'{v:.0f}' for v in sub]}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            continue
        results[name] = r
        torch.save(r, os.path.join(OUT, f"{name}.pt"))

    save_figures(results, OUT, "fast_probe")
    summary = {n: {"final_res": r['residuals'][-1] if r['residuals'] else None,
                   "residuals": r['residuals']} for n, r in results.items()}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
