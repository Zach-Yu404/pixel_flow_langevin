#!/usr/bin/env python
"""Re-measure the gamma^2 table (eq. 56) on a SINGLE image and overwrite
gamma2_meas.json with it.

The committed table averaged ||v - d_exact||^2 / n over the 7 shared images;
the sweeps and the debug-box harness all run on junco alone, so this rebuilds
the table for exactly that image. The measurement follows main.py's onestep
loop bit-for-bit: same eps_for(image, stage) noise, same tau grid
(set_timesteps per stage), same CFG call (raw guidance_scale — the per-stage
class_guidance_scale happens inside make_velocity_fn), and gamma^2 is recorded
at every grid point including sigma_tau < sigma_min, as main.py does.

The previous table is backed up to test/results/ before overwriting.

    PYTHONHASHSEED=0 python remeasure_gamma2.py [--image junco]
"""

import argparse
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
A2 = os.path.dirname(HERE)
if A2 not in sys.path:
    sys.path.insert(0, A2)


def _retry_import(name, tries=40, delay=3):
    last = None
    for attempt in range(1, tries + 1):
        try:
            __import__(name)
            return sys.modules[name]
        except OSError as exc:
            if getattr(exc, "errno", None) != 121:
                raise
            last = exc
            print(f"[boot] ceph EIO importing {name} ({attempt}/{tries})", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"ceph EIO persisted importing {name}") from last


alg2 = _retry_import("main")
utils = _retry_import("utils")
torch = _retry_import("torch")
_local = [p for p in sys.path if p and not p.startswith("/CBIG-Standard-ECE")]
_mounted = [p for p in sys.path if not p or p.startswith("/CBIG-Standard-ECE")]
sys.path[:] = _local + _mounted

import copy                                                    # noqa: E402
import json                                                    # noqa: E402
from omegaconf import OmegaConf                                # noqa: E402
from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_H_tau, compute_sigma_tau, make_velocity_fn,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="junco")
    cli = ap.parse_args()

    with open(os.path.join(A2, "config.json")) as f:
        cfg = json.load(f)
    paths = {k: alg2.resolve_path(v.replace("{IP_PACKAGE}", alg2.base.IP_PACKAGE)
                                  .replace("{HERE}", A2))
             for k, v in cfg["paths"].items()}
    alg2.PATHS = paths
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kw = cfg["sampler_kw"]

    demos = alg2.load_demo_images(resolution=256, demo_dir=paths["demo_dir"])
    demo = {d["short_name"]: d for d in demos}[cli.image]
    gt = demo["gt"].unsqueeze(0).to(device)
    model_cfg = OmegaConf.load(os.path.join(paths["model_dir"], "config.yaml"))
    model = alg2._load_model(model_cfg, device)
    print("[setup] model loaded", flush=True)

    num_stages = int(model_cfg.scheduler.num_stages)
    scheduler = PixelFlowScheduler(model_cfg.scheduler.num_train_timesteps,
                                   num_stages=num_stages, gamma=-1 / 3)
    pyr = alg2.base.gt_stage_pyramid(gt, num_stages)
    shift = float(kw["shift"])
    guidance_scale = float(kw["guidance_scale"])
    do_cfg = guidance_scale > 0
    g_bypass = bool(kw["g_bypass_stage3"])
    labels = torch.tensor([int(demo["class_idx"])], dtype=torch.int32, device=device)
    pe = (torch.cat([int(model.num_classes) * torch.ones_like(labels), labels], dim=0)
          if do_cfg else labels)

    table_path = os.path.join(A2, "gamma2_meas.json")
    old_tab = json.load(open(table_path))["table"] if os.path.exists(table_path) else {}

    gamma2_tab = {}
    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        steps_si = int(alg2.base.per_stage(kw["ode_steps_per_stage"], si, num_stages))
        sc.set_timesteps(steps_si, si, device=device, shift=shift)
        sk, ek = float(sc.start_t[si]), float(sc.end_t[si])
        eff_si = si if g_bypass else None
        x1_gt = pyr[si]
        h, w = x1_gt.shape[-2:]
        x0 = alg2.base.eps_for([cli.image], si, x1_gt.shape).to(device)
        size_tensor, rope_pos = alg2.base.rope_for(model, h, w, device)
        d_exact = utils.apply_B(x1_gt, sk, ek, eff_si) - (ek - sk) * x0

        for step_idx in range(len(sc.Timesteps)):
            T = sc.Timesteps[step_idx]
            tau = float(sc.t[step_idx])
            sigma_tau = compute_sigma_tau(tau, sk, ek)
            x_tau = apply_H_tau(x1_gt, tau, sk, ek, eff_si) + sigma_tau * x0
            vfn = make_velocity_fn(model, T, pe, size_tensor, rope_pos,
                                   do_cfg, guidance_scale, si)
            with torch.no_grad():
                v = vfn(x_tau)
            g2 = float(((v - d_exact) ** 2).mean(dim=(1, 2, 3)).mean())
            gamma2_tab.setdefault(si, {})[round(tau, 6)] = g2
            old = old_tab.get(str(si), {}).get(f"{round(tau, 6)}", float("nan"))
            print(f"[s{si} tau={tau:.3f}] gamma2={g2:.6f} (7-img avg was {old:.6f})",
                  flush=True)

    bak_dir = os.path.join(HERE, "results")
    os.makedirs(bak_dir, exist_ok=True)
    bak = os.path.join(bak_dir, "gamma2_meas_prev_7img.json")
    if os.path.exists(table_path) and not os.path.exists(bak):
        shutil.copy2(table_path, bak)
        print(f"[backup] old table -> {bak}", flush=True)

    json.dump({"note": "gamma2_meas is task-independent (no A/y in v or d_exact); "
                       f"keyed stage -> tau -> ||v - d_exact||^2 / n; measured on "
                       f"the single image '{cli.image}' (test/remeasure_gamma2.py)",
               "table": gamma2_tab},
              open(table_path, "w"), indent=1)
    print(f"[done] -> {table_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
