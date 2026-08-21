#!/usr/bin/env python
"""Does gamma^2 ever help? Sweep it against the real network error.

The earlier test only fed back the measured gamma^2 and found it harmful at
every grid point. That leaves the harder question open: is the measured value
simply wrong, or is no positive ridge useful at all?

So at each (stage, tau) this sweeps gamma^2 over several decades — including the
measured value and multiples of it — and reports the argmin of ||x0_hat - x0||^2
against the network's own v. Because

    x1_hat - x1 = -sigma_tau H_tau^-1 (x0_hat - x0)

exactly, the ridge's relative effect on x1_hat is identical, so measuring x0_hat
settles both.

Cor. 8 derives the ridge under eps ~ N(0, gamma^2 I) independent of X0. The real
velocity error is neither white nor independent, so an argmin at zero does not
refute Cor. 8 — it says the assumption does not hold here.

    PYTHONHASHSEED=0 python gamma2_sweep.py [--image junco]
"""

import argparse
import os
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
measurement = _retry_import("measurement")
torch = _retry_import("torch")
_retry_import("csv")
_local = [p for p in sys.path if p and not p.startswith("/CBIG-Standard-ECE")]
_mounted = [p for p in sys.path if not p or p.startswith("/CBIG-Standard-ECE")]
sys.path[:] = _local + _mounted

import copy                                                    # noqa: E402
import csv                                                     # noqa: E402
import json                                                    # noqa: E402
from omegaconf import OmegaConf                                # noqa: E402
from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_H_tau, compute_sigma_tau, make_velocity_fn,
)
from ms_posterior_sampling_utils import class_guidance_scale   # noqa: E402
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="junco")
    ap.add_argument("--out", default="test/results/gamma2_sweep")
    cli = ap.parse_args()

    with open(os.path.join(A2, "config.json")) as f:
        cfg = json.load(f)
    paths = {k: alg2.resolve_path(v.replace("{IP_PACKAGE}", alg2.base.IP_PACKAGE)
                                  .replace("{HERE}", A2))
             for k, v in cfg["paths"].items()}
    alg2.PATHS = paths
    measurement.configure(cfg["tasks_setup"], cfg["algorithm"]["measurement_seed"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kw, alg = cfg["sampler_kw"], cfg["algorithm"]

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
    guidance = float(kw["guidance_scale"])
    do_cfg = guidance > 0
    labels = torch.tensor([int(demo["class_idx"])], dtype=torch.int32, device=device)
    prompt_embeds = (torch.cat([int(model.num_classes) * torch.ones_like(labels),
                                labels], dim=0) if do_cfg else labels)
    g = torch.Generator(device="cpu").manual_seed(int(alg["seed"]))
    gamma2_tab = json.load(open(paths["gamma2_table"]))["table"]

    rows = []
    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        steps = int(float(utils._per_stage(kw["ode_steps_per_stage"], si, num_stages)))
        sc.set_timesteps(steps, si, device=device, shift=float(kw["shift"]))
        s_k, e_k = float(sc.start_t[si]), float(sc.end_t[si])
        eff_si = si if bool(kw["g_bypass_stage3"]) else None
        x1_gt = pyr[si]
        h, w = x1_gt.shape[-2:]
        size_tensor, rope_pos = alg2.base.rope_for(model, h, w, device)
        g2_stage = gamma2_tab[str(si)]

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx])
            sigma_tau = float(compute_sigma_tau(tau, s_k, e_k))
            if sigma_tau < float(alg["sigma_min"]):
                continue
            x0 = torch.randn(x1_gt.shape, generator=g).to(device)
            x_tau = apply_H_tau(x1_gt, tau, s_k, e_k, eff_si) + sigma_tau * x0
            vfn = make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos,
                                   do_cfg, class_guidance_scale(guidance, si), si)
            with torch.no_grad():
                v = vfn(x_tau)

            g2_meas = float(g2_stage.get(f"{round(tau, 6)}",
                                         list(g2_stage.values())[step_idx]))
            grid = sorted({0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0,
                           g2_meas / 4, g2_meas / 2, g2_meas,
                           2 * g2_meas, 4 * g2_meas})
            errs = {}
            for g2 in grid:
                x0_hat = utils.score_solve(x_tau, v, s_k, e_k, tau, g2, eff_si,
                                           float(kw["cg_tol"]), int(kw["cg_max_iter"]))
                errs[g2] = float(((x0_hat - x0) ** 2).mean())
            best = min(errs, key=errs.get)
            e0, eb, em = errs[0.0], errs[best], errs[g2_meas]
            # At stage 0, tau=0 the state IS the noise (H_0 = s_0 G = 0 since
            # s_0 = 0), so x0 is recovered exactly and e0 is 0. Percentages are
            # meaningless there.
            deg = e0 == 0.0
            rows.append(dict(stage=si, tau=round(tau, 6), sigma_tau=sigma_tau,
                             gamma2_meas=g2_meas, best_gamma2=best,
                             err_at_0=e0, err_at_best=eb, err_at_meas=em,
                             gain_best_pct=float("nan") if deg else (e0 - eb) / e0 * 100,
                             loss_meas_pct=float("nan") if deg else (em - e0) / e0 * 100))
            print(f"[s{si} tau={tau:.3f}] best gamma2={best:.2e} "
                  f"(meas {g2_meas:.4f}) | err0={e0:.5f} best={eb:.5f} "
                  f"meas={em:.5f} | gain={rows[-1]['gain_best_pct']:+.2f}% "
                  f"meas cost={rows[-1]['loss_meas_pct']:+.2f}%", flush=True)

    out = os.path.join(A2, cli.out)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"gamma2_sweep_{cli.image}.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    ok = [r for r in rows if r["loss_meas_pct"] == r["loss_meas_pct"]]   # drop NaN
    n_help = sum(1 for r in ok if r["best_gamma2"] > 0)
    print(f"\n[summary] {len(rows)} grid points ({len(ok)} non-degenerate)")
    print(f"  argmin gamma^2 > 0 at: {n_help}/{len(ok)}")
    if n_help:
        gains = [r["gain_best_pct"] for r in ok if r["best_gamma2"] > 0]
        print(f"  gain from the best positive ridge: max {max(gains):+.3f}%, "
              f"mean {sum(gains) / len(gains):+.3f}%")
    print(f"  measured gamma^2 costs on average: "
          f"{sum(r['loss_meas_pct'] for r in ok) / len(ok):+.2f}%")
    print(f"[done] -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
