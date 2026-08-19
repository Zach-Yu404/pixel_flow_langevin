#!/usr/bin/env python
"""GT diagnostic: is Block 1 the bottleneck, or is it prior propagation?

Builds x_tau from the GROUND TRUTH at every (stage, tau),

    x_tau^k = H_tau^k x1_gt^k + sigma_tau^k x0^k ,

and compares three estimates of x1 against x1_gt, split into the observed
region and the measurement null space (the hole):

  1. Block-1 conditional mean   — solve M_tau m = b_tau   (deterministic RHS)
  2. Block-1 exact sample       — solve M_tau x = b_tilde (xi_y / xi_h, Lem. 5)
  3. network-implied x1_hat     — one velocity call, Tweedie clean endpoint

If (1) and (2) already reconstruct the hole from a GT-consistent x_tau, then
Block 1 is fine and the failure lives in getting the prior into x_tau, i.e. in
Block-2 mixing. If they do not, Block 1 itself cannot fill the null space and
no amount of Block-2 mixing will help.

The hole column is also compared against sigma_tau/(tau*e_k), the predicted
null-space amplification (on ker G, H_tau = tau*e_k I).

    PYTHONHASHSEED=0 python gt_diagnostic.py [--task box_inpainting] [--image junco]
"""

import argparse
import json
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
measurement = _retry_import("measurement")
utils = _retry_import("utils")
_retry_import("forward_operator.motionblur.motionblur")
torch = _retry_import("torch")
_retry_import("csv")
_local = [p for p in sys.path if p and not p.startswith("/CBIG-Standard-ECE")]
_mounted = [p for p in sys.path if not p or p.startswith("/CBIG-Standard-ECE")]
sys.path[:] = _local + _mounted

import copy                                                    # noqa: E402
import csv                                                     # noqa: E402
import torch.nn.functional as F                                # noqa: E402
from omegaconf import OmegaConf                                # noqa: E402
from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_H_tau, compute_sigma_tau, direct_estimate_x1, make_velocity_fn,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402
from ms_posterior_sampling_utils import class_guidance_scale   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="box_inpainting")
    ap.add_argument("--image", default="junco")
    ap.add_argument("--out", default="test/results/gt_diagnostic")
    cli = ap.parse_args()

    with open(os.path.join(A2, "config.json")) as f:
        cfg = json.load(f)
    paths = {k: alg2.resolve_path(v.replace("{IP_PACKAGE}", alg2.base.IP_PACKAGE)
                                  .replace("{HERE}", A2))
             for k, v in cfg["paths"].items()}
    alg2.PATHS = paths
    alg2.TASKS_SETUP = cfg["tasks_setup"]
    alg2.SAMPLER_KW = cfg["sampler_kw"]
    measurement.configure(cfg["tasks_setup"], cfg["algorithm"]["measurement_seed"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kw = cfg["sampler_kw"]
    spec = cfg["tasks_setup"][cli.task]
    sigma_n = float(spec["sigma_n"])
    alg_cfg = cfg["algorithm"]

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

    op, mask, y, _, _, make_Ak_fns_fn, _ = measurement.build_setup_and_measurement(
        cli.task, spec["operator"], demo, sigma_n, 256, device)
    hole_full = (1.0 - mask).to(device)

    guidance = float(kw["guidance_scale"])
    do_cfg = guidance > 0
    labels = torch.tensor([int(demo["class_idx"])], dtype=torch.int32, device=device)
    if do_cfg:
        prompt_embeds = torch.cat([int(model.num_classes) * torch.ones_like(labels),
                                   labels], dim=0)
    else:
        prompt_embeds = labels

    g = torch.Generator(device="cpu").manual_seed(int(alg_cfg["seed"]))

    def randn_like_cpu(x):
        return torch.randn(x.shape, generator=g).to(x.device)

    rows = []
    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        steps = int(float(utils._per_stage(kw["ode_steps_per_stage"], si, num_stages)))
        sc.set_timesteps(steps, si, device=device, shift=float(kw["shift"]))
        s_k, e_k = float(sc.start_t[si]), float(sc.end_t[si])
        eff_si = si if bool(kw["g_bypass_stage3"]) else None
        x1_gt = pyr[si]
        h, w = x1_gt.shape[-2:]
        hole = F.interpolate(hole_full, size=(h, w), mode="nearest")
        obs = 1.0 - hole
        Ak, ATk = make_Ak_fns_fn(op, y, (1, 3, h, w), device)
        size_tensor, rope_pos = alg2.base.rope_for(model, h, w, device)

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx])
            sigma_tau = compute_sigma_tau(tau, s_k, e_k)
            if sigma_tau < float(alg_cfg["sigma_min"]):
                continue
            x0 = randn_like_cpu(x1_gt)
            x_tau = apply_H_tau(x1_gt, tau, s_k, e_k, eff_si) + sigma_tau * x0

            # (1) Block-1 conditional mean: deterministic RHS
            m_hat = utils.clean_image_solve(
                x_tau, Ak, ATk, y, sigma_n, sigma_tau, tau, s_k, e_k, eff_si,
                torch.zeros_like(x1_gt), float(kw["cg_tol"]),
                ridge_rel=float(alg_cfg["ridge_rel"]),
                max_iter=int(alg_cfg["cg_max_iter_l14"]))

            # (2) Block-1 exact sample: same M_tau, RHS with the Lemma-5 noise
            M_tau, inv_e2, inv_s2, epsilon = utils.make_M_tau(
                Ak, ATk, sigma_n, sigma_tau, tau, s_k, e_k, eff_si,
                x1_gt.shape, device, float(alg_cfg["ridge_rel"]))
            xi_y, xi_h = randn_like_cpu(y), randn_like_cpu(x1_gt)
            b_tilde = (utils.data_rhs(ATk, y, x_tau, inv_e2, inv_s2,
                                      tau, s_k, e_k, eff_si)
                       + (1.0 / sigma_n) * ATk(xi_y)
                       + (1.0 / float(sigma_tau)) * apply_H_tau(xi_h, tau, s_k, e_k, eff_si))
            if epsilon:
                b_tilde = b_tilde + epsilon ** 0.5 * randn_like_cpu(x1_gt)
            x1_draw = utils.cg_solve(M_tau, b_tilde, x0=torch.zeros_like(x1_gt),
                                     tol=float(kw["cg_tol"]),
                                     max_iter=int(alg_cfg["cg_max_iter_l14"]))

            # (3) network-implied x1_hat: one velocity call (Tweedie clean endpoint)
            vfn = make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos,
                                   do_cfg, class_guidance_scale(guidance, si), si)
            with torch.no_grad():
                v = vfn(x_tau)
            x1_model = direct_estimate_x1(x_tau - tau * v, x_tau + (1.0 - tau) * v,
                                          s_k, e_k)

            def split(a):
                return (utils.mse_masked(a, x1_gt, hole),
                        utils.mse_masked(a, x1_gt, obs))

            mh, mo = split(m_hat)
            dh, do_ = split(x1_draw)
            nh, no = split(x1_model)
            rows.append(dict(stage=si, step=step_idx, tau=round(tau, 6),
                             sigma_tau=float(sigma_tau),
                             predicted_amp=float(sigma_tau / (tau * e_k)) if tau > 0 else float("inf"),
                             mean_hole=mh, mean_obs=mo,
                             draw_hole=dh, draw_obs=do_,
                             model_hole=nh, model_obs=no))
            print(f"[s{si} tau={tau:.3f} sig={float(sigma_tau):.3f}] "
                  f"hole  mean={mh:.4f} draw={dh:.4f} model={nh:.4f} | "
                  f"obs  mean={mo:.4f} draw={do_:.4f} model={no:.4f}", flush=True)

    out = os.path.join(A2, cli.out)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"gt_diag_{cli.task}_{cli.image}.csv")
    with open(path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    print(f"[done] {len(rows)} rows -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
