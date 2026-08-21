#!/usr/bin/env python
"""Section 7.2 (draft p.20): the score solve in isolation, one step only.

For every (stage k, tau) the ground truth alone builds the state — no sampler,
no measurement, no A:

    x_tau = H_tau x1_gt + sigma_tau x0        d_tau = B_k x1_gt - (e_k - s_k) x0   (49)

Two solves of (38) with gamma = 0, then the implied clean image of p.20:

    x1_hat = (H_tau)^-1 (x_tau - sigma_tau x0_hat)

  exact : x0_hat from d_tau. Prop. 7 is an identity here, so x1_hat must equal
          x1_gt to solver tolerance. "Any discrepancy is a bug, not an
          approximation."
  model : x0_hat from v_theta. ||x0_hat - x0|| is the network error that defines
          gamma^2; x1_hat is what that error looks like as an image.

The paper's acceptance criteria for the model column are qualitative — it should
be "recognizably the right image, blurry at small tau and sharpening as tau -> 1,
and never noisier than x_tau itself" — so the run reports x1_hat's MSE against
the GT next to x_tau's own, splits both by the box mask so the unobservable
region is visible on its own, and writes the panels to look at.

    PYTHONHASHSEED=0 python score_x1hat.py [--image junco] [--stages 0,1,2,3]
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
    """The mount throws EIO in bursts while Python scans sys.path."""
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
import numpy as np                                             # noqa: E402
import torch.nn.functional as F                                # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from omegaconf import OmegaConf                                # noqa: E402
from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_H_tau, compute_sigma_tau, make_velocity_fn,
)
from ms_posterior_sampling_utils import class_guidance_scale   # noqa: E402
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402


def to_img(t):
    """[-1,1] CHW tensor -> HWC uint8-ish float for imshow."""
    return ((t.detach().float().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="junco")
    ap.add_argument("--task", default="box_inpainting",
                    help="only used to split the metrics by mask; A never enters 7.2")
    ap.add_argument("--stages", default="0,1,2,3")
    ap.add_argument("--out", default="test/results/score_x1hat")
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
    kw, alg = cfg["sampler_kw"], cfg["algorithm"]
    spec = cfg["tasks_setup"][cli.task]

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

    # mask only splits the metrics; 7.2 never touches A
    _, mask, _, _, _, _, _ = measurement.build_setup_and_measurement(
        cli.task, spec["operator"], demo, float(spec["sigma_n"]), 256, device)
    hole_full = (1.0 - mask).to(device)

    guidance = float(kw["guidance_scale"])
    do_cfg = guidance > 0
    labels = torch.tensor([int(demo["class_idx"])], dtype=torch.int32, device=device)
    prompt_embeds = (torch.cat([int(model.num_classes) * torch.ones_like(labels),
                                labels], dim=0) if do_cfg else labels)

    g = torch.Generator(device="cpu").manual_seed(int(alg["seed"]))
    gamma2_tab = json.load(open(paths["gamma2_table"]))["table"]
    stages = [int(s) for s in cli.stages.split(",")]

    rows, panels = [], {}
    for si in range(num_stages):
        if si not in stages:
            continue
        sc = copy.deepcopy(scheduler)
        steps = int(float(utils._per_stage(kw["ode_steps_per_stage"], si, num_stages)))
        sc.set_timesteps(steps, si, device=device, shift=float(kw["shift"]))
        s_k, e_k = float(sc.start_t[si]), float(sc.end_t[si])
        eff_si = si if bool(kw["g_bypass_stage3"]) else None
        x1_gt = pyr[si]
        h, w = x1_gt.shape[-2:]
        hole = F.interpolate(hole_full, size=(h, w), mode="nearest")
        obs = 1.0 - hole
        size_tensor, rope_pos = alg2.base.rope_for(model, h, w, device)
        g2_stage = gamma2_tab[str(si)]
        panels[si] = []

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx])
            sigma_tau = float(compute_sigma_tau(tau, s_k, e_k))
            # H_tau is singular at tau=0 unless G=I, so x1_hat is undefined there
            if tau == 0.0 and eff_si != 3:
                print(f"[s{si} tau=0.000] skipped: H_0 = s_k G is singular", flush=True)
                continue

            x0 = torch.randn(x1_gt.shape, generator=g).to(device)
            x_tau = apply_H_tau(x1_gt, tau, s_k, e_k, eff_si) + sigma_tau * x0
            d_exact = utils.apply_B(x1_gt, s_k, e_k, eff_si) - (e_k - s_k) * x0   # (49)

            vfn = make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos,
                                   do_cfg, class_guidance_scale(guidance, si), si)
            with torch.no_grad():
                v = vfn(x_tau)

            g2 = float(g2_stage.get(f"{round(tau, 6)}",
                                    list(g2_stage.values())[step_idx]))
            L = int(kw["cg_max_iter"])
            solves = {
                "exact": utils.score_solve(x_tau, d_exact, s_k, e_k, tau, 0.0,
                                           eff_si, float(kw["cg_tol"]), L),
                "model": utils.score_solve(x_tau, v, s_k, e_k, tau, 0.0,
                                           eff_si, float(kw["cg_tol"]), L),
                "model_g2": utils.score_solve(x_tau, v, s_k, e_k, tau, g2,
                                              eff_si, float(kw["cg_tol"]), L),
            }
            rec = dict(stage=si, step=step_idx, tau=round(tau, 6),
                       sigma_tau=sigma_tau, gamma2=g2,
                       x_tau_mse=float(((x_tau - x1_gt) ** 2).mean()),
                       x_tau_hole=utils.mse_masked(x_tau, x1_gt, hole))
            imgs = {}
            for name, x0_hat in solves.items():
                x1_hat = utils.apply_H_tau_inv(x_tau - sigma_tau * x0_hat,
                                               tau, s_k, e_k, eff_si)
                imgs[name] = x1_hat
                rec[f"x0err_{name}"] = float(((x0_hat - x0) ** 2).mean())
                rec[f"x1_{name}"] = float(((x1_hat - x1_gt) ** 2).mean())
                rec[f"x1_{name}_hole"] = utils.mse_masked(x1_hat, x1_gt, hole)
                rec[f"x1_{name}_obs"] = utils.mse_masked(x1_hat, x1_gt, obs)
            rows.append(rec)
            panels[si].append((tau, x_tau[0], imgs["exact"][0], imgs["model"][0]))
            print(f"[s{si} tau={tau:.3f} sig={sigma_tau:.3f}] "
                  f"exact: x0err={rec['x0err_exact']:.2e} x1={rec['x1_exact']:.2e} | "
                  f"model: x0err={rec['x0err_model']:.4f} "
                  f"x1={rec['x1_model']:.4f} (hole {rec['x1_model_hole']:.4f}, "
                  f"obs {rec['x1_model_obs']:.4f}) | x_tau={rec['x_tau_mse']:.4f}",
                  flush=True)

    out = os.path.join(A2, cli.out)
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, f"score_x1hat_{cli.image}.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    for si, items in panels.items():
        if not items:
            continue
        n = len(items)
        fig, ax = plt.subplots(4, n, figsize=(1.7 * n, 7.2))
        ax = np.atleast_2d(ax)
        for j, (tau, xt, xe, xm) in enumerate(items):
            for i, (im, lab) in enumerate([(pyr[si][0], "GT"), (xt, "x_tau"),
                                           (xe, "x1_hat exact"), (xm, "x1_hat model")]):
                ax[i, j].imshow(to_img(im)); ax[i, j].axis("off")
                if j == 0:
                    ax[i, j].set_ylabel(lab)
                    ax[i, j].axis("on"); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[0, j].set_title(f"τ={tau:.2f}", fontsize=8)
        fig.suptitle(f"7.2 one-step x1_hat — stage {si}, {cli.image}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"panel_stage{si}_{cli.image}.png"), dpi=110)
        plt.close(fig)

    print(f"[done] {len(rows)} rows -> {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
