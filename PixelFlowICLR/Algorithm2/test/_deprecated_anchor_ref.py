#!/usr/bin/env python
"""VERBATIM copy of the deprecated Tweedie anchor, for comparison only.

`sample_alg2` as it stood at commit 74a0c39
(`PixelFlowICLR/Algorithm2/full_ip_compare.py`), before 8ff8091 removed it.
Nothing here is used by the shipped samplers -- it exists so
test_wv_alg2/test_reg_alg2 can run the historical code against the current
`run_posterior_reg_sampling_alg2` on an identical setup and show exactly where
the two differ.

Only the import plumbing is adapted: the old `algorithm2` module was later
consolidated into `utils`, so `alg.` -> `utils.`. Every line of the sampler
body below is unchanged, including the two things the current version fixes
(no sqrt(ridge) RHS term at tau=0; xi_a drawn only when anchor > 0).
"""
import copy
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ALG2 = os.path.dirname(HERE)
if ALG2 not in sys.path:
    sys.path.insert(0, ALG2)

import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
import utils as alg                                             # noqa: E402
import onestep_mse_vs_t as base                                 # noqa: E402
from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_H_tau, compute_sigma_tau, cg_solve, make_velocity_fn,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler   # noqa: E402

H0 = 0.1


def sample_alg2(model, config, gt, y, setup, kw, eta, gamma2_tab, device,
                seed=42, record=False, h0=H0, anchor=0.0):
    # anchor > 0: EXPERIMENTAL Tweedie-anchored Block 1 (task debug-box-alg2-hole).
    # Treats x1_model = direct_estimate_x1(v) as a Gaussian pseudo-observation with
    # precision `anchor`: M += anchor*I, b += anchor*x1_model + sqrt(anchor)*xi_a
    # (keeps the Lem.-5 exact-draw semantics for the anchored conditional; zero
    # extra NFE — reuses the l.10 velocity). anchor=0.0 reproduces the paper's
    # Algorithm 2 unchanged.
    """Algorithm 2 (PDF p.13) for ONE image [1,3,256,256]."""
    num_stages = int(config.scheduler.num_stages)
    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=num_stages, gamma=-1 / 3)
    shift = float(kw.get("shift", 1.0))
    guidance_scale = float(kw.get("guidance_scale", 0.0))
    do_cfg = guidance_scale > 0
    g_bypass = bool(kw.get("g_bypass_stage3", True))
    cg_tol = float(kw.get("cg_tol", 1e-5))
    L = int(kw.get("cg_max_iter", 50))
    S = int(float(kw.get("num_langevin", 10)))
    grid_n = int(float(kw.get("ode_steps_per_stage", 10)))
    class_label = int(kw.get("class_label", 10))
    num_classes = int(model.num_classes)

    pyr = base.gt_stage_pyramid(gt, num_stages)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def randn_like_cpu(x):
        return torch.randn(x.shape, generator=g).to(x.device)

    x1 = torch.zeros_like(pyr[0])                       # line 1 (stage-0 res)
    rows, traj = [], []
    for k in range(num_stages):
        x0 = randn_like_cpu(pyr[k])                     # line 3 / line 20 fresh draw
        sc = copy.deepcopy(scheduler)
        sc.set_timesteps(grid_n, k, device=device, shift=shift)
        sk, ek = float(sc.start_t[k]), float(sc.end_t[k])
        eff = k if g_bypass else None
        h, w = pyr[k].shape[-2:]
        A_fn, AT_fn = setup["mkA"](setup["op"], setup["y"], pyr[k][:1].shape, device)
        size_tensor, rope_pos = base.rope_for(model, h, w, device)
        pe = torch.tensor([class_label], dtype=torch.int32, device=device)
        if do_cfg:
            pe = torch.cat([num_classes * torch.ones_like(pe), pe], dim=0)
        g2_stage = gamma2_tab[str(k)]

        for step_idx in range(len(sc.Timesteps)):
            tau = float(sc.t[step_idx])
            sigma_t = compute_sigma_tau(tau, sk, ek)
            T = sc.Timesteps[step_idx]
            vfn = make_velocity_fn(model, T, pe, size_tensor, rope_pos,
                                   do_cfg, guidance_scale, k)
            g2 = float(g2_stage.get(f"{round(tau, 6)}",
                                    list(g2_stage.values())[step_idx]))
            x0_hat = None
            if sigma_t >= alg.SIGMA_MIN:
                inv_e2, inv_s2 = 1.0 / eta ** 2, 1.0 / float(sigma_t) ** 2

                def M0(x):
                    return inv_e2 * AT_fn(A_fn(x)) + inv_s2 * apply_H_tau(
                        apply_H_tau(x, tau, sk, ek, eff), tau, sk, ek, eff)
                ridge = alg.power_iter_norm(M0, x1.shape, device) * 1e-6 \
                    if tau == 0.0 else 0.0
                diag = ridge + anchor
                M_fn = (lambda x: M0(x) + diag * x) if diag else M0

                for s in range(S):
                    x_tau = apply_H_tau(x1, tau, sk, ek, eff) + sigma_t * x0  # l.9
                    with torch.no_grad():
                        v = vfn(x_tau)                                        # l.10
                    x0_hat = alg.score_solve(x_tau, v, sk, ek, tau, g2, eff,
                                             cg_tol, L)                       # l.11
                    xi_y = randn_like_cpu(y)                                  # l.12
                    xi_h = randn_like_cpu(x1)
                    b = inv_e2 * AT_fn(setup["y"]) + \
                        inv_s2 * apply_H_tau(x_tau, tau, sk, ek, eff) + \
                        (1.0 / eta) * AT_fn(xi_y) + \
                        (1.0 / float(sigma_t)) * apply_H_tau(xi_h, tau, sk, ek, eff)  # l.13
                    if anchor > 0:
                        x1_model = alg.direct_estimate_x1(
                            x_tau - tau * v, x_tau + (1.0 - tau) * v, sk, ek)
                        b = b + anchor * x1_model + \
                            math.sqrt(anchor) * randn_like_cpu(x1)
                    x1 = cg_solve(M_fn, b, x0=x1.clone(), tol=cg_tol,
                                  max_iter=200 if tau == 0.0 else L)          # l.14
                    x0 = (x_tau - apply_H_tau(x1, tau, sk, ek, eff)) / float(sigma_t)  # l.15
                    xi0 = randn_like_cpu(x0)                                  # l.16
                    x0 = x0 - (h0 / 2.0) * (x0 + x0_hat) + math.sqrt(h0) * xi0  # l.17
            mse = float(((x1 - pyr[k]) ** 2).mean())
            rows.append(dict(stage=k, step=step_idx, tau=tau,
                             sigma_tau=float(sigma_t), mse_x1=mse))
            if record:
                x_tau_rec = apply_H_tau(x1, tau, sk, ek, eff) + float(sigma_t) * x0
                traj.append((x_tau_rec[0].cpu(), x1[0].cpu(),
                             (x0_hat if x0_hat is not None else x0)[0].cpu()))
        if k < num_stages - 1:                            # line 20: U^(1) + fresh x0
            x1 = F.interpolate(x1, scale_factor=2, mode="nearest")
    return x1, rows, traj
