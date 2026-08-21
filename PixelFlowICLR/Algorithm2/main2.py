#!/usr/bin/env python
"""Algorithm 3 (modified draft p.15) — sampler and an A/B against Algorithm 2.

Algorithm 3 keeps Algorithm 2's operators, its score solve and its exact draw
untouched. It changes the order of the two blocks and inserts two lines:

    Alg 2                              Alg 3
    ------------------------------     ------------------------------
    9  x_tau <- H x1 + sigma x0        9  x_tau <- H x1 + sigma x0
    10 v     <- v_theta(x_tau)         10 v     <- v_theta(x_tau)
    11 score solve -> x0_hat           11 score solve -> x0_hat        (identical)
    -- Block 1: exact draw of x1 --    -- Block 1: Langevin on x0 --
    12 xi_y, xi_h                      12 xi_0
    13 b_tilde                    (34) 13 x0 <- x0 - h0/2 (x0+x0_hat) + sqrt(h0) xi_0
    14 solve M x1 = b_tilde            14 x1_hat := H^-1 (x_tau - sigma x0_hat)   (51)
    15 x0 <- (x_tau - H x1)/sigma      15 x_tau <- H x1_hat + sigma x0
    -- Block 2: Langevin on x0 --      -- Block 2: exact draw of x1 --
    16 xi_0                            16 xi_y, xi_h
    17 x0 <- x0 - h0/2 (x0+x0_hat)     17 b_tilde                            (34)
          + sqrt(h0) xi_0        (36)  18 solve M x1 = b_tilde
                                       19 x0 <- (x_tau - H x1)/sigma

So two things change. The exact draw now runs last, and — the substantive part —
line 15 rebuilds the state from x1_hat, the MMSE denoising of the interpolant,
instead of leaving it built from the previously drawn x1. The prior therefore
enters x_tau directly rather than only through x0.

One consequence is structural: line 14 needs (H_tau)^-1, and H_0 = s_k G is
singular wherever ker(G) is non-trivial, so Algorithm 3 cannot evaluate tau=0
outside stage 3 (where g_bypass makes G = I). The grid starts at the first
tau > 0 there, which is what 7.13 recommends anyway.

    PYTHONHASHSEED=0 python main2.py --task box_inpainting --image junco
    PYTHONHASHSEED=0 python main2.py --compare          # Alg 2 vs Alg 3, same seed
"""

import os

ORIG_CWD = os.getcwd()

import argparse   # noqa: E402
import copy       # noqa: E402
import csv        # noqa: E402
import json       # noqa: E402
import math       # noqa: E402
import sys        # noqa: E402
import time       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import main as alg2_main                                       # noqa: E402
import utils                                                   # noqa: E402
import measurement                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402
from omegaconf import OmegaConf                                # noqa: E402
from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_H_tau, compute_sigma_tau, make_velocity_fn, make_Ak_fns,
)
from ms_posterior_sampling_utils import class_guidance_scale   # noqa: E402
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402


def run_posterior_sampling_alg3(
    model, config, gt, y, operator, eta, device,
    *,
    num_langevin=10, ode_steps_per_stage=10, shift=1.0,
    guidance_scale=0.0, class_label=10, g_bypass_stage3=True,
    cg_tol=1e-5, cg_max_iter=50, make_Ak_fns_fn=None, seed=42,
    record_trajectory=False, gamma2_tab=None, h0=utils.H0,
    sigma_min=utils.SIGMA_MIN, ridge_rel=1e-6, cg_max_iter_l14=200,
    terminal_replace_weight=0.0, check_52=True, rebuild_state=True,
    recompute_x0=True, **unused_kw,
):
    """Algorithm 3 (modified draft p.15).

    Line numbers in the comments are Algorithm 3's own. Everything the two
    algorithms share is called through the same helpers Algorithm 2 uses, so a
    difference in the results can only come from the reordering and from
    lines 14-15.

    ``check_52`` asserts the draft's free consistency identity (52),
    H_tau (x1_hat - x1) = sigma_tau (x0 - x0_hat), which holds by construction
    and catches a sign or scaling slip in (51) immediately.
    """
    if make_Ak_fns_fn is None:
        make_Ak_fns_fn = make_Ak_fns
    B = gt.shape[0]
    S = int(float(num_langevin))
    L = int(cg_max_iter)
    num_stages = int(config.scheduler.num_stages)
    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=num_stages, gamma=-1 / 3)

    pe_labels = torch.tensor([int(class_label)] * B, dtype=torch.int32, device=device)
    do_cfg = guidance_scale > 0
    if do_cfg:
        prompt_embeds = torch.cat([int(model.num_classes) * torch.ones_like(pe_labels),
                                   pe_labels], dim=0)
    else:
        prompt_embeds = pe_labels

    target_h, target_w = int(gt.shape[-2]), int(gt.shape[-1])
    init = 2 ** (num_stages - 1)
    h, w = target_h // init, target_w // init

    g = torch.Generator(device="cpu").manual_seed(int(seed))

    def randn_like_cpu(x):
        return torch.randn(x.shape, generator=g).to(x.device)

    pyr = utils.base.gt_stage_pyramid(gt, num_stages)
    x1 = torch.zeros((B, 3, h, w), device=device)                    # l.1
    rows, traj = [], []
    skipped_tau0 = 0

    for si in range(num_stages):                                     # l.2
        sc = copy.deepcopy(scheduler)
        steps = int(float(utils._per_stage(ode_steps_per_stage, si, num_stages)))
        sc.set_timesteps(steps, si, device=device, shift=shift)
        s_k, e_k = float(sc.start_t[si]), float(sc.end_t[si])
        eff_si = si if g_bypass_stage3 else None

        if si > 0:                                                   # l.22: U^(1)
            h *= 2
            w *= 2
            x1 = F.interpolate(x1, size=(h, w), mode="nearest")
        x0 = randn_like_cpu(pyr[si])                                 # l.3

        Ak, ATk = make_Ak_fns_fn(operator, y, (B, 3, h, w), device)
        size_tensor, rope_pos = utils.base.rope_for(model, h, w, device)
        gamma2_stage = gamma2_tab[str(si)]

        for step_idx, T in enumerate(sc.Timesteps):                  # l.4
            tau = float(sc.t[step_idx])
            sigma_tau = compute_sigma_tau(tau, s_k, e_k)             # l.5
            if sigma_tau < sigma_min:
                continue
            # l.14 needs H_tau^-1; H_0 = s_k G is singular unless G = I.
            if tau == 0.0 and eff_si != 3:
                skipped_tau0 += 1
                continue

            gamma2 = float(gamma2_stage.get(
                f"{round(tau, 6)}", list(gamma2_stage.values())[step_idx]))
            M_tau, inv_e2, inv_s2, epsilon = utils.make_M_tau(       # l.7
                Ak, ATk, eta, sigma_tau, tau, s_k, e_k, eff_si,
                x1.shape, device, ridge_rel)
            x0_hat = None

            for s in range(S):                                       # l.8
                x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + sigma_tau * x0  # l.9
                x1_at9, x0_at9 = x1, x0        # (52) compares against these
                with torch.no_grad():
                    v = model_velocity(model, T, prompt_embeds, size_tensor,
                                       rope_pos, do_cfg, guidance_scale, si)(x_tau)
                x0_hat = utils.score_solve(x_tau, v, s_k, e_k, tau, gamma2,
                                           eff_si, cg_tol, L)        # l.11

                # ── Block 1: Langevin step for x0 ──────────────────────────
                xi_0 = randn_like_cpu(x0)                            # l.12
                x0 = x0 - (h0 / 2.0) * (x0 + x0_hat) + math.sqrt(h0) * xi_0  # l.13

                # ── (51): the MMSE denoising of the interpolant ────────────
                x1_hat = utils.apply_H_tau_inv(x_tau - sigma_tau * x0_hat,
                                               tau, s_k, e_k, eff_si)  # l.14
                if rebuild_state:                                    # l.15
                    x_tau = apply_H_tau(x1_hat, tau, s_k, e_k, eff_si) + sigma_tau * x0

                # ── Block 2: exact draw of x1 ──────────────────────────────
                xi_y = randn_like_cpu(y)                             # l.16
                xi_h = randn_like_cpu(x1)
                b_tilde = utils.data_rhs(ATk, y, x_tau, inv_e2, inv_s2,
                                         tau, s_k, e_k, eff_si) + \
                    (1.0 / eta) * ATk(xi_y) + \
                    (1.0 / float(sigma_tau)) * apply_H_tau(xi_h, tau, s_k, e_k,
                                                           eff_si)    # l.17
                if epsilon:
                    b_tilde = b_tilde + math.sqrt(epsilon) * randn_like_cpu(x1)
                x1 = utils.cg_solve(M_tau, b_tilde, x0=x1.clone(), tol=cg_tol,
                                    max_iter=cg_max_iter_l14 if tau == 0.0 else L)  # l.18
                if recompute_x0:                                     # l.19
                    # The amplifier: x_tau was rebuilt from x1_hat at l.15, the
                    # draw returns x1 != x1_hat, and that difference reaches x0
                    # divided by sigma_tau. Keeping the Langevin x0 instead
                    # leaves it bounded (0.69-0.80 against 1051 with l.19 on).
                    x0 = (x_tau - apply_H_tau(x1, tau, s_k, e_k, eff_si)) / float(sigma_tau)
                if check_52 and s == 0:
                    # (52): H(x1_hat - x1) == sigma(x0 - x0_hat), against the x1
                    # and x0 that built x_tau at l.9 — not the updated ones.
                    lhs = apply_H_tau(x1_hat - x1_at9, tau, s_k, e_k, eff_si)
                    rhs = float(sigma_tau) * (x0_at9 - x0_hat)
                    scale = max(float(rhs.abs().max()), 1e-12)
                    resid = float((lhs - rhs).abs().max()) / scale
                    # relative: (52) is exact, so only float32 round-off remains
                    if resid > 1e-4:
                        raise AssertionError(
                            f"(52) violated at stage {si} tau={tau}: "
                            f"relative residual {resid:.3e}")

            rows.append(dict(stage=si, step=step_idx, tau=tau,
                             sigma_tau=float(sigma_tau),
                             mse_x1=float(((x1 - pyr[si]) ** 2).mean())))
            if record_trajectory:
                x_tau_rec = apply_H_tau(x1, tau, s_k, e_k, eff_si) + float(sigma_tau) * x0
                traj.append((x_tau_rec[0].cpu(), x1[0].cpu(),
                             (x0_hat if x0_hat is not None else x0)[0].cpu()))

    if skipped_tau0:
        print(f"[alg3] skipped {skipped_tau0} tau=0 grid points "
              f"(H_0 = s_k G singular; 7.13 recommends starting at tau > 0)",
              flush=True)

    if terminal_replace_weight > 0:
        m = operator.get_mask(x=y).float().to(x1.device)
        x1 = terminal_replace_weight * (m * y + (1.0 - m) * x1) + \
            (1.0 - terminal_replace_weight) * x1
    return x1, rows, traj


def model_velocity(model, T, prompt_embeds, size_tensor, rope_pos, do_cfg,
                   guidance_scale, si):
    return make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos,
                            do_cfg, class_guidance_scale(guidance_scale, si), si)


# ═════════════════════════════ harness ═══════════════════════════════════════
def build(cfg, task, image, device):
    paths = {k: alg2_main.resolve_path(v.replace("{IP_PACKAGE}", utils.base.IP_PACKAGE)
                                       .replace("{HERE}", HERE))
             for k, v in cfg["paths"].items()}
    alg2_main.PATHS = paths
    measurement.configure(cfg["tasks_setup"], cfg["algorithm"]["measurement_seed"])
    demos = alg2_main.load_demo_images(resolution=256, demo_dir=paths["demo_dir"])
    demo = {d["short_name"]: d for d in demos}[image]
    gt = demo["gt"].unsqueeze(0).to(device)
    spec = cfg["tasks_setup"][task]
    op, mask, y, _, _, mkA, _ = measurement.build_setup_and_measurement(
        task, spec["operator"], demo, float(spec["sigma_n"]), 256, device)
    model_cfg = OmegaConf.load(os.path.join(paths["model_dir"], "config.yaml"))
    model = alg2_main._load_model(model_cfg, device)
    gamma2_tab = json.load(open(paths["gamma2_table"]))["table"]
    return demo, gt, op, mask, y, mkA, model, model_cfg, gamma2_tab, spec


def metrics(x1, gt, mask):
    hole = (1.0 - mask).to(x1.device)
    obs = 1.0 - hole
    return (utils.mse_masked(x1, gt, hole), utils.mse_masked(x1, gt, obs),
            float(((x1 - gt) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--task", default="box_inpainting")
    ap.add_argument("--image", default="junco")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--num-langevin", type=int, default=None,
                    help="override S (inner iterations); the l.15/l.19 feedback compounds over it")
    ap.add_argument("--no-l15", action="store_true",
                    help="ablation: skip line 15 (do not rebuild x_tau from x1_hat)")
    ap.add_argument("--compare", action="store_true",
                    help="run Algorithm 2 on the same setup and seed")
    ap.add_argument("--out", default="results/alg3")
    cli = ap.parse_args()

    cfg_path = cli.config or os.path.join(HERE, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    alg2_main._check_config_keys(cfg)
    alg2_main._check_hash_seed("alg3")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    (demo, gt, op, mask, y, mkA, model, model_cfg,
     gamma2_tab, spec) = build(cfg, cli.task, cli.image, device)
    print("[setup] model loaded", flush=True)

    kw = {**cfg["sampler_kw"], **spec.get("kw", {})}
    if cli.num_langevin is not None:
        kw["num_langevin"] = cli.num_langevin
    alg = cfg["algorithm"]
    common = dict(gamma2_tab=gamma2_tab, make_Ak_fns_fn=mkA,
                  h0=alg["h0"], sigma_min=alg["sigma_min"],
                  ridge_rel=alg["ridge_rel"],
                  cg_max_iter_l14=alg["cg_max_iter_l14"],
                  terminal_replace_weight=float(spec["terminal_replace_weight"]),
                  class_label=int(demo["class_idx"]),
                  **{k: v for k, v in kw.items() if k not in ("class_label", "seed")})

    out = os.path.join(HERE, cli.out)
    os.makedirs(out, exist_ok=True)
    results = []
    for seed in [int(s) for s in cli.seeds.split(",")]:
        t0 = time.time()
        x1_3, rows3, _ = run_posterior_sampling_alg3(
            model, model_cfg, gt, y, op, float(spec["sigma_n"]), device,
            seed=seed, rebuild_state=not cli.no_l15, **common)
        h3, o3, f3 = metrics(x1_3, gt, mask)
        rec = dict(seed=seed, alg="alg3-no-l15" if cli.no_l15 else "alg3", hole=h3, obs=o3, full=f3,
                   secs=round(time.time() - t0, 1))
        print(f"[{rec['alg']} seed={seed}] hole={h3:.4f} obs={o3:.4f} full={f3:.4f} "
              f"({rec['secs']}s)", flush=True)
        results.append(rec)

        if cli.compare:
            t0 = time.time()
            x1_2, rows2, _ = utils.run_posterior_sampling_alg2(
                model, model_cfg, gt, y, op, float(spec["sigma_n"]), device,
                seed=seed, **common)
            h2, o2, f2 = metrics(x1_2, gt, mask)
            rec2 = dict(seed=seed, alg="alg2", hole=h2, obs=o2, full=f2,
                        secs=round(time.time() - t0, 1))
            print(f"[alg2 seed={seed}] hole={h2:.4f} obs={o2:.4f} full={f2:.4f} "
                  f"({rec2['secs']}s)   -> alg3 vs alg2 hole "
                  f"{(h3 - h2) / h2 * 100:+.1f}%", flush=True)
            results.append(rec2)

    path = os.path.join(out, f"alg3_{cli.task}_{cli.image}.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader(); wr.writerows(results)
    print(f"[done] -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
