#!/usr/bin/env python
"""WV Algorithm 2: the (x0, x1)-coupling / velocity-uncertainty precision term.

x_tau = H_tau x1 + sigma_tau x0 and v_theta = B_k x1 - (e_k - s_k) x0 + eps_v
with eps_v ~ N(0, gamma^2 I). Eliminating x0 leaves a SECOND Gaussian
observation of x1 that costs no extra network call:

    r_tau := (e_k - s_k) x_tau + sigma_tau v_theta = N_k x1 + sigma_tau eps_v

so Block 1 gains precision N^T N / (sigma_tau^2 gamma^2) and the matching RHS
term. gamma^2 is the measured Cor.-8 table, so no hand-tuned weight enters.

Three arms on one image / one seed / the live config, plus an operator-level
check of the identity the whole derivation rests on:

    --check   V-WV1 r_tau == N_k x1 under the oracle displacement
              V-WV2 the exact draw's RHS noise covariance == M_tau^wv
    --run     baseline (utils.run_posterior_sampling_alg2, untouched),
              WV x1_hat_method="direct", WV x1_hat_method="inverse",
              WV with Lemma 5's noise dropped (block1_noise=False)
    --report  comparison.csv + trajectories.png + the panels

    PYTHONHASHSEED=0 python test_wv_alg2.py            # all three
    PYTHONHASHSEED=0 python test_wv_alg2.py --check    # operator checks only
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

import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
import matplotlib                                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from omegaconf import OmegaConf                                 # noqa: E402

import main as alg2_main                                        # noqa: E402
import measurement                                              # noqa: E402
import utils                                                    # noqa: E402
from utils import (apply_G, apply_B, apply_N, apply_H_tau, compute_sigma_tau,
                   make_M_tau, make_M_tau_wv, data_rhs,
                   cg_solve, mse_masked)                        # noqa: E402

OUT = os.path.join(HERE, "results_wv_alg2")
TASK, IMAGE = "box_inpainting", "junco"
# (s_k, e_k) per stage, as PixelFlowScheduler produces them for num_stages=4.
SCHED = [(0.0, 0.25), (0.142857, 0.5), (0.333333, 0.75), (0.6, 1.0)]


# ══════════════════════════ operator-level checks ═══════════════════════════
def check(res=32, seed=0):
    """V-WV1 / V-WV2: the coupling identity and the draw's noise covariance."""
    torch.manual_seed(seed)
    rows, ok = [], True
    print("== V-WV1: r_tau == N_k x1 under the oracle displacement ==")
    for k, (s_k, e_k) in enumerate(SCHED):
        for tau in (0.0, 0.444, 0.999):
            x1 = torch.randn(1, 3, res, res)
            x0 = torch.randn(1, 3, res, res)
            sig = compute_sigma_tau(tau, s_k, e_k)
            x_tau = apply_H_tau(x1, tau, s_k, e_k, k) + sig * x0
            d_tau = apply_B(x1, s_k, e_k, k) - (e_k - s_k) * x0     # oracle v
            r_tau = (e_k - s_k) * x_tau + sig * d_tau
            ref = apply_N(x1, s_k, e_k, k)
            err = float((r_tau - ref).abs().max())
            scale = float(ref.abs().max())
            rel = err / max(scale, 1e-12)
            ok &= rel < 1e-5
            rows.append(dict(check="V-WV1", stage=k, tau=tau, value=rel, tol=1e-5,
                             passed=rel < 1e-5))
            print(f"  [stage {k} tau={tau:.3f}] rel max|r_tau - N_k x1| = {rel:.3e} "
                  f"{'PASS' if rel < 1e-5 else 'FAIL'}")

    print("== V-WV2: RHS noise covariance == M_tau^wv (projection Monte-Carlo) ==")
    n_draw = 4000
    for k, (s_k, e_k) in enumerate(SCHED):
        tau, gamma2, eta = 0.444, 0.012, 0.05
        s_k_, e_k_ = s_k, e_k
        sig = compute_sigma_tau(tau, s_k_, e_k_)
        mask = (torch.rand(1, 1, res, res) > 0.5).float()           # stand-in A
        A = (lambda x: mask * x)
        AT = (lambda r: mask * r)
        y_probe = torch.zeros(1, 3, res, res)
        M_fn, ie2, is2, iv2, _ = make_M_tau_wv(A, AT, eta, sig, tau, s_k_, e_k_, k,
                                               (1, 3, res, res), "cpu", 0.0, gamma2)
        u = torch.randn(1, 3, res, res)
        u = u / u.norm()
        target = float((u * M_fn(u)).sum())                         # u^T M u
        acc = 0.0
        for _ in range(n_draw):
            xi_y = torch.randn_like(y_probe)
            xi_h = torch.randn_like(u)
            xi_v = torch.randn_like(u)
            z = ((1.0 / eta) * AT(xi_y)
                 + (1.0 / sig) * apply_H_tau(xi_h, tau, s_k_, e_k_, k)
                 + (1.0 / (sig * gamma2 ** 0.5)) * apply_N(xi_v, s_k_, e_k_, k))
            acc += float((u * z).sum()) ** 2
        emp = acc / n_draw
        rel = abs(emp - target) / max(abs(target), 1e-12)
        tol = 0.06                       # 4000 draws -> ~2.2% s.e. on a variance
        ok &= rel < tol
        rows.append(dict(check="V-WV2", stage=k, tau=tau, value=rel, tol=tol,
                         passed=rel < tol))
        print(f"  [stage {k}] Var<u,noise> {emp:.6g} vs u^T M u {target:.6g}  "
              f"rel {rel:.3e} {'PASS' if rel < tol else 'FAIL'}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "operator_checks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


# ═══════════════════════════════ the three arms ═════════════════════════════
def build(device):
    cfg = json.load(open(os.path.join(ALG2, "config.json")))
    paths = {k: alg2_main.resolve_path(v.replace("{IP_PACKAGE}", utils.base.IP_PACKAGE)
                                       .replace("{HERE}", ALG2))
             for k, v in cfg["paths"].items()}
    alg2_main.PATHS = paths
    measurement.configure(cfg["tasks_setup"], cfg["algorithm"]["measurement_seed"])
    demos = alg2_main.load_demo_images(resolution=256, demo_dir=paths["demo_dir"])
    demo = {d["short_name"]: d for d in demos}[IMAGE]
    gt = demo["gt"].unsqueeze(0).to(device)
    spec = cfg["tasks_setup"][TASK]
    op, mask, y, _, _, mkA, _ = measurement.build_setup_and_measurement(
        TASK, spec["operator"], demo, float(spec["sigma_n"]), 256, device)
    mcfg = OmegaConf.load(os.path.join(paths["model_dir"], "config.yaml"))
    model = alg2_main._load_model(mcfg, device)
    gamma2_tab = json.load(open(paths["gamma2_table"]))["table"]
    return cfg, demo, gt, op, mask, y, mkA, model, mcfg, gamma2_tab, spec


def run(arm, device="cuda:0"):
    cfg, demo, gt, op, mask, y, mkA, model, mcfg, gamma2_tab, spec = build(device)
    print(f"[{arm}] model loaded", flush=True)
    kw = {**cfg["sampler_kw"], **spec.get("kw", {})}
    alg = cfg["algorithm"]
    common = dict(gamma2_tab=gamma2_tab, make_Ak_fns_fn=mkA, h0=alg["h0"],
                  sigma_min=alg["sigma_min"], ridge_rel=alg["ridge_rel"],
                  cg_max_iter_l14=alg["cg_max_iter_l14"],
                  terminal_replace_weight=float(spec["terminal_replace_weight"]),
                  class_label=int(demo["class_idx"]), seed=alg["seed"],
                  record_trajectory=True,
                  **{k: v for k, v in kw.items() if k not in ("class_label", "seed")})
    eta = float(spec["sigma_n"])
    if arm == "baseline":
        x1, rows, traj = utils.run_posterior_sampling_alg2(
            model, mcfg, gt, y, op, eta, device, **common)
    elif arm == "wv_nonoise":
        # Lemma 5's noise dropped from b_tau: l.14 returns the conditional mean.
        x1, rows, traj = utils.run_posterior_wv_sampling_alg2(
            model, mcfg, gt, y, op, eta, device,
            x1_hat_method="direct", block1_noise=False, **common)
    else:
        x1, rows, traj = utils.run_posterior_wv_sampling_alg2(
            model, mcfg, gt, y, op, eta, device,
            x1_hat_method=arm.split("_", 1)[1], **common)

    hole = (1.0 - mask).to(device).float()
    ns = int(mcfg.scheduler.num_stages)
    pyr = utils.base.gt_stage_pyramid(gt, ns)
    out_rows = []
    for gi, (r, t) in enumerate(zip(rows, traj)):
        k = r["stage"]; tau = r["tau"]; sig = r["sigma_tau"]
        s_k, e_k = SCHED[k]
        xk = t[1].unsqueeze(0).to(device)
        m_k = F.interpolate(hole, size=xk.shape[-2:], mode="nearest")
        Ak, ATk = mkA(op, y, tuple(xk.shape), device)
        row = dict(arm=arm, gstep=gi, stage=k, step=r["step"], tau=tau, sigma_tau=sig,
                   mse_total=float(((xk - pyr[k]) ** 2).mean()),
                   mse_hole=mse_masked(xk, pyr[k], m_k),
                   mse_vis=mse_masked(xk, pyr[k], 1.0 - m_k),
                   meas_resid=float(((Ak(xk) - y) ** 2).mean()))
        if len(t) > 5:                       # WV arms record the extra channels
            x1_hat, mu, xt = t[3], t[4], t[5]
            row["x1hat_hole"] = ("" if x1_hat is None else
                                 mse_masked(x1_hat.unsqueeze(0).to(device), pyr[k], m_k))
            if mu is not None:
                mu = mu.unsqueeze(0).to(device)
                row["mu_wv_hole"] = mse_masked(mu, pyr[k], m_k)
                row["mu_wv_vis"] = mse_masked(mu, pyr[k], 1.0 - m_k)
            if xt is not None:
                # SAME-STATE control: the baseline Block-1 mean at the very state
                # mu_wv was solved at -- only M_tau and b_tau differ.
                xt = xt.unsqueeze(0).to(device)
                eff = k
                Mb, ie2, is2, _ = make_M_tau(Ak, ATk, eta, sig, tau, s_k, e_k, eff,
                                             xt.shape, device, alg["ridge_rel"])
                bb = data_rhs(ATk, y, xt, ie2, is2, tau, s_k, e_k, eff)
                mub = cg_solve(Mb, bb, x0=xk.clone(), tol=kw["cg_tol"],
                               max_iter=alg["cg_max_iter_l14"] if tau == 0.0
                               else kw["cg_max_iter"])
                row["mu_base_hole"] = mse_masked(mub, pyr[k], m_k)
                if x1_hat is not None:      # range(G)/ker(G) split of x1_hat error
                    d = x1_hat.unsqueeze(0).to(device) - pyr[k]
                    Gd = apply_G(d, stage_idx=eff)
                    row["x1hat_hole_range"] = float(((Gd ** 2) * m_k).sum()
                                                    / (m_k.sum() * 3))
                    row["x1hat_hole_ker"] = float((((d - Gd) ** 2) * m_k).sum()
                                                  / (m_k.sum() * 3))
        out_rows.append(row)
        if r["step"] == 9 or gi == len(rows) - 1:
            panel(arm, gi, k, r["step"], tau, gt, y, pyr[k], t, xk, m_k)

    fin = dict(arm=arm, hole=mse_masked(x1, gt, hole),
               vis=mse_masked(x1, gt, 1.0 - hole),
               full=float(((x1 - gt) ** 2).mean()),
               meas_resid=float(((op(x1) - y) ** 2).mean()))
    d = os.path.join(OUT, arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "trajectory.csv"), "w", newline="") as f:
        keys = sorted({kk for rr in out_rows for kk in rr})
        keys = ["arm", "gstep", "stage", "step", "tau"] + \
            [kk for kk in keys if kk not in ("arm", "gstep", "stage", "step", "tau")]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for rr in out_rows:
            w.writerow({kk: rr.get(kk, "") for kk in keys})
    json.dump(fin, open(os.path.join(d, "final.json"), "w"), indent=1)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for A, (t_, ttl) in zip(ax, [(gt, "GT"), (y, "measurement y"),
                                 (x1, f"{arm} final\nhole {fin['hole']:.4f} "
                                      f"vis {fin['vis']:.5f}")]):
        A.imshow(img(t_)); A.set_title(ttl, fontsize=9); A.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(d, "final.png"), dpi=140)
    plt.close(fig)
    print(f"[{arm}] FINAL {json.dumps(fin)}", flush=True)


def img(t):
    return ((t[0].detach().cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1)


def panel(arm, gi, k, step, tau, gt, y, gt_k, t, xk, m_k):
    """GT | measurement | x1_hat | Block-1 mean | sampled x1 | error map."""
    items = [(gt_k, "GT (stage)"), (y, "measurement y")]
    if len(t) > 5:
        items.append((t[3].unsqueeze(0) if t[3] is not None else None, "x1_hat"))
        items.append((t[4].unsqueeze(0) if t[4] is not None else None, "Block-1 mean"))
    items.append((xk, "sampled x1"))
    fig, ax = plt.subplots(1, len(items) + 1, figsize=(3.1 * (len(items) + 1), 3.4))
    for A, (t_, ttl) in zip(ax, items):
        if t_ is None:
            A.text(.5, .5, "undefined\n(H_tau singular)", ha="center", va="center",
                   fontsize=9)
        else:
            A.imshow(img(t_.to(xk.device) if hasattr(t_, "to") else t_))
        A.set_title(ttl, fontsize=9); A.axis("off")
    err = (((xk - gt_k) ** 2).mean(1, keepdim=True) * m_k)[0, 0].detach().cpu()
    im = ax[-1].imshow(err, cmap="magma"); ax[-1].set_title("hole sq-err", fontsize=9)
    ax[-1].axis("off"); fig.colorbar(im, ax=ax[-1], fraction=0.046)
    fig.suptitle(f"{arm} — stage {k} step {step} tau={tau:.4f}", fontsize=11)
    fig.tight_layout()
    d = os.path.join(OUT, arm); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"stage{k}_step{step}.png"), dpi=130)
    plt.close(fig)


# ══════════════════════════════════ report ══════════════════════════════════
def report():
    arms = ["baseline", "wv_direct", "wv_inverse", "wv_nonoise"]
    fins, allrows = [], []
    for a in arms:
        p = os.path.join(OUT, a, "final.json")
        if os.path.exists(p):
            fins.append(json.load(open(p)))
            allrows += list(csv.DictReader(open(os.path.join(OUT, a, "trajectory.csv"))))
    with open(os.path.join(OUT, "comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "hole", "vis", "full", "meas_resid"])
        w.writeheader(); w.writerows(fins)
    print(f"{'arm':12s} {'hole':>12s} {'vis':>12s} {'full':>12s} {'meas_resid':>12s}")
    for d in fins:
        print(f"{d['arm']:12s} {d['hole']:12.6f} {d['vis']:12.7f} "
              f"{d['full']:12.6f} {d['meas_resid']:12.7f}")

    def g(r, k):
        v = r.get(k, "")
        return float(v) if v not in ("", None) else float("nan")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a in arms:
        rs = [r for r in allrows if r["arm"] == a]
        if not rs:
            continue
        gs = [int(r["gstep"]) for r in rs]
        ax[0].semilogy(gs, [g(r, "mse_hole") for r in rs], marker=".", ms=3, label=a)
        ax[1].semilogy(gs, [g(r, "meas_resid") for r in rs], marker=".", ms=3, label=a)
        if a != "baseline":
            ax[2].semilogy(gs, [g(r, "mu_wv_hole") for r in rs], marker=".", ms=3,
                           label=f"{a}: mu_wv")
            ax[2].semilogy(gs, [g(r, "mu_base_hole") for r in rs], ls="--", lw=1,
                           label=f"{a}: mu_base (same state)")
            ax[2].semilogy(gs, [g(r, "x1hat_hole") for r in rs], ls=":", lw=1,
                           label=f"{a}: x1_hat")
    for A, ttl in zip(ax, ["sampled x1 — HOLE MSE", "measurement residual",
                           "Block-1 mean, HOLE (same state)"]):
        for b in (10, 20, 30):
            A.axvline(b, color="gray", lw=.6, alpha=.5)
        A.set_title(ttl, fontsize=10); A.grid(alpha=.3)
        A.legend(fontsize=7); A.set_xlabel("global step")
    fig.suptitle("WV Algorithm 2 — box_inpainting / junco / seed 42")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trajectories.png"), dpi=140)
    print(f"[report] -> {OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", default=None,
                    help="comma-separated arms: baseline,wv_direct,wv_inverse,wv_nonoise")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not (a.check or a.run or a.report):
        a.check = a.report = True
        a.run = "baseline,wv_direct,wv_inverse,wv_nonoise"
    os.makedirs(OUT, exist_ok=True)
    rc = 0
    if a.check:
        rc = check()
    if a.run:
        for arm in a.run.split(","):
            run(arm)
    if a.report:
        report()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
