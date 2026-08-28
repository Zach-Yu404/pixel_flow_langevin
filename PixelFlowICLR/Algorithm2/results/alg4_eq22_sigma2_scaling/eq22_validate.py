#!/usr/bin/env python
"""Eq.(22) sigma^2-scaling validation battery.

Phase A: baseline-off 4-seed runs (must reproduce recorded 0.1339/0.1382).
Phase B: operator-level identities on the REAL frame operators grabbed from a
         live run (fp64 + fp32), solve comparison, magnitude probe.
Phase C: scaled-on 4-seed runs, per-seed vs baseline.
"""
import os, sys, json, math
import numpy as np, torch

A = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2"
sys.path.insert(0, A)
os.chdir(A)
import s_prior_methods as SP
import main4, utils

O = A + "/results/alg4_eq22_sigma2_scaling"
os.makedirs(O, exist_ok=True)
rep = open(os.path.join(O, "validation.md"), "w")


def emit(s):
    print(s, flush=True)
    rep.write(s + "\n")
    rep.flush()


SP._init_globals()
device = "cuda:0"
config, model, gamma2_tab, S = SP._load_sampling(device)
stats = json.load(open(SP._stats_path()))
K = int(config.scheduler.num_stages)
s2_fns = {arm: SP._build_s_ops(arm, stats, K)
          for arm in ("spectral", "pooled_junco")}
hole = S["hole"].to(device).float()
gt = S["gt"].to(device)


def four_seed(arm, over, label):
    S["kw"].pop("eq22_sigma2_scale", None)
    S["kw"].update(over)
    finals, holes = [], []
    ncbad, iters_max = 0, 0
    for seed in (42, 43, 44, 45):
        x1, rows, _ = main4._run_once(model, config, S, device,
                                      s2_fn=s2_fns[arm],
                                      gamma2_tab=gamma2_tab, seed=seed)
        finals.append(x1)
        holes.append(float((((x1 - gt) ** 2) * hole).sum() / (hole.sum() * 3)))
        ncbad = max(ncbad, sum(1 for r in rows if not r["blk1_cg_converged"]))
        iters_max = max(iters_max, max(int(r["blk1_cg_iters"]) for r in rows))
    stack = torch.stack(finals)
    spread = float((stack.std(dim=0, unbiased=True) * hole).sum() / hole.sum() / 3)
    emit(f"[{label}] {arm} holes=" + "/".join(f"{h:.4f}" for h in holes)
         + f" mean4={np.mean(holes):.4f}±{np.std(holes):.4f}"
           f" spread={spread:.4f} cg_bad={ncbad} cg_iters_max={iters_max}")
    return holes, finals


emit("# Eq.(22) sigma^2-scaling validation\n\n## Phase A: baseline off (records: spectral 0.1339±0.0047, pooled 0.1382±0.0054)")
base = {}
for arm in ("spectral", "pooled_junco"):
    base[arm] = four_seed(arm, {}, "A-baseline")

emit("\n## Phase B: operator identities on real frame operators")


class Grab:
    def __init__(self, wanted):
        self.wanted, self.store = set(wanted), {}

    def on_frame_setup(self, **kw):
        if kw["frame"] in self.wanted and kw["frame"] not in self.store:
            self.store[kw["frame"]] = kw

    def on_inner(self, **kw):
        pass

    def on_frame_end(self, **kw):
        pass


WANT = (9, 25, 31, 38)
for arm in ("pooled_junco", "spectral"):
    grab = Grab(WANT)
    S["kw"].pop("eq22_sigma2_scale", None)
    main4._run_once(model, config, S, device, s2_fn=s2_fns[arm],
                    gamma2_tab=gamma2_tab, seed=42, diag=grab)
    for f in sorted(grab.store):
        kw = grab.store[f]
        si, tau = kw["stage"], kw["tau"]
        s_k, e_k, sig = kw["s_k"], kw["e_k"], float(kw["sigma_tau"])
        eta, eff = kw["eta"], kw["eff_si"]
        Af, ATf, shape = kw["A_fn"], kw["AT_fn"], kw["shape"]
        s_ret = s2_fns[arm](si, sig)
        s_op = s_ret if isinstance(s_ret, utils.SOperator) \
            else utils.ScalarSOp(float(s_ret))
        M, Cinv, inv_e2, _, _ = utils.make_M_tau_den(
            Af, ATf, eta, sig, tau, s_k, e_k, eff, s_op)
        Mt, c, sc_e2 = utils.make_M_tau_den_sigma2(
            Af, ATf, eta, sig, tau, s_k, e_k, eff, s_op)
        g = torch.Generator(device="cpu").manual_seed(1234 + f)
        v = torch.randn(shape, generator=g).to(device)
        xh = torch.randn(shape, generator=g).to(device)
        xi_y0 = torch.randn(shape, generator=g).to(device)
        xi_y = Af(xi_y0)  # measurement-shaped via A range
        xi_h = torch.randn(shape, generator=g).to(device)
        xi_s = torch.randn(shape, generator=g).to(device)

        def rel(a, b):
            return float((a - b).norm() / b.norm().clamp(min=1e-30))

        rM32 = rel(Mt(v), c * M(v))
        q = (inv_e2 * ATf(S["y"].to(device)) + Cinv(xh)
             + (1.0 / eta) * ATf(xi_y)
             + (1.0 / sig) * utils.apply_H_tau(xi_h, tau, s_k, e_k, eff)
             + s_op.apply_S_inv_sqrt(xi_s)) if shape == tuple(S["y"].shape) or True else None
        qt = (sc_e2 * ATf(S["y"].to(device))
              + utils.apply_H_tau(utils.apply_H_tau(xh, tau, s_k, e_k, eff),
                                  tau, s_k, e_k, eff)
              + c * s_op.apply_S_inv(xh)
              + (c / eta) * ATf(xi_y)
              + sig * utils.apply_H_tau(xi_h, tau, s_k, e_k, eff)
              + c * s_op.apply_S_inv_sqrt(xi_s))
        rq32 = rel(qt, c * q)
        # fp64 identity (scalar S only; spectral buffers are fp32)
        r64 = ""
        if isinstance(s_op, utils.ScalarSOp):
            vd = v.double()
            r64 = f" relM64={rel(Mt(vd), c * M(vd)):.2e}"
        # solve comparison, same q
        Minv_b = utils.make_jacobi_precond(M, shape, device)
        Minv_s = utils.make_jacobi_precond(Mt, shape, device, floor=c * 1e-12)
        xb, itb, rb = utils.pcg_solve(M, q, Minv_b, tol=1e-6, max_iter=600)
        xs, its, rs = utils.pcg_solve(Mt, qt, Minv_s, tol=1e-6, max_iter=600,
                                      clamp_floor=c * 1e-12)
        rx = rel(xs, xb)
        mag_b = float(M(v).abs().max())
        mag_s = float(Mt(v).abs().max())
        emit(f"[B] {arm} f{f} st{si} tau={tau:.3f} sig={sig:.3e}: "
             f"relM32={rM32:.2e}{r64} relRHS32={rq32:.2e} "
             f"relX={rx:.2e} it_b/s={itb}/{its} res_b/s={rb:.1e}/{rs:.1e} "
             f"max|Mv|={mag_b:.2e} max|Mtv|={mag_s:.2e}")

emit("\n## Phase C: eq22_sigma2_scale=True, same seeds")
sc = {}
for arm in ("spectral", "pooled_junco"):
    sc[arm] = four_seed(arm, {"eq22_sigma2_scale": True}, "C-scaled")

emit("\n## Per-seed deltas (scaled - baseline)")
for arm in ("spectral", "pooled_junco"):
    hb, fb = base[arm]
    hs, fs = sc[arm]
    dmax = max(float((a - b).abs().max()) for a, b in zip(fs, fb))
    emit(f"[D] {arm} dhole=" + "/".join(f"{s - b:+.4f}" for s, b in zip(hs, hb))
         + f" max|dx|={dmax:.3e}")

emit("\nEQ22 VALIDATE DONE")
rep.close()
