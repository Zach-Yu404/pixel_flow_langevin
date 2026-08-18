#!/usr/bin/env python
"""Component verification for Algorithm 2 (user-requested audit, 2026-08-18).

Answers, with dense-matrix evidence (float64, 16x16 grid unless noted):
  V1  Is G (and hence H_tau, B, N — polynomials in G) self-adjoint?
      => is the nested apply_H_tau(apply_H_tau(x)) EXACTLY (H^T H) x,
         and score_solve's N(N(x)) EXACTLY (N^T N) x?  Does transpose matter?
  V2  compute_sigma_tau vs the paper formula (l.5).
  V3  make_velocity_fn CFG semantics (stub model): u + scale*(c-u),
      scale = class_guidance_scale(base, stage).
  V4  Adjoint tests for Ak/ATk: default make_Ak_fns (mask x bilinear-up with
      exact interpolate-adjoint) + box/random mask operators.
  V5  M0 closure vs dense (1/eta^2) A^T A + (1/sigma^2) H^T H (explicit
      transposes) — validates the l.7 implementation.
  V6  power_iter_norm vs numpy's exact largest eigenvalue of dense M.
  V7  score_solve: current N∘N form vs explicit-adjoint N^T N normal equation
      (dense solve) — quantifies the difference; S1 identity via both.
  V8  gamma2 matters: v = d_exact + gamma_true*noise; solver gamma2 swept —
      shows x0_hat error is minimized near gamma2 ~ gamma_true^2 (i.e. the
      gamma2 value is a real hyperparameter that must be tested).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch

import algorithm2 as alg  # noqa: E402  (pulls IP_package onto sys.path, chdirs)
from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_G, apply_H_tau, compute_sigma_tau, make_Ak_fns, make_velocity_fn,
)
from ms_posterior_sampling_utils import class_guidance_scale  # noqa: E402

torch.set_default_dtype(torch.float64)
RES = 16                      # dense grid (16x16 -> 256x256 matrices)
N_PIX = RES * RES
STAGES = [(0, 0.0, 0.25), (1, 0.142857, 0.5), (2, 0.333333, 0.75), (3, 0.6, 1.0)]


def dense(op, n=N_PIX, shape=(1, 1, RES, RES)):
    """Materialize a linear operator (acting on [1,1,R,R]) as an (n, n) matrix."""
    M = np.zeros((n, n))
    for j in range(n):
        e = torch.zeros(shape)
        e.view(-1)[j] = 1.0
        M[:, j] = op(e).reshape(-1).numpy()
    return M


def report(name, val, tol, unit="max|Δ|"):
    ok = val < tol
    print(f"  [{name:<46}] {unit} = {val:.3e}  {'PASS' if ok else '** FAIL **'} (tol {tol:g})")
    return ok


def main():
    all_ok = True
    print("== V1: self-adjointness of G / H_tau / B / N (dense, float64) ==")
    for si, s_k, e_k in STAGES:
        eff = si                      # g_bypass_stage3 convention
        G = dense(lambda x: apply_G(x, stage_idx=eff))
        all_ok &= report(f"G symmetry (stage {si})", np.abs(G - G.T).max(), 1e-12)
    si, s_k, e_k, tau = 1, 0.142857, 0.5, 0.4     # representative non-bypass stage
    H = dense(lambda x: apply_H_tau(x, tau, s_k, e_k, 1))
    B = dense(lambda x: alg.apply_B(x, s_k, e_k, 1))
    Nm = dense(lambda x: alg.apply_N(x, s_k, e_k, 1))
    all_ok &= report("H_tau symmetry", np.abs(H - H.T).max(), 1e-12)
    all_ok &= report("B symmetry", np.abs(B - B.T).max(), 1e-12)
    all_ok &= report("N symmetry", np.abs(Nm - Nm.T).max(), 1e-12)
    HH = dense(lambda x: apply_H_tau(apply_H_tau(x, tau, s_k, e_k, 1), tau, s_k, e_k, 1))
    NN = dense(lambda x: alg.apply_N(alg.apply_N(x, s_k, e_k, 1), s_k, e_k, 1))
    all_ok &= report("nested H∘H  vs  H^T H", np.abs(HH - H.T @ H).max(), 1e-12)
    all_ok &= report("nested N∘N  vs  N^T N", np.abs(NN - Nm.T @ Nm).max(), 1e-12)
    print("  => G is self-adjoint (bilinear-down-by-2 == 2x2 avg-pool, and its")
    print("     adjoint == nearest-up/4); polynomials in G inherit symmetry, so")
    print("     X∘X == X^T X exactly — transpose form is equivalent HERE, and the")
    print("     nested implementation is correct. (Any future G that is NOT")
    print("     self-adjoint WOULD break this — V1 is the guard.)")

    print("== V2: compute_sigma_tau vs paper l.5 ==")
    err = max(abs(compute_sigma_tau(t, s, e) - ((1 - t) * (1 - s) + t * (1 - e)))
              for _, s, e in STAGES for t in np.linspace(0, 1, 11))
    all_ok &= report("sigma_tau formula", err, 1e-15)

    print("== V3: make_velocity_fn CFG semantics (stub model) ==")

    class Stub(torch.nn.Module):
        def forward(self, x, timestep=None, class_labels=None,
                    latent_size=None, pos_embed=None):
            half = x.shape[0] // 2 if x.shape[0] > 1 else x.shape[0]
            out = x.clone()
            out[:half] = 1.0          # "uncond" half
            out[half:] = 3.0          # "cond" half
            return out

    for si in range(4):
        base_scale = 2.0
        vfn = make_velocity_fn(Stub(), torch.tensor(500.0), None, None, None,
                               True, base_scale, si)
        got = float(vfn(torch.zeros(1, 1, 4, 4))[0, 0, 0, 0])
        want = 1.0 + class_guidance_scale(base_scale, si) * (3.0 - 1.0)
        all_ok &= report(f"CFG stage {si}: u + scale*(c-u)", abs(got - want), 1e-12)

    print("== V4: adjoint tests Ak/ATk ==")
    g = torch.Generator().manual_seed(0)

    class MaskOp:                     # box/random inpainting stand-in (mask mult)
        def __init__(self, mask):
            self.mask = mask

        def get_mask(self, x=None):
            return self.mask

    mask = (torch.rand(1, 1, RES, RES, generator=g) > 0.3).double()
    y_probe = torch.randn(1, 1, RES, RES, generator=g)
    for stage_res in (RES // 4, RES // 2, RES):
        Ak, ATk = make_Ak_fns(MaskOp(mask), y_probe, (1, 1, stage_res, stage_res), "cpu")
        u = torch.randn(1, 1, stage_res, stage_res, generator=g)
        w = torch.randn(1, 1, RES, RES, generator=g)
        gap = abs(float((Ak(u) * w).sum()) - float((u * ATk(w)).sum()))
        all_ok &= report(f"make_Ak_fns adjoint (stage res {stage_res})", gap, 1e-10)

    print("== V5: M0 closure vs dense (1/eta^2)A^T A + (1/s^2)H^T H ==")
    eta, sigma = 0.05, 0.6
    Ak, ATk = make_Ak_fns(MaskOp(mask), y_probe, (1, 1, RES, RES), "cpu")

    def M0(x):
        return (1 / eta ** 2) * ATk(Ak(x)) + (1 / sigma ** 2) * apply_H_tau(
            apply_H_tau(x, tau, s_k, e_k, 1), tau, s_k, e_k, 1)
    Mdense = dense(M0)
    A = dense(Ak)
    Mref = (1 / eta ** 2) * A.T @ A + (1 / sigma ** 2) * H.T @ H
    all_ok &= report("M0 vs explicit-transpose reference",
                     np.abs(Mdense - Mref).max() / np.abs(Mref).max(), 1e-12, "rel")

    print("== V6: power_iter_norm vs exact lambda_max ==")
    lam_true = float(np.linalg.eigvalsh(Mdense).max())
    lam_est = alg.power_iter_norm(M0, (1, 1, RES, RES), "cpu", iters=20)
    rel = abs(lam_est - lam_true) / lam_true
    print(f"  lambda_max exact {lam_true:.4f}  power-iter(20) {lam_est:.4f}  rel err {rel:.2e}")
    all_ok &= report("power_iter_norm(20) within 5%", rel, 5e-2, "rel")
    print("     (ridge = 1e-6*estimate — a few % bias is immaterial)")

    print("== V7: score_solve — N∘N form vs explicit N^T N dense solve; S1 ==")
    x1_true = torch.randn(1, 1, RES, RES, generator=g)
    x0_true = torch.randn(1, 1, RES, RES, generator=g)
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)   # NOT V5's test constant
    x_tau = apply_H_tau(x1_true, tau, s_k, e_k, 1) + sigma_tau * x0_true
    d_exact = alg.apply_B(x1_true, s_k, e_k, 1) - (e_k - s_k) * x0_true
    for g2 in (0.0, 0.02):
        x0_cg = alg.score_solve(x_tau, d_exact, s_k, e_k, tau, g2, 1, 1e-12, 4000)
        rhs = (Nm.T @ (B @ x_tau.reshape(-1).numpy()
                       - H @ d_exact.reshape(-1).numpy()))
        lhs = Nm.T @ Nm + g2 * (H.T @ H)
        x0_dense = np.linalg.solve(lhs, rhs)
        gap = np.abs(x0_cg.reshape(-1).numpy() - x0_dense).max()
        all_ok &= report(f"CG(N∘N) vs dense(N^T N), gamma2={g2}", gap, 1e-6)
    x0_s1 = alg.score_solve(x_tau, d_exact, s_k, e_k, tau, 0.0, 1, 1e-12, 4000)
    all_ok &= report("S1 identity: x0_hat == x0 (v=d_exact, g2=0)",
                     float((x0_s1 - x0_true).abs().max()), 1e-6)
    print("  => with self-adjoint N/H the un-transposed implementation solves the")
    print("     SAME normal equation as the N^T N form — no difference beyond CG tol.")

    print("== V8: gamma2 is a real knob (v noisy => best gamma2 > 0) ==")
    for gamma_true in (0.05, 0.15):
        noise = torch.randn(1, 1, RES, RES, generator=g)
        v_noisy = d_exact + gamma_true * noise
        errs = {}
        for g2 in (0.0, 1e-4, 1e-3, 1e-2, gamma_true ** 2, 4 * gamma_true ** 2, 1.0):
            x0_hat = alg.score_solve(x_tau, v_noisy, s_k, e_k, tau, g2, 1, 1e-12, 4000)
            errs[g2] = float(((x0_hat - x0_true) ** 2).mean())
        best = min(errs, key=errs.get)
        line = "  ".join(f"g2={k:.1e}:{v:.4f}" for k, v in errs.items())
        print(f"  gamma_true={gamma_true}: {line}")
        print(f"    best gamma2 = {best:.1e} (gamma_true^2 = {gamma_true**2:.1e}) "
              f"-> zero-gamma2 is {errs[0.0]/errs[best]:.2f}x worse")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "** SOME CHECKS FAILED **"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
