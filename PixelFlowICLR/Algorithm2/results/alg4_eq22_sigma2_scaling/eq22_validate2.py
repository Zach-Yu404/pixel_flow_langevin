#!/usr/bin/env python
"""Eq.(22) sigma^2-scaling: round-2 validation covering codex code-review gaps.

[F]  fallback scaling-law unit tests (codex 1D counterexample + 64-dim SPD)
[A2] post-fix 4-seed regression, baseline + scaled, both arms
[G]  frame grabs {9,15,25,31,38,39} identities incl. spectral fp64
[S8] sigma=1e-8 stress on real f39 operators
[DN] dense fp64 (stage 0, pooled): Mt=cM Frobenius, direct-solve identity,
     MC Var(u^T x) vs analytic u^T M^-1 u  (validates Cov(zeta)=M)
[T]  tol sweep on f9 spectral, fixed RHS, vs fp64 dense reference
[BL] gaussian_blur operator identity
[CB] diag_noise_off=["xi_h"] combo, off vs on
[H]  final-tensor SHA256 hashes archived
"""
import os, sys, json, math, hashlib
import numpy as np, torch

A = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2"
sys.path.insert(0, A)
os.chdir(A)
import s_prior_methods as SP
import main4, utils

O = A + "/results/alg4_eq22_sigma2_scaling"
rep = open(os.path.join(O, "validation2.md"), "w")


def emit(s):
    print(s, flush=True)
    rep.write(s + "\n")
    rep.flush()


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


emit("# Round-2 validation (post codex code review fixes)\n")
emit("Fixes under test: builder gated into else (no scalar 1/sigma^2 on "
     "scaled branch); guard-failure fallback M_inv = I/c; sigma>0 guard.\n")

# ---------------- [F] fallback unit tests (CPU) ----------------
emit("## [F] fallback scaling law")
# codex 1D counterexample: M=1, b=1e-3, c=1e-4
M1 = lambda x: x
b1 = torch.full((1, 1, 1, 1), 1e-3)
c = 1e-4
Mt1 = lambda x: c * x
xb, itb, rb = utils.pcg_solve(M1, b1, None, tol=1e-6, max_iter=8)
xs_bad, _, rs_bad = utils.pcg_solve(Mt1, c * b1, None, tol=1e-6, max_iter=8,
                                    clamp_floor=c * 1e-12)
xs_fix, its, rs = utils.pcg_solve(Mt1, c * b1, lambda r: r / c, tol=1e-6,
                                  max_iter=8, clamp_floor=c * 1e-12)
emit(f"[F] 1D: baseline x={float(xb):.9e} resid={rb:.2e} | "
     f"None-fallback(old bug) x={float(xs_bad):.9e} resid={rs_bad:.2e} | "
     f"I/c fallback x={float(xs_fix):.9e} resid={rs:.2e} "
     f"bitwise_equal={bool(torch.equal(xs_fix, xb))}")
# random 64-dim SPD, fp32
g = torch.Generator().manual_seed(7)
R = torch.randn(64, 64, generator=g)
Ms = R @ R.T + 64 * torch.eye(64)
v0 = torch.randn(1, 1, 1, 64, generator=g)
Mf = lambda x: (x.reshape(1, 64) @ Ms.T).reshape(1, 1, 1, 64)
c2 = 1e-6
Mtf = lambda x: c2 * Mf(x)
xb, itb, rb = utils.pcg_solve(Mf, v0, None, tol=1e-6, max_iter=200)
xs, its, rs = utils.pcg_solve(Mtf, c2 * v0, lambda r: r / c2, tol=1e-6,
                              max_iter=200, clamp_floor=c2 * 1e-12)
emit(f"[F] 64-dim SPD c=1e-6: it_b/s={itb}/{its} resid={rb:.2e}/{rs:.2e} "
     f"relX={rel(xs, xb):.2e} bitwise={bool(torch.equal(xs, xb))}")

# ---------------- setup ----------------
SP._init_globals()
device = "cuda:0"
config, model, gamma2_tab, S = SP._load_sampling(device)
stats = json.load(open(SP._stats_path()))
K = int(config.scheduler.num_stages)
s2_fns = {arm: SP._build_s_ops(arm, stats, K)
          for arm in ("spectral", "pooled_junco")}
hole = S["hole"].to(device).float()
gt = S["gt"].to(device)
y_full = S["y"].to(device)


def four_seed(arm, over, label, keep=False):
    for k in ("eq22_sigma2_scale", "diag_noise_off", "diag_noise_off_from_stage"):
        S["kw"].pop(k, None)
    S["kw"].update(over)
    holes, finals = [], []
    ncbad, iters_max = 0, 0
    for seed in (42, 43, 44, 45):
        x1, rows, _ = main4._run_once(model, config, S, device,
                                      s2_fn=s2_fns[arm],
                                      gamma2_tab=gamma2_tab, seed=seed)
        holes.append(float((((x1 - gt) ** 2) * hole).sum() / (hole.sum() * 3)))
        if keep or seed == 42:
            finals.append(x1)
        ncbad = max(ncbad, sum(1 for r in rows if not r["blk1_cg_converged"]))
        iters_max = max(iters_max, max(int(r["blk1_cg_iters"]) for r in rows))
    emit(f"[A2] {arm}/{label} holes=" + "/".join(f"{h:.4f}" for h in holes)
         + f" mean4={np.mean(holes):.4f}±{np.std(holes):.4f}"
           f" cg_bad={ncbad} it_max={iters_max}")
    return holes, finals


emit("\n## [A2] post-fix 4-seed regression")
fin = {}
for arm in ("spectral", "pooled_junco"):
    _, fb = four_seed(arm, {}, "base")
    _, fs = four_seed(arm, {"eq22_sigma2_scale": True}, "scaled")
    fin[arm] = (fb[0], fs[0])
    emit(f"[A2] {arm} seed42 max|dx|={float((fs[0]-fb[0]).abs().max()):.3e}")

emit("\n## [H] final-tensor SHA256 (seed42; archival reference)")
for arm, (fb, fs) in fin.items():
    hb = hashlib.sha256(fb.cpu().numpy().tobytes()).hexdigest()[:16]
    hs = hashlib.sha256(fs.cpu().numpy().tobytes()).hexdigest()[:16]
    emit(f"[H] {arm} base={hb} scaled={hs}")


# ---------------- grabs ----------------
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


def spec_double(s_op, x, sqrt=False):
    P = s_op.power.to(x.device).double()
    X = torch.fft.fft2(x, norm="ortho")
    return torch.fft.ifft2(X / (P.sqrt() if sqrt else P), norm="ortho").real


def build_pair(kw, s_op, sig=None):
    sig = float(kw["sigma_tau"]) if sig is None else sig
    tau, s_k, e_k, eff = kw["tau"], kw["s_k"], kw["e_k"], kw["eff_si"]
    eta, Af, ATf = kw["eta"], kw["A_fn"], kw["AT_fn"]
    M, Cinv, inv_e2, _, _ = utils.make_M_tau_den(
        Af, ATf, eta, sig, tau, s_k, e_k, eff, s_op)
    Mt, c, sc_e2 = utils.make_M_tau_den_sigma2(
        Af, ATf, eta, sig, tau, s_k, e_k, eff, s_op)
    return M, Mt, c, sc_e2, inv_e2, Cinv


def ident_checks(tag, kw, s_op, y_ref, sig=None):
    sig_v = float(kw["sigma_tau"]) if sig is None else sig
    tau, s_k, e_k, eff = kw["tau"], kw["s_k"], kw["e_k"], kw["eff_si"]
    eta, Af, ATf, shape = kw["eta"], kw["A_fn"], kw["AT_fn"], kw["shape"]
    M, Mt, c, sc_e2, inv_e2, Cinv = build_pair(kw, s_op, sig)
    g = torch.Generator(device="cpu").manual_seed(999)
    v = torch.randn(shape, generator=g).to(device)
    xh = torch.randn(shape, generator=g).to(device)
    xi_y = Af(torch.randn(shape, generator=g).to(device))
    xi_h = torch.randn(shape, generator=g).to(device)
    xi_s = torch.randn(shape, generator=g).to(device)
    H1 = lambda z: utils.apply_H_tau(z, tau, s_k, e_k, eff)
    rM32 = rel(Mt(v), c * M(v))
    q = (inv_e2 * ATf(y_ref) + Cinv(xh) + ATf(xi_y) / eta
         + H1(xi_h) / sig_v + s_op.apply_S_inv_sqrt(xi_s))
    qt = (sc_e2 * ATf(y_ref) + H1(H1(xh)) + c * s_op.apply_S_inv(xh)
          + (c / eta) * ATf(xi_y) + sig_v * H1(xi_h)
          + c * s_op.apply_S_inv_sqrt(xi_s))
    rq32 = rel(qt, c * q)
    # fp64: both arms (spectral via double wrapper). Blur operators are
    # float32 conv2d modules — for those, run the A term in fp32 and cast
    # back (H/S terms stay fp64; both sides share the same A path, so the
    # Mt = c M comparison stays meaningful).
    vd = v.double()
    if isinstance(s_op, utils.ScalarSOp):
        Sinv_d = lambda z: z / float(s_op.scalar_equiv)
    else:
        Sinv_d = lambda z: spec_double(s_op, z)

    def A2d(z):
        try:
            return ATf(Af(z))
        except RuntimeError:
            return ATf(Af(z.float())).double()

    Md = lambda z: inv_e2 * A2d(z) + H1(H1(z)) / sig_v ** 2 + Sinv_d(z)
    Mtd = lambda z: sc_e2 * A2d(z) + H1(H1(z)) + c * Sinv_d(z)
    rM64 = rel(Mtd(vd), c * Md(vd))
    Minv_b = utils.make_jacobi_precond(M, shape, device)
    Minv_s = utils.make_jacobi_precond(Mt, shape, device, floor=c * 1e-12)
    if Minv_s is None:
        Minv_s = (lambda r, _c=c: r / _c)
        fb = " FALLBACK"
    else:
        fb = ""
    xb, itb, rb = utils.pcg_solve(M, q, Minv_b, tol=1e-5, max_iter=600)
    xs, its, rs = utils.pcg_solve(Mt, qt, Minv_s, tol=1e-5, max_iter=600,
                                  clamp_floor=c * 1e-12)
    nb = int((~torch.isfinite(xs)).sum())
    emit(f"[{tag}] sig={sig_v:.3e}: relM32={rM32:.2e} relM64={rM64:.2e} "
         f"relRHS32={rq32:.2e} relX={rel(xs, xb):.2e} it={itb}/{its} "
         f"res={rb:.1e}/{rs:.1e} nonfinite={nb}{fb} "
         f"max|Mv|={float(M(v).abs().max()):.2e} "
         f"max|Mtv|={float(Mt(v).abs().max()):.2e}")
    return kw


emit("\n## [G] frame identities incl. spectral fp64 (tol=1e-5, prod-like)")
WANT = (9, 15, 25, 31, 38, 39)
grabs = {}
for arm in ("pooled_junco", "spectral"):
    grab = Grab(WANT)
    for k in ("eq22_sigma2_scale",):
        S["kw"].pop(k, None)
    main4._run_once(model, config, S, device, s2_fn=s2_fns[arm],
                    gamma2_tab=gamma2_tab, seed=42, diag=grab)
    grabs[arm] = grab.store
    for f in sorted(grab.store):
        kw = grab.store[f]
        s_ret = s2_fns[arm](kw["stage"], float(kw["sigma_tau"]))
        s_op = s_ret if isinstance(s_ret, utils.SOperator) \
            else utils.ScalarSOp(float(s_ret))
        ident_checks(f"G:{arm}:f{f}:st{kw['stage']}", kw, s_op, y_full)

emit("\n## [S8] sigma=1e-8 stress on real f39 operators")
for arm in ("pooled_junco", "spectral"):
    kw = grabs[arm][39]
    s_ret = s2_fns[arm](kw["stage"], 1e-8)
    s_op = s_ret if isinstance(s_ret, utils.SOperator) \
        else utils.ScalarSOp(float(s_ret))
    ident_checks(f"S8:{arm}", kw, s_op, y_full, sig=1e-8)

# ---------------- [DN] dense fp64, stage 0, pooled ----------------
emit("\n## [DN] dense fp64 stage-0 (pooled): Mt=cM, direct-solve, MC cov")
kw = grabs["pooled_junco"][9]
s_ret = s2_fns["pooled_junco"](0, float(kw["sigma_tau"]))
s_op = utils.ScalarSOp(float(s_ret)) if not isinstance(s_ret, utils.SOperator) else s_ret
sig = float(kw["sigma_tau"])
tau, s_k, e_k, eff = kw["tau"], kw["s_k"], kw["e_k"], kw["eff_si"]
eta, Af, ATf, shape = kw["eta"], kw["A_fn"], kw["AT_fn"], kw["shape"]
D = int(np.prod(shape[1:]))
H1 = lambda z: utils.apply_H_tau(z, tau, s_k, e_k, eff)
inv_e2 = 1.0 / eta ** 2
c = sig ** 2
Md_fn = lambda z: inv_e2 * ATf(Af(z)) + H1(H1(z)) / sig ** 2 + z / float(s_op.scalar_equiv)
Mtd_fn = lambda z: (c / eta ** 2) * ATf(Af(z)) + H1(H1(z)) + (c / float(s_op.scalar_equiv)) * z
cols = []
cols_t = []
B = 256
eye = torch.eye(D, dtype=torch.float64, device=device)
for i in range(0, D, B):
    E = eye[i:i + B].reshape(-1, *shape[1:])
    cols.append(Md_fn(E).reshape(E.shape[0], D))
    cols_t.append(Mtd_fn(E).reshape(E.shape[0], D))
Mdense = torch.cat(cols).T
Mtdense = torch.cat(cols_t).T
emit(f"[DN] D={D} frob rel |Mt - c M| = "
     f"{float((Mtdense - c * Mdense).norm() / (c * Mdense).norm()):.2e} "
     f"sym_err={float((Mdense - Mdense.T).abs().max()):.2e}")
L = torch.linalg.cholesky(Mdense)
g = torch.Generator(device="cpu").manual_seed(5150)
xh = torch.randn(shape, generator=g).to(device).double()
b_det = (inv_e2 * ATf(y_full.double()) + H1(H1(xh)) / sig ** 2
         + xh / float(s_op.scalar_equiv)).reshape(D, 1)
bt_det = ((c / eta ** 2) * ATf(y_full.double()) + H1(H1(xh))
          + (c / float(s_op.scalar_equiv)) * xh).reshape(D, 1)
NMC = 4000
U = torch.randn(D, 32, generator=g, dtype=torch.float64).to(device)
U = U / U.norm(dim=0, keepdim=True)
W = torch.cholesky_solve(U, L)                      # W = M^-1 U
var_ana = (U * W).sum(dim=0)                        # u^T M^-1 u
maxdx = 0.0
zs = []
yshape = tuple(y_full.shape[1:])       # xi_y lives in MEASUREMENT space:
for i in range(0, NMC, 100):           # Cov(A^T xi_y / eta) = A^T A / eta^2,
    n = min(100, NMC - i)              # matching the production Lemma-9 RTO.
    xi_y = torch.randn((n,) + yshape, generator=g).to(device).double()
    xi_h = torch.randn((n,) + tuple(shape[1:]), generator=g).to(device).double()
    xi_s = torch.randn((n,) + tuple(shape[1:]), generator=g).to(device).double()
    zeta = (ATf(xi_y) / eta + H1(xi_h) / sig
            + xi_s / math.sqrt(float(s_op.scalar_equiv))).reshape(n, D)
    x_un = torch.cholesky_solve(b_det + zeta.T, L)
    x_sc = torch.linalg.solve(Mtdense, bt_det + c * zeta.T)
    maxdx = max(maxdx, float((x_sc - x_un).abs().max()))
    zs.append(zeta @ W)                              # (b+zeta)^T M^-1 u parts
zz = torch.cat(zs)
var_emp = zz.var(dim=0, unbiased=True)
dev = ((var_emp - var_ana) / var_ana).abs()
emit(f"[DN] direct-solve per-realization max|x_sc - x_un| = {maxdx:.2e} "
     f"(N={NMC})")
emit(f"[DN] MC cov (Cov(zeta)=M check): max|var_emp/var_ana - 1| = "
     f"{float(dev.max()):.3f} (32 dirs, N={NMC}, MC 1-sigma ~ "
     f"{math.sqrt(2.0 / NMC):.3f})")

# ---------------- [T] tol sweep, f9 spectral ----------------
emit("\n## [T] tol sweep on f9 spectral (fixed RHS, vs fp64 dense ref)")
kw = grabs["spectral"][9]
s_ret = s2_fns["spectral"](0, float(kw["sigma_tau"]))
s_op = s_ret
sig = float(kw["sigma_tau"])
tau, s_k, e_k, eff = kw["tau"], kw["s_k"], kw["e_k"], kw["eff_si"]
eta, Af, ATf, shape = kw["eta"], kw["A_fn"], kw["AT_fn"], kw["shape"]
D = int(np.prod(shape[1:]))
c = sig ** 2
H1 = lambda z: utils.apply_H_tau(z, tau, s_k, e_k, eff)
Sd = lambda z: spec_double(s_op, z)
Md_fn = lambda z: ATf(Af(z)) / eta ** 2 + H1(H1(z)) / sig ** 2 + Sd(z)
cols = []
for i in range(0, D, B):
    E = eye[i:i + B].reshape(-1, *shape[1:])
    cols.append(Md_fn(E).reshape(E.shape[0], D))
Mdense = torch.cat(cols).T
Mdense = 0.5 * (Mdense + Mdense.T)
g = torch.Generator(device="cpu").manual_seed(4242)
xh = torch.randn(shape, generator=g).to(device)
xi_y = Af(torch.randn(shape, generator=g).to(device))
xi_h = torch.randn(shape, generator=g).to(device)
xi_s = torch.randn(shape, generator=g).to(device)
M, Mt, cc, sc_e2, inv_e2, Cinv = build_pair(kw, s_op)
q = (inv_e2 * ATf(y_full) + Cinv(xh) + ATf(xi_y) / eta + H1(xi_h) / sig
     + s_op.apply_S_inv_sqrt(xi_s))
qt = (sc_e2 * ATf(y_full) + H1(H1(xh)) + cc * s_op.apply_S_inv(xh)
      + (cc / eta) * ATf(xi_y) + sig * H1(xi_h)
      + cc * s_op.apply_S_inv_sqrt(xi_s))
x_ref = torch.linalg.solve(Mdense, q.double().reshape(D, 1)).reshape(shape).float()
Minv_b = None
_ones = torch.ones(shape, device=device)
_d = (M(_ones) - s_op.apply_S_inv(_ones) + s_op.inv_diag_mean() * _ones)
Minv_b = (lambda r, _i=(1.0 / _d): _i * r)
_dt = (Mt(_ones) - cc * s_op.apply_S_inv(_ones)
       + cc * s_op.inv_diag_mean() * _ones)
Minv_s = (lambda r, _i=(1.0 / _dt): _i * r)
for tol in (1e-5, 1e-6, 1e-7):
    xb, itb, rb = utils.pcg_solve(M, q, Minv_b, tol=tol, max_iter=2000)
    xs, its, rs = utils.pcg_solve(Mt, qt, Minv_s, tol=tol, max_iter=2000,
                                  clamp_floor=cc * 1e-12)
    emit(f"[T] tol={tol:.0e}: it={itb}/{its} "
         f"relX(s,b)={rel(xs, xb):.2e} "
         f"err_b(vs dense)={rel(xb, x_ref):.2e} err_s={rel(xs, x_ref):.2e}")

# ---------------- [BL] gaussian_blur identity ----------------
emit("\n## [BL] gaussian_blur operator identity (junco, f9/f38)")
Sb = main4._task_setup("gaussian_blur", "junco", device, config)
grab = Grab((9, 38))
main4._run_once(model, config, Sb, device, s2_fn=s2_fns["pooled_junco"],
                gamma2_tab=gamma2_tab, seed=42, diag=grab)
for f in sorted(grab.store):
    kwb = grab.store[f]
    s_ret = s2_fns["pooled_junco"](kwb["stage"], float(kwb["sigma_tau"]))
    s_op2 = s_ret if isinstance(s_ret, utils.SOperator) \
        else utils.ScalarSOp(float(s_ret))
    ident_checks(f"BL:f{f}", kwb, s_op2, Sb["y"].to(device))

# ---------------- [CB] diag_noise_off combo ----------------
emit("\n## [CB] diag_noise_off=['xi_h'] combo (spectral seed42)")
res = {}
for lab, over in (("off", {"diag_noise_off": ["xi_h"],
                           "diag_noise_off_from_stage": 0}),
                  ("on", {"diag_noise_off": ["xi_h"],
                          "diag_noise_off_from_stage": 0,
                          "eq22_sigma2_scale": True})):
    for k in ("eq22_sigma2_scale", "diag_noise_off",
              "diag_noise_off_from_stage"):
        S["kw"].pop(k, None)
    S["kw"].update(over)
    x1, rows, _ = main4._run_once(model, config, S, device,
                                  s2_fn=s2_fns["spectral"],
                                  gamma2_tab=gamma2_tab, seed=42)
    h = float((((x1 - gt) ** 2) * hole).sum() / (hole.sum() * 3))
    res[lab] = (x1, h)
    emit(f"[CB] eq22={lab}: hole={h:.4f}")
emit(f"[CB] max|dx|={float((res['on'][0] - res['off'][0]).abs().max()):.3e}")

for k in ("eq22_sigma2_scale", "diag_noise_off", "diag_noise_off_from_stage"):
    S["kw"].pop(k, None)

emit("\nEQ22 VALIDATE2 DONE")
rep.close()
