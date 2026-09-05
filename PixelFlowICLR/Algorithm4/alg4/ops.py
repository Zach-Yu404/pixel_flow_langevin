"""Linear operators of the interpolant and the solvers. Every function here is numerically identical to the
reference implementation (PixelFlowICLR/Algorithm2/utils.py + IP_package/ms_posterior_sampling_article_version_final_utils.py).

Interpolant (stage k, time tau in [0,1]):
    H_tau = (1-tau) s_k G + tau e_k I           G = nearest_up(bilinear_down(x)) -- a low-pass projection
    sigma_tau = (1-tau)(1-s_k) + tau(1-e_k)
    B_k = e_k I - s_k G  (= dH/dtau)            N_k = e_k(1-s_k) I - s_k(1-e_k) G
"""
import math
import torch
import torch.nn.functional as F


# ── interpolant operators ─────────────────────────────────────────────────────
def apply_G(x, scale_factor=2):
    """G = nearest_up . bilinear_down: same low-pass projection at every stage (G^2 = G, G^T = G)."""
    _, _, H, W = x.shape
    small = (H // scale_factor, W // scale_factor)
    return F.interpolate(
        F.interpolate(x, size=small, mode="bilinear", align_corners=False),
        size=(H, W), mode="nearest",
    )


def apply_H_tau(x, tau, s_k, e_k):
    """H_tau^k x = (1-tau) s_k G x + tau e_k x."""
    return (1.0 - tau) * s_k * apply_G(x) + tau * e_k * x


def apply_B(x, s_k, e_k):
    """B_k x = e_k x - s_k G x  (= dH_tau/dtau)."""
    return e_k * x - s_k * apply_G(x)


def apply_N(x, s_k, e_k):
    """N_k x = e_k (1-s_k) x - s_k (1-e_k) G x."""
    return e_k * (1.0 - s_k) * x - s_k * (1.0 - e_k) * apply_G(x)


def compute_sigma_tau(tau, s_k, e_k):
    """sigma_tau = (1-tau)(1-s_k) + tau(1-e_k)."""
    return (1.0 - tau) * (1.0 - s_k) + tau * (1.0 - e_k)


def per_stage(val, stage_idx, num_stages):
    """Resolve a scalar / length-num_stages list at the given stage."""
    if isinstance(val, (list, tuple)):
        if len(val) != num_stages:
            raise ValueError(f"per-stage list length {len(val)} != num_stages {num_stages}: {val}")
        return val[stage_idx]
    return val


def gt_stage_pyramid(gt, num_stages):
    """Stage targets x1^k by chain-of-bilinear-halving (the training-time collate), k = 0 coarsest."""
    pyr = {num_stages - 1: gt}
    cur = gt
    for k in range(num_stages - 2, -1, -1):
        cur = F.interpolate(cur, size=(cur.shape[-2] // 2, cur.shape[-1] // 2), mode="bilinear")
        pyr[k] = cur
    return pyr


# ── solvers ──────────────────────────────────────────────────────────────────
def cg_solve(A_fn, b, x0=None, tol=1e-5, max_iter=50):
    """Plain conjugate gradients for an SPD matrix-free operator (batched over dim 0)."""
    B = b.shape[0]
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A_fn(x)
    p = r.clone()
    rs_old = (r * r).reshape(B, -1).sum(dim=1)
    b_norm = (b * b).reshape(B, -1).sum(dim=1).sqrt().clamp(min=1e-12)
    for _ in range(max_iter):
        Ap = A_fn(p)
        pAp = (p * Ap).reshape(B, -1).sum(dim=1).clamp(min=1e-12)
        alpha = (rs_old / pAp).reshape(B, 1, 1, 1)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).reshape(B, -1).sum(dim=1)
        if (rs_new.sqrt() / b_norm).max() < tol:
            break
        beta = (rs_new / rs_old.clamp(min=1e-12)).reshape(B, 1, 1, 1)
        p = r + beta * p
        rs_old = rs_new
    return x


def pcg_solve(A_fn, b, M_inv=None, x0=None, tol=1e-5, max_iter=50):
    """Preconditioned CG. Returns (x, iterations, relative residual) so a truncated solve is visible."""
    B = b.shape[0]

    def dot(u, v):
        return (u * v).reshape(B, -1).sum(dim=1)

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A_fn(x)
    z = r if M_inv is None else M_inv(r)
    p = z.clone()
    rz = dot(r, z)
    b_norm = dot(b, b).sqrt().clamp(min=1e-12)
    rel = float((dot(r, r).sqrt() / b_norm).max())
    it = 0
    for it in range(1, int(max_iter) + 1):
        Ap = A_fn(p)
        alpha = (rz / dot(p, Ap).clamp(min=1e-12)).reshape(B, 1, 1, 1)
        x = x + alpha * p
        r = r - alpha * Ap
        rel = float((dot(r, r).sqrt() / b_norm).max())
        if rel < tol:
            break
        z = r if M_inv is None else M_inv(r)
        rz_new = dot(r, z)
        p = z + (rz_new / rz.clamp(min=1e-12)).reshape(B, 1, 1, 1) * p
        rz = rz_new
    return x, it, rel


def make_jacobi_precond(M_fn, shape, device, floor=1e-12):
    """Diagonal (Jacobi) preconditioner from M applied to the all-ones vector; None if the probe is not
    strictly positive (caller then falls back to plain CG). Any SPD preconditioner leaves the solution unchanged."""
    d = M_fn(torch.ones(shape, device=device))
    if not bool(torch.isfinite(d).all()) or float(d.min()) <= floor:
        return None
    inv = 1.0 / d
    return lambda r: inv * r


def make_exact_AT(A_fn, x_shape):
    """Exact adjoint of a linear A_fn via autograd (used for the blur operators, whose analytic flip(K)
    adjoint ignores that reflection padding is not self-adjoint at the border)."""
    def AT(r):
        det = torch.are_deterministic_algorithms_enabled()
        wo = torch.is_deterministic_algorithms_warn_only_enabled()
        if det and not wo:
            torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            with torch.enable_grad():
                xp = torch.zeros(x_shape, device=r.device, dtype=r.dtype, requires_grad=True)
                Ax = A_fn(xp)
                (grad,) = torch.autograd.grad(Ax, xp, grad_outputs=r.detach())
        finally:
            if det and not wo:
                torch.use_deterministic_algorithms(True, warn_only=False)
        return grad.detach()
    return AT


# ── metrics used inside the sampler ──────────────────────────────────────────
def mse_masked(a, b, m):
    """Mean squared error over mask m (broadcast [1,1,h,w]); NaN if the mask is empty."""
    n = m.sum() * a.shape[1]
    if float(n) == 0:
        return float("nan")
    return float(((a - b) ** 2 * m).sum() / n)


def measurement_residual(A_fn, x1, y, eta):
    """||A x1 - y|| / (eta sqrt(m)), averaged over the batch; should settle near 1."""
    B = y.shape[0]
    m = y[0].numel()
    r = ((A_fn(x1) - y) ** 2).reshape(B, -1).sum(dim=1).sqrt()
    return float((r / (eta * math.sqrt(m))).mean())
