"""Algorithm 4 — the sampler (draft Sec. 7). Numerically identical to the reference
(PixelFlowICLR/Algorithm2/utils.run_posterior_sampling_alg4 with every diagnostic switch off).

    l.1   x1 <- 0
    l.2   for stage k = 0..K-1:
    l.3     x0 ~ N(0, I)                                   [fresh endpoint noise per stage]
    l.4     for tau on the stage grid:
    l.5       H_tau, sigma_tau;  l.6 N_k;  l.7 C^-1 = H^T H/sigma^2 + S^-1,  M^den = A^T A/eta^2 + C^-1
    l.8       x_tau <- H_tau x1 + sigma_tau x0             [once per tau]
    l.9       for s = 1..S_it:
    l.10        v <- v_theta(x_tau, tau, k)                 [one network evaluation]
    l.11        [N^2 + gamma^2 H^2] x1_hat = N[(e-s) x_tau + sigma v] + gamma^2 H x_tau      (19)
    l.12        xi_y ~ N(0, I_m);  xi_h, xi_s ~ N(0, I_n)
    l.13        M^den x1 = A^T y/eta^2 + C^-1 x1_hat + A^T xi_y/eta + H^T xi_h/sigma + S^-1/2 xi_s   (22)  [PCG]
    l.14        xi_0 ~ N(0, I_n);  x_tau <- H_tau x1 + sigma_tau xi_0                       (23)
    l.16      x0 <- (x_tau - H_tau x1)/sigma_tau
    l.18    x1 <- nearest-upsample(x1)                      [ancestral transition to the next stage]
"""
import copy
import torch
import torch.nn.functional as F

from .ops import (apply_G, apply_H_tau, apply_N, compute_sigma_tau, per_stage, gt_stage_pyramid,
                  cg_solve, pcg_solve, make_jacobi_precond, mse_masked, measurement_residual)
from .prior import SOperator, SpectralSOp
from pixelflow.scheduling_pixelflow import PixelFlowScheduler

NFE = {"n": 0}


def count_nfe_hook(module, args, kwargs=None):
    NFE["n"] += 1


# ── velocity network wrapper (CFG with the per-stage guidance schedule) ─────────────────────────
def class_guidance_scale(base_scale, stage_idx):
    scale_dict = {0: 0.0, 1: 1.0 / 6.0, 2: 2.0 / 3.0, 3: 1.0}
    return (base_scale - 1.0) * scale_dict.get(stage_idx, 1.0) + 1.0


def rope_for(model, h, w, device):
    from diffusers.models.embeddings import get_2d_rotary_pos_embed
    size_tensor = torch.tensor([h // model.patch_size], dtype=torch.int32, device=device)
    pos = get_2d_rotary_pos_embed(
        embed_dim=model.attention_head_dim,
        crops_coords=((0, 0), (h // model.patch_size, w // model.patch_size)),
        grid_size=(h // model.patch_size, w // model.patch_size),
        device=device, output_type="pt",
    )
    return size_tensor, torch.stack(pos, -1)


def make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos, do_cfg, guidance_scale_base, stage_idx):
    def velocity_fn(x_tau):
        inp = torch.cat([x_tau] * 2) if do_cfg else x_tau
        ts = T.expand(inp.shape[0]).to(inp.dtype)
        with torch.no_grad():
            pred = model(inp, timestep=ts, class_labels=prompt_embeds, latent_size=size_tensor, pos_embed=rope_pos)
        if do_cfg:
            scale = class_guidance_scale(guidance_scale_base, stage_idx)
            u, c = pred.chunk(2)
            pred = u + scale * (c - u)
        return pred
    return velocity_fn


def stage_schedule(config, si, ode_steps, shift, device):
    """The scheduler state for one stage: (s_k, e_k, tau grid, Timesteps)."""
    sc = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                            num_stages=int(config.scheduler.num_stages), gamma=-1 / 3)
    sc.set_timesteps(int(ode_steps), si, device=device, shift=shift)
    return sc


# ── the two linear systems ───────────────────────────────────────────────────────────────────
def make_endpoint_operator(s_k, e_k, tau, gamma2):
    """u -> [N_k^2 + gamma^2 H_tau^2] u  -- SPD at every tau (N_k has eigenvalues e_k-s_k, e_k(1-s_k) > 0)."""
    def op(u):
        Nu = apply_N(apply_N(u, s_k, e_k), s_k, e_k)
        if gamma2 > 0:
            Hu = apply_H_tau(apply_H_tau(u, tau, s_k, e_k), tau, s_k, e_k)
            Nu = Nu + gamma2 * Hu
        return Nu
    return op


def clean_endpoint_solve(x_tau, v, sigma_tau, s_k, e_k, tau, gamma2, cg_tol, L, x1_warm=None):
    """(19): [N^2 + gamma^2 H^2] x1_hat = N[(e_k - s_k) x_tau + sigma_tau v] + gamma^2 H x_tau."""
    op = make_endpoint_operator(s_k, e_k, tau, gamma2)
    rhs = apply_N((e_k - s_k) * x_tau + float(sigma_tau) * v, s_k, e_k)
    if gamma2 > 0:
        rhs = rhs + gamma2 * apply_H_tau(x_tau, tau, s_k, e_k)
    return cg_solve(op, rhs, x0=x1_warm, tol=cg_tol, max_iter=L)


def make_M_tau_den(A_fn, AT_fn, eta, sigma_tau, tau, s_k, e_k, s_op):
    """(12)+(21): C^-1 = H^T H/sigma^2 + S^-1,  M^den = A^T A/eta^2 + C^-1.  Returns (M, Cinv, 1/eta^2, 1/sigma^2)."""
    inv_e2, inv_s2 = 1.0 / eta ** 2, 1.0 / float(sigma_tau) ** 2

    def Cinv(x):
        return inv_s2 * apply_H_tau(apply_H_tau(x, tau, s_k, e_k), tau, s_k, e_k) + s_op.apply_S_inv(x)

    def M(x):
        return inv_e2 * AT_fn(A_fn(x)) + Cinv(x)

    return M, Cinv, inv_e2, inv_s2


def _block1_preconditioner(M_den, s_op, x1, device):
    """Jacobi preconditioner for M^den. For a spectral S the all-ones probe would read S^-1's DC response
    instead of its constant diagonal mean_w 1/P(w), so the S^-1 part of the probe is replaced exactly."""
    if isinstance(s_op, SpectralSOp):
        ones = torch.ones_like(x1)
        d = M_den(ones) - s_op.apply_S_inv(ones) + s_op.inv_diag_mean() * ones
        if bool(torch.isfinite(d).all()) and float(d.min()) > 1e-12:
            inv_d = 1.0 / d
            return lambda r, _inv_d=inv_d: _inv_d * r
        return None
    return make_jacobi_precond(M_den, x1.shape, device)


# ── the sampler ─────────────────────────────────────────────────────────────────────────────
def run_algorithm4(model, config, gt, y, operator, eta, device, *,
                   make_Ak_fns, s2_fn, gamma2_tab,
                   num_langevin, ode_steps_per_stage=10, shift=1.0, guidance_scale=2.0, class_label=0,
                   cg_tol=1e-5, cg_max_iter=300, cg_max_iter_endpoint=200, sigma_min=1e-8,
                   seed=42, hole_mask=None, terminal_replace_weight=0.0, record_trajectory=False):
    """Returns (x1, rows, traj). rows: one dict per (stage, step) with mse_x1, meas_resid, x0_rms, Block-1 CG
    receipts and (with hole_mask) mse_hole / mse_obs at the stage resolution. traj (optional): (x_tau, x1, x1_hat)."""
    B = gt.shape[0]
    L = int(cg_max_iter)
    num_stages = int(config.scheduler.num_stages)
    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps, num_stages=num_stages, gamma=-1 / 3)

    pe_labels = torch.tensor([int(class_label)] * B, dtype=torch.int32, device=device)
    do_cfg = guidance_scale > 0
    if do_cfg:
        uncond_label = int(model.num_classes)
        prompt_embeds = torch.cat([uncond_label * torch.ones_like(pe_labels), pe_labels], dim=0)
    else:
        prompt_embeds = pe_labels

    target_h, target_w = int(gt.shape[-2]), int(gt.shape[-1])
    init_factor = 2 ** (num_stages - 1)
    h, w = target_h // init_factor, target_w // init_factor

    g = torch.Generator(device="cpu").manual_seed(int(seed))     # one CPU noise stream for every xi

    def randn_like_cpu(x):
        return torch.randn(x.shape, generator=g).to(x.device)

    pyr = gt_stage_pyramid(gt, num_stages)
    x1 = torch.zeros((B, 3, h, w), device=device)                                     # l.1
    rows, traj = [], []
    frame = -1
    for si in range(num_stages):                                                      # l.2
        sc = copy.deepcopy(scheduler)
        ode_steps_si = int(float(per_stage(ode_steps_per_stage, si, num_stages)))
        S_it_entry = per_stage(num_langevin, si, num_stages)
        if isinstance(S_it_entry, (list, tuple)) and len(S_it_entry) != ode_steps_si:
            raise ValueError(f"per-frame num_langevin for stage {si} has {len(S_it_entry)} entries, "
                             f"stage has {ode_steps_si} steps")
        sc.set_timesteps(ode_steps_si, si, device=device, shift=shift)
        s_k, e_k = float(sc.start_t[si]), float(sc.end_t[si])

        if si > 0:                                                                    # l.18
            h *= 2
            w *= 2
            x1 = F.interpolate(x1, size=(h, w), mode="nearest")
        x0 = randn_like_cpu(pyr[si])                                                  # l.3

        Ak, ATk = make_Ak_fns(operator, y, (B, 3, h, w), device)
        size_tensor, rope_pos = rope_for(model, h, w, device)
        gamma2_stage = gamma2_tab[str(si)]
        hole_k = None if hole_mask is None else F.interpolate(hole_mask.to(device).float(), size=(h, w), mode="nearest")

        for step_idx, T in enumerate(sc.Timesteps):                                   # l.4
            frame += 1
            S_it = int(float(S_it_entry[step_idx])) if isinstance(S_it_entry, (list, tuple)) else int(float(S_it_entry))
            tau = float(sc.t[step_idx])
            sigma_tau = compute_sigma_tau(tau, s_k, e_k)                              # l.5
            velocity_fn = make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos, do_cfg, guidance_scale, si)
            gamma2 = float(gamma2_stage.get(f"{round(tau, 6)}", list(gamma2_stage.values())[step_idx]))
            x1_hat = None
            s2 = float("nan")
            cg_iters, cg_resid = 0, 0.0
            if sigma_tau >= sigma_min:
                s_op = s2_fn(si, float(sigma_tau))
                if not isinstance(s_op, SOperator):
                    raise TypeError("s2_fn must return an SOperator (the per-class spectral S)")
                s2 = float(s_op.scalar_equiv)
                M_den, Cinv, inv_e2, inv_s2 = make_M_tau_den(Ak, ATk, eta, sigma_tau, tau, s_k, e_k, s_op)   # l.6-7
                M_inv = _block1_preconditioner(M_den, s_op, x1, device)
                x_tau = apply_H_tau(x1, tau, s_k, e_k) + sigma_tau * x0                                    # l.8

                for _ in range(S_it):                                                                      # l.9
                    with torch.no_grad():
                        v = velocity_fn(x_tau)                                                             # l.10
                    x1_hat = clean_endpoint_solve(x_tau, v, sigma_tau, s_k, e_k, tau, gamma2,
                                                  cg_tol, cg_max_iter_endpoint, x1_warm=x1.clone())        # l.11
                    xi_y = randn_like_cpu(y)                                                               # l.12
                    xi_h = randn_like_cpu(x1)
                    xi_s = randn_like_cpu(x1)
                    b_tilde = (inv_e2 * ATk(y) + Cinv(x1_hat)
                               + (1.0 / eta) * ATk(xi_y)
                               + (1.0 / float(sigma_tau)) * apply_H_tau(xi_h, tau, s_k, e_k)
                               + s_op.apply_S_inv_sqrt(xi_s))                                              # l.13 (22)
                    x1, cg_it, cg_rel = pcg_solve(M_den, b_tilde, M_inv, x0=x1.clone(), tol=cg_tol, max_iter=L)
                    cg_iters = max(cg_iters, cg_it)
                    cg_resid = max(cg_resid, cg_rel)
                    xi_0 = randn_like_cpu(x0)                                                              # l.14
                    x_tau = apply_H_tau(x1, tau, s_k, e_k) + float(sigma_tau) * xi_0                       # (23)

                x0 = (x_tau - apply_H_tau(x1, tau, s_k, e_k)) / float(sigma_tau)                          # l.16

            row = dict(stage=si, step=step_idx, tau=tau, sigma_tau=float(sigma_tau), s2=s2, gamma2=gamma2,
                       mse_x1=float(((x1 - pyr[si]) ** 2).mean()),
                       meas_resid=measurement_residual(Ak, x1, y, eta),
                       x0_rms=float((x0 ** 2).mean().sqrt()),
                       blk1_cg_iters=cg_iters, blk1_cg_resid=cg_resid,
                       blk1_cg_converged=int(cg_iters == 0 or cg_resid < cg_tol))
            if hole_k is not None:
                row["mse_hole"] = mse_masked(x1, pyr[si], hole_k)
                row["mse_obs"] = mse_masked(x1, pyr[si], 1.0 - hole_k)
            row["frame"] = frame
            rows.append(row)
            if record_trajectory:
                x_tau_rec = apply_H_tau(x1, tau, s_k, e_k) + float(sigma_tau) * x0
                traj.append((x_tau_rec[0].cpu(), x1[0].cpu(), (x1_hat if x1_hat is not None else x1)[0].cpu()))

    if terminal_replace_weight > 0:          # optional post-sampling projection onto the observed pixels
        m = operator.get_mask(x=y).float().to(x1.device)
        x1 = terminal_replace_weight * (m * y + (1.0 - m) * x1) + (1.0 - terminal_replace_weight) * x1
    return x1, rows, traj
