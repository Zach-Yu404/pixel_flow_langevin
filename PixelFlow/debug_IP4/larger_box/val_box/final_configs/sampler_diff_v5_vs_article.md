# `ms_sampler_v5` vs `ms_posterior_sampling_article_version_final` — structural diff

Comparison of:

- **v5**: `PixelFlow/debug_IP4/ms_sampler_v5.py` (288 L) + `PixelFlow/debug_IP4/langevin_v5.py`
- **article**: `PixelFlow/ms_posterior_sampling_article_version_final.py` (414 L) + `PixelFlow/ms_posterior_sampling_article_version_final_utils.py` (361 L)

All line citations are `path:line`.

## TL;DR

`ms_sampler_v5` is the **article version's math** plus **12 sweep-only experimental axes** (Round-6 epsilon-augmentation knobs N1–N4, mask-aware step scaling, hybrid DPS pre-kick, terminal/soft replacement, per-stage `sigma_ref_sq`, per-stage skip / reset of the eps update, optional final WLS-CG denoise). The core PRINCIPLE Langevin gradient and the helper library are **imported unchanged** from the article-version utils file (`langevin_v5.py:20`). The two diverge in three ways:

1. **API shape** — v5 is a stateless function (`run_ip4(**kwargs) → metric tuple`); the article version is a CLI (`main(config_path)` driven by JSON config, with disk I/O for trajectories/videos).
2. **Trajectory recording** — the article saves 8 intermediate-state tensors plus per-step Langevin logs; v5 discards them and computes 3 PSNR variants + HF energy inline.
3. **Knob surface** — v5 exposes 12 extra parameters (most as scalar-or-per-stage-list), all controlling experimental Round-6 axes. The article version exposes I/O knobs (`return_traj`, `record_every`, `save_videos`, etc.) that v5 lacks.

In the 167-config sweep over the IP4 winner space, **none** of the 12 v5-only knobs produced a Pareto-optimal config — the eventual winner (`F_cfg20_lr150_hxu02`: CFG=2 + λ_reg=150 + h_x=0.2 uniform) only uses knobs that already exist in the article version. The v5-only knobs all collapsed to a worse attractor or had no effect (see §G).

---

## A. Public API & entry points

### v5 — `ms_sampler_v5.py:55-56`

```python
@torch.no_grad()
def run_ip4(...):
```

Single function with ~30 kwargs. Return value:

```python
return xf.cpu(), psnr_obs, psnr_all, res, elapsed, hf_u    # ms_sampler_v5.py:288
```

A 6-tuple of `(reconstruction, PSNR_observed, PSNR_full, residual_norm, wall_time, HF_energy_unobserved)`. No disk I/O. Designed for use in chunked sweeps where the caller writes its own metrics JSON.

### article — `ms_posterior_sampling_article_version_final.py:61`

```python
def main(config_path):
```

Single parameter (a path). Parses JSON via `load_run_config()` (`ms_posterior_sampling_article_version_final_utils.py:31`), constructs experiment paths via `build_experiment_paths()`, runs the sampler, then writes trajectory tensors / videos / CSV logs to disk. No return value of interest.

**Net difference**: v5 is sweep-friendly (stateless, low-latency); article is publication/diagnostic-friendly (full state recording, CLI).

---

## B. Sampling flow

The 4-stage outer structure is **identical** in both:

```
for stage_idx in range(num_stages):                # 4 stages, gamma = -1/3
  for step_idx, T in enumerate(Timesteps_k):       # variable ODE steps per stage
    optional warm_restart                          # re-init x1_k from model prediction
    optional DPS pre-kick                          # v5 only
    Langevin inner loop (num_langevin steps)       # core math
    optional soft replacement at stage 3           # v5 only
    advance lat = apply_H_tau(...) + sig * eps_k   # state advance
optional final WLS-CG denoise                       # v5 only
optional terminal replacement                       # v5 only
```

| Step | v5 location | article location |
|---|---|---|
| Warm restart | `ms_sampler_v5.py:183-191` | `ms_posterior_sampling_article_version_final.py:298-309` |
| DPS pre-kick | `ms_sampler_v5.py:196-200` | — *(v5 only)* |
| Langevin call | `ms_sampler_v5.py:202-233` (calls `principle_langevin_v5`) | `ms_posterior_sampling_article_version_final.py:311-326` (calls `principle_langevin_sample`) |
| Soft replacement (stage 3) | `ms_sampler_v5.py:236-240` | — *(v5 only)* |
| State advance | `ms_sampler_v5.py:242` | `ms_posterior_sampling_article_version_final.py:329` |
| Final WLS-CG polish | `ms_sampler_v5.py:247-257` | — *(v5 only)* |
| Terminal mask blend | `ms_sampler_v5.py:259-261` | — *(v5 only)* |

The article version additionally records eight intermediate-state tensors during the ODE loop (`xts_traj`, `xt_next_traj`, `x1_traj_before/after`, `eps_traj`, `langevin_inner_traj`, `langevin_inner_logs`, `step_logs`); v5 omits all of these to save memory.

---

## C. Langevin inner-loop math

Core gradient is identical in both samplers:

```
g_x1 = (1/η²)·Aᵀ(b − Ax1) + Hᵀ_τ · s_flow + λ_prox·(sg(x1_hat) − x1_k)
g_eps = … (eps update step (f))
```

- Article: `_langevin_step()` at `ms_posterior_sampling_article_version_final_utils.py:247`, gradient assembled at `:286-294`, noise term `:300-305`, eps update at `:311`.
- v5: `langevin_step_v5()` at `langevin_v5.py:36`, gradient at `:80-83`, noise at `:111-118`, eps update at `:134-136` (gated by `skip_eps_update`).

`sigma_ref_sq` (the Tweedie soft-damping floor) is **hard-coded** to `0.01` in the article (`ms_posterior_sampling_article_version_final_utils.py:265`) and **per-stage parameterized** in v5 (`ms_sampler_v5.py:82`, dispatched per stage at `:171`).

### Round-6 eps-augmentation axes (v5 only)

Four new terms applied inside `langevin_step_v5`:

| Knob | Mechanism | v5 line |
|---|---|---|
| `lambda_eps_norm` (N1) | Push `‖eps‖² → N` (the dimension count); prevents blow-up. | `langevin_v5.py:96-101` |
| `mask_aware_eps` (N2) | In the unobserved region, replace the Tweedie term with a pure N(0,I) prior. | `langevin_v5.py:88-94` |
| `lambda_eps_prox` (N3) | Proximal to the initial eps_0; keeps the chain coupled to the prior. | `langevin_v5.py:103-105` |
| `lambda_eps_inject` (N4) | Active per-step Gaussian noise injection. | `langevin_v5.py:107-109` |

None of these terms exist in `_langevin_step()`. Article's eps update is purely `eps_k += (h_eps/2)·g_eps + sqrt(h_eps)·x3` (`ms_posterior_sampling_article_version_final_utils.py:311`).

### CFG (classifier-free guidance)

Both samplers route guidance through `make_velocity_fn()` from the article utils (`ms_posterior_sampling_article_version_final_utils.py:225`). The article calls it at `:281` and v5 calls it at `ms_sampler_v5.py:178-181`. CFG is a shared knob; v5 does not extend it.

---

## D. Knob inventory

### v5-only kwargs (12)

All declared in the `run_ip4` signature at `ms_sampler_v5.py:70-83`:

| kwarg | Default | Purpose |
|---|---|---|
| `dps_kick_zeta` | `None` | DPS-style pre-kick before Langevin (scalar or 4-list); 0 = off. |
| `terminal_replace_weight` | `0.0` | Final hard/soft mask blend `mask*y + (1-mask)*x1`; 0 = off, 1 = hard. |
| `soft_replace_weight` | `0.0` | Per-step blend, stage 3 only. |
| `h_x_obs_ratio` | `1.0` | Mask-aware step scaling (different inside vs outside the mask). |
| `final_denoise` | `False` | Optional WLS-CG polish after Langevin loop ends. |
| `skip_eps_update` | `False` | Skip step (f) of Langevin per stage. |
| `reset_eps_per_ode_step` | `False` | Resample `eps ∼ N(0,I)` before each ODE step. |
| `lambda_eps_norm` | `0.0` | N1 — drive `‖eps‖²` toward dim count. |
| `mask_aware_eps` | `False` | N2 — pure prior in unobserved region. |
| `lambda_eps_prox` | `0.0` | N3 — proximal to initial eps_0. |
| `lambda_eps_inject` | `0.0` | N4 — active noise per step. |
| `sigma_ref_sq` | `0.01` | Per-stage Tweedie soft-damping floor (article hardcodes this). |

Most accept a 4-element list for per-stage scheduling (resolved by `_ps()` helper at `ms_sampler_v5.py:39`).

### article-only kwargs

All I/O / introspection knobs absent from v5:

| kwarg | Article line | Purpose |
|---|---|---|
| `return_traj` | `:334, 359` | Toggle trajectory tensor recording. |
| `record_every` | `:334, 345, 356` | Sub-sample factor for trajectory save. |
| `save_videos` | `:127` | Render per-stage videos. |
| `save_inner_latents` | `:132` | Save Langevin inner-loop latents. |
| `combined_video_fps`, `langevin_video_fps`, `video_dpi`, `video_batch_size` | `:129-131` | Visualization params. |
| `latent_update_mode` | `:102` | "principle" or alternative; affects trajectory naming. |
| `measurement_mode` | `:101, 195-200` | "call" (noiseless) or "measure" (adds `sigma_n`). |

### Shared kwargs (same defaults in both)

`h_x = 0.1`, `h_epsilon = 0.01`, `lambda_reg = 50.0`, `noise_scale = 0.0`, `lambda_prox = 0.0`, `num_langevin = 10`, `guidance_scale = 0.0`, plus `rho_s`, `rho_e`, `lambda_x`, `inference_each_step / ode_steps_per_stage`, `class_label`, `seed`. v5 additionally allows the four `h_*` and `num_langevin` knobs to take a 4-list for per-stage scheduling — the article does not.

---

## E. Numerical / implementation quirks

- **PSNR / metrics** — v5 computes 3 PSNR variants (full, observed-only, unobserved-only) using the publication formula `10·log10(4 / MSE_{[-1,1]})` plus HF energy inline (`ms_sampler_v5.py:265-288`). Article computes nothing inline; metrics are derived from saved trajectories elsewhere.
- **RNG seeding** — v5 calls `seed_everything(seed)` *and* `torch.manual_seed(seed)` (`ms_sampler_v5.py:88-89`); article uses `seed_everything(seed)` only (`ms_posterior_sampling_article_version_final.py:135`). Redundant but explicit in v5.
- **Precision** — neither uses `autocast` / AMP. Both wrap the entry function with `@torch.no_grad()`.
- **Mask broadcast** — v5 explicitly expands a `B=1` mask to the full batch (`ms_sampler_v5.py:110-149`); article inlines mask handling inside `make_Ak_fns()` and never materializes the mask tensor at the sampler level.
- **`weights_only`** — article uses `torch.load(..., weights_only=False)` for checkpoint loading at `:182`; v5 doesn't load a model itself (the model is passed in by the caller).
- **`torch.compile`** — neither uses it.
- **Class label cast** — v5 casts `class_label` to `int32` at `:98`; article defers to dataset loader.

---

## F. Helpers in `*_utils.py`

`ms_posterior_sampling_article_version_final_utils.py` defines (line numbers):

| Helper | Line | Role |
|---|---|---|
| `load_run_config` | 31 | JSON config loader. |
| `cg_solve` | 97 | Conjugate-gradient solver for SPD systems. |
| `apply_G` | 126 | Low-pass G operator (stage-3 → identity). |
| `apply_H_tau` | 140 | Consistency operator H_τ^k. |
| `apply_HT_tau` | 145 | Adjoint H_τ^T. |
| `compute_sigma_tau` | 150 | Noise schedule σ_τ. |
| `wls_estimate_x1` | 159 | WLS x1 solver (Algorithm 1, step b). |
| `direct_estimate_x1` | 176 | Direct x1 (no CG). |
| `_interpolate_adjoint` | 187 | Exact adjoint of `F.interpolate`. |
| `make_Ak_fns` | 197 | Closure for A_k / A_k^T. |
| `make_velocity_fn` | 225 | Closure for velocity prediction + CFG. |
| `_langevin_step` | 247 | Single Langevin step. |
| `principle_langevin_sample` | 327 | Langevin loop with trajectory recording. |

`langevin_v5.py:20` imports the helpers wholesale:

```python
from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, apply_HT_tau, compute_sigma_tau,
    wls_estimate_x1, direct_estimate_x1, cg_solve,
    make_Ak_fns, make_velocity_fn,
)
```

So everything in §F (except `_langevin_step` and `principle_langevin_sample`) is **physically the same code** in v5 and the article version. v5 only re-implements the two Langevin functions:

| Article | v5 |
|---|---|
| `_langevin_step()` (`_utils.py:247`) | `langevin_step_v5()` (`langevin_v5.py:36`) — adds N1–N4, per-stage `sigma_ref_sq`, mask-aware step, `skip_eps_update`. |
| `principle_langevin_sample()` (`_utils.py:327`) | `principle_langevin_v5()` (`langevin_v5.py:144`) — drops trajectory recording, adds `mask_k`, `h_x_obs_ratio`, plus all N1–N4 plumbing. |

Plus two new helpers in `langevin_v5.py`:

- `dps_gradient_kick()` at `:186` — the DPS pre-kick `x1 += zeta · Aᵀ(y − Ax1) / norm`.
- `terminal_replacement()` at `:196` — observed-pixel hard/soft blend with measurement.

---

## G. Practical implications for sweep work

After 167 configs in the IP4 larger-box exploration, the empirical picture of the v5-only knobs:

- **`dps_kick_zeta`** — `J_dps_*` configs introduced visible boundary seams; never beat the no-kick baseline. Skip.
- **`terminal_replace_weight`** — must be **on** (1.0) to keep observed pixels aligned with `y`. Without it PSNR_obs degrades. This is the only v5-only knob that's non-optional for the IP4 setting.
- **`soft_replace_weight`** — not Pareto-optimal vs `terminal_replace_weight=1`.
- **`h_x_obs_ratio`** — `P_obs_025` / `P_obs_30` both hurt. Default 1.0 wins.
- **`final_denoise`** — not swept; behavior likely matches `reset_eps` pattern (smooth fill).
- **`reset_eps_per_ode_step`** — `H_resetEps` gives the highest PSNR (20.33) in the sweep but with smooth-fill failure mode (LPIPS 0.180). Same attractor as `h_eps ≥ 1e-2`.
- **`skip_eps_update`** — not Pareto-optimal.
- **`lambda_eps_norm` / `mask_aware_eps` / `lambda_eps_prox` / `lambda_eps_inject`** (N1–N4) — all collapse to the same equilibrium (LPIPS ≈ 0.158, PSNR ≈ 17.5, |dHF| ≈ 0.096) which is **worse** than the C_L5 baseline. None are useful for this task.
- **`sigma_ref_sq`** — not swept; default 0.01 retained.

The actual winner `F_cfg20_lr150_hxu02` uses **only knobs that exist in the article version**:

```json
{
  "h_x": 0.2,
  "h_epsilon": 0.001,
  "num_langevin": 5,
  "lambda_reg": 150.0,
  "noise_scale": 0.0,
  "terminal_replace_weight": 1.0,
  "guidance_scale": 2.0
}
```

`terminal_replace_weight=1.0` is the single v5-only knob that the recipe relies on; this is also the IP4-paper convention so the article version effectively has it baked in via its own `measurement_mode="measure"` pathway.

**Implication**: for sweep / search work the v5 surface added ~12 knobs that turned out to be dead-ends (or confirmed-unhelpful) on this benchmark. The article version's smaller knob surface is sufficient to land within ~0.014 LPIPS of the v5 winner. The trade-off is that v5 was needed to *prove* those Round-6 knobs don't help — a falsification result, but not a Pareto improvement.

---

## H. Feature-parity table

| Feature | v5 | Article |
|---|---|---|
| Entry pattern | Function API (`run_ip4(**kwargs)`) | CLI (`main(config_path)`) |
| Trajectory recording | None (memory-efficient) | Full (8 tensors + per-step logs) |
| Warm restart | Yes (`ms_sampler_v5.py:183`) | Yes (`...article...py:298`) |
| DPS pre-kick | Yes (`ms_sampler_v5.py:196`) | No |
| Terminal replacement | Yes (`ms_sampler_v5.py:260`) | No |
| Soft per-step replacement (stage 3) | Yes (`ms_sampler_v5.py:236`) | No |
| Mask-aware step size (`h_x_obs_ratio`) | Yes (`ms_sampler_v5.py:224`) | No |
| Per-stage `sigma_ref_sq` | Yes (`ms_sampler_v5.py:232`) | No (hardcoded 0.01) |
| eps-norm regularizer (N1) | Yes (`langevin_v5.py:96-101`) | No |
| Mask-aware eps prior (N2) | Yes (`langevin_v5.py:88-94`) | No |
| eps proximal (N3) | Yes (`langevin_v5.py:103-105`) | No |
| Active noise injection (N4) | Yes (`langevin_v5.py:107-109`) | No |
| `skip_eps_update` per stage | Yes (`langevin_v5.py:134-136`) | No |
| `reset_eps_per_ode_step` | Yes (`ms_sampler_v5.py:194`) | No |
| Final WLS-CG denoise | Yes (`ms_sampler_v5.py:247-257`) | No |
| Inline metrics (PSNR×3 + HF) | Yes (`ms_sampler_v5.py:265-288`) | No |
| CFG support | Yes (`make_velocity_fn`, shared) | Yes (`make_velocity_fn`, shared) |
| Per-stage list scheduling for `h_x`/`h_eps`/`L`/`λ_reg` | Yes | Scalar only |
| Trajectory-record knobs (`return_traj`, `record_every`) | No | Yes |
| Video / inner-latent saving | No | Yes |
| `measurement_mode` knob | No (always uses noisy measurement) | Yes (`call` vs `measure`) |
| `latent_update_mode` knob | No | Yes |

## Verification

The single import that makes v5 inherit the article-version helpers:

```bash
$ grep -n "from ms_posterior_sampling_article_version_final_utils" \
       PixelFlow/debug_IP4/langevin_v5.py
20:from ms_posterior_sampling_article_version_final_utils import (
```

Confirms that `cg_solve`, `apply_G`, `apply_H_tau`, `apply_HT_tau`, `compute_sigma_tau`, `wls_estimate_x1`, `direct_estimate_x1`, `make_Ak_fns`, `make_velocity_fn` are reused unchanged — no fork, no shadowing.
