"""18 OAT configs (anchor + 17 single-axis variants) for random_inpainting.

Anchor (defaults from box winners + sensible PixelFlow defaults):
  h_epsilon=0.01, num_langevin=10, h_x=0.1, sigma_ref_sq=1e-4,
  final_denoise=False, terminal_replace_weight=1.0, noise_scale=1.0

OAT axes (each tested keeping others at anchor):
  h_epsilon: 0.1, 0.01, 0.001, 0.0001
  num_langevin: 1, 5, 10, 15, 20
  h_x: 0.05, 0.1, 0.2, 0.3, 0.35
  sigma_ref_sq: 1e-2, 1e-3, 1e-4, 1e-5
  final_denoise: 0, 1
  terminal_replace_weight: 0, 1
  noise_scale: 0, 1
"""

ANCHOR = dict(
    num_langevin=10,
    h_x=0.1,
    h_epsilon=0.01,
    lambda_reg=150.0,
    noise_scale=1.0,
    ode_steps_per_stage=10,
    shift=1.0,
    guidance_scale=2.0,
    warm_restart=True,
    g_bypass_stage3=True,
    x1_init_mode="model",
    lambda_x=0.01,
    rho_s=1.0, rho_e=1.0,
    cg_tol=1e-5, cg_max_iter=50,
    dps_kick_zeta=None,
    terminal_replace_weight=1.0,
    soft_replace_weight=0.0,
    h_x_obs_ratio=1.0,
    lambda_prox=150.0,
    skip_eps_update=False,
    reset_eps_per_ode_step=False,
    lambda_eps_norm=0.0,
    mask_aware_eps=False,
    lambda_eps_prox=0.0, lambda_eps_inject=0.0,
    sigma_ref_sq=1e-4,
    final_denoise=False,
)

OAT_AXES = [
    ("h_epsilon",               [0.1, 0.01, 0.001, 0.0001]),
    ("num_langevin",            [1, 5, 10, 15, 20]),
    ("h_x",                     [0.05, 0.1, 0.2, 0.3, 0.35]),
    ("sigma_ref_sq",            [1e-2, 1e-3, 1e-4, 1e-5]),
    ("final_denoise",           [False, True]),
    ("terminal_replace_weight", [0.0, 1.0]),
    ("noise_scale",             [0.0, 1.0]),
]


def _fmt(v):
    if isinstance(v, bool):  return "T" if v else "F"
    if isinstance(v, int):   return str(v)
    if isinstance(v, float):
        if v == 0.0: return "0"
        if abs(v) >= 0.001 and abs(v) < 1000:
            return f"{v:g}"
        return f"{v:.0e}".replace("e-0", "e-")
    return str(v)


def _build():
    seen = set()
    out = []
    # 1) anchor
    out.append(("anchor", dict(ANCHOR)))
    seen.add(_key(ANCHOR))
    # 2) for each axis, set value and emit if new
    for axis, vals in OAT_AXES:
        for v in vals:
            kw = dict(ANCHOR)
            kw[axis] = v
            k = _key(kw)
            if k in seen:
                continue
            seen.add(k)
            short = _fmt(v)
            tag = f"oat_{axis}={short}"
            out.append((tag, kw))
    return out


def _key(kw):
    return tuple(sorted(kw.items(), key=lambda x: x[0]))


CONFIGS = _build()


if __name__ == "__main__":
    print(f"Total: {len(CONFIGS)}")
    for n, kw in CONFIGS:
        print(f"  {n:<35s}  h_eps={kw['h_epsilon']}  L={kw['num_langevin']}  "
              f"h_x={kw['h_x']}  srs={kw['sigma_ref_sq']:.0e}  "
              f"fd={kw['final_denoise']}  tr={kw['terminal_replace_weight']}  "
              f"ns={kw['noise_scale']}")
