"""Superresolution-4x parameter sweep — anchor + single-axis + targeted combos.

Anchor (article default extended for SR):
  h_x = 0.1, h_epsilon = 0.01, num_langevin = 10, lambda_reg = 50,
  noise_scale = 0, terminal_replace_weight = 0, sigma_ref_sq = 0.01,
  final_denoise = False, guidance_scale = 0.

User axes (cartesian = 3200, infeasible). Strategy:
  * 7 single-axis sweeps anchored on the default → ~22 configs
  * Targeted multi-axis combos around the most promising regions → ~32 configs

Total ~55 configs. ~3 min per config × 55 / 8 GPUs ≈ 25 min wall.
"""

ANCHOR = dict(
    h_x=0.1,
    h_epsilon=0.01,
    num_langevin=10,
    lambda_reg=50.0,
    noise_scale=0.0,
    terminal_replace_weight=0.0,
    sigma_ref_sq=0.01,
    final_denoise=False,
    guidance_scale=0.0,
)


def _cfg(extra):
    out = dict(ANCHOR); out.update(extra); return out


# ===== Single-axis sweeps =====

GROUP_TR = [   # terminal_replace_weight (anchor=0)
    ("S_tr0",            _cfg(dict(terminal_replace_weight=0.0))),
    ("S_tr1",            _cfg(dict(terminal_replace_weight=1.0))),
]

GROUP_HE = [   # h_epsilon (anchor=0.01)
    ("S_he1e-1",         _cfg(dict(h_epsilon=1e-1))),
    ("S_he1e-2",         _cfg(dict(h_epsilon=1e-2))),
    ("S_he1e-3",         _cfg(dict(h_epsilon=1e-3))),
    ("S_he1e-4",         _cfg(dict(h_epsilon=1e-4))),
]

GROUP_L = [    # num_langevin (anchor=10)
    ("S_L1",             _cfg(dict(num_langevin=1))),
    ("S_L5",             _cfg(dict(num_langevin=5))),
    ("S_L10",            _cfg(dict(num_langevin=10))),
    ("S_L15",            _cfg(dict(num_langevin=15))),
    ("S_L20",            _cfg(dict(num_langevin=20))),
]

GROUP_SRS = [  # sigma_ref_sq (anchor=1e-2)
    ("S_srs1e-2",        _cfg(dict(sigma_ref_sq=1e-2))),
    ("S_srs1e-3",        _cfg(dict(sigma_ref_sq=1e-3))),
    ("S_srs1e-4",        _cfg(dict(sigma_ref_sq=1e-4))),
    ("S_srs1e-5",        _cfg(dict(sigma_ref_sq=1e-5))),
]

GROUP_FD = [   # final_denoise (anchor=False)
    ("S_fd0",            _cfg(dict(final_denoise=False))),
    ("S_fd1",            _cfg(dict(final_denoise=True))),
]

GROUP_HX = [   # h_x (anchor=0.1)
    ("S_hx005",          _cfg(dict(h_x=0.05))),
    ("S_hx01",           _cfg(dict(h_x=0.10))),
    ("S_hx02",           _cfg(dict(h_x=0.20))),
    ("S_hx03",           _cfg(dict(h_x=0.30))),
    ("S_hx035",          _cfg(dict(h_x=0.35))),
]

GROUP_NS = [   # noise_scale (anchor=0)
    ("S_ns0",            _cfg(dict(noise_scale=0.0))),
    ("S_ns1",            _cfg(dict(noise_scale=1.0))),
]

STAGE_S = (GROUP_TR + GROUP_HE + GROUP_L + GROUP_SRS + GROUP_FD
           + GROUP_HX + GROUP_NS)


# ===== Targeted multi-axis combos =====
# Hypothesis: SR is much closer to a smoothing/denoising task than box inpainting.
# Lower h_eps + higher h_x usually balances perceptual vs distortion.
# tr=1 means: replace x1 with bicubic-upsampled y (data injection).
# Best combos to probe: (h_eps,L) × tr=1 / fd=1 / large h_x.

GROUP_C_he_L = [   # h_eps × L grid (3×3)
    ("C_he1e-2_L5",      _cfg(dict(h_epsilon=1e-2, num_langevin=5))),
    ("C_he1e-2_L15",     _cfg(dict(h_epsilon=1e-2, num_langevin=15))),
    ("C_he1e-3_L5",      _cfg(dict(h_epsilon=1e-3, num_langevin=5))),
    ("C_he1e-3_L10",     _cfg(dict(h_epsilon=1e-3, num_langevin=10))),
    ("C_he1e-3_L15",     _cfg(dict(h_epsilon=1e-3, num_langevin=15))),
    ("C_he1e-4_L10",     _cfg(dict(h_epsilon=1e-4, num_langevin=10))),
]

GROUP_C_he_hx = [   # h_eps × h_x grid (3×3)
    ("C_he1e-3_hx02",    _cfg(dict(h_epsilon=1e-3, h_x=0.20))),
    ("C_he1e-3_hx03",    _cfg(dict(h_epsilon=1e-3, h_x=0.30))),
    ("C_he1e-3_hx005",   _cfg(dict(h_epsilon=1e-3, h_x=0.05))),
    ("C_he1e-2_hx02",    _cfg(dict(h_epsilon=1e-2, h_x=0.20))),
    ("C_he1e-2_hx03",    _cfg(dict(h_epsilon=1e-2, h_x=0.30))),
    ("C_he1e-4_hx02",    _cfg(dict(h_epsilon=1e-4, h_x=0.20))),
]

GROUP_C_tr_X = [    # tr=1 × every other knob in motion
    ("C_tr1_he1e-3",     _cfg(dict(terminal_replace_weight=1.0, h_epsilon=1e-3))),
    ("C_tr1_he1e-3_hx02",_cfg(dict(terminal_replace_weight=1.0, h_epsilon=1e-3, h_x=0.20))),
    ("C_tr1_L15",        _cfg(dict(terminal_replace_weight=1.0, num_langevin=15))),
    ("C_tr1_fd1",        _cfg(dict(terminal_replace_weight=1.0, final_denoise=True))),
    ("C_tr1_srs1e-3",    _cfg(dict(terminal_replace_weight=1.0, sigma_ref_sq=1e-3))),
    ("C_tr1_hx03",       _cfg(dict(terminal_replace_weight=1.0, h_x=0.30))),
]

GROUP_C_fd_X = [
    ("C_fd1_he1e-3",     _cfg(dict(final_denoise=True, h_epsilon=1e-3))),
    ("C_fd1_he1e-3_hx02",_cfg(dict(final_denoise=True, h_epsilon=1e-3, h_x=0.20))),
    ("C_fd1_L15",        _cfg(dict(final_denoise=True, num_langevin=15))),
]

GROUP_C_srs = [     # σ_ref² combos with the strong perceptual axis
    ("C_srs1e-3_he1e-3", _cfg(dict(sigma_ref_sq=1e-3, h_epsilon=1e-3))),
    ("C_srs1e-4_he1e-3", _cfg(dict(sigma_ref_sq=1e-4, h_epsilon=1e-3))),
    ("C_srs1e-3_L15",    _cfg(dict(sigma_ref_sq=1e-3, num_langevin=15))),
]

GROUP_C_ns = [      # noise_scale=1 alongside tr=1 / fd=1 / strong perceptual
    ("C_ns1_tr1",        _cfg(dict(noise_scale=1.0, terminal_replace_weight=1.0))),
    ("C_ns1_he1e-3",     _cfg(dict(noise_scale=1.0, h_epsilon=1e-3))),
]

GROUP_C_3way = [    # 3-axis combos: best perceptual recipe candidates
    ("C_he1e-3_hx02_L15",      _cfg(dict(h_epsilon=1e-3, h_x=0.20, num_langevin=15))),
    ("C_he1e-3_hx03_L15",      _cfg(dict(h_epsilon=1e-3, h_x=0.30, num_langevin=15))),
    ("C_he1e-3_hx02_tr1",      _cfg(dict(h_epsilon=1e-3, h_x=0.20, terminal_replace_weight=1.0))),
    ("C_he1e-3_hx03_tr1_fd1",  _cfg(dict(h_epsilon=1e-3, h_x=0.30, terminal_replace_weight=1.0, final_denoise=True))),
    ("C_he1e-3_hx02_srs1e-3",  _cfg(dict(h_epsilon=1e-3, h_x=0.20, sigma_ref_sq=1e-3))),
    ("C_he1e-3_L15_srs1e-3",   _cfg(dict(h_epsilon=1e-3, num_langevin=15, sigma_ref_sq=1e-3))),
]

STAGE_C = (GROUP_C_he_L + GROUP_C_he_hx + GROUP_C_tr_X + GROUP_C_fd_X
           + GROUP_C_srs + GROUP_C_ns + GROUP_C_3way)


CONFIGS = STAGE_S + STAGE_C

if __name__ == "__main__":
    print(f"Total configs: {len(CONFIGS)}")
    seen = set()
    for n, kw in CONFIGS:
        if n in seen: print(f"  ! duplicate name: {n}")
        seen.add(n)
        print(f"  {n:<30s}  he={kw['h_epsilon']:.0e}  L={kw['num_langevin']:>2d}  "
              f"hx={kw['h_x']:.2f}  srs={kw['sigma_ref_sq']:.0e}  tr={kw['terminal_replace_weight']:.0f}  "
              f"fd={int(kw['final_denoise'])}  ns={kw['noise_scale']:.1f}")
