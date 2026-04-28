"""Exploration sweep around the larger_box LPIPS anchors.

Anchors from larger_box/results/WINNERS.md (128x128 box, res=256, 2-img GT):
  C_L5     : h_eps=1e-3, L=5,  h_x=[.1,.1,.1,.7]   LPIPS=0.1295 PSNR=17.51 |dHF|=0.011
  C_L10    : h_eps=1e-3, L=10, h_x=[.1,.1,.1,.7]   LPIPS=0.1252 PSNR=16.60 |dHF|=0.041
  A_he2e-3 : h_eps=2e-3, L=10, h_x=[.1,.1,.1,.7]   LPIPS=0.1352 PSNR=19.20

We target the BALANCED corner (semantic content + acceptable PSNR + not grainy).
Earlier sweep covered h_epsilon and num_langevin only. Here we add the other axes
left untouched: noise_scale, lambda_reg, h_x shape, lambda_prox, eps refresh,
ode_steps_per_stage, dps_kick, structural ablations. Plus the IP4 ref_baseline_F1
re-run in the 128x128 setting (it was previously only evaluated on the 64x64 box).
"""

# Perceptual base = WINNER3 = C_L5 anchor minus the swept axis
BASE_C5 = dict(
    h_x=[0.1, 0.1, 0.1, 0.7],
    h_epsilon=1e-3,
    num_langevin=5,
    lambda_reg=50.0,
    noise_scale=0.0,
    terminal_replace_weight=1.0,
)

BASE_C10 = dict(BASE_C5); BASE_C10["num_langevin"] = 10
BASE_A2e3 = dict(BASE_C5); BASE_A2e3["h_epsilon"] = 2e-3; BASE_A2e3["num_langevin"] = 10


def _cfg(base, extra):
    out = dict(base)
    out.update(extra)
    return out


# ---- Group E: noise_scale -----------------------------------------------
GROUP_E = [
    ("E_N01_C5",   _cfg(BASE_C5,   dict(noise_scale=0.1))),
    ("E_N03_C5",   _cfg(BASE_C5,   dict(noise_scale=0.3))),
    ("E_N01_C10",  _cfg(BASE_C10,  dict(noise_scale=0.1))),
    ("E_N03_C10",  _cfg(BASE_C10,  dict(noise_scale=0.3))),
    ("E_N01_A2e3", _cfg(BASE_A2e3, dict(noise_scale=0.1))),
]

# ---- Group F: lambda_reg around the default 50 --------------------------
GROUP_F = [
    ("F_lr25_C5",  _cfg(BASE_C5, dict(lambda_reg=25.0))),
    ("F_lr100_C5", _cfg(BASE_C5, dict(lambda_reg=100.0))),
    ("F_lr200_C5", _cfg(BASE_C5, dict(lambda_reg=200.0))),
]

# ---- Group G: h_x shape (existing uses [.1,.1,.1,.7]) -------------------
GROUP_G = [
    ("G_hx_lo",    _cfg(BASE_C5, dict(h_x=[0.1, 0.1, 0.1, 0.4]))),
    ("G_hx_hi",    _cfg(BASE_C5, dict(h_x=[0.1, 0.1, 0.1, 1.0]))),
    ("G_hx_uni3",  _cfg(BASE_C5, dict(h_x=0.3))),
    ("G_hx_late2", _cfg(BASE_C5, dict(h_x=[0.1, 0.1, 0.3, 0.7]))),
]

# ---- Group H: lambda_prox / eps strategies (untouched axes) -------------
GROUP_H = [
    ("H_lprox01",  _cfg(BASE_C5, dict(lambda_prox=0.1))),
    ("H_lprox05",  _cfg(BASE_C5, dict(lambda_prox=0.5))),
    ("H_resetEps", _cfg(BASE_C5, dict(reset_eps_per_ode_step=True))),
    ("H_einject",  _cfg(BASE_C5, dict(lambda_eps_inject=0.1))),
]

# ---- Group I: ref_baseline_F1 in 128x128 setting ------------------------
# Defaults (per ms_sampler_v5 signature): h_x=0.1, h_epsilon=0.01, L=10,
# lambda_reg=50, noise_scale=0.0. + terminal_replace_weight=1.0.
GROUP_I = [
    ("I_ref_F1", dict(
        h_x=0.1,
        h_epsilon=0.01,
        num_langevin=10,
        lambda_reg=50.0,
        noise_scale=0.0,
        terminal_replace_weight=1.0,
    )),
]

# ---- Group J: dps_kick on C_L5 base -------------------------------------
GROUP_J = [
    ("J_dps_s3_2", _cfg(BASE_C5, dict(dps_kick_zeta=[0, 0, 0, 2.0]))),
    ("J_dps_all5", _cfg(BASE_C5, dict(dps_kick_zeta=5.0))),
]

# ---- Group K: more ODE steps at stage-3 ---------------------------------
GROUP_K = [
    ("K_ode_s3_15", _cfg(BASE_C5, dict(ode_steps_per_stage=[10, 10, 10, 15]))),
    ("K_ode_s3_20", _cfg(BASE_C5, dict(ode_steps_per_stage=[10, 10, 10, 20]))),
]

# ---- Group L: structural ablations on C_L5 ------------------------------
GROUP_L = [
    ("L_no_warm",   _cfg(BASE_C5, dict(warm_restart=False))),
    ("L_no_bypass", _cfg(BASE_C5, dict(g_bypass_stage3=False))),
]

# Sanity re-run of the existing winner so a seeded re-run has a control.
GROUP_M = [
    ("M_C5_repeat",  dict(BASE_C5)),
    ("M_C10_repeat", dict(BASE_C10)),
]

# ---- Group N: parameter COMBINATIONS (multi-axis) -----------------------
# Anchor = C_L5. We pair noise_scale, lambda_reg, h_x shape, ode_steps,
# lambda_prox, h_x_obs_ratio, x1_init_mode, guidance_scale and per-stage
# h_epsilon / num_langevin schedules. 20 configs.
GROUP_N = [
    # 2-axis: noise_scale × lambda_reg
    ("N_n01_lr100",        _cfg(BASE_C5, dict(noise_scale=0.1, lambda_reg=100.0))),
    ("N_n03_lr100",        _cfg(BASE_C5, dict(noise_scale=0.3, lambda_reg=100.0))),
    ("N_n01_lr200",        _cfg(BASE_C5, dict(noise_scale=0.1, lambda_reg=200.0))),
    # 2-axis: noise_scale × h_x shape
    ("N_n01_hx1",          _cfg(BASE_C5, dict(noise_scale=0.1, h_x=[0.1, 0.1, 0.1, 1.0]))),
    ("N_n01_hxlate",       _cfg(BASE_C5, dict(noise_scale=0.1, h_x=[0.1, 0.1, 0.3, 0.7]))),
    # 2-axis: noise_scale × ode_steps
    ("N_n01_ode15",        _cfg(BASE_C5, dict(noise_scale=0.1, ode_steps_per_stage=[10, 10, 10, 15]))),
    # 2-axis: lambda_reg × h_x shape
    ("N_lr100_hx1",        _cfg(BASE_C5, dict(lambda_reg=100.0, h_x=[0.1, 0.1, 0.1, 1.0]))),
    ("N_lr200_hxlo",       _cfg(BASE_C5, dict(lambda_reg=200.0, h_x=[0.1, 0.1, 0.1, 0.4]))),
    # 2-axis: lambda_prox × noise_scale
    ("N_lp01_n01",         _cfg(BASE_C5, dict(lambda_prox=0.1, noise_scale=0.1))),
    # 2-axis: h_x_obs_ratio (NEW axis — never tested)
    ("N_obs_lo",           _cfg(BASE_C5, dict(h_x_obs_ratio=0.5))),
    ("N_obs_hi",           _cfg(BASE_C5, dict(h_x_obs_ratio=2.0))),
    # x1_init_mode = wls (instead of "model")
    ("N_wls",              _cfg(BASE_C5, dict(x1_init_mode="wls"))),
    # CFG (guidance_scale)
    ("N_cfg2",             _cfg(BASE_C5, dict(guidance_scale=2.0))),
    ("N_cfg2_n01",         _cfg(BASE_C5, dict(guidance_scale=2.0, noise_scale=0.1))),
    # 3-axis combos
    ("N_n01_lr100_hx05",   _cfg(BASE_C5, dict(noise_scale=0.1, lambda_reg=100.0, h_x=[0.1, 0.1, 0.1, 0.5]))),
    ("N_n03_lr100_ode15",  _cfg(BASE_C5, dict(noise_scale=0.3, lambda_reg=100.0,
                                              ode_steps_per_stage=[10, 10, 10, 15]))),
    # h_eps schedule + L combos
    ("N_he_dec_L5",        _cfg(BASE_C5, dict(h_epsilon=[0.005, 0.002, 0.001, 0.001]))),
    ("N_he_inc_n01",       _cfg(BASE_C5, dict(h_epsilon=[0.001, 0.001, 0.002, 0.005], noise_scale=0.1))),
    # Per-stage L
    ("N_L_5510",           _cfg(BASE_C5, dict(num_langevin=[5, 5, 5, 10]))),
    ("N_L_3_5_5_10",       _cfg(BASE_C5, dict(num_langevin=[3, 5, 5, 10]))),
]

CONFIGS = (
    GROUP_E + GROUP_F + GROUP_G + GROUP_H +
    GROUP_I + GROUP_J + GROUP_K + GROUP_L +
    GROUP_M + GROUP_N
)

if __name__ == "__main__":
    print(f"Total configs: {len(CONFIGS)}")
    for name, kw in CONFIGS:
        print(f"  {name:<14s}  {kw}")
