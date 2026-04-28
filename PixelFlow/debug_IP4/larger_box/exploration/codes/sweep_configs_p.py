"""Deep parameter sweep — Stage P (single-axis) + Stage Q (multi-axis) anchored
on N_cfg2 (the current winner). 75 configs.

For each parameter listed in the task spec, we sweep 5-8 values around the
default. Then we run 2-3 axis combinations on the most promising axes.
Goal: produce per-axis curves showing how each knob affects perceptual
(LPIPS) vs. PSNR vs. |dHF| relative to the N_cfg2 anchor.

Anchor (N_cfg2):
    h_x = [0.1, 0.1, 0.1, 0.7]
    h_epsilon = 1e-3
    num_langevin = 5
    lambda_reg = 50
    noise_scale = 0
    terminal_replace_weight = 1.0
    guidance_scale = 2.0
"""

ANCHOR = dict(
    h_x=[0.1, 0.1, 0.1, 0.7],
    h_epsilon=1e-3,
    num_langevin=5,
    lambda_reg=50.0,
    noise_scale=0.0,
    terminal_replace_weight=1.0,
    guidance_scale=2.0,
)


def _cfg(extra):
    out = dict(ANCHOR)
    out.update(extra)
    return out


# ===== Stage P — single-axis fine sweeps =====

# P_he: h_epsilon fine sweep
GROUP_P_he = [
    ("P_he_3e-4",  _cfg(dict(h_epsilon=3e-4))),
    ("P_he_5e-4",  _cfg(dict(h_epsilon=5e-4))),
    ("P_he_7e-4",  _cfg(dict(h_epsilon=7e-4))),
    ("P_he_1.5e-3",_cfg(dict(h_epsilon=1.5e-3))),
    ("P_he_2e-3",  _cfg(dict(h_epsilon=2e-3))),
    ("P_he_3e-3",  _cfg(dict(h_epsilon=3e-3))),
    ("P_he_5e-3",  _cfg(dict(h_epsilon=5e-3))),
]

# P_L: num_langevin fine sweep (existing covers L=1,3,5,10,15,20,30)
GROUP_P_L = [
    ("P_L2",  _cfg(dict(num_langevin=2))),
    ("P_L4",  _cfg(dict(num_langevin=4))),
    ("P_L7",  _cfg(dict(num_langevin=7))),
    ("P_L8",  _cfg(dict(num_langevin=8))),
]

# P_hx: h_x stage-3 magnitude
GROUP_P_hx = [
    ("P_hx_s3_00",  _cfg(dict(h_x=[0.1, 0.1, 0.1, 0.0]))),
    ("P_hx_s3_02",  _cfg(dict(h_x=[0.1, 0.1, 0.1, 0.2]))),
    ("P_hx_s3_05",  _cfg(dict(h_x=[0.1, 0.1, 0.1, 0.5]))),
    ("P_hx_s3_09",  _cfg(dict(h_x=[0.1, 0.1, 0.1, 0.9]))),
    ("P_hx_s3_12",  _cfg(dict(h_x=[0.1, 0.1, 0.1, 1.2]))),
]

# P_hxu: h_x uniform sweep
GROUP_P_hxu = [
    ("P_hxu_005", _cfg(dict(h_x=0.05))),
    ("P_hxu_01",  _cfg(dict(h_x=0.1))),
    ("P_hxu_02",  _cfg(dict(h_x=0.2))),
    ("P_hxu_04",  _cfg(dict(h_x=0.4))),
    ("P_hxu_06",  _cfg(dict(h_x=0.6))),
]

# P_lreg: lambda_reg fine sweep (extending F_lr)
GROUP_P_lreg = [
    ("P_lreg_10",  _cfg(dict(lambda_reg=10.0))),
    ("P_lreg_75",  _cfg(dict(lambda_reg=75.0))),
    ("P_lreg_150", _cfg(dict(lambda_reg=150.0))),
    ("P_lreg_300", _cfg(dict(lambda_reg=300.0))),
]

# P_cfg: guidance_scale fine sweep around the new winner CFG=2
GROUP_P_cfg = [
    ("P_cfg_0",   _cfg(dict(guidance_scale=0.0))),
    ("P_cfg_05",  _cfg(dict(guidance_scale=0.5))),
    ("P_cfg_10",  _cfg(dict(guidance_scale=1.0))),
    ("P_cfg_15",  _cfg(dict(guidance_scale=1.5))),
    ("P_cfg_25",  _cfg(dict(guidance_scale=2.5))),
    ("P_cfg_30",  _cfg(dict(guidance_scale=3.0))),
    ("P_cfg_40",  _cfg(dict(guidance_scale=4.0))),
    ("P_cfg_60",  _cfg(dict(guidance_scale=6.0))),
]

# P_lp: lambda_prox big values (small values saturated; need to push up)
GROUP_P_lp = [
    ("P_lp_005", _cfg(dict(lambda_prox=0.05))),
    ("P_lp_2",   _cfg(dict(lambda_prox=2.0))),
    ("P_lp_5",   _cfg(dict(lambda_prox=5.0))),
    ("P_lp_20",  _cfg(dict(lambda_prox=20.0))),
    ("P_lp_100", _cfg(dict(lambda_prox=100.0))),
]

# P_lx: lambda_x (default 0.01 — totally untouched)
GROUP_P_lx = [
    ("P_lx_0001", _cfg(dict(lambda_x=0.001))),
    ("P_lx_005",  _cfg(dict(lambda_x=0.05))),
    ("P_lx_01",   _cfg(dict(lambda_x=0.1))),
    ("P_lx_05",   _cfg(dict(lambda_x=0.5))),
    ("P_lx_10",   _cfg(dict(lambda_x=1.0))),
    ("P_lx_50",   _cfg(dict(lambda_x=5.0))),
]

# P_rho: rho_s, rho_e endpoint weighting (default 1.0, 1.0)
GROUP_P_rho = [
    ("P_rho_05_05", _cfg(dict(rho_s=0.5, rho_e=0.5))),
    ("P_rho_05_1",  _cfg(dict(rho_s=0.5, rho_e=1.0))),
    ("P_rho_1_05",  _cfg(dict(rho_s=1.0, rho_e=0.5))),
    ("P_rho_1_2",   _cfg(dict(rho_s=1.0, rho_e=2.0))),
    ("P_rho_2_1",   _cfg(dict(rho_s=2.0, rho_e=1.0))),
    ("P_rho_2_2",   _cfg(dict(rho_s=2.0, rho_e=2.0))),
]

# P_ode: ode_steps_per_stage scalar
GROUP_P_ode = [
    ("P_ode_5",   _cfg(dict(ode_steps_per_stage=5))),
    ("P_ode_8",   _cfg(dict(ode_steps_per_stage=8))),
    ("P_ode_12",  _cfg(dict(ode_steps_per_stage=12))),
]

# P_obs: h_x_obs_ratio (default 1, scaling step size in observed region)
GROUP_P_obs = [
    ("P_obs_025", _cfg(dict(h_x_obs_ratio=0.25))),
    ("P_obs_30",  _cfg(dict(h_x_obs_ratio=3.0))),
]

STAGE_P = (GROUP_P_he + GROUP_P_L + GROUP_P_hx + GROUP_P_hxu + GROUP_P_lreg
           + GROUP_P_cfg + GROUP_P_lp + GROUP_P_lx + GROUP_P_rho + GROUP_P_ode
           + GROUP_P_obs)

# ===== Stage Q — multi-axis combinations =====

# Q_cfg_x_hx: best new axis (CFG) × h_x stage-3 magnitude
GROUP_Q_cfg_hx = [
    ("Q_cfg15_hx04", _cfg(dict(guidance_scale=1.5, h_x=[0.1, 0.1, 0.1, 0.4]))),
    ("Q_cfg15_hx05", _cfg(dict(guidance_scale=1.5, h_x=[0.1, 0.1, 0.1, 0.5]))),
    ("Q_cfg25_hx04", _cfg(dict(guidance_scale=2.5, h_x=[0.1, 0.1, 0.1, 0.4]))),
    ("Q_cfg25_hx05", _cfg(dict(guidance_scale=2.5, h_x=[0.1, 0.1, 0.1, 0.5]))),
    ("Q_cfg30_hx05", _cfg(dict(guidance_scale=3.0, h_x=[0.1, 0.1, 0.1, 0.5]))),
]

# Q_cfg_x_lreg: CFG × lambda_reg
GROUP_Q_cfg_lreg = [
    ("Q_cfg2_lreg25",  _cfg(dict(guidance_scale=2.0, lambda_reg=25.0))),
    ("Q_cfg2_lreg75",  _cfg(dict(guidance_scale=2.0, lambda_reg=75.0))),
    ("Q_cfg2_lreg150", _cfg(dict(guidance_scale=2.0, lambda_reg=150.0))),
    ("Q_cfg3_lreg100", _cfg(dict(guidance_scale=3.0, lambda_reg=100.0))),
]

# Q_cfg_x_L: CFG × num_langevin
GROUP_Q_cfg_L = [
    ("Q_cfg2_L7",  _cfg(dict(guidance_scale=2.0, num_langevin=7))),
    ("Q_cfg2_L10", _cfg(dict(guidance_scale=2.0, num_langevin=10))),
    ("Q_cfg3_L10", _cfg(dict(guidance_scale=3.0, num_langevin=10))),
]

# Q_cfg_x_he: CFG × h_epsilon
GROUP_Q_cfg_he = [
    ("Q_cfg2_he5e-4", _cfg(dict(guidance_scale=2.0, h_epsilon=5e-4))),
    ("Q_cfg2_he5e-3", _cfg(dict(guidance_scale=2.0, h_epsilon=5e-3))),
    ("Q_cfg3_he1e-3", _cfg(dict(guidance_scale=3.0, h_epsilon=1e-3))),
]

# Q_cfg_x_lprox (high lambda_prox + CFG)
GROUP_Q_cfg_lp = [
    ("Q_cfg2_lp5",  _cfg(dict(guidance_scale=2.0, lambda_prox=5.0))),
]

# Q_3axis: triple combos
GROUP_Q_3 = [
    ("Q_cfg2_hx05_lr75",   _cfg(dict(guidance_scale=2.0, h_x=[0.1, 0.1, 0.1, 0.5], lambda_reg=75.0))),
    ("Q_cfg2_hx04_lr100",  _cfg(dict(guidance_scale=2.0, h_x=[0.1, 0.1, 0.1, 0.4], lambda_reg=100.0))),
    ("Q_cfg3_hx05_lr100",  _cfg(dict(guidance_scale=3.0, h_x=[0.1, 0.1, 0.1, 0.5], lambda_reg=100.0))),
]

# Q_cfg_x_rho
GROUP_Q_cfg_rho = [
    ("Q_cfg2_rho_05",  _cfg(dict(guidance_scale=2.0, rho_s=0.5, rho_e=0.5))),
    ("Q_cfg2_rho_22",  _cfg(dict(guidance_scale=2.0, rho_s=2.0, rho_e=2.0))),
]

STAGE_Q = (GROUP_Q_cfg_hx + GROUP_Q_cfg_lreg + GROUP_Q_cfg_L + GROUP_Q_cfg_he
           + GROUP_Q_cfg_lp + GROUP_Q_3 + GROUP_Q_cfg_rho)

# ===== Sanity / calibration =====
GROUP_R = [
    ("R_anchor_N_cfg2", dict(ANCHOR)),                                      # winner re-run
    ("R_anchor_C_L5",   dict(h_x=[0.1, 0.1, 0.1, 0.7], h_epsilon=1e-3,
                              num_langevin=5, lambda_reg=50.0, noise_scale=0.0,
                              terminal_replace_weight=1.0)),                # C_L5 re-run (no CFG)
]

CONFIGS = STAGE_P + STAGE_Q + GROUP_R

if __name__ == "__main__":
    print(f"Total configs: {len(CONFIGS)}")
    for name, kw in CONFIGS:
        diff = {k: v for k, v in kw.items() if k not in ANCHOR or ANCHOR.get(k) != v}
        print(f"  {name:<22s}  {diff}")
