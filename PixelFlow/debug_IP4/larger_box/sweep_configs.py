"""Configurations for larger_box (128x128) sweep over h_epsilon and num_langevin.

All configs share the same "perceptual" base inherited from the debug_IP4
WINNER3 setting (h_x stage-3 amplification + terminal_replace), since that
is the established recipe for visual content. We then vary:

  Group A  : h_epsilon  (scalar)               -- 10 configs
  Group B  : h_epsilon  (per-stage schedule)   -- 8 configs
  Group C  : num_langevin (scalar)             -- 7 configs
  Group D  : num_langevin (per-stage schedule) -- 7 configs

Total: 32 configs. Each config is a (name, kw) tuple where kw is forwarded
verbatim to ms_sampler_v5.run_ip4.
"""

# Perceptual base: WINNER3 minus the swept axis defaults.
BASE = dict(
    h_x=[0.1, 0.1, 0.1, 0.7],   # X4 stage-3 amplification (visual content)
    lambda_reg=50.0,             # default
    noise_scale=0.0,             # off
    terminal_replace_weight=1.0, # observed pixels hard-replaced at end
)


def _cfg(extra):
    out = dict(BASE)
    out.update(extra)
    return out


# Group A: h_epsilon scalar sweep (num_langevin fixed at 10)
GROUP_A = [
    ("A_he1e-5", _cfg(dict(h_epsilon=1e-5,  num_langevin=10))),
    ("A_he5e-5", _cfg(dict(h_epsilon=5e-5,  num_langevin=10))),
    ("A_he1e-4", _cfg(dict(h_epsilon=1e-4,  num_langevin=10))),
    ("A_he5e-4", _cfg(dict(h_epsilon=5e-4,  num_langevin=10))),
    ("A_he1e-3", _cfg(dict(h_epsilon=1e-3,  num_langevin=10))),  # W1 reference
    ("A_he2e-3", _cfg(dict(h_epsilon=2e-3,  num_langevin=10))),
    ("A_he5e-3", _cfg(dict(h_epsilon=5e-3,  num_langevin=10))),
    ("A_he1e-2", _cfg(dict(h_epsilon=1e-2,  num_langevin=10))),  # default
    ("A_he5e-2", _cfg(dict(h_epsilon=5e-2,  num_langevin=10))),
    ("A_he1e-1", _cfg(dict(h_epsilon=1e-1,  num_langevin=10))),
]

# Group B: h_epsilon per-stage schedules
GROUP_B = [
    ("B_he_dec",       _cfg(dict(h_epsilon=[1e-2, 5e-3, 2e-3, 1e-3], num_langevin=10))),
    ("B_he_dec_steep", _cfg(dict(h_epsilon=[1e-2, 1e-3, 1e-4, 1e-5], num_langevin=10))),
    ("B_he_inc",       _cfg(dict(h_epsilon=[1e-3, 2e-3, 5e-3, 1e-2], num_langevin=10))),
    ("B_he_s3low",     _cfg(dict(h_epsilon=[1e-2, 1e-2, 1e-2, 1e-3], num_langevin=10))),
    ("B_he_s3vlow",    _cfg(dict(h_epsilon=[1e-2, 1e-2, 1e-2, 1e-4], num_langevin=10))),
    ("B_he_lowall",    _cfg(dict(h_epsilon=[1e-3, 1e-3, 1e-3, 1e-4], num_langevin=10))),
    ("B_he_v",         _cfg(dict(h_epsilon=[1e-3, 1e-2, 1e-2, 1e-3], num_langevin=10))),
    ("B_he_w1+s3vlow", _cfg(dict(h_epsilon=[1e-3, 1e-3, 1e-3, 1e-5], num_langevin=10))),
]

# Group C: num_langevin scalar sweep (h_epsilon fixed at W1 = 1e-3, the perceptual sweet spot)
GROUP_C = [
    ("C_L1",  _cfg(dict(h_epsilon=1e-3, num_langevin=1))),
    ("C_L3",  _cfg(dict(h_epsilon=1e-3, num_langevin=3))),
    ("C_L5",  _cfg(dict(h_epsilon=1e-3, num_langevin=5))),
    ("C_L10", _cfg(dict(h_epsilon=1e-3, num_langevin=10))),  # default
    ("C_L15", _cfg(dict(h_epsilon=1e-3, num_langevin=15))),
    ("C_L20", _cfg(dict(h_epsilon=1e-3, num_langevin=20))),
    ("C_L30", _cfg(dict(h_epsilon=1e-3, num_langevin=30))),
]

# Group D: num_langevin per-stage schedules (h_epsilon=W1=1e-3)
GROUP_D = [
    ("D_L_s3skip",  _cfg(dict(h_epsilon=1e-3, num_langevin=[10, 10, 10, 1]))),
    ("D_L_s3low",   _cfg(dict(h_epsilon=1e-3, num_langevin=[10, 10, 10, 5]))),
    ("D_L_s3hi",    _cfg(dict(h_epsilon=1e-3, num_langevin=[10, 10, 10, 20]))),
    ("D_L_s3max",   _cfg(dict(h_epsilon=1e-3, num_langevin=[10, 10, 10, 30]))),
    ("D_L_rampup",  _cfg(dict(h_epsilon=1e-3, num_langevin=[3, 5, 10, 20]))),
    ("D_L_focuss3", _cfg(dict(h_epsilon=1e-3, num_langevin=[5, 5, 5, 30]))),
    ("D_L_lowall",  _cfg(dict(h_epsilon=1e-3, num_langevin=[3, 3, 3, 3]))),
]

CONFIGS = GROUP_A + GROUP_B + GROUP_C + GROUP_D

if __name__ == "__main__":
    import json
    print(f"Total configs: {len(CONFIGS)}")
    for name, kw in CONFIGS:
        print(f"  {name:<20s}  he={kw['h_epsilon']!s:<35s}  L={kw['num_langevin']!s}")
