"""Final-decision mini-sweep — co-tenancy fair comparison + best-of-best combos.

12 configs run in one batch so the GPU contention level is identical → the
between-config metric deltas reflect real parameter effects rather than
co-tenancy noise. This produces the clean number for the WINNER decision.
"""

ANCHOR = dict(
    h_x=[0.1, 0.1, 0.1, 0.7],
    h_epsilon=1e-3,
    num_langevin=5,
    lambda_reg=50.0,
    noise_scale=0.0,
    terminal_replace_weight=1.0,
)


def _cfg(extra):
    out = dict(ANCHOR); out.update(extra); return out


CONFIGS = [
    # Same-contention anchors (3 baselines for fair comparison)
    ("F_anchor_C_L5",    dict(ANCHOR)),
    ("F_anchor_N_cfg2",  _cfg(dict(guidance_scale=2.0))),
    ("F_anchor_P_cfg25", _cfg(dict(guidance_scale=2.5))),
    ("F_anchor_lreg300", _cfg(dict(lambda_reg=300.0))),

    # Combos: CFG × lambda_reg
    ("F_cfg20_lr150", _cfg(dict(guidance_scale=2.0, lambda_reg=150.0))),
    ("F_cfg20_lr300", _cfg(dict(guidance_scale=2.0, lambda_reg=300.0))),
    ("F_cfg25_lr150", _cfg(dict(guidance_scale=2.5, lambda_reg=150.0))),
    ("F_cfg25_lr300", _cfg(dict(guidance_scale=2.5, lambda_reg=300.0))),

    # Combos with h_x variations
    ("F_cfg25_hxu02", _cfg(dict(guidance_scale=2.5, h_x=0.2))),
    ("F_cfg25_lr150_hxs05", _cfg(dict(guidance_scale=2.5, lambda_reg=150.0,
                                       h_x=[0.1, 0.1, 0.1, 0.5]))),

    # Aggressive triple combos
    ("F_cfg25_lr200_L7",   _cfg(dict(guidance_scale=2.5, lambda_reg=200.0,
                                      num_langevin=7))),
    ("F_cfg20_lr150_hxu02", _cfg(dict(guidance_scale=2.0, lambda_reg=150.0,
                                       h_x=0.2))),
]

if __name__ == "__main__":
    print(f"Total: {len(CONFIGS)}")
    for name, kw in CONFIGS:
        print(f"  {name:<24s}  CFG={kw.get('guidance_scale',0)} lreg={kw.get('lambda_reg',50)} h_x={kw.get('h_x','-')} L={kw.get('num_langevin',5)}")
