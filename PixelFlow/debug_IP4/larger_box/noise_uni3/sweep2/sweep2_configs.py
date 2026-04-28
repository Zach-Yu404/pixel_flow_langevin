"""sweep2: anti-grain at noise_scale=1.0, on top of S1_h01_L15.

Fixed:  lambda_reg=150, lambda_prox=150, noise_scale=1.0, guidance_scale=2.0
Sweep:  h_epsilon ∈ {0.01, 0.001},  h_x ∈ {0.1, 0.2},
        num_langevin ∈ {5, 10, 15}, terminal_replace_weight ∈ {0, 1}
Total:  2 * 2 * 3 * 2 = 24 configs.
"""

FIXED = dict(
    lambda_reg=150.0,
    lambda_prox=150.0,
    noise_scale=1.0,
    guidance_scale=2.0,
)

H_EPS_LIST = [0.01, 1e-3]
H_X_LIST   = [0.1, 0.2]
L_LIST     = [5, 10, 15]
TR_LIST    = [0.0, 1.0]


def _name(he, hx, L, tr):
    he_s = "1e-2" if he == 0.01 else "1e-3"
    hx_s = f"hx{hx}".replace(".", "")  # hx01 / hx02
    return f"he{he_s}_{hx_s}_L{L}_tr{int(tr)}"


CONFIGS = []
for he in H_EPS_LIST:
    for hx in H_X_LIST:
        for L in L_LIST:
            for tr in TR_LIST:
                kw = dict(FIXED)
                kw.update(
                    h_epsilon=he,
                    h_x=hx,
                    num_langevin=L,
                    terminal_replace_weight=tr,
                )
                CONFIGS.append((_name(he, hx, L, tr), kw))


if __name__ == "__main__":
    print(f"Total: {len(CONFIGS)}")
    for n, kw in CONFIGS:
        print(f"  {n:<28s} {kw}")
