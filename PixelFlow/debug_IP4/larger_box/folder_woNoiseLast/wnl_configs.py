"""folder_woNoiseLast: same fixed knobs as sweep2 but drop tr=1, add sigma_ref_sq axis.

Fixed:  lambda_reg=150, lambda_prox=150, noise_scale=1.0, guidance_scale=2.0,
        terminal_replace_weight=0.0   (NO terminal hard-replace)
Sweep:  h_epsilon ∈ {0.01, 0.001},  h_x ∈ {0.1, 0.2},
        num_langevin ∈ {5, 10, 15},  sigma_ref_sq ∈ {1e-3, 1e-4}
Total:  2 * 2 * 3 * 2 = 24 configs.

Hypothesis: reducing sigma_ref_sq strengthens the Tweedie pull (1/(σ_τ² + σ_ref²)
caps at 1/σ_ref²), which may suppress Brownian grain without needing tr=1.
"""

FIXED = dict(
    lambda_reg=150.0,
    lambda_prox=150.0,
    noise_scale=1.0,
    guidance_scale=2.0,
    terminal_replace_weight=0.0,
)

H_EPS_LIST = [0.01, 1e-3]
H_X_LIST   = [0.1, 0.2]
L_LIST     = [5, 10, 15]
SRS_LIST   = [1e-3, 1e-4]


def _name(he, hx, L, srs):
    he_s = "1e-2" if he == 0.01 else "1e-3"
    hx_s = f"hx{hx}".replace(".", "")
    srs_s = "srs1e-3" if srs == 1e-3 else "srs1e-4"
    return f"he{he_s}_{hx_s}_L{L}_{srs_s}"


CONFIGS = []
for he in H_EPS_LIST:
    for hx in H_X_LIST:
        for L in L_LIST:
            for srs in SRS_LIST:
                kw = dict(FIXED)
                kw.update(
                    h_epsilon=he,
                    h_x=hx,
                    num_langevin=L,
                    sigma_ref_sq=srs,
                )
                CONFIGS.append((_name(he, hx, L, srs), kw))


if __name__ == "__main__":
    print(f"Total: {len(CONFIGS)}")
    for n, kw in CONFIGS:
        print(f"  {n:<32s} {kw}")
