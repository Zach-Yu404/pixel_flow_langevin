"""Generate fully-expanded config JSONs for all best_box test settings.

Single source of truth: DEFAULTS reflect run_ip4() signature defaults
(see debug_IP4/ms_sampler_v5.py:run_ip4). Per-config OVERRIDES come from
documented top results in debug_IP4. The merged 'kw' dict is what
run_eval.py forwards to run_ip4 (class_label and seed are runtime-only).
"""
import json, os, copy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "configs")
os.makedirs(OUT, exist_ok=True)

# Defaults must match run_ip4() signature in debug_IP4/ms_sampler_v5.py
DEFAULTS = dict(
    num_langevin=10,
    h_x=0.1,
    h_epsilon=0.01,
    lambda_reg=50.0,
    noise_scale=0.0,
    ode_steps_per_stage=10,
    shift=1.0,
    guidance_scale=0.0,
    warm_restart=True,
    g_bypass_stage3=True,
    x1_init_mode="model",
    lambda_x=0.01,
    rho_s=1.0,
    rho_e=1.0,
    cg_tol=1e-5,
    cg_max_iter=50,
    dps_kick_zeta=None,
    terminal_replace_weight=0.0,
    soft_replace_weight=0.0,
    h_x_obs_ratio=1.0,
    lambda_prox=0.0,
    skip_eps_update=False,
    reset_eps_per_ode_step=False,
    lambda_eps_norm=0.0,
    mask_aware_eps=False,
    lambda_eps_prox=0.0,
    lambda_eps_inject=0.0,
)

CONFIGS = [
    dict(
        name="A0_reference_WINNER3",
        desc="WINNER3 reference: W1 (h_eps=0.001) + X4 stage3 h_x=0.7 + terminal_replace=1.0",
        source="debug_IP4/final_ana/visual_best_box  (single-image PSNR=20.56, HF=0.684)",
        overrides=dict(
            h_epsilon=0.001,
            h_x=[0.1, 0.1, 0.1, 0.7],
            terminal_replace_weight=1.0,
        ),
    ),
    dict(
        name="E1_W1+lam20",
        desc="W1 + lambda_reg stage3=20 (CG softer at stage 3 -> bigger Langevin steps)",
        source="debug_IP4/final_ana/visual_best_box  (single-image PSNR=22.17, HF=0.631)  -- BEST PSNR",
        overrides=dict(
            h_epsilon=0.001,
            lambda_reg=[50, 50, 50, 20],
            terminal_replace_weight=1.0,
        ),
    ),
    dict(
        name="C1_W1+hx_multistage",
        desc="W1 + per-stage h_x ramp [0.1, 0.2, 0.4, 0.7]: progressive prior amplification",
        source="debug_IP4/final_ana/visual_best_box  (single-image PSNR=20.06, HF=0.703)",
        overrides=dict(
            h_epsilon=0.001,
            h_x=[0.1, 0.2, 0.4, 0.7],
            terminal_replace_weight=1.0,
        ),
    ),
    dict(
        name="H1_W1+X4+noise0.3",
        desc="WINNER3 + noise_scale stage3=0.3: Langevin noise injection for stochastic texture",
        source="debug_IP4/final_ana/visual_best_box  (single-image PSNR=20.31, HF=0.692)",
        overrides=dict(
            h_epsilon=0.001,
            h_x=[0.1, 0.1, 0.1, 0.7],
            noise_scale=[0, 0, 0, 0.3],
            terminal_replace_weight=1.0,
        ),
    ),
    dict(
        name="S3_stage3_only",
        desc="h_epsilon stage-only schedule: HI=0.01 stages 0-2, LO=0.001 stage 3 + terminal_replace=1.0",
        source="debug_IP4/results_round7  (single-image PSNR=21.52, HF=0.624)  -- top non-E1 PSNR",
        overrides=dict(
            h_epsilon=[0.01, 0.01, 0.01, 0.001],
            terminal_replace_weight=1.0,
        ),
    ),
]


def merge(defaults, overrides):
    out = copy.deepcopy(defaults)
    out.update(overrides)
    return out


CFG_SWEEP = [1.0, 2.0, 3.0, 4.0]   # classifier-free guidance scales to add per base config


def write_one(name, desc, source, overrides):
    kw = merge(DEFAULTS, overrides)
    out = dict(name=name, desc=desc, source=source,
               overrides=overrides, kw=kw)
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


def main():
    n = 0
    # Base configs (CFG=0, original 5)
    for cfg in CONFIGS:
        write_one(cfg["name"], cfg["desc"], cfg["source"], cfg["overrides"])
        n += 1

    # CFG sweep variants: add guidance_scale override on top of base overrides.
    for cfg in CONFIGS:
        for gs in CFG_SWEEP:
            tag = f"cfg{int(gs)}"
            ov = dict(cfg["overrides"])
            ov["guidance_scale"] = gs
            write_one(
                name=f"{cfg['name']}_{tag}",
                desc=f"{cfg['desc']}  +  guidance_scale={gs}",
                source=f"{cfg['source']}  (CFG sweep variant)",
                overrides=ov,
            )
            n += 1
    print(f"\n{n} configs generated  ({len(CONFIGS)} base + {len(CONFIGS)*len(CFG_SWEEP)} CFG variants).")


if __name__ == "__main__":
    main()
