# PixelFlowICLR

ICLR-bound analysis experiments on top of the existing PixelFlow/MSFlow code.
Nothing here re-implements method logic — all math is imported from
`../PixelFlow/IP_package` (sampler utils, scheduler, demo image loader).

## Experiment 1 — one-step x̂₁ᵏ prediction MSE vs t

`onestep_mse_vs_t.py` · submit with `sbatch run_onestep_mse.sbatch`

Pipeline (per task × image × stage k):
x_GT → x₁ᵏ (chain-bilinear halving, = training collate `data_in1k.py`)
→ x_tᵏ = H_t(x₁ᵏ) + σ_t·ε (one ε per image+stage, shared across tasks)
→ ONE velocity call → {x̂₁,WLS (wls_estimate_x1), x̂₁,model (direct_estimate_x1)}
→ MSE vs x₁ᵏ. No Langevin / no iterative refinement / no ODE rollout.

- 5 tasks (box/random inpainting, gaussian/motion blur, SR), each using its
  `tasks/<task>/configs/LPIPS_king*.json` kw (ode_steps_per_stage, shift,
  guidance_scale, rho_s/rho_e, lambda_x, cg_*, g_bypass_stage3).
- 7 GT images = the set used by every existing `IP_package/playground_runs`
  experiment (union of their meta.json "images"; loaded via
  `demo_runner.load_demo_images` from `IP_package/demo`).
- Outputs under `results/onestep_mse/`:
  `<task>/raw_mse.json`, `<task>/curves/<image>_stage<k>.png` (WLS + Model on
  one axes), `<task>/overview_mean.png`, `all_mse.csv`.

Registered in `../.research/experiments/2026-08-15-onestep-mse-vs-t.md`.
