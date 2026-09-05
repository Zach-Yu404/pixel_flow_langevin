# Algorithm 4 — finalized configuration (2026-09-05)

Everything below is what `PYTHONHASHSEED=0 python main4.py` (mode `full_ip`) runs with `config_alg4.json` as committed.
Reference numbers (last section) come from exactly this configuration (`results/alg4_final/`).

## 1. Fixed choices (not config options)

| item | final | where |
|---|---|---|
| prior covariance S of (12)/(22) | per-class spectral S: S_k = F^H diag(P_k) F, P_k = floored centred mean power spectrum of the image's own ImageNet synset (50 val images/class), stages 32/64/128/256 | `main4.S_STATS`, `default_s2_fn`, `_bind_s2`; data `s_stats/spectral_power_labelled.npz` + `LOC_synset_mapping.txt` |
| γ²(k, τ) of (19)/(22) | ImageNet-val 50k table (mean over all classes) | `config_alg4.json paths.gamma2_table = {HERE}/gamma2_stats/gamma2_all.json` |
| Block 1 | exact RTO draw of x₁ from π(x₁ \| x_τ, y): PCG on M_τ^den = AᵀA/η² + C⁻¹, ζ = Aᵀξ_y/η + Hᵀξ_h/σ_τ + S^{-1/2}ξ_s (Lemma 9), Jacobi preconditioner, no ridge | `utils.run_posterior_sampling_alg4` l.12–13 |
| Block 2 | exact draw (23): x_τ = H_τ x₁ + σ_τ ξ₀, ξ₀ ~ N(0, I); no Langevin step, no h₀ | l.14 |
| clean endpoint (19) | [N_k² + γ²H_τ²] x̂₁-solve, CG cap 200 | `clean_endpoint_solve` |
| interpolant | H_τ = (1−τ)s_k G + τe_k I, σ_τ = (1−τ)(1−s_k) + τ(1−e_k); stage bounds s = (0, 1/7, 1/3, 3/5), e = (1/4, 1/2, 3/4, 1); REAL G (no G=I bypass) | `utils` |
| withdrawn (not in code) | Eq.(22) σ_τ² scaling, Block-2 adaptive pCN/OU refresh, Block-2 Langevin (x₀ − h₀/2(x₀+x̂₀)), C⁻¹ without (1/σ²)HᵀH, ξ₀:=ξ_h — all reverted byte-exact | `.research/tasks/*` |

## 2. Config (`config_alg4.json`)

| section | value |
|---|---|
| sampler_kw | num_langevin (S_it per stage) **[2, 2, 1, 1]** for every task (the old random_inpainting task-level override `num_langevin: 15` was removed on 2026-09-05), ode_steps_per_stage 10, shift 1.0, guidance_scale 2.0 (CFG), cg_tol 1e-5, cg_max_iter 300 |
| algorithm | sigma_min 1e-8, seed 42, measurement_seed 42, cg_max_iter_endpoint 200 |
| tasks_setup (all σ_n = 0.05, terminal_replace_weight 0) | box_inpainting: centred 128×128 mask · random_inpainting: mask_prob 0.7 · gaussian_blur: k 61, std 3.0 · motion_blur: k 61, intensity 0.5, DAPS kernel seed 42 · superresolution: ×4 antialias |
| images | breastplate_armor, crane_structure, ibex_horns, junco, lakeside_beach, sea_anemone, shetland_sheepdog (IP_package/demo, ImageNet val) |
| paths | model_dir {IP_PACKAGE}/pretrained_models/c2img, demo_dir {IP_PACKAGE}/demo, gamma2_table {HERE}/gamma2_stats/gamma2_all.json |

Diagnostic probe kwargs in `run_posterior_sampling_alg4` (`diag_noise_off`, `diag_noise_off_from_stage`, `diag_xi0_use_xih`,
`diag_xi0_off_from_frame`) all default OFF; with defaults the sampler is the exact scheme above.

## 3. Statistics behind the fixed choices (how to rebuild on a fresh machine)

Both are single-file, one pass over ImageNet val (`/CBIG-Standard-ECE/Zach_dataset/Zach_dataset/imageNet256`, LOC_synset_mapping.txt, LOC_val_solution.csv):

```
cd PixelFlowICLR/Algorithm2
python s_stats/compute_s_stats.py                       # ~7 min, no inference  -> s_stats/{s_pooled_statistics_*.json, spectral_power_{labelled,all}.npz}
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 python gamma2_stats/compute_gamma2_stats.py   # one process per GPU; TF32 matmul; ~60 GPU-h for 50k
PYTHONHASHSEED=0 python gamma2_stats/compute_gamma2_stats.py --merge                 # -> gamma2_stats/gamma2_all.json (committed), gamma2_labelled.json
```
`spectral_power_labelled.npz` (334 MB) is gitignored and **must** be rebuilt before `main4.py` runs (CONSTRAINTS.md 2026-09-03).
`gamma2_all.json` is committed, so the γ² step is only needed to re-measure. Note: `compute_gamma2_stats.py` initialises main4's
config (which resolves `paths.gamma2_table`), so keep the existing `gamma2_all.json` in place while re-measuring; the merge step
overwrites it at the end. Other untracked runtime inputs: `PixelFlow/IP_package/` (demo_runner, pipeline, demo images + labels.json,
pretrained_models/c2img/{config.yaml, model.pt}; sha256 of model.pt 37863b53…), the DAPS clone for motion-blur kernels
(`IP_package/baselines/DAPS`), and the conda env `pixelflow`.

- S statistics: preprocessing evaluate.cca (BOX halving + BICUBIC + centre-crop 256) + Normalize(0.5, 0.5); stage pyramid = chain of bilinear halving; P_k = (1/N)Σ|F x|² − |F μ|² per class, floor 1e-8·max.
- γ² definition: γ²(k, τ) = mean_images mean_pixels ‖v_θ(x_τ) − (B_k x₁ − (e_k − s_k)x₀)‖², x_τ = H_τ x₁ + σ_τ x₀, x₀ = eps_for(name, k), CFG 2.0, class embedding = own class; τ grid = the sampler's own schedule. 74 classes fp32 + 926 TF32 (same-class cross-check: relative difference ≤ 7e-5).

## 4. How to run

```
cd PixelFlowICLR/Algorithm2
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 python main4.py            # mode full_ip -> results/alg4 (full_ip_final.csv, full_ip_metrics.csv, frames)
PYTHONHASHSEED=0 python main4.py --mode diagnose                    # junco box per-frame diagnosis + montage
```
PYTHONHASHSEED=0 is mandatory (mask seeds use hash()). Conda env `pixelflow`. `full_ip` resumes from `results/alg4/full_ip_final.csv`
and skips cells already listed there; to regenerate, rename/remove `results/alg4` first (the pre-final Aug-24 output is archived as
`results/alg4_pre_final_0824`: old pooled S, old γ² table, num_langevin 10 — stale, reference only). `--mode diagnose` compares against
`results/alg4/full_ip_final.csv`, so run `full_ip` first.

Metrics conventions used in the tests (`rerun_imageNet/metrics.py`): PSNR/SSIM on [0,1] (piq, data_range 1), LPIPS official AlexNet
(piq-VGG also stored, never quoted); hole MSE in [−1,1] units on the raw sampler output.

## 5. Reference results with this configuration

(filled from `results/alg4_final/full_ip_final.csv`, seed 42, single sample per cell — see section 6 for the multi-seed tables)

REFERENCE_TABLE_PLACEHOLDER

## 6. Multi-seed evidence already measured with S = spectral_class (γ² = 7-image table; γ² = gamma2_all differs by ≤ 0.03 dB)

- `gamma2_stats/test/summary_est.md`: 5 tasks × 7 images; PSNR single-sample 21.42, MMSE5 24.45, MMSE10 25.06, ξ₀=0 23.49, 3ξ=0 25.64 (baseline table); gamma2_all −0.03 / −0.02 / −0.01 / +0.00 / −0.00 dB.
- `s_stats/test/s4_per_task.md`: S type × schedule sweep (spectral_class best; schedule 2222 vs 2211 within 0.1 dB).
- `results/all_img_tests/MMSE_results/`: earlier S (14-image spectral) baselines and noise-condition probes.

## 7. Files

`main4.py`, `utils.py`, `config_alg4.json`, `s_prior_methods.py` (ablation arms only), `s_stats/compute_s_stats.py`,
`gamma2_stats/compute_gamma2_stats.py`, `gamma2_stats/gamma2_all.json`, `gamma2_meas_alg4.json` (old 7-image reference, unused by default).
Research memory: `MSFlow/.research/` (STATE.yaml, CURRENT.md, decisions/2026-09-03-fixed-spectral-class-S.md, decisions/2026-09-05-gamma2-all-default.md).
