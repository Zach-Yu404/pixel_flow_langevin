# Algorithm 4 — clean-endpoint posterior sampler for cascaded flow priors (PixelFlow)

Self-contained, finalized implementation (2026-09-05) of the Algorithm-4 sampler for linear inverse problems
(box / random inpainting, Gaussian / motion blur, 4× super-resolution) on 256×256 ImageNet with the
class-conditional PixelFlow prior. Numerically identical to the reference run it was extracted from
(`PixelFlowICLR/Algorithm2/main4.py`, results in `tests/reference_final.csv`); `tests/acceptance.py` checks that.

## What the sampler does (per stage k = 0..3, resolution 32→256)

```
x1 ← 0
for stage k:   x0 ~ N(0, I)
  for tau on the stage grid (10 steps):
    x_tau ← H_tau x1 + sigma_tau x0                      H_tau = (1-tau) s_k G + tau e_k I,  sigma_tau = (1-tau)(1-s_k) + tau(1-e_k)
    for s in 1..S_it[k]:                                 S_it = [2, 2, 1, 1]
      v ← v_theta(x_tau, tau, k)                         one network call (CFG 2.0, per-stage guidance schedule)
      [N_k² + γ²(k,tau) H_tau²] x1_hat = N_k[(e_k-s_k) x_tau + sigma_tau v] + γ² H_tau x_tau        (19)  CG
      x1 ← solve M^den x1 = Aᵀy/η² + C⁻¹ x1_hat + Aᵀξ_y/η + Hᵀξ_h/sigma + S^{-1/2} ξ_s               (22)  PCG, exact RTO draw
           C⁻¹ = HᵀH/sigma² + S⁻¹,  M^den = AᵀA/η² + C⁻¹
      x_tau ← H_tau x1 + sigma_tau ξ_0                                                            (23)  exact draw
    x0 ← (x_tau − H_tau x1)/sigma_tau
  x1 ← nearest-upsample(x1)
```

Fixed inputs (not tunable):
* **S** — prior covariance surrogate: per-class spectral S, `S_k = Fᴴ diag(P_k) F`, with `P_k` the floored, centred mean power
  spectrum of the image's own ImageNet synset at stage k, measured on all 50 val images of the class
  (`data/s_stats/spectral_power_labelled.npz`, built by `scripts/compute_s_stats.py`).
* **γ²(k, τ)** — velocity-error variance of the network, measured on the full ImageNet val set (`data/gamma2_all.json`,
  built by `scripts/compute_gamma2_stats.py`).

## Layout

```
run.py                      CLI: tasks × images → results/<out>/{final.csv, metrics.csv, nfe.json, *.png}
config.json                 finalized settings (paths, algorithm, sampler_kw, tasks_setup, tasks, images)
alg4/ops.py                 G, H_tau, B_k, N_k, sigma_tau, CG / PCG / Jacobi, exact adjoint, stage pyramid, masked MSE
alg4/prior.py               SpectralSOp, ClassSpectralPrior (S per class, bound per image)
alg4/sampler.py             run_algorithm4 (the listing above), velocity wrapper (CFG), endpoint solve, M^den
alg4/operators.py           inpainting / blur / SR operators, masks, A_k with exact adjoints, measurement construction
alg4/data.py, model.py      demo images (center-crop, [-1,1]); PixelFlow model loading
alg4/metrics.py             PSNR / SSIM (piq), LPIPS (official AlexNet + piq-VGG), FID helpers
pixelflow/                  vendored PixelFlow model, stage scheduler, config loader (unmodified)
scripts/compute_s_stats.py  S statistics from ImageNet val (one pass, no inference, ~7 min on one GPU)
scripts/compute_gamma2_stats.py   γ² table from ImageNet val (≈60 GPU-h on A100 with TF32; multi-GPU by claims)
scripts/make_motion_kernel.py     regenerate the DAPS motion-blur PSF data file (only if missing)
data/                       gamma2_all.json, motion kernel, synsets.txt, demo/ (7 images + labels.json), s_stats/ (npz, not in git)
tests/acceptance.py         reproduces tests/reference_final.csv (35 cells) within tolerance
```

## Setup

1. Environment: python ≥ 3.10, PyTorch with CUDA, `pip install -r requirements.txt` (conda env `pixelflow` on the cluster).
2. Model weights: `config.json → paths.model_dir` must hold PixelFlow's `config.yaml` + `model.pt`
   (class-conditional 256, 4 stages; default path points at `../../PixelFlow/IP_package/pretrained_models/c2img`).
   They are the public release `ShoufaChen/PixelFlow-Class2Image` on Hugging Face (model.pt 2.7 GB,
   sha256 37863b53bf26a60de50c5c700efde6a42157630c32ce68979020c10ec7cd30bb).
3. **S statistics (required, not in git):**
   ```
   python scripts/compute_s_stats.py          # needs ImageNet val + LOC_synset_mapping.txt / LOC_val_solution.csv (paths at the top of the script)
   ```
   → `data/s_stats/spectral_power_labelled.npz` (334 MB). Nothing runs without it.
4. γ² table: `data/gamma2_all.json` ships with the repo. To re-measure (optional, ≈60 GPU-h):
   ```
   PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=<g> python scripts/compute_gamma2_stats.py     # one process per GPU
   python scripts/compute_gamma2_stats.py --merge
   ```
5. Motion-blur kernel: `data/motion_kernel_k61_i0.5_seed42.npy` ships with the repo (DAPS random-walk PSF, seed 42);
   `scripts/make_motion_kernel.py --daps <DAPS repo>` regenerates it.

## Run

```
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 python run.py                       # all 5 tasks × 7 images → results/run
PYTHONHASHSEED=0 python run.py --tasks box_inpainting --images junco --save-png --out results/junco
PYTHONHASHSEED=0 python run.py --seed 43                                   # another posterior sample per cell
```
`PYTHONHASHSEED=0` is mandatory: random-inpainting masks are seeded with `hash(image_name)`.
`final.csv` columns: `mse_full`, `mse_hole` / `mse_obs` (inpainting; [-1,1] units), `psnr` / `ssim` / `lpips_alex` / `lpips_piq`
(on [0,1]), `meas_resid` (‖Ax−y‖/(η√m), ≈1 when consistent), `nfe`, `blk1_cg_bad` (Block-1 solves that did not
reach `cg_tol`; expected 0). Cells already in `final.csv` are skipped on rerun (resume).

## Acceptance test

```
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 python tests/acceptance.py             # ~15 min; all 35 cells
PYTHONHASHSEED=0 python tests/acceptance.py --tasks box_inpainting --images junco
```
Compares full / hole MSE with `tests/reference_final.csv` (tolerance 2e-4 absolute; identical code on the same GPU
differs by ~1e-5 from run-to-run non-determinism in the blur/SR adjoints).

Reference numbers (seed 42, mean over the 7 images; PSNR = 10·log10(4/MSE_full)):

| task | full MSE | PSNR (dB) | hole MSE |
|---|---|---|---|
| box_inpainting | 0.0458 | 19.85 | 0.1744 |
| random_inpainting | 0.0174 | 23.97 | 0.0232 |
| gaussian_blur | 0.0339 | 21.01 | – |
| motion_blur | 0.0358 | 20.69 | – |
| superresolution | 0.0324 | 21.17 | – |

## Conventions worth knowing

* Measurements: y = A x + N(0, σ_n² I), σ_n = 0.05 for every task; inpainting noise seed = sha256(task/image/measurement_seed);
  blur/SR seeds are fixed constants (see `alg4/operators.py`). Box mask: centred 128×128. Random mask: 70 % missing.
* Blur adjoints are exact autograd adjoints of the full chain (upsample → reflection-padded convolution);
  SR uses antialiased bicubic downsampling (DAPS-aligned).
* All sampler noise comes from one CPU `torch.Generator(seed)`; the draw order is x0 (per stage), then per inner
  iteration ξ_y, ξ_h, ξ_s, ξ_0. Changing that order changes the sample.
* The network runs in fp32 (TF32 matmul off) during sampling; the γ² table was measured with TF32 matmul
  (relative effect ≈1e-5, cross-checked).
* Not included on purpose (all evaluated and rejected in the research log): Langevin/pCN variants of Block 2,
  σ_τ² rescaling of (22), scalar or pooled S, per-class γ² tables, S_it schedules other than [2,2,1,1].
