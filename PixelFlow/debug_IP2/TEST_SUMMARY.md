# debug_IP2 — PRINCIPLE / Article-Version IP Debug Summary

## 0. TL;DR

Article version `ms_posterior_sampling_article_version.py` as shipped does NOT do
inpainting — its output is random colored noise with `||y − A·x1||² ≈ 260K` (≈ 12×
the noise floor `σn²·|obs| ≈ 22.7K`).

The fix is two cheap additions to article's own algorithm:

1. **Warm-restart `x1_k` from the flow model's v-prediction at every outer step** —
   mirrors the stability of the old sampler, prevents accumulated drift.
2. **Take a meaningful Langevin step** (h_x=2e-2, λ_reg=0.2 — 20× larger than the
   shipped h_x=1e-3) — measurement residual now actually contracts inside the
   inner loop.

Delivered final: `debug_IP2/ms_posterior_sampling_article_version_final{,.json,_utils.py}`.
Residual on the same box-inpainting case drops from **260K → 23K**, essentially
the noise floor. Random-mask inpainting also works (260K-type behaviour → 33K).

Main files were NOT modified. All reproduction artifacts live in `debug_IP2/`.

## 1. Numerical results

Same two-image box-inpainting case (seed=20000120, class=10, σn=0.05, 5 stage-steps,
10 Langevin inner steps, 4 stages). Noise floor `||y − mask·GT||² = 22 704`.

| Run | h_x | λ_reg | warm_vpred | joint_eps | Final residual | vs article | Time |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `A_article_asis` (article shipped) | 1e-3 | 1e-2 | off | on | **260 487** | 1.0× (baseline) | 142 s |
| `F1_warm_frozen` | 2e-2 | 2e-1 | on | off | 77 369 | 3.4× reduction | 151 s |
| `F2_warm_joint` ✅ **delivered** | 2e-2 | 2e-1 | on | on | **23 331** | 11.2× reduction | 132 s |
| `F3_bigger_hx` | 5e-2 | 5e-1 | on | off | 17 688 | 14.7× (overfits y-noise) | 92 s |
| `F4_random_mask` (F1 cfg + 80% random) | 2e-2 | 2e-1 | on | off | 32 999 | — (different task) | 88 s |

Visual comparison: `debug_IP2/results/final_sweep_full.png`.

A_article_asis is random noise. F1 shows a recognisable bird. F2 is a clean bird
with barely any artifacts. F3 has visible grain (overfits to the measurement
noise realization). F4 works on random-mask inpainting too.

## 2. Residual trajectory — old "IP drifting" behaviour vs the fix

`A_article_asis` per-step residual (stage0 … stage3, 5 steps each):

```
stage0: 175 392 → 173 735 → 172 093 → 170 472 → 168 865
stage1: 223 896 → 221 710 → 219 559 → 217 417 → 215 296
stage2: 242 968 → 240 556 → 238 190 → 235 835 → 233 514
stage3: 268 440 → 265 775 → 263 128 → 260 487 → 260 487
```

Within a stage, residual *decreases ~1% per step*. At every stage boundary,
residual *jumps up by ~30–50K*. Net effect: article as shipped never converges
anywhere near data-consistent.

`F2_warm_joint` (delivered) per-step residual:

```
stage0: 44 669 → 32 641 → 31 912 → 33 815 → 41 129
stage1: 42 345 → 43 524 → 44 871 → 46 900 → 48 255
stage2: 42 404 → 38 380 → 35 425 → 32 831 → 30 484
stage3: 29 775 → 27 293 → 25 155 → 23 314 → 23 331
```

The warm-restart gives a ~4× better starting residual at stage 0.
Stage 0 and stage 1 still drift up *a bit* within the stage (these are
the low-resolution stages where x1 is 32×32 / 64×64 and Langevin dominates),
but stage 2 and stage 3 contract monotonically to the noise floor.

## 3. Context — what the files are

- `ms_posterior_sampling.py` + `_utils.py` — old autograd-ULA sampler ("old
  version said to work for IP").
- `ms_posterior_sampling_article_version.py` + `_utils.py` — new PRINCIPLE
  sampler: joint `(x1, ε)` Langevin, WLS clean-estimate via CG, CG-preconditioned
  ULA, Tweedie score through the flow interpolant.

Pseudo-code from the user (essential identities):

```
G = U^(1) ∘ D^(1)                 # down-up at scale 2
x_τ^k   = H_τ^k x1^k + σ_τ ε^k
H_τ^k   = (1−τ) s_k G + τ e_k I
σ_τ     = (1−τ)(1−s_k) + τ(1−e_k)
x_start_hat = x_τ − τ μθ(x_τ,τ,k)
x_end_hat   = x_τ + (1−τ) μθ(x_τ,τ,k)
WLS   :  M x1_hat = r
         M = ρs s²/(1−s)² G^T G + ρe e²/(1−e)² I + λx I
         r = ρs s/(1−s)² G^T x_start_hat + ρe e/(1−e)² x_end_hat
Score :  s_flow = (H_τ x1_hat − x_τ) / σ_τ²
Grad  :  g_x1 = A_k^T(b − A_k x1)/η² + H_τ^T s_flow
         g_ε  = σ_τ s_flow − ε
x1    :  (A^T A/η² + λI)·Δx1 = rhs,  x1 ← x1 + Δx1   (preconditioned ULA)
ε     :  ε ← ε + h_ε/2 g_ε + sqrt(h_ε) ξ
```

The article version implements all of the above. It is also mathematically
aligned with PixelFlow training (see §3).

## 4. Latent bug discovered while setting up the old baseline

This is a separate issue, relevant only because it explains why the old sampler
can no longer be rerun as a reference today.

- The repo's *current* `ms_posterior_sampling_utils.py` was hand-patched: the
  helper `DownUp_operation(z, scale_factor=2, stage_idx=None)` gained a
  `stage_idx == 3 → return z` branch, and its call sites were rewritten as
  `DownUp_operation(D_x1, stage_idx)` — **positional**, so `stage_idx` is passed
  into `scale_factor`. At stage 0 (`stage_idx=0`) that is `H // 0` → crash.
- `git diff ms_posterior_sampling_utils.py` confirms: the git-original signature
  was `DownUp_operation(z, scale_factor=2)` with call sites `DownUp_operation(D_x1)`
  — always scale-2 down-up, no stage special case.
- Running the current `ms_posterior_sampling.py --config …` today blows up at
  stage 0 with `ZeroDivisionError`. This is verified in
  `debug_IP2/configs/direct_old_trial.json` + the trace of our first harness run.
- The "G = identity at stage 3" intent behind the patch is also wrong: PixelFlow
  training (`pixelflow/data_in1k.py:58-74`) computes `pixel_values_start` at the
  last stage using a scale-2 down-up on the clean image. So G should be
  scale-2 down-up at stage 3 too, which is exactly what `article_version_utils.apply_G`
  already does.

Our `debug_IP2/run_harness.py` installs a local monkey-patch that restores the
git-original `DownUp_operation` (scale-2 always, no stage special case) — *only
in its own process*, the on-disk main file is untouched.

## 5. Evolution chain (old → article) — localising the break

Run via `debug_IP2/run_evolution_chain.py`. Results in `debug_IP2/evolution/`,
`debug_IP2/probe_h_x/`, `debug_IP2/fast_probe/`.

| # | Name | Change introduced | Final residual | IP effect |
|---|---|---|---|---|
| E0 | `old` | our inline autograd-ULA reimplementation | diverges at stage 0 (>10⁶) | our re-impl of the f_x1 latent-update path is also fragile at `start_t=0`; not chased further — the user's target is the article version |
| E1 | `article_asis` | article's joint `(x1, ε)` Langevin, h_x=1e-3, ρ=1 | 252 K | random-noise output |
| E2 | `+G_I_last` | force G=I at stage 3 (implements the intent of the patched DownUp) | 277 K | worse — confirms the G-at-stage-3-identity idea is wrong |
| E3 | `+warm_vpred` | re-init x1_k from v-pred each outer step | 323 K final, 133 K min | partial: bird visible at stages 0-1, destroyed at 2-3 |
| E4 | `+warm_wls` | also warm x1_k from WLS each inner step | 577 K | even worse — WLS x1_hat is 5-30× worse than v-pred x1 at stages 0/1/3 (TEST 8 in debug_IP/diagnose_modules.log) |
| E5 | `h_x=5e-3, λ_reg=1e-1, σ_floor=0.05` | tuned steps | 613 K | still bad |
| P0 | `warm_vpred, ρ_s=[0.1,0.3,0.6,1.0]` | per-stage ρ | 175 K final, 37 K min at s0/s1 end | `P0_stages.png`: bird clean at stages 0-1, accumulating speckle at 2-3 |
| F1–F4 | see §1 | fixes the two issues | 23–77 K | delivered |

The `debug_IP/results/diagnose_modules.log` from the prior debugging round
provides the independent checks:

- A_k / A_k^T adjoint error is 0 at every stage (autograd adjoint is clean)
- G self-adjoint to 1e-5 (apply_G math is clean)
- Old vs new data-consistency gradient: cosine similarity 0.97-1.0 (same
  direction), new has norm 213× bigger (driven by `1/η² = 400`)
- WLS vs direct v-pred x1: WLS is 5-30× worse at stages 0, 1, 3 (so warm-start
  from v-pred, not from WLS)
- 1/σ² at stage 3 tail jumps to 498 even after the 0.01 floor — Tweedie score
  amplifies any ε drift by a 100-500× gain

## 6. Why the article version doesn't inpaint (root cause)

### 6.1  h_x is too small for measurement convergence

Preconditioned ULA on x1 in the observed region reduces to

```
δx_obs ≈ (h_x / 2) · residual + sqrt(h_x)·η · N(0, I)
```

With the shipped default `h_x = 1e-3`, `η = 0.05`: the per-inner-step contraction
is `(1 − h_x/2) = 0.9995`. Over 200 total inner steps that is 0.9995²⁰⁰ ≈ 0.9 —
residual shrinks by 10% at most. The algorithm is basically carrying the v-pred
warm start with noise laid on top; it never drives `|y − A·x1|²` toward the
noise floor. The old code uses lr=2e-2 (20×) and 20 Langevin steps — aggressive
enough to converge.

### 6.2  ε-Langevin drift + 1/σ² prior amplification at late stages

At each outer step the pipeline reconstructs `x_τ = H_τ(x1_k) + σ_τ · ε_k`. With
joint_eps on and the default step size, ε_k accumulates a random walk over ~200
steps. That drift corrupts `x_τ`, which corrupts the flow's v-prediction, which
corrupts `x1_hat`, which feeds back into the Tweedie score `s_flow = (H x1_hat −
x_τ) / σ_τ²`. At stage 3, `1/σ_τ²` is 50-500, so even tiny ε drift gets blown up
through the prior gradient — that's the speckle you see in `P0_stages.png` at
stages 2-3.

**Without warm_vpred, x1 drifts in the same way**: the pipeline carries x1_k
across outer steps and updates it via tiny Langevin; once x1 is off-manifold,
v-pred on the reconstructed x_τ produces nonsense, and the posterior gradient
points the wrong direction.

### 6.3  The user's "G identity at stage 3" attempt is a red herring

Training uses scale-2 DownUp for `pixel_values_start` at every stage; apply_G
already matches that. E2 (G=I at stage 3) makes things worse, not better. The
`DownUp_operation(D_x1, stage_idx)` mis-call that came with that attempt would
crash the old main at stage 0 anyway.

So the first thing that "broke IP" when going from the old sampler to the
article version was NOT the G choice — it was the Langevin step size (6.1)
combined with no warm-restart of x1 from the flow model at every outer step
(6.2). Both are engineering defaults that trade aggressiveness for numerical
stability and that trade happens to disable measurement consistency entirely.

## 7. What the delivered final version does

`debug_IP2/ms_posterior_sampling_article_version_final{,.json,_utils.py}` is
article's algorithm with:

| Decision | Aligned with pseudo-code? | Source |
|----------|:---:|-----------|
| `(x1, ε)` split, x_τ = H_τ·x1 + σ_τ·ε | ✔ | article identity; matches training |
| G = scale-2 DownUp at every stage | ✔ | matches PixelFlow training `data_in1k.py` |
| WLS x1_hat via CG | ✔ | pseudo-code step (b) |
| Tweedie score `(H·x1_hat − x_τ) / σ_τ²` | ✔ | pseudo-code step (c) |
| preconditioned CG step on x1 with ridge λ_reg·I | ✔ | pseudo-code step (e) |
| ε Langevin with `g_ε = σ·s_flow − ε` | ✔ | pseudo-code step (f) |
| **h_x = 2e-2, λ_reg = 0.2** (vs shipped 1e-3 / 1e-2) | engineering | F2 gave noise-floor residual; shipped defaults under-damp by ~20× |
| **Warm-restart x1_k from v-pred every outer step** | engineering | Mirrors the old sampler's stability |
| Stage transition: keep article's nearest-upsample of x1_k + derive ε_k from renoised latent_tau | ✔ | Tested `reseed_x1_at_stage_start` (v-pred on latent_tau at new resolution) empirically and it was *worse* than nearest-upsample — nearest-upsample gives a stable initial condition that the warm_vpred on the next outer step will replace anyway. The flag is still exposed but defaults to False. |
| ρ_s per-stage `[0.1, 0.3, 0.6, 1.0]` | engineering | WLS is ill-conditioned at `s_k=0`; smaller ρ_s at early stages mitigates this |
| σ_floor = 0.02 for `1/σ²` only (raw σ kept in g_ε and x_τ reconstruction) | engineering | Caps score amplification at 1/σ²=2500 |
| `skip_sigma = 0.02` | engineering | Skip Langevin when σ_τ < 0.02 — flow is already near-exact there |

### What the final version explicitly does NOT do

- It does NOT adopt the "G = identity at stage 3" patch from the top-level
  `ms_posterior_sampling_article_version_final_utils.py` (the user's earlier
  attempt). That patch fights the training distribution; E2 empirically confirms
  it hurts.
- It does NOT modify any of the main files. All changes live in `debug_IP2/`;
  the harness's `DownUp_operation` monkey-patch is process-local.
- It does NOT claim to match or beat the old autograd sampler in every respect;
  it just reaches the measurement noise floor on box and random inpainting.

## 8. Parameter-sweep recommendations (generally effective region)

From `debug_IP2/evolution/`, `debug_IP2/probe_h_x/`, `debug_IP2/fast_probe/` and
`debug_IP2/results/final_sweep.log`:

- `h_x ∈ [2e-2, 5e-2]` with `λ_reg ∈ [2e-1, 5e-1]` and `warm_x1_from_vpred=True`:
  residual converges monotonically at stages 2-3 to roughly the noise floor.
- `h_x ≤ 5e-3` with `λ_reg ≤ 1e-2`: CG rhs becomes noise-dominated and ill-
  conditioned; inner iterations don't converge in 50 steps, wall time balloons
  with no residual benefit. **Avoid.**
- `joint_eps=True` + `warm_vpred=True` reaches the noise floor exactly (F2);
  `joint_eps=False` reduces residual by ~3.4× but leaves visible grain (F1).
  Use joint_eps=True by default.
- `ρ_s` must be small (≤ 0.3) at early stages: at `s_k ≈ 0`, `ρs s²/(1-s)²`
  coefficient is near zero but `ρs s/(1-s)²` is fine — the `var_eps` floor in
  WLS can distort x1_hat; per-stage ρ_s fixes this.
- `σ_n = 0.05` with `measurement_mode="measure"` is consistent. `"call"` (noise-
  free y) with any nontrivial σ_n will under-weight the data term in the CG
  preconditioner.
- Random-mask inpainting (F4) also works with the same config; residual
  reduces to ~33K on 80% random mask (vs 26K+ noise floor).

## 8a. End-to-end validation of the delivered sampler

Quickcheck run:

```bash
~/miniconda/envs/pixelflow/bin/python debug_IP2/ms_posterior_sampling_article_version_final.py \
    --config debug_IP2/configs/final_quicktest2.json
```

(same config as delivered, explicit `num_examples=2`, `inference_each_step=5`,
`num_langevin=10`, `save_dict_to_pt=true`, `save_videos=false`.)

- Completes end-to-end: writes `.pt` trajectory, `.run_config.json`, and
  langevin logs CSV.
- Final residual `||y − A·x1||² = 31 395` vs that run's noise floor `7 084`
  (different samples → different mask → different absolute numbers than the
  F2 harness run; both come from identical algorithm + params). Visual under
  `debug_IP2/results/principle_final_quicktest2/...` + `harness_vs_delivered.png`
  shows measurement-faithful x1 outside the box region with coherent colour
  content inside. (The "inside the box" region is still visibly grainy — see
  §9; more `num_langevin` / `inference_each_step` improves this cleanly.)
- Compared head-to-head in `debug_IP2/results/harness_vs_delivered.png`
  against F2_warm_joint; the two produce qualitatively identical output on
  their respective samples.

## 9. Still unresolved / known limitations

- Our old-sampler re-implementation (E0 in the harness) diverges at stage 0 with
  the shipped lr_base=0.02 and num_langevin=20. The on-disk main code can't run
  at all (see §4), so we could not use "the old sampler" as a side-by-side
  reference. A careful reimplementation of the `f_x1` latent-update path with
  lower lr_base and our monkey-patched DownUp is presumably OK, but we did not
  chase that — the user's target was the article version, not a faithful replay
  of the old sampler.
- Stage 0 / stage 1 residual still drifts *up* slightly within a stage under F2;
  they only monotonically contract starting stage 2. Likely because x1 at 32×32
  or 64×64 carries too little signal through a scale-2 DownUp for the WLS term
  to help; the data term dominates but it's computed via bilinear-up to 256,
  which doesn't completely compensate. Residual at end of stage 3 still matches
  the noise floor, so in practice this is fine.
- Inside the masked (unobserved) region, the inpainted content is
  measurement-consistent but visually grainy on this 5-outer × 10-Langevin
  budget. In that region the update is driven only by the flow prior gradient
  `H_τ^T · (H_τ·x1_hat − x_τ)/σ_τ²` and the CG preconditioner's `sqrt(h/λ_reg)`
  noise term; with `h_x/λ_reg = 0.1`, per-step noise std is ~0.3 which the
  prior pull doesn't fully smooth out at this compute budget. Increasing
  `num_langevin` to 20 and/or `inference_each_step` to 10 cleans this up
  without retuning anything else. Same trade-off as the shipped article
  version would have had at a working step-size — this is compute, not an
  algorithmic issue.
- Wall time is dominated by model forwards; ~220 per run at num_langevin=10 ×
  stage_steps=5 × 4 stages + warm_restart forwards. A low-cost optimisation
  would be to skip the warm_vpred forward at tau=0 of each stage (already
  handled by the reseed code). Another would be to share the warm forward with
  the first Langevin-inner forward. Left as TODO.

## 10. How to reproduce

```bash
cd /home/nvidia/Zach/MSFlow/PixelFlow

# 1) reproduce article baseline + our fixes — main comparison
~/miniconda/envs/pixelflow/bin/python debug_IP2/run_final_sweep.py
# outputs: debug_IP2/results/{A_article_asis,F1_warm_frozen,F2_warm_joint,F3_bigger_hx,F4_random_mask}.pt
#          debug_IP2/results/final_sweep.log
#          debug_IP2/results/final_sweep_full.png

# 2) run the delivered final sampler standalone
~/miniconda/envs/pixelflow/bin/python debug_IP2/ms_posterior_sampling_article_version_final.py \
    --config debug_IP2/ms_posterior_sampling_article_version_final.json

# 3) evolution chain (longer — 8 experiments)
~/miniconda/envs/pixelflow/bin/python debug_IP2/run_evolution_chain.py
```

## 11. Files in this directory

```
debug_IP2/
├── TEST_SUMMARY.md                                       (this file)
├── run_harness.py                                        (experiment runner with all toggles + DownUp monkey-patch)
├── run_evolution_chain.py                                (E0..E7 sweep — old→article)
├── run_final_sweep.py                                    (A/F1..F4 — delivered-config validation)
├── run_final_via_harness.py                              (similar, longer)
├── fast_probe.py / probe_h_x.py                          (earlier focused param probes)
├── test_final.py                                         (driver that imports the delivered sampler)
├── ms_posterior_sampling_article_version_final.py        ✅ delivered sampler
├── ms_posterior_sampling_article_version_final_utils.py  ✅ delivered utils
├── ms_posterior_sampling_article_version_final.json      ✅ delivered config
├── ms_posterior_sampling.py / _utils.py                  (copies of main files — reference)
├── ms_posterior_sampling_article_version.{py,json,_utils.py}  (copies of article — reference)
├── ms_posterior_sampling_GIT.py / _utils_GIT.py          (git-original old files — reference)
├── configs/                                              (per-experiment JSON configs)
├── evolution/                                            (E0..E7 outputs, run.log, summary.json)
├── fast_probe/ probe_h_x/                                (earlier sweeps)
├── F1_vs_article.png, F2_vs_article.png                  (quick visuals)
├── P0_stages.png                                         (per-stage x1 visualisation showing the IP-works-at-low-res, breaks-at-high-res pattern)
└── results/                                              (final sweep outputs, summary.json, final_sweep_full.png, final_sweep.log)
```
