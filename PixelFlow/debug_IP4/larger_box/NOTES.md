# larger_box — h_epsilon × num_langevin sweep (memory)

128×128 box inpainting on 2-image debug GT (baseline_05). 32 configs across
4 axes. Base = WINNER3 perceptual recipe: `h_x=[0.1, 0.1, 0.1, 0.7]`,
`lambda_reg=50`, `terminal_replace=1.0`, `noise_scale=0`. Random box position
seeded with `torch.manual_seed(7919)`.

Perceptual = LPIPS-ranked. PSNR is reported but is *not* the primary axis.

## TL;DR — what to use for 128×128 box

| Need | Pick | Rationale |
|------|------|-----------|
| Best LPIPS | `C_L10` (h_eps=1e-3, L=10) | rank-1, identical to WINNER3 reference |
| LPIPS + 2× faster | `C_L5` (h_eps=1e-3, L=5) | rank-2 LPIPS, half the runtime, best |dHF| of the LPIPS top-3 |
| Balanced PSNR/LPIPS | `A_he2e-3` to `A_he5e-3` | LPIPS≈0.135, PSNR up to 19.9 |
| Best PSNR | `A_he1e-2` (h_eps=1e-2, L=10) | PSNR=20.70 but LPIPS jumps to 0.146 |
| Best HF match | `B_he_dec_steep` ([1e-2,1e-3,1e-4,1e-5]) | |dHF|=0.004, decent PSNR |

Default recommendation for 128×128 perceptual work: **C_L10** = exactly
WINNER3, no Langevin tweak helps.

## Key findings

1. **L=10 is already optimal.** L=1 and L=30 both lose ~0.08 LPIPS vs L=10.
   L=5 is nearly as good as L=10 at half the cost. L=15/20/30 hurt LPIPS
   monotonically, suggesting over-correction.

2. **h_epsilon sweet spot for LPIPS is 1e-3** (≈ W1 reference). Going lower
   (1e-4, 1e-5) collapses LPIPS to ~0.173 (under-correction). Going higher
   (1e-2, 5e-2, 1e-1) over-shoots and pushes LPIPS to 0.15→0.21 even though
   PSNR keeps climbing. Classic perception/distortion tradeoff.

3. **PSNR and LPIPS disagree.** All of the top-PSNR configs (h_eps≥5e-3) have
   LPIPS *worse* than h_eps=1e-3 with the same L. If you optimize PSNR you
   get blurry-but-aligned reconstructions; LPIPS prefers slightly less
   aligned but more structured fills.

4. **Per-stage h_epsilon schedules don't beat the scalar.** None of group B
   beats `C_L10`/`C_L5` on LPIPS. `B_he_dec_steep` is the only schedule
   worth remembering — it gives near-perfect HF energy match (|dHF|=0.004)
   while PSNR stays at 19.9, useful when HF-energy preservation matters.

5. **Per-stage num_langevin schedules all hurt.** Ramping L up at stage-3
   (`D_L_s3max/s3hi/focuss3`) blows up |dHF| (0.18–0.23) and degrades LPIPS.
   Ramping down (`D_L_s3skip/s3low`) is closer to baseline but still worse
   than uniform L=5–10. Stage-3 wants *fewer*, not more, Langevin steps.

6. **Stage-3 sensitivity is asymmetric.** `B_he_w1+s3vlow` (h_eps=1e-3
   except 1e-5 at stage-3) stays at LPIPS 0.143 — close to the C_L5/C_L10
   region. So *reducing* h_eps at the final stage is safe; raising L there
   is not.

## Why some hypotheses didn't pan out

- "More Langevin → better posterior approximation" — false beyond L=10. The
  CG solve already gets to tol=1e-5 in <50 iters per call (see
  `best_box/instrument_cg.py`), so extra outer Langevin steps just
  re-correct the same well-converged update.
- "Lower h_eps at stage-3 helps perceptual quality" — only marginally
  (`B_he_w1+s3vlow` ≈ 0.143). Bigger gain comes from keeping L moderate.
- "Increasing schedule (`B_he_inc`) lets early stages explore" — LPIPS=0.170,
  one of the worst. Early stages benefit from *more* correction, not less.

## Files

- `sweep_configs.py` — single source of truth, `CONFIGS = A+B+C+D` (32).
- `run_chunk.py` — per-GPU runner, computes PSNR/SSIM/LPIPS/PSNR_unobs/HF/|dHF|.
- `launch.sh` — 8 GPUs × 4 configs serial-per-GPU, ~10 min total wall time.
- `aggregate.py` — produces this WINNERS.md, sweep_summary.json, grids.
- `results/sweep_summary.json` — full kw + metrics for all 32 (machine-readable).
- `results/WINNERS.md` — top-8 by LPIPS / PSNR / |dHF| + full sorted table.
- `results/<name>.{json,pt,png}` — per-config metrics, xf tensor, GT|Meas|recon|crop grid.

## What this *doesn't* tell you

- Sample size = 2 GT images. Treat rankings within ±0.005 LPIPS as ties.
- Single fixed mask seed (7919). A different box position may shift winners.
- Only varies `h_epsilon` and `num_langevin`. `h_x`, `lambda_reg`,
  `noise_scale`, `terminal_replace` were held at WINNER3. For a wider
  sweep see `best_box/` (CFG sweep) or new sibling folders.

## If you continue this sweep

- Add `noise_scale ∈ {0.0, 0.1, 0.3}` × top-3 LPIPS configs — `H1` from
  best_box hinted that noise=0.3 helps perceptual; not yet tested at 128.
- Run top-3 on 100-image FID protocol (mirror `best_box/launch.sh`) — the
  2-image LPIPS rankings need FID validation.
- The `B_he_dec_steep` HF-match property is intriguing; worth checking on a
  texture-heavy mask region.
