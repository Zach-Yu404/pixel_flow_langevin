# fp32 vs TF32 same-class cross-check (3 classes x 40 (stage,tau) points)

- n01440764: fp32→tf32: rel diff mean +5.40e-06, max|.| 7.17e-05; stage3 τ=0 0.01907 vs 0.01907, τ=0.999 0.15082 vs 0.15083
- n01443537: fp32→tf32: rel diff mean +5.53e-06, max|.| 2.95e-05; stage3 τ=0 0.00636 vs 0.00636, τ=0.999 0.14768 vs 0.14769
- n01484850: fp32→tf32: rel diff mean +8.78e-06, max|.| 4.80e-05; stage3 τ=0 0.00616 vs 0.00616, τ=0.999 0.13747 vs 0.13747
- all 120 points: mean rel diff +6.57e-06, max |rel diff| 7.17e-05
