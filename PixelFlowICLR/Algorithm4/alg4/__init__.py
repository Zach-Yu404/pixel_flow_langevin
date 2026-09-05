"""Algorithm 4 — clean-endpoint posterior sampler for cascaded (PixelFlow) flow priors.

Finalized configuration (2026-09-05): per-class spectral prior covariance S from ImageNet-val statistics,
gamma^2(k, tau) table measured on ImageNet val (50k), exact RTO draw for x1 (Block 1, PCG) and exact draw
for x_tau (Block 2), S_it = [2, 2, 1, 1] inner iterations per stage. See README.md.
"""
__version__ = "1.0.0"
