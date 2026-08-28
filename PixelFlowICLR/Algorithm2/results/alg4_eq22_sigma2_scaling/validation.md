# Eq.(22) sigma^2-scaling validation

## Phase A: baseline off (records: spectral 0.1339±0.0047, pooled 0.1382±0.0054)
[A-baseline] spectral holes=0.1334/0.1284/0.1415/0.1325 mean4=0.1339±0.0047 spread=0.2391 cg_bad=0 cg_iters_max=172
[A-baseline] pooled_junco holes=0.1331/0.1338/0.1467/0.1392 mean4=0.1382±0.0054 spread=0.2470 cg_bad=0 cg_iters_max=19

## Phase B: operator identities on real frame operators
[B] pooled_junco f9 st0 tau=0.999 sig=7.502e-01: relM32=2.59e-07 relM64=1.53e-16 relRHS32=2.47e-07 relX=4.56e-07 it_b/s=23/23 res_b/s=9.9e-07/9.9e-07 max|Mv|=4.42e+04 max|Mtv|=2.49e+04
[B] pooled_junco f25 st2 tau=0.555 sig=4.354e-01: relM32=7.32e-08 relM64=1.39e-16 relRHS32=1.06e-07 relX=2.59e-07 it_b/s=23/23 res_b/s=5.9e-07/5.9e-07 max|Mv|=4.17e+03 max|Mtv|=7.91e+02
[B] pooled_junco f31 st3 tau=0.111 sig=3.556e-01: relM32=6.09e-08 relM64=1.22e-16 relRHS32=8.89e-08 relX=1.09e-07 it_b/s=3/3 res_b/s=1.2e-10/1.8e-10 max|Mv|=2.24e+03 max|Mtv|=2.83e+02
[B] pooled_junco f38 st3 tau=0.888 sig=4.480e-02: relM32=6.13e-08 relM64=1.12e-16 relRHS32=8.40e-08 relX=1.11e-07 it_b/s=3/3 res_b/s=5.1e-10/6.7e-10 max|Mv|=4.18e+03 max|Mtv|=8.38e+00
[B] spectral f9 st0 tau=0.999 sig=7.502e-01: relM32=1.95e-07 relRHS32=2.40e-07 relX=1.28e-04 it_b/s=600/600 res_b/s=1.3e-06/1.4e-06 max|Mv|=4.43e+04 max|Mtv|=2.49e+04
[B] spectral f25 st2 tau=0.555 sig=4.354e-01: relM32=7.47e-08 relRHS32=9.49e-08 relX=2.90e-06 it_b/s=312/312 res_b/s=9.9e-07/9.9e-07 max|Mv|=4.61e+03 max|Mtv|=8.74e+02
[B] spectral f31 st3 tau=0.111 sig=3.556e-01: relM32=5.65e-08 relRHS32=8.14e-08 relX=3.46e-06 it_b/s=181/181 res_b/s=9.3e-07/9.3e-07 max|Mv|=3.68e+03 max|Mtv|=4.65e+02
[B] spectral f38 st3 tau=0.888 sig=4.480e-02: relM32=5.78e-08 relRHS32=8.29e-08 relX=2.26e-07 it_b/s=20/20 res_b/s=6.1e-07/6.1e-07 max|Mv|=5.58e+03 max|Mtv|=1.12e+01

## Phase C: eq22_sigma2_scale=True, same seeds
[C-scaled] spectral holes=0.1334/0.1284/0.1415/0.1325 mean4=0.1339±0.0047 spread=0.2391 cg_bad=0 cg_iters_max=172
[C-scaled] pooled_junco holes=0.1331/0.1338/0.1467/0.1392 mean4=0.1382±0.0054 spread=0.2470 cg_bad=0 cg_iters_max=19

## Per-seed deltas (scaled - baseline)
[D] spectral dhole=-0.0000/-0.0000/-0.0000/+0.0000 max|dx|=2.026e-03
[D] pooled_junco dhole=+0.0000/+0.0000/-0.0000/+0.0000 max|dx|=1.137e-05

EQ22 VALIDATE DONE
