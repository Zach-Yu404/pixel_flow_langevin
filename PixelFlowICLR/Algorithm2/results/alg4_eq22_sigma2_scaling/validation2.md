# Round-2 validation (post codex code review fixes)

Fixes under test: builder gated into else (no scalar 1/sigma^2 on scaled branch); guard-failure fallback M_inv = I/c; sigma>0 guard.

## [F] fallback scaling law
[F] 1D: baseline x=1.000000047e-03 resid=0.00e+00 | None-fallback(old bug) x=2.351821167e-04 resid=7.65e-01 | I/c fallback x=1.000000047e-03 resid=0.00e+00 bitwise_equal=True
[F] 64-dim SPD c=1e-6: it_b/s=14/14 resid=8.55e-07/8.55e-07 relX=1.10e-07 bitwise=False

## [A2] post-fix 4-seed regression
[A2] spectral/base holes=0.1334/0.1284/0.1415/0.1325 mean4=0.1339±0.0047 cg_bad=0 it_max=172
[A2] spectral/scaled holes=0.1334/0.1284/0.1415/0.1325 mean4=0.1339±0.0047 cg_bad=0 it_max=172
[A2] spectral seed42 max|dx|=2.172e-04
[A2] pooled_junco/base holes=0.1331/0.1338/0.1467/0.1392 mean4=0.1382±0.0054 cg_bad=0 it_max=19
[A2] pooled_junco/scaled holes=0.1331/0.1338/0.1467/0.1392 mean4=0.1382±0.0054 cg_bad=0 it_max=19
[A2] pooled_junco seed42 max|dx|=1.656e-05

## [H] final-tensor SHA256 (seed42; archival reference)
[H] spectral base=49a738066340c31b scaled=81ad4dae3aecce35
[H] pooled_junco base=3876a9122e97beb5 scaled=d27ab4b9962583ba

## [G] frame identities incl. spectral fp64 (tol=1e-5, prod-like)
[G:pooled_junco:f9:st0] sig=7.502e-01: relM32=2.49e-07 relM64=2.64e-16 relRHS32=2.68e-07 relX=5.72e-07 it=19/19 res=7.6e-06/7.6e-06 nonfinite=0 max|Mv|=4.99e+04 max|Mtv|=2.81e+04
[G:pooled_junco:f15:st1] sig=6.589e-01: relM32=1.40e-07 relM64=1.63e-16 relRHS32=1.42e-07 relX=2.94e-07 it=18/18 res=8.7e-06/8.7e-06 nonfinite=0 max|Mv|=1.27e+04 max|Mtv|=5.52e+03
[G:pooled_junco:f25:st2] sig=4.354e-01: relM32=7.94e-08 relM64=1.47e-16 relRHS32=1.05e-07 relX=2.58e-07 it=18/18 res=9.5e-06/9.5e-06 nonfinite=0 max|Mv|=3.59e+03 max|Mtv|=6.81e+02
[G:pooled_junco:f31:st3] sig=3.556e-01: relM32=6.15e-08 relM64=1.30e-16 relRHS32=8.80e-08 relX=1.07e-07 it=3/3 res=1.0e-10/2.2e-10 nonfinite=0 max|Mv|=1.92e+03 max|Mtv|=2.43e+02
[G:pooled_junco:f38:st3] sig=4.480e-02: relM32=6.12e-08 relM64=1.17e-16 relRHS32=8.38e-08 relX=1.15e-07 it=3/3 res=6.1e-10/4.0e-10 nonfinite=0 max|Mv|=3.80e+03 max|Mtv|=7.63e+00
[G:pooled_junco:f39:st3] sig=4.000e-04: relM32=5.94e-08 relM64=1.04e-16 relRHS32=8.45e-08 relX=1.04e-07 it=2/2 res=1.3e-10/7.1e-11 nonfinite=0 max|Mv|=2.86e+07 max|Mtv|=4.57e+00
[G:spectral:f9:st0] sig=7.502e-01: relM32=2.62e-07 relM64=1.85e-16 relRHS32=1.90e-07 relX=2.04e-04 it=509/509 res=9.8e-06/9.5e-06 nonfinite=0 max|Mv|=5.03e+04 max|Mtv|=2.83e+04
[G:spectral:f15:st1] sig=6.589e-01: relM32=1.09e-07 relM64=1.63e-16 relRHS32=1.50e-07 relX=1.06e-05 it=544/545 res=1.0e-05/1.0e-05 nonfinite=0 max|Mv|=1.31e+04 max|Mtv|=5.68e+03
[G:spectral:f25:st2] sig=4.354e-01: relM32=7.33e-08 relM64=1.38e-16 relRHS32=9.49e-08 relX=2.76e-06 it=245/245 res=9.8e-06/9.8e-06 nonfinite=0 max|Mv|=4.03e+03 max|Mtv|=7.64e+02
[G:spectral:f31:st3] sig=3.556e-01: relM32=5.64e-08 relM64=1.08e-16 relRHS32=8.16e-08 relX=3.55e-06 it=151/151 res=9.3e-06/9.4e-06 nonfinite=0 max|Mv|=3.69e+03 max|Mtv|=4.67e+02
[G:spectral:f38:st3] sig=4.480e-02: relM32=5.80e-08 relM64=1.08e-16 relRHS32=8.26e-08 relX=2.28e-07 it=17/17 res=5.6e-06/5.6e-06 nonfinite=0 max|Mv|=5.39e+03 max|Mtv|=1.08e+01
[G:spectral:f39:st3] sig=4.000e-04: relM32=5.96e-08 relM64=1.04e-16 relRHS32=8.41e-08 relX=1.09e-07 it=2/2 res=4.8e-08/4.8e-08 nonfinite=0 max|Mv|=2.86e+07 max|Mtv|=4.57e+00

## [S8] sigma=1e-8 stress on real f39 operators
[S8:pooled_junco] sig=1.000e-08: relM32=6.18e-08 relM64=1.45e-16 relRHS32=6.20e-08 relX=6.24e-08 it=2/2 res=8.0e-11/7.6e-11 nonfinite=0 max|Mv|=4.57e+16 max|Mtv|=4.57e+00
[S8:spectral] sig=1.000e-08: relM32=6.18e-08 relM64=1.48e-16 relRHS32=6.20e-08 relX=6.24e-08 it=2/2 res=8.0e-11/7.6e-11 nonfinite=0 max|Mv|=4.57e+16 max|Mtv|=4.57e+00

## [DN] dense fp64 stage-0 (pooled): Mt=cM, direct-solve, MC cov
[DN] D=3072 frob rel |Mt - c M| = 5.17e-17 sym_err=0.00e+00
[DN] direct-solve per-realization max|x_sc - x_un| = 9.16e-15 (N=4000)
[DN] MC cov (Cov(zeta)=M check): max|var_emp/var_ana - 1| = 0.046 (32 dirs, N=4000, MC 1-sigma ~ 0.022)

## [T] tol sweep on f9 spectral (fixed RHS, vs fp64 dense ref)
[T] tol=1e-05: it=84/84 relX(s,b)=1.16e-05 err_b(vs dense)=1.09e-03 err_s=1.09e-03
[T] tol=1e-06: it=107/108 relX(s,b)=1.43e-05 err_b(vs dense)=1.24e-04 err_s=1.13e-04
[T] tol=1e-07: it=126/126 relX(s,b)=3.99e-06 err_b(vs dense)=8.37e-06 err_s=7.55e-06

## [BL] gaussian_blur operator identity (junco, f9/f38)
[BL:f9] sig=7.502e-01: relM32=2.49e-07 relM64=2.43e-07 relRHS32=1.93e-07 relX=4.77e-06 it=47/47 res=9.7e-06/9.7e-06 nonfinite=0 max|Mv|=3.47e+04 max|Mtv|=1.96e+04
[BL:f38] sig=4.480e-02: relM32=6.11e-08 relM64=2.24e-10 relRHS32=8.68e-08 relX=1.35e-07 it=7/7 res=9.8e-06/9.8e-06 nonfinite=0 max|Mv|=2.02e+03 max|Mtv|=4.06e+00

## [CB] diag_noise_off=['xi_h'] combo (spectral seed42)
[CB] eq22=off: hole=0.1063
[CB] eq22=on: hole=0.1063
[CB] max|dx|=5.043e-03

EQ22 VALIDATE2 DONE
