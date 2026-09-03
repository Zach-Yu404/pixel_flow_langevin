# CG max_iteration test — Block-1 (27)/(22) PCG, spectral baseline, all noise on

config: junco/box, num_langevin [2,2,1,1] (config default), seeds 42-45, cg_tol 1e-5, only `cg_max_iter` varied. L=300 = current reference (converges, ~170 iters).

| L | hole mean4±std | seeds | resid_max | unconverged/40 | seed42 stage-ends |
|---|---|---|---|---|---|
| 5 | 0.1115±0.0021 | 0.1096/0.1092/0.1140/0.1131 | 6.49e-02 | 39 | 0.1867/0.1200/0.1137/0.1096 |
| 10 | 0.1233±0.0033 | 0.1246/0.1188/0.1277/0.1220 | 3.23e-02 | 39 | 0.2216/0.1316/0.1287/0.1246 |
| 20 | 0.1246±0.0074 | 0.1299/0.1177/0.1338/0.1170 | 1.24e-02 | 38 | 0.2462/0.1281/0.1328/0.1299 |
| 50 | 0.1261±0.0026 | 0.1287/0.1229/0.1286/0.1241 | 2.52e-03 | 34 | 0.2540/0.1243/0.1243/0.1287 |
| 300 | 0.1340±0.0047 | 0.1334/0.1284/0.1415/0.1325 | 9.99e-06 | 0 | 0.2551/0.1228/0.1291/0.1334 |

Reading: hole MSE is monotone in L — fewer CG iterations = lower hole MSE (L=5: −17%). A truncated PCG from the warm start x1 does not realise the full RTO draw; it injects only a fraction of the posterior-variance noise (39/40 frames unconverged, resid 6e-2). Same injection mechanism as the xi_h / pCN probes, now via solver inexactness. NOT an exact draw (Lemma 9 needs convergence) and not a legal sampler change; an implicit-regularisation diagnostic. obs MSE unchanged (measurement-pinned).
