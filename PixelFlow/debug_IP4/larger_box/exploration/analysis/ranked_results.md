# Combined ranked results — 167-config corpus

`larger_box` (32) + `exploration/runs` (45) + `exploration/runs_p` (78) + `exploration/runs_f` (12) = **167** configs.
Balanced = LPIPS + 0.02·max(0, 17−PSNR) + 0.5·max(0, |dHF|−0.05). Lower=better.

## Top-30 by balanced score

| Rank | Config | Source | PSNR | PSNRu | SSIM | LPIPS | |dHF| | Balanced |
|------|--------|--------|------|-------|------|-------|-------|----------|
| 1 | `F_cfg20_lr150_hxu02` | exploration_v3_final | 17.18 | 11.16 | 0.844 | **0.1200** | 0.010 | **0.1200** |
| 2 | `F_cfg25_lr200_L7` | exploration_v3_final | 17.81 | 11.78 | 0.848 | **0.1267** | 0.034 | **0.1267** |
| 3 | `C_L5` | larger_box | 17.51 | 11.49 | 0.844 | **0.1295** | 0.011 | **0.1295** |
| 4 | `P_lreg_300` | exploration_v2_deep | 17.60 | 11.58 | 0.829 | **0.1328** | 0.035 | **0.1328** |
| 5 | `P_hxu_02` | exploration_v2_deep | 17.38 | 11.36 | 0.848 | **0.1298** | 0.056 | **0.1329** |
| 6 | `P_lreg_150` | exploration_v2_deep | 17.55 | 11.53 | 0.831 | **0.1329** | 0.029 | **0.1329** |
| 7 | `C_L10` | larger_box | 16.60 | 10.57 | 0.851 | **0.1252** | 0.041 | **0.1333** |
| 8 | `P_lreg_75` | exploration_v2_deep | 17.51 | 11.49 | 0.832 | **0.1335** | 0.020 | **0.1335** |
| 9 | `N_cfg2` | exploration_v1 | 18.29 | 12.27 | 0.850 | **0.1339** | 0.017 | **0.1339** |
| 10 | `A_he1e-3` | larger_box | 18.38 | 12.36 | 0.872 | **0.1345** | 0.044 | **0.1345** |
| 11 | `N_L_5510` | exploration_v1 | 17.47 | 11.45 | 0.847 | **0.1350** | 0.017 | **0.1350** |
| 12 | `F_cfg20_lr300` | exploration_v3_final | 17.70 | 11.68 | 0.833 | **0.1351** | 0.030 | **0.1351** |
| 13 | `A_he2e-3` | larger_box | 19.20 | 13.18 | 0.876 | **0.1352** | 0.040 | **0.1352** |
| 14 | `C_L15` | larger_box | 16.87 | 10.85 | 0.854 | **0.1330** | 0.043 | **0.1357** |
| 15 | `P_cfg_25` | exploration_v2_deep | 17.30 | 11.28 | 0.831 | **0.1357** | 0.003 | **0.1357** |
| 16 | `F_anchor_N_cfg2` | exploration_v3_final | 18.91 | 12.89 | 0.853 | **0.1360** | 0.017 | **0.1360** |
| 17 | `G_hx_uni3` | exploration_v1 | 16.83 | 10.81 | 0.844 | **0.1328** | 0.007 | **0.1363** |
| 18 | `F_anchor_C_L5` | exploration_v3_final | 18.94 | 12.92 | 0.853 | **0.1365** | 0.025 | **0.1365** |
| 19 | `P_cfg_15` | exploration_v2_deep | 17.54 | 11.52 | 0.834 | **0.1366** | 0.014 | **0.1366** |
| 20 | `F_cfg20_lr150` | exploration_v3_final | 17.59 | 11.57 | 0.834 | **0.1368** | 0.024 | **0.1368** |
| 21 | `P_cfg_10` | exploration_v2_deep | 17.62 | 11.60 | 0.835 | **0.1380** | 0.018 | **0.1380** |
| 22 | `P_cfg_0` | exploration_v2_deep | 17.62 | 11.60 | 0.835 | **0.1380** | 0.018 | **0.1380** |
| 23 | `N_L_3_5_5_10` | exploration_v1 | 17.71 | 11.69 | 0.848 | **0.1383** | 0.020 | **0.1383** |
| 24 | `F_anchor_P_cfg25` | exploration_v3_final | 17.50 | 11.48 | 0.833 | **0.1388** | 0.026 | **0.1388** |
| 25 | `A_he5e-3` | larger_box | 19.94 | 13.92 | 0.882 | **0.1394** | 0.039 | **0.1394** |
| 26 | `N_he_inc_n01` | exploration_v1 | 17.72 | 11.70 | 0.835 | **0.1404** | 0.050 | **0.1405** |
| 27 | `N_cfg2_n01` | exploration_v1 | 18.28 | 12.26 | 0.838 | **0.1416** | 0.016 | **0.1416** |
| 28 | `P_cfg_05` | exploration_v2_deep | 17.66 | 11.64 | 0.836 | **0.1418** | 0.021 | **0.1418** |
| 29 | `N_wls` | exploration_v1 | 18.01 | 11.99 | 0.845 | **0.1430** | 0.014 | **0.1430** |
| 30 | `P_lp_100` | exploration_v2_deep | 17.24 | 11.22 | 0.833 | **0.1432** | 0.026 | **0.1432** |

## Top-15 by LPIPS

| Rank | Config | Source | PSNR | LPIPS | |dHF| |
|------|--------|--------|------|-------|-------|
| 1 | `F_cfg20_lr150_hxu02` | exploration_v3_final | 17.18 | **0.1200** | 0.010 |
| 2 | `P_hxu_06` | exploration_v2_deep | 15.65 | **0.1238** | 0.051 |
| 3 | `C_L10` | larger_box | 16.60 | **0.1252** | 0.041 |
| 4 | `F_cfg25_lr200_L7` | exploration_v3_final | 17.81 | **0.1267** | 0.034 |
| 5 | `P_hxu_04` | exploration_v2_deep | 16.45 | **0.1275** | 0.071 |
| 6 | `C_L5` | larger_box | 17.51 | **0.1295** | 0.011 |
| 7 | `P_hxu_02` | exploration_v2_deep | 17.38 | **0.1298** | 0.056 |
| 8 | `P_lreg_300` | exploration_v2_deep | 17.60 | **0.1328** | 0.035 |
| 9 | `G_hx_uni3` | exploration_v1 | 16.83 | **0.1328** | 0.007 |
| 10 | `P_lreg_150` | exploration_v2_deep | 17.55 | **0.1329** | 0.029 |
| 11 | `C_L15` | larger_box | 16.87 | **0.1330** | 0.043 |
| 12 | `P_lreg_75` | exploration_v2_deep | 17.51 | **0.1335** | 0.020 |
| 13 | `N_cfg2` | exploration_v1 | 18.29 | **0.1339** | 0.017 |
| 14 | `A_he1e-3` | larger_box | 18.38 | **0.1345** | 0.044 |
| 15 | `N_L_5510` | exploration_v1 | 17.47 | **0.1350** | 0.017 |

## Top-15 by PSNR

| Rank | Config | Source | PSNR | LPIPS | |dHF| |
|------|--------|--------|------|-------|-------|
| 1 | `I_ref_F1` | exploration_v1 | **20.76** | 0.1736 | 0.074 |
| 2 | `A_he1e-2` | larger_box | **20.70** | 0.1459 | 0.042 |
| 3 | `B_he_dec` | larger_box | **20.41** | 0.1629 | 0.037 |
| 4 | `H_resetEps` | exploration_v1 | **20.33** | 0.1799 | 0.014 |
| 5 | `B_he_s3vlow` | larger_box | **20.26** | 0.1703 | 0.071 |
| 6 | `B_he_s3low` | larger_box | **20.25** | 0.1710 | 0.070 |
| 7 | `B_he_v` | larger_box | **20.21** | 0.1527 | 0.061 |
| 8 | `A_he5e-2` | larger_box | **19.99** | 0.2022 | 0.090 |
| 9 | `A_he5e-3` | larger_box | **19.94** | 0.1394 | 0.039 |
| 10 | `B_he_dec_steep` | larger_box | **19.89** | 0.1601 | 0.004 |
| 11 | `N_he_dec_L5` | exploration_v1 | **19.57** | 0.1473 | 0.013 |
| 12 | `Q_cfg2_he5e-3` | exploration_v2_deep | **19.33** | 0.1707 | 0.103 |
| 13 | `A_he2e-3` | larger_box | **19.20** | 0.1352 | 0.040 |
| 14 | `P_he_5e-3` | exploration_v2_deep | **19.17** | 0.1654 | 0.075 |
| 15 | `A_he1e-1` | larger_box | **19.10** | 0.2078 | 0.094 |

## Same-batch F sweep (clean co-tenancy)

These 12 configs ran in one batch → metric deltas reflect real param effects, not co-tenancy noise.

| Config | PSNR | LPIPS | |dHF| | Balanced |
|--------|------|-------|-------|----------|
| `F_cfg20_lr150_hxu02` | 17.18 | **0.1200** | 0.010 | **0.1200** |
| `F_cfg25_lr200_L7` | 17.81 | **0.1267** | 0.034 | **0.1267** |
| `F_cfg20_lr300` | 17.70 | **0.1351** | 0.030 | **0.1351** |
| `F_anchor_N_cfg2` | 18.91 | **0.1360** | 0.017 | **0.1360** |
| `F_anchor_C_L5` | 18.94 | **0.1365** | 0.025 | **0.1365** |
| `F_cfg20_lr150` | 17.59 | **0.1368** | 0.024 | **0.1368** |
| `F_anchor_P_cfg25` | 17.50 | **0.1388** | 0.026 | **0.1388** |
| `F_anchor_lreg300` | 17.95 | **0.1435** | 0.065 | **0.1510** |
| `F_cfg25_lr150_hxs05` | 17.47 | **0.1526** | 0.109 | **0.1819** |
| `F_cfg25_hxu02` | 16.82 | **0.1576** | 0.092 | **0.1821** |
| `F_cfg25_lr150` | 17.27 | **0.1551** | 0.109 | **0.1846** |
| `F_cfg25_lr300` | 17.39 | **0.1535** | 0.117 | **0.1871** |

## All 167 configs sorted by balanced score

| Config | Source | PSNR | LPIPS | |dHF| | Balanced |
|--------|--------|------|-------|-------|----------|
| `F_cfg20_lr150_hxu02` | exploration_v3_final | 17.18 | 0.1200 | 0.010 | 0.1200 |
| `F_cfg25_lr200_L7` | exploration_v3_final | 17.81 | 0.1267 | 0.034 | 0.1267 |
| `C_L5` | larger_box | 17.51 | 0.1295 | 0.011 | 0.1295 |
| `P_lreg_300` | exploration_v2_deep | 17.60 | 0.1328 | 0.035 | 0.1328 |
| `P_hxu_02` | exploration_v2_deep | 17.38 | 0.1298 | 0.056 | 0.1329 |
| `P_lreg_150` | exploration_v2_deep | 17.55 | 0.1329 | 0.029 | 0.1329 |
| `C_L10` | larger_box | 16.60 | 0.1252 | 0.041 | 0.1333 |
| `P_lreg_75` | exploration_v2_deep | 17.51 | 0.1335 | 0.020 | 0.1335 |
| `N_cfg2` | exploration_v1 | 18.29 | 0.1339 | 0.017 | 0.1339 |
| `A_he1e-3` | larger_box | 18.38 | 0.1345 | 0.044 | 0.1345 |
| `N_L_5510` | exploration_v1 | 17.47 | 0.1350 | 0.017 | 0.1350 |
| `F_cfg20_lr300` | exploration_v3_final | 17.70 | 0.1351 | 0.030 | 0.1351 |
| `A_he2e-3` | larger_box | 19.20 | 0.1352 | 0.040 | 0.1352 |
| `C_L15` | larger_box | 16.87 | 0.1330 | 0.043 | 0.1357 |
| `P_cfg_25` | exploration_v2_deep | 17.30 | 0.1357 | 0.003 | 0.1357 |
| `F_anchor_N_cfg2` | exploration_v3_final | 18.91 | 0.1360 | 0.017 | 0.1360 |
| `G_hx_uni3` | exploration_v1 | 16.83 | 0.1328 | 0.007 | 0.1363 |
| `F_anchor_C_L5` | exploration_v3_final | 18.94 | 0.1365 | 0.025 | 0.1365 |
| `P_cfg_15` | exploration_v2_deep | 17.54 | 0.1366 | 0.014 | 0.1366 |
| `F_cfg20_lr150` | exploration_v3_final | 17.59 | 0.1368 | 0.024 | 0.1368 |
| `P_cfg_10` | exploration_v2_deep | 17.62 | 0.1380 | 0.018 | 0.1380 |
| `P_cfg_0` | exploration_v2_deep | 17.62 | 0.1380 | 0.018 | 0.1380 |
| `N_L_3_5_5_10` | exploration_v1 | 17.71 | 0.1383 | 0.020 | 0.1383 |
| `F_anchor_P_cfg25` | exploration_v3_final | 17.50 | 0.1388 | 0.026 | 0.1388 |
| `A_he5e-3` | larger_box | 19.94 | 0.1394 | 0.039 | 0.1394 |
| `N_he_inc_n01` | exploration_v1 | 17.72 | 0.1404 | 0.050 | 0.1405 |
| `N_cfg2_n01` | exploration_v1 | 18.28 | 0.1416 | 0.016 | 0.1416 |
| `P_cfg_05` | exploration_v2_deep | 17.66 | 0.1418 | 0.021 | 0.1418 |
| `N_wls` | exploration_v1 | 18.01 | 0.1430 | 0.014 | 0.1430 |
| `P_lp_100` | exploration_v2_deep | 17.24 | 0.1432 | 0.026 | 0.1432 |
| `G_hx_hi` | exploration_v1 | 17.81 | 0.1433 | 0.050 | 0.1433 |
| `B_he_w1+s3vlow` | larger_box | 17.90 | 0.1434 | 0.019 | 0.1434 |
| `N_n01_lr100_hx05` | exploration_v1 | 18.56 | 0.1445 | 0.010 | 0.1445 |
| `P_cfg_30` | exploration_v2_deep | 17.05 | 0.1448 | 0.003 | 0.1448 |
| `G_hx_late2` | exploration_v1 | 17.69 | 0.1454 | 0.042 | 0.1454 |
| `Q_cfg3_L10` | exploration_v2_deep | 17.50 | 0.1458 | 0.007 | 0.1458 |
| `A_he1e-2` | larger_box | 20.70 | 0.1459 | 0.042 | 0.1459 |
| `P_lp_20` | exploration_v2_deep | 17.41 | 0.1465 | 0.001 | 0.1465 |
| `C_L20` | larger_box | 17.26 | 0.1418 | 0.061 | 0.1472 |
| `N_he_dec_L5` | exploration_v1 | 19.57 | 0.1473 | 0.013 | 0.1473 |
| `Q_cfg2_L10` | exploration_v2_deep | 17.67 | 0.1478 | 0.010 | 0.1478 |
| `P_cfg_40` | exploration_v2_deep | 16.77 | 0.1434 | 0.009 | 0.1480 |
| `P_lp_5` | exploration_v2_deep | 17.41 | 0.1481 | 0.006 | 0.1481 |
| `P_lp_2` | exploration_v2_deep | 17.41 | 0.1484 | 0.007 | 0.1484 |
| `P_lp_005` | exploration_v2_deep | 17.41 | 0.1486 | 0.007 | 0.1486 |
| `P_lx_0001` | exploration_v2_deep | 17.41 | 0.1486 | 0.007 | 0.1486 |
| `P_lx_005` | exploration_v2_deep | 17.41 | 0.1486 | 0.007 | 0.1486 |
| `P_hxu_04` | exploration_v2_deep | 16.45 | 0.1275 | 0.071 | 0.1489 |
| `P_L8` | exploration_v2_deep | 18.16 | 0.1393 | 0.070 | 0.1492 |
| `Q_cfg2_L7` | exploration_v2_deep | 17.50 | 0.1503 | 0.034 | 0.1503 |
| `F_anchor_lreg300` | exploration_v3_final | 17.95 | 0.1435 | 0.065 | 0.1510 |
| `P_hxu_06` | exploration_v2_deep | 15.65 | 0.1238 | 0.051 | 0.1512 |
| `G_hx_lo` | exploration_v1 | 17.97 | 0.1416 | 0.070 | 0.1517 |
| `P_hx_s3_00` | exploration_v2_deep | 18.10 | 0.1523 | 0.026 | 0.1523 |
| `F_lr100_C5` | exploration_v1 | 17.99 | 0.1408 | 0.074 | 0.1528 |
| `P_L7` | exploration_v2_deep | 17.31 | 0.1535 | 0.039 | 0.1535 |
| `P_hxu_01` | exploration_v2_deep | 18.12 | 0.1551 | 0.026 | 0.1551 |
| `F_lr200_C5` | exploration_v1 | 18.03 | 0.1399 | 0.082 | 0.1557 |
| `M_C10_repeat` | exploration_v1 | 17.74 | 0.1566 | 0.027 | 0.1566 |
| `P_ode_12` | exploration_v2_deep | 17.44 | 0.1552 | 0.053 | 0.1568 |
| `N_n03_lr100_ode15` | exploration_v1 | 18.28 | 0.1568 | 0.004 | 0.1568 |
| `P_hx_s3_02` | exploration_v2_deep | 18.12 | 0.1569 | 0.027 | 0.1569 |
| `Q_cfg3_hx05_lr100` | exploration_v2_deep | 16.97 | 0.1573 | 0.045 | 0.1578 |
| `B_he_v` | larger_box | 20.21 | 0.1527 | 0.061 | 0.1581 |
| `P_obs_30` | exploration_v2_deep | 16.54 | 0.1480 | 0.055 | 0.1599 |
| `B_he_dec_steep` | larger_box | 19.89 | 0.1601 | 0.004 | 0.1601 |
| `P_hx_s3_05` | exploration_v2_deep | 18.08 | 0.1603 | 0.036 | 0.1603 |
| `P_he_7e-4` | exploration_v2_deep | 16.99 | 0.1592 | 0.055 | 0.1618 |
| `P_cfg_60` | exploration_v2_deep | 16.15 | 0.1448 | 0.026 | 0.1618 |
| `Q_cfg2_lp5` | exploration_v2_deep | 17.52 | 0.1620 | 0.051 | 0.1626 |
| `B_he_dec` | larger_box | 20.41 | 0.1629 | 0.037 | 0.1629 |
| `Q_cfg2_lreg25` | exploration_v2_deep | 17.90 | 0.1590 | 0.058 | 0.1630 |
| `Q_cfg2_rho_05` | exploration_v2_deep | 17.51 | 0.1621 | 0.052 | 0.1630 |
| `Q_cfg2_rho_22` | exploration_v2_deep | 17.51 | 0.1621 | 0.052 | 0.1630 |
| `R_anchor_N_cfg2` | exploration_v2_deep | 17.51 | 0.1621 | 0.052 | 0.1630 |
| `P_he_5e-4` | exploration_v2_deep | 16.80 | 0.1598 | 0.051 | 0.1644 |
| `Q_cfg2_he5e-4` | exploration_v2_deep | 17.38 | 0.1548 | 0.069 | 0.1644 |
| `Q_cfg30_hx05` | exploration_v2_deep | 17.28 | 0.1519 | 0.076 | 0.1648 |
| `Q_cfg3_he1e-3` | exploration_v2_deep | 17.52 | 0.1570 | 0.066 | 0.1649 |
| `P_hx_s3_09` | exploration_v2_deep | 17.97 | 0.1612 | 0.058 | 0.1654 |
| `P_lx_01` | exploration_v2_deep | 17.43 | 0.1656 | 0.051 | 0.1659 |
| `P_lx_05` | exploration_v2_deep | 17.43 | 0.1656 | 0.051 | 0.1659 |
| `P_lx_10` | exploration_v2_deep | 17.43 | 0.1656 | 0.051 | 0.1659 |
| `P_lx_50` | exploration_v2_deep | 17.43 | 0.1656 | 0.051 | 0.1659 |
| `P_rho_05_05` | exploration_v2_deep | 17.43 | 0.1656 | 0.051 | 0.1659 |
| `P_rho_05_1` | exploration_v2_deep | 17.43 | 0.1656 | 0.051 | 0.1659 |
| `P_rho_1_05` | exploration_v2_deep | 17.43 | 0.1656 | 0.051 | 0.1659 |
| `P_rho_1_2` | exploration_v2_deep | 17.43 | 0.1656 | 0.051 | 0.1659 |
| `P_rho_2_1` | exploration_v2_deep | 17.43 | 0.1656 | 0.051 | 0.1659 |
| `P_rho_2_2` | exploration_v2_deep | 17.43 | 0.1656 | 0.051 | 0.1659 |
| `Q_cfg2_hx05_lr75` | exploration_v2_deep | 17.47 | 0.1619 | 0.059 | 0.1667 |
| `Q_cfg2_hx04_lr100` | exploration_v2_deep | 17.44 | 0.1612 | 0.061 | 0.1668 |
| `Q_cfg25_hx05` | exploration_v2_deep | 17.42 | 0.1516 | 0.081 | 0.1672 |
| `P_he_3e-4` | exploration_v2_deep | 16.60 | 0.1603 | 0.048 | 0.1683 |
| `Q_cfg25_hx04` | exploration_v2_deep | 17.47 | 0.1512 | 0.085 | 0.1686 |
| `B_he_lowall` | larger_box | 17.89 | 0.1604 | 0.067 | 0.1687 |
| `D_L_s3hi` | larger_box | 17.41 | 0.1689 | 0.038 | 0.1689 |
| `D_L_lowall` | larger_box | 17.63 | 0.1693 | 0.042 | 0.1693 |
| `Q_cfg3_lreg100` | exploration_v2_deep | 17.59 | 0.1528 | 0.087 | 0.1712 |
| `R_anchor_C_L5` | exploration_v2_deep | 17.65 | 0.1670 | 0.059 | 0.1713 |
| `Q_cfg15_hx05` | exploration_v2_deep | 17.72 | 0.1518 | 0.091 | 0.1723 |
| `Q_cfg2_lreg75` | exploration_v2_deep | 17.79 | 0.1559 | 0.084 | 0.1727 |
| `Q_cfg2_lreg150` | exploration_v2_deep | 17.78 | 0.1524 | 0.091 | 0.1731 |
| `Q_cfg15_hx04` | exploration_v2_deep | 17.75 | 0.1512 | 0.094 | 0.1731 |
| `A_he5e-4` | larger_box | 16.70 | 0.1671 | 0.031 | 0.1732 |
| `N_obs_hi` | exploration_v1 | 16.85 | 0.1560 | 0.079 | 0.1736 |
| `F_lr25_C5` | exploration_v1 | 17.20 | 0.1705 | 0.056 | 0.1736 |
| `P_he_1.5e-3` | exploration_v2_deep | 17.73 | 0.1621 | 0.073 | 0.1736 |
| `P_hxu_005` | exploration_v2_deep | 18.51 | 0.1748 | 0.029 | 0.1748 |
| `K_ode_s3_20` | exploration_v1 | 17.22 | 0.1750 | 0.051 | 0.1753 |
| `D_L_s3low` | larger_box | 17.65 | 0.1602 | 0.082 | 0.1764 |
| `P_ode_8` | exploration_v2_deep | 17.70 | 0.1535 | 0.096 | 0.1766 |
| `P_hx_s3_12` | exploration_v2_deep | 17.91 | 0.1619 | 0.081 | 0.1775 |
| `P_L4` | exploration_v2_deep | 17.09 | 0.1672 | 0.071 | 0.1777 |
| `P_he_5e-3` | exploration_v2_deep | 19.17 | 0.1654 | 0.075 | 0.1781 |
| `N_n01_ode15` | exploration_v1 | 17.66 | 0.1613 | 0.084 | 0.1783 |
| `N_n01_hxlate` | exploration_v1 | 17.63 | 0.1671 | 0.075 | 0.1794 |
| `N_n01_hx1` | exploration_v1 | 17.58 | 0.1688 | 0.071 | 0.1795 |
| `E_N03_C10` | exploration_v1 | 17.41 | 0.1795 | 0.049 | 0.1795 |
| `P_he_2e-3` | exploration_v2_deep | 18.08 | 0.1665 | 0.076 | 0.1796 |
| `E_N01_C10` | exploration_v1 | 17.49 | 0.1796 | 0.041 | 0.1796 |
| `K_ode_s3_15` | exploration_v1 | 17.27 | 0.1726 | 0.064 | 0.1798 |
| `H_resetEps` | exploration_v1 | 20.33 | 0.1799 | 0.014 | 0.1799 |
| `P_he_3e-3` | exploration_v2_deep | 18.55 | 0.1668 | 0.077 | 0.1801 |
| `B_he_s3vlow` | larger_box | 20.26 | 0.1703 | 0.071 | 0.1809 |
| `B_he_s3low` | larger_box | 20.25 | 0.1710 | 0.070 | 0.1809 |
| `J_dps_s3_2` | exploration_v1 | 17.51 | 0.1584 | 0.096 | 0.1812 |
| `H_lprox05` | exploration_v1 | 17.52 | 0.1585 | 0.096 | 0.1814 |
| `H_lprox01` | exploration_v1 | 17.52 | 0.1585 | 0.096 | 0.1814 |
| `H_einject` | exploration_v1 | 17.52 | 0.1585 | 0.096 | 0.1814 |
| `E_N01_A2e3` | exploration_v1 | 18.49 | 0.1745 | 0.065 | 0.1819 |
| `F_cfg25_lr150_hxs05` | exploration_v3_final | 17.47 | 0.1526 | 0.109 | 0.1819 |
| `N_n01_lr200` | exploration_v1 | 17.66 | 0.1609 | 0.092 | 0.1821 |
| `F_cfg25_hxu02` | exploration_v3_final | 16.82 | 0.1576 | 0.092 | 0.1821 |
| `D_L_s3skip` | larger_box | 17.78 | 0.1579 | 0.102 | 0.1841 |
| `N_n01_lr100` | exploration_v1 | 17.66 | 0.1646 | 0.089 | 0.1842 |
| `E_N01_C5` | exploration_v1 | 17.13 | 0.1687 | 0.081 | 0.1843 |
| `F_cfg25_lr150` | exploration_v3_final | 17.27 | 0.1551 | 0.109 | 0.1846 |
| `M_C5_repeat` | exploration_v1 | 17.31 | 0.1720 | 0.075 | 0.1848 |
| `B_he_inc` | larger_box | 18.57 | 0.1703 | 0.079 | 0.1848 |
| `P_lreg_10` | exploration_v2_deep | 17.72 | 0.1681 | 0.085 | 0.1855 |
| `I_ref_F1` | exploration_v1 | 20.76 | 0.1736 | 0.074 | 0.1857 |
| `N_n03_lr100` | exploration_v1 | 17.61 | 0.1693 | 0.084 | 0.1862 |
| `F_cfg25_lr300` | exploration_v3_final | 17.39 | 0.1535 | 0.117 | 0.1871 |
| `N_lr100_hx1` | exploration_v1 | 17.73 | 0.1547 | 0.117 | 0.1880 |
| `J_dps_all5` | exploration_v1 | 17.32 | 0.1713 | 0.084 | 0.1883 |
| `E_N03_C5` | exploration_v1 | 17.11 | 0.1759 | 0.075 | 0.1885 |
| `N_lp01_n01` | exploration_v1 | 17.75 | 0.1599 | 0.109 | 0.1895 |
| `N_lr200_hxlo` | exploration_v1 | 17.80 | 0.1523 | 0.126 | 0.1905 |
| `L_no_bypass` | exploration_v1 | 17.45 | 0.1725 | 0.088 | 0.1914 |
| `A_he1e-4` | larger_box | 15.99 | 0.1733 | 0.033 | 0.1934 |
| `C_L3` | larger_box | 18.13 | 0.1593 | 0.119 | 0.1940 |
| `A_he5e-5` | larger_box | 15.87 | 0.1735 | 0.032 | 0.1961 |
| `Q_cfg2_he5e-3` | exploration_v2_deep | 19.33 | 0.1707 | 0.103 | 0.1974 |
| `A_he1e-5` | larger_box | 15.79 | 0.1732 | 0.032 | 0.1974 |
| `C_L30` | larger_box | 18.20 | 0.1804 | 0.094 | 0.2023 |
| `P_ode_5` | exploration_v2_deep | 18.05 | 0.1700 | 0.124 | 0.2072 |
| `P_L2` | exploration_v2_deep | 17.17 | 0.1861 | 0.095 | 0.2087 |
| `D_L_rampup` | larger_box | 17.04 | 0.1655 | 0.145 | 0.2130 |
| `N_obs_lo` | exploration_v1 | 18.36 | 0.1687 | 0.140 | 0.2139 |
| `A_he5e-2` | larger_box | 19.99 | 0.2022 | 0.090 | 0.2220 |
| `A_he1e-1` | larger_box | 19.10 | 0.2078 | 0.094 | 0.2299 |
| `D_L_focuss3` | larger_box | 16.82 | 0.1655 | 0.182 | 0.2350 |
| `P_obs_025` | exploration_v2_deep | 18.36 | 0.1952 | 0.132 | 0.2363 |
| `C_L1` | larger_box | 17.66 | 0.2032 | 0.145 | 0.2506 |
| `D_L_s3max` | larger_box | 15.71 | 0.1496 | 0.229 | 0.2649 |
| `L_no_warm` | exploration_v1 | 14.03 | 0.2499 | 0.095 | 0.3316 |