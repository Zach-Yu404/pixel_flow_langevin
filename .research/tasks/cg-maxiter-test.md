# cg-maxiter-test（2026-08-27，done）

## 用户原始要求（逐字）
「与此同时做个测试基于baseline的spectral调整equation 27的CG solve，max_iteration
设置5，10，20，50，保留所有噪声，结果保存在alg4_box_cgtest」

## 解读
草稿 (27) = Lemma-9 RTO 噪声 ζ，"代入 (22) 后一次线性求解" ⟹ Block-1 PCG 的
`cg_max_iter`（非端点 (19) 的 cg_max_iter_endpoint）。基线 = 现行 config
（spectral、[2,2,1,1]、全噪声、cg_tol 1e-5），仅改 L；4 seeds；seed42 montage。

## 结果（results/alg4_box_cgtest/summary.md + cgtest_block1.csv + traj_block1_L*/）
| L | hole mean4±std | unconverged/40 | resid_max |
| 5 | 0.1115±0.0021 | 39 | 6.5e-2 |
| 10 | 0.1233±0.0033 | 39 | 3.2e-2 |
| 20 | 0.1246±0.0074 | 38 | 1.2e-2 |
| 50 | 0.1261±0.0026 | 34 | 2.5e-3 |
| 300(参考) | 0.1340±0.0047 | 0 | 1e-5 |
hole 随 L 单调：越截断越好（L=5 −17%）；seed42 stage 末 L=5：0.1867/0.1200/
0.1137/0.1096 全程单调下降；obs 不变。机制：从 warm start x1 出发的截断 PCG
只实现 RTO draw 的一部分——注入的后验方差噪声更少（与 ξ_h 探针、pCN 同一
注入机制，改由 solver 不精确实现）。**非 exact draw**（Lemma 9 需收敛），
非合法采样器改动，属隐式正则化诊断；参考臂 L=300 逐位复现档案 0.1334。
