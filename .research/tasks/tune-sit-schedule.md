# tune-sit-schedule（2026-08-25，done）

## 用户原始要求（逐字，按时间序）

1. 「还是撤回per-tau gamma2， 另外先尝试S_it：1，2，5，10」
2. 「我现在要traj Sit1/2/3/4/5的traj图片」「与此同时计算每个stage的MSE」
3. 「那我采用10，10，2，2呢」
4. 「跑一下10-2-2-2」
5. 「还是全部都是2的，在此基础之上，…」（转入 compare-s-prior-methods，基线定为均匀 2）

## 结论

- γ² 撤回 per-τ 表（config paths.gamma2_table → gamma2_meas_alg4.json）。
- 六档 S_it（1/2/3/4/5/10，box/junco，seed42，全同帧号 montage）最终 hole：
  0.1545 / **0.1495** / 0.1622 / 0.1749 / 0.1831 / 0.2192 —— V 形谷底 = 2
  （1=混合不足；≥3 每加一次内循环给棘轮多喂一转，恶化近线性）。
  obs 六档持平（0.0030–0.0034）⟹ S_it 只作用于 ker/hole 方向。
- 逐 stage 末 hole（stage0/1/2/3）：最优 S_it 分 stage，分界= b=1：
  b≥1.15 的 stage0/1 迭代多有益（stage1 最好 0.1093@10），b<1 的 stage2/3 单调有害。
- **跨 stage 损伤不可加**：10,10,2,2 → 0.1644（差于均匀 2；毒在 stage1 的 10 次——
  其末段 b_ker≈1.15 贴边界，存下的内容被后段加倍惩罚）；10,2,2,2 → **0.1485**
  （微超均匀 2；stage0 b=3–4 远离边界，随便迭代都安全）。
  经验规则：S_it 按 b 的余量分配。
- 用户最终裁决：**基线取均匀 S_it=2**（config num_langevin=2）。

## 代码

- utils.py：`num_langevin` 标量或长度 num_stages 列表（复用 `_per_stage`，与
  ode_steps_per_stage 同约定）；标量路径行为逐位不变（回归 2,2,2,2 复现）。

## 产物

results/alg4_box_stage3_diagnosis/：sit_comparison.csv（六档）、
trajectory_metrics_Sit{1..5}.csv、trajectory_montage_Sit{1..5}.png、
trajectory_metrics_Sit{10-10-2-2,10-2-2-2,2-2-2-2}.csv + 对应 montage、
stage_end_mse_by_sit.csv。
