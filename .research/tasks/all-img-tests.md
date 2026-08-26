# all-img-tests（2026-08-26，running）

## 用户原始要求（逐字）
「继续用[2,2,1,1],seed42，两种S在剩下的几张图上测试box inpainting和其他4各task，
全都放在/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/results/all_img_tests
下边分folder， 有和没有xi_h和有没有3个xi，以及从stage 20以后有没有xi_h和3个xi」

## 展开（我方解读，已同步用户）
- 网格：5 task（box_inpainting/random_inpainting/gaussian_blur/motion_blur/
  superresolution）× 6 图（config images 除 junco：breastplate_armor/
  crane_structure/ibex_horns/lakeside_beach/sea_anemone/shetland_sheepdog）
  × 2 S 臂 × 5 噪声条件 = 300 cell，seed42，num_langevin 强制 [2,2,1,1]
  （覆盖 random_inpainting 的任务级 kw num_langevin=15）。
- 5 条件：baseline / ξ_h=0 全程 / 3ξ=0 全程 / ξ_h=0 自 f20（stage≥2）/
  3ξ=0 自 f20。**「3个ξ」= Block-1 RTO 三噪声 ξ_y/ξ_h/ξ_s**（置零后 Block 1
  变确定性后验均值，ξ_0 仍在）——已向用户声明此解读。
- 「从stage 20以后」= frame 20 = stage 2 起（与此前 f20 探针同口径）。

## 实现
scratchpad/all_img_tests.py：复用 main4._task_setup(task,image)（_box_setup
即其 box 特例）+ SP._init_globals/_load_sampling/_build_s_ops + utils
diag_noise_off 开关。GPU 2（空闲），detached+40 次 EIO 重试，幂等（summary.csv
已有行跳过）。产物每 (task,image)：gt.png/measurement.png/10 张终态 PNG/
comparison.jpg（GT+y+2 臂×5 条件网格），全局 summary.csv（hole/obs/full MSE、
PSNR(range2)、meas_resid、cg_bad、秒）。~60MB，≈6h。

## 口径备注
- spectral S 校准集 = 14 张非 junco 图（s_statistics.json；其 CAVEAT 文字
  "6 张"是陈旧的），**包含全部 6 张测试图 ⟹ spectral 在本网格是校准泄漏**；
  pooled_junco 只用 junco ⟹ 干净迁移。解读对比时必须带此标注。
- γ² 表 gamma2_meas_alg4.json 按 box 口径测得，直接沿用于其余 task。
- blur/SR 无 hole，指标只有 full MSE/PSNR；blur 任务伴随 = make_exact_AT
  自动微分精确伴随（motion flip(K) 约束由工厂结构保证）。
- 噪声条件均为诊断探针（draw-then-zero，非 exact draw）。

## 结果
待填。
