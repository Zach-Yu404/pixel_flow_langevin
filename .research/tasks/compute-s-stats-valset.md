# compute-s-stats-valset（2026-08-31，done）

## 用户原始要求（逐字）
「我现在要4个表格存在…/Algorithm2/s_stats/ 保存 s_pooled_statistics_labelled.json;
s_pooled_statistics_all.json; spectral_power_labelled.npz; spectral_power_all.npz
按照之前的方式来计算整个validation set的这值，inference的话只用一遍，分别按照
class label计算两种s和全部和在一起计算2种s，在计算之前，再次确定方法没有问题，
和现在的implementation也是一致的，就开始做，给我个预计完成时间」+「代码尽量放在
一个文件可以跑完」

## 实现
单文件 s_stats/compute_s_stats.py（自包含：内联 evaluate.cca / gt_stage_pyramid /
label map，逐字复刻）。方法核验四点与现行实现一致：pooled=measure_s2 原式
（E[x²]−E[x]²，float64，[-1,1]）；spectral=cmd_measure 原式（均值图中心化、
mean|Fz|² ortho、floor 1e-8×max），一遍式用精确恒等 (1/N)Σ|Fx|²−|Fμ|²；预处理
= evaluate.cca+Normalize；金字塔=bilinear 减半链。数据 = 本机
/CBIG-Standard-ECE/Zach_dataset/Zach_dataset/imageNet256 全 50,000 val +
LOC_val_solution 标签。类序单遍流式，逐类 shard + 每25类 checkpoint 断点续跑。
实跑 433 秒（~115 张/秒，无模型 inference——纯数据统计）。

## 结果（s_stats/ 四文件，共 ~350MB，主要是 labelled npz 1000 类×4 stage 谱）
- ALL pooled s²(k) = {0.2674, 0.2844, 0.2975, 0.3075}（50k 图）
- ALL 谱均值 stage3 = 0.3017（Parseval ✓）
- per-class stage3 s²：min 0.137 / median 0.294 / max 0.487（类间 3.5×）
- **关键对比**：junco 表（现行 pooled_junco）= {0.0397…0.0554}——**比 val 全集
  低 ~5.5×**！junco 是异常低方差图（简单背景鸟图）；demo14 谱均值 0.243-0.276
  也偏低 ~10%。⟹ 现行 pooled_junco 先验严重低估真实图像先验方差
  （S⁻¹ 过强 5.5×），这可能解释 pooled 臂此前在噪声路径上的额外脆弱性。
