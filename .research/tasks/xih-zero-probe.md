# xih-zero-probe（2026-08-26，running）

## 用户原始要求（逐字）
1. 「另外现在尝试让f20开始的epislon_h = 0做个对比」
2. 「回退，并且尝试让整个过程xi_h = 0展示一下结果」（回退指 f20 起点版）

## 机制
utils.py 常驻诊断开关 `diag_noise_off`（list ⊆ {xi_y,xi_h,xi_s,xi_0}）+
`diag_noise_off_from_stage`（默认 2）。draw-then-zero：仍按原顺序抽样再置零，
RNG 流对齐 ⟹ 默认 None 逐位不变（回归 0.1447/0.1143/0.1291/0.1495 已验）。
**定性：diagnostic probe，非 exact draw**——置零 ξ_h 而不做 FDT 补偿即篡改
条件协方差（codex R4 结论），结果只作 ξ_h 贡献上界，不作为 fix 保留。

## 第一版：ξ_h=0 从 f20 起（stage≥2）——已按用户要求回退
[2,2,1,1]、4 seeds：spectral 0.0923±0.0084（obs 0.0023，spread 0.1576）、
pooled 0.0965±0.0047（obs 0.0027，spread 0.1546），全收敛。基线 0.1339/0.1382
⟹ stage≥2 的 ξ_h 携带 ~0.04 终态 hole MSE 与 ~1/3 spread。seed42 轨迹
stage 末 0.2551/0.1228/0.0998/0.0890——末两 stage 首次全程单调去噪。
结果目录已删，数字保留于本文件与新 probe_results.csv 的 reverted-reference 行。

## 第二版：ξ_h=0 全程（from_stage=0）——完成
[2,2,1,1]、4 seeds：spectral 0.0940±0.0105（spread 0.1643）、pooled
0.0966±0.0045（spread 0.1548），全收敛。**与 f20 版均值持平（0.0923/
0.0965）**⟹ 终态收益几乎全部来自 stage≥2 的 ξ_h；早期置零不带来额外
收益（spectral 方差反而变大 0.0084→0.0105）。seed42 stage 末轨迹：
spectral 0.2437/0.1282/0.1176/0.1063、pooled 0.1445/0.1117/0.1023/0.0900
——两臂四 stage 全程单调去噪（基线在 stage2/3 回升）。
产物：results/alg4_noise_probe_xih/{probe_results.csv, spectral/, pooled_junco/}
（各臂 trajectory/inner/precision CSV + montage jpg），~2.6MB。

## 第三版：ξ_h=0 全程 × S_it 扫描（用户「在此基础上尝试不同的S_it」）
检验机理预测：无 ξ_h 注入后棘轮死亡 ⟹ 晚期多迭代应从有害转有益，V 形谷
底应右移。配置：uniform 1/2/5/10 + 晚期加重 [2,2,5,5]/[2,2,10,10]，双臂
× 4 seeds，全叠加 diag_noise_off=[xi_h] from_stage=0。产物追加
results/alg4_noise_probe_xih/sit_sweep_results.csv（幂等断点续跑）。结果待填。
