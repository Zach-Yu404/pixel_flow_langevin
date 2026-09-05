# clean-alg4-project（2026-09-05，running）

## 用户原始要求（逐字）
「/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm4写一个clean version的版本，完整的一个项目」

## 做法
从定稿代码路径（main4.run_full_ip / _task_setup / utils.run_posterior_sampling_alg4 及其传递依赖）抽出自包含项目
PixelFlowICLR/Algorithm4：
- alg4/{ops, prior, sampler, operators, data, model, metrics}.py：算子与解法器、按类谱 S、采样器（去掉全部 diag 探针、
  Alg-2 变体、消融臂）、五个 task 的算子/mask/测量构造（逐字保留 seed 与 RNG 顺序）、demo 图加载、模型加载、指标；
- pixelflow/：vendored model.py / scheduling_pixelflow.py / utils/config.py（未改）；
- run.py + config.json（定稿设置，S_it [2,2,1,1] 全任务）；scripts/{compute_s_stats, compute_gamma2_stats（重写为只依赖
  alg4）, make_motion_kernel}；data/{gamma2_all.json, motion_kernel（DAPS PSF 缓存为 npy，运行时不再依赖 DAPS 仓库）,
  synsets.txt, demo/ 7 图 + labels.json, s_stats/npz（gitignore）}；tests/acceptance.py（以 results/alg4 35 格为基准，
  容差 2e-4）；README（含 S_stats 必做步骤、γ² 可选重测、PYTHONHASHSEED=0 契约、参考数字）。
- 依赖图由 workflow wf_a6a5def9 三视角独立映射。

## 验证
- smoke：box/junco hole 0.1401 / MSE 0.0370 / resid 0.928 = 参考（0.14009 / 0.03699 / 0.9275）。
- 35 格验收（tests/acceptance.py，GPU 0）：35/35 在 2e-4 内，实际 |Δ| ≤ 8e-6（GPU 运行噪声量级）。
- 逐行等价审查（workflow wf_0f94c900，两视角）：采样器/先验/解法器与算子/测量/数据均"无偏差"；CPU 位级检查
  35 格的 gt/mask/y/A_k/A_kᵀ 全部 torch.equal，motion kernel 与 DAPS 现生成逐位相同；采纳的小修：acceptance
  空重叠判失败、labels.json 头部计数。有意的简化：只支持居中 box；use_daps_kernel/blur 的 measurement_mode 不再暴露。

## 状态
done（2026-09-05）。Algorithm4 已入库（data/s_stats npz 与 results/ 走 .gitignore）。
