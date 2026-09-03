# default-spectral-class-s（2026-09-03，done）

## 用户原始要求（逐字）
「现在默认使用spectral_class」；澄清：「我的意思是，这个不用作为一个选项，默认使用
spectral_class的s_stats作为指定选项」。

## 实现（main4.py / config_alg4.json / s_prior_methods.py）
- S 固定：S_STATS = {spectral_npz: s_stats/spectral_power_labelled.npz, synset_map:
  imageNet256/LOC_synset_mapping.txt}；default_s2_fn(K) 返回 _SpectralS2（未绑定），
  _bind_s2(s2_fn, class_idx) 在 _run_once 入口和 full_ip 循环内按 demo class_idx 绑定
  （contraction / cg_audit 在 task setup 后绑定，并改为 SOperator-aware：
  make_M_tau_den 直接收 SOperator，S^{-1/2}ξ_s 用 apply_S_inv_sqrt）。
- config_alg4.json 删除 S_prior 段；CONFIG_SCHEMA/加载/打印同步；S_PRIOR 全局移除。
  make_s2_fn 保留四种标量 prescription 作为 diversity 消融臂（S_PRIOR_KEYS 检查改名为
  diversity arm）。s_prior_methods.py 的 incumbent 臂改用 default_s2_fn。
- 第一版曾把 spectral_all/spectral_class 做成 S_prior 的两种 mode，用户澄清后撤掉，
  改为硬编码。

## 验证（scratchpad/spectral_class_check.py, determinism_check.py, smoke_cfg.json）
- shetland_sheepdog(n02105855)/sea_anemone(n01914609)：4 个 stage 的功率谱张量与
  s_stats/test/run_s4_test.py::make_s2fn 构造 torch.equal 全 True；Tr(S_k)/D =
  0.217/0.230/0.239/0.246（shetland）。
- 未绑定调用按设计抛 RuntimeError。
- 完整采样（box/shetland/seed42/[2,2,1,1]）default vs S4 构造 max|dx| 1.2e-4~1.4e-3，
  hole MSE 0.19453 vs 0.19454；同一 s2_fn 重复跑本身差 6.6e-4/8.0e-4 ⟹ 差异即 GPU
  运行间噪声，S 等价。
- CLI 端到端：PYTHONHASHSEED=0 main4.py --config <full_ip smoke> 跑通（box/junco 40 帧，
  打印 S=spectral_class(...)，full_ip_final.csv 生成）。
- 注意：spectral_power_labelled.npz（334MB）按 .gitignore 不入库，本机路径依赖；
  换机需先跑 s_stats/compute_s_stats.py（433 s）。
