# implement-algorithm4

state: review
type: implementation
owner: claude
issue: null
pr: null
created: 2026-08-21

## 【用户原始要求】
> 按照main.py调用utils.py的方式写main4.py并在utils.py补充posterior_sampling的算法，把page4的algorithm实现出来，结果放在results/alg4

## 已确认 Interpretation

- "page4 的 algorithm" = `PixelFlowICLR/Algorithm2/results/algorithm.4pdf.pdf` §7 的伪代码
  （草稿内部把它编号为 "Algorithm 1"；本项目按文件名称其为 **Algorithm 4**）。
  精读见 `references/2026-08-21-algorithm4-clean-endpoint-sampler.md`。
- "按照 main.py 调用 utils.py 的方式" = main4.py 自带严格 config 契约与 RUNNERS 分发，
  **所有数学都 import 自 `Algorithm2/utils.py`**，本文件只做配置、instrumentation 与作图
  —— 与 main.py 对 Algorithm 2 的关系完全一致。
- "在 utils.py 补充 posterior_sampling 的算法" = **纯新增** `run_posterior_sampling_alg4`，
  既有三个采样器（alg2 / wv / reg）一行未动。
- "结果放在 results/alg4" = `PixelFlowICLR/Algorithm2/results/alg4/`（与既有 `results/alg3` 同级）。

**这条要求同时是执行授权**：上一轮任务 `read-algorithm4-draft` 记的是
`execution_allowed: false`（用户当时只说"阅读"），本轮用户明确要求实现，故置 true。
本机 codex 不可用（single-agent degraded），据此同时置
`single_agent_execution_authorized: true`，`authorized_agent: claude`。

## Scope

包含：`utils.py` 新增 Alg-4 算子与采样器；`main4.py`；`config_alg4.json`；
CPU verify 审计；GPU full_ip 全量跑；S_prior 测量与敏感性对照。

不包含：改动 `run_posterior_sampling_alg2` / `_wv_` / `_reg_`（CONSTRAINTS 采样器纪律）；
改动 `main.py` / `main2.py` / `config.json`。

## 约束

- CONSTRAINTS §采样器纪律：永不 fork 既有采样器/算子 —— 遵守，全部 import。
- CONSTRAINTS §环境："登录节点无 GPU，必须走 Slurm" 是**旧集群**条款；本挂载机
  已记录为"无 Slurm，本地 4×A100 直跑"（CURRENT 2026-08-18），故本轮直接本地跑，
  用 `CUDA_VISIBLE_DEVICES` 避开他人占用的卡。
- `PYTHONHASHSEED=0` 硬契约（demo_runner 的 mask seed 走 hash()）。
- 本机 ceph 间歇 `Remote I/O error`（见 `context/facts.md`）。

## 执行记录

- 新增 `PixelFlowICLR/Algorithm2/{main4.py, config_alg4.json}`；
  `utils.py` 纯新增 `make_endpoint_operator` / `clean_endpoint_solve` /
  `make_M_tau_den` / `measurement_residual` / `run_posterior_sampling_alg4`。
  既有采样器（alg2 / wv / reg）一行未动。
- commits：`069f529`（实现）、`2da86bd`（增量落盘 + S_prior 扫描）、本轮 report/记忆。
  branch `IP_branch`，**未 push**（本机无 gh/网络）。
- **入库口径**：`utils.py` 用 git plumbing 只入库了 Alg-4 的 380 行；
  工作区里此前遗留的 525 行 WIP（wv/reg 采样器 + `run_posterior_sampling_alg2`
  内循环那段 `d819043` 自述"neither beats the baseline"的 x_tau 重建）
  **仍未提交**，不是我该替用户决定的。入库前实测过 staged 版本能独立通过 verify。
- tests：`main4.py --mode verify` 17 项 dense float64 全过（A2 把 Prop. 7 验到 5e-15；
  A4 显示 τ=0 处 `λ_min(H₀ᵀH₀/σ²) = -4.6e-18` 而 `λ_min(M^den) = 1/s²`）。
- 运行：`measure_s2`（CPU）+ `full_ip` 5 任务 × 7 图 35 格全部完成
  + S_prior 敏感性 5 个 arm（box/junco）。

## 结果

【结果】完整报告见 `PixelFlowICLR/Algorithm2/results/alg4/report.md`。

**实现按草稿 §7 落地，三处结构差异都到位**：l.8 移出内循环、Block 1 中心是 x̂₁、
Block 2 是直接抽样 (23)（无 h₀）。`h0`/`ridge_rel`/`cg_max_iter_l14` 被双重拒绝而非静默忽略；
`S_prior` 无代码默认值。

**实现正确性有正面证据**：草稿 §7 自己的诊断 `‖x₀‖²/n ≈ 1` 在 1365 个 step 上
均值 **0.9991**（范围 [0.970, 1.022]）——Block 2 与 l.16 完全按设计工作。

**主要发现：全部亏损集中在 stage 3，五个任务无一例外**（含没有 null space 的 blur/SR）。
各 stage 末 `mse_x1`（7 图均值）：box 0.070/0.071/0.136/**0.922**；
random 0.012/0.022/0.077/**2.016**；gaussian 0.012/0.065/0.211/**2.295**；
motion 0.043/0.084/0.201/**2.341**；SR 0.013/0.032/0.193/**2.391**。
stage 2 在整段上是平的，stage 3 单调劣化。

**起点不能用 σ_τ→0 或 γ² 变大解释**：stage 3 内最大跳变在 step 0→1（box 3.5×、SR 5.0×），
而那一步 σ_τ 只从 0.400 降到 0.356、γ² 只从 0.0094 升到 0.0109。
后续单调恶化与两者一致，但"起始"与"延续"必须分开说。

**与 anchored Alg 2 同 λ 对照（box/junco，`s² = 1/λ`）**：
stage 0–2 上 Alg 4 持平或略胜（λ_eq=400：0.064/0.052/0.088 vs anchor 0.064/0.061/0.095），
**stage 3 差 10–28 倍**（1.140 vs 0.112）。

**实测 S_prior = 0.200/0.211/0.220/0.228（λ_eq 4.4–5.0）**，远弱于 anchor 扫描偏好的
λ=100–200。草稿 §8.2 建议"宁可取大"，但 `s²=1.0` 每个 stage 都不比实测值更好，
有用的方向是更小的 s²。**这不反驳草稿**——§8.2 正好警告该区间会"重建指标更好、
多样性丧失"，分辨两者需要 §8.6 的 ker(A_k) spread 诊断，**该诊断尚未跑**。

【需要用户决定】

1. **是否允许为诊断关掉 `g_bypass_stage3` 跑一格。** stage 3 是唯一 `G = I` 的 stage，
   H_τ / N_k 退化为标量、interpolant 失去全部低通作用，而 l.18 的 `U⁽¹⁾` 是 nearest 上采样；
   Alg 4 的 Block 1 又完全以 x̂₁ 为中心，没有 Alg 2 那个保留已抽状态的 `H_τᵀx_τ/σ_τ²` 项。
   这是目前最可能的机制但**未经检验**，而 CONSTRAINTS §采样器纪律明令
   "`g_bypass_stage3=True` 不关"——作为诊断而非调参是否可以破例，需要用户裁决。
2. 是否跑 §8.6 的多样性诊断（不跑则"更小 s² 更好"不能写进任何结论）。
3. 工作区 `utils.py` 那 525 行遗留 WIP 如何处置。
