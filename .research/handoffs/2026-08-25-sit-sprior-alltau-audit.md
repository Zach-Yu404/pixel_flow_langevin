# Handoff 2026-08-25 · S_it 程序 + τ≥0.78 全因审计 + S_prior 对比 + 双 agent 恢复

**目标**：稳定 Algorithm 4 的 late-τ 行为并确定 S 构造；恢复双 agent。
**状态**：三个任务 done（tune-sit-schedule / diagnose-late-tau-all-causes /
compare-s-prior-methods），全录见各 task 文件与 CURRENT.md 08-25 节。

## 完成

- 基线定为 per-τ γ² + S_it=2（config 已改）；当前最优 spectral S：hole 0.1491
  （4 种子 0.1411±0.005），obs 0.0026。
- 全因审计判定：算法固有（b∝σ→0 的移除坍塌 + 单位增益随机游走），实现忠实；
  草稿级缺陷：(19) 无 S⁻¹ 收缩 vs (12) 有 —— 代理不自洽。
- SOperator 接口进 utils（标量路径逐位回归通过）；s_prior_methods.py 双臂入口；
  六臂对比后按用户指令清理至 pooled_junco + spectral；全部产物数值复核通过。
- 残余诊断关键发现：pooled_junco 的 x̂₁@f31–f33 = hole 0.083（超 Alg2 0.102），
  被 f34–f39 毁掉 —— 读出点（b≈1 处读 x̂₁）是候选最大免费杠杆，待用户裁决。
- 基础设施：codex 0.149.1 装好登录（danger-full-access 用户授权），handshake 全绿，
  dual-agent 恢复；预检上游化 research-init agent/dual-agent-preflight。

## 修改文件（全部未提交，与既往未提交改动同批待用户定 SHA）

utils.py（per-stage num_langevin、SOperator、Block-1 两行）、s_prior_methods.py（新）、
config_alg4.json（γ² 表指回 per-τ、num_langevin=2）、
results/alg4_box_stage3_diagnosis/（S_it 六档+调度对比）、
results/alg4_box_s_prior_methods/（双臂全套）。

## 进行中（本 handoff 写就时刚接到）

- **x0carry-ablation**（用户 2026-08-25 晚指令）：为 Alg4 增加 x0_carry_mode
  （sampled 默认不变 / endpoint_raw = N⁻¹z 闭式 / endpoint_gamma = (18) 式 CG），
  先做 oracle 校验（d_exact 替换 v_out 时 endpoint_raw 须精确还原 xi_0），
  box/junco、S_it=2、pooled_junco+spectral 两 S，输出 results/alg4_x0carry_ablation/。

## 已知风险 / 假设

- 所有 S 统计 DEMO_ONLY/LEAKAGE；spectral 正式采用前须训练集重测。
- 逐帧数值 ±20% 轨迹敏感度（同机同 GPU 可 4 位小数复现）。
- γ² 查表静默 positional fallback（5 处）在改 grid/表时会咬人。

## 推荐下一步

x0carry ablation → 读出点实验（x̂₁@b≈1）→ 定 SHA、推送、发起 codex 精确复审。
