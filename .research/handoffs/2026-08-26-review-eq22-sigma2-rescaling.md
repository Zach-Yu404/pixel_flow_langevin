# 2026-08-26 · review-eq22-sigma2-rescaling 交接

## 目标与状态

- 目标：只依据用户列出的实现事实，独立审查 Eq. (22) 整体乘
  \(\sigma_\tau^2\) 的代数、统计与 fp32/PCG 后果。
- 状态：review；Codex 独立归档稿已完成，未修改任何采样实现。
- execution_allowed: false。

## 已完成

- 逐项推导 \(\widetilde M,\widetilde b,\widetilde\zeta\)。
- 区分逐路径相同、仅分布相同及错误的
  \(\operatorname{Cov}(\widetilde\zeta)=\widetilde M\) 三种构造。
- 推导均值、协方差与条件数；给出一维统计反例和 PCG clamp 反例。
- 审计相对残差、三个绝对 clamp、Jacobi floor、spectral 探针及
  \(\sigma=10^{-8}\) 的具体量级。

## 修改文件

- .research/tasks/review-eq22-sigma2-rescaling.md
- .research/references/2026-08-26-codex-eq22-sigma2-rescaling-review.md
- .research/STATE.yaml、CURRENT.md、CONSTRAINTS.md
- 本 handoff

branch：无新 branch；commit：待本轮记忆提交；tests：无代码测试，完成 Markdown/公式人工复核。

## 假设与已知问题

- 技术推导只采用用户题面；\(\sigma_\tau>0\)，且原 \(S^{-1}\)、\(S^{-1/2}\) 运算有效。
- research-doctor --agents-only 10/10 通过。
- research-peer plan --agent codex 三次均因
  .git/research-os/peer/worktrees 的 Ceph Remote I/O error 失败，未生成结构化双 agent
  plan/compare artifact；任务因此保留 pending_review_by: claude。

## 下一步

- Ceph worktree 恢复后，只需对本 task 重新运行双方 plan/compare；不要重做 Codex 推导。
- 若用户随后授权实现，应另开 implementation task；正确噪声必须满足
  \(\operatorname{Cov}(\widetilde\zeta)=\sigma_\tau^4M\)，并同步审计所有绝对尺度保护。
