# 未解决问题

（协议 §1–§3 已闭合；以下为双方确认的**非阻塞**遗留项，不影响当前两主臂结论）

- [适用范围] batch>1、假想 sigma_scaled S、spectral fp64 RHS/MC 未完整覆盖
  ——扩展开关适用范围或设默认前需补。
- [nit] `c * apply_S_inv(x)` 未融合：极小谱功率/S∝σ² 时大中间张量仍会先
  形成；如需该组合应给 SOperator 提供融合操作（标量 (c/s2)x、谱域 (c/P)X）。
- [nit] blur 的 fp64 恒等检查 A 项回退 fp32（conv2d float32 权重）——
  [BL] 不是纯 fp64 证明。
- [nit] 归档脚本未单独断言非正 σ 必抛异常（guard 已实现，缺单测）。
- [设计债，缩放前即存在] pcg_solve 的 `.clamp(min=...)` 会把负内积静默改正、
  不区分 NaN/Inf；长期应换相对 breakdown 判据——会改 baseline 行为，
  需另立任务由用户裁决。
- [用户裁决] 是否在某些场景启用 eq22_sigma2_scale（共识：不改默认）。
