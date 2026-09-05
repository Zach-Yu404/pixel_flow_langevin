# 2026-09-05-gamma2-all-default

任务：gamma2-tables-test / compute-gamma2-valset
状态：已采纳（用户直接指令）

## 【共识】
目标：Algorithm 4 的 γ²(k,τ) 表改用 ImageNet-val 全量 50k 的测量表 gamma2_stats/gamma2_all.json。

确认方案：
- config_alg4.json paths.gamma2_table: {HERE}/gamma2_meas_alg4.json → {HERE}/gamma2_stats/gamma2_all.json
  （同 table 格式，key str(round(τ,6)) 与采样器查表逐点匹配；不用 per-class 表）。
- 旧 7 图表 gamma2_meas_alg4.json 保留在仓库作为论文可比口径的参照。
- 用户原话：「现在gamma2我用gamma2_all的，现在给我一个finalize的版本和我check一下」。

是否需要实现：是（已实现，config 一行）

## 依据
gamma2-tables-test：γ² 表对结果不敏感（全噪声单样本 −0.03 dB、MMSE10 −0.01、无噪持平/略优，GPU 噪声地板 0）；
7 图表对测试图 100% 样本内，val 表是无泄漏的诚实估计（早期 τ 高 25–55%）；per-class 表方差大无收益。

## 参与
【Claude｜实现+验证】；Codex 未参与。
