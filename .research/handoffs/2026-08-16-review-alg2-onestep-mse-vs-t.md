# 2026-08-16 review alg2-onestep-mse-vs-t

- 目标：审查 Issue #2 指定的 `458d5af..9ece9d6`。
- 当前状态：re-review 已 approve，任务完成。
- 完成了什么：核对 Algorithm 2 算子、line 11/14、伴随、S1/S2、跨任务缓存与 raw CSV；发现 3 处证据口径必改项。
- 修改文件：`.research/tasks/alg2-onestep-mse-vs-t.md`、`.research/STATE.yaml`、本 handoff。
- branch / base / head：`IP_branch` / `458d5af` / `9ece9d6`。
- tests / 核验：对 `alg2_mse.csv` 独立复算 pooled means 与 γ² score-error 差异；最大均值曲线相对差 10.94%，单图最大 19.60%。
- 结果：实现主体未发现与任务规格不一致；实验记录中的 3 处必改项均已修正，增量复审通过。
- assumptions：本次仅审查 Issue #2 指定范围，不将既有 `apply_G` 的自伴假设扩展为 scope 外阻塞项。
- known concerns：当前执行环境将 `.git` 挂载为只读，`git pull` 和最终 commit 可能被权限拦截。
- remaining work：无。
- recommended next step：按用户决定选择下一项研究任务。
