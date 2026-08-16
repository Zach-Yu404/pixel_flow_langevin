# 2026-08-16 review-onestep-mse-vs-t

- 目标：按 Issue #1 审查 `cba4e45..4167561` 的实验 1/1b 增量。
- 当前状态：request changes，等待 Claude 修正后 re-review。
- 完成内容：核对 base+diff、训练金字塔与 sampler utils、5 份配置消费参数、1400 行 raw MSE；
  已准备【Codex｜Review】结论并要求挂 `触发:needs-fix`。
- 修改文件：`.research/STATE.yaml`、`.research/CURRENT.md`、
  `.research/tasks/onestep-mse-vs-t.md`、本 handoff。
- branch / HEAD：`IP_branch` / `4167561`。
- tests：三个脚本 `py_compile` 通过；1400 rows = 280 完整实验键 × 5 tasks，跨任务 MSE 全等；
  CPU self-test 在当前受限环境未产出结果；未重跑 GPU job。
- 结果：实现数学路径未见 blocker；实验结论的单调性/一致胜出表述不符合 raw data，且仅由 MSE
  相等不能声称预测 tensor bitwise identical。
- assumptions：review 对象严格按 Issue #1，无 PR。
- known concerns：当前环境 `.git` 只读，无法 pull/commit；工作树另有大量用户改动，未触碰。
- remaining work：Claude 修正文案/证据口径后，Codex 仅 review 增量。
- recommended next step：修复上述两项并在 Issue #1 通知 re-review。
