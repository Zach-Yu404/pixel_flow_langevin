# 2026-08-16 rereview-onestep-mse-vs-t

- 目标：完成 Issue #1 的增量复审，最终范围 `526ca24..458d5af`。
- 当前状态：done；Codex approve。
- 完成内容：核对两轮增量 diff，独立复算 `all_mse.csv` 的胜负、单调性与跨任务 MSE 一致性，并验证最终 README 模板修复。
- 修改文件：`.research/STATE.yaml`、`.research/CURRENT.md`、`.research/tasks/onestep-mse-vs-t.md`、本 handoff。
- branch / HEAD：`IP_branch` / `458d5af`。
- tests：受影响脚本 `py_compile` 通过；最终模板与磁盘 README 均为 1031 字节且逐行无 diff；`git show --check 458d5af` 通过。
- 结果：README 模板已限定为 MSE rows 一致且注明未直接比较 prediction tensors；最终复审通过。
- assumptions：无 PR，review 目标严格按 Issue #1；未重复 GPU 实验。
- known concerns：原工作区 `.git` 只读，使用同一远端的临时 clone 提交 canonical memory；未触碰工作区其他改动。
- remaining work：无。
- recommended next step：按项目研究优先级选择下一项任务。
