# 当前状态（人类可读，保持 ≤1 页）

更新时间：2026-08-14

## 正在进行

- （无明确进行中的任务——本次为 Research OS 接入 + 记忆初始化。上一次实质工作是 2026-07 的迁移验证与 benchmark 打包）

## 最近完成

- 2026-08-14：接入 Research OS（.research/ 初始化、GitHub label、watcher 启动、记忆引导）
- 2026-07：旧机器 → CBIG 集群迁移完成；MRI prior checkpoint 恢复；`github_project_local/` 复现 benchmark 打包并验证（AFHQ SR bit-exact；ImageNet/CelebA batch-parity bit-exact；MRI 归一化契约验证）
- 2026-04~05（迁移前，见 git log 与 debug_IP4）：五任务 ImageNet 调参收敛（`memory_blur.md` 有 best config 与禁区）、MRI val_final_8 参考协议跑完、`IP_package/Experiments.md` 写成 paper 风格

## 下一步

- （待用户确认研究方向：例如跑 `--full` 复现、补 baseline 表格、写 paper draft、或继续某个任务的调参）
- 处理外层 repo 未提交状态（见"阻塞"第 2 条）

## 阻塞 / 需要用户决定

0. **远端有一个待覆盖的误提交**（2026-08-14）：ed4df6b 被误用 `git commit -a` 推上了 origin/IP_branch，
   混入了未决的实验区改动。本地已修正为 a7db725（只含 ARCHITECTURE.md 修改），但 force push 被权限
   拦截。二选一：**(a)** 在项目目录跑 `git push --force-with-lease origin IP_branch` 覆盖远端（推荐，
   恢复"实验改动待决"状态）；**(b)** 若愿意接受那次实验改动入库，跑 `git reset --hard origin/IP_branch`
   回到远端版本（会把本地 a7db725 之后的记忆提交一并对齐，需重新提交 CURRENT.md 的本节更新）
1. **只在用户脑中的上下文**：接入时未获得"上次做到一半"的口头信息——如果有 in-flight 的实验/改动（例如 `ms_posterior_sampling.json`、`pipeline_pixelflow.py`、`train.py` 的未提交修改意图），请告知以补进记忆
2. **外层 repo 工作区大量未提交变更**：`PixelFlow_train_code/` 整目录删除（−5226 行）、`IP_package/`、`debug_IP4/` 多数实验目录 untracked。是否按现状提交（大文件已被 .gitignore 排除）由用户决定
3. codex CLI 已安装（2026-08-14，v0.147.0，GitHub releases musl 二进制 → `~/.local/bin/codex`），
   但**尚未登录**——需用户本人跑 `codex login`（涉及账号，agent 不能代办）。登录后把 STATE.yaml
   改回 `mode: dual-agent` 并清掉 unavailable_agent
