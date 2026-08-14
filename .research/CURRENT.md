# 当前状态（人类可读，保持 ≤1 页）

更新时间：2026-08-14

## 正在进行

- （无明确进行中的任务——本次为 Research OS 接入 + 记忆初始化。上一次实质工作是 2026-07 的迁移验证与 benchmark 打包）

## 最近完成

- 2026-08-14：**全量记忆导入**——13 路并行读完项目内 130+ 份 .md(含前代 agent 的 SESSION_HANDOFF/global_memory)+ 关键代码/未提交 diff → 写入 experiments/{imagenet-5task-tuning, imagenet-baselines, multimodal-celeba-afhq-mri}.md、context/{authoritative-entries, contradictions-registry}.md、ARCHITECTURE/CONSTRAINTS 增补
- 2026-08-14：接入 Research OS（.research/ 初始化、GitHub label、watcher 启动、codex 装好并登录、dual-agent 就绪）
- 2026-07：旧机器 → CBIG 集群迁移完成；MRI prior checkpoint 恢复；`github_project_local/` 复现 benchmark 打包并验证（AFHQ SR bit-exact；ImageNet/CelebA batch-parity bit-exact；MRI 归一化契约验证）
- 2026-04~05（迁移前，见 git log 与 debug_IP4）：五任务 ImageNet 调参收敛（`memory_blur.md` 有 best config 与禁区）、MRI val_final_8 参考协议跑完、`IP_package/Experiments.md` 写成 paper 风格

## 下一步

- （待用户确认研究方向：例如跑 `--full` 复现、补 baseline 表格、写 paper draft、或继续某个任务的调参）
- 处理外层 repo 未提交状态（见"阻塞"第 2 条）

## 阻塞 / 需要用户决定

1. **in-flight 工作已从 diff 复原(机械意图)**：未提交改动 = random_inpainting **第二阶段 OAT sweep**(数据契约 .pt→PNG + stage2_configs + 断点续跑)、train.py 训练韧性(resume/epoch-ckpt/epoch-eval)、CFG 空类 token 泛化、legacy 采样配置开 CFG=2.0(详见 ARCHITECTURE"工作区未提交改动"节)。**待用户确认**:stage2 sweep 跑到哪一步了、动机实验是什么(MRI 先验训练?)
2. **外层 repo 工作区大量未提交变更**：`PixelFlow_train_code/` 整目录删除（−5226 行）、`IP_package/`、`debug_IP4/` 多数实验目录 untracked。是否按现状提交（大文件已被 .gitignore 排除）由用户决定
3. ~~codex CLI 待装/待登录~~ 已解决（2026-08-14）：v0.147.0 装于 `~/.local/bin/codex`，
   ChatGPT 账号已登录，5 个 MCP server 双侧注册，STATE 已切回 `dual-agent`
