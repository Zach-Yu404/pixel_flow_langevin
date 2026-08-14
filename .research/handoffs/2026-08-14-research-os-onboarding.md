# Handoff · 2026-08-14 · Research OS 接入 + 记忆初始化

**by**: claude ·  **phase**: onboarding → done

## 做了什么

1. `research-init` 装到 `~/research-init`；`research-setup-machine` 跑完（11 skills、3 plugins；MCP：context7(plugin)/arxiv/semantic-scholar/huggingface 连通；exa/wandb 缺 key 跳过；codex CLI 本机不存在）
2. `research-onboard` 建 `.research/` 骨架；`research-discover-paths` 自动发现 153 项大文件 → `local.yaml`（已补 dataset_root、MRI prior、celeba model、conda env、无 GPU 说明）
3. GitHub：复用已有 remote `Zach-Yu404/pixel_flow_langevin`（IP_branch 已 track origin），13 个标准 label 建好，watcher 启动（`research-watch status` 查看）
4. 记忆引导：吸收 `PRINCIPLE_MANUAL.md`、`github_project_local/{README,LOCAL_REPRODUCTION}.md` + docs/、`debug_IP4/memory_blur.md`、`IP_package/Experiments.md`、git 历史、Claude auto-memory（`msflow-migration-map` 已迁入并留指针）→ 填充 PROJECT / ARCHITECTURE / CURRENT / CONSTRAINTS / artifacts.yaml（c2img 与 MRI prior 已算 sha256）
5. git 身份配好（Zach-Yu404 / zeqiu.yu18@gmail.com，repo-local）

## 关键事实（后续 agent 必读）

- `github_project_local/` 是**独立嵌套 repo**（pixelflow-ip-benchmark），不进外层 git
- 外层工作区有大量未提交变更（`PixelFlow_train_code/` 删除、实验目录 untracked）——处置待用户决定
- 登录节点无 GPU，跑实验走 Slurm；复现契约见 CONSTRAINTS.md

## 待办 / 待用户

- 用户口头补充 in-flight 上下文（未提交修改的意图）
- codex CLI 未装 → degraded mode；装好后更新 STATE.yaml
- exa/wandb MCP 缺 API key
