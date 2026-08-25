# Shared Facts

只放事实，不放分析（分析各 agent 独立做）。追加式。

## 2026-08-21 · claude · 仓库导入（github.com/Zach-Yu404/pixel_flow_langevin）

- Shared Fact: **本 repo 的 `origin` 就是
  `https://github.com/Zach-Yu404/pixel_flow_langevin.git`**（`git remote -v`）。
  用户要求"导入"的 GitHub repo 与本工作区是同一个 repo，不是外部资料。
- Shared Fact: 分支两条。`IP_branch`（当前 HEAD，`d819043`，
  **ahead of `origin/IP_branch` by 9 commits**）与 `main`（`6fcd539`，与 origin 同步）。
  `main..IP_branch` = 56 commits；`IP_branch` 全history 58 commits，
  首个 commit `581ab11` 2026-04-14 "Initial commit: PixelFlow + training code"。
- Shared Fact: 9 个未 push 的 commit（新→旧）：
  `d819043` Test both x_tau rebuild variants after line 11; neither beats the baseline /
  `39f61c3` Diagnose main.py against main2.py on one matched run /
  `5734af2` research: the tau=0 ridge is not what holds the hole at 1.0 /
  `73e7853` research: record what the H_tau_inv diagnostic settled /
  `f6c3b58` Guard apply_H_tau_inv before the divide, not after /
  `437151f` Implement Algorithm 3 from the modified draft and test it against Algorithm 2 /
  `93fa7d2` Sweep gamma^2, and correct the loss curves to compare like with like /
  `ca733a5` Answer whether gamma^2 helps: it does not, at any of the 37 points /
  `c6c26aa` Add the p.20 implied clean image and test the score solve in isolation
- Shared Fact: `github_project_local/` 是**独立嵌套 repo**，不在上述历史内（PROJECT.md 已载）。

## 2026-08-21 · claude · 本机环境（挂载服务器，路径见 `local.yaml` 的 `external.mount_root`）

- Shared Fact: **`gh` 未安装、`codex` 未安装**（`command -v` 均为空；
  `PATH` 含 `~/.local/bin`，其中只有 `claude` 一个符号链接）。
  HOME 见 `local.yaml` 的 `external.home`，不与旧机共享。
  ⟹ 本机事实上 **single-agent degraded**，且**记忆只能 commit，无法 push**。
  `git fetch` 报 `could not read Username for 'https://github.com'`。
  - 【2026-08-25 更新】codex 0.149.1 已自行安装（`npm install -g --prefix ~/.local
    @openai/codex`，/usr 前缀 root 所有装不进），伴生二进制 codex-code-mode-host +
    codex-resources/bwrap 已从 GitHub release rust-v0.149.1 补装到 `~/.local/bin`。
    登录有效（`codex login status` = ChatGPT；凭证 `~/.codex` 本就在此 home）。
    ⟹ “HOME 不与旧机共享”对凭证不成立：`~/.codex`、`~/.claude` 都在且可用。
  - 【2026-08-25】**本机 bwrap 沙箱不可用**：`unshare --user` 成功但
    `--map-root-user`（uid_map 写入）被拒 ⟹ bwrap "setting up uid map: Permission
    denied" ⟹ `codex exec` 在 read-only/workspace-write 任何沙箱模式下都无法读文件/
    执行命令（纯 LLM 回复正常）。唯一出路 = `-s danger-full-access`。
    **用户已授权（全局）**：`~/.codex/config.toml` `sandbox_mode = "danger-full-access"`
    + `~/.bashrc` `export RESEARCH_CODEX_SANDBOX=danger-full-access`（research-peer/
    orchestrator 显式 flag 用它，否则 workspace-write 会覆盖 config）。
  - 【2026-08-25】codex ChatGPT 登录态曾在当天 ~14:32 失效（token 刷新失败被清，
    VSCode ChatGPT 扩展侧同步掉线）；用 `codex login --device-auth`（设备码，无需
    端口转发）由用户在浏览器授权恢复。再失效时同法处理。
  - 【2026-08-25】research-doctor（新版 --agents-only）三级 handshake **全绿**：
    ①claude -p 返 nonce ②codex exec 返 nonce ③codex 读回 claude 写入的共享文件。
    ⟹ **dual-agent 恢复**（STATE.yaml memory_version 39）。
- Shared Fact: `research-doctor`（2026-08-21）= **25 通过 / 2 警告 / 1 失败**。
  失败 = gh 未安装；警告 = gh 未登录、codex 未安装。项目侧 15 项全部 ✅。
- Shared Fact: **`git status` 的 untracked 扫描在本机稳定失败**：
  `fatal: Invalid path '.../.git': Remote I/O error`，连续 5 次重试全败。
  `git status -uno` 正常。根因是 ceph 挂载上若干目录读取返回 `Remote I/O error`，
  已观察到的有 `PixelFlowICLR/Algorithm2/__pycache__`、
  `PixelFlowICLR/Algorithm2/test/__pycache__`、`.git/research-os/sessions`、
  `.git/logs/refs`（含 `heads`/`remotes`）以及 baselines 下多个嵌套 `.git`。
  这些是**间歇性**的（同一路径 `ls` 有时成功有时失败），不是权限问题。
  ⟹ 任何依赖 `git status` 全量扫描的自动化（含 Research OS Stop gate）在本机可能非确定性失败；
  用 `-uno` 或显式路径的 `git add` 可以绕开。
- Shared Fact: `research-init` 工具的本地 clone 在
  `local.yaml` 的 `external.research_init_clone`（**不是** HANDBOOK 说的 `~/research-init`），
  分支 `agent/efficient-dual-agent`，HEAD `dd4e766`，领先 `origin/main` 3 个 commit（未 push）。

## 2026-08-21 · claude · 工作区未提交状态

- Shared Fact: `git diff --stat` = 6 个文件。其中 5 个是 Research OS bundle v2 升级产物
  （`CLAUDE.md`、`AGENTS.md`、`.research/RULES.md`、`.claude/settings.json`、`.gitignore`）。
- Shared Fact: 第 6 个是 **`PixelFlowICLR/Algorithm2/utils.py`，+525 行纯新增**：
  `data_rhs_matchx1`、`make_M_tau_wv`、`data_rhs_wv`、`_x1_hat_diag`、
  `direct_estimate_x1`、`run_posterior_wv_sampling_alg2`、`_make_anchor_P`、
  `run_posterior_reg_sampling_alg2`。
- Shared Fact: **同一个 diff 里 `run_posterior_sampling_alg2` 的内循环也被改了**（l.11 之后插入 5 行）：
  用 `direct_estimate_x1` 从 v 重算 `x1_hat`，再用 `x_tau ← H_τ·x1_hat + σ_τ·x0`
  **重建 x_tau** 后才进 Block 1。
- Shared Fact: HEAD commit `d819043` 的正文明确写着两件事：
  ① "Test both x_tau rebuild variants after line 11; **neither beats the baseline**"
  （变体 A = `H·x1_hat + σ·x0_hat`，结束于 hole 664.6；变体 B = 用 (51) 求 x1_hat 配
  state 自己的 x0，结束于 NaN；baseline 是 1.047）；
  ② "**utils.py is back to its committed state.**"
  ⟹ 工作区现在这 +525 行是那次 commit **之后**重新加上去的。
  工作区的插入用 `direct_estimate_x1` 配 state 自己的 x0，
  **与 A、B 都不完全相同**（A 用 x0_hat，B 用 (51)），是第三种组合，其结果未见记录。
  ⟹ 工作区里的"baseline 采样器"**不是 baseline**。

## 2026-08-22 · claude · results/ 盘点与清理

- Shared Fact: `PixelFlowICLR/Algorithm2/results/` 下 15 个目录**全部**能追溯到某个 commit
  或某个 `.research/tasks/` 文件，**没有孤儿目录**。
  逐条归属见本轮 handoff；`inverse_diagnosis`(32M) 是 commit `39f61c3`/`d819043`
  「Diagnose main.py against main2.py」的证据且部分已 git 跟踪。
- Shared Fact: **唯一可安全删除的大块是 PNG 帧转储**，全部由本轮 Alg-4 工作产生：
  `alg4/frames_tmp`(19M, 80 帧) 与 `alg4/s2_sens/*/frames_tmp`(5 臂 × 40 帧, 42M)。
  未被任何记忆/报告引用、未被 git 跟踪（`*.png` 已 gitignore）、由
  `main4.py --mode full_ip` 可重新生成。已删除，`results/alg4` 从 **61M → 866K**，
  全部 CSV/JSON/report 完好。
- Shared Fact: `algorithm2_diagnosis`(8.0M) 与 `prior_injection`(14M) 被
  `tasks/algorithm2-line15-diagnosis.md`、`tasks/prior-injection-b-tau.md` 引用，
  但**此前 0 个文件进 git** —— 结论只存在于这块间歇性 EIO 的 ceph 盘上。
  已把两者的 `report.md` + `comparison.csv`（共 ~70KB，其余 21MB 是 gitignored 的 PNG）入库。
- Shared Fact: 远端 `origin/IP_branch` 与本地**已分叉**：远端有 3 个 workflow/entrypoint
  基础设施提交（PR #4/#5/#6，`5ed3313`/`e7d21c3`/`d25ec85`），本地有 20 个研究提交。
  备份因此推到**新分支** `agent/alg4-clean-endpoint-2026-08-22`（纯增量，未动 `IP_branch`）。
  远端另有 `sync/algorithm2-d819043`、`agent/research-workflow-v7` 等分支。
  合并 IP_branch 需要用户决定，未执行。
