# implement-algorithm4

state: working
type: implementation
owner: claude
issue: null
pr: null
created: 2026-08-21

## 【用户原始要求】
> 按照main.py调用utils.py的方式写main4.py并在utils.py补充posterior_sampling的算法，把page4的algorithm实现出来，结果放在results/alg4

## 已确认 Interpretation

- "page4 的 algorithm" = `PixelFlowICLR/Algorithm2/results/algorithm.4pdf.pdf` §7 的伪代码
  （草稿内部把它编号为 "Algorithm 1"；本项目按文件名称其为 **Algorithm 4**）。
  精读见 `references/2026-08-21-algorithm4-clean-endpoint-sampler.md`。
- "按照 main.py 调用 utils.py 的方式" = main4.py 自带严格 config 契约与 RUNNERS 分发，
  **所有数学都 import 自 `Algorithm2/utils.py`**，本文件只做配置、instrumentation 与作图
  —— 与 main.py 对 Algorithm 2 的关系完全一致。
- "在 utils.py 补充 posterior_sampling 的算法" = **纯新增** `run_posterior_sampling_alg4`，
  既有三个采样器（alg2 / wv / reg）一行未动。
- "结果放在 results/alg4" = `PixelFlowICLR/Algorithm2/results/alg4/`（与既有 `results/alg3` 同级）。

**这条要求同时是执行授权**：上一轮任务 `read-algorithm4-draft` 记的是
`execution_allowed: false`（用户当时只说"阅读"），本轮用户明确要求实现，故置 true。
本机 codex 不可用（single-agent degraded），据此同时置
`single_agent_execution_authorized: true`，`authorized_agent: claude`。

## Scope

包含：`utils.py` 新增 Alg-4 算子与采样器；`main4.py`；`config_alg4.json`；
CPU verify 审计；GPU full_ip 全量跑；S_prior 测量与敏感性对照。

不包含：改动 `run_posterior_sampling_alg2` / `_wv_` / `_reg_`（CONSTRAINTS 采样器纪律）；
改动 `main.py` / `main2.py` / `config.json`。

## 约束

- CONSTRAINTS §采样器纪律：永不 fork 既有采样器/算子 —— 遵守，全部 import。
- CONSTRAINTS §环境："登录节点无 GPU，必须走 Slurm" 是**旧集群**条款；本挂载机
  已记录为"无 Slurm，本地 4×A100 直跑"（CURRENT 2026-08-18），故本轮直接本地跑，
  用 `CUDA_VISIBLE_DEVICES` 避开他人占用的卡。
- `PYTHONHASHSEED=0` 硬契约（demo_runner 的 mask seed 走 hash()）。
- 本机 ceph 间歇 `Remote I/O error`（见 `context/facts.md`）。

## 执行记录

（见"结果"；branch IP_branch，未 push——本机无 gh/网络）

## 结果

（进行中）
