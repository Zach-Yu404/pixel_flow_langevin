# 2026-08-21-research-os-verify-and-algorithm4-read

【Claude｜交接】

目标：按用户原话——接入 Research OS（工具 github.com/Zach-Yu404/research-init）、
初始化记忆、阅读 MSFlow 下的代码、把 github.com/Zach-Yu404/pixel_flow_langevin 导入记忆、
仔细阅读 `PixelFlowICLR/Algorithm2/results/algorithm.4pdf.pdf`。

当前状态：四项全部完成，记忆已写入并 commit。**未 push**（本机无 gh / 无 GitHub 网络）。
未实现任何算法代码——用户只要求阅读。

已完成：
- Research OS 核验：`research-upgrade-project` → bundle v2 且幂等（无新增 diff）；
  `research-doctor` → 25 通过 / 2 警告 / 1 失败，失败与警告全部是本机 gh/codex 缺失，
  项目侧 15 项全 ✅。**项目 08-14 就已接入，本次是确认而非首次接入。**
- 仓库导入：确认 `pixel_flow_langevin` **就是本工作区的 origin**；采集分支/未 push
  commit/工作区 diff 事实 → `context/facts.md`。
- 代码阅读：`PixelFlowICLR/` 全部（README、onestep_*、consolidate）、
  `Algorithm1/main_alg1.py`、`Algorithm2/{main,main2,utils,measurement}.py` + 两个 config +
  `gamma2_meas.json`、`Algorithm2/test/`（含两份 report.md）。
- 草稿精读：`algorithm.4pdf.pdf` 15 页逐页读完；另读 `modified.pdf` 确认 Alg.1/2/3 谱系。

修改文件：
- 新增 `.research/references/2026-08-21-algorithm4-clean-endpoint-sampler.md`（精读 note）
- 新增 `.research/tasks/read-algorithm4-draft.md`
- 新增 `.research/context/facts.md`
- 更新 `.research/CURRENT.md`（补齐 08-19～08-21，新增本轮两节）
- 更新 `.research/STATE.yaml`（memory_version 28→29；mode → single-agent-degraded）
- 更新 `.research/ARCHITECTURE.md`（草稿谱系与代码映射、环境事实）
- Research OS bundle v2 升级产物：`CLAUDE.md`、`AGENTS.md`、`.research/RULES.md`、
  `.claude/settings.json`、`.gitignore`、`.research/{bin,policy.json,system.json}`
- **未动任何源码**（`PixelFlowICLR/Algorithm2/utils.py` 的 +525 行未提交改动保持原样，未 commit）

branch：`IP_branch`
commit：见下（本轮记忆 commit）
tests：无（纯阅读 + 记忆任务）

结果：

**Algorithm 4 = clean-endpoint sampler。** 把耦合从 flat-prior 高斯换成真实的
denoising conditional ⟹ learned marginal `p_τ` 相消 ⟹ **step size 与先验平滑同时消失**，
两个 conditional 都是精确抽样（Lemma 8/9，randomize-then-optimize）。
唯一近似：`p(x_1|x_τ) ≈ N(x̂_1, C)`。核心贡献是给 C 一个模型自带的上界
（Lemma 3，Brascamp–Lieb：`Cov ⪯ σ_τ²H_τ⁻²`）和一个无可调参数的取法
`C⁻¹ = H_τᵀH_τ/σ_τ² + S_prior⁻¹`（Prop. 4：处处正定、admissible、高斯先验下精确）。

**与本项目最有价值的对应：我们经验调出来的 Tweedie anchor λ 就是 S_prior⁻¹。**
`λI = S_prior⁻¹` ⟺ `s² = 1/λ`。草稿要求这个量**被测量**（该 scale 训练图的逐像素方差
或功率谱），不是被搜索。我们扫出的最优 λ≈100 ⟹ s²=0.01；图像在 [-1,1]，
assumption-free 上界是 s²≤1。**λ=100 落在草稿 §8.2 明确警告的"低估 S_prior"一侧**，
其后果是 Block 1 收缩向去噪估计——**重建指标反而变好、样本多样性丧失，
且该失败在我们现在报告的 hole/visible MSE 上看不出来**。

assumptions：
- 把仓库文件名 `algorithm.4` 当作 Algorithm 4；草稿内部自编号为 "Algorithm 1"，
  且 Table 2 表头是 `Alg. 1 | Alg. 2 | Alg. 1`（排版 bug），已在 note 里消歧。
- "导入 pixel_flow_langevin"解释为从本地 clone 采集仓库事实——因为它就是 origin，
  且本机无网络。若用户本意是别的东西（例如某个未在本地的分支/release），需重来。

known concerns：
- **本机 single-agent degraded**：无 gh、无 codex、无 GitHub 网络。记忆只能 commit
  不能 push；`IP_branch` 已积压 **9 个未 push commit**；Issue #3 的增量复审在本机推不动。
- **`git status` 全量扫描在本机稳定失败**（ceph `Remote I/O error`，5/5 次）。
  `git status -uno` 可用。依赖全量 `git status` 的自动化（含 Stop gate）可能非确定性失败。
- **工作区 baseline 被污染**：`utils.py` 的未提交 diff 里，
  `run_posterior_sampling_alg2` 内循环 l.11 后插了 5 行重建 x_tau，
  而 HEAD commit `d819043` 自述该改动 "neither beats the baseline"。
- 草稿 §8.4 自陈：Alg.4 的 Block 1 中心是 x̂_1，**全权重承担 velocity error**，
  优势随 γ 增大而减小并会反转。我们 γ² 表 stage 3 末端 = 0.145（全表最大）。

remaining work：
- 实现 Algorithm 4（**未授权**，`execution_allowed: false`）。
- 测量 `S_prior`（4 个 stage 各一次）——这是实现前置数据，且本身不需要新采样器。
- 把未 push 的 9 个 commit 推上去（需 gh / 网络）。

recommended next step：
**先做一件不需要写采样器、也不需要网络的事**：按 4 个 stage 各测一次训练图的逐像素方差
（或 (13) 的功率谱）得到 `S_prior`，把 `s² = 1/λ` 直接放到 `results_reg_alg2` 已有的
λ 曲线上看落点。这一步就能证伪或坐实"λ≈100 = 低估 S_prior、正在牺牲多样性换指标"这条预测，
并顺带补上草稿 §8.6 要求报告的三个量之一。
若要判定多样性是否已经丧失，再加 §8.6 的诊断：固定 y 重复采样，量 `ker(A_k)` 内的 spread。
