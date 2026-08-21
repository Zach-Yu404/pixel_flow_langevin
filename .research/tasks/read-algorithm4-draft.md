# read-algorithm4-draft

state: done
type: research
owner: claude
issue: null
pr: null
created: 2026-08-21

## 【用户原始要求】
> 把这个项目接入 Research OS，工具在 https://github.com/Zach-Yu404/research-init，
> 接入后初始化记忆，阅读/CBIG-Standard-ECE/Zach/MSFlow下的代码，把https://github.com/Zach-Yu404/pixel_flow_langevin导入记忆，仔细阅读/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/results/algorithm.4pdf.pdf

## 已确认 Interpretation

四件事，其中前两件是**确认/补齐**而不是从零做：

1. **接入 Research OS**：MSFlow 早在 2026-08-14 已接入；本次跑
   `research-upgrade-project` 确认已在 bundle v2 且幂等，`research-doctor` 25 通过。
2. **初始化记忆**：全量导入同样在 2026-08-14 完成（13 路并行读 130+ .md）。
   本次做的是**增量补齐**：记忆停在 2026-08-19，而 08-19～08-21 的工作（尤其
   `results_reg_alg2` 与新草稿）尚未进入 CURRENT/STATE。
3. **导入 pixel_flow_langevin**：`MSFlow` 的 `origin` **就是** 该 repo，
   本机无网络/无 gh，因此从本地 clone 采集仓库事实 → `context/facts.md`。
4. **精读 algorithm.4pdf.pdf**：15 页全文逐页读完
   → `references/2026-08-21-algorithm4-clean-endpoint-sampler.md`。

## Scope

包含：Research OS 状态核验与升级；代码阅读（PixelFlowICLR 全部 + Algorithm2 数学核心）；
仓库事实采集；PDF 精读并与现有实现逐条对照；记忆写入。

**不包含：实现 Algorithm 4。** 用户只说"仔细阅读"，未要求实现，
故本任务 `execution_allowed: false`（见 STATE.yaml）。要动代码需用户明确开绿灯。

## 约束

- CONSTRAINTS §采样器纪律：永不 fork `run_posterior_sampling`／算子。
  Alg.4 若实现，应与 `run_posterior_wv_sampling_alg2` / `run_posterior_reg_sampling_alg2`
  同样走"纯新增函数"的路子，不动 `run_posterior_sampling_alg2`。
- 本机（挂载服务器）无 gh、无 codex → 事实上 single-agent degraded，
  记忆只能 commit 不能 push。

## 执行记录

- `research-upgrade-project /CBIG-Standard-ECE/Zach/MSFlow` → bundle v2，幂等（无新增 diff）。
- `research-doctor` → 25 通过 / 2 警告（gh 未登录、codex 未安装）/ 1 失败（gh 未安装）。
- 读代码：`PixelFlowICLR/{README,onestep_mse_vs_t,consolidate_results,onestep_visual}`、
  `Algorithm1/{main_alg1.py,config_alg1.json}`、
  `Algorithm2/{main.py,main2.py,utils.py,measurement.py,config.json,gamma2_meas.json}`、
  `Algorithm2/test/`（含 `results_reg_alg2/report.md` 与
  `results_reg_alg2/regularization_final/report.md`）。
- 读草稿：`results/algorithm.4pdf.pdf`（15 页）与 `results/modified.pdf`（CLIMB-Flow，
  确认 Alg.1/2/3 编号谱系）。
- 采集仓库事实：branch / unpushed / 工作区 diff。

## 结果

【结果】

**A. 草稿定位。** algorithm.4 是 "A clean-endpoint posterior sampler"。它把耦合从
flat-prior 高斯换成真实的 denoising conditional，`p_τ` 因此相消，
**step size 与先验平滑同时消失**，两个 conditional 都是精确抽样。
代价是唯一一个近似：把 `p(x_1|x_τ)` 当高斯，需要它的协方差 C。
核心贡献是给 C 一个**模型自带的上界**（Lemma 3，Brascamp–Lieb）和一个
**在整个 schedule 上正定、高斯先验下精确、无可调参数**的取法
`C⁻¹ = H_τᵀH_τ/σ_τ² + S⁻¹`（(12), Prop. 4）。

**B. 与本项目实现的三条硬对应（详见 references note §Relation）**

1. `score_solve` 就是草稿 (18)；`gamma2_meas.json` 就是 (20)。**已完全一致。**
2. **`run_posterior_reg_sampling_alg2` 里那个经验调出来的 Tweedie anchor `λ`，
   就是 (12) 的 `S⁻¹`：`λI = S⁻¹` ⟺ `s² = 1/λ`。**
   草稿说这个量应当被**测量**（该 scale 训练图的逐像素方差／功率谱），不是被搜索。
   我们扫出的最优 λ≈100 对应 s²=0.01；图像在 [-1,1]，assumption-free 上界是 s²≤1
   （λ≥1）。**λ=100 落在草稿 §8.2 明确警告的"低估 S"一侧**——那会把 Block 1
   收缩向去噪估计，重建指标变好而样本多样性丧失，且该失败在 hole/visible MSE 上看不出来。
3. 但**代码是"加"、草稿是"换"**：现有 `b = Aᵀy/η² + H_τᵀx_τ/σ_τ² + λx̂_1`；
   草稿 (22) 是 `b = Aᵀy/η² + (H_τᵀH_τ/σ_τ² + S⁻¹)x̂_1`。
   工作区未提交的 `data_rhs_matchx1` 恰好是 (22) 缺了 `S⁻¹x̂_1` 的那一半。

**C. 两个长期困扰按构造被消除**（如果 Alg.4 成立）

- **τ=0 的 ridge**：Prop. 4(a) 使 M 处处正定，`ridge_rel`/`power_iter_norm`/
  `epsilon`/`√ε ξ_eps`/`cg_max_iter_l14=200` 全部不再需要。
  与已记录的 "τ=0 ridge 不是把洞压在 1.0 的原因"（`5734af2`）互相印证。
- **h₀**：Block 2 变成直接重加噪 (23)，没有步长。
  `regularization_final` 那 48 格网格的头号结论是"h₀ 主导、太小则 x₀ 冻住"——
  这个病理在 Alg.4 里不存在。

**D. 草稿自身的两个问题（建议反馈作者）**

- Table 2 三列表头字面是 `Alg. 1 | Alg. 2 | Alg. 1`，第三列应是本文的算法。
- §8.2 的 Popoviciu 界 `s²≤1/4` 假设像素在 [0,1]；本项目在 [-1,1]，对应 `s²≤1`。

**E. §8.4 的自陈风险，对我们特别相关**：Alg.4 的 Block 1 中心是 x̂_1，
**全权重承担 velocity error**，而 interpolant-coupled 的中心 x_τ 是精确的。
优势随 γ 增大而减小并会反转。我们的 γ² 表显示 stage 3 末端冲到 0.145（全表最大），
正好在"末端决定重建"的位置 —— **实现前应先用 γ² 表经验定位 crossover。**

【需要用户决定】

1. **是否实现 Algorithm 4？** 现在 `execution_allowed: false`（用户只说了"阅读"）。
2. 若实现，第一步建议是**先测量 S_prior**（4 个 stage 各测一次逐像素方差／功率谱），
   然后把测得的 s² 直接放到现有 λ 曲线上看落点 —— 这一步不需要新采样器，
   用现有 `results_reg_alg2` 的扫描即可证伪第 B.2 条预测。
