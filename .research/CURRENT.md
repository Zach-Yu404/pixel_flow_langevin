# 当前状态（人类可读，保持 ≤1 页）

更新时间：2026-08-22

## 2026-08-24：Alg 4 收敛已修复（sigma_min=0.39，b≥1 截止）+ measurement 下界分析

- **修复**：`sigma_min` 0.01→0.39（既有 key）。判据：只在 (19) 的 velocity 权重
  b = N·σ_τ/(N²+γ²h²) ≥ 1 的步上运行（σ_τ ≥ N_k ≈ 0.4）；0.39 = 0.4 − float32 容差
  （保住 f30——唯一仍收缩、携带全分辨率数据项的 stage-3 帧）。
- **验证**：box/junco hole 3.53→**0.288**（12×）、resid 0.994、stage-3 增长 1.00×、NFE 270。
  **4 seeds**：出界 99.2%→**2.88%**、洞内 std 0.453（GT 0.478）、跨 seed spread 0.356
  → **多样性存活**，此前"无法判定"的 §7.7 已可回答。
- **起点更正**（用户观察触发）：恶化从 **f26** 开始（b<1 首帧），f31 只是最大跳变；
  b≥1/b<1 把 40 帧二分（x̂₁ 洞内下限中位 0.068 vs 1.356）。
- **measurement consistency 已饱和**：真实 A_k 的 LS 下界 vs 实测——stage 0 差 0.2% 且
  frame 0 起即平；stage 3 下界 0.5（op.measure 洞内也加噪）。早期高 resid 是表示极限。
- **让步**：采样器实际停在 σ_τ≈0.4（反驳草稿 §8.3）；hole 0.272 仍差 anchored Alg2 (0.102) 2.7×。
- CG 收敛（另一问题）也已闭环：cg_max_iter=300，全 1365 步收敛（motion 最大需 175）。
- 详见 report §11–§12、`tasks/alg4-box-stage3-diagnosis.md`。**待跑**：smin=0.39 的 5 任务全量。

## 2026-08-22（第二轮）：Alg 4 stage-3 根因查清，并更正上一轮的机制解释

用户："探索并debug原因"。三重证据（解析 + 数值 A8 + 受控探针）：

- **根因：σ_τ→0 时 (19) 把网络整个挤出去。** 用草稿恒等式 (3) `N = σB + (e−s)H`，
  σ→0 ⟹ **`x̂₁ = H_τ⁻¹x_τ`，对任意 G、与 v_θ 无关**；velocity 只经 `σ·N·v` 进入，
  权重 **`b ∝ σ_τ`**。已固化为 `verify` 的 **A8**（σ 每降 10×，b 与 ‖x̂₁−H⁻¹x_τ‖ 同步降 10×）。
  **草稿只讨论过 γ→∞ 的这个极限（§5.2/A.3），没讨论 σ_τ→0**，而 schedule 保证它必然发生。
- **只有 stage 3 会踩到**：`σ_τ(τ=1)=1−e_k` = 0.750/0.500/0.250/**0.000**，只有它 `e_k=1`。
  **与 `g_bypass_stage3`/G=I 无关**（上一轮记的"最后一个 stage 没有重置"只是表象）。
- **逐帧收支精确闭合**：`Δhole = 端点步 + Block1 注入`。端点步是唯一移除机制，
  随 b 单调衰减、**frame 35 变号**；frame 31 的跳变 = 移除能力 −1.699→−0.143（12×）而注入没变。
  Block 1 注入在**全 40 帧**都等于 `1/C⁻¹`（0.80–1.09）。
- **更正上一轮**：受控一步映射显示白噪声下**每帧都有全局吸引不动点**
  （frame 31 在 δ=3.2 时 out/in=0.165），**无域可逃**，所以"边缘稳定+随机逃逸"是错的。
  **真实机制是结构性棘轮**：replay 臂（注入采样器自己发散出的 x₁）vs 同量级白噪声，
  `hat/in` 在 frame 31 是 **0.537 vs 0.122（4.4×）**、frame 38 是 **1.007 vs 0.872**。
  Block 1 注白噪声 → 端点步转成**连贯的离流形内容** → 下一轮去噪器撤不掉 → 棘轮累积。
  S_it 是棘轮转数。观测区完好是因为 `‖AᵀA‖/η²`≈400 压倒 `C⁻¹`，x₁ 被 y 钉住。
- **更正后的最小下一步**：`S_it` 10→1 仍是第一步（理由改为"减少棘轮转数"）；
  另一个不引入新超参数的方向：**stage 3 在 σ_τ 很小处仍在放 step**
  （frames 35–38 端点步已变号），`ode_steps_per_stage`/`sigma_min` 都是既有 key。
  这与草稿 §8.3"误差在 cascade 末端最小"直接冲突，值得作为可证伪实验报出去。
- 新增：`main4.py --mode contraction`（含 replay 臂）、verify A8、
  `results/alg4_box_stage3_diagnosis/contraction/`。

## 2026-08-22：Algorithm 4 stage-3 失效诊断（box_inpainting）

用户指令：只测 box_inpainting；**把 `g_bypass_stage3` 从变量面删掉、走正常流程**；
不重新加入 `h0`/`ridge_rel`/新超参数；查清"frame 20 多仍正常、之后逐渐恶化、问题集中在 stage 3"。
（歧义读法已判定并记录在 `tasks/alg4-box-stage3-diagnosis.md`，需用户复核。）

- **失效点 = frame 31（stage 3, τ=0.111）**，单帧 4.68×，全轨迹最大跳变。
  frames 0–30 平在 mse_full 0.074–0.127，**frame 30 是最好的一帧（0.0786）**。
  观测区全程单调改善、`resid` 单调→1、`‖x₀‖²/n` 全程 0.99–1.02
  ⟹ **失效 100% 在 measurement null space 内**。
- **Block 1 注入的正是它自己的后验方差 `1/C⁻¹`**（洞内无数据项），
  实测/预测比 **0.98–1.08** 跨全部 stage ⟹ **Block 1 按规范工作，不是 bug**。
- **失效瞬间 Block 1 先坏、x̂₁ 后坏**（frame 31 的 s=0/1 上 x̂₁ 仍改善输入，s=2 才转坏）。
- **判别变量是去噪器在洞内的收缩率**（0.27–0.45 → 0.8 → ≥1），随 σ_τ 下降而衰减；
  **stage 转移重置 σ_τ 从而救回收缩率，stage 3 之后没有重置**。
  不动点公式 `hole(x̂₁)+1/C⁻¹` 精确成立（0.1488+0.1508=0.2996 vs 实测 0.2997）。
  **排除**"stage 3 注入方差最大"：C⁻¹ 全局最小在 frame 10（注入 0.210 > frame 31 的 0.130）却稳定。
- **是 inner-loop feedback 累积**：S_it 10/2/1 → stage 3 增长 **11.23×/3.36×/1.80×**，
  S_it=1 在 frame 31 完全无跳变。
- **是真发散不是多样性**：最终洞内 **99.1% 像素出 [-1,1]**（std 1.855，GT std 0.478）；
  S_prior 四个 arm 全部发散 ⟹ **"更强 precision 是否牺牲 diversity"本数据无法判定**。
- **最小下一步**：`num_langevin` 10→1（既有 config key，非新超参数）。
  但它修的是**发散**不是质量——S_it=1 最终 hole 仍 0.826，远差于 anchored Alg 2 的 0.10。
- 产物：`PixelFlowICLR/Algorithm2/results/alg4_box_stage3_diagnosis/`（report + 6 CSV + 5 PNG
  + `s_prior_sensitivity/` + `diversity/`）。入口 `main4.py --mode diagnose|diversity`。
  新增 `Algorithm2/alg4_diag.py`（recorder + metric + 全部诊断绘图）。
  **只读性已证**：与已提交 full_ip 同格相对差 2.5e-5 / 2.5e-5 / 2.0e-7。

### 更正：撤回"删目录会毒坏父目录"的说法

上一轮 `results/alg4/report.md` 写过"在这个 ceph 挂载上删除目录会让父目录 readdir 返回 EIO"。
**证据不支持，已撤回**：会话最开始就已在从未删除过的目录上观察到 EIO。
能站住的只有"readdir 会间歇性 EIO，随机命中任意目录"。
真正有效的缓解是**进程内重试 import**（所有失败都在 importlib 的 `_fill_cache`；
进程级重试要白白重载 2.7GB checkpoint）＋ `PYTHONDONTWRITEBYTECODE=1`。

## 2026-08-21（晚）：Algorithm 4 实现完成并首次全量跑通

用户指令："按照main.py调用utils.py的方式写main4.py并在utils.py补充posterior_sampling的
算法，把page4的algorithm实现出来，结果放在results/alg4"。**这同时是执行授权**
（上一轮 `read-algorithm4-draft` 记的是 `execution_allowed: false`）；本机 codex 不可用，
已记 `single_agent_execution_authorized: true`。

- **实现**：`Algorithm2/{main4.py, config_alg4.json}` + `utils.py` 纯新增五个函数。
  既有 alg2/wv/reg 三个采样器一行未动。三处结构差异全部到位：l.8 移出内循环、
  Block 1 中心是 x̂₁ 而非 x_τ、Block 2 是直接抽样 (23)（**无 h₀**）。
  `h0`/`ridge_rel`/`cg_max_iter_l14` 被 config 契约与采样器双重拒绝；`S_prior` 无代码默认值。
- **verify 17 项 dense float64 全过**：A2 把 Prop. 7 验到 `5e-15`；
  A4 显示 τ=0 处 `λ_min(H₀ᵀH₀/σ²) = -4.6e-18`（逼 Alg 2 加 ridge 的奇异性）
  而 `λ_min(M^den)` 恰等于 `1/s²`。
- **实测 S_prior**（`--mode measure_s2`）= 0.200/0.211/0.220/0.228，**λ 等效仅 4.4–5.0**，
  远弱于 anchor 扫描偏好的 λ=100–200。口径警告：在 demo/eval 图上测的，非训练集。
- **主要发现：全部亏损集中在 stage 3，五个任务无一例外**（含无 null space 的 blur/SR）。
  各 stage 末 mse_x1（7 图均值）：box 0.070/0.071/0.136/**0.922**、
  random 0.012/0.022/0.077/**2.016**、gaussian 0.012/0.065/0.211/**2.295**、
  motion 0.043/0.084/0.201/**2.341**、SR 0.013/0.032/0.193/**2.391**。
  stage 2 全段是平的。
- **起点不能用 σ_τ→0 或 γ² 变大解释**：stage 3 最大跳变在 step 0→1（box 3.5×、SR 5.0×），
  而该步 σ_τ 只 0.400→0.356、γ² 只 0.0094→0.0109。延续段则与两者一致。**两句分开说。**
- **同 λ 对照（`s² = 1/λ`，box/junco）**：stage 0–2 上 Alg 4 持平或略胜 anchored Alg 2
  （λ_eq=400：0.064/0.052/0.088 vs 0.064/0.061/0.095），**stage 3 差 10–28 倍**。
- **实现正确性的正面证据**：草稿 §7 诊断 `‖x₀‖²/n ≈ 1` 在 1365 个 step 上均值 **0.9991**。
- **§8.6 的量已报告**：measurement residual 末步 box 0.910 / random 1.060 / gaussian 1.118 /
  motion 1.291 / **SR 3.747（明显不满足"应稳定在 1 附近"）**。
- 详见 `PixelFlowICLR/Algorithm2/results/alg4/report.md` 与 `tasks/implement-algorithm4.md`。

### 本轮踩到的运行问题（已修，值得记住）

- **第一次全量跑在 21/35 处崩溃并丢掉全部 21 格**——沿用了 `main.py`"最后统一写盘"的结构。
  已改为每格立即落盘 + 跳过已完成格 + 单格重试，作图从磁盘回读。
- **不要在这个 ceph 挂载上删目录**：删掉 `__pycache__` 后其**父目录** readdir 开始返回 EIO，
  把后续所有 import 打挂。用 `PYTHONDONTWRITEBYTECODE=1` 即可。
- **杀 wrapper 不会杀 python 子进程**：三个孤儿进程并发往同一 CSV 追加，把每格写了三遍。
  已加 `flock` 单实例锁 + `trap` 连带杀子进程。附带收获：这三次并发同 seed 同 config 的
  重复格，各指标最大相对差 **4.3e-4**，与本项目"GPU 非 bit-exact、四位小数一致"口径吻合。

## 2026-08-21：Research OS 核验 + Algorithm 4 草稿精读（本轮）

- **Research OS**：早在 08-14 已接入；本轮 `research-upgrade-project` 确认在 bundle v2
  且幂等，`research-doctor` = 25 通过 / 2 警告 / 1 失败（全部失败与警告都在
  **本机 gh / codex 未安装**，项目侧 15 项全 ✅）。工具本地 clone 路径见
  `local.yaml` 的 `external.research_init_clone`（分支 `agent/efficient-dual-agent`）。
- **本机是 single-agent degraded**：无 gh、无 codex、无 GitHub 网络
  → 记忆**只能 commit 不能 push**，Issue #3 的增量复审在本机无法推进。
  另：`git status` 全量扫描因 ceph 的 `Remote I/O error` 稳定失败（`-uno` 可用）。
  详见 `context/facts.md`。
- **`pixel_flow_langevin` 导入**：该 GitHub repo **就是本工作区的 origin**。
  `IP_branch` 领先 `origin/IP_branch` **9 个 commit（未 push）**，
  逐条清单见 `context/facts.md`。
- **algorithm.4pdf.pdf 精读完成（15 页全文）** → `references/2026-08-21-algorithm4-clean-endpoint-sampler.md`。
  草稿把耦合从 flat-prior 高斯换成真实 denoising conditional，`p_τ` 相消，
  **step size 与先验平滑同时消失**，两个 conditional 都精确抽样；唯一近似是把
  `p(x_1|x_τ)` 当高斯，其协方差由 `C⁻¹ = H_τᵀH_τ/σ_τ² + S⁻¹` 给出（Lemma 3 + Prop. 4）。
  三条与本项目的硬对应：
  ① `score_solve` 就是草稿 (18)、`gamma2_meas.json` 就是 (20)，**已完全一致**；
  ② **我们经验调出来的 Tweedie anchor λ 就是 (12) 的 S⁻¹**（`λI = S⁻¹` ⟺ `s²=1/λ`），
     草稿要求**测量**而非搜索该量；我们最优的 λ≈100 ⟹ s²=0.01，图像在 [-1,1] 的
     assumption-free 上界是 s²≤1，**λ=100 落在草稿 §8.2 警告的"低估 S"一侧**
     （会收缩向去噪估计：重建指标变好而多样性丧失，且该失败在 MSE 上看不见）；
  ③ 代码是"**加**"anchor 到 interpolant 项上，草稿是用 `C⁻¹x̂_1` **替换**它——
     未提交的 `data_rhs_matchx1` 恰是 (22) 缺了 `S⁻¹x̂_1` 的那一半。
  两个长期困扰按构造消失：**τ=0 的 ridge 整套机制**（Prop. 4(a)；与 `5734af2`
  "ridge 不是把洞压在 1.0 的原因"互相印证）与 **h₀**（Block 2 变直接重加噪；
  `regularization_final` 的头号结论"h₀ 主导、太小则 x₀ 冻住"即此病理）。
  草稿自陈风险：Block 1 中心是 x̂_1，**全权重承担 velocity error**，优势随 γ 增大反转；
  我们 γ² 表 stage 3 末端冲到 0.145（全表最大）。
  详见 `tasks/read-algorithm4-draft.md`。**用户只要求"阅读"，未要求实现
  → `execution_allowed: false`。**

## 2026-08-19~21：记忆补记（此前未进 CURRENT）

- `algorithm3-from-modified-draft`（`437151f`）、`diag-h-tau-inv`、
  `algorithm2-line15-diagnosis`、`prior-injection-b-tau`、
  `wv-coupling-precision-alg2`、`tweedie-anchor-reg-alg2`、`alg3-parameter-sweep`
  七个任务文件已在 `tasks/` 内，但 CURRENT 停在 08-18/19。
- **`results_reg_alg2`（Tweedie anchor λ 扫描，box/junco/seed42）**：
  洞区 **1.0062 → 0.1023（λ=100，9.8×）**，可见区与 measurement residual **同向单调改善**
  （可见区 −20%），即 anchor 在观测区没有与数据打架。stage-end：λ=0 第一个 stage 末就是 15.7，
  λ≥5 全程压在 0.55 以下。P=identity 略优于 P=nullspace（0.1150 vs 0.1160）。
- **`results_reg_alg2/regularization_final`（48 格 λ×h₀×S_iter 网格，随机项全保留）**：
  最优 **λ=100, h₀=0.01, S_iter=10 → hole 0.084308**；综合最好 **λ=100, h₀=0.1, S_iter=5**。
  三条结构性结论：(1) λ 在 16/16 个 (h₀,S_iter) 格上单调有帮助；
  (2) **h₀ 是主导因子，1e-4/1e-3 全线不可用**（x₀ 冻住，λ 再大救不回来）；
  (3) h₀ 与 S_iter 干净交互——小 h₀ 要大 S_iter、大 h₀ 要小 S_iter，最优点 `S_iter·h₀ ∈ [0.1,0.5]`。
  S_iter 是 hole 与 visible 的权衡旋钮（每次内迭代注入一次 Lemma-5 噪声）。
- **术语冲突警告**：草稿 algorithm.4 的 `S` = 先验协方差 surrogate；
  本项目代码与报告的 `S` = 内迭代次数 `num_langevin`。今后一律写 `S_prior` / `S_iter`。
- **工作区 `utils.py` 有 +525 行未提交**（wv / reg 两个采样器族 + 若干 helper），
  且**同一 diff 改了 `run_posterior_sampling_alg2` 的内循环**（l.11 后重建 x_tau），
  而 HEAD commit `d819043` 自述该改动 "neither beats the baseline"
  ⟹ **工作区里的 baseline 采样器不是 baseline**。见 `context/facts.md`。


## 正在进行

- **measurement-alignment-inpainting（2026-08-18 晚）**：查实我们的两个 inpainting 任务**观测无噪**
  （`demo_runner` 走 `y=op(gt)`，算子前向就是 `mask*x`，`sigma` 存而不用；实测可见区
  `max|y−GT|=0.0`），而 8 个 baseline 全部带噪（五个 σ=0.05、三个 σ=0.10）——
  **我们的 inpainting 指标是在更容易的问题上取得的**，仓库自查报告早已标为
  "CRITICAL — in our favor" 但修复只落地了 mask 那一半。
  按用户裁决：只对齐我们这侧（σ=0.05 + box 居中 128×128），**全部代码新增在
  `Algorithm2/test/`，既有文件零改动**（含 PixelFlow `set_timesteps`，用户明令不动）。
  回归护栏通过：三个带噪任务逐位未变，random_inpainting 的 y 变化 std=0.0497(=σ)。
  τ=0 的 ridge 问题改用扫描既有 `algorithm.ridge_rel` 来测——正是论文 §7.13 要求"报告"的量。
  进行中：full_ip（5 任务 × 3 图）与 ridge sweep（4 ε × 4 seed）。
  详见 `tasks/measurement-alignment-inpainting.md`

- **Issue #3 增量复审仍为 request changes**：复审 `6a5a927..a8ac308`；生产公式、full_ip
  post 输出、显式 config/out 路径和 V8 变量隔离已修，但 V9 未测试生产抽样路径，K sweep
  的 RNG/NFE 契约不闭环，config 只验键且 anchor 仍可 silent no-op，S2 未覆盖 τ=0/残差/稠密
  参考，mask checksum/bbox 未落盘，V8 stdout 仍把 diagnostic 混入 `ALL CHECKS PASSED`。
  证据侧 n=4 单图极差不能称通用 5% 地板，K/anchor/γ² 仍只能作描述性观察；多处
  bit-exact/8 位有效数字/ULP 口径尚待修正。Review：Issue #3 comment `5337002138`，已加
  `触发:needs-fix`。详见 `tasks/review-alg2-debug-directory.md` 与本轮 handoff。

## 2026-08-18（晚）：新挂载机路径兼容（用户指令）

- 环境：同一 ceph 盘挂到 `/CBIG-Standard-ECE/`（旧前缀 `/sfs/ceph/standard/`、`/standard/`
  在本机不存在）；**无 Slurm**，本地 4×A100 直跑；HOME 不共享（`/home2/rqx6rs`，本机无 gh）。
- 真正断点是**绝对符号链接**（共享盘上存的是旧机写的 `/sfs/...` 目标）：74 个已改为相对目标，
  一次修改两机通用。致命项 `IP_package/pixelflow`（模型包本身）此前在新机是死链。
- 代码侧加 `SHARE_ROOTS` + `resolve_path()`（存在即原样返回 → 旧机行为零变化；否则按候选根
  依次重试；全失败列出所有尝试过的路径）。
- 端到端验证：新机 GPU 复现集群 trw 实验，数值接近但非逐位；两机整条 CSV 最大绝对差
  `mse_x1=3.10e-6`、`hole=1.43e-5`、`obs=8.92e-8`，不能表述为“同到 8 位有效数字”。
  详见 `tasks/path-compat-mounted-server.md`。

## 2026-08-18（Codex review Issue #3）

- 审查区间 `b0c9351..6a5a927`。CPU V1–V8 与 one-step self-test 可运行并通过，但未覆盖
  τ=0 抽样协方差、config 值迁移、post-projection 落盘和 fixed-score K 扩展等关键风险。
- 必改主项：τ=0 ridge 随机 RHS 漏 `sqrt(epsilon)` 噪声；K>1 复用冻结 score；active anchor
  配置是 silent no-op；full_ip 丢弃 post-projection x1；相对 out 会写到调用者 CWD；缺严格
  config/seed 契约测试；oracle/anchor/K 结论需补可复算与公平对照或降级口径。
- 已提交 CSV 反证“bit-exact/逐位一致”：trw 两次独立运行最大轨迹指标差约 `1.05e-5`；
  f9299d7 与 6a5a927 末端 hole/obs 也仅四位小数一致。详见
  `tasks/review-alg2-debug-directory.md` 与同日 handoff。

## 2026-08-18（晚）：GPU 终验闭环 + trw 实验 + 协作层二次修复

- **cg_max_iter_l14 回归**：终验 18650178 得 hole 0.9704≠0.9691 → 根因=参数内联把 τ=0 的
  l.14 CG 上限从硬编码 200 写成 config 里的 50；修复后 job 18651717 复现到四位小数
  0.9691/0.0018（非 bit-exact），终验数值判据通过。同批修回 box/random trw=1.0。
- **terminal_replace_weight 0 vs 1.0 实测**（用户指令，job 18651908）：两次独立运行的
  pre-projection 轨迹数值接近但非逐位一致；trw=1 洞区不变 0.9691、观测区 post obs=**0.0**。
  顺带发现 demo 管线
  inpainting 的 y **无加性噪声**（η-模型失配）。详见 tasks/test-terminal-replace-weight.md
- **协作层二次修复**：Issue #3 首次 codex 派发静默失败（exit 0 无评论）+ 标签被误置
  状态:done——三个缺陷：①错误评论原文引用 marker 导致后续子串校验误判 ②疑似双 watcher
  竞态 ③watcher 在 login node 反复被清（846355 已死，日志无痕）。修复=orchestrator 补
  输出留痕/行首 marker 校验/单实例锁 + **watcher 迁移 scrontab**（计算节点已验证 gh/codex 可用）。

## 2026-08-16 协作层修复记录

- 用户发现"GitHub 上看不到项目 & Codex 零参与"。根因三件：
  1. default branch 停在 `main`（初始骨架），工作全在 `IP_branch` → 已切 default=IP_branch。
     repo 名是 `pixel_flow_langevin`（非 MSFlow）——改名与否待用户定
  2. watcher 8月14日启动后即死：orchestrator 轮询裸奔 gh 调用，一次 TLS 超时整个进程崩
     → research-init 已修（watch 模式容错重试，commit 4a47632）并重启
  3. Issue/PR 从未创建过，label 状态机零触发 → Codex 自然零参与；另修掉
     `状态:review` 每轮重复调 codex 的烧 token bug（commit e881ef0）
- research-init 三个修复已推上游（含 doctor 的 BSD stat bug）

## 最近完成

- 2026-08-18~19：**Algorithm2 目录治理四连（原执行记录；Issue #3 复审待修）**——results 170M→18M 只留关键证据；
  main 剥离 WLS/Model（纯 Alg2）；config.json 成为唯一配置源（4 模式 JSON 合一 → 全参数内联，
  路径指针废除）；用户抓出 projection（terminal_replace_weight box/random=1.0）遗漏已补；
  S2 stage-1 漂移判决=hash 种子调用失误（契约 PYTHONHASHSEED=0）。GPU 指标仅四位小数一致，
  并非 bit-exact；其余必改项见本页顶部。commits f9299d7/e3f23b9。
  GPU 终验 job 18650178 被 A100 维护窗口挂起（判据 hole 0.9691）。
  详见 tasks/cleanup-algorithm2-directory.md（含补记说明）

- 2026-08-17：**debug-box-alg2-hole 原调试阶段结论（Issue #3 复审待修）**——固定 demo/seed 下
  Tweedie anchor 的已存 CSV 显示洞区 MSE 池化 1.097→0.178（6.2×），但 oracle 目标、K>1
  实现、RNG 配对和多 seed 证据均不足，暂不能据此确认“结构性缺陷/唯一有效干预”。详见
  tasks/debug-box-alg2-hole.md 与 review-alg2-debug-directory.md。

- 2026-08-16：**Issue #2 re-review 通过**：Algorithm 2 one-step 估计器实验的 3 处证据口径
  已修正；Codex 增量复审确认 10.94%/19.60% 可复算，结论 approve。
- 2026-08-16：**Issue #1 review 完成并关闭——双 agent 回路首次完整闭环**：3 轮
  （request changes ×2 → approve），全程 watcher/orchestrator 自动调度。
  ①结论表述过强 ×2 + "bitwise identical" 证据越界 → 修正 526ca24（修正前逐项复算核实）；
  ②README 生成模板漏改（会回归旧口径）→ 修正 458d5af；③approve（模板与磁盘逐字节一致）。

- 2026-08-15：**实验 1b：可视化 + 结果瘦身**——每个 t 的一步恢复图合成 `onestep_predictions.mp4`
  （40 帧 = 4 stage × 10 t，帧内 [GT|x_t|WLS|Model] × 7 图 + 逐图 MSE，job 18568633）；
  145 张冗余曲线 PNG 合并为 1 张 `mse_vs_t_summary.png`（合并前逐位一致性校验）。
  结果目录收敛到 6 个文件；png/mp4 按仓库纪律 gitignored，见 artifacts.yaml
- 2026-08-15：**PixelFlowICLR 实验 1（onestep-mse-vs-t）完成**——one-step x̂₁ MSE vs t，
  5 任务 × 7 图 × 4 stage，A100 job 18567584。核心发现：Model(direct) 一致优于 WLS
  （stage0 差 ~50×，stage3 打平）；5 任务曲线因 kw 相同而完全一致（one-step 不经过 operator）。
  详见 experiments/2026-08-15-onestep-mse-vs-t.md；数据 PixelFlowICLR/results/onestep_mse/

- 2026-08-14：**全量记忆导入**——13 路并行读完项目内 130+ 份 .md(含前代 agent 的 SESSION_HANDOFF/global_memory)+ 关键代码/未提交 diff → 写入 experiments/{imagenet-5task-tuning, imagenet-baselines, multimodal-celeba-afhq-mri}.md、context/{authoritative-entries, contradictions-registry}.md、ARCHITECTURE/CONSTRAINTS 增补
- 2026-08-14：接入 Research OS（.research/ 初始化、GitHub label、watcher 启动、codex 装好并登录、dual-agent 就绪）
- 2026-07：旧机器 → CBIG 集群迁移完成；MRI prior checkpoint 恢复；`github_project_local/` 复现 benchmark 打包并验证（AFHQ SR bit-exact；ImageNet/CelebA batch-parity bit-exact；MRI 归一化契约验证）
- 2026-04~05（迁移前，见 git log 与 debug_IP4）：五任务 ImageNet 调参收敛（`memory_blur.md` 有 best config 与禁区）、MRI val_final_8 参考协议跑完、`IP_package/Experiments.md` 写成 paper 风格

## 下一步

- **（最小下一步）`num_langevin` 10→1 重跑 box_inpainting**，再在 S_it=1 下重跑 diversity。
- **（待用户裁决）是否允许为诊断关掉 `g_bypass_stage3` 跑一格。** stage 3 是唯一 `G = I`
  的 stage，H_τ/N_k 退化为标量、interpolant 失去全部低通作用，而 l.18 的 `U⁽¹⁾` 是 nearest
  上采样；Alg 4 的 Block 1 又完全以 x̂₁ 为中心。这是 stage-3 崩溃目前最可能的机制但未经检验，
  而 CONSTRAINTS §采样器纪律明令"`g_bypass_stage3=True` 不关"。
- **跑 §8.6 的多样性诊断**（固定 y 重复采样，量 ker(A_k) 内 spread）。
  不跑它，"更小的 s² 更好"就不能写进任何结论——§8.2 说过那正是指标变好而多样性丧失的样子。
- γ² 通道 ablation（冻结 γ² 重跑 stage 3），分离 σ_τ→0 与 velocity error 两条通道。
- `S_prior` 换成训练集统计，消除当前 demo/eval 图的泄漏口径。
- 修复 Issue #3 的必须修改项后请求 Codex 只复审增量（**本机无 gh/codex，需换机或先装工具**）。
- （待用户确认研究方向：例如跑 `--full` 复现、补 baseline 表格、写 paper draft、或继续某个任务的调参）
- 处理外层 repo 未提交状态（见"阻塞"第 2 条）

## 阻塞 / 需要用户决定

0. **（2026-08-21，本机）无 gh / 无 codex / 无 GitHub 网络**：记忆只能 commit 不能 push；
   `IP_branch` 已积压 9 个未 push commit；dual-agent 回路在本机不可用。
   需要用户在本机 `gh auth login` + 安装 codex，或换回有工具的机器。
0b. **工作区 `utils.py` 的 baseline 污染**：`run_posterior_sampling_alg2` 带着一个
   HEAD 自述"无收益"的未提交改动（l.11 后重建 x_tau）。是否 revert 由用户定。

1. **in-flight 工作已从 diff 复原(机械意图)**：未提交改动 = random_inpainting **第二阶段 OAT sweep**(数据契约 .pt→PNG + stage2_configs + 断点续跑)、train.py 训练韧性(resume/epoch-ckpt/epoch-eval)、CFG 空类 token 泛化、legacy 采样配置开 CFG=2.0(详见 ARCHITECTURE"工作区未提交改动"节)。**待用户确认**:stage2 sweep 跑到哪一步了、动机实验是什么(MRI 先验训练?)
2. **外层 repo 工作区大量未提交变更**：`PixelFlow_train_code/` 整目录删除（−5226 行）、`IP_package/`、`debug_IP4/` 多数实验目录 untracked。是否按现状提交（大文件已被 .gitignore 排除）由用户决定
3. ~~codex CLI 待装/待登录~~ 已解决（2026-08-14）：v0.147.0 装于 `~/.local/bin/codex`，
   ChatGPT 账号已登录，5 个 MCP server 双侧注册，STATE 已切回 `dual-agent`
