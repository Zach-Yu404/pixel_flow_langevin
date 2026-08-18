# debug-box-alg2-hole

state: done · owner: claude · type: debug

## 【用户原始要求】（2026-08-17，逐字）

> 先debug box inpainting，algorithm2中间没有很好的填充reasonable的东西，另外，
> 所有内容都在/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2下变动，
> 不再加claude落款就行

（scope 解释：代码与产物改动仅限 Algorithm2/；.research 任务/实验登记为流程簿记照常。）

## 【Claude｜根因分析】（systematic debugging Phase 1-2）

症状：full_ip box 帧（如 frame_0035，step 36/40）Alg2 的 x̂₁ 洞区始终为饱和彩噪，从未被
模型内容填充；对照 Alg1-Model 洞区填充完好。metrics 佐证：box s0-s2 Alg2 MSE 7.29/3.12/1.45。

排除项：
- 实现≠论文？否——full_ip_compare.run_posterior_sampling_alg2 与草稿 p.13 Algorithm 2 逐行一致（含 l.17 的
  (x₀+x̂₀) 号——式(35)(36) 推导自洽，x̂₀ 项即对 x_τ 的去噪方向，非符号错误）。
- one-step 版 line14 不用 x̂₀？按 spec 如此（b 只含 y,x_t），非本 bug 主因。

根因（结构性，仅在 AᵀA=0 的洞区暴露）：
1. Alg1 Block1 梯度含 −Hᵀx̂₀/σ ⇒ 模型内容直灌 x₁；Alg2 Block1 改为精确条件采样
   π(x₁|x_τ,y)，洞区条件方差 σ²H⁻²：早期 stage H⁻¹=1/(τe)（τ=0.111 时 36×）⇒ 每内步
   注入巨方差；晚期 σ→0 ⇒ x₁←H⁻¹x_τ≈x₁ 自复制冻结。
2. 模型先验唯一通道 = Block2 x₀-Langevin，对 x_τ 的拉力 ∝ σ·h₀/2/步；h₀=0.1 单值未扫
  （代码自注 "raise toward 1e-1...unswept"）。论文 §7.5 预言同款症状。

## 【Claude｜Phase 3 假设与最小验证】

假设：洞区失败=Block2 欠混合，h₀（论文自带旋钮）过小。
最小实验：box·junco 单图，h₀∈{0.1,0.5,1.0}（S 不变），记录 obs/hole 分离 MSE 轨迹 + 末帧。
预期：h₀↑ ⇒ hole MSE 显著下降、末帧洞区出现语义内容；若否→回 Phase 1。

## 【Phase 3 记录】

### 轮 1：h₀ 假设 —— 证伪
job 18634904（A100, 6m24s）：h₀∈{0.1,0.5,1.0}，洞区 MSE 0.969/2.790/3.285（obs 恒 ~0.002）。
h₀↑ 反而更差 ⇒ h₀ 非杠杆。

### 轮 1.5：标量 oracle 仿真（CPU，两项决定性证据）
1. 原算法 + 完美 score：mean→μ_p ✓ 但 std 4.8 vs 目标 0.2（24×）——洞区 x₁ 链为
   无收缩随机游走（Block1 精确采样只经 AᵀA 收缩；l.15 重算 x₀ 每步清零 Block2 进展）。
   结论：算法级缺陷（论文 (62) 的条件分布不含 x₁ 的图像先验），(S,h₀) 无解。
   Prop.4 核对：τ>0 时 ker(H)∩ker(A)={0}，实现只在 τ=0 加岭与论文一致，非实现偏差。
2. Tweedie 锚定变体（M+=λI, b+=λ·x̂₁_model+√λξ，保持 Lem.5 精确采样语义，零额外 NFE）：
   oracle 下 λ=25 时 mean 1.002 / std 0.200 —— 与目标后验完全一致。

### 轮 2：anchor 假设（进行中）
偏离登记：run_posterior_sampling_alg2 新增 anchor 参数（默认 0.0 = 论文原算法逐位不变；实验性变体
明确标注）。job 18635053：box·junco，anchor∈{0,1,5,25}，h₀=0.1，S=10，seed 42。
判据：anchor>0 使洞区 MSE 显著下降且末图洞区出现语义内容。
算法修改的最终采纳（是否写进论文/成为默认）待用户裁决。

### 轮 2 结果：anchor 假设 —— 证实（junco 单图）
job 18635053（A100, 8m22s）：anchor∈{0,1,5,25}，h₀=0.1，同 seed。
洞区 MSE：0.9691（=原算法基线，逐位复现轮 1）→ 0.8862 → 0.4516 → **0.0861**（11.3×），
obs 区恒 0.0018（锚定不伤观测区）。剂量-响应单调，与 oracle 预测一致。
视觉：anchor=25 洞区出现语义合理填充（鸟腹延续/树枝贯通/背景重建）vs anchor=0 纯彩噪。
产物：results/debug_box_anchor/（h0_sweep.csv + mse 曲线 + final_x1 面板）。

### 轮 3：7 图全量复核（进行中）
job 18635142：全部 7 张 playground 图，anchor∈{0, 25}，其余全同。判据：逐图洞区 MSE
一致性下降 + 无观测区回退。

### 轮 3 结果：7 图全量复核 —— 证实，7/7 全胜
job 18635142（A100）。洞区 MSE（anchor=0 → 25）：breastplate 1.144→0.083（13.7×）、
crane 1.220→0.385（3.2×）、ibex 1.056→0.249（4.2×）、junco 0.969→0.086（11.3×）、
lakeside 0.965→0.076（12.8×）、sea_anemone 1.276→0.160（8.0×）、sheepdog 1.051→0.205（5.1×）。
池化均值 1.097→0.178（6.2×）；obs 区全部 ~0.0019 无回退。
产物：results/debug_box_confirm/<image>/。

## 结论

box 洞区不填充的根因是**算法级**的：论文 Algorithm 2 的 Block 1 条件分布（式 62）在
ker(A) 内不含 x₁ 的图像先验，洞区 x₁ 链为无收缩随机游走（oracle 仿真：完美 score 下
std 仍 24× 超标）；l.15 的 x₀ 重算每内步清零 Block 2 进展，(S, h₀) 均非杠杆（h₀ 扫描
已证伪）。实现与论文逐字一致（含 Prop.4 的 τ=0 岭），非实现 bug。

修复（实验性，默认关闭）：Tweedie 锚定 Block 1——把 l.10 已算出的 v 得到的
x̂₁_model 作为精度 λ 的高斯伪观测（M += λI，b += λx̂₁ + √λξ_a，保持 Lem.5 精确采样
语义，零额外 NFE）。λ=25 时 7/7 图洞区 MSE 降 3.2×–13.7×，观测区无回退，洞区出现
语义填充。λ 的角色与 Alg1/PRINCIPLE 的 λ_reg 同源。

**待用户裁决**：是否把锚定项写进论文 Algorithm 2（即修改 (33)/(34) 加先验伪观测项），
以及 λ 的取值/调度（本轮只试了常数 λ∈{1,5,25}，未做 per-stage 调度与更大 λ）。

state: done（调试闭环；采纳与否待用户）

## 【后记：重构与命名统一】（2026-08-17，用户指令）

1. 变量名统一：先与旧代码对齐，再按用户升级指令与**论文符号**对齐
   （S/L/h0/gamma2/eta/epsilon/x1/x0/x_tau/sigma_tau/M_tau/b_tilde/xi_*；kw/config 键名不变）。
2. `sample_alg2` → `run_posterior_sampling_alg2`：签名与段落结构对齐旧 inference
   `run_posterior_sampling`（CFG setup→尺寸推导→stage 循环→内循环），论文行号注释标差异；
   `alg2_x1_solve` → `clean_image_solve`（论文 "clean-image solve"）。
3. 重构中发现并修复 **class_label 公平性 bug**：full_ip main 里 Alg2 兜底类别 10 vs
   wls/model 真实类别 → 旧 full_ip_metrics.csv 的 alg2 行引用前需重跑
   （debug/anchor 系列不受影响——该路径显式传了正确类别）。
4. 行为等价验证：job 18640239 ✅ 通过——junco anchor=0 洞区 MSE 0.9691 / obs 0.0018，与重构前逐位一致。

## 【组件审计】（2026-08-18，用户逐项点名核查）

新增 `Algorithm2/verify_components.py`（V1-V8，float64 稠密矩阵证据，全部 PASS）：

| 问题 | 结论 | 证据 |
|---|---|---|
| M0 嵌套 apply_H_tau 是否= (H_τ)ᵀH_τ | **等价且正确**：G 逐位自伴（bilinear½↓=2×2 avg-pool，其伴随=nearest↑/4；且 G²=G 投影），H/B/N 为 G 的多项式继承自伴 | G 对称误差 0.0；H∘H vs HᵀH 1.4e-17；N∘N vs NᵀN 1.7e-18 |
| score_solve 的 N²/H² 要不要 transpose | **当前 G 下无区别**；CG(N∘N) 与稠密 NᵀN 正规方程解一致；V1 为未来换非自伴 G 的守卫 | 差 1.1e-15；S1 恒等式 1.1e-15（首跑 FAIL 是测试自身 σ 用错，已修） |
| make_Ak_fns/Ak/ATk | 伴随测试全过 | <Ax,w>vs<x,ATw> ≤1.8e-15（stage res 4/8/16） |
| compute_sigma_tau / make_velocity_fn | 公式=论文 l.5；CFG=u+scale(c−u)，scale=class_guidance_scale(stage) | 0 误差（stub 模型） |
| power_iter_norm | 20 迭代 vs 精确 λ_max 相对误差 2e-4 | ridge=1e-6·估计，偏差无关紧要 |
| γ² 取值 | 白噪声探针下 γ²=0 最优（N 谱 [0.36,0.43] 条件良好，γ²=1.0 明显有害）；但真实 v 误差是结构化的（verify_gamma2 实测 11%/20% 差异）→ **须真模型上扫**，已加 gamma2_scale 旋钮 | V8 表 + 既有 verify_gamma2 |

## 【用户修订与新实验】（2026-08-18）

1. 用户注释停用 anchor 路径（回到论文纯版）；其 τ=0 ridge 改写引入语法错误已按原意修复
   （epsilon 仅 τ=0，M_tau 不含 anchor——与停用的 b_tilde 保持一致）。
2. S 语义修正：S=论文 inner iterations（l.8，由 kw num_langevin 供值），非 Langevin 步数；
   注释已改。
3. 新参数：x0_langevin_steps（K 次 l.16-17，l.15 仅一次——循环 l.15 会重置 x0 使迭代退化，
   按此语义实现）；gamma2_scale（γ² 表乘子，0=关）。
4. GPU 扫描 job 18641220：box·junco，K∈{1,3,5}×γ²scale∈{1,0}，h0=0.1，anchor 停用。
   （注：K>1 正对准此前诊断的"l.15 清零 Block2 进展"问题——若 K↑ 显著填洞，
   即论文机制内的修复路径，可替代/补充 anchor 方案。）

### x0-Langevin K 与 γ² 扫描结果（job 18641220，2026-08-18）
box·junco 洞区 MSE：K=1/γ²meas **0.9691**（=基线逐位复现，新参数默认路径零漂移）；
K=1/γ²=0 0.9782（γ² 影响 ~1%，与 V8 白噪声探针一致）；
K=3 2.13/2.24；K=5 2.49/2.54——**K↑ 单调恶化**（与 h₀ 扫描同签名：Block-2 churn 增多、
x₁ 无收缩机制不变）。

**最终结论**：论文机制内三旋钮（h₀↑、x₀-Langevin K↑、γ²）对 box 洞区全部无效或有害；
唯一有效干预仍是 Tweedie anchor（7/7 图 6.2×，当前被用户停用）。结构性诊断
（Block-1 条件分布在 ker(A) 无 x₁ 先验）三重确认。算法层修改待用户裁决。

## 【目录重组】（2026-08-18，用户指令）

Algorithm2/ 收敛为 utils.py + main.py + 5 个 JSON（4 模式配置 + gamma2_meas 数据表）+ results/。
代码体逐行搬运（非重写）：数学件+采样器 → utils.py；四个入口 → main.py 模式函数
（onestep/full_ip/debug_box/verify，`python main.py --config <mode>.json`）；15 个旧文件
git rm（历史保留）；sbatch 改用 `sbatch --wrap` 一行式（见 main.py docstring，含
PYTHONHASHSEED=0）。SUMMARY.md 的论文映射浓缩进 utils.py docstring。
等价性证据：CPU verify ALL PASS；onestep self-test S1 1.08e-07 / S2 0.0110（=历史记录）；
GPU job 18643397 box K=1 洞区 MSE 0.9691 / obs 0.0018（=重组前逐位）。
