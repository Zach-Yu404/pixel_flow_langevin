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
- 实现≠论文？否——full_ip_compare.sample_alg2 与草稿 p.13 Algorithm 2 逐行一致（含 l.17 的
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
偏离登记：sample_alg2 新增 anchor 参数（默认 0.0 = 论文原算法逐位不变；实验性变体
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
