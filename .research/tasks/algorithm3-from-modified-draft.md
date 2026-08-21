# algorithm3-from-modified-draft

state: done · owner: claude · type: experiment · 2026-08-20

## 【用户原始要求】（逐字）

> 阅读/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/results/modified.pdf page13,algorithm 3,这个是基于algorithm2改的，对比差别，然后在/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2写一个main2.py并测试

（注：Algorithm 3 实际在 modified.pdf 第 15 页；第 13 页是 Algorithm 1。
modified.pdf 中 Algorithm 2 与 Algorithm 3 同时存在，可逐行对照。）

## 与 Algorithm 2 的差别（三处）

| | Alg 2 | Alg 3 |
|---|---|---|
| l.10–11 score solve | 相同 | **完全相同** |
| 块顺序 | Block1=精确抽 x₁ → Block2=x₀ Langevin | **对调** |
| 新增 l.14 | — | `x̂₁ := (H_τ)⁻¹(x_τ − σ_τx̂₀)` ▷ (51) |
| 新增 l.15 | — | `x_τ ← H_τx̂₁ + σ_τx₀`（用先验估计重建状态） |

论文对 (51) 的定位：x̂₁ 是 `E[X1|x_τ]` 的隐含估计，即"插值的 MMSE 去噪"。
另有 (52)：`H_τ(x̂₁ − x₁) = σ_τ(x₀ − x̂₀)`——与我 8/19 独立推出并验证的恒等式相同。

**结构性约束**：l.14 需要 (H_τ)⁻¹，而 `H_0 = s_k G` 在 ker(G) 非平凡时奇异，
故 Alg 3 在 stage 0–2 无法求值 τ=0（stage 3 因 g_bypass 使 G=I 除外）。
实测跳过 3 个格点，与 §7.13"网格从 τ>0 开始"的建议一致。

## 实现忠实性证据

`main2.py` 的 `run_posterior_sampling_alg3` 逐行对应，共享 Alg 2 的全部算子
（`make_M_tau`/`data_rhs`/`score_solve`/`apply_H_tau_inv`），故差异只可能来自那三处。
内置 (52) 断言：**全程相对残差 ~1e-7（float32 精度）**，证明 (51) 的实现无符号/缩放错误。
（初版用绝对阈值 1e-3 误报，已改为相对阈值。）

## 结果：Alg 3 在标准配置下发散

box_inpainting / junco / seed 42，其余参数取 config.json 现值：

| S（内迭代） | Alg 3 洞区 | Alg 3 可见区 | Alg 2 洞区 | Alg 2 可见区 |
|---|---|---|---|---|
| 1 | **1.2541** | 1.0207 | 317.46 | **0.0044** |
| 3 | 98.66 | 5.3e11（发散中） | — | — |
| 10（默认） | **NaN** | NaN | 1.0469 | 0.0036 |

发散自 **stage 0 的 τ=0.555** 起按几何速度增长：stage 0 末 \|x₀\|=5.8e4、
stage 1 末 1e11、stage 2 变 NaN。

## 机制（解析 + 数值双重确认）

把 l.14 代入 l.15：`x_τ' = x_τ + σ_τ(x₀' − x̂₀)`，其中 x₀' 是 Langevin 更新后的值。
展开得 `x_τ' = x_τ + σ(x₀ − x̂₀) − σ(h₀/2)(x₀ + x̂₀) + σ√h₀ξ`。

对照 Alg 2 的相应更新：`x_τ' = x_τ + σ(x₀' − x₀)`，是 **O(h₀)** 的小量。
Alg 3 多出的 **σ(x₀ − x̂₀) 是 O(1) 且不随 h₀ 减小**——唯一的阻尼被移除。

数值确认（n=3·32·32 蒙特卡洛）：x̂₀ 与 x₀ 一致时两者更新相同；失配比 0→1 时
Alg 2 的更新维持 0.29 不变，Alg 3 的从 0.29 涨到 0.95（3.2×），与失配同阶。

而 x₀ 与 x̂₀ 的失配来自 l.19：`x₀ ← (x_τ − H x₁)/σ` 吸收了 `H(x̂₁ − x₁)/σ`，
x₁ 在 ker(A) 上的条件方差极大（早前实测：给 GT x_τ 时洞区仍达 40–435），
于是失配被放大后灌回状态 → 正反馈。

**Alg 3 放大的恰是它想修复的那个病理。**

## 消融：l.15 是必需的，同时也是发散源

`--no-l15`（跳过 l.15）：不再发散，但洞区 **215.49**（Alg 2 为 1.05）。
原因：没有 l.15，l.13 的 Langevin 更新会被 l.19 从未包含该更新的 x_τ 重算而**完全覆盖**，
Block 1 的工作全部作废。故 l.15 承担着"把 Langevin 更新送进 x_τ"的职责，删不得。

## 核心诊断：l.15 在注入先验的同时抹掉了数据一致性

`x̂₁ = H⁻¹(x_τ − σx̂₀)`，其中 x̂₀ 由 `score_solve(x_τ, v)` 得到——**全程不含 y**。
所以 l.15 用一个纯先验估计重建状态，等于丢弃上一次抽样获得的数据拟合。

这解释了 S=1 的对照：洞区好 253×（先验确实被注入了，意图达成），
可见区坏 232×（数据项被抹掉）。S 增大本可让重复抽样找回数据项，
但失配的正反馈会在找回之前发散。

## 建议的改法（未实现，待用户裁决）

l.15 只在 **ker(A)** 上用 x̂₁ 重建、在可观测子空间保留 x₁：

    x_τ ← H_τ[(I − P_A)x̂₁ + P_A x₁] + σ_τ x₀

其中 P_A 由算子给出（inpainting 即 mask），**不引入新超参**。
这样先验只填数据管不到的地方，数据拟合不被抹除，同时 (x̂₁ − x₁) 在 ker(A) 之外为零、
l.19 的失配来源被切断，正反馈随之消失。

## 产物

`main2.py`（Alg 3 采样器 + A/B harness + `--no-l15` / `--num-langevin` 消融开关）；
`results/alg3/alg3_box_inpainting_junco.csv`。
