# implement-algorithm1

state: working · owner: claude · type: implementation · 开始：2026-08-19

## 【用户原始要求】（逐字）

> 按照我的算法实现algorithm1，写成/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm1/main_alg1.py，其中的代码可以引用utils.py的也可以对进行补充

随后（用户指出 draft 位置，逐字）：

> 在/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/results/Formatting_Instructions_for_ICLR_2027_Conference_Submissions.pdf page12

## 走过的弯路（必须记下来，避免重犯）

第一版把 Algorithm 1 认成了 **PRINCIPLE**——因为库内 `PRINCIPLE_MANUAL.md:29`、
`ARCHITECTURE.md:6`、`Experiments.md:5`、`_langevin_step` 的行内注释**全都写着
"Algorithm 1"**，而我当时判定 ICLR draft 不在本机（只找了 `*.pdf` 里名字像论文的，
把 `Algorithm2/results/Formatting_Instructions_for_ICLR_2027_Conference_Submissions.pdf`
当成了 ICLR 模板文件跳过了）。**那份文件就是 draft 本体**（26 页，p.12=Alg 1、p.13=Alg 2）。

教训：① 文件名不可信，`pdfinfo`/首页一秒就能证伪；② 库内注释说的 "Algorithm 1" 是
**上一版（NeurIPS 投稿版）**的编号，与当前 draft 的编号不是一回事——Table 3 里
"NeurIPS version" 这一列就是 PRINCIPLE。第一版实现已整体作废重写。

## Algorithm 1 是什么（draft p.12）

**Preconditioned Langevin within Gibbs.** 与 Algorithm 2 "differ in three lines and share
everything else, including Block 2 and the score evaluation"（§4）。

```
 1: x1 <- 0
 2: for k = 0..K-1:
 3:   x0 ~ N(0,I)
 4:   for tau on the stage grid:
 5:     H_tau = (1-tau)s_k G + tau e_k I;  sig_tau = (1-tau)(1-s_k) + tau(1-e_k)
 6:     B_k = e_k I - s_k G;  N_k = e_k(1-s_k)I - s_k(1-e_k)G
 7:     M_tau = (1/eta^2)A^T A + (1/sig^2)H^T H + eps I        <- (25), no lambda_pre
 8:     for s = 1..S:
 9:       x_tau = H_tau x1 + sig_tau x0
10:       v = v_theta(x_tau, tau, k)
11:       solve [N^2 + gamma^2 H^2] x0_hat = N(B x_tau - H v)  <- Cor. 8, 与 Alg2 逐字相同
   Block 1（全部差异所在）
12:       xi_y ~ N(0,I_m); xi_h ~ N(0,I_n)
13:       g = (1/eta^2)A^T(y - A x1) - (1/sig)H^T x0_hat       <- (21), Prop. 6
14:       solve M_tau delta = (h1/2) g + sqrt(h1)[(1/eta)A^T xi_y + (1/sig)H^T xi_h]
15:       x1 <- x1 + delta                                     <- invariance up to O(h1)
   Block 2（与 Alg2 逐字相同）
16-17:    x0 <- x0 - (h0/2)(x0 + x0_hat) + sqrt(h0) xi_0       <- (36)
20:   x1 <- U^(1)(x1);  x0 ~ N(0,I)
```

与 Algorithm 2 的差异**恰好是**：l.13–15 换掉 Alg2 的 l.13–14（确定性右端 + 精确解），
且 Alg2 的 l.15（`x0 <- (x_tau - H x1)/sig`）在这里**不存在**——Alg 1 是 (x1,x0) 上的
Gibbs sweep，x1 动时 x0 固定，无需重算。M_tau 是同一个算子：这里是 Prop. 2 选的
预条件子，那里是 Prop. 3 高斯条件的精度矩阵。

**符号相反不是笔误**（p.14）：Alg 1 条件在 x0 上，插值态随 x1 移动，drift 要抵消网络
报告的噪声，故 `-(1/sig)H^T x0_hat`；Alg 2 条件在 x_tau 上把它冻住，同一项是
`+(1/sig)H^T x0`。**即使估计完美 x0_hat = x0 两者也不相等**（verify B2 实测差 1.046e+01）。

## 交付

- `PixelFlowICLR/Algorithm1/main_alg1.py`（单文件入口：sampler + 三个 mode + 配置契约）
- `PixelFlowICLR/Algorithm1/config_alg1.json`（唯一配置源）

共享算子**全部 import 自 `Algorithm2/utils.py`**（`make_M_tau`、`score_solve`、
`apply_B`、`apply_N`、`data_rhs`、`power_iter_norm`、`make_exact_AT`、`mse_masked`、
`adjoint_test`、NFE hook），本文件只写 Block 1。**没有改动 Algorithm2 或任何既有文件。**
gamma^2 直接读 `Algorithm2/gamma2_meas.json`（同一张表，Alg1/Alg2 的 score solve 相同）。

## 对 l.14 的一处修正（与 Alg2 已做的那处同源）

(22) 要求 Block-1 噪声 `zeta ~ N(0, P^-1) = N(0, M_tau)`，这样推过 P = M_tau^-1 才得到
协方差 `h1 M_tau^-1`。l.14 写出的两项只实现 `M_tau - eps I`。**ridge 生效时（只在 tau=0，
H_0 = s_k G 奇异）补第三项 `sqrt(eps) xi_eps`**，协方差才精确等于 M_tau。
eps=0 时该项消失，l.14 即为原文。verify B1 稠密验证，并实测"去掉该项协方差偏 1.0e-06"，
确保这条检查不是空的。

## 超参出处

- `h1`、`S`：draft §6.4 引用的 reported settings——"(1-0.05)^15 for h1=0.1, S=15"
  （random inpainting）与 "(1-0.1)^10 for h1=0.2, S=10"（其余）；box inpainting 取 h1=0.1。
  Table 3 只认证 `h1 ∈ (0,2)`，不给定值，故 config 可调、代码里做区间校验。
- `h0=0.1`、`ridge_rel=1e-6`、`cg_max_iter_l14=200`、gamma^2 表：与 Algorithm 2 的 config 同值。
- 测量口径与 `terminal_replace_weight=0`：沿用已对齐的那套（inpainting σ_n=0.05、box 居中），
  两个算法才可比。见 `tasks/measurement-alignment-inpainting.md`。

## 验证

**verify B1–B7（CPU dense float64）ALL CHECKS PASSED**：

| 检查 | 断言 | 实测 |
|---|---|---|
| B1 | zeta 协方差 == M_tau（含 ridge）；`block1_noise` 就是该线性映射 | rel 1.4e-16 |
| B2 | (43) + (2/sig)H^T x0 == (44)，且两者**确实不等** | 2.3e-13 / 差 1.046e+01 |
| B3 | v 精确、gamma^2=0 时 score solve 还原 x0（Prop. 7） | 1.1e-15 |
| B4 | (44) 梯度下 Block-1 一步恰好留下 (1-h1/2) 的 gap（§6.4 的预测基础），h1∈{0.1,0.2,1.0,1.9} | ≤1.8e-15 |
| B4b | `block1_step` == M^-1[(h1/2)g + sqrt(h1)zeta]，即 sampler 实际调用的那条 | ≤5.3e-15 |
| B5 | ridge 恰好在 M_tau 奇异处生效：**tau=0 时 lambda_min(M0)=-1.36e-13，tau>0 全正** | PASS（Table 3 的说法实测坐实） |
| B6 | U^(1) 是 2x 最近邻；x_tau - H x1 - sig x0 == 0 | ≤2.2e-16 |
| B7 | A_k/A_k^T 伴随（三个 stage 分辨率） | ≤2.8e-17 |

与 Algorithm 2 共享的算子（G/H/B/N 自伴随、sigma_tau、CFG、M0 vs 稠密、power_iter_norm、
score_solve vs 显式 N^T N）**不重复检查**，由 `Algorithm2/main.py --mode verify` 覆盖。

**变异测试**（改生产代码看 verify 是否变红，全部命中）：
翻转 l.13 先验项符号 → 2 红；`sqrt(eps)`→`eps` → 2 红；l.14 去掉 1/2 → 4 红；
`sqrt(h1)`→`h1` → 4 红。为此把 Block 1 抽成 `block1_drift` / `block1_noise` /
`block1_step`，**verify 调的是生产函数**（V8 被删就是因为没做到这点）。

**配置契约**：键 + **类型 + 区间**，10 个用例实测全部 fail-fast——`center:"false"`、
`trw=-1.0`、`h1=2.0`（超出认证区间）、`num_langevin=True`、`cg_max_iter=0`、
blur 任务设 trw>0、operator 键拼错、per-task kw 未知键、缺 `algorithm.seed`。
（这正是 Codex 在 Alg2 提出、当时没落地的那一项，新文件直接补齐。）

**GPU smoke**（box_inpainting/junco，S=2）：NFE=78 = 4×10×2−2 ✓。

## 已观察到的现象（待完整跑完确认）

smoke（S=2）下 `post_obs=0.0043`（数据一致性正常）但 `post_hole=29.0`——洞区被
**tau=0 的 ridge 注入**打爆。机制清楚且可解析算：stage 0 的 s_0=0 ⇒ **H_0 = 0**，
故该格点 M_tau = (1/eta^2)A^T A + eps I，洞内 A^T A=0 ⇒ M=eps I 且 drift 恒 0，
只剩噪声，每步 std = sqrt(h1/eps) = sqrt(0.1/4e-4) ≈ 15.8。
Alg 2 在同一格点是 `x1 = M^-1 b`（std = 1/sqrt(eps) = 50，不累积）；
**Alg 1 是 `x1 <- x1 + delta`，会逐步累积**。这正是 Table 3 / §7.13 说的
"eps only at tau = 0; unnecessary if the grid starts at tau > 0"，
且对 Alg 1 比对 Alg 2 更严重。用户已明令 `set_timesteps` 不动，故不改网格，
如实记录该依赖（§7.13 要求"报告"的正是这个量）。

## 结果（2026-08-19，5 任务 × 3 图，NFE 390/585，测量与 Alg2 对齐）

**Algorithm 1 按 draft 自己给的 reported settings 跑，重建不出来。** 数据在 [-1,1]，
故 MSE≈1.4 比直接输出全零还差；观测区一致性正常（post_obs 0.004–0.005），坏的是其余部分。

| task | h1 | post_mse | post_hole | post_obs |
|---|---|---|---|---|
| box_inpainting | 0.1 | 0.72–0.74 | 2.87–2.94 | 0.0039–0.0041 |
| random_inpainting | 0.1 | 0.48–0.64 | 0.69–0.92 | 0.0051–0.0052 |
| gaussian_blur | 0.2 | 1.38–1.48 | — | — |
| motion_blur | 0.2 | 1.34–1.40 | — | — |
| superresolution | 0.2 | 1.28–1.36 | — | — |

### 【重要更正，2026-08-19 晚】上面这张表**不能**读作"Algorithm 1 的性能"

Algorithm 2 对照（同一 measurement.py、同一 gamma^2 表、同一组算子、同 trw=0、同一张图，
**只有 Block 1 不同**）与平凡基线：

| | box | random | gaussian | motion | SR |
|---|---|---|---|---|---|
| 平凡基线：直接用 y | 0.0217 | 0.0401 | 0.0122 | 0.0231 | 0.0078 |
| 平凡基线：全零 | 0.0561 | 0.0561 | 0.0561 | 0.0561 | 0.0561 |
| **Algorithm 2** | 0.264 | 0.572 | 0.970 | 0.828 | 1.005 |
| **Algorithm 1** | 0.720 | 0.546 | 1.479 | 1.399 | 1.361 |

**两个算法都比"什么都不做"差 10–100 倍。** 也就是说共享管线本身有一个支配性缺陷，
早于本任务存在（与上一轮 GT 诊断的结论一致：采样器洞区 0.96 vs Block 1 能做到 0.0044，差 217 倍）。
**我先前把绝对数字表述成"Algorithm 1 重建不出来"是越界的，已收回。**

### 收回：tau=0 的 ridge 注入**不是** Algorithm 1 特有的

Algorithm 2 逐 stage 轨迹显示**同样的 tau=0 爆炸，量级几乎相同**：

| stage, tau=0 | Alg 1 | Alg 2 |
|---|---|---|
| box, stage 1 | 28.51 | **28.56** |
| box, stage 2 | 111.08 | **109.53** |
| gaussian, stage 0 末 → stage 1 tau=0 | 0.012 → 0.446 | 0.012 → **1.042** |

gaussian_blur 在 stage 0 结束时 Alg2 = 0.0118（与平凡基线 0.0122 相当，**说明 stage 0 是好的**），
随后每次 stage 转移 + tau=0 就被摧毁一次。这是**共享缺陷**，不是 Block 1 的区别。

### 真正属于 Block 1 的差异：**挨打之后能不能爬回来**

两者吃同样的 tau=0 冲击，只有 Alg 2 爬得回来：

| box_inpainting/junco | stage 2 tau=0 | stage 2 末 |
|---|---|---|
| Alg 1 | 111.08 | **5.92** |
| Alg 2 | 109.53 | **0.504** |

这正是 B4c 的解析预测：Alg 2 是"解"出 x1，与 sigma_tau 无关直接落到条件均值；
Alg 1 的增量是 O(h1 sigma_tau^2)，末 stage 收缩率已坍缩到 6.4e-06。

**两者可以合起来解释**：把 tau=0 去掉后 Alg 1 的 box = **0.276** vs Alg 2 = **0.264**，
差距几乎消失。所以 Alg1/Alg2 的差别主要是**能修复多少损伤**，而不是损伤本身不同。

### h1 扫描：调不出来，而且越调越差（B4c 的预测成立）

| task | h1=0.1/0.2 | h1=0.5 | h1=1.0 | h1=1.9 |
|---|---|---|---|---|
| box_inpainting | 0.720 | 0.620 | 0.786 | 0.674 |
| random_inpainting | **0.546** | 0.999 | 1.600 | 1.682 |
| gaussian_blur | **1.479** | 1.966 | 2.337 | 2.682 |
| motion_blur | **1.399** | 2.137 | 2.494 | 2.508 |
| superresolution | **1.361** | 2.014 | 2.459 | 2.768 |

噪声按 sqrt(h1) 涨，而收缩率被 M^-1 M_A ≈ 0 卡住涨不动 ⇒ 单调变差。
box 那一列在 seed 地板内平坦。**Table 3 说 h1 "tuned, certified range (0,2)"，
但在本管线上没有任何 h1 能救它。**

### 尚未解释的第三个问题

即便去掉 tau=0，blur/SR 两个算法都仍在 MSE ≈ 1.0（平凡基线 0.012）。
**这不能归给任一算法的 Block 1**，是共享路径里另一个我尚未定位的缺陷。

### 两个互相独立的原因（已分离）

**(a) tau=0 格点的 ridge 注入。** stage 0 的 s_0=0 ⇒ H_0 = 0；k>0 时 H_0 = s_k G 秩亏。
在 ker(A)∩ker(G) 上 M_tau = eps I、drift 恒 0，只剩噪声，每内步 std = sqrt(h1/eps) ≈ 15.8，
且**每个 stage 都重新注入一次**。逐 stage 轨迹（box_inpainting/junco，mse_x1）：

| stage | tau=0 | stage 末 |
|---|---|---|
| 0 | 8.04 | 4.63 |
| 1 | **28.51** | 1.93 |
| 2 | **111.08** | 5.92 |
| 3 | 3.72 | 0.72 |

按 §7.13 的办法让网格从第一个 tau>0 开始（新增 `skip_tau0`，**默认 false = 原文逐字**；
实现方式是不执行该格点，与既有的 `sigma_min` 跳步同理，**`set_timesteps` 一行未动**）：

| task | tau=0 计入 | tau=0 跳过 | 变化 |
|---|---|---|---|
| box_inpainting | 0.720（hole 2.87） | **0.276**（hole 1.09） | −62% |
| random_inpainting | 0.546（hole 0.78） | **0.437**（hole 0.62） | −20% |
| gaussian_blur | 1.479 | 1.319 | −11% |
| motion_blur | 1.399 | 1.256 | −10% |
| superresolution | 1.361 | 1.343 | −1% |

**(b) 收缩率坍缩——这条是结构性的，跳过 tau=0 修不了。**
§6.4 预测残留 gap = (1-h1/2)^S = 0.599。实测（diag 模式，按 §6.4 的 "How to check"
每内步多解一次 CG 求 M^-1 b_tau）：

| stage | tau | gap(s=0) | gap(s=S-1) | 实测比值 |
|---|---|---|---|---|
| 0 | 0.000 | 50.9 | 157.1 | **3.09** |
| 1 | 0.000 | 237.1 | 571.1 | **2.41** |
| 2 | 0.000 | 732.8 | 2326.2 | **3.18** |
| 0–3 | >0 | — | — | **0.996–1.14** |

即 tau>0 时 Block 1 **一点 gap 都没合上**，tau=0 时 gap 还翻三倍。

**verify B4c 给出解析原因**：(43) 对 x1 是仿射的，`g = c - M_A x1`（`M_A = A^T A/eta^2`），
所以更新的 Jacobian 是 `-(h1/2) M_tau^-1 M_A`。**只有 (44)**（同一个 M 出现在两处）
才化简成 §6.4 假设的 `-(h1/2)I`。而 sigma_tau -> 0 时 `M_tau ~ H^T H/sigma^2` 压倒 `M_A`，
收缩率随之坍缩。末 stage（返回的就是它的 x1）稠密实测：

| tau | sigma_tau | 最好方向 | 最差方向 | 与 h1/2 之比 |
|---|---|---|---|---|
| 0.000 | 0.400 | 9.94e-02 | 0 | 0.994 |
| 0.900 | 0.040 | 4.10e-02 | 0 | 0.410 |
| 0.990 | 0.004 | 6.41e-04 | 0 | **0.0064** |
| 0.999 | 0.0004 | 6.40e-06 | 0 | **0.0001** |

最差方向恒为 0 = ker(A)。**结构上 Alg 1 的增量是 O(h1 sigma_tau^2)，恰好在必须收尾的地方消失；
Alg 2 是"解"出 x1，与 sigma_tau 无关直接落到条件均值上。**
draft 把 Alg 1 的代价说成 "a bias of order h1"；实测更强：在小 sigma_tau 处根本不收敛。

**注意 B4c 的口径**：该 Jacobian 把 x0_hat 冻住（算法在一次迭代内正是这么算的），
跨迭代 x0_hat 经 v_theta 依赖 x1，故最差方向为 0 只表示 **x1 自身没有回复力**，
不等于 ker(A) 完全不受约束。

### 其余 §6.5 指标（box_inpainting/junco）

- §6.3 Block-2 是否在动：max ||x0-x0(0)||/||x0(0)|| = **2.769**——h0=0.1 下动得很充分，
  draft 担心的 "h0 too small for the budget"（那是 h0=1e-3~1e-2 的情形）在这里不成立。
- 数据一致性 ||Ax1-y||/(eta sqrt(m))：4.377 -> **1.088**（目标 1）✓
- 噪声端点尺度 ||x0||/sqrt(n) = **1.345**（目标 1，偏高 35%）
- bookkeeping ||x_tau - H x1 - sig x0|| = 4.77e-07 = float32 舍入 ✓

## 进行中（结论未定，等这三组）

1. **ridge sweep** `ridge_rel ∈ {1e-8, 1e-6, 1e-4, 1e-2}`（box + gaussian，junco）——
   §7.13 要求"报告"的那个 eps 依赖量。
2. **Algorithm 2 对照**：同一份 measurement.py、同一张 gamma^2 表、同一组算子、
   同样 trw=0、同一张图，只有 Block 1 不同。**这是判定"是 Alg 1 的 Block 1 还是我的
   共享管线有问题"的关键实验**，在此之前不得下结论。
3. **h1 sweep** `h1 ∈ {0.5, 1.0, 1.9}`——Table 3 说 h1 是"tuned, certified range (0,2)"，
   所以 0.2 不是定值；不试这个旋钮就说算法不行是不公平的。
   B4c 预测它救不了末期坍缩（收缩率 ∝ h1 · M^-1 M_A，小 sigma 处 M^-1M_A ~ 1e-4），
   能救多少本身就是对 B4c 的检验。

## 未做（明确记下来，避免被当成已覆盖）

- 只跑了 3 张图 / 单 seed，没有 seed 地板估计；上一轮实测洞区 MSE 的 seed 地板约 5%
  （n=4），本轮差异（−62%、−20%）远超该量级，但 blur/SR 的 −1%~−11% **不能**据此判读。
- 没做 oracle-velocity 端到端测试（用精确速度替掉网络）。曾考虑，但点质量/高斯先验两种
  构造下 Block 2 的 x0~N(0,I) 假设都不自洽，解释会有歧义；改用 Algorithm 2 对照来隔离。
