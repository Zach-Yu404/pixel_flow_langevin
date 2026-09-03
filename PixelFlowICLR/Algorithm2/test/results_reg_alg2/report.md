# Tweedie pseudo-observation anchor for Algorithm 2

`box_inpainting` / `junco` / seed 42 / `config.json` 现值。所有臂共用同一 image / seed /
mask / measurement / config / h0 / CG / transition / line 15 / stage schedule；
**唯一变量是 λ**（外加 λ=25 的一次 P=I vs P=nullspace 对照）。

## Implementation

**这是恢复原来那版做法**：`sample_alg2(anchor=lam)`，加于 commit `74a0c39`
（"Debug box Alg2 hole: root-cause + Tweedie-anchored Block 1 (opt-in)"），
随后在 `8ff8091`（"Drop the approaches that did not work"）被删除。原式为

```python
diag = ridge + anchor
M_fn = lambda x: M0(x) + diag * x                       # M += anchor * I
...
if anchor > 0:
    x1_model = direct_estimate_x1(x_tau - tau*v, x_tau + (1-tau)*v, sk, ek)
    b = b + anchor * x1_model + math.sqrt(anchor) * randn_like_cpu(x1)
```

当时的记录：junco 洞区 **0.969 → 0.086**（λ=25，11.3×），7 图 pooled 1.097 → 0.178。

现在写进 `utils.py` 的 `run_posterior_reg_sampling_alg2`（**纯新增，
`run_posterior_sampling_alg2` 与 `run_posterior_wv_sampling_alg2` 未动**）：

```
M_tau  = M_tau^(0) + lambda * P
b_tau  = b_tau^(0) + lambda * P x1_model
b_tilde = b_tau + (1/eta) A^T xi_y + (1/sigma) H^T xi_h + sqrt(lambda) P xi_a
```

`anchor_P="identity"` 是**默认**，即上面的原式（P=I ⟹ `λ P = λ I`、
`sqrt(λ) P ξ_a = sqrt(λ) ξ_a`）。`M^(0)` / `b^(0)` 用原 `make_M_tau` / `data_rhs`
（`H^T x_tau` 形式）；`x1_model` 复用 l.10 的 v，**零额外 NFE**，
`direct_estimate_x1` 未修改。

`anchor_P="nullspace"` 保留为可选：P 为 stage 分辨率上的 hole 指示器。
它是精确正交投影（`P²−P`、自伴均为 0.00e+00），但因 `A_k = A∘U` 且 U 是插值上采样，
"落在 ker(A_k) 内"**只在 stage 3 精确**，stage 0–2 泄漏 16%–42%（`checks.csv`）。
blur/SR 的 `get_mask` 无 hole 时 fallback 到 P=I，并在 `rows["anchor_P"]` 标出，不静默。

### 与 74a0c39 的两处有意差异（都是后来的修复，不属于 anchor 本身）

1. **τ=0 的 ridge 保留 `sqrt(epsilon) xi_eps`**。74a0c39 把 `epsilon` 放进 M 却没有配
   对应的右端噪声，那样抽到的协方差是 `M⁻¹M₀M⁻¹` 而不是 `M⁻¹`。
2. **`xi_a` 在每个 λ（含 0）都抽**，只乘 `sqrt(lambda)`。74a0c39 只在 `anchor > 0`
   时抽，导致不同 λ 之后的 RNG 流错位、彼此不可比。现在所有 λ 每步消耗完全相同
   数量/shape/顺序的随机数，扫描是 paired-noise 比较；代价是相对
   `run_posterior_sampling_alg2` 每次内迭代多一次抽样，所以 **λ=0 是本族内的基线，
   而非那个函数的逐位复现**（λ=0 实测洞区 1.0062，与历史记录 0.9691/1.0276/1.047 同水平）。

如果这两处也要回到原样，改回来各是一行。

## Lambda sweep（P = I，原形式）

| lambda | hole MSE | visible MSE | full MSE | meas. resid | hole sample RMS |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.006200 | 0.0036286 | 0.254272 | 0.0021282 | 0.0469 |
| 5 | 0.516544 | 0.0035943 | 0.131832 | 0.0021267 | 0.0467 |
| 10 | 0.159661 | 0.0035559 | 0.042582 | 0.0021242 | 0.0464 |
| 25 | 0.115043 | 0.0034351 | 0.031337 | 0.0021161 | 0.0457 |
| 50 | 0.107578 | 0.0032436 | 0.029327 | 0.0021058 | 0.0446 |
| **100** | **0.102337** | **0.0028924** | **0.027753** | **0.0020962** | 0.0426 |

洞区 **1.0062 → 0.1023（9.8×）**。可见区与 measurement residual **也随 λ 单调改善**
（可见区 −20%），即 anchor 在观测区并没有与数据打架。

（`hole sample RMS` 取最后一个有 Block-1 求解的 step，即 stage 3 step 8；那里 σ_τ 已很小，
各 λ 几乎相同——真正有区分度的是中间步，见下。）

## Stage-end results

| lambda | stage 0 | stage 1 | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 0 | 15.7067 | 3.2063 | 1.9761 | 1.0062 |
| 5 | 0.2416 | 0.2151 | 0.1905 | 0.5165 |
| 10 | 0.1647 | 0.1428 | 0.1512 | 0.1597 |
| 25 | 0.1169 | 0.0950 | 0.1165 | 0.1150 |
| 50 | 0.0984 | 0.0770 | 0.1016 | 0.1076 |
| 100 | 0.0853 | 0.0672 | 0.0942 | 0.1023 |
| 25 nullspace | 0.1170 | 0.0950 | 0.1167 | 0.1160 |

**λ=0 第一个 stage 末就是 15.7**；只要 λ≥5 全程压在 0.55 以下。

## 代表性 late-τ step 的 mean / variance 分解

`gstep 19`（stage 1 最后一步，τ=0.999）：

| lambda | `x1_model` hole | Block-1 mean hole | sampled hole | rms(x1−μ) hole | 1/√λ |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.53859 | 2.37677 | 3.20633 | 0.92807 | — |
| 5 | 0.06275 | 0.07945 | 0.21506 | 0.37692 | 0.4472 |
| 10 | 0.06272 | 0.06956 | 0.14283 | 0.27958 | 0.3162 |
| 25 | 0.06271 | 0.06453 | 0.09497 | **0.18360** | 0.2000 |
| 50 | 0.06273 | 0.06181 | 0.07703 | 0.13247 | 0.1414 |
| 100 | 0.06279 | 0.05974 | 0.06718 | 0.09527 | 0.1000 |

`gstep 38`（stage 3 step 8，σ_τ 已很小）：

| lambda | `x1_model` hole | mean hole | sampled hole | rms(x1−μ) |
|---:|---:|---:|---:|---:|
| 0 | **0.99983** | 1.00462 | 1.00620 | 0.04689 |
| 25 | 0.11103 | 0.11298 | 0.11504 | 0.04572 |
| 100 | 0.09816 | 0.10051 | 0.10234 | 0.04255 |

## Isotropic vs nullspace（λ=25）

| P | hole MSE | visible MSE | meas. resid |
|---|---:|---:|---:|
| **identity（原形式）** | **0.115043** | **0.0034351** | **0.0021161** |
| nullspace | 0.116042 | 0.0036451 | 0.0021319 |


## 与 deprecated 原版的逐点对照（`ref_*` 臂）

为了不靠推断，`test/_deprecated_anchor_ref.py` 把 74a0c39 的 `sample_alg2`
**逐字**搬了出来（只把已合并的 `algorithm2` 模块名改成 `utils`，采样体一行未动，
**包括**当前版本修掉的那两处：τ=0 无 `sqrt(ridge)` 右端项、`xi_a` 只在 `anchor>0` 时抽），
在**当前**的 image / seed / measurement / config 下跑：

| arm | hole | visible | full | meas. resid |
|---|---:|---:|---:|---:|
| `ref_0`（原版逐字，λ=0） | 0.975018 | 0.0036298 | 0.246477 | 0.0021246 |
| `lambda_0`（本版，λ=0） | 1.006200 | 0.0036286 | 0.254272 | 0.0021282 |
| `ref_25`（原版逐字，λ=25） | 0.135513 | 0.0034128 | 0.036438 | 0.0021136 |
| `lambda_25`（本版，λ=25） | **0.115043** | 0.0034351 | 0.031337 | 0.0021161 |

stage-end hole：`ref_0` 15.983 / 3.254 / 1.032 / 0.975，`ref_25` 0.1129 / 0.1106 / 0.1173 / 0.1355。

**结论三条**：

1. **两版算法一致，数值差异只来自那两处有意保留的修复。** λ=0 差 3.2%
   （0.9750 vs 1.0062），落在已测过的 seed 散布内（4-seed 扫描 sd 2.4%、极差 5.3%）；
   λ=25 差 15%（0.1355 vs 0.1150），本版更好。两处修复都改变了 RNG 流的位置，
   所以差异本来就不该为零。
2. **与历史记录 0.086 的差距不是代码问题。** 逐字原版在**今天的 config** 下也只给到
   **0.1355**。74a0c39 当时走的是 `demo_runner.build_setup_and_measurement` +
   `configs_best_out/box_inpainting_best.json`，**box 不居中**；`center: true` 与
   `measurement_mode` 是后来 measurement 对齐那一轮才加的。基线也同样平移
   （当时 0.969，现在 ref 0.975 / 本版 1.006）。
3. 图：`ref_vs_mine.png`（四格对照）、`lambda_gallery.png`（λ 全扫描）。

## Analysis

**1. 洞区填不上的主因是不是 Block-1 variance 太大？——是触发因素，不是最终项。**
λ=0、gstep 19 处 `sample = mean + noise²` 精确闭合：2.377 + 0.928² = 3.238 ≈ 3.206，
**均值占 74%、方差只占 26%**。更关键的是 gstep 38：λ=0 的 `rms(x1−μ)=0.0469`，
与 λ=100 的 0.0426 几乎一样——**末端方差本来就很小**，洞区却是 1.006。
真正的链条是：早/中期 precision 弱 → 抽样噪声进入状态 → 网络看到的 x_τ 偏离流形 →
它自己的 `x1_model` 退化到 **0.99983**（等于"纯噪声"）→ 均值随之退化。
**λ 是在早中期给 precision 加下限，阻断这条级联，而不是在末端压方差。**

**2. λ 增大时均值变化小、方差显著下降？——分两段。**
λ 0→5：**均值变化极大**（2.377 → 0.079，30×），方差只降 2.5×——这一步是把整条轨迹救回来。
λ 25→100 才符合你的描述：均值几乎不动（0.0645 → 0.0597，−7%），方差减半（0.1836 → 0.0953，−48%）。

**3. 最佳 λ。** hole 单调降到 λ=100（0.1023），但边际收益衰减：25→50 改善 6.5%，
50→100 再改善 4.9%。**拐点在 λ≈25–50**。不过与 nullspace 版不同，**iso 的 visible 和
measurement residual 也随 λ 单调改善**，所以在本任务上没有"λ 越大越伤观测区"的代价，
λ=100 在三个指标上都是最优。

**4. λ=25 → std≈0.2 的理论与实测一致。** 标量预测（`checks.csv`）在 m≪λ 的方向上给出
0.1916–0.1937；gstep 19 实测 0.1836。λ=5/10/50/100 实测 0.3769/0.2796/0.1325/0.0953
对预测 0.4472/0.3162/0.1414/0.1000，一律略低于 1/√λ——正是因为 `m>0` 还额外贡献了
precision。**Var = 1/(m+λ) 成立。**

**5. nullspace 相对 isotropic 没有优势。** λ=25 下 iso 的 hole 0.115043 略优于
nullspace 的 0.116042，visible 0.0034351 **也优于** 0.0036451，meas. resid 同样更好。
即**原来的各向同性形式在三个指标上都不差于 nullspace 版**，
与"λI 会把观测区拉向 model prediction 而受害"的担心相反。
原因可解释：inpainting 的 `y = op(gt)` **无加性噪声**，观测区那 0.0036 的误差来自
采样器自身的抽样噪声而非测量，所以在观测区也加 anchor 只是又补了一份信息。
**这一条不能外推到测量真有噪声的任务**（blur / SR）——那里 iso 会与数据对抗，
而且那些算子没有 hole mask，nullspace 会 fallback 到 P=I。

**与 74a0c39 原记录的对照**：当时 junco 是 0.969 → 0.086（11.3×），现在是
1.0062 → 0.1150（8.7×，λ=25）。基线与终值都略高，方向与量级一致；
差异来自此后的 config 变更（measurement 与 8 条 baseline 对齐、γ² 表、τ=0 ridge 噪声修复）。


## 随机项的必要性 / h0 / S（全部 λ=25，其余不变，paired-noise）

`run_posterior_reg_sampling_alg2` 新增两个消融开关 `block1_noise` / `block2_noise`；
两个 xi 无论开关都照抽，只是乘 0，所以 RNG 流位置对齐。

### 1. 随机项

| arm | 去掉了什么 | hole | visible | meas. resid | rms(x1−μ) |
|---|---|---:|---:|---:|---:|
| `lambda_25` | —（全噪声） | 0.115043 | 0.0034351 | 0.0021161 | 0.0457 |
| `lambda_25_nob1` | `(1/η)Aᵀξ_y + (1/σ_τ)H_τξ_h` | 0.108649 | 0.0021820 | 0.0012159 | 0.0104 |
| `lambda_25_nob2` | `√h0·ξ0` | 0.075504 | 0.0025529 | 0.0016868 | 0.0457 |
| **`lambda_25_nob12`** | **两个都去** | **0.072351** | **0.0013697** | **0.0008323** | 0.0104 |

两个都去：hole **−37%**、visible **−60%**、measurement residual **−61%**。
单独看，去掉 Block-2 的 Langevin 噪声（`nob2`，0.0755）比去掉 Block-1 的
Lemma-5 噪声（`nob1`，0.1086）对洞区更有效。

自洽性检查：`nob2` 的 `rms(x1−μ)` 保持 0.0457（它不动 Block-1 的抽样），
`nob1`/`nob12` 都降到 0.0104（剩下的是 anchor 自己的 `√λ·P·ξ_a`，未关）。
开关的行为与声明一致。

### 2. h0（全噪声，S=10）

| h0 | hole | visible | meas. resid |
|---:|---:|---:|---:|
| 1e-4 | 0.369294 | 0.0047104 | 0.0025130 |
| 1e-3 | 0.321963 | 0.0046080 | 0.0024513 |
| 1e-2 | 0.159325 | 0.0040339 | 0.0021442 |
| **0.1（config 现值）** | **0.115043** | 0.0034351 | 0.0021161 |
| 0.3 | 0.318332 | 0.0032002 | 0.0021816 |
| 1.0 | 2.040828 | 0.0033976 | 0.0023909 |

干净的 U 形，**最小值恰好落在现值 0.1**。两端都灾难：h0=1e-4 时 Block 2 几乎不动 x0、
状态冻住；h0=1.0 时 stage 3 直接崩掉（stage2 0.154 → stage3 2.041）。

### 3. S（全噪声，h0=0.1）

| S | hole | visible | meas. resid |
|---:|---:|---:|---:|
| 5 | 0.110887 | 0.0036241 | 0.0020910 |
| 10 | 0.115043 | 0.0034351 | 0.0021161 |
| 15 | 0.143474 | 0.0033098 | 0.0021196 |

**S 越大洞区越差**（0.111 → 0.115 → 0.143）。这与 §1 一致：每次内迭代都注入一次
Lemma-5 噪声，迭代越多累积越多。S=5 与 S=10 的差落在 seed 散布内，S=15 的 +29% 不是。

### 结论

随机项**对 MSE 不但不必要，而且有害**。但去掉它们就不再是目标分布的 Gibbs sampler，
而是确定性的交替极小化：要点估计就关，要后验样本就留。
`h0=0.1` 与 `S=10`（config 现值）在全噪声下已经是最优，无需调整。

## Decision

```
BEST LAMBDA:
    lambda = 100 在 hole / visible / meas-resid 三项上都最优
    （0.102337 / 0.0028924 / 0.0020962）。
    但边际收益已很小（50->100 只改善 4.9%），拐点在 lambda = 25-50。

NULLSPACE OR ISOTROPIC:
    ISOTROPIC（P = I，原形式）。lambda=25 下它在 hole、visible、meas-resid
    三项上都不差于 nullspace。nullspace 的理论优势（观测区零 bias）在本任务上
    不兑现，因为 inpainting 的 y 无噪声。对有噪声测量的任务结论可能反转。

WHY:
    anchor 给弱约束方向的 precision 加下限 Var = 1/(m+lambda)。
    早/中期 m 只有 1.7-2.6，lambda=25 把 std 从 0.61-0.77 压到 0.19，
    抽样噪声不再污染状态，网络的 x1_model 从 0.99983 恢复到 0.111。

DOES ANCHOR MAINLY FIX MEAN OR VARIANCE:
    两者都修，因果方向是"先压方差、再救均值"。
    lambda 0->5：均值改善 30 倍、方差只改善 2.5 倍（轨迹被救回）。
    lambda 25->100：均值几乎不动（-7%）、方差减半（-48%）。
    末端（gstep 38）方差在所有 lambda 下都只有 ~0.046——单纯的末端方差
    从来不是瓶颈；瓶颈是早中期方差造成的状态退化级联。

KEEP / REJECT run_posterior_reg_sampling_alg2:
    KEEP。lambda=0 退化回 Algorithm 2 的算法路径（洞区 1.0062，与历史基线同水平），
    lambda>0 把洞区从 1.006 降到 0.102-0.115，且 visible 与 measurement residual
    同时改善。
```

**未做**：没有把最佳 λ 写回任何其它 sampler，没有开下一轮实验。
