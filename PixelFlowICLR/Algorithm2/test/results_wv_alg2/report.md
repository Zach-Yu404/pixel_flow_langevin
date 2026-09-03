# WV Algorithm 2 —— (x0,x1) coupling / velocity-uncertainty precision

`box_inpainting` / `junco` / seed 42 / `config.json` 现值（`trw=0`、`num_langevin=10`、
`ode_steps_per_stage=10`、`guidance_scale=2.0`、`h0=0.1`、`ridge_rel=1e-6`、
γ² 取 `gamma2_meas.json`）。三条轨迹共用同一 GT / measurement / mask / operator /
stage / time grid，且共享 RNG 流（ξ_v 走独立第二条流，见下）。

## Implemented

`utils.py`（**纯新增，274 行插入、0 行删除**——`run_posterior_sampling_alg2` 一字未动）：

- `make_M_tau_wv(...)` → `M^wv = AᵀA/η² + HᵀH/σ² + NᵀN/(σ²γ²)`，返回多一个标量 `inv_v2`
- `data_rhs_wv(...)` → `b^wv = Aᵀy/η² + Hᵀx_τ/σ² + Nᵀ[(e−s)x_τ + σ_τ v]/(σ²γ²)`
- `_x1_hat_diag(...)` → 诊断用的 clean endpoint estimate（`direct` / `inverse`）
- `run_posterior_wv_sampling_alg2(...)`

`test/test_wv_alg2.py`（`--check` / `--run` / `--report`）。

### 三个实现决定

1. **ξ_v 用独立的第二条 RNG 流**（`seed + 104729`）。共享流的位置顺序
   （x0 → ξ_y → ξ_h → ridge → ξ_0）因此与 `run_posterior_sampling_alg2` 逐位一致，
   baseline 与 WV 看到**同样的** x0/ξ_y/ξ_h/ξ_0，差别只来自 M_τ 和 b_τ。
2. **WV 版建立在未改动的 Algorithm 2 主体上**，不含当前 baseline 里那句
   "先用 x̂₁ 重建 x_τ 再抽样"。理由是数学的：r_τ 的恒等式要求 x_τ 和 v 是同一个状态，
   而 v 是在 l.9 的 x_τ 上求的。
3. **`x1_hat_method` 只影响报告，不影响采样**。coupling RHS 直接由 (x_τ, v) 构造，
   不需要 clean endpoint estimate。实测证实：两臂最终洞区 0.07541140 与 0.07541135
   （相对差 7e-7，仅来自诊断多做的一次 CG）。

## Tested

`baseline`（`utils.run_posterior_sampling_alg2`，原样）、`wv_direct`、`wv_inverse`，
外加 operator-level 检查。

## Results location

- `test/results_wv_alg2/report.md`（本文件）
- `test/results_wv_alg2/comparison.csv`、`operator_checks.csv`、`trajectories.png`
- `test/results_wv_alg2/{baseline,wv_direct,wv_inverse}/`：`trajectory.csv`、
  `final.json`、`final.png`、每个 stage 末的面板图
  （GT | measurement | x1_hat | Block-1 mean | sampled x1 | error map）
- 测试代码保留在 `test/test_wv_alg2.py`

## Key results

| Method | Hole MSE | Visible MSE | Full MSE | meas. resid |
|---|---:|---:|---:|---:|
| baseline（当前 utils.py） | **NaN** | NaN | NaN | NaN |
| baseline（上一轮实测值，见下） | 0.098428 | 0.0019472 | 0.026067 | — |
| **WV + direct** | **0.075411** | **0.0015545** | **0.020019** | 0.0029660 |
| **WV + inverse** | **0.075411** | **0.0015545** | **0.020019** | 0.0029660 |

**当前的 `run_posterior_sampling_alg2` 发散**：洞区 31.66 → 107.9（stage 0 末）
→ 9.78e10（stage 1 末），在 **stage 1 step 7（gstep 17, τ=0.777）** 变成 NaN。
所以表里第二行用的是上一轮在**同一** image/seed/config 下实测的 baseline
（`data_rhs_matchx1` 版本，该版本现已不在代码里）。相对它，WV 洞区改善 **1.31×**、
可见区改善 **1.25×**、全图改善 **1.30×**。

### 每个 stage 末的洞区 MSE

| stage | baseline（当前） | WV |
|---|---:|---:|
| 0 | 107.86 | **0.13504** |
| 1 | 9.78e10 | **0.07738** |
| 2 | NaN | **0.07230** |
| 3 | NaN | **0.07541** |

WV 的洞区**第一步就是 0.2155**，全程从未超过 0.22。此前所有变体在 stage 边界都会冲到
31.66 / 106 / 435。

### 同状态 Block-1 mean（决定性）

同一个 x_τ（WV 采样器记录的 l.9 状态）、同一套 CG 设置，只换 M_τ 和 b_τ：

| gstep | stage | τ | `mu_wv` 洞区 | `mu_base` 洞区 | 倍数 |
|---|---|---|---:|---:|---:|
| 0 | 0 | 0.000 | 0.1017 | 0.2163 | 2.1× |
| 9 | 0 | 0.999 | 0.0862 | 2.824 | **32.8×** |
| 14 | 1 | 0.444 | 0.0657 | 3.372 | **51.4×** |
| 20 | 2 | 0.000 | 0.0695 | 0.7388 | 10.6× |
| 34 | 3 | 0.444 | 0.0715 | 0.1139 | 1.6× |

### mean vs sampling noise

| gstep | `mu_wv` 洞区 | 抽样 x1 洞区 | 比值 |
|---|---:|---:|---:|
| 9 | 0.0862 | 0.135 | 1.57× |
| 19 | 0.0603 | 0.0774 | 1.28× |
| 29 | 0.0677 | 0.0723 | 1.07× |
| 34 | 0.0715 | 0.0762 | 1.07× |

对照上一轮：抽样/mean 曾是 **54×**（5.44 对 0.10），stage 2 起点更是 **3700×**
（435 对 0.118）。

### direct vs inverse 的 x1_hat

| gstep | stage | direct 总 | direct range(G) | direct ker(G) | inverse 总 | inverse range(G) | inverse ker(G) |
|---|---|---:|---:|---:|---:|---:|---:|
| 9 | 0 | 0.06204 | 0.04832 | 0.01372 | **0.06186** | 0.04814 | 0.01372 |
| 19 | 1 | 0.06545 | 0.05612 | 0.00933 | **0.06464** | 0.05573 | 0.00891 |
| 29 | 2 | 0.06872 | 0.06176 | 0.00696 | **0.06845** | 0.06170 | 0.00675 |
| 34 | 3 | 0.07174 | 0.07174 | 0.00000 | **0.07145** | 0.07145 | 0.00000 |

`inverse` 在有定义处一致略优（0.3–1%），但在 **3 个 τ=0 网格点**（stage 0/1/2）
H_τ 奇异、(51) 不可辨识，按要求报告为 unavailable 而**不**加 ridge/伪逆；
采样器本身照常运行，全程无 NaN/Inf。

### Operator-level 检查（`operator_checks.csv`）

- **V-WV1**：oracle displacement `d_τ = B_k x1 − (e−s)x0` 下
  `r_τ = (e−s)x_τ + σ_τ d_τ` 与 `N_k x1` 的相对最大误差
  **5.8e-08 … 2.0e-07**（12 个 (stage, τ) 组合，float32 精度）。恒等式成立。
- **V-WV2**：抽样 RHS 噪声在随机探针 u 上的方差与 `uᵀM^wv u` 的相对偏差
  **1.5e-03 … 2.0e-02**（4000 次抽样，容差 6%）。协方差匹配。

## Analysis

**coupling-derived WV term 有效，而且是本项目至今唯一一次把 sampling noise 压下去的改动。**

机制是清楚的：洞区落在 ker(A)，那里原来的 `M_τ = HᵀH/σ²` 很弱，于是后验又宽又偏
（`mu_base` 洞区一路涨到 3.37）。加进 `NᵀN/(σ²γ²)` 之后，因为 γ² ≈ 0.007–0.017，
这一项比 `HᵀH/σ²` 大一个量级以上，洞区的后验精度整体抬升 → **均值变准（最多 51×）、
方差同时变小（抽样/mean 从 54× 降到 1.07–1.57×）**。而且权重是 γ²（实测表）给的，
不是手调 λ。

**这解决的正是前几轮定位的瓶颈。** 之前的结论是"mean 好、抽样是噪声主导"；WV 把
这两头一起修好了：mean 更好，且抽样不再淹没它。洞区第一步就是 0.2155，
不再有 31.66 / 435 那种 stage 边界爆炸。

**direct vs inverse：inverse 略好，但差距很小，且不是全域可用。**
差异集中在 ker(G)：stage 1/2 上 inverse 的 ker 分量一致更低
（0.00891 vs 0.00933；0.00675 vs 0.00696），这与之前测出的
`direct_estimate_x1` 在 ker(G) 上放大 1.2× 的偏差一致。stage 3（G=I，ker 为空）
inverse 仍略优（0.07145 vs 0.07174），说明还有一小部分来自 range(G)——
`inverse` 经过 x̂₀，带上了 score solve 的 γ² 正则。
但由于两者只作诊断用、不进 RHS，**对最终结果没有任何影响**（0.07541140 vs 0.07541135）。

**顺带的一个结构性观察**：WV 的 RHS 在洞区实际给出的 clean 估计是
`N⁻¹[(e−s)x_τ + σ_τ v]`，在 range(G) 上与 `direct_estimate_x1` 完全相同，
在 ker(G) 上恰好差 `(e−s)/(e(1−s))` = 1/1.2。也就是说 **coupling 推导自动带上了
G 感知的正确缩放**——不需要手工修正 `direct_estimate_x1` 的 1.2× 偏差。

**问题现在来自哪里**：不再是 mean，也不再是 sampling noise（两者都已收敛到 0.07 附近）。
最终洞区 0.0754 与逐步 mean 0.0715 已经非常接近——**剩下的是 x̂₁ 本身的误差**
（全程 0.062–0.114，且 86–90% 在 range(G)）。视觉上，洞区现在是一片连贯、可信的背景，
树枝穿过洞区接得上，鸟被移除了（见 `wv_direct/final.png`）——这是合理的 inpainting 行为，
但它不是被遮挡物体的重建。

**data consistency 没有被牺牲**：可见区 0.0015545（上一轮 baseline 0.0019472，改善 1.25×），
measurement residual 0.0029660，与 σ_n² = 0.0025 同量级。

## Decision

```
BEST CURRENT VARIANT:
    run_posterior_wv_sampling_alg2（x1_hat_method 取哪个都一样；
    要报告最准的 clean estimate 就用 "inverse"，但它在 3 个 tau=0 点不可用）
    洞区 0.075411 / 可见区 0.0015545 / 全图 0.020019

WHY:
    coupling 项 N^T N/(sigma^2 gamma^2) 同时修好了 mean 和方差：
    同状态下 Block-1 mean 的洞区误差降低 1.6-51.4 倍，
    抽样/mean 之比从 54 倍降到 1.07-1.57 倍。
    权重由实测 gamma^2 给出，没有引入手调 lambda。
    operator 级恒等式 r_tau = N_k x1 与噪声协方差均已数值验证。

KEEP BASELINE:
    yes —— run_posterior_sampling_alg2 一字未动（274 行插入、0 行删除）。
    但请注意：它当前的形态会在 stage 1 step 7 发散成 NaN。

NEXT ISSUE:
    x1_hat 自身的误差（全程 0.062-0.114，86-90% 落在 range(G)）。
    mean 和 sampling noise 都已不是瓶颈，最终洞区 0.0754 已贴着逐步 mean 0.0715。
```

---

## 追加：b_tau 不加那三项随机噪声（`block1_noise=False`）

`wv_nonoise` 臂：`b_tilde = b_det`，即去掉 `(1/eta)A^T xi_y`、`(1/sigma)H^T xi_h`、
`(1/(sigma*gamma))N^T xi_v`（tau=0 的 `sqrt(epsilon)*xi_eps` 一并去掉）。
ξ 仍然照抽、只是不用，所以 RNG 流位置与采样版逐位对齐，两条轨迹只差这一项。

| Method | Hole MSE | Visible MSE | Full MSE | meas. resid |
|---|---:|---:|---:|---:|
| WV + direct（抽样） | 0.075411 | 0.0015545 | 0.020019 | 0.0029660 |
| **WV 无噪声（取均值）** | **0.072602** | **0.0011736** | **0.019031** | **0.0026928** |
| 改善 | 3.7% | 24.5% | 4.9% | 9.2% |

每 stage 末洞区：

| stage | 抽样 | 无噪声 |
|---|---:|---:|
| 0 | 0.13504 | **0.08613** |
| 1 | 0.07738 | **0.06040** |
| 2 | 0.07230 | **0.06755** |
| 3 | 0.07541 | **0.07260** |

**实现自检**：无噪声臂每一步 `x1 − mu` 恒为 `0.00000`，确认 l.14 现在返回的正是条件均值。

**抽样与均值的差距随进程收缩**：`x1 − mu` 在抽样臂上是
+0.114（gstep 0）→ +0.049（gstep 9）→ +0.017（gstep 19）→ +0.0047（gstep 34）。

**两条轨迹的均值几乎重合**（gstep 0：0.10172 vs 0.10121；gstep 4：0.08562 vs 0.08555），
说明噪声主要是在每一步上叠加方差，并没有把链条推到不同的区域——WV 精度把状态束得很紧。

### 怎么理解这个结果

**MSE 变好几乎是构造上必然的**：后验均值是 MSE 意义下的最优点估计，所以拿均值代替抽样
本来就应该降 MSE。3.7% 的洞区增益**不能**被解读成"Lemma 5 的噪声是错的"。

**代价是它不再是 sampler**：去掉噪声之后 Block 1 返回的是条件均值而非后验样本，
整条链不再是目标分布的 Gibbs sampler，而是一个交替求条件均值的确定性格式。
随之失去的是：后验样本、不确定性量化、以及 Lemma 5 / 论文的采样保证。
视觉上两者非常接近（见两个 `final.png`），无噪声版略平滑一点。

**取舍取决于目标**：要报 MSE/PSNR 这类失真指标，无噪声版更好且更省事；
要的是"从后验里采样"这件事本身（论文的命题、多样性、误差棒），就必须保留噪声。
