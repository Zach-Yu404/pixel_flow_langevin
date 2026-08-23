# b_tau 的 prior injection —— baseline vs modified

`box_inpainting` / `junco` / seed 42 / `config.json` 现值（`terminal_replace_weight=0`,
`num_langevin=10`, `ode_steps_per_stage=10`, `guidance_scale=2.0`, `h0=0.1`,
`ridge_rel=1e-6`, `gamma2` 用 `gamma2_meas.json`）。两次运行共用同一 GT、measurement、
mask、operator、seed、stage 与 time grid、RNG 流——唯一变量是 `b_tau` 的确定性部分。

改动（用户在 `utils.py` 中所做）：

```
baseline :  b_tau = (1/eta^2) A^T y + (1/sigma_tau^2) H_tau^T x_tau
modified :  b_tau = (1/eta^2) A^T y + (1/sigma_tau^2) H_tau^T H_tau x1_hat
            x1_hat = direct_estimate_x1(x_tau - tau*v, x_tau + (1-tau)*v, s_k, e_k)
```

`direct_estimate_x1` 不消耗随机数，所以两条轨迹的 xi 序列逐位相同。

## Tested

1. **baseline 全轨迹**（40 步）——`utils.py` 修改前的 `b_tau`。
2. **modified 全轨迹**（40 步）——用户当前的 `b_tau`。
3. **同状态 rhs 对照**：在每一步的最后一次 inner iteration，用**同一个** `x_tau`、
   **同一个** `M_tau`、**同一套** CG 设置（warm start = 当前 x1，`tol=1e-5`，
   `max_iter=50`，tau=0 时 200），分别解出
   `mu_old = M^-1 b_old` 与 `mu_new = M^-1 b_new`，并记录 `x1_hat`。
   两条轨迹上都做——所以 "只换 rhs" 的效果不依赖于状态好坏。
4. **mean / sampling-noise 分解**：同一 (k,tau) 下比较 `mu` 与实际抽样 `x1`，
   洞区 MSE 与 `rms(x1 - mu)`。
5. **`direct_estimate_x1` 的算子级校验**（CPU，与采样无关）：用精确端点构造，
   检查它在 range(G) / ker(G) 上是否还原 x1。

仪表全部在一个临时脚本里完成（`utils.py` / `main.py` / `main2.py` 未被本次测试改动）。
该脚本的 baseline 复现了已提交的基线数字 **1.0468701 / 0.0036223**（记录值 1.047 / 0.00362），
说明这份带仪表的循环拷贝与 `run_posterior_sampling_alg2` 是同一条路径。

## Results

### 最终重建

| | 洞区 MSE | 可见区 MSE | 全图 MSE |
|---|---|---|---|
| baseline | 1.0469 | 0.0036223 | 0.26443 |
| **modified** | **0.098427** | **0.0019472** | **0.026067** |
| 倍数 | **10.6×** | **1.86×** | **10.1×** |

可见区（data consistency）不但没有变差，还改善了 1.86 倍。

### 每个 stage 末的洞区 MSE

| stage | baseline | modified |
|---|---|---|
| 0 | 14.35 | 5.44 |
| 1 | 3.276 | 0.9729 |
| 2 | 1.993 | **0.1785** |
| 3 | 1.047 | **0.09843** |

### 同状态 rhs 对照（决定性的一组数）

同一 `x_tau`、同一 `M_tau`，只换右端项：

| 轨迹 | gstep | tau | `x1_hat` | `mu_old` | `mu_new` | mu_old/mu_new |
|---|---|---|---|---|---|---|
| baseline | 8 | 0.888 | 0.0720 | 13.34 | **0.1432** | 93× |
| baseline | 9 | 0.999 | 0.0917 | 9.248 | **0.1189** | 78× |
| baseline | 24 | 0.444 | 1.589 | 36.76 | **1.905** | 19× |
| modified | 19 | 0.999 | 0.0664 | 0.5331 | **0.07226** | 7.4× |
| modified | 29 | 0.999 | 0.0654 | 0.1276 | **0.0686** | 1.9× |

`mu_new` 的洞区 MSE 每一处都贴着 `x1_hat`（stage 3 上两者精确相等——那里 G=I，
`H_tau` 是 identity 的标量倍，在 ker(A) 上 `mu = x1_hat` 是恒等式）。

### mean vs sampling noise

`rms(x1 - mu)` 在两条轨迹上**逐点相同**（5.616/5.616、2.314/2.313、10.3/10.3、
20.85/20.85、0.3311/0.3311）——协方差是 `M_tau^-1`，与 `b` 无关，符合理论。

| 轨迹 | gstep | mu 洞区 | 抽样 洞区 | noise rms | mu + noise² |
|---|---|---|---|---|---|
| modified | 9 | 0.1005 | 5.44 | 2.313 | 5.45 |
| modified | 20 | 0.1175 | 435.0 | 20.85 | 435.0 |
| modified | 29 | 0.0686 | 0.1785 | 0.331 | 0.178 |
| modified | 38 | 0.09621 | 0.09843 | 0.0469 | 0.0984 |

分解精确闭合：`sample = mu + noise²`。小 tau / 早期 stage 上噪声项比 mean 的误差
大 3–4 个数量级（gstep 20：0.118 对 435）。

### x1_hat 自身的走向

| | stage0 末 | stage1 末 | stage2 末 | stage3 末 |
|---|---|---|---|---|
| baseline `x1_hat` 洞区 | 0.0917 | 0.4797 | 0.8081 | 1.039 |
| modified `x1_hat` 洞区 | 0.0633 | 0.0664 | **0.0654** | 0.0962 |

baseline 的 x1_hat 在 stage 2 内一度涨到 **1.589**（gstep 24），最后停在 1.039。
baseline 的状态越差 → 网络的 clean estimate 越差 → rhs 越差，是一个正反馈；
modified 全程稳定在 0.063–0.10。

### ||x0||_rms（目标 1）

baseline 全程 0.69–0.93，从未到 1（这正是之前诊断出的 Block-2 混合不足）。
modified 升到 0.86–1.12，末步 **1.119**。x0 是由 `(x_tau - H_tau x1)/sigma_tau` 反解的，
x1 变好之后 x0 自然回到单位尺度，Block-2 才工作在它被设计的区间里。

### `direct_estimate_x1` 的算子级校验

用精确端点（无网络误差）构造，检查还原度：

| stage | (s_k, e_k) | range(G) 上误差 | ker(G) 上误差 | ker(G) 增益 |
|---|---|---|---|---|
| 0 | (0, 0.25) | 3.6e-07 | 4.8e-07 | 1.0000 |
| 1 | (0.142857, 0.5) | 2.4e-07 | 6.2e-01 | **1.2000** |
| 2 | (0.333333, 0.75) | 3.6e-07 | 7.5e-01 | **1.2000** |
| 3 | (0.6, 1.0) | 2.4e-07 | 0（ker 为空） | — |

推导对得上：`((1-s)x_e - (1-e)x_s)/(e-s) = N_k x1/(e-s)`，在 range(G) 上系数是 1，
在 ker(G) 上是 `e(1-s)/(e-s)`，stage 1 和 2 都恰好等于 1.2。**stage 1/2 注入的先验，
其高频（ker G）分量被系统性放大 20%**；stage 0 和 stage 3 精确。

## Analysis

**先验确实进入了洞区。** 同状态 rhs 对照是干净的因果证据：状态、算子、CG 设置全部不变，
只把 `H^T x_tau` 换成 `H^T H x1_hat`，Block-1 mean 的洞区误差就从 9.2 掉到 0.12。
原因是明确的：洞区落在 ker(A)，那里 `M_tau ≈ H^T H/sigma^2 + eps`，于是
`mu_hole ≈ (H^T H)^-1 H^T H x1_hat = x1_hat`；而 baseline 的
`mu_hole ≈ H_tau^-1 x_tau = x1 + sigma_tau H_tau^-1 x0`，小 tau 下 `H_tau^-1` 把 x0
放大成主导项——这正是之前诊断中"洞区收敛到单位方差噪声"的来源。

**mean 是对的，而且带语义。** 不只是 MSE：stage 2 末的 `x1_hat` / Block-1 mean 图里，
洞区是一片连贯的背景，两根树枝穿过洞区接得上（见
`modified/stage2_step9_tau0.9990.png`）。模型没有幻觉出一只鸟——它把洞补成了背景，
这在 inpainting 上是合理行为，也是 MSE 0.065 的来源。baseline 同一位置的 x1_hat 是 0.808。

**sampling noise 现在是主要瓶颈。** 分解精确闭合，而且噪声项与 baseline 逐点相同：
`b` 的改动只动 mean，不动协方差。gstep 20 上 mean 的洞区误差是 0.118，实际抽样是 435——
**噪声比 mean 的误差大 3700 倍**。之所以端到端仍然改善 10 倍，是因为噪声尺度随
tau→1 和 stage 推进单调下降（20.9 → 0.33 → 0.047），而 mean 携带的结构会通过
`x_tau = H_tau x1 + sigma_tau x0` 传到下一步。换句话说：**每一步的抽样几乎是纯噪声，
但每一步的 mean 是好的，链条靠 mean 逐步收敛。**

**Block 2 不再是破坏者。** x0 的尺度从 0.69–0.93 回到 0.86–1.12，Langevin 现在
作用在一个尺度正确的 x0 上。stage 内洞区单调下降，没有看到 Block-2 造成的回退。

**真正丢失语义的地方是 stage 2 → stage 3 的过渡。** stage 2 末 x1_hat 是连贯背景
（0.0654）；经过 nearest 上采样 + 重新抽 x0 之后，stage 3 第 0 步的 x1_hat 变成细粒度
散斑（0.091，见 `modified/stage3_step0_tau0.0000.png`），到 stage 3 第 4 步固化成一片
**周期性洋红点阵**（`modified/stage3_step4_tau0.4440.png`、`modified/final.png`）。
MSE 看不见这个 artifact（振幅低，0.0962 与 stage 2 的 0.0654 同量级），但视觉上
stage 3 没有把 stage 2 已经建立的连贯背景继承下来。**最终图里的洞区不是鸟，也不是
干净背景，而是灰底 + 点阵**——比 baseline 的饱和 RGB 噪声好一个量级，但仍不是
可用的重建。

**一个必须记下的性质改变（不是缺陷判断，是事实）**：`b_new` 丢掉了
`sigma_tau H_tau^T x0` 这一项，Block-1 不再是论文模型在给定 `x_tau` 下的精确 Gibbs
条件分布。它对 `x_tau` 的依赖变成了非线性的、通过网络的依赖（`x1_hat = f(v(x_tau))`）。
所以修改后的采样器是一个 plug-and-play / proximal 变体，而不是 Algorithm 2 所述的
Gibbs sampler；它采的目标分布与论文写的那个不是同一个。

## Conclusion

**SUCCESS（在 conditional mean 与端到端重建这两个层面上都成立）**——
新的 `b_tau` 把网络的 clean estimate 真正注入了洞区：同状态下 Block-1 mean 的洞区误差
降低 19–93 倍，端到端洞区 MSE 从 1.047 降到 0.0984（10.6×），可见区同时改善 1.86×。

**下一个最值得解决的唯一问题：stage 2 → stage 3 的过渡。** 语义内容在 stage 2 末已经
建立好（连贯背景，x1_hat 洞区 0.065），却在进入 stage 3 后退化成周期性点阵。
这一步是当前唯一一处"先验已经拿到、随后被丢掉"的地方，而且它发生在最终输出的
分辨率上——修好它直接决定最终图像质量。（`direct_estimate_x1` 在 stage 1/2 上把
ker(G) 分量放大 1.2 倍是另一个已确认的偏差，但它不在 stage 3，量级也小得多。）
