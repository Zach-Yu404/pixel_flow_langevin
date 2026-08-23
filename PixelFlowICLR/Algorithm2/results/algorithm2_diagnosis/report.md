# Algorithm 2 最小诊断 —— transition 对比 / line 15 / direct_estimate_x1

`box_inpainting` / `junco` / seed 42 / `config.json` 现值（`trw=0`, `num_langevin=10`,
`ode_steps_per_stage=10`, `guidance_scale=2.0`, `h0=0.1`, `ridge_rel=1e-6`）。
被诊断对象是**当前的 `utils.py`**，即已含 `data_rhs_matchx1` + `direct_estimate_x1`
的 prior-injection 版本。

## 1. Stage transition —— 与原始实现逐行对比

参考实现：`PixelFlow/IP_package/ms_posterior_sampling_article_version_final.py`
的 `run_posterior_sampling`（l.448 起）。用户给的路径
`PixelFlow/ms_posterior_sampling_article_version_final.py` 是同一文件的 439 行前缀，
**不含采样器**（只有 `_per_stage` 和 `main`）；`utils.py` 实际 import 的也是 IP_package 那份。

| 项 | 参考 `run_posterior_sampling` | Algorithm 2 `utils.py` | 一致？ |
|---|---|---|---|
| `x1` upsample | l.602 `F.interpolate(x1_k, (h,w), mode="nearest")` | l.325 `F.interpolate(x1, (h,w), mode="nearest")` | **一致** |
| upsample mode | `nearest` | `nearest` | **一致** |
| stage index / `g_bypass` | l.592 `eff_si = si if g_bypass_stage3 else None` | l.320 同一行 | **一致** |
| `x_tau` 构造式 | l.634 `apply_H_tau(x1,τ,s,e,eff_si) + σ_τ·eps` | l.351 同一式 | **一致** |
| 新 stage 的 `x0` | **不是 fresh noise**：l.596–603 把上一 stage 的 `latent_tau` 上采样后做 renoise，再由它反解 `eps_k` | l.326 `x0 = randn_like_cpu(pyr[si])`，每个 stage 全新 iid | **不一致** |
| noise 分布 / 相关性 | `sample_block_noise`：每个 2×2 块内 4 像素协方差 `(1−γ)I + γ11ᵀ`，γ=−1/3 | `torch.randn`：iid | **不一致** |

参考的 stage transition（l.596–603）完整是：

```python
latent_tau = F.interpolate(latent_tau, (h, w), mode="nearest")
ost = sc.original_start_t[si]; gam = sc.gamma          # gam = -1/3
alpha = 1 / (sqrt(1 - 1/gam) * (1 - ost) + ost)
beta  = alpha * (1 - ost) / sqrt(-gam)
noise = sample_block_noise(sc, B, 3, h, w)
latent_tau = alpha * latent_tau + beta * noise         # PixelFlow renoise
x1_k  = F.interpolate(x1_k, (h, w), mode="nearest")
eps_k = (latent_tau - s_k * apply_G(x1_k, eff_si)) / max(1 - s_k, 1e-8)
```

**block noise 的性质（实测，CPU）**：每像素边缘方差 1.0083，但
`‖G·n‖/‖n‖ = 4.9e-04`，每个 2×2 块的均值 `max|avg_pool2d(n,2)| = 2.0e-03`。
即 **block noise 精确地只活在 ker(G) 里**（块内和为零），range(G) 上一点不加。
对照 iid 高斯：`‖G·x‖/‖x‖ = 0.5046`，`max|avg_pool| = 1.715`。

所以两者的差别不是"换了个随机种子"，而是结构性的：

- 参考实现跨 stage **保留** `latent_tau` 的粗尺度内容（range(G)），只在新出现的
  高频子空间 ker(G) 注入噪声——这正是 PixelFlow 金字塔的 ancestral transition。
- Algorithm 2 每进一个 stage 重抽 iid `x0`，**一半噪声能量落在 range(G)**，
  把上一 stage 已经建立的粗尺度内容重新盖掉。

这是 **有意的、已在 docstring 里写明的偏离**（`utils.py` 的 sampler docstring：
"the ancestral transition U^(1) + fresh x0 (l.20) instead of the renoise transition"），
对应论文 Algorithm 2 的 l.20。**不是 bug**。同理 l.1 的 `x1 = 0`（参考是
`x1_k = torch.randn`）也是论文规定。

**结论**：transition 在 upsample、mode、stage index、`g_bypass`、`x_tau` 构造式上
逐行一致；`x0` 的来源与噪声相关性按论文 l.20 有意不同。按要求不为它设计额外实验。

### 与 §2 直接相关的一处补充（同属代码对比）

参考实现里 **有** l.15 的对应物，但位置和角色都不同：

- `run_posterior_sampling` l.654：`eps_k = (x_tau_k − H_tau x1_k)/sigma_t`
  —— **每个 ODE step 只做一次**，紧跟在 `warm_restart` 用
  `x1_k = direct_estimate_x1(...)` 整体替换 x1 之后，作用是让 (x1, eps) 与网络刚看过的
  x_tau 保持一致。
- 内层 Langevin（`_langevin_step`，`..._utils.py` l.247–331）**没有**这个 restore：
  `(f)` 对 eps 做梯度/Langevin 更新，`(g)` 用 **新的 x1 和新的 eps** 重建
  `x_tau_k = H_tau x1_k + sigma_tau eps_k`。

即：参考实现的内层迭代里，新的 x1 **会**进入 x_tau；Algorithm 2 的 l.15 把 restore
放进了内层每一次迭代。这正是 §2 要测的结构差异。

---

## 2. line 15 —— 它是否抵消掉 Block-1 的 semantic information

两臂，除 l.15 外**完全相同**（同 image / seed / config / measurement / time grid；
l.15 不消耗随机数，两臂 xi 流逐位对齐）：

- `l15`：保持现状 `x0 <- (x_tau_old − H_tau x1_new)/sigma_tau`
- `no_l15`：不做 restore，沿用当前 x0，令 `x_tau_next = H_tau x1_new + sigma_tau x0`

Block 2、h0、ridge、CG 一律未动。

### 直接测量：‖x_τ^{after Block1} − x_τ^{before Block1}‖ / ‖x_τ‖

（每个 ODE step 内 10 次 inner iteration 的均值）

| arm | 范围 |
|---|---|
| `l15` | **1.6e-08 … 9.6e-08**（float32 舍入，等于精确为 0） |
| `no_l15` | 8.0e-04 … **7.9e-01** |

**用户假设的机制被证实**：l.15 确实把 Block-1 对 x_τ 的改动**逐位抵消掉**。

但 x_τ 并没有被冻住。同一表里 `d_step_rel`（相邻 inner iteration 之间 x_τ 的实际相对变化，
即 Block-2 之后）在 `l15` 臂是 **0.091 – 0.451**——x_τ 照常在动，只是全部动量走 x₀ 这条路。

### 结果

| | 最终洞区 | 最终可见区 | 全图 |
|---|---|---|---|
| `l15`（现状） | **0.098428** | **0.0019472** | **0.026067** |
| `no_l15` | 3.1247 | 0.0021883 | 0.78281 |
| | **差 31.7×** | | |

每 stage 末洞区：

| stage | `l15` | `no_l15` |
|---|---|---|
| 0 | 5.44 | 5.459 |
| 1 | 0.9729 | 1.604 |
| 2 | **0.1785** | 1.445 |
| 3 | **0.09843** | 3.125（stage 3 内单调**上升**：1.513 → 3.125） |

网络自己的 clean estimate `x1_hat` 洞区：

| | stage0 末 | stage1 末 | stage2 末 | stage3 末 |
|---|---|---|---|---|
| `l15` | 0.0633 | 0.0664 | 0.0654 | 0.0962 |
| `no_l15` | 0.0808 | 0.6253 | 1.336 | 3.123 |

**假设的结论方向被证伪**：去掉 restore 之后洞区不是改善，而是差 32 倍。

---

## 3. direct_estimate_x1 —— 只判断是否值得改

先确认一件事：`utils.py` 里的 `direct_estimate_x1` 与参考实现
`ms_posterior_sampling_article_version_final_utils.py:176` 的函数**逐字相同**（含 docstring）。
ker(G) 上 1.2× 的系统缩放是**原实现自带**的，不是本轮引入的。

`l15` 臂上把 `x1_hat − GT` 在洞区分解为 range(G) / ker(G)，并给出精确修正
`x1_fix = G x1_hat + (I−G)x1_hat / 1.2` 的收益：

| stage | step | `x1_hat` 洞区 | range(G) | ker(G) | ker 占比 | `/1.2` 修正后 | 收益 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.09895 | 0.08927 | 0.00968 | 9.8% | 0.09876 | 0.19% |
| 1 | 9 | 0.06639 | 0.05704 | 0.00935 | 14.1% | 0.06624 | 0.23% |
| 2 | 0 | 0.08461 | 0.07604 | 0.00857 | 10.1% | 0.08398 | 0.74% |
| 2 | 9 | 0.06544 | 0.05724 | 0.00820 | 12.5% | 0.06500 | 0.67% |
| 3 | 任意 | 0.081–0.091 | 全部 | **0.00000** | 0% | 不变 | 0% |

stage 1/2 的误差 86–90% 在 range(G)，`/1.2` 修正最多只拿回 **0.74%**；
stage 3（产出最终图的那一级）G=I，ker(G) 为空，修正**完全无效**。
相对最终洞区 0.0984 和 μ 的 0.07–0.16，这个量级差 2–3 个数量级。

---

## Analysis

**观察 1**：`l15` 臂的 `‖x_τ^{after} − x_τ^{before}‖/‖x_τ‖ = 1.6e-08 … 9.6e-08`。
→ 排除"l.15 只是近似恢复"的可能，它是精确恒等式：`H_τx₁_new + σ_τ·(x_τ−H_τx₁_new)/σ_τ ≡ x_τ`。
→ 支持用户提出的机制：Block-1 对 x_τ 的改动确实被完全抵消。

**观察 2**：去掉 restore 后洞区从 0.0984 变成 3.125（差 32×），且 stage 3 内单调上升。
→ 排除"l.15 是瓶颈"。
→ 支持相反的解释，而上一轮的数据正好给出了原因：**Block-1 的抽样是噪声主导的**
（stage 2 step 0：mean 洞区 0.12，抽样 435，噪声 rms 20.85）。l.15 抵消掉的那部分，
**绝大部分是抽样噪声，不是语义**。

**观察 3**：`no_l15` 臂里 `x1_hat` 洞区 0.081 → 0.625 → 1.336 → 3.123 单调恶化，
且与 `mu_hole` 几乎逐点相等（如 stage2 末 1.336 / 1.332）。
→ 排除"Block-1 求解出问题"（mean 一直忠实跟着 x̂₁）。
→ 支持正反馈链：噪声主导的 x₁ 进入 x_τ → 网络输入偏离流形 → x̂₁ 变差 →
`b_τ = A^T y/η² + H^T H x̂₁/σ²` 变差 → x₁ 更差。

**观察 4**：`l15` 臂 `d_step_rel` 恒在 0.09–0.45。
→ 排除"l.15 把链条冻死"。语义靠 **x₁ 本身**（它就是状态）向前传，x_τ 只是网络的
查询点；x_τ 的更新全部由 Block-2 对 x₀ 的 Langevin 提供，幅度受控。

**观察 5**：stage 1/2 的 `x1_hat` 误差 86–90% 落在 range(G)，ker(G) 修正只值 0.19–0.74%，
stage 3 上为 0。
→ 排除 `direct_estimate_x1` 的 1.2× 偏差是当前瓶颈。

**综合**：l.15 不是缺陷，而是当前配置下的保护机制——它把噪声主导的 Block-1 抽样挡在
网络输入之外。真正的瓶颈仍是上一轮定位的那个：Block-1 的采样协方差 `M_τ^{-1}` 太大，
以及 stage 2→3 过渡时语义内容的丢失。

---

## Decision

**MUST FIX**：无。本轮没有任何实验证明当前实现存在 correctness 问题。

**KEEP**：
1. **line 15 原样保留**——去掉它使洞区差 31.7×（0.0984 → 3.1247）。
   按既定判据（"如果没有明显改善，则保留 line 15"）结论明确。
2. **`direct_estimate_x1` 原样保留**——ker(G) 的 1.2× 偏差在 stage 1/2 只占误差的
   10–14%，精确修正只值 0.19–0.74%，stage 3 上为 0；且该函数与参考实现逐字相同，
   改它会偏离所有 baseline 共用的代码路径。
3. **stage transition 原样保留**——upsample / mode / stage index / `g_bypass` /
   `x_tau` 构造式与参考实现逐行一致；fresh iid `x0` 与 block noise 的差别是论文
   Algorithm 2 l.20 明文规定的偏离，已在 docstring 记录。

**OPTIONAL**（有理论依据、当前证据不足以支持动手，**未实施**）：
- 参考实现的 renoise transition 只在 ker(G) 注入噪声（实测 `‖G·n‖/‖n‖ = 4.9e-04`），
  而 Algorithm 2 的 iid `x0` 有一半能量落在 range(G)，会盖掉上一 stage 的粗尺度内容。
  这与"stage 2→3 语义丢失"的现象方向一致，但本轮按要求未为 transition 设计实验，
  所以只是相关性，不是证据。
- `direct_estimate_x1` 若要保留 G 的最小修正：
  `x1 = G·x̂₁ + (I−G)·x̂₁ · (e_k−s_k)/(e_k(1−s_k))`（stage 1/2 上除数恰为 1.2，
  stage 0/3 为 1，即自动无操作）。一行，可逆，但按上表不值得。
