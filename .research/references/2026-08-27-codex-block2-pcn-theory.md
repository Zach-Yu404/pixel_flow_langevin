# PixelFlow Algorithm 4：自适应 pCN/OU Block-2 refresh 独立理论审查

> 日期：2026-08-27  
> 审查方：Codex（独立理论审查；未读取 Claude 推导）  
> 范围：题面给定的现行 Algorithm 4、两种候选 \(\lambda\)、precision 数值口径与最低验证标准；不审查尚未给出的实现 diff。

## 0. 结论摘要

1. 对固定的新中心 \(m=H_\tau x_1^{\rm new}\)、\(\sigma_\tau>0\) 和常数
   \(0\leq\lambda\leq1\)，题定更新是标准 Gaussian OU/pCN 核。它保持
   \(N(m,\sigma_\tau^2I)\) 不变，并满足细致平衡。\(\lambda=1\) 仍不变且可逆，
   但只是恒等核，完全不遍历。
2. 用该核替换独立 Block 2 后，固定 \((k,\tau)\) 的 Gaussian-surrogate 联合 Gibbs
   目标不变；但系统扫描的整轮组合一般不可逆。它不是独立 conditional draw：
   \(\lambda>0\) 从非平稳输入出发不会一步得到正确 conditional，而会保留
   \(\rho=\sqrt\lambda\) 比例的旧 residual。
3. `sigma_pcn` 在合法 schedule 上 well-defined；clip 在精确算术下多余、在浮点中可作
   防御。仅当 clip 未触发且 \(\sigma_\tau\geq0\) 时，
   \(\sigma_\tau\sqrt{1-\lambda}=\sigma_\tau^2/\sigma_{k,0}\) 严格成立。
4. `precision_pcn` 在 \(S\succ0,\sigma_\tau>0\) 时 well-defined。若 \(S\) 在每个 stage
   内固定，给定四组边界下 \(q\) 在 stage 内必严格单调；任何实测下降都是实现或数据
   口径异常。stage 0 的 \(\tau\to0\) 不退化，真正奇点是 stage 3 的
   \(\tau\to1\)。
5. **当前 precision 数值输入契约未通过。** 指定的 `s_statistics.json` 没有所需的
   \(\operatorname{mean}(1/P)\)，也不能用 \(1/\operatorname{mean}(P)\) 代替；题面还未
   列出十个精确 \(\tau\)。因此不能诚实地声称仅由指定两份 JSON 得到了“实际 4×10
   双表”。本稿给出：严格闭式、同目录原始谱的补充复算值，以及在明确网格假设下的
   条件性 4×10 表。**在 exact tau grid 与 `mean_inverse_power` 被持久化并逐帧核对前，
   应按 fail-closed 原则停用 `precision_pcn`，只保留诊断。**
6. 即使补齐数据，stage 3 末帧也会使 precision 核接近 identity。条件表的 pooled
   末帧 \(\lambda\approx0.999996607\)，纯 OU 的理论 IACT 约 \(1.18\times10^6\)；
   这不是非单调 bug，而是严重的 mixing 代价。
7. fp32 下，“`independent` 跨帧逐位复现旧 baseline”与“l.16 对所有模式直接 carry
   原始 `z_new`”通常不能同时成立。若逐位兼容是硬约束，`independent` 必须保留原
   residual carry 的 legacy 分支；pCN 两模式才直接 carry `z_new`。
8. 固定目标的平稳态下，不变核不能改善终态边缘分布，只能改变耦合和 ESS。完整 40-frame
   Algorithm 4 是改变 \((k,\tau,H,\sigma,S)\) 且每帧只走两步的非齐次有限时链；若指标
   改善，只能归因于非平衡路径、较少 fresh-noise 注入及 warm-coordinate persistence，
   不能归因于改变后的 stationary target 更好。

---

## 1. 固定中心 OU/pCN 核：不变性、可逆性与 Gibbs 组合

令

\[
m=H_\tau x_1^{\rm new},\qquad
z=\frac{x_\tau-m}{\sigma_\tau},\qquad
\rho=\sqrt\lambda.
\]

提议核写成

\[
z'=\rho z+\sqrt{1-\rho^2}\,\xi,\qquad \xi\sim N(0,I).
\]

### 1.1 条件不变性

若 \(z\sim N(0,I)\) 且 \(\xi\perp z\)，则

\[
\mathbb E z'=0,\qquad
\operatorname{Cov}(z')=\rho^2I+(1-\rho^2)I=I.
\]

线性高斯性给出 \(z'\sim N(0,I)\)，故

\[
x_\tau'=m+\sigma_\tau z'
\sim N(m,\sigma_\tau^2I).
\]

所以答案是肯定的，但条件必须写全：固定 \(x_1^{\rm new}\)、固定合法 \(\lambda\)，且
\(\sigma_\tau>0\)。在 \(\sigma_\tau=0\) 处，标准化坐标本身无定义。

### 1.2 细致平衡

当 \(0\leq\rho<1\) 时，标准正态密度 \(\phi\) 与转移密度 \(k_\rho\) 满足

\[
\phi(z)k_\rho(z,z')\propto
\exp\!\left[-\frac{
\|z\|^2+\|z'\|^2-2\rho z^\top z'
}{2(1-\rho^2)}\right].
\]

右端交换 \(z,z'\) 不变，因此满足详细平衡。仿射变换回 \(x_\tau\) 后，核对
\(N(m,\sigma_\tau^2I)\) 仍可逆。

两个端点要单独说明：

- \(\lambda=0\)：\(z'=\xi\)，恰为独立 conditional draw；
- \(\lambda=1\)：\(z'=z\)，是恒等核。它作为退化 Markov 核仍满足详细平衡，却不遍历，
  从错误初值永远不会收敛到目标 conditional。

### 1.3 替换 Gibbs 子核后联合目标是否不变

设固定 frame 的 surrogate 联合目标为 \(\pi(x_1,x_\tau\mid y)\)，且题面两个 Block 确为
它的 conditional：

\[
x_1\mid x_\tau,y\sim\pi(dx_1\mid x_\tau,y),\qquad
x_\tau\mid x_1\sim N(H_\tau x_1,\sigma_\tau^2I).
\]

精确 Block 1 保持 \(\pi\)。Block 1 后，固定新 \(x_1\) 使用保持第二个 conditional 的
\(K_{x_1}\)，亦保持 \(\pi\)。因此系统扫描组合 \(P=P_1P_2\) 及其两次迭代均保持
\(\pi\)。不过，两个可逆子核的有序乘积通常不交换，所以整轮 \(P\) 一般不满足详细平衡；
能推出的是联合不变性，不是整轮可逆性。

这里的 \(\pi\) 是 Algorithm 4 的 Gaussian surrogate 联合目标。该结论不会自动把 surrogate
升级为原始 PixelFlow posterior 的严格 exact sampler；Block 1 的截断线性求解若不精确，
结论也只近似成立。

### 1.4 与独立 conditional draw 的本质区别

独立抽样无视旧 \(x_\tau\)，从任意输入出发一步就把 Block 2 conditional 重置正确。
pCN 则有

\[
x_\tau'=m+\rho(x_\tau-m)
+\sigma_\tau\sqrt{1-\rho^2}\,\xi.
\]

它只是 conditional-invariant Markov move。若输入均值相对目标有误差 \(d\)，一步后仍保留
\(\rho d\)；若中心固定、连续做两步，则旧 residual 的系数是 \(\rho^2=\lambda\)。因此它用
更小 fresh jump 换取更慢的 conditional mixing。

---

## 2. 两种 \(\lambda\) 是否 well-defined

### 2.1 `sigma_pcn`

记 \(\sigma_{k,0}=1-s_k\)。因四个 stage 都有 \(e_k-s_k=1/4\)，

\[
\sigma_\tau=1-s_k-\frac\tau4
=\sigma_{k,0}-\frac\tau4.
\]

对 \(0\leq\tau\leq1\)，四个 \(\sigma_{k,0}=1,3/4,1/2,1/4\) 均为正，且
\(0\leq\sigma_\tau\leq\sigma_{k,0}\)。故精确算术下

\[
\lambda=1-\left(\frac{\sigma_\tau}{\sigma_{k,0}}\right)^2\in[0,1].
\]

clip 在合法理论域内不需要；其合理用途只是吸收极小浮点越界。clip 不应掩盖负
\(\sigma\)、越界 \(\tau\)、NaN 或 schedule 错误。

写 \(c_k=(e_k-s_k)/(1-s_k)\)，则 \(\lambda=1-(1-c_k\tau)^2\)。理论端点为：

| stage | \(c_k\) | \(\lambda(0)\) | \(\lambda(1)\) |
|---:|---:|---:|---:|
| 0 | \(1/4\) | 0 | \(7/16=0.4375\) |
| 1 | \(1/3\) | 0 | \(5/9\) |
| 2 | \(1/2\) | 0 | \(3/4\) |
| 3 | 1 | 0 | 1 |

stage 3 若真到 \(\tau=1\)，\(\sigma=0\) 且 \(z\) 无定义；`lambda=1` 或 clip 都不能修复
除零。题面给出的实际最小正 \(\sigma\approx4\times10^{-4}\) 避开了严格端点。

当 clip 未触发且 \(\sigma_\tau\geq0\) 时，

\[
\sigma_\tau\sqrt{1-\lambda}
=\sigma_\tau\sqrt{\left(\frac{\sigma_\tau}{\sigma_{k,0}}\right)^2}
=\frac{\sigma_\tau^2}{\sigma_{k,0}}.
\]

所以题面恒等式成立。若 clip 因非法输入饱和，或代码先在 fp32 中形成
`lambda = 1-u` 再由 `1-lambda` 恢复 \(u\)，则不保证逐位恒等。更稳定的实现口径是直接保存
\(u=1-\lambda=(\sigma/\sigma_{k,0})^2\)，并直接用 \(\sqrt u\) 作为 innovation 系数。

### 2.2 `precision_pcn`

若 \(\sigma_\tau>0\)、\(S\succ0\)，且 pooled 的 \(s^2\) 或 spectral 的全部
\(P(\omega)\) 均为有限正数，则

\[
q_{k,\tau}=\frac1n\operatorname{Tr}\left(
\frac{H_\tau^\top H_\tau}{\sigma_\tau^2}+S^{-1}
\right)>0.
\]

于是 \(q_{k,0}>0\)，且

\[
\lambda=
\begin{cases}
0,&q_{k,\tau}\le q_{k,0},\\[1mm]
1-q_{k,0}/q_{k,\tau},&q_{k,\tau}>q_{k,0}.
\end{cases}
\]

因此有限正 \(q\) 下 \(\lambda\in[0,1)\)，stage 首帧为 0，\(q\to\infty\) 时趋近 1。
`min(1,q0/q)` 是必要的单侧防御：若 \(q\) 非单调，它避免负 \(\lambda\)；但生产诊断必须
同时保存 raw ratio 和 raw \(q\)，否则会把非单调 bug 静默伪装成 \(\lambda=0\)。若出现
\(q\le0\)、NaN/Inf、\(\sigma\le0\) 或 float32 将 \(\lambda\) 舍入成 1，应报错/停用，
而不是再 clip。

precision 模式同样应直接计算 fresh variance fraction

\[
u=1-\lambda=\min(1,q_{k,0}/q_{k,\tau}),
\]

然后用 \(\sqrt u\)，不要从 fp32 的 `1-lambda` 反推它。

---

## 3. `precision_pcn` 数值审查

### 3.1 可复算闭式与单调性

在 \(\operatorname{range}(G)\) 与 \(\ker(G)\) 上，\(H_\tau\) 的特征值分别是

\[
h_G=s_k+\frac\tau4,\qquad h_\perp=\tau e_k,
\]

且两类方向占比为 \(1/4\) 与 \(3/4\)。令

\[
c_k=\frac1n\operatorname{Tr}(S^{-1}),\qquad
r_k(\tau)=\frac1n\operatorname{Tr}(H_\tau^\top H_\tau)/\sigma_\tau^2,
\]

则 \(q_k=c_k+r_k\)，并可化为：

| stage | \(\sigma_k(\tau)\) | \(r_k(\tau)\) |
|---:|---:|---:|
| 0 | \((4-\tau)/4\) | \(\displaystyle\frac{\tau^2}{(4-\tau)^2}\) |
| 1 | \((3-\tau)/4\) | \(\displaystyle\frac{1+2\tau+13\tau^2}{4(3-\tau)^2}\) |
| 2 | \((2-\tau)/4\) | \(\displaystyle\frac{1+\tau+7\tau^2}{(2-\tau)^2}\) |
| 3 | \((1-\tau)/4\) | \(\displaystyle\frac{9+6\tau+49\tau^2}{4(1-\tau)^2}\) |

更直接地，\(h_G/\sigma\) 与 \(h_\perp/\sigma\) 的导数均为正，因为

\[
\frac d{d\tau}\frac{h}{\sigma}
=\frac{h'\sigma+h/4}{\sigma^2}>0
\]

（有效域内至少一项严格为正）。所以只要 \(c_k\) 在 stage 内固定，四个 stage 的
\(q_k(\tau)\) 都严格递增。若实际日志出现显著下降，应检查 \(\tau\) 排序、stage 索引、
\(S\) 口径或 trace 公式，而不能把它解释成模型现象。

stage 0 没有 \(0/0\)：

\[
q_0(\tau)=c_0+\frac{\tau^2}{(4-\tau)^2}\longrightarrow c_0>0.
\]

真正的奇点是 stage 3 的 \(\tau\to1\)，此时 \(q\to\infty\)、\(\lambda\to1\)。

### 3.2 真实统计文件审计

`s2_meas.json` 实际给出每个 stage 一个 pooled 方差，而不是 40 个逐 \(\tau\) 值：

| stage | \(s_k^2\) | \(c_k^{\rm pool}=1/s_k^2\) |
|---:|---:|---:|
| 0 | 0.03969954352354733 | 25.189206505784167 |
| 1 | 0.04718059426719991 | 21.195154820150350 |
| 2 | 0.05210772396874556 | 19.191012844848192 |
| 3 | 0.05543326541794339 | 18.039709413840637 |

`s_statistics.json` 只保存 `mean_power`、`max_power`、`floor`、`n_floored_bins`，**没有**

\[
c_k^{\rm spec}=\operatorname{mean}_\omega(1/P_k(\omega)).
\]

尤其 Jensen 给出 \(\operatorname{mean}(1/P)\ge1/\operatorname{mean}(P)\)，一般不取等。
若误用 JSON 中 `mean_power` 的倒数，会得到约

\[
(4.1121,\ 3.8852,\ 3.7414,\ 3.6295),
\]

这不是 spectral trace precision。

为量化该缺项，本稿补充读取同目录原始谱
`configs/spectral_power.npz`（SHA-256
`98eca187f2b192495fb0ae512463588434abaf09099dfe7d435e4cfd5c931454`），以 float64 对已存
float32 \(P\) 做 reciprocal-mean，得到：

| stage | \(c_k^{\rm spec}=\operatorname{mean}(1/P)\) | \(1/\operatorname{mean}(P)\) |
|---:|---:|---:|
| 0 | 110.65794904006324 | 4.112092187890551 |
| 1 | 132.04004161161134 | 3.885184226877824 |
| 2 | 162.15674078175431 | 3.741402335815208 |
| 3 | 261.76484148511651 | 3.629472567828700 |

差异达约 27–72 倍，足以根本改变 \(\lambda\)。因此 `mean(1/P)` 必须由真实 \(P\) 计算并
持久化，绝不能 silent fallback 到 `1/mean(P)`。另有 provenance 异常：JSON 的 `CAVEAT`
文字称 6 张 non-junco 图，但 `calibration_images` 与 `n_images` 都是 14；这虽不改变上述
代数，也应在重新生成统计时修正。

### 3.3 为什么题面尚不足以唯一给出“实际”4×10 表

题面给出“每 stage 10 个正 \(\tau\)”和“最小正 \(\sigma\approx4\times10^{-4}\)”，但没有
列出十个 \(\tau_j\)。这不能唯一恢复网格。例如

\[
(0.1,0.2,\ldots,0.9,0.9984)
\]

与

\[
(0.01,0.02,0.04,0.08,0.16,0.32,0.64,0.9,0.99,0.9984)
\]

都满足十个正点及同一末端 \(\sigma_3=0.0004\)，但 \(q_0\) 和全部中间 \(\lambda\) 不同。
此外，指定 JSON 本身没有 spectral reciprocal-mean。故仅凭题面指定两份 JSON，实际双表
不可识别。

为给出量级审查而不伪装成实际表，下面采用唯一显式的**条件性重建假设**：

\[
\boxed{\tau_j=0.9984\,j/10,\quad j=1,\ldots,10.}
\]

该假设使 stage 3 最后一点恰有 \(\sigma=0.0004\)。实际启用前必须用持久化的 exact
`tau_grid` 重生成；若 exact grid 不同，下面数值不能作为生产 golden table。

### 3.4 Pooled：条件性 4×10 \(q,\lambda\) 表

每个 stage 的 reference \(q_{k,0}\) 取表中第一行，所以第一行 \(\lambda=0\)。

| k | frame | \(\tau\) | \(\sigma\) | \(q\) | \(\lambda\) | \(\rho=\sqrt\lambda\) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.09984 | 0.9750400 | 25.189861812 | 0.000000000 | 0.000000000 |
| 0 | 2 | 0.19968 | 0.9500800 | 25.191967267 | 0.000083576 | 0.009142015 |
| 0 | 3 | 0.29952 | 0.9251200 | 25.195757927 | 0.000234012 | 0.015297458 |
| 0 | 4 | 0.39936 | 0.9001600 | 25.201508336 | 0.000462136 | 0.021497348 |
| 0 | 5 | 0.49920 | 0.8752000 | 25.209540119 | 0.000780590 | 0.027939035 |
| 0 | 6 | 0.59904 | 0.8502400 | 25.220231278 | 0.001204171 | 0.034701164 |
| 0 | 7 | 0.69888 | 0.8252800 | 25.234027615 | 0.001750248 | 0.041835964 |
| 0 | 8 | 0.79872 | 0.8003200 | 25.251456856 | 0.002439267 | 0.049388935 |
| 0 | 9 | 0.89856 | 0.7753600 | 25.273146203 | 0.003295371 | 0.057405321 |
| 0 | 10 | 0.99840 | 0.7504000 | 25.299844301 | 0.004347161 | 0.065933002 |
| 1 | 1 | 0.09984 | 0.7250400 | 21.234664858 | 0.000000000 | 0.000000000 |
| 1 | 2 | 0.19968 | 0.7000800 | 21.256291908 | 0.001017442 | 0.031897371 |
| 1 | 3 | 0.29952 | 0.6751200 | 21.289953038 | 0.002596914 | 0.050959927 |
| 1 | 4 | 0.39936 | 0.6501600 | 21.338282179 | 0.004855935 | 0.069684542 |
| 1 | 5 | 0.49920 | 0.6252000 | 21.404541124 | 0.007936459 | 0.089086807 |
| 1 | 6 | 0.59904 | 0.6002400 | 21.492794958 | 0.012010076 | 0.109590492 |
| 1 | 7 | 0.69888 | 0.5752800 | 21.608145337 | 0.017284245 | 0.131469559 |
| 1 | 8 | 0.79872 | 0.5503200 | 21.757044044 | 0.024009658 | 0.154950502 |
| 1 | 9 | 0.89856 | 0.5253600 | 21.947719371 | 0.032488775 | 0.180246428 |
| 1 | 10 | 0.99840 | 0.5004000 | 22.190763289 | 0.043085423 | 0.207570285 |
| 2 | 1 | 0.09984 | 0.4750400 | 19.514951686 | 0.000000000 | 0.000000000 |
| 2 | 2 | 0.19968 | 0.4500800 | 19.647265643 | 0.006734472 | 0.082063828 |
| 2 | 3 | 0.29952 | 0.4251200 | 19.857593329 | 0.017254943 | 0.131358072 |
| 2 | 4 | 0.39936 | 0.4001600 | 20.172953255 | 0.032618009 | 0.180604565 |
| 2 | 5 | 0.49920 | 0.3752000 | 20.631078356 | 0.054099289 | 0.232592539 |
| 2 | 6 | 0.59904 | 0.3502400 | 21.285579338 | 0.083184377 | 0.288417019 |
| 2 | 7 | 0.69888 | 0.3252800 | 22.214152257 | 0.121508151 | 0.348580193 |
| 2 | 8 | 0.79872 | 0.3003200 | 23.532021718 | 0.170706541 | 0.413166481 |
| 2 | 9 | 0.89856 | 0.2753600 | 25.414732568 | 0.232140191 | 0.481809289 |
| 2 | 10 | 0.99840 | 0.2504000 | 28.138376297 | 0.306464898 | 0.553592718 |
| 3 | 1 | 0.09984 | 0.2250400 | 21.152020387 | 0.000000000 | 0.000000000 |
| 3 | 2 | 0.19968 | 0.2000800 | 22.782716210 | 0.071576006 | 0.267536924 |
| 3 | 3 | 0.29952 | 0.1751200 | 25.780630108 | 0.179538270 | 0.423719565 |
| 3 | 4 | 0.39936 | 0.1501600 | 31.352330169 | 0.325344551 | 0.570389824 |
| 3 | 5 | 0.49920 | 0.1252000 | 42.168466980 | 0.498392474 | 0.705969174 |
| 3 | 6 | 0.59904 | 0.1002400 | 64.967058397 | 0.674419299 | 0.821230357 |
| 3 | 7 | 0.69888 | 0.0752800 | 120.403161876 | 0.824323381 | 0.907922563 |
| 3 | 8 | 0.79872 | 0.0503200 | 296.045154375 | 0.928551371 | 0.963613704 |
| 3 | 9 | 0.89856 | 0.0253600 | 1328.877116991 | 0.984082787 | 0.992009469 |
| 3 | 10 | 0.99840 | 0.0004000 | \(6.233780290\times10^6\) | 0.999996607 | 0.999998303 |

### 3.5 Spectral：条件性 4×10 \(q,\lambda\) 表

本表使用上文从原始 `spectral_power.npz` 补算的真实
\(\operatorname{mean}(1/P)\)，不是 `1/mean_power`。

| k | frame | \(\tau\) | \(\sigma\) | \(q\) | \(\lambda\) | \(\rho=\sqrt\lambda\) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.09984 | 0.9750400 | 110.658604346 | 0.000000000 | 0.000000000 |
| 0 | 2 | 0.19968 | 0.9500800 | 110.660709801 | 0.000019026 | 0.004361905 |
| 0 | 3 | 0.29952 | 0.9251200 | 110.664500461 | 0.000053279 | 0.007299260 |
| 0 | 4 | 0.39936 | 0.9001600 | 110.670250870 | 0.000105236 | 0.010258473 |
| 0 | 5 | 0.49920 | 0.8752000 | 110.678282653 | 0.000177797 | 0.013334068 |
| 0 | 6 | 0.59904 | 0.8502400 | 110.688973812 | 0.000274368 | 0.016564045 |
| 0 | 7 | 0.69888 | 0.8252800 | 110.702770149 | 0.000398958 | 0.019973944 |
| 0 | 8 | 0.79872 | 0.8003200 | 110.720199390 | 0.000556313 | 0.023586280 |
| 0 | 9 | 0.89856 | 0.7753600 | 110.741888737 | 0.000752059 | 0.027423687 |
| 0 | 10 | 0.99840 | 0.7504000 | 110.768586835 | 0.000992903 | 0.031510367 |
| 1 | 1 | 0.09984 | 0.7250400 | 132.079551649 | 0.000000000 | 0.000000000 |
| 1 | 2 | 0.19968 | 0.7000800 | 132.101178699 | 0.000163716 | 0.012795148 |
| 1 | 3 | 0.29952 | 0.6751200 | 132.134839830 | 0.000418422 | 0.020455376 |
| 1 | 4 | 0.39936 | 0.6501600 | 132.183168970 | 0.000783892 | 0.027998070 |
| 1 | 5 | 0.49920 | 0.6252000 | 132.249427916 | 0.001284514 | 0.035840120 |
| 1 | 6 | 0.59904 | 0.6002400 | 132.337681750 | 0.001950541 | 0.044164932 |
| 1 | 7 | 0.69888 | 0.5752800 | 132.453032128 | 0.002819720 | 0.053101037 |
| 1 | 8 | 0.79872 | 0.5503200 | 132.601930836 | 0.003939454 | 0.062765069 |
| 1 | 9 | 0.89856 | 0.5253600 | 132.792606162 | 0.005369685 | 0.073278137 |
| 1 | 10 | 0.99840 | 0.5004000 | 133.035650081 | 0.007186784 | 0.084774900 |
| 2 | 1 | 0.09984 | 0.4750400 | 162.480679623 | 0.000000000 | 0.000000000 |
| 2 | 2 | 0.19968 | 0.4500800 | 162.612993580 | 0.000813674 | 0.028524971 |
| 2 | 3 | 0.29952 | 0.4251200 | 162.823321266 | 0.002104377 | 0.045873488 |
| 2 | 4 | 0.39936 | 0.4001600 | 163.138681192 | 0.004033388 | 0.063508959 |
| 2 | 5 | 0.49920 | 0.3752000 | 163.596806293 | 0.006822423 | 0.082597962 |
| 2 | 6 | 0.59904 | 0.3502400 | 164.251307275 | 0.010779991 | 0.103826737 |
| 2 | 7 | 0.69888 | 0.3252800 | 165.179880194 | 0.016340977 | 0.127831830 |
| 2 | 8 | 0.79872 | 0.3003200 | 166.497749655 | 0.024126873 | 0.155328274 |
| 2 | 9 | 0.89856 | 0.2753600 | 168.380460505 | 0.035038394 | 0.187185453 |
| 2 | 10 | 0.99840 | 0.2504000 | 171.104104233 | 0.050398701 | 0.224496550 |
| 3 | 1 | 0.09984 | 0.2250400 | 264.877152458 | 0.000000000 | 0.000000000 |
| 3 | 2 | 0.19968 | 0.2000800 | 266.507848281 | 0.006118753 | 0.078222461 |
| 3 | 3 | 0.29952 | 0.1751200 | 269.505762179 | 0.017174437 | 0.131051276 |
| 3 | 4 | 0.39936 | 0.1501600 | 275.077462241 | 0.037081590 | 0.192565808 |
| 3 | 5 | 0.49920 | 0.1252000 | 285.893599051 | 0.073511428 | 0.271129909 |
| 3 | 6 | 0.59904 | 0.1002400 | 308.692190468 | 0.141937630 | 0.376746109 |
| 3 | 7 | 0.69888 | 0.0752800 | 364.128293948 | 0.272571902 | 0.522084190 |
| 3 | 8 | 0.79872 | 0.0503200 | 539.770286447 | 0.509278004 | 0.713637165 |
| 3 | 9 | 0.89856 | 0.0253600 | 1572.602249062 | 0.831567612 | 0.911903291 |
| 3 | 10 | 0.99840 | 0.0004000 | \(6.234024015\times10^6\) | 0.999957511 | 0.999978755 |

### 3.6 数值判定

- **stage 内单调性：通过。** 两张条件表均严格递增；这也是闭式定理，不依赖网格等距。
- **stage 边界：有重置跳变，但不是错误。** pooled 的 \(c_k\) 逐 stage 下降，按理论端点
  比较，\(q_{k+1}(0)-q_k(1)\) 约为 \(-4.0774,-2.7541,-7.9013\)；spectral 的
  reciprocal-mean 反而上升，边界会向上跳。因为每个 stage 重新定义 \(q_{k,0}\) 并令
  \(\lambda=0\)，不同维度/不同 \(S\) 的跨-stage \(q\) 本就不要求全局单调。
- **stage 0：不退化。** \(S^{-1}\) 给出严格正下界；若错误漏掉它，\(q_0\to0\) 会令后续
  \(\lambda\) 错变成 1。
- **stage 3：极端但符合公式。** 条件表最后一点 pooled 有
  \(\rho\approx0.999998303\)，纯 OU
  \(\mathrm{IACT}=(1+\rho)/(1-\rho)\approx1.18\times10^6\)，两次更新累计 innovation
  标准差 \(\sqrt{1-\lambda^2}\approx0.002605\)。spectral 仍有 IACT 约
  \(9.41\times10^4\)。这几乎是 identity kernel，必须显式报告 ESS，不能称为温和 refresh。
- **启用判定：当前停用、保留诊断。** 原因不是发现了非单调，而是生产所需 exact tau 与
  `mean_inverse_power` 没有在指定审计文件中持久化；同时末端 mixing 风险极强。若补齐的
  exact 表复算后与闭式一致，可将 precision 模式恢复为次要诊断 arm，但不得静默改公式或
  另加未授权 cap。

最低应逐帧保存：`tau, sigma, h2_trace_over_n, mean_inverse_power, q, q_ref,
q_ref_over_q, fresh_variance=1-lambda, lambda, sqrt_lambda, sqrt_fresh`，并保存原始谱 hash、
floor 顺序和计算 dtype。

---

## 4. 为什么不能直接令物理 \(\sigma_\tau\to w\sigma_\tau\)

原 Block 2 conditional 是

\[
N(m,\sigma_\tau^2I).
\]

若直接抽 \(x_\tau=m+w\sigma_\tau z\)，保持的是
\(N(m,w^2\sigma_\tau^2I)\)。除非 \(w=1\)，这已改变 conditional covariance；而 Block 1
仍按原物理 \(\sigma_\tau\) 构造，两子核不再对应同一个联合目标，Gibbs 不变性论证失效。

更一般地，若写

\[
x'=m+\rho(x-m)+w\sigma_\tau\xi,
\]

其平稳协方差 \(V\) 满足

\[
V=\rho^2V+w^2\sigma_\tau^2I
\quad\Longrightarrow\quad
V=\frac{w^2}{1-\rho^2}\sigma_\tau^2I.
\]

要保持原 \(V=\sigma_\tau^2I\)，必须令 \(w=\sqrt{1-\rho^2}=\sqrt{1-\lambda}\)。所以合法
自由度是“相关保留与 innovation 方差的配对分解”，不是任意缩小物理条件标准差。

直接缩放还会使 l.16 的 \(x_0\) 方差变为 \(w^2I\)，下一帧 warm reconstruction 不再携带
模型定义的标准噪声坐标。

---

## 5. 为什么 \(\lambda\) 不应依赖当前 state

“每个固定 \(\lambda\) 的核都保持目标”不能推出“按当前被更新状态选择 \(\lambda\) 后仍保持
目标”。一维标准正态给出反例。令

\[
\lambda(z)=\mathbf 1_{\{|z|>1\}},\qquad
z'=\sqrt{\lambda(z)}z+\sqrt{1-\lambda(z)}\xi.
\]

若 \(Z\sim N(0,1)\)，则

\[
\mathbb E[(Z')^2\mid Z=z]=1+\lambda(z)(z^2-1),
\]

从而

\[
\mathbb E[(Z')^2]
=1+\mathbb E[(Z^2-1)\mathbf1_{\{|Z|>1\}}]
=1+2\phi(1)>1.
\]

一步后方差已不再为 1。正反转移分别使用 \(\lambda(z)\) 与 \(\lambda(z')\)，一般也破坏
详细平衡。因此依赖当前 \(x_\tau\)、residual、MSE、由当前输入得到的 \(v_\theta\) 等，若无
MH 接受率或专门的可逆状态依赖构造，就没有不变性保证。依赖 GT 的指标还会引入不可部署的
信息泄漏。

依赖 \((k,\tau,H,\sigma,S)\) 则不同：这些量在固定 frame 的 Block 2 内是外生常数，核仍是
常系数 OU，前述详细平衡逐 frame 成立。

严格数学上有一个例外，值得避免口径过强：若 \(\lambda=f(x_1^{\rm new})\) 且在给定
\(x_1^{\rm new}\) 的 Block 2 内固定，则对每个 conditioning value 仍可保持 conditional。
所以“任何 state dependence 必错”不是定理；题面的全禁令是更强、易复核的设计规则。
当前 \(x_\tau\)/residual/\(v_\theta\) 依赖仍属于一般不合法情形。

---

## 6. l.16：直接 carry `z_new` 与 residual 是否同一坐标

精确算术且 \(\sigma_\tau\ne0\) 时，

\[
x_\tau^{\rm new}=H_\tau x_1^{\rm new}+\sigma_\tau z_{\rm new}
\quad\Longrightarrow\quad
\frac{x_\tau^{\rm new}-H_\tau x_1^{\rm new}}{\sigma_\tau}=z_{\rm new}.
\]

二者是同一个标准化 residual 坐标，前提是使用最后一次 Block 2 的同一个
\((H, x_1^{\rm new},\sigma)\)，中间没有再修改状态。

fp32 下二者不逐位相等。若 \(m=Hx_1=O(1)\)，构造
\(\widehat x=\operatorname{fl}(m+\sigma z)\) 后再相减、除以小 \(\sigma\)，误差量级为

\[
|\widehat z_{\rm resid}-z|
=O\!\left(u\frac{|m|}{\sigma}+u|z|\right),
\]

其中 \(u\) 是机器精度。直接 carry 避开大数相消，\(\sigma\) 越小越有利；在
\(\sigma\approx4\times10^{-4}\) 时，标准化误差的粗量级已可达几 \(10^{-4}\)。当
\(\sigma\) 小于 \(m\) 附近半个 ULP 时，\(m+\sigma z\) 甚至直接舍入为 \(m\)，residual
路径会丢掉整个 \(z\)。

但这产生一个硬兼容冲突。独立模式下 \(z_{\rm new}=\xi\)，旧 baseline 跨帧 carry 的实际值是

\[
x_0^{\rm old}=\operatorname{fl}\!\left(
\frac{\operatorname{fl}(m+\sigma\xi)-m}{\sigma}
\right),
\]

通常不与 \(\xi\) 逐位相同。下一帧用 \(H',\sigma'\) warm rebuild 后，轨迹即分叉。因此以下
三项不能无条件同时满足：

1. 完全沿用现行 fp32 baseline；
2. `independent` 跨帧逐位一致；
3. l.16 对 `independent` 也直接 carry 原始 \(z_{\rm new}=\xi\)。

若“逐位”是硬标准，`independent` 必须显式走 legacy 路径：原 RNG 次序、原
`x_tau = m + sigma*xi`、原 residual l.16；pCN 模式才 direct-carry。不能只把通用公式中的
\(\lambda\) 设成 0，因为仍可能多算 `z_old`、改变浮点操作或在异常输入上产生 `0*Inf=NaN`。

另外，**每次 inner 的 `z_old` 仍必须用新的 Block-1 输出重新定中心**：

\[
z_{\rm old}=\frac{x_{\tau,\rm old}-H x_{1,\rm new}}\sigma.
\]

不能复用上次 carry 的 \(x_0\)，因为它通常是相对旧 \(x_1\) 的 residual。若错用旧中心
\(m_{\rm old}=m_{\rm new}+\delta\)，平稳输入更新后的均值会偏成
\(m_{\rm new}-\sqrt\lambda\,\delta\)。

residual 恒等式应作为只读诊断；容差需随 \(u|m|/\sigma\) 缩放，并同时报告 max/RMS 误差，
不能把 fp32 逐位相等当数学正确性的必要条件。

---

## 7. 预期效果的理论边界

### 7.1 平稳态不能改善固定目标的终态边缘

对固定 \((k,\tau)\)，合法 pCN 与独立 Block 2 保持同一联合目标。若链已平稳，更新前后边缘
完全相同；若固定最后一个 frame 无限迭代、\(\lambda<1\) 且整体链遍历，两模式应收敛到同一
目标。故不变核不能在平稳态“改善”终态边缘 MSE、视觉质量或 posterior 正确性。

### 7.2 完整 Algorithm 4 不是单一平稳链

完整算法逐 frame 改变 \((k,\tau,H,\sigma,S,\lambda)\)，每帧只做 \(S_{it}=2\)，并用 l.16/l.8
在不同 frame 间 warm reconstruct。它是非齐次有限时链；题面没有给出一个把前一 frame 联合
目标精确运输到下一目标的定理。因此最终边缘可以因 kernel choice 而不同。

若 pCN 改善单条有限轨迹，合理归因是：

- 减少每个有限 inner step 的 fresh-noise jump；
- 在相邻 frame 间保留标准化噪声坐标；
- 改变瞬时目标追踪误差及有限步 bias–variance 权衡；
- 小 \(\sigma\) 下避免 l.16 的相消损失。

不能表述为缩小了目标 covariance，也不能称 stationary target 更好。相反，pCN 也可能保留
过时 residual，使新 frame 适应更慢。

### 7.3 autocorrelation / ESS 代价

固定中心纯 OU 有

\[
\operatorname{Corr}(z_t,z_{t+h})=\rho^h,
\qquad \rho=\sqrt\lambda.
\]

对线性观测量，

\[
\tau_{\rm int}=\frac{1+\rho}{1-\rho},
\qquad
\frac{\mathrm{ESS}}N=\frac{1-\rho}{1+\rho},
\qquad
\mathbb E[(z'-z)^2]=2(1-\rho).
\]

所以 fresh noise 越小，ESS 通常越差；\(\lambda=1\) 时 ESS 为零。实际 alternating Gibbs 中
\(x_1\) 会改变中心，上式不能替代整链实测，但它是 Block 2 persistence 的精确基准。特别要
避免把 \(\lambda\) 本身误称 lag-1 correlation；相关系数是 \(\sqrt\lambda\)。

---

## 8. 最低有说服力的实现验证清单

### 8.1 A. `independent` baseline 逐位回归

最低配置：

- 同一硬件、软件栈、fp32、determinism 设置；旧实现与新 `independent` 并排；
- pooled 与 spectral 两条路径，各至少 3 个固定 seed；
- 完整 \(4\times10\) frame、每 frame 2 inner，覆盖实际最小 \(\sigma\)；
- 判据用 `torch.equal`/逐字节一致，而不是 `allclose`；
- 每次 Block 2 比较 RNG 前后状态、\(\xi_0,m,x_\tau^{\rm out}\)、帧末 \(x_0\)、下一帧
  warm-rebuild \(x_\tau\) 和最终输出；
- 诊断必须 RNG-free，不能因日志、norm probe 或模式分支多消耗一次随机数；
- `independent` 使用上述 legacy l.16 compatibility branch。若不保留，就只能声称数学等价，
  不能声称逐位 baseline。

还应检查 `lambda=0` 不进入通用 `z_old` 算法，避免额外算术和 `0*Inf`。

### 8.2 B. Dense fp64 conditional invariance / reversibility

取最小 \(n=4\)，例如

\[
G=\operatorname{diag}(1,0,0,0),
\]

恰满足 rank ratio \(1/4\)。用非零、非对称固定 \(x_1^{\rm new}\)，使 \(m\ne0\)。至少覆盖：

- \(\lambda=0\)、一个内部值、实际 stage-3 最大 \(\lambda\)；
- \(\lambda=1\) 作为“可逆但不遍历”的负面测试；
- 常规 \(\sigma\)、\(4\times10^{-4}\)，以及 \(10^{-8}\) 的数值退化诊断；
- stale-center 反例：强制 \(x_1^{\rm old}\ne x_1^{\rm new}\)，确认实现以新中心算 `z_old`；
- sigma/precision 两种 schedule 产生的系数，并验证
  `rho^2 + fresh_variance = 1`。

最有力的一步 ensemble 测试是独立抽
\(z_{\rm old},\xi\sim N(0,I)\)，构造 stationary input 后更新一次，检验

\[
E[z_{\rm new}]=0,\quad
\operatorname{Cov}(z_{\rm new})=I,\quad
\operatorname{Cov}(z_{\rm old},z_{\rm new})=\sqrt\lambda I.
\]

最低建议 \(N=2^{18}\) replicas，并按理论标准误设 5σ 阈值。线性高斯下联合一、二阶矩已完全
刻画分布。另在 fp64 随机点验证

\[
\log\pi(z)+\log K(z'\mid z)
=\log\pi(z')+\log K(z\mid z')
\]

对所有 \(\lambda<1\) 成立；\(\lambda=1\) 不套非退化转移密度公式，而单验 identity 与
非收敛性。

### 8.3 C. 固定目标 stationary joint-chain 与 mixing

建立可解析小型联合高斯模型：

\[
x_1\sim N(\mu,C),\qquad
x_\tau\mid x_1\sim N(Hx_1,\sigma^2I).
\]

其真值为

\[
E[x_1]=\mu,\quad E[x_\tau]=H\mu,
\]

\[
\operatorname{Cov}(x_1)=C,\quad
\operatorname{Cov}(x_\tau)=HCH^\top+\sigma^2I,\quad
\operatorname{Cov}(x_1,x_\tau)=CH^\top.
\]

Block 1 conditional 可解析：

\[
V=(C^{-1}+H^\top H/\sigma^2)^{-1},
\]

\[
E[x_1\mid x_\tau]
=V(C^{-1}\mu+H^\top x_\tau/\sigma^2).
\]

最低设计：

- \(n=4,\operatorname{rank}(G)=1\)；
- 一个 scalar \(C\)，以及一个与 \(G\) 不对易的 dense SPD \(C\)，避免全对角测试掩盖错误；
- 冻结 stage-0 early、stage-2 mid、stage-3 late 三组参数；
- independent、sigma-pCN、precision-pCN 三模式；每个冻结核检查 1 和 2 次 full sweep，后者
  匹配实际 \(S_{it}=2\)。

验证分两层：

1. **stationary-start invariance**：从解析联合目标独立抽至少 \(2^{18}\) 个 replicas，各做
   1/2 sweeps，比较完整 joint mean/covariance。独立 replicas 避免把高度相关的长链样本误当
   成大样本。
2. **nonstationary convergence / mixing**：至少 4 个过度分散初值，分别监测
   \(\operatorname{range}(G)\) 与 \(\ker(G)\) 上的 \(x_1,x_\tau\)，报告
   \(\hat R<1.01\)、bulk/tail ESS、ACF 及基于 ESS 的均值/协方差置信区间。

还可直接对照解析均值转移矩阵。令

\[
A=VH^\top/\sigma^2,\qquad
T=\rho I+(1-\rho)HA.
\]

则冻结链满足

\[
E[x_{\tau,t+1}\mid x_{\tau,t}]
=\text{const}+T x_{\tau,t}.
\]

实测均值衰减和 lag-1 covariance 应与 \(T\) 一致。\(\rho=1\) 时 \(T=I\)，直接证明不遍历。
对末帧近 identity 的 \(\lambda\)，短链不可能给出可信 ESS；应以解析谱半径为主，并明确需要
远大于 IACT 的链长，而不是用几千步“验证收敛”。

最后，固定-kernel stationary 测试只能证明局部子核正确，不能证明非齐次 40-frame 输出无偏。
完整算法还需相同初始分布、相同随机流的 paired finite-horizon ensemble，分别报告终态 MSE、
spread、ACF/ESS；单条轨迹变好而跨链 spread/ESS 下降，必须如实归类为 finite-step
regularization–mixing trade-off。

---

## 9. 最终审查意见

- `sigma_pcn`：理论上可作为主候选进入实现与验证；应直接计算 fresh variance，严格避开
  \(\sigma=0\)，并接受其提高 autocorrelation 的代价。
- `precision_pcn`：公式本身 well-defined，且给定 stage-constant \(S\) 时 \(q\) 严格单调；
  但当前指定审计数据缺 exact tau 与 persisted `mean(1/P)`，所以**暂不启用，只保留诊断**。
  补齐后若闭式、trace、单调性与 fp32 系数均通过，可恢复为次要对照；stage 3 末端近 identity
  必须作为预期风险而非异常静默隐藏。
- `independent`：必须是显式 legacy compatibility path，才能兑现跨 frame 逐位 baseline；
  generic \(\lambda=0\) 的数学等价不足以保证 fp32/RNG 逐位等价。
- l.16：pCN 路径直接 carry `z_new` 在小 \(\sigma\) 下数值更稳；residual 只作带尺度容差的
  correctness check。每个 inner 的 `z_old` 仍必须相对本次 `x1_new` 重新计算。

上述意见不改变物理 \(\sigma_\tau\)、不缩小目标 conditional covariance，也不宣称有限时
Algorithm 4 已处于平稳态。
