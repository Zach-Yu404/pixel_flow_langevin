# Codex(gpt-5.6-sol/ultra) 三轮交换存档 · Alg4 末两 stage 程序（2026-08-26）

只保留每轮最终答复；过程探索已删。提问与全程见会话记录。

═══════════════ codex_round1_proposal ═══════════════

结论：第一优先应做 **Block 2 白化残差上的负 Gaussian overrelaxation**。它直接命中首要噪声源 \(\xi_0\)，不改 \(H_\tau x_1\) 中心、不减 conditional 方差，只把独立刷新换成保持原 \(N(H_\tau x_1,\sigma_\tau^2I)\) 不变的相关核。

先明确精确性口径：草稿自己承认，网络代理使两个 conditional 一般不来自同一联合分布；因此这里能严格保证的是用户定义的“每个 conditional 精确抽样或 invariant”，不能升级宣称全局真 posterior invariant。[草稿的精确性边界](/CBIG-Standard-ECE/Zach/MSFlow/.research/references/2026-08-21-algorithm4-clean-endpoint-sampler.md:143)

## 排序

| 排名 | 候选 | 预期收益 | 合法性置信度 |
|---|---|---:|---:|
| 1 | Block 2 标量负过松弛 | 高 | 很高 |
| 2 | 投影/两带非中心辅助旋转 | 高，但依赖子空间对齐 | 很高 |
| 3 | 按 \(b_\tau\) 的逐帧 \(S_{\rm it}\) 调度 | 中等 | 最高，纯调度 |
| 4 | Block 1 Gaussian invariant kernel | 中低 | 很高 |
| 5 | Block 1 局部 PX-DA 辅助变量 | 低 | 高 |

## 1. Block 2 白化残差负过松弛

固定刚更新的 \(x_1^+\)，令

\[
r=\frac{x_\tau-H_\tau x_1^+}{\sigma_\tau},
\qquad
r^+=\rho r+\sqrt{1-\rho^2}\,\varepsilon,\quad
\varepsilon\sim N(0,I),
\]

\[
x_\tau^+=H_\tau x_1^++\sigma_\tau r^+,
\qquad -1<\rho<0.
\]

### 合法性

保持层级：**原 Block 2 conditional 的 invariant kernel**。

若 \(r\sim N(0,I)\)，则 \(r^+\sim N(0,I)\)。而且 \((r,r^+)\) 的联合协方差为

\[
\begin{bmatrix}I&\rho I\\ \rho I&I\end{bmatrix},
\]

交换前后状态不变，因此满足 detailed balance。这里 \(\rho=0\) 恰是现有 Eq. (23) 的独立精确刷新；\(\rho\neq0\) 不是从任意输入一步得到独立 conditional draw，但严格保持同一个 conditional 不变，符合给定合法性定义。这是经典 Gaussian overrelaxation/pCN 家族。[Neal 1995](https://arxiv.org/abs/bayes-an/9506004)，[Cotter et al. 2013](https://arxiv.org/abs/1202.0709)

它与被否决的 antithetic center 本质不同：这里完全不碰 \(\hat x_1\)、\(M\)、\(d\)，Block 2 的均值和方差仍逐字是 \(H_\tau x_1,\sigma_\tau^2I\)。

### 为什么可能有效

线性化一个完整 sweep 的残差传播为 \(c\approx1\) 时，修改后的系数近似

\[
c_\rho=\rho+(1-\rho)c
      =1-(1-\rho)(1-c).
\]

负 \(\rho\) 最多把微弱的收缩 gap \(1-c\) 放大近两倍，同时新鲜 \(\xi_0\) 的幅度由 \(1\) 变成 \(\sqrt{1-\rho^2}\)。这直接对准实测第一噪声源。[噪声归因](/CBIG-Standard-ECE/Zach/MSFlow/.research/tasks/attribute-late-noise-sources.md:16)

### 实现

替换 [utils.py 的原 Block 2 位置](/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/utils.py:1391)：

- 围绕更新后的 `Hx1` 计算残差，不能围绕 `x1_in`。
- 所有臂仍照常抽 `xi_0`，保持 RNG 对齐。
- `rho==0` 单独走原表达式，要求 bitwise baseline。
- 只在 stage 2、3 启用；stage 0、1 保持 \(\rho=0\)。
- l.16 不改，但此时 `x0 == r_plus`，不再等于最后的 `xi_0`。

### 风险

- 平稳 conditional 方差完全没减少；收益只能来自有限 \(S_{\rm it}=2\) 和移动 \(\tau\) 下的负相关。
- \(\rho\to-1\) 是周期二反射，不遍历，正式方案不能取 \(-1\)。
- 负 \(\rho\) 会把新 \(x_1\) 中的 Block-1 噪声以 \(1-\rho\) 放进下一 \(x_\tau\)，可能把瓶颈从 \(\xi_0\) 转移到 \(\xi_h\)。
- 当 \(b_\tau\approx0.001\) 时，最多两倍 contraction gap 仍可能不够，不能声称根治末端。

## 2. 投影化非中心辅助旋转

这是此前“非中心化辅助变量”想法中真正可实现、严格合法的版本。

取一个精确正交投影 \(P\)。box inpainting 可取 hole/\(\ker(A)\) 掩码；通用版本可取现有 \(G\) 或 \(I-G\)，代码已验证 \(G=G^\top=G^2\)。[G 的验证](/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/utils.py:16)

定义

\[
R=\rho_P P+\rho_C(I-P),
\]

\[
L=\sqrt{1-\rho_P^2}P+\sqrt{1-\rho_C^2}(I-P),
\]

\[
r^+=Rr+Lp,\qquad p\sim N(0,I).
\]

可令受损子空间 \(\rho_P<0\)，补空间 \(\rho_C=0\)，即观测/安全方向仍独立刷新。

### 合法性

保持层级：**原 Block 2 conditional 的非中心辅助变量 invariant kernel**。

扩充目标为 \(\phi(r)\phi(p)\)，并执行正交旋转

\[
\begin{bmatrix}r^+\\p^+\end{bmatrix}
=
\begin{bmatrix}R&L\\-L&R\end{bmatrix}
\begin{bmatrix}r\\p\end{bmatrix}.
\]

由于 \(R^2+L^2=I\)，该变换保持两份标准高斯；积分掉 \(p\) 后仍是原 Block 2 conditional。网络继续看到训练形式 \(H_\tau x_1+\sigma_\tau r\)，没有改变其平稳输入分布。

### 实现与风险

- 复用 `apply_G`，或对 box 使用现成 hole mask；无 CG、无额外 NFE。
- 先验证标量候选，再升级投影版本，避免同时引入两个解释变量。
- \(\ker(A)\)、\(\ker(G)\) 和 spectral \(S\) 的频带不完全重合；选错投影可能无收益或产生边界接缝。
- \(\rho_P,\rho_C\) 必须只依赖 stage、\(\tau\)、测量表和固定算子，不能根据当前图像误差自适应。

## 3. 按 \(b_\tau\) 的逐帧内循环调度

利用 \(G\) 的两个特征子空间，令 \(g\in\{0,1\}\)：

\[
h_g=(1-\tau)s_kg+\tau e_k,\qquad
n_g=e_k(1-s_k)-s_k(1-e_k)g,
\]

\[
b_g(k,\tau)=
\frac{n_g\sigma_\tau}
     {n_g^2+\gamma^2(k,\tau)h_g^2}.
\]

首个无新连续旋钮的规则：

\[
S_{\rm it}(k,\tau)=
\begin{cases}
2,& b_{\rm ker}(k,\tau)\ge1,\\
1,& b_{\rm ker}(k,\tau)<1.
\end{cases}
\]

stage 3 虽从 \(0.94\) 开始，但仍执行一次 sweep，而不是停止或改读出；stage 2 则在 \(\tau\approx0.44\) 后由 2 降到 1。

### 合法性

保持层级：**纯调度**。每次保留下来的 Block 1/2 操作原封不动，仍是原 conditional 的精确抽样。

### 实现

当前 `num_langevin` 只支持逐 stage `_per_stage`；需扩成 `(stage,tau)->int` 或明确表。自定义网格后必须删除 γ² 的 positional fallback，缺少精确 \((k,\tau)\) key 应直接报错，不能静默错配。

### 风险

这是减小棘轮次数，不改变单次噪声平衡，收益上限有限。历史上全局 \(S_{\rm it}=1\) 比 2 差，而逐 stage 的 `10,2,2,2` 也只微弱改善；新意仅在“逐 \(\tau\) 跨 \(b=1\) 切换”。[历史调度扫描](/CBIG-Standard-ECE/Zach/MSFlow/.research/tasks/tune-sit-schedule.md:13)

## 4. Block 1 的精确 Gaussian overrelaxation

固定 \(x_\tau\)，记

\[
q_1(x_1)=N(\mu,M^{-1}),\qquad M\mu=d,
\]

\[
d=A^\top y/\eta^2+C^{-1}\hat x_1(x_\tau),
\]

\[
\zeta=A^\top\xi_y/\eta+
H_\tau^\top\xi_h/\sigma_\tau+
S^{-1/2}\xi_s\sim N(0,M).
\]

用一次原 PCG 即可执行

\[
M x_1^+
=
\rho_1 Mx_1+(1-\rho_1)d+
\sqrt{1-\rho_1^2}\,\zeta .
\]

### 合法性

保持层级：**原 Block 1 Gaussian conditional invariance**。标准化后仍是

\[
x^+-\mu=\rho_1(x-\mu)+\sqrt{1-\rho_1^2}\eta,
\quad \eta\sim N(0,M^{-1}).
\]

### 实现与风险

在 [Block 1 RHS/PCG](/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/utils.py:1380) 拆出 `d` 与 `zeta`，加入 `rho1*M_den(x1_old)`；仍只需一次 PCG。

这里更合理的是小正 \(\rho_1\)：同时降低新 \(\xi_h/\xi_s\) 和对当前 noisy center 的瞬时追随。但它也按同样比例减慢已证有效的确定性收缩，本质接近合法 under-mixing；负 \(\rho_1\) 又会把当前首要 \(\xi_0\) 通路放大。因此只适合作为第二噪声源的机制探针。

## 5. 局部 Block 1 PX-DA

若希望真正测试此前的 \(z\mid x_1\) 辅助变量方向，只能限定在当前 Gaussian \(q_1\) 内：

\[
z\mid x_1\sim
N(\sqrt{\lambda}\,S^{-1/2}x_1,I).
\]

积分 \(z\) 恒为 1，所以 \(q_1(x_1)\) 边缘不变。然后精确抽

\[
x_1\mid z\sim
N\!\left(
P^{-1}[d+\sqrt{\lambda}S^{-1/2}z],P^{-1}
\right),
\quad
P=M+\lambda S^{-1}.
\]

RTO 需在原三路噪声外加
\(\sqrt{\lambda}S^{-1/2}\xi_a\)。

### 合法性

保持层级：**同一个局部 Gaussian conditional 的辅助变量精确因子分解**；两步组合是 \(q_1\)-invariant kernel。

### 实现与风险

现有 `SOperator` 已提供 \(S^{-1}\) 与 \(S^{-1/2}\)，无需矩阵开方；但要重新构造 \(P\) 的预条件器。它只产生正相关 persistence，通常比现有一步独立精确 RTO 混合更慢，且引入 \(\lambda\)。数学合法，但收益预期最低。

## 首个应实现的方案与判据

先实现候选 1 的标量版，仅 stage 2、3 启用：

\[
\rho\in\{0,-0.5,-0.9,+0.5\}.
\]

其中 \(+0.5\) 只是区分“少刷新”与“负相关”的机制对照。固定 box/junco、两种 \(S\)、\(S_{\rm it}=2\)、4 seeds，所有臂抽取完全相同数量和顺序的随机数。

先过合法性门：

- \(\rho=0\) 对 baseline bitwise。
- dense float64 验证 \(E[r^+]=0\)、\(\operatorname{Cov}(r^+)=I\)、\(\operatorname{Cov}(r,r^+)=\rho I\)。
- l.16 重构 \(x_\tau=H_\tau x_1+\sigma_\tau x_0\) 到数值精度。
- 所有 Block-1 PCG 继续收敛；`x0_rms` 不产生系统漂移。

采用门槛：

- stage 2、3 各自都不再出现净 hole 恶化，而不只是最终帧偶然变好；
- 4-seed paired final-hole 改善超过已知约 \(0.003\) 的轨迹地板，置信区间不跨 0；
- pooled 与 spectral 同方向；
- obs、measurement residual 不劣于基线区间；
- \(\ker(A)\) spread 至少保留基线的 90%；
- 进阶目标是 spectral hole \(\le0.102\)，真正超过 Alg2。

若 \(-0.9\) 仍不能让 stage 3 变成净去噪，或收益完全被 \(\xi_h\) 接管，应停止深扫 \(\rho\)：该族能把 contraction gap 放大的理论上限只有约 2 倍。

## 确定不值得试

- 改 \(\hat x_1\) center、antithetic center、确定性 x0 carry、返回 \(\hat x_1\)/均值：前两类改变 conditional，最后一类改变读出对象；已有负结果无需重做。
- 单独把 \(\xi_0,\xi_h,\xi_s\) 缩小、置零或少乘系数：直接缩小 conditional covariance；没有对应 residual carry 补偿时非法。
- 直接令跨步噪声 \(\xi_t=-\xi_{t-1}\)：若没有扩展状态和 invariant-kernel 证明，会使新噪声条件于当前状态而非标准高斯。
- 截断 CG 来“降噪”：RTO 精确性要求精确解，截断不是合法正则。
- ULA/Langevin、加回 \(h_0/h_1\)：草稿已证明不存在同时匹配 Gaussian drift 和 variance 的步长。[草稿 §6.3](/CBIG-Standard-ECE/Zach/MSFlow/.research/references/2026-08-21-algorithm4-clean-endpoint-sampler.md:127)
- Schur 补抽 \(x_\tau\mid y\)：当前不存在可供边缘化的共同联合。必要相容条件是
  \[
  C^{-1}D\hat x_1(x_\tau)=H_\tau^\top/\sigma_\tau^2,
  \]
  网络代理一般不满足；此前“另一精确因子分解”提议应撤回。
- 全局 \(z\mid x_1\sim N(Rx_1,\Omega)\)：虽然积分 \(z\) 为 1，但精确 \(x_1\mid z,y\) 仍含不可求的 flow prior；把它改写成高斯就是换目标。
- 对 antithetic center 做全维 independence-MH 校正：技术上可恢复合法性，但图像维度下中心位移的 Mahalanobis 范数会使接受率接近零，同时付出双倍 NFE。
- 继续均匀增加 \(S_{\rm it}\)、任意缩小测得的 \(S\)：前者已有 V 形负结果；后者会用欠散布换 MSE，属于草稿明确警告的 diversity collapse。

附带说明：读取期间，共享 [utils.py](/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/utils.py:1144) 出现了一段尚未验证的 `block2_alpha` WIP，公式正是候选 1，但不是本轮提案实施结果。其注释中的“sampler stationary law unchanged”表述过强，只能写“Block 2 conditional invariant”；同时 l.16 的“等于最后 `xi_0`”在 \(\rho\neq0\) 时也不再成立。
tokens used
148,796
结论：第一优先应做 **Block 2 白化残差上的负 Gaussian overrelaxation**。它直接命中首要噪声源 \(\xi_0\)，不改 \(H_\tau x_1\) 中心、不减 conditional 方差，只把独立刷新换成保持原 \(N(H_\tau x_1,\sigma_\tau^2I)\) 不变的相关核。

先明确精确性口径：草稿自己承认，网络代理使两个 conditional 一般不来自同一联合分布；因此这里能严格保证的是用户定义的“每个 conditional 精确抽样或 invariant”，不能升级宣称全局真 posterior invariant。[草稿的精确性边界](/CBIG-Standard-ECE/Zach/MSFlow/.research/references/2026-08-21-algorithm4-clean-endpoint-sampler.md:143)

## 排序

| 排名 | 候选 | 预期收益 | 合法性置信度 |
|---|---|---:|---:|
| 1 | Block 2 标量负过松弛 | 高 | 很高 |
| 2 | 投影/两带非中心辅助旋转 | 高，但依赖子空间对齐 | 很高 |
| 3 | 按 \(b_\tau\) 的逐帧 \(S_{\rm it}\) 调度 | 中等 | 最高，纯调度 |
| 4 | Block 1 Gaussian invariant kernel | 中低 | 很高 |
| 5 | Block 1 局部 PX-DA 辅助变量 | 低 | 高 |

## 1. Block 2 白化残差负过松弛

固定刚更新的 \(x_1^+\)，令

\[
r=\frac{x_\tau-H_\tau x_1^+}{\sigma_\tau},
\qquad
r^+=\rho r+\sqrt{1-\rho^2}\,\varepsilon,\quad
\varepsilon\sim N(0,I),
\]

\[
x_\tau^+=H_\tau x_1^++\sigma_\tau r^+,
\qquad -1<\rho<0.
\]

### 合法性

保持层级：**原 Block 2 conditional 的 invariant kernel**。

若 \(r\sim N(0,I)\)，则 \(r^+\sim N(0,I)\)。而且 \((r,r^+)\) 的联合协方差为

\[
\begin{bmatrix}I&\rho I\\ \rho I&I\end{bmatrix},
\]

交换前后状态不变，因此满足 detailed balance。这里 \(\rho=0\) 恰是现有 Eq. (23) 的独立精确刷新；\(\rho\neq0\) 不是从任意输入一步得到独立 conditional draw，但严格保持同一个 conditional 不变，符合给定合法性定义。这是经典 Gaussian overrelaxation/pCN 家族。[Neal 1995](https://arxiv.org/abs/bayes-an/9506004)，[Cotter et al. 2013](https://arxiv.org/abs/1202.0709)

它与被否决的 antithetic center 本质不同：这里完全不碰 \(\hat x_1\)、\(M\)、\(d\)，Block 2 的均值和方差仍逐字是 \(H_\tau x_1,\sigma_\tau^2I\)。

### 为什么可能有效

线性化一个完整 sweep 的残差传播为 \(c\approx1\) 时，修改后的系数近似

\[
c_\rho=\rho+(1-\rho)c
      =1-(1-\rho)(1-c).
\]

负 \(\rho\) 最多把微弱的收缩 gap \(1-c\) 放大近两倍，同时新鲜 \(\xi_0\) 的幅度由 \(1\) 变成 \(\sqrt{1-\rho^2}\)。这直接对准实测第一噪声源。[噪声归因](/CBIG-Standard-ECE/Zach/MSFlow/.research/tasks/attribute-late-noise-sources.md:16)

### 实现

替换 [utils.py 的原 Block 2 位置](/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/utils.py:1391)：

- 围绕更新后的 `Hx1` 计算残差，不能围绕 `x1_in`。
- 所有臂仍照常抽 `xi_0`，保持 RNG 对齐。
- `rho==0` 单独走原表达式，要求 bitwise baseline。
- 只在 stage 2、3 启用；stage 0、1 保持 \(\rho=0\)。
- l.16 不改，但此时 `x0 == r_plus`，不再等于最后的 `xi_0`。

### 风险

- 平稳 conditional 方差完全没减少；收益只能来自有限 \(S_{\rm it}=2\) 和移动 \(\tau\) 下的负相关。
- \(\rho\to-1\) 是周期二反射，不遍历，正式方案不能取 \(-1\)。
- 负 \(\rho\) 会把新 \(x_1\) 中的 Block-1 噪声以 \(1-\rho\) 放进下一 \(x_\tau\)，可能把瓶颈从 \(\xi_0\) 转移到 \(\xi_h\)。
- 当 \(b_\tau\approx0.001\) 时，最多两倍 contraction gap 仍可能不够，不能声称根治末端。

## 2. 投影化非中心辅助旋转

这是此前“非中心化辅助变量”想法中真正可实现、严格合法的版本。

取一个精确正交投影 \(P\)。box inpainting 可取 hole/\(\ker(A)\) 掩码；通用版本可取现有 \(G\) 或 \(I-G\)，代码已验证 \(G=G^\top=G^2\)。[G 的验证](/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/utils.py:16)

定义

\[
R=\rho_P P+\rho_C(I-P),
\]

\[
L=\sqrt{1-\rho_P^2}P+\sqrt{1-\rho_C^2}(I-P),
\]

\[
r^+=Rr+Lp,\qquad p\sim N(0,I).
\]

可令受损子空间 \(\rho_P<0\)，补空间 \(\rho_C=0\)，即观测/安全方向仍独立刷新。

### 合法性

保持层级：**原 Block 2 conditional 的非中心辅助变量 invariant kernel**。

扩充目标为 \(\phi(r)\phi(p)\)，并执行正交旋转

\[
\begin{bmatrix}r^+\\p^+\end{bmatrix}
=
\begin{bmatrix}R&L\\-L&R\end{bmatrix}
\begin{bmatrix}r\\p\end{bmatrix}.
\]

由于 \(R^2+L^2=I\)，该变换保持两份标准高斯；积分掉 \(p\) 后仍是原 Block 2 conditional。网络继续看到训练形式 \(H_\tau x_1+\sigma_\tau r\)，没有改变其平稳输入分布。

### 实现与风险

- 复用 `apply_G`，或对 box 使用现成 hole mask；无 CG、无额外 NFE。
- 先验证标量候选，再升级投影版本，避免同时引入两个解释变量。
- \(\ker(A)\)、\(\ker(G)\) 和 spectral \(S\) 的频带不完全重合；选错投影可能无收益或产生边界接缝。
- \(\rho_P,\rho_C\) 必须只依赖 stage、\(\tau\)、测量表和固定算子，不能根据当前图像误差自适应。

## 3. 按 \(b_\tau\) 的逐帧内循环调度

利用 \(G\) 的两个特征子空间，令 \(g\in\{0,1\}\)：

\[
h_g=(1-\tau)s_kg+\tau e_k,\qquad
n_g=e_k(1-s_k)-s_k(1-e_k)g,
\]

\[
b_g(k,\tau)=
\frac{n_g\sigma_\tau}
     {n_g^2+\gamma^2(k,\tau)h_g^2}.
\]

首个无新连续旋钮的规则：

\[
S_{\rm it}(k,\tau)=
\begin{cases}
2,& b_{\rm ker}(k,\tau)\ge1,\\
1,& b_{\rm ker}(k,\tau)<1.
\end{cases}
\]

stage 3 虽从 \(0.94\) 开始，但仍执行一次 sweep，而不是停止或改读出；stage 2 则在 \(\tau\approx0.44\) 后由 2 降到 1。

### 合法性

保持层级：**纯调度**。每次保留下来的 Block 1/2 操作原封不动，仍是原 conditional 的精确抽样。

### 实现

当前 `num_langevin` 只支持逐 stage `_per_stage`；需扩成 `(stage,tau)->int` 或明确表。自定义网格后必须删除 γ² 的 positional fallback，缺少精确 \((k,\tau)\) key 应直接报错，不能静默错配。

### 风险

这是减小棘轮次数，不改变单次噪声平衡，收益上限有限。历史上全局 \(S_{\rm it}=1\) 比 2 差，而逐 stage 的 `10,2,2,2` 也只微弱改善；新意仅在“逐 \(\tau\) 跨 \(b=1\) 切换”。[历史调度扫描](/CBIG-Standard-ECE/Zach/MSFlow/.research/tasks/tune-sit-schedule.md:13)

## 4. Block 1 的精确 Gaussian overrelaxation

固定 \(x_\tau\)，记

\[
q_1(x_1)=N(\mu,M^{-1}),\qquad M\mu=d,
\]

\[
d=A^\top y/\eta^2+C^{-1}\hat x_1(x_\tau),
\]

\[
\zeta=A^\top\xi_y/\eta+
H_\tau^\top\xi_h/\sigma_\tau+
S^{-1/2}\xi_s\sim N(0,M).
\]

用一次原 PCG 即可执行

\[
M x_1^+
=
\rho_1 Mx_1+(1-\rho_1)d+
\sqrt{1-\rho_1^2}\,\zeta .
\]

### 合法性

保持层级：**原 Block 1 Gaussian conditional invariance**。标准化后仍是

\[
x^+-\mu=\rho_1(x-\mu)+\sqrt{1-\rho_1^2}\eta,
\quad \eta\sim N(0,M^{-1}).
\]

### 实现与风险

在 [Block 1 RHS/PCG](/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/utils.py:1380) 拆出 `d` 与 `zeta`，加入 `rho1*M_den(x1_old)`；仍只需一次 PCG。

这里更合理的是小正 \(\rho_1\)：同时降低新 \(\xi_h/\xi_s\) 和对当前 noisy center 的瞬时追随。但它也按同样比例减慢已证有效的确定性收缩，本质接近合法 under-mixing；负 \(\rho_1\) 又会把当前首要 \(\xi_0\) 通路放大。因此只适合作为第二噪声源的机制探针。

## 5. 局部 Block 1 PX-DA

若希望真正测试此前的 \(z\mid x_1\) 辅助变量方向，只能限定在当前 Gaussian \(q_1\) 内：

\[
z\mid x_1\sim
N(\sqrt{\lambda}\,S^{-1/2}x_1,I).
\]

积分 \(z\) 恒为 1，所以 \(q_1(x_1)\) 边缘不变。然后精确抽

\[
x_1\mid z\sim
N\!\left(
P^{-1}[d+\sqrt{\lambda}S^{-1/2}z],P^{-1}
\right),
\quad
P=M+\lambda S^{-1}.
\]

RTO 需在原三路噪声外加
\(\sqrt{\lambda}S^{-1/2}\xi_a\)。

### 合法性

保持层级：**同一个局部 Gaussian conditional 的辅助变量精确因子分解**；两步组合是 \(q_1\)-invariant kernel。

### 实现与风险

现有 `SOperator` 已提供 \(S^{-1}\) 与 \(S^{-1/2}\)，无需矩阵开方；但要重新构造 \(P\) 的预条件器。它只产生正相关 persistence，通常比现有一步独立精确 RTO 混合更慢，且引入 \(\lambda\)。数学合法，但收益预期最低。

## 首个应实现的方案与判据

先实现候选 1 的标量版，仅 stage 2、3 启用：

\[
\rho\in\{0,-0.5,-0.9,+0.5\}.
\]

其中 \(+0.5\) 只是区分“少刷新”与“负相关”的机制对照。固定 box/junco、两种 \(S\)、\(S_{\rm it}=2\)、4 seeds，所有臂抽取完全相同数量和顺序的随机数。

先过合法性门：

- \(\rho=0\) 对 baseline bitwise。
- dense float64 验证 \(E[r^+]=0\)、\(\operatorname{Cov}(r^+)=I\)、\(\operatorname{Cov}(r,r^+)=\rho I\)。
- l.16 重构 \(x_\tau=H_\tau x_1+\sigma_\tau x_0\) 到数值精度。
- 所有 Block-1 PCG 继续收敛；`x0_rms` 不产生系统漂移。

采用门槛：

- stage 2、3 各自都不再出现净 hole 恶化，而不只是最终帧偶然变好；
- 4-seed paired final-hole 改善超过已知约 \(0.003\) 的轨迹地板，置信区间不跨 0；
- pooled 与 spectral 同方向；
- obs、measurement residual 不劣于基线区间；
- \(\ker(A)\) spread 至少保留基线的 90%；
- 进阶目标是 spectral hole \(\le0.102\)，真正超过 Alg2。

若 \(-0.9\) 仍不能让 stage 3 变成净去噪，或收益完全被 \(\xi_h\) 接管，应停止深扫 \(\rho\)：该族能把 contraction gap 放大的理论上限只有约 2 倍。

## 确定不值得试

- 改 \(\hat x_1\) center、antithetic center、确定性 x0 carry、返回 \(\hat x_1\)/均值：前两类改变 conditional，最后一类改变读出对象；已有负结果无需重做。
- 单独把 \(\xi_0,\xi_h,\xi_s\) 缩小、置零或少乘系数：直接缩小 conditional covariance；没有对应 residual carry 补偿时非法。
- 直接令跨步噪声 \(\xi_t=-\xi_{t-1}\)：若没有扩展状态和 invariant-kernel 证明，会使新噪声条件于当前状态而非标准高斯。
- 截断 CG 来“降噪”：RTO 精确性要求精确解，截断不是合法正则。
- ULA/Langevin、加回 \(h_0/h_1\)：草稿已证明不存在同时匹配 Gaussian drift 和 variance 的步长。[草稿 §6.3](/CBIG-Standard-ECE/Zach/MSFlow/.research/references/2026-08-21-algorithm4-clean-endpoint-sampler.md:127)
- Schur 补抽 \(x_\tau\mid y\)：当前不存在可供边缘化的共同联合。必要相容条件是
  \[
  C^{-1}D\hat x_1(x_\tau)=H_\tau^\top/\sigma_\tau^2,
  \]
  网络代理一般不满足；此前“另一精确因子分解”提议应撤回。
- 全局 \(z\mid x_1\sim N(Rx_1,\Omega)\)：虽然积分 \(z\) 为 1，但精确 \(x_1\mid z,y\) 仍含不可求的 flow prior；把它改写成高斯就是换目标。
- 对 antithetic center 做全维 independence-MH 校正：技术上可恢复合法性，但图像维度下中心位移的 Mahalanobis 范数会使接受率接近零，同时付出双倍 NFE。
- 继续均匀增加 \(S_{\rm it}\)、任意缩小测得的 \(S\)：前者已有 V 形负结果；后者会用欠散布换 MSE，属于草稿明确警告的 diversity collapse。

附带说明：读取期间，共享 [utils.py](/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/utils.py:1144) 出现了一段尚未验证的 `block2_alpha` WIP，公式正是候选 1，但不是本轮提案实施结果。其注释中的“sampler stationary law unchanged”表述过强，只能写“Block 2 conditional invariant”；同时 l.16 的“等于最后 `xi_0`”在 \(\rho\neq0\) 时也不再成立。

═══════════════ codex_round2 ═══════════════

总判断：正 \(\rho\) 的收益不是“更强线性收缩”，而是让网络连续处理同一个局部流形图，同时压低移动 conditional center 的噪声；负 \(\rho\) 则把内插变成外推，并放大 Block‑1 噪声。旧预测漏掉的正是这两项。

## 1. 正负 \(\rho\) 为何反转

记网络输入 \(q_s=x_{\tau,s}\)，把 Block‑1 写成

\[
x_{1,s+1}=m_\tau(q_s)+u_s,\qquad
\operatorname{Cov}(u_s)=\Sigma_1=M_\tau^{-1},
\]

并令 \(F_\tau(q)=H_\tau m_\tau(q)\)。实际 Block‑2 等价于

\[
q_{s+1}
=\rho q_s+(1-\rho)F_\tau(q_s)
 +(1-\rho)H_\tau u_s
 +\sigma_\tau\sqrt{1-\rho^2}\,\varepsilon_s. \tag{1}
\]

旧式

\[
c_\rho=\rho+(1-\rho)c
\]

只保留了 (1) 的确定性部分，并假定 \(F_\tau\) 是固定仿射映射、\(F'_\tau=c\)。代数没错，错的是用这一项闭合有限时 MSE。真实局部误差还有

\[
\begin{aligned}
\mathbb E\|\delta q^+\|^2
\approx&\ \|[\rho I+(1-\rho)J_F]\delta q\|^2\\
&+(1-\rho)^2\operatorname{tr}(H\Sigma_1H^\top)
 +(1-\rho^2)\sigma_\tau^2d
 +\text{曲率/交叉项}. \tag{2}
\end{aligned}
\]

对相同 \(|\rho|\)：

- fresh \(\xi_0\) 方差 \(1-\rho^2\) 完全相同；
- Block‑1 移动中心噪声却乘 \((1-\rho)^2\)。

例如正 \(0.7/0.85\) 的倍率是 \(0.09/0.0225\)；负 \(-0.5/-0.9\) 是 \(2.25/3.61\)。仅此已给出正确符号。

再把状态写成流形慢变量与法向快变量 \(x=\phi_\tau(z)+n\)：

\[
n_{s+1}=C_\tau n_s+K_{\perp,\tau}\delta q_s+\cdots ,
\]

\[
z_{s+1}=z_s-a_\tau g_\tau(z_s)
 +K_{\parallel,\tau}\delta q_s
 +\tfrac12\mathcal H_\tau[\delta q_s,\delta q_s]+\cdots .
\]

对白化残差 AR(1)，

\[
\operatorname{Cov}(r_{s+1}-r_s)=2(1-\rho)I,
\]

所以网络所见输入粗糙度近似

\[
Q_\rho
\approx2\sigma_\tau^2(1-\rho)I
 +(1-\rho)^2H\Sigma_1H^\top. \tag{3}
\]

正 \(\rho\) 让两次输入留在同一 local chart；负 \(\rho\) 以步长 \(1-\rho>1\) 越过 \(F(q)\)，进入流形另一侧，放大曲率偏移和 basin switch。

标量快模甚至可解出

\[
n_{s+1}=cn_s+\kappa\sigma_\tau\Delta r_{s+1}
\]

对应

\[
\frac{\operatorname{Var}_\rho(n)}
     {\operatorname{Var}_0(n)}
=
\frac{1-\rho}{1-c\rho},\qquad 0<c<1,
\]

严格随 \(\rho\) 增大而下降。全关噪声的 \(0.071/0.080\) 证明慢模确定性项本来就在收缩；正 \(\rho\) 只是让它从扩散和曲率偏移下显露出来。冻结 oracle 只验证 \(r\) 的边缘分布和 lag‑1，不约束这些 transition-functional，因此不矛盾。

\(\rho\to1\) 后反弹也自然：快扰动继续下降，但 (1) 中真正朝 \(F(q)\) 前进的步长 \(1-\rho\) 趋零，旧残差和错误 basin 被冻结。

Stage 1 的大贡献来自 Eq. (19)：

\[
\delta\hat x_1=b_\tau\,\delta\varepsilon_v,\qquad
b_\tau=\frac{N\sigma_\tau}{N^2+\gamma_\tau^2h^2}.
\]

更一般地网络响应算子含 \(b_\tau J_v\)，扰动能量约按 \(b_\tau^2Q_\rho\) 放大。\(b_\tau\ge1.15\) 时：

- \(\rho=.7\) 令残差跳变 RMS 降到基线的 \(\sqrt{0.3}=0.548\)，移动中心噪声幅度降到 \(0.3\)；
- \(\rho=-.7\) 则分别变为 \(1.304\) 和 \(1.7\)。

Stage 1 写入的是上采样后保留的粗尺度语义/流形分支；stage 2/3 的 \(b<1\)、自增益近 1，已无能力纠正。因此其 \(0.02+\) 终点贡献合理，并应在 stage 1 末已经可见。

## 2. 无自由旋钮的 \(\rho\) 规则

建议定义成“网络响应时间内，旧残差和新创新各占一半”。

连续 \(q\) 次 pCN 后

\[
r_{s+q}=\rho^q r_s+\sqrt{1-\rho^{2q}}\,\tilde\varepsilon.
\]

令 retained variance = refreshed variance：

\[
\rho^{2q}=\frac12
\quad\Longrightarrow\quad
\rho(q)=2^{-1/(2q)}. \tag{4}
\]

响应次数直接由已有 \(b\) 表决定：

\[
q_{k,\tau}
=
\min\!\left\{
S_{\rm it},
\max\!\left(1,\left\lceil\frac1{b^{\rm eff}_{k,\tau}}\right\rceil\right)
\right\}. \tag{5}
\]

标量核取相关脆弱模式中的最小 \(b\)。因为 \(b\) 已包含 \(\sigma_\tau\)，无需再乘一次 \(\sigma\)。

当前 \(S_{\rm it}=2\) 只产生两档：

\[
b^{\rm eff}\ge1:\quad \rho=2^{-1/2}=0.7071,
\]

\[
b^{\rm eff}<1:\quad \rho=2^{-1/4}=0.8409.
\]

再加两个结构门：

\[
s_k=0\ \text{或}\ \sigma_{k,\tau}=0
\quad\Longrightarrow\quad \rho=0.
\]

因此自动得到

\[
(\rho_0,\rho_1,\rho_2,\rho_3)
=(0,\ 0.707,\ 0.841,\ 0.841),
\]

几乎就是实测最优 \(0:0.7:0.85:0.85\)。逐 \(\tau\) 版在 stage 2 的 \(b=1\) crossover 前取 .707、之后单向 latch 到 .841；若只支持 per-stage 常数，就取该 stage 的最大 \(q\)。即使 \(b\to0\)，规则也封顶 .841，天然排除 .95/.98 冻结区。

## 3. 投影分带

仍值得做，但定位应是“质量–spread Pareto”，不再预期显著击败标量版 MSE。

理论首选精确正交投影

\[
P=\Pi_{\ker(A_k)\cap\ker(G)}.
\]

损伤位于数据 nullspace，而 Eq. (19) 最脆弱的是 \(\ker G\)；交集最有针对性，同时允许低频 hole 语义继续混合。

建议：

\[
\begin{array}{c|cc}
k&\rho_P&\rho_C\\ \hline
0&0&0\\
1&0.707&0.707\\
2,3&0.841&0.707
\end{array}
\]

强诊断臂可用 stage 2/3 的 \((0.841,0)\)。

注意：

- 不能直接取 \(P_{\rm hole}(I-G)\)，除非验证二者交换；否则不是正交投影。
- 若精确交投影太贵，通用 Pareto fallback 取 \(I-G\)；若只关心 box 的 raw hole-MSE，可用 hole mask，但它也会持续化低频 hole 语义，保 spread 较差。

相对标量 \(0:.7:.85:.85\)，\((.841,.707)\) 预计 hole MSE 持平至略差 \(0\)–\(0.004\)，但 spread 回升约 5%–15%；\((.841,0)\) 可能差 \(0.005\)–\(0.015\)，换取更多混合。故若只追 MSE，可跳过分带。

## 4. Block‑1 under-relaxation

推荐 \(\boxed{\rho_1=0}\)。

若法向条件均值响应近似 \(\mu(x)=ax,\ 0<a<1\)，则

\[
\lambda_{\rho_1}=a+\rho_1(1-a)
\]

更接近 1，确定性收缩变慢；稳态法向方差满足

\[
\frac{V_{\rho_1}}{V_0}
=
\frac{(1+\rho_1)(1+a)}
     {1+a+\rho_1(1-a)}
>1.
\]

同时，朝均值移动按 \(1-\rho_1\) 一阶减少，而创新 SD

\[
\sqrt{1-\rho_1^2}=1-\rho_1^2/2+\cdots
\]

仅二阶减少：\(\rho_1=.3\) 是收缩少 30%，创新 SD 只少 4.6%。

即使按 spectral “全 Block‑1 关噪”收益 \(0.043\) 给最乐观上界，\(\rho_1=.3\) 只减少 9% 新创新方差，收益不超过约 \(0.004\)，且 Block‑2 正 \(\rho\) 已削弱其传播。预测：

- \(\rho_1=.3\)：\(\Delta\)hole \(0\) 到 \(+0.008\)，多半变差；
- \(\rho_1=.5\)：约 \(+0.008\) 到 \(+0.02\)。

不值得多一次 \(Mx_1\) 算子应用；最多跑一个 .3 机制臂。

## 5. Spread 与保混合变体

冻结 pCN 核的确定量是

\[
\mathrm{IACT}=\frac{1+\rho}{1-\rho},\qquad
\mathrm{ESS}/n=\frac{1-\rho}{1+\rho}.
\]

所以

\[
\rho=.7:\ 5.67,\ 0.176;\qquad
\rho=.85:\ 12.33,\ 0.081.
\]

但跨 seed 终点 spread 不必按此比例下降：若初始 \(r_0\sim N(0,I)\)，任意有限步仍有 \(\operatorname{Var}(r_t)=I\)。必然下降的是混合率，不是残差边缘方差；完整 \(x_1\) spread 必须实测。

基于当前 MSE 改善，预期：

- spectral：\(0.247\to0.20\)–\(0.23\)，下降约 7%–20%；
- pooled：\(0.265\to0.23\)–\(0.26\)，下降约 0%–15%。

尺度极限：若 spectral 的 \(0.026\) 改善全是方差，则为

\[
\sqrt{0.247^2-0.026}=0.187;
\]

若全是 bias，spread 可仍为 \(0.247\)。

诚实口径应写成：每个 persistent kernel 保持冻结 Gaussian conditional 不变，oracle 验证 mean/variance/lag‑1；但有限、非齐次的 40-frame 链降低混合，不能据此宣称终点输出分布或 posterior spread 不变。固定同一 \(y\)/mask、等 NFE，报告至少 16 seeds 的 hole/obs、\(G/(I-G)\) 分带 spread，并附冻结 IACT/ESS。

保 spread 的首选是每两 sweep 一次 persist、一次 refresh。复合核的 frame-boundary 相关系数为 0，逐 sweep 相位平均

\[
\mathrm{IACT}=1+\rho,
\]

所以 \(\rho=.85\) 时是 \(1.85\)，远小于 \(12.33\)。

建议：

- stage 0：\((0,0)\)；
- stage 1：保留 \((.707,.707)\)，因为其质量贡献最大；
- stage 2/3：首测 \((.841,0)\)，即第一次 Block‑2 persist、第二次 refresh。这样第二次网络调用直接看到平滑输入，而最终 refresh 恢复 frame-boundary 混合。
- 反序 \((0,.841)\) 保留跨-\(\tau\) 连续性，也合法，但末帧最后一次 persist 不再被网络读取，预计 raw MSE 略弱；可作为“跨-\(\tau\) coherence”机制臂。

预计该交替版保留后期 full-persist 收益的约 40%–70%，同时恢复大部分有限时 spread。
tokens used
78,564
总判断：正 \(\rho\) 的收益不是“更强线性收缩”，而是让网络连续处理同一个局部流形图，同时压低移动 conditional center 的噪声；负 \(\rho\) 则把内插变成外推，并放大 Block‑1 噪声。旧预测漏掉的正是这两项。

## 1. 正负 \(\rho\) 为何反转

记网络输入 \(q_s=x_{\tau,s}\)，把 Block‑1 写成

\[
x_{1,s+1}=m_\tau(q_s)+u_s,\qquad
\operatorname{Cov}(u_s)=\Sigma_1=M_\tau^{-1},
\]

并令 \(F_\tau(q)=H_\tau m_\tau(q)\)。实际 Block‑2 等价于

\[
q_{s+1}
=\rho q_s+(1-\rho)F_\tau(q_s)
 +(1-\rho)H_\tau u_s
 +\sigma_\tau\sqrt{1-\rho^2}\,\varepsilon_s. \tag{1}
\]

旧式

\[
c_\rho=\rho+(1-\rho)c
\]

只保留了 (1) 的确定性部分，并假定 \(F_\tau\) 是固定仿射映射、\(F'_\tau=c\)。代数没错，错的是用这一项闭合有限时 MSE。真实局部误差还有

\[
\begin{aligned}
\mathbb E\|\delta q^+\|^2
\approx&\ \|[\rho I+(1-\rho)J_F]\delta q\|^2\\
&+(1-\rho)^2\operatorname{tr}(H\Sigma_1H^\top)
 +(1-\rho^2)\sigma_\tau^2d
 +\text{曲率/交叉项}. \tag{2}
\end{aligned}
\]

对相同 \(|\rho|\)：

- fresh \(\xi_0\) 方差 \(1-\rho^2\) 完全相同；
- Block‑1 移动中心噪声却乘 \((1-\rho)^2\)。

例如正 \(0.7/0.85\) 的倍率是 \(0.09/0.0225\)；负 \(-0.5/-0.9\) 是 \(2.25/3.61\)。仅此已给出正确符号。

再把状态写成流形慢变量与法向快变量 \(x=\phi_\tau(z)+n\)：

\[
n_{s+1}=C_\tau n_s+K_{\perp,\tau}\delta q_s+\cdots ,
\]

\[
z_{s+1}=z_s-a_\tau g_\tau(z_s)
 +K_{\parallel,\tau}\delta q_s
 +\tfrac12\mathcal H_\tau[\delta q_s,\delta q_s]+\cdots .
\]

对白化残差 AR(1)，

\[
\operatorname{Cov}(r_{s+1}-r_s)=2(1-\rho)I,
\]

所以网络所见输入粗糙度近似

\[
Q_\rho
\approx2\sigma_\tau^2(1-\rho)I
 +(1-\rho)^2H\Sigma_1H^\top. \tag{3}
\]

正 \(\rho\) 让两次输入留在同一 local chart；负 \(\rho\) 以步长 \(1-\rho>1\) 越过 \(F(q)\)，进入流形另一侧，放大曲率偏移和 basin switch。

标量快模甚至可解出

\[
n_{s+1}=cn_s+\kappa\sigma_\tau\Delta r_{s+1}
\]

对应

\[
\frac{\operatorname{Var}_\rho(n)}
     {\operatorname{Var}_0(n)}
=
\frac{1-\rho}{1-c\rho},\qquad 0<c<1,
\]

严格随 \(\rho\) 增大而下降。全关噪声的 \(0.071/0.080\) 证明慢模确定性项本来就在收缩；正 \(\rho\) 只是让它从扩散和曲率偏移下显露出来。冻结 oracle 只验证 \(r\) 的边缘分布和 lag‑1，不约束这些 transition-functional，因此不矛盾。

\(\rho\to1\) 后反弹也自然：快扰动继续下降，但 (1) 中真正朝 \(F(q)\) 前进的步长 \(1-\rho\) 趋零，旧残差和错误 basin 被冻结。

Stage 1 的大贡献来自 Eq. (19)：

\[
\delta\hat x_1=b_\tau\,\delta\varepsilon_v,\qquad
b_\tau=\frac{N\sigma_\tau}{N^2+\gamma_\tau^2h^2}.
\]

更一般地网络响应算子含 \(b_\tau J_v\)，扰动能量约按 \(b_\tau^2Q_\rho\) 放大。\(b_\tau\ge1.15\) 时：

- \(\rho=.7\) 令残差跳变 RMS 降到基线的 \(\sqrt{0.3}=0.548\)，移动中心噪声幅度降到 \(0.3\)；
- \(\rho=-.7\) 则分别变为 \(1.304\) 和 \(1.7\)。

Stage 1 写入的是上采样后保留的粗尺度语义/流形分支；stage 2/3 的 \(b<1\)、自增益近 1，已无能力纠正。因此其 \(0.02+\) 终点贡献合理，并应在 stage 1 末已经可见。

## 2. 无自由旋钮的 \(\rho\) 规则

建议定义成“网络响应时间内，旧残差和新创新各占一半”。

连续 \(q\) 次 pCN 后

\[
r_{s+q}=\rho^q r_s+\sqrt{1-\rho^{2q}}\,\tilde\varepsilon.
\]

令 retained variance = refreshed variance：

\[
\rho^{2q}=\frac12
\quad\Longrightarrow\quad
\rho(q)=2^{-1/(2q)}. \tag{4}
\]

响应次数直接由已有 \(b\) 表决定：

\[
q_{k,\tau}
=
\min\!\left\{
S_{\rm it},
\max\!\left(1,\left\lceil\frac1{b^{\rm eff}_{k,\tau}}\right\rceil\right)
\right\}. \tag{5}
\]

标量核取相关脆弱模式中的最小 \(b\)。因为 \(b\) 已包含 \(\sigma_\tau\)，无需再乘一次 \(\sigma\)。

当前 \(S_{\rm it}=2\) 只产生两档：

\[
b^{\rm eff}\ge1:\quad \rho=2^{-1/2}=0.7071,
\]

\[
b^{\rm eff}<1:\quad \rho=2^{-1/4}=0.8409.
\]

再加两个结构门：

\[
s_k=0\ \text{或}\ \sigma_{k,\tau}=0
\quad\Longrightarrow\quad \rho=0.
\]

因此自动得到

\[
(\rho_0,\rho_1,\rho_2,\rho_3)
=(0,\ 0.707,\ 0.841,\ 0.841),
\]

几乎就是实测最优 \(0:0.7:0.85:0.85\)。逐 \(\tau\) 版在 stage 2 的 \(b=1\) crossover 前取 .707、之后单向 latch 到 .841；若只支持 per-stage 常数，就取该 stage 的最大 \(q\)。即使 \(b\to0\)，规则也封顶 .841，天然排除 .95/.98 冻结区。

## 3. 投影分带

仍值得做，但定位应是“质量–spread Pareto”，不再预期显著击败标量版 MSE。

理论首选精确正交投影

\[
P=\Pi_{\ker(A_k)\cap\ker(G)}.
\]

损伤位于数据 nullspace，而 Eq. (19) 最脆弱的是 \(\ker G\)；交集最有针对性，同时允许低频 hole 语义继续混合。

建议：

\[
\begin{array}{c|cc}
k&\rho_P&\rho_C\\ \hline
0&0&0\\
1&0.707&0.707\\
2,3&0.841&0.707
\end{array}
\]

强诊断臂可用 stage 2/3 的 \((0.841,0)\)。

注意：

- 不能直接取 \(P_{\rm hole}(I-G)\)，除非验证二者交换；否则不是正交投影。
- 若精确交投影太贵，通用 Pareto fallback 取 \(I-G\)；若只关心 box 的 raw hole-MSE，可用 hole mask，但它也会持续化低频 hole 语义，保 spread 较差。

相对标量 \(0:.7:.85:.85\)，\((.841,.707)\) 预计 hole MSE 持平至略差 \(0\)–\(0.004\)，但 spread 回升约 5%–15%；\((.841,0)\) 可能差 \(0.005\)–\(0.015\)，换取更多混合。故若只追 MSE，可跳过分带。

## 4. Block‑1 under-relaxation

推荐 \(\boxed{\rho_1=0}\)。

若法向条件均值响应近似 \(\mu(x)=ax,\ 0<a<1\)，则

\[
\lambda_{\rho_1}=a+\rho_1(1-a)
\]

更接近 1，确定性收缩变慢；稳态法向方差满足

\[
\frac{V_{\rho_1}}{V_0}
=
\frac{(1+\rho_1)(1+a)}
     {1+a+\rho_1(1-a)}
>1.
\]

同时，朝均值移动按 \(1-\rho_1\) 一阶减少，而创新 SD

\[
\sqrt{1-\rho_1^2}=1-\rho_1^2/2+\cdots
\]

仅二阶减少：\(\rho_1=.3\) 是收缩少 30%，创新 SD 只少 4.6%。

即使按 spectral “全 Block‑1 关噪”收益 \(0.043\) 给最乐观上界，\(\rho_1=.3\) 只减少 9% 新创新方差，收益不超过约 \(0.004\)，且 Block‑2 正 \(\rho\) 已削弱其传播。预测：

- \(\rho_1=.3\)：\(\Delta\)hole \(0\) 到 \(+0.008\)，多半变差；
- \(\rho_1=.5\)：约 \(+0.008\) 到 \(+0.02\)。

不值得多一次 \(Mx_1\) 算子应用；最多跑一个 .3 机制臂。

## 5. Spread 与保混合变体

冻结 pCN 核的确定量是

\[
\mathrm{IACT}=\frac{1+\rho}{1-\rho},\qquad
\mathrm{ESS}/n=\frac{1-\rho}{1+\rho}.
\]

所以

\[
\rho=.7:\ 5.67,\ 0.176;\qquad
\rho=.85:\ 12.33,\ 0.081.
\]

但跨 seed 终点 spread 不必按此比例下降：若初始 \(r_0\sim N(0,I)\)，任意有限步仍有 \(\operatorname{Var}(r_t)=I\)。必然下降的是混合率，不是残差边缘方差；完整 \(x_1\) spread 必须实测。

基于当前 MSE 改善，预期：

- spectral：\(0.247\to0.20\)–\(0.23\)，下降约 7%–20%；
- pooled：\(0.265\to0.23\)–\(0.26\)，下降约 0%–15%。

尺度极限：若 spectral 的 \(0.026\) 改善全是方差，则为

\[
\sqrt{0.247^2-0.026}=0.187;
\]

若全是 bias，spread 可仍为 \(0.247\)。

诚实口径应写成：每个 persistent kernel 保持冻结 Gaussian conditional 不变，oracle 验证 mean/variance/lag‑1；但有限、非齐次的 40-frame 链降低混合，不能据此宣称终点输出分布或 posterior spread 不变。固定同一 \(y\)/mask、等 NFE，报告至少 16 seeds 的 hole/obs、\(G/(I-G)\) 分带 spread，并附冻结 IACT/ESS。

保 spread 的首选是每两 sweep 一次 persist、一次 refresh。复合核的 frame-boundary 相关系数为 0，逐 sweep 相位平均

\[
\mathrm{IACT}=1+\rho,
\]

所以 \(\rho=.85\) 时是 \(1.85\)，远小于 \(12.33\)。

建议：

- stage 0：\((0,0)\)；
- stage 1：保留 \((.707,.707)\)，因为其质量贡献最大；
- stage 2/3：首测 \((.841,0)\)，即第一次 Block‑2 persist、第二次 refresh。这样第二次网络调用直接看到平滑输入，而最终 refresh 恢复 frame-boundary 混合。
- 反序 \((0,.841)\) 保留跨-\(\tau\) 连续性，也合法，但末帧最后一次 persist 不再被网络读取，预计 raw MSE 略弱；可作为“跨-\(\tau\) coherence”机制臂。

预计该交替版保留后期 full-persist 收益的约 40%–70%，同时恢复大部分有限时 spread。

═══════════════ codex_round3 ═══════════════

先更新总裁定：最后一批已经改变了问题前提。`0.1411` 不是合法类下界，尾段减少 sweep 是稳定、合法的突破口。

| 配置 | hole，4 seeds | obs | spread | NFE |
|---|---:|---:|---:|---:|
| spectral `[2,2,2,2]` | 0.1411±0.0047 | 0.0026 | 0.2468 | 80 |
| spectral `[2,2,1,1]` | **0.1317±0.0038** | 0.0027 | 0.2379 | 60 |
| pooled `[2,2,2,2]` | 0.1499±0.0007 | 0.0032 | 0.2646 | 80 |
| pooled `[2,2,2,1]` | **0.1423±0.0027** | 0.0034 | 0.2513 | 70 |

所以真正稳固的不是“均匀 \(S_{\rm it}=2\)”本身，而是把所有 stage 绑在同一调度时形成的局部折中。解开后，前两 stage 用 2 次、后两 stage 用 1 次，恰好避开尾段第二次净注入。

## 1. 鞍点、注入—移除平衡与理论下界

在 \(\ker A\) 的单个共同谱模态上，记 \(h,N,s^2\) 分别为 \(H_\tau,N_k,S\) 的特征值，\(\Delta=e_k-s_k\)。由 (12)，

\[
C=\left(\frac{h^2}{\sigma^2}+\frac1{s^2}\right)^{-1}
=\frac{\sigma^2s^2}{h^2s^2+\sigma^2}.
\]

这是每次 fresh Block-1 draw 必须注入的方差。

把 (19) 的 endpoint map 记为 \(f(x_\tau)=\hat x_1\)，在当前轨迹附近线性化：

\[
K=J_f=
\frac{N\Delta+\gamma^2h+N\sigma J_v}
     {N^2+\gamma^2h^2}.
\]

利用 \(N=\sigma B+\Delta h\)，一轮反馈的旧误差增益为

\[
a=hK
=1-b(B-hJ_v),\qquad
b=\frac{N\sigma}{N^2+\gamma^2h^2}.
\]

因此真正的移除率是 \(1-a=b(B-hJ_v)\)。尾段 \(b\to0\) 时，有限 \(J_v\) 下 \(a\to1\)：旧误差不再被清除。线性误差递推为

\[
e_{j+1}=ae_j+K\sigma\xi_{0,j}+\zeta_j+b\epsilon_{v,j},
\qquad \zeta_j\sim\mathcal N(0,C),
\]

所以

\[
V_{j+1}=a^2V_j+q,\qquad
q=K^2\sigma^2+C+b^2\gamma^2.
\]

执行 \(m\) 个 sweep 后，

\[
V_m=a^{2m}V_0+
q\frac{1-a^{2m}}{1-a^2}.
\]

第二次 sweep 相对第一次的风险增量近似为

\[
R_2-R_1
=a^2\left[q-(1-a^2)E_0^2\right].
\]

- 早段：\(1-a^2\) 大、初始偏差大，第二次 sweep 的移除超过注入，故有益。
- 晚段：\(a\simeq1\)，上式近似 \(a^2q>0\)，第二次 sweep 几乎是纯注入。

这正是 `[2,2,1,1]` 胜出的理论原因：保留所有 \(\tau\) 点和每点一次数据精化，只删掉末两 stage 的第二份 \(q\)。

更重要的是，(12)/(19) 存在可写成定理的兼容性缺口。若

\[
x_1\mid x_\tau\sim\mathcal N(f(x_\tau),C),\qquad
x_\tau\mid x_1\sim\mathcal N(hx_1,\sigma^2)
\]

真是同一个 Gaussian joint 的两个 conditionals，则 mixed Hessian 必须满足

\[
C^{-1}J_f=\frac h{\sigma^2}.
\]

唯一相容斜率是

\[
K_*=\frac{Ch}{\sigma^2}
=\frac{hs^2}{h^2s^2+\sigma^2},\qquad
a_*=\frac{h^2s^2}{h^2s^2+\sigma^2}<1,
\]

并恰好满足 fluctuation–dissipation 平衡：

\[
K_*^2\sigma^2+C=(1-a_*^2)s^2.
\]

实际失衡量可写成

\[
\mathcal B
=q-(1-a^2)s^2
=(h^2s^2+\sigma^2)(K^2-K_*^2)+b^2\gamma^2.
\]

(19) 没有 (12) 的 \(S^{-1}\) 收缩；网络权重塌缩后 \(K\to1/h,\ a\to1\)，但 (12) 仍按 \(C\) 注入。因此不是“网络把图越推越坏”，而是“网络清除通道关闭后，先前注入被冻结”。

理论下界的严格裁定是：

- **不存在覆盖全部合法 schedule/invariant kernels 的正数值 universal lower bound。** 恒等核本身也 invariant，有限非齐次轨迹可以靠少做更新改变终态；而且 \(C\to0\) 可随 \(\sigma\to0\)。`0.1411` 已被 `0.1317` 直接反证。
- 对固定 conditional 的 fresh exact Block-1 draw，确有条件方差下界。若
  \[
  M=A^\top A/\eta^2+C^{-1},
  \]
  则
  \[
  \mathbb E[\|P(x^+-x^\star)\|^2\mid x_\tau]
  \ge \operatorname{tr}(PM^{-1}P^\top).
  \]
- 若真是正确 posterior 的独立单样本，则其单-GT Bayes 风险为 \(2\operatorname{tr}\operatorname{Var}(X\mid y)\)。但当前两个 surrogate conditionals 一般不相容，不能用该式证明 `0.1317` 是下界。

旧 spectral 的确定性探针 \(0.0797+0.2468^2=0.1406\) 与 0.1411 的吻合很醒目，但只能称 practical-floor proxy。当前 spread 是“逐坐标 sample std 的算术平均”，不是 RMS；四种子严格分解是

\[
\overline{\mathrm{MSE}}
=\mathrm{MSE}(\bar x,x^\star)+\frac34\operatorname{mean}_i(s_i^2),
\quad \operatorname{mean}(s_i^2)\ge\operatorname{mean}(s_i)^2.
\]

## 2. 自身轨迹上测 γ²/S 是否合法

结论分两种协议：

- **独立 pilot 轨迹测量，随后冻结表再正式评测：按你们当前的窄合法性定义，合法。**
- **同一正式链在线即测即改：不合法。** conditional 随历史变化，不再是“同一个 conditional”，除非把表纳入扩展状态并重新证明 invariance。

轨迹 γ² 可定义为

\[
d_{\rm exact}=Bx_1-\Delta x_0,\qquad
\gamma^2_{\rm traj}(k,\tau)
=\frac1n\mathbb E_{\rm pilot}\|v_\theta(x_\tau)-d_{\rm exact}\|^2.
\]

但它不再是草稿 (20) 所称的 task-independent network property，而是依赖 \(A,y,S,S_{\rm it}\) 和调度的 on-policy calibration；\(\epsilon\perp X_0\) 的推导也丢失。必须用独立 calibration inverse problems/seeds，不能拿 box/junco 评测本身制表再报结果。

方向上，我预测 late/off-manifold \(\gamma^2_{\rm traj}\) 更大且有色。因为

\[
\frac{\partial b}{\partial\gamma^2}
=-\frac{N\sigma h^2}{(N^2+\gamma^2h^2)^2}<0,
\]

更大的 γ² 会进一步关掉唯一移除通道、把 \(K\) 推向 \(1/h\)。对新 `[2,2,1,1]`，预计 hole 变化中位数为持平至恶化 \(0.003\!-\!0.005\)，坏情形约 \(+0.01\)；不看好它继续提升。

S 同理：独立 pilot 后冻结 SPD 谱，在代数上仍能做 exact Gaussian draw，故窄口径合法；但 sampler trajectory covariance 是 posterior/算法噪声，不是 prior covariance，Prop. 4(c) 和“由 prior 测量、非调参”的解释失效。

\[
C_\omega=\frac{\sigma^2S_\omega}{h_\omega^2S_\omega+\sigma^2},
\qquad
\frac{\partial C_\omega}{\partial S_\omega}
=\frac{\sigma^4}{(h_\omega^2S_\omega+\sigma^2)^2}>0.
\]

轨迹谱会把 hole 高频 ratchet 噪声重新写入 \(S\)，从而增大 \(C\)、提前 crossover。预测 hole 恶化约 \(0.005\!-\!0.03\)，spread 上升约 5–20%。若测得更小的 S 而 MSE 下降，那更可能是 posterior-specific overconfidence/spread collapse，不是 prior 校准改善。

## 3. 仍值得做的合法干预

有，而且现在已经不是“合法类内到头”。

首选是逐帧、无新连续参数的阈值调度：

\[
S_{\rm it}(k,\tau)=
\begin{cases}
2,&k=0,1,\\
2,&k=2\ \text{且}\ b_{k,\tau}\ge1,\\
1,&\text{其他}.
\end{cases}
\]

按现表即 stage 2 在约 \(\tau=0.44\) 后由 2 降为 1，stage 3 全部为 1；约 64 NFE。预测：

- 打败旧 0.1411：概率约 80–90%；
- 打败新 0.1317：概率约 35–50%；
- hole 约 0.128–0.134，obs 0.0026–0.0028，spread 约 0.23–0.24。

第二，补跑 spectral `[2,2,2,1]`。本轮只是远端 I/O 连续失败，并非负结果；pooled 同配置已经从 0.1499 降到 0.1423。预测 spectral 0.133–0.139，打败旧基线概率约 70%，但打败 `[2,2,1,1]` 概率低于 25%。

如果还允许一次表方向实验，我只推荐 held-out 的有色速度误差谱，而不是自身轨迹标量 γ：

\[
\Gamma_{k,\tau}(\omega)
=\mathbb E_{\rm calib}\left|\mathcal F(v-d_{\rm exact})(\omega)\right|^2,
\]

并把 (19) 改写为

\[
[N^2+H\Gamma H]\hat x_1
=N[\Delta x_\tau+\sigma v]+\Gamma Hx_\tau.
\]

草稿在 (20) 后已经允许 \(\Sigma_\epsilon\) 推广；冻结的 FFT 对角 \(\Gamma\) 无新 NFE、无调参。它打败旧 0.1411 的概率约 40%，但打败新 0.1317 只有约 20–30%。

`ode_steps=[10,10,14,14]` 当前不是负结果，而是未运行成功：γ² 表只有 10 个 τ 键，新增点触发 positional fallback 越界。必须先在实际 14 点重测并冻结 γ²，才能解释。

最有价值的非合法选项仍是 antithetic endpoint center：

\[
x_\tau^-=2Hx_1-x_\tau,\qquad
\tilde x_1=\tfrac12\{\hat x_1(x_\tau)+\hat x_1(x_\tau^-)\}.
\]

历史 spectral 4-seed 为 hole **0.0875**、obs 0.0022、spread 0.115、NFE 160。代价是：

- 2×网络调用；
- 明确改变原 conditional mean，不再保持原 conditional；
- spread 从 0.247 减半，质量收益中包含显著的欠散布/方差压缩；
- 应称“quality-biased reconstruction option/机制验证”，不能称 posterior sampler 改进。

## 4. 最终报告建议

主结论应改成：

> Kernel correlation 与粗 τ 重分配没有可复现均值收益；但分阶段减少尾段 exact refresh 在两种 S 上均稳定改善。推荐配置为 spectral S、\(\rho=0\)、均匀 τ、\(S_{\rm it}=[2,2,1,1]\)。它以更低 NFE 将 hole MSE 从 0.1411 降至 0.1317，但仍未达到 Alg2 的 0.102。

值得写进论文/草稿的机理：

- “每块 exact”不等于“全局 exact posterior”：两个 Gaussian surrogate conditionals 必须满足 \(C^{-1}J_f=H/\sigma^2\)，当前一般不满足。草稿自己也承认无共同 invariant joint；标题、摘要和贡献中的 “exact posterior sampler” 应降为“exact updates for frozen surrogate conditionals”。
- \(b\to0,\ a\to1\) 的注入—移除失衡，以及第二次 late sweep 变成净 \(q\) 注入。
- `[2,2,1,1]` 与 pooled `[2,2,2,1]` 对该模型的跨 S 验证。
- spectral S 的价值：按真实频率功率减少不存在的高频 prior innovation；正式论文必须改用独立训练/校准集，不能保留 demo-only 统计。
- persistence 的四种子彩票：冻结 conditional 的 invariance 不保证有限、非齐次、网络反馈链的终态 MSE。
- reflection 只宜写“与网络在 reflected/off-trajectory 状态上的外推失效一致”；若没有直接测 reflected-state γ²，不要写成已证明因果。
- 调度的双重职责：减少 late frames/sweeps 可降注入，但偏低 τ 会损失末端 obs 精化；“保留全部 τ、只删第二 sweep”因此优于 shift。

正文结果表只需保留：

- spectral/pooled 两基线；
- spectral `[2,2,1,1]` 与 pooled `[2,2,2,1]`；
- persistence ladder/global 各一个；
- combo 与 sched-only 对照；
- shift；
- 14-step 标为“未运行：γ 网格不匹配”，不能列作负结果。

可删或降附录：

- seed42 的密集 ρ 排名及 \(q\approx1/b\) 规则；
- 所有反射强度、banded/alternating 推演；
- warm start、均值 γ²、sigma 门限微扫、float32/CG/查表调试；
- 重复 montage、逐帧原始 CSV 和中间配置。

最后修正一个统计措辞：你写的“跨种子方差 ×5–6”实际是**标准差**约 ×5.3–6.3；对应方差约 ×28–39。并明确 \(n=4\)、“±”是 sample std 不是置信区间，且当前只覆盖单图、单 mask、单任务。
tokens used
170,355
先更新总裁定：最后一批已经改变了问题前提。`0.1411` 不是合法类下界，尾段减少 sweep 是稳定、合法的突破口。

| 配置 | hole，4 seeds | obs | spread | NFE |
|---|---:|---:|---:|---:|
| spectral `[2,2,2,2]` | 0.1411±0.0047 | 0.0026 | 0.2468 | 80 |
| spectral `[2,2,1,1]` | **0.1317±0.0038** | 0.0027 | 0.2379 | 60 |
| pooled `[2,2,2,2]` | 0.1499±0.0007 | 0.0032 | 0.2646 | 80 |
| pooled `[2,2,2,1]` | **0.1423±0.0027** | 0.0034 | 0.2513 | 70 |

所以真正稳固的不是“均匀 \(S_{\rm it}=2\)”本身，而是把所有 stage 绑在同一调度时形成的局部折中。解开后，前两 stage 用 2 次、后两 stage 用 1 次，恰好避开尾段第二次净注入。

## 1. 鞍点、注入—移除平衡与理论下界

在 \(\ker A\) 的单个共同谱模态上，记 \(h,N,s^2\) 分别为 \(H_\tau,N_k,S\) 的特征值，\(\Delta=e_k-s_k\)。由 (12)，

\[
C=\left(\frac{h^2}{\sigma^2}+\frac1{s^2}\right)^{-1}
=\frac{\sigma^2s^2}{h^2s^2+\sigma^2}.
\]

这是每次 fresh Block-1 draw 必须注入的方差。

把 (19) 的 endpoint map 记为 \(f(x_\tau)=\hat x_1\)，在当前轨迹附近线性化：

\[
K=J_f=
\frac{N\Delta+\gamma^2h+N\sigma J_v}
     {N^2+\gamma^2h^2}.
\]

利用 \(N=\sigma B+\Delta h\)，一轮反馈的旧误差增益为

\[
a=hK
=1-b(B-hJ_v),\qquad
b=\frac{N\sigma}{N^2+\gamma^2h^2}.
\]

因此真正的移除率是 \(1-a=b(B-hJ_v)\)。尾段 \(b\to0\) 时，有限 \(J_v\) 下 \(a\to1\)：旧误差不再被清除。线性误差递推为

\[
e_{j+1}=ae_j+K\sigma\xi_{0,j}+\zeta_j+b\epsilon_{v,j},
\qquad \zeta_j\sim\mathcal N(0,C),
\]

所以

\[
V_{j+1}=a^2V_j+q,\qquad
q=K^2\sigma^2+C+b^2\gamma^2.
\]

执行 \(m\) 个 sweep 后，

\[
V_m=a^{2m}V_0+
q\frac{1-a^{2m}}{1-a^2}.
\]

第二次 sweep 相对第一次的风险增量近似为

\[
R_2-R_1
=a^2\left[q-(1-a^2)E_0^2\right].
\]

- 早段：\(1-a^2\) 大、初始偏差大，第二次 sweep 的移除超过注入，故有益。
- 晚段：\(a\simeq1\)，上式近似 \(a^2q>0\)，第二次 sweep 几乎是纯注入。

这正是 `[2,2,1,1]` 胜出的理论原因：保留所有 \(\tau\) 点和每点一次数据精化，只删掉末两 stage 的第二份 \(q\)。

更重要的是，(12)/(19) 存在可写成定理的兼容性缺口。若

\[
x_1\mid x_\tau\sim\mathcal N(f(x_\tau),C),\qquad
x_\tau\mid x_1\sim\mathcal N(hx_1,\sigma^2)
\]

真是同一个 Gaussian joint 的两个 conditionals，则 mixed Hessian 必须满足

\[
C^{-1}J_f=\frac h{\sigma^2}.
\]

唯一相容斜率是

\[
K_*=\frac{Ch}{\sigma^2}
=\frac{hs^2}{h^2s^2+\sigma^2},\qquad
a_*=\frac{h^2s^2}{h^2s^2+\sigma^2}<1,
\]

并恰好满足 fluctuation–dissipation 平衡：

\[
K_*^2\sigma^2+C=(1-a_*^2)s^2.
\]

实际失衡量可写成

\[
\mathcal B
=q-(1-a^2)s^2
=(h^2s^2+\sigma^2)(K^2-K_*^2)+b^2\gamma^2.
\]

(19) 没有 (12) 的 \(S^{-1}\) 收缩；网络权重塌缩后 \(K\to1/h,\ a\to1\)，但 (12) 仍按 \(C\) 注入。因此不是“网络把图越推越坏”，而是“网络清除通道关闭后，先前注入被冻结”。

理论下界的严格裁定是：

- **不存在覆盖全部合法 schedule/invariant kernels 的正数值 universal lower bound。** 恒等核本身也 invariant，有限非齐次轨迹可以靠少做更新改变终态；而且 \(C\to0\) 可随 \(\sigma\to0\)。`0.1411` 已被 `0.1317` 直接反证。
- 对固定 conditional 的 fresh exact Block-1 draw，确有条件方差下界。若
  \[
  M=A^\top A/\eta^2+C^{-1},
  \]
  则
  \[
  \mathbb E[\|P(x^+-x^\star)\|^2\mid x_\tau]
  \ge \operatorname{tr}(PM^{-1}P^\top).
  \]
- 若真是正确 posterior 的独立单样本，则其单-GT Bayes 风险为 \(2\operatorname{tr}\operatorname{Var}(X\mid y)\)。但当前两个 surrogate conditionals 一般不相容，不能用该式证明 `0.1317` 是下界。

旧 spectral 的确定性探针 \(0.0797+0.2468^2=0.1406\) 与 0.1411 的吻合很醒目，但只能称 practical-floor proxy。当前 spread 是“逐坐标 sample std 的算术平均”，不是 RMS；四种子严格分解是

\[
\overline{\mathrm{MSE}}
=\mathrm{MSE}(\bar x,x^\star)+\frac34\operatorname{mean}_i(s_i^2),
\quad \operatorname{mean}(s_i^2)\ge\operatorname{mean}(s_i)^2.
\]

## 2. 自身轨迹上测 γ²/S 是否合法

结论分两种协议：

- **独立 pilot 轨迹测量，随后冻结表再正式评测：按你们当前的窄合法性定义，合法。**
- **同一正式链在线即测即改：不合法。** conditional 随历史变化，不再是“同一个 conditional”，除非把表纳入扩展状态并重新证明 invariance。

轨迹 γ² 可定义为

\[
d_{\rm exact}=Bx_1-\Delta x_0,\qquad
\gamma^2_{\rm traj}(k,\tau)
=\frac1n\mathbb E_{\rm pilot}\|v_\theta(x_\tau)-d_{\rm exact}\|^2.
\]

但它不再是草稿 (20) 所称的 task-independent network property，而是依赖 \(A,y,S,S_{\rm it}\) 和调度的 on-policy calibration；\(\epsilon\perp X_0\) 的推导也丢失。必须用独立 calibration inverse problems/seeds，不能拿 box/junco 评测本身制表再报结果。

方向上，我预测 late/off-manifold \(\gamma^2_{\rm traj}\) 更大且有色。因为

\[
\frac{\partial b}{\partial\gamma^2}
=-\frac{N\sigma h^2}{(N^2+\gamma^2h^2)^2}<0,
\]

更大的 γ² 会进一步关掉唯一移除通道、把 \(K\) 推向 \(1/h\)。对新 `[2,2,1,1]`，预计 hole 变化中位数为持平至恶化 \(0.003\!-\!0.005\)，坏情形约 \(+0.01\)；不看好它继续提升。

S 同理：独立 pilot 后冻结 SPD 谱，在代数上仍能做 exact Gaussian draw，故窄口径合法；但 sampler trajectory covariance 是 posterior/算法噪声，不是 prior covariance，Prop. 4(c) 和“由 prior 测量、非调参”的解释失效。

\[
C_\omega=\frac{\sigma^2S_\omega}{h_\omega^2S_\omega+\sigma^2},
\qquad
\frac{\partial C_\omega}{\partial S_\omega}
=\frac{\sigma^4}{(h_\omega^2S_\omega+\sigma^2)^2}>0.
\]

轨迹谱会把 hole 高频 ratchet 噪声重新写入 \(S\)，从而增大 \(C\)、提前 crossover。预测 hole 恶化约 \(0.005\!-\!0.03\)，spread 上升约 5–20%。若测得更小的 S 而 MSE 下降，那更可能是 posterior-specific overconfidence/spread collapse，不是 prior 校准改善。

## 3. 仍值得做的合法干预

有，而且现在已经不是“合法类内到头”。

首选是逐帧、无新连续参数的阈值调度：

\[
S_{\rm it}(k,\tau)=
\begin{cases}
2,&k=0,1,\\
2,&k=2\ \text{且}\ b_{k,\tau}\ge1,\\
1,&\text{其他}.
\end{cases}
\]

按现表即 stage 2 在约 \(\tau=0.44\) 后由 2 降为 1，stage 3 全部为 1；约 64 NFE。预测：

- 打败旧 0.1411：概率约 80–90%；
- 打败新 0.1317：概率约 35–50%；
- hole 约 0.128–0.134，obs 0.0026–0.0028，spread 约 0.23–0.24。

第二，补跑 spectral `[2,2,2,1]`。本轮只是远端 I/O 连续失败，并非负结果；pooled 同配置已经从 0.1499 降到 0.1423。预测 spectral 0.133–0.139，打败旧基线概率约 70%，但打败 `[2,2,1,1]` 概率低于 25%。

如果还允许一次表方向实验，我只推荐 held-out 的有色速度误差谱，而不是自身轨迹标量 γ：

\[
\Gamma_{k,\tau}(\omega)
=\mathbb E_{\rm calib}\left|\mathcal F(v-d_{\rm exact})(\omega)\right|^2,
\]

并把 (19) 改写为

\[
[N^2+H\Gamma H]\hat x_1
=N[\Delta x_\tau+\sigma v]+\Gamma Hx_\tau.
\]

草稿在 (20) 后已经允许 \(\Sigma_\epsilon\) 推广；冻结的 FFT 对角 \(\Gamma\) 无新 NFE、无调参。它打败旧 0.1411 的概率约 40%，但打败新 0.1317 只有约 20–30%。

`ode_steps=[10,10,14,14]` 当前不是负结果，而是未运行成功：γ² 表只有 10 个 τ 键，新增点触发 positional fallback 越界。必须先在实际 14 点重测并冻结 γ²，才能解释。

最有价值的非合法选项仍是 antithetic endpoint center：

\[
x_\tau^-=2Hx_1-x_\tau,\qquad
\tilde x_1=\tfrac12\{\hat x_1(x_\tau)+\hat x_1(x_\tau^-)\}.
\]

历史 spectral 4-seed 为 hole **0.0875**、obs 0.0022、spread 0.115、NFE 160。代价是：

- 2×网络调用；
- 明确改变原 conditional mean，不再保持原 conditional；
- spread 从 0.247 减半，质量收益中包含显著的欠散布/方差压缩；
- 应称“quality-biased reconstruction option/机制验证”，不能称 posterior sampler 改进。

## 4. 最终报告建议

主结论应改成：

> Kernel correlation 与粗 τ 重分配没有可复现均值收益；但分阶段减少尾段 exact refresh 在两种 S 上均稳定改善。推荐配置为 spectral S、\(\rho=0\)、均匀 τ、\(S_{\rm it}=[2,2,1,1]\)。它以更低 NFE 将 hole MSE 从 0.1411 降至 0.1317，但仍未达到 Alg2 的 0.102。

值得写进论文/草稿的机理：

- “每块 exact”不等于“全局 exact posterior”：两个 Gaussian surrogate conditionals 必须满足 \(C^{-1}J_f=H/\sigma^2\)，当前一般不满足。草稿自己也承认无共同 invariant joint；标题、摘要和贡献中的 “exact posterior sampler” 应降为“exact updates for frozen surrogate conditionals”。
- \(b\to0,\ a\to1\) 的注入—移除失衡，以及第二次 late sweep 变成净 \(q\) 注入。
- `[2,2,1,1]` 与 pooled `[2,2,2,1]` 对该模型的跨 S 验证。
- spectral S 的价值：按真实频率功率减少不存在的高频 prior innovation；正式论文必须改用独立训练/校准集，不能保留 demo-only 统计。
- persistence 的四种子彩票：冻结 conditional 的 invariance 不保证有限、非齐次、网络反馈链的终态 MSE。
- reflection 只宜写“与网络在 reflected/off-trajectory 状态上的外推失效一致”；若没有直接测 reflected-state γ²，不要写成已证明因果。
- 调度的双重职责：减少 late frames/sweeps 可降注入，但偏低 τ 会损失末端 obs 精化；“保留全部 τ、只删第二 sweep”因此优于 shift。

正文结果表只需保留：

- spectral/pooled 两基线；
- spectral `[2,2,1,1]` 与 pooled `[2,2,2,1]`；
- persistence ladder/global 各一个；
- combo 与 sched-only 对照；
- shift；
- 14-step 标为“未运行：γ 网格不匹配”，不能列作负结果。

可删或降附录：

- seed42 的密集 ρ 排名及 \(q\approx1/b\) 规则；
- 所有反射强度、banded/alternating 推演；
- warm start、均值 γ²、sigma 门限微扫、float32/CG/查表调试；
- 重复 montage、逐帧原始 CSV 和中间配置。

最后修正一个统计措辞：你写的“跨种子方差 ×5–6”实际是**标准差**约 ×5.3–6.3；对应方差约 ×28–39。并明确 \(n=4\)、“±”是 sample std 不是置信区间，且当前只覆盖单图、单 mask、单任务。
