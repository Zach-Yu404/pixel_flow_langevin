# PixelFlow Algorithm 4 Block 1：Eq. (22) 乘 \(\sigma_\tau^2\) 的独立理论审查

> 证据边界：以下推导只使用任务题面列出的实现事实，不使用源码、实验结果或外部资料。

## 审查结论摘要

令 \(\sigma=\sigma_\tau>0\)、\(c=\sigma^2\)。若目标是把**同一个随机线性方程**整体乘以
\(c\)，则必须使用

\[
\widetilde M=cM,\qquad \widetilde b=cb,\qquad
\widetilde\zeta=c\zeta.
\]

关键点是

\[
\operatorname{Cov}(\widetilde\zeta)=c^2M=c\widetilde M,
\]

而不是 \(\operatorname{Cov}(\widetilde\zeta)=\widetilde M\)。前者给出原目标协方差
\(M^{-1}\)；后者会给出 \(c^{-1}M^{-1}=\sigma^{-2}M^{-1}\)。当
\(\sigma=10^{-8}\) 时，误用后者会把输出协方差放大 \(10^{16}\) 倍、标准差放大
\(10^8\) 倍。

因此，正确提议在精确算术下不改变任何样本或统计性质；它应称为**全局标量数值重缩放**或
**代数等价的方程归一化**，不是新的采样算法，也不改变条件数。它可能缓解 fp32 的大数中间量
和 reduction 溢出，但不能修复相对量级差、加法吸收或病态性；若不同时处理绝对 clamp 与
预条件器 floor，还可能显著恶化 PCG。

## 记号

下文简写

\[
H=H_\tau,\qquad \widehat x=\widehat x_1,\qquad c=\sigma^2.
\]

原系统为

\[
M=\eta^{-2}A^\top A+\sigma^{-2}H^\top H+S^{-1},
\]

\[
b=\eta^{-2}A^\top y+\sigma^{-2}H^\top H\widehat x+S^{-1}\widehat x,
\]

\[
\zeta=\eta^{-1}A^\top\xi_y+\sigma^{-1}H^\top\xi_h+S^{-1/2}\xi_s,
\qquad \operatorname{Cov}(\zeta)=M.
\]

## 1. \(\widetilde M\)、\(\widetilde b\)、\(\widetilde\zeta\) 的逐项形式

整体乘 \(c=\sigma^2\) 后，算子应为

\[
\boxed{
\widetilde Mv
=\frac{\sigma^2}{\eta^2}A^\top Av
+H^\top Hv
+\sigma^2S^{-1}v
}.
\]

确定 RHS 应为

\[
\boxed{
\widetilde b
=\frac{\sigma^2}{\eta^2}A^\top y
+H^\top H\widehat x
+\sigma^2S^{-1}\widehat x
}.
\]

随机 RHS 应为

\[
\boxed{
\widetilde\zeta
=\frac{\sigma^2}{\eta}A^\top\xi_y
+\sigma H^\top\xi_h
+\sigma^2S^{-1/2}\xi_s
}.
\]

这里第二项使用了 \(\sigma^2/\sigma=\sigma\)。

两类 \(S\) 都可直接实现：

- 若 \(S=s^2I\)，则三个相关项分别是
  \(\sigma^2v/s^2\)、\(\sigma^2\widehat x/s^2\)、\(\sigma^2\xi_s/s\)。
- 若 \(S\) 在 FFT 域由 \(P(\omega)\) 对角化，则算子项与确定 RHS 在原
  `apply_S_inv` 输出上乘 \(\sigma^2\)，随机项在原 `apply_S_inv_sqrt` 输出上乘
  \(\sigma^2\)。原实现若可定义，所需条件仍是相应的 \(P(\omega)>0\)。

但有一个容易混淆的限制：正确随机项是
\(\sigma^2S^{-1/2}\xi_s\)，不是
\((\sigma^2S^{-1})^{1/2}\xi_s=\sigma S^{-1/2}\xi_s\)。后者是为
\(\widetilde M\) 构造“协方差等于精度”的标准 RTO 噪声，会改变目标协方差。故不能把
\(\widetilde M\) 当成一个新的目标精度后，原封不动复用“噪声因子等于各精度项平方根”的逻辑。

## 2. 两个缩放恒等式何时严格成立

在精确算术下，逐项展开立即给出

\[
\boxed{\widetilde M=\sigma^2M},\qquad
\boxed{\widetilde b+\widetilde\zeta=\sigma^2(b+\zeta)}.
\]

需要区分三种“相同”：

1. **代数相同。** \(\sigma\) 必须是该系统上的同一个正标量，且所有算子、\(y\)、
   \(\widehat x\) 不变。
2. **逐样本、逐路径相同。** 必须使用同一组已经实现出来的
   \(\xi_y,\xi_h,\xi_s\)。要让后续 Block 2 的 \(\xi_0\) 也与基线对齐，还必须保持三次
   抽样的顺序、形状、次数及 CPU generator 的推进量不变。
3. **仅分布相同。** 可以换成另一组彼此独立且与当前状态独立的标准高斯，再使用上节的正确
   缩放系数。此时条件分布不变，但固定 seed 的样本路径不相同；若抽样次数或顺序也变化，后续
   \(\xi_0\) 也不会对齐。

在 float32 中还要再区分“数学相等”和“逐比特相等”。即使复用同一组 \(\xi\)，

\[
\operatorname{fl}\!\left(c\,\operatorname{fl}(b+\zeta)\right)
\]

一般也不等于把每一项先重构为缩放系数、再逐项相加的结果；\(c/\sigma\) 与直接使用
\(\sigma\)、不同的加法次序也会引入不同舍入。若先形成原 RHS 再乘 \(c\)，可以获得最直接的
张量级对应，但会失去避免原式大中间量的主要数值收益。

## 3. 解的均值与协方差

使用正确缩放噪声时，

\[
\widetilde x
=(cM)^{-1}(cb+c\zeta)
=M^{-1}(b+\zeta).
\]

所以同一组 \(\xi\) 下，精确解逐样本完全相同。于是

\[
\boxed{\mathbb E[\widetilde x]=M^{-1}b}.
\]

又因为

\[
\operatorname{Cov}(\widetilde\zeta)
=\operatorname{Cov}(c\zeta)=c^2M,
\]

故

\[
\begin{aligned}
\operatorname{Cov}(\widetilde x)
&=(cM)^{-1}(c^2M)(cM)^{-1}\\
&=M^{-1}.
\end{aligned}
\]

这里使用了题面中 \(M\) 为对称正定精度算子的事实语义。结论依赖于精确求解；有限精度、有限
迭代 PCG 若不再保持缩放齐次性，会产生额外的均值和协方差误差。

## 4. 应该缩放原噪声，还是为 \(\widetilde M\) 重建 RTO 噪声

### 4.1 正确方案：\(\widetilde\zeta=c\zeta\)

同一组 \(\xi\) 给出逐路径相同；换一组独立标准高斯但仍使用

\[
\frac{\sigma^2}{\eta}A^\top\xi_y'
+\sigma H^\top\xi_h'
+\sigma^2S^{-1/2}\xi_s'
\]

则只保证分布相同。两者均满足

\[
\operatorname{Cov}(\widetilde\zeta)=c^2M=c\widetilde M,
\]

输出协方差均为 \(M^{-1}\)。

### 4.2 错误方案：直接令新噪声的协方差为 \(\widetilde M\)

为 \(\widetilde M\) 按标准 RTO 因子重建会得到

\[
\zeta_{\mathrm{RTO},\widetilde M}
=\frac{\sigma}{\eta}A^\top\xi_y'
+H^\top\xi_h'
+\sigma S^{-1/2}\xi_s',
\]

其协方差确为 \(\widetilde M=cM\)。均值仍然是

\[
(cM)^{-1}(cb)=M^{-1}b,
\]

但输出协方差变为

\[
\boxed{
(cM)^{-1}(cM)(cM)^{-1}
=(cM)^{-1}=c^{-1}M^{-1}=\sigma^{-2}M^{-1}
}.
\]

一维反例足够说明问题。取 \(M=1,b=0\)：

- 正确缩放是 \(cx=c\zeta\)、\(\zeta\sim\mathcal N(0,1)\)，所以
  \(x\sim\mathcal N(0,1)\)。
- 若令 RHS 噪声 \(\varepsilon\sim\mathcal N(0,c)\)，则
  \(cx=\varepsilon\)，从而 \(\operatorname{Var}(x)=1/c\)。

若确实想复用“\(\operatorname{Cov}=\widetilde M\)”的噪声生成器，可以先生成上面的
\(\zeta_{\mathrm{RTO},\widetilde M}\)，再额外乘 \(\sqrt c=\sigma\)。所得协方差是
\(c\widetilde M=c^2M\)，这才等价于正确方案。

## 5. 条件数与 fp32 动态范围

### 5.1 精确条件数不变

对 \(c>0\)，

\[
\lambda_i(cM)=c\lambda_i(M),\qquad
\boxed{\kappa_2(cM)=\kappa_2(M)}.
\]

所以整体缩放不改变谱跨度，也不改变三项的相对权重。系数跨度本身也不等同于矩阵条件数：若
\(A=H=I\) 且 \(S^{-1}\) 也是标量恒等项，\(M\) 仍是恒等阵的标量倍数，条件数为 1；若
\(H^\top H\) 有零空间，而数据或先验只在该零空间提供较小特征值，才会出现与系数比相近的
巨大条件数。整体乘 \(c\) 对这两种情况都无改善。

### 5.2 晚期的具体系数量级

当 \(\sigma=10^{-8}\)、\(c=10^{-16}\)、\(\eta=0.05\) 时：

| 通道 | 原 \(M\) 系数 | \(\widetilde M\) 系数 | 原 \(\zeta\) 系数 | 正确 \(\widetilde\zeta\) 系数 |
|---|---:|---:|---:|---:|
| \(A\) | \(400\) | \(4\times10^{-14}\) | \(20\) | \(2\times10^{-15}\) |
| \(H\) | \(10^{16}\) | \(1\) | \(10^8\) | \(10^{-8}\) |
| \(S\) | \(q_S\) | \(10^{-16}q_S\) | \(\sqrt{q_S}\) | \(10^{-16}\sqrt{q_S}\) |

这里标量情形 \(q_S=1/s^2\)，谱情形逐频率为 \(q_S(\omega)=1/P(\omega)\)。确定 RHS
中相应精度项的系数与 \(M\) 两列相同。

例如假设相关算子输出的量级相近，则数据项与 \(H\) 项之比在原系统是
\(400/10^{16}=4\times10^{-14}\)，在缩放系统仍是 \(4\times10^{-14}/1\)。

### 5.3 能改善什么

float32 最大有限数约为 \(3.4\times10^{38}\)，最小正规数约为
\(1.18\times10^{-38}\)，最小次正规数约为 \(1.4\times10^{-45}\)，1 附近相邻数间距约为
\(1.19\times10^{-7}\)。\(10^{16}\) 与 \(10^{-16}\) 本身均可表示；风险主要来自后续乘法、
平方和与 reduction。

若未缩放 RHS 的 \(N\) 个分量均约为 \(10^{16}\)，则

\[
\operatorname{dot}(b,b)\approx N10^{32}.
\]

当 \(N\gtrsim3.4\times10^6\) 时，float32 reduction 就可能溢出；缩放后对应平方和约为
\(N\)。直接逐项构造缩放系统还能避免先产生 \(10^{16}\) 级中间量，再乘 \(10^{-16}\)。若
原中间量已经成为 `Inf`，事后乘 \(c\) 不可能恢复。

### 5.4 不能改善什么

缩放不能恢复相对过小而被吸收的项。标量反例是

\[
10^{16}+400
\quad\longleftrightarrow\quad
1+4\times10^{-14}.
\]

在前者中，\(10^{16}\) 附近的 float32 间距远大于 400；在后者中，1 附近的间距远大于
\(4\times10^{-14}\)。小项在两边都可能被完全吸收。因此它不能改善：

- 三个精度项的相对量级差；
- 小特征值、近零空间和条件数；
- 由相消造成的相对误差；
- \(A\)、\(H\) 或 FFT 算子内部、发生在外部标量乘法之前的 reduction 误差。

它还可能把弱方向移入次正规或下溢区。反例：若 \(H^\top Hv=0\)、
\(A^\top Av=10^{-32}\)，则原数据项为 \(4\times10^{-30}\)，仍是正规数；缩放后为
\(4\times10^{-46}\)，低于最小次正规数，会成为 0。

### 5.5 逐项重构优于事后整体相乘，但二者各有风险

- `c * original_Mv` 或 `c * original_rhs` 不能防止原表达式中的溢出、吸收和舍入；它只会缩小
  已经形成的结果。
- 为获得主要收益，应直接使用 \(H^\top H\)、\(\sigma H^\top\xi_h\)、
  \(\sigma^2/\eta^2\) 等消去巨大倒数后的系数，而不是先算
  \(\sigma^{-2}\) 再乘 \(\sigma^2\)。
- 对谱项，\(c(v/P)\) 与 \((c/P)v\) 在 fp32 中不必相等。前者可能在 \(v/P\) 处先溢出，
  后者可能让 \(c/P\) 过早进入次正规区；应根据 \(P\) 的实际范围选择稳定顺序。

结论是：缩放只移动绝对指数范围，不压缩相对动态范围。

## 6. PCG 停机、clamp 与预条件器是否要同步修改

为避免与确定项 \(b\) 重名，令送入 `pcg_solve` 的总 RHS 为

\[
q=b+\zeta,\qquad \widetilde q=cq.
\]

### 6.1 精确算术下不需要改的量

无绝对 clamp 时，

\[
\frac{\|\widetilde r\|_2}{\|\widetilde q\|_2}
=\frac{c\|r\|_2}{c\|q\|_2}
=\frac{\|r\|_2}{\|q\|_2}.
\]

因此：

- `cg_tol=1e-5` 是无量纲相对阈值，不应乘 \(c\)；
- `max_iter=300` 没有数学上的尺度修改；
- 每个 batch 元素的相对残差不变，故 `max_batch` 也不变；
- 若预条件器随系统只差一个整体正标量，预条件后的条件数不变。

以自然的 Jacobi 缩放为例。设原逆预条件器为 \(K=D^{-1}\)。若
\(\widetilde D=cD\)，则 \(\widetilde K=K/c\)。从相同初值出发，有

\[
\widetilde r=cr,\quad \widetilde z=z,\quad \widetilde p=p,
\quad \widetilde{p^\top Mp}=c(p^\top Mp),
\quad \widetilde{r^\top z}=c(r^\top z).
\]

故 \(\alpha\)、\(\beta\) 以及每一步 \(x\) 都与原 PCG 相同。

### 6.2 三个固定 `1e-12` 不具缩放不变性

在上述自然预条件器尺度下，要复现原来的分支行为，应有

\[
\begin{aligned}
\epsilon_{\|q\|,\mathrm{new}}&=c\times10^{-12},\\
\epsilon_{p^\top Mp,\mathrm{new}}&=c\times10^{-12},\\
\epsilon_{r^\top z,\mathrm{new}}&=c\times10^{-12}.
\end{aligned}
\]

在 \(c=10^{-16}\) 时三者都是 \(10^{-28}\)，而不是 \(10^{-12}\)。保留原常数相当于把
原坐标中的触发门槛从 \(10^{-12}\) 提到 \(10^4\)，提高了 16 个数量级。

若故意不缩放逆预条件器，则精确 PCG 的 \(x\) 迭代仍可保持等价，但此时
\(r^\top z\) 缩放为 \(c^2\)，\(p^\top Mp\) 缩放为 \(c^3\)；相应 floor 应分别为
\(10^{-44}\) 与 \(10^{-60}\)，后者连 float32 次正规数都无法表示。这说明绝对常数本质上
不是尺度稳健的 breakdown 判据。

一个一维晚期反例可单独隔离 `alpha` clamp 的问题。取

\[
M=10^{16},\quad q=10^8,\quad K=M^{-1}=10^{-16},\quad x_0=0.
\]

原 PCG 有 \(r=10^8,z=p=10^{-8}\)，且
\(rz=pMp=1\)，一步得到 \(x=10^{-8}\)。缩放后

\[
\widetilde M=1,\quad \widetilde q=10^{-8},\quad \widetilde K=1,
\]

所以 \(r=z=p=10^{-8}\)，正确的 \(rz=p\widetilde Mp=10^{-16}\)，精确
\(\alpha=1\)。但固定 clamp 给出

\[
\alpha_{\mathrm{impl}}=\frac{10^{-16}}{\max(10^{-16},10^{-12})}=10^{-4}.
\]

第一步只走到 \(10^{-12}\)。此时 `b_norm=1e-8` 尚未触发 floor，故误差完全由内积的绝对
clamp 造成。

严格等价所需的 \(c\times10^{-12}\) 只是“复现旧 floor 语义”的答案，不是最稳健的设计。
更稳健的做法是：范数对零 RHS 单独处理；对 \(p^\top Mp\)、\(r^\top z\) 使用与操作数范数和
机器精度相关的相对 breakdown 判据，并显式处理非正值、`Inf`、`NaN`。否则
`.clamp(min=...)` 还会把负内积静默改成正数。仅把 floor 改成 \(10^{-28}\) 也不能防止
`dot` 在取平方根之前已经下溢，故 reduction 的稳定累加或更高精度仍需单独考虑。

### 6.3 Jacobi floor 必须同步尺度化

若原定义是

\[
K=\frac{1}{\max(D,10^{-12})},
\]

则保持 \(\widetilde K=K/c\) 需要

\[
\boxed{
\widetilde K
=\frac{1}{\max(cD,c\times10^{-12})}
}.
\]

晚期保留固定 `1e-12` 会产生非均匀畸变。以算子对角量级为 1 的方向为例：

- \(H\) 主导方向：缩放对角约为 1，floor 不触发；
- 只有数据项的方向：缩放对角约为 \(4\times10^{-14}\)，正确逆对角为
  \(2.5\times10^{13}\)，固定 floor 只给 \(10^{12}\)，相差约 25 倍；
- 只有 \(S^{-1}\) 且 \(q_S\approx1\) 的方向：缩放对角约为 \(10^{-16}\)，正确逆对角为
  \(10^{16}\)，固定 floor 只给 \(10^{12}\)，相差 \(10^4\) 倍。

因为 floor 只在部分方向触发，这不是一个对 PCG 无害的全局常数倍，而会改变预条件器形状、
收敛轨迹，并可能在 300 次上限内改变近似解。

### 6.4 Spectral 预条件探针必须所有项一起缩放

记 \(\bar q_S=\operatorname{mean}(1/P)\)。原探针是

\[
d=M\mathbf1-S^{-1}\mathbf1+\bar q_S\mathbf1.
\]

缩放系统对应的探针必须是

\[
\boxed{
\widetilde d
=\widetilde M\mathbf1-cS^{-1}\mathbf1+c\bar q_S\mathbf1
=cd
}.
\]

于是 \(\widetilde M_{\mathrm{inv,prec}}=1/\widetilde d=(1/c)(1/d)\)。若只把第一项换成
\(\widetilde M\mathbf1\)，却保留未缩放的减项与补项，所得探针一般不是 \(cd\)；晚期相关谱
先验替代项可错 \(10^{16}\) 倍。若探针对角也应用 floor，该 floor 同样要尺度化或改成相对 floor。

## 总判断：统计性质、命名与对 hole MSE 的预期

### 精确算术

若 \(\widetilde M=cM\)、\(\widetilde b=cb\)、\(\widetilde\zeta=c\zeta\)，则同一组随机数下
每个 Block 1 解逐样本相同。因此 Block 1 条件分布、整个算法的转移核以及所有统计性质都不变；
同一路径的 hole MSE 等指标也应严格相同。

这项改动应称为：

- **全局标量数值重缩放**；
- **代数等价的线性系统归一化**；或
- 在同步修正数值保护后，**实现层的数值稳定性修正**。

它不应称为新的采样算法、统计改进或改善条件数的预条件化方法。

### float32 与有限 PCG

- 若当前主要问题是大中间量、平方 reduction 溢出、`Inf/NaN` 或由绝对尺度造成的假收敛，
  正确的逐项重构和尺度一致保护可能提高求解忠实度。
- 若主要问题是 \(H\)、数据、先验三项的相对分离、加法吸收或巨大条件数，则整体缩放没有根治作用。
- 若保留原 `1e-12` clamps/floor，缩放可能让它们在晚期成为主导，反而恶化解和采样协方差。
- 若改用 \(\operatorname{Cov}=\widetilde M\) 的错误噪声，晚期方差会灾难性放大，hole MSE 等指标
  预期显著恶化。

最后一点还可对单次 Block 1 条件分布定量化。若 \(P_h\) 是抽取 hole 像素的线性算子、真值固定，
则

\[
\mathbb E\|P_h(x-x_{\mathrm{gt}})\|^2
=\|P_h(\mu-x_{\mathrm{gt}})\|^2
+\operatorname{tr}(P_h\Sigma P_h^\top).
\]

错误噪声不改均值，却把 \(\Sigma\) 从 \(M^{-1}\) 变为 \(c^{-1}M^{-1}\)，故该条件 MSE 的方差项
也放大 \(c^{-1}=10^{16}\) 倍（只要 hole 上方差非零）。后续非线性迭代的最终 MSE 无法仅由题面
严格推出，但方向上没有理由期待这种错误构造改善质量。

因此，对正确重缩放不能预言 hole MSE 有系统性下降：理想情况下应完全不变；若换用新的独立但
分布正确的 \(\xi\)，单次 MSE 会有 Monte Carlo 波动，但其分布不变；实际 fp32 指标的变化只反映
数值误差、停止分支和后续轨迹敏感性，方向没有理论保证。若基线已经无溢出且可靠收敛，预期收益
主要是数值安全裕量，而不是采样质量本身。
