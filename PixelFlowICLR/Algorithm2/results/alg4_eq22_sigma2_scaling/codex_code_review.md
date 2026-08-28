日期：2026-08-26｜模型：OpenAI Codex（GPT-5）

# Eq. (22) `sigma_tau^2` 缩放开关：独立实现审查

## 审查对象与总裁决

- 源码：`PixelFlowICLR/Algorithm2/utils.py`，本次审查字节对应 Git blob
  `47c57d1f2cc921ae16dc13967445048408a6cb43`；工作区基准 HEAD 为
  `1e9efb963503617971c5147df4fa601f65f19c16`。实现尚未绑定独立 commit，故以 blob
  标识精确版本。
- 验证记录：`results/alg4_eq22_sigma2_scaling/validation.md`，Git blob
  `18ce3565b9d99d7ac46e310ea68d95261a3d1f0c`。
- 方法：直接逐行审查当前源码，静态追踪 RNG，并独立推导 PCG 缩放律；另用一维 fp32
  反例复核异常分支。理论共识文件只作为预期规格，不作为实现正确性的替代证据。

总裁决：当前已验证的 `pooled_junco` 与 `spectral` 主路径中，算子、RHS、随机流、谱探针、
预条件器及默认关闭路径均实现正确；没有发现会改变这些主路径 Gaussian 目标的漏乘或多乘。
但存在一个真实的 **should-fix**：若 Jacobi/spectral 探针 guard 失败并退回
`M_inv=None`，统一的 `clamp_floor=c*1e-12` 不再遵守未预条件 CG 的缩放律，可能严重改变
scaled 解。现有验证没有覆盖该分支。此外，验证资产不足以证明“全面 Monte Carlo 等价”或
“baseline 逐位一致”。

## 发现汇总

| 严重度 | 发现 | 影响 |
|---|---|---|
| **should-fix** | 预条件器 guard 失败后仍以 `M_inv=None` 配合统一 `c*1e-12` | 未预条件时 `rz` 与 `p^T M p` 分别按 `c^2`、`c^3` 缩放；绝对 clamp 可主导 PCG，破坏路径等价 |
| **should-fix（验证）** | 无可复跑脚本、原始 full-precision 产物或 tensor hash；未测 fallback、f39/极小 sigma、谱臂 fp64 RHS/算子、条件均值/协方差 | 当前记录是很好的 smoke test，但不是完整的 bitwise/Monte Carlo 证明 |
| **nit** | scaled 分支仍先调用旧 `make_M_tau_den`，在标量层计算 `1/sigma_tau^2` | 不会产生大张量，当前数值收益仍在；但“完全不形成该中间量”的字面承诺不成立，极端 sigma 仍受旧构造限制 |
| **nit / 组合启用前应修** | `c*S^{-1}` 当前按 `c * apply_S_inv(x)` 求值 | 对当前有 floor 的固定 S 安全；`sigma_scaled` S 或极小谱功率时，`S^{-1}` 的大中间张量仍可能先形成 |

## 1. 算子与 RHS 的逐项系数

令 `sigma=sigma_tau>0`、`c=sigma^2`。`H_tau` 在本仓库中自伴，源码头部
`utils.py:L16-L20` 记录并由 dense float64 guard 验证，因此两次
`apply_H_tau` 对应 `H_tau^T H_tau`。

### 1.1 算子

`make_M_tau_den_sigma2`（`L1115-L1138`）的实现为：

| 项 | 应有系数 | 源码 | 结论 |
|---|---:|---|---|
| `A^T A v` | `c/eta^2` | `sc_e2=c/eta^2`（`L1129-L1133`） | 正确 |
| `H^T H v` | `1` | 两次 `apply_H_tau`，无额外系数（`L1134-L1135`） | 正确 |
| `S^{-1}v` | `c` | `c * s_op.apply_S_inv(x)`（`L1136`） | 正确 |

所以该闭包确实实现

\[
\widetilde Mv=(c/\eta^2)A^TAv+H^THv+cS^{-1}v=cMv.
\]

新函数自身没有计算显式 `1/sigma^2`，也没有先求未缩放的 `M(v)` 再事后乘 `c`。

### 1.2 确定与随机 RHS

scaled 分支 `L1478-L1493` 逐项为：

| 项 | 应有系数 | 源码系数 | 结论 |
|---|---:|---:|---|
| `A^T y` | `c/eta^2` | `_sc_e2`（`L1485`） | 正确 |
| `H^T H x1_hat` | `1` | 两次 `apply_H_tau`（`L1486-L1488`） | 正确 |
| `S^{-1}x1_hat` | `c` | `_c22`（`L1489`） | 正确 |
| `A^T xi_y` | `c/eta` | `_c22/eta`（`L1490`） | 正确 |
| `H^T xi_h` | `sigma` | `_sig`（`L1491-L1492`） | 正确；即 `c/sigma=sigma` |
| `S^{-1/2}xi_s` | `c` | `_c22`（`L1493`） | 正确；没有误写成 `sigma` |

因此源码实现的是同一组噪声 realization 下的

\[
\widetilde b+\widetilde\zeta=c(b+\zeta),
\]

而不是为 `M_tilde` 重建 `Cov(zeta)=M_tilde` 的错误噪声。未发现漏乘或多乘。

## 2. seed、三组 `xi` 与 `xi_0`

结论：**随机流完全对齐**。

- 唯一 CPU generator 在 `L1323` 以相同 `seed` 创建；`randn_like_cpu` 在
  `L1325-L1326` 未变。
- 每阶段的初始 `x0` 在 `L1361` 抽取。
- 每个 inner iteration 都在缩放分支之前，无条件按固定顺序抽取
  `xi_y(shape=y)`、`xi_h(shape=x1)`、`xi_s(shape=x1)`（`L1464-L1466`）。
- `eq22_sigma2_scale` 分支自身不调用 RNG；PCG 也不调用 RNG。迭代数不同不会改变 generator
  推进量。
- `xi_0(shape=x0)` 仍只在 Block 1 solve 后抽一次（`L1507`），位置、次数和形状不变。
- `diag_noise_off` 的组合也是先抽再置零（`L1467-L1473`、`L1508-L1510`），故不会让任一
  arm 少消费随机数。

需要区分“随机张量相同”和“状态相同”：`xi_0` 本身相同，但此前 fp32/PCG 造成的 `x1`
微差会进入 `x_tau=H x1+sigma xi_0`，再经 `L1523` 的相减和除以 `sigma` 传播。后续轨迹可以
不同，但这不是 seed 或 RNG stream 错位。

## 3. spectral `SOperator` 与预条件器

### 3.1 正常构造成功的路径：通过

谱探针的 scaled 分支（`L1430-L1433`）是

\[
\widetilde d=\widetilde M\mathbf1-cS^{-1}\mathbf1
  +c\,\operatorname{mean}(1/P)\mathbf1=c d.
\]

三项全部带 `c`，没有只缩第一项。guard 使用 `_floor22`（`L1437`），非谱 Jacobi 也显式传
`floor=_floor22`（`L1443-L1444`）；`_floor22` 在开关开/关时分别为
`c*1e-12` 与 `1e-12`（`L1403-L1413`）。构造成功时
`K_tilde=K/c`，从而 `r_tilde=cr`、`z_tilde=z`、`p_tilde=p`，`rz` 和
`p^T M p` 都按 `c` 缩放；此时三处统一使用 `c*1e-12` 正确。

### 3.2 **should-fix：`M_inv=None` fallback 的缩放律错误**

谱 guard 在 `L1437-L1441`、Jacobi helper 在 `L1591-L1595` 都明确允许回退
`M_inv=None`。但无预条件器时，scaled 迭代满足

\[
\widetilde r=cr,\quad \widetilde z=cr,\quad \widetilde p=cp,
\]

因而

\[
\widetilde{r^Tz}=c^2r^Tz,\qquad
\widetilde{p^T\widetilde M p}=c^3p^TMp.
\]

此时 `b_norm` floor 应按 `c` 缩放，但 `rz` 与 `p^T M p` 若要复现旧绝对门槛，应分别按
`c^2`、`c^3` 缩放，不能继续共用 `c*1e-12`。

独立 fp32 一维反例：取 `M=1`、`b=1e-3`、`x0=0`、`c=1e-4`、只跑一步。

- baseline：`x=1.000000047e-3`，相对残差 `0`；
- 当前 scaled + `M_inv=None`：`x=9.999999747e-6`，相对残差 `0.99`；
- scaled fallback 改用 `K_tilde(r)=r/c`：结果与 baseline 逐位相同。

推荐修复：scaled 的 guard-failure 分支不要返回 `None`，而使用
`lambda r: r/_c22`，即把 baseline 的恒等预条件器按自然规律缩成 `I/c`；或者为 PCG 的
norm、`rz`、`pAp` 分设尺度与 breakdown 判据。并增加强制触发两类 guard 的单测。

现有 A/C 记录未显示该 fallback 被触发，迭代行为也与正常预条件器路径相符；但验证没有显式
记录 guard 结果。因此此项不推翻已报主路径结果，却仍是 API 支持范围内的真实正确性缺口。

## 4. 默认 `False` 的 baseline 路径

源码层面结论：**通过，数值操作保持原路径**。

- 新 kwarg 默认 `False`（`L1192`）。
- 关闭时 `M_blk1=M_den` 是直接别名，不是重新包一层算子；`_floor22=1e-12`
  （`L1412-L1413`）。
- spectral 关闭分支仍逐字使用
  `M_den(1)-S^{-1}(1)+mean(1/P)*1`（`L1434-L1436`），guard 读到的
  `_floor22` 就是旧值 `1e-12`（`L1437`）。
- Jacobi 接收同一个 `M_den` 对象的别名，并显式得到同值 `floor=1e-12`
  （`L1443-L1444`）。
- `b_tilde` 的 else 分支（`L1494-L1499`）保留原项、原括号和原求值顺序：
  `inv_e2*A^Ty + Cinv(x1_hat) + A^Txi_y/eta + H^Txi_h/sigma + S^{-1/2}xi_s`。
- `pcg_solve` 新参数默认仍为 `1e-12`（`L1598-L1599`），三处替换恰是
  `||b||`、`p^TAp`、`rz`（`L1623`、`L1628`、`L1636`）；关闭路径显式传同一数值，其他调用者
  不传时也沿用旧值。

因此默认关闭时未发现会改变浮点运算或 RNG 的代码差异。Phase A 的历史指标复现与此一致。
但现存档案只保存四位小数，不能仅凭档案把“逐位一致”升级为实验事实；严格 bitwise 声明仍应
保存旧/新 final tensor 的 `torch.equal` 结果或内容 hash。

## 5. 对现有验证的充分性判断

### 5.1 已足以支持的结论

- Phase A 两种 S、4 seeds 复现历史 hole 均值/标准差与 spread，足以作为默认关闭路径的端到端
  指标回归。
- Phase B 在真实 frame 算子上得到 pooled `relM64=1.1e-16~1.5e-16`，并在两臂得到
  fp32 `relM/relRHS` 约 `0.6e-7~2.6e-7`；这强力支持逐项代数恒等，没有数量级错误。
- 8 个检查点的 baseline/scaled PCG 迭代数成对一致，且 pooled 的单步 `relX` 至多
  `4.56e-7`；对主路径的数值齐次性是正面证据。
- Phase C/D 的 paired 4-seed hole 指标四位一致，且 pooled 最终
  `max|Delta x|=1.137e-5`，足以排除错误噪声协方差、漏乘 `c` 等灾难性实现错误。
- f38 的 `max|Mv|=4.18e3 -> max|Mtilde v|=8.38` 直接展示了绝对动态范围缩小。

### 5.2 仍存在的缺口（**should-fix：验证资产/覆盖**）

1. 目录中只有 25 行摘要，没有可复跑脚本、命令、环境/设备版本、误差范数定义、完整精度
   CSV、原始 tensor 或 hash。结果不能由第三方从该目录独立复算。
2. Phase A 与 `program_results.csv` 都只保存四位聚合值；这证明档案指标复现，不证明
   tensor bitwise equality。
3. `relM64` 只出现在 pooled 行；spectral 没有记录 fp64 算子恒等，两个 arm 都没有记录
   fp64 RHS/直接解恒等。记录标题是“real frame operators”，也没有持久化所谓 dense matrix
   的构造与结果。
4. 4 seeds 的最终 MSE 是 paired smoke test，不是条件 Gaussian 的 Monte Carlo 均值/协方差
   检验。若要以实验声称 covariance 不变，需要小维 direct solve 的解析均值/协方差对照，或
   足够多 draws 及置信区间。
5. 局部 Phase B 最小只测到 f38 的 `sigma=4.48e-2`；实际轨迹 f39 仍运行且
   `sigma=4.0e-4`，配置 `sigma_min=1e-8` 的边界更未触及。最能检验 floor/下溢和动态范围的
   区域没有局部算子/PCG 证据。
6. 没有覆盖 stage 1、guard-failure fallback、`diag_noise_off` 各组合、`sigma_scaled` S、
   batch>1，以及 Algorithm 4 支持的更多 measurement operator。
7. spectral f9 的记录实际是 `relX=1.28e-4`、`it_b/s=600/600`，显著高于其余检查点；
   `validation.md` 又未记录该 probe 的 tol/cap 定义。它是 solver 敏感压力点，不能概括成
   “所有单步解差都在约 1e-6”。

建议最低补充集：提交可复跑测试；保存 baseline tensor hash；加入 f39、`sigma=1e-8`、
`sigma_min` 上下边界与强制 fallback；对 scalar/spectral S 做小维 fp64 direct solve 和大量 draw
的均值/协方差核对；再扩到至少一个非 inpainting 算子和 batch>1。

## 6. spectral `max|Delta x|=2.0e-3` 的归因

判断：**可合理归因于 fp32 求值重排、有限精度 PCG/停止截断及后续非线性轨迹放大；不能归因于
算法目标改变，也不应归因于 RNG stream 改变。** 目前证据对该判断很强，但尚未把
“算子舍入”“solver 截断”“GPU 非确定性”三者定量拆开。

依据如下：

- 源码已证明 `M_tilde=cM`、`q_tilde=cq` 的精确算术目标不变；同一 `xi` 下精确解逐样本相同。
- 三组 Block-1 `xi` 与 `xi_0` 的抽样顺序/次数/形状完全一致，排除了随机流漂移。
- 实测算子/RHS 的 fp32 差正处于约 `1e-7` 的舍入尺度。
- spectral PCG 明显更长：端到端最大迭代数 172，而 pooled 为 19；spectral f9 的单步
  `relX=1.28e-4`，pooled 最大约 `4.56e-7`。最终差异分别为 `2.026e-3` 与
  `1.137e-5`，与 solver 敏感度高度相关。
- 逐项重构改变了浮点乘法/加法顺序；即使迭代次数相同，迭代向量也无需逐位相同。微差随后进入
  velocity 网络及 40 帧反馈，`L1523` 的相减再除以小 `sigma` 也可放大坐标误差。
- 四个 paired seed 的 hole MSE 均保持到四位，未呈现与错误 target 相符的系统偏差。

最有判别力的补充实验是：在同一捕获 frame、同一 `x1_hat` 和同一组 `xi` 上固定 RHS，同时以
fp64/direct solve 为共同参考；扫描 PCG tol/max_iter，并记录每帧共同坐标下的真实残差和首个
分歧点。若 tol 收紧后两臂 `Delta x` 系统下降，即可直接坐实 solver 截断归因。另应做同 arm、
同 seed 重复运行以测 GPU 非确定性底噪，并保存四组 `xi` 的 hash，作为随机流对齐的运行时证据。

## 7. 边界情形与其他开关

- **旧 builder 仍被调用（nit）**：`L1401-L1402` 在开关开启时仍先调用
  `make_M_tau_den`，后者在 `L1101` 计算 Python 标量 `1/sigma^2`。scaled 分支不求值
  `M_den/Cinv`，所以不会形成 `1/sigma^2` 级大张量，当前动态范围收益没有丢失；但可把两个
  builder 移入 if/else，以兑现“完全不形成 reciprocal”的字面契约并移除无谓失败点。
- **`sigma_min` 与下溢**：标准配置传入 `sigma_min=1e-8`，此时
  `c*1e-12=1e-28` 仍是正常可表示的 fp32 数；当前轨迹最小正 sigma 是约 `4e-4`，更安全。
  若允许 `sigma` 低于约 `1.1e-13`，floor 会进入 fp32 次正规区；低于约 `3.7e-17`
  时 floor 可下溢为零；低于约 `3.7e-23` 时 `c` 本身在 fp32 张量运算中可下溢为零。若
  `sigma_min<=0` 并让
  `sigma=0` 进入分支，旧 builder 会除零，`L1523` 也会除零。应显式约束支持域为正 sigma，
  或改用 dtype-aware guard/更高精度 reduction。
- **`sigma_scaled` S / 极端谱（组合启用前应修）**：当前写法
  `c * s_op.apply_S_inv(x)` 与 `c * s_op.apply_S_inv_sqrt(xi)` 会先执行 `S^{-1}`。
  当 `s2` 自身正比于 `sigma^2`，或谱功率极小时，大中间量可能先形成再被 `c` 缩小。
  可为 scaled SOperator 提供融合操作：标量用 `(c/s2)*x`，谱算子在频域用 `(c/P)*X`；
  sqrt 项同理。现有 spectral 数据有相对 floor、当前两臂未见问题。
- **`diag_noise_off`**：抽样后置零，和缩放开关组合时 RNG 对齐且每个保留项仍满足同一 `c`
  缩放；静态审查通过。但它本身按设计改变条件协方差，不能用其结果证明生产 Gaussian target，
  且目前没有组合回归测试。
- **标准入口可用性（范围外观察）**：kwarg API 已存在，但 `main4.py` 的严格
  `sampler_kw` schema 未暴露 `eq22_sigma2_scale`；当前只能由直接调用者传入。用户明确限定本次
  实现在 `utils.py`，故不把它判为本次代码缺陷；若要从标准 JSON/CLI 启用，需另行接线。
- **既有 PCG breakdown 语义**：`.clamp(min=...)` 会把负的 `p^TAp`/`rz` 静默改成正数，
  也没有显式区分 NaN/Inf。这是缩放前即存在的问题，不由本补丁引入；长期更稳健的方案仍是
  相对 breakdown 判据和显式非正/非有限检查。

## 最终结论

对于现有已验证配置，Eq. (22) 缩放实现保持原 Gaussian draw 的精确算术目标，RHS 噪声、RNG
与预条件器主路径都正确；默认关闭路径也保持 baseline 操作。它只能称为 numerical rescaling，
证据支持其降低绝对中间量，但不支持改善条件数或解决 frozen-center。

在把开关推广到所有算子、设为默认值或运行正式结论性实验前，应先修复 `M_inv=None` fallback，
并补齐可复现验证、极小 sigma/guard 边界与真正的条件均值/协方差测试。现有 spectral
`max|Delta x|=2.0e-3` 没有显示目标改变，最可信归因是浮点/solver 截断后的轨迹放大。

无 blocker

## 共同解释:Codex 最终意见(2026-08-27)

**总裁决**：修复后实现在当前 `pooled_junco`/`spectral` 范围正确，无残留 blocker；
它应保留为 opt-in numerical rescaling，暂不改默认。

1. **Gaussian draw：精确算术下严格保持，fp32 实现不保证逐位一致。** 对 `sigma>0`、
同一组 `xi_y/xi_h/xi_s` 且精确求解，`Mt=cM`、`qt=cq` 使每个 realization 的解完全相同；
换独立 `xi` 则只是同分布。[DN] 给出 `5.17e-17` 的稠密 fp64 恒等误差、4000 个直接解
`max|dx|=9.16e-15` 及与 MC 误差相容的协方差检验。有限容差 PCG/fp32 仍只是同一目标的数值近似，
64 维单测也是 `relX=1.10e-7, bitwise=False`。

2. **float32：真正改善的是绝对动态范围/溢出安全裕量，未证明当前轨迹精度或收敛改善。**
[G] 的实际末帧 `sigma=4e-4` 将 `max|Mv|` 从 `2.86e7` 降到 `4.57`；[S8] `sigma=1e-8`
将 `4.57e16` 降到 `4.57`。但两臂均 `nonfinite=0`，[S8] 仍是 `it=2/2`、
`res=8.0e-11/7.6e-11`、`relX=6.24e-8`；条件数和相对吸收也不变。

3. **CG 迭代/残差的近乎成对相等来自缩放齐次性，不是条件数改善。** `Ktilde=K/c`、
同比缩放 floor 及修复后的 fallback `I/c` 使精确 PCG 的方向、步长和相对残差不变。
[G] spectral f15 的 `544/545` 和 [T] 的 `107/108` 这类 ±1 差异，来自约 `1e-7`
的算子/RHS 舍入、reduction/递推误差及 tol 边界；两轮数据不支持 CG 加速。

4. **trajectory：现有证据不表明超过合理数值误差。** spectral 的两轮 `max|dx|`
为 `2.026e-3` 和 `2.172e-4`，hole 指标四位一致；长 PCG 的舍入/截断差可经 40 帧
非线性反馈放大。再结合已报告的 baseline 同 seed 跨运行 hash 不同且 GPU 非确定性
底噪与缩放差同量级，这些差异不能归因于 Gaussian target 改变；hash 只证明非 bitwise。

5. **默认值：不建议现在改为 `True`。** 当前 `sigma_min=1e-8` 只是阈值，实际最小正
`sigma=4e-4`；baseline 从未溢出、4-seed 的 `cg_bad=0`且指标无改善。应在真正触及更小
`sigma`、更大 reduction 或出现 nonfinite 时作为防御性开关，而非改变当前默认复现口径。

6. **定性：只能称 numerical rescaling/代数等价方程归一化。** 它不改变精确 transition kernel、
条件中心或条件数，因而不能称为解决 stage 2/3 frozen-center；当前也没有“修掉数值伪冻结”的证据。

### 销项与保留

- **销项—唯一代码 should-fix**：guard-failure 已统一 fallback 到 `Ktilde=I/c`；[F] 1D 与 baseline
  逐位相等，64 维 SPD 的迭代/残差成对一致。
- **销项—两项边界/nit**：旧 builder 已移入 `else`，scaled 分支不再形成标量 `1/sigma^2`；
  `sigma>0` guard 实现正确（但归档脚本未单独断言非正 sigma 必抛异常）。
- **销项—验证 should-fix 的当前任务范围**：脚本、修复后回归、f15/f39/S8、spectral fp64
  算子、pooled 稠密直接解+MC、tol、blur、diag 组合和 final hash 均已归档，足以支持上述裁决。
- **保留为非阻塞适用范围限制**：尚无 batch>1、假想 `sigma_scaled S`、spectral fp64 RHS/MC
  的完整覆盖；`c*apply_S_inv(...)` 的未融合中间量 nit 仍在。blur 的 A 臂会回退 fp32，
  故 [BL] 不是纯 fp64 证明。这些不推翻当前两主臂结论，但也不支持扩展为全配置默认。

**与 Claude 的分歧：无实质分歧。** 我只明确收紧共同结论的边界：第 2 点不能从“安全裕量增加”
升格为“当前配置已实证精度/收敛改善”，第 5 点也不支持现在改默认。
