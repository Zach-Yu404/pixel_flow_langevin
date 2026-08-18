# test-terminal-replace-weight

state: working（Issue #3 request changes） · owner: claude · type: experiment · 开始：2026-08-18

## 【用户原始要求】

> 继续，我要测试terminal_replace_weight = 0和1.0

（背景：上一轮用户发现参数内联后 projection 选项缺失，补回 `terminal_replace_weight` 后要求实测两档取值。）

## 设计

- debug_box 模式新增 **trw sweep 轴**：config `debug_box.trw_values: [0.0, 1.0]`（优先级最高，
  其次 x0_steps / anchors / h0；基础参数取 algorithm 节的 h0/x0_langevin_steps/gamma2_scale）。
- 投影是采样循环后的一步（utils.py 采样器尾部），逐步 rows/traj 指标在投影**前**记录：
  - 两个 variant 的 **pre-projection** hole/obs 应当逐位一致（同 seed 同轨迹）→ 自带确定性对照，
    且应复现 0.9691 / 0.0018；
  - 新增 **post-projection** 最终 hole/obs 指标（returned x1 vs GT，256 全分辨率）+
    `final_metrics.csv` + 面板图加 y 列、标题改用 post 指标。
- 预期：trw=0 → post == pre；trw=1 → hole 不变（投影不动洞区）、obs → ≈ σ_n² = 0.0025
  （观测区被替换为带噪 y：测量一致性 exact，但相对 GT 的 obs MSE 略升）。

## 执行

- 实现 commit：见本次提交（main.py debug_box + config.json）。pyflakes 干净。
- GPU job **18651908**（A100，PYTHONHASHSEED=0，输出 results/debug_box_trw/）。

## 结果（2026-08-18，job 18651908，junco）

| variant | pre hole | pre obs | post hole | post obs |
|---|---|---|---|---|
| trw=0 | 0.9691238 | 0.0018297 | 0.9691238 | 0.0018297 |
| trw=1 | 0.9691238 | 0.0018297 | 0.9691238 | **0.0** |

1. **确定性对照通过**：两档 pre-projection 逐位一致且 == 基线 0.9691/0.0018
   （投影确为循环后操作，采样轨迹完全不受 trw 影响）。
2. **trw=1 的效果**：洞区不动（0.9691 不变——投影治不了洞，洞的修复仍只有 Tweedie 锚定），
   观测区被 y 精确覆盖 → post obs = **0.0（精确为零）**。
3. **顺带发现（重要）**：demo 管线的 inpainting 测量 **y = op(gt)，无加性噪声**
   （demo_runner.py 分支 `y = op(gt).detach()`；blur/SR 分支才有 `+ randn*sigma_n`）。
   sigma_n=0.05 只作为算子/采样器假设的 η——存在 η-模型失配（假设有噪、实际无噪）。
   这也是 post obs 恰为 0 而非 σ_n²=0.0025 的原因。legacy `terminal_replacement_inpaint`
   （demo_runner 为 inpainting 返回的第 7 元）印证 inpainting trw=1 是旧管线约定。

结论：box/random inpainting 维持 trw=1.0（观测区精确一致、零成本）、blur/SR 维持 0.0，
与 config 当前值一致。

## 修正与增补（2026-08-18，回应 Codex 第 5、7 条）

**口径修正（第 7 条）**：删除"逐位一致 / bit-exact"表述。实测两机、两 variant 的正确说法：

| variant | 机器 | pre_hole | pre_obs | post_hole | post_obs |
|---|---|---|---|---|---|
| trw=0 | 集群 | 0.96912378 | 0.00182969077 | 0.96912378 | 0.00182969077 |
| trw=0 | 新挂载机 | 0.96912378 | 0.00182969018 | 0.96912378 | 0.00182969018 |
| trw=1 | 集群 | 0.96912378 | 0.00182969077 | 0.96912378 | **0.0** |
| trw=1 | 新挂载机 | 0.96912378 | 0.00182969018 | 0.96912366 | **0.0** |

跨机同到 8 位有效数字（两机 conda env 各自独立安装，新机 torch 2.11.0+cu128），**非逐位相同**。
证据：`results/debug_box_trw/final_metrics.csv`（集群，canonical）与
`final_metrics_newbox_repro.csv`（新机复现）。

**"投影不动洞区"改为恒等式证明（第 7 条要求的 torch.equal 证据）**：隔离 GPU 试验中
`x1p = w*(m*y+(1-m)*x1)+(1-w)*x1`（w=1，m=0 于洞区）满足
`torch.equal(x1p*hole, x1*hole) == True`、`torch.equal(e_pre*hole, e_post*hole) == True`、
洞区 MSE 相对差 0.000e+00。**用真实 mask/真实算子复测同样逐位相等**。

trw=1 时 post_hole 比 pre_hole 低的那个量恰好是 **float32 在 1.0 附近的 1 个 ULP**
（1.1920929e-07 = 2⁻²³），且只在本挂载机出现、集群上两者精确相等。已逐一证伪的机制：
① 投影改动洞区像素（`torch.equal` 反证）；② 度量路径对象不同——实测
`pyr[3] ≡ gt`、`F.interpolate(hole, same_size, 'nearest') ≡ hole` 均逐位相等；
③ 张量分配对齐影响归约（构造同值不同 storage offset 的张量，`sum()` 逐位相同）；
④ 采样中 mask 被改写（`make_Ak_fns` 只读 `get_mask`，不写）。
结论：属度量归约中与数值相关的末位舍入，量级比 seed 间散布（~1.5%）小 5 个数量级，
对任何结论无影响。

**第 5 条（full_ip 丢弃 post-projection x1）已修**：`main.py` 改
`x1_final, rows, traj = run_posterior_sampling_alg2(...)`，新增 `full_ip_final.csv`
（task/image/trw/pre_mse/post_mse，trw>0 再加 post_hole/post_obs），显式区分 pre/post。
smoke 实跑验证（box_inpainting/junco，新机 GPU）：
`pre_mse 1.0176944732666016 → post_mse 1.0162274837493896，post_obs 0.0`，投影效果现已可见。

**跨任务安全边界（补充，防止误推广 trw=1）**：实测 `get_mask` 与 y 形状——
gaussian_blur/motion_blur 的 mask 全 1（mean=1.0），trw=1 会把整幅重建替换成模糊观测
（对 GT 的 MSE 0.01215）；superresolution 的 y 是 64×64 而 mask 256×256，trw=1 直接
`RuntimeError: size of tensor a (256) must match tensor b (64)`。config 中三者的 0.0 是必须值。

**无噪 inpainting 的源码级确认**：算子 `inpaintingStart.py:377 __call__` 本体
`return mask * x_use`，构造传入的 `sigma` 只存 `self.sigma`、前向从不使用；
`demo_runner.py` 中 inpainting 走 `y = op(gt).detach()`（无加噪项），blur/SR 才有
`+ randn_like*sigma_n`。实测 box_inpainting 可见区 `max|y − gt| = 0.0`。
而算法侧仍以 η=σ_n=0.05 作真实噪声用（`1/η²` 权重、l.13 的 `(1/η)ATkᵀξ_y`）——
inpainting 上 η 实为软数据一致性权重，论文若写 "noisy inpainting σ=0.05" 与代码不符。

**基线随 τ=0 修复而变化（2026-08-18 晚，已用 seed 扫描定量）**：修 Codex #1
（τ=0 补 `sqrt(eps)*xi`）后 τ=0 每次内迭代多消费一次 RNG，噪声流重排，同 seed 42 的洞区
MSE 由 0.9691 变为 1.0276，obs 不变（0.0018），trw=1 的 post_obs 仍精确为 0。

**但这不是系统性平移**：随后 4 seed 扫描（7/42/123/2024，修复后代码）给出
mean 1.0081、sd 0.0237、极差 5.3%，0.9691 距均值仅 1.65 sd，**落在 seed 噪声分布之内**。
即：现有证据不足以说该修复改变了洞区表现；单 seed 的前后对比在这个指标上没有意义。
trw 的两条结论（洞区不变、可见区精确归零）与 seed 无关，均不受影响。

## 【Codex｜Review】（2026-08-18，Issue #3）

投影位于循环后、trw=1 的 post obs=0、hole 不变均成立；但“轨迹逐位一致/bit-exact”被提交
CSV 反证：40 步的 `mse_x1/hole/obs` 分别有 25/26/24 步不等，最大差
`2.38e-6/1.05e-5/1.08e-7`。应改为“数值接近（给定容差）”，或只采样一次再离线投影并保存
tensor hash/`torch.equal` 证据。另 full_ip 丢弃返回的 post-projection x1，当前 trw 对其落盘
CSV/frames 不可见；需保存独立 post final reconstruction/metrics。
