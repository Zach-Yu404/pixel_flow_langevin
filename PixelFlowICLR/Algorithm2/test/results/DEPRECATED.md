# 已废弃 / 已证伪的实验与方案

记录在此，代码从 `Algorithm2/` 移除。历史数据与推导保留在 git 历史与
`.research/tasks/debug-box-alg2-hole.md`。

判据统一：洞区 MSE 的 **seed 地板约 5%**（n=4，sd 0.0237），效应量小于它的结论一律不成立。

---

## 1. Tweedie anchor（`anchor`）

**是什么**：在 Block 1 的 b̃_τ 上加 `λ·x̂₁_model + √λ·ξ`、M 上加 `λI`，
把模型的 Tweedie 估计当作 ker(A) 上的高斯先验注入。

**曾经的结果**：λ=25 时 7/7 图修复洞区（池化 1.097→0.178，6.2×），零额外 NFE。

**为何废弃**：

- 用户明确停用（改成注释），且它**改变了后验目标**——是加了一个启发式先验项，
  不是同一个分布的更好采样；新的约束是"不新增可调超参、不加启发式 anchor、不改后验目标"。
- λ 是一个必须调的超参（150–350 在论文 Alg-1 里也是 tuned）。
- GT 诊断给出了更根本的替代路径：Block 1 本来就有能力填洞（末期 τ 达 0.0044），
  问题在 Block-2 mixing，**不需要外挂先验**。

**代码处理**：删除 `anchor` 参数、注释掉的 b̃ 分支、fail-fast 守卫、config 键、
debug_box 的 `anchors` 扫描轴。

## 2. x₀-Langevin 多步（`x0_langevin_steps` > 1 / `x0_langevin_recompute`）

**是什么**：把 l.16-17 重复 K 次。原实现在 K 步内复用第一子步的 `x̂₀`（frozen score）；
后来补了 `x0_langevin_recompute=True` 的真多步版本（每子步多一次 NFE）。

**结果**：K=3/5 单调恶化（0.97→2.13/2.49，远超 5% 地板）。真多步版本从未实测。

**为何废弃**：

- 论文 Algorithm 2 的 l.16-17 就是**一步**，K>1 不是论文写法；
- frozen-score 的 K>1 在数学上不是"对当前 x₀ 的多步 Langevin"，其结论只能算消融；
- 真正的 mixing 问题由 §2 的指数积分器解决，且**零额外 NFE**，比 K× NFE 的路线更优。

**代码处理**：删除两个参数与 K 循环，恢复为论文的单步；删除 debug_box 的 `x0_steps` 轴。

## 3. γ² 缩放（`gamma2_scale`）

**是什么**：给实测 γ²(k,τ) 表整体乘一个系数（0 ⇒ 关闭 Cor. 8 的网络误差项）。

**结果**：γ²×1 与 ×0 的洞区差异约 1%（0.9691 vs 0.9782）——**低于 5% 的 seed 地板，不可分辨**。
verify 的 V8 白噪声探针也显示最优 γ² 在测试区间内几乎平坦。

**为何废弃**：它是论文之外的缩放旋钮；γ² 本身（Cor. 8 的实测表）保留。

**代码处理**：删除 `gamma2_scale` 参数与 config 键、debug_box 的 `gamma2_scales` 轴；
`gamma2` 直接取实测表。

## 4. h₀ 扫描

**结果**：h₀ 0.1→0.5→1.0 单调恶化（0.97→…→3.29）。

**为何不再作为"旋钮扫描"保留**：原因已定位为 **ULA 的稳态方差偏差 1/(1−h₀/4)**
（h₀=1 时 +33%，h₀→4 发散），而非 h₀ 本身"不好"。换成指数积分器后该偏差消失，
h₀ 的作用完全改变，旧扫描结论**不可迁移**。

**代码处理**：debug_box 的 `h0` 扫描轴删除；`h0` 仍是论文参数，保留在 config。

## 5. 终端投影（`terminal_replace_weight`）

**是什么**：采样结束后把可见区替换成 y（inpainting 旧管线约定 =1.0）。

**结果**：加噪对齐 baseline 后，投影会把测量噪声原样抄进可见区——
可见区 MSE 由采样器达到的 0.0018 抬到 **σ²=0.0025**（三图两任务实测 0.00246–0.00252）。
且它对洞区是恒等式（`torch.equal` 证明逐位不变）。

**处理**：用户裁决 **trw=0**，config 里两个 inpainting 已改 0.0。
参数保留（blur/SR 本来就是 0，且它是有据可查的管线约定，不是死代码），但不再使用。

## 6. one-step 估计器实验（`onestep` 模式的 WLS/Model 对比部分）

**结果**（实验 1，已完成并归档）：Model（`direct_estimate_x1`）一致优于 WLS，
stage 0 差约 50×，stage 3 打平；5 个任务曲线完全一致（one-step 不经过 operator）。

**现状**：WLS 分支早已删除；`onestep` 模式现在只剩 **S1/S2/adjoint 自检**，
这部分是 Algorithm 2 的接线验证，**保留**。

---

## 保留的非论文项（有理由，不删）

| 项 | 理由 |
|---|---|
| `g_bypass_stage3` | PixelFlow 的 stage-3 语义（G=I），属实现契约而非旋钮 |
| `sigma_min` 跳过 | σ_τ→0 时 1/σ² 溢出的数值保护 |
| `cg_tol` / `cg_max_iter` / `cg_max_iter_l14` | 论文 Algorithm 2 的 `Require:` 明列 CG iterations L |
| `ridge_rel` | Prop. 4 的 ε，论文明列于 `Require:` |
| `measurement_mode` / `operator.center` / `measurement_seed` | 与 baseline 对齐的测量设定，非算法参数 |
| `block2_integrator` / `block1_final_draw` | 离散化选择（非可调超参），保留以便复现论文原样 |
