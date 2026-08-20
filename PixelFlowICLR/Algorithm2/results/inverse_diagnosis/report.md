# main.py (Alg 2) vs main2.py (Alg 3) — 逆问题诊断

box_inpainting / junco / seed 42 / config.json 现值。两个采样器共用同一 GT、measurement、
mask、operator、seed、stage 与 time grid、metric —— 唯一变量是算法本身。

## Tested

1. `main.py` 的 `run_posterior_sampling_alg2` —— 全轨迹 trace（40 步）
2. `main2.py` 的 `run_posterior_sampling_alg3` —— 全轨迹 trace（36 步；τ=0 在 stage 0–2 被跳过）
3. 一个因果测试：`main2` 关闭 l.15/l.19 的 x₀ 重算（`recompute_x0=False`），其余完全不变

trace 全部由既有的 `record_trajectory=True` 产出，`x₀` 与 `x̂₁` 离线导出
（`x₀=(x_τ−H_τx₁)/σ_τ`，`x̂₁=H_τ⁻¹(x_τ−σ_τx̂₀)`）——**未改动任何正式代码**。

## Key results

| | main (Alg 2) | main2 (Alg 3) | main2 无 x₀ 重算 |
|---|---|---|---|
| 首次明显退化 | 无（单调改善） | **stage 0, τ=0.444** | stage 3, τ=0.222 |
| stage 0 第 1 步洞区 | 31.58 | **0.098** | 0.101 |
| stage 2 末洞区 | 1.993 | 1.3e8（发散中） | **0.911** |
| 最终洞区 | **1.047** | NaN | 3.250 |
| 最终可见区 | 0.00362 | NaN | 0.00623 |
| `‖x₀‖/√n`（目标 1） | 0.69–0.93 | 0.77 → **1051 → ∞** | 0.69–0.80 |

**最重要的一个数**：stage 0 第 1 步，main2 的洞区是 **0.098**，main 是 **31.58**——
相差 **322 倍**。用 (51) 的 x̂₁ 重建状态确实能把洞填好，问题出在之后。

**x̂₁ 与抽样 x₁ 的对比（main）**：x̂₁ 全程优于抽样结果，stage 2 处达 **20×**
（4.52 vs 91.31），但优势随 τ 单调衰减，**最后一步归零**（0.2644 vs 0.2644）。
stage 3 内同分辨率下，τ=0.111 的 x̂₁（0.246）已优于最终输出（0.264）。

## Analysis

**观察 1**：main 的 `‖x₀‖/√n` 全程 0.69–0.93，从未到 1。
→ 排除数值发散（x₁ 范数正常）。
→ 支持 Block-2 混合不足：x₀ 是先验进入洞区的唯一通道，而它没被推到目标尺度。

**观察 2**：main 的洞区在每个 stage 内单调下降但收敛到 ~1.0 即停；
GT∈[−1,1] 时洞区 MSE 1.05 ⇒ 洞里是 RMS≈1.02 的随机量，即"单位方差噪声、零图像结构"。
→ 排除"起点太差救不回"（ridge 扫描已证：ε=1e-2 起点 0.34 反被推高到 0.89）。
→ 支持"链条主动收敛到噪声水平"。

**观察 3**：main2 用 x̂₁ 重建后第 1 步洞区即达 0.098。
→ **排除 endpoint/score 有问题**：模型的去噪估计足够好，注入它立刻见效。
→ **排除 Block-1 无能力**：拿到含先验的 x_τ，精确抽样能填洞。

**观察 4**：main2 的 `‖x₀‖/√n` 从 0.77 起呈几何增长（0.85→1.49→5.0→16.3→55→217→1051），
洞区同步崩坏；关掉 x₀ 重算后 `‖x₀‖/√n` 立刻回到 0.69–0.80 且不再发散。
→ **确证 l.15/l.19（`x₀ ← (x_τ−H_τx₁)/σ_τ`）是发散源**：x_τ 被 x̂₁ 重建，
抽样所得 x₁ ≠ x̂₁，其差被除以 σ_τ 灌进 x₀ 形成正反馈。

**观察 5**：因果臂在 stage 2 末达 **0.911**（优于 main 的最终 1.047），
随后 stage 3 单调恶化到 3.250，同时可见区从 0.00496 升到 0.00623。
→ stage 3 时 G=I、ker(G) 为空，重建把**整幅** x₁ 换成 x̂₁（含可见区），
数据拟合被抹掉。这是 stage-transition 之后的**语义问题**，不是数值问题。

## Final conclusion

**Main bottleneck**（main.py）：不是 Block-1、不是 score、不是 τ=0 的 ridge，
而是**先验无法有效进入 x_τ**。x₀ 是唯一通道且其尺度停在 0.7–0.9（目标 1），
洞区因此收敛到噪声水平 ~1.0。

**main.py vs main2.py 的差异**：main2 在 l.11 后用 (51) 把 x̂₁ 注入状态，
这**在第一步就把洞区做到 0.098（322× 优于 main）**；但它保留了 l.19 的 x₀ 重算，
后者把 x̂₁ 与抽样 x₁ 的差除以 σ_τ 反馈进 x₀，导致几何发散。
**两者的差距从 stage 0 第 1 步就已出现，方向与最终结果相反。**

**MUST FIX**：main2 的 x₀ 重算与 (51) 重建不相容——二者同时存在必然发散。
若保留重建，必须去掉重算（或反之）。

**KEEP AS IS**：main.py 的 Block-1 精确抽样、score solve、closed-form inverse、
τ=0 的 Prop.-4 ridge。这些都已被证明正确或非瓶颈。

**No need to change**：G 与 closed-form 代数（已验证为正交投影）、solver（CG 与闭式同量级）、
额外 regularization（ridge 有害）。

**Leading hypothesis（证据尚不充分）**：把 (51) 的重建**限制在 ker(A) 内、且不重算 x₀**，
可能同时保住 0.09 量级的洞区与可见区的数据拟合。本轮只测了"全局重建 + 不重算"
（stage 2 末 0.911，stage 3 因抹掉可见区数据拟合而退化到 3.25），
未测"仅 ker(A) 重建 + 不重算"的组合。单图单 seed，需多 seed 确认。

## 产物

- 逐时间点图片（`GT | x1 | x1_hat | x_tau | x0_hat | |err|`，文件名 `stage{K}_tau{T}.png`）：
  `main/`（40 张）、`main2/`（36 张）、`main2_no_x0recompute/`（36 张）
- 指标：`summary.csv`（main + main2，76 行）、`causal_test.csv`（因果臂，36 行）
- 对照图：`comparison/trajectories.png`（洞区 / 可见区 / ‖x₀‖·n^{-1/2}，三臂）
