# test-terminal-replace-weight

state: working · owner: claude · type: experiment · 开始：2026-08-18

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
