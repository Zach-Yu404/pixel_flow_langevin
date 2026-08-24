# alg4-box-stage3-diagnosis

state: done
type: debug
owner: claude
issue: null
pr: null
created: 2026-08-22

## 【用户原始要求】
> 请基于当前 **Algorithm 4 / clean-endpoint sampler** 做一次集中的诊断实验。**只测试 box_inpainting，禁止引入或修改 `g_bypass_stage3`，删除这个变量和其相关的影响，走正常流程，不要重新加入 `h0`、`ridge_rel` 或新的可调超参数。**
>
> 目标：查清为什么结果在 **frame id 20 多附近看起来仍正常，但后续逐渐恶化，且主要问题集中在 stage 3**。
>
> （全文含 8 节：1 定位失效时间点；2 检查 h/Hτ 影响；3 拆解错误产生位置；4 测 inner-loop feedback（Sit=1/2/默认）；5 测 S_prior 并同时看 diversity；6 图像诊断 montage；7 最终结论必须回答 9 个问题；8 文件与代码整理。原文见本轮对话与 handoff。）

## 已确认 Interpretation

**关于 `g_bypass_stage3` 的歧义读法（已判定，需用户复核）**：
"删除这个变量和其相关的影响" 与 §2 明写的 "特别检查 stage 3（**G=I 时**），H_τ = h_τ I"
是绑定的 —— 只有 bypass **仍然生效** 时 stage 3 才有 G=I、H_τ 才是标量。
故执行为：**把 `g_bypass_stage3` 从 Alg-4 采样器的参数面与 config 中彻底删除
（不再是可调变量，config 也改不动它），行为固定为当前正常流程 `eff_si = si`。**
数值与已跑结果完全一致，不引入任何新旋钮。
若用户本意是 "连 G=I 的效果也去掉"，则 §2 的标量 h_τ 分析不成立，需回退重做。

其余：只跑 box_inpainting；不加 h0 / ridge_rel / 任何新可调超参数；
`Sit` 与 `S_prior` 是**已有**参数，用户明确要求对照，不算新增旋钮。

## Scope

包含：Alg-4 采样器加只读诊断钩子（不消耗 RNG）；新模块 `alg4_diag.py`（recorder + metric + plotting）；
`main4.py` 新增 box-inpainting 诊断入口；四组实验（trajectory / Sit / S_prior×diversity / montage）；
结果统一进 `results/alg4_box_stage3_diagnosis/`；代码整理。

不包含：改动 alg2 / wv / reg 采样器；改动 `main.py` / `main2.py` / `config.json`；
碰工作区中与本任务无关的 WIP（`utils.py` 那 525 行）。

## 约束

- CONSTRAINTS §采样器纪律：不 fork 既有采样器/算子。
- 诊断钩子**不得消耗共享 RNG 流**，否则与已提交结果不可比 —— 用逐位复现验证。
- `PYTHONHASHSEED=0` 硬契约。
- 本机 ceph 间歇 EIO（见 `context/facts.md`）：跑批必须增量落盘 + 续跑 + flock 单实例。

## 执行记录

- `utils.py`：Alg-4 采样器删掉 `g_bypass_stage3` 参数（`eff_si = si` 固定，行为不变，
  且若有人传该 kwarg 会**报错**而非静默忽略）；加 `diag=None` 只读钩子（3 处）。
- 新增 `Algorithm2/alg4_diag.py`：recorder + metric + **全部诊断绘图**。
- `main4.py`：新增 `--mode diagnose`（box-inpainting 唯一诊断入口）与 `--mode diversity`；
  schema 去掉 `g_bypass_stage3` 并把它列入 `DEAD_ALG2_KEYS`（config 设它会报错）。
- 结果目录 `PixelFlowICLR/Algorithm2/results/alg4_box_stage3_diagnosis/`。
- **只读性已证**：与已提交 full_ip 同格结果相对差 2.5e-5 / 2.5e-5 / 2.0e-7。
- `--mode verify` 17 项仍全过。
- 清理：删除 `results/alg4/_smoke`（本任务前一轮的 scratch，未入 git）。
  未动 `test/` 下 alg2/wv/reg 的脚本与结果（不属本任务）。

## 结果

【结果】完整报告见 `PixelFlowICLR/Algorithm2/results/alg4_box_stage3_diagnosis/report.md`。

**失效点 = frame 31（stage 3, τ=0.111）**，单帧跳变 4.68×，全轨迹最大。
frames 0–30 全程平在 mse_full 0.074–0.127，**frame 30 反而是最好的一帧（0.0786）**。
观测区自始至终单调改善、`resid` 单调走向 1、`‖x₀‖²/n` 全程 0.99–1.02
—— **失效 100% 在 measurement null space 内**。

**机制（证据链）**：
1. **Block 1 注入的正是它自己的后验方差 `1/C⁻¹`**（洞内无数据项）。
   实测/预测比 0.98–1.08，跨全部 stage。**Block 1 按规范工作，不是 bug。**
2. **失效瞬间是 Block 1 先坏、x̂₁ 后坏**：frame 31 的 s=0/s=1 上 x̂₁ 仍改善输入
   （−0.035、−0.160），Block 1 各加 +0.136 / +0.130；x̂₁ 到 s=2 才转坏。
3. **判别变量是去噪器在洞内的收缩率**，不是 h_τ/σ_τ：
   frames 0–24 收缩率 0.27–0.45 → frame 28–29 已退化到 0.79–0.82 → frame 31 是 0.88 → 之后 ≥1。
   C⁻¹ 的**全局最小在 frame 10**（注入 0.210 > frame 31 的 0.130）却稳定，
   所以"stage 3 精度低/注入大"被排除。
4. **stage 转移会把 σ_τ 重置回高位从而救回收缩率**（0.250→0.400），
   stage 0→1→2 每次都被救；**stage 3 之后没有重置**。
   不动点公式 `hole(x̂₁)+1/C⁻¹` 精确成立（frame 30：0.1488+0.1508=0.2996 vs 实测 0.2997）。
5. **是 inner-loop feedback 累积**：S_it 10/2/1 → stage 3 增长 11.23×/3.36×/1.80×，
   **S_it=1 在 frame 31 完全没有跳变**。

**这是真发散不是多样性**：最终 x₁ 洞内 **99.1% 像素出 [-1,1]**（std 1.855，GT std 0.478）。
S_prior 四个 arm（λ_eq 4.4/25/100/400）**全部发散**（出界 99.2%/97.9%/37.3%/36.9%）
⟹ **"更强 precision 是否牺牲 diversity"在本数据上无法判定**，
测到的 spread 是发散幅度而非后验宽度。必须先消除发散再问。

**尚未排除**：stage 3 上 G=I 的结构影响（需关 `g_bypass_stage3`，本轮明令不动）；
"最后一个 stage 无 σ_τ 重置"是主因还是相关（n=1 轨迹无法区分逃逸是随机还是必然）。

【需要用户决定 / 下一步最小修改】

**把 `num_langevin`（S_it）从 10 降到 1 重跑** —— 一行 config 改动，既有 key，
不是新超参数、不是新算法。S_it=1 把 stage 3 增长 11.23×→1.80×、PSNR 6.56→12.82 dB。
**但它修的是发散不是质量**：S_it=1 的最终 hole 仍是 0.826，远差于 anchored Alg 2 的 0.10。

随后两步（都不引入新超参数）：①在 S_it=1 下重跑 diversity，那时 §7.7 才问得出来；
②确认 S_it=1 下 x₁ 是否仍出界——若仍出界，问题回到 S_prior 的口径（训练集统计而非 demo/eval）。


## 追加（2026-08-22 第二轮：探索并 debug 根因）

【用户原始要求】
> 探索并debug原因

**根因（解析可证 + 数值验证 + 受控探针三重证据）**：

1. **σ_τ→0 时 (19) 把网络整个挤出去。** 用草稿恒等式 (3) `N = σB + (e−s)H`，
   σ→0 ⟹ `x̂₁ = H_τ⁻¹x_τ`，**对任意 G、与 v_θ 无关**。velocity 只经 `σ·N·v` 进入 (19)，
   权重 `b ∝ σ_τ`。已加进 `main4.py --mode verify` 作为 **A8** 并通过。
   草稿只在 §5.2/A.3 讨论 γ→∞ 的这个极限，**没有讨论 σ_τ→0**，而 schedule 保证它必然发生。
2. **只有 stage 3 会踩到**：`σ_τ(τ=1) = 1 − e_k` = 0.750/0.500/0.250/**0.000**，
   只有 stage 3 的 `e_k = 1`。**与 `g_bypass_stage3` / G=I 无关。**
   （上一轮记的"最后一个 stage 没有 σ_τ 重置"只是表象。）
3. **逐帧收支闭合**：`Δhole = 端点步 + Block 1 注入`，逐帧精确。
   端点步是唯一移除机制，随 b 单调衰减并在 **frame 35 变号**；
   frame 31 的跳变 = 移除能力 −1.699 → −0.143（12×）而注入几乎没变。
   Block 1 的注入在**全部 40 帧**上都等于 `1/C⁻¹`（比值 0.80–1.09）。

**推翻了上一轮的解释**：受控一步映射显示白噪声污染下**每一帧都有全局吸引的稳定不动点**
（frame 31 在 δ=3.2 时 out/in=0.165），**不存在可逃逸的吸引域**，
所以"边缘稳定 + 一次坏抽样逃逸"是错的。

**真实机制是结构性棘轮**：把采样器自己发散出的 x₁ 注入同一帧（replay 臂），
与同量级白噪声对照——`hat/in` 在 frame 31 是 **0.537 vs 0.122（4.4×）**，
frame 38 是 **1.007 vs 0.872**。即：Block 1 注入白噪声 → 端点步把它转成**连贯的、
离开数据流形的内容**（最终洞内 99.1% 出 [-1,1]）→ 下一轮去噪器无法撤销 → 棘轮累积。
σ_τ→0 使 b→0，棘轮再无回摆。S_it 是棘轮的转数。

**同时解释观测区为何完好**：那里 `‖AᵀA‖/η²`≈400 压倒 `C⁻¹`≈4–459，x₁ 被 y 钉住。

**尚未测量**：端点步为何会把白噪声转成连贯离流形内容（需看 x̂₁ 的频谱/流形距离，非仅 MSE）。

【更正后的最小下一步】
`S_it` 10→1 仍是第一步，但理由变了（减少棘轮转数，而非避免逃逸）。
另一个同样不引入新超参数的方向：**stage 3 的时间网格在 σ_τ 很小处仍在放 step**
（frames 35–38 的 σ_τ = 0.178/0.134/0.089/0.045，端点步在那里已变号 +0.075…+0.164，
这些 step 只会加误差）。`ode_steps_per_stage` 与 `sigma_min` 都是既有 config key。
这与草稿 §8.3"两种误差都在 cascade 末端最小"的预期直接冲突，值得作为可证伪实验报出去。

新增产物：`results/alg4_box_stage3_diagnosis/contraction/{contraction_map.csv, contraction_map.png}`；
`main4.py --mode contraction`（含 `replay_x1` 臂）；verify A8。


## 追加（2026-08-23/24：修复收敛 + measurement 下界）

【用户原始要求】
> 修正测试一下，帮我修改过来保证他收敛，另外早期是不是对measurement consistency的match不够
> 没用的话就保持原样，撤销修改

**修复 = `sigma_min` 0.01 → 0.39（既有 key，判据推导而非搜索）**：只在 (19) 的
velocity 权重 b = N·σ_τ/(N²+γ²h²) ≥ 1 的步上运行，即 σ_τ ≥ N_k ≈ 0.4；
0.39 = 0.4 − float32 容差（scheduler s_k=0.600000024 使 f30 σ=0.39999998，
0.40 会误跳这个唯一仍收缩、携带全分辨率数据项的 stage-3 帧）。

**验证（box/junco）**：hole 3.53→**0.288**（12×）、mse_full 0.885→0.076、
resid 0.994（回到 §8.6 目标）、stage-3 增长 1.00×、NFE 390→270。
**4 seeds**：hole 0.262–0.288、出界 99.2%→**2.88%**、洞内 std 0.453（GT 0.478）、
跨 seed spread 0.356（真后验宽度，未塌缩）→ §7.7 可答：多样性存活。
另测 smin=0.40（f30 被毛刺跳过）：hole 0.319 但 obs/resid 差（0.0104/1.86），确认 f30 必须保留。

**起点更正**（用户观察 f24 触发）：恶化从 **frame 26** 开始（b<1 的第一帧），
f31 只是最大单帧跳变；领先指标是 x̂₁ 洞内下限（f26 翻倍），b≥1/b<1 把 40 帧干净二分
（下限中位 0.068 vs 1.356）。b 与 σ_τ 在轨迹上共线，靠受控探针分离：
网络贡献 (a·h)²−实测 从 0.300（f30）塌到 0.058（f38），与 b 同步。

**measurement consistency（问题 2）**：用真实 A_k 求 LS 下界
（`measurement_floor.json`）——stage 0 实测距下界 **0.2%** 且 frame 0 起即平，
**早期已饱和在表示极限，不是权重不够**；stage 3 下界为 0.5 而非 0
（op.measure 洞内也加噪，√0.25），§8.6 口径需按此读。损伤全在 ker(A)。

**概念让步（已写进 report §12.5）**：修复后采样器实际停在 σ_τ≈0.4，从未走到
schedule 末端——直接反驳草稿 §8.3"误差在末端最小"。hole 0.272 仍比
anchored Alg2 的 0.102 差 2.7×：修的是发散，不是反超。

产物：`fix_smin04/`、`fix_smin039/`（含 diversity 4 seeds）、`measurement_floor.json`、
report §12。config_alg4.json：sigma_min=0.39、cg_max_iter=300（均为验证后落盘）。
