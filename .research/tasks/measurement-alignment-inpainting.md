# measurement-alignment-inpainting

state: working · owner: claude · type: experiment · 开始：2026-08-18

## 【用户原始要求】（逐字）

> 论文里 "noisy inpainting σ=0.05" 与代码不符 —— 要么改措辞，要么真的给 inpainting 加噪。什么意思，另外你说的数学缺陷：τ=0 的 ridge 让方程解 M = M0 + εI，但右端噪声只有 M0 的协方差——所谓"精确抽样"实际抽的是 M⁻¹M₀M⁻¹。补 √ε·ξ 后新增 V9 经验协方差测试，证实旧实现确实抽错了分什么意思

> 给我全部中文的解释

（随后在 plan 审批中逐条追加的约束，逐字）

> 其中你发现了operator和baseline的不统一，我现在同意你给统一后开始实施

> 所有的实验在Algorithm2下的test中跑，不许改之前的代码，跑algorithm2，根据/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2下的inverse problem的代码进行试验

> set_timesteps这个逻辑还按照Pixelflow本身的逻辑来，以免出错

裁决（AskUserQuestion）：统一范围＝**只改我们的（加 σ=0.05）**；box 位置＝**改为居中 128×128**；
重跑范围＝**"这个algorithm2先跑3张图junco, shetland_sheepdog和ibex_horns"**。

## 背景：两个被查实的问题

### 1. inpainting 无噪，与 8 个 baseline 不同难度

`demo_runner.py` 两个 inpainting 分支走 `y = op(gt)`；算子前向 `inpaintingStart.py:377` 就是
`return mask * x_use`，构造传入的 `sigma` 只存 `self.sigma`、**前向从不使用**。
实测 box 可见区 `max|y − GT| = 0.0`。而算法侧仍以 η=σ_n=0.05 作真实噪声用
（`1/η²` 权重、l.13 的 `(1/η)Aᵀξ_y`）。

Baseline 侧实测（逐个查代码）：

| 方法 | inpainting 噪声 | 来源 |
|---|---|---|
| DPS、FPS-SMC | σ=0.05 | 读共享 `demo15_measurements_cca/y.npy`（noise 已烘焙） |
| DAPS、PSLD、ReSample | σ=0.05 | 各自 `measure()` / `noiser()` |
| DDRM、DDNM、DiffPIR | **σ=0.10（2×）** | 各自 runner 里 `2 * SIGMA_N` |
| **我们** | **0** | `y = op(gt)` |

噪声阶梯 **我们(0) < 五个(0.05) < 三个(0.10)**：我们的 inpainting 指标是在更容易的问题上取得的。
仓库自查报告 `baselines/baseline_audit_report.md` 早已列为 "CRITICAL — in our favor"，
且其修复方案**只落地了 mask 反转那一半**（2026-06-23），加噪那半从未实施；
故 `baseline_metric_summary.md` 里 35.48 dB 的 random-inpainting 同时带**旧反转 mask + 无噪**两个优势。
`Experiments.md:25-31` 我们自己的表仍标 σ_n=0.05。

另一个同向差异：box 洞位置我们随机（`make_box_mask` 用 `hash()` 种子），
共享协议与 DPS/FPS/DDRM/DDNM/DiffPIR 用**居中** 128×128。

### 2. τ=0 使 ridge ε 变成建模参数

论文 **Prop. 4（p.9）**："A ridge εI with ε > 0 is therefore added to M_τ.
**It is required by the formulation, not by numerics.**" → 上一轮补的 `√ε·ξ` 符合论文，保留。

论文 **§7.13（p.26）**："**the cleanest remedy is not to add anything but to start the time grid
at the first τ > 0**" 与判据 "if performance depends appreciably on ε, then ε is **doing modelling
work rather than conditioning work**, and that dependence **should be reported**"。

稠密实测（RES=16，box 遮一半，τ=0）：

| stage | σ_τ | rank(M0)/dim |
|---|---|---|
| 0 | 1.000 | **128/256** |
| 1 | 0.857 | 160/256 |
| 2 | 0.667 | 160/256 |
| 3 | 0.400 | 256/256（满秩） |

两版协方差按子空间分解（stage 0，τ=0，ε=4.0e-04）：

| 子空间 | 修复版（含 √ε） | 旧版 |
|---|---|---|
| 可辨识 | 2.4999980e-03 | 2.4999950e-03（相对差 1e-6，**两者都对**） |
| 零空间（＝洞区） | **2500**（每像素 std 50） | **0** |

即差异**完全**局限于数据约束不到的方向；且注入量 = 1/ε 由数值旋钮决定，
**ε 越小注入越大，ε→0 极限不存在**。CG 截断不压制（洞区 std≈35，稠密精确解 48.7）。
实测网格**含 τ=0**（每 stage 第一步，`linspace(t_start=0, …)`）。

→ 裁决：**不动 `set_timesteps`**（用户指令），改为扫描既有 `algorithm.ridge_rel` 旋钮，
直接测出 §7.13 要求报告的那个依赖量。

## 实现二次调整：参数统一进 `Algorithm2/config.json`（2026-08-18，用户指令）

> 把operator的参数也在/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/config.json里统一

裁决（AskUserQuestion）：**config 存参数 + main.py 真正消费它**——而非"参数放 config、机制留 test"，
因为后者会让直接跑 `main.py` 的人拿到一份**静默失效**的配置（正是 Codex 抓过的 anchor no-op 那类问题）。

落地方式（commit `3c66cc5`）：

- 新增 `Algorithm2/measurement.py` 承载对齐逻辑；`main.py` 与 `utils.py` **只改 import 一行**
  （`from demo_runner import …` → `from measurement import …`），**所有调用点不动**。
- `config.json` 新增：每任务 `measurement_mode`、box 的 `operator.center`、
  `algorithm.measurement_seed`；两个 inpainting 的 `terminal_replace_weight` 改为 0.0。
- 契约同步扩展（`TASK_KEYS` / `TASK_OPERATOR_KEYS` / `CONFIG_SCHEMA["algorithm"]`），
  实测三类错误均 fail-fast：漏 `measurement_mode`、`center` 拼成 `centre`、漏 `measurement_seed`。
- blur/SR 若设 `measurement_mode="call"` 直接拒绝——它们的 y 本就带噪，
  "honour" 它会静默改掉所有既有结果。
- `test/overrides.py` 随之删除，test 层只剩"跑哪些图/哪些 sweep"的选择。

**顺带修掉一个真 bug**：`utils.py` 只把 `PixelFlowICLR/` 加进 `sys.path`、未加自身所在的
`Algorithm2/`，换上下文导入时 `measurement` 不可见；按该文件既有写法补 `sys.path.insert(0, HERE)`。

回归护栏数值在逻辑上移前后**完全一致**（random 0.0497 / box 0.1243，blur/SR 逐位不变），
证明只是搬家、未改行为；V1–V9 与 self-test 均通过。

## 实现一（历史）：`Algorithm2/test/` 下的临时覆盖层

复用现成物件，不重写任何数学：

| 需求 | 复用 | 位置 |
|---|---|---|
| inpainting 加噪 | `op.measure(x)` | `inpaintingStart.py:379-383` |
| box 居中 | `make_box_mask(..., short_name=None)` | `demo_runner.py:215-227` |
| 换算子 mask | `_force_mask(op, mask)` | `pipeline.py:454-456` |
| ε 依赖 | 既有配置键 `algorithm.ridge_rel` | `utils.py:316-318` |

- `test/overrides.py`：包装 `build_setup_and_measurement` 后重绑定
  `main.` 与 `utils.` 两处同名引用 → 既有 runner 原样生效。
  噪声种子用 **sha256** 派生（不用 Python `hash()`，后者跨进程不稳定）；
  调用前后保存/恢复全局 RNG 状态，避免污染采样器噪声流。
- `test/config.json`：test 专属开关；派生配置严格符合既有 schema。
- `test/run_experiments.py`：驱动，sweep 带 ceph 重试，
  **成功判据＝该次运行自己的 CSV 是否存在**（不 grep 共享日志——上一轮就是这么假成功的）。

## 验证

**回归护栏（关键验收）**：对齐只作用于两个 inpainting，其余零影响。

| 任务 | 结果 |
|---|---|
| gaussian_blur / motion_blur / superresolution | mask 与 y **逐位未变** |
| random_inpainting | mask 不变；y 变化 std **0.0497**（= σ_n 0.05） |
| box_inpainting | mask 变居中；y 变化 std 0.1243（洞移位＋噪声） |

**CPU**：verify V1–V9 `ALL CHECKS PASSED`；self-test 通过
（adjoint 5.59e-09；S1 最差 1.21e-07；S2 stage3 观测 RMSE 0.0110，stage1 box 0.0653 / random 0.0948）。

**schema 契约立功**：第一版驱动给 `verify` 段塞了 `out` 键，被
`KeyError: config.json "verify": unknown=['out']` 当场拦下，而非静默跑错配置。

## 进行中 / 待回填

- `full_ip`：5 任务 × 3 图（junco / shetland_sheepdog / ibex_horns），新测量下基线。
- `ridge_sweep`：`ridge_rel ∈ {1e-8, 1e-6, 1e-4, 1e-2}` × 4 seed（7/42/123/2024）。
  判读须用多 seed 均值：旧测量下洞区 MSE 的 seed 地板实测 5.3%（n=4，sd 0.0237），
  本轮由 sweep 自身的组内 sd 重新给出。

## 用户裁决

**`terminal_replace_weight = 0`**（2026-08-18，用户指令，逐字："terminal_replace_weight =0"）。

理由：观测加噪后，投影 `x1 ← m·y + (1−m)·x1` 会把测量噪声**原样抄进可见区**，
可见区误差从"精确 0"（无噪时代）变成 ~σ²=0.0025，即投影由"免费的数据一致性"变成"注入噪声"。
落实位置：`test/config.json` 的 `terminal_replace_weight`，由驱动写进派生配置的全部
`tasks_setup.*`——**不改 `Algorithm2/config.json`**（该文件里 inpainting 仍是 1.0，属既有代码）。

注：本轮 full_ip 在该裁决之前启动，跑的是 trw=1，因此 `full_ip_final.csv` 同时给出
`pre_mse`（投影前＝trw=0 的等价结果，投影是循环后一次性操作，轨迹完全相同）与
`post_mse/post_obs`（投影后）——正好用同一次运行的数据佐证该裁决，无需补跑。
其后所有运行（含 ridge sweep）均为 trw=0。

## 待用户裁决

- 若 ε 依赖显著超过 seed 地板：论文是如实报告该依赖，还是采纳 §7.13 建议改网格从 τ>0 起
  （后者需改 PixelFlow 调度逻辑，本轮明确不做）。

## 遗留（已知不一致，本轮不动）

`rerun_imageNet/runner.py:328` 同样硬编码 inpainting 无噪。本轮不跑该管线故未动，
**将来重跑 5×100 验证表前必须先做同样对齐**，否则大表与本轮小表口径不一。
