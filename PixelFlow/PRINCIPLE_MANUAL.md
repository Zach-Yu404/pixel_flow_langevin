# PRINCIPLE 实用手册

**PRogressive INterpolant with CG-Inner Preconditioned Langevin Estimation**

---

## 1. 文件结构

```
PixelFlow/
├── ms_posterior_sampling_article_version.py        # 主脚本
├── ms_posterior_sampling_article_version_utils.py   # 工具函数 (CG, WLS, Langevin)
├── ms_posterior_sampling_article_version.json       # 配置文件
├── ms_posterior_sampling.py                         # 旧版脚本 (对比参考)
└── ms_posterior_sampling_utils.py                   # 旧版工具函数 (部分被 re-export)
```

---

## 2. 快速运行

```bash
python ms_posterior_sampling_article_version.py
python ms_posterior_sampling_article_version.py --config /path/to/config.json
```

---

## 3. 算法流程 (Algorithm 1)

```
x1_k = randn(),  eps_k = randn()

for stage k = 0..K-1:
│ if k > 0: upsample x1_k, renoise eps_k (PixelFlow alpha/beta + block noise)
│ A_k, AT_k = make_Ak_fns(operator, stage_shape)
│
│ for step i, T in Timesteps_k:
│ │ t_curr = scheduler.t[i]
│ │ sigma_t = (1-t)(1-s_k) + t(1-e_k)
│ │ x_tau = H_t(x1_k) + sigma_t * eps_k
│ │
│ │ for s = 1..S_k:                    ← Langevin inner loop
│ │ │ (a) mu = model(x_tau, T)         ← velocity prediction
│ │ │     xs_hat = x_tau - t*mu
│ │ │     xe_hat = x_tau + (1-t)*mu
│ │ │ (b) x1_hat = CG(M, r)           ← WLS clean estimate
│ │ │ (c) s_flow = (H·x1_hat - x_tau) / sigma_safe²
│ │ │ (d) g_x1 = AT_k(b - A_k·x1)/η² + HT(s_flow)
│ │ │     g_eps = sigma_t * s_flow - eps_k
│ │ │ (e) delta = CG(P⁻¹, rhs)        ← preconditioned ULA
│ │ │     x1_k += delta
│ │ │ (f) eps_k += h_eps/2 * g_eps + noise
│ │ │ (g) x_tau = H_t(x1_k) + sigma_t * eps_k
```

---

## 4. 变量对照表

### 伪代码 → 代码

| 伪代码 | 代码 | 说明 |
|--------|------|------|
| $b$ | `y` | 观测值 |
| $A, A^T$ | `A_k_fn, AT_k_fn` | 前向/伴随算子 (AT 用 autograd 精确伴随) |
| $\eta$ | `sigma_n` | 观测噪声 |
| $K$ | `num_stages` | stage 数 |
| $N_k$ | `stage_inference_steps` | 时间步数/stage |
| $S_k$ | `num_langevin` | Langevin 步数/时间点 |
| $\tau$ | `t_curr` | stage 内归一化时间 ∈ [0,1] |
| $s_k, e_k$ | `start_t, end_t` | stage 系数 (from scheduler) |
| $x_1^k$ | `x1_k` | 干净图像估计 |
| $\epsilon^k$ | `eps_k` | 噪声变量 |
| $x_\tau^k$ | `x_tau_k` | 插值态 |
| $G$ | `apply_G()` | 低通投影 (self-adjoint, 已验证) |
| $H_\tau^k$ | `apply_H_tau()` | $(1-\tau)s_k G + \tau e_k I$ |
| $\sigma_\tau$ | `sigma_t` | $(1-\tau)(1-s_k) + \tau(1-e_k)$ |
| $h_x, h_\epsilon$ | `h_x, h_epsilon` | Langevin 步长 |
| $\lambda_x$ | `lambda_x` | WLS 岭正则 |
| $\lambda$ | `lambda_reg` | CG 预条件岭正则 |
| $\rho_s^k, \rho_e^k$ | `rho_s, rho_e` | WLS 端点权重 |

---

## 5. 配置参数

### 核心 Langevin

| 参数 | 默认 | 调参 |
|------|------|------|
| `num_langevin` | 20 | 先试 5-10 |
| `h_x` | 1e-3 | 太大发散，太小慢。1e-4 ~ 1e-2 |
| `h_epsilon` | 1e-3 | 通常 ≤ h_x |

### 正则化

| 参数 | 默认 | 作用 |
|------|------|------|
| `lambda_x` | 0.01 | WLS 岭正则 |
| `lambda_reg` | 0.01 | CG 预条件器岭参数 |
| `rho_s` | 1.0 | WLS start 端权重 (支持 per-stage list) |
| `rho_e` | 1.0 | WLS end 端权重 (支持 per-stage list) |

### CG

| 参数 | 默认 | 说明 |
|------|------|------|
| `cg_tol` | 1e-5 | 相对收敛容差 ‖r‖/‖b‖ |
| `cg_max_iter` | 50 | 最大迭代 |

### measurement_mode 与 sigma_n

| mode | y 生成 | sigma_n 含义 | 一致性 |
|------|--------|-------------|:---:|
| `"measure"` | y = A(gt) + σ_n·ε | likelihood 噪声 σ | **一致** |
| `"call"` | y = A(gt) (无噪声) | ⚠️ 不应 > 0 | **不一致** |

默认为 `"measure"`。如用 `"call"` + `sigma_n > 0`，config loader 会发出 warning。

---

## 6. 算子和伴随设计决策

### G = apply_G (self-adjoint)

```
G(x) = nearest_up( bilinear_down(x) )
```

**数值验证**：G^T ≈ G，伴随误差 ~1e-7（float32 精度），所有分辨率。不需要单独的 GT 实现。

### A_k / A_k^T (exact adjoint via autograd)

```
A_k(x)   = mask · bilinear_upsample(x)
A_k^T(r) = _interpolate_adjoint(mask · r, stage_size)
```

`_interpolate_adjoint` 用 `torch.autograd.grad` 精确计算 bilinear upsample 的转置。保证 AT@A 是 SPD（CG 收敛所需）。

**验证结果**：

| Scale | 旧 (bilinear↓) AT@A 对称误差 | 新 (autograd) |
|:---:|:---:|:---:|
| 8x | 7% | ~0 |
| 4x | 26% | ~0 |
| 2x | 122% | ~0 |

开销：单次 ~0.18ms (GPU)，总占比 ~5%。

### sigma_tau_safe 设计

`sigma_tau_safe = max(sigma_tau, 1e-4)` 仅用于 score 中的 `1/σ²` 除法。原始 `sigma_tau` 用于 `g_eps` 的乘法和 `x_tau` 重构。

---

## 7. CG solver

- **Batch-aware**: per-sample alpha/beta，shape `(B,1,1,1)` 广播
- **相对收敛**: `‖r‖/‖b‖ < tol`（尺度无关）
- **SPD 保证**: WLS 的 M = coeff·G²+coeff·I (SPD); step(e) 的 AT@A/η²+λI (SPD, exact adjoint)

---

## 8. 输出

```
{dict_path}/{exp_name}/
├── *_mode-{mode}.pt                    # 轨迹 (if save_dict_to_pt)
├── *_mode-{mode}.run_config.json       # 配置副本
├── *_mode-{mode}_langevin_logs.csv     # loss/grad_norm/delta_norm per step
├── *_mode-{mode}_with_y.mp4           # 采样视频
└── *_mode-{mode}_langevin_inner.mp4   # Langevin 内部视频
```

### Langevin 日志关键信号

| 信号 | 正常 | 异常 |
|------|------|------|
| `loss` | 持续下降 | 振荡/发散 → 减小 h_x |
| `delta_norm` | 稳定 O(0.01-0.1) | 极小 → lambda_reg 太大或 CG 没收敛 |
| `grad_norm` | 有界 | 爆炸 → sigma_tau 太小，检查时间调度 |

---

## 9. 调参流程

### Step 1: 快速验证
```json
{"num_langevin": 5, "inference_each_step": 5, "num_examples": 2,
 "save_dict_to_pt": false, "save_videos": true}
```

### Step 2: 扫描 h_x
```
h_x = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
```
选 loss 下降最快且不发散的值。

### Step 3: 调正则化
- 结果模糊 → 减小 lambda_x
- 结果有伪影 → 增大 lambda_x / lambda_reg

### Step 4: 加量
```json
{"num_langevin": 20, "inference_each_step": 10, "num_examples": 4}
```

---

## 10. 计算量

每 stage 模型调用: $N_k \times S_k$ (Langevin 内) + 1 (velocity_fn 创建但在 Langevin 中调用)

默认 (N=10, S=20, K=4): **800 次模型 forward**

每 Langevin step 有 2 次 CG (WLS + preconditioner)，每次 CG ≤ 50 次矩阵乘。

---

## 11. 与旧版区别

| | 旧版 | PRINCIPLE |
|---|---|---|
| 主变量 | latents (插值态) | x1_k + eps_k |
| Langevin | autograd ULA, 仅 x1 | 解析梯度, 联合 x1+eps |
| x1 估计 | 直接 velocity prediction | WLS + CG |
| 预条件 | 无 | CG (A^T A/η² + λI)⁻¹ |
| A_k^T | bilinear↓ (近似) | autograd 精确 |
| stage 转换 | alpha/beta + block noise | 同左 (保留 PixelFlow renoise) |
