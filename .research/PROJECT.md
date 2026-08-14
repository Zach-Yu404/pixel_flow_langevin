# MSFlow

## 这个项目是什么

**PRINCIPLE**（PRogressive INterpolant with CG-Inner Preconditioned Langevin Estimation）——
基于 PixelFlow（多 stage 像素空间 flow model）先验的图像逆问题后验采样方法。
在 flow 的每个时间步内嵌 Langevin 内循环：WLS+CG 求干净图像估计、预条件 ULA 联合更新
(x1, eps)、autograd 精确伴随算子。目标是做成 paper-ready 的 benchmark：
与 DAPS / PSLD / ReSample 等 baseline 在统一算子、统一 metric、统一 seed 下对比。

## 目标

- ImageNet-256 五个线性逆问题（SR×4、Gaussian deblur、Motion deblur、random/box inpainting）上与 DAPS 对齐对比（DAPS 的原始 100 张 val 图，统一 PSNR/SSIM/LPIPS）
- 扩展模态：AFHQ-512、CelebA-256、MRI-384（acc-8 SENSE 多线圈，非线性）
- 可复现打包（`github_project_local/`，独立 repo）→ 未来 official release

## 范围（明确不做什么）

- baseline 代码不 vendor 进 repo（引用原始路径，license 各自适用）
- 大文件（checkpoint/dataset/轨迹 .pt）不进 git —— 见 `artifacts.yaml` + `local.yaml`
- PSLD/ReSample 用 FFHQ 人脸先验跑 ImageNet（OOD、非 prior-matched），**不做 naive 对比展示**
- 定量表格不进 repo（在复现机器的 `results/` 和 paper draft 里）

## 关键入口（原文件是权威，.research/ 只存要点）

| 内容 | 位置 |
|---|---|
| 方法手册（算法/变量对照/调参） | `PixelFlow/PRINCIPLE_MANUAL.md` |
| 主采样脚本 | `PixelFlow/ms_posterior_sampling_article_version.py` (+`_utils.py`, `.json`) |
| 实验总述（paper 风格，per-task 超参表） | `PixelFlow/IP_package/Experiments.md` |
| 打包实验代码（tasks/configs/runs） | `PixelFlow/IP_package/` |
| Blur 实验记忆（best config / 禁区） | `PixelFlow/debug_IP4/memory_blur.md` |
| per-task 调参工作区 | `PixelFlow/debug_IP4/`（superresolution_4, larger_box, random_inpainting, Gaussian_blur, Motion_blur, MRI…） |
| MRI 参考协议 | `PixelFlow/debug_IP4/MRI/val_final_8/` |
| 可复现 benchmark（**独立嵌套 repo**） | `github_project_local/` → github.com/Zach-Yu404/pixelflow-ip-benchmark |
| 复现路径契约（绝对路径唯一来源） | `github_project_local/configs/paths.local.yaml` |
| 训练代码 | `PixelFlow/train.py`（PixelFlow_train_code/ 已在工作区删除、未提交） |

## 外部资源

- 本 repo GitHub：github.com/Zach-Yu404/pixel_flow_langevin（private，当前分支 IP_branch）
- 嵌套 repo：github.com/Zach-Yu404/pixelflow-ip-benchmark（`github_project_local/`，**不属于本 repo 的 git 历史**）
- 上游：github.com/ShoufaChen/PixelFlow（Apache-2.0）
- 大文件位置：见 `local.yaml`（gitignored，本机映射）+ `artifacts.yaml`（逻辑名）
