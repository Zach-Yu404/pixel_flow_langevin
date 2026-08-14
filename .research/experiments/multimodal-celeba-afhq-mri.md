# 多模态扩展:CelebA-256 / AFHQ-512 / MRI-384(已完成/部分)

> 权威文件:`IP_package/celeba_results/{METRIC_AUDIT.md,code/RESUME_NOTES.md,metrics/metrics_summary.md}`、
> `github_project_local/docs/datasets_manifest.md`、`debug_IP4/MRI/`、
> `github_project_local/results/previous_results/{afhq,celeba,mri}/`。

## CelebA-256(500 图/config,完成)

- 先验 exp_155024/model_epoch0020.pt(num_classes=2,patch_size=2 → ~16× 慢,~445 s/img);**锁定决策:batch_size=25、SR 是 ×2、box 是固定居中 40×40 洞、random σ_n=0.01**
- 结果:box balanced_perceptual 39.01/0.9887/0.0053 全指标胜;gaussian LPIPS_king 全胜(连 PSNR);random PSNR_king 全胜;SR LPIPS_king 全胜
- 悬决:random LPIPS_king 的 srs=1e-5 与物理噪声 1e-4 不一致(改需重跑 5 组)——owner call
- ⚠️ celeba_results/ 根目录会被外部进程周期性重置;持久物只放 code/ 与符号链接的数据盘

## AFHQ-512(18 demo 图,完成;box 只有 3/4 configs)

- 先验 model_epoch80_fid39.02.pt(3 类);batch_size=1 → 复现**逐位精确**;算子按 512 缩放(gaussian k=121 σ6、box 256²)
- 最优:random 全指标 LPIPS_king;box PSNR/SSIM 翻到 pareto_dual_king(与 ImageNet 不同);其余任务 PSNR_king/LPIPS_king 各守本指标

## MRI-384(FastMRI 脑部 AXT2,SENSE acc-8,16 coils)

- 参考协议 = `debug_IP4/MRI/val_final_8/`:16 样本/slice(seeds 42-57),报 MMSE;参考指标 per-sample ~27.0-27.2,MMSE ~27.3-27.5
- 采样配置:σ_n=0.01、S=10、λ=200、h_eps=0.01(ImageNet 的 10 倍)、h_x=0.3、srs=1e-3、**ns=0**、fd=true
- **归一化契约(已验证,勿再归一化)**:x 峰值归一(|x|max=1)、csm 单位功率;per-slice scale(~1302-1385)只乘到测量 y;PSNR 幅值域 data_range=gt_mag.max()
- 先验 checkpoint 曾在迁移中丢失,**2026-07 已恢复**(`github_project_local/checkpoints/mri/CHECKPOINT_RESTORED.txt`;docs 里所有 "MISSING/reference-only" 表述已过时)→ run_mri.py 可重采样
- sweep(48 configs)**不完整且聚合坏**:SSIM 全 0、ULA/MALA 基线 0、8 个 ns1_λ50 组合全崩;best = ns0_l400_he1e-02_hx0.3_srs1e-03(PSNR 26.46);轴向:ns=0 压倒 ns=1(24.7 vs 20.6)
- MRI/ 下 dnnlib、torch_utils 是从 Sahil 的 fastmri 项目复制的(避免跨用户依赖)

## 复现基准(github_project_local,2026-06-29 建成 + 2026-07 迁移后验证)

- 设计:不重实现,33 个只读符号链接 + sys.path bootstrap,跑的就是产生参考数字的原代码;绝对路径只在 configs/paths.local.yaml
- 验证:metric 复评 ≤0.004 dB;GT/测量/掩码逐位一致;batch-parity 下采样逐位一致(gaussian max|Δ|=0);SR 残差 ≤0.074 dB 完全来自非确定 CUDA kernel;AFHQ 逐位;批组成混淆(batch-of-2 vs 8)是 ~0.5-0.8 dB 假差异的根源
- cleanup 红线(勿动):seeds、sys.path 顺序、os.chdir、model-load 分支、operator flags、metric 数学、已存输出字节
- release 迁移(剪符号链接、vendor 代码、相对路径)是**已写方案未执行**(docs/release_migration_notes.md)
