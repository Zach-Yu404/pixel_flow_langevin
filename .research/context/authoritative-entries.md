# 权威入口索引(哪个问题读哪个文件)

> 2026-08-14 记忆导入产物。相对路径基于 repo 根。旧文档内的绝对路径全部是旧机器(/home/nvidia/...),按 local.yaml 映射。

## 方法与代码
| 问题 | 文件 |
|---|---|
| 算法/变量对照/调参手册 | `PixelFlow/PRINCIPLE_MANUAL.md` |
| 基准重跑实际 import 的规范 sampler | `PixelFlow/ms_posterior_sampling_article_version_final.py`(+`_utils.py`) |
| 论文式实验规格(per-task 超参表、OAT 消融、MRI/CelebA 配置) | `PixelFlow/IP_package/Experiments.md` |
| IP pipeline 怎么跑(config schema、锁定 seed 约定) | `PixelFlow/IP_package/README.md` |
| all-seeds-42 政策唯一源 | `PixelFlow/IP_package/rerun_imageNet/seeding.py` |
| rerun 硬件/度量审计(3 把 LPIPS 尺子、FID 偏差) | `PixelFlow/IP_package/rerun_imageNet/METRIC_AUDIT.md` |
| v5 debug sampler vs article 版结构差异(12 个死旋钮) | `PixelFlow/debug_IP4/larger_box/val_box/final_configs/sampler_diff_v5_vs_article.md` |

## 前代 agent 记忆(最高密度)
| 内容 | 文件 |
|---|---|
| 记忆索引(读哪份 handoff) | `PixelFlow/IP_package/baselines/results/memories/README.md` |
| MMSE/不确定性/DC 战役全记录 | `.../memories/SESSION_HANDOFF.md` |
| 100 图全 baseline 战役(env vars、启动、clobber 事故) | `.../memories/SESSION_HANDOFF_100img_all_baselines.md` |
| 复现基准项目(bit-exact 验证、红线、MRI 契约) | `.../memories/SESSION_HANDOFF_github_project_local.md` |
| debug_IP4 战役地图 / run_ip4 旋钮 / val 协议 / SR4 / 掩码 bug / 低 PSNR 归因 | `.../memories/global_memory/*.md` |

## 实验结论
| 内容 | 文件 |
|---|---|
| 各任务 N=500 验证赢家 | `PixelFlow/IP_package/tasks/*/validation_summary.md`(+同目录 metric_logic_audit.md,**头部陈旧时以 audit 为准**) |
| box 128² 完整调参史 | `PixelFlow/debug_IP4/larger_box/`(val_box/NOTES.md 为入口) |
| eps 熵机制(为何 he=1e-3 出语义内容) | `PixelFlow/debug_IP4/exploration/W1_DIAGNOSIS.md` |
| 公平性审计 + 修复清单 | `PixelFlow/IP_package/baselines/baseline_audit_report.md` |
| 最终 3-way 对比报告(citable) | `PixelFlow/IP_package/baselines/results/updated_all_results/FINAL_REPORT.md` |
| 全方法 100 图统一表(AlexNet LPIPS) | `.../updated_all_results/metrics_table_100.md` |
| demo15 任务规格(locked) | `PixelFlow/IP_package/baselines/task_definitions_demo15.md` |
| 低 PSNR 归因(非 bug 证据链) | `PixelFlow/IP_package/baselines/low_psnr_diagnostics_report.md` |
| CelebA LPIPS 尺子政策 | `PixelFlow/IP_package/celeba_results/METRIC_AUDIT.md` |
| 复现契约与容差(15 问答) | `github_project_local/docs/local_reproduction_report.md` |
| 算子对齐明细(vs DAPS 的 max\|Δ\|) | `github_project_local/docs/operator_alignment_notes.md` |
| MRI 归一化契约 | `github_project_local/docs/mri_normalization_notes.md` |

## 陷阱与矛盾
见 `.research/context/contradictions-registry.md`(哪些文件陈旧、哪些数字不可混用)。
