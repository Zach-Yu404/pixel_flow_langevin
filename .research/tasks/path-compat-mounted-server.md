# path-compat-mounted-server（2026-08-18）

## 【用户原始要求】（逐字）

> 当前服务器是 mount 过来的环境，之前代码里使用的绝对路径可能不适用于这里。
>
> 请根据当前服务器实际目录结构检查并修复所有相关路径。已知可能存在：
>
> * 旧服务器：`/standard/CBIG-Standard-ECE/Zach/...` 或 `/sfs/ceph/standard/CBIG-Standard-ECE/Zach/...`
> * 当前服务器：`/CBIG-Standard-ECE/Zach/...`
>
> 要求：
>
> 1. 不要直接把旧路径硬改成新路径，否则旧服务器会无法运行。
> 2. 对需要访问的路径做兼容处理，优先使用 `try/except` 或候选路径检测，依次尝试可能的路径，直到找到真实存在且可访问的路径。
> 3. 优先根据当前文件/项目位置动态推断路径，尽量减少硬编码。
> 4. 如果所有候选路径都不可用，给出明确报错，并打印尝试过的路径。
> 5. 修改后必须保证：
>    * 当前 mount 服务器可以正常运行；
>    * 原来的旧服务器也可以继续运行；
>    * 不改变原有代码逻辑，只修复路径兼容性。
> 6. 先检查当前目录结构和现有代码中的路径使用情况，再修改；不要凭假设批量替换路径。
> 7. 修改完成后，在当前服务器实际运行相关代码验证路径解析成功，并说明最终采用了什么兼容方案。

（同轮在途请求，尚未闭环时并行推进：「继续，我要测试terminal_replace_weight = 0和1.0」——
trw 实验代码与 config 已在 6a5a927 提交，本机 GPU 1 正在跑 junco trw∈{0,1}。）

## 环境事实（排查结论，先查后改）

- 新机器 = 同一 ceph 共享盘挂载在 `/CBIG-Standard-ECE/`（`Zach` 与 `Zach_dataset` 都在）；
  旧前缀 `/sfs/ceph/standard/...`、`/standard/...` 在本机均不存在。
- 本机：4×A100-80GB 本地直跑，**无 Slurm**；HOME=`/home2/rqx6rs`（旧机 `/home/rqx6rs` 不存在，
  home 不共享）；conda env `pixelflow` 本机已有（`/home2/rqx6rs/anaconda3/envs/pixelflow`），
  torch 2.11.0+cu128 CUDA 可用；缺 `piq` 已 pip 装上。
- 活跃代码链（Algorithm2 → onestep_mse_vs_t → IP_package）本身全部 `__file__` 相对推断 +
  config 占位符 `{IP_PACKAGE}/{HERE}`，无硬编码绝对路径；grep 命中的旧路径全在
  `temp/`、`rerun_imageNet/`、`baselines/` 遗留脚本（未改动，`operators.py` 的 DAPS
  死路径已被 utils.py 预插桩免疫）。
- **真正断点 = 绝对符号链接**（链接内容存于共享盘，旧机写的 `/sfs/...` 目标在新机死链）。

## 兼容方案（两层）

1. **符号链接改相对目标**（一次修改、两台机器同时生效，不动任何数据）：
   - `PixelFlow/IP_package/pretrained_models` → `../pretrained_models`
   - `PixelFlow/IP_package/pixelflow` → `../pixelflow`（**致命项**：调度器/模型包本身，
     不修则新机 import 必挂）
   - `PixelFlow/IP_package/demo_daps100/*.JPEG`（64 个）→ `../../../../../Zach_dataset/...`
   - `PixelFlow/IP_package/celeba_results/{logs,groups,runs}` → 同上 Zach_dataset 相对形
   - `github_project_local/checkpoints/mri/{config.yaml,model_epoch40.pt}` → 同上
     （modeL_epoch40.pt 经相对链实测 7.3GB 可读）
   - `baselines/` 下 7 个（DAPS demo-imagenet15 / DAPS checkpoints×2 / PSLD samples /
     ReSample×3）→ `realpath --relative-to` 生成相对形
2. **代码侧候选前缀重映射**（`Algorithm2/main.py`：`SHARE_ROOTS` + `resolve_path()`）：
   config.json 路径先做占位符替换；若结果存在→原样返回（旧机行为零变化）；不存在且
   前缀命中已知挂载根之一→依次尝试其余根；全部失败→ `FileNotFoundError` 打印全部
   候选。只对输入路径（model_dir/demo_dir/gamma2_table）生效，输出目录本就 HERE 相对。

## 验证（新机实测）

- `resolve_path` 四例全过：本机路径直通；`/sfs/...` 与 `/standard/...` 旧绝对路径均正确
  重映射到 `/CBIG-Standard-ECE/...`；缺失路径报错并列出 3 个尝试过的候选。
- import 链（main→utils→base→demo_runner→pixelflow 包）在新机全通（经修复后的相对链接）。
- 端到端：本机 GPU 1 real run `main.py`（mode=debug_box，trw sweep）进行中，结果待回填。
- 深扫（maxdepth 6）修复后未再发现旧前缀链接（收尾时复扫确认）。

## 踩坑记录

- `while IFS=' -> ' read` 把 `-` 也当分隔符 → `demo-imagenet15`、`diffusion-posterior-sampling`
  两条路径被截断漏修；改为逐个显式处理后修复。
- 中断轮的后台命令实际已执行（符号链接与 GPU 启动各生效一次）→ 出现重复 GPU 进程，
  已杀新留旧；后续以 `pgrep` 实查为准，不凭会话记忆假设进程状态。
- 本机 ceph 偶发 `Remote I/O error`（git objects 读取一次性抖动，重试即过）。

## 端到端验证结果（新机 GPU 1 实跑，2026-08-18）

`main.py`（mode=debug_box, trw sweep, junco）在新机跑通，与集群同一实验对照：

| variant | 机器 | pre_hole | pre_obs | post_hole | post_obs |
|---|---|---|---|---|---|
| trw=0 | 集群 | 0.96912378 | 0.00182969077 | 0.96912378 | 0.00182969077 |
| trw=0 | 新机 | 0.96912378 | 0.00182969018 | 0.96912378 | 0.00182969018 |
| trw=1 | 集群 | 0.96912378 | 0.00182969077 | 0.96912378 | **0.0** |
| trw=1 | 新机 | 0.96912378 | 0.00182969018 | 0.96912366 | **0.0** |

- 跨机一致性：**同到 8 位有效数字，但非逐位相同**（两台的 conda env 是各自独立安装，
  新机 torch 2.11.0+cu128）。obs 相对差 3e-8。**不要对外宣称 bit-exact**。
- trw=1 的 post_hole 比 pre_hole 低 1.2e-7：隔离 GPU 试验证明投影
  `w*(m*y+(1-m)*x1)+(1-w)*x1` 在 m=0 处**逐位不改动 x1**（`torch.equal` True，MSE 相对差 0），
  故该残差是度量路径的 float32 归约舍入，非真实效应。

## 状态

- [x] 排查（目录结构 + 代码 grep + 符号链接三级扫描）
- [x] 修复（74 个符号链接 → 相对；main.py resolve_path）
- [x] resolve_path 单测 + import 链验证
- [x] 端到端 GPU run 通过（新机复现集群 trw 实验，数值 8 位有效数字一致）
- [ ] commit + Codex review Issue
