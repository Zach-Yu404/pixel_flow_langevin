# cleanup-algorithm2-directory

state: done（GPU 终验被集群维护窗口挂起，CPU 证据已闭环） · owner: claude · type: implementation

## 【用户原始要求】（2026-08-18~19，逐字，按时间序）

> 把/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2整理干净，只保留utils.py,
> main.py和配套.json文件以及results放结果

> /standard/.../Algorithm2/results的结果很乱，给整理下，只要关键内容，另外.json的config文件
> 我只要1个，现在太多了，配合整理目前的.py文件，另外main里还有wls和model相关的内容，
> 给我清理掉，我只呀algorithm2的内容

> main用到的信息，/standard/.../Algorithm2/config.json没有包含，config信息只可以用这里的

> 也就是说这里也包含了很多参数，现在我不要config.json里调用两个路径，把其中用到的所有参数
> 全部放在config里，方便我一次性调整

> num_langevin，cg_max_iter_l14和cg_max_iter重复吗，另外这个num_langevin有用到嘛，
> 对于box_inpainting,我记得还有个projection，这个选项我没有看到

## 【Claude｜执行记录】

1. **目录收敛**（commit f9299d7）：utils.py + main.py + config + results/；15 旧文件 git rm；
   等价性：CPU verify ALL PASS、self-test S1 1.08e-07/S2 0.0110（=历史）、GPU job 18643397
   hole 0.9691 逐位。
2. **results 清理 + WLS/Model 剥离 + config 单一化**（并入 e3f23b9）：results 170M→18M
   只留五份关键证据（anchor/confirm/h0/x0steps/unified + ICLR PDF）；full_ip 旧数据
   （含 class_label 不公平 alg2 行）与中间帧删除（git 历史保留）；main 剥离 Alg1-WLS/Model
   全部内容（onestep 的 x1_model 仅保留 l.14 warm start 论文角色；被删路径均为确定性 CG，
   不消耗随机流，alg2 数字可证明不变）；4 个模式 JSON → 1 个 config.json。
3. **参数全内联 + projection 补全**（commit e3f23b9）：lpips_king_root/best_cfg_dir 指针废除，
   sampler_kw + tasks_setup 内联（值程序化提取；差异处理：group_format 死键弃、gaussian
   kernel_std 浮点噪声归一 3.0、random num_langevin=15 任务级覆盖）；用户指出的
   terminal_replace_weight（projection，box/random=1.0）遗漏已补——采样器循环后投影，
   逐步指标不受影响；utils:327 硬编码 200 → algorithm.cg_max_iter_l14。
4. **S2 stage-1 漂移判决**：无种子 hash('junco') 两进程两值（779425382/3471873873），
   PYTHONHASHSEED=0 恒 1032905628 ⇒ 漂移=某次 CPU 调用漏设种子（调用失误，非代码）；
   复现契约 = PYTHONHASHSEED=0（所有 sbatch 均设）。
5. **在途**：GPU 终验 job 18650178（判据 hole 0.9691；投影为循环后操作，行指标可证不变）
   被 A100 维护窗口（50 节点 maint）挂起，通过后回填。

## 备注
- final_denoise（box/gaussian/motion=true）为 Alg1 末端 WLS 去噪，按"只要 algorithm2"未纳入。
- 本文件为补记：目录重组后的四轮改动当时未按纪律同步 .research，由用户指出后补齐。
