# Clean Backbone 上 VS Stage II 三组控制结果

完成时间：2026-10-13T22:13:34+08:00。独立实验目录：[全部源码、日志、数组与模型](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013)。

## 1. 结果与协议范围

完成一个**随机初始化的 seed42 Clean Stage I（200epochs）**，基于1980张官方TRAIN留出验证选最佳；之后完成**seed123 Uniform-PerSample、High-VS、Low-VS 各10epochs**。旧StageI、原VS、SMART和H2T结果完整保留。没有加载CIFAR100(train=False)、解码官方TEST图像或标签、计算TEST accuracy、使用TEST选模。prepare只对官方test文件作1次不透明MD5校验；训练/统计/StageII的TEST文件访问拦截记录均为0。

**固定epoch10结论：** High-VS相对Uniform ΔOverall=-0.1515pp、ΔTail=-0.6897pp、ΔFew=-1.0000pp；Low-VS相对Uniform ΔOverall=+0.3030pp、ΔTail=+1.0345pp、ΔFew=+1.0000pp。本轮复现了固定第10epoch Overall的正向趋势。该结论只针对一个Clean基座与一个StageII随机流，不能称多种子稳定改善或官方测试收益。

全部准确率为**99类、1980张官方TRAIN留出验证**，class1–99各20张，class0缺失；100维输出保留。Head有效35类、OfficialMedium35类、Tail29类；Many34类、频次Medium35类、Few30类。Overall与Macro99相同。不能与论文100类TEST准确率直接比较。

## 2. Clean Stage I

新模型从随机初始化训练，未载入旧epoch198；Python/NumPy/Torch seed42，LT的官方seed0索引在独立RandomState中核对，冻结数组加载不改变训练随机流。ResNet32、CE+MixUp alpha1、SGD lr0.1/momentum0.9/weight_decay0.0002、batch128、workers0、FP32、200epochs、官方5epoch warmup和160/180下降。实际train与scheduler从复制的兼容源码AST逐字编译复用。

RNG实现细节：旧官方Dataset构造会将NumPy全局状态重置到subset seed0，进而影响后续MixUp的Beta抽样序列。本轮加载已冻结数组，训练NumPy流明确按用户指定seed42运行，LT seed0只用于独立索引核对。MixUp公式与Beta分布保持不变；旧/新运行的差异还包括这一随机流路径，不能解释为只更换epoch选择数据。

训练10847张LT，100类，max500/min5，IF100；raw IDs、标签数量和像素SHA与历史StageI逐项一致。验证TRAIN raw ID和精确图像与LT均零重叠。

- 最佳epoch：**184**，按验证Overall、严格变大才替换、平局最早。没有early stopping。
- 最佳验证Overall=40.0505%、Head=69.2857%、Medium=38.8571%、Tail=6.2069%。
- 最佳模型：[clean_stage1_best.pth](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/stage1/checkpoints/clean_stage1_best.pth)，SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。
- 最终模型：[clean_stage1_final.pth](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/stage1/checkpoints/clean_stage1_final.pth)，SHA256 `c7c157460c1da606704107395e067eee83dcbbd12e10ed3ff6b486fa8c18d504`。
- 17000次正式更新，loss与全部梯度有限；验证不更新BN。真实数据1batch预检查使用丢弃模型，正式模型重新初始化。
- 协议：[CLEAN_STAGE1_PROTOCOL.md](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/CLEAN_STAGE1_PROTOCOL.md)、[clean_stage1_protocol.json](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/clean_stage1_protocol.json)；200epoch指标/最佳间隔记录：[epoch_validation_metrics.csv](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/stage1/epoch_validation_metrics.csv)。

## 3. 新统计与相同 VS 规则

根据用户确认，旧统计文件保持不动，仅替换统计来源。新μ、方差和epsilon全部从**本轮Clean最佳Backbone对10847张LT的官方eval transform特征**计算；类别统计没有使用验证或TEST。统计冻结SHA256 `e81a4897a965d642f832bd6429e0e5a7d136725daaa069d9ded96f87ab64aef9`。

保持原定义：GAP64、float64统计；类内variance_ddof0；保存ddof1诊断；pooled=sum(n×var0)/(10847−100)；lambda=5/(n+5)；v_reg=(1−lambda)var0+lambda×pooled；epsilon=1e−6×mean(global LT per-channel variance ddof1)=4.26226893569e-07。

在线评分 `S=v_reg/(v_reg+(mu_h−f_t_current_augmented_GAP)^2+epsilon)`；Uniform全64、High Top48、Low Bottom48，使用相同Gumbel基础随机数不放回取16个。rho0.25；每次完整复制供体对应8×8通道，标签始终为BS接收者。供体n<20或评分极差≤1e−12时使用同一Uniform掩码；非有限评分停止。IS类别分布没有过滤。

核心数学函数 [reused_vs_math.py](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/reused_vs_math.py) 与前轮逐字相同；改动diff见 [online_policy.adaptation.diff](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/online_policy.adaptation.diff)（独立三组接口与Clean统计路径，评分/排序/Top48/fallback规则相同）。统计来源与提取正确性：[clean_statistics_provenance.json](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/clean_statistics_provenance.json)。Clean最佳分类器的缓存验证预测逐项复现；StageI与StageII的cudnn设置沿用各自原配置，记录logits/CE的数值差异。

## 4. 固定第10epoch主比较

Medium在此表指原编号分组36–70；Few按LT频次<20定义。单位%。

| Method | Overall | Head | Medium（编号） | Tail | Few |
|---|---:|---:|---:|---:|---:|
| Uniform-PerSample | 47.0707 | 60.4286 | 50.8571 | 26.3793 | 27.0000 |
| High-VS | 46.9192 | 62.7143 | 48.7143 | 25.6897 | 26.0000 |
| Low-VS | 47.3737 | 59.7143 | 51.5714 | 27.4138 | 28.0000 |

相对同一Clean Uniform的差值，单位pp：

| Method | ΔOverall | ΔHead | ΔMedium（编号） | ΔTail | ΔFew |
|---|---:|---:|---:|---:|---:|
| High-VS | -0.1515 | +2.2857 | -2.1429 | -0.6897 | -1.0000 |
| Low-VS | +0.3030 | -0.7143 | +0.7143 | +1.0345 | +1.0000 |

按训练频次及Macro99：

| Method | Many (>100) | Medium (20–100) | Few (<20) | Macro99 |
|---|---:|---:|---:|---:|
| Uniform-PerSample | 61.3235 | 50.4286 | 27.0000 | 47.0707 |
| High-VS | 63.6765 | 48.5714 | 26.0000 | 46.9192 |
| Low-VS | 60.4412 | 51.2857 | 28.0000 | 47.3737 |

## 5. 最佳验证epoch（描述性）

每组只按同一验证Overall选择，平局最早；主要结论使用固定epoch10。

| Method | Best epoch | Overall | Head | Medium（编号） | Tail | Few |
|---|---:|---:|---:|---:|---:|---:|
| Uniform-PerSample | 3 | 47.2222 | 61.0000 | 50.4286 | 26.7241 | 27.3333 |
| High-VS | 9 | 46.9697 | 62.8571 | 48.8571 | 25.5172 | 25.8333 |
| Low-VS | 9 | 47.4242 | 59.8571 | 51.4286 | 27.5862 | 28.1667 |

完整逐类结果：[per_class_epoch10.csv](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/per_class_epoch10.csv)；全部29个Tail类的变化：[tail_class_changes_epoch10.csv](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/tail_class_changes_epoch10.csv)。每类20张，一张图对应该类5pp变化，class0留空，不将缺失类当成0%平均。配对得失图像数：[paired_changed_images_epoch10.csv](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/paired_changed_images_epoch10.csv)；完整epoch与指标：[clean_stage2_metrics.csv](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/clean_stage2_metrics.csv)和各arm的epoch_validation_metrics.csv。

## 6. 采样组成、实际掩码与公平性

三组独立加载同一个Clean最佳权重及fc，空SGD state；没有加载另一组StageII模型。共850个batch/组、108470个融合样本/组。沿用原seed123的BS/IS计划与Gumbel随机流；训练增强原RandomCrop32 padding4、随机翻转和Normalize，private CPU seed123+300009，IS先增强、BS后增强。评分与掩码不消耗增强流。

每batch在SGD前核查：BS/IS索引与标签、增强图像、冻结特征、增强状态、环境CPU/CUDA随机状态、Gumbel、当前GAP、S和全部三组掩码，均与Uniform逐项SHA匹配。[pairing_verification.json](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/pairing_verification.json)验证每组850个配对batch。所有掩码16个唯一通道；[sampling_masks.npz](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/sampling_masks.npz)保存全部计划与三组实际掩码。预检查还验证选中通道逐位复制供体、未选通道逐位保留接收者、标签不变和原池化/fc调用一致。

以下平均所有10epochs的实际选择；V是**收缩方差和**，B是**平方均值偏移和**，ratio=V/(V+B+epsilon)，不把该比值解释成最终分类收益。

| Method | V_selected | B_selected | variance fraction | mean selected S | mean overlap Uniform /16 |
|---|---:|---:|---:|---:|---:|
| Uniform-PerSample | 3.717814 | 10.095437 | 0.305399 | 0.531702 | 16.0000 |
| High-VS | 3.633123 | 3.252773 | 0.553224 | 0.650824 | 12.1136 |
| Low-VS | 3.820990 | 13.241554 | 0.249736 | 0.407275 | 12.1084 |

掩码完全相同率、组合覆盖、fallback和逐通道频率见 [sampling_diagnostics.json](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/sampling_diagnostics.json)、[channel_selection_frequency.csv](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/channel_selection_frequency.csv)、[selected_variation_shift.csv](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/selected_variation_shift.csv)。

全部Backbone可学习参数、BN affine、running_mean/running_var/num_batches_tracked在每epoch和保存模型中逐位保持Clean最佳状态，只更新fc。各组status记录classifier每参数最大变化和BB/BN检查；[stage2_dryrun_verification.json](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/stage2_dryrun_verification.json)验证实际全通道复制和参数更新。每组最佳/最终checkpoint重新加载，仅在同一1980张TRAIN验证缓存上复核，指标完全匹配各组checkpoint_validation.json。

## 7. 时间与显存

| Method | 正式训练秒 | 相对Uniform秒 | peak allocated MiB | peak reserved MiB |
|---|---:|---:|---:|---:|
| Uniform-PerSample | 103.91 | +0.00 | 78.98 | 98.00 |
| High-VS | 103.85 | -0.06 | 79.45 | 116.00 |
| Low-VS | 103.45 | -0.46 | 79.45 | 116.00 |

各组都计算三组掩码以核对输入与评分，计时包括线上增强、特征、评分、融合和SGD；差值会包含系统负载与执行顺序波动，不能当成纯算法额外成本。[timing_and_memory.csv](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/timing_and_memory.csv)保存分项计时及峰值；StageI对应显存见其epoch表。PyTorch张量峰值不等于nvidia-smi整个GPU占用。

## 8. 判断与限制

1. **High-VS是否优于Uniform？** 固定epoch10 ΔOverall=-0.1515pp，ΔTail=-0.6897pp，ΔFew=-1.0000pp；分别按指标判断，不能只看Overall。
2. **Low-VS是否优于Uniform？** 固定epoch10 ΔOverall=+0.3030pp，ΔTail=+1.0345pp，ΔFew=+1.0000pp。
3. **之前Low优势趋势是否保留？** 之前同99类协议seed123的ΔOverall约+0.3535pp、ΔTail约+1.5517pp；本轮复现了固定第10epoch Overall的正向趋势。新基座在不使用TEST选checkpoint时仍观察到的正向差值可作为进一步支持，但单次结果不能证明过去收益完全独立于选择偏差。若优势消失，本次也无法区分基座选择、seed42初始化、验证集复用和随机波动的作用。
4. **Tail/Few是否稳定提升？** Low在10个epoch中Tail正差值9/10、Few正差值9/10；High为Tail 0/10、Few 0/10。这些epoch高度相关，只是轨迹描述；只有一个新基座和一个StageII seed，**没有跨种子稳定性证据**。

本轮同时改变基座训练seed（旧123→新42）和StageI epoch选择数据，因此旧/新差值不是“只移除TEST选模”的因果估计。当前三组内部的共享控制可靠，仍受固定单基座、单StageII随机流、1980张验证规模、class0缺失及历史验证集反复使用影响。“Clean”仅说明新基座和统计的训练/选模数据链没有依赖旧TEST所选模型，不能将该验证集称为新独立最终测试集，也不能提前宣称提高官方TEST或论文成绩。

## 9. Checkpoint与复现

| Arm / kind | Epoch | 文件 | SHA256 |
|---|---:|---|---|
| Uniform-PerSample / best | 3 | [model_best.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/arms/Uniform-PerSample/checkpoints/model_best.pth.tar) | `0ee67f8bd3bcfeaf3e991fc5b97882e5524712a750390af6fa966d92c096d76d` |
| Uniform-PerSample / final | 10 | [model_final.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/arms/Uniform-PerSample/checkpoints/model_final.pth.tar) | `9b14053acdfc9b04036da629d133d0fa69cc2197d623348aa8b5821e73fe52c1` |
| High-VS / best | 9 | [model_best.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/arms/High-VS/checkpoints/model_best.pth.tar) | `a88ef0e4ffe3f58f8a50f39acc85a94efe9b7258bb88b595ba7976ab5eab8a5d` |
| High-VS / final | 10 | [model_final.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/arms/High-VS/checkpoints/model_final.pth.tar) | `87b8804d1bd1c67068657b2186d2ef3ce414c5891c56bfe28e2fbf6d6b0ac727` |
| Low-VS / best | 9 | [model_best.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/arms/Low-VS/checkpoints/model_best.pth.tar) | `56668ccb641e6d5682123cd4a7dcdede896777870ee819f9acf20fdf130551f9` |
| Low-VS / final | 10 | [model_final.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/arms/Low-VS/checkpoints/model_final.pth.tar) | `1cb077ba6ccbda645b1026d6851f93f29bc9cd1bf1a27230aaca3997ff9cd82b` |

全部模型、日志与统计都在独立目录。命令在该目录执行，使用保存源码和现有冻结输入；入口拒绝覆盖已完成训练，重放须另建独立实验目录。

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONHASHSEED='42'
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u prepare_clean.py
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u clean_stage1.py dryrun
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u clean_stage1.py train
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u build_clean_statistics.py
$env:PYTHONHASHSEED='123'
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u clean_stage2.py dryrun
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u clean_stage2.py train
& 'D:\app\anaconda\envs\pytorch\python.exe' -B -u report_clean_results.py
```

[source_lock.json](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/source_lock.json)、[stage2_source_lock.json](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/stage2_source_lock.json)、两套source_snapshot及diff保存源码；[preflight.json](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/preflight.json)、[protocol.json](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/protocol.json)保存数据与完整配置；stage1_full_console.log、stage2_full_console.log保存完整控制台日志。历史完整性：[historical_verification.json](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/historical_verification.json)。没有运行Original-H2T、Shuffled-VS、SMART、其他供体训练或额外seed。

本任务完成后停止，[STOP_COMPLETED.json](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/STOP_COMPLETED.json)锁定本轮结束状态。
