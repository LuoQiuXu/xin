# H2T 供体选择 Stage II 控制实验结果

完成时间：2026-10-12T23:28:08+08:00。实验目录：[selector_fixed_bn_20261012](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012>)。

本轮在独立 Git 实验分支中实现 Random-of-4、Low-D、Mid-D、High-D。只训练 Stage II 分类器，原 Stage I、参考分类器、历史官方复现和 BN 审计文件均保留。结果用于阶段性机制审计，不把 99 类验证准确率与论文 100 类测试准确率直接比较。

**主要结果：** 12 次独立训练（4策略×3种子，每次10epochs）完成；原预声明首轮规则无入选者，四组后续补充重复的范围扩展已披露。固定第10epoch的三种子验证ΔOverall相对Random-of-4为 Low-D -0.08 pp、Mid-D -0.10 pp、High-D -0.10 pp。当前没有稳定额外Overall收益的证据，也不能据此认定某种供体必然有害或普遍无效。

## 1. 验证集与评估口径

采用用户确认的 99 类未见验证方案：从原始 CIFAR-100 **训练文件**中、原 LT 子集之外固定抽取类别 1–99 各 20 张，共 1,980 张；缺失类别 ID=0。第 0 类 500 张原训练图像均在锁定的 Stage I 子集中，无法提供真正未见的同类验证样本。

仍使用原 100 类分类器和 **100 维 logits**。argmax 遍历所有输出，预测到第 0 类照常计错；没有删除、屏蔽或重新归一化任何输出类别。Macro 为实际有验证图像的 99 类准确率平均；每类样本数相同，故 Macro 与 Overall 数值相同。

训练集 / 验证集原始索引交集=0；供体池 / 验证集交集=0；参考分类器训练样本与验证集交集=0；与训练图像完全相同的内容交集=0。LT 索引按原始 `rand_number=0` 重建，图像字节 SHA256 与历史审计的训练子集 SHA256 一致，固定文件可供后续消融复用。

| 项目 | 记录 |
| --- | --- |
| 固定索引文件 | [split_indices.npz](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/split_indices.npz>) |
| 索引文件 SHA256 | c8579feef60d7705472ec00aeac85fa00a66c243761e0a8d76c462c663286ca1 |
| 验证索引数组 SHA256 | 46f557118f83617d1b685d62ba5a5715b512e71c9fd4647dd9ba35cd7e3f535e |
| 划分种子 | 20261012 |
| 验证样本数 / 缺失类别 | 1980 / [0] |
| 各类样本数 | ID0:0；ID1–99:每类20 |
| 类计数及分组 | [validation_class_protocol.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/validation_class_protocol.csv>) |
| 逐样本原始ID | [fixed_sample_ids.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/fixed_sample_ids.csv>) |
| 索引与重叠审计 | [split_audit.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/split_audit.json>) |
| 后续消融固定协议 | [VALIDATION_PROTOCOL.md](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/VALIDATION_PROTOCOL.md>) |

Many/Medium/Few 按 **LT 训练频次**定义；为衔接旧报告，另附官方 YAML 的 Head/Medium/Tail 口径，两个 Medium 不应混用。

| 口径 | 定义 | 总类别数 | 验证有效类别数 |
| --- | --- | --- | --- |
| Many | 训练频次 >100 | 35 | 34 |
| Medium | 20≤训练频次≤100 | 35 | 35 |
| Few | 训练频次 <20 | 30 | 30 |
| Head（官方） | ID [0,36) | 36 | 35 |
| Medium（官方） | ID [36,71) | 35 | 35 |
| Tail（官方） | ID [71,100) | 29 | 29 |

## 2. 锁定模型、训练配置与策略

| 项目 | 配置 |
| --- | --- |
| Stage I | epoch198，[model_best.pth.tar](<D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428/ckps/model_best.pth.tar>) |
| Stage I SHA256 | 0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e |
| 参考分类器 | 固定 BN DR 第10 epoch，[model_final.pth.tar](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/runtime/saved/cifar100_if100_stage2_fixed_bn_20261011_2255_dr_202610112255/ckps/model_final.pth.tar>) |
| 参考 SHA256 | 1b054d8b45753c25dedd9897e3052e5f33c73b0c38b47acaa277c0f61c50283d |
| 数据 / Backbone | CIFAR-100-LT IF100，原10847图，ResNet-32 |
| 预算 | 每组10 epochs，128 batch，85更新/epoch，最后batch95，共850次更新 |
| SGD | base lr0.1；momentum0.9；weight_decay0.0005；nesterov=False |
| 调度 / CE | 直接编译官方cosine调度函数，lr_factor0.2；原CrossEntropyLoss，无重加权/MixUp |
| BN / rho | 固定BN及affine，冻结全部Backbone参数；rho=0.25，64通道替换16 |
| 种子 | [123, 124, 125]；只改变Stage II数据/增强/掩码流，Stage I和参考模型固定 |
| 完整预声明协议 | [protocol.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/protocol.json>) |
| 原YAML有效配置 | [effective_source_config.yaml](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/effective_source_config.yaml>) |

每个分类器均独立加载 Stage I 的分类器参数，并创建空状态 SGD；不从参考 DR 或另一策略的 Stage II 参数初始化。参考分类器仅用于冻结评分。训练时所有 Backbone 前向均 `eval()+no_grad()`，参考分类器 `eval()+requires_grad_(False)`；仅训练分类器为 `train()`。

BS 沿用均匀类别后均匀类内实例的规则。IS 供体类别取自每 epoch 对完整 LT 索引的随机排列，精确保留原 IS 每类频次；然后在该类内有放回抽 K=4 个候选。**这里改变了原 H2T 的类内实例选择方式**。保留同类供体，不使用验证图像构建候选。

在线应用源码中的 RandomCrop、RandomHorizontalFlip、ToTensor 和原 CIFAR-10 normalization；并非复用第一阶段固定增强特征库。每个 batch 的四个候选使用同一原始通道掩码，全部接收者也共享该掩码。融合调用未修改的 `methods.H2T`，标签一直为 BS 接收者标签。

D = 冻结参考分类器的原 Margin − 融合 Margin。Low-D 取最小 D；High-D 取最大 D；Mid-D 在排序第2/第3位均匀随机选择，**不是离0最近的候选**；Random-of-4 独立均匀抽候选位置。排序稳定，平局按候选位置处理。

## 3. 先导实验：99 类独立验证集

准确率单位 %；主要对照固定第 10 epoch。最佳 checkpoint 只按验证 Overall 选 epoch，出现并列时保留最早 epoch。

| 策略 | 最佳epoch | 最佳Overall | 第10 Overall | 第10 Macro | Many | Medium | Few |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random-of-4 | 6 | 48.74 | 48.33 | 48.33 | 64.85 | 51.00 | 26.50 |
| Low-D | 6 | 48.48 | 48.18 | 48.18 | 65.29 | 50.43 | 26.17 |
| Mid-D | 6 | 48.59 | 48.28 | 48.28 | 64.85 | 51.00 | 26.33 |
| High-D | 6 | 48.64 | 48.28 | 48.28 | 64.12 | 51.57 | 26.50 |

官方类别区间的固定第 10 epoch 验证结果：

| 策略 | Head（35有效类） | Medium（35类） | Tail（29类） |
| --- | --- | --- | --- |
| Random-of-4 | 64.14 | 50.43 | 26.72 |
| Low-D | 64.71 | 49.71 | 26.38 |
| Mid-D | 64.14 | 50.43 | 26.55 |
| High-D | 63.57 | 50.86 | 26.72 |

所有运行最佳验证 checkpoint 的完整类别指标（单位%，仅按验证Overall选择epoch）：

| seed | 策略 | epoch | Overall | Macro | Many | Medium | Few | Head | Medium（官方） | Tail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 123 | Random-of-4 | 6 | 48.74 | 48.74 | 64.71 | 51.43 | 27.50 | 64.00 | 51.14 | 27.41 |
| 123 | Low-D | 6 | 48.48 | 48.48 | 65.15 | 50.43 | 27.33 | 64.43 | 50.14 | 27.24 |
| 123 | Mid-D | 6 | 48.59 | 48.59 | 64.26 | 51.43 | 27.50 | 63.57 | 51.14 | 27.41 |
| 123 | High-D | 6 | 48.64 | 48.64 | 63.82 | 52.00 | 27.50 | 63.29 | 51.71 | 27.24 |
| 124 | Random-of-4 | 7 | 48.54 | 48.54 | 65.00 | 51.14 | 26.83 | 64.29 | 50.57 | 27.07 |
| 124 | Low-D | 6 | 48.69 | 48.69 | 65.00 | 51.29 | 27.17 | 64.29 | 50.71 | 27.41 |
| 124 | Mid-D | 2 | 48.64 | 48.64 | 64.41 | 51.14 | 27.83 | 63.71 | 50.57 | 28.10 |
| 124 | High-D | 4 | 48.69 | 48.69 | 63.24 | 52.14 | 28.17 | 62.86 | 51.14 | 28.62 |
| 125 | Random-of-4 | 6 | 48.69 | 48.69 | 64.12 | 51.29 | 28.17 | 63.43 | 51.00 | 28.10 |
| 125 | Low-D | 3 | 48.38 | 48.38 | 63.97 | 50.71 | 28.00 | 63.57 | 50.29 | 27.76 |
| 125 | Mid-D | 6 | 48.48 | 48.48 | 63.82 | 51.29 | 27.83 | 63.14 | 51.00 | 27.76 |
| 125 | High-D | 6 | 48.48 | 48.48 | 63.38 | 52.00 | 27.50 | 62.86 | 51.43 | 27.59 |

首轮固定epoch10预测到缺失真实类别ID0的图像数（照常计错）：Random-of-4:9；Low-D:9；Mid-D:9；High-D:9。

验证集基线（不同于历史测试数字）：

| 模型 | Overall/Macro | Many | Medium | Few |
| --- | --- | --- | --- | --- |
| StageI | 40.96 | 71.47 | 41.14 | 6.17 |
| frozen_DR_reference | 47.83 | 67.65 | 48.71 | 24.33 |

![先导学习曲线](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/figures/01_pilot_validation_curves.png>)

## 4. 预声明选择与多种子复核

预声明的首轮规则：只使用 seed123 固定 epoch10 验证 Overall；相对 Random-of-4 增益严格 >0 的策略进入复核。没有依据最佳测试 epoch、Tail 测试准确率或已有论文成绩筛选。选择锁 [selection_lock.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/selection_lock.json>) 在测试图像读取前保存。

| 策略 | 先导第10验证ΔOverall（pp） | 原预声明规则入选 |
| --- | --- | --- |
| Random-of-4 | +0.00 | 对照 |
| Low-D | -0.15 | 否 |
| Mid-D | -0.05 | 否 |
| High-D | -0.05 | 否 |

**首轮没有策略入选。** 三个差值仅相当于1–3张验证图，因此随后将四组全部纳入seed124、125的补充重复实验，检查训练随机性。这次复核范围扩展发生在首轮之后，[supplemental_replication_protocol.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/supplemental_replication_protocol.json>) 在新增训练前保存；它不是预注册阳性确认，也没有改变原筛选阈值或“无入选”选择锁。所有组采用同一模型、配置、预算、划分和随机流控制，所有结果均报告。

| seed | 策略 | 最佳epoch | 最佳Overall | 第10 Overall | Macro | Many | Medium | Few |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 124 | Random-of-4 | 7 | 48.54 | 48.48 | 48.48 | 64.56 | 51.43 | 26.83 |
| 124 | Low-D | 6 | 48.69 | 48.59 | 48.59 | 65.00 | 51.00 | 27.17 |
| 124 | Mid-D | 2 | 48.64 | 48.48 | 48.48 | 64.85 | 51.43 | 26.50 |
| 124 | High-D | 4 | 48.69 | 48.23 | 48.23 | 63.68 | 51.43 | 27.00 |
| 125 | Random-of-4 | 6 | 48.69 | 48.28 | 48.28 | 64.12 | 51.43 | 26.67 |
| 125 | Low-D | 3 | 48.38 | 48.08 | 48.08 | 64.41 | 50.86 | 26.33 |
| 125 | Mid-D | 6 | 48.48 | 48.03 | 48.03 | 63.68 | 51.14 | 26.67 |
| 125 | High-D | 6 | 48.48 | 48.28 | 48.28 | 63.38 | 52.14 | 26.67 |

相对同种子 Random-of-4 的固定 epoch10 验证差值，单位pp：

| seed | 策略 | ΔOverall | ΔMacro | ΔMany | ΔMedium | ΔFew |
| --- | --- | --- | --- | --- | --- | --- |
| 123 | Low-D | -0.15 | -0.15 | +0.44 | -0.57 | -0.33 |
| 123 | Mid-D | -0.05 | -0.05 | +0.00 | +0.00 | -0.17 |
| 123 | High-D | -0.05 | -0.05 | -0.74 | +0.57 | +0.00 |
| 124 | Low-D | +0.10 | +0.10 | +0.44 | -0.43 | +0.33 |
| 124 | Mid-D | +0.00 | +0.00 | +0.29 | +0.00 | -0.33 |
| 124 | High-D | -0.25 | -0.25 | -0.88 | +0.00 | +0.17 |
| 125 | Low-D | -0.20 | -0.20 | +0.29 | -0.57 | -0.33 |
| 125 | Mid-D | -0.25 | -0.25 | -0.44 | -0.29 | +0.00 |
| 125 | High-D | +0.00 | +0.00 | -0.74 | +0.71 | +0.00 |

仅两个新种子的均值与样本标准差（未混入用于选策略的先导种子）：

| 策略 | 复核ΔOverall 均值±SD | ΔMany 均值 | ΔMedium 均值 | ΔFew 均值 | Overall正增益种子数 |
| --- | --- | --- | --- | --- | --- |
| Low-D | -0.05 ± 0.21 | +0.37 | -0.50 | +0.00 | 1/2 |
| Mid-D | -0.13 ± 0.18 | -0.07 | -0.14 | -0.17 | 0/2 |
| High-D | -0.13 ± 0.18 | -0.81 | +0.36 | +0.08 | 0/2 |

三种子固定 epoch10 验证准确率（含初筛seed123；补充重复范围在首轮后决定）：

| 策略 | 种子数 | Overall 均值±SD | Many 均值±SD | Medium 均值±SD | Few 均值±SD |
| --- | --- | --- | --- | --- | --- |
| Random-of-4 | 3 | 48.37 ± 0.11 | 64.51 ± 0.37 | 51.29 ± 0.25 | 26.67 ± 0.17 |
| Low-D | 3 | 48.28 ± 0.27 | 64.90 ± 0.45 | 50.76 ± 0.30 | 26.56 ± 0.54 |
| Mid-D | 3 | 48.27 ± 0.23 | 64.46 ± 0.68 | 51.19 ± 0.22 | 26.50 ± 0.17 |
| High-D | 3 | 48.27 ± 0.03 | 63.73 ± 0.37 | 51.71 ± 0.38 | 26.72 ± 0.25 |

三种子成对差值的均值与样本标准差（单位pp）：

| 策略 | ΔOverall 均值±SD | ΔMany 均值 | ΔMedium 均值 | ΔFew 均值 |
| --- | --- | --- | --- | --- |
| Low-D | -0.08 ± 0.16 | +0.39 | -0.52 | -0.11 |
| Mid-D | -0.10 ± 0.13 | -0.05 | -0.10 | -0.17 |
| High-D | -0.10 ± 0.13 | -0.78 | +0.43 | +0.06 |

![成对复核增益](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/figures/02_paired_validation_gains.png>)

只有两个新的 Stage II 随机种子，不能宣称已证明统计显著或跨 Backbone 随机种子稳定。固定同一验证集复核反映训练流敏感性，不是新的独立数据验证。首轮选择与之后补充覆盖的数据依赖均需披露，故将新种子均值与包含先导的三种子均值分开。

## 5. 配对、冻结和掩码证据

| seed | 比较策略 | 匹配batch数 | 核验内容 |
| --- | --- | --- | --- |
| 123 | Random-of-4, Low-D, Mid-D, High-D | 850/850 | BS/候选索引、完整增强输入、特征图、掩码、增强状态、CPU/CUDA RNG逐batch一致 |
| 124 | Random-of-4, Low-D, Mid-D, High-D | 850/850 | BS/候选索引、完整增强输入、特征图、掩码、增强状态、CPU/CUDA RNG逐batch一致 |
| 125 | Random-of-4, Low-D, Mid-D, High-D | 850/850 | BS/候选索引、完整增强输入、特征图、掩码、增强状态、CPU/CUDA RNG逐batch一致 |

每个 epoch 比较全部 Backbone state_dict 的字节，包含 Conv、BN affine、running_mean、running_var、num_batches_tracked；所有组均逐位不变。31个 BN 计数器始终为16830。参考分类器权重和bias全程哈希不变且无梯度；训练分类器 weight 和 bias 两个张量每 epoch 均发生变化，所有850次更新的CE及梯度均有限。保存后独立重新加载最佳及最终 checkpoint 验证，并再次检查BN字节。

四组共享事先生成的计划文件，在线增强、通道、Random选择和Mid选择采用隔离随机数流。原 H2T 的 CPU randperm 在 `fork_rng(devices=[])` 中使用保存的独立状态，调用后恢复环境 RNG。Low/Mid/High 的候选D矩阵逐位一致。记录的是实际逐batch输入字节和状态匹配，而不只是声明同seed。

实验使用官方 `cudnn.benchmark=False / deterministic=False`。配对哈希验证本次输入和特征一致；不承诺其他驱动/设备重放时有完全相同的浮点结果。

候选抽样摘要（每seed108470个接收样本；同类和重复候选均保留）：

| seed | 同类供体组数 | 重复候选索引组数 | 比例 |
| --- | --- | --- | --- |
| 123 | 1120 | 5551 | 5.12% |
| 124 | 1080 | 5649 | 5.21% |
| 125 | 1056 | 5599 | 5.16% |

同一原图重复候选在本轮可使用不同在线增强，因此重复索引不一定产生相同特征或D。不同候选共享同一掩码；并未筛选通道或调整rho。完整在线D、排序选择位置及采样计划均保留。

## 6. 实测耗时与显存

各策略为独立顺序运行，下面是实际10-epoch训练墙钟时间，含原在线增强、K=4候选特征提取、融合、评分、SGD及配对诊断；不含验证和保存checkpoint。所有组都提取四个候选，因此 Random-of-4 也有K=4提取成本。该对照不等于经过推理裁剪、只提取已随机选中一个供体的原始H2T速度。

| seed | 策略 | 训练秒 | 相对Random额外秒 | 倍数 | 参考评分秒 | 峰值allocated MiB | 峰值reserved MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 123 | Random-of-4 | 209.04 | +0.00 | 1.00 | 0.00 | 89.12 | 116.00 |
| 123 | Low-D | 213.67 | +4.63 | 1.02 | 2.72 | 97.71 | 134.00 |
| 123 | Mid-D | 235.76 | +26.73 | 1.13 | 3.38 | 97.71 | 134.00 |
| 123 | High-D | 233.89 | +24.85 | 1.12 | 2.27 | 97.71 | 134.00 |
| 124 | Random-of-4 | 228.93 | +0.00 | 1.00 | 0.00 | 89.12 | 116.00 |
| 124 | Low-D | 233.79 | +4.86 | 1.02 | 2.25 | 97.71 | 134.00 |
| 124 | Mid-D | 235.16 | +6.23 | 1.03 | 3.37 | 97.71 | 134.00 |
| 124 | High-D | 234.56 | +5.63 | 1.02 | 2.23 | 97.71 | 134.00 |
| 125 | Random-of-4 | 231.06 | +0.00 | 1.00 | 0.00 | 90.07 | 134.00 |
| 125 | Low-D | 234.09 | +3.03 | 1.01 | 2.22 | 97.71 | 134.00 |
| 125 | Mid-D | 235.66 | +4.60 | 1.02 | 3.41 | 97.71 | 134.00 |
| 125 | High-D | 234.93 | +3.87 | 1.02 | 2.21 | 97.71 | 134.00 |

合计分类器训练 120 epochs、2760.54 秒；含每epoch验证及checkpoint保存的运行合计 2776.36 秒（不含共享采样计划构造、模块初始化、预热、最终重新加载验证）。顺序运行受到GPU功耗、温度、系统负载和CPU数据增强影响，耗时差不能全部视作Margin计算的纯开销；多次融合及逐batch复制正确性检查也计入训练时间。评分秒单列，[training_times.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/training_times.csv>) 保存各阶段明细。系统级 nvidia-smi 使用量含桌面进程，和本脚本allocated/reserved不同。

![训练耗时](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/figures/03_measured_training_time.png>)

## 7. 锁定后的一次终端测试（100类，描述性）

四组先导及复核全部完成、[confirmation_lock.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/confirmation_lock.json>) 保存后，才第一次在本轮读取官方test图像与标签。在全部10000图、每类100图上重新加载本轮每个最佳（按验证选择）与最终checkpoint评估。验证所有保存的Backbone张量与同一冻结Backbone逐位一致后，复用一次完整测试特征提取；每个分类器从其checkpoint独立加载。测试结果没有参与候选、策略或epoch选择，此后不再训练。

以下为100类的**另一评估协议**，不与99类验证数字直接比较，也不作论文Table/README复现成绩对比。历史 Stage I 与参考来源曾按官方测试选模型，所以这些测试数字属于描述性外部检查，不能声称整个研究历史都未触碰测试集。

| seed | 策略 | Checkpoint | epoch | Overall | Head | Medium（官方） | Tail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 123 | High-D | best | 6 | 47.06 | 61.19 | 49.34 | 26.76 |
| 123 | High-D | final | 10 | 47.38 | 61.81 | 49.94 | 26.38 |
| 123 | Low-D | best | 6 | 46.96 | 61.83 | 48.80 | 26.28 |
| 123 | Low-D | final | 10 | 47.11 | 62.22 | 49.00 | 26.07 |
| 123 | Mid-D | best | 6 | 46.94 | 61.36 | 48.91 | 26.66 |
| 123 | Mid-D | final | 10 | 47.25 | 61.92 | 49.60 | 26.21 |
| 123 | Random-of-4 | best | 6 | 46.99 | 61.44 | 48.94 | 26.69 |
| 123 | Random-of-4 | final | 10 | 47.29 | 61.97 | 49.74 | 26.10 |
| 124 | High-D | best | 4 | 47.37 | 61.11 | 49.51 | 27.72 |
| 124 | High-D | final | 10 | 47.17 | 61.50 | 49.34 | 26.76 |
| 124 | Low-D | best | 6 | 47.26 | 62.00 | 49.43 | 26.34 |
| 124 | Low-D | final | 10 | 47.01 | 62.36 | 48.66 | 25.97 |
| 124 | Mid-D | best | 2 | 47.04 | 61.11 | 49.49 | 26.62 |
| 124 | Mid-D | final | 10 | 47.10 | 61.67 | 48.97 | 26.76 |
| 124 | Random-of-4 | best | 7 | 47.22 | 61.94 | 49.29 | 26.45 |
| 124 | Random-of-4 | final | 10 | 47.18 | 61.89 | 49.23 | 26.45 |
| 125 | High-D | best | 6 | 47.20 | 61.11 | 49.66 | 26.97 |
| 125 | High-D | final | 10 | 47.25 | 60.97 | 49.94 | 26.97 |
| 125 | Low-D | best | 3 | 46.89 | 61.14 | 48.80 | 26.90 |
| 125 | Low-D | final | 10 | 47.08 | 61.86 | 49.06 | 26.34 |
| 125 | Mid-D | best | 6 | 47.10 | 61.31 | 49.20 | 26.93 |
| 125 | Mid-D | final | 10 | 47.20 | 61.22 | 49.57 | 26.93 |
| 125 | Random-of-4 | best | 6 | 47.15 | 61.14 | 49.54 | 26.90 |
| 125 | Random-of-4 | final | 10 | 47.26 | 61.50 | 49.43 | 26.97 |

按训练频次的 Many/Medium/Few、Macro 及全部测试预测见 [terminal_test_metrics.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/terminal_test_metrics.csv>) 和 [results.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/test_evaluations/results.json>)。

## 8. Checkpoint 与独立实验分支

每次运行均保存完整日志、配置、每epoch验证logits/预测、最佳和第10epoch模型以及SGD状态。完整文件路径和SHA256见 [checkpoint_manifest.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/checkpoint_manifest.csv>)。

| seed | 策略 | 最佳epoch | 最佳checkpoint | 最终checkpoint |
| --- | --- | --- | --- | --- |
| 123 | Random-of-4 | 6 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/123/arms/Random-of-4/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/123/arms/Random-of-4/checkpoints/model_final.pth.tar>) |
| 123 | Low-D | 6 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/123/arms/Low-D/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/123/arms/Low-D/checkpoints/model_final.pth.tar>) |
| 123 | Mid-D | 6 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/123/arms/Mid-D/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/123/arms/Mid-D/checkpoints/model_final.pth.tar>) |
| 123 | High-D | 6 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/123/arms/High-D/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/123/arms/High-D/checkpoints/model_final.pth.tar>) |
| 124 | Random-of-4 | 7 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/124/arms/Random-of-4/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/124/arms/Random-of-4/checkpoints/model_final.pth.tar>) |
| 124 | Low-D | 6 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/124/arms/Low-D/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/124/arms/Low-D/checkpoints/model_final.pth.tar>) |
| 124 | Mid-D | 2 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/124/arms/Mid-D/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/124/arms/Mid-D/checkpoints/model_final.pth.tar>) |
| 124 | High-D | 4 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/124/arms/High-D/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/124/arms/High-D/checkpoints/model_final.pth.tar>) |
| 125 | Random-of-4 | 6 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/125/arms/Random-of-4/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/125/arms/Random-of-4/checkpoints/model_final.pth.tar>) |
| 125 | Low-D | 3 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/125/arms/Low-D/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/125/arms/Low-D/checkpoints/model_final.pth.tar>) |
| 125 | Mid-D | 6 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/125/arms/Mid-D/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/125/arms/Mid-D/checkpoints/model_final.pth.tar>) |
| 125 | High-D | 6 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/125/arms/High-D/checkpoints/model_best.pth.tar>) | [epoch10](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/seeds/125/arms/High-D/checkpoints/model_final.pth.tar>) |

当前工作区上层不是有效Git仓库，因此在本次独立实验目录建立专用本地Git仓库，从审计通过的48文件快照提交 `compat-baseline`，再建立 `experiment/donor-margin-fixed-bn`。没有修改或覆盖原运行目录。官方归档commit标识为 `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`，来源为此前下载归档的commit标记，未将其冒充本地独立仓库的提交。

48个兼容源码文件与历史 runtime_manifest SHA256 完全相同，实验改动全部在外部脚本和新配置。历史 815 个文件前后SHA256一致，包括上一轮供体机制审计、特征缓存及所有旧 checkpoint/日志/报告。实验 Diff：[experiment.diff](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/experiment.diff>)；源码提交标识：[git_provenance.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/git_provenance.json>)。

## 9. 结论与限制

原预声明先导规则没有阳性入选者；以下为全部策略补充重复的描述性结论，不能事后改写成预注册确认。

Low-D：新种子验证第10epoch ΔOverall 均值 -0.05 pp，1/2种子为正；ΔFew均值 +0.00 pp。新种子平均结果未优于同种子Random-of-4。

Mid-D：新种子验证第10epoch ΔOverall 均值 -0.13 pp，0/2种子为正；ΔFew均值 -0.17 pp。新种子平均结果未优于同种子Random-of-4。

High-D：新种子验证第10epoch ΔOverall 均值 -0.13 pp，0/2种子为正；ΔFew均值 +0.08 pp。新种子平均结果未优于同种子Random-of-4。

本次配对消除了组间BS样本、供体候选、增强、通道、初始参数、BN更新和参考模型变化的差异，主要干预为**从同一候选集合选择不同供体实例**。若出现本轮组间差异，可在当前控制条件下联系供体选择规则；不能推断Low-D必然安全或High-D必然有害，也不能泛化为原始shift_bn=True设置的收益。

仍受固定Stage I/参考分类器适配性、只有两个新增训练种子、有限验证规模、类0缺失、同一验证集用于多策略选择、共享batch通道和候选依赖影响。机制差异明显并不保证泛化提升；类别内D排序的效果与参考模型定义有关。本轮没有训练Backbone、供体网络、通道选择网络，也没有自适应rho。

## 10. 可复查文件与运行命令

| 内容 | 文件 |
| --- | --- |
| 实现 | [selector_stage2.py](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/selector_stage2.py>) |
| 补充多种子复核 | [replicate_stage2.py](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/replicate_stage2.py>) |
| 汇总/绘图/报告 | [summarize_stage2.py](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/summarize_stage2.py>) |
| 预声明配置 | [protocol.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/protocol.json>) |
| 训练/验证固定索引 | [split_indices.npz](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/split_indices.npz>) |
| 每epoch指标 | [epoch_validation_metrics.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/epoch_validation_metrics.csv>) |
| 成对增益 | [paired_validation_gains.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/paired_validation_gains.csv>) |
| 候选选择统计 | [online_choice_statistics.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/online_choice_statistics.csv>) |
| 训练耗时 | [training_times.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/training_times.csv>) |
| 完整汇总 | [suite_summary.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/suite_summary.json>) |
| 交付文件校验 | [delivery_manifest.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_stage2_runs/selector_fixed_bn_20261012/delivery_manifest.json>) |

以下命令用于复查流程；同一输出目录拒绝覆盖训练或重复测试，复现时须复制源码到新的独立目录并保持固定split协议。

```powershell
Set-Location -LiteralPath 'D:\xiangmuxunlian\新\H2T_repro_audit\donor_stage2_runs\selector_fixed_bn_20261012'
$env:PYTHONHASHSEED='123'
$pythonExe='D:\app\anaconda\envs\pytorch\python.exe'
& $pythonExe selector_stage2.py prepare
& $pythonExe selector_stage2.py pilot
& $pythonExe replicate_stage2.py
& $pythonExe selector_stage2.py test
& $pythonExe selector_stage2.py verify
& $pythonExe summarize_stage2.py
```

本轮完成后停止，不根据终端测试结果追加实验或更改策略。
