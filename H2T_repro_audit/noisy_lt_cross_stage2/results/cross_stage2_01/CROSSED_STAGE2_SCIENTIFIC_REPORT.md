# H2T：Stage I来源 × Stage II标签条件交叉诊断

**结论：Few相对收益衰减的优先干预位置是Stage II的观测标签＋采样条件，并存在强烈的R×S条件依赖。** 不支持用“Stage I噪声表征单独造成衰减”概括本轮Few结果。该结论限于两份固定Stage I源模型、两个Stage II seed及既有开发验证集，不是损失函数单因素因果结论，也不是已验证的新方法。

## 已完成的实验与真实来源

唯一分支 `experiment/noisy-h2t-crossed-stage2-20261029`；开始时已在目标分支，起始HEAD `71bd1e56147d90ad9adf7620a43fa8ca864485de`，工作区干净。本轮只新增R=clean/S=noisy和R=noisy/S=clean的8次Stage II训练，各10 epochs，seed124/125、No-Fusion/H2T各一；**Stage I新增训练为0，原有两个对角单元没有重跑。**

R同时指定Stage I骨干、初始FC及BN缓冲区来源。已实际检查CPU可读性、权重有限性、状态哈希与epoch200，来源SHA精确匹配上一轮：

| Stage I来源 | 固定epoch | SHA256 |
|---|---:|---|
| clean | 200 | `378efb8ac7fba039b3809ec925e87e06818374b6364414113c2a2edeb7604eb1` |
| noisy | 200 | `2a3a587865314e87397b8e9b5ea958756f69bffefb3657d8576ba6f901bfa0d7` |

来源核查见 `SOURCE_AUDIT.json`；新8个模型的来源、最终SHA、路径、耗时及显存见 `CHECKPOINT_INDEX.csv`。没有猜造或重新训练缺失权重，没有改写历史结果。

## 2×2主要结果：Few相对收益

定义 `D(R,S)=Acc(H2T|R,S)-Acc(NoFusion|R,S)`，单位pp。R是Stage I源模型条件，S是Stage II观测标签及balanced receiver池的整体条件。

| R：Stage I来源 | S：Stage II标签/采样 | seed124 | seed125 | 均值 |
|---|---|---:|---:|---:|
| clean | clean | +3.6667 | +4.5000 | +4.0833 |
| clean | noisy | -2.6667 | -2.6667 | -2.6667 |
| noisy | clean | +1.1667 | +3.1667 | +2.1667 |
| noisy | noisy | -0.1667 | +1.0000 | +0.4167 |

Clean R + Noisy S中两个seed均为−2.6667pp（各少16张Few正确分类），提供了S条件会抑制H2T Few相对收益的直接交叉证据。Noisy R + Clean S仍有+1.1667/+3.1667pp收益，不能将噪声Stage I源模型视为必然消除H2T收益。

原全Noisy对角单元的seed125仍为+1.0000pp，保留这一反例；不把本轮交叉条件的负收益改写成全Noisy实际流程已有稳定负迁移。

本轮是相对收益诊断。固定clean R时，S由clean变noisy，H2T Few均值27.5000%→31.3333%，No-Fusion为23.4167%→34.0000%；相对收益下降并不意味着这两个模型的Few绝对准确率在此干预下都下降。四个单元、两个seed的Overall H2T−NoFusion均为正，不支持“全部噪声增强有害”的概括。

## 每个seed及均值：所有频组的H2T−NoFusion

| R | S | seed | Overall D | Many D | Medium D | Few D |
|---|---|---|---:|---:|---:|---:|
| clean | clean | 124 | +0.9596 | -2.3529 | +1.8571 | +3.6667 |
| clean | clean | 125 | +1.6667 | -2.3529 | +3.1429 | +4.5000 |
| clean | clean | mean | +1.3131 | -2.3529 | +2.5000 | +4.0833 |
| clean | noisy | 124 | +1.5657 | +4.1176 | +2.7143 | -2.6667 |
| clean | noisy | 125 | +1.3131 | +3.6765 | +2.4286 | -2.6667 |
| clean | noisy | mean | +1.4394 | +3.8971 | +2.5714 | -2.6667 |
| noisy | clean | 124 | +0.0505 | -1.4706 | +0.5714 | +1.1667 |
| noisy | clean | 125 | +0.0505 | -3.2353 | +0.5714 | +3.1667 |
| noisy | clean | mean | +0.0505 | -2.3529 | +0.5714 | +2.1667 |
| noisy | noisy | 124 | +0.2020 | +0.0000 | +0.7143 | -0.1667 |
| noisy | noisy | 125 | +0.2525 | -0.4412 | +0.2857 | +1.0000 |
| noisy | noisy | mean | +0.2273 | -0.2206 | +0.5000 | +0.4167 |

完整NF/H2T准确率见 `analysis/factorial_accuracy.csv`（32行，每行含两个方法，共64个准确率）；新8臂原始指标在 `crossed_final_metrics.csv`。每单元/频组均值见 `analysis/factorial_cell_means.csv`。提供的分析程序将绝对准确率列命名为 `*_accuracy_pp`，这些列的值实际是准确率%；只有差值及contrast使用pp。

Few与Overall的全部绝对准确率如下，避免仅展示差值：

| R | S | seed | Few NF % | Few H2T % | Overall NF % | Overall H2T % |
|---|---|---:|---:|---:|---:|---:|
| clean | clean | 124 | 23.3333 | 27.0000 | 45.4545 | 46.4141 |
| clean | clean | 125 | 23.5000 | 28.0000 | 45.6061 | 47.2727 |
| clean | noisy | 124 | 33.8333 | 31.1667 | 43.8889 | 45.4545 |
| clean | noisy | 125 | 34.1667 | 31.5000 | 43.9899 | 45.3030 |
| noisy | clean | 124 | 23.8333 | 25.0000 | 38.5859 | 38.6364 |
| noisy | clean | 125 | 23.8333 | 27.0000 | 38.6364 | 38.6869 |
| noisy | noisy | 124 | 22.3333 | 22.1667 | 35.3535 | 35.5556 |
| noisy | noisy | 125 | 21.6667 | 22.6667 | 34.6970 | 34.9495 |

Many/Medium的绝对值完整保留在CSV，没有按类别或seed筛选。新8个模型的800行逐类结果在 `crossed_per_class.csv`，class0的验证分母为0、准确率留空。

## 阶段效应与交互

按预定公式：

```text
R = [(D(noisy,clean)-D(clean,clean))+(D(noisy,noisy)-D(clean,noisy))]/2
S = [(D(clean,noisy)-D(clean,clean))+(D(noisy,noisy)-D(noisy,clean))]/2
I = D(noisy,noisy)-D(noisy,clean)-D(clean,noisy)+D(clean,clean)
对角衰减 = D(noisy,noisy)-D(clean,clean) = R+S
```

I是非加性对照，不应再次加到R+S中充当第三项对角贡献。以下均为pp；seed124的Few R≈−1.8e−15是浮点舍入量，显示为0。

| 分组 | seed | R主效应 | S主效应 | R×S交互 | 对角变化 R+S |
|---|---|---:|---:|---:|---:|
| Overall | 124 | -1.1364 | +0.3788 | -0.4545 | -0.7576 |
| Overall | 125 | -1.3384 | -0.0758 | +0.5556 | -1.4141 |
| Overall | mean | -1.2374 | +0.1515 | +0.0505 | -1.0859 |
| Many | 124 | -1.6176 | +3.9706 | -5.0000 | +2.3529 |
| Many | 125 | -2.5000 | +4.4118 | -3.2353 | +1.9118 |
| Many | mean | -2.0588 | +4.1912 | -4.1176 | +2.1324 |
| Medium | 124 | -1.6429 | +0.5000 | -0.7143 | -1.1429 |
| Medium | 125 | -2.3571 | -0.5000 | +0.4286 | -2.8571 |
| Medium | mean | -2.0000 | +0.0000 | -0.1429 | -2.0000 |
| Few | 124 | +0.0000 | -3.8333 | +5.0000 | -3.8333 |
| Few | 125 | +1.1667 | -4.6667 | +5.0000 | -3.5000 |
| Few | mean | +0.5833 | -4.2500 | +5.0000 | -3.6667 |

Few均值：R=+0.5833pp，S=−4.2500pp，I=+5.0000pp，对角变化−3.6667pp。S的平均负向效应与两个seed的方向一致，R的平均主效应没有显示同样的稳定负向压制。不过I很大，必须看条件效应：

| seed | S效应，固定clean R | S效应，固定noisy R | R效应，固定clean S | R效应，固定noisy S |
|---:|---:|---:|---:|---:|
| 124 | -6.3333 | -1.3333 | -2.5000 | +2.5000 |
| 125 | -7.1667 | -2.1667 | -1.3333 | +3.6667 |

S在两种R、两个seed下都降低Few相对收益；在clean R下抑制更强。R在clean S下为负、在noisy S下为正，说明模型来源效应随S翻转。**+5pp正交互表示两项变化联合时的衰减小于各自单独变化相加，不是“联合协同恶化”的证据。** 可报告S为下一步优先位置，同时保留强条件依赖，不能将平均主效应当作完整机制分解。

该归因也不能推广至所有分组：Overall平均R效应−1.2374pp、S效应+0.1515pp；Medium平均R效应−2.0000pp、S效应0；Many平均S效应+4.1912pp。Few问题与整体准确率退化的阶段关联并不相同。

## 数据背景与解释限度

本轮复用上一轮同一10847张IF100图像、clean/noisy标签及1980张不相交TRAIN holdout；没有新split、噪声等级或TEST。评价按benchmark clean true counts：Many>100、Medium20..100、Few<20，分母1980/680/700/600。class0缺失，评估99类，Many仅34个有验证样本的类。与论文20样本边界的定义差异沿用并披露，不将这些开发集指标称为论文官方Few结果。

训练结束后的独立数据背景核对确认：原30个Few类对应的Noisy observed池共1234张，仅164张真实类别与观测类别匹配；实例加权纯度13.2901%，均匀类采样加权纯度15.8950%（`DATA_CONTEXT.json`）。这说明S确实同时改变监督质量和接收者曝光，不能证明具体哪种H2T内部机制导致结果。

R包括骨干、初始FC与BN来源，并非只改变纯骨干表示；S同时改变targets与采样池，并非只改CE里的label。干净R或干净S使用可信clean训练标签，均为**oracle诊断条件/上界，不是可部署的无oracle抗噪算法**。Noisy S训练窗口仅解码noisy目标并据此建类池；来源预检只对两份标签文件做SHA核查，不将clean目标传入Noisy S训练。

两种Stage II seed共享同一Stage I seed42；没有独立表示重复、显著性检验或独立官方TEST确认。固定TRAIN开发集已被历史研究使用，不据此声称普遍性、论文级稳健性或PI-H2T失效。

## 训练协议、验证及必要的入口补充

- 沿用冻结作者ResNet32 OptionA、GAP+带bias100类FC；Stage II原hard receiver CE，原class-balanced receiver/instance donor，原batch-shared16/64完整通道替换rho=.25；不启用MixUp、PIF、额外loss或其他模块。
- 只训练6500个FC参数，原SGD(momentum=.9,wd=.0005)、cosine10轮、首轮有效lr=.02；batch128、workers0。骨干参数冻结、BN-shift=True，两臂都按receiver→donor做两次前向。No-Fusion只跳过替换并消耗相同未使用mask随机draw，保持后续RNG流。
- 正式运行前CPU测试4/4通过，原3个工具语法编译通过，真实配对dryrun两臂各1次更新且初始状态/控制哈希/最终BN相等。末次新增核验工具也完成语法编译。
- 8臂各850次更新、1700次骨干前向；31层BN从17000到18700。4对共3400个批次精确比较receiver/donor raw-ID、target、增强、CPU RNG、特征及mask SHA，不接受allclose；最终BN缓冲区和整个骨干state逐项相等，骨干参数与各自Stage I完全相同。
- 额外确认固定S/seed/method时，新交叉与旧对角实验的采样、增强、mask流相同（特征随R变化，不要求跨R特征相等）。核对全部末轮logits、32行新指标、800行逐类计数及factorial恒等式；`VALIDATION.json`为PASS。
- 8个正式模型全部完成并锁定SHA后才构建holdout评价；每个epoch10只eval一次，保存(1980,100) FP32有限logits与同一raw-ID/label，没有选best。
- 指定解释器 `D:\app\minicondaaa\envs\xiaoju\python.exe`；实际Torch2.14.1+cu132、CUDA13.2、cuDNN92400、RTX5070 Laptop。FP32无AMP，cudnn deterministic=False/benchmark=False/TF32=True，matmul TF32=False，与旧Stage II一致，逐run JSON记录实际环境。
- 只补充当前入口的来源锁、IO保护、运行计数/耗时/显存和logits落盘，不改变原训练核心。旧源码与旧输出均未编辑；来源数据及权重SHA复核通过，20712项历史文件路径/大小/mtime_ns无变化（这是元数据普查，不是全部内容SHA普查）。

8次正式run内记录的耗时合计608.10s，约10.14分钟；单次74.13–78.64s。PyTorch allocated峰值124.47MiB。合计不含解释器导入、独立预检、评价及报告时间，不等同完整墙钟时长。

## 交付与下一步边界

全部结果在 `H2T_repro_audit/noisy_lt_cross_stage2/results/cross_stage2_01/`：

- `analysis/factorial_accuracy.csv`、`factorial_contrasts.csv`、`factorial_cell_means.csv`、`summary.json`：完整2×2结果和每seed/均值分析。
- `CHECKPOINT_INDEX.csv`、`RUN_MANIFEST.json`、`SOURCE_AUDIT.json`、`LOCAL_ONLY_INDEX.csv`：新模型SHA、真实来源、代码和本地二进制索引。
- 各run `epochs.csv/RUN.json`、`CPU_TEST_RESULTS.txt`、`DRYRUN.json`、`EVALUATION_AUDIT.json`、`VALIDATION.json`：逐轮结果与验证。
- 41个本地文件共32,488,336 bytes，包含真实权重、logits、控制日志和历史元数据快照；大文件不提交，`local_only/`已加入专用ignore。恢复/复核命令见README，不从GitHub重造缺失二进制。

**本轮研究出口：优先Stage II接收者可靠性／融合监督这一处干预位置，并保留R×S条件依赖。** 下一轮最多围绕一个可证伪问题：在真实Noisy标签与原有采样条件下，类别感知的接收者可靠性监督能否恢复原H2T的Few相对收益，并同时报告Overall。这里仅确定位置，不在本轮定义、实现或训练新模块，不搜索donor、mask、融合比例、额外seed或噪声强度。
