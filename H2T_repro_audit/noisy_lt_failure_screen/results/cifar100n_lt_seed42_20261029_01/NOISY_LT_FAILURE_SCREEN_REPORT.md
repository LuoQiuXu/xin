# CIFAR-100N-LT IF100 原始 H2T 失效初筛

**最终决策：NO-GO。有效性核验：PASS。** 本轮观察到 Few 相对收益衰减，但没有满足预定的双seed稳定失效门槛。固定场景在本轮停止，不触发独立Stage I seed确认，不开展新模块、噪声等级搜索或PI-H2T实验。

## 科学结论

定义 `D_g(regime)=Acc(H2T)-Acc(NoFusion)`，`Interaction_g=D_g(noisy)-D_g(clean)`。

- Few 的 D(clean) 为 +3.6667/+4.5000 pp；D(noisy) 为 −0.1667/+1.0000 pp；交互为 −3.8333/−3.5000 pp（S2 seed124/125）。两次交互都达到≤−2pp，但第二个seed的Noisy H2T高于No-Fusion，严格的合取门槛未通过。
- Noisy/Few 的净改判为 −1/+6 张（分母各600）。单个seed的一张净损失不能写成稳定失败。Noisy/Overall 的 H2T−No-Fusion仍为 +0.2020/+0.2525 pp。
- 有证据支持本次Stage I表示与开发集上的相对收益衰减；没有证实稳定的Few收益反转、原H2T普遍失效或PI-H2T失效。这里没有进行独立Stage I重复、显著性检验或论文级独立TEST验证。
- Clean/Noisy条件同时改变Stage I监督、观测类频、Stage II balanced receiver流与hard targets；结果不能单独归因为标签损失、供体语义、mask、BN或某个融合机制。

提供的 `analysis/summary.json` 使用合并状态 `NO_GO_OR_INCONCLUSIVE`；由于本轮完整运行和有效性核验均通过，最终在 `SCIENTIFIC_DECISION.json` 明确归类为 **NO-GO**，而非缺证据导致的INCONCLUSIVE。未更改原分析程序或判定阈值。

## 数据来源与真实噪声

人工标签直接来自 [CIFAR-N 作者仓库](https://github.com/UCSC-REAL/cifar-10-100n) 的 `data/CIFAR-100_human.pt`，Git blob `5dac9db8af8f7224f982fbac9794c67f60f8bd45`，文件SHA256为 `bd2d80409754d420292d622e15e25248ba21e37d27429efac49c8da723f44394`。NumPy数值类型的受限兼容处理、文件大小与来源URL均记录在 `INPUT_ATTESTATION.json`。

全部50,000项 `clean_label` 与原始 Python TRAIN fine labels 相同，确认使用正确的图像顺序。固定seed0 IF100 LT包含10847图，复用既有1980张clean TRAIN holdout，raw-ID与相同像素跨集合重叠均为0。没有新抽split或读取官方TEST数据。

真实LT错标3902/10847=**35.9730801%**，全量40.2%。Noisy观测类频为14–445，无空类，L1类频变化4012。按任务阈值，clean构造类数Many/Medium/Few为35/35/30，noisy观测类数为39/58/3；IF100描述的是clean基准图像构造，观测noisy最大/最小类频比约31.79，不能仍假设观测标签是IF100。100类明细见 `CLASS_FREQUENCY_AUDIT.csv` 与 `protocol/DATA_AUDIT.json`。

Noisy训练仅载入images-only缓存和 `noisy_train.npz`，类池与loss target均为observed noisy label。Clean真值只用于基准构造、数据核查和最后评价，没有oracle修补采样类别或调整噪声率。

## 原始训练规则与成对公平性

方法证据以 [H2T原论文](https://ojs.aaai.org/index.php/AAAI/article/view/29262/30382) 及冻结的作者实现为准；完整文件/行号、显式适配与环境差异见 `PROTOCOL_AUDIT.md`。实际直接复用原模型、CE、MixUp、采样getitem、train核心、学习率调度及 `methods.H2T`，不是另写一种融合公式。

- Stage I：seed42，两条件从同一个新随机初始化分别200 epochs；ResNet32 OptionA+GAP+带bias100类FC，CE+MixUp(alpha1)，SGD(.1,.9,wd.0002)，batch128，workers0，原5轮warmup及161/181轮开始衰减。两个末轮权重各来自17,000次更新，200轮图像增强与MixUp系数流全部精确一致；Noisy未从Clean训练模型微调。
- Stage II：每条件seed124/125各做No-Fusion/H2T 10 epochs，共8次；同regime+seed共享epoch200 BB与FC、空SGD初始状态；只训练6500个FC参数，原hard receiver CE，SGD(momentum.9,wd.0005)，原cosine首轮有效lr=.02，无MixUp。
- 原作者默认BN-shift=True。两臂均执行receiver→donor两次前向且更新BN，保持骨干参数冻结。H2T调用原始batch-shared16/64完整通道替换(rho=.25)；No-Fusion只跳过替换，并消耗相同未使用mask随机draw以维持后续增强/采样配对，没有少做供体前向。
- 全部4对、3400个批次在分类器更新前精确核对raw-ID、target、增强图像、CPU RNG、receiver/donor特征和mask SHA；4对最终BN哈希均相等。每臂850次FC更新、1700次BB前向，31层BN计数由17000增至18700，骨干参数逐项未变。
- Windows/RTX5070 Laptop 8GB，指定Python3.12解释器，Torch2.14.1+cu132、CUDA13.2、cuDNN92400；FP32无AMP。Stage I/II分别保留作者deterministic=True/False，统一benchmark=False、cudnn TF32=True、matmul TF32=False。现代运行环境及NumPy显式seed适配均已披露，不声称原作者历史逐比特重放。

没有改进donor、mask、loss、融合比例、分类器初始化来源或训练预算，没有训练PIF/PI-H2T。

## 固定末轮结果

全部10个正式模型完成且SHA锁定后才读取验证输入。Stage II仅epoch10各eval一次；(1980,100) FP32有限logits与raw-ID/label逐模型保存本地，未选best。分组固定为clean true LT counts的Many>100、Medium20..100、Few<20，分母为1980/680/700/600。class0缺失，评估99类；Many构造35类但评价仅34类。20样本边界按任务归Medium，与论文Tail≤20口径不同，不能直接等同论文分组指标。

| 条件 | S2 seed | 方法 | Overall | Many | Medium | Few |
|---|---:|---|---:|---:|---:|---:|
| clean | 124 | no_fusion | 45.4545 | 62.5000 | 47.8571 | 23.3333 |
| clean | 124 | h2t | 46.4141 | 60.1471 | 49.7143 | 27.0000 |
| clean | 125 | no_fusion | 45.6061 | 62.5000 | 48.1429 | 23.5000 |
| clean | 125 | h2t | 47.2727 | 60.1471 | 51.2857 | 28.0000 |
| noisy | 124 | no_fusion | 35.3535 | 47.2059 | 35.0000 | 22.3333 |
| noisy | 124 | h2t | 35.5556 | 47.2059 | 35.7143 | 22.1667 |
| noisy | 125 | no_fusion | 34.6970 | 45.8824 | 35.0000 | 21.6667 |
| noisy | 125 | h2t | 34.9495 | 45.4412 | 35.2857 | 22.6667 |

单位均为准确率%。完整32行CSV见 `evaluation/final_metrics.csv`；1000行逐类准确率（含两个Stage I的描述项，缺class0为空）见 `evaluation/per_class_accuracy.csv`，400行逐类净改判见 `evaluation/per_class_reclassifications.csv`。

| S2 seed | 分组 | D(clean), pp | D(noisy), pp | Interaction, pp |
|---:|---|---:|---:|---:|
| 124 | Overall | +0.9596 | +0.2020 | -0.7576 |
| 124 | Many | -2.3529 | +0.0000 | +2.3529 |
| 124 | Medium | +1.8571 | +0.7143 | -1.1429 |
| 124 | Few | +3.6667 | -0.1667 | -3.8333 |
| 125 | Overall | +1.6667 | +0.2525 | -1.4141 |
| 125 | Many | -2.3529 | -0.4412 | +1.9118 |
| 125 | Medium | +3.1429 | +0.2857 | -2.8571 |
| 125 | Few | +4.5000 | +1.0000 | -3.5000 |

效应单位为百分点。没有把交互负值单独当作H2T失败证据，也没有把两个方法在Noisy下共同降分当作H2T特有缺陷。净改判的wrong→right、right→wrong与分母见 `evaluation/paired_reclassifications.csv`。

Stage I epoch200评价仅作描述，发生在全部Stage II训练完成后：

| 条件 | Overall | Many | Medium | Few |
|---|---:|---:|---:|---:|
| clean | 38.5354 | 68.6765 | 36.1429 | 7.1667 |
| noisy | 31.8687 | 54.1176 | 29.2857 | 9.6667 |

## Checkpoint 来源与真实产物

| 固定末轮模型 | Epoch | SHA256 |
|---|---:|---|
| stage1_clean | 200 | `378efb8ac7fba039b3809ec925e87e06818374b6364414113c2a2edeb7604eb1` |
| stage1_noisy | 200 | `2a3a587865314e87397b8e9b5ea958756f69bffefb3657d8576ba6f901bfa0d7` |
| stage2_clean_seed124_no_fusion | 10 | `6045717983d5db54b680d42e1d2f5e50a9c9b4c0a191ce02c7bdd62d1a9bbff3` |
| stage2_clean_seed124_h2t | 10 | `cbc0e88dd53f94f085e7df8e1ae028e09026af2fa93e0573ed823586880a3acb` |
| stage2_clean_seed125_no_fusion | 10 | `235cceb3051c567041b1cd74a496dc82839769943d112fb55c519e764cc2165e` |
| stage2_clean_seed125_h2t | 10 | `709e7d0d9ccfc826c0e51698eb282b160b8dcd2c8ea164752fc3966321460c46` |
| stage2_noisy_seed124_no_fusion | 10 | `c2c3cbf9c959b1483ac4de3dec8150ec1adca816be04164640bbe6bcb4fd9106` |
| stage2_noisy_seed124_h2t | 10 | `cae55660fc6717d9a91d2e02840a99ac01efbd309f76d3f77c273aebedfa80a1` |
| stage2_noisy_seed125_no_fusion | 10 | `56f75eeaab8b2d03573ede71c6121beb7e834c4170ae3493f9ab7f644651f455` |
| stage2_noisy_seed125_h2t | 10 | `c3344557c943192a7dc39837b375fa5b27155bf456338ae8bb385e0039be82ee` |

随机epoch0初始化、干跑权重、输入NPY/NPZ、全部真实logits、原生训练日志和逐batch控制日志只留本地，不写入Git。`RUN_MANIFEST.json` 索引58个local-only文件，共97,063,520 bytes，并记录实际SHA256与路径；其 `checkpoints`、`INITIALIZATION.json`、各run `START.json/RUN.json` 可以逐级复核来源。没有复用任何历史训练checkpoint。

所有新产物均位于 `H2T_repro_audit/noisy_lt_failure_screen/`；结果目录为 `results/cifar100n_lt_seed42_20261029_01/`。数据、权重及primary_cache被既有 `.gitignore` 规则忽略；只提交本轮新代码和小体积报告/核验结果。

## 验证与历史证据边界

CPU契约测试9/9通过，新增与原有Python工具共10个文件语法检查通过（含最后的报告生成器）；正式训练前6个真实单batch干跑通过，权重未复用。`VALIDATION.json` 从持久化权重、850批控制日志和logits重新检查配对/冻结/指标/改判结果，通过全部5类综合门禁。原始作者文件SHA未变，20543项受保护历史文件的路径/大小/mtime_ns元数据与开始快照一致；另对使用的原始TRAIN与历史split执行SHA复核。这不是对全部历史文件做内容哈希的声明。

本轮没有重新解释此前的语义几何NO-GO、IF200固定BN结果或Office-Home不稳定域效应。尤其IF200使用固定BN，本轮使用作者默认BN-shift，协议不同，不能把跨任务差值当作重复验证。上述历史资产只读且未覆盖。

主要限制：单个Stage I seed；两个Stage II seed共享同一表示；固定TRAIN开发验证集曾被历史研究使用且缺class0；无官方独立TEST；真实噪声改变观测类频与采样分布。根因未被单独识别，稳定反转未被证实。完整PI-H2T作者一致性仍未建立，本轮结果不适用于PI-H2T。

**执行决策：NO-GO，结束本固定场景；不追加noise sweep、IF搜索、机制消融或创新模块。**
