# H2T Variation–Shift Guided Fusion：Stage II seed123

完成时间：2026-10-13T17:50:02+08:00。五组 **各10 epochs、850次SGD更新**；固定BN；100维logits；仅既有1980张99类验证集。主指标固定epoch10，未用各组最佳epoch拼接。

## 1. 五个问题的回答

1. **High-VS相对Uniform：本种子未优于Uniform。** epoch10 Overall -0.1010pp，Tail -1.0345pp，Few -1.1667pp，净正确图片 -2。

2. **High-VS相对Shuffled：本种子未优于Shuffled。** Overall -0.1010pp，Tail -1.0345pp，Few -1.1667pp。

3. **本轮较低均值偏移的High没有改善Overall/Tail/Few。** 它提高Head/Many、降低Medium/Tail/Few；Low-VS名称指低S，本轮具有更大的B且出现相反头尾权衡。未观察到“lower mean shift带来更好主要泛化指标”的收益证据。这里同时改变被选通道身份与迁移组成，不能识别纯粹B下降的因果效应。判别仅根据新训练Classifier的固定epoch10验证结果，未依据旧DR即时Margin挑策略。

4. **融合过温和/增强难度下降只是候选解释。** High在线B显著较低，训练CE更低而验证CE更高，符合较温和融合的描述；但训练/验证输入分布不同，也没有只改变B的干预，不能确认“过温和造成泛化下降”或过拟合的因果解释。

5. **是否值得再跑两种子：** Low-VS具有正向Overall/Tail/Few单种子点估计，值得在确认后按原协议完整配对复核124/125，重点检验Low的头尾权衡；仅净多7张总体图像，尚不能认定有效。 本轮只123，尚无种子方差、总体优势或最终官方测试证据；已停止等待确认。

## 2. 固定epoch10结果

单位为%；Medium-ID与Medium-n是两个不同定义：

| Method | Overall | Head | Medium-ID | Tail | Many | Medium-n | Few | Macro | val CE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original-H2T | 48.2828 | 64.0000 | 50.5714 | 26.5517 | 64.7059 | 51.1429 | 26.3333 | 48.2828 | 2.0720 |
| Uniform-PerSample | 48.2323 | 64.0000 | 50.5714 | 26.3793 | 64.5588 | 51.2857 | 26.1667 | 48.2323 | 2.0725 |
| High-VS | 48.1313 | 66.5714 | 48.5714 | 25.3448 | 67.2059 | 49.4286 | 25.0000 | 48.1313 | 2.1028 |
| Low-VS | 48.5859 | 62.8571 | 51.4286 | 27.9310 | 63.3824 | 52.0000 | 27.8333 | 48.5859 | 2.0675 |
| Shuffled-VS | 48.2323 | 64.1429 | 50.4286 | 26.3793 | 64.8529 | 51.0000 | 26.1667 | 48.2323 | 2.0724 |


![固定第10epoch](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/figures/02_fixed_epoch10.png)

配对差值（准确率单位pp，图片为新增正确/新增错误/净正确）：

| 比较 | Overall | Head | Med-ID | Tail | Many | Med-n | Few | 总体图片 | Tail图片 | Few图片 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-A | -0.0505 | +0.0000 | +0.0000 | -0.1724 | -0.1471 | +0.1429 | -0.1667 | 7/8/-1 | 0/1/-1 | 0/1/-1 |
| C-B | -0.1010 | +2.5714 | -2.0000 | -1.0345 | +2.6471 | -1.8571 | -1.1667 | 31/33/-2 | 5/11/-6 | 5/12/-7 |
| D-B | +0.3535 | -1.1429 | +0.8571 | +1.5517 | -1.1765 | +0.7143 | +1.6667 | 19/12/+7 | 10/1/+9 | 11/1/+10 |
| C-E | -0.1010 | +2.4286 | -1.8571 | -1.0345 | +2.3529 | -1.5714 | -1.1667 | 30/32/-2 | 5/11/-6 | 5/12/-7 |
| C-D | -0.4545 | +3.7143 | -2.8571 | -2.5862 | +3.8235 | -2.5714 | -2.8333 | 42/51/-9 | 5/20/-15 | 5/22/-17 |


B−A包含掩码共享粒度与随机实现差异；C−B为High评分池作用；D−B为Low；C−E检查真实评分与通道绑定相对打乱；C−D比较两个评分方向。A与C/D/E之间不能将所有差异归因VS评分。

逐验证raw ID的新旧预测、每个变化图片、100类准确率与全部五个对比差值见 [paired_predictions_epoch10.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/paired_predictions_epoch10.csv)、[paired_changed_images_epoch10.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/paired_changed_images_epoch10.csv)、[per_class_epoch10.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/per_class_epoch10.csv)、[paired_epoch10_contrasts.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/paired_epoch10_contrasts.csv)。缺失class0的逐类准确率留空，预测class0照常记错。

## 3. 锁定配置与评估口径

| 项目 | 配置 |
| --- | --- |
| Stage I | epoch198，原CE+MixUp200epochs，ResNet32；同一Backbone/Classifier初始化 |
| Stage I SHA256 | 0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e |
| 训练集 | 原10847张CIFAR100LT IF100；原raw IDs与顺序；无新划分 |
| 冻结 | 全部Conv、BN affine、running_mean/var/counter固定；仅原fc100×64+b训练 |
| SGD | base_lr0.1 / momentum0.9 / decay0.0005 / nesterovFalse；新空momentum |
| scheduler | 原cos factor0.2；实际epoch1 lr0.02，epoch10 lr0.000489434837 |
| 预算 | 5组×10epochs；128batch，85步/epoch，末95；每组850更新 |
| 损失与比例 | 原CE、不重加权、不MixUp、无额外Loss；rho0.25，16/64完整8×8通道 |
| 统计 | 原VS审计文件逐字节复制；eval μ、v_reg、epsilon、n冻结 |
| 统计SHA256 | 695aa8fb861a06f615d4ec24782545e92d83aa1f6533b8218c853314cfa287bf |
| 在线S | 每次实际增强receiver的原GAP64；float64复用原评分函数 |
| 收缩/回退 | lambda5/(n+5)及原v_pool；n<20使用精确Uniform mask；Top48保持 |
| seed | 仅123；不自动124/125 |
| 验证 | 1980图像，class1..99各20，class0缺失；保留100logits |
| split SHA256 | c8579feef60d7705472ec00aeac85fa00a66c243761e0a8d76c462c663286ca1 |
| 验证raw ID SHA256 | 46f557118f83617d1b685d62ba5a5715b512e71c9fd4647dd9ba35cd7e3f535e |


Head有效35类/700图，Medium-ID35/700，Tail29/580；Many34/680，Medium-n35/700，Few30/600；Overall=99类Macro。Many n>100、Medium-n20..100、Few<20；HeadID0..35、Medium-ID36..70、Tail71..99。未换分组，未使用官方测试选Checkpoint或方法。

训练/验证raw ID交集0，图像内容hash交集0；原划分和验证银行从之前封存副本复制并校验。验证来源图像未参加Stage I或StageII训练，原Stage I最佳epoch曾依据官方test选择，既有验证也有历史机制评估，结果不应当作全新未暴露终检。

[training_config.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/training_config.json)、[configuration_comparison.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/configuration_comparison.json)、[effective_source_config.yaml](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/effective_source_config.yaml)、[split_audit.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/split_audit.json)、[validation_class_protocol.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/validation_class_protocol.csv) 保存完整配置及类ID。配置冲突检查结果为空。

## 4. 在线评分和真实训练路径

每batch先执行原顺序IS图像增强、BS图像增强，再冻结Backbone各前向一次。**用于S的GAP来自随后同次融合的receiver张量**，没有重新增强或第二次Backbone提取，不使用原eval/augA/augB固定接收特征或审计固定mask。重新池化同一张量只用于一致性断言。

`S=v_reg/[v_reg+(mu_h−GAP(receiver_aug))²+epsilon]` 复用原审计函数源代码；mu/v_reg/epsilon/n完全冻结且数组readonly。v_reg按原lambda5/(n+5)与原pooled within-class目标逐元素重新核对一致，但不重新估计统计量。均值/方差描述eval有限训练集合，不把它们冒充完整训练增强分布的总体参数。

B/C/D/E每draw用同一Gumbel64噪声：B全部64池；C高S48；D低S48；E按原固定(raw receiver ID,donor class ID)置换S绑定后高48。候选内取噪声Top16，等权无放回。E不重排真实特征。n<20的30类每epoch305供体图像保留全部，三引导组精确回退B的mask；同类融合不删除。

RNG独立：BS123、IS offset100003、增强300009、原A通道400012、Gumbel700021、Shuffled830029。10epochs BS/IS与原规则重放核对；B基础noise和A共享randperm状态沿用以前锁定流；首epochShuffled绑定与VS审计相同，但**实际选中池/掩码由新在线S动态生成**。

Dry-run使用真实LT首128张paired draws，含3个n<20供体；五组分别从Stage I初始化，classifier前后向/一次SGD更新pass，参数变化只在fc；全部Backbone/BN与输入标签逐位不变，原H2T与外部共享mask融合逐元素相同，16个唯一通道完整复制验证通过。在线S与真实增强GAP字面公式相同，且与fixed eval评分有差异，防止误用缓存。

[dryrun_verification.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/dryrun_verification.json)；正式组重新调用原models()、重新load Stage I和空SGD，丢弃dry-run更新，未续接旧StageII。

五组4250步loss、logits和classifier梯度全部finite；各组850/850 batch的索引、图像、增强、冻结feature map、在线GAP、在线S、B/C/D/E全部在线mask哈希、基础噪声与CPU/CUDA RNG均相同。采样策略使用不同mask导致分类器梯度不同是预期。[pairing_verification.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/pairing_verification.json)。

每epoch和每个保存Checkpoint核对93个Backbone learned tensors及93个BN buffers（31套running_mean、var、counter）逐位不变；counter保持16830；classifier仅2个tensor/6500参数更新。冻结统计文件的SHA也保持不变。A/B的最终classifier及全部epoch准确率与上一轮SMART中的同设置基线逐位/逐值复现，见 [baseline_reproduction_checks.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/baseline_reproduction_checks.json)。

## 5. 全部逐epoch结果

| 组 | epoch | Overall=Macro | Head | Med-ID | Tail | Many | Med-n | Few | train CE | val CE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original-H2T | 1 | 47.6263 | 61.4286 | 50.5714 | 27.4138 | 61.7647 | 51.4286 | 27.1667 | 1.41576 | 2.09373 |
| Original-H2T | 2 | 48.0808 | 62.8571 | 50.0000 | 27.9310 | 63.3824 | 50.7143 | 27.6667 | 1.23922 | 2.07590 |
| Original-H2T | 3 | 48.3333 | 63.8571 | 49.8571 | 27.7586 | 64.2647 | 50.5714 | 27.6667 | 1.19215 | 2.07496 |
| Original-H2T | 4 | 47.8788 | 63.4286 | 49.5714 | 27.0690 | 64.1176 | 50.0000 | 27.0000 | 1.16298 | 2.07346 |
| Original-H2T | 5 | 47.6768 | 62.7143 | 50.8571 | 25.6897 | 63.3824 | 51.4286 | 25.5000 | 1.13185 | 2.07315 |
| Original-H2T | 6 | 48.7374 | 64.2857 | 51.0000 | 27.2414 | 64.8529 | 51.4286 | 27.3333 | 1.11942 | 2.07416 |
| Original-H2T | 7 | 48.5354 | 64.1429 | 50.8571 | 26.8966 | 64.5588 | 51.4286 | 27.0000 | 1.09601 | 2.07338 |
| Original-H2T | 8 | 48.4343 | 64.1429 | 50.5714 | 26.8966 | 64.5588 | 51.4286 | 26.6667 | 1.10926 | 2.07155 |
| Original-H2T | 9 | 48.2323 | 64.1429 | 50.2857 | 26.5517 | 64.8529 | 50.8571 | 26.3333 | 1.09138 | 2.07216 |
| Original-H2T | 10 | 48.2828 | 64.0000 | 50.5714 | 26.5517 | 64.7059 | 51.1429 | 26.3333 | 1.07535 | 2.07203 |
| Uniform-PerSample | 1 | 47.8788 | 61.4286 | 51.2857 | 27.4138 | 61.7647 | 52.0000 | 27.3333 | 1.40495 | 2.09048 |
| Uniform-PerSample | 2 | 47.8283 | 63.0000 | 49.2857 | 27.7586 | 63.5294 | 50.0000 | 27.5000 | 1.23224 | 2.07648 |
| Uniform-PerSample | 3 | 48.1818 | 63.5714 | 49.7143 | 27.7586 | 63.9706 | 50.4286 | 27.6667 | 1.19072 | 2.07237 |
| Uniform-PerSample | 4 | 48.0808 | 63.5714 | 50.1429 | 26.8966 | 64.2647 | 50.5714 | 26.8333 | 1.16578 | 2.07386 |
| Uniform-PerSample | 5 | 47.8788 | 63.5714 | 50.4286 | 25.8621 | 64.1176 | 51.1429 | 25.6667 | 1.13048 | 2.07406 |
| Uniform-PerSample | 6 | 48.3838 | 63.5714 | 50.4286 | 27.5862 | 64.2647 | 51.0000 | 27.3333 | 1.11507 | 2.07484 |
| Uniform-PerSample | 7 | 48.2323 | 63.5714 | 50.8571 | 26.5517 | 64.1176 | 51.4286 | 26.5000 | 1.07524 | 2.07319 |
| Uniform-PerSample | 8 | 48.7374 | 64.4286 | 51.0000 | 27.0690 | 65.0000 | 51.7143 | 26.8333 | 1.10340 | 2.07159 |
| Uniform-PerSample | 9 | 48.3333 | 64.0000 | 50.5714 | 26.7241 | 64.5588 | 51.2857 | 26.5000 | 1.08500 | 2.07247 |
| Uniform-PerSample | 10 | 48.2323 | 64.0000 | 50.5714 | 26.3793 | 64.5588 | 51.2857 | 26.1667 | 1.07645 | 2.07255 |
| High-VS | 1 | 46.9192 | 64.2857 | 48.1429 | 24.4828 | 64.8529 | 49.0000 | 24.1667 | 0.79098 | 2.11132 |
| High-VS | 2 | 47.6768 | 64.4286 | 48.1429 | 26.8966 | 65.1471 | 48.7143 | 26.6667 | 0.64289 | 2.09776 |
| High-VS | 3 | 47.6263 | 65.0000 | 48.1429 | 26.0345 | 65.5882 | 48.8571 | 25.8333 | 0.61366 | 2.09888 |
| High-VS | 4 | 47.4242 | 65.0000 | 47.1429 | 26.5517 | 65.7353 | 47.8571 | 26.1667 | 0.58876 | 2.10078 |
| High-VS | 5 | 47.0707 | 64.8571 | 47.7143 | 24.8276 | 65.5882 | 48.4286 | 24.5000 | 0.57035 | 2.10152 |
| High-VS | 6 | 47.7273 | 65.5714 | 48.7143 | 25.0000 | 66.3235 | 49.2857 | 24.8333 | 0.55614 | 2.10196 |
| High-VS | 7 | 48.1818 | 66.5714 | 48.7143 | 25.3448 | 67.2059 | 49.5714 | 25.0000 | 0.54380 | 2.10139 |
| High-VS | 8 | 48.1313 | 66.5714 | 48.4286 | 25.5172 | 67.2059 | 49.2857 | 25.1667 | 0.54371 | 2.10151 |
| High-VS | 9 | 48.1313 | 66.5714 | 48.5714 | 25.3448 | 67.2059 | 49.4286 | 25.0000 | 0.53919 | 2.10282 |
| High-VS | 10 | 48.1313 | 66.5714 | 48.5714 | 25.3448 | 67.2059 | 49.4286 | 25.0000 | 0.53095 | 2.10278 |
| Low-VS | 1 | 47.7778 | 60.7143 | 51.4286 | 27.7586 | 61.0294 | 52.1429 | 27.6667 | 1.73405 | 2.08947 |
| Low-VS | 2 | 48.1313 | 62.2857 | 50.4286 | 28.2759 | 62.6471 | 51.1429 | 28.1667 | 1.55535 | 2.07475 |
| Low-VS | 3 | 47.8788 | 62.1429 | 49.8571 | 28.2759 | 62.5000 | 50.5714 | 28.1667 | 1.51238 | 2.06939 |
| Low-VS | 4 | 48.1818 | 61.5714 | 51.0000 | 28.6207 | 62.2059 | 51.2857 | 28.6667 | 1.47929 | 2.07168 |
| Low-VS | 5 | 48.1818 | 61.5714 | 51.7143 | 27.7586 | 62.0588 | 52.4286 | 27.5000 | 1.45619 | 2.07206 |
| Low-VS | 6 | 48.5354 | 62.1429 | 51.8571 | 28.1034 | 62.6471 | 52.4286 | 28.0000 | 1.42510 | 2.07323 |
| Low-VS | 7 | 48.5859 | 62.8571 | 51.8571 | 27.4138 | 63.2353 | 52.4286 | 27.5000 | 1.38726 | 2.06904 |
| Low-VS | 8 | 48.3333 | 62.4286 | 51.4286 | 27.5862 | 62.9412 | 52.1429 | 27.3333 | 1.42186 | 2.06744 |
| Low-VS | 9 | 48.3838 | 62.5714 | 51.4286 | 27.5862 | 63.0882 | 52.1429 | 27.3333 | 1.40595 | 2.06755 |
| Low-VS | 10 | 48.5859 | 62.8571 | 51.4286 | 27.9310 | 63.3824 | 52.0000 | 27.8333 | 1.39583 | 2.06753 |
| Shuffled-VS | 1 | 47.9798 | 61.5714 | 51.2857 | 27.5862 | 61.9118 | 52.0000 | 27.5000 | 1.40066 | 2.08920 |
| Shuffled-VS | 2 | 48.0808 | 63.2857 | 49.5714 | 27.9310 | 63.8235 | 50.2857 | 27.6667 | 1.23240 | 2.07487 |
| Shuffled-VS | 3 | 48.3838 | 63.5714 | 50.2857 | 27.7586 | 63.9706 | 51.0000 | 27.6667 | 1.18391 | 2.07051 |
| Shuffled-VS | 4 | 48.2828 | 63.7143 | 50.1429 | 27.4138 | 64.4118 | 50.5714 | 27.3333 | 1.15870 | 2.07234 |
| Shuffled-VS | 5 | 47.7273 | 63.4286 | 50.2857 | 25.6897 | 64.1176 | 50.8571 | 25.5000 | 1.13218 | 2.07507 |
| Shuffled-VS | 6 | 48.3838 | 63.0000 | 51.1429 | 27.4138 | 63.5294 | 51.8571 | 27.1667 | 1.10203 | 2.07444 |
| Shuffled-VS | 7 | 48.2323 | 63.8571 | 50.7143 | 26.3793 | 64.4118 | 51.4286 | 26.1667 | 1.06823 | 2.07273 |
| Shuffled-VS | 8 | 48.5354 | 64.5714 | 50.7143 | 26.5517 | 65.1471 | 51.4286 | 26.3333 | 1.10232 | 2.07210 |
| Shuffled-VS | 9 | 48.3838 | 64.1429 | 50.7143 | 26.5517 | 64.8529 | 51.2857 | 26.3333 | 1.07861 | 2.07237 |
| Shuffled-VS | 10 | 48.2323 | 64.1429 | 50.4286 | 26.3793 | 64.8529 | 51.0000 | 26.1667 | 1.07171 | 2.07236 |


[epoch_validation_metrics.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/epoch_validation_metrics.csv) 还含LR、time/memory、在线组成和Uniform重合。[channel_selection_frequency.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/channel_selection_frequency.csv) 保存每epoch每通道实际计数。各组batches.csv有完整850步Loss、时间与全部配对hash。

![逐epoch验证曲线](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/figures/01_validation_curves.png)

## 6. 实际在线掩码与组成

以下108470个在线draw统计均含预声明回退；B为均值偏移平方代理，V为冻结正则化供体方差之和。比例是每draw的 `V/(V+B+epsilon)` 再平均，不是平均V除以平均(V+B)。

| 策略 | 平均共享Uniform/16 | mask完全相同率 | 所选原绑定S均值 | V均值 | B均值 | V份额均值 | 回退率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original-H2T | 4.0043 | 0.0000% | 0.5294 | 3.7913 | 10.5047 | 0.3024 | 2.8118% |
| Uniform-PerSample | 16.0000 | 100.0000% | 0.5291 | 3.7894 | 10.4980 | 0.3022 | 2.8118% |
| High-VS | 12.1104 | 3.2599% | 0.6477 | 3.6959 | 3.3542 | 0.5507 | 2.8118% |
| Low-VS | 12.1129 | 3.2627% | 0.4046 | 3.8984 | 13.7534 | 0.2468 | 2.8118% |
| Shuffled-VS | 12.1120 | 3.2691% | 0.5293 | 3.7906 | 10.4870 | 0.3023 | 2.8118% |


原A共享mask只是粒度参考；其“回退率”列是遇到低n条件的draw频率，A仍使用原共享mask，没有切换为B。High更高S主要降低B，不等于增加绝对V；Low是更大偏移，Shuffled在原始通道绑定下的平均评分/组成接近Uniform。实际mask不是几乎不变的弱概率干预。

![实际在线组成](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/figures/03_online_composition.png)

![训练和验证CE](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/figures/04_losses.png)

训练Loss是在融合后的BS输入上、验证CE是在未融合的固定验证图像上，二者不可当作相同输入分布的直接泛化gap估计。更低训练CE与更小B仅描述较温和融合，不证明哪个因素导致准确率变化。

## 7. 描述性最佳epoch和独立Checkpoint

仅按验证Overall选Best，平局取早。主结果仍epoch10；每组best与final都已重新加载再验证，100 logits、指标精确一致，不覆盖旧文件：

| 策略 | Best epoch | Best Overall | 最终epoch10 Overall | 最佳模型 | 最终模型 |
| --- | --- | --- | --- | --- | --- |
| Original-H2T | 6 | 48.7374 | 48.2828 | [best](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/arms/Original-H2T/checkpoints/model_best.pth.tar) | [final](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/arms/Original-H2T/checkpoints/model_final.pth.tar) |
| Uniform-PerSample | 8 | 48.7374 | 48.2323 | [best](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/arms/Uniform-PerSample/checkpoints/model_best.pth.tar) | [final](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/arms/Uniform-PerSample/checkpoints/model_final.pth.tar) |
| High-VS | 7 | 48.1818 | 48.1313 | [best](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/arms/High-VS/checkpoints/model_best.pth.tar) | [final](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/arms/High-VS/checkpoints/model_final.pth.tar) |
| Low-VS | 7 | 48.5859 | 48.5859 | [best](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/arms/Low-VS/checkpoints/model_best.pth.tar) | [final](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/arms/Low-VS/checkpoints/model_final.pth.tar) |
| Shuffled-VS | 8 | 48.5354 | 48.2323 | [best](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/arms/Shuffled-VS/checkpoints/model_best.pth.tar) | [final](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/arms/Shuffled-VS/checkpoints/model_final.pth.tar) |


全部模型、源Stage I及统计SHA在 [checkpoint_manifest.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/checkpoint_manifest.csv)、[checkpoint_integrity.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/checkpoint_integrity.json)；各组checkpoint_validation.json记录独立reload结果。

## 8. 训练时间与显存

| 策略 | 训练s | 相对B额外s | 在线评分s | 峰值allocated MiB | reserved MiB |
| --- | --- | --- | --- | --- | --- |
| Original-H2T | 105.849 | +0.074 | 3.678 | 78.97 | 98.00 |
| Uniform-PerSample | 105.775 | +0.000 | 3.934 | 79.44 | 116.00 |
| High-VS | 105.157 | -0.618 | 3.975 | 79.44 | 116.00 |
| Low-VS | 105.953 | +0.178 | 4.061 | 79.44 | 116.00 |
| Shuffled-VS | 105.475 | -0.300 | 3.787 | 79.44 | 116.00 |


![时间统计](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/figures/05_measured_time.png)

计时含在线增强、两次冻结Backbone前向、在线S和全部4个逐样本mask的配对审计、classifier更新及hash；不含epoch后验证/保存。五组共同执行在线S/四mask用于公平审计，因此评分成本是本次实现的统一审计开销；组间time差不能简单归因算法计算成本。PyTorchallocated/reserved不是整卡占用，桌面/驱动另占显存。全程RTX5070 Laptop8GB，无OOM。

## 9. 文件、差异与原文件保护

新目录：`D:\xiangmuxunlian\新\H2T_repro_audit\variation_shift_stage2_runs\vs_stage2_seed123`；分支 `experiment/variation-shift-stage2-seed123`。保留五组完整日志、10个best/final Checkpoint、50个逐epoch预测/100logits NPZ、实际在线掩码与逐sample V/B/比例NPZ、输入pair hash、训练配置和源码快照。

**训练主循环复用**上一轮经过验证的固定BN training.py，SGD、LR、增强、冻结检查、checkpoint和验证流程保持；改动限于在线VS掩码/统计与validation CE输出。[training.py.adaptation.diff](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/training.py.adaptation.diff)、[common.py.adaptation.diff](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/common.py.adaptation.diff) 可对照 reference_trainer 的未修改副本；原VS数学助手函数逐字复制，未导入无训练审计入口或其禁止训练guard。

48个兼容runtime文件原样复制；methods.py与官方文件字节相同。[source_lock.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/source_lock.json)、[git_training_provenance.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/git_training_provenance.json)、[experiment.diff](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/experiment.diff)、[historical_verification.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/historical_verification.json)、[delivery_manifest.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/delivery_manifest.json) 保存输入/源码及所有历史文件校验。原StageI、DR、SMART、VS无训练审计、原H2T及旧报告未覆盖。

主要原始数据：[online_scores_A_float64.npy](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/online_scores_A_float64.npy)（A记录全部在线S，五组逐batch S/GAP hash相同）；[sampling_and_masks.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/sampling_and_masks.npz)（只有BS/IS和原A状态，无固定VS masks）；[gumbel_noise_float64.npy](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/gumbel_noise_float64.npy)；[shuffled_permutations.npy](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/shuffled_permutations.npy)；[class_feature_statistics.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/class_feature_statistics.npz)；各组actual_masks.npz、sampling_composition.npz与batches.csv。

复查命令（从只复制脚本的全新独立目录执行；当前目录已完成，拒绝重复覆盖/追加）：

```powershell
$py = "D:\app\anaconda\envs\pytorch\python.exe"
$run = "D:\xiangmuxunlian\新\H2T_repro_audit\variation_shift_stage2_runs\vs_stage2_seed123"
& $py "$run\build_sources.py"
& $py "$run\adapt_trainer.py"
& $py "$run\vs_stage2.py" prepare
& $py "$run\vs_stage2.py" dryrun
& $py "$run\lock_source.py"
& $py "$run\vs_stage2.py" train
& $py "$run\analyze_results.py"
& $py "$run\finalize_results.py"
```

## 10. 当前证据与停止

实际迁移组成改变及单种子分类结果均已观察，但“均值偏移低→更好泛化”的因果结论没有被证明。High/Low改变通道身份、各维扰动和信息结构；λ统计先验、n<20回退、固定StageI与99类验证限制仍存在。旧冻结分类器即时Margin只提供此前机制背景，未用于本轮策略判断。

每类只有20个验证图像，缺失class0；前后多个实验与监控已有验证集暴露。报告全部组和固定epoch10结果，没有挑epoch、调Top48、修改评分/正则化或增加Loss。单种子净图片差不能替代多种子方差、独立验证或官方终检。

[STOP_AFTER_SEED123.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_stage2_runs/vs_stage2_seed123/STOP_AFTER_SEED123.json) 标记五组全部完成并停止。没有seed124/125、官方测试集评估、Backbone重训或额外方案；下一阶段需明确确认。
