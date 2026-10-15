# H2T：SMART 启发的协方差通道实验（seed 123）

完成时间：2026-10-13T15:38:05+08:00。四组各 **10 epochs / 850 次 SGD 更新**。本轮已停止；没有 seed 124/125、额外方案、超参数搜索或官方测试集评估。

## 1. 当前判断

固定第 10 epoch、1980 张验证图像：Cov-High 相对 Uniform-PerSample **+0.0505 pp**，Cov-Low **-0.0505 pp**。目前缺少稳定收益证据：Cov-High 的正向点估计仅净多一张验证图像。若继续，它只有保持同协议作低成本多种子机制复核的有限价值，不能据此推荐协方差引导替代 Uniform。

这里检验的是固定收缩、列余弦代理、T=1、alpha=0.5 的组合。不能外推为所有协方差指导均有效/无效，也不能认定高分通道语义安全。

## 2. SMART 原文与实验代理

详细公式、页码、Softmax 轴与温度证据边界见 [方法核对报告](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/SMART_TO_H2T_METHOD_CHECK.md)。

SMART 在正文 §3.3 使用类别数量相关在线均值/协方差更新、`Σ_t^T Σ_h`、最后维 Softmax、Hadamard 加权和标准化后的稠密特征变换；附录 A.4 讨论协方差对齐。它不复制选出的 16 个供体通道。本文离线 GAP 协方差、列余弦、混合权重和 Gumbel 子集都是本实验适配，未冒称原 SMART 公式。温度 τ 的指数实现约定在正文未展开，本轮独立定义常规 `exp(s/T)`，T=1。

## 3. 锁定输入、训练和独立验证协议

| 项目 | 值 |
| --- | --- |
| Stage I | epoch 198 / CE+MixUp 200 epochs / ResNet-32 |
| Stage I SHA256 | 0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e |
| 原训练子集 | 10847 图像；100 类；IF=100；原 raw ID 顺序 |
| 验证集 | 原固定 split 逐字节复制；1980 图像；类 1..99 每类20；缺失ID0 |
| Split SHA256 | c8579feef60d7705472ec00aeac85fa00a66c243761e0a8d76c462c663286ca1 |
| 验证索引 SHA256 | 46f557118f83617d1b685d62ba5a5715b512e71c9fd4647dd9ba35cd7e3f535e |
| 输出与指标 | 保留100 logits；预测0计错；Overall=99类Macro |
| BS / IS | 均匀类后类内 BS；每epoch全部LT样本一次 IS 随机排列，单供体 |
| BN / Backbone | eval + requires_grad=False；全部 Conv、BN affine、running mean/var/count 冻结 |
| Stage II | 每组10 epochs，BS128，85步/epoch，末batch95 |
| 优化器 | SGD base lr0.1、momentum0.9、weight decay0.0005、nesterov=False |
| 实际LR | 官方cos factor0.2；首epoch0.02；第10epoch0.000489434837 |
| 损失 | 官方CE，reweight=False、MixUp=False；无SMART额外Loss |
| 供体和比例 | 原单实例IS；rho0.25；恒16/64通道；标签恒为BS接收者 |
| 评估边界 | 仅固定未见训练来源验证；不与论文100类官方测试准确率直接比较 |


[preflight.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/preflight.json)；[effective_source_config.yaml](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/effective_source_config.yaml)；[protocol.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/protocol.json)；[split_audit.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/split_audit.json)；[preparation_checks.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/preparation_checks.json)。

按 raw ID 检查训练/验证交集 0；逐像素验证复制数组与原始 train 内容一致；图像内容哈希交集亦为 0。原 Stage I 及历史 DR 已见过整个 LT 训练子集；验证图像未进入这些训练。Stage I 最佳 epoch 的既往选择曾使用官方测试，历史验证集也已有多次机制评估，因此本协议并非完全未暴露的最终泛化检验。

分组口径（数值均为有效类别平均准确率；此验证每类样本相等，也等于相应图像准确率）：

| 分组 | 定义 | 原100类中的类数 | 验证有效类数 | 图像数 |
| --- | --- | --- | --- | --- |
| Head | ID0..35 | 36 | 35 | 700 |
| Medium（官方） | ID36..70 | 35 | 35 | 700 |
| Tail | ID71..99 | 29 | 29 | 580 |
| Many | LT n>100 | 35 | 34 | 680 |
| Medium（频次） | 20≤LT n≤100 | 35 | 35 | 700 |
| Few | LT n<20 | 30 | 30 | 600 |


有效 class ID 全表及逐类验证数量见 [validation_class_protocol.csv](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/validation_class_protocol.csv)；固定 raw ID 见 [fixed_sample_ids.csv](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/fixed_sample_ids.csv)。

## 4. 协方差估计与尾类不确定性

每张 LT 训练图像使用官方 eval transform，经锁定 Backbone 得到 `[64,8,8]`；按原分类器 `avg_pool2d(...,8).view(B,64)` GAP。float32 神经网络推理，转 float64 后累积：`S_c=X_centered^T X_centered/(n_c−1)`。不使用验证或测试图像估计协方差；空间 token 不作独立样本。

事先固定正则化：`Σ_c=0.5 S_c+0.5 trace(S_c)/64·I+εI`；`ε=1e−6 × 全LT的64维无偏方差均值=4.42654486782e-07`，全局方差均值 0.442654486782。所有策略共享同一冻结统计文件，训练不再更新。

| 训练频次组 | 类别数 | Raw LOO类均值平均 | 收缩后LOO类均值平均 | 收缩后条件数均值 |
| --- | --- | --- | --- | --- |
| Many | 35 | 0.02127 | 0.01549 | 9.124 |
| Medium | 35 | 0.08851 | 0.07120 | 12.401 |
| Few | 30 | 0.32555 | 0.29267 | 20.266 |


100 类 finite/对称/PSD 容差、正则化条件数 <100、LOO最大相对变化 <3 的预声明数值门均通过。类别频次与收缩后 LOO 敏感性 Spearman=-0.99939（描述关系，不代表因果）。这些门只用于排除数值失效/极端敏感性，不是总体协方差准确性的检验。

30 个 Few 类只有 5..19 张图像，样本协方差秩最多 n−1；4 个类仅5张，秩最多4。Few 收缩后删一图像相对 Frobenius 变化平均约29.3%，全类最极端单次删除仍达80.3%，不能把正定矩阵当作可靠总体估计。正则化条件数最高33.212。各类秩、特征值、对角方差、原谱条件与正则化条件、全部逐图 LOO 数据见 [covariance_diagnostics.csv](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/covariance_diagnostics.csv)、[covariance_leave_one_out.csv](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/covariance_leave_one_out.csv)。保留这项风险执行一次受控数值代理测试；没有声称尾类真实语义协方差已可靠恢复。

冻结统计 [covariance.npz](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/covariance.npz)（SHA256 `34f4cf84aae5220739519a8fe491170de8e860d4a8d43d385fb85414fb78a95e`）；[covariance_metadata.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/covariance_metadata.json)；[training_GAP_features_float64.npy](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/training_GAP_features_float64.npy)。

![协方差敏感性](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/figures/05_covariance_sensitivity.png)

## 5. 四组通道规则与公平性

| 组 | 掩码 | 供体 | 权重 |
| --- | --- | --- | --- |
| A Original-H2T | 原函数 randperm64 首16；batch共享 | 同一单IS实例 | uniform |
| B Uniform-PerSample | 逐样本Gumbel-Top16无放回 | 同A | 1/64 |
| C Cov-High | 与B同噪声、同Top16机制 | 同A | 0.5/64+0.5softmax(+s) |
| D Cov-Low | 与B同噪声、同Top16机制 | 同A | 0.5/64+0.5softmax(−s) |


`s_k=cos(Σ_t[:,k],Σ_h[:,k])` 沿协方差行索引归一化；字面列向量单元核对通过；替换的确是对应 k，复制完整8×8通道。代理对 t/h 对称，不等价于SMART方向变换。T=1、alpha=0.5，无任何调整。数值异常/近零列整对回退均匀；全等分数回退。此次只有100个同类对触发全等回退，0个非有限或近零回退；实际同类融合及回退为 1.0325%，未删除。采样用的回退分数记0；[diagnostic_literal_column_cosine.npz](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/diagnostic_literal_column_cosine.npz) 另保存原始余弦（同类应为1）以免混淆。

B/C/D 使用 `argsort(−(log(q)+g),stable=True)[:16]`；同一 float64 `g=−log(−log(U))`。q 是采样权重，实际包含率由 Top16 产生。A 保留隔离的原 CPU randperm 状态，并不使用该 Gumbel 噪声。

BS/IS/原A掩码复用并重新核对以前的 seed123 流。BS seed123；IS offset100003；增强300009；A通道400012；Gumbel700021。所有流互相隔离，增强按每draw“单IS先、BS后”的官方顺序重放。以前K4需4个供体增强，本次单供体使增强序列相对K4历史改变；四个新组之间完全一致。没有 Margin 教师、K4筛选或额外学习网络。

四组所有 **850/850** batch 的接收/供体索引、标签、增强后像素、冻结特征、基础噪声和CPU/CUDA RNG哈希相同；分类器与空 SGD momentum 独立从同一 Stage I 初始化。不同组掩码故意不同；A/B的随机实现与共享粒度不同。[pairing_verification.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/pairing_verification.json)。

正确性 [dryrun.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/dryrun.json)：四组单batch前后向及classifier更新；同共享掩码新/原融合逐元素相同；16供体/48接收通道逐位正确；标签未改；全数据alpha0时C/D掩码与B精确相同；finite统计/输入/logits/loss/梯度。正式3400步均finite，全部93个Backbone参数张量和93个BN缓冲区每epoch及两种保存Checkpoint逐位不变，31个BN counter保持16830，分类器2个参数张量均变化。

注意官方 Stage II 的 shift_bn=True 在这里被明确设置False，这是四组共同的固定BN机制控制；A保留官方融合代码，但本轮A不是历史官方shift_bn=True完整复现分数。

## 6. 逐 epoch 验证结果

所有epoch均在同一1980图像、100 logits上评估；Macro与Overall相等。两个Medium口径分列。

| 组 | Epoch | Overall/Macro | Head | Med-ID | Tail | Many | Med-n | Few | CE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original-H2T | 1 | 47.6263 | 61.4286 | 50.5714 | 27.4138 | 61.7647 | 51.4286 | 27.1667 | 1.415765 |
| Original-H2T | 2 | 48.0808 | 62.8571 | 50.0000 | 27.9310 | 63.3824 | 50.7143 | 27.6667 | 1.239221 |
| Original-H2T | 3 | 48.3333 | 63.8571 | 49.8571 | 27.7586 | 64.2647 | 50.5714 | 27.6667 | 1.192147 |
| Original-H2T | 4 | 47.8788 | 63.4286 | 49.5714 | 27.0690 | 64.1176 | 50.0000 | 27.0000 | 1.162982 |
| Original-H2T | 5 | 47.6768 | 62.7143 | 50.8571 | 25.6897 | 63.3824 | 51.4286 | 25.5000 | 1.131855 |
| Original-H2T | 6 | 48.7374 | 64.2857 | 51.0000 | 27.2414 | 64.8529 | 51.4286 | 27.3333 | 1.119421 |
| Original-H2T | 7 | 48.5354 | 64.1429 | 50.8571 | 26.8966 | 64.5588 | 51.4286 | 27.0000 | 1.096009 |
| Original-H2T | 8 | 48.4343 | 64.1429 | 50.5714 | 26.8966 | 64.5588 | 51.4286 | 26.6667 | 1.109262 |
| Original-H2T | 9 | 48.2323 | 64.1429 | 50.2857 | 26.5517 | 64.8529 | 50.8571 | 26.3333 | 1.091376 |
| Original-H2T | 10 | 48.2828 | 64.0000 | 50.5714 | 26.5517 | 64.7059 | 51.1429 | 26.3333 | 1.075348 |
| Uniform-PerSample | 1 | 47.8788 | 61.4286 | 51.2857 | 27.4138 | 61.7647 | 52.0000 | 27.3333 | 1.404954 |
| Uniform-PerSample | 2 | 47.8283 | 63.0000 | 49.2857 | 27.7586 | 63.5294 | 50.0000 | 27.5000 | 1.232237 |
| Uniform-PerSample | 3 | 48.1818 | 63.5714 | 49.7143 | 27.7586 | 63.9706 | 50.4286 | 27.6667 | 1.190717 |
| Uniform-PerSample | 4 | 48.0808 | 63.5714 | 50.1429 | 26.8966 | 64.2647 | 50.5714 | 26.8333 | 1.165782 |
| Uniform-PerSample | 5 | 47.8788 | 63.5714 | 50.4286 | 25.8621 | 64.1176 | 51.1429 | 25.6667 | 1.130482 |
| Uniform-PerSample | 6 | 48.3838 | 63.5714 | 50.4286 | 27.5862 | 64.2647 | 51.0000 | 27.3333 | 1.115071 |
| Uniform-PerSample | 7 | 48.2323 | 63.5714 | 50.8571 | 26.5517 | 64.1176 | 51.4286 | 26.5000 | 1.075240 |
| Uniform-PerSample | 8 | 48.7374 | 64.4286 | 51.0000 | 27.0690 | 65.0000 | 51.7143 | 26.8333 | 1.103402 |
| Uniform-PerSample | 9 | 48.3333 | 64.0000 | 50.5714 | 26.7241 | 64.5588 | 51.2857 | 26.5000 | 1.084999 |
| Uniform-PerSample | 10 | 48.2323 | 64.0000 | 50.5714 | 26.3793 | 64.5588 | 51.2857 | 26.1667 | 1.076446 |
| Cov-High | 1 | 47.9798 | 61.7143 | 51.0000 | 27.7586 | 62.0588 | 51.7143 | 27.6667 | 1.405593 |
| Cov-High | 2 | 47.8788 | 63.0000 | 49.2857 | 27.9310 | 63.5294 | 50.0000 | 27.6667 | 1.233956 |
| Cov-High | 3 | 48.1313 | 63.4286 | 49.8571 | 27.5862 | 63.8235 | 50.5714 | 27.5000 | 1.192600 |
| Cov-High | 4 | 48.2828 | 63.5714 | 50.2857 | 27.4138 | 64.2647 | 50.7143 | 27.3333 | 1.167861 |
| Cov-High | 5 | 47.8283 | 63.4286 | 50.4286 | 25.8621 | 64.1176 | 51.0000 | 25.6667 | 1.129847 |
| Cov-High | 6 | 48.3838 | 63.4286 | 50.7143 | 27.4138 | 64.1176 | 51.2857 | 27.1667 | 1.112338 |
| Cov-High | 7 | 48.3333 | 63.5714 | 51.0000 | 26.7241 | 64.1176 | 51.5714 | 26.6667 | 1.074687 |
| Cov-High | 8 | 48.6869 | 64.4286 | 50.8571 | 27.0690 | 65.0000 | 51.5714 | 26.8333 | 1.104235 |
| Cov-High | 9 | 48.2828 | 63.8571 | 50.5714 | 26.7241 | 64.5588 | 51.1429 | 26.5000 | 1.086207 |
| Cov-High | 10 | 48.2828 | 64.0000 | 50.5714 | 26.5517 | 64.5588 | 51.2857 | 26.3333 | 1.074726 |
| Cov-Low | 1 | 47.8283 | 61.5714 | 51.1429 | 27.2414 | 61.9118 | 51.8571 | 27.1667 | 1.402186 |
| Cov-Low | 2 | 47.9293 | 63.0000 | 49.4286 | 27.9310 | 63.5294 | 50.1429 | 27.6667 | 1.228201 |
| Cov-Low | 3 | 48.0808 | 63.4286 | 49.7143 | 27.5862 | 63.8235 | 50.4286 | 27.5000 | 1.189066 |
| Cov-Low | 4 | 48.1313 | 63.5714 | 50.2857 | 26.8966 | 64.2647 | 50.7143 | 26.8333 | 1.164702 |
| Cov-Low | 5 | 47.8788 | 63.5714 | 50.4286 | 25.8621 | 64.1176 | 51.1429 | 25.6667 | 1.128967 |
| Cov-Low | 6 | 48.1818 | 63.1429 | 50.5714 | 27.2414 | 63.8235 | 51.1429 | 27.0000 | 1.111913 |
| Cov-Low | 7 | 48.1313 | 63.5714 | 50.7143 | 26.3793 | 64.1176 | 51.2857 | 26.3333 | 1.077921 |
| Cov-Low | 8 | 48.5859 | 64.4286 | 50.7143 | 26.8966 | 65.0000 | 51.4286 | 26.6667 | 1.103653 |
| Cov-Low | 9 | 48.2828 | 64.0000 | 50.4286 | 26.7241 | 64.5588 | 51.1429 | 26.5000 | 1.085697 |
| Cov-Low | 10 | 48.1818 | 64.0000 | 50.4286 | 26.3793 | 64.5588 | 51.1429 | 26.1667 | 1.075492 |


[epoch_validation_metrics.csv](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/epoch_validation_metrics.csv) 包含逐epoch LR、时间、显存；[channel_frequency_all_epochs.csv](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/channel_frequency_all_epochs.csv) 保存逐epoch每通道计数/包含率。每组 batches.csv 保存全部850步Loss、配对哈希与计时，training.log和 full_console.log 保留运行日志。

![逐epoch曲线](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/figures/01_epoch_curves.png)

## 7. 主要结果：固定第10 epoch

| 组 | Overall | Macro | Head | Med-ID | Tail | Many | Med-n | Few |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original-H2T | 48.2828 | 48.2828 | 64.0000 | 50.5714 | 26.5517 | 64.7059 | 51.1429 | 26.3333 |
| Uniform-PerSample | 48.2323 | 48.2323 | 64.0000 | 50.5714 | 26.3793 | 64.5588 | 51.2857 | 26.1667 |
| Cov-High | 48.2828 | 48.2828 | 64.0000 | 50.5714 | 26.5517 | 64.5588 | 51.2857 | 26.3333 |
| Cov-Low | 48.1818 | 48.1818 | 64.0000 | 50.4286 | 26.3793 | 64.5588 | 51.1429 | 26.1667 |


![第10epoch准确率](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/figures/02_fixed_epoch10_accuracy.png)

配对差值，单位pp；不是从各指标最佳epoch拼接：

| 比较 | Overall | Head | Med-ID | Tail | Many | Med-n | Few | 新增正确/新增错误 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B − A | -0.0505 | +0.0000 | +0.0000 | -0.1724 | -0.1471 | +0.1429 | -0.1667 | 7/8 |
| C − B | +0.0505 | +0.0000 | +0.0000 | +0.1724 | +0.0000 | +0.0000 | +0.1667 | 1/0 |
| D − B | -0.0505 | +0.0000 | -0.1429 | +0.0000 | +0.0000 | -0.1429 | +0.0000 | 0/1 |
| C − D | +0.1010 | +0.0000 | +0.1429 | +0.1724 | +0.0000 | +0.1429 | +0.1667 | 2/0 |


B−A：掩码粒度和随机实现变化的联合效果。C−B：同逐样本机制下偏高关联的效果。D−B：偏低关联。C−D：关联方向。A−C/D还混入共享粒度，不能将全部差异归因协方差。配对噪声减少了本轮随机流混杂，但只有一个训练种子，不能排除初始化/样本增强流对效应方向的影响；同验证集上的个位数正确图像差也不构成稳定收益。

逐类与配对预测 [per_class_epoch10.csv](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/per_class_epoch10.csv)；[paired_predictions_epoch10.csv](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/paired_predictions_epoch10.csv)；[paired_epoch10_contrasts.csv](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/paired_epoch10_contrasts.csv)。

## 8. 描述性最佳epoch与Checkpoint

最佳只按验证Overall择epoch，平局取最早。原始历史模型未覆盖。八个模型均已独立重新加载，保存的Backbone/BN与Stage I字节相同，验证指标与原记录精确一致。

| 组 | Best epoch | Best Overall | 第10epochOverall | 最佳Checkpoint | 最终Checkpoint |
| --- | --- | --- | --- | --- | --- |
| Original-H2T | 6 | 48.7374 | 48.2828 | [best](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/arms/Original-H2T/checkpoints/model_best.pth.tar) | [final](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/arms/Original-H2T/checkpoints/model_final.pth.tar) |
| Uniform-PerSample | 8 | 48.7374 | 48.2323 | [best](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/arms/Uniform-PerSample/checkpoints/model_best.pth.tar) | [final](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/arms/Uniform-PerSample/checkpoints/model_final.pth.tar) |
| Cov-High | 8 | 48.6869 | 48.2828 | [best](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/arms/Cov-High/checkpoints/model_best.pth.tar) | [final](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/arms/Cov-High/checkpoints/model_final.pth.tar) |
| Cov-Low | 8 | 48.5859 | 48.1818 | [best](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/arms/Cov-Low/checkpoints/model_best.pth.tar) | [final](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/arms/Cov-Low/checkpoints/model_final.pth.tar) |


各组 effective.yaml、experiment_config.json、status.json、checkpoint_validation.json 与模型同目录。

## 9. 实际通道干预强度

实际BS/IS对中分数通道极差中位数 0.30951；q_high/low 距Uniform的平均TV分别 0.01331/0.01342。分数并非完全相等，但T=1、混合0.5使概率偏移小。

| 组 vs B | 子集完全相同率 | 平均共享/16 | 平均Jaccard | 平均所选有效评分 |
| --- | --- | --- | --- | --- |
| Original-H2T | 0.0000% | 4.004278 | 0.146413 | 0.712405 |
| Uniform-PerSample | 100.0000% | 16.000000 | 1.000000 | 0.712422 |
| Cov-High | 76.6719% | 15.760782 | 0.971935 | 0.714628 |
| Cov-Low | 76.5567% | 15.759666 | 0.971803 | 0.710199 |


Cov-High提高实际所选平均评分、Cov-Low降低，说明权重确已生效；不过约76.6%的子集与B完全一致、平均只替换约0.24个不同通道，干预很弱。这个现象在正式训练前已经报告，未调T/alpha。不能用小效应否定所有协方差信息。[sampling_diagnostics.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/sampling_diagnostics.json)；[mask_overlap_by_receiver_group.csv](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/mask_overlap_by_receiver_group.csv)。

![掩码干预](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/figures/03_mask_intervention.png)

![实际通道频率](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/figures/04_actual_channel_frequency.png)

## 10. 时间、显存与额外成本

| 组 | 训练秒数 | 比B增加秒数 | 比B增加% | 融合累计秒 | 峰值allocated MiB | 峰值reserved MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Original-H2T | 102.790 | -5.401 | -4.99 | 7.644 | 78.97 | 98.00 |
| Uniform-PerSample | 108.191 | +0.000 | +0.00 | 9.580 | 79.44 | 116.00 |
| Cov-High | 112.694 | +4.503 | +4.16 | 9.609 | 79.44 | 116.00 |
| Cov-Low | 110.343 | +2.152 | +1.99 | 9.172 | 79.44 | 116.00 |


离线特征、协方差、列评分、全10epoch掩码生成总计 7.124s；其中保存协方差前 3.508s（含模型加载和特征提取）。该一次性成本四组共享，不包含正式优化；没有在线第二评分网络。[timing_and_memory.csv](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/timing_and_memory.csv)。

训练计时含在线官方增强、两次冻结Backbone前向、逐batch配对哈希及元素检查；不含epoch后验证/模型保存。C/D权重与掩码离线预生成，在线和B走同一融合实现，运行差异还受GPU时钟、系统负载和审计开销影响；这里不是专用加速基准。allocated/reserved是PyTorch进程指标，不等于nvidia-smi的整卡占用（另有桌面/驱动占用）。全部训练在8GB RTX5070 Laptop上完成，无OOM。

## 11. 可复查文件与原文件保护

独立运行目录：`D:\xiangmuxunlian\新\H2T_repro_audit\smart_channel_runs\smart_cov_20261013_seed123`；[git_training_provenance.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/git_training_provenance.json)；[source_lock.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/source_lock.json)；[experiment.diff](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/experiment.diff)。48个兼容runtime源文件逐字节复制；原methods.py及所有历史目录未编辑。新增掩码代码在外部脚本中，不覆盖原算法。[historical_verification.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/historical_verification.json) 记录历史文件全量哈希复核。

主要原始数据 [sampling_and_masks.npz](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/sampling_and_masks.npz)（接收/供体索引、A状态及BCD子集）；[gumbel_noise_float64.npy](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/gumbel_noise_float64.npy)；[channel_weights.npz](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/channel_weights.npz)；统计与所有epoch预测NPZ、日志和图表PNG/SVG均保留。完整SHA文件清单在 [delivery_manifest.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/delivery_manifest.json)。

准确复查命令（只适用于一个全新、未运行的独立副本，当前脚本拒绝覆盖或追加）：

```powershell
$py = "D:\app\anaconda\envs\pytorch\python.exe"
$run = "D:\xiangmuxunlian\新\H2T_repro_audit\smart_channel_runs\smart_cov_20261013_seed123"
& $py "$run\build_helpers.py"
& $py "$run\smart_channels.py" prepare
& $py "$run\smart_channels.py" stats
& $py "$run\smart_channels.py" dryrun
& $py "$run\lock_source.py"
& $py "$run\smart_channels.py" train
& $py "$run\analyze_results.py"
```

## 12. 停止与后续判断

目前缺少稳定收益证据：Cov-High 的正向点估计仅净多一张验证图像。若继续，它只有保持同协议作低成本多种子机制复核的有限价值，不能据此推荐协方差引导替代 Uniform。 任何 seed124/125 必须等待明确确认，并保持相同验证索引、统计估计规则与超参数；不能根据本轮结果调整T、alpha或供体。当前仅保存候选点估计与风险，不宣称新算法或原SMART复现。

[STOP_AFTER_SEED123.json](D:/xiangmuxunlian/新/H2T_repro_audit/smart_channel_runs/smart_cov_20261013_seed123/STOP_AFTER_SEED123.json) 明确锁定停止；没有新的Backbone训练、Stage I改写、Margin供体筛选、SMART额外Loss、PI融合比例或官方测试策略选择。
