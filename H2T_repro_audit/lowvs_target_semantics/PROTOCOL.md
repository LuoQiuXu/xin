# Low-VS Target-Semantics Pilot：预注册协议

## 1. 单一问题

在固定的 Low-VS 融合输入上，尾类接收者统一使用硬标签是否会限制融合样本的训练价值？本轮只替换监督目标，不再改变 donor 选择、通道分数、候选池、融合比例或保护规则。

本实验是机制筛选，不是最终方法声明。当前开发集曾被历史实验重复使用，因此任何性能结果都只算探索性证据。

## 2. 三个实验臂

全部实验只对尾类接收者（训练标签 ID >= 71）的 fused CE 目标做干预。非尾类接收者在三臂中均保持原硬标签。

### A. HardRecipient

原始基线：
loss = CE(stage2_logits(fused), y_receiver)

### B. DonorMixTarget

对尾类接收者使用：
target = 0.75 * one_hot(y_receiver) + 0.25 * one_hot(y_donor)

对非尾类接收者仍用接收者 one-hot。若 donor 与 receiver 同类，目标自然退化为 one-hot。0.25 是固定的通道占比代理（16/64），并非声称特征通道替换等价于图像/语义 MixUp；本轮不扫描该值。

### C. FirstCrossedCompetitor

使用**冻结且训练来源可核实的 balanced cRT reference head**，分别计算 clean receiver GAP 特征与 fused GAP 特征的 logits。对尾类接收者：

1. 只有当 clean reference 对真实接收类别具有严格正 margin（真实类 logit 严格高于所有非真实类）时，才检查 crossing。
2. 对每个竞争类 j != y，定义 m_j(0)=z_y(clean)-z_j(clean)，m_j(1)=z_y(fused)-z_j(fused)。
3. 若 m_j(0)>0 且 m_j(1)<=0，则该类沿 clean-to-fused 线性特征路径的交叉比例为 alpha_j = m_j(0) / (m_j(0) - m_j(1))。
4. 选择最小 alpha_j 对应的竞争类；并列时取较小 class ID。
5. 若存在首次交叉，目标为 0.75 * one_hot(y_receiver) + 0.25 * one_hot(j_first)；若 clean reference 已误判、没有交叉或参考 logits 非法，则该样本回退到接收者硬标签。非尾类一律硬标签。

由于 reference head 是线性分类器，GAP 特征沿 clean-to-fused 线性路径变化时，其 logits 也沿直线变化。该 target 是参考分类器条件下的训练假设，不是真实语义标签，也不证明首次竞争类一定应该获得标签质量。

### Loss scaling

三个实验臂统一用单次交叉熵形式。软目标损失：
loss_i = -sum_c target_ic * log_softmax(logits_i)_c。
每行 target 总质量必须为 1。禁止额外增加 clean anchor、risk weight、label smoothing 或任何辅助 loss，以免混杂梯度尺度和监督语义。

## 3. 固定训练协议

- 数据：真实 CIFAR-100-LT，IF=100；训练子集 10,847 张。
- Backbone：ResNet-32，复用已锁定 Clean Stage-I seed 42 / epoch 184 checkpoint；报告中记录的 SHA256 为 7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88，运行前必须核验本地实际文件。
- Stage-II seeds：124、125。
- 每臂：10 epochs，每 epoch 85 个更新，合计 850 次真实 optimizer.step。
- batch size：128，末批按原训练集长度处理。
- Backbone、BN affine、running statistics、num_batches_tracked 全部冻结并处于 eval；只训练原 100-way linear classifier。
- 初始化、SGD、scheduler、学习率缩放、数据增强、receiver/donor schedule、Gumbel 随机流、Low-VS 在线分数、Bottom48 候选池、16 个完整通道替换和 fallback 规则，全部复用现有封存的 LowVS-Full 实现。
- 每个 seed 的三臂从相同 Stage-I classifier 初始化，独立训练；三臂的每个 batch 必须有完全相同的 receiver/donor IDs、增强后输入、冻结特征、Gumbel、原始 VS 分数、实际融合 mask。允许不同的只有目标分布和由其导致的梯度/分类器状态。
- reference cRT 只用于生成 C 的目标，不参与 Stage-II 输出 logits、梯度或优化器更新。必须检查它与 Stage-I 特征来源一致、只用 TRAIN 训练、未用官方 TEST 选择。
- 如果 balanced cRT reference 的本地 checkpoint 与训练来源无法核实，立即停止，不得改用 Stage-I head 或临时新训一个未锁定 reference 来继续本协议。

## 4. 数据隔离和文件保护

- 官方 CIFAR-100 TEST 禁止加载、解码、哈希、统计、评估或用于选择模型。历史研究曾使用该测试集，因此不能将它称为全新未见终检。
- 开发集只允许复用已封存的 1,980 张官方 TRAIN 留出图像（class ID 1–99 每类 20 张，class 0 缺失），仅作探索性结果评估；不允许用它构建 donor bank、reference、target、mask 或训练计划。
- 新输出只能写入 H2T_repro_audit/lowvs_target_semantics/results/。任何预期输出已存在时拒绝覆盖，不自动 resume。
- 不修改历史脚本、checkpoint、采样计划、Gumbel、缓存或报告。任何源 hash、配对签名、冻结状态或 TEST guard 失败时立即停止，不跳过、不自动重试。
- 不执行 git checkout/fetch/pull/push/commit/branch/merge 等版本控制操作；由用户管理版本同步。

## 5. 必需输出

每个 seed/arm：
- config.json、effective.yaml、epochs.csv、batch_signatures.csv、target_diagnostics.csv；
- final_fc.pth、validation_epoch10.npz、summary.json；
- console stdout/error log。

全局：
- preflight.json、source_manifest.json、reference_provenance.json、environment.json；
- epoch10.csv、paired.csv、per_class.csv、paired_group_transitions.csv；
- target_diagnostics_summary.csv、scientific_summary.json；
- LOWVS_TARGET_SEMANTICS_RESULTS.md 与中文 LOWVS_TARGET_SEMANTICS_SCIENTIFIC_REPORT.md；
- 成功停止标记 STOP_COMPLETED.json；若异常则写 FAILED_STOP.json。

报告主结果固定使用 epoch 10。Best epoch 仅可作为诊断，不作为主比较。

## 6. 必须报告的指标

- Overall、Head、OfficialMedium、Tail、Many、Medium、Few、Macro、validation CE；
- 每张开发图像的配对 wins/losses/net correct；
- Tail-only wins/losses/net correct；
- 每个类别的正确数变化、尾类最差类别及主要混淆去向；
- B/A、C/A 的每 seed 差值，不能只报平均；
- B 中 donor 与 receiver 异类比例、软目标实际激活率；
- C 中 clean reference 正确率、crossing 率、first-crossed competitor 类别分布、硬标签回退比例；
- 三臂所有输入/mask 签名一致性和冻结状态检查。

一张图像在 1,980 张总体开发集上约为 0.0505 pp；在 580 张有效 Tail 子集上约为 0.1724 pp。几张图的差异不能当作稳定提升。两个 seed 共享同一开发图像，不得把重复评估计作独立样本，也不得在 n=2 下做显著性声明。

## 7. 预声明决策规则

候选 B 或 C 只有在以下条件同时成立时才值得下一轮确认：
1. 相对 A 的 Tail net-correct 在 seed 124 和 125 都为正；
2. 每个 seed 的 Tail 净收益都至少超过 2 张图像，避免把 1–2 张图的波动当成信号；
3. 两个 seed 的 Overall 都不低于 A；若尾类提升但 Overall/Head 下降，必须明确记为 trade-off，不宣称总体改善；
4. 收益不是只来自一个类别：报告所有类别变化，若净收益几乎全由单一类驱动，则只视为类别特例；
5. 实现审计确认除 target 外没有输入或训练协议差异。

若方向跨 seed 翻转、增益只涉及 1–2 张图，或两个新目标都不优于 A，则停止本候选，不扫描 donor_mass、epoch、类别阈值或 reference 置信度。即使通过，也只说明该目标值得在独立评估协议上确认，不代表已证明泛化或达到发表标准。
