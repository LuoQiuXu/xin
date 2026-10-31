# Fish4Knowledge-16: H2T / PI-H2T 外部长尾对照实验协议

日期：2026-10-31
实验分支：`experiment/fish16-h2t-pih2t-cross-domain-20261031`
基线分支：`experiment/h2t-cross-domain-data-gate-20261030`
研究阶段：外部领域性能验证；不进行方法改进。

## 0. 当前数据资格与决策

本轮只在 Fish4Knowledge 的 16 类 primary 范围开展探索性训练。

- 官方全量：23 类、27,370 张图像、8,725 条原始轨迹。
- primary：16 类、27,133 张图像、8,680 条独立轨迹。
- 固定分组划分：train 16,269 图像 / 5,208 轨迹；val 5,396 / 1,736；test 5,468 / 1,736。
- 固定数据划分 seed=42，60/20/20，按 tracking trajectory 分组，不按图片随机切分。
- primary 的训练类频 Head / Medium / Tail 为 6/5/5 类三分位；不能称为 H2T 论文 CIFAR 的 Many/Medium/Few 绝对阈值。
- 其余 7 类继续保留在完整审计清单，但不参与 primary 训练和主指标。

ISIC 2019 当前为 NO_GO_DATA_INTEGRITY：存在两对字节完全相同、标签分别冲突为 MEL/NV 的图像，并连接两个 lesion ID。没有 train/val/test 划分。禁止训练 ISIC、手工删图、改标签或将其描述为 H2T 方法失效。ISIC 只有在冲突来源被独立查明并取得可信数据版本后，才能另立实验。

审计报告：
`H2T_repro_audit/cross_domain_failure_boundary/results/formal_data_audit_20261031_01/CROSS_DOMAIN_DATA_AUDIT_REPORT.md`

本地审计清单在仓库外：
`D:\\h2t_cross_domain_audit\\fish4knowledge\\manifest_primary.csv`

原始图像：
`D:\\xiangmuxunlian\\xin\\data\\fishRecognition_GT\\fish_image\\`

## 1. 核心研究问题

检验 H2T 与 PI-H2T 的头尾迁移收益能否从 CIFAR-100-LT 迁移到水下鱼类视觉领域，并定位收益消失或反转的条件。

本轮只回答：
1. H2T 的随机通道替换在 Fish4Knowledge 上是否仍有尾类收益？
2. PI-H2T 的 PIF + H2TF 是否比 H2T 更稳健，还是也会出现尾类负迁移？
3. 如果融合后性能退化，退化集中在哪些类别、哪些测试样本、哪些训练 donor 关系或融合比例区间？

本轮禁止设计新损失、新 donor 选择器、新通道选择器或新融合公式。先获得可信的原方法外部表现，再讨论改进。

## 2. 代码与方法来源

- H2T 的 Stage-I/Stage-II 训练结构、ResNet50 feature extractor、H2T 通道替换函数来自仓库中的 `H2T_official_clean/H2T-main/`。
- PI-H2T 的 PIF 与 H2TF 公式实现参考 `PI_H2T_review/H2T-main/`。
- 现有 PI-H2T review runner 仅支持 CIFAR-100 / ResNet-32，不得直接将其当作已支持 Fish 的训练入口；本轮需要在新目录实现通用 ResNet50 适配。
- 原始 H2T 与 PI-H2T 目录均只读，不得直接修改其历史代码。

## 3. 公平比较：六个必要条件

采用两个 Stage-I 表征分支，每个分支做 Stage-I 直接评估和 Stage-II 对照：

| ID | Stage-I 表征 | Stage-II | 目的 |
|---|---|---|---|
| A0 | Plain ResNet50 | 不做 Stage-II | 原始基础模型 |
| A1 | Plain ResNet50 | cRT，无融合 | 排除平衡分类器重训练的影响 |
| A2 | Plain ResNet50 | H2T 通道替换 | 测 H2T 本身的迁移收益 |
| B0 | PIF-ResNet50 | 不做 Stage-II | 单独观察 PIF 表征作用 |
| B1 | PIF-ResNet50 | cRT，无融合 | 排除 PIF 分支中的分类器重训练影响 |
| B2 | PIF-ResNet50 | PI-H2TF | 测 PI-H2T 的融合收益 |

不能只比较 A2 和 B2：PIF 本身会改变 Stage-I 表征。核心增益必须相对于各自的无融合控制计算：

`Delta_H2T_Tail = Tail(A2) - Tail(A1)`

`Delta_PI_H2T_Tail = Tail(B2) - Tail(B1)`

同时报告相对 Stage-I 的变化，但不得把它替代 matched cRT 对照。

## 4. Backbone 与共同训练协议

- Backbone：仓库 H2T 的 ResNet50_FE，2048 维 pooled feature；使用 ImageNet 预训练权重。
- Plain 与 PIF 两条分支必须从完全相同的 ResNet50 预训练权重初始化；PIF 分支只额外添加现有 PI-H2T review 中的 PIF 模块。
- PIF 放在 ResNet50 layer4 feature map 后、GAP 前；不得改变 ResNet 主干层数或通道数。
- 输入统一为 224×224。
- Stage-I 训练增强：RandomResizedCrop(224)、RandomHorizontalFlip、ColorJitter(0.4)、ImageNet normalization；MixUp alpha=1.0。
- 验证/测试：Resize(256)、CenterCrop(224)、ImageNet normalization。
- 采用相同的优化器、学习率计划、batch、增强、epoch 和随机种子；Plain 与 PIF 的差异仅为 PIF。
- 首轮 pilot：训练 seed=42；Stage-I 90 epochs，SGD，lr=0.01，momentum=0.9，weight_decay=1e-4，5-epoch warmup + cosine；Stage-II 10 epochs，SGD，lr=0.01，momentum=0.9，weight_decay=5e-4，cosine。
- 资源默认：batch_size=16，AMP，num_workers=4；若 OOM，退到 batch_size=8 并使用 gradient accumulation=2，必须记录真实 micro-batch 与 effective batch。所有方法使用相同配置。
- 预训练权重若本地不存在，先报告权重来源/缓存状态；不得静默改为随机初始化。记录权重文件 SHA256。
- 这是固定的外部领域受控适配协议，不宣称复现 H2T/PI-H2T 论文的所有原始 benchmark 配方。

## 5. Stage-II 采样与冻结规则

按原 H2T 双采样思想构造 recipient / donor：

- Recipient：按类别均衡采样（uniform class sampling with replacement）。
- Donor：从 Stage-I train split 的自然长尾实例分布抽样（instance-proportional sampling with replacement）。
- H2T 与 PI-H2T 使用相同 seed 下预生成的 recipient/donor index pairs，确保 donor 数据流可比较。
- cRT 对照只使用同一 recipient，不使用 donor。
- donor label 不参与 recipient 的 CE 监督；只用于记录 donor class 与 donor frequency group。
- 不额外强制 donor 必须是 Head 类，也不强制 donor 与 recipient 不同类/不同轨迹；这些约束会改变原方法。必须记录 same-image、same-track、same-class 的配对比例，以便解释结果。
- Stage-II 所有方法统一冻结 feature extractor 参数与 BN running statistics（model.eval）；只优化 classifier。此设置用于控制融合效应，须在报告中说明 H2T 原仓库配置存在 shift_bn=True 的 BN 更新行为；本轮主比较不把 BN 更新差异混入融合差异。只有主实验完成后，才可单独做 H2T BN-shift sensitivity。
- Stage-II cRT、H2T、PI-H2T 使用相同 recipient sampling schedule、epoch 数和 optimizer。
- H2T：调用原始 H2T channel replacement 核心函数，rho=0.25；保留“同一 batch 共享随机通道索引”的原始行为，不增加选择机制。
- PI-H2T：保留现有 paper_theoretical_bounds H2TF 公式。先 GAP，再用当前 classifier 的类别权重计算 cosine；`r=(1-cosine)/2`，`fused=r*f_recipient+(1-r)*f_donor`；ratio_detach=True。PIF 使用现有两共享参数公式，不发明新初始化或新参数。

## 6. 评估协议

### 主指标

- Overall image Top-1 accuracy
- Macro accuracy（各类准确率算术平均）
- Head / Medium / Tail accuracy（按训练频次三分位 6/5/5 类）
- 每类 Top-1 accuracy
- Worst-class accuracy
- Confusion matrix

### 轨迹相关性

同一 trajectory 的多张图像并非独立样本。必须同时：
- 报告 image-level 指标（主指标，便于与图像分类方法对照）。
- 报告 trajectory-level 辅助指标：对同一 test trajectory 的 logits 求平均后分类。
- 按 test trajectory 做 cluster bootstrap（至少 1,000 次），给 Macro / Tail 指标置信区间；不能按单张图像独立 bootstrap。
- 每类同时显示 test image 数与独立 trajectory 数，特别标出只有 3 条 test trajectories 的类别。结论不得忽略该统计限制。

### 负迁移定位

对同一测试样本比较融合模型与对应 cRT：
- correct→wrong、wrong→correct 数量；
- per-class accuracy delta；
- target logit margin 的变化；
- confidence / margin 分布变化。

训练阶段同时记录：
- H2T 每个 batch 的替换通道数、通道索引摘要；
- PI-H2TF 的 r 与 donor mass（1-r）整体及按类别分布；
- recipient class、donor class、donor Head/Medium/Tail、same-class、same-track 配对率。
这些记录用于解释退化，不得据此在本轮修改算法。

## 7. 训练与测试纪律

- 先做真实数据 1-batch forward/backward、真实 1-epoch smoke；检查所有六个条件的数据/模型路径。
- smoke 全部通过后，只跑 seed=42 的 pilot。
- pilot 完成后先分析是否存在候选退化；如果存在，再固定同一协议跑 seed=43、44。只有多 seed 方向一致，才称为稳定退化。
- Stage-I 和 Stage-II 均使用 validation 选择 best checkpoint。
- Test split 只在模型和 checkpoint 固定后评估一次；不得用 test accuracy 选择 epoch、调超参数或选择方法。
- 任何训练异常、OOM、NaN、数据缺失都必须停止并报告，不得静默跳过样本。
- 不下载、不复制、不修改原始数据；不把 manifests、图像、预训练权重、checkpoints 或大体积训练结果提交 Git。

## 8. 预先定义的科学判据

- H2T 负迁移候选：`Delta_H2T_Tail < 0`，且 Tail / Macro / per-class 变化能由样本级预测转换或 margin 变化支持。
- PI-H2T 负迁移候选：`Delta_PI_H2T_Tail < 0`，并检查 r / donor mass 是否退化到极端、是否在特定类别上集中。
- H2T 相对 PI-H2T 的差异必须在各自 matched cRT 基线后解释，不能只比绝对准确率。
- 单 seed 只标记 candidate signal；至少 3 个训练 seeds 方向一致并给出轨迹级置信区间后，才称为稳定结果。
- 若融合方法没有性能退化，不人为制造失败；报告其适用范围和未发现盲区，再考虑第二个有效外部数据集。
- 不能把 ISIC 的数据标签冲突、Fish 的极少类别支持或训练代码 bug 归因于 H2T / PI-H2T 算法。

## 9. 交付

输出目录放在仓库外，例如：
`D:\\h2t_cross_domain_results\\fish16_pilot_seed42\\`

必须交付：
- run_manifest.json（分支、commit、输入 manifest SHA、数据路径、模型权重 SHA、配置、seed、环境）
- per_epoch_metrics.csv
- best_validation_metrics.json
- final_test_metrics.json
- per_class_metrics.csv
- confusion_matrix.csv
- trajectory_metrics.json
- cluster_bootstrap_ci.json
- sample_transition_audit.csv
- h2t_channel_audit.json
- pi_ratio_audit.json
- donor_pair_audit.csv
- checkpoints（只保存在仓库外）
- TRAINING_REPORT.md（含所有偏差、异常、限制）

本轮目标是找到 H2T / PI-H2T 的真实外部性能边界，不是改进方法。任何新算法改动必须等结果分析后单独立项。
