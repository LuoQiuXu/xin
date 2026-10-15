# PI-H2T 完整 PIF setting：ratio-only A/B/C 复现验证

## 结论先行

本轮已经完成真实 PIF Stage-I，再冻结该 PIF checkpoint 运行与 CE 阶段完全相同的 ratio-only A/B/C、Stage-II seeds 42/43/44。

结果是：

- `B > A` 在 3/3 个 Stage-II seed 中成立，平均 `+0.39 ± 0.09` 个百分点；
- 但关键的 correspondence 阴性对照 `B > C` **没有复现**：三 seed 分别为 `+0.01`、`-0.14`、`+0.22` 个百分点，平均只有 `+0.03 ± 0.18` 个百分点，且 seed43 反向；
- 因此，完整 PIF setting 下预先规定的严格 gate（`B>A` 且 `B>C` 在全部 seed 中成立）**不通过**。

可以说 PIF setting 中 pair-conditioned allocation 相对 PI-H2T recipient-only allocation 存在一个小而方向一致的信号；但不能说它已被证明来自“正确的 recipient--donor correspondence”，更不能声称该修正稳定提升完整 PI-H2T。

这与 CE setting 的结果不同：CE 下 `B-A=+0.77±0.16`、`B-C=+0.51±0.14`，且二者在 3/3 seed 都为正；PIF 下 B-C 的优势消失。

## 这次如何构成真实 PIF Stage-I

论文中的 PIF 在最后 pooling 前先对 feature map 做 channel-wise mean，得到 permutation-invariant feature，再按 Eq. (6) 对 `[F, F-F_PI]` 做 1D convolution 得到融合 feature map。H2TF 只在 Stage-II 使用。

本地 H2T 仓库有 PIF 预留分支，但存在两个阻碍完整运行的实现问题：

1. PIF 分支计算了 `out_hat`，却返回了未融合的 `out`；已改为 PIF 时返回 `out_hat`，严格对应论文 Eq. (6)。
2. Stage-I 没有可选择的 PIF backbone factory；新增 `resnet32_pif_fe`，仅令现有 `ResNet_fe(feature_tuning=True)` 可被原 Stage-I 训练器调用。

这些是启用论文已定义 PIF 的最小修复，不改动 H2TF ratio、A/B/C、损失、sampling 或 reference head。还将旧代码的 `pathlib2` backport 改为 Python 标准库 `pathlib`，并将 checkpoint 的 tensor 标量转为 JSON 数值；二者均为运行兼容性修复。

### Stage-I 固定设置与审计

- 数据：CIFAR100-LT，IF=100。
- 架构：官方 CIFAR ResNet-32 feature extractor + PIF + 原 classifier。
- 训练：CE + MixUp，batch 128，200 epoch，SGD momentum 0.9，weight decay `2e-4`，初始 LR `0.1`，epoch 160/180 分别乘以 `0.1`。
- checkpoint：`saved/cifar100_imb001_stage1_ce_mixup_pif_202608082320/ckps/model_best.pth.tar`。
- 最佳 checkpoint：epoch 164，Top-1 `40.27%`；独立严格加载后的复测也是 `40.27%`。
- PIF 审计：checkpoint 含 6 个 `conv1d.*` 参数键；PIF 前向测试的 `conv1d` 梯度绝对值和为正；现有 Stage-II 脚本严格加载后识别为 `stage1_pif=True`。

说明：公开 H2T 仓库的主训练代码是 AAAI-24 H2T，而非一个完整、可直接运行的 PI-H2T 端到端发布版。因此本轮是基于论文 Eq. (6) 与仓库预留 PIF 分支完成的**论文忠实复现实现**，不应描述为“直接运行官方 PI-H2T 完整代码”。

## Stage-II：与 CE A/B/C 完全保持一致的部分

- 同一个冻结的 PIF Stage-I backbone 与 Stage-I classifier head；后者既计算原 PI ratio，也计算 pair geometry。
- class-balanced recipient + instance-wise long-tailed donor；冻结 backbone、BN shift、仅训练 classifier、CE、10 epochs。
- Stage-II seeds：42、43、44；batch 128，workers 0，`ratio_bins=4`，`delta=0`。
- A：原 PI-H2T recipient-only ratio。
- B：在 `Many/Medium/Few × 原 PI recipient-ratio 四分层` 内，保存原生 PI donor-mass 多重集合，依 `lambda_critical(f_B,f_I)` 排序分配。
- C：保存 B 的相同 ratio 集合、相同 recipient、相同 donor，仅在同一控制分层内打乱 ratio--pair correspondence。
- 没有 donor selection、候选 donor、risk loss、clean anchor、cRT/balanced reference、固定 lambda、额外 sampling 或超参搜索。
- 主指标仍为预注册的第 10 epoch Last Overall。

## PIF Stage-II 主结果

| Stage-II seed | A：PI recipient-only | B：pair-conditioned | C：shuffled correspondence | B-A | B-C |
|---:|---:|---:|---:|---:|---:|
| 42 | 34.82 | **35.23** | 35.22 | +0.41 | +0.01 |
| 43 | 34.96 | 35.25 | **35.39** | +0.29 | -0.14 |
| 44 | 35.35 | **35.81** | 35.59 | +0.46 | +0.22 |
| 均值 ± 样本标准差 | 35.04 ± 0.27 | **35.43 ± 0.33** | 35.40 ± 0.19 | **+0.39 ± 0.09** | **+0.03 ± 0.18** |

频次组的跨 seed 均值如下（单位：%）：

| 组别 | Overall | Many | Medium | Few |
|---|---:|---:|---:|---:|
| A | 35.04 ± 0.27 | 26.47 ± 0.34 | 45.23 ± 0.81 | 33.39 ± 0.40 |
| B | **35.43 ± 0.33** | **27.22 ± 0.49** | 45.37 ± 0.64 | **33.62 ± 0.27** |
| C | 35.40 ± 0.19 | 26.96 ± 0.22 | **45.59 ± 0.85** | 33.57 ± 0.41 |

## 审计结果

| 审计项 | seed42 | seed43 | seed44 |
|---|---:|---:|---:|
| A/B/C ratio 多重集合最大误差 | 0 | 0 | 0 |
| B 的 `lambda_critical -> lambda_pair` 单调违规数 | 0 | 0 | 0 |
| C 层内实际打乱比例 | 95.63% | 95.23% | 95.40% |
| A/B/C recipient、donor、target、donor-label 流校验和 | 一致 | 一致 | 一致 |
| 原生 PI 平均 donor mass | 0.769666 | 0.769682 | 0.769856 |

因此，PIF 三臂之间的差异不是由样本、增强、recipient/donor 数据流、总体 donor mass、频次组 donor mass、ratio 多重集合或不同 reference head 造成的。

## 与 CE 结果的直接比较

| Stage-I setting | B-A（均值 ± 样本标准差） | B-C（均值 ± 样本标准差） | 严格 gate |
|---|---:|---:|---|
| 无 PIF 的 CE checkpoint | +0.77 ± 0.16 | +0.51 ± 0.14 | 通过，3/3 seed 均正 |
| 本轮完整 PIF checkpoint | +0.39 ± 0.09 | +0.03 ± 0.18 | **不通过**，seed43 为负 |

这里的 CE/PIF 并列只用于比较两轮 gate 的稳定性，**不是** PIF 的绝对性能因果估计：此前 CE checkpoint 使用的是本地 Option-B ResNet-32，而本轮 PIF 使用发布 H2T 配方的默认 feature extractor。因此，只有每一轮内部严格控制的 A/B/C 比较可以作因果解释。

PIF 下原生 PI donor mass 约为 `0.77`，高于 CE setting 的约 `0.655`。与此同时，PIF Stage-I 初始 Overall 为 40.27%，而 10 epoch Stage-II 后三臂均约为 35%。这说明在当前论文原 ratio 尺度下，PIF representation 的 H2TF 训练十分激进；它是一个值得记录的观察，但本轮没有调节或归因该尺度，也不以此解释 B-C 失败。

## 应如何解读

本轮完整 PIF 实验否定的是一个过强的说法：

> 不能说“只要加入 PIF，pair-conditioned ratio 就会稳定优于 recipient-only，且这种优势已被 shuffled correspondence 对照验证”。

更精确的当前结论是：

> CE representation 下，donor-aware ratio correspondence 有清晰的三 seed 证据；在本轮 PIF representation 下，B 相对原 PI ratio 的小幅优势存在，但无法跨 shuffled 对照稳定复现。因此，PIF 下对“正确 correspondence 是收益来源”的因果主张尚不成立。

这不是失败后需要追加 donor selection、risk-weighted loss 或 ratio 调参的理由。相反，结果指出下一步若要继续，应先明确研究问题：是复核 PIF Stage-I/Stage-II 基线是否与论文报告一致，还是专门研究 PIF 改变原生 PI ratio 尺度后为何让 H2TF 训练变得更激进。二者都应是新的、预注册的对照，而不应事后修改本轮 A/B/C。

## 可复核文件

- `models/resnet_cifar.py`：PIF feature-map 返回与 `resnet32_pif_fe` factory。
- `config/cifar100/cifar100_imb001_stage1_ce_mixup_pif.yaml`：PIF Stage-I 配方。
- `saved/cifar100_imb001_stage1_ce_mixup_pif_202608082320/ckps/model_best.pth.tar`：冻结 PIF Stage-I checkpoint。
- `outputs/pi_h2t_ratio_only_pif_stage1seed123_seed42/summary.csv`
- `outputs/pi_h2t_ratio_only_pif_stage1seed123_seed43/summary.csv`
- `outputs/pi_h2t_ratio_only_pif_stage1seed123_seed44/summary.csv`
- `train_pi_h2t_ratio_only_stage2.py`：固定 A/B/C Stage-II 训练器。
