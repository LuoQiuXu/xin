# H2T 供体判别扰动机制审计（第一阶段）

完成时间：2026-10-12T19:14:04+08:00。独立审计目录：[donor_perturbation_20261012_phase1_seed123](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123>)。

## 1. 回答本轮问题

**是：在这两个已锁定模型、训练子集、相同接收特征、相同候选供体类别、相同融合通道及 rho=0.25 下，不同候选供体产生了有实际判别含义的扰动差异。**

主分析四候选 D 极差中位数为 **1.381**，95 分位为 **3.354**（logit units）；作为尺度参考，原始 |Margin| 中位数为 2.969。
**5,404/20,000 组（27.02%）**在同一接收样本上，四个候选使冻结参考分类器出现部分判对、部分判错；Tail 为 **1,844/5,792 组（31.84%）**。这是决策事件的差异，没有用人为选定的 D 阈值定义“明显”。

本轮结论支持进一步研究候选供体的判别差异，**没有证明 Low-D、Mid-D 或 High-D 的训练效果更好，也没有把正 D 或大 D 等同于有害融合**。没有训练任何新模型，没有选择或部署供体策略。

## 2. 实际模型、数据、特征及控制协议

### 锁定 Checkpoint

- 特征提取：Stage I epoch 198 最佳 Checkpoint（历史 Overall 40.49%），[Stage I 文件](<D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428/ckps/model_best.pth.tar>)。
- SHA-256：`0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e`。
- 参考分类器：固定 BN 的 DR 第 10 epoch Checkpoint，[固定 BN DR epoch 10 文件](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/runtime/saved/cifar100_if100_stage2_fixed_bn_20261011_2255_dr_202610112255/ckps/model_final.pth.tar>)；仅取其中的 Classifier 权重。
- SHA-256：`1b054d8b45753c25dedd9897e3052e5f33c73b0c38b47acaa277c0f61c50283d`。
- ResNet-32 `resnet32_fe()`：[5,5,5] BasicBlocks，feature_tuning=False；输出最后一层 ReLU 后 **[B,64,8,8]** 特征图。
- 使用原始 `Classifier(feat_in=64,num_classes=100)`：`F.avg_pool2d(x,x.size()[3])` → flatten [B,64] → `fc`；没有在审计中重新定义分类器或用别的投影代替。
- 实例化并严格加载 DataParallel 模型，所有模块 `eval()`、参数 requires_grad=False、全程 no_grad；从未创建 optimizer、调用 backward 或训练。参考模型始终冻结。
- 冻结 DR Checkpoint 的整个 Backbone state_dict 与 Stage I 逐位一致，加载时再次核验；两者的 BN counters 均为 16830。

### 数据与预处理

- 只实例化 CIFAR-100 的 train=True 长尾子集：IF=100，100 类，10,847 个实例，每类 500→5。子集像素字节 SHA-256 与此前复现记录一致。训练索引在本报告中均为重排后 LT 子集索引，**不是 CIFAR 原始 50,000 张序号**。
- 未实例化测试 dataset，未解码或用于分析测试图像/标签；历史测试表现未参与候选构造、排序或参数选择。torchvision 的数据完整性检查可能读取测试文件字节校验 MD5，这不涉及测试样本评分或策略选择。
- **主分析**：对训练子集使用官方 eval_transform（ToTensor + 原始 Normalize），不作随机裁剪/翻转，以隔离供体身份差异。
- **敏感性核验**：每个训练实例使用一次固定 seed=129 的官方 train_transform（RandomCrop(32,padding=4) + RandomHorizontalFlip + ToTensor + 原始 Normalize）；固定缓存后沿用同一接收者、同一候选、同一掩码重新评分。
- 两套 transform 表达式从未改动的官方 `CIFAR100_LT.__init__` AST 中直接提取并实例化。Normalize 仍为均值 (0.4914,0.4822,0.4465)、标准差 (0.2023,0.1994,0.2010)。
- 特征图以 float32 保存，每套完整银行 [10847,64,8,8]；同一实例在一套银行内复用相同增强/特征，避免重复候选因不同增强而产生额外差异。这与实际 Stage II 每次抽样都重新增强的在线训练不同。

### 抽样与掩码

- 抽取 **20,000 个 BS 接收样本组**：Python Random 按官方顺序均匀抽类别，再在该类内随机抽实例（有放回）。20,000 是抽样次数，覆盖 6611 个不同训练实例。
- 每组先按 `P(c_h=c)=n_c/10847` 抽一个 IS 供体类别，再类内均匀、有放回抽 K=4 个供体实例。同组四个供体属于同一类别。允许重复、同类和自供体，全部保留。
- 每次调用仍为 **原始 methods.H2T**，rho=0.25；64 通道随机排列取前 16，对接收特征做原地赋值，保留其余 48 通道，融合后标签保持接收者标签。没有改变融合函数签名或传入新选择规则。
- 为保留官方函数的 batch 共享掩码，32 个接收组 × K4 合成 B=128 调用原始函数，一批 32 组共用一份 M；因此每套数据有 625 份实际随机掩码，而不是 20,000 份独立掩码。每组内 K4 完全共享通道，满足控制要求。
- 通道 CPU RNG 单独用 seed=126，并在 fork_rng 中调用原始 H2T；用独立 predictor 核验原始 randperm 的 16 个通道及 RNG 消耗，两套预处理均重置同一通道流。环境数据 RNG 不被 H2T 扰动。
- 固定随机流：`bs=123; is_class=124; within_class=125; channel=126; random_candidate=127; mid_candidate=128; train_augmentation=129`；PYTHONHASHSEED=123。后端配置：`{'cudnn_benchmark': False, 'cudnn_deterministic': False, 'deterministic_algorithms': False}`。

**与原始 BS+IS 在线过程的区别必须保留在解释中**：IS 的类别边缘概率相同，但原仓库 IS 来自每 epoch 打乱的实例遍历；本审计独立、有放回抽类别，并一次在类内构造四个候选，再对冻结缓存评分。类内候选联合选择方式、跨 batch 的遍历过程和增强复用均改变。随机从 K4 选一个的供体边缘分布仍是 IS；Low/Mid/High 条件排序后的类内实例分布可能改变。本审计不是原始训练轨迹的重放。

## 3. 正确性与保护核验

- 独立 runtime 复制已审计的 48 个文件，与原副本和历史 manifest 的 SHA-256 一致。核心源码新增修改为 0；沿用已批准 Windows 兼容性补丁。原 commit 标记 `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8` 来自原审计中的官方 archive。
- Feature forward / 直接原始池化+fc 与 Classifier forward 的 logits **逐位相同**；B=128 的同一接收特征复用四份，融合前四个 Margin 一致，避免不同 batch shape 造成数值差异。四候选由相同基准特征独立复制后一起融合，不会在同一接收特征上累积四次原地替换。
- 主分析与增强核验各 625 个融合 batch：每次检查 16 替换通道等于供体，48 保留通道等于接收者，供体及标签不变、环境 RNG 不变。1250 次原始 H2T 调用全部通过。
- 重复供体索引在相同缓存和掩码下的 D 完全相同。5 个自供体候选的 D 与特征 L2 均恰为 0；额外身份融合检查亦为 0，未用数值噪声制造差异。
- 同时验证 D=(真实类 logit 下降)+(最大竞争类 logit 上升)；最大 float32 代数误差：主分析 0.000000954，增强核验 0.000000954。
- Backbone 和参考 Classifier 的完整 state_dict 字节哈希前后相同，涵盖 Conv、BN affine、running_mean、running_var、num_batches_tracked 和分类器全部权重。
- 708 个历史文件（原始/固定 BN 实验的源码、日志、报告、Checkpoint 等）前后 SHA-256 一致；Stage I 和参考 DR Checkpoint 单独复核一致。
- 证据：[correctness_verification.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/correctness_verification.json>)；[historical_before.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/historical_before.json>) / [historical_after.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/historical_after.json>)。

## 4. K4 构造与采样分布验证

BS 均匀类别分布：χ²=94.350，df=99，p=0.6134；IS 的 n_c/10847 分布：χ²=101.082，df=99，p=0.4230。类别频数与声明分布没有明显失配证据；这不等于证明有限抽样频数完全一致。概率来自实际子集 n_c，且逐候选验证标签确属抽取类别。

| 接收组 | BS 接收次数 | 不同接收实例 | IS 供体类别次数 |
| --- | ---: | ---: | ---: |
| Head | 7306 | 4733 | 16340 |
| Medium | 6902 | 1592 | 3104 |
| Tail | 5792 | 286 | 556 |

Head=[0,36)，Medium=[36,71)，Tail=[71,100)。这里的供体次数为抽类别次数，每次对应 4 个候选；Tail 的 5792 次接收只覆盖 286 个不同训练实例，不能当作同等数量的独立图像。

625 个掩码中每通道出现 122–180 次，均匀抽样期望为 625×16/64=156.25；完整掩码及频数保存，没有按通道表现筛选。

![BS 与 IS 类别分布验证](D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/figures/01_sampling_distributions.png)

## 5. Margin 扰动总体与类别组分布

定义 m=z_y−max_{j≠y}z_j，D=m_before−m_after；D>0 为真实类别 Margin 下降，D<0 为上升，D=0 为不变。竞争类别在计算前明确排除 y。单位是锁定分类器的 logits，换分类器或权重尺度后不可直接跨模型比较绝对 D。

| 接收组 | 候选数 | D 均值 | D P05 | D P50 | D P95 | D>0 (%) | D<0 (%) | D=0 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 80000 | 1.739 | -0.755 | 1.481 | 5.134 | 85.04 | 14.96 | 0.006 |
| Head | 29224 | 1.611 | -0.947 | 1.333 | 5.149 | 81.72 | 18.27 | 0.010 |
| Medium | 27608 | 1.790 | -0.807 | 1.509 | 5.308 | 84.50 | 15.49 | 0.007 |
| Tail | 23168 | 1.839 | -0.419 | 1.613 | 4.865 | 89.86 | 10.14 | 0.000 |

| 接收组 | Margin 前 P50 | Margin 后 P50 | 前 \|Margin\| P50 | 前参考分类器判对 (%) |
| --- | ---: | ---: | ---: | ---: |
| All | 2.965 | 1.239 | 2.969 | 95.54 |
| Head | 2.996 | 1.401 | 3.000 | 94.22 |
| Medium | 3.145 | 1.345 | 3.147 | 94.68 |
| Tail | 2.747 | 0.973 | 2.747 | 98.20 |

融合前 Margin 每个接收组计一次；融合后四候选全部纳入。上表判对率为训练子集上的冻结参考分类器诊断值，非测试准确率，不能解释成泛化性能。主分析 D 的全部连续分位数、均值、标准差及极值保存在 analysis_summary.json。

## 6. 同一接收样本内部的候选差异

| 接收组 | 极差 P05 | 极差 P25 | 极差 P50 | 极差 P75 | 极差 P95 | 极差 P99 | 候选 D_std P50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 0.447 | 0.905 | 1.381 | 2.032 | 3.354 | 4.585 | 0.539 |
| Head | 0.460 | 0.935 | 1.445 | 2.094 | 3.442 | 4.657 | 0.564 |
| Medium | 0.436 | 0.902 | 1.364 | 2.013 | 3.287 | 4.453 | 0.531 |
| Tail | 0.445 | 0.876 | 1.336 | 1.982 | 3.328 | 4.542 | 0.520 |

| 接收组 | 极差>0 组 (%) | 同时含 D<0 与 D>0 (%) | 候选判对/判错混合组 (%) | 候选预测类别不同组 (%) |
| --- | ---: | ---: | ---: | ---: |
| All | 100.00 | 22.87 | 27.02 | 34.49 |
| Head | 100.00 | 26.27 | 25.54 | 33.21 |
| Medium | 100.00 | 22.98 | 24.54 | 32.01 |
| Tail | 100.00 | 18.46 | 31.84 | 39.07 |

主分析极差最小 0.007925，最大 8.476721；所有 20,000 组极差非零，但“非零”本身不作为有意义的唯一证据。候选组内标准差按 ddof=0 计算，描述这四个候选的离散程度，不将其当作总体标准差的无偏估计。
每组保存 D_min、D_max、极差、标准差及完整四候选排序；并列使用原候选位置稳定排序，排序只用于描述。没有给 D 极差设定一个人为通过阈值。判对/判错混合以及 D 跨 0 使用的是原定义中的决策事件，具有直接解释。

**Tail 并非极差最大的一组**：其极差中位数低于 Head，但候选判对/判错混合比例更高。现有证据说明 Tail 也存在候选判别差异，不能据此推断差异仅发生于 Tail 或对 Tail 的训练收益最大。

![D 分布及同一接收样本候选极差 ECDF](D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/figures/02_margin_and_within_receiver_range.png)

## 7. 相似度、L2 距离与 D 的关系

主定义：余弦相似度为接收/供体完整 [64,8,8] 展平后的 cosine；L2 为融合前后完整特征图差值的欧氏范数。同时报告分类器所见 GAP 后 [64] 向量的 cosine / L2，避免把空间差异与分类器输入差异混为一谈。

| 度量 × D（All） | Pearson | Spearman | 组内去均值 Pearson |
| --- | ---: | ---: | ---: |
| cosine_spatial | -0.101 | -0.088 | -0.173 |
| cosine_gap | -0.126 | -0.111 | -0.247 |
| l2_spatial | 0.473 | 0.439 | 0.385 |
| l2_gap | 0.527 | 0.486 | 0.517 |

| 接收组 | 空间 cosine Pearson | 空间 L2 Pearson | GAP cosine Pearson | GAP L2 Pearson |
| --- | ---: | ---: | ---: | ---: |
| Head | -0.099 | 0.422 | -0.109 | 0.477 |
| Medium | -0.082 | 0.503 | -0.101 | 0.549 |
| Tail | -0.131 | 0.503 | -0.169 | 0.572 |

组内去均值 Pearson 先在每个 K4 组内减去该指标和 D 的各自均值，再合并计算，用于区分候选组内变化与不同接收者之间的变化。相关性为描述性结果，**不代表因果关系，也不能直接用 cosine 或 L2 替代 Margin 评分**。两者均由相同特征及掩码决定，可能存在数学耦合。重复接收实例、重复候选和共享 batch 掩码使行之间有关联，未把 80,000 行当作独立样本报告相关性显著性 p 值。

![相似度和融合 L2 与 D 的关系](D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/figures/03_similarity_and_distance_vs_D.png)

### 相似度接近但 D 不同的候选

对每组 6 个候选对，选择**空间 cosine 差最小的一对**，不依据 D 来选，也不用人工绝对相似度阈值。以下两列为同一批选中候选对的边缘统计，不暗示它们的中位数来自同一个候选对。

| 接收组 | 最近对 \|Δcos\| P50 | 最近对 \|ΔD\| P50 | 最近对 \|ΔD\| P95 |
| --- | ---: | ---: | ---: |
| All | 0.00758 | 0.596 | 2.262 |
| Head | 0.00738 | 0.621 | 2.337 |
| Medium | 0.00786 | 0.585 | 2.178 |
| Tail | 0.00758 | 0.579 | 2.243 |

因此相近的余弦值并不保证相近的判别 Margin 扰动。全部 120,000 个组内候选对按 |Δcos| 排成 10 个等数量区间后，也报告每区间 |ΔD| 连续分布（下图及 JSON），没有只保留一个有利阈值区间。

![同组候选对的余弦差与判别扰动差](D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/figures/04_similar_cosine_different_D.png)

## 8. 决策边界与最强竞争类别变化

“跨越参考分类器决策边界”主定义为 argmax 预测类别与融合前不同；另外区分正确→错误、错误→正确和真实类别 Margin 的符号跨越，后两类条件率使用各自融合前状态的候选数作分母。最强竞争类别是排除 y 后的 argmax，可能在整体预测仍正确时变化。

| 接收组 | 预测类别变化 (%) | 正确→错误 条件率 (%) | 错误→正确 条件率 (%) | 最强竞争类变化 (%) |
| --- | ---: | ---: | ---: | ---: |
| All | 25.80 | 24.30 | 22.28 | 61.65 |
| Head | 24.95 | 22.68 | 25.65 | 63.72 |
| Medium | 25.13 | 23.49 | 18.60 | 58.37 |
| Tail | 27.68 | 27.18 | 21.63 | 62.94 |

All：预测类别变化 20,641/80,000；正确→错误 18,570/76,428；错误→正确 796/3,572；最强竞争类变化 49,320/80,000。
这些是融合输入上的即时参考分类器行为。合成特征仍按接收者标签解释；模型误判与后续训练是否有益是不同的问题，本轮没有估计训练或泛化收益。完整 margin 符号跨越率、零 margin 数及分母均在 JSON 中保留。

![决策事件与预处理敏感性](D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/figures/06_decisions_and_augmentation_sensitivity.png)

## 9. 原始随机选择与 Low / Mid / High D 的描述性比较

Random 为每组用独立 RNG 在四候选位置中均匀选一个（位置计数 [4944, 5030, 4981, 5045]）；Low-D 为原始 D 最小的候选，High-D 为最大；Mid-D 在 D 升序第 2 或第 3 名中均匀选择，**不是最接近零的 D**。重复得分按稳定位置排序。
Random 的供体边缘分布与本审计声明的 IS 相同，因此刻画原始随机融合抽样思想下的扰动；其类别/实例遍历节奏和冻结缓存仍与在线 Stage II 不同，不能声称这里重放了原始训练的每个 batch。

| 选择（All） | D 均值 | D P05 | D P50 | D P95 | 融合后参考判对 (%) | 预测类别变化 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 1.735 | -0.758 | 1.461 | 5.158 | 73.18 | 25.88 |
| Low-D | 0.967 | -1.256 | 0.753 | 3.912 | 86.09 | 13.94 |
| Mid-D | 1.719 | -0.617 | 1.467 | 4.941 | 74.18 | 24.85 |
| High-D | 2.540 | -0.010 | 2.272 | 6.032 | 59.07 | 39.43 |

| 选择（Tail） | D 均值 | D P05 | D P50 | D P95 |
| --- | ---: | ---: | ---: | ---: |
| Random | 1.828 | -0.449 | 1.590 | 4.875 |
| Low-D | 1.086 | -0.850 | 0.913 | 3.671 |
| Mid-D | 1.824 | -0.293 | 1.607 | 4.687 |
| High-D | 2.618 | 0.327 | 2.384 | 5.725 |

Low/Mid/High 的 D 分布顺序由排序定义产生，其参考判对率差异也受所用标签 Margin 直接影响，不能当作独立训练验证。特别是较大正 D 可能只是使冻结分类器面对更难的训练输入；较小或负 D 也不自动代表更好的增强。没有预设中等扰动最优。

![Random 与三种 D 排序选择的描述性分布](D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/figures/05_random_low_mid_high_D.png)

## 10. 重复候选、同类别和自供体

| 项目 | 计数 / 分母 | 比例 (%) |
| --- | ---: | ---: |
| 存在重复候选的接收组 | 1017/20000 | 5.085 |
| 超出不同实例数的重复候选位置 | 1070/80000 | 1.3375 |
| 供体类别=接收类别的组 | 185/20000 | 0.925 |
| 包含自供体的组 | 5/20000 | 0.025 |
| 自供体候选 | 5/80000 | 0.00625 |

按实际 IS 类别概率和类内有放回采样公式，重复组率理论期望为 5.0966%；实际 5.0850%。每组不同实例数直方图（1/2/3/4）为 {'1': 0, '2': 53, '3': 964, '4': 18983}。本配置最少 5 个实例；代码允许更小类别有放回构造 K4，不会因不足 K 而排除。

| 同类情况 | 接收组数 | D 均值 | D P50 | 极差 P50 | 预测类别变化 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| same_class | 185 | -0.289 | -0.254 | 1.313 | 6.35 |
| different_class | 19815 | 1.758 | 1.498 | 1.381 | 25.98 |

同类融合 Head / Medium / Tail 接收组数为 [155, 29, 1]；Tail 同类只有 1 组，不能作稳定的 Tail 同类结论。所有同类和重复候选均包含在总体统计内，并单独标记，未删除。

## 11. 典型候选差异实例

示例按 Head / Medium / Tail 各自 **D 极差排序的 10%、50%、90% 位置**机械选取（9 组），未按正负 D、是否判对或某个候选“有利”选择。实例用于说明机制；全部总体统计才是结论依据。候选位置与排名均为 0-based 位置、1-based 名次，索引对应缓存 LT 子集。

| 组 / 极差位置 | 接收组 ID | 接收 / 供体类别 | 四个 D（原位置） | 极差 | D 升序位置 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Head q10 | 4933 | 30 / 6 | -0.173, 0.271, 0.430, -0.027 | 0.603 | [0, 3, 1, 2] |
| Head q50 | 10317 | 34 / 16 | 3.585, 3.169, 2.139, 2.678 | 1.445 | [2, 3, 1, 0] |
| Head q90 | 9370 | 8 / 37 | 3.251, 4.789, 6.095, 3.221 | 2.873 | [3, 0, 1, 2] |
| Medium q10 | 9380 | 40 / 4 | -0.548, 0.041, -0.197, -0.174 | 0.589 | [0, 2, 3, 1] |
| Medium q50 | 8957 | 37 / 5 | 1.187, 1.715, 0.589, 0.352 | 1.364 | [3, 2, 0, 1] |
| Medium q90 | 16598 | 37 / 2 | -0.590, 0.272, 0.006, -2.486 | 2.758 | [3, 0, 2, 1] |
| Tail q10 | 15156 | 81 / 16 | 0.919, 0.851, 0.396, 0.338 | 0.580 | [3, 2, 1, 0] |
| Tail q50 | 13914 | 95 / 75 | 0.629, 0.924, 1.965, 0.629 | 1.336 | [0, 3, 1, 2] |
| Tail q90 | 4980 | 74 / 26 | 1.294, 1.945, 1.555, -0.802 | 2.747 | [3, 0, 2, 1] |

例如 Tail 的中位极差示例中，重复供体保持同分，同时另一个候选使 Margin 变为负值；图中也包含四候选都判对、都判错、Margin 上升等情况，展示不一致的扰动方向，而非选择一种结果证明策略有效。每例最近 cosine 候选对的 |Δcos| / |ΔD|、完整候选索引与 logits-derived Margin 存在 analysis_summary.json。

![按极差分位位置机械选择的九组供体候选实例](D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/figures/07_systematic_candidate_examples.png)

## 12. 固定官方训练增强的敏感性核验

接收索引、K4 供体索引、供体类别、Random/Mid 选择流和 625 份 M 均不变，仅将训练图像的预处理换为固定缓存的官方训练增强。不是另一个训练实验。

| 预处理 / 接收组 | D 均值 | 极差 P50 | 极差 P95 | 候选判对/判错混合 (%) |
| --- | ---: | ---: | ---: | ---: |
| eval 主分析 / All | 1.739 | 1.381 | 3.354 | 27.02 |
| eval 主分析 / Tail | 1.839 | 1.336 | 3.328 | 31.84 |
| 固定 train augmentation / All | 1.586 | 1.268 | 3.079 | 26.90 |
| 固定 train augmentation / Tail | 1.647 | 1.203 | 3.006 | 32.06 |

两模式逐候选 D 的 Pearson=0.870，逐组极差的 Pearson=0.687。Low-D 被选候选**位置**一致率 60.81%，High-D 为 61.10%。

差异存在的总体证据在这次增强核验中仍成立，但具体候选排序明显依赖增强。没有证明相同实例在所有增强、所有随机通道或所有分类器下都有稳定的 Low/High 身份。

## 13. 是否值得第二阶段训练，以及尚不能归因的内容

**值得在获得下一步确认后检验 Low-D / Mid-D / High-D 的训练效果。**依据是连续极差的尺度、组内判别决策变化、增强核验中的差异延续和相近 cosine 候选仍有不同 D，说明候选排序具有可研究的空间。

这些证据只证明“供体不同会产生不同的冻结参考分类器扰动”。它们没有给出供体好坏的因果标签：参考分类器在该训练子集上训练过，训练 Margin 具有拟合偏差；对训练合成输入的即时判对/判错不是未来泛化效果；本轮只有一个抽样种子组合、一个通道流和一个固定增强缓存。重复接收与 batch 共用掩码也限制独立样本解释。
后续若获授权，公平对照应包含同样 K4 构造下的 Random-of-4，控制 BS/IS 类别、接收者增强、候选增强、通道掩码、BN 条件、训练预算及参考分类器冻结方式，并报告多随机种子的变化。Low/Mid/High 均应作为待检验条件，不提前指定最优方向。当前没有启动这些训练，也没有修改官方 H2T。

## 14. 可复查产物、命令和资源

- 实际评分脚本：[audit_donors.py](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/audit_donors.py>)；统计脚本：[analyze_donors.py](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/analyze_donors.py>)；绘图：[plot_donors.py](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/plot_donors.py>)；报告：[report_donors.py](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/report_donors.py>)。
- 全部抽样索引及 Random/Mid 位置：[sampling_plan.npz](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/sampling_plan.npz>)；16 通道掩码：[channel_masks.npy](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/channel_masks.npy>)；每类分布：[sampling_distribution.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/sampling_distribution.csv>)。
- 主分析原始候选：[eval_preprocess_candidates.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/eval_preprocess_candidates.csv>) / [eval_preprocess_statistics.npz](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/eval_preprocess_statistics.npz>)；每组极差和排序：[eval_preprocess_receiver_groups.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/eval_preprocess_receiver_groups.csv>) / [eval_preprocess_ranked_groups.npz](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/eval_preprocess_ranked_groups.npz>)。
- 增强核验原始候选：[fixed_train_aug_candidates.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/fixed_train_aug_candidates.csv>) / [fixed_train_aug_statistics.npz](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/fixed_train_aug_statistics.npz>)；每组排序：[fixed_train_aug_receiver_groups.csv](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/fixed_train_aug_receiver_groups.csv>)。
- 完整统计（全部分位数、相关性、分母、示例、分布验证）：[analysis_summary.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/analysis_summary.json>)；完整 feature maps：[eval_preprocess_feature_maps.npy](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/eval_preprocess_feature_maps.npy>) / [fixed_train_aug_feature_maps.npy](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/fixed_train_aug_feature_maps.npy>)。
- 图表：7 组 PNG + SVG 位于 [图表目录](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/figures>)；评分日志：[extraction_scoring.log](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/extraction_scoring.log>)。
- 配置及软件版本：[preflight.json](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/preflight.json>) / [effective_source_config.yaml](<D:/xiangmuxunlian/新/H2T_repro_audit/donor_audit_runs/donor_perturbation_20261012_phase1_seed123/effective_source_config.yaml>)。YAML 为加载的源配置，其中 Stage II 训练超参数没有执行；实际审计数量、预处理、K4、模型引用和 RNG 流以 preflight 为准。

GPU：NVIDIA GeForce RTX 5070 Laptop GPU；PyTorch 2.11.0+cu130 / torchvision 0.26.0+cu130 / CUDA 13.0。两套提取、评分及结尾完整性核验耗时 19.56 秒；PyTorch peak allocated 65.47 MiB，peak reserved 98.00 MiB。这是进程 allocator 的峰值，不是含桌面的全 GPU 显存。

已执行命令（PowerShell；工作目录为本工作区）：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONHASHSEED = '123'
$auditRoot = 'D:\xiangmuxunlian\新\H2T_repro_audit\donor_audit_runs\donor_perturbation_20261012_phase1_seed123'
& 'D:\app\anaconda\envs\pytorch\python.exe' -u "$auditRoot\audit_donors.py" prepare
& 'D:\app\anaconda\envs\pytorch\python.exe' -u "$auditRoot\audit_donors.py" run
& 'D:\app\anaconda\envs\pytorch\python.exe' -u "$auditRoot\analyze_donors.py"
& 'D:\app\anaconda\envs\pytorch\python.exe' -u "$auditRoot\plot_donors.py"
& 'D:\app\anaconda\envs\pytorch\python.exe' -u "$auditRoot\report_donors.py"
```

评分启动器拒绝覆盖已有运行和根目录报告。独立复查可利用保存的 feature maps、抽样计划、M、Classifier 权重重算；若复制脚本到新目录重新提取，应同时将复制脚本中的报告输出名及报告存在性保护改为新名称，保留本次记录。这只涉及独立审计脚本的输出保护，不涉及 H2T 核心源码。

**完成状态：第一阶段特征提取、K4 构造、Margin/距离计算、统计与图表全部完成。已停止，没有启动 Low-D / Mid-D / High-D 训练或任何其他改进实验。**
