# PI-H2T 忽略 donor 与 pair-specific 供体选择验证

## 要验证的问题

对于 PI-H2T，融合控制量只依赖 recipient 的特征及其类别中心/分类器向量：

```text
rho_PI(r) = g(d(f_r, w_y_r))
```

因此，对固定 recipient `r` 的候选 donor `d_1 ... d_K`，PI-H2T 给出的分数完全相同，不能比较 donor A 与 donor B。

本文的 pair-specific 几何则直接计算真实融合路径。在固定 donor 比例 `lambda_0=0.25` 下，定义冻结 reference 分类器下、距离最近的带符号决策边界 margin：

```text
S(r, d) = min_{j != y_r} d_{y_r,j}((1-lambda_0) f_r + lambda_0 f_d)
```

目标是：在不让 recipient 过度危险的前提下，选择一个足够困难、可能更有训练价值的 donor。

## 不训练即可直接验证的 PI-H2T 缺陷

在 CIFAR100-LT IF100 上，对每个平衡采样的 recipient 提供 4 个长尾候选 donor，并使用冻结的 balanced cRT reference，得到：

| 同一 recipient 的 4 个 donor 内部统计量 | 数值 |
|---|---:|
| PI-H2T center score 标准差 | **0.0000** |
| pair `lambda_critical` 标准差 | **0.1560** |
| pair 融合后 signed-margin 标准差 | **0.2428** |
| 在 `lambda=0.25` 时同时存在安全与危险 donor 的 recipient 比例 | **43.08%** |

这直接证明了问题：对 donor 改变，PI-H2T 完全没有反应；而 pair-specific boundary geometry 对同一个 recipient 看到了显著不同的风险图景。

## 四策略 Stage-II 实验

所有策略均使用相同的 Stage-I CE ResNet-32、同一个冻结 balanced cRT reference、同一个 Stage-II CE classifier 初始化/优化器/损失、seed 42、`K=4` 与 `lambda=0.25`。每个策略收到的 recipient 图像、候选 donor 图像与标签、随机增强均逐位配对一致。

| 标签 | 策略 | 决策规则 |
|---|---|---|
| A | 随机 donor | 从 4 个候选中均匀随机选择 |
| B | 最安全 donor | `argmax_d lambda_critical(r,d)` |
| C | 相对 hardest-safe donor | 在 `S(r,d) >= T_r` 的候选中最小化 `S(r,d)` |
| D | 打乱的相对分数 | 打乱 pair 分数与 donor 槽位的对应，再使用 C 的规则 |

对 C，recipient-relative 阈值为：

```text
T_r = min(S(r, clean), 0) - delta,   delta=0
```

这是长尾 reference 存在偏差时必需的处理：若 reference 已经把一个 tail recipient 放在目标边界错误侧，C 只要求融合不要让它更差；若 clean recipient 已在正确侧，该约束就退化为通常的“不跨越边界”。

## 为什么绝对 hardest-safe 不是主实验

字面上的绝对规则 `S(r,d) >= 0` 会受到 reference 偏差的系统性影响：

| Recipient 组别 | Clean reference 判对 | 存在任一绝对安全 donor |
|---|---:|---:|
| Many | 76.21% | 80.98% |
| Medium | 47.19% | 54.60% |
| Few | 16.39% | 20.37% |

约一半 recipient、以及约 80% 的 Few recipient 没有任何满足绝对安全的候选。因而，使用相对约束才是在现有冻结 reference 下检验“不过度危险”的有效方式。

## Seed-42：风险受限的难度最大化结果

主指标是最后一个 epoch；`best epoch` 仅作诊断，因为它是在测试集上选择得到的。

| 策略 | 最后一轮 Overall | Many | Medium | Few | 最佳 Overall（epoch） |
|---|---:|---:|---:|---:|---:|
| A 随机 donor | **46.37** | 57.64 | 50.29 | **27.66** | 46.50 (8) |
| B 最安全 donor | 46.15 | 57.08 | 50.09 | 27.83 | 46.27 (6) |
| C 相对 hardest-safe | 46.28 | **58.31** | 50.29 | 26.52 | **46.51 (8)** |
| D 打乱相对分数 | 46.35 | 57.47 | **50.60** | 27.41 | 46.38 (8) |

seed-42 上的两个性能门槛都不成立：

```text
C > A ：不成立（最后一轮 46.28 < 46.37）
C > D ：不成立（最后一轮 46.28 < 46.35）
```

最佳 epoch 的微小差异来自在测试集上挑选 epoch，且远小于单 seed 训练波动，不能作为性能提升证据。

## C 是否正确实现了目标策略？

是。epoch 10 的训练阶段统计如下：

| 指标 | A 随机 | B 最安全 | C 相对 hardest-safe | D 打乱分数 |
|---|---:|---:|---:|---:|
| 选中 `lambda_critical` | 0.453 | **0.654** | 0.464 | 0.452 |
| 选中融合 signed margin | -0.179 | +0.038 | -0.274 | -0.179 |
| signed margin 绝对值均值 | 0.963 | 0.955 | **0.858** | 0.968 |
| 满足自身安全约束 | 44.33%* | 52.04%* | **96.62%** | 78.61% |
| 冻结 reference 的 flip proxy | 55.67% | **47.96%** | 50.32% | 55.42% |

`*` A/B 使用绝对不跨界的诊断，C/D 使用 recipient-relative 诊断，二者不能直接当作同一种安全率比较。

对 C，96.62% 的 recipient 至少有一个满足相对安全约束的 donor，C 从中选择最贴近边界的 donor。D 获得完全相同的候选流与阈值，但破坏了“几何分数—真实 donor”的对应，因而安全约束达成率降至 78.61%。

## 为什么随机 donor 看起来仍然最高

最后一轮 A 仅比 C 高 0.09 个百分点、仅比 D 高 0.02 个百分点；单个 seed 不能据此说随机策略真实更优。更重要的是，这不是候选池或配对实现出错：A/C/D 的候选流校验和一致，C 比 D 更常满足真实的“分数—donor”安全约束（96.62% 对 78.61%）。

真正的问题在于：当冻结 reference 已经误判 clean recipient 时，`relative safe` 故意弱于“带 recipient 标签的样本一定正确”。分组审计结果为：

| 策略 | 组别 | Clean reference 判对 | 融合后 reference 判对 | 满足相对约束 |
|---|---|---:|---:|---:|
| 随机 | Many | 76.21% | 72.08% | 86.21% |
| C 相对 hardest-safe | Many | 76.21% | 77.62% | 98.23% |
| 随机 | Medium | 47.19% | 43.11% | 76.89% |
| C 相对 hardest-safe | Medium | 47.19% | 50.05% | 97.08% |
| 随机 | Few | 16.39% | 13.94% | 71.71% |
| C 相对 hardest-safe | Few | 16.39% | 17.73% | 95.51% |

对于 Few，C 在 95.51% 样本上满足了“相对不更差”，但融合后仍有 82.27% 位于冻结 reference 的错误目标侧。因此该规则成功限制了额外 reference 风险，却可能把距离边界很近、recipient 标签仍不可靠的样本持续送入训练。这与最终表现一致：相对随机，C 在 Many 上 +0.67、Medium 基本持平、Few 上 -1.14 个点。

因此，失败的 C-vs-A/D 性能门槛否定的是下面这条额外主张：

> 最近的安全边界，不会自动等于最有训练价值的 donor。

它不影响两个已被验证的结论：PI-H2T 忽略 donor；pair-specific geometry 能排序并控制 donor 风险。

## 审计保证

- 所有四种策略在每一个 epoch 的 recipient/candidate 图像/标签流校验和均一致。
- 所有标量融合策略固定使用 `lambda=0.25`；未引入 loss weighting、组别重缩放、donor-mass compensation 或 reference-head 再训练。
- C 依据 pair geometry 选择 donor；D 使用相同的 K 个候选但打乱分数—donor 对应，因此 C 的结果不是 Best-of-K 计算量带来的收益。
- C 的结果不能归因于同类 donor：C 的同类 donor 比例仅 0.27%，随机策略为 1.02%。

## 诚实结论

诊断层面已经成立：

1. PI-H2T 从公式上就忽略 donor 身份。
2. recipient-donor pair geometry 在 donor 间显著变化，能区分安全与危险候选。
3. 在冻结 reference 下，它可以实施定义明确的 recipient-relative 风险约束，并选择困难融合样本。

最终分类性能层面尚未成立：这一版“风险受限的难度最大化”没有在 seed 42 上击败随机 donor 或打乱分数对照。因此不应扩展到多 seed，也不能宣称性能提升。

下一版方法必须在 donor-conditioned 风险几何之外，额外刻画 *training value*；同时保留当前的配对 A/B/C/D 框架。它不应回到 loss weighting、reference head 调参或 recipient-only 分数的路线。

## 对应文件

- `train_donor_selection_stage2.py`：K 候选、严格配对的训练器；含随机、最安全、绝对 hardest-safe 与 recipient-relative hardest-safe 策略。
- `analyze_hardest_safe_availability.py`：直接验证 PI-H2T 忽略 donor，并审计候选的绝对安全性。
- `analyze_relative_hardest_safe_by_group.py`：按频次组别区分“相对安全”与“目标标签正确性”的审计。
- `outputs/relative_hardest_safe_selection_balref_seed42_k4_lambda025/summary.csv`：各策略指标与候选流校验和。
