# PI-H2T：Center-r vs Boundary-r（PIF，seed 42）

## 目的

本轮只回答一个前置问题：在 **不使用 donor 信息** 的条件下，recipient 的 signed boundary margin 是否比原 PI-H2T 的类别中心 cosine 信息更适合决定 H2TF ratio。

因此没有使用 donor-conditioned geometry、`lambda_critical`、boundary clipping、`delta`、OOF reference、reliability `q`、controlled crossing、donor selection 或额外 loss。

## 固定条件

- CIFAR100-LT，IF=100，PIF Stage-I checkpoint，seed 42；
- frozen PIF backbone + released H2T Stage-II BN shift；
- class-balanced recipient、instance-wise donor、原 H2TF、CE、SGD、10 epochs；
- trainable classifier 始终从同一原 Stage-I CE classifier 初始化；
- 原 Stage-I CE classifier 同时用于 PI center score 与 boundary margin；
- 固定第 10 epoch 评估一次测试集，不用 test 选 epoch；
- 每个 epoch 显式重置训练 RNG，因此三组的 recipient/donor/augmentation stream checksum 完全一致。

PIF 已冻结的全局匹配尺度为：

```text
lambda_A = kappa_match * (1 - r_PI),  kappa_match = 0.9267667644
```

它对 A/B/C 完全相同，不是本轮调参。

## A/B/C 定义

### A：原 center-r

使用发布版 PI-H2T 的 recipient center cosine score 得到 `r_PI`，再按上面的固定 `kappa_match` 得到 donor mass `lambda_A`。

### B：recipient boundary-r

对每个 recipient pooled feature `f_i`，在原 Stage-I CE head 下计算：

```text
m_i = min_{j != y_i}
      ((w_y-w_j)^T f_i + (b_y-b_j)) / (||w_y-w_j||_2 + eps)
```

`m_i > 0` 表示在正确一侧，`m_i = 0` 在最近决策边界，`m_i < 0` 在错误一侧。

在每个 `Many/Medium/Few × lambda_A-bin` 内，保留 A 的 **全部 donor-mass multiset**，把更大的 `lambda_A` 分配给更大的 `m_i`：

```text
m_i ↑  ->  lambda_B,i ↑  ->  r_B,i = 1-lambda_B,i ↓
```

所以 B 只替换 center information 为 boundary information，不改变平均融合强度、频率组质量或 ratio 分布。

### C：shuffled boundary-r

保留 B 在每一个 stratum 内的 exact ratio multiset，使用 random derangement 打乱 `lambda_B -> recipient` 的对应关系。实际 donor、recipient、fusion pair 都不变；不会重新计算、clip 或投影 ratio。

## 结果：固定第 10 epoch

| 方法 | Overall | Many | Medium | Few | H-mean |
|---|---:|---:|---:|---:|---:|
| A：center-r | 37.20 | 30.44 | 47.03 | 33.72 | 32.00 |
| B：boundary-r | 37.12 | 30.19 | 46.86 | **33.97** | 31.97 |
| C：shuffled boundary-r | **37.56** | **30.86** | **47.49** | 33.90 | **32.31** |

关键差值（百分点）：

- `B - A`：Overall -0.08，Many -0.25，Medium -0.17，Few +0.24，H-mean -0.03；
- `B - C`：Overall -0.44，Many -0.67，Medium -0.63，Few +0.07，H-mean -0.34；
- `C - A`：Overall +0.36，Many +0.42，Medium +0.46，Few +0.17，H-mean +0.31。

## 因果审计

末轮 A/B/C 的 donor mass mean 都为 0.714177；Many/Medium/Few donor mass 也逐项一致：

| 组别 | Many | Medium | Few |
|---|---:|---:|---:|
| A | 0.726151 | 0.715895 | 0.697219 |
| B | 0.726151 | 0.715895 | 0.697219 |
| C | 0.726151 | 0.715895 | 0.697219 |

- B 的 A/B ratio multiset 最大误差：0；
- C 的 B/C ratio multiset 最大误差：0；
- B 的 `m -> lambda_B` 单调性违例：0；
- C 最后一轮可打乱样本比例：99.9908%，实际移动比例：100%，ratio 真正重分配比例：99.9908%；
- A/B/C 每个 epoch 的 recipient、donor、target stream checksum 完全相同。

因此结果不能归因于 donor mass、ratio 分布、频率组质量、actual fusion pair 或数据增强流的不同。

B 也并非退化为 A：在 100 个类别上，B 相对 A 的类别平均 donor-mass 改变量绝对值均值为 0.01016，Few 组为 0.01474；单类别改变量范围为 -0.04002 到 +0.02564。也就是说，boundary margin 确实重新安排了 ratio，只是这种重新安排没有转化成 B 相对于 A/C 的稳定优势。

## 判断

本轮的预注册成功条件是：

```text
B > A：H-mean 和 Few 提升，Medium 不下降；
B > C：H-mean、Few、Medium 提升。
```

结果为：

- B 只获得了 +0.24 的 Few，H-mean 与 Medium 反而下降，因此 `B > A` 不成立；
- C 在 Overall、Many、Medium、H-mean 上高于 B，因此 `B > C` 不成立。

所以目前 **不能声称 boundary-r 优于原 PI-H2T center-r**。更精确地说，当前的 signed-margin 排序可能带来很小的 Few 倾向，但这个效应小于/不稳定于将同一 ratio 集合随机重排所带来的训练波动，尚没有“正确 boundary-to-recipient correspondence 有效”的证据。

按照先前冻结的主线，此处不进入 donor 讨论，也不直接扩展 seed 43/44。下一步应先审视 boundary-r 的 recipient-level 映射本身，例如 margin 的跨类别可比性或是否应以 boundary hardness 而非“安全程度”排序；但这属于新假设，需在改公式前单独设计新 A/B/C，而不是从当前结果继续叠加 donor 模块。

## 文件

- 实验配置：[experiment.json](outputs/center_boundary_r_pif_seed42/experiment.json)
- 汇总结果：[summary.csv](outputs/center_boundary_r_pif_seed42/summary.csv)
- 每类 margin/ratio 审计：[boundary_r class audit](outputs/center_boundary_r_pif_seed42/boundary_r/last_epoch_class_audit.csv)
- 训练代码：[train_center_boundary_r_stage2.py](train_center_boundary_r_stage2.py)
