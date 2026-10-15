# Donor-conditioned fusion risk 最小验证报告

## 结论

核心假设通过，但闭式解必须作为上界使用，不能直接取边界端点。

- 风险预测：通过。donor-conditioned score 在三个重采样 seed 上稳定优于 PI-H2T center ratio 和 recipient-only boundary ratio。
- 风险控制：在 `gamma <= 0.85` 时通过；推荐实际使用 `lambda_safe = 0.8 * lambda_critical`。
- 原始端点 `gamma = 1`：不通过。此时融合特征恰好落到最近决策边界，容易产生 tie，并放大 reference head 与独立 probe 的边界失配。
- 因此，值得进入完整训练，但完整训练不应使用裸 `lambda_critical`，应使用带内部安全收缩的比例。

## 固定实验协议

- 数据：CIFAR100-LT，imbalance factor 100，类别数量从 500 递减到 5。
- 固定表示：本地已完成的 ResNet-32/CE Stage-I 特征。
- 固定 reference classifier：对应 Stage-I 线性分类头。
- 特征源测试准确率：40.75%。
- recipient：从 LT 训练特征按类别均匀采样。
- donor：从 LT 训练特征按实例均匀采样，因此保持 head-biased 分布。
- 每个 seed 采样 30,000 对，共使用 seed 42、43、44。
- 独立 probe：每类随机抽取 50 个测试特征，共 5,000 个特征拟合 balanced multinomial logistic regression。
- harmful fusion：probe 在融合前预测正确、融合后预测错误。
- 该 probe 使用测试特征仅用于 go/no-go 诊断，不可作为最终论文测试协议；正式实验应改用独立 validation split。

## 三种 ratio

统一记 donor mass 为 `lambda`，融合为：

```text
f(lambda) = (1 - lambda) * f_recipient + lambda * f_donor
```

1. PI-H2T center ratio：按论文 Eq. (8)，将 recipient 到其 ground-truth classifier weight 的 cosine distance 用理论范围 `[0, 2]` 线性归一化，再转换成 donor mass。
2. Recipient-only boundary ratio：仅使用 recipient 到最近 reference decision boundary 的距离，并将分布匹配到 PI-H2T ratio，避免因尺度不同造成不公平。
3. Donor-conditioned closed-form ratio：同时使用 recipient 和 donor 相对每个 `y-vs-j` 决策面的有符号距离，取所有危险方向约束的最小闭式上界。

## Harmful fusion 预测

以下为三个 seed 的均值 ± 标准差。AUPRC 越高越好。

| candidate lambda | PI-H2T center | recipient-only | donor-conditioned |
|---:|---:|---:|---:|
| 0.1 | 0.0830 ± 0.0011 | 0.0688 ± 0.0072 | **0.0831 ± 0.0076** |
| 0.2 | 0.1978 ± 0.0050 | 0.1755 ± 0.0041 | **0.2145 ± 0.0085** |
| 0.3 | 0.3602 ± 0.0064 | 0.3371 ± 0.0081 | **0.4154 ± 0.0108** |
| 0.4 | 0.5574 ± 0.0102 | 0.5402 ± 0.0094 | **0.6401 ± 0.0096** |

对应 AUROC：

| candidate lambda | PI-H2T center | recipient-only | donor-conditioned |
|---:|---:|---:|---:|
| 0.1 | 0.6303 | 0.6151 | **0.6483** |
| 0.2 | 0.6209 | 0.6135 | **0.6604** |
| 0.3 | 0.6138 | 0.6110 | **0.6815** |
| 0.4 | 0.6064 | 0.6047 | **0.6983** |

seed 42、`lambda=0.4` 的 500-bootstrap AUPRC 95% 区间为：

- PI-H2T center：0.5504–0.5737
- recipient-only：0.5382–0.5599
- donor-conditioned：0.6362–0.6583

donor-conditioned 与两条基线的区间不重叠。

## 匹配 donor mass 的控制能力

每一行都让三种方法具有相同的平均 donor mass。预算比例是相对于 donor-conditioned 原始闭式上界均值的 25%、50%、75%、100%。

### Harmful flip rate（越低越好）

| 预算 | PI-H2T center | recipient-only | donor-conditioned |
|---:|---:|---:|---:|
| 25% | 0.0565 | 0.0565 | **0.0509** |
| 50% | 0.1603 | 0.1598 | **0.1426** |
| 75% | 0.3405 | 0.3413 | **0.3244** |
| 100% | **0.5873** | 0.5882 | 0.6043 |

### Probe accuracy（越高越好）

| 预算 | PI-H2T center | recipient-only | donor-conditioned |
|---:|---:|---:|---:|
| 25% | 0.6765 | 0.6760 | **0.6805** |
| 50% | 0.6182 | 0.6186 | **0.6315** |
| 75% | 0.4949 | 0.4944 | **0.5073** |
| 100% | **0.3153** | 0.3152 | 0.3026 |

归一化 harmful-flip 控制曲线面积（越低越好）：

- PI-H2T center：0.20829 ± 0.00445
- recipient-only：0.20870 ± 0.00471
- donor-conditioned：**0.19991 ± 0.00388**

## 为什么不能直接使用 lambda_critical

闭式解给出的是满足约束等号的最大比例。直接取 `lambda=lambda_critical` 时，大量样本恰好满足最近 margin 为 0，而不是处在决策区域内部。

seed 42 的诊断：

- `lambda_critical` 中位数：0.4475；90% 分位数：0.6078。
- 65.36% 的 pair 具有 `lambda_critical > 0.4`。
- reference accuracy 在融合前为 99.88%。
- `gamma <= 0.85` 时 reference accuracy 约为 99.90%。
- `gamma = 1` 时 reference accuracy 降到 45.63%，原因是大量样本落在边界 tie 上。

三 seed 的控制 frontier 显示：

- `gamma = 0.85` 是当前 0.05 网格中，对每个 seed 都同时满足 harmful flip 不增、probe accuracy 不降的最大值。
- `gamma = 0.9` 已在 seed 42 和 44 失效。
- 推荐 `gamma = 0.8`，其平均 absolute flip gain 为 0.0114，accuracy gain 为 0.0085，probe-loss-delta gain 为 0.0798。

最终建议：

```text
lambda_safe = 0.8 * min(lambda_critical, lambda_cap)
```

`0.8` 是当前最小验证得到的安全工作点，不应直接视为跨数据集通用常数；正式训练中应在独立 validation split 上校准，并报告 `0.75/0.80/0.85` 敏感性。

## 可复现文件

- `fusion_risk_minival.py`：三种风险预测与匹配预算控制评估。
- `export_minival_pairs_from_archive.py`：从既有固定特征生成 recipient/donor pair。
- `summarize_fusion_risk_minival.py`：三 seed 聚合与安全系数 frontier。
- `outputs/fusion_risk_minival/cifar100_if100_aggregate/`：聚合 CSV 与 JSON。

