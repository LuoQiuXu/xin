# PI-H2T：纯 Center-r vs Boundary-r（PIF，seed 42）

## 为什么重做

前一版在 `Many/Medium/Few × lambda_A-bin` 内做 boundary rank matching。由于 `lambda_A-bin` 本身来自 PI-H2T center score，那一版只能检验 **center 局部条件下的 boundary 增量信息**，不能作为纯 boundary-r 的结论。

本报告对应的新实验完全移除了 `lambda_A-bin`。旧输出 `outputs/center_boundary_r_pif_seed42/` 保留为历史的 `center-bin-conditioned` 对照，不用于本报告结论。

## 纯 A/B/C 定义

唯一分层：

```text
g ∈ {Many, Medium, Few}
```

### A：Center-r

保持原 PI-H2T center score 以及已经冻结的 PIF 全局尺度：

```text
lambda_A,i = kappa_match * lambda_PI,i
kappa_match = 0.9267667644
```

### B：Boundary-r

以原 Stage-I CE classifier 的 signed normalized closest-boundary margin：

```text
m_i = min_{j != y_i}
      ((w_y-w_j)^T f_i + (b_y-b_j)) / (||w_y-w_j||_2 + eps)
```

在每个 frequency group 内，将 A 的 ratio 仅作为 **无身份数值袋**：

```text
L_g = sort({lambda_A,i : i ∈ g})
```

按 `m_i` 升序排列 recipient，按 `L_g` 升序分配：

```text
m_i ↑  ->  lambda_B,i ↑
r_B,i = 1 - lambda_B,i
```

因此 B 不查看 recipient 自身的 `lambda_A,i`、center score 或 `lambda_A-bin`；center 信息只保留为每个频率组的全局 ratio 分布控制。

### C：Random-r

在同一个 Many/Medium/Few 组内，对 B 的 ratio 做 random derangement：

```text
lambda_C = Shuffle_g(lambda_B)
```

actual recipient、donor、fusion pair、sampling、augmentation、backbone、classifier、loss、optimizer、BN shift 和训练 epoch 均不变。

## 固定 endpoint 结果

| 方法 | Overall | Many | Medium | Few | H-mean |
|---|---:|---:|---:|---:|---:|
| A：Center-r | 37.20 | 30.44 | 47.03 | 33.72 | 32.00 |
| B：纯 Boundary-r | 37.06 | 30.06 | 46.97 | 33.79 | 31.81 |
| C：Random-r | **37.85** | **31.25** | **47.91** | **33.90** | **32.52** |

关键差值（百分点）：

- `B - A`：Overall -0.14，Many -0.39，Medium -0.06，Few +0.07，H-mean -0.19；
- `B - C`：Overall -0.79，Many -1.19，Medium -0.94，Few -0.10，H-mean -0.70；
- `C - A`：Overall +0.65，Many +0.81，Medium +0.89，Few +0.17，H-mean +0.52。

## 纯度与因果审计

末轮 A/B/C 的 group-level donor-mass multiset 完全相同：

| 指标 | A | B | C |
|---|---:|---:|---:|
| Overall donor mass mean | 0.714177 | 0.714178 | 0.714177 |
| Many donor mass mean | 0.726151 | 0.726151 | 0.726151 |
| Medium donor mass mean | 0.715895 | 0.715895 | 0.715895 |
| Few donor mass mean | 0.697219 | 0.697219 | 0.697219 |
| ratio multiset 最大误差 | 0 | 0 | 0 |

recipient-level Spearman correlation（各频率组 macro 平均）：

| 关系 | A | B | C |
|---|---:|---:|---:|
| ρ(m, lambda) | 0.724 | **1.000** | 0.004 |
| ρ(lambda_A, lambda) | 1.000 | 0.724 | 0.008 |
| mean |lambda-lambda_A| | 0 | 0.02223 | 0.04595 |
| ratio changed fraction | 0% | 99.917% | 99.991% |

此外：

- B 的 margin-to-ratio 单调性违例为 0；
- C 每个 epoch 的 shuffle eligible fraction 和 moved fraction 都为 100%；
- A/B/C 每个 epoch 的 recipient、donor、target、donor-label stream checksum 完全一致。

这证明 B 已经真正替换了 recipient-level center correspondence，而 C 同时去除了 center 与 boundary 的样本级对应；C 的优势不能由 ratio 分布、频率组 donor mass、donor pair 或数据流差异解释。

## 结论

纯 boundary-r 的预注册条件仍未通过：

```text
B > A：不成立；
B > C：不成立。
```

因此现有证据不支持：

> 将更大的 signed boundary safety margin 映射为更大的 donor mass，能够取代 PI-H2T 的 center-r。

这次结论比上一轮更强，因为 center-bin 混杂已被去除。它不等于“所有 boundary 信息都无价值”，而是说明目前这一个 **单调安全 margin → donor mass** 的 recipient-r 映射不对或不充分。尤其 C 同时提升 Many、Medium、Few，提示 ratio 的有效性不只是“越安全越多融合”。

按主线，暂不进入 donor-conditioned ratio。若继续，应先提出一个新的、预注册的 recipient-only r 假设，例如 boundary hardness / near-boundary band 的非单调映射，再用完全相同的 groupwise A/B/C 控制验证。

## 产物

- 实验配置：[experiment.json](outputs/center_boundary_r_purefreq_pif_seed42/experiment.json)
- 汇总结果：[summary.csv](outputs/center_boundary_r_purefreq_pif_seed42/summary.csv)
- B 的 class audit：[last_epoch_class_audit.csv](outputs/center_boundary_r_purefreq_pif_seed42/boundary_r/last_epoch_class_audit.csv)
- 实验实现：[train_center_boundary_r_stage2.py](train_center_boundary_r_stage2.py)
