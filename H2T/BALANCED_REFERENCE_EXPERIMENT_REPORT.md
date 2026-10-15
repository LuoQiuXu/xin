# Balanced Reference Head 因果实验报告

## 结论

本轮实验成功隔离并检验了以下假设：

> donor-conditioned geometry 没有形成稳定分类增益，是否主要因为原始 CE Reference Head 的尾类边界不可靠？

结果表明：

1. **CE Reference 的边界可靠性问题确实存在，而且可以被 balanced cRT head 明显改善。**
2. **更好的外部 harmful 预测没有转化成最终 Overall 提升。**
3. Balanced Reference 稳定地把一部分性能从 Many 转移给 Medium/Few，但三种子平均 Overall 略降 0.027 个百分点。

因此，“CE reference bias 是唯一瓶颈”这一解释不成立。当前更可能的瓶颈位于 Stage-II 的训练目标、融合样本难度或 recipient-only 标签监督，而不是 reference boundary 本身。

## 1. 因果隔离

### 相同部分

- CIFAR100-LT IF=100；训练集 10,847，测试集 10,000。
- 同一个 CE ResNet-32 Stage-I checkpoint，Stage-I Overall 40.75%。
- 相同 Backbone、Stage-II output head 初始化、optimizer、数据顺序、图像增强、BN shift、融合均值 `mu=0.25`、几何权重 `beta=0.5`。
- output head 始终由原始 CE classifier 初始化；Balanced Reference 从不参与输出预测或梯度更新。
- Stage-II seeds 为 42、43、44；主结果固定使用 epoch 10。

### 唯一变化

- 条件 B：原始 CE head 作为冻结 Reference Head。
- 条件 C：同一 CE Backbone 冻结特征上重新训练的 class-balanced cRT linear head 作为冻结 Reference Head。

逐张量检查表明，B/C 在每个 seed 的最终 Backbone（包括 BN running statistics）198 个状态张量全部 bitwise identical，最大差异为 0。因而两组之间没有 Backbone 或 BN 混杂。

## 2. Balanced cRT Reference

cRT head 丢弃原始 imbalanced CE classifier，只在固定 64 维 CE 特征上使用类别均衡 recipient sampling 训练 10 epochs；不依据测试集选择 checkpoint，固定保存最后 epoch。

| Reference Head | Overall | Many | Medium | Few |
|---|---:|---:|---:|---:|
| 原始 CE | 40.75 | 66.86 | 38.97 | 10.48 |
| Balanced cRT | 42.73 | 62.44 | 42.46 | 18.59 |

它在相同表示空间中显著改善 Medium/Few，说明确实得到了一个不同且更均衡的 reference boundary，而不是更换 Backbone 所产生的效果。

## 3. 完整 Stage-II 结果

数值为三种 Stage-II seeds 的均值和样本标准差，单位为百分点。

### Epoch 10（主结果）

| 条件 | Reference / Fusion | Overall | Many | Medium | Few |
|---|---|---:|---:|---:|---:|
| A | Fixed lambda=0.25 | **46.343 +/- 0.107** | 57.380 +/- 0.208 | **50.543 +/- 0.346** | 27.575 +/- 0.600 |
| B | CE Ref + Donor Boundary | 46.327 +/- 0.095 | **58.537 +/- 0.170** | 50.143 +/- 0.114 | 26.563 +/- 0.369 |
| C | Balanced Ref + Donor Boundary | 46.300 +/- 0.122 | 58.130 +/- 0.267 | 50.219 +/- 0.214 | 26.885 +/- 0.502 |
| D | Official H2T | 46.283 +/- 0.169 | 57.213 +/- 0.267 | 50.457 +/- 0.249 | **27.678 +/- 0.502** |

条件 C 相对 B 的严格配对差值：

```text
Overall: -0.04, -0.07, +0.03；mean -0.027 +/- 0.051
Many:    -0.31, -0.42, -0.50；mean -0.407 +/- 0.098
Medium:  -0.06, +0.09, +0.20；mean +0.076 +/- 0.129
Few:     +0.31, +0.17, +0.48；mean +0.322 +/- 0.155
```

Balanced Reference 在三个 seed 上都降低 Many、提高 Few，但 Overall 没有净提升。这是稳定的 head-to-tail 再分配，而不是最终性能突破。

### Test-selected Best（仅作诊断）

| 条件 | Overall |
|---|---:|
| A：Fixed | 46.383 +/- 0.038 |
| B：CE Reference | **46.437 +/- 0.006** |
| C：Balanced Reference | 46.413 +/- 0.040 |
| D：Official H2T | **46.443 +/- 0.038** |

即使使用不够严格的 test-selected Best，Balanced Reference 也没有超过 CE Reference 或 Official H2T。

## 4. 独立 External Harmful 诊断

为了避免 reference 自我定义 harmful，额外使用一个不属于 CE/cRT reference 的 evaluator：在冻结训练特征上计算类别中心，用余弦最近类中心分类器评价 test recipient 与长尾 train donor 的融合路径。

harmful 标签定义为：external evaluator 原本正确分类 recipient，但在固定 `lambda=0.25` 融合后预测错误。reference 只输出 closed-form critical ratio，不参与 harmful 真值定义。

### harmful 预测能力

| Reference | Overall AUROC | Overall AUPRC | Few AUROC | Few AUPRC |
|---|---:|---:|---:|---:|
| CE | 0.828 +/- 0.007 | 0.392 +/- 0.012 | 0.751 +/- 0.018 | 0.417 +/- 0.030 |
| Balanced cRT | **0.886 +/- 0.005** | **0.505 +/- 0.003** | **0.840 +/- 0.002** | **0.520 +/- 0.024** |

Balanced Reference 对独立 harmful 标签的预测提升非常明确，尤其是 Few 类。这证明它不是简单改变分类先验，而是得到与独立语义 evaluator 更一致的 donor-path 排序。

### 未训练融合特征的 external evaluator 表现

在保持每个频率组平均 donor mass 都为 0.25 的情况下：

| Reference policy | External accuracy | Harmful flip rate |
|---|---:|---:|
| CE boundary | 44.003% +/- 0.150 | 8.227% +/- 0.409 |
| Balanced boundary | **44.307% +/- 0.115** | **7.675% +/- 0.354** |

Balanced policy 生成的原始融合特征语义质量更高，但最终训练分类头仍未获得 Overall 增益。

## 5. 假设判断

### 得到支持

- 原 CE head 的 reference boundary 对 Few 类明显不可靠。
- 同一 Backbone 上的 balanced cRT head 能显著改善独立 harmful AUROC/AUPRC。
- Balanced Reference 能减少外部 evaluator 下的 harmful fusion，并恢复部分 Few 性能。

### 未得到支持

- “只要 reference boundary 更可靠，最终分类准确率就会提高”没有成立。
- 当前 donor-conditioned ratio control 仍未稳定优于 Fixed 或 Official H2T。

### 更合理的新解释

风险预测和最终训练收益并不是同一个目标。H2T 需要具有训练价值的困难增强样本；更安全、更符合 recipient 语义的融合可能同时削弱 head 类的 hard-augmentation 正则化。另一方面，对已经接近或越过语义边界的融合样本继续使用纯 recipient hard label，也可能产生监督噪声。

## 6. 下一步建议

不建议继续训练更多 reference head 或继续扫描 `mu/beta`。下一轮应固定 Balanced Reference，直接检验 Stage-II 如何使用融合样本：

1. 当前 recipient-only hard label（现有条件 C）。
2. clean-recipient anchor：同时优化原 recipient 与 fused recipient。
3. risk-weighted fused loss：对高 external/reference risk 的 fused loss 降权。

这样才能区分两种解释：

- 更安全的融合不够“难”，因此正则化收益下降；
- 融合接近边界时，recipient-only hard label 存在语义噪声。

在该问题解决前，最严谨的结论仍然是：**Balanced Reference 改善了 donor-conditioned risk modeling 和 Few 保护，但没有改善最终 Overall。**

