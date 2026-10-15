# PI-H2T ratio-only A/B/C 验证协议

## 研究问题

只验证一个因果问题：PI-H2T 的 H2TF 融合比例只看 recipient，忽略实际 donor，是否会损害最终分类性能？

本协议不包含 donor selection、risk-weighted loss、clean anchor、balanced cRT reference、固定 `lambda=0.25`，也不引入新的采样或训练损失。

## 固定不变的部分

- 同一个 Stage-I checkpoint；当前第一步使用现有 CE checkpoint，PIF 为关闭状态。
- 同一个 Stage-I classifier head 既作为 PI-H2T ratio 的 `w_y`，也作为 pair boundary geometry 的冻结几何 head。
- 同一个官方 H2T 双分支采样：class-balanced `f_B` 是带标签 recipient，instance-wise `f_I` 是 donor。
- 同一套随机增强、recipient/donor 数据流、Stage-II classifier 初始化、BN-shift、CE loss、优化器和训练轮数。
- H2TF 一律按论文 Eq. (7) 在 pooled feature 上计算：

```text
f_tilde = r f_B + (1-r) f_I = (1-lambda) f_B + lambda f_I
```

## 三组实验

| 组别 | ratio | donor |
|---|---|---|
| A `pi_recipient_only` | 原始 PI-H2T recipient-only ratio | 与 B/C 完全相同 |
| B `pair_stratified_rankmatch` | pair-conditioned ratio | 与 A/C 完全相同 |
| C `pair_stratified_rankmatch_shuffled` | B 的 ratio 打乱对应 | 与 A/B 完全相同 |

原 PI-H2T 使用 recipient 权重：

```text
r_PI = (1 - cos(f_B, w_y)) / 2
lambda_PI = 1 - r_PI
```

这里 `lambda` 是 donor mass。它不是固定超参数，而是原 PI-H2T 在当前 batch 为每个样本生成的真实 donor-mass 集合。

## B：分组、分层的 RankMatch

先用冻结 Stage-I classifier head 计算每个实际 pair 的 closed-form `lambda_critical(f_B, f_I)`。然后按：

```text
frequency group (Many / Medium / Few)
x native PI recipient-ratio bin
```

分层。在每一个层内，保留 A 的 `lambda_PI` 数值多重集合；将其从小到大分配给按 `lambda_critical` 从小到大排序的真实 pair：

```text
lambda_critical up  ->  lambda_pair up
r_pair = 1 - lambda_pair
```

因此 safer donor 能获得更多 donor mass，同时 B 在每个层内与 A 完全具有相同的 ratio 分布。这样不会让 Many/Few 的不同 `lambda_critical` 分布混入全局重新分配的混杂因素。

## C：打乱 correspondence 的阴性对照

C 保留 B 在每一个相同分层内的 ratio 数值多重集合、相同 recipient 与相同 donor；只使用独立局部随机数发生器在层内打乱 ratio 分配。它不改变 A/B/C 的数据增强和采样随机流。

## 预注册判断

- `B > A`：同一融合强度分布下，pair-conditioned ratio allocation 优于 PI-H2T 的 recipient-only allocation。
- `B > C`：增益来自正确的 recipient-donor correspondence，而不是换了一种 ratio 分布。
- 主指标为第 10 个 epoch 的 Last Overall；Best 只作为测试集选择偏差下的诊断。

训练器会记录以下审计量：每个频次组的 donor mass、每 batch/分层的 ratio 多重集合最大误差、B 的单调分配违例数、C 的实际打乱比例，以及 recipient/donor/label 数据流校验和。

## 两步决策

1. 先用现有 CE Stage-I checkpoint 执行 seed-42、43、44 的 A/B/C go/no-go。它们只改变 Stage-II 随机性，是 H2TF ratio mechanism isolation，而不是完整 PIF 复现。
2. 不对 seed-42 的结果追加任何 ratio、reference 或 loss 调参。仅当三个 Stage-II seeds 都维持 `B>A` 与 `B>C` 的方向时，才训练/获得 PIF Stage-I checkpoint，并原样重复同一协议；届时才能就完整 PI-H2T 提出性能结论。

## 状态更新：PIF 完整复验已完成

真实 PIF Stage-I checkpoint 已训练并完成同一 A/B/C 的 Stage-II seeds 42、43、44。PIF setting 中 `B>A` 为 3/3，但 `B>C` 在 seed43 为负，三 seed 平均只有 `+0.03±0.18` 个百分点，故严格 gate 不通过。完整实现、审计和结果见 `PI_H2T_RATIO_ONLY_PIF_FULL_REPORT.md`；不应基于 CE setting 的结果直接声称已经改进完整 PI-H2T。
