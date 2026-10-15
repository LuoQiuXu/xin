# PI-H2T ratio-only A/B/C：seed-42 机制验证

## 这一轮只回答什么

只回答 PI-H2T 的 H2TF 融合比例忽略实际 donor，是否会造成可观测的最终性能损失。

没有加入 donor selection、risk-weighted loss、clean anchor、balanced cRT reference、固定 `lambda=0.25`、额外候选 donor，或新的训练目标。

## 与旧实验相比的关键修正

1. 几何只使用**冻结的原 Stage-I classifier head**，不使用 cRT 或任何 balanced reference。
2. A 使用论文原始 PI-H2T ratio，而不是固定融合强度：

   ```text
   r_PI = (1 - cos(f_B, w_y)) / 2
   lambda_PI = 1 - r_PI
   f_tilde = r f_B + (1-r) f_I
   ```

3. B 不做全 batch 的 global RankMatch。先按 `Many/Medium/Few × 原 PI recipient-ratio 的 4 个固定区间` 分层；在每个层中保留 A 的 `lambda_PI` 数值多重集合，再按实际 pair 的 `lambda_critical(f_B,f_I)` 升序分配 donor mass：

   ```text
   lambda_critical up  ->  lambda_pair up
   r_pair = 1 - lambda_pair
   ```

4. C 保留 B 在每一个相同层中的完整 ratio 集合、同一 recipient、同一 donor，只在层内使用局部随机数打乱 ratio--pair 对应。

这样 A/B/C 的平均融合强度、每个频次组的融合强度、原 PI-ratio 分层分布、recipient/donor 样本和训练流程均一致；唯一干预是 ratio 是否分配给正确的 recipient-donor pair。

## 设置

- 数据集：CIFAR100-LT IF=100。
- Stage-I：现有 CE ResNet-32 checkpoint，seed 42，epoch 184；该 checkpoint 没有 PIF 参数。
- Stage-II：官方 H2T 双分支采样，class-balanced recipient + instance-wise donor；冻结 backbone、BN shift、仅训练原 Stage-I 初始化的 classifier、CE、10 epochs。
- 几何：冻结 Stage-I classifier 的权重与 bias；训练中的 classifier 更新不会改变 ratio 几何。
- 原始 PI-H2T 平均 recipient mass：0.3456；相应平均 donor mass：0.6544。三组均严格使用这套原生 ratio 分布，不与 `lambda=0.25` 比较。

## 结果：预注册 gate 通过

主指标为第 10 个 epoch（Last）：

| 组别 | Overall | Many | Medium | Few |
|---|---:|---:|---:|---:|
| A 原 PI-H2T recipient-only ratio | 40.36 | 39.08 | 50.69 | 29.48 |
| B pair-conditioned ratio | **40.96** | **40.42** | 50.63 | 29.97 |
| C 打乱 pair correspondence | 40.55 | 39.31 | 50.43 | **30.17** |

```text
B - A = +0.60 个百分点
B - C = +0.41 个百分点
```

B 在 10 个 epoch 的每一轮都领先：

| Epoch | A | B | C | B-A | B-C |
|---:|---:|---:|---:|---:|---:|
| 1 | 43.30 | 43.47 | 43.44 | +0.17 | +0.03 |
| 2 | 43.24 | 43.46 | 43.29 | +0.22 | +0.17 |
| 3 | 42.11 | 42.61 | 42.42 | +0.50 | +0.19 |
| 4 | 41.53 | 42.12 | 41.49 | +0.59 | +0.63 |
| 5 | 40.90 | 41.74 | 41.24 | +0.84 | +0.50 |
| 6 | 40.79 | 41.33 | 40.97 | +0.54 | +0.36 |
| 7 | 40.32 | 40.89 | 40.33 | +0.57 | +0.56 |
| 8 | 40.23 | 40.76 | 40.37 | +0.53 | +0.39 |
| 9 | 40.29 | 40.98 | 40.52 | +0.69 | +0.46 |
| 10 | 40.36 | 40.96 | 40.55 | +0.60 | +0.41 |

## 审计：为什么这可以归因于 pair correspondence

第 10 个 epoch：

| 审计项 | A | B | C |
|---|---:|---:|---:|
| ratio 多重集合最大误差 | 0 | 0 | 0 |
| `lambda_critical ↑ -> lambda_pair ↑` 违例数 | 5135* | **0** | 5079* |
| C 层内实际打乱比例 | - | - | 95.20% |
| recipient 流校验和 | -10473771.4922 | -10473771.4922 | -10473771.4922 |
| donor 流校验和 | -9299823.6602 | -9299823.6602 | -9299823.6602 |
| target / donor-label 校验和 | 537099 / 215936 | 537099 / 215936 | 537099 / 215936 |

`*` A 本来不按 `lambda_critical` 分配，C 则故意破坏对应，故非零是预期现象。

因此：

- B 与 A 的差异不可能来自更大的总体 donor mass、Many/Few 间 donor mass 再分配、不同 donor，或不同采样/增强流；
- B 与 C 的差异不可能来自 ratio 分布本身；
- 现有证据支持：**正确的 recipient-donor ratio allocation 本身带来了收益。**

## 结论的边界

这个 seed-42 结果通过了当前的 go/no-go：在相同原 PI-H2T 融合强度分布下，pair-conditioned ratio 同时优于原 recipient-only ratio 和打乱 correspondence 对照。

它已经说明 donor blindness 不只是一个公式层面的缺口，而会影响 H2TF 的训练性能。

## 不能过度解释的地方

### 这不是完整 PI-H2T

当前 Stage-I checkpoint 没有 PIF。因此准确表述只能是：

> 在一个不含 PIF 的 H2TF ratio-isolation setting 中，pair-conditioned ratio allocation 优于 recipient-only allocation。

不能表述为“已经改进完整 PI-H2T”。

### B 的收益可能是减少额外伤害，而非创造净正向 augmentation

三组都随 10 轮 Stage-II 训练下降：

| 组别 | Epoch 1 | Epoch 10 |
|---|---:|---:|
| A | 43.30 | 40.36 |
| B | 43.47 | 40.96 |
| C | 43.44 | 40.55 |

原生 PI-H2T ratio 的平均 recipient mass 为 0.3456，即 donor mass 为 0.6544。这个较强的融合分布在当前 CE、无 PIF representation 下可能本身就偏激进。因而现有证据更稳妥的解释是：错误 correspondence 造成额外损伤，而正确 correspondence 减少了该损伤；尚未证明 pair-conditioned allocation 相对正常 PI-H2T 会产生净正向 augmentation 收益。

### 当前收益不应称为 Tail 专属收益

B 相对 A 的组别变化为：Many `+1.34`、Medium `-0.06`、Few `+0.49`；Overall 的主要贡献来自 Many。B 相对 C 则为 Many `+1.11`、Medium `+0.20`、Few `-0.20`。因此可以说 pair-conditioned allocation 提升了当前设置的 Overall，但不能说“考虑 donor 后明显改善 Tail”。

## 已预注册的下一步

现在不改变 `lambda_critical`、reference、sampling、loss、ratio-bin 数、RankMatch 或任何超参数。

1. 复用**同一个** CE Stage-I checkpoint，只改变 Stage-II 的独立随机 seed 为 43 和 44，原样运行 A/B/C。
2. 主指标仍为 epoch 10 Last；分别报告每个 seed 的 `B-A` 与 `B-C`。这检验的是 Stage-II 随机性，而不是独立的 Stage-I 表征随机性。
3. 只有当三个 Stage-II seeds 都维持 `B>A` 且 `B>C` 的方向时，才训练/取得真实 PIF Stage-I checkpoint，并**不改一行 A/B/C ratio-only 协议**地重复。

PIF setting 若再次通过，才能把结论上升为：PI-H2T 的 recipient-only fusion ratio 是会造成可测量性能损失的设计局限，pair-conditioned ratio 能够修正它。

## 对应文件

- `train_pi_h2t_ratio_only_stage2.py`：专用 A/B/C 训练器。
- `PI_H2T_RATIO_ONLY_PROTOCOL.md`：预注册实验协议。
- `outputs/pi_h2t_ratio_only_ce_seed42/summary.csv`：汇总指标和全部审计量。

## 状态更新：CE 三随机种子复验已完成

上述预注册的 seed43、seed44 已在**完全不改变该协议**的情况下完成。第 10 epoch Last Overall 的三个 seed 均满足 `B>A` 和 `B>C`：

| seed | B-A | B-C |
|---:|---:|---:|
| 42 | +0.60 | +0.41 |
| 43 | +0.91 | +0.67 |
| 44 | +0.79 | +0.46 |

汇总和审计见 `PI_H2T_RATIO_ONLY_CE_MULTISEED_REPORT.md`。因此 CE setting 的 gate 已通过；后续仅需在获得真实 PIF Stage-I checkpoint 后，原样复验 A/B/C，而不应再在 CE setting 中追加调参或新机制。
