# PI-H2T：PIF、融合强度与 Stage-II 性能下降的因果定位

## 结论摘要

本轮实验回答了三个问题：

1. PIF Stage-II 从约 40% 降到约 35%，不是 classifier balanced retraining 或 BN shift 导致的；主要下降来自 H2TF 融合。
2. 在严格同架构、同初始化、同训练配方控制下，PIF 确实提高了 feature-class-weight cosine，并系统性增大 PI-H2T 映射出的 donor mass。
3. 把 PIF-on donor mass 均值缩放回 matched PIF-off 水平，可以恢复部分性能，但不能恢复到 no-fusion 水平。因此 donor-mass 增大是部分原因，不是完整解释；当前不满足继续做 A/B/C pair-correspondence 实验的预注册门槛。

## 1. 实验控制

### 1.1 Endpoint 定位

所有 endpoint 均使用同一个 PIF Stage-I checkpoint、seed 42、相同 recipient/donor/label 流、相同 classifier 初始化和 10-epoch cosine 学习率流程。

- `T0`：PIF Stage-I checkpoint 直接测试。
- `T1`：donor mass 为 0，训练 classifier，BN 固定。
- `T2`：donor mass 为 0，训练 classifier，BN 更新。
- `T3`：原生 PI-H2T donor mass，BN 更新。
- `T4`：预注册的 matched-scale donor mass，BN 更新。

`T1/T2` 虽然 donor mass 为 0，仍保留 recipient 和 donor 两次 backbone forward；因此 `T2/T3/T4` 的 BN 输入方式相同。第 10 轮的四个数据流 checksum 在 `T1/T2/T3` 间逐值一致：

| Audit | T1 | T2 | T3 |
|---|---:|---:|---:|
| recipient checksum | -10497553.703125 | -10497553.703125 | -10497553.703125 |
| donor checksum | -9339508.480469 | -9339508.480469 | -9339508.480469 |
| target checksum | 537099 | 537099 | 537099 |
| donor-label checksum | 215936 | 215936 | 215936 |

### 1.2 严格 matched PIF-off

PIF ON/OFF 都实例化 Option-A、PIF-capable ResNet-32。两组都创建 `conv1d` PIF 参数，以保证 RNG 消耗顺序一致；唯一 intervention 是 forward 中的 `pif_active=True/False`。

初始化审计结果：

- backbone state 逐张量完全一致；
- classifier state 逐张量完全一致；
- 两份 Stage-I YAML 除实验名和 `pif_active` 外完全一致；
- 数据、seed 123、CE+MixUp、200 epochs、optimizer、LR schedule、backbone 和 classifier 均相同。

PIF-off 最佳 Stage-I Overall 为 39.40%，PIF-on 为 40.27%，即 PIF-on 高 0.87 点。PIF 并没有让 Stage-I 表征性能变差。

## 2. Endpoint 结果

最后一轮是预注册主结果，best epoch 仅作辅助观察。

| Endpoint | BN | donor mass | Last Overall | Many | Medium | Few | Best Overall |
|---|---|---:|---:|---:|---:|---:|---:|
| T0：Stage-I direct | checkpoint | — | 40.27 | 66.53 | 40.06 | 7.93 | — |
| T1：no fusion | fixed | 0 | 44.80 | 65.00 | 47.00 | 17.07 | 44.85 |
| T2：no fusion | update | 0 | 45.69 | 65.22 | 47.80 | 18.90 | 45.85 |
| T3：Native PI-H2T | update | 0.7697 | 34.82 | 26.33 | 44.40 | 33.79 | 40.99 |
| T4：matched scale | update | 0.7133 | 37.26 | 30.86 | 46.71 | 33.79 | 42.18 |

关键差值：

- `T1 - T0 = +4.53`：balanced classifier retraining 本身有益。
- `T2 - T1 = +0.89`：当前 BN shift 没有造成性能下降。
- `T3 - T2 = -10.87`：原生 H2TF 融合是主要下降来源。
- `T4 - T3 = +2.44`：matched scaling 能恢复部分性能。
- `T4 - T2 = -8.43`：仅修正全局 donor-mass 均值仍远未消除融合损失。

以 `T2-T3` 的 10.87 点为待恢复缺口，matched scaling 恢复约 22.4%。它主要恢复 Many/Medium，Few 基本不变。原生融合确实提高 Few，但以 Many 的巨大下降为代价。

## 3. Matched PIF OFF/ON cosine 与 ratio audit

PI-H2T 使用：

\[
\lambda_{\mathrm{PI}}=\frac{1+\cos(f,w_y)}{2}.
\]

两个 checkpoint 接收逐张量相同的图片。模型保持 eval，audit 不更新 BN。

### 3.1 Deterministic test set

| Arm | Accuracy | cosine mean | lambda mean | P10 | P25 | P50 | P75 | P90 | P(lambda > 0.75) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matched PIF-off | 39.40 | 0.2719 | 0.6360 | 0.5210 | 0.5737 | 0.6388 | 0.7020 | 0.7474 | 9.29% |
| PIF-on | 40.27 | 0.3551 | 0.6775 | 0.5389 | 0.6064 | 0.6864 | 0.7569 | 0.8039 | 27.21% |

### 3.2 Operational Stage-II balanced recipient stream

| Arm | lambda mean | Std | P10 | P25 | P50 | P75 | P90 | P(lambda > 0.75) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matched PIF-off | 0.7128 | 0.0482 | 0.6470 | 0.6820 | 0.7159 | 0.7466 | 0.7729 | 22.73% |
| PIF-on | 0.7691 | 0.0465 | 0.7082 | 0.7399 | 0.7719 | 0.8016 | 0.8273 | 68.01% |

按频率组的平均 donor mass：

| Group | PIF-off | PIF-on | ON - OFF |
|---|---:|---:|---:|
| Many | 0.7328 | 0.7775 | +0.0448 |
| Medium | 0.7172 | 0.7724 | +0.0552 |
| Few | 0.6832 | 0.7550 | +0.0718 |

PIF 对 Few recipient 的 donor-mass 增幅最大。Stage-II balanced stream 中，PIF-on 正确样本的平均 donor mass 为 0.7778，错误样本为 0.7102；即 PI 映射会让与本类权重更对齐的 recipient 接收更大的 donor mass。

因此可以在 matched control 下确认：

> PIF 提高 feature-class-weight cosine，并通过 PI-H2T 的单调映射显著增大 donor mass；这种变化不是 Option-A/Option-B 架构差异造成的。

## 4. 预注册 matched-scale 控制

缩放系数只由 operational Stage-II balanced recipient stream 的两个 checkpoint 决定：

\[
\kappa_{\mathrm{match}}
=\frac{\mathbb E[\lambda_{\mathrm{PIF-off}}]}
{\mathbb E[\lambda_{\mathrm{PIF-on}}]}
=\frac{0.7128000}{0.7691256}
=0.9267667644.
\]

训练中使用：

\[
\lambda'=\kappa_{\mathrm{match}}\lambda_{\mathrm{PI}}.
\]

没有 sweep，也没有根据准确率调整该值。第 10 轮实际 donor mass 为 0.7133，与目标 0.7128 一致。缩放后的 PIF-on P10/P25/P50/P75/P90 与 matched PIF-off 也非常接近，因此它不仅大致匹配均值，也大幅缩小了分布尺度差异。

然而 Last Overall 只从 34.82% 恢复到 37.26%，没有接近 no-fusion 的 45.69%。所以不能把全部下降归因于一个全局 donor-mass scale mismatch。

## 5. 当前可支持与不可支持的结论

### 可以支持

1. PIF Stage-II 的主要性能下降来自 H2TF fusion，而非 classifier retraining 或 BN shift。
2. PIF 因果性地改变 cosine/ratio 几何，并显著提高 PI-H2T donor mass，尤其影响 Few recipient。
3. 原生 donor mass 过大确实是问题的一部分：预注册缩放恢复 2.44 点。
4. 仅做全局缩放不足以修复性能；ratio 对具体 recipient/donor 的分配或 PIF 特征上的线性融合兼容性仍可能有问题。

### 目前不能支持

1. 不能声称 matched scaling 已解决 PI-H2T 的 PIF 问题。
2. 不能声称 donor blindness 已通过最终性能实验得到修复。
3. 不能在当前结果上直接宣称 pair-conditioned B 优于 shuffled C；native PIF 下该控制此前没有稳定复现，而本轮 matched-scale 没有通过恢复门槛。

## 6. Go/No-Go

本轮预注册门槛是：只有 matched-scale 能把融合性能拉回合理区间，才在相同尺度下重新运行 A/B/C 多 seed。

该门槛没有通过：`T4=37.26%`，仍比 `T2=45.69%` 低 8.43 点。因此本轮在这里停止，没有追加 A/B/C，也没有引入 donor selection、risk-weighted loss 或其他调参。

下一步若继续，应先用最小 matched control 区分两种剩余解释：

- matched PIF-off 在其 native ratio 下是否同样发生大幅 Stage-II collapse；
- 若 PIF-off 不 collapse，则剩余问题来自 PIF feature interpolation / recipient-level ratio correspondence，而不只是 donor-mass 分布均值。

## 7. 产物

- Endpoint T1：`outputs/pif_endpoint_seed42_T1_k0_bnfixed/`
- Endpoint T2：`outputs/pif_endpoint_seed42_T2_k0_bnshift/`
- Native T3：`outputs/pi_h2t_ratio_only_pif_stage1seed123_seed42/pi_recipient_only/`
- Matched-scale T4：`outputs/pif_endpoint_seed42_T4_kappa_matched/`
- Matched PIF-off Stage-I：`saved/cifar100_imb001_stage1_ce_mixup_pif_off_matched_202608092042/`
- 完整 audit JSON：`outputs/pif_matched_cosine_ratio_audit_seed42.json`
