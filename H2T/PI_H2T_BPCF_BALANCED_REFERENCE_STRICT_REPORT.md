# PI-H2T BPCF：严格 balanced-reference 修复实验

## 结论

原 Stage-I reference 对 Few 的边界偏置确实是问题的一部分。只把 boundary geometry head 换成同一 PIF 特征上由 class-balanced recipient 训练得到并冻结的 T2 head，同时严格保持原 PI-H2T base ratio、trainable classifier 初始化、backbone、donor、sampling 和 loss 不变后：

- D 的 Few 从 22.24% 提升到 27.90%；
- D 的三组 H-mean 从 37.19% 提升到 41.05%；
- D 相对 no-fusion T2 的 Medium/Few 分别提高 3.69/9.00 点；
- D 仍比 exact-multiset shuffled E 高 1.01 点 Overall。

但残留问题没有完全解决：D 的 Few 仍比 E 低 1.69 点，H-mean 低 0.35 点。因此 balanced reference 修复了长尾平衡，却还不能证明正确 boundary correspondence 对 Few 有益。

## 1. 严格控制

正式实验固定：

- PI ratio head：原 PIF Stage-I classifier weight；
- boundary geometry head：T2 最后一个 epoch 的 frozen class-balanced classifier；
- trainable Stage-II classifier：仍从原 Stage-I classifier 初始化；
- `lambda_0 = 0.9267667644 * lambda_PI`；
- D：统一最坏-margin projection；
- E：在 `Many/Medium/Few × lambda_0-bin` 内直接打乱 D 的最终 ratio，不重新投影；
- fused-only CE，无 donor selection、risk loss、额外监督或 slack。

T2 reference 只使用训练集训练，不使用测试标签。放回原 Stage-I PIF backbone 直接评估时：

| Reference | Overall | Many | Medium | Few |
|---|---:|---:|---:|---:|
| 原 Stage-I head | 40.27 | 66.53 | 40.06 | 7.93 |
| balanced T2 head | 44.91 | 63.36 | 47.91 | 18.38 |

说明 balanced head 的确改善了 Medium/Few reference geometry，而不是只提升头类。

实现核查时曾发现一次污染：初版接入 external reference 时误把 PI cosine weight 也换成 balanced head，使 base donor mass 从 0.7133 变为约 0.7221。该组输出 `outputs/bpcf_balref_unified_margin_pif_seed42/` 被标记为预实验，不进入正式结论。随后已将 `pi_weight` 与 `reference_weight` 完全拆开，并用新目录严格重跑。

严格重跑中，D/E/T4 的 base donor mass 均为 0.713301，数据流 checksum 与既有 T4 逐值一致；D/E 的最终 ratio 多重集合最大误差为 0。

## 2. 正式 seed42 结果

主结果使用第 10 轮。

| 方法 | Overall | Many | Medium | Few | H-mean | donor mass |
|---|---:|---:|---:|---:|---:|---:|
| T2 no-fusion | 45.69 | 65.22 | 47.80 | 18.90 | 33.64 | 0 |
| T4 matched PI ratio | 37.26 | 30.86 | 46.71 | 33.79 | 35.97 | 0.7133 |
| 原 Stage-I-ref D | 46.67 | 62.17 | 50.97 | 22.24 | 37.19 | 0.3563 |
| 原 Stage-I-ref E | 46.23 | 57.36 | 52.34 | 25.03 | 39.22 | 0.3563 |
| balanced-ref D | **46.32** | **56.14** | **51.49** | 27.90 | 41.05 | 0.4587 |
| balanced-ref E | 45.31 | 52.08 | 51.37 | **29.59** | **41.40** | 0.4587 |

balanced-ref D 与 E 的直接差值：

| D-E | Overall | Many | Medium | Few | H-mean |
|---|---:|---:|---:|---:|---:|
| 差值 | +1.01 | +4.06 | +0.11 | -1.69 | -0.35 |

balanced-ref D 相对原 Stage-I-ref D：

| 新D-原D | Overall | Many | Medium | Few | H-mean |
|---|---:|---:|---:|---:|---:|
| 差值 | -0.35 | -6.03 | +0.51 | +5.66 | +3.86 |

这说明更平衡的 reference 成功把模型从 Many 向 Medium/Few 重新平衡，但正确 pair correspondence 的剩余收益仍主要来自 Many。

## 3. Ratio 与约束审计

| Audit | balanced-ref D | balanced-ref E |
|---|---:|---:|
| base donor mass | 0.713301 | 0.713301 |
| used donor mass | 0.458718 | 0.458718 |
| Many donor mass | 0.474500 | 0.474500 |
| Medium donor mass | 0.459505 | 0.459505 |
| Few donor mass | 0.438314 | 0.438314 |
| lambda=0 fraction | 1.50% | 1.50% |
| ratio multiset max error | 0 | 0 |
| reference violation | 0 | 47.54% |

balanced reference 将 Few donor mass 从原 reference D 的 0.2173 提高到 0.4383，解释了 Few 与 H-mean 的明显恢复。

## 4. 进一步可靠性审计

一个自然的参数为零修复是：仅当 frozen reference 正确识别当前 recipient 时才允许其裁剪 ratio，否则回退 `lambda_0`。但真实 Stage-II 训练流审计显示 balanced reference 的训练样本正确覆盖率为：

| Overall | Many | Medium | Few |
|---:|---:|---:|---:|
| 95.47% | 92.58% | 95.45% | 99.03% |

Few 训练覆盖率反而接近 100%，与测试 Few=18.38% 严重不一致，表明训练内 correctness 被小样本过拟合污染。直接使用它做 reliability gate 几乎不会关闭错误的 tail geometry，因此该方案在完整训练前被否决，没有继续增加一次事后调参实验。

## 5. 当前结论与下一约束

balanced reference 是有效修复，但只解决了一部分问题：

1. 它显著改善了 Few 和 H-mean，证明原 Stage-I reference 偏置确实存在。
2. D>E 的 Overall 信号从 +0.44 扩大到 +1.01，Medium 也由负差变为轻微正差。
3. 但 Few 与 H-mean 仍是 D<E，不能声称 donor-conditioned boundary correspondence 已服务于尾类。

剩余问题不能再用训练内 correctness、任意 group coefficient 或测试集阈值解决。若继续，应使用训练集内部的 out-of-fold/held-out reliability 来决定 reference geometry 的可信度，然后在完全冻结该 reliability 后重跑 D/E。否则会把 Few 的过拟合 reference 当成高置信边界。

## 6. 产物

- 正式输出：`outputs/bpcf_balref_strict_pif_seed42/`
- 严格 smoke：`outputs/bpcf_balref_strict_smoke_seed42/`
- 训练内 reliability audit：`outputs/bpcf_balref_trust_audit_seed42/`
- 受污染、仅保留审计的预实验：`outputs/bpcf_balref_unified_margin_pif_seed42/`
- 实现：`train_pi_h2t_ratio_only_stage2.py`

