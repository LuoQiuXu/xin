# PI-H2T 统一最坏边界约束：PIF seed42 决定性验证

## 结论摘要

本轮首次得到明确的最终分类性能正向结果，但尚不足以宣布完整通过。

- T4 全局尺度校准：37.26%。
- D 统一最坏-margin BPCF：46.67%。
- E 保持 D 的 ratio 多重集合、打乱 pair correspondence：46.23%。
- T2 no-fusion：45.69%。

因此，seed42 上同时观察到：

1. D-T4 = +9.41 个百分点，pair-specific boundary clipping 显著修复了过强融合。
2. D-E = +0.44 个百分点，在完全相同 ratio 多重集合下，正确 recipient--donor correspondence 出现正向信号。
3. D-T2 = +0.98 个百分点，修正后的融合首次超过 no-fusion。

但是，D 的 Few accuracy 为 22.24%，虽然高于 T2 的 18.90%，却低于 T4 的 33.79%；并且 D 相比 E 的 0.44 点 Overall 优势主要来自 Many，Few 反而下降 2.79 点。按照预先约定的“不能靠 Few 大幅下降换取总体提升”门槛，本轮暂不扩展 seed43/44。

## 1. 冻结的实验定义

所有实验使用同一个 PIF Stage-I checkpoint、seed42、相同 recipient/donor/label 流、相同 classifier 初始化、10-epoch cosine 学习率、BN 更新及 fused-only CE。

PI-H2T 全局校准基线为：

\[
\lambda_0=\kappa_{\mathrm{match}}\lambda_{\mathrm{PI}},
\qquad
\kappa_{\mathrm{match}}=0.9267667644209227.
\]

对 frozen Stage-I linear head，定义：

\[
a_j=d_{yj}(f_r),
\qquad
s_j=d_{yj}(f_d)-d_{yj}(f_r),
\]

以及 recipient 的统一最坏 reference margin：

\[
m_0=\min_{j\ne y}a_j,
\qquad
\tau=\min(m_0,0).
\]

D 只向下裁剪 \(\lambda_0\)：

\[
\lambda_D=
\max\left[
0,
\min\left(
\lambda_0,
\min_{j:s_j<0}\frac{a_j-\tau}{-s_j}
\right)
\right].
\]

等价约束为：

\[
m(f(\lambda_D))\ge \min(m(f_r),0).
\]

E 保留相同 recipient、donor 和 D 的最终 ratio，仅在 `Many/Medium/Few × lambda_0-bin` 内打乱 ratio-to-pair correspondence；打乱后不重新投影、不重新缩放。

## 2. 严格控制审计

第 10 轮 D/E/T4 的四项数据流 checksum 完全一致：

| Audit | T4 | D | E |
|---|---:|---:|---:|
| recipient | -10497553.703125 | -10497553.703125 | -10497553.703125 |
| donor | -9339508.480469 | -9339508.480469 | -9339508.480469 |
| target | 537099 | 537099 | 537099 |
| donor label | 215936 | 215936 | 215936 |

D/E 的实际 donor mass 均值均为 0.356336，Many/Medium/Few 均值也逐组一致；每个 batch、每个分层的 ratio 多重集合最大误差为 0。

| Audit | D | E |
|---|---:|---:|
| base donor mass | 0.713301 | 0.713301 |
| projected/used donor mass | 0.356336 | 0.356336 |
| Many donor mass | 0.463352 | 0.463352 |
| Medium donor mass | 0.362265 | 0.362265 |
| Few donor mass | 0.217284 | 0.217284 |
| clipping fraction | 97.56% | 97.56% |
| lambda=0 fraction | 10.02% | 10.02% |
| ratio multiset max error | 0 | 0 |
| shuffled moved fraction | 0 | 95.94% |
| reference-boundary violation | 0 | 46.94% |

这些审计说明 D/E 的唯一变化是 ratio 与具体 recipient--donor pair 的对应关系。

## 3. 最终分类结果

主结果使用第 10 轮，不按 test accuracy 选择 epoch。

| 方法 | Overall | Many | Medium | Few | donor mass |
|---|---:|---:|---:|---:|---:|
| T2：no-fusion | 45.69 | 65.22 | 47.80 | 18.90 | 0 |
| T4：matched PI ratio | 37.26 | 30.86 | 46.71 | 33.79 | 0.7133 |
| D：统一最坏-margin BPCF | **46.67** | 62.17 | 50.97 | 22.24 | 0.3563 |
| E：shuffled D ratio | 46.23 | 57.36 | 52.34 | 25.03 | 0.3563 |

关键差值：

| 比较 | Overall | Many | Medium | Few |
|---|---:|---:|---:|---:|
| D-T4 | +9.41 | +31.31 | +4.26 | -11.55 |
| D-E | +0.44 | +4.81 | -1.37 | -2.79 |
| D-T2 | +0.98 | -3.06 | +3.17 | +3.34 |

D 的最佳 epoch 为第 6 轮，Overall=46.74%；预注册主结果仍保留第 10 轮的 46.67%。

## 4. 能支持与不能支持的结论

### 可以支持

1. 原生/全局缩放后的 PI-H2T ratio 仍然过强。统一最坏-margin clipping 将平均 donor mass 从 0.7133 降到 0.3563，并恢复 9.41 点 Overall。
2. 修正后的融合不是简单 no-fusion：平均 donor mass 仍为 0.3563，且相对 T2 同时提高 Overall、Medium 和 Few。
3. D 在完全相同 ratio 多重集合下比 E 高 0.44 点，说明正确 pair correspondence 存在初步正向性能信号。
4. reference 几何约束按定义工作：D 的违规率为 0；打乱对应关系后 E 的违规率升至 46.94%。

### 目前不能支持

1. 不能声称 donor blindness 已稳定修复；D-E 只有单 seed 的 0.44 点，尚无跨 seed 置信度。
2. 不能声称 pair correspondence 对所有频率组都有益。D-E 的收益来自 Many，Medium/Few 均下降。
3. 不能把 D-T4 的全部 9.41 点归因于 correspondence；其中大部分可能来自将 donor mass 从 0.7133 大幅降至 0.3563。D-E 才是 correspondence 的直接证据。
4. 当前 frozen Stage-I reference 对 Few 的约束可能过度保守：Few donor mass 只有 0.2173，并且正确 correspondence 相对 shuffled control 没有带来 Few 增益。

## 5. Go/No-Go

seed42 满足：

- D > T4；
- D > E；
- D > T2。

但不满足预先约定的尾类保护条件：D 相对 T4 的 Few 大幅下降，并且 D-E 的小幅优势来自 Many 而不是 Few。因此本轮停止在 seed42，不立即补 seed43/44，也不进行阈值、slack 或 group-wise 系数搜索。

这不是“没有正向结果”，而是得到了一项有边界的正向结果：

> 统一最坏-margin clipping 能把 PIF 下失效的 H2TF 拉回并略超 no-fusion；正确 pair correspondence 对 Overall 有单 seed 正向信号，但当前 Stage-I reference geometry 的收益偏向 Many，尚未成为可靠的长尾改进方法。

## 6. 产物

- 实现：`train_pi_h2t_ratio_only_stage2.py`
- smoke：`outputs/bpcf_unified_margin_smoke_seed42/`
- 完整 D/E：`outputs/bpcf_unified_margin_pif_seed42/`
- T4：`outputs/pif_endpoint_seed42_T4_kappa_matched/`
- T2：`outputs/pif_endpoint_seed42_T2_k0_bnshift/`

