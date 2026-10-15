# PI-H2T H2TF ratio-only：CE Stage-I 下的三随机种子复验

## 结论

在固定同一个无 PIF 的 CE Stage-I checkpoint、只改变 Stage-II 随机种子（42、43、44）的前提下，pair-conditioned ratio（B）在三个 seed 的第 10 epoch Last Overall 都同时高于：

- 原始 PI-H2T 的 recipient-only ratio（A）；
- 保持完全相同 ratio 集合、但打乱 ratio--pair 对应关系的对照（C）。

三 seed 的平均结果为 `B = 41.12 ± 0.16`，`A = 40.36 ± 0.13`，`C = 40.61 ± 0.18`（均值 ± 样本标准差，单位：%）。对应的配对提升为 `B-A = +0.77 ± 0.16` 个百分点、`B-C = +0.51 ± 0.14` 个百分点。

因此，当前 CE setting 的预注册 gate 通过：**在不改变原始 PI-H2T 融合强度分布的条件下，把 ratio 分配给正确的 recipient--donor pair 是有效的；PI-H2T 的 recipient-only ratio 确实遗漏了会影响最终性能的 donor 信息。**

这仍是 H2TF ratio mechanism isolation，而不是完整 PI-H2T 的最终性能结论：当前 Stage-I checkpoint 不含 PIF。

## 固定协议与范围

- 数据：CIFAR100-LT，IF=100。
- Stage-I：同一个 CE ResNet-32 checkpoint，`cifar100_rho100_resnet32_ce_seed42/best.pth`，epoch 184；`stage1_has_pif=False`。
- 几何：全部使用冻结的原 Stage-I classifier head；没有使用 cRT、balanced reference 或 EMA reference。
- Stage-II：官方 H2T 的 class-balanced recipient + instance-wise donor 数据流；冻结 backbone、BN shift、CE、10 epochs、batch size 128、`workers=0`。
- 融合公式固定为论文 H2TF 的 pooled-feature 形式：

  ```text
  f_tilde = r f_B + (1-r) f_I = (1-lambda) f_B + lambda f_I
  ```

- 没有加入 donor selection、候选 donor、risk-weighted loss、clean anchor、固定 lambda、额外 sampling、reference 校准或任何超参搜索。
- 三次复验仅将 Stage-II 随机种子改为 42、43、44；每个 seed 内的 A/B/C 使用同一 recipient/donor/augmentation 数据流。
- 主指标预先固定为第 10 epoch 的 **Last Overall**；Best 仅保留作诊断，不参与结论。

### 三个对照组

| 组别 | ratio 分配 | 目的 |
|---|---|---|
| A：`pi_recipient_only` | 原始 PI-H2T 的 recipient-only ratio | 基线 |
| B：`pair_stratified_rankmatch` | 在 `Many/Medium/Few × 原 PI recipient-ratio 四分层` 内，将原生 `lambda_PI` 多重集合按 `lambda_critical(f_B,f_I)` 升序匹配给实际 pair | 只修正 recipient--donor correspondence |
| C：`pair_stratified_rankmatch_shuffled` | 保留 B 的同一 ratio 多重集合、同一 recipient、同一 donor；仅在相同分层内打乱 ratio--pair 对应 | 排除“仅换了 ratio 分布”的解释 |

其中 `lambda` 为 donor mass。B 满足 `lambda_critical ↑ → lambda_pair ↑`，即更安全的实际 pair 获得更大的 donor mass；A/C 不要求这一单调对应关系。

## 主结果：Last Overall

| Stage-II seed | A：PI recipient-only | B：pair-conditioned | C：shuffled correspondence | B - A | B - C |
|---:|---:|---:|---:|---:|---:|
| 42 | 40.36 | **40.96** | 40.55 | +0.60 | +0.41 |
| 43 | 40.23 | **41.14** | 40.47 | +0.91 | +0.67 |
| 44 | 40.48 | **41.27** | 40.81 | +0.79 | +0.46 |
| 均值 ± 样本标准差 | 40.36 ± 0.13 | **41.12 ± 0.16** | 40.61 ± 0.18 | **+0.77 ± 0.16** | **+0.51 ± 0.14** |

`B>A` 和 `B>C` 在 3/3 个 seed 都成立，故这不是 seed42 的单次偶然排序。

## 频次组结果：跨 seed 均值 ± 样本标准差

| 组别 | Overall | Many | Medium | Few |
|---|---:|---:|---:|---:|
| A：PI recipient-only | 40.36 ± 0.13 | 39.17 ± 0.19 | 50.86 ± 0.17 | 29.16 ± 0.62 |
| B：pair-conditioned | **41.12 ± 0.16** | **40.74 ± 0.40** | 50.90 ± 0.32 | **29.79 ± 0.36** |
| C：shuffled correspondence | 40.61 ± 0.18 | 39.42 ± 0.17 | **50.87 ± 0.46** | 29.71 ± 0.60 |

相对 A，B 的平均变化为 Many `+1.57`、Medium `+0.05`、Few `+0.63` 个百分点；相对 C 为 Many `+1.32`、Medium `+0.04`、Few `+0.08` 个百分点。当前 Overall 优势主要由 Many 贡献，因而不能将本轮表述为“已经证明 Tail 专属收益”。

## 审计：为什么可归因于 correspondence

每个 seed 的最后 epoch 均满足：

| 审计项 | seed42 | seed43 | seed44 |
|---|---:|---:|---:|
| A/B/C ratio 多重集合最大误差 | 0 | 0 | 0 |
| B 的 `lambda_critical → lambda_pair` 单调分配违规数 | 0 | 0 | 0 |
| C 层内实际打乱比例 | 95.20% | 95.32% | 95.20% |
| A/B/C recipient、donor、target、donor-label 数据流校验和 | 一致 | 一致 | 一致 |
| 三组平均 donor mass | 0.654428 | 0.655274 | 0.655041 |

因此：

1. B 对 A 的提升不能由更大的总 donor mass、不同的 Many/Medium/Few donor mass、不同样本对或不同训练流解释；
2. B 对 C 的提升不能由 ratio 分布本身解释，因为 C 保留了 B 在每个控制分层中的完整 ratio 多重集合；
3. 唯一有意改变的变量是：每个原生 PI ratio 是否分配给与其实际 recipient--donor 几何相匹配的 pair。

## 应如何解释，以及不应如何解释

可以得出的结论是：

> PI-H2T 的 recipient-only fusion ratio 并非只是公式上的信息缺失。在当前 H2TF ratio-isolation setting 中，它会造成可复现的最终分类性能损失；pair-conditioned ratio allocation 能降低这部分损失。

仍不能得出的结论是：

- 不能说已经改进了完整 PI-H2T，因为 Stage-I PIF 尚未复现；
- 不能说 B 已经证明“净正向 augmentation”效应。三臂都经历了从早期 Stage-II 测试峰值向 epoch10 回落，较稳妥的解释是 B 减少了错误 correspondence 带来的额外 harmful fusion；
- 不能把结果包装为 Tail-only 改进，当前主要 Overall 增益来自 Many。

## 下一步（不改 A/B/C 协议）

三 seed 的 CE gate 已通过。下一步应训练或取得一个真实的 **PIF Stage-I checkpoint**，然后不改动 ratio 公式、4 个分层、RankMatch、sampling、loss、reference head 或任何超参数，原样重复同一 A/B/C 试验。若 PIF setting 仍满足 `B>A` 且 `B>C`，才可以把结论上升到完整 PI-H2T 的设计缺陷与修正。

## 可复核文件

- `train_pi_h2t_ratio_only_stage2.py`
- `PI_H2T_RATIO_ONLY_PROTOCOL.md`
- `outputs/pi_h2t_ratio_only_ce_seed42/summary.csv`
- `outputs/pi_h2t_ratio_only_ce_seed43/summary.csv`
- `outputs/pi_h2t_ratio_only_ce_seed44/summary.csv`
