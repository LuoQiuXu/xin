# PI-H2T：OOF reference 与 controlled crossing 实验报告（seed 42）

## 1. 本轮要回答的问题

本轮不引入 donor selection、新 loss 或新 classifier head，只替换 PI-H2T 的融合比例：

1. recipient-OOF balanced reference 能否减轻 reference head 对尾类训练样本的过拟合；
2. 在严格安全比例上加入 reliability-controlled crossing，能否改善尾类；
3. 在 ratio 分布完全一致时，正确的 recipient-donor correspondence 是否优于错误 correspondence。

## 2. 固定不变的条件

- CIFAR100-LT，IF=100，seed 42；
- 同一个 PIF Stage-I checkpoint；
- frozen backbone，Stage-II BN shift；
- class-balanced recipient 与 instance-wise donor；
- 同一 PI-H2T H2TF、CE、SGD、10 epochs；
- Stage-II classifier 始终从原 Stage-I classifier 初始化；
- 测试指标只采用固定第 10 epoch，不使用 test 选 checkpoint。

## 3. OOF reference 与可靠度

训练集按类别分层为 5 folds。第 k 个 balanced cRT head 不使用 fold k；对 fold k 的温度 `T_k`，只使用其他四个 fold 的 OOF logits/labels 拟合 class-balanced NLL。样本可靠度固定为：

```text
q_i = p_T^OOF(y_i | f_i)
```

没有 correctness hard gate，也不会随 Stage-II augmentation 逐 epoch 重算。

OOF head 指标：Overall 96.66%，Many 96.97%，Medium 95.04%，Few 89.86%。温度为 0.3723–0.4096。虽然比原 balanced head 的训练流 Few 99.03% 低，但仍远高于独立测试 Few；原因是只对 reference head 做了 OOF，Stage-I backbone 仍见过全部训练样本。

Stage-II 最后一轮采样到的静态 q 均值为：All 0.9124，Many 0.9425，Medium 0.9299，Few 0.8548。

## 4. 实验组

- A：原 PI-H2T recipient-only ratio；
- T2：相同 Stage-II/BN 流下不融合；
- D0：recipient-OOF strict pair boundary；
- E0：保留 D0 的 exact ratio multiset，只用错误 donor geometry 重新分配 ratio，实际融合仍使用原 donor；
- Dcc：`lambda_safe + alpha(1-q)(lambda0-lambda_safe)`；
- Ecc：保留 Dcc 的 exact ratio multiset，只破坏 donor correspondence。

E0/Ecc 在 `Many/Medium/Few × 4个lambda0-bin × 2个q-bin` 内做 epoch-level reallocation。

## 5. alpha 的训练内选择

固定特征 5-fold Stage-II OOF 校准结果：

| alpha | Overall | Many | Medium | Few | H-mean |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 92.69 | 92.53 | 93.72 | 92.94 | 92.73 |
| 0.5 | 92.37 | 92.26 | 93.50 | 92.48 | 92.37 |
| 1.0 | 92.02 | 91.94 | 93.67 | 91.64 | 91.79 |

按预注册规则会选择 `alpha=0`，即退化为 D0。但这些数值仍受到 Stage-I backbone 训练集泄漏影响，不能视为可信的独立泛化校准。因而正式保留 A/T2/D0/E0；另外把 `alpha=1` 作为预先声明的端点诊断运行 Dcc/Ecc，不用测试集在多个非零 alpha 中选优。

## 6. 固定第 10 epoch 结果

实现 sanity check：新训练引擎的 A 为 37.00%（旧 T4 为 37.26%），T2 为 45.56%（旧 T2 为 45.69%），差异分别为 -0.26/-0.13 点。新实验内部所有 arm 每一轮的 recipient/donor stream checksum 完全一致，因此 D/E 因果比较不受该小幅基线漂移影响。

| 方法 | Overall | Many | Medium | Few | H-mean |
|---|---:|---:|---:|---:|---:|
| A | 37.00 | 30.50 | 46.51 | **33.59** | 31.97 |
| T2 | **45.56** | **64.94** | 47.86 | 18.72 | 29.07 |
| D0 | 40.45 | 39.06 | 49.06 | 31.79 | 35.05 |
| E0 | **40.76** | **39.56** | **49.17** | **32.10** | **35.44** |
| Dcc (`alpha=1`) | 40.19 | 38.47 | 48.77 | 31.97 | 34.92 |
| Ecc (`alpha=1`) | **40.56** | **39.00** | **49.09** | **32.21** | **35.28** |

粗体只表示各自对照对中更高者，不表示统计显著。

关键差值（百分点）：

- `D0 - E0`：Overall -0.31，Many -0.50，Medium -0.11，Few -0.31，H -0.39；
- `Dcc - Ecc`：Overall -0.37，Many -0.53，Medium -0.31，Few -0.24，H -0.36；
- `Dcc - D0`：Overall -0.26，Many -0.58，Medium -0.29，Few +0.17，H -0.13；
- `Dcc - A`：Overall +3.19，Many +7.97，Medium +2.26，Few -1.62，H +2.95。

## 7. 因果审计

| 审计项 | D0 | E0 | Dcc | Ecc |
|---|---:|---:|---:|---:|
| donor mass mean | 0.59839 | 0.59839 | 0.60793 | 0.60793 |
| ratio multiset 最大误差 | 0 | 0 | 0 | 0 |
| 错 donor 可移动比例（末轮） | — | 100% | — | 100% |
| 错 donor 实际移动比例 | — | 100% | — | 100% |
| ratio 被重新分配比例 | — | 98.36% | — | 99.85% |
| boundary violation | 0% | 21.02% | 25.91% | 22.95% |
| crossing depth mean | 0 | 0.08934 | 0.01253 | 0.09417 |

D/E 每个频率组的 donor mass 也一致；全部 epoch 的 multiset error 均为 0，可移动比例最低为 99.97%，实际移动比例始终为 100%。因此 D/E 的差异不能归因于总体融合量、频率组质量或 ratio 分布。

## 8. 结论

本轮没有通过预注册 go/no-go：

1. `D0 > E0` 不成立；
2. `Dcc > Ecc` 在 Medium、Few、H-mean 上均不成立；
3. `Dcc > D0` 只在 Few 上增加 0.17 点，同时 Medium、Overall、H-mean 均下降；
4. Dcc 虽比 A 的 Overall/H-mean 高，但 Few 低 1.62 点，不能证明 donor-conditioned correspondence 改善尾类。

目前更符合数据的解释是：**严格按正确 pair geometry 保护 reference margin 仍然过于保守；错误 correspondence 造成的更深边界穿越反而略有利。** 这支持“尾类需要受控困难”的大方向，但不支持当前这版 `safe boundary + OOF q relaxation` 已经找到了正确的 donor-conditioned 控制规律。

seed 42 已未通过机制 gate，因此暂不扩 seed 43/44。下一次修改应先解决两个问题：可靠度需要来自真正未被 Stage-I representation 训练过的验证样本；比例目标需要显式控制 crossing depth 区间，而不是从 strict-safe 单向、且被过高 q 强烈抑制地放松。

## 9. 产物

- OOF reference：`outputs/oof_balanced_reference_seed42/recipient_oof_5fold.pth`
- A/T2/D0/E0：`outputs/oof_cc_pif_seed42/summary.csv`
- alpha=1 Dcc/Ecc：`outputs/oof_cc_pif_seed42_alpha1/summary.csv`
- OOF 训练脚本：`train_oof_balanced_reference.py`
- Stage-II 脚本：`train_oof_controlled_crossing_stage2.py`
