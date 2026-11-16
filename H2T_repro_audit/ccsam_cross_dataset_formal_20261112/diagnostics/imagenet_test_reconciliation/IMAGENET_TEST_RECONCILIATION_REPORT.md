# ImageNet-LT 官方测试逐图片对账与跨数据集终评

**ImageNet-LT 对账 50,000/50,000 通过。ImageNet-LT 和 Places-LT 的官方终评均已完成。** 两套评估使用已保存的 6 个 Stage-I backbone 与 18 个 Stage-II 分类器，训练执行次数为 0。两次评估在新的目录下独立保存。[原终评失败状态](../../runs/amp_gradient_fix_20261112_01/final_evaluation/status.json)

## 原始错误与对账结果

官方清单使用 `ILSVRC2012_val_<八位编号>.JPEG`，本地 15 个测试 Parquet 分片的每个文件名都在同一个图像编号后附加了 `_<WNID>`。原加载器要求原样 basename 和类别 ID 完全相同，因文件名格式不一致而停止。单独改变目录前缀、大小写或扩展名均无法解释这批差异。

| 指标 | 官方测试清单 | Parquet 测试分片 |
| --- | ---: | ---: |
| 行数 | 50,000 | 50,000 |
| 唯一原始文件名 | 50,000 | 50,000 |
| 重复文件名 | 0 | 0 |
| 空路径 | 0 | 0 |
| 类别数 | 1,000 | 1,000 |
| 每类样本数 | 50 | 50 |

| 逐图片交叉检查 | 原样 basename | 应用已验证的 WNID 后缀规则 |
| --- | ---: | ---: |
| 文件名匹配 | 0 | 50,000 |
| 同名记录的类别 ID 不同 | 0 | 0 |
| 官方文件名缺失 | 50,000 | 0 |
| Parquet 多余文件名 | 50,000 | 0 |
| 命名冲突 | 0 | 0 |

原样比较中，仅目录前缀变化、仅大小写变化、仅扩展名变化分别为 0、0、0。符合 WNID 后缀规则的记录有 50,000 条。诊断只投影了 `label` 和 `image.path` 两列，覆盖 15 个分片的 510 个 row group。[对账统计与逐类计数](metadata_20261116_01/reconciliation.json) · [前 100 个差异样本](metadata_20261116_01/first_100_differences.json)

### 第一个报错样本

| 查询 | 结果 |
| --- | --- |
| 原路径等于 `ILSVRC2012_val_00029930.JPEG` | 0 条 |
| 任意路径下有完全相同的 basename | 0 条 |
| 完全相同的 basename 对应不同标签 | 0 条 |
| 规范化后同名 | 1 条，`ILSVRC2012_val_00029930_n01440764.JPEG`，标签 `0`，WNID `n01440764` |

该条记录位于 `data/test-00007-of-00015.parquet` 的 `row_group=11`、`row_index=79`，坐标从 0 起算。官方路径为 `val/n01440764/ILSVRC2012_val_00029930.JPEG`，标签为 `0`。

### 类别 ID 和修复门禁

- 官方 Train 清单构建了 WNID 到类别 ID `0–999` 的一一映射。15 个分片的 Hugging Face `ClassLabel.names` 顺序全部一致。50,000 个 Parquet 原始标签均为 Arrow `int64`，与按图片对应的官方 ID 之差全部为 0，附加 WNID 冲突为 0。编号基准和类别顺序均与官方一致。
- 规范化只接受 `ILSVRC2012_val_<八位编号>_<WNID>.JPEG` 中经过核对的 WNID 后缀。完整对账检查原始编号、官方目录 WNID、Parquet 文件名 WNID、标签和物理坐标。通过后，将可信的原始 Parquet basename 与清单标签按官方顺序交给已有 `official_entries` 严格成员检查，并逐条比对其实际物理索引。
- 两侧全部记录唯一且完整匹配，所需补充图像数为 0，本地分片足以完成终评。逐图片对账记录使用从 0 起算的 `row_group` 和 `row_index`。完整对账 CSV 和逐图预测保存在本地，文件路径与 SHA256 均登记在清单中。[`ARTIFACT_SHA256.csv`](ARTIFACT_SHA256.csv)

配对内容 SHA256 为 `5d49cc0a4c196d1e96623625d21cf6c5fc90e6f7d7a9c763499214c4a9b037f0`。官方到 Parquet 的 CSV 文件 SHA256 为 `eab3803bb81822b435104dc8c6c8adc1f3d91f60fe08ef861418a63328965b04`。[诊断产物校验](metadata_20261116_01/status.json)

## 官方测试执行

两套终评均在 RTX 5070 Laptop GPU 上执行固定 checkpoint 推理。Stage I 固定第 90 轮，Stage II 固定第 30 轮，三个种子为 `123`、`124`、`125`。Places-LT 使用官方 `Places_LT_test.txt` 与训练清单产生的 365 个 class-folder 映射，经唯一性检查后从解包的 `val` 目录读取 36,500 个测试图像。新评估使用原配置的 batch size `16`、4 个 loader worker 和固定的 224 像素中心裁剪。

| 数据集 | 已预测的测试图像 | 测试类别 | 开始（北京时间） | 结束（北京时间） | 退出码 |
| --- | ---: | ---: | --- | --- | ---: |
| ImageNet-LT | 50,000 | 1,000 | 2026-11-16 22:09:21 | 2026-11-16 22:14:10 | 0 |
| Places-LT | 36,500 | 365 | 2026-11-16 22:14:24 | 2026-11-16 22:18:27 | 0 |

[ImageNet-LT 终评状态](../../runs/imagenet_test_reconciled_20261116_01/imagenet_lt/status.json) · [Places-LT 终评状态](../../runs/imagenet_test_reconciled_20261116_01/places_lt/status.json) · [CPU 测试日志](../../runs/imagenet_test_reconciled_20261116_01/validation/cpu_tests.log)

原有 22 项 CPU 测试和新增 5 项对账回归测试均通过。终评状态和文件 SHA256 已复核，从保存的逐图预测独立复算了 24 组 Overall、Many、Medium、Few 和逐类准确率，并核对预测文件的标签顺序与对应官方清单一致。

### 三个种子的均值 ± 样本标准差（%）

Overall 为全体测试样本的 top-1 准确率，Many、Medium、Few 对对应类别的准确率做宏平均。类别分组由训练样本数决定，Many >100，Medium 为 20–100，Few <20。两个数据集的 Many、Medium、Few 类别数分别为 385/479/136 和 131/163/71。

| 数据集 | 组别 | Overall | Many | Medium | Few |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ImageNet-LT | Stage-I | 42.09 ± 0.61 | 65.73 ± 0.72 | 33.68 ± 0.63 | 4.79 ± 0.43 |
| ImageNet-LT | D0 | 48.27 ± 0.63 | 61.05 ± 0.78 | 45.05 ± 0.66 | 23.41 ± 0.24 |
| ImageNet-LT | D1 | 49.58 ± 0.49 | 58.67 ± 0.76 | 47.07 ± 0.40 | 32.73 ± 0.17 |
| ImageNet-LT | D2 | 49.14 ± 0.53 | 59.92 ± 0.86 | 45.96 ± 0.47 | 29.84 ± 0.14 |
| Places-LT | Stage-I | 23.79 ± 0.13 | 43.51 ± 0.27 | 17.94 ± 0.05 | 0.86 ± 0.11 |
| Places-LT | D0 | 31.71 ± 0.24 | 39.46 ± 0.11 | 32.57 ± 0.40 | 15.40 ± 0.50 |
| Places-LT | D1 | 34.24 ± 0.20 | 37.75 ± 0.27 | 35.35 ± 0.32 | 25.25 ± 0.78 |
| Places-LT | D2 | 33.56 ± 0.19 | 40.23 ± 0.19 | 33.44 ± 0.12 | 21.49 ± 0.96 |

### 固定种子配对差值的均值（百分点）

| 数据集 | 对比 | Overall | Many | Medium | Few |
| --- | --- | ---: | ---: | ---: | ---: |
| ImageNet-LT | D1-D0 | +1.32 | -2.38 | +2.02 | +9.31 |
| ImageNet-LT | D2-D0 | +0.87 | -1.14 | +0.91 | +6.43 |
| ImageNet-LT | D2-D1 | -0.45 | +1.25 | -1.11 | -2.89 |
| Places-LT | D1-D0 | +2.54 | -1.71 | +2.77 | +9.85 |
| Places-LT | D2-D0 | +1.85 | +0.77 | +0.87 | +6.09 |
| Places-LT | D2-D1 | -0.69 | +2.48 | -1.90 | -3.76 |

### 逐种子结果（%）

| 数据集 | Seed | 组别 | Overall | Many | Medium | Few |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ImageNet-LT | 123 | Stage-I | 41.5200 | 64.9506 | 33.1649 | 4.6176 |
| ImageNet-LT | 123 | D0 | 47.6020 | 60.1766 | 44.3633 | 23.4118 |
| ImageNet-LT | 123 | D1 | 49.0600 | 57.8130 | 46.6722 | 32.6912 |
| ImageNet-LT | 123 | D2 | 48.5560 | 58.9299 | 45.5115 | 29.9118 |
| ImageNet-LT | 124 | Stage-I | 42.7400 | 66.3792 | 34.3758 | 5.2794 |
| ImageNet-LT | 124 | D0 | 48.8520 | 61.6883 | 45.6910 | 23.6471 |
| ImageNet-LT | 124 | D1 | 50.0320 | 59.2727 | 47.4656 | 32.9118 |
| ImageNet-LT | 124 | D2 | 49.5840 | 60.4364 | 46.4426 | 29.9265 |
| ImageNet-LT | 125 | Stage-I | 42.0100 | 65.8701 | 33.4864 | 4.4853 |
| ImageNet-LT | 125 | D0 | 48.3460 | 61.2987 | 45.0814 | 23.1765 |
| ImageNet-LT | 125 | D1 | 49.6600 | 58.9299 | 47.0605 | 32.5735 |
| ImageNet-LT | 125 | D2 | 49.2760 | 60.3844 | 45.9123 | 29.6765 |
| Places-LT | 123 | Stage-I | 23.8548 | 43.5573 | 17.9877 | 0.9718 |
| Places-LT | 123 | D0 | 31.7562 | 39.5344 | 32.5337 | 15.6197 |
| Places-LT | 123 | D1 | 34.3836 | 37.4809 | 35.6564 | 25.7465 |
| Places-LT | 123 | D2 | 33.5699 | 40.0534 | 33.4785 | 21.8169 |
| Places-LT | 124 | Stage-I | 23.8822 | 43.7557 | 17.9387 | 0.8592 |
| Places-LT | 124 | D0 | 31.9123 | 39.3282 | 32.9939 | 15.7465 |
| Places-LT | 124 | D1 | 34.3315 | 37.7481 | 35.3681 | 25.6479 |
| Places-LT | 124 | D2 | 33.7370 | 40.2061 | 33.5399 | 22.2535 |
| Places-LT | 125 | Stage-I | 23.6438 | 43.2137 | 17.8896 | 0.7465 |
| Places-LT | 125 | D0 | 31.4466 | 39.5191 | 32.1963 | 14.8310 |
| Places-LT | 125 | D1 | 34.0164 | 38.0153 | 35.0123 | 24.3521 |
| Places-LT | 125 | D2 | 33.3589 | 40.4351 | 33.3129 | 20.4085 |

[ImageNet-LT 原始指标](../../runs/imagenet_test_reconciled_20261116_01/imagenet_lt/ImageNet_LT_metrics.csv) · [Places-LT 原始指标](../../runs/imagenet_test_reconciled_20261116_01/places_lt/Places_LT_metrics.csv) · [ImageNet-LT 逐类准确率](../../runs/imagenet_test_reconciled_20261116_01/imagenet_lt/per_class_accuracy.csv) · [Places-LT 逐类准确率](../../runs/imagenet_test_reconciled_20261116_01/places_lt/per_class_accuracy.csv)

## 权重复用与原产物校验

两次独立评估都核验了 6 个 Stage-I checkpoint、18 个 Stage-II checkpoint、6 份训练特征缓存及其标签数组，保留 D0/D1/D2 的原始采样、同种子配对和 D2 teacher 绑定。权重在推理前后均与归档 SHA256 一致，已加载的 encoder、BN 缓冲区和分类器状态也通过评估后哈希核验。原归档索引中的 305 项文件均通过复核，包括原 `final_evaluation` 的失败状态与日志。[ImageNet-LT 权重复用证据](../../runs/imagenet_test_reconciled_20261116_01/imagenet_lt/checkpoint_reuse.json) · [Places-LT 权重复用证据](../../runs/imagenet_test_reconciled_20261116_01/places_lt/checkpoint_reuse.json)

| 数据集 | Seed | checkpoint | 原始 SHA256 |
| --- | ---: | --- | --- |
| ImageNet-LT | 123 | Stage I · epoch 90 | `eca33a8df37074b3a6b23b135dde0ed6b62114fa208754e8607a7fd7421a054c` |
| ImageNet-LT | 123 | D0 · epoch 30 | `2ce8bfe579e7b34513c87b6f9fb8efe4d6c2be7c639fc10b617a6a12c4398e5d` |
| ImageNet-LT | 123 | D1 · epoch 30 | `e35eff2912fd26d9d0db9f0623bc027e667f27168d64e2c5e7be86a82a853bbd` |
| ImageNet-LT | 123 | D2 · epoch 30 | `98fd9f8780a7f3dfeb2e0642ccf4d666ff4228e8aad9dd0fa7965602eb3a97ec` |
| ImageNet-LT | 124 | Stage I · epoch 90 | `7ec0f9b8a4e6da14ade93f12a28d731f47086987d61b6acb49247180fb1b2bae` |
| ImageNet-LT | 124 | D0 · epoch 30 | `e40502ff82103f851d25ac9bd4e1bffa02cac57daedc4c33774c335ac4450a54` |
| ImageNet-LT | 124 | D1 · epoch 30 | `6af6f4dac9b282c9250d7886f92d2af8ec977514be7e6428790105d78b1790ea` |
| ImageNet-LT | 124 | D2 · epoch 30 | `dbf34dce2c42bb72bdef924a91464d8e9c38c10875c11e0366d629085897dde6` |
| ImageNet-LT | 125 | Stage I · epoch 90 | `a22fb0b48b5da48db6b195ea089a76b6bdace16be9a89cd068329255bd13d67c` |
| ImageNet-LT | 125 | D0 · epoch 30 | `cf4b9b93a37c38bbdf1f997cc9cdb6b8f31ec7829f9ca6925d52d2f2a8bd81df` |
| ImageNet-LT | 125 | D1 · epoch 30 | `d2d763d1d59bd04bb8ee98db413fc2489661de3cd9cd0decdf5f7d3717533982` |
| ImageNet-LT | 125 | D2 · epoch 30 | `0b7f5ddaa80ec5cb1f0f926b91249de6ecaf470e0ea713da0e8686eb4a235661` |
| Places-LT | 123 | Stage I · epoch 90 | `2147090d49e6f6e3b3bfdbaaa55f096e9d4cf190b2e668e24cef577dd6f7e439` |
| Places-LT | 123 | D0 · epoch 30 | `822e663e6d733493ed542a5aa1853b6ac59ef4ea151de429c2e178018183f820` |
| Places-LT | 123 | D1 · epoch 30 | `ef4ede6897209f40b77a1e5601221012a4b738fd9cb293a5193c40d75d54072f` |
| Places-LT | 123 | D2 · epoch 30 | `a278547a62daa72d7982b754c28df3979ebc0ed08cd57ecdfcf11f28883043e6` |
| Places-LT | 124 | Stage I · epoch 90 | `b6195fa59b177ab6bc36bff72f768c930c1a4db156e5f0f7dbaa630941b0790f` |
| Places-LT | 124 | D0 · epoch 30 | `c525362e7f32386ef15afe42cb67761db4ecaaf0be696d956635aea92356bb05` |
| Places-LT | 124 | D1 · epoch 30 | `3612c9c0bf20ed8c08fed2d989d1bf5783799dac62784458ebe0709d71d1fbf9` |
| Places-LT | 124 | D2 · epoch 30 | `675c6313efa14e7f3f68441b89e1c6c94c3cd26b6d56eb1c4754a94244e436da` |
| Places-LT | 125 | Stage I · epoch 90 | `03ee8235207596dbc5d51b74fdb98719c06ed936cf4b2bc832f0a69ef083e3e5` |
| Places-LT | 125 | D0 · epoch 30 | `a442bdcd63ba681de0d7306f461ccb97c98b775846bda708d8f139b121d95290` |
| Places-LT | 125 | D1 · epoch 30 | `bdfb0f3f497a2d2c1074ba00e684454995929f068019de11baaec425f0b1f34f` |
| Places-LT | 125 | D2 · epoch 30 | `88f752499a76bc75cf079a2b98bb7c012a13d8568538abb9cd7f2862a69ac3a0` |

分支为 `experiment/ccsam-cross-dataset-formal-20261112`，执行时 HEAD 为 `73db1e68851a80694185130c580fcd0e46df5bcd`。新增程序位于本诊断目录，原训练和终评失败产物保留在原有运行目录。[只读诊断](diagnose_imagenet_test.py) · [独立评估入口](evaluate_reconciled.py) · [回归测试](test_imagenet_test_reconciliation.py)
