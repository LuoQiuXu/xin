# CC-SAM Stage II 正式执行完成记录

九组正式实验全部完成，原生命令退出码 0；固定末轮 epoch=30。

| Seed | 条件 | Overall (%) | Few (%) | Tail (%) |
| --- | --- | ---: | ---: | ---: |
| 123 | C0_BALANCED_CE | 47.070707 | 24.333333 | 24.137931 |
| 123 | C1_CCSAM_FEATURE_GRAD | 46.313131 | 30.166667 | 29.827586 |
| 123 | C2_WEIGHTED_CE_NO_PERTURB | 47.020202 | 25.500000 | 25.344828 |
| 124 | C0_BALANCED_CE | 46.767677 | 24.333333 | 24.137931 |
| 124 | C1_CCSAM_FEATURE_GRAD | 46.363636 | 29.833333 | 29.482759 |
| 124 | C2_WEIGHTED_CE_NO_PERTURB | 46.868687 | 25.833333 | 25.689655 |
| 125 | C0_BALANCED_CE | 46.767677 | 24.833333 | 24.655172 |
| 125 | C1_CCSAM_FEATURE_GRAD | 46.363636 | 30.166667 | 29.827586 |
| 125 | C2_WEIGHTED_CE_NO_PERTURB | 46.818182 | 26.333333 | 26.206897 |

## 配对效应（百分点，3 个 seed 的均值）

| 比较 | Overall | Few | Tail |
| --- | ---: | ---: | ---: |
| C1_minus_C0_full_transfer | -0.521886 | +5.555556 | +5.402299 |
| C1_minus_C2_feature_perturbation | -0.555556 | +4.166667 | +3.965517 |
| C2_minus_C0_weighting_and_curriculum | +0.033670 | +1.388889 | +1.436782 |

实际完成 270 个 Epoch、45,900 次 optimizer.step、2,928,690 次样本使用；每组 5,100 个批次轨迹，C1/C2 与同 seed C0 的索引、增强图像、GAP 特征和初始化均精确配对。

官方 TEST 打开/实例化尝试均为 0；原始 checkpoint、prepared、采样计划和生产源码哈希不变。原 Smoke 未重跑，未扩展任何种子、轮数或机制。

主比较为 C1−C2，隔离特征扰动的增量效应；C2−C0 为类别加权和 alpha 混合的效应。数值来自原分析程序的固定末轮评价，未选择最佳 validation epoch。

三个 seed 共享一个 Clean Stage I backbone；1980 张 validation 来自官方 TRAIN 且不含 class 0，并非独立 TEST。三种子结果不构成独立表征或跨数据集推广证据，不宣称统计显著性。

完整机器结果见 ../summary.json、../final_epoch_metrics.csv、../paired_deltas.csv；原分析报告为 ../CCSAM_STAGE2_RESULTS.md。执行证据见 START.json、END.json、formal.log、VERIFICATION.json；所有本地 checkpoint/批次轨迹 SHA256 见 ARTIFACT_SHA256.csv。
