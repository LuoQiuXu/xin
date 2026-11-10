# CC-SAM-TPR 残差空间版执行报告

执行前本地和 GitHub 均位于 `experiment/ccsam-tpr-20261110` 的 `627eaa6cd94e9e94b0a02334c494b99e22e9166e`。本轮使用已同步的实现和协议。初次导入失败日志逐字节保留。

## 阶段和退出码

所有命令从本目录执行，python 指任务指定的 xiaoju 解释器。时间使用 UTC+08:00。

| 阶段 | 命令 | 开始 | 结束 | 退出码 |
|---|---|---|---|---:|
| syntax | `python -B -m py_compile train_tpr.py test_tpr.py analyze_tpr.py finalize_tpr.py` | 2026-11-10T20:54:26+08:00 | 2026-11-10T20:54:26+08:00 | 0 |
| unit_tests | `python -B -m unittest -v test_tpr` | 2026-11-10T20:54:26+08:00 | 2026-11-10T20:54:31+08:00 | 0 |
| import_evidence | `python -B -X utf8 -` 〔真实导入两个模块〕 | 2026-11-10T20:54:31+08:00 | 2026-11-10T20:54:35+08:00 | 0 |
| preflight | `python -B -u train_tpr.py --preflight` | 2026-11-10T20:54:35+08:00 | 2026-11-10T20:54:41+08:00 | 0 |
| preflight_evidence | `python -B -X utf8 -` 〔独立核验 16 项锁定输入〕 | 2026-11-10T20:54:41+08:00 | 2026-11-10T20:54:41+08:00 | 0 |
| smoke | `python -B -u train_tpr.py --smoke-only` | 2026-11-10T20:54:41+08:00 | 2026-11-10T20:54:50+08:00 | 0 |
| smoke_evidence | `python -B -X utf8 -` 〔核验 6 个配对批次、图及 guard〕 | 2026-11-10T20:54:50+08:00 | 2026-11-10T20:54:50+08:00 | 0 |
| formal | `python -B -u train_tpr.py --formal` | 2026-11-10T20:54:50+08:00 | 2026-11-10T21:08:24+08:00 | 0 |
| analysis | `python -B -u analyze_tpr.py` | 2026-11-10T21:08:24+08:00 | 2026-11-10T21:08:29+08:00 | 0 |

12 项实际 CPU 单元测试全部通过。两个回归测试分别确认 C3 helper 来自 ab_head_preservation.train_ab，以及 Smoke 直接传递 protected_inputs。独立导入检查确认两个模块引用同一函数。

Preflight 核验三份同 seed C2 checkpoint、原采样计划与固定 sampler 的完整顺序、准备数组和源文件。独立核对通过 16 项原输入 SHA256。C0–C4 的 52、26、31 项封存条目在运行前后均与发布索引一致，另一个 TPR 分支留在本机的 10 项产物和 140 项历史未跟踪文件也保持一致。

真实 Smoke 从 Stage II TRAIN 构造 65 个非 Many 类的 top-2 无向图，共 96 条边。三个 seed 各两个连续增强批次精确匹配 C1 与 C4。每组完成一次丢弃的 A/B 更新。原 C1 解析梯度校验、梯度有限性和冻结门禁通过。

## 实际训练和哈希

实际记录合计 90 轮、15,300 次优化器更新和 976,230 个采样位置。完整轨迹与同 seed 的 C1 和 C4 文件 SHA256 一致。90 轮均通过 C2 基座、Many 输出行、Backbone 和 BN 校验。合并 checkpoint 的 Many 权重行和全量偏置另经逐字节核验。

| Seed | Epoch | 实际步数 | 配对批次 | 采样位置 |
|---:|---:|---:|---:|---:|
| 123 | 30 | 5,100 | 5,100 | 325,410 |
| 124 | 30 | 5,100 | 5,100 | 325,410 |
| 125 | 30 | 5,100 | 5,100 | 325,410 |

| Seed | C2 基座 checkpoint SHA256 | TPR checkpoint SHA256 | C1/C4/TPR 轨迹 SHA256 |
|---:|---|---|---|
| 123 | e4a935871c80d6f62a09ecb34b070348f1b11687fef0e59184cd8cf49c9970fb | 32f9ce871f49f8597e4f497c04b7ad65fb18fe9652828fad19f5ec8566fa616c | e422f9c106cf90399831df7e9d6c6377b8d98e1d677d8ba929d120d55bf4dcdc |
| 124 | 0da2c30eff7d7de1b2eafdb963690d128cde72a30557cc33ddbc0625d96a5eed | ce5c34cf6672eef75025c02b4f65f8aa40f9951f0f03ccf4f133f9b0014079ab | ffa484e36bcb88d1e1c889a49a34c315602de38f105e67ee24a11049d55e44e9 |
| 125 | 9613a92beb407fd9fc7b10158f3e8235a32701dfd0435dc001670c67a6d17c6b | 8fe8908aeb66b593d3c19f22bc0ebc5c960dd1597289c59c54c8b9170eac834b | abbab224efec845ea9297005938a0aeaabb055c8221bb0b11feaee9b38637888 |

Smoke 和 Formal 的 TEST 文件打开与 CIFAR100(train=False) 计数均为 0。固定第 30 轮分析读取已存储的 TRAIN holdout 结果。

## 第 30 轮结果

准确率为百分比，表中为三 seed 均值 ± 样本标准差。验证使用同一 1,980 张 TRAIN holdout 和同一冻结 Backbone。class 0 参与训练，holdout 中该类样本数为 0。

| 条件 | Overall | Macro | Many | MediumN | Few | Head | OfficialMedium | Tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 46.869 ± 0.175 | 46.869 ± 0.175 | 64.804 ± 0.340 | 48.619 ± 0.675 | 24.500 ± 0.289 | 63.762 ± 0.297 | 48.667 ± 0.595 | 24.310 ± 0.299 |
| C1 | 46.347 ± 0.029 | 46.347 ± 0.029 | 58.431 ± 0.306 | 48.571 ± 0.247 | 30.056 ± 0.192 | 57.619 ± 0.297 | 48.857 ± 0.247 | 29.713 ± 0.199 |
| C2 | 46.902 ± 0.105 | 46.902 ± 0.105 | 63.235 ± 0.441 | 49.048 ± 0.918 | 25.889 ± 0.419 | 62.286 ± 0.429 | 49.048 ± 0.918 | 25.747 ± 0.434 |
| C3 | 46.296 ± 0.117 | 46.296 ± 0.117 | 57.892 ± 0.370 | 48.619 ± 0.218 | 30.444 ± 0.347 | 57.095 ± 0.360 | 48.905 ± 0.218 | 30.115 ± 0.359 |
| C4 | 46.852 ± 0.146 | 46.852 ± 0.146 | 61.569 ± 0.170 | 48.952 ± 0.218 | 27.722 ± 0.536 | 60.667 ± 0.165 | 49.190 ± 0.297 | 27.356 ± 0.653 |
| TPR | 46.869 ± 0.101 | 46.869 ± 0.101 | 63.137 ± 0.370 | 48.857 ± 0.795 | 26.111 ± 0.385 | 62.190 ± 0.360 | 48.857 ± 0.795 | 25.977 ± 0.398 |

### 预设探索阈值

| 阈值 | 实测 | 结果 |
|---|---:|---|
| Many 均值相对 C2 ≥ −0.2 pp | -0.098 pp | PASS |
| Few 均值相对 C2 ≥ +2.0 pp | +0.222 pp | NOT MET |
| Overall 均值相对 C2 ≥ −0.2 pp | -0.034 pp | PASS |
| 至少两个 seed 的 Many 相对 C2 ≥ −0.2 pp | 3/3 | PASS |
| 至少两个 seed 的 Few 相对 C2 为正 | 2/3 | PASS |
| Few 均值相对 C4 ≥ −0.5 pp | -1.611 pp | NOT MET |

逐 seed 配对差值、逐类结果和 MORE 分量见 [固定轮次结果](analysis/TPR_RESULTS.md)、[配对表](analysis/paired_deltas.csv)、[逐类表](analysis/per_class.csv) 和 [末轮指标](analysis/final_epoch_metrics.csv)。

## 日志和归档

TPR_EXECUTION.log 是原失败记录。TPR_EXECUTION_627eaa6.log 保留本轮语法检查至分析的完整输出。主日志关闭后执行原版 finalize_tpr.py，使其记录的是稳定的日志字节。finalize 的命令、输出和退出码另存在 TPR_FINALIZE_627eaa6.log。

Git 保留新报告和指标的原始字节。原版清单对已跟踪源码记录本机执行字节，Git 可能对既有源码的 CRLF 换行做标准化。两个训练 transcript、最终权重和配对轨迹保留在本机。

[Preflight](preflight_report.json) · [Smoke](smoke_report.json) · [TRAIN 图](tail_neighbor_graph.json) · [正式完成标记及 TEST guard](formal_completed.json) · [SHA256 清单](ARTIFACT_SHA256.csv)
