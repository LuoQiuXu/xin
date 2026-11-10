# CC-SAM-TPR 执行与完整性报告

执行分支为 `experiment/ccsam-tpr-lowrank-20261110`，开始时 HEAD 为 `672e4df8e26975524b44243ae52970f48f64baec`，C4 基线为 `4ab34ad00c86bcac60d765123f153b76ff097f47`。

C0–C4 的三份独立清单分别核验 52、26、31 项，归档前复核显示受保护文件变更数为 0。16 项锁定输入和 140 项历史未跟踪文件也与原记录一致。

任务附件 E 节与冻结协议第 3 节指定混淆图只使用全部 10,847 张未增强 Stage II TRAIN 图像。协议第 5 节第 8 条仍写 holdout，本次以正文的 TRAIN 规则执行。采用双向行归一化平均、正分混淆边及类别 ID 平局规则。平方余弦损失等权归约。

每个 seed 的 A 使用该 seed 的确定性正交初始化，B 初始为零。固定子空间来自一次 CPU FP32 SVD，两项正则的数值保护使用 epsilon 1e-12。基础 C1 损失和解析梯度沿用原实现。

## 实际执行命令

命令均在 TPR 目录运行。python 指任务指定的 xiaoju 解释器，时间使用 UTC+08:00。

| 阶段 | 命令 | 开始 | 结束 | 退出码 |
|---|---|---|---|---:|
| syntax | `python -B -m py_compile tpr_model.py tpr_inputs.py train_tpr.py analyze_tpr.py test_tpr_model.py` | 2026-11-10T14:42:25+08:00 | 2026-11-10T14:42:26+08:00 | 0 |
| unit_tests | `python -B -m pytest -q -p no:cacheprovider test_tpr_model.py` | 2026-11-10T14:42:26+08:00 | 2026-11-10T14:42:41+08:00 | 0 |
| preflight | `python -B -u train_tpr.py --phase preflight --seeds 123 124 125` | 2026-11-10T14:42:41+08:00 | 2026-11-10T14:42:48+08:00 | 0 |
| smoke | `python -B -u train_tpr.py --phase smoke --seeds 123 124 125 --prepare-missing` | 2026-11-10T14:42:48+08:00 | 2026-11-10T14:42:58+08:00 | 0 |
| formal | `python -B -u train_tpr.py --phase formal --seeds 123 124 125` | 2026-11-10T14:42:58+08:00 | 2026-11-10T14:56:05+08:00 | 0 |
| analysis | `python -B -u train_tpr.py --phase analyze --seeds 123 124 125` | 2026-11-10T14:56:05+08:00 | 2026-11-10T14:56:11+08:00 | 0 |

9 项 CPU 测试通过。三个 seed 的真实 Smoke 各核验 2 个 C1 配对 batch。初始 logits 与 C2 逐位一致，第一步 A 梯度为零，第二步 A、B 均有有限非零梯度且更新。两步均通过原 C1 解析梯度与 Autograd 的原容差门禁。

| Seed | 第一步 B 梯度 L2 | 第二步 A 梯度 L2 | 第二步 B 梯度 L2 |
|---:|---:|---:|---:|
| 123 | 0.08780134 | 0.008764601 | 254.577 |
| 124 | 0.08334184 | 0.005183439 | 141.3357 |
| 125 | 0.07912831 | 0.004604285 | 134.2225 |

正式训练完成 90 轮和 15,300 次优化器更新。采样位置共 976,230 个。三个完整轨迹的 SHA256 均与对应 C1 轨迹相同。每轮通过 C2 基座、Backbone、BN 和 Many 参数行的逐位校验。

下表报告前 10 批的最大加权正则占比，以及全程 B 梯度峰值。梯度峰值为实际记录的数值瞬态，与标量正则占比是不同指标。

| Seed | max(0.01·L_sub / L_C1) | max(0.01·L_div / L_C1) | 全程 max ‖∇B‖₂ | 出现位置（轮/批，批从 0 起） |
|---:|---:|---:|---:|---|
| 123 | 0.738% | 1.403% | 254.577 | 1/1 |
| 124 | 0.431% | 3.033% | 141.3357 | 1/1 |
| 125 | 0.246% | 1.266% | 134.2225 | 1/1 |

Preflight、TRAIN 图、Smoke、正式训练和分析的两个 TEST 访问计数均为 0。

## 固定第 30 轮

| Seed | Overall | Macro | Many | MediumN | Few | Head | OfficialMedium | Tail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 123 | 45.202 | 45.202 | 60.294 | 45.714 | 27.500 | 59.429 | 45.857 | 27.241 |
| 124 | 46.566 | 46.566 | 61.765 | 47.857 | 27.833 | 60.857 | 48.143 | 27.414 |
| 125 | 46.667 | 46.667 | 61.912 | 48.000 | 27.833 | 61.000 | 48.286 | 27.414 |

这些结果来自同一 TRAIN holdout 和同一冻结表征。class 0 参与训练，holdout 缺少该类。C2 基座的 Many 参数行固定时，其他类别的 logits 仍可改变 Many top-1。单一 TPR 条件给出整套方法的联合结果，无法分解各正则的独立贡献。

[固定轮次对照](analysis/TPR_RESULTS.md) · [完整配对差值](analysis/paired_deltas.csv) · [逐类结果](analysis/per_class.csv) · [损失与梯度曲线](analysis/training_loss_gradient_curves.svg)

## 归档范围

最终 A/B checkpoint、完整 C1 配对轨迹、逐批诊断和终端 transcript 均登记为 local_only。清单的 sha256 栏采用执行机器上的原字节。Git 对源码换行规范化时，github_blob_sha256 栏记录仓库字节的哈希。

[输入与源码 SHA256](preflight_report.json) · [正式完成记录](formal_completed.json) · [TEST guard](formal_test_guard.json) · [产物 SHA256 清单](ARTIFACT_SHA256.csv)
