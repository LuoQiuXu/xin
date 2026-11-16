# CC-SAM 跨数据集实验运行记录

**训练已完成。官方测试评估因 ImageNet-LT 的测试清单一致性检查失败而中止。**

运行于北京时间 2026-11-12 21:34:51 启动，2026-11-16 12:04:28 停止。实验从 `a3977ae07f3fd33c1cac9673909a300792818c7f` 上的工作区源码运行。提交本次产物时核对了 19 个运行源文件的原始 SHA256，可在运行配置中查看。[运行配置](../runs/amp_gradient_fix_20261112_01/formal_execution/config.json)

| 数据集 | Stage I | 训练特征缓存 | Stage II |
| --- | ---: | ---: | ---: |
| ImageNet-LT | 3/3（各 90 轮） | 3/3（115,846 × 2,048） | 9/9（各 30 轮） |
| Places-LT | 3/3（各 90 轮） | 3/3（62,500 × 2,048） | 9/9（各 30 轮） |

- 三个种子均为 `123`、`124`、`125`。Stage II 的 D0、D1、D2 使用同种子的 Stage I checkpoint、初始分类器和训练特征。18 份逐轮采样记录共 540 轮，均与所属计划的索引哈希一致。每个 D2 的 teacher 均与同数据集、同种子的 D0 checkpoint 匹配。
- 运行门禁包括 22/22 项 CPU 测试、两套训练集预检、独立 CUDA 自检、两套真实 GPU Smoke。AMP 梯度溢出及 GradScaler 恢复证据已归档。[AMP 梯度诊断](../diagnostics/AMP_GRADIENT_DIAGNOSIS_REPORT.md) · [CUDA 修复记录](../CUDA_SMOKE_FIX_REPORT.md)

## 终评状态

训练及 checksum 门禁在 2026-11-16 12:04:25 核验了 6 个 Stage I 和 18 个 Stage II。ImageNet-LT 终评随后开始读取官方测试清单和 Parquet 的文件名及标签，并在建立测试数据集时失败。日志给出的首个缺失键为 `('ILSVRC2012_val_00029930.JPEG', 0)`。目前仅证实文件名与类别 ID 的组合未按清单匹配，差异来源仍未确定。ImageNet-LT 完成 0 条预测，Places-LT 终评尚未开始。两套测试指标、逐类准确率和配对差值均未生成。[终评状态](../runs/amp_gradient_fix_20261112_01/final_evaluation/status.json) · [失败 traceback](../runs/amp_gradient_fix_20261112_01/validation/run-formal_stderr.log)

## 可核验产物

- [正式执行报告和 24 个 checkpoint 的 SHA256](../runs/amp_gradient_fix_20261112_01/FORMAL_EXECUTION_REPORT.md) · [正式运行日志](../runs/amp_gradient_fix_20261112_01/validation/run-formal_console.log)
- [Stage I 逐轮训练指标](../runs/amp_gradient_fix_20261112_01/stage1) · [六份特征缓存状态](../runs/amp_gradient_fix_20261112_01/features) · [Stage II 逐轮训练指标和采样轨迹](../runs/amp_gradient_fix_20261112_01/stage2)
- [产物 SHA256 清单](ARTIFACT_SHA256.csv) 的路径相对于实验目录。`published=true` 的条目是本次收录的证据文件。`published=false` 的条目可用本地原件复核，包括权重、特征数组和采样计划。
