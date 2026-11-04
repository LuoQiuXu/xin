# Fish16 Clean Launch修复后Smoke报告

**结论：PASS_SMOKE_ONLY。六组按A0、B0、A1、A2、B1、B2顺序，各完成一个真实Epoch和VAL评价，随后停止。**

执行日期：2026-11-04（Asia/Shanghai；目录名称保留任务规定的20261103）。
分支：`experiment/fish16-clean-launch-fix-20261103`。执行HEAD：`960b1be97058202ee97db88f97a424bf1e6c37fb`。初始工作区干净；启动前的两处修复见STARTUP_FIXES.md，执行后14个源文件及8个上游文件指纹保持一致。
Python按用户确认固定为`D:\app\minicondaaa\envs\xiaoju\python.exe`。环境：`{"python_executable": "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe", "python_version": "3.12.12", "torch": "2.14.1+cu132", "cuda_build": "13.2", "cudnn": 92400, "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU"}`。

## 门禁与执行次数

| 阶段 | 真实结果 | 退出码 |
|---|---|---:|
| STATIC_PROTOCOL | PASS，所有活动autocast显式禁用 | 0 |
| SYNTAX | 14/14文件 | 0 |
| CPU | AMP12 + contract10 + formal14 + diagnostic5 = 41/41 | 0 |
| Preflight | PASS_PREFLIGHT_ONLY | 0 |
| 六组GPU Smoke | COMPLETE_SMOKE_ONLY / 6组真实result | 0 |

CPU、预检及GPU Smoke各执行一次。未手动更换unittest导入上下文；本轮启动前的两处工具记录修复得到用户授权，未改算法或实验参数。

预检：manifest SHA=62d9a2e9fffaaca22ef0c99c0ce4134b4d0e1ecc308333a7cb57b10cf95f0f30；TRAIN16269/5208轨迹、VAL5396/1736轨迹、TEST5468/1736轨迹；原始TRAIN/VAL共21665张图像字节哈希逐一通过，跨split泄漏0。TEST5468行仅metadata，路径无resolve/stat/open/hash，图像字节读取0。
官方权重SHA：`19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`。
转换适配器SHA：`c1bc33a75ef4d4dd607aa1cb0635ed42d547265a41ad0467081276e5009a7b3e`。
转换来源、strict_load、仅去掉ImageNet fc、265个主干权重/浮点buffer精确保留的来源记录、PIF两参数及共同初始化均通过；8个上游SHA一致，无网络权重下载。

## 六组真实结果

accuracy单位为%，显存为PyTorch peak allocated MiB。用时列为训练循环耗时，排除后续VAL和checkpoint保存；全程wall time另列。

| 条件 | Epoch | 样本 | 尝试/真实更新 | 跳步 | Loss | VAL Overall | Macro | Worst | Tail图像 | Peak MiB | 训练秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 1/1 | 16269 | 1017/1017 | 0 | 1.014550 | 94.310600 | 69.464532 | 0.000000 | 23.000000 | 1631.490 | 228.520 |
| B0 | 1/1 | 16269 | 1017/1017 | 0 | 1.004769 | 97.590813 | 70.511323 | 0.000000 | 32.000000 | 1635.991 | 230.970 |
| A1 | 1/1 | 16269 | 1017/1017 | 0 | 0.388465 | 97.516678 | 93.433760 | 66.666672 | 88.000000 | 338.646 | 70.241 |
| A2 | 1/1 | 16269 | 1017/1017 | 0 | 0.489448 | 94.347672 | 91.824405 | 66.666672 | 82.000000 | 348.665 | 102.739 |
| B1 | 1/1 | 16269 | 1017/1017 | 0 | 0.314383 | 98.424759 | 93.937456 | 54.545456 | 92.000000 | 338.646 | 70.625 |
| B2 | 1/1 | 16269 | 1017/1017 | 0 | 1.730579 | 77.297997 | 89.504503 | 57.928120 | 89.000000 | 348.666 | 103.883 |

主六条件总样本97614，总尝试/真实optimizer更新6102/6102，有限梯度审计6102次；所有事件kind=optimizer_step，before/after真实计数逐一递增。NaN=0、Inf=0、缺失梯度门禁未触发；skip=0、Scale恒1、Scale转换0。内置四个一次性单batch检查单独计，不并入六个Epoch的更新统计。
全程父进程记录wall time=1099.072544秒。
六组均完成恰好5396张VAL评价；evaluate_validation每次完成均assert len(labels)==5396，result仅在评价成功返回后保存。16类准确率与6/5/5分组完整保存在每组result.json及SMOKE_SUMMARY.json。

## 配对与冻结

以下全部为真实控制序列逐项相等，完整序列及独立聚合SHA见PAIRING_CHECKS.json和各组input_controls.json。

- A0_B0_recipient_sha256: True
- A0_B0_mixup_rng_sha256: True
- A0_B0_mixup_lambda: True
- A1_A2_recipient_views: True
- B1_B2_recipient_views: True
- A1_B1_recipient_views: True
- A1_B2_recipient_views: True
- A2_B2_donor_views: True
- A1_A2_extractor_initial_sha256: True
- A1_A2_classifier_initial_sha256: True
- B1_B2_extractor_initial_sha256: True
- B1_B2_classifier_initial_sha256: True
- A1_extractor_matches_A0_final: True
- A2_extractor_matches_A0_final: True
- B1_extractor_matches_B0_final: True
- B2_extractor_matches_B0_final: True

四组Stage-II提取器完整state（参数和BN所有buffer）初末SHA均相同；分类头实际更新。A1/A2来自同一A0 Smoke checkpoint，B1/B2来自同一B0 Smoke checkpoint，并严格加载。所有训练recipient/donor索引都落在TRAIN；保存采样计划经metadata复核无TEST索引。

FP32：manifest明确amp=false、autocast_enabled=false、grad_scaler_enabled=false；四个单batch feature dtype均torch.float32。保持micro/effective16、accumulation1、workers4；Stage-I首Epoch LR0.002、Stage-II首Epoch LR0.01、momentum0.9、原weight decay与原90/10日程分母，未重缩放为1轮日程。A2保持H2T共享512通道；B2保持原PI-H2TF、ratio_detach=True及原bounds模式。

## 科学范围及记录细节

这些是one-Epoch Smoke的真实指标，不能替代90+10正式实验。B2本轮Overall低于B1的结果被如实保留，未据准确率调整公式、PIF、BN、融合比例、loss或LR；尚不能据此认定方法缺陷或跨域退化机制。

原始ACTUAL_EXECUTION_STATUS顶层为COMPLETE_SMOKE_ONLY，六条件完成列表及全部result、SMOKE_SUMMARY和exit0一致。辅助batch_smoke字段仍残留RUNNING；BATCH_SMOKE.json实际保存四个成功路径及FP32/分类头更新/提取器冻结证明。这是辅助状态未收尾的日志字段，原始记录按字节保留，不将其解释为仍有训练在运行，也没有为显示字段重跑GPU。

每组训练循环和完整评价成功的有限性来自真实门禁返回及6102条事件；没有额外模型推理来生成本报告。原始大checkpoint与npz保留外部，路径及SHA见CHECKPOINT_SHA256.csv/OUTPUT_SHA256.csv。

## 精确命令与输出


```text
cpu: D:\app\minicondaaa\envs\xiaoju\python.exe -B -X utf8 -u H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\validate_cpu.py --output-dir D:\h2t_cross_domain_results\fish16_clean_launch_cpu_20261103_02
cwd=D:\xiangmuxunlian\xin
exit_code=0
```

```text
preflight: D:\app\minicondaaa\envs\xiaoju\python.exe -B -X utf8 -u H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\preflight.py --output-dir D:\h2t_cross_domain_results\fish16_clean_launch_preflight_20261103_02
cwd=D:\xiangmuxunlian\xin
exit_code=0
```

```text
smoke: D:\app\minicondaaa\envs\xiaoju\python.exe -B -X utf8 -u H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\smoke.py --output-dir D:\h2t_cross_domain_results\fish16_clean_launch_smoke_20261103_02 --preflight-record D:\h2t_cross_domain_results\fish16_clean_launch_preflight_20261103_02\preflight.json --cpu-validation-record D:\h2t_cross_domain_results\fish16_clean_launch_cpu_20261103_02\CPU_VALIDATION.json --micro-batch 16
cwd=D:\xiangmuxunlian\xin
exit_code=0
```

CPU目录：`D:\h2t_cross_domain_results\fish16_clean_launch_cpu_20261103_02`。
预检目录：`D:\h2t_cross_domain_results\fish16_clean_launch_preflight_20261103_02`。
Smoke目录：`D:\h2t_cross_domain_results\fish16_clean_launch_smoke_20261103_02`。
归档内CPU/preflight/manifest/status/summary/result/events/controls等是原始文件的逐字节复制，路径与SHA对应ARCHIVE_PROVENANCE.json；额外的汇总/检查文件单独标明来源。OUTPUT_SHA256.csv的root字段区分外部CPU、preflight、Smoke与archive；不上传checkpoint、图像、数据集和大型张量。

**TEST图像字节读取=0，TEST模型评价=0；seed43/44=0；正式90+10未启动。所有Smoke已停止，等待结果审阅。**
