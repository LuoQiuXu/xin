# Fish16 Clean Launch执行报告（2026-11-03）

**结论：STOPPED_CPU_IMPORT_GATE / FAIL_VALIDATION，未启动GPU Smoke。** 本轮未修改任何源码、配置、数据、权重、manifest或任务说明，未修补导入环境后重试。

## 执行身份与指纹

分支：`experiment/fish16-clean-launch-20261103`。HEAD：`9f1a7ac8373f087274b6f919b048c45a14b46e2e`。
开始和结束Git工作区均干净，本地与抓取时远端一致。执行前后22个项目/上游Python文件SHA256一致；8个原始H2T/PI-H2T文件与既有锁定指纹一致。完整清单见EXECUTION_IDENTITY_BEFORE.json和SOURCE_FINGERPRINTS_AFTER.json。
环境：`{"python_executable": "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe", "python": "3.12.12", "torch": "2.14.1+cu132", "cuda_build": "13.2", "cudnn": 92400, "gpu_inventory_command": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], "gpu_inventory": "NVIDIA GeForce RTX 5070 Laptop GPU, 591.74, 8151 MiB", "cuda_initialized_after_environment_query": true, "environment_query_is_not_model_forward_backward_or_training": true}`。这里只查询软件版本和GPU清单，没有调用模型前向、反向或训练；环境查询可能初始化CUDA上下文，不应将其计作GPU Smoke。

## 规定检查的真实结果

| 检查 | 结果 | 退出码 |
|---|---|---:|
| 规定11文件py_compile | PASS 11/11 | 0 |
| test_amp_recovery | 12通过 | 包含于整套命令 |
| test_contract | 10通过 | 包含于整套命令 |
| test_formal_train | 14通过 | 包含于整套命令 |
| diagnostics.test_diagnostic_stats | 模块导入失败；5方法未加载 | 整套命令1 |

unittest报告Ran37条记录：36个真实测试方法通过，另1条为加载器生成的失败占位记录（errors=1）。不能表述为37个真实用例完整执行；5个诊断方法尚未获得断言结果。

精确执行命令和工作目录：

```text
D:\app\minicondaaa\envs\xiaoju\python.exe -B -X utf8 -m py_compile D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\protocol_config.py D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\preflight.py D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\data.py D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\models.py D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\amp_recovery.py D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\smoke.py D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\formal_train.py D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\test_amp_recovery.py D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\test_contract.py D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\test_formal_train.py D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t\diagnostics\test_diagnostic_stats.py
cwd=D:\xiangmuxunlian\xin
exit_code=0
```

```text
D:\app\minicondaaa\envs\xiaoju\python.exe -B -X utf8 -m unittest test_amp_recovery test_contract test_formal_train diagnostics.test_diagnostic_stats -v
cwd=D:\xiangmuxunlian\xin\H2T_repro_audit\cross_domain_failure_boundary\fish16_h2t_pih2t
exit_code=1
```

## 首个实际失败

`diagnostics/test_diagnostic_stats.py:4`使用顶层导入：

```python
from diagnose_b0 import tensor_stats, gradient_records, gradient_groups, cpu_tree
```

本次采用的包式unittest命令未将diagnostics目录作为顶层模块搜索目录，无法解析diagnose_b0并触发ModuleNotFoundError。CPU_ALL_SUITES.log和FAILURE.json保存完整traceback。该结果说明记录下的启动上下文导入失败，不能据此判断5个诊断方法的断言结果、模型算法错误或FP32数值稳定性。

按任务“任一门禁失败即停止，不得修补后重跑”，未修改源码、增补PYTHONPATH、换命令重试，也未调用preflight或smoke。合法测试启动上下文及后续完整预检门禁仍待处理。

## 未执行阶段与六组状态

完整数据/权重/原始代码preflight：NOT_RUN。CPU契约包含权重身份和严格兼容性通过，但不等价于完整preflight；全量图像SHA、split/no-leakage及路径检查没有本轮preflight结果。不得沿用旧结果或记为PASS。

| 条件 | 状态 | 完成Smoke Epoch | 尝试窗口 | 真实optimizer更新 | Loss / Validation / VRAM |
|---|---|---:|---:|---:|---|
| A0 | NOT_STARTED | 0/1 | 0 | 0 | 未测量 |
| B0 | NOT_STARTED | 0/1 | 0 | 0 | 未测量 |
| A1 | NOT_STARTED | 0/1 | 0 | 0 | 未测量 |
| A2 | NOT_STARTED | 0/1 | 0 | 0 | 未测量 |
| B1 | NOT_STARTED | 0/1 | 0 | 0 | 未测量 |
| B2 | NOT_STARTED | 0/1 | 0 | 0 | 未测量 |

GPU配对control hash和Stage-II BN/extractor冻结验证均NOT_RUN；训练loss、validation和峰值VRAM为null/未测量，GPU训练wall time为0。没有训练失败snapshot，因为GPU根本未启动；CPU导入traceback已完整保存。

任务指定的preflight和smoke目录均未创建，避免产生貌似运行过的结果。本报告位于独立的新validation目录。未启动formal90+10、seed43/44或额外诊断实验。未执行preflight的TEST字节hash路径，也未进行TEST模型评价；CPU真实视图用例使用训练recipient/donor。

## 初始源码读取的额外备注（未执行验证）

已读源码还显示：smoke manifest仍写amp=True且未显式记录NUMERIC_MODE，实际autocast上下文已禁用；preflight仍包含全manifest图像字节hash及旧ResNet50_FE/原factory探针。仅将这些已观察内容保存在INITIAL_SOURCE_REVIEW_NOTES.json，不将其当作执行过的预检结果，也不在CPU门禁失败后继续运行来探查。

本轮停止，无源码修补、CPU重跑、GPU Smoke或正式训练。原始日志、前后指纹、环境、结构化结果、六组零启动状态及本报告均列入OUTPUT_SHA256.csv。
