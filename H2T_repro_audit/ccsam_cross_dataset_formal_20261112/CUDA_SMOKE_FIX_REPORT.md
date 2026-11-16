# CC-SAM 跨数据集 CUDA Smoke 修复报告

## 执行结论

**CUDA 初始化与显存统计已在 RTX 5070 Laptop 实测通过。真实数据 Smoke 仍失败，正式训练准入未通过。**

| 检查 | 实测结果 |
| --- | --- |
| Python 语法 | 16 个文件通过，退出码 `0` |
| CPU 单元测试 | 19/19 通过，退出码 `0` |
| ImageNet-LT Preflight | 115,846 条训练记录通过 |
| Places-LT Preflight | 62,500 条训练记录通过 |
| 独立 CUDA 自检 | 通过，退出码 `0` |
| ImageNet-LT GPU Smoke | seed 123，第 1 批反向后的梯度检查失败，完成批次 0/2 |
| Places-LT GPU Smoke | ImageNet-LT 失败后停止，完成批次 0/2 |
| 正式实验 | Stage I 0/6，Stage II 0/18，官方 TEST 数据加载 0 次 |

## 首轮错误和完整调用链

首轮 `RuntimeError: Invalid device argument` 出现在原版 `experiment_common.py` 第 200 行。`run_cross_dataset.py` 的 `smoke` 命令经 `run_smoke()` 进入 `record_run()`，随后调用 `torch.cuda.reset_peak_memory_stats(0)`。安装环境的 PyTorch `torch/cuda/memory.py` 第 397 行将该调用直接交给 CUDA 分配器。原入口的 `is_available()` 仅确认 CUDA 可用性，创建 `torch.device('cuda', 0)` 也不会初始化 PyTorch CUDA 分配器。重置发生在分配器初始化前，首轮因此在模型和数据迭代之前退出。首轮已显式传入 `0`，本轮也实测该编号有效。[首轮 traceback](validation/smoke_console.log)

已检查整个实验目录中的设备和统计调用。模块导入阶段仅创建设备描述对象。预检环境记录查询设备元数据，独立 CUDA 自检、真实 Smoke、Stage I、特征提取、Stage II 和终评共用设备与显存统计实现。所有峰值重置和读取均传入明确编号。

## 修复范围

- `cuda_statistics.py` 集中执行 CUDA 可用性检查、显式初始化、设备数量和编号校验、`set_device(index)`、`synchronize(index)` 及峰值重置。设备编号缺失时读取已初始化的当前设备，再显式选定编号。峰值分配和保留量使用同一编号。异常保留原始 traceback 并附加设备诊断。
- `experiment_common.py` 用实际训练设备绑定每次运行的统计，先保存运行状态，再初始化和重置。`cross_dataset_smoke.py`、`train_stage1.py`、`extract_features.py`、`train_stage2.py` 和 `evaluate_final.py` 将实际设备传入记录器。
- `run_cross_dataset.py` 增加独立 `cuda-self-check` 命令并验证其报告。`experiment_common.py`、各阶段入口及 `prepare_experiment.py`、`analyze_results.py` 支持按运行名称隔离产物。本次在新目录运行，原始预检、Smoke 和报告留在原路径。`README.md` 记录执行顺序，`.gitignore` 忽略运行目录。
- `test_pipeline.py` 扩充设备编号缺失、越界、初始化顺序、统计设备一致性及失败记录的 CPU 回归。D0/D1/D2 目标函数和冻结协议文件通过原始哈希比对，两个数据集指纹和六份官方采样计划与首轮一致。

## 真机证据

- 环境为 `D:\app\minicondaaa\envs\xiaoju\python.exe`，Python 3.12.12，PyTorch 2.14.1+cu132，torchvision 0.29.1+cu132，CUDA 13.2，cuDNN 报告值 92400。NVIDIA 驱动为 591.74。PyTorch 识别出 1 个可见设备，当前与实际设备均为 `cuda:0`，GPU 为 NVIDIA GeForce RTX 5070 Laptop GPU。`CUDA_VISIBLE_DEVICES` 未设置。
- CUDA 自检在 `cuda:0` 上依次分配 1,024 和 2,048 字节的张量，在两次分配之间同步并重置峰值，随后同步并读取到 3,072 字节峰值分配量和 2,097,152 字节峰值保留量。自检只使用这两个张量。[CUDA 自检 JSON](runs/cuda_smoke_fix_20261112_01/cuda_self_check/cuda_statistics.json)
- 真实 Smoke 成功初始化 CUDA 并进入 ImageNet-LT 的第一批训练图像。前向特征、logits、损失的有限性检查通过，随后完成反向与 `unscale_`。原有梯度检查在 `train_stage1.py:75` 抛出 `Missing or nonfinite Stage-I Smoke gradient at AMP scale 65536.0`，早于 optimizer step。记录器在失败后从同一设备读出峰值分配量 974,838,272 字节、峰值保留量 1,178,599,424 字节。[真实 Smoke 日志](runs/cuda_smoke_fix_20261112_01/validation/smoke_console.log)
- 现有错误合并了梯度缺失与非有限两种情况。具体失败参数、NaN 或 Inf 类型以及是否为 AMP 溢出仍未确定。Places-LT 的 GPU 前向与反向、特征梯度等价性、D0/D1/D2 更新、D2 teacher 和 Frozen Encoder/BN 验收均未执行。[执行报告](runs/cuda_smoke_fix_20261112_01/FORMAL_EXECUTION_REPORT.md)

## Git 与产物保护

- 分支为 `experiment/ccsam-cross-dataset-formal-20261112`，HEAD 为 `a3977ae07f3fd33c1cac9673909a300792818c7f`。原有工作区修改均保留，暂存区为空。本次修复只涉及实验目录，目录外 Git 状态与开始时一致。
- 首轮保存的 22 项产物按原 SHA256 索引全部复核通过。新尝试使用 `runs/cuda_smoke_fix_20261112_01`，两轮日志分别保留。Smoke 失败后按协议停止。
