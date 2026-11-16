# CC-SAM Stage-I AMP 梯度诊断与验收

## 结论

ImageNet-LT 的首批故障由 AMP 缩放反向中 `encoder.conv1.weight` 的 Inf 引起。两次同批对照中，FP32 的 161 个可训练参数张量梯度全部有限。AMP 的首轮出现 1 个非有限梯度张量、2,630 个 Inf、0 个 NaN、0 个缺失梯度。GradScaler 自动降到 `8192` 后完成有效更新。ImageNet-LT 与 Places-LT 的 GPU Smoke 均通过，正式训练准入通过。

## 同批次证据

- 使用 ImageNet-LT seed `123` 的第一批官方 TRAIN 图像，16 张，`224×224`，4 个加载 worker。两次对照和修复后的真实 ImageNet-LT Smoke 首批的输入、标签、MixUp 后的输入及初始模型状态哈希一致。MixUp lambda 为 `0.9268907886795259`，原始标签和 permutation 已保存。两组诊断从同一初始 state_dict 的独立副本开始。[批次身份](amp_gradient_probe/imagenet_lt_seed123_acceptance_probe_01/batch_identity.json)
- `Missing gradient` 指实际可训练参数的 `grad is None`。`Nonfinite gradient` 指已生成的梯度包含 NaN 或 Inf。逐参数检查来自 `named_parameters()`，并核对优化器成员资格。BN 缓冲区只用于复原重试前状态。
- 唯一异常参数是 `encoder.conv1.weight`，形状 `[64,3,7,7]`，参数与梯度均为 `torch.float32`，`requires_grad=True`，属于优化器。`65536` 下解缩放前后均有 2,630 个 Inf。前向输入、特征、logits 和损失有限。FP32 loss 为 `7.293352127075195`，AMP loss 为 `7.289465427398682`，首次 scaled loss 为 `477722.40625`。[FP32 逐参数记录](amp_gradient_probe/imagenet_lt_seed123_acceptance_probe_01/fp32.json) · [首次 AMP 逐参数记录](amp_gradient_probe/imagenet_lt_seed123_acceptance_probe_01/amp_attempt_00.json)

| 模式 | Scale 变化 | 缺失梯度张量 | 非有限梯度张量 | NaN 元素 | Inf 元素 | SGD 更新 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| FP32 | 不适用 | 0 | 0 | 0 | 0 | 对照只执行前后向 |
| AMP 首轮 | 65,536 → 32,768 | 0 | 1 | 0 | 2,630 | 跳过 |
| AMP 重试 1 | 32,768 → 16,384 | 0 | 1 | 0 | 686 | 跳过 |
| AMP 重试 2 | 16,384 → 8,192 | 0 | 1 | 0 | 12 | 跳过 |
| AMP 重试 3 | 8,192 → 8,192 | 0 | 0 | 0 | 0 | 实际执行 |

- 三次跳步均完成 `scaler.update()`，优化器 step hook 均未触发，0 个参数改变。成功重试的 hook 触发一次，161 个参数张量共 25,542,567 个元素发生变化，最大绝对参数差 `0.009492173790931702`，更新后参数全部有限。两次对照合计 6 次诊断重试。[原始对照](amp_gradient_probe/imagenet_lt_seed123_probe_01/summary.json) · [验收对照](amp_gradient_probe/imagenet_lt_seed123_acceptance_probe_01/summary.json) · [恢复后的逐参数记录](amp_gradient_probe/imagenet_lt_seed123_acceptance_probe_01/amp_attempt_03.json)

## 修复与重新验收

- Smoke 对缺失梯度单独报错。梯度含 NaN/Inf 时，先核实 GradScaler 跳过更新、参数保持原值且 scale 下降，再用同一批输入、同一 MixUp 和批前 BN 缓冲区状态重试。每批至多 8 次重试。最终通过要求解缩放梯度有限、真实 step 和可测的有限参数更新。正式 Stage-I 继续使用原有 CUDA AMP、默认 GradScaler 初始 scale、每批一次 step/update 和既定训练预算。正式日志以优化器 hook 计数。
- 18 个 Python 文件通过语法检查，CPU 测试 22/22 通过。两套 Preflight 分别核验 115,846 条和 62,500 条训练记录，六份采样计划 SHA256 与上一轮一致。CUDA 独立自检在 RTX 5070 Laptop `cuda:0` 上通过，PyTorch 2.14.1+cu132，CUDA 13.2。[CPU 测试日志](../runs/amp_gradient_fix_20261112_01/validation/cpu_tests.log) · [Preflight](../runs/amp_gradient_fix_20261112_01/preflight/status.json) · [CUDA 自检](../runs/amp_gradient_fix_20261112_01/cuda_self_check/cuda_statistics.json)

| 数据集 | Stage-I 有效更新批次 | 首批各次 Inf 元素 | 两批实际更新张量数 | 实际特征梯度等价性最大绝对误差 | D0/D1/D2、teacher、Encoder/BN |
| --- | ---: | --- | --- | ---: | --- |
| ImageNet-LT | 2/2 | 2,630 → 686 → 12 → 0 | 161 / 161 | 0 | 通过 |
| Places-LT | 2/2 | 2,456 → 789 → 183 → 0 | 161 / 161 | 0 | 通过 |

两套首批均在 `8192` 恢复，第二批一次完成。两个数据集的所有初始 Inf 均位于 `encoder.conv1.weight`，两次有效更新的梯度均有限。D2 teacher 保持 eval、冻结且无梯度，KL 覆盖对应数据集的全部输出类别。真实与合成特征的等价性使用原 `rtol=1e-5`、`atol=1e-7`。验收阶段官方 TEST 数据访问 0 次，正式 Stage-I 和 Stage-II checkpoint 生成 0 个。[ImageNet-LT Smoke](../runs/amp_gradient_fix_20261112_01/smoke/imagenet_lt/status.json) · [Places-LT Smoke](../runs/amp_gradient_fix_20261112_01/smoke/places_lt/status.json) · [真实 Smoke 日志](../runs/amp_gradient_fix_20261112_01/validation/smoke_console.log)

## 正式实验启动

两套 Smoke 通过后已启动 `run-formal --all`。正式 runner PID 35268，输出目录为 `runs/amp_gradient_fix_20261112_01`。截至 2026-11-12 21:36:09（北京时间），ImageNet-LT seed 123 已记录 epoch 1 的第 500/7,241 批、497 次实际 SGD 更新和 3 次 AMP 跳步。Stage I 完成 0/6，Stage II 完成 0/18，官方 TEST 数据访问 0 次。终评入口由全部 Stage-II 完成状态和校验和放行。[正式实时日志](../runs/amp_gradient_fix_20261112_01/validation/run-formal_console.log) · [正式阶段状态](../runs/amp_gradient_fix_20261112_01/formal_execution/status.json)

首个后台启动器在创建正式运行状态前退出，其启动记录已保留。随后启动的唯一正式 runner 已写入上述实际批次进度。

## 修改文件与保护核验

- `train_stage1.py` 增加同批 AMP 重试和实际 SGD step 计数。`cross_dataset_smoke.py` 写入逐参数证据并在准入时校验。`experiment_common.py` 将共享证据代码纳入运行源码 SHA256。
- `test_pipeline.py` 增加 Inf 恢复、缺失梯度立即停止和重试预算耗尽三项回归测试。`.gitignore` 忽略诊断运行产生的本地 CUDA 和编译缓存。
- `diagnostics/amp_gradient_probe/gradient_evidence.py` 与 `diagnostics/amp_gradient_probe/probe_stage1.py` 提供逐参数统计、FP32/AMP 对照和可核验的实际参数差异。
- 当前分支为 `experiment/ccsam-cross-dataset-formal-20261112`，HEAD 为 `a3977ae07f3fd33c1cac9673909a300792818c7f`。已有修改仍在工作区，暂存区为空。两份历史 SHA256 索引的 22 项和 29 项均复核通过。CUDA 修复、冻结协议、采样策略和 D0/D1/D2 目标函数通过原始哈希比对。
