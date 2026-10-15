# H2T Stage I 正式训练结果

## 完成状态

- 完成 **200/200 epochs**，执行 17,000 个 SGD step，累计训练样本访问 2,169,400 次；未加载旧 Checkpoint。
- 用户授权范围为 Stage I。本轮没有启动任何 Stage II 正式训练，也没有改变 H2T 融合策略、供体/通道规则或 rho。
- 运行 ID：`formal_20261011_1429_r2`；开始：`2026-10-11T14:28:34+0800`；结束：`2026-10-11T14:55:08+0800`；训练及每 epoch 评估耗时 26.57 分钟（不含之后的独立模型复核评估）。

## 最佳与最终模型

| Checkpoint | Epoch（从 1 开始） | Overall | Head | Medium | Tail |
|---|---:|---:|---:|---:|---:|
| 最佳 model_best.pth.tar | 198 | 40.490% | 68.806% | 40.114% | 5.793% |
| 最终 model_final.pth.tar | 200 | 40.090% | 68.306% | 39.400% | 5.897% |

最佳 Epoch 为 **198**；原日志 `Best epoch` 从 0 开始，对应值为 **197**。每 epoch 以官方 CIFAR-100 test split 的 Overall 监控最佳模型；它不是独立 validation split 上的选择。正式保存的最佳和最终模型均经过原 `eval-modifed.py` 独立重新加载、全量评估，结果与对应训练 epoch 一致。

评估共 10,000 张图像，每类 100 张。Head [0,36)、Medium [36,71)、Tail [71,100) 为官方 YAML 的零基半开区间；各组按逐类准确率平均，Overall 按全部样本计算。没有改用按样本数阈值划分的其他 Head/Medium/Tail 定义。评估不使用 H2T 或 MixUp，且 BN buffers 不变。

- [最佳 Checkpoint](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428/ckps/model_best.pth.tar)
- [最终 Checkpoint](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428/ckps/model_final.pth.tar)
- [最佳模型逐类结果](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/best_evaluation.json)
- [最终模型逐类结果](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/final_evaluation.json)

## 配置与算法一致性

使用已审计独立兼容副本 `H2T_repro_audit/runtime`。官方归档 Commit 为 `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`；该值来自本地官方 ZIP comment，网络限制使此前未能以 Git 对象/远程 HEAD 独立核验。48 个官方归档文件及兼容副本的逐文件 SHA256 已留存，本轮运行前后核对兼容副本和运行源码快照，均通过。

| 项目 | 本轮实际设置 |
|---|---|
| 原 YAML | cifar100_imb001_stage1_ce_mixup.yaml，字节未修改 |
| 数据 / Backbone | CIFAR-100-LT IF=100 / resnet32_fe（ResNet-32） |
| epochs / batch / workers | 200 / 128 / 0 |
| Loss | 原 CrossEntropyLoss，无类别重权重 |
| MixUp | True，alpha=1.0，原 Beta 采样和双标签加权 CE |
| SGD | lr=0.1，momentum=0.9，weight_decay=2e-4，原默认 nesterov=False |
| LR | 5 epoch warmup；第 1–5 epoch 为 0.02/0.04/0.06/0.08/0.1；6–160 为 0.1；161–180 为 0.01；181–200 为 0.001 |
| 随机设置 | 官方 deterministic=True，SetRandomSeed(123)；长尾子集构造 NumPy seed=0 |
| GPU 包装 | YAML gpu=null，原 DataParallel CUDA:0 路径 |
| Backbone / classifier | 两者都进入 Stage I SGD，PIF/feature_tuning=False |
| resume | 显式空字符串；训练期间 checkpoint load 调用为 0 |
| CLI 覆盖 | 仅绝对 data_path、空 resume、独立运行 name |

没有新增或修改核心训练代码。本轮新增外部运行记录脚本，调用原 Stage I 的 train / validate / scheduler / save_checkpoint，记录 Loss、指标、显存和输出位置；未改动采样、梯度、优化器、随机数流、前向结果或 epoch 数。额外保存的 model_final.pth.tar 是官方 current.pth.tar 的字节复制。

独立副本沿用原审计的五个兼容文件修改：ResNet_fe 构造参数错误修补、pathlib2→标准库 pathlib，以及训练/评估 torch.load 显式 weights_only=False。兼容 Diff 见 [compatibility.diff](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/compatibility.diff)。

首次外部启动器在参数解析阶段因 `sys.argv` 多包含 Python 的 `-u` 参数而退出；当时尚未创建模型/训练目录，也未执行训练。已仅修复外部启动器，并使用新的运行 ID 从头启动；失败记录保存在 `formal_runs/formal_20261011_1429`，未与本轮结果合并。

本轮原 Stage I 主函数完整完成 200 epochs 后，外部归档核验因官方 save_code 默认忽略 PDF/PPTX，而外部脚本要求核验全部归档文件，触发 FileNotFoundError，启动器因此退出码为 1。随后仅补齐 3 个非代码演示文档、复制 epoch-200 current 为 final，重新核验全部源码、Checkpoint 和指标；没有重跑训练、没有修改模型 tensor。原异常状态保存在 status_before_finalize_recovery.json，恢复记录保存在 status.json 的 postprocessing_recovery。完成的训练结果有效，外部记录错误已修复。实际执行的启动器快照保持原字节；后续修正版本另存为 formal_stage1_after_run_fix.py，未冒充本次执行源码。

## 环境、数据与显存

- Python：`D:\app\anaconda\envs\pytorch\python.exe`，版本 3.12.13。
- 系统：Windows-11-10.0.26200-SP0；torch 2.11.0+cu130、torchvision 0.26.0+cu130、NumPy 2.3.1。
- GPU：NVIDIA GeForce RTX 5070 Laptop GPU，显存 8151 MiB；CUDA runtime 13.0，compute capability=[12, 0]；驱动见保存的 nvidia-smi。
- 数据路径：`D:\xiangmuxunlian\新\H2T\data\cifar100`。train/test/meta 完整性验证通过；训练 10,847 张、100 类、最大 500 / 最小 5、实际 IF=100；每 epoch 85 batch（末 batch=95）。测试 10,000 张，每类 100。
- PyTorch 峰值 allocated：**333.30 MiB**；reserved：**416.00 MiB**。
- nvidia-smi 每 5 秒采样的 GPU 全局占用最高：**1971 MiB**，包含桌面及其他 GPU 程序；不是本训练进程的独占显存。WDDM 下 nvidia-smi 不提供可用的逐进程显存数字。
- 所有 17,000 个 MixUp 训练 Loss 均为有限值，batch Loss 范围 0.310150–7.248970；最终 epoch 平均 Loss=1.769750。全部最佳/最终 Checkpoint 状态 tensor 有限。
- 最终每个 BN num_batches_tracked=17,000，符合 Stage I 每个训练 batch 一次前向；31 个 BN 均符合此计数。评估不增加计数。

## 运行命令

实际执行的记录运行器（内部调用下列官方入口，配置一致）：

```powershell
Set-Location -LiteralPath 'D:\xiangmuxunlian\新'
& 'D:\app\anaconda\envs\pytorch\python.exe' -u .\H2T_repro_audit\formal_stage1.py run --run-id formal_20261011_1429_r2
```

等价的官方入口配置命令（展示用；**本轮已经运行完成，不要重复执行**）：

```powershell
Set-Location -LiteralPath 'D:\xiangmuxunlian\新\H2T_repro_audit\runtime'
& 'D:\app\anaconda\envs\pytorch\python.exe' -u .\train_stage1.py --cfg .\config\cifar100\cifar100_imb001_stage1_ce_mixup.yaml data_path 'D:\xiangmuxunlian\新\H2T\data\cifar100' resume "''" name cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2
```

独立评估通过外部只读观察运行器调用原 `eval-modifed.py`；best/final 两次分别保存官方评估日志和机器可读结果，未执行任何训练。

## 保存清单与追溯

- [完整官方训练日志](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428/logs/cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2.txt)
- [完整 stdout/stderr](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/stdout_stderr.log)
- [原 YAML](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/original.yaml)、[实际完整配置](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/effective.yaml)
- [运行源码快照目录](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428/codes)、[逐文件源码 hash](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/runtime_manifest.json)
- [环境与数据预检](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/preflight.json)、[显存采样](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/gpu_samples.jsonl)
- [完整运行状态与逐 epoch 数据](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/status.json)、[epoch CSV](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/epochs.csv)
- [Checkpoint / LR / 源码验证](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/artifact_verification.json)
- [最佳模型独立评估日志](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/best_evaluation.official.log)、[最终模型独立评估日志](D:/xiangmuxunlian/新/H2T_repro_audit/formal_runs/formal_20261011_1429_r2/final_evaluation.official.log)
- 本轮外部运行、验证及报告脚本各自在证据目录保存了一份；源文件与输出文件 hash 见 artifact_manifest.json。

| 文件 | Checkpoint epoch | SHA256 |
|---|---:|---|
| model_best.pth.tar | 198 | `0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e` |
| current.pth.tar | 200 | `0bbc992651b358cd7ab7a89466dbfbbd405ad1f1c459461840e3bf1ffd259a55` |
| model_final.pth.tar | 200 | `0bbc992651b358cd7ab7a89466dbfbbd405ad1f1c459461840e3bf1ffd259a55` |

官方 Checkpoint 不保存 optimizer/RNG 状态，其 resume 分支也不能精确续训。本轮从头连续完成 200 epochs，不存在将不完整 resume 误报为完整训练的问题。

## 下一阶段方案与 BatchNorm 记录

[DR 与 DR+H2T 公平对照方案](D:/xiangmuxunlian/新/H2T_DR_COMPARISON_PLAN.md) 已给出：同一 Stage I Checkpoint / 相同初始分类器 / 相同 BS 标签 / 原 SGD 与 10 epoch 调度，DR 使用已有 `H2T 0.0` 开关，DR+H2T 保持官方 `H2T=0.25`。

官方 Stage II 的 shift_bn=True 会更新 Backbone BN 统计量。DR 每 batch 做 BS 一次前向，每个 BN 每 epoch +85；DR+H2T 做 BS→IS 两次前向，每个 BN每 epoch +170。Conv 和 BN affine 权重不更新，但 running_mean / running_var / num_batches_tracked 会变动。因此仓库默认对照的性能变化包含 BN 更新流程的影响。

方案另外列出了冻结 BN 统计量的配置对照，以及通过独立运行器匹配 BS→IS BN 更新的控制对照；均明确区分于官方默认流程，且本轮未实施。相同 seed 也不自动保证采样/增强相同，因为 H2T 的 CPU randperm 会额外消耗 RNG；严格配对方案对此作了记录。

**Stage I 已完成，Stage II 未启动，等待下一步确认。**
