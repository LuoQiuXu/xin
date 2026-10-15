# H2T BN 机制审计

完成时间：2026-10-11 23:04:43 +0800；控制实验编号：`fixed_bn_20261011_2255`。

本次完成两组独立的固定 BN 控制实验，每组 10 epochs，并分别重新加载最佳及第 10 epoch Checkpoint，在全部 10,000 张 CIFAR-100 测试图像上评估。原始 DR / DR+H2T 运行、源码和结果保留。本报告属于机制控制实验，不替代官方 `shift_bn=True` 复现。

## 1. 结果

所有准确率单位为 %；差值单位为百分点（pp）。Head 为类别 [0,36)，Medium 为 [36,71)，Tail 为 [71,100)，三组为各类准确率的平均值。固定 epoch 10 为等训练时长的主要对照，最佳 epoch 为补充。

### 最佳 Checkpoint

| 条件 | 最佳 epoch | Overall | Head | Medium | Tail |
|---|---:|---:|---:|---:|---:|
| Stage I | 198 | 40.49 | 68.81 | 40.11 | 5.79 |
| 原始 DR | 7 | 47.10 | 65.28 | 47.40 | 24.17 |
| 原始 DR+H2T | 9 | 48.06 | 61.81 | 50.66 | 27.86 |
| 固定 BN DR | 7 | 46.55 | 64.11 | 47.11 | 24.07 |
| 固定 BN DR+H2T | 8 | 47.30 | 61.89 | 49.69 | 26.31 |

### 固定第 10 epoch

| 条件 | epoch | Overall | Head | Medium | Tail |
|---|---:|---:|---:|---:|---:|
| 原始 DR | 10 | 47.00 | 65.28 | 47.26 | 24.00 |
| 原始 DR+H2T | 10 | 47.87 | 61.83 | 50.60 | 27.24 |
| 固定 BN DR | 10 | 46.49 | 64.06 | 47.31 | 23.69 |
| 固定 BN DR+H2T | 10 | 47.25 | 61.81 | 49.77 | 26.14 |

### H2T 相对 DR 的增益

| 条件 / 选择规则 | ΔOverall | ΔHead | ΔMedium | ΔTail |
|---|---:|---:|---:|---:|
| 原始 BN 更新 / 最佳 | +0.96 | -3.47 | +3.26 | +3.69 |
| 原始 BN 更新 / epoch 10 | +0.87 | -3.44 | +3.34 | +3.24 |
| 固定 BN + 配对输入 / 最佳 | +0.75 | -2.22 | +2.57 | +2.24 |
| 固定 BN + 配对输入 / epoch 10 | +0.76 | -2.25 | +2.46 | +2.45 |
| 固定条件增益 − 原始增益 / epoch 10 | -0.11 | +1.19 | -0.89 | -0.79 |

### 固定 BN 结果相对 Stage I 的变化

| 固定 BN 条件 / 选择规则 | ΔOverall | ΔHead | ΔMedium | ΔTail |
|---|---:|---:|---:|---:|
| DR / 最佳 | +6.06 | -4.69 | +7.00 | +18.28 |
| DR / epoch 10 | +6.00 | -4.75 | +7.20 | +17.90 |
| DR+H2T / 最佳 | +6.81 | -6.92 | +9.57 | +20.52 |
| DR+H2T / epoch 10 | +6.76 | -7.00 | +9.66 | +20.34 |

## 2. 锁定配置与隔离方式

- Stage I：epoch 198，Overall 40.49%；[锁定 Checkpoint](<D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428/ckps/model_best.pth.tar>)。
- SHA-256：`0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e`。两组各自新进程只加载该文件一次，初始化 Backbone 和分类器与其一致。
- CIFAR-100-LT IF=100，ResNet-32，真实训练子集 10,847 张，100 类，每类 500→5 张；每个 epoch 85 batches（128×84+95），workers=0，10 epochs；测试每类 100 张。
- SGD 仅包含分类器 2 个可学习张量，共 6,500 参数；momentum=0.9、weight_decay=0.0005、nesterov=False。CE 不重加权；Stage II 不使用 MixUp。
- 保留官方 cosine 调度：配置 lr=0.1、lr_factor=0.2，实际 epoch 1 lr=0.02，epoch 10 lr≈0.0004894348373。
- 两组使用原始 YAML，命令行共同覆盖数据路径、epoch-198 resume、独立名称和 `shift_bn=False`。DR 额外设置 `H2T=0.0`；H2T 保留 `H2T=0.25`。
- 独立运行目录：[固定 BN 控制实验目录](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255>)；其中 `runtime/` 只复制已审计的 48 个源码文件，不复制原始 saved 目录。与已审计副本的 SHA-256 全部一致；核心源码新增修改为 0。
- 原始 commit：`e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`（沿用原审计中官方 archive 的 commit 标记）；Windows 兼容补丁沿用已审计的五文件补丁，未新增算法补丁。
- 使用代码已有的 `shift_bn=False` 分支，Backbone `eval()`，分类器 `train()`；保留原始 `torch.no_grad()` 及仅分类器 SGD。Conv / BN affine 的 `requires_grad` 标志保持原始值，但无梯度、未进入优化器，参数逐位不变。
- DR 保留原始 dual_sample 数据准备：依然先准备 IS 供体和增强，再准备 BS 接收者；`H2T=0` 跳过供体 GPU forward 和融合。这样两组数据 RNG 的调用路径相同，未改变 BS/IS 采样规则。

### H2T 函数及 RNG 控制

H2T 仍调用原始 `methods.H2T`：`randperm(64)` 取前 `int(0.25×64)=16` 通道，在接收者 [B,64,8,8] 上原地复制供体同位置通道；保留 48 通道，整个 batch 共用掩码。BS 标签用于 CE，IS 标签不参与融合后的监督；无供体或通道筛选。

唯一额外执行控制为外部 observer 的 CPU RNG 隔离：独立 CPU Generator 种子 123；用 `torch.random.fork_rng(devices=[])` 暂存环境 CPU RNG，将独立状态交给原始函数，保存推进后的通道状态，再恢复环境状态。每次调用均核验原始随机排列的 RNG 消耗、16/48 通道替换关系、供体不变和环境 CPU RNG 逐位不变。抽样分布和融合公式保持不变，具体掩码序列与历史官方复现不同，因此明确列为新增控制条件。

两组 Python、NumPy、torch CPU、CUDA 与 PYTHONHASHSEED 均为 123；保留 YAML `deterministic=False`，未启用额外确定性算法。后端实际设置：`{"cudnn_deterministic": false, "cudnn_benchmark": false, "deterministic_algorithms": false}`。

### 全训练数据配对实证

| 项目 | 相同 batch / 总 batch |
|---|---:|
| BS 标签 | 850 / 850 |
| BS 增强后图像（完整 float tensor 字节哈希） | 850 / 850 |
| IS 标签 | 850 / 850 |
| IS 增强后图像（完整 float tensor 字节哈希） | 850 / 850 |
| batch 就绪时 torch CPU RNG 状态 | 850 / 850 |

两组每个 epoch 的开始和结束，Python / NumPy / torch CPU / CUDA RNG 状态均相同（各 10/10）。这里的“输入一致”经全部 batch 哈希确认，不仅依据相同 seed。BS 样本身份以图像与标签 tensor 的完整一致性核验；未改变 dataset 返回值加入索引。验证证据：[paired_inputs_verification.json](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/paired_inputs_verification.json>)；逐 batch 哈希和全部 H2T 掩码存于各组 status.json。

## 3. 参数与 BN 的逐位核验

| 状态项 | 原始 DR / 原始 H2T（epoch 10） | 固定 BN DR / 固定 BN H2T（最佳及 epoch 10） |
|---|---|---|
| Backbone 可学习张量 | 均 0/93 改变 | 均 0/93 改变，原始字节一致 |
| BN running_mean | 均 31/31 改变 | 均 0/31 改变，原始字节一致 |
| BN running_var | 均 31/31 改变 | 均 0/31 改变，原始字节一致 |
| BN num_batches_tracked | DR：16830→17680；H2T：16830→18530 | 两组全部 31 层保持 16830 |
| 分类器可学习张量 | 均 2/2 改变 | 均 2/2 改变，每个 epoch 有有限梯度 |

训练时逐次 Backbone forward 检查 `no_grad` 及所有 BN 的 eval 状态；每个 epoch 后核验 Backbone 参数、93 个 BN buffers 的完整字节状态哈希与 Stage I 一致，计数增量为 0。每次验证也核验 BN 不变；最后对最佳与最终文件中的每个张量再次按 dtype、shape、原始字节比较，不依赖容差或仅最大差值。四次完整测试 reload 通过。

Conv/BN affine 的全部 463,504 参数和三个 BN buffer 集合均不变，两组最终 Backbone state_dict 也逐位相同。每组 850 个训练 loss 为有限值，CE 标签始终为 BS，分类器每个 epoch 均更新。详细逐张量 SHA-256 和差值：[artifact_verification.json](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/artifact_verification.json>)。

| 固定 BN 组 | 最终分类器最大绝对变化（相对 Stage I） | GPU peak allocated MiB | GPU peak reserved MiB | 全 GPU 采样峰值 MiB |
|---|---:|---:|---:|---:|
| DR | 0.394815505 | 73.46 | 126.00 | 1557 |
| DR+H2T | 0.328750908 | 79.58 | 144.00 | 1566 |

全 GPU 采样峰值包含桌面及其他进程，采样间隔 5 秒；allocated/reserved 为训练进程 PyTorch CUDA 的峰值。

## 4. 机制结论与归因边界

在固定 Stage I BN、相同 Backbone、逐 batch 配对的 BS/IS 样本及增强条件下，H2T 在第 10 epoch 相对 DR 的 Overall 增益为 **+0.76 pp**，最佳模型增益为 **+0.75 pp**；本次单种子控制结果显示 H2T 带来额外收益。

这一固定 BN 条件内的差异可以归因于“加入原始随机通道融合”的训练干预及其对分类器学习轨迹的影响：BN、Conv/BN affine、数据和增强保持一致，测试时没有 H2T 融合，两组相同的固定 Backbone 只配合各自训练出的分类器。该结论以本次指定 seed、独立通道掩码流和软件/硬件环境为条件。

原始条件第 10 epoch H2T 增益为 +0.87 pp，本次固定条件为 +0.76 pp；二者相差 -0.11 pp。这不是可单独识别的 BN 因果贡献：历史原始实验使用共享 CPU RNG，H2T 调用会改变后续增强及 IS 顺序；本次同时固定 BN 并隔离通道 RNG。历史与控制之间同时改变了这些因素。未执行“BN 更新 + 同一隔离 RNG/配对协议”实验，不能将这个差值全部归因于 BN。

原始 `shift_bn=True` 使训练 forward 使用当前 batch 统计量，并更新用于测试的 running buffers；H2T 组每 batch 两次 Backbone forward，DR 一次。因此原始结果变化涉及训练特征归一化、BN 统计量迁移和分类器调整。固定 BN 不仅禁止 buffers 更新，还使训练特征使用 Stage I running statistics；不宜把原始全部性能变化解释为分类器权重调整。

仍保留的随机性与选择影响：仅运行一个 seed 和一条随机通道流，无多种子方差或置信区间；未启用确定性 CUDA 算法，可能有数值非确定性。最佳 epoch 由同一测试集逐 epoch 选择，且两组可能不同；最佳对比带选择效应，固定第 10 epoch 为本次主要对比。本次证据证明了输入配对和状态冻结，不代表跨 seed 的普遍因果效应。

## 5. 产物、保护核验及完整配置

原始套件的 delivery manifest、其 32 份交付文件、原始最佳/最终/当前 Checkpoint、源码快照、日志及已有报告等合计 146 个文件，训练前后 SHA-256 一致；原始 runtime 的 48 个源码文件也保持一致。证据：[protected_original_before.json](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/protected_original_before.json>)、[protected_original_after.json](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/protected_original_after.json>)。原始报告：[H2T_STAGE2_RESULTS.md](<D:/xiangmuxunlian/新/H2T_STAGE2_RESULTS.md>)。

### 新增模型和日志

| 固定 BN 组 | 最佳 epoch | 最佳 Checkpoint | 第 10 epoch Checkpoint | 配置 / 逐 epoch 记录 / 完整日志 |
|---|---:|---|---|---|
| DR | 7 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/runtime/saved/cifar100_if100_stage2_fixed_bn_20261011_2255_dr_202610112255/ckps/model_best_stage2.pth.tar>) | [epoch 10](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/runtime/saved/cifar100_if100_stage2_fixed_bn_20261011_2255_dr_202610112255/ckps/model_final.pth.tar>) | [YAML](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/dr/effective.yaml>) / [CSV](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/dr/epochs.csv>) / [日志](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/dr/stdout_stderr.log>) |
| DR+H2T | 8 | [best](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/runtime/saved/cifar100_if100_stage2_fixed_bn_20261011_2255_h2t_202610112300/ckps/model_best_stage2.pth.tar>) | [epoch 10](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/runtime/saved/cifar100_if100_stage2_fixed_bn_20261011_2255_h2t_202610112300/ckps/model_final.pth.tar>) | [YAML](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/h2t/effective.yaml>) / [CSV](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/h2t/epochs.csv>) / [日志](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/h2t/stdout_stderr.log>) |

最佳模型使用 `model_best_stage2.pth.tar`；最终模型为 current.pth.tar 的字节副本 `model_final.pth.tar`。没有重置原始全局 best_acc1 的 Stage I 阈值，observer 单独保存 10 epochs 内的最大值；本次与原始 model_best.pth.tar 一致。源码快照位于各组 saved 运行目录的 codes/，48 个文件均验证。每个模型路径和 SHA-256 见 artifact_verification.json。

### 已执行命令

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONHASHSEED = '123'
$auditRoot = 'D:\xiangmuxunlian\新\H2T_repro_audit\bn_audit_runs\fixed_bn_20261011_2255'
& 'D:\app\anaconda\envs\pytorch\python.exe' -u "$auditRoot\bn_audit_runner.py" prepare --suite-id fixed_bn_20261011_2255
& 'D:\app\anaconda\envs\pytorch\python.exe' -u "$auditRoot\bn_audit_runner.py" run --suite-id fixed_bn_20261011_2255 --arm dr
& 'D:\app\anaconda\envs\pytorch\python.exe' -u "$auditRoot\bn_audit_runner.py" run --suite-id fixed_bn_20261011_2255 --arm h2t
```

上述目录已完成，启动器会拒绝覆盖。重新执行需要通过 create_bn_audit.py 创建新的唯一 suite-id。每组原始训练入口 argv、加载路径和 SGD 参数保存在 status.json；训练入口仍为原始 train_stage2.py。

外部 observer 差异：[control_observer.diff](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/control_observer.diff>)；独立评估器差异：[evaluation_observer.diff](<D:/xiangmuxunlian/新/H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/evaluation_observer.diff>)。这些文件展示控制协议的改动，核心源码未修改。环境：3.12.13 | packaged by Anaconda, Inc. | (main, Mar 19 2026, 20:16:45) [MSC v.1942 64 bit (AMD64)]；PyTorch 2.11.0+cu130 / torchvision 0.26.0+cu130 / CUDA 13.0；NVIDIA GeForce RTX 5070 Laptop GPU。

### DR 固定 BN 完整有效配置

```yaml
H2T: 0.0
ResLT:
  gamma: 0.7
alpha: null
backbone: resnet32_fe
batch_size: 128
classifier: Classifier
data_path: "D:\\xiangmuxunlian\\\u65B0\\H2T\\data\\cifar100"
dataset: cifar100
deterministic: false
dist_backend: nccl
dist_url: null
distributed: false
gpu: null
head_class_idx:
- 0
- 36
imb_factor: 0.01
log_dir: logs
loss:
  CE:
    reweight_CE: false
  LDAM:
    max_m: 0.5
    reweight_epoch: -1
    s: 30
  RIDE:
    additional_diversity_factor: -0.2
    base_diversity_temperature: 1.0
    base_loss_factor: 1.0
    max_m: 0.5
    reweight: false
    reweight_epoch: -1
    reweight_factor: 0.05
    s: 30
  ResLT:
    beta: 0.99
    smooth: 0.0
  add_extra_info: false
  type: CrossEntropyLoss
lr: 0.1
lr_scheduler:
  lr_factor: 0.2
  lr_step:
  - 160
  - 180
  type: cos
med_class_idx:
- 36
- 71
mixup: false
mode: stage2
model_dir: ckps
momentum: 0.9
multiprocessing_distributed: false
name: cifar100_if100_stage2_fixed_bn_20261011_2255_dr
num_classes: 100
num_epochs: 10
num_experts: 1
print_freq: 40
rank: -1
resume: "D:\\xiangmuxunlian\\\u65B0\\H2T_repro_audit\\runtime\\saved\\cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428\\\
  ckps\\model_best.pth.tar"
sampler:
  dual_sample:
    enable: true
    type: long-tailed
  type: weighted sampler
  weighted_sampler:
    type: balance
shift_bn: false
smooth_head: null
smooth_tail: null
tail_class_idx:
- 71
- 100
use_norm: false
weight_decay: 0.0005
workers: 0
world_size: -1
```

### DR+H2T 固定 BN 完整有效配置

```yaml
H2T: 0.25
ResLT:
  gamma: 0.7
alpha: null
backbone: resnet32_fe
batch_size: 128
classifier: Classifier
data_path: "D:\\xiangmuxunlian\\\u65B0\\H2T\\data\\cifar100"
dataset: cifar100
deterministic: false
dist_backend: nccl
dist_url: null
distributed: false
gpu: null
head_class_idx:
- 0
- 36
imb_factor: 0.01
log_dir: logs
loss:
  CE:
    reweight_CE: false
  LDAM:
    max_m: 0.5
    reweight_epoch: -1
    s: 30
  RIDE:
    additional_diversity_factor: -0.2
    base_diversity_temperature: 1.0
    base_loss_factor: 1.0
    max_m: 0.5
    reweight: false
    reweight_epoch: -1
    reweight_factor: 0.05
    s: 30
  ResLT:
    beta: 0.99
    smooth: 0.0
  add_extra_info: false
  type: CrossEntropyLoss
lr: 0.1
lr_scheduler:
  lr_factor: 0.2
  lr_step:
  - 160
  - 180
  type: cos
med_class_idx:
- 36
- 71
mixup: false
mode: stage2
model_dir: ckps
momentum: 0.9
multiprocessing_distributed: false
name: cifar100_if100_stage2_fixed_bn_20261011_2255_h2t
num_classes: 100
num_epochs: 10
num_experts: 1
print_freq: 40
rank: -1
resume: "D:\\xiangmuxunlian\\\u65B0\\H2T_repro_audit\\runtime\\saved\\cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428\\\
  ckps\\model_best.pth.tar"
sampler:
  dual_sample:
    enable: true
    type: long-tailed
  type: weighted sampler
  weighted_sampler:
    type: balance
shift_bn: false
smooth_head: null
smooth_tail: null
tail_class_idx:
- 71
- 100
use_norm: false
weight_decay: 0.0005
workers: 0
world_size: -1
```

## 6. 完成状态

两组固定 BN 控制实验、四次 10,000 张测试集重新加载评估、原始产物保护核验、参数及 buffers 逐位核验、850 batches 配对核验均已完成。仅执行本次指定的两组控制；未开展其他改进实验。完成后停止，等待下一步指示。
