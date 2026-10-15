# H2T Stage II 正式复现结果

## 1. 完成状态与结论

完成实验 A（DR）和实验 B（DR+H2T），各 **10 epochs / 850 个 SGD step**。两组分别在新进程中从同一 Stage I 最佳 Checkpoint 独立初始化，未加载对方的 Stage II 权重。所有训练 Loss 有限，源码 hash、参数冻结、分类器更新及 BN 更新核验通过。所有最佳和最终模型均已通过原 `eval-modifed.py` 在 **全部 10,000 张测试图像** 上独立重新加载评估。

本次单 seed 重跑中，原始 DR+H2T 流程在固定第 10 epoch 的 Overall 上带来额外收益：**+0.870 个百分点**；最佳模型对比为 **+0.960 个百分点**。这是单次固定 seed、仓库默认训练流程的观测结果。两臂 BN 更新次数和数据增强 RNG 消耗不同，结果不能单独归因于 H2T 通道融合或分类器权重调整。

本轮没有供体/通道筛选、自适应 rho、Backbone 结构更改或其他创新性修改。训练完成后停止；**未执行额外 BN 控制实验或机制审计**。

## 2. 输入 Checkpoint、配置与种子

Stage I epoch=198，Overall=40.490%。输入文件为 [同一 Stage I 最佳模型](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428/ckps/model_best.pth.tar)，SHA256=`0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e`。两臂分别核对 Backbone、classifier 和全部 BN buffers 与此 Checkpoint 逐 tensor 完全相等；SGD 从新的空状态开始。原源码加载 epoch=198 后，Stage II 仍执行自己的 0–9 epoch 循环，Checkpoint epoch 字段为 Stage II 的 1–10。

| 项目 | A：DR | B：DR+H2T |
|---|---|---|
| 数据 / 模型 | CIFAR-100-LT IF=100 / ResNet-32 | 同左 |
| Stage II / batch / workers | 10 epochs / 128 / 0 | 同左 |
| Loss / MixUp | CE，无类别重权重 / False | 同左 |
| SGD | lr=0.1，momentum=0.9，wd=5e-4，nesterov=False | 同左 |
| 分类器实际 LR | 原 cosine × lr_factor=0.2，首 epoch=0.02 | 同左 |
| 训练对象 | 仅 classifier 进入 SGD，Backbone no_grad | 同左 |
| 采样 | BS 接收；保留原 IS 元数据生成 | BS 接收 + IS 供体 |
| H2T | 仅 CLI 设 0.0；融合调用=0 | 原 YAML 0.25；融合调用=850 |
| Backbone 前向 / batch | BS 一次 | BS→IS 两次 |
| shift_bn | True | True |
| Seed | Python / NumPy / torch CPU / CUDA 初始 seed=123，PYTHONHASHSEED=123 | 同左 |
| deterministic | YAML False；原 CuDNN / deterministic algorithms 默认值保留 | 同左 |
| gpu | YAML null，单 GPU DataParallel | 同左 |

两臂的最终完整配置只在 `H2T` 和输出 `name` 上不同。数据路径和 resume 均一致。A 保留原 `sampler.dual_sample.enable=True`，数据集仍生成 IS 图像和标签，但原 `H2T>0` 条件为 False，跳过供体 GPU 前向和 H2T；CE 仅使用 BS 特征及 BS 标签。没有再关闭双采样配置而引入额外差异。

完整配置如下，并分别保存为独立 effective.yaml。

### 实验 A 完整配置

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
name: cifar100_if100_stage2_official_20261011_1525_dr
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
shift_bn: true
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

### 实验 B 完整配置

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
name: cifar100_if100_stage2_official_20261011_1525_h2t
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
shift_bn: true
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

## 3. 准确率、最佳 Epoch 与 Checkpoint

| 模型 | Stage II Epoch（从 1 开始） | Overall | Head | Medium | Tail |
|---|---:|---:|---:|---:|---:|
| Stage I 最佳 | 输入 epoch 198 | 40.490% | 68.806% | 40.114% | 5.793% |
| DR 最佳 | 7 | 47.100% | 65.278% | 47.400% | 24.172% |
| DR+H2T 最佳 | 9 | 48.060% | 61.806% | 50.657% | 27.862% |
| DR 最终 | 10 | 47.000% | 65.278% | 47.257% | 24.000% |
| DR+H2T 最终 | 10 | 47.870% | 61.833% | 50.600% | 27.241% |

独立评估的每类样本数均为 100；Overall 为全部样本准确率，Head [0,36)、Medium [36,71)、Tail [71,100) 为官方 YAML 的逐类准确率均值。评估使用干净特征，不执行 MixUp 或 H2T；BN buffers 在评估期间完全不变。独立评估四组指标与对应训练 epoch 一致。

- DR：[最佳](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_if100_stage2_official_20261011_1525_dr_202610111525/ckps/model_best_stage2.pth.tar)、[最终](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_if100_stage2_official_20261011_1525_dr_202610111525/ckps/model_final.pth.tar)；最佳 Epoch=7。
- DR+H2T：[最佳](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_if100_stage2_official_20261011_1525_h2t_202610111527/ckps/model_best_stage2.pth.tar)、[最终](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_if100_stage2_official_20261011_1525_h2t_202610111527/ckps/model_final.pth.tar)；最佳 Epoch=9。

“最佳”按 Stage II 10 个 epoch 内最高 Overall 选择，首次达到最高值时保留。外部记录器只复制原 current 文件为 model_best_stage2，未改变原 best_acc1、训练或保存逻辑。原 Stage II best_acc1 继承 Stage I 的 40.49% 门槛；本轮两个实验均超过该门槛，原 model_best.pth.tar 与独立 Stage II 最佳副本 SHA256 完全一致。最终 model_final.pth.tar 与原 current.pth.tar 字节相同。最佳和最终均保存并重新评估。

原日志 Best epoch 为零基编号；本报告最佳 Epoch 已加 1。主要 DR/H2T 对比采用固定第 10 epoch 的最终模型，最佳模型作为同时报告的结果。每 epoch 监控使用的是官方 CIFAR-100 test split，最佳选择并非独立 validation split。

## 4. 相对 Stage I 的变化与 H2T 的额外收益

以下均为**准确率百分点变化（pp）**，不是相对百分比增长。

| 对比 | Overall | Head | Medium | Tail |
|---|---:|---:|---:|---:|
| DR 最佳 − Stage I | +6.610 | -3.528 | +7.286 | +18.379 |
| DR+H2T 最佳 − Stage I | +7.570 | -7.000 | +10.543 | +22.069 |
| DR 最终 − Stage I | +6.510 | -3.528 | +7.143 | +18.207 |
| DR+H2T 最终 − Stage I | +7.380 | -6.972 | +10.486 | +21.448 |
| 最佳：H2T − DR | +0.960 | -3.472 | +3.257 | +3.690 |
| 最终：H2T − DR | +0.870 | -3.444 | +3.343 | +3.241 |

必须同时观察 Overall 和各类别组变化；Head 的下降与 Medium/Tail 的提升反映原训练流程对分类决策的重新平衡。本报告只陈述本轮测得的差值，不把单 seed 结果解释为多次实验的统计显著性。

## 5. 学习参数与 BN 统计量

| 实验（最终模型） | Backbone 学习参数变化 | classifier 参数变化 | running_mean 变化 | running_var 变化 | 每个 BN 计数增量 | 最终计数 |
|---|---:|---:|---:|---:|---:|---:|
| DR | 0/93 | 2/2 | 31/31 | 31/31 | 850 | 17680 |
| DR+H2T | 0/93 | 2/2 | 31/31 | 31/31 | 1700 | 18530 |

输入 Stage I 每个 BN num_batches_tracked=16,830。两组均包含 31 个 BatchNorm2d。每 epoch A 增加 85、B 增加 170；10 epochs 分别增加 850/1700。Conv 权重和 BN affine 参数逐 tensor 保持与 Stage I 完全一致，Backbone 梯度始终 None；classifier 的权重和 bias 都变化，梯度有限。源码的 Backbone requires_grad 标志仍为 True，实际冻结由 no_grad 和 classifier-only SGD 实现，未另行改写该机制。

- DR：classifier 最大绝对参数变化 0.397375；BN running_mean 最大绝对变化 0.188970，running_var 最大绝对变化 2.638661。
- DR+H2T：classifier 最大绝对参数变化 0.359021；BN running_mean 最大绝对变化 0.100773，running_var 最大绝对变化 2.444742。

`shift_bn=True` 保持 Backbone.train()；torch.no_grad 并不冻结 running_mean / running_var / num_batches_tracked。DR 只执行 BS 前向，H2T 先 BS 再 IS。默认 BN momentum=0.1 的均值累计分别为：

```text
DR:       μ_new = 0.9 μ_old + 0.1 μ_BS
DR+H2T:   μ_new = 0.81 μ_old + 0.09 μ_BS + 0.1 μ_IS
```

因此差异涉及统计量更新次数、输入分布和先后顺序。最终模型间 running_mean / running_var 的差异及逐层最大变化保存在 artifact_verification.json。这些 buffers 会参与 eval 模式推理，不能将 Stage II 的全部收益或 DR/H2T 差值都归因于 classifier 权重。本轮按要求保留此官方行为，没有统一或冻结 BN 统计量。

## 6. RNG 消耗与采样记录

两组初始 Python、NumPy、torch CPU/CUDA RNG 状态 hash 完全一致，后端设置一致。原 CIFAR 子集构造会将 NumPy seed 重设为 0，两组均经历此过程。种子控制通过独立运行器实现；未将 YAML deterministic 改为 True，也没有强制启用额外确定性算法。

H2T 的原 CPU torch.randperm(64) 每个训练 batch 消耗一次 CPU RNG；DR 没有该调用。数据增强和 DataLoader 也使用 CPU RNG，所以不能宣称全程逐 batch 图像完全相同。本轮没有增加虚拟 randperm、重放数据或隔离随机流。

- 实际比较了 850 对 batch 的指纹：BS 标签相同 850 对，IS 标签顺序相同 85 对，batch 就绪时 CPU RNG 状态相同 1 对。
- 每个 epoch 的前 2 个 batch 共 20 对图像做 hash 检查：BS 图像相同 1 对，IS 图像相同 1 对。这只是被检查的图像 batch 范围，没有对其余图像作“完全配对”声明。
- 首个 BS 图像 batch 相同：True；第二个 BS 图像 batch 相同：False。
- IS 每 epoch 的类别直方图均严格等于完整 10,847 张长尾子集计数；BS 每 epoch 的类别直方图和全部标签指纹分别记录。相同标签序列不等于相同的增强图像。

## 7. 与论文 Table 5 / README 的对比

论文 Table 5 给出的 CIFAR-100-LT IF=100 结果如下。[论文 Table 5](https://arxiv.org/html/2306.06963v3)

| 对比 | Overall | Head | Medium | Tail |
|---|---:|---:|---:|---:|
| 论文 Table 5 Stage I CE | 39.55% | 67.86% | 38.51% | 5.66% |
| 本次 Stage I − 论文 | +0.940 pp | +0.946 pp | +1.604 pp | +0.133 pp |
| 论文 Table 5 DR | 45.68% | 63.67% | 46.63% | 22.21% |
| 本次 DR 最佳 − 论文 | +1.420 pp | +1.608 pp | +0.770 pp | +1.962 pp |
| 本次 DR 最终 − 论文 | +1.320 pp | +1.608 pp | +0.627 pp | +1.790 pp |
| 论文 Table 5 DR+H2T | 47.73% | 62.00% | 50.17% | 27.07% |
| 本次 DR+H2T 最佳 − 论文 | +0.330 pp | -0.194 pp | +0.487 pp | +0.792 pp |
| 本次 DR+H2T 最终 − 论文 | +0.140 pp | -0.167 pp | +0.430 pp | +0.171 pp |

论文 Table 5 的 DR→BS+IS Overall 增益为 +2.05 pp。本轮最佳对比 +0.960 pp，固定最终对比 +0.870 pp；用户指定的 rho=0.25 为仓库 YAML 值，论文 §4.4 最佳比例为 0.3，两个设置不能视为完全相同的超参数复现。[论文 §4.4](https://arxiv.org/html/2306.06963v3)

官方 README 的 IF=100 Stage I=39.55%、Stage II CE+H2T=47.80%，没有提供对应 DR-only 结果。本轮 Stage I=40.49%（+0.94 pp），H2T 最佳 48.060%（相对 README +0.260 pp）、最终 47.870%（+0.070 pp）。作者说明仓库配置重跑略优于论文。[官方 README](https://github.com/Keke921/H2T#results-and-models)

## 8. 复现偏差与解释边界

1. 使用本机 Stage I epoch-198 最佳模型，不是作者发布的 39.55% 模型；上游表征不同，后续绝对准确率可能不同。两实验内部始终使用同一文件。
2. 按用户和原 YAML 使用 rho=0.25；论文最佳 0.3 对 64 通道替换 19 个，而当前替换 16 个。本轮没有据结果调参。
3. 运行环境为 Windows、NVIDIA GeForce RTX 5070 Laptop GPU、torch 2.11.0+cu130、torchvision 0.26.0+cu130、CUDA runtime 13.0、Python 3.12.13。原论文训练环境未由仓库完整锁定；不同库/硬件的数值路径可能导致偏差。
4. Stage II 原 YAML 没固定 seed，本轮按用户要求固定为 123；依然保留 deterministic=False。报告为一次原始流程重跑，不提供重复 seed 的均值/方差。
5. 相同种子不消除 H2T 对 CPU RNG 的额外消耗。BN 统计量也存在官方规定的单/双前向差异。这两点限制“纯融合机制”的因果归因。
6. 仍使用审计锁定的官方归档 Commit `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`，兼容 Diff 仅修复构造器、pathlib 导入、现代 torch.load 的运行问题。Commit 来自 ZIP comment，未以远程 Git 对象独立核验；逐文件 SHA256、运行快照和方法/采样/模型源文件均核对未变。

## 9. 保存位置、日志和命令

证据目录：`D:\xiangmuxunlian\新\H2T_repro_audit\stage2_runs\official_20261011_1525`。

### DR

- 开始 2026-10-11T15:25:42+0800，结束 2026-10-11T15:27:30+0800；耗时 107.52 秒。
- PyTorch 峰值 allocated=73.46 MiB、reserved=126.00 MiB；GPU 全局采样占用峰值=1682 MiB（包含桌面程序，非本进程独占显存）。
- [原训练日志](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_if100_stage2_official_20261011_1525_dr_202610111525/logs/cifar100_if100_stage2_official_20261011_1525_dr.txt)、[完整 stdout/stderr](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/dr/stdout_stderr.log)、[逐 epoch / BN / RNG 记录](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/dr/status.json)、[epoch CSV](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/dr/epochs.csv)。
- [完整配置](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/dr/effective.yaml)、[原 YAML](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/dr/original.yaml)、[运行源码快照](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_if100_stage2_official_20261011_1525_dr_202610111525/codes)。
- [最佳模型评估](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/dr/best_evaluation.json)、[最终模型评估](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/dr/final_evaluation.json)，含全部逐类正确计数；对应 stdout 日志位于相同证据目录。

实际运行命令（已执行完毕，展示追溯用途）：

```powershell
Set-Location -LiteralPath 'D:\xiangmuxunlian\新'
$env:PYTHONHASHSEED = '123'
& 'D:\app\anaconda\envs\pytorch\python.exe' -u .\H2T_repro_audit\formal_stage2.py run --suite-id official_20261011_1525 --arm dr
```

运行器调用原 train_stage2.py；完整原入口 argv 保存于 status.json。最佳模型保存副本及证据 hook 不更改训练计算。直接单独运行原入口而不设置所有 RNG 的种子，会失去本轮的相同随机种子设置。

### DR+H2T

- 开始 2026-10-11T15:27:52+0800，结束 2026-10-11T15:29:48+0800；耗时 115.32 秒。
- PyTorch 峰值 allocated=79.58 MiB、reserved=144.00 MiB；GPU 全局采样占用峰值=1683 MiB（包含桌面程序，非本进程独占显存）。
- [原训练日志](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_if100_stage2_official_20261011_1525_h2t_202610111527/logs/cifar100_if100_stage2_official_20261011_1525_h2t.txt)、[完整 stdout/stderr](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/h2t/stdout_stderr.log)、[逐 epoch / BN / RNG 记录](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/h2t/status.json)、[epoch CSV](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/h2t/epochs.csv)。
- [完整配置](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/h2t/effective.yaml)、[原 YAML](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/h2t/original.yaml)、[运行源码快照](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/cifar100_if100_stage2_official_20261011_1525_h2t_202610111527/codes)。
- [最佳模型评估](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/h2t/best_evaluation.json)、[最终模型评估](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/h2t/final_evaluation.json)，含全部逐类正确计数；对应 stdout 日志位于相同证据目录。

实际运行命令（已执行完毕，展示追溯用途）：

```powershell
Set-Location -LiteralPath 'D:\xiangmuxunlian\新'
$env:PYTHONHASHSEED = '123'
& 'D:\app\anaconda\envs\pytorch\python.exe' -u .\H2T_repro_audit\formal_stage2.py run --suite-id official_20261011_1525 --arm h2t
```

运行器调用原 train_stage2.py；完整原入口 argv 保存于 status.json。最佳模型保存副本及证据 hook 不更改训练计算。直接单独运行原入口而不设置所有 RNG 的种子，会失去本轮的相同随机种子设置。

[环境 / 数据 / 输入预检](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/preflight.json)、[参数 / BN / Checkpoint 核验](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/artifact_verification.json)、[官方源码兼容 Diff](D:/xiangmuxunlian/新/H2T_repro_audit/stage2_runs/official_20261011_1525/compatibility.diff)。

全部 Checkpoint 的 SHA256、epoch 和参数/BN 比较记录在核验 JSON，输出证据逐文件 SHA256 记录在 delivery_manifest.json。Stage I 输入 hash 在训练前后均保持不变。外部运行、评估、报告脚本及引用数据各保存一份。

**任务完成后停止；等待下一步机制审计任务。**
