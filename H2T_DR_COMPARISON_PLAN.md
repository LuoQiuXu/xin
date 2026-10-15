# 同一 Stage I Checkpoint 的 DR / DR+H2T 对照方案

更新时间：2026-10-11。**本文件记录执行前的方案。后续已按用户确认完成原始 DR / DR+H2T 两臂各 10 epochs，见 [Stage II 结果](D:/xiangmuxunlian/新/H2T_STAGE2_RESULTS.md)。本方案的额外 BN 控制实验尚未实施，原始 H2T 算法未修改。**

## 1. 固定输入与复现口径

代码使用 [已审计兼容副本](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/train_stage2.py)，对应官方归档 Commit `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`。Commit 来源和独立验证限制见 [原审计报告](D:/xiangmuxunlian/新/H2T_REPRO_AUDIT.md)。

建议两臂都加载本轮 Stage I 的 **model_best.pth.tar 的同一份字节文件**。正式结果报告将记录其 SHA256 和 epoch；不为两臂各选不同的 Stage I 模型。Backbone、分类器权重及 BN buffers 必须全部一致，分别复制到两个独立模型中，不能共用可变 BN 状态。按作者源码加载原分类器，不自行随机重置。

Stage I 文件位置：

`D:\xiangmuxunlian\新\H2T_repro_audit\runtime\saved\cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428\ckps\model_best.pth.tar`

Stage I 已完成：该文件的 Checkpoint epoch=198，SHA256=`0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e`。其独立评估 Overall=40.490%、Head=68.806%、Medium=40.114%、Tail=5.793%；每个 BN 的初始 num_batches_tracked=16,830。后续两臂必须核对这一相同输入文件，详见 [Stage I 结果](D:/xiangmuxunlian/新/H2T_STAGE1_RESULTS.md)。

两臂共同设置：CIFAR-100-LT IF=100、同一 NumPy seed=0 产生的 10,847 张子集、ResNet-32、batch 128、10 epochs、SGD momentum 0.9、weight decay 5e-4、原 cosine 调度、CE、MixUp=False、相同初始化和分类器 SGD 空状态。基础配置 lr=0.1、lr_factor=0.2，分类器首个 epoch 的实际 LR=0.02；不更改此调度。

采样与标签均保持官方方式：BS 为均匀选类别再均匀选该类实例的接收分支；IS 为原长尾实例分布的供体分支，允许同类供体。损失使用 BS 标签，IS 标签不进入 CE。DR 分类器直接接收 BS 特征；DR+H2T 接收原 `methods.H2T` 的融合特征。H2T=0.25 表示 64 个通道中随机替换 16 个完整 8×8 通道图，整个 batch 共用该通道 mask。

## 2. 仓库默认设置下的两臂

| 项目 | DR | DR+H2T |
|---|---|---|
| Stage I checkpoint | 同一文件、同一 SHA256 | 同左 |
| BS / IS 数据配置 | 保留双采样配置 | 保留双采样配置 |
| H2T 配置 | `H2T 0.0` | YAML `H2T=0.25` |
| 特征 | 直接使用 BS 特征 | 原随机通道替换后的 BS 特征 |
| 训练标签 | BS 标签 | BS 标签 |
| SGD 参数 | 仅 classifier | 仅 classifier |
| Backbone 学习参数 | 不更新 | 不更新 |
| BN 模式 | `shift_bn=True`，train | `shift_bn=True`，train |
| 每 batch Backbone 前向 | BS 一次 | BS 一次，再 IS 一次 |
| 每个 BN 的计数增量 / epoch | 85 | 170 |
| 主要结果的选择 | 固定第 10 epoch | 固定第 10 epoch |

这是从现有源码直接运行得到的原始方法配置对照。它能比较两个完整训练流程的性能，但其中 **特征融合和 BN 统计量更新流程同时不同**，不能把全部性能差归因为分类器权重更新或通道替换的单独作用。

本轮不会执行下面的命令。下一轮确认后可使用独立 name 和输出目录：

```powershell
Set-Location -LiteralPath 'D:\xiangmuxunlian\新\H2T_repro_audit\runtime'
$h2tPython = 'D:\app\anaconda\envs\pytorch\python.exe'
$h2tStage1Checkpoint = 'D:\xiangmuxunlian\新\H2T_repro_audit\runtime\saved\cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428\ckps\model_best.pth.tar'

# DR：仅通过已有配置关闭 H2T，其他 YAML 设置保持原值。
& $h2tPython -u .\train_stage2.py --cfg .\config\cifar100\cifar100_imb001_stage2_ce_H2T.yaml data_path 'D:\xiangmuxunlian\新\H2T\data\cifar100' resume $h2tStage1Checkpoint H2T 0.0 name dr_if100_pending_run_id

# DR+H2T：保留官方 H2T=0.25、shift_bn=True。
& $h2tPython -u .\train_stage2.py --cfg .\config\cifar100\cifar100_imb001_stage2_ce_H2T.yaml data_path 'D:\xiangmuxunlian\新\H2T\data\cifar100' resume $h2tStage1Checkpoint name dr_h2t_if100_pending_run_id
```

`H2T 0.0` 必须写成 float，避免 yacs 将 `0` 解码为 int 后触发类型不匹配。这些命令复现官方执行路径；严格配对的随机性控制还需要下文的外部实验运行器。

## 3. 必须记录的 BatchNorm 问题

[Stage II train](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/train_stage2.py:297) 在 `shift_bn=True` 时调用 `model.train()`。`torch.no_grad()` 阻止学习参数产生梯度，**并不阻止 BN running_mean、running_var、num_batches_tracked 更新**。Backbone 的 Conv 与 BN affine 参数保持不变，31 个 BN 的三种 buffers 会变动。

DR+H2T 的执行顺序是 BS 前向，然后 IS 前向，再融合。DR 的 `H2T=0.0` 分支只做 BS 前向。即使数据集仍返回 IS 图像，也不会自动得到相同的 BN 更新。

本结构使用默认 BN momentum=0.1。以某层 running_mean 为例，一对 batch 的更新为：

```text
DR：       μ_new = 0.9 μ_old + 0.1 μ_BS
DR+H2T：   μ_new = 0.81 μ_old + 0.09 μ_BS + 0.1 μ_IS
```

running_var 也按同样动量顺序累计各分支的方差估计。因此差异既涉及更新次数，也涉及分布与顺序。评估切为 `model.eval()` 后，网络使用这些不同的累计统计量，预测会受到影响。

此前真实数据 Stage II dry-run 已验证：85 个 batch 中每个 BN 的计数增加 170，93/93 个 BN buffers 发生变化，而 Backbone 的 93 个学习参数均未变化。这不是兼容性错误修复对象，本轮保留作者行为。

## 4. 隔离 BN 影响的公平性控制

建议先报告第 2 节的官方流程对照，再增加 **预先声明的 BN 控制对照**。以下是后续可选实验设计，本轮不实施。

### A. 使用已有配置冻结 BN，作为补充诊断

两臂同时设 `shift_bn False`，分别用 H2T=0.0 / 0.25；两者从同一个 Stage I BN 状态开始，整个 Stage II 都不更新 BN buffers。仅 classifier 被 SGD 更新，原 H2T 随机通道替换保持不变。无需修改源码。

此配置改变了仓库默认 BN 训练模式，应明确标为 **“BN 统计量固定的补充对照”**，不能把其结果当成仓库 `shift_bn=True` 的原始复现结果。该对照可更清楚地观察同一固定表征下，融合训练输入对分类器学习的影响。

### B. 匹配两臂 BN 前向流程，作为另一种控制

如果需要保持 `shift_bn=True`，可在独立实验运行器中使 DR 也依次计算 BS、IS 两次 Backbone 前向，丢弃 IS 特征，仅把 BS 特征送给分类器；DR+H2T 完全调用原来的 `methods.H2T`。两臂使用相同输入、顺序和初始 BN 状态，使 BN 更新可以逐步一致。

这不改变 H2T 的供体、mask、rho 或标签规则，但它改变了原 DR 分支的前向及 BN 统计量更新流程，因此应单独标为 **“DR（匹配 BN 更新）”**。不能覆盖官方脚本，也不能与未经此控制的 DR 混报。本轮仅提出方案，后续确认后才编写独立运行器及 Diff。

仅在训练结束后给两臂复制相同 BN buffers，不能充分替代上述控制：若训练时采用 eval BN，统计量会影响训练表征；统一 buffers 还可能使模型评估分布与训练分布不一致。必须事先明确训练与评估协议。

## 5. 随机性与严格配对

Stage II YAML 的 `deterministic=False`，原入口没有主动固定种子。单次不固定随机性的比较可以作为源码重跑，但不足以支持精确性能归因。

即使给两臂设相同 seed，也不保证之后看到完全一样的 batch：原 H2T 使用 CPU `torch.randperm(64)`，CPU RNG 同时参与数据增强和 DataLoader 的随机过程；DR 不执行此调用，两臂的随机序列会分叉。

严格对照应使用独立运行器：

1. 在运行前为每个 epoch / batch 声明或记录相同的 IS 索引顺序、BS 抽样和增强随机状态；两臂重放相同增强后的输入与 BS 标签，保留官方采样分布，不按类别难度或供体质量筛选。
2. 原 H2T 继续产生均匀随机通道 mask。在 DR 臂匹配该随机调用的消耗或用独立 RNG 隔离 mask 与数据 RNG，记录具体实现；不要把固定 seed 误写成已实现输入配对。
3. 两臂使用相同的一组重复实验 seed；初始分类器、BN 状态及学习率一致。需要统计可靠性时按同一批 seed 重复，报告均值、标准差以及配对差值。
4. 为每臂记录数据子集 hash、每 epoch 标签直方图、采样/增强配对证据、BN buffer 差异、梯度与学习参数变化。若使用匹配 BN 流程，应确认每个 BN 的计数增量一致，并逐项核对 mean / var，而不只比较计数。

以上只控制实验可比性，不引入供体筛选、通道筛选、困难样本感知或自适应 rho。

## 6. 评估、Checkpoint 和归因

- 主要终点预先固定为各臂 Stage II 第 10 epoch，使用同一完整 CIFAR-100 test split。Overall 为样本准确率；Head [0,36)、Medium [36,71)、Tail [71,100) 为逐类准确率平均值。
- 保留完整 10 epoch 日志、current/final 和最佳文件，但不因某臂成绩临时改变训练时长、学习率、rho 或选择规则。
- 官方 Stage II 会继承 Stage I 的 `best_acc1` 作为最佳模型阈值。若 Stage II 全程没有超过 Stage I，该臂可能根本没有 model_best.pth.tar。比较时不能误用缺失、旧文件或另一臂的最佳模型。主终点直接使用各自第 10 epoch current/final，独立记录 Stage II 内最佳 epoch 作为次要结果。
- 官方入口每 epoch 使用 CIFAR-100 **test split** 监控并选择最佳 checkpoint；这里的“最佳”不是独立验证集上的模型选择。采用固定 Stage II 第 10 epoch 作为主终点有助于统一两臂选择规则。
- 报告 Stage I → DR 的增量、Stage I → DR+H2T 的增量，以及 DR → DR+H2T 的差值。仓库默认对照的差值包含分类器学习、融合训练输入和 BN 统计量流程的联合影响；补充 BN 控制对照提供更明确的归因证据。

## 7. 本轮边界

用户已确认的 Stage I 200 epochs 和 Stage II 原始 DR / DR+H2T 各 10 epochs 均已完成，结果与文件位置已保存。额外 BN 控制臂和机制审计均未启动，未改动原始 H2T 核心源码；等待下一步任务。
