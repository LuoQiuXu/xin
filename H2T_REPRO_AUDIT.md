# H2T 原始方法复现审计

> 后续状态更新（2026-10-11）：已用本报告的兼容副本完成 Stage I 200 epochs（最佳 Epoch=198、Overall=40.490%），见 [Stage I 结果](D:/xiangmuxunlian/新/H2T_STAGE1_RESULTS.md)。随后按用户确认从同一 Stage I 最佳模型完成 Stage II DR / DR+H2T 各 10 epochs：最佳 Overall=47.100% / 48.060%，最终=47.000% / 47.870%，见 [Stage II 结果](D:/xiangmuxunlian/新/H2T_STAGE2_RESULTS.md)。BN 控制实验及下一步机制审计未启动。以下原审计及 1-epoch dry-run 记录保留 2026-10-09 的历史状态。

日期：2026-10-09，Asia/Shanghai。目标：CIFAR-100-LT / IF=100 / ResNet-32 / Stage I 200 epochs / Stage II 10 epochs / batch size 128 / SGD。设备：Windows 11、RTX 5070 Laptop GPU、8GB 显存。

## 1. 结论与当前状态

**必要兼容性处理和真实数据短测已完成。具备开始官方仓库配置的 Stage I 200-epoch 训练的技术条件，完整训练尚未启动，等待用户确认。**

- 官方仓库包含本目标所需的两个训练阶段、CIFAR-100-LT 构造、ResNet-32、CE、MixUp、H2T 与评估实现，但原样代码不能直接运行：ResNet 构造存在两处错误，当前环境还存在 pathlib2 和 checkpoint 加载兼容问题。
- 在独立副本中做最小兼容修补后，实际执行了 Stage I **完整 1 epoch**、Stage II **完整 1 epoch**，各 85 次 SGD 更新；均通过。本次 Stage II 使用刚生成的 Stage I 1-epoch checkpoint。
- 独立评估入口在全部 10,000 张真实测试图像上执行成功，结果与训练时评估以及逐样本独立计数一致。
- 本次采用官方仓库发布的 **rho=0.25、shift_bn=True**。论文最佳比例为 0.3，因此本次应标记为“原始 H2T、官方仓库配置复现”，不能称为论文全部超参数的逐项一致复现。
- 完整训练后的论文准确率尚未验证；本次短测准确率仅证明流程能够运行。

审计使用的入口位于 [独立运行副本](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/train_stage1.py)。现有 H2T 两个目录的源码未修改；兼容副本中的 H2T 函数、CIFAR-100 数据加载器、损失函数、两个 YAML 和训练函数主体均保持官方实现。

## 2. 官方来源、Commit 与追溯边界

官方论文：[AAAI 2024 出版页](https://ojs.aaai.org/index.php/AAAI/article/view/29262)。补充材料对照使用 [arXiv v3 正文和附录](https://arxiv.org/html/2306.06963v3)，重点为 §3.3、Algorithm 1、§4.2、§4.4、Appendix A Listing 1。官方仓库：[Keke921/H2T](https://github.com/Keke921/H2T)。

| 项目 | 记录 |
|---|---|
| 本次固定的 Commit Hash | `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8` |
| Commit 记录来源 | 工作区 GitHub ZIP 的 archive comment |
| 源 ZIP | [H2T-main.zip](D:/xiangmuxunlian/新/H2T-main.zip) |
| ZIP SHA256 | `3f16f6759cfba1a5fbaa77b3d4a3bd5c43597b829298fd078c2459e12a79adcc` |
| 提取内容 | 48 个文件；其中源码、YAML、README 共 40 个 |
| 原始 H2T methods.py SHA256 | `d8266450190511f60771d7e632280c37a4d38a190daf4e5e55898656b00461a1` |
| 原始文件完整性 | 48/48 文件与提取时 SHA256 清单一致 |

已尝试 git clone，但 GitHub 网络连接失败；GitHub connector 也未成功连接。现有目录没有可用 Git checkout，不能用 `git rev-parse HEAD` 得到本地提交。故从带 Commit 标识的 ZIP 重新提取官方快照，并通过可访问的官方网页对照模型错误、H2T、配置和训练实现。**此 Commit 来自 ZIP 元数据；本次未独立通过 Git 对象验证该 Commit，也未确认最新远端 HEAD。** 不虚构本地 Git 历史或作者提交时间。

原始快照位于 [official](D:/xiangmuxunlian/新/H2T_repro_audit/official/README.md)；完整文件清单见 [source_manifest.json](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/source_manifest.json)。每阶段还保留了作者 `save_code` 生成的实际运行源码快照、配置、日志、checkpoint 和 SHA256；训练时的代码快照与最终兼容副本之间仅可能存在后续 checkpoint 加载兼容补丁的差异。

## 3. 论文、README 与实际配置

论文 §4.2 的目标设置为 ResNet-32、200+10 epochs、batch 128、SGD momentum 0.9；Stage I 的基础 LR 为 0.1，在 160/180 附近降低。原始方法使用 BS 接收分支、IS 供体分支及接收标签，Appendix A 使用随机通道替换。§4.4 最佳比例为 0.3，Table 1 的 IF=100 DR+H2T 为 47.73%。[论文](https://arxiv.org/html/2306.06963v3)

| 配置项 | Stage I 实际配置 | Stage II 实际配置 | 审计判断 |
|---|---|---|---|
| 数据集 / IF | cifar100 / imb_factor=0.01 | 相同 | IF 是 500/5=100；配置数值为 IF 的倒数 |
| Backbone | resnet32_fe | resnet32_fe | 普通 ResNet-32，无额外 PIF 层参与前向 |
| 正式 epochs | 200 | 10 | 与目标一致；本次均只运行 1 |
| Batch size | 128 | 128 | 最后一个 batch 为 95；保留官方 drop_last=False |
| SGD | momentum=0.9，weight_decay=2e-4 | momentum=0.9，weight_decay=5e-4 | 保留原配置，nesterov=False |
| YAML lr | 0.1 | 0.1 | 不能仅凭这个字段判断第一步实际 LR |
| 实际 LR 调度 | multistep，5 epoch warmup | cosine，lr_factor=0.2 | Stage I 默认配置来自 utils/logger.py；Stage II 只有 classifier 参数组 |
| 第一 epoch 实际 LR | 0.02 | 0.02 | 两个短测与正式训练第一 epoch 一致 |
| Stage I 衰减 | 第 160/180 个一基 epoch 仍为前一档；第 161/181 个起降档 | 不适用 | 作者代码使用 epoch+1 后严格 `>` 判断；保留该边界 |
| 损失 | 普通 CrossEntropyLoss | 普通 CrossEntropyLoss | 无 CE 类别重加权、无 label smoothing |
| MixUp | True，alpha=1.0 | False，alpha=null | Stage I 是 CE + 图像 MixUp |
| 主采样 | 实例 shuffle，不重采样接收类别 | weighted sampler / balance | Stage II 在 __getitem__ 内按类均匀抽样 |
| 供体分支 | 关闭 | dual_sample.enable=True / long-tailed | 一个原始 IS index 对应一个供体 |
| H2T 比例 | 不调用 H2T | YAML H2T=0.25 | 覆盖函数默认 0.3；保持仓库 YAML |
| BN | backbone.train() | shift_bn=True → backbone.train() | Stage II 权重冻结，BN 统计量更新 |
| 随机性 | deterministic=True → seed=123 | deterministic=False | 未额外改成确定性 Stage II；抽样子集内部固定 rand_number=0 |
| Windows loader | workers=0 | workers=0 | 与官方 YAML 一致，不使用 NCCL 分布式 |
| GPU 包装 | gpu=null → 单卡 DataParallel | 同上 | checkpoint 键含 module.，后续加载保持相同包装 |

Stage I 的 5-epoch warmup 和严格 `>` 的边界是仓库实现细节；论文没有完整说明这些细节。Stage I 正式调度实测函数值：一基 epoch 1/2/5/160/161/180/181/200 分别为 0.02/0.04/0.1/0.1/0.01/0.01/0.001/0.001。Stage II 正式 10-epoch 调度从约 0.02 到 0.000489435，公式分母为 10；本次只执行了其第一 epoch。

README 的 IF=100 Stage I 参考为 39.55%，Stage II 为 47.80%，并说明仓库配置重跑略优于论文。[官方 README](https://github.com/Keke921/H2T#results-and-models) 本次保留 0.25，未主动换成 0.3。

对应源码：[Stage I 调度](D:/xiangmuxunlian/新/H2T_repro_audit/official/train_stage1.py:457)、[Stage II 调度](D:/xiangmuxunlian/新/H2T_repro_audit/official/train_stage2.py:487)、[默认配置](D:/xiangmuxunlian/新/H2T_repro_audit/official/utils/logger.py:78)、[Stage I YAML](D:/xiangmuxunlian/新/H2T_repro_audit/official/config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml)、[Stage II YAML](D:/xiangmuxunlian/新/H2T_repro_audit/official/config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml)。

## 4. 两阶段与 H2T 的完整性审计

### 4.1 数据和模型

CIFAR-100 原始训练集 50,000 张，每类 500 张；作者按类编号 i=0…99 保留 `int(500 * 0.01**(i/99))` 张。每类原始索引使用 NumPy seed=0 洗牌后截取，测试集不做长尾下采样。两阶段实际使用完全相同的训练子集，数组内容和标签一致，子集 SHA256 已记录。

训练变换是 RandomCrop(32,padding=4)、RandomHorizontalFlip、ToTensor 和作者原来的 Normalize；评估只有 ToTensor 与相同 Normalize。均值 `(0.4914,0.4822,0.4465)`、标准差 `(0.2023,0.1994,0.2010)` 保留，没有更换成另一套 CIFAR-100 统计量。

模型由 3×3 stem、16/32/64 通道的三个 residual stage 构成，每个 stage 5 个 BasicBlock，使用 option A shortcut。池化前输出 `[B,64,8,8]`，分类器进行全局均值池化及 Linear(64,100)。Backbone 463,504 个参数，classifier 6,500 个参数，总计 470,004；31 个 BatchNorm2d。最小修补后 PIF 默认关闭，未构造 conv1d 参数。

源码：[数据集构造](D:/xiangmuxunlian/新/H2T_repro_audit/official/datasets/cifar100.py:14)、[网络与分类器](D:/xiangmuxunlian/新/H2T_repro_audit/official/models/resnet_cifar.py:126)。

### 4.2 Stage I：CE + MixUp

实例采样的图像先按 Beta(1,1) 得到 lambda，然后在同一个 batch 内使用随机 permutation 做图像 MixUp。损失是两组标签 CE 的 lambda 加权和。SGD 包含 backbone 和 classifier，两者都通过正常计算图反向传播。CE 默认没有 class weight；本次 85 个 batch 均实际调用了 MixUp。原始训练日志中的 Acc@1 使用原始标签，只是训练监测值，正式比较使用干净测试集评估。

源码：[Stage I train](D:/xiangmuxunlian/新/H2T_repro_audit/official/train_stage1.py:285)、[MixUp](D:/xiangmuxunlian/新/H2T_repro_audit/official/methods.py:8)、[CE](D:/xiangmuxunlian/新/H2T_repro_audit/official/loss.py:8)。

### 4.3 Stage II：BS + IS 双采样和标签

入口仍使用 `dataset.train_instance`。其 DataLoader 对长尾样本索引做不放回 shuffle；`__getitem__` 先以传入 index 生成 `meta['sample_image']` 和 `sample_label`，作为 **IS 供体**，随后均匀抽一个类别、再在类内随机抽一个实例，返回 `(img,target)`，作为 **BS 接收样本**。每对样本按 batch 行对应，无供体筛选；供体可以来自 head、medium、tail 或同类。

故名称 `train_instance` 不代表 Stage II 的接收标签仍是长尾分布，`train_balance` 这个独立 ClassAwareSampler loader 也不是本配置实际训练入口。Stage II 特征顺序为 `model(BS_images)`、`model(IS_images)`，调用 H2T 接收前者、注入后者。损失只读取 BS target，供体标签未进入 CE，也未混合标签。

源码：[双采样顺序](D:/xiangmuxunlian/新/H2T_repro_audit/official/datasets/cifar100.py:75)、[Stage II 实际 loader](D:/xiangmuxunlian/新/H2T_repro_audit/official/train_stage2.py:231)、[Stage II 特征与损失](D:/xiangmuxunlian/新/H2T_repro_audit/official/train_stage2.py:321)。

### 4.4 rho 的实际含义与随机通道替换

**源码中的 rho 是 IS 供体替换进 BS 接收特征的通道比例。** 一次调用用 CPU 上的 `torch.randperm(C)` 抽一个通道排列，再搬到默认 CUDA，取前 `int(rho*C)` 个通道。所有 batch 样本共享这一组通道；替换同编号通道的整个空间特征图，不抽空间区域、不做线性插值、不逐样本重新抽 mask。函数原地覆盖 x1 并返回同一对象，x2 保持不变。

本配置 C=64、rho=0.25：替换 16 个、保留 48 个；不是保留 16 个。rho=0.3 在同一结构上会替换 19 个，有效比例为 19/64=29.6875%，因为代码使用 int 截断。

论文主文 Eq.(1) 将 M_p 乘接收特征，其后文字把 M_p 的 1 通道数写为 [d*p]，按该段字面会得到 p 是保留比例，与附录替换代码存在符号口径冲突。**本次按作者 Appendix A 和官方可执行代码解释为替换比例，不把 rho 变成 1-rho。** 附录用 Tensor 类型检查，仓库用 `len(x1.shape)==4`；本目标的 4D tensor 分支随机数流和替换行为一致。[论文正文与附录](https://arxiv.org/html/2306.06963v3)

官方 methods.py 在两个本地目录、原始快照和兼容副本之间一致。本次未添加 RATIO_DETACH；活动源码/YAML 搜索也没有该开关。固定浮点 rho 不参与优化，不存在需新增 detach 开关的可学习比例。官方文件中的 H2Tv2 预留接口未被本配置调用。

源码：[原始 H2T](D:/xiangmuxunlian/新/H2T_repro_audit/official/methods.py:31)。

### 4.5 冻结、计算图、原地赋值和 BN

Stage II backbone 前向和 H2T 均位于 `torch.no_grad()`；classifier 前向位于该上下文外，SGD 只包含 classifier。Backbone 的 requires_grad 标志仍为 True，但没有梯度，也没有进入优化器；这是作者采用的有效权重冻结方式。本次所有 backbone 参数逐位相等，classifier weight/bias 都更新。

原地赋值在本路径下作用于无计算图的特征，实测没有 autograd 版本冲突。Stage I 不调用 H2T。若在未冻结 backbone 的路径直接复用该原地函数，可能产生计算图风险；不属于本配置，本次未扩展或修改。

`shift_bn=True` 使 Stage II 保持 backbone.train()，两分支各做一次前向。因此 31 个 BN 的 running_mean、running_var、num_batches_tracked 均会变化；每个 BN 在 85 个 batch 中累计 **170 次**。BN 的可学习 affine 参数仍冻结。评估时 model.eval()，BN buffers 不再更新。**将整个 Stage II 改成 model.eval() 会改变作者默认训练策略，本次保留原行为。**

源码：[官方模式切换](D:/xiangmuxunlian/新/H2T_repro_audit/official/train_stage2.py:305)、[SGD 参数组](D:/xiangmuxunlian/新/H2T_repro_audit/official/train_stage2.py:241)。

### 4.6 评估与分组口径

评估使用完整、均衡的 CIFAR-100 test split，干净特征直接池化分类，不执行 H2T 或 MixUp。Overall 为样本 top-1，Head/Medium/Tail 为各组逐类准确率的均值。官方 CIFAR YAML 的分组是零基半开区间：Head [0,36)、Medium [36,71)、Tail [71,100)。

本次全部 100 类均有 100 个测试样本，无除零问题。按该分组重新计数与原程序打印值一致。它与按训练样本量 `>100`、`20<n<=100`、`<=20` 分组并不完全相同：例如类 35 有 98 张却归 Head，类 69/70 有 20/19 张却归 Medium。保留官方 CIFAR YAML 分组，比较其他报告时必须核对区间。论文对应阈值描述针对大规模数据集，不能直接覆盖本 YAML。

源码：[Stage II validate](D:/xiangmuxunlian/新/H2T_repro_audit/official/train_stage2.py:404)、[实际评估文件](D:/xiangmuxunlian/新/H2T_repro_audit/official/eval-modifed.py:232)。

## 5. 官方源码与当前本地代码差异

对 ZIP 中全部 40 个源码、YAML、README 文件做了逐文件、换行归一化比较。

| 本地目录 | 改动的官方文件 | 影响 |
|---|---|---|
| H2T | datasets/cifar100.py | 增加多供体候选和实例索引 metadata；候选数为 1 时接近原单供体路径，但本次直接使用原始数据加载器 |
| H2T | models/resnet_cifar.py | 修补原构造错误，同时增加 PIF active 开关、PIF backbone 工厂，启用时返回 out_hat；属于扩展架构 |
| H2T | train_stage1.py | 增加 set_pif_active 调用、checkpoint PIF 标记 |
| H2T | utils/logger.py | 默认 pif_active=True；普通 resnet32_fe 配合新增调用会触发 ValueError |
| H2T | utils/util.py | 将 pathlib2 替换为标准库 pathlib |
| H2T_official_clean/H2T-main | models/resnet_cifar.py | 删除错误 Conv2d 参数并移除整段 PIF 初始化/前向分支；可运行但不是逐字未修改版本 |

本地 H2T 的普通默认 Stage I 模型设置已实测失败：`cannot activate PIF on a backbone without PIF parameters`。本次不在该扩展目录中绕过开关训练，使用独立原始副本。

两目录的原始 methods.py、train_stage2.py、loss.py、IF=100 YAML 和 eval-modifed.py 与 ZIP 相同。除改动的官方文件外，H2T 还包含 36 个新增源码/配置/报告，H2T_official_clean 包含 12 个新增源码/报告，均未进入本次训练入口。活动源码与 YAML 中 RATIO_DETACH 命中为 0；此搜索排除历史 saved/outputs/data/tmp，并不宣称旧归档中从未出现该名称。

完整证据：[H2T_diff](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/H2T_vs_official.diff)、[H2T_official_clean_diff](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/H2T_official_clean_vs_official.diff)、[逐文件比较](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/local_comparison.json)、[新增文件与开关搜索](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/local_added_files.json)、[本地 PIF 错误](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/local_default_pif_probe.json)。Diff 中最后一行粘连来自原文件缺少末尾换行，比较 JSON 已区分文本内容变化。

## 6. 发现的问题、影响与处理

| 问题 | 影响范围 | 处理或建议 |
|---|---|---|
| Conv2d 收到无效 feature_tuning 参数 | 两阶段和评估无法创建 ResNet-32，实测 TypeError | 独立补丁删除该 Conv2d 参数 |
| ResNet_fe 构造函数引用未定义 feature_tuning | 修复上一项后仍会 NameError | 独立补丁在 ResNet_fe 的签名补默认 False，保持 PIF 关闭 |
| pathlib2 未安装、无依赖锁文件 | utils 导入失败；官方运行环境无法直接追溯 | 独立补丁换成标准库 pathlib；记录现有完整版本信息 |
| 新 PyTorch 的 weights_only=True 默认行为 | 本次真实 checkpoint 含 NumPy ECE 标量，实测 UnpicklingError | 仅兼容副本显式 weights_only=False，加载本次自产、已记 SHA256 的 checkpoint |
| README 的 config 路径缺少 cifar100 子目录，Stage I 命令缺 .yaml | 按说明复制命令找不到配置 | 提供实际存在的完整相对路径 |
| README 的 eval-modified.py、eval.py 与文件名不符 | 无法启动评估；实际文件为 eval-modifed.py | 保留文件名，提供正确入口和 IF=100 cfg，覆盖其默认 IF=200 配置 |
| Stage I YAML 默认 resume=current.pth.tar | 若 cwd 意外已有该文件，会加载旧模型，不能确认从头训练 | 正式从头命令显式传空 resume |
| Stage II 或 eval 的 resume 文件缺失时仍继续 | 可能在随机 backbone 上训练或评估而不报错退出 | 命令先 Test-Path，短测 runner 也强制存在；不修改作者核心 |
| Stage II shift_bn=True | 权重冻结不等于 BN 状态固定，会影响结果 | 保留并实测；不把 backbone.eval() 当兼容补丁 |
| YAML rho=0.25、函数默认/论文最佳 0.3；主文 mask 口径冲突 | 超参数不逐项一致，误解可能反向替换比例 | 明确按源码取“供体替换比例”；不自动改 rho |
| .cuda() 默认设备与 CPU 宣称不一致 | 无 CUDA 时会失败；换非默认卡可能出现 index 设备问题 | 本机 CUDA:0 单卡实测可用；保留 CPU randperm 后 cuda 的随机流，不改为 CUDA 原生 randperm |
| gpu=null 与 gpu=0 包装不同 | module. checkpoint 键不匹配；eval gpu=0 分支还有 best_acc1 整数 .to 问题 | 全流程保留官方 gpu=null 的单卡 DataParallel，不混用包装 |
| resume 不恢复 optimizer、epoch 调度、RNG | 作者代码不是精确中断续训：加载权重后重新从 epoch 0 开始；checkpoint 无 SGD momentum state | 记录为完整训练的操作限制；本次不扩展续训逻辑，不能把 resume 当等价续训 |
| Stage II best_acc1 继承 Stage I | 若 Stage II 始终未超过 Stage I，可能没有本阶段 model_best | current.pth.tar 总会保存；评估命令必要时退回本阶段 current，禁止借用 Stage I best |
| 输出目录时间戳只有分钟 | 同一分钟同名重跑可能合并日志/覆盖 checkpoint | 短测与正式运行用不同 name；重复正式实验应使用独立 name |
| H2T 多专家 fallback 先访问 x1.shape | list 输入会在进入 else 前报错；其他 loss/backbone 接口也未完整集成 | 不影响本目标 4D tensor；记录，不扩展其他数据集/专家路径 |
| weighted sampler 的非目标 long-tailed 分支疑似 sample_class 未赋值，dual_sample 节点真值判断及重复 reset_epoch | 非目标配置可能出错，目标 balance+long-tailed 路径没有触发 | 记录，不修改本次原始数据加载逻辑 |

修补前的问题、范围和建议已先写入 [pre_patch_findings.md](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/pre_patch_findings.md)；默认 checkpoint 加载错误证据见 [checkpoint_load_probe.json](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/checkpoint_load_probe.json)。

## 7. 独立兼容性补丁与算法不变检查

补丁仅改动 5 个原始文件：模型构造函数 2 行；标准库导入 1 行；三个入口各 2 个 torch.load 调用加入 weights_only=False。训练循环、调度、SGD、CE、采样、H2T、标签与 BN 模式保持原代码。

[compatibility.diff](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/compatibility.diff) 给出官方 → 兼容副本的全部 Diff；[prepare_compatibility.py](D:/xiangmuxunlian/新/H2T_repro_audit/prepare_compatibility.py) 可从保留快照重建副本，已有 runtime 时拒绝覆盖。

完整运行环境：[environment.json](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/environment.json)。实际 Python 为 `D:\app\anaconda\envs\pytorch\python.exe`，版本 3.12.13；torch 2.11.0+cu130、torchvision 0.26.0+cu130、NumPy 2.3.1、Pillow 11.1.0、PyYAML 6.0.3、yacs 0.1.8。GPU compute capability=(12,0)，wheel 包含 sm_120，CUDA 实际张量运算通过；驱动 591.74。没有安装或升级环境。

依赖记录见 [requirements-repro.txt](D:/xiangmuxunlian/新/H2T_repro_audit/requirements-repro.txt)。本配置不依赖 tensorboard；pathlib2 已通过兼容导入处理。Windows 使用 workers=0、单卡非分布式训练。

未增加供体筛选、困难样本感知、自适应融合比例、PIF 前向、H2T 改进或新训练策略。[runtime_manifest.json](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/runtime_manifest.json) 和最终追溯清单验证 methods.py、datasets/cifar100.py、loss.py、默认配置和 YAML 一致，以及两个 train 函数/调度函数 AST 相同。

## 8. 真实数据 1-epoch dry-run 结果

通过 torchvision 官方文件 MD5 验证 train、test、meta。训练共 10,847 张、100 类、实际 IF=100，最大 500、最小 5；测试 10,000 张、每类 100。每阶段 84 个 128-batch 加最后一个 95-batch；均执行 85 次 optimizer.step，没有截断为几步 smoke test。完整类别分布与训练子集哈希见 [preflight.json](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/preflight.json)。

审计 runner 调用作者真实 main/train/validate；通过只读快照、loader 观察和函数包装记录证据。没有代写训练算法。H2T mask 验证使用独立 CPU Generator 复制已有 RNG 状态，不额外消耗训练随机数；额外计数和断言会影响少量耗时/显存，因此下表时间是带审计的单次记录。

| 检查 | Stage I | Stage II |
|---|---|---|
| 完整训练数据遍历 | 10,847 张 / 85 步 | BS 接收 10,847 张 / IS 供体 10,847 张 / 85 步 |
| 样本加权平均训练 Loss | 4.201193（实际加权 MixUp CE） | 4.645771（普通 CE） |
| batch Loss 范围 | 3.643492–7.248970 | 4.294616–5.545026 |
| Loss 和梯度 | 全部有限 | 全部有限 |
| Backbone 参数张量更新 | 93/93 | 0/93，最大绝对差=0 |
| Classifier 参数张量更新 | 2/2 | 2/2，最大绝对变化约 0.187849 |
| Backbone 梯度 | 有 | 无；SGD 不含 backbone |
| Classifier 梯度 | 有 | 有 |
| BN tracking increment | 每个 BN +85 | 每个 BN +170 |
| BN buffers 更新 | 正常训练更新 | 93/93 buffers 更新，符合 shift_bn=True |
| MixUp / H2T 次数 | MixUp 85，H2T 0 | MixUp 0，H2T 85 |
| Checkpoint 加载 | 从头初始化 | epoch=1；加载后 backbone 参数/buffers、classifier 与源 checkpoint 逐位相等 |
| 全部训练+评估耗时 | 11.18 秒 | 11.60 秒 |

Stage II BS 分支 100 类均出现，每类 83–140 次；Head/Medium/Tail 抽样次数为 **3898/3732/3217**（35.936%/34.406%/29.658%），接近官方分组的均衡期望 36%/35%/29%。IS 次数为 **8922/1639/286**（82.253%/15.110%/2.637%），逐类恰等于原长尾训练分布，证明传入 IS 索引完整遍历。

H2T 第一个真实 batch 形状为 `[128,64,8,8]`。被替换通道为 `[54,47,41,46,19,18,24,7,1,26,27,60,14,28,61,40]`，共 16 个；逐位验证替换后等于同位置 IS 特征，其他 48 个通道等于原 BS 特征，供体未改变，返回对象与接收对象地址相同。所有 Stage II batch 的 CE target 均等于原 BS target；首 batch 损失与普通 CE 逐位一致。

| 10,000 张真实测试图像 | Overall | Head | Medium | Tail |
|---|---:|---:|---:|---:|
| Stage I 1 epoch 后 | 3.580% | 9.917% | 0.029% | 0.000% |
| Stage II 1 epoch 后 | 5.150% | 9.556% | 4.086% | 0.966% |
| 独立 eval-modifed.py 读取 Stage II checkpoint | 5.150% | 9.556% | 4.086% | 0.966% |

每次评估均在 eval/no_grad 下运行，BN buffers 没有变化。Stage II 与独立入口的逐类正确数完全一致。这些准确率来自尚未完成表征学习的 1-epoch backbone，不能用来判断论文 39.55%/47.73%/47.80% 的完整复现成败。

机器可读结果：[stage1.json](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/stage1.json)、[stage2.json](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/stage2.json)、[eval.json](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/eval.json)。完整日志：[Stage I](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/stage1.stdout.log)、[Stage II](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/stage2.stdout.log)、[独立评估](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/eval.stdout.log)。

Checkpoint：

- [Stage I current.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/repro_audit_stage1_1ep_202610092250/ckps/current.pth.tar)，SHA256 `d295dbb8c06d21e793a4e6164803e576fc61d9abdf01040d142012e06065d285`。
- [Stage II current.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/runtime/saved/repro_audit_stage2_1ep_202610092252/ckps/current.pth.tar)，SHA256 `35ea38cbbebe20c8c3192d136c6f975b2e9540cc489f97e250cca0f9274296c8`。

### 8.1 显存

| 阶段 | PyTorch max allocated | PyTorch max reserved | 1 秒监测样本的最大整卡占用 |
|---|---:|---:|---:|
| Stage I | 333.30 MiB | 416.00 MiB | 1924 MiB |
| Stage II | 79.58 MiB | 144.00 MiB | 1641 MiB |
| 独立评估 | 64.66 MiB | 108.00 MiB | 1599 MiB |

整卡占用含 Windows 桌面和其他图形应用，与 PyTorch 分配值不是同一口径；WDDM 下 nvidia-smi 未提供可靠的逐进程占用。PyTorch 峰值来自 CUDA API；整卡值是离散监测样本最大值。本机 nvidia-smi 显示总显存 8151 MiB，batch 128 的短测有充足余量。完整 200 epochs 的长期稳定性尚未实测，显存不会因 epoch 数简单乘以 200。

## 9. 准确运行命令

已验证全部 5 段 PowerShell 命令的语法及正式配置的 YACS 合并结果；检查过程中未执行正式训练。命令显式使用已验证的 pytorch 环境解释器，系统 PATH 的默认 Python 指向未安装 torch 的 Anaconda base 环境。证据见 [命令语法检查](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/powershell_command_check.json)。

### 9.1 本次实际执行的短测

以下都是 PowerShell 命令，工作目录为 `D:\xiangmuxunlian\新`。runner 的训练命令只允许一个 epoch，不启动完整训练。

```powershell
Set-Location -LiteralPath 'D:\xiangmuxunlian\新'
$h2tPython = 'D:\app\anaconda\envs\pytorch\python.exe'
& $h2tPython -u .\H2T_repro_audit\audit_original.py preflight
& $h2tPython -u .\H2T_repro_audit\audit_original.py stage1
$h2tStage1Audit = Get-Content .\H2T_repro_audit\evidence\stage1.json -Raw | ConvertFrom-Json
& $h2tPython -u .\H2T_repro_audit\audit_original.py stage2 --checkpoint $h2tStage1Audit.checkpoint.path
$h2tStage2Audit = Get-Content .\H2T_repro_audit\evidence\stage2.json -Raw | ConvertFrom-Json
& $h2tPython -u .\H2T_repro_audit\audit_original.py eval --checkpoint $h2tStage2Audit.checkpoint.path
```

runner 内部调用的 Stage I cfg 为 `config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml`，传 `num_epochs 1`、空 resume；Stage II 为 `config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml`，传 `num_epochs 1` 和实际 Stage I checkpoint。仅附加审计实验名、现有数据的绝对路径；实际 argv 和最终配置均在 JSON 中。

### 9.2 正式 Stage I：200 epochs，从头训练（未执行）

```powershell
Set-Location -LiteralPath 'D:\xiangmuxunlian\新\H2T_repro_audit\runtime'
$h2tPython = 'D:\app\anaconda\envs\pytorch\python.exe'
& $h2tPython -u .\train_stage1.py --cfg .\config\cifar100\cifar100_imb001_stage1_ce_mixup.yaml data_path 'D:\xiangmuxunlian\新\H2T\data\cifar100' resume "''" name cifar100_imb001_stage1_ce_mixup_formal
```

`resume "''"` 会被 YACS 解码为空字符串，避免误加载 cwd 下的旧 checkpoint；epochs、学习率、优化器、MixUp 等均直接继承官方 YAML/默认配置。保持 gpu=null，不增加 `gpu 0`。若重复正式实验，应把 name 换成唯一实验标识，避免同分钟覆盖。

### 9.3 正式 Stage II：10 epochs，加载完整 Stage I（未执行）

以下定位已完成的正式 Stage I，检查其 current 的 epoch=200，然后使用该运行的 model_best。与短测 checkpoint 完全分开，不把 1-epoch 文件用于正式复现。

```powershell
Set-Location -LiteralPath 'D:\xiangmuxunlian\新\H2T_repro_audit\runtime'
$h2tPython = 'D:\app\anaconda\envs\pytorch\python.exe'
$h2tStage1Run = Get-ChildItem -LiteralPath '.\saved' -Directory -Filter 'cifar100_imb001_stage1_ce_mixup_formal_*' | Sort-Object Name -Descending | Select-Object -First 1
if (-not $h2tStage1Run) { throw '未找到正式 Stage I 运行目录' }
$h2tStage1Last = Join-Path $h2tStage1Run.FullName 'ckps\current.pth.tar'
if (-not (Test-Path -LiteralPath $h2tStage1Last)) { throw 'Stage I current checkpoint 不存在' }
& $h2tPython -c "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); assert c['epoch']==200, 'Stage I has not completed 200 epochs'" $h2tStage1Last
if ($LASTEXITCODE -ne 0) { throw 'Stage I 尚未完成 200 epochs' }
$h2tStage1Best = Join-Path $h2tStage1Run.FullName 'ckps\model_best.pth.tar'
if (-not (Test-Path -LiteralPath $h2tStage1Best)) { throw 'Stage I best checkpoint 不存在' }
& $h2tPython -u .\train_stage2.py --cfg .\config\cifar100\cifar100_imb001_stage2_ce_H2T.yaml data_path 'D:\xiangmuxunlian\新\H2T\data\cifar100' resume $h2tStage1Best name cifar100_imb001_stage2_ce_h2t_formal
```

保留 rho=0.25、shift_bn=True、10 epochs、SGD 及原始 cosine 分类器 LR；这里选择的是完成 200 epochs 的 Stage I 运行中保存的最佳验证 checkpoint，其自身 epoch 字段可能早于 200。阶段连接没有恢复 Stage I SGD 状态，符合作者重新建立 classifier SGD 的路径。

### 9.4 正式模型评估（未执行）

Stage I：

```powershell
& $h2tPython -u .\eval-modifed.py --cfg .\config\cifar100\cifar100_imb001_stage1_ce_mixup.yaml data_path 'D:\xiangmuxunlian\新\H2T\data\cifar100' resume $h2tStage1Best name cifar100_imb001_stage1_eval_formal
```

Stage II：

```powershell
$h2tStage2Run = Get-ChildItem -LiteralPath '.\saved' -Directory -Filter 'cifar100_imb001_stage2_ce_h2t_formal_*' | Sort-Object Name -Descending | Select-Object -First 1
if (-not $h2tStage2Run) { throw '未找到正式 Stage II 运行目录' }
$h2tStage2Checkpoint = Join-Path $h2tStage2Run.FullName 'ckps\model_best.pth.tar'
if (-not (Test-Path -LiteralPath $h2tStage2Checkpoint)) {
    $h2tStage2Checkpoint = Join-Path $h2tStage2Run.FullName 'ckps\current.pth.tar'
}
if (-not (Test-Path -LiteralPath $h2tStage2Checkpoint)) { throw 'Stage II checkpoint 不存在' }
& $h2tPython -u .\eval-modifed.py --cfg .\config\cifar100\cifar100_imb001_stage2_ce_H2T.yaml data_path 'D:\xiangmuxunlian\新\H2T\data\cifar100' resume $h2tStage2Checkpoint name cifar100_imb001_stage2_eval_formal
```

## 10. 正式训练就绪判断

**可以在用户确认后开始 Stage I 200 epochs。** 启动阻断已通过独立兼容补丁解决，真实数据、CUDA、前向/反向、参数更新、阶段 checkpoint 连接、双采样、原始通道替换、标签和四项评估均已实测通过。

正式任务应采用本报告固定的原始快照和兼容 Diff，并明确使用仓库发布配置 rho=0.25；若要逐项采用论文最佳 rho=0.3，需要用户明确选择，当前没有自动改变。本次 Stage II 对 BN 的更新是作者默认行为，不能以“完全冻结所有状态”的描述代替。

当前工作停留在审计和 1+1 epoch 短测。200-epoch Stage I、10-epoch Stage II、论文准确率验证及中断续训支持均未自动启动或补做。完整训练时需要保留该次实际运行的源码、配置、日志和 checkpoint；作者的 resume 不构成精确续训。

最终一致性与证据哈希见 [final_verification.json](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/final_verification.json) 和 [execution_source_manifest.json](D:/xiangmuxunlian/新/H2T_repro_audit/evidence/execution_source_manifest.json)。
