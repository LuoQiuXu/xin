# CIFAR-100N-LT 协议核查

本协议在最终评价前记录。唯一分支为 `experiment/h2t-cifar100n-lt-failure-screen-20261029`；起始 HEAD 为 `23c1281fcd376d788d2ff996ad288613de3064dd`。训练代码及作者来源文件的 SHA256 已在 `TRAINING_LOCK.json` 冻结，正式训练进程逐次校验。没有运行作者的 `main()`、官方 TEST loader 或 best-checkpoint 选择逻辑。

## 数据门槛

- 人工标签来自 [CIFAR-N 作者仓库](https://github.com/UCSC-REAL/cifar-10-100n) 的 Python/PyTorch 文件 `data/CIFAR-100_human.pt`。下载字节与官方 Git blob `5dac9db8af8f7224f982fbac9794c67f60f8bd45` 一致；实际 SHA256、来源 URL、安全反序列化信息见 `INPUT_ATTESTATION.json`。没有使用 TFDS 重排标签或第三方数据副本。
- `clean_label` 与本地官方 Python TRAIN 的全部 50,000 个 fine label 逐项相等。固定 LT raw-ID 与 seed0 的官方逐类 shuffle 截取一致；10847 张 LT 和1980张固定 TRAIN holdout raw-ID 无交集，像素缓存逐项等于 raw TRAIN，跨集合完全相同的像素图像为0。
- 使用既有 `clean_s1_seed42_vs_seed123_20261013/split_indices.npz`；没有重新划分验证集。该验证集缺 class0，其他99类各20张。
- 本轮 `prepare_protocol.py` 原文件未改动，真实输出在 `protocol/DATA_AUDIT.json`。LT 噪声为3902/10847=35.9730801%，全量为40.2%；不可混用。观测类频14–445，无空类。按相同计数阈值，clean类别为35/35/30，noisy观测类别为39/58/3，L1计数变化4012。
- 只在数据准备/审计阶段读取全部 clean truth。训练使用独立 images-only 缓存和本条件 `*_train.npz`；Noisy 的 targets、receiver 类池及类频只由 `noisy_train.npz` 生成。训练进程 open 审计钩子拒绝人工原始文件、raw TRAIN、clean truth 中间件及任何验证输入；Noisy 还拒绝 `clean_train.npz`。每次真实运行保留允许的数值文件读取清单及拒绝记录。

## 论文—实际调用路径

[H2T 原论文](https://ojs.aaai.org/index.php/AAAI/article/view/29262/30382) 的 Basic Settings（PDF第5页，印刷页13585）给出 CIFAR 的 ResNet32、200/10 epochs 等基本预算。具体 MixUp、优化器、BN 及随机通道实现以冻结的作者代码/YAML 为直接证据；不把本文的开发集实验等同论文官方测试指标。

| 项目 | 作者证据（本仓库 `H2T_official_clean/H2T-main/` 下） | 本轮实际调用及状态 |
|---|---|---|
| Backbone/分类头 | `models/resnet_cifar.py:114–171` | 直接导入 `resnet32_fe()` Option A 和 `Classifier(64,100)`；GAP+带bias FC，无PIF、归一化头或可学习缩放 |
| Stage I 核心 | `train_stage1.py:285–383` | AST原样提取 `train` 函数，未改函数体；原 CE 与 `methods.mixup_data/mixup_criterion`；alpha=1 |
| Stage I SGD | `train_stage1.py:231–234,457–475`；Stage I YAML | 两组参数（BB、FC），SGD lr=.1、momentum=.9、wd=.0002；原调度器5轮warmup，epoch161/181开始衰减 |
| Stage II 双采样 | `datasets/cifar100.py:74–127`；Stage II YAML | 原 `__getitem__` 函数体；DataLoader shuffle 的 instance donor，Python随机均匀类+类内随机 receiver。先增强 donor，后增强 receiver。没有更换 donor 策略 |
| Stage II BN | `train_stage2.py:303–318,354–378`；Stage II YAML `shift_bn: True` | BB `.train()` + `no_grad()`；两臂同样 receiver→donor 两次前向。BB所有参数冻结；31层BN的运行统计照原设置更新 |
| H2T 融合 | `methods.py:31–50` | 直接调用原 `H2T`；CPU `torch.randperm(64).cuda()`，随机batch-shared16个完整通道替换，rho=.25；非标量混合 |
| No-Fusion 配对 | 任务清单步骤2明确要求同BN路径 | 保留同一两次前向，融合回调只返回 receiver。消耗同样的未使用 `randperm(64)`，使随后采样/增强流不因关融合改变；有效替换为0。不是少一次 donor forward 的原始 rho=0 快捷路径 |
| Stage II 监督/SGD | `train_stage2.py:239–244,354–406,487–498` | 原hard receiver CE；单FC参数组，SGD momentum=.9、wd=.0005，原cosine，首轮实际lr=.1×.2=.02；不启用MixUp或额外loss |
| Checkpoint/预算 | 本任务固定epoch要求 | Clean/Noisy从同一个新建seed42随机初始化分别200轮；同regime两臂从同一epoch200 BB+FC开始，空SGD动量；seed124/125各两臂10轮，85 updates/epoch |
| 末轮评价 | 本任务步骤3 | 两个Stage I与八个Stage II全部结束并锁定SHA后，才构建验证输入；每个最终模型单次eval，禁止best选择 |

## 显式适配与计算设置

1. 原始 CIFAR wrapper 会实例化官方 TEST、生成split并隐式重置 NumPy seed0；本轮按任务复用锁定 split，替换为 images-only/observed-label IO 适配。训练随机源明确设置为本次seed42（Stage I）或124/125（Stage II），包括NumPy；不声称逐比特重放原作者seed123脚本。原始增强、采样getitem、MixUp函数及学习率调度函数保持原样。
2. 原作者 YAML 的 Stage I `deterministic=True`、Stage II `False`分别保留。统一 `cudnn.benchmark=False`、`cudnn.allow_tf32=True`、`cuda.matmul.allow_tf32=False`，FP32，无AMP。解释器 `D:\app\minicondaaa\envs\xiaoju\python.exe`，Torch2.14.1+cu132，CUDA13.2，cuDNN92400，RTX5070 Laptop GPU；每次 `START.json` 记录生效值。现代软硬件环境不等同原作者论文环境。
3. `AuditedSGD` 在原SGD更新前额外检查梯度有限性，不改变SGD公式。无梯度裁剪、EMA、额外正则、阈值修改或混合精度。
4. 正式学习之前5项CPU契约测试、语法检查及6个真实单批次干跑通过。之后补充4项CPU训练契约检查，共9项通过。干跑权重独立保存，正式训练未复用干跑更新后的模型。
5. 清单中的SHA256按本地实际文件字节计算。仓库启用了Git文本换行归一化，部分文本的Git内容字节与本地CRLF字节不同；`SOURCE_EOL_AUDIT.csv` 同时记录锁定来源/工具的本地SHA、Git索引内容SHA及LF归一化SHA，并核对差异仅为换行。未转换训练时的源文件；二进制数据与模型SHA不涉及文本换行规则。

## 配对门禁与评价口径

- 同regime+seed的H2T在**每个FC更新前**，逐项精确比较 No-Fusion 的 receiver/donor raw-ID、hard target、增强图像、CPU RNG、两次BB特征及mask的SHA256；不接受allclose替代。
- 每epoch检查更新次数和31层BN计数，Stage II BB参数保持不变且无梯度。两臂最终BN状态SHA必须相等。任何失败先保留 `FAILURE.json` 后停止；不自动放宽、续跑或更换配置。
- Many>100、Medium20..100、Few<20，始终使用 benchmark clean true counts；不按Noisy观测类别重新分组。验证分母固定1980/680/700/600，class0缺失明确记录。
- 该20样本边界按本任务清单执行：class69的20张归Medium。论文第5页写的是Medium>20、Tail≤20，作者YAML又使用固定索引区间；三者不可混称完全相同的分组指标。本轮不把统计分组差异用于改变训练或进行结果选择。
- 主结果只取Stage II epoch10，`stage1_sha`、验证raw-ID哈希、组分母及epoch交给原 `compare_outcomes.py` 校验。Stage I epoch200评价只作描述。
- 所有模型/输入数组/logits和逐batch大日志只留本地，提供SHA索引；小体积epoch CSV、来源清单、协议审计与科学报告提交同一分支。

## 预先固定的科学判定

`D_g(regime)=Acc(H2T)-Acc(NoFusion)`，`Interaction_g=D_g(noisy)-D_g(clean)`。

只有两个Stage II seed均满足 Few交互≤−2pp且Noisy条件H2T Few严格低于同源No-Fusion，才记为候选失效并建议另一独立Stage I seed确认；不授权新模块。有效运行未满足条件则NO-GO；有效性失败则BLOCKED/INCONCLUSIVE。

Noisy改变监督标签及观测类频，balanced receiver流会随之改变；因此这是整个真实噪声场景效应，不能分离为纯label-loss效应、donor效应或某个融合机制根因。两种Stage II seed共享一个Stage I，不是独立表示训练重复。固定TRAIN开发集曾在历史研究使用，缺class0且非独立官方TEST；不报告论文级显著性或推广到PI-H2T。
