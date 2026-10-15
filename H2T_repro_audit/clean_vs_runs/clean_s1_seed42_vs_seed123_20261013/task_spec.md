# Codex任务：重新训练 Clean Stage I Backbone，并复现实验 VS Stage II 三组验证

## 一、任务目的

当前 VS Stage II(seed123)实验已经完成。

数据协议审计发现：

- CIFAR-100-LT 数据和划分正确。
- Stage II没有直接使用官方TEST。
- 但是Stage I backbone的best checkpoint曾根据官方TEST accuracy选择。

因此当前实验存在历史model selection leakage。

本任务目标：

重新训练一个完全clean的Stage I backbone：

1. 不读取官方CIFAR-100 TEST标签。
2. 不使用TEST accuracy选择epoch。
3. 使用TRAIN留出validation进行checkpoint选择。
4. 在clean backbone基础上重新训练VS Stage II。

本轮只验证核心机制：

Uniform-PerSample
High-VS
Low-VS

不运行：
- Original-H2T
- Shuffled-VS
- 其他供体实验
- SMART实验


==================================================
二、严格实验原则
==================================================

1. 新实验必须独立目录。

禁止覆盖：

- 原Stage I checkpoint
- 原VS seed123结果
- 原SMART结果
- 原H2T复现实验


2. 官方TEST禁止参与：

禁止：

- 加载CIFAR100(train=False)
- 计算TEST accuracy
- 根据TEST选择epoch
- 保存TEST best checkpoint


允许：

- 文件完整性MD5校验
- 不读取test图像和标签


3. 所有模型选择必须基于：

CIFAR-100 TRAIN中预留validation

即：

1980 validation images

class 1-99

class0缺失

沿用已有split_indices.npz。


==================================================
三、Stage I Clean重新训练
==================================================

数据：

CIFAR100-LT

IF=100

ResNet-32


训练数据：

严格使用已有LT：

10847 images

必须验证：

raw indices
label distribution
pixel hash

与之前Stage I一致。


Validation：

1980 TRAIN留出validation

只用于：

epoch选择
early stopping记录


禁止：

TEST evaluation


==================================================
四、Stage I训练配置
==================================================

保持与之前Stage I一致。

不要改变训练recipe。

包括：

Backbone:
ResNet-32


Optimizer:
SGD


learning rate:
0.1


momentum:
0.9


weight decay:
保持原配置


epochs:
200


batch size:
保持原配置


loss:
保持原Stage I配置


data augmentation:
保持原配置


seed:

使用新的固定seed：

42


训练过程中：

每epoch:

1.
训练LT

2.
validation evaluation

3.
保存：

best_validation_overall checkpoint


同时保存：

epoch_validation_metrics.csv


必须记录：

epoch
train loss
validation overall
validation head
validation medium
validation tail


最终输出：

clean_stage1_best.pth


==================================================
五、Stage I必须完成协议审计
==================================================

训练完成后输出：

CLEAN_STAGE1_PROTOCOL.md


必须回答：

1.
训练期间是否加载TEST？

2.
best checkpoint依据什么指标？

3.
validation和training是否重叠？

4.
LT数量分布是否正确？

5.
checkpoint hash是多少？


如果发现TEST调用：

立即停止，不进入Stage II。


==================================================
六、Clean Stage I之后执行Stage II VS
==================================================


只运行三组：


A.
Uniform-PerSample


B.
High-VS


C.
Low-VS



共同条件：

固定：

clean Stage I backbone


冻结：

Backbone

BN


只训练：

Classifier


保持：

CE loss

SGD

10 epochs

batch size=128


保持：

IS供体

BS接收者


rho=0.25


每次：

替换16/64 channel


==================================================
七、VS机制保持原实验版本
==================================================

不要修改：

评分公式：

S_k =
v_h,k /
(v_h,k +(mu_h,k-f_t,k)^2+epsilon)


不要修改：

Top48候选池


不要修改：

小样本fallback规则


不要修改：

统计文件


统计量：

只能来自LT训练集


禁止：

使用validation/test计算类别统计。


==================================================
八、Stage II三组定义
==================================================


1.
Uniform-PerSample

每个样本：

64 channel随机选择16


均匀随机


----------------


2.
High-VS


根据当前接收样本feature计算S_k。


选择：

Top48 channel


随机选择16 channel


----------------


3.
Low-VS


选择：

Bottom48 channel


随机选择16 channel



==================================================
九、Stage II公平性要求
==================================================

三组必须共享：

same:

- clean Stage I checkpoint
- classifier initialization
- receiver images
- IS donor images
- data augmentation
- batch order
- optimizer
- learning rate


不同：

只有channel selection mask


保存：

sampling_masks.npz


并验证：

每个sample：

16 unique channels


==================================================
十、Stage II输出
==================================================


输出：

CLEAN_VS_STAGE2_RESULTS.md


包含：


固定epoch10:

|Method|Overall|Head|Medium|Tail|Few|
|-|-|-|-|-|-|
|Uniform|
|High-VS|
|Low-VS|


同时输出：

best validation epoch


per-class accuracy


tail class变化


sampling statistics


==================================================
十一、最终判断标准
==================================================


不要只看Overall。


重点：

1.

High-VS 是否 > Uniform


2.

Low-VS 是否 > Uniform


3.

是否保持之前Low-VS优势趋势


4.

Tail/Few是否稳定提升


如果clean backbone下：

Low-VS仍然提升：

说明VS机制不是依赖TEST选择的偶然现象。


如果优势消失：

说明之前提升可能来自：

- backbone选择偏差
- validation过拟合
- 随机波动


==================================================

完成：

1. clean Stage I训练
2. clean协议审计
3. Uniform/High-VS/Low-VS Stage II

后停止。

不要自动增加seed。
不要修改公式。
不要调参。
不要使用官方TEST。