# Codex任务：Clean Stage I固定下 Low-VS vs Uniform 多种子稳定性验证

## 一、实验目的

当前已经完成：

1. H2T原始方法复现。
2. SMART协方差通道引导实验。
3. Variation-Shift (VS) 通道机制设计。
4. Clean Stage I重新训练，避免历史TEST选择泄漏。
5. Clean backbone下首次VS Stage II实验。

当前观察：

Clean Stage I + seed123:

Uniform-PerSample:
Overall = 47.0707
Tail = 26.3793
Few = 27.00


Low-VS:
Overall = 47.3737
Tail = 27.4138
Few = 28.00


Low-VS相对于Uniform:

Overall +0.3030 pp
Tail +1.0345 pp
Few +1.0 pp


当前假设：

H2T收益可能来自具有方向性的跨类别特征扰动，而不是简单减少供体偏移。

但是seed123单次结果不足以证明。

本轮目标：

验证：

Low-VS是否在不同随机种子下稳定优于Uniform-PerSample。


==================================================
二、本轮只做稳定性验证
==================================================

只运行两个方法：

A:
Uniform-PerSample


B:
Low-VS


不运行：

- Original-H2T
- High-VS
- Shuffled-VS
- SMART
- 其他采样策略


不修改VS公式。

不修改候选池。

不修改统计规则。


==================================================
三、固定实验基础
==================================================

必须复用：

Clean Stage I Backbone


路径：

<当前clean_stage1_best.pth>


禁止：

重新训练Stage I。


必须复用：

Clean Stage I重新计算得到的统计文件：

- class mean
- class variance
- variance shrinkage
- epsilon


禁止：

使用旧TEST选择Stage I生成的统计文件。


==================================================
四、Stage II训练配置
==================================================

保持与seed123完全一致。


Dataset:

CIFAR-100-LT

IF=100


Backbone:

ResNet-32


Stage I:

冻结：

- Backbone
- BN running statistics
- BN affine参数


只训练：

Classifier


Loss:

CrossEntropy


Optimizer:

SGD


epoch:

10


batch size:

128


rho:

0.25


接收者：

BS


供体：

IS


每次融合：

16/64 channels


特征：

最后层[64,8,8]


==================================================
五、随机种子设置
==================================================

本轮运行：

seed=124

seed=125


要求：

所有随机源同步设置：

- python random
- numpy
- torch
- cuda


每个seed独立训练。

禁止：

使用seed123结果作为初始化。


==================================================
六、Uniform-PerSample定义
==================================================

保持原实现：

对于每个接收样本：

从64个channel中：

均匀随机

无放回选择16个channel。


替换：

receiver channel

←

donor对应channel


保持：

相同IS donor序列。


==================================================
七、Low-VS定义
==================================================

严格复用seed123版本。


评分：

S_k =


v_h,k /
(v_h,k + (mu_h,k - f_t,k)^2 + epsilon)


其中：

v_h,k:
供体类别方差


mu_h,k:
供体类别均值


f_t,k:
当前训练batch中接收样本GAP后的feature


epsilon:

使用clean统计文件中的固定值。


选择：

Bottom48 channels


然后：

从Bottom48中均匀随机选择16个channel。


保持：

- 同样receiver
- 同样donor
- 同样batch顺序
- 同样augmentation


只改变channel selection。


==================================================
八、实验公平性要求
==================================================

Uniform和Low-VS必须共享：

1.
同一个Clean Stage I checkpoint


2.
同一个classifier初始化


3.
同一个训练数据顺序


4.
同一个receiver batch


5.
同一个IS donor


6.
同一个augmentation


7.
同一个optimizer


8.
同一个scheduler


9.
同一个epoch数量


唯一差异：

channel mask。


保存：

pairing_hash.json


证明：

两组使用相同训练输入。


==================================================
九、Dry-run要求
==================================================

正式训练前：

每个seed执行一次真实batch检查。


验证：

1.
模型加载正确。


2.
Backbone无梯度。


3.
BN状态不变化。


4.
Classifier可以更新。


5.
Uniform每个样本16个channel。


6.
Low-VS每个样本16个channel。


7.
未选channel保持receiver。


8.
选中channel来自donor。


9.
loss无NaN。


10.
GPU显存正常。


dry-run结束：

重新加载clean Stage I checkpoint。

清空optimizer状态。

再开始正式训练。


==================================================
十、训练记录
==================================================

每个epoch保存：

- train loss
- validation Overall
- Head
- Medium
- Tail
- Many
- Medium-n
- Few


保存：

seed124:

uniform_seed124/
lowvs_seed124/


seed125:

uniform_seed125/
lowvs_seed125/


==================================================
十一、结果输出
==================================================

生成：

CLEAN_VS_MULTI_SEED_STABILITY_REPORT.md


包含：


## 1. 实验配置

说明：

Clean Stage I

无TEST选择


## 2. 固定epoch10结果


表：

|Seed|Method|Overall|Head|Medium|Tail|Few|
|-|-|-|-|-|-|-|
|124|Uniform||||
|124|Low-VS||||
|125|Uniform||||
|125|Low-VS||||


## 3. 增益统计


计算：

Low-VS - Uniform


分别报告：

Overall delta

Tail delta

Few delta


## 4. 类别级分析


比较：

哪些tail类别提升。

哪些类别下降。


输出：

per_class_delta.csv


## 5. 判断


明确回答：


问题1:

Low-VS是否在两个seed均优于Uniform？


问题2:

Tail/Few提升是否稳定？


问题3:

是否支持：

“方向性扰动可能是H2T收益来源之一”


问题4:

是否值得继续进行论文方法设计？


==================================================
十二、停止条件
==================================================

完成seed124和seed125后停止。


禁止：

- 增加seed126
- 调Top48
- 改评分公式
- 调epsilon
- 调融合比例
- 改训练epoch
- 使用TEST


等待下一步分析。


最终目标：

不是寻找最高准确率。

而是验证：

Low-VS提升是否可重复。