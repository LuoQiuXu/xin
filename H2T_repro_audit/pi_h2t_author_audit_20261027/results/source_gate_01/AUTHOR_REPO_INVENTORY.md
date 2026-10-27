# 作者仓库清单与调用链核查

结论：**已找到作者公开的PIF/H2Tv2片段，未在所核查作者指定公开来源定位到与arXiv v2相符、可直接运行的完整PI-H2T两阶段发布。** 不作“作者从未发布”或全球不存在的断言。

## 身份与覆盖范围

作者v2 PDF p1源码声明直接指向[该仓库](https://github.com/Keke921/H2T)。实时GitHub branches/API及git ls-remote --heads --tags一致：只有main，HEAD=e0c849235d9655fca31cd9fa8a5f38f92a77c7e8；没有advertised tag；releases API为空。完整递归树58项、truncated=false，具体每项路径/Git对象SHA见AUTHOR_TREE.json与AUTHOR_REPO_INVENTORY.csv。

main可达提交API返回79项（per_page=100），从2023-03-06初始提交到2025-03-22 README更新；最早项无父提交，当前链全部可见。已逐项核查提交日期/消息/父链，重点读取了以下两次源码diff：
- [3dafa708，2024-10-16，Add PIF](https://github.com/Keke921/H2T/commit/3dafa70841e96034477698639db5169ceda195f8)；
- [f96309af，2024-10-14，methods.py加入H2Tv2](https://github.com/Keke921/H2T/commit/f96309af41d1c8bada5c4c80e9ac25310a2ade51)。

41个Python/YAML/README/LICENSE文本全部按“blob <字节数>\0<UTF8字节>”复算Git blob SHA1并与固定树匹配；另记录SHA256。33份Python作AST解析。空__init__.py由树的size=0和标准空blob SHA确认，连接器未返回普通文本内容，未伪造下载成功。其余7个blob是四份ImageNet/Places数据清单及三份AAAI slides/poster文件，只核对树身份，未下载。未将AST/关键词筛查冒充全文件运行验证。

## 作者真正公开了什么

|部分|固定源码|实际证据|判定|
|---|---|---|---|
|项目身份|[README.md:15-24](https://github.com/Keke921/H2T/blob/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8/README.md#L15-L24)|自述AAAI2024 sample code，原H2T为核心；PI论文链接同库不能改变当前运行入口身份|原始H2T公开实现|
|通道替换H2T|[methods.py:31-47](https://github.com/Keke921/H2T/blob/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8/methods.py#L31-L47)|按int(rho*C)随机挑通道，原位复制供体整张map|不是Eq7全向量插值|
|PIF构造|[models/resnet_cifar.py:128-145](https://github.com/Keke921/H2T/blob/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8/models/resnet_cifar.py#L128-L145)|Conv2d非法feature_tuning参数；feature_tuning无形参/模块绑定；三层Conv1d共25参数|公开但不完整/不可直接构造|
|PIF返回|[models/resnet_cifar.py:156-168](https://github.com/Keke921/H2T/blob/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8/models/resnet_cifar.py#L156-L168)|计算out_hat但return out，未把PIF结果接到分类器|活动PIF未成立|
|H2Tv2|[methods.py:50-67](https://github.com/Keke921/H2T/blob/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8/methods.py#L50-L67)|有线性向量混合；dis变换为.2+.8/(1+dis)；参数dis由外部提供|作者原型片段；不是经证明的Eq8实现|
|StageI|[train_stage1.py:113-115](https://github.com/Keke921/H2T/blob/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8/train_stage1.py#L113-L115)，[train_stage1.py:320-335](https://github.com/Keke921/H2T/blob/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8/train_stage1.py#L320-L335)|配置选resnet32_fe→ResNet_fe；MixUp→model→Classifier；构造在CPU已抛TypeError|不能直接跑PI StageI|
|StageII|[train_stage2.py:305-378](https://github.com/Keke921/H2T/blob/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8/train_stage2.py#L305-L378)|CIFAR/无MixUp支路model(receiver/donor)→H2T(...rho=config.H2T)→Classifier→receiver CE；BN由shift_bn控制|仍是原H2T活动路径|
|配置|[config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml:8-45](https://github.com/Keke921/H2T/blob/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8/config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml#L8-L45)，[config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml:14-60](https://github.com/Keke921/H2T/blob/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8/config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml#L14-L60)|原CE+MixUp、H2T=.25、shift_bn=True；无PIF开关/两参数初始化或H2Tv2集成|不能标为完整PI配置|
|模型链接|[README.md:92-119](https://github.com/Keke921/H2T/blob/e0c849235d9655fca31cd9fa8a5f38f92a77c7e8/README.md#L92-L119)|Results and Models标注CE+MixUp/CE+H2T及AAAI结果；未绑定论文PIF42.19或PI49.04的epoch/SHA|作者PI checkpoint UNVERIFIED|

唯一原型CPU探针只执行已检查的ResNet_fe类定义并尝试构造，没有导入训练入口、forward、反传、数据或CUDA查询。实际异常：Conv2d.__init__() got an unexpected keyword argument 'feature_tuning'。这是公开快照可运行性证据，不是本地历史准确率下降的因果证明。feature_tuning未定义和return out属于另两项静态问题，未通过修改作者代码来“依次跑通”。原文件和冻结档案均未改。

## 实际调用路径与没有完成的连接

StageI parse_args/config→main_worker(getattr resnet32_fe)→ResNet_fe→最后conv stage→feature map→Classifier(GAP+Linear)。PIF分支既构造有错，返回值也不生效。StageII config H2T=.25/mixup=False→dataset原索引IS供体+均衡接收者→两个model前向→methods.H2T→Classifier→hard CE。H2Tv2没有在33份Python的AST调用中出现，也没有被所选YAML动态指名；实际分支逐段检查证实仍调用H2T。这一结论同时依据配置、工厂、数据返回值、融合函数、分类器返回值和优化器边界，而非仅搜索名字。

三层Conv1d之间没有激活，因此可以折叠为仿射混合；不能仅由25参数或拼接顺序倒置就称其数学不可表示Eq3。这里的确定差异是参数化/bias、构造失败及输出未接入，而非宣称作者发表方法存在精度缺陷。

## 冻结档案与作者当前main的区别

同41个文本逐字节比较（仅归一化CRLF）发现3个不同路径：models/resnet_cifar.py、utils/Unzip.py、utils/util.py。冻结ResNet_fe没有公开main的PIF草稿，构造可被本地wrapper使用；本地两参数PaperPIF是我们额外写入的独立实现。不能把H2T_official_clean档案或PI_H2T_review/H2T/中的活动PIF，当成当前main已经可运行的证明。所有文件哈希与差异范围见PUBLIC_SOURCE_STATIC_AUDIT.json。

## 其他作者来源和范围边界

[作者publications页](https://keke921.github.io/publications/)把TPAMI PI-H2T列为论文链接；AAAI H2T条目另带该GitHub code链接。arXiv v2 PDF与作者TeX包只给Keke921/H2T这一代码URL，TeX包没有.py/.yaml/.sh/.ipynb或checkpoint。GitHub作者仓库名检索返回25项；独立PI-H2T仓库名查询为空，二者只是检索范围记录，并非全球穷尽证明。没有将第三方复现、我们自己的xin、LIFT等同名/相关fork冒充作者PI发布。

期刊正式网页/PDF本轮未取得可读全文，独立IEEE补充附件是否另有代码仍UNVERIFIED；不把HTTP202空体、418或索引无结果等同于作者未公开。没有访问私有代码、联系作者、创建issue/PR、修复作者源或运行GPU。
