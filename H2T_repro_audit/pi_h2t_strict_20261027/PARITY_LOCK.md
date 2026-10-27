# PI-H2T v2 训练前一致性与协议锁

日期：2026-10-27（Asia/Shanghai）。目标分支：`repro/pi-h2t-cifar100-paper-parity-20261027`；开始HEAD：`5bdbe5c8749cafb613215e157e7b3290f70626d6`。正式run-id：`pih2t_if100_seed123_42_v1`。

本文件在本轮任何新模型的官方CIFAR100 TEST准确率产生前建立。执行目标是**PAPER_ALIGNED_INDEPENDENT_RECONSTRUCTION_NOT_AUTHOR_VERIFIED**；准确率接近不能替代作者实现/协议核实。

## 来源

原论文：<https://arxiv.org/pdf/2506.00625v2>，2026-07-18，17页。已取得全文并在前轮按页对照，本轮确认缓存SHA-256为`b68e80b98b31ad2a317f6462cab93e59f15ac4f456b5bbd1d23386339f3d5dc8`；核心是PDF p4 Eq3/6、p5 Eq7/8及footnote2、p16 Algorithm1。复核依据还包括前轮`AUTHOR_MECHANISM_CROSSWALK.md`、`PI_PAPER_CODE_PARITY.md`和原始component audit报告。理论段/pseudocode记号差异不被用作替换已核实Eq3/6/7/8的依据。

原作者库：<https://github.com/Keke921/H2T>。本轮只读`git ls-remote --heads`得到唯一公开head main=`e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`。这是分支身份旁证，**没有证明它是完整PI作者训练代码**。训练直接依赖本地冻结`H2T_official_clean/H2T-main/`的模型/数据构造器，其内容哈希另记录，绝不修改。

## 字段锁定

|字段|固定选择|证据类别|
|---|---|---|
|PIF输入/输出|CIFAR最后卷积图[B,64,8,8]，输出同形状，位于最后stage后/GAP前|VERIFIED_FROM_PI_PAPER（Eq3/6）；具体64/8来自冻结ResNet32|
|PIF均值/拼接|mean(dim=1,keepdim=True)，[F-mean_C(F),F]→[B*C,2,H*W]|VERIFIED_FROM_PI_PAPER（p4 Eq3–6）|
|PIF参数|单个Conv1d(2,1,kernel_size=1,bias=False)，总计2参数|VERIFIED_FROM_PI_PAPER（p5）；不用历史25参数结构/权重|
|PIF初值|PyTorch Conv1d默认初始化，StageI从零训练|RECONSTRUCTION_ASSUMPTION（作者精确初值UNVERIFIED）|
|PIF性质|均值对通道置换不变，含原F项的整个输出是等变，不宣称完全不变|VERIFIED_FROM_PI_PAPER的公式代数解释|
|H2TF|GAP后f_tilde=r*f_receiver+(1-r)*f_donor|VERIFIED_FROM_PI_PAPER（p5 Eq7，p16算法）|
|r/lambda|r=(1-cos(f_receiver,w_y))/2，接收者保留权重；lambda=1-r为供体权重|VERIFIED_FROM_PI_PAPER（Eq8、理论边界归一化脚注2的字面代数）|
|比例对W求导|ratio位于no_grad；冻结StageI W作为reference|RECONSTRUCTION_ASSUMPTION（作者detach/reference语义UNVERIFIED）|
|监督|CE；仅接收者hard label，不混合供体标签|VERIFIED_FROM_PI_PAPER（p5 Sec3.4）|
|数据|现有官方缓存`D:/xiangmuxunlian/xin/H2T/data/cifar100`，CIFAR100 TRAIN构造LT，IF100 exp，selection seed0|INHERITED_FROM_H2T（冻结dataset）；不下载/不换train-dev|
|StageII采样|BS接收者/IS供体，dataset先保留原索引供体再均衡接收者|VERIFIED_FROM_PI_PAPER（Eq9）+INHERITED_FROM_H2T|
|骨干/头|冻结ResNet32 feature factory（[5,5,5] BasicBlock，Option-A）；原GAP+带bias普通Linear(64,100)|INHERITED_FROM_H2T；作者未列全bias/shortcut细节|
|StageI|seed123，200epochs，batch128，CE+输入MixUp(alpha1)，SGD .1/.9/wd.0002|PI论文基础设置+INHERITED_FROM_H2T|
|StageI调度|5轮warmup(.02,.04,.06,.08,.1)，epoch161起.01，181起.001|INHERITED_FROM_H2T（默认multistep，源码epoch+1后>160/>180），不是cosine|
|StageII|seed42，两组各10epochs，batch128，classifier-only SGD，initial lr.02/momentum.9/wd.0005，cosine|INHERITED_FROM_H2T（含.2倍率）；作者全部细节未完全核实|
|BN|默认update，两组都在train/no_grad中按receiver→donor前向；骨干/PIF参数固定，BN buffers可更新|INHERITED_FROM_H2T + RECONSTRUCTION_ASSUMPTION（PI作者buffers策略UNVERIFIED）|
|两组唯一干预|no_fusion直接用receiver向量；h2tf用Eq7；两组都算r并都前向donor，不引入其他机制|RECONSTRUCTION_ASSUMPTION（matched BN traffic对照，**不是作者未改动DR路径**）|
|随机/数值|禁AMP，batch不降；cuDNN benchmark=False、deterministic=True；实际TF32/backend设置由环境核验记录，不按精度修改|RECONSTRUCTION_ASSUMPTION（配对可重复性控制）|
|checkpoint|本run新训练fixed epoch200 `stage1_final_epoch200.pth`；两组严格载入同一SHA、同一FC初值|RECONSTRUCTION_ASSUMPTION（作者best选择路径UNVERIFIED）；严禁历史25参数替代|
|评估|只在StageI epoch200和每组StageII epoch10汇报官方TEST，不择best、不按TEST改公式/设置；100图/类与label顺序校验|RECONSTRUCTION_ASSUMPTION（报告协议，不宣称独立未使用测试集）|

本轮核心公式没有选择另一个版本或反转r的自由度。作者未公开/未核实项保持UNVERIFIED，并在上述透明独立重建范围内执行。CPU测试、数据真实性、CUDA/真实单batch及配对/来源门禁未通过时均不得进入正式训练；OOM、非有限值、数据/代码/checkpoint漂移立即停止。所有修复仅在当前study目录，先记录失败及回归验证，再固化代码哈希，正式StageI开始后不修改执行源码。
