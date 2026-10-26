# 作者机制对照：H2T、PI-H2T、FeatRecon、SEL、DBG

日期2026-10-26；读本与SHA见`PAPER_ACCESS.json`。页码一律为所下载PDF的物理页码。五个指定主来源均取得完整PDF：H2T AAAI9页；PI-H2T **arXiv:2506.00625v2，2026-07-18，17页**；FeatRecon ICLR33页（含附录）；SEL ICCV11页；DBG CVPR10页。另外取得H2T arXiv v3 13页用于补充Listing1/Table7，不能与AAAI主文页码混用。全文按页抽取并查阅正文/附录，关键机制页另渲染复核；参考文献只作索引，数学证明的阅读不等于独立形式化验证。逐页范围见`PAPER_PAGE_LEDGER.csv`。本轮未使用PI摘要代替公式，也未把CVF网页403误写成PDF不可取得：指定解释器的直接PDF请求已成功。

## 1. H2T：阶段II的表示利用，不是新的表示学习器

**作者主张**：尾类语义稀疏导致分类边界偏置；在固定表示上借用头偏置分支语义调整分类头。[AAAI原文](https://ojs.aaai.org/index.php/AAAI/article/download/29262/30382)，PDF pp3–4 Methodology、Eq1–2、Algorithm1。输入为池化前图F_B/F_I，输出为部分整通道替换的图；监督只有接收分支标签。Eq2是BS=1/C、IS=n_i/N，不承诺只抽头/尾类。

**公式/代码边界**：Eq1的M_p极性与邻文计数存在保留/替换歧义；arXiv v3 p10 Listing1和冻结`methods.py:31-47`明确复制int(rho*C)个供体通道，batch共享随机索引。不能用PI-H2T的全向量线性插值反过来描述原始H2T。

**假设及已有结论**：AAAI p7 Conclusion明确假设表示已充分学习，并承认尾类改善伴随头类牺牲。p7 Fig6/Table5已研究强度和供体采样；p5基础配方为ResNet32/CIFAR100-LT、200+10轮、batch128、SGD。arXiv附录Table7已有固定/随机通道比较。因此“固定表示限制”“强融合伤头类”“随机通道未研究”均不能作为未经作者讨论的新问题。

**本地四方对照**：作者说StageII固定表示参数；冻结源所选shift_bn=True仍更新BN buffers（既有component report及`train_stage2.py:313-370`）。这是实现边界，不证明作者所有实验保持统计固定，更不证明已发表收益无效。

## 2. PI-H2T：两个机制分别作用于表示和分类器

[PI-H2T v2完整原文](https://arxiv.org/pdf/2506.00625v2)。作者以PIF补足原H2T不改变表示的限制，并将H2TF改为pooling后的自适应向量融合（p2 extension，p4–5 Sec3.3/3.4，p16 Algorithm1）。

PIF（p4 Eq3–6）：F_PI=mean_channel(F)，F_fuse=a(F-F_PI)+bF；2→1、kernel1的Conv1d共享于全部通道和空间坐标，主实现无bias为2参数；p5明确允许含bias时3参数。输入/输出维度相同，StageI学习并在推理保留。它融合均值和顺序相关F，不能将整个输出说成完全置换不变。

H2TF（p5 Eq7–9）：f_tilde=r f_B+(1-r)f_I；r=Norm[dist(f_B,w_y)]，footnote2明确使用理论边界而非batch统计的线性min-max。cosine distance范围[0,2]，所以**本轮代数推导**r=(1-cos)/2、donor mass lambda=(1+cos)/2。这里只是输入/比例语义的确认，不是作者明确给出的stop-gradient或reference-head约定。BS/IS和接收者硬标签沿用；只在StageII使用H2TF，推理不融合。

**理论适用边界**：p6 Remark1将学习到的权重视作类中心，并假设相关PI项更靠近正确中心；这些假设不能改写为任意已训练特征上的保证。Eq10漏写b项而Eq12包含，p16算法StageI line5写F而非F_fuse；与正文一致的活动PIF路径有直接支持，但这些笔误/记号歧义应保留，不能据一处伪代码令PIF失活。没有从论文恢复a,b精确初值、W是否固定用于r、r的detach或BN buffers设置。

**协议与消融**：p7为CNN ResNet32、CIFAR100-LT、CE/MixUp、200+10轮、batch128、SGD .9；不能套用ViT预训练/AutoAugment配方。p10 Table6给CIFAR100IF100 PIF StageI42.19→StageII49.22；Table8比较mean/max/min。p12 Table10还比较Conv1D无PI、SE、learned pooling，与PIF收益，不可再称作者未控制参数增加。p11 Table9是**CIFAR10**的PIF+DR82.50与PI-H2TF83.04，不能移作CIFAR100控制。

**作者已承认的限制**：p11及p16 AppendixD描述过小r/过度供体混合产生语义噪声、局部混淆和部分头类错分；p13 Conclusion指出ECE有时略升，以及PIF与分类器校准的理论交互仍开放。这些是作者原有边界，不能包装为本地新发现。p12 Fig9/10的CIFAR100 r约.65指**接收者**；本地lambda≈.77指供体。仅对齐均值、交换权重或反转公式来追逐.65均不构成复现。

## 3. FeatRecon：表示坍缩与支持域覆盖已有系统研究

[ICLR2025全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/adb2075b6dd31cb18dfa727240d2887e-Paper-Conference.pdf)。pp4–6 Sec3 Eq1–6、Theorem1/2在单位球、3≤K≤h+1、强encoder、全量batch、tau=1及one-vs-all样本量条件下分析SC loss最优几何；达到阈值时尾中心重合。它不是对CE/PIF/冻结StageII H2TF普适坍缩的定理。

pp6–8 Sec4 Eq7–10估计类中心和角度分位数，用邻近头类统计正则尾类support，将生成特征限制在球冠（角度上界1/2 acos(-1/(K-1))）；Eq11调温，Eq12–13用logit-compensated CE和SC联合训练表示。AppendixA pp14–15的Algorithm1/2及Eq14–16给迭代估计、截断角度与球面采样。输入是训练表示/统计，输出是受support约束的合成特征和更新后的表示，不是H2TF的某个更好r。

pp8–10实验：CIFAR ResNet32+128维projection，batch256，200/400轮，分类头AutoAug/Cutout与projection的SimAug；Table5比较naive upsample、feature generation和temperature。原论文附录p15 A.3明确理论仍限one-vs-all，general class-size geometry未解。**我们的推论边界**：generic“尾类中心不可靠/生成特征越界”已被研究，但该文未验证本地PIF的25参数版本为何在StageII退化，也不证明PI-H2T在其原协议下失效。

## 4. SEL：邻类和候选可靠性不是空白概念

[ICCV2025全文](https://openaccess.thecvf.com/content/ICCV2025/papers/Jian_Supervised_Exploratory_Learning_for_Long-Tailed_Visual_Recognition_ICCV_2025_paper.pdf)。pp3–5 Sec3 Eq1：z_e=z_s+r(z_s'-z_s)，这里r是**候选/供体**位移权重，与PI Eq7的r极性不同。Eq2按类频补足目标数；Eq3用batch类中心L2距离选邻类；Eq4按正确类置信度采候选；Eq5–6把接收者标签探索CE与原任务loss加权。作者关注区域覆盖及边界清晰，允许representation训练或classifier调整阶段。

p6 Sec4.2.2固定seed2024，CIFAR100最后20轮、CIFAR10额外20轮、ImageNet/iNat额外10轮；lambda=.9，邻类数按数据集设置。p6 Table2在其H2T+CE IF100协议下42.27→42.45（+0.18），不是原H2T的47.73同源对照；p7 Table5隔离CbC/FwS，p8研究邻类数/起始epoch/权重。作者指出合成增加计算量（p8）；邻域与confidence合理性的讨论不等于对所有PIF pair和比例的严格边界保证。**我们的解释**：泛称“引入donor几何/可靠性”不足以证明新颖；该文没有解决当前paper-code parity缺口，也没有做同checkpoint同ratio multiset的本地PIF验证。

## 5. DBG：生成器迁移副作用不能直接归因原H2T

[CVPR2026全文](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Decision_Boundary-aware_Generation_for_Long-tailed_Learning_CVPR_2026_paper.pdf)。pp1–4 Sec3的对象是**diffusion-based** head-to-tail generation。Eq1–2拟合vMF/BC衡量类间重叠；Eq3–4用类内近邻距离标准化定义outlier；Eq5–7修改CFG并以平衡classifier测confidence/credibility。其图中H2T表示生成迁移条件，不是AAAI随机通道函数的直接受控消融。

pp4–6 Sec4 Eq8–11从中间扩散步去分类噪声并向混淆类别重建，Eq12–13和置信度分支过滤生成样本，再以合成图补原训练集、重训分类器。输入/输出是图像/扩散轨迹及经清洗的辅助数据，区别于固定表示上的即时通道替换。p6–7为CIFAR ResNet32/200轮/batch128，以及ViT/100轮/batch32；Table5比较generator、prototype-distance、confidence-credibility。p8 Conclusion承认生成器长尾偏置仍限制效果。**可支持**：生成式迁移需要考虑边界损伤；**不支持**：以该文作为原H2T/本地PI-H2T退化的直接因果证据。表中也存在个别非正增益，不能把总体叙述扩写为每个设置全胜。

## 四方交叉表

|耦合|作者论文证据|实际本地代码|既有受控观察|仍未解的因果范围|
|---|---|---|---|---|
|H2T固定表示与BN|AAAI p7假设，p4算法固定参数|冻结源shift_bn=True，train/no_grad|本轮无训练；既有component audit为静态证据|buffers与参数边界不能合一；并非新发表缺陷|
|PIF参数化|PI p4–5单层/2或3参数|历史H2T三层25；独立review单层2|PIF历史checkpoint40.27，原论文42.19，协议尚未对齐|差异是否导致StageII退化未证实|
|比例方向|PI p5 Eq7/8+脚注2|历史r=(1-cos)/2，lambda=1-r|本地lambda≈.77；r≈.23|公式未反转；分布差异可能来自特征、W与协议|
|reference与BN|PI未明确freeze-reference/detach/buffers|历史冻结StageI W+更新BN；review当前W+冻结BN|T2/T3同BN流；T1/T2只作无融合BN对照|不排除BN×fusion交互；完整作者配置仍缺|
|pair correspondence|SEL已研究邻类/FwS；PI理论交互开放|历史B配对排序、C层内打乱|CE gate通过；PIF gate失败|不能升格完整PI缺陷/改进|
|support/ambiguity|FeatRecon支持域；DBG扩散迁移分析|与当前25参数PIF代码不同|无同源本地反事实|只提供问题背景，不能跨机制迁移因果|
