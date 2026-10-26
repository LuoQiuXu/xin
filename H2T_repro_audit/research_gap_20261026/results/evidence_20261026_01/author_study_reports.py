"""Render human-reviewed mechanism evidence, not a model or experiment runner."""
import csv
import hashlib
import json
from pathlib import Path

RUN=Path(__file__).resolve().parent
REPO=RUN.parents[3]

def write(name,text):
    with (RUN/name).open('x',encoding='utf-8',newline='\n') as h: h.write(text.rstrip()+'\n')

PARITY=[]
def row(field,paper,claim,historical,status,review,review_status,qualification):
    assert status in {'MATCH','MISMATCH','UNVERIFIED'} and review_status in {'MATCH','MISMATCH','UNVERIFIED'}
    PARITY.append(dict(field=field,paper_reference=paper,paper_rule=claim,historical_code=historical,
                       historical_status=status,review_code=review,review_status=review_status,qualification=qualification))

row('PIF channel mean','PI v2 PDF p4 Eq4-5','沿特征通道取均值，保留每个空间位置',
    'H2T/models/resnet_cifar.py:173','MATCH','PI_H2T_review/H2T-main/models/resnet_cifar.py:172','MATCH','该均值不等于把整个输出特征变成置换不变；保留原F项')
row('PIF residual equation','PI v2 p4 Eq3,6; p5 Sec3.3','a(F-mean)+bF；单层2→1的kernel1 Conv1d',
    'H2T/models/resnet_cifar.py:144','MISMATCH','PI_H2T_review/H2T-main/models/resnet_cifar.py:167','MATCH','历史是2→4→2→1三层；无激活可约成有效仿射混合，不能从结构差异直接推断性能根因')
row('PIF parameterization/bias','PI v2 p5 Sec3.3; p12 Table10','主实现2参数无bias；正文另允许含bias时3参数',
    'H2T/models/resnet_cifar.py:145','MISMATCH','PI_H2T_review/H2T-main/models/resnet_cifar.py:167','MATCH','历史18个weight+7个bias=25；单层审核包2；后者没有被用于历史PIF结果')
row('PIF concat order','PI v2 p4 Eq6','[F-mean,F]',
    'H2T/models/resnet_cifar.py:174','MISMATCH','PI_H2T_review/H2T-main/models/resnet_cifar.py:173','MATCH','历史[F,F-mean]为顺序差异，可交换有效系数重参数化；不是单独数学错误或精度因果证据')
row('PIF tensor dimensions','PI v2 p4-5 Eq3,6','每个通道/空间坐标共享两个系数，输出保留原形状',
    'H2T/models/resnet_cifar.py:172','MATCH','PI_H2T_review/H2T-main/models/resnet_cifar.py:175','MATCH','CIFAR静态[B,64,8,8]→[B*64,2,64]→原形状；本轮未测真实张量')
row('PIF active return','PI v2 p4 Eq3; p16 Algorithm1','Stage I与干净推理保留PIF输出',
    'H2T/models/resnet_cifar.py:180','MATCH','PI_H2T_review/H2T-main/models/resnet_cifar.py:184','MATCH','历史训练快照codes/models/resnet_cifar.py:170也return out_hat；不能把上游草稿return out误算为该历史运行未启用PIF')
row('PIF initialization','PI v2 p4-5 Sec3.3; p16 Algorithm1','模型随机初始化；未明确给a,b的精确数值/分布或种子顺序',
    'H2T/models/resnet_cifar.py:150','UNVERIFIED','PI_H2T_review/H2T-main/models/resnet_cifar.py:165','UNVERIFIED','历史_weights_init只处理Conv2d/Linear，Conv1d默认初始化；审核包也默认Conv1d；均未恢复作者初值')
row('H2TF location','PI v2 p2 extension; p16 Algorithm1 lines12-14','PIF后GAP，Stage II在pooled feature线性融合',
    'H2T/train_pi_h2t_ratio_only_stage2.py:533','MATCH','PI_H2T_review/H2T-main/pi_h2t.py:29','MATCH','p5 Eq7后文字仍称再pooling；按p2与算法明确流程解释，不因文字不一致擅改原文')
row('H2TF receiver/donor polarity','PI v2 p5 Eq7','f_tilde=r*f_receiver+(1-r)*f_donor',
    'H2T/train_pi_h2t_ratio_only_stage2.py:608','MATCH','PI_H2T_review/H2T-main/pi_h2t.py:38','MATCH','本地lambda是donor mass，r=1-lambda；未发现写反')
row('Ratio cosine/theoretical bounds','PI v2 p5 Eq8 footnote2','cosine distance 1-cos范围[0,2]；理论边界线性归一化',
    'H2T/train_pi_h2t_ratio_only_stage2.py:292','MATCH','PI_H2T_review/H2T-main/pi_h2t.py:19','MATCH','数学推出r=(1-cos)/2，lambda=(1+cos)/2；无batch min/max，不把相似度当距离')
row('Ratio clipping','PI v2 p5 Eq8','r有界[0,1]',
    'H2T/train_pi_h2t_ratio_only_stage2.py:298','MATCH','PI_H2T_review/H2T-main/pi_h2t.py:22','MATCH','native A只是数值范围clip；扩展D/E另有边界clip，历史A/B/C未选择D/E')
row('Ratio reference classifier','PI v2 p5 Eq8; p16 Algorithm1','w_y为对应class classifier权重；没有明确另设冻结reference头',
    'H2T/train_pi_h2t_ratio_only_stage2.py:750','UNVERIFIED','PI_H2T_review/H2T-main/train_pi_h2t.py:37','UNVERIFIED','历史固定Stage-I W；审核包随当前Stage-II W更新；论文未明确stop-gradient/reference约定，不能称二者同时已验证官方等价')
row('Ratio gradient/detach','PI v2 p5 Eq8; p16 Algorithm1','固定extractor、调整classifier；未明确r对W求导',
    'H2T/train_pi_h2t_ratio_only_stage2.py:532','UNVERIFIED','PI_H2T_review/H2T-main/pi_h2t.py:25','UNVERIFIED','历史ratio在no_grad且W冻结；审核包默认detach=True；这不是论文逐项声明')
row('Supervision labels/loss','PI v2 p5 Sec3.4; p7 Sec4.2','只用BS接收者标签；CE为允许的宿主loss',
    'H2T/train_pi_h2t_ratio_only_stage2.py:627','MATCH','PI_H2T_review/H2T-main/train_pi_h2t.py:40','MATCH','donor-label只供审计；没有按lambda混合标签')
row('BS/IS sampling','PI v2 p5 Eq9','接收者类别均匀，供体实例原分布',
    'H2T/train_boundary_h2t_stage2.py:176','MATCH','PI_H2T_review/H2T-main/train_pi_h2t.py:99','MATCH','H2T dataset:98保留原索引、后续接收者重采样；不是head-only/tail-only，也不是多候选选择')
row('CNN backbone and head','PI v2 p7 Sec4.2; p4 notation','CIFAR100-LT/ResNet32，线性头',
    'H2T/train_pi_h2t_ratio_only_stage2.py:240','UNVERIFIED','PI_H2T_review/H2T-main/models/resnet_cifar.py:178','UNVERIFIED','PIF历史Option-A；CE外部checkpoint可能Option-B，代码按state判定；论文未给全部shortcut/bias细节。审核包普通FC带bias')
row('Stage I optimizer/MixUp','PI v2 p7 Sec4.2','CE+MixUp，200epochs，batch128，SGD .1/momentum .9，160/180衰减',
    'H2T/config/cifar100/cifar100_imb001_stage1_ce_mixup_pif.yaml:37','MATCH','PI_H2T_review/H2T-main/train_pi_h2t.py:89','MATCH','Match仅限明确列出的配方；wd、warmup、增强全细节不能仅凭正文同名认定全协议匹配')
row('Stage II optimizer/schedule','PI v2 p7 Sec4.2','10epochs，SGD .9；未给所有StageII参数组倍率',
    'H2T/train_pi_h2t_ratio_only_stage2.py:753','UNVERIFIED','PI_H2T_review/H2T-main/train_pi_h2t.py:97','UNVERIFIED','历史及审核包都继承.1×.2首轮.02和cosine；论文正文没有确认这套倍率/wd的完整作者配置')
row('Backbone gradient freeze','PI v2 p16 Algorithm1 line15','StageII只微调分类器参数',
    'H2T/train_pi_h2t_ratio_only_stage2.py:264','MATCH','PI_H2T_review/H2T-main/pi_h2t.py:42','MATCH','requires_grad/no_grad/optimizer边界一致，不由此推断BN统计固定')
row('BN running buffers','PI v2 p6 and p16 Algorithm1','固定representation参数；未单列BN buffers规则',
    'H2T/train_pi_h2t_ratio_only_stage2.py:477','UNVERIFIED','PI_H2T_review/H2T-main/pi_h2t.py:44','UNVERIFIED','历史默认train更新BN；T1额外fixed；审核包eval固定；两者是已证实的本地差异，但谁匹配作者尚未确定')
row('Checkpoint lineage/selection','PI v2 p12 Fig9-10','论文并列best/last；未提供可逐字节锁定的本地StageI checkpoint',
    'H2T/outputs/pi_h2t_ratio_only_pif_stage1seed123_seed42/experiment.json:2','UNVERIFIED','PI_H2T_review/H2T-main/train_pi_h2t.py:16','UNVERIFIED','历史epoch164/40.27；审核包要求pif.fuse.weight，历史conv1d.*，严格loader拒绝互换；只核验当前文件hash，未反序列化')
row('Inference no H2TF','PI v2 p6; p16 AppendixD','推理保留PIF、移除H2TF',
    'H2T/train_pi_h2t_ratio_only_stage2.py:430','MATCH','PI_H2T_review/H2T-main/train_pi_h2t.py:104','MATCH','当前只读审计，没有跑evaluate或读取TEST')
row('Accuracy/evaluation protocol','PI v2 p8 Table2; p10 Table6; p12 Fig9','CIFAR100IF100 StageI42.19，StageII best49.22/last49.04',
    'H2T/outputs/pi_h2t_ratio_only_pif_stage1seed123_seed42/summary.csv:2','UNVERIFIED','PI_H2T_review/H2T-main/PI_H2T_REVIEW_ZH.md:1','UNVERIFIED','本地40.27→last34.82是该实现/协议结果；原审核包未证明论文精度，数值差异不是随机误差显著性或作者缺陷证据')

CROSSWALK=r'''# 作者机制对照：H2T、PI-H2T、FeatRecon、SEL、DBG

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
'''

PARITY_INTRO=r'''# PI-H2T 论文—代码一致性审计

状态对象是**具体字段**，不是整套复现的通行证。MATCH表示已读论文明确规则与对应静态实现相符；MISMATCH表示有可定位的实现/参数化差异；UNVERIFIED表示论文/作者配置/运行归属不足。论文公式确认不等于历史模型达到论文协议。最终Stage-II异常状态：**REPRODUCTION_UNRESOLVED**。

## 代码谱系

1. `H2T_official_clean/H2T-main/`：前次component audit冻结的AAAI H2T路径；仍以通道替换为主证据，未改动。
2. `PI_H2T_review/H2T-main/review/upstream_provenance.json`：记录2026-09-30 codeload ZIP，commit comment `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`，archive SHA256 `3f16f6759cfba1a5fbaa77b3d4a3bd5c43597b829298fd078c2459e12a79adcc`。本轮验证该记录文件hash；没有重新取得ZIP或核对GitHub该commit字节，不能把记录自证为当前官方完整PI发布。
3. `review/pi_h2t_vs_official_main.diff:7-34`保留当时上游草稿非法Conv2d参数、未定义feature_tuning、三层Conv1d和计算out_hat却return out的差异证据。独立审核包删掉草稿，另外写单层PIF、独立入口和冻结BN。其自述明确为**复现候选**，不是作者原码。主来源URL为 https://github.com/Keke921/H2T ，本轮网络旁证请求未完成，未据其假装确认归属。
4. `H2T/`是实际历史训练使用的本地改写分支：`resnet32_pif_fe`→PIF-capable ResNet32/Option-A→三层Conv1d→活动返回out_hat。训练时保存的`H2T/saved/cifar100_imb001_stage1_ce_mixup_pif_202608082320/codes/models/resnet_cifar.py:139-170`也含三层与有效返回。**因此历史PIF确实有活动实现证据，但不是审核包那份两参数PIF**。六个conv1d.* checkpoint键只证明三层结构存在，不证明等价作者参数化。

## 关键代数与因果边界

历史三层无激活，令输入x=[F,F-mean]，则y=W3 W2 W1 x+W3 W2 b1+W3 b2+b3。可以折叠成c_F F+c_R(F-mean)+c_0；故不能声称它必然是更复杂的非线性融合，或25参数本身证明论文方程完全不可表示。论文p5允许单层含bias的3参数版本，函数族可能相同；但是**分解参数化、默认初始化、bias、SGD/weight decay作用和随机数顺序不同**，训练轨迹/初始有效系数没有证明相同。拼接顺序也可以通过系数交换等价，单独把它叫“公式写反”同样不严谨。

`STATIC_PARITY_CHECKS.json`从AST数得历史25、review2；标量边界cos=-1/0/1时r=1/.5/0、lambda=0/.5/1。只是源码与代数核验，不是从真实checkpoint提取a,b,c，不是模型前向，不是因果实验。未对ratio调参、交换权重、换loss或写入原代码。

历史模型和头使用strict state loading（ratio-only:260-264）；旧路径中的`新`与当前仓库`xin`不同，当前候选checkpoint文件位置、大小及SHA见`CHECKPOINT_FILE_METADATA.json`，仅原始字节哈希。没有历史SHA作逐字节同一性证明，也没有反序列化权重；作者checkpoint身份仍未确定。review loader:21只接受pif.fuse.weight，历史为conv1d.*，两套模型不能静默互换。

原论文p16 StageI算法line5的F/F_fuse和p6 Eq10/b项记号不一致保留为文稿歧义；活动PIF依正文Eq3/6及随后推理说明核对，不能用伪代码笔误“证明”应返回未融合F。reference W、detach、BN、初始化的未定项也不能靠本地任选约定写成MATCH。

## 逐字段状态表

'''

PRIOR=r'''# 历史实验事实及重新解释

本表来自只读历史报告与原始summary.csv/experiment.json，10个来源锚点先经既有verify_evidence程序通过。`HISTORICAL_CONTRASTS.csv`有18行A/B/C Last结果；`HISTORICAL_RECALC.json`用stdlib重算均值、样本SD和配对差。没有重新训练、加载预测或访问官方TEST。数值为百分比/百分点；这是已有记录的复算，不是统计显著性检验。

## 同源PIF endpoint（仅StageII seed42）

同一StageI seed123、PIF checkpoint epoch164（40.27%），分类头同初始化，batch128，10轮cosine，SGD .1×.2，BS receiver/IS donor。T1/T2即使donor mass=0也保留两次骨干前向，避免简单关闭H2T少一次BN更新的混杂。摘要表的数值必须来自这一份causal diagnostic和对应输出，不能换成其他seed或best结果。

|端点|BN|donor mass|Last Overall|Many|Medium|Few|Best（辅助）|
|---|---|---:|---:|---:|---:|---:|---:|
|T0 StageI direct|checkpoint|—|40.27|66.53|40.06|7.93|—|
|T1 no fusion|fixed|0|44.80|65.00|47.00|17.07|44.85|
|T2 no fusion|update|0|45.69|65.22|47.80|18.90|45.85|
|T3 local native H2TF|update|.7697|34.82|26.33|44.40|33.79|40.99|
|T4 matched scale|update|.7133|37.26|30.86|46.71|33.79|42.18|

历史`H2T/PI_H2T_PIF_CAUSAL_DIAGNOSTIC_REPORT.md`记录T1/T2/T3末轮receiver/donor/target/donor-label checksum逐值相同；原CSV复核也相同。它们是浮点加和校验，不是抗碰撞SHA，也不是本轮重新生成的逐样本重放证明。

T2−T1=+.89，T3−T2=−10.87，T4−T3=+2.44，T4−T2=−8.43。**局部可支持**：该实现、该checkpoint、该seed下，启用所实现融合和比例映射对应主要下降；无融合条件的BN update未导致下降。**不能支持**：“BN在所有融合设置下无影响”或“作者PI-H2T已经失效”。没有T3的完整BN交互反事实；而PIF参数化、reference W和作者协议未完全对齐。matched-scale由操作流均值比.7128000/.7691256=.9267667644预先确定，恢复约22.4%的该局部差距，但不是因果中介份额，也未修复完整落差。

匹配PIF-off/on StageI控制创建同一含PIF结构，仅pif_active开关不同，历史报告称初始化逐张量一致。on/off最佳StageI分别40.27/39.40；在相同eval图像和冻结BN的既有cosine audit中cos均值.3551/.2719，对应lambda .6775/.6360；操作BS流lambda .7691/.7128。它支持在该单对训练实例下PIF开关改变了几何，不能证明作者两参数PIF有同样符号/尺度，也不能把1个StageI seed视为跨训练随机性的因果规律。两侧各自best epoch不同（164/198），选择过程属于控制边界。

## CE与PIF的A/B/C保持分别汇报

A=recipient-only native比例，B在频次组×原r四分层内按pair critical geometry重排同一ratio multiset，C层内打乱B的对应。三组均用冻结StageI W作ratio/reference，BN更新且仅训练FC。它们是本地机制干预，不是作者完整PI程序；B/C改变对应而非开发一个本轮新模块。

|setting / StageII seed|A|B|C|B−A|B−C|
|---|---:|---:|---:|---:|---:|
|CE /42|40.36|40.96|40.55|+.60|+.41|
|CE /43|40.23|41.14|40.47|+.91|+.67|
|CE /44|40.48|41.27|40.81|+.79|+.46|
|CE mean±sampleSD|40.36±.13|41.12±.16|40.61±.18|+.77±.16|+.51±.14|
|PIF /42|34.82|35.23|35.22|+.41|+.01|
|PIF /43|34.96|35.25|35.39|+.29|−.14|
|PIF /44|35.35|35.81|35.59|+.46|+.22|
|PIF mean±sampleSD|35.04±.27|35.43±.33|35.40±.19|+.39±.09|+.03±.18|

CE为另一份无PIF checkpoint（external CE seed42 epoch184，metadata best40.76），可走Option-B shortcut；PIF为本地Option-A/25参数PIF seed123 epoch164。因此跨setting的均值差不隔离PIF。每个setting只有一个StageI checkpoint、三个StageII seeds；3/3正号不构成总体显著性。每个seed三臂数据checksum匹配、ratio multiset最大误差0，B排序违例0，C打乱约95%；这支持受控分配问题，但不能排除浮点累加碰撞或重放细节未记录。

CE严格gate（所有seed B>A且B>C）通过，主要Overall增益来自Many，不是Tail专属结论。PIF的B>A为3/3，但B>C只有2/3，原门槛**FAILED**。不能筛掉seed43，不能把+.03±.18描述为稳定配对优势，也不能继续借CE正结果宣称改善了完整PI-H2T。

## boundary-r负结果（不改写成新假设）

`H2T/PI_H2T_CENTER_VS_BOUNDARY_R_SEED42_REPORT.md`及原CSV：PIF seed42、同checkpoint/原StageI head、固定kappa=.9267667644，保留group×lambda-bin的ratio多重集合；B仅用recipient signed-boundary margin排序，C打乱，不用donor geometry。A/B/C的Overall为37.20/37.12/37.56；H-mean32.00/31.97/32.31；Few33.72/33.97/33.90，Medium47.03/46.86/47.49。B−A Overall−.08，B−C−.44，H-mean与Medium也未过原门槛。

末轮lambda均值.714177一致、multiset误差0、B单调违例0，数据流校验一致。它说明这一个已规定的margin重分配没有获得要求优势，不证明“边界几何不重要”，也不授权换排序方向、加donor模块或在同一开发流程继续试探。未扩展其他seed。

## 数据使用与因果语言

历史ratio-only脚本`H2T/train_pi_h2t_ratio_only_stage2.py:783,808`初始及每轮都调用evaluate，best只是记录；选定主结果为固定Last。boundary脚本只在训练结束评估（`H2T/train_center_boundary_r_stage2.py:760`），但其StageI checkpoint及多个前期研究已用官方test查看/选模，不能称整个研究链条都拥有未污染的独立测试集。此处只读这些记录，没有读数据集本体。

历史报告中“native”“official ratio”“缺陷”“不是BN导致”等措辞在本轮被限制为特定本地实现/对照范围；不回写历史报告，也不抹去负结果。尚未恢复作者全流程parity，所以因果定位的最终分类仍是**REPRODUCTION_UNRESOLVED**。
'''

DECISION=r'''# ONE_QUESTION_DECISION

**判断：STOP_AND_REPAIR_PARITY。** 不是GO；也不是已经证明一切机制问题均被解决。当前最强证据是一个**可定位的复现一致性缺口**，尚不是已确认的发表方法缺陷或新研究空白。

|门槛|状态|依据与限制|
|---|---|---|
|G1 作者机制已验证|PASS（核心公式）|已打开PI v2 p4–5 Eq3/6/7/8与脚注2，原H2T公式/阶段及三篇相关工作全文；W-reference、detach、BN不在核心公式核验之外偷填为确定|
|G2 parity建立或缺口精确隔离|PARTIAL / NOT ESTABLISHED|历史PIF三层25参数vs论文单层2/3明确；review两参数版本与历史训练版本分离。初始化、BN、W-reference、checkpoint作者归属仍未定，完整parity未建立|
|G3 问题可重复且不由实现不一致解释|FAIL|PIF同一StageI checkpoint三StageII seeds存在低准确率，但B>C严格gate失败；单seed endpoint落差仍可能依赖PIF参数化/协议。没有排除复现偏差的反事实|
|G4 后续工作尚未解决该具体问题|NOT ESTABLISHED|PI已处理固定表示/随机通道/手调ratio，且承认过度融合及理论交互；FeatRecon研究支持域，SEL邻类/可靠性，DBG生成边界。没有能跨越parity缺口的一个新颖具体命题|
|G5 单因素、8GB可证伪验证可预注册|WITHHELD|当前首先需要文档/作者配置/代码身份核对；多个未知项同时变化时，训练对比不能被称为单因素因果检验。本轮无显存实测或训练资源承诺|

唯一保留的待澄清问题（**复现问题，不称新颖性已证实**）：

> 历史PIF+H2TF的StageII退化，在严格恢复论文PIF参数化、明确ratio所用W及BN/训练/checkpoint协议后，是否仍成立？

它可被“parity对齐后退化消失”反驳；但这句话不是本轮安排新训练的授权，也不把未对齐时的34.82%视作作者方法的反例。当前只确认参数化不同，**没有证实其就是退化根因**。

本轮不选择TailOff、MarginProtect、boundary-r重命名或新的donor/attention机制，也不同时保留其他随机漏洞。作者原文开放问题不自动意味着我们有新的可发表解决；未主张期刊级新颖性、总体统计显著性或实证提升。

## parity修复所需的证据（不是新实验配方）

先核对作者发布版本/配置与PIF确切层数、bias和初值；明确Eq8 W是当前还是冻结、r是否detach、BN buffers状态、StageI/II实际LR倍率及checkpoint选择；将这些字段和候选checkpoint SHA绑定。若只能得到本地独立重建，应继续标UNVERIFIED，不得改历史输出或强行校准r到.65。任何未来训练须另行授权，且不得以多个未决字段同时改动后准确率上升就宣称发现新机制。

由于G2/G3/G4未满足，没有制定NEXT_EXPERIMENT的arms、阈值或搜索范围；`NEXT_EXPERIMENT_PREREG.md`只记录未注册状态。
'''

OVERVIEW=r'''# H2T / PI-H2T 深度机制研究报告

2026-10-26；分支`research/h2t-pih2t-gap-20261026`；审计基准HEAD `de2198a85117a46972d294fb4ae09951b3dbb018`。**结论：STOP_AND_REPAIR_PARITY；历史StageII异常归类REPRODUCTION_UNRESOLVED。** 已确认本地参数化差异，尚未证实性能下降根因，不将复现异常称为已发表PI-H2T缺陷。

## 六阶段完成情况

1. 先执行既有5项单元测试；默认temp路径被沙箱拒绝，改为本新结果目录内的临时fixture后5项全过，测试逻辑未改。既有verify_evidence成功核对10个来源锚点，生成EVIDENCE_INDEX/SOURCE_MANIFEST。锚点通过只证明原记录存在，不证明科学命题为真。
2. 取得五篇指定主论文全部PDF并锁定版本/SHA，逐页定位机制正文/附录，11张关键页图像复核；H2T额外用arXiv附录。作者主张、输入输出、阶段、假设、配方、对照和自述限制见`AUTHOR_MECHANISM_CROSSWALK.md`及页账本。
3. 沿本地实际工厂/forward/ratio/reference/optimizer/loader/evaluate/checkpoint调用追踪，输出`PI_PAPER_CODE_PARITY.md/.csv`，不把独立review包当作历史训练或作者官方代码。
4. 只读40份核心源/元数据/历史结果，重算18行A/B/C和endpoint差，见`PRIOR_EXPERIMENT_EVIDENCE.md`、`HISTORICAL_CONTRASTS.csv`及`HISTORICAL_RECALC.json`。没有合并不同checkpoint/不同评估协议。
5. 明确G1–G5；G3失败且完整parity未建立。不设计新模块，不注册新训练，只保留一个复现问题。
6. 完整性、引用、AST/JSON/CSV和提交范围检查见`EXECUTION_REPORT.md`及最终JSON；仅发布本轮目录中的小体积报告、CSV/JSON与审计脚本，PDF/渲染/全文抽取留本地忽略。已获用户授权提交推送同名分支，不PR/merge。

## 最强证据链

论文PI v2 p4–5规定单层Conv1d(2,1,1)，主实现2参数无bias（可选含bias3参数）。产生历史PIF实验的`H2T/models/resnet_cifar.py:144-147`及训练保存代码却是带bias的2→4→2→1，共**25参数**；独立`PI_H2T_review/`才是2参数实现，但不是那些历史结果的训练来源。三层间无非线性，可以代数折叠成有效仿射混合，故这个发现证明了**训练参数化不一致**，不证明函数表达能力必然不同或它就是准确率下降的原因。

Eq7/8及脚注2核验后，本地比例方向本身是匹配的：r=(1-cos)/2为接收者保留量，lambda=(1+cos)/2为供体；不能将r和lambda混同，不能通过交换二者贴合论文图中r≈.65。历史r≈.23、lambda≈.77与论文分布不同，是诊断线索，不足以定位原因。ratio的W是否用当前head、detach、BN、初值与作者checkpoint身份仍不完全可确认。

在同一历史PIF checkpoint、StageII seed42中，T2无融合45.69、T3所实现native融合34.82、T4尺度匹配37.26。两次骨干前向及BN输入对齐使这个局部对照有解释力：损失主要伴随所实现融合出现，尺度缩小只恢复2.44点。但它不能反驳论文，也不能排除BN与融合交互或未对齐参数化的作用。

PIF A/B/C三种子B−C为+.01/−.14/+.22，严格gate失败；CE三个正差来自另一无PIF checkpoint，不能补足完整PI证据。boundary-r负结果同样保留。论文PI已经承认过度融合语义噪声及PIF—校准理论交互；FeatRecon、SEL、DBG分别覆盖表示支持域、邻类可靠性与生成边界问题，泛称“融合有害/供体不盲目/几何重要”不足以建立新颖机制问题。

## 保留的一个问题与当前停止点

唯一待澄清项是：**严格恢复论文PIF与明确全流程协议后，历史StageII退化是否仍存在？** 当前是parity修复问题，不能包装为已确认研究空白。没有任何新模块、随机机制、训练arms或参数搜索。只有在未来parity证据和可重复因果问题同时成立时，才有资格讨论一个具体可证伪研究命题；本报告没有越过这道门槛。

## 阅读导航

- `AUTHOR_MECHANISM_CROSSWALK.md`：作者证据与我们的解释分栏，五篇完整机制比较。
- `PI_PAPER_CODE_PARITY.md/.csv`：23个字段逐项MATCH/MISMATCH/UNVERIFIED。
- `PRIOR_EXPERIMENT_EVIDENCE.md`：历史控制、局限、负结果和测试集使用。
- `ONE_QUESTION_DECISION.md`：五门槛与明确停止判断。
- `EXECUTION_REPORT.md`：命令退出码、源文件与论文hash、保护及发布验证。

本轮没有GPU调用、模型前向、训练、checkpoint反序列化或官方TEST读取。历史test使用只通过源码与已有报告识别；原始H2T、PI review与全部历史实验资产均未修改。
'''

def main():
    write('AUTHOR_MECHANISM_CROSSWALK.md',CROSSWALK)
    with (RUN/'PI_PAPER_CODE_PARITY.csv').open('x',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(PARITY[0]));w.writeheader();w.writerows(PARITY)
    table=['|字段|论文规则/位置|历史H2T/代码|状态|独立review代码|状态|限定|','|---|---|---|---|---|---|---|']
    for item in PARITY:
        table.append('|'+ '|'.join(str(item[k]).replace('|','\\|') for k in ['field','paper_reference','historical_code','historical_status','review_code','review_status','qualification'])+'|')
    write('PI_PAPER_CODE_PARITY.md',PARITY_INTRO+'\n'.join(table)+'\n\n字段对应的原文规则详见同名CSV的paper_rule列。')
    write('PRIOR_EXPERIMENT_EVIDENCE.md',PRIOR)
    write('ONE_QUESTION_DECISION.md',DECISION)
    write('RESEARCH_REPORT.md',OVERVIEW)
    write('NEXT_EXPERIMENT_PREREG.md','# 未注册新实验\n\n状态：WITHHELD / STOP_AND_REPAIR_PARITY。G2/G3/G4尚未满足，因此没有训练arms、接受阈值、搜索空间或GPU命令。先完成论文—实现一致性的文档核对；本轮不执行训练。')
    entries=json.loads((RUN/'PAPER_ACCESS.json').read_text(encoding='utf-8'))
    topics={'H2T':[(1,2,'Introduction/related work'),(3,4,'Eq1/2 and training algorithm/rationale'),(5,7,'Protocol/results/ablation/conclusion'),(8,9,'References')],
            'PI_H2T':[(1,3,'Introduction/extension/related work'),(4,6,'PIF Eq3/6; H2TF Eq7/8; rationale'),(7,9,'Protocol and benchmark tables'),(10,12,'Ablations/ratios/failure cases'),(13,13,'Conclusion/limitations and references'),(14,15,'References'),(16,17,'Appendix algorithm/failure cases/balanced data')],
            'FEATRECON':[(1,3,'Motivation/related work'),(4,6,'SC theory assumptions/geometry/distribution gap'),(7,8,'Support Eq7-10; loss Eq11-13'),(9,10,'Protocol/results/ablation/conclusion'),(11,13,'References'),(14,16,'Algorithms/limitations/visualization'),(17,23,'Theorem1 proofs and lemmas'),(24,33,'Theorem2 proofs and critical threshold')],
            'SEL':[(1,3,'Motivation/related work'),(4,5,'OFA/CbC/FwS Eq1-6 and rationale'),(6,8,'Protocol/H2T comparisons/ablation/conclusion'),(9,11,'References')],
            'DBG':[(1,2,'Diffusion-transfer motivation/related work'),(3,4,'Overlap/outlier/CFG metrics and generator'),(5,6,'Filtering/protocol/main results'),(7,8,'Ablations/analysis/limitations'),(9,10,'References')],
            'H2T_ARXIV_V3':[(1,9,'H2T companion main paper/references'),(10,10,'Appendix Listing1/Table7'),(11,13,'Supplementary visualizations')]}
    ledger=[]
    for item in entries:
        if item['status']!='FULL_PDF_TEXT_OPENED': continue
        pages=json.loads((RUN/(item['key']+'_PAGES.json')).read_text(encoding='utf-8'))
        for p in pages:
            topic=next(t for a,b,t in topics[item['key']] if a<=p['page']<=b)
            ledger.append(dict(paper=item['key'],page=p['page'],topic=topic,text_characters=len(p['text']),
                               text_sha256=hashlib.sha256(p['text'].encode('utf-8')).hexdigest(),
                               review_level='REFERENCE_INDEX' if topic=='References' else 'TEXT_MECHANISM_OR_CONTEXT_REVIEW',
                               visual_review=(RUN/'rendered'/f"{item['key']}_{p['page']:02d}.png").exists(),
                               qualification='Text reading/page location is not independent theorem verification or a measured reproduction'))
    with (RUN/'PAPER_PAGE_LEDGER.csv').open('x',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    print(json.dumps({'parity_fields':len(PARITY),'page_ledger_rows':len(ledger),'decision':'STOP_AND_REPAIR_PARITY'},ensure_ascii=False))

if __name__=='__main__':main()
