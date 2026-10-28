# 机制新颖性对照：已研究、尚未回答、尚不足以立项

**没有候选同时通过科学证据和新颖性门槛。** 文献覆盖某类主题不代表已经实现了我们的每一个具体算子；反过来，一个具体实现不同，也不自动形成有价值的研究空白。以下把作者算法事实、我们的比较推论和未确定项分开。所有页码均为 PDF 从1起的页码，SHA、版本和访问限制见 [PAPER_FILES.json](PAPER_FILES.json)、[LITERATURE_ACCESS.json](LITERATURE_ACCESS.json)。只引用原论文/作者或正式会议来源。

## 1. 论文机制与实际边界

|原始来源与核查位置|作者已明确研究的具体操作|与本地候选的关系及不能外推之处（本轮判断）|
|---|---|---|
|[H2T，AAAI 2024][H2T]，p3 Eq1–2、p4 Alg1；p7 Table5/Fig6/结论；[arXiv v3][H2T_V3] p10 Listing1/Table7|BS receiver、IS donor；池化前完整通道替换，保留 receiver hard label；Stage II 调整分类器。附录比较固定位置与随机通道。p7 明示固定表示及 Head/Tail 代价。|“融合强度/供体分布会改变类间边界”已有明确研究。作者代码4D输入的 mask 为 batch shared；本地逐样本 LowVS 是不同干预，必须有 Uniform-PerSample 等对照。trade-off 本身不是新发现。|
|[PI-H2T arXiv2506.00625v2][PI]，2026-07-18；p4 Eq3/6、p5 Eq7–9及脚注2，p12 Fig9–10/Table10、p13 结论、p16 Alg1/Appendix D|PIF：`a(F−mean_channel(F))+bF`，单个2→1 kernel，2参数无bias/3参数有bias。H2TF：`r f_receiver+(1−r) f_donor`；理论边界 min-max 的 cosine-distance r，BS/IS，hard receiver label。作者已讨论过量 donor 语义噪声以及表示与分类器校准的未解交互。|替换随机通道、适应融合比例、改善 Stage-I 表示这些宽泛目标已覆盖。论文 r 是 **receiver** 权重，按脚注推导 `r=(1−cos)/2`；这不恢复 W 引用、detach、BN 等作者实现。期刊正式正文/独立补充材料的等价性 **UNVERIFIED**，不以 v2 冒充全部正式协议。|
|[SEL，ICCV 2025][SEL]，p3–5 Eq1–6，p7 Table5|`z_e=z_s+r(z_s'−z_s)`；这里 r 是 **donor** 权重。按类缺额补充 target（CbC），batch 类中心 L2 取邻类，再按 donor 真实类预测概率采样（FwS），保留 target hard label；辅助 CE 与原任务损失加权。|邻近供体、类别频次及可靠性选择已有具体实现。它不是 VS Bottom48、不是 GU 梯度内积，也不是保证每次融合标签正确的定理。不能仅以“都考虑困难度”断言与 RAVS 完全相同；但只提“选更可靠邻类”不足以主张新颖。|
|[FeatRecon，ICLR 2025][FEATRECON]，p4–6 Theorem1–2；p7 Eq7–10、p8 Eq11–13|理论分析归一化 SC 特征的失衡几何；由训练类中心/角度分位数估计球冠支持域，用邻近 Head 统计正则 Tail，再采样特征；联合 SC、带 logit compensation 的 CE 及温度调整。|表示塌缩、Tail 支持不足、借 Head 稳定统计已被研究；不是 frozen H2T 的在线 donor 风险筛选。理论有 one-vs-all、容量和维数等假设，不能直接套到本地 CE/64维100类模型，宣称其坍缩已被证明。|
|[DBG，CVPR 2026][DBG]，p3–4 Eq1–7；p4–6 Eq8–13|扩散中间时刻抑制源类别语义再向混淆类重建；用 prototype distance 与 confidence/credibility 双路清洗后重训分类器；报告边界含混/分布偏移指标。|“生成或迁移可能造成边界混淆、需要筛选”已有算法。该文 head-to-tail 对象是 **diffusion generation**，不是 AAAI H2T 的16/64 frozen-feature替换。不能拿它的失效或 TEST 几何分析直接证明本项目 H2T 失败，也未验证本项目8GB最小设置。|
|[SelMix，ICLR 2024][SELMIX]，p5 Eq2–4、p6 Eq6–8、p7 Alg1/Theorem4.2|保持首样本标签做 feature mixup；以混合损失产生参数更新方向 `Vij=−∂Lmix_ij/∂W`，估计目标 ψ 沿该方向的 Gain；`P(i,j)=softmax(sG)ij` 选择类别对，周期使用 validation 的 confusion/centroid 信息。|已具体处理“混合的方向是否提升所需结局”和选择分布。与本地 GU 的 TRAIN balanced CE reference、逐样本三档强度不同；Office 输入 CE 梯度也不同。差异不等于新颖缺口。其结论依赖近似与优化假设，不能据此说本地贪心规则应当有效。|
|[Tailoring Mixup to Data for Calibration，ICLR 2025][SK]，p4 Theorem3.1–3.2，p5–6 Eq2–5|Similarity Kernel Mixup 用样本对归一化距离决定 τ，再以 inverse-Beta-CDF 将均匀 λ 映射为配对相关的对称 Beta 插值系数；图像与标签同权混合。|距离/分布外混合/标签不确定性与校准之间的权衡已有具体处理；不是 H2T hard-label 通道替换，也不是判别梯度方向效用。不能据此证明 LowVS 语义正确，更不能以“只混近邻”当新问题。|
|[Decoupling/cRT，ICLR 2020][CRT]，p4 §4、p5 Eq3–4；[Logit Adjustment，ICLR 2021][LA] arXiv v2 p5 Eq9、p6 Eq10|cRT 重训分类头；τ-normalization/LWS 仅改变 W 类尺度。Logit adjustment 使用 `f_y−τ logπ_y` 的类常数偏移或训练损失的 prior offset。|说明“b 不解释”远不足以排除分类器偏置。W 范数与方向需要区别于加性 logits 偏置；两者也不是完全相同方法。本地尚未产生排除这些解释后的稳健残差。|
|[CurrMix，CVPR Findings 2026][CURRMIX]，仅会议索引摘要可得|索引描述交替容易/困难样本参与多图 MixUp。|**FULL_MECHANISM_UNVERIFIED**：本次正文入口403，没有核实公式、页码和控制实验。不把它用作某一候选已被完整实现的决定性证据；不编造其与 RAVS/LowVS 等价。|

PIF 的通道均值是 permutation-invariant 的聚合量；完整 Eq3 仍含 F，不能把它误读为“所有输出已不依赖通道置换”或任意语义注意力。这个公式层面的区分不是本项目新实证缺陷，也没有因此设计模块。

## 2. 候选问题去重矩阵

以下候选仅用于审查既有叙事，不是下一组实验安排。G1 是本轮 README 的去重门槛，不是旧报告中同名但定义不同的 G1。

|候选命题|G1：已有哪个相同干预/反事实？|G2：现象与更简单解释|G3：具体文献重叠及未回答部分|本轮处理|
|---|---|---|---|---|
|LowVS 独有安全方向可改善 Tail|MASK、NORM、BOUNDARY 已直接研究掩码/范数/方向|Full 小幅正，Matched 变号；静态边界干扰不预测训练好坏|H2T 通道选择、PI 全向量比例、SEL 邻类/置信度是不同算子；不存在自动的新颖收益证明|停止该收益叙事，不改 Bottom-K 或换方向评分|
|风险/梯度效用决定 donor 或融合强度|K4、GU、RAVS、Office risk 已覆盖相应诊断或规则|GU/RAVS 原门槛失败；Office 高 AUROC 只对应同头即时 CE；剂量/代理/时间尺度尚未隔离|SelMix 的参数更新方向与 ψ Gain 已覆盖效用导向选择；SEL/DBG 覆盖其他可靠性筛选。尚未有独立训练收益残差|不得以 RiskOff 等名称重启试错|
|Tail receiver 应跳过直接融合|RECEIVER 已预注册 TailOff−RandomOffMatched|没有结果；局部受损类样本极少，不构成 Tail 群体稳定失效|SEL 改 target 次数，PI 调比例，均非精确同一 skip 反事实；这不补上缺失的实证|保留 PROTOCOL_ONLY，不重新立项或自动执行|
|融合标签应分配给 donor/first-crossed class|TARGET 已预注册；CLEAN_CE 检验的是不同辅助损失|TARGET 没结果；clean-CE 改善 Tail 同时损失 Overall，未证明真语义目标|Mixup/SK 用软目标；H2T/PI/SEL/SelMix 保留 receiver 标签。特定 crossing 规则的新颖性未确立|不把未跑写成负结果，也不据此 GO|
|H2T 收益超越预测偏置重分配|NORM W/b hybrid 已做；另有 bias-vs-discrimination 提案|只排除保存终点 b-only；广义类偏置/W尺度/方向/选择尚未排除。组内排序和 bias-envelope 的提案没有结果|cRT、LWS/τ-normalization、logit adjustment 已具体研究校准。狭义组偏置不变量能否出现稳定残差尚未知|不是本轮新问题；不执行 oracle 网格，也不将其推广成新模块|
|IF200 必然触发 H2T 负迁移|IF200 四臂已直接测试该单点|Shared−DR 两 seed Overall/Tail 均正；跨 IF 混杂且极小类变化|DBG 的 diffusion 负迁移不同；不能借其故事替代本地效应|停止 IF sweep 或另造压力数据寻找失败|
|PI 的过量融合/表示交互造成作者方法失效|旧 PIF ratio/endpoint 重分配和新2参数重建已有；作者来源审计另做|两参数重建仍退化，但作者完整 parity 未建立；没有已证实根因|PI 已讨论 self mass、语义噪声、校准交互；这个作者开放问题不等于我们已有新结论|BLOCKED_PARITY，先保留归属边界，不扩展机制|
|Office domain mismatch 可成为训练效用筛选器|Office risk 已用 receiver/class/mask 配对 domain 条件|donor identity 与 domain 同时变化，固定模型即期风险不等于训练收益；本地 VAL 已用|SEL/FeatRecon/DBG 是相关但不同的邻域/支持/筛选机制；没有一篇被冒称解决了精确 Office 反事实|局部诊断有效，但 G2 与独立验证路径不足，不能 GO|

## 3. 真正仍未知的部分为何没有变成本轮 GO

“即时难度不等于长期训练效用”“冻结特征上的收益是否超越简单校准”“表示与分类器校准如何交互”仍然没有被本地证据完全解释。文献中也没有因此获得适用于本项目的无条件定理。但问题未完全解决只是必要条件；本轮还需要未重复、可复核的强现象和独立评价路径。现有证据不能同时满足这些条件。

此次检索截至 2026-10-28，围绕判别方向、增强效用、选择性 Mixup 与指定五篇论文进行定向核查，不声称穷尽全部文献。正式 PI 版本和 CurrMix 细节未确定，会限制“全领域无研究空白”的断言；它们不影响本轮更窄结论：**尚无一项由现有证据支持、可通过 G1/G2/G3 的候选。**

[H2T]: https://ojs.aaai.org/index.php/AAAI/article/download/29262/30382
[H2T_V3]: https://arxiv.org/pdf/2306.06963v3
[PI]: https://arxiv.org/pdf/2506.00625v2
[SEL]: https://openaccess.thecvf.com/content/ICCV2025/papers/Jian_Supervised_Exploratory_Learning_for_Long-Tailed_Visual_Recognition_ICCV_2025_paper.pdf
[FEATRECON]: https://proceedings.iclr.cc/paper_files/paper/2025/file/adb2075b6dd31cb18dfa727240d2887e-Paper-Conference.pdf
[DBG]: https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Decision_Boundary-aware_Generation_for_Long-tailed_Learning_CVPR_2026_paper.pdf
[SELMIX]: https://proceedings.iclr.cc/paper_files/paper/2024/file/44a8ab79377a7fa99575b939f1b98a21-Paper-Conference.pdf
[SK]: https://proceedings.iclr.cc/paper_files/paper/2025/file/1eac91b56dc20d4f2046951453dde527-Paper-Conference.pdf
[CRT]: https://arxiv.org/pdf/1910.09217v2
[LA]: https://arxiv.org/pdf/2007.07314v2
[CURRMIX]: https://openaccess.thecvf.com/content/CVPR2026F/html/Jian_CurrMix_Curriculum-Enhanced_MixUp_for_Long-Tailed_Visual_Recognition_CVPRF_2026_paper.html
