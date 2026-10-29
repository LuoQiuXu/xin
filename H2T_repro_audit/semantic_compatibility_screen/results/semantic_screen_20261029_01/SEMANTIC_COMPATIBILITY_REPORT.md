# SEMANTIC COMPATIBILITY REPORT

日期：2026-10-29（Asia/Shanghai）。唯一工作分支：`experiment/semantic-compatibility-screen-20261029`。起始提交 `c2c1b4e92b54e764baab27ebe3fbace614bad071`；本轮未切换分支。执行依据为上级两份原始[任务TXT](../../CODEX_SEMANTIC_COMPATIBILITY_SCREEN.txt)和[协议](../../SEMANTIC_SCREEN_PROTOCOL.md)，以及用户对210张/21类受限范围的明确确认，见 [SCOPE_RESOLUTION.json](SCOPE_RESOLUTION.json)。

**最终科学判断：NO-GO。** 本轮完成真实冻结特征筛查，未得到足以将“语义兼容性”作为当前关键失效原因、并推进Stage-II验证的联合证据。受体等权的同superclass−异superclass标签一致性为 **−2.3054 pp**，按fine类聚类重采样的95%区间 **[−5.6334, +1.3445] pp**；margin差为 **−0.096405**，区间 **[−0.210702, +0.002282]**。两个区间均跨零，标签一致性的点估计也未达到事前锁定的5pp实际效应门槛。不能把这个NO-GO写成“已证明语义完全无关”，也不能把反向点估计写成“不同superclass必然更安全”。

## 1. 独立覆盖率发现及获批范围

**独立发现：21/30个Few类具备同superclass的Many供体。** 这一结果仅由官方CIFAR100 fine/coarse标签、类号与IF100指数LT计数决定，尚未使用模型预测，因此不是H2T或PI-H2T失效证据。

原始TRAIN共50,000图，100个fine类各500图；官方coarse标签逐fine类唯一，20个superclass各含5个fine类。原始TRAIN SHA256为 `735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b`，MD5为`16019d7e3df5f24257cddd939b257f8d`；meta MD5为`7973b15100ade9c7d40fb424638fde48`，与本机torchvision官方数据定义一致。映射直接读取TRAIN的`coarse_labels`，名称来自已校验meta，没有按名称猜关系或使用外部语义embedding。

按历史seed0重建并逐像素/标签核验10,847张LT TRAIN。**Few严格为训练类频<20，Many严格为>100**：Few30类305图，Many35类8,824图。可得同/异superclass至少各4个合法Many供体的Few为21类210图，即类覆盖70.00%、图像覆盖68.85%。seed20261029优先抽取的300张Few中只有205张满足条件，因此本轮先提出范围澄清；收到用户方案2确认后固定全部210张合格图像及其21类，未按后续结果挑类、挑正确受体或引入Medium供体。

排除9类95图如下，原因均为同superclass Many供体数为0：

|fine ID|官方fine名称|官方superclass|LT样本数|排除原因|
|---:|---|---|---:|---|
|70|rose|flowers|19|该superclass无Many类/图像|
|74|shrew|small_mammals|15|该superclass无Many类/图像|
|80|squirrel|small_mammals|12|该superclass无Many类/图像|
|81|streetcar|vehicles_2|11|该superclass无Many类/图像|
|82|sunflower|flowers|11|该superclass无Many类/图像|
|85|tank|vehicles_2|9|该superclass无Many类/图像|
|89|tractor|vehicles_2|7|该superclass无Many类/图像|
|92|tulip|flowers|6|该superclass无Many类/图像|
|96|willow_tree|trees|5|该superclass无Many类/图像|

每个superclass的实际供体覆盖完整记录如下；没有Few图像的superclass不被算作本轮受体范围：

|coarse ID|官方名称|Many类数|Many图像|Few类数|Few图像|可覆盖Few图像|
|---:|---|---:|---:|---:|---:|---:|
|0|aquatic_mammals|2|538|2|23|23|
|1|fish|2|589|2|23|23|
|2|flowers|0|0|3|36|0|
|3|food_containers|4|1014|0|0|0|
|4|fruit_and_vegetables|1|500|1|10|10|
|5|household_electrical_devices|1|179|2|17|17|
|6|household_furniture|3|749|2|16|16|
|7|insects|5|1378|0|0|0|
|8|large_carnivores|1|434|2|13|13|
|9|large_man-made_outdoor_things|2|512|1|14|14|
|10|large_natural_outdoor_scenes|2|278|1|18|18|
|11|large_omnivores_and_herbivores|4|760|0|0|0|
|12|medium_mammals|1|102|1|15|15|
|13|non-insect_invertebrates|1|149|3|30|30|
|14|people|2|754|1|5|5|
|15|reptiles|2|271|2|19|19|
|16|small_mammals|0|0|2|27|0|
|17|trees|0|0|1|5|0|
|18|vehicles_1|2|617|1|7|7|
|19|vehicles_2|0|0|3|27|0|

全100类映射/类频、全305个Few的逐图覆盖、优先300抽样及全部superclass见 `FINE_CLASS_COVERAGE.csv`、`RECIPIENT_COVERAGE.csv`、`PREFERRED_300_COVERAGE.csv`、`SUPERCLASS_COVERAGE.csv`。初始`COVERAGE_SUMMARY.json`保留了澄清前状态；最终获批边界由独立的`SCOPE_RESOLUTION.json`明确覆盖。

## 2. 合法Stage I与真实特征

采用既有 **Clean Stage-I seed42、best epoch184**，文件：

`H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013/stage1/checkpoints/clean_stage1_best.pth`

SHA256：`7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。它从随机初始化训练200 epochs，按重复使用的1,980张TRAIN holdout选best（99个观察类，平局取最早），训练及选模没有使用官方TEST标签；历史prepare曾有一次不透明TEST字节MD5完整性检查，本轮没有打开TEST。没有改选旧TEST-best、Stage-II或PIF权重。

从同一历史runtime原模型类strict=True加载ResNet32 Option-A、`feature_tuning=False`及其原100类带bias分类头，权重(100,64)、bias(100,)。只读保留模型源码，不调用旧训练器或会自动构建TEST的Dataset包装器。使用原eval `ToTensor -> Normalize`，mean=(.4914,.4822,.4465)、std=(.2023,.1994,.2010)，不重新随机增强。

先验证8个受体及候选涉及的498张真实图像，再处理其余5,995张，合计 **6,493个唯一LT原始图像**（210个Few受体、6,283个Many供体）。每raw ID只前向一次，完整过程逐字节复用pilot特征。输出池化特征(6493,64)、float32，有限；模型与分类头eval、requires_grad=False、无Dropout、optimizer、反向传播或参数更新，BN及全部状态哈希前后相同。原分类头输出与显式池化特征输入的GPU线性输出逐位一致。

GPU batch64，PyTorch2.14.1+cu132、torchvision0.29.1+cu132、CUDA13.2，RTX5070 Laptop 8GB；cudnn deterministic=True、benchmark=False、TF32=False，matmul TF32=False/precision=highest，deterministic_algorithms=True，CUBLAS_WORKSPACE_CONFIG=:4096:8。完整环境、批次shape和状态哈希在`PILOT_VALIDATION.json`、`FEATURE_EXTRACTION.json`。

诊断NPZ将缓存float32特征及原float32分类头**精确转换为float64数值**，供NumPy统一计算；这没有增加已提取特征的有效精度。GPU FP32头与NumPy FP64重算的最大logit绝对差约`3.22702715e-06`，argmax分歧合计0，不把近似数值一致称为历史哈希重放。

- 特征bank NPZ SHA256：`d31ca4c2df0cfd65ea5c868b1065f26b28623e82fcc204a145971bf8ebf86e00`。
- 输入pairs NPZ SHA256：`984b771257ea03de2461dc1b18dc9bacb6c885a30af7a0677d2f17be6fe0ef97`。
- 每raw ID特征、原始权重及标签的可追溯核验见`ARTIFACT_VALIDATION.json`；大数组只留本地并被忽略，`ARTIFACT_INDEX.csv`保留全部大小/SHA。

## 3. 固定候选、固定比例与距离匹配

`PAIR_PLAN.json`和`PRE_ANALYSIS_PLAN.json`在真实前向前写入；分析计划SHA256 `7973c8b5eeb78f9351297efd20777935a9ef6d33f935d063b1f369cfa9eaee6a`。这属于本轮事前锁定，未声称独立公开注册或确认性检验。

固定全部210个受体，对每个受体按raw-ID局部RNG seed20261029，从合法Many池中无放回抽32个同coarse不同fine供体、32个异coarse供体；同raw ID/同fine排除。各受体完整的同coarse Many池为102–754图、异coarse池为8,070–8,722图。抽样只准备审计候选，不训练或实现新的donor选择策略。

共13,440候选对，两语义臂各6,720；每受体固定同一原图/特征和同一接收标签，donor mass始终严格 **0.25**。原始`screen.py`不改一字，采用默认4个受体内距离分位层，在层内确定性贪心最近距离、无放回匹配；本轮调用前检查新输出目录不存在，未覆盖旧结果。

这里使用任务指定的 `0.75*f_rec + 0.25*f_donor` **冻结标量几何proxy**；没有替换原H2T随机通道融合，也没有执行PI-H2T训练。标量混合点与随机通道mask的风险分布不等价，不能把本轮一致性直接当成原算法最终accuracy。原生PI可选诊断在前向前已锁定跳过：本轮无PIF的Clean基座不提供作者完整PIF/H2TF一致性证据，未据结果另选ratio。

事前质量门槛：≥30不同受体、≥10个fine类、≥5个coarse类、最大fine类受体占比≤20%；全匹配相对距离gap中位数≤5%、90分位≤10%；clean正确比例≥50%。相对gap为`2*abs(d_same-d_diff)/(d_same+d_diff)`。不满足时必须INCONCLUSIVE，不能靠换比例、供体或类范围补救。

事前联合信号门槛：受体等权标签一致性差绝对值≥5pp、标准化margin差≥.2、两指标同方向、≥2/3 fine类及coarse类margin方向一致，且两项fine类聚类重采样95%区间均在相同方向排除0。它是开发筛查推进规则，不是已训练方法的显著性证明。

## 4. 实际匹配质量与proxy有效性

得到 **4,138个匹配对、210个不同受体、21个fine类、13个coarse类**。每受体匹配数6–32，中位20；最大fine类占18/210=8.57%，未由单个受体类别覆盖主导。受体是主要计数单位；共享Many供体、类别与superclass仍造成依赖，不能把4,138行当作独立试验。

|质量指标|真实结果|
|---|---:|
|逐匹配donor mass|same=diff=0.25，逐个严格相等|
|平均绝对Euclidean距离gap|0.089298|
|平均有符号距离gap（same−diff）|−0.007201|
|相对gap均值|1.2354%|
|相对gap中位数|0.4298%|
|相对gap90分位|3.1027%|
|相对gap最大值|23.9048%|
|预设≤5%相对gap敏感性样本|3,906对，仍覆盖210受体|
|clean冻结头判对|108/210，51.4286%|
|clean已错误|102/210，48.5714%|
|clean margin均值 / 中位数|0.087850 / 0.084094|

匹配总体通过事前质量门槛，但不是精确距离控制：232对的gap>5%，个别残差较大。冻结头对将近半数clean受体已经错误，刚超过事前50%门槛，proxy仍明显脆弱。没有为了改善clean正确率而改选权重或删除这102个受体。每受体候选数、匹配数、距离均值/gap、clean margin及同/异指标完整保存于 [RECIPIENT_ANALYSIS.csv](analysis_v2/RECIPIENT_ANALYSIS.csv)，逐匹配残差在 [MATCH_DISTANCE_AUDIT.csv](analysis_v2/MATCH_DISTANCE_AUDIT.csv)。

## 5. Observed：同−异语义差异

冻结目标margin为接收类logit减最大非接收类logit；正的同−异差表示同superclass在此旧分类头下更有利。`margin_drop=margin_mixed-margin_clean`；同受体两臂clean相同，因此其同−异差与mixed margin差是代数相同的量，不算两份独立证据。

|估计口径|样本|标签一致性差 pp|margin差|
|---|---|---:|---:|
|距离匹配前，受体等权|210受体，各32+32候选|−0.4613|−0.033602|
|原始screen按匹配行等权|4,138行|−0.7250|−0.031992|
|**事前主要口径：受体等权**|210受体|**−2.3054**|**−0.096405**|
|按21个fine类等权|21类|−1.7357|−0.103878|
|预设相对距离gap≤5%敏感性，受体等权|3,906行 / 210受体|−2.4768|−0.099275|

匹配行数因受体而异，所以保留原始行等权结果，同时遵守事前受体等权估计；没有观察后挑选更有利的汇总口径。fine类聚类重采样固定2,000次、seed20261029：

|主要指标|受体等权点估计|95%类聚类重采样区间|
|---|---:|---|
|同−异标签一致性|−2.305384 pp|[−5.633435, +1.344499] pp|
|同−异margin|−0.096405|[−0.210702, +0.002282]|
|标准化margin差|−0.206570|开发效应量，非训练收益|

这是固定模型/训练图像内、对类异质性稳定性的描述性重采样，不是训练种子、总体泛化或因果区间。Many供体复用和superclass依赖超出该聚类层级，不能将区间作为严格独立实验的置信保证。

121/210个受体的margin差为负、89为正；14/21个fine类与11/13个coarse类均值为负，方向并非普遍一致。逐类结果全部列出：

|fine ID|N受体|clean正确 %|匹配对|同−异标签一致性 pp|同−异margin|
|---:|---:|---:|---:|---:|---:|
|71|18|61.11|318|-10.3429|-0.217449|
|72|17|0.00|396|+0.7799|-0.046947|
|73|16|56.25|326|-3.1962|+0.014045|
|75|15|100.00|346|-4.1856|-0.079504|
|76|14|92.86|224|-4.2106|-0.098078|
|77|13|23.08|280|-0.0753|+0.057359|
|78|13|76.92|336|+11.7764|+0.299839|
|79|12|33.33|280|+4.9242|+0.022196|
|83|10|80.00|108|-20.6667|-0.905462|
|84|10|70.00|132|-13.6905|-0.290423|
|86|9|66.67|118|+3.3069|-0.040350|
|87|8|12.50|144|-3.5714|-0.091380|
|88|8|50.00|204|-8.4010|-0.159135|
|90|7|42.86|132|+3.7338|-0.016021|
|91|7|14.29|130|-1.2987|-0.320969|
|93|6|33.33|130|+5.4563|+0.203495|
|94|6|83.33|102|-5.8036|+0.011012|
|95|6|0.00|132|+0.0000|-0.085587|
|97|5|60.00|100|+2.5758|+0.028556|
|98|5|0.00|88|+0.0000|-0.411790|
|99|5|60.00|112|+6.4394|-0.054854|

例如sweet_pepper(class83)点估计−20.67pp，而snake(class78)为+11.78pp；这显示类别异质性，不能择取其中一个方向作为结论。仅按clean正确性分层的诊断为：108个clean正确受体标签一致性差−4.8290pp、margin−0.120781；102个clean错误受体为+0.3667pp、margin−0.070596。该分层不用于筛选受体或另立GO。

## 6. Interpretation：A / B / C / D

**A：同一受体面对相关/无关供体，是否有系统性冻结几何损害差异？** 点估计呈轻度反向：同coarse平均较差；更严格距离残差口径方向相近。但不同受体/类存在反向，两个主要区间均跨0，且标签一致性的实际效应不足5pp。当前只能记录不确定的负向关联提示，没有稳定“语义相关供体更安全”的支持，也没有证明“语义相异一定有害”或“语义完全无效”。

**B：是否独立于距离，是否可能由类别混淆、频次或模型错误造成？** 近似匹配后与≤5%残差口径结果相近，说明负向点估计没有在这一Euclidean控制下消失；这不等于语义关系的独立因果作用。匹配后同/异供体所属Many类训练计数均值约251.96/311.89，特征norm均值8.1594/7.5746，仍有频次、norm和类别几何差异。接收标签及语义coarse关系不能在同一对真实图像上随机操控；还存在已训练头的类别混淆与102个clean错误受体。未将这些剩余关联归为纯语义效应，也未追加新的匹配特征或ratio调参。

**C：是否支持“这个场景下H2T随机donor有潜在问题”？** 本轮没有达到足以推进该机制主张的联合门槛。覆盖21/30类是数据可得性，不能当作原方法失效；冻结标量混合的标签一致性和旧head margin不等于随机通道H2T或PI-H2T的Stage-II训练效用。已有donor安全选择、PIF对应打乱对照的否定证据仍成立，不能用本轮proxy绕过。参考 [PRIOR_EVIDENCE_REVIEW.json](PRIOR_EVIDENCE_REVIEW.json)，未继承旧报告中未经作者一致性验证的“完整官方PI-H2T缺陷”措辞。

**D：下一阶段GO还是NO-GO？NO-GO。** 设计/距离/范围质量达到本轮阈值，因此不是因少量无效匹配强行下结论；但联合语义信号要求失败：标签一致性点估计仅2.31pp、两个类聚类区间均跨0。结论是**不足以推进**，不是已排除所有小效应。未进入Stage-II，不设计新模块、donor策略、融合方式或新的训练实验。

## 7. 证据边界及交付

- 全部结论**仅限获批的21个具备同coarse Many供体的Few类**。9类被排除是结构性选择偏差；不能推广至全部尾类、其他IF或其他数据集。
- 受体/供体全部来自此模型实际LT训练集，这是训练图像上的冻结几何诊断，不是独立验证集/官方TEST准确率，更不是Stage-II收益。
- 一个Stage-I seed、一个固定比例/候选计划/距离分层；没有正式多seed、训练或搜索。没有修改历史模型、权重、H2T/PI-H2T方法或旧结果。
- 原始`screen.py`中的`screen_sufficient_pairs=True`只计匹配行数，不等于科学GO；本轮另检查210个不同受体、21类、13superclass、距离质量、proxy和联合效应。
- 初次CPU测试有一个fixture漏设异coarse Many类，已修正fixture，真实覆盖逻辑未改。初次分析导出因NumPy布尔JSON类型失败；第一次输出原地保留、忽略并记录哈希，修正后写入新的`analysis_v2/`。数值CSV/summary在两次导出间逐字节相同，未再提取特征或修改统计门槛；不要使用根目录那份未完成的决策JSON。

正式分析文件位于 **`analysis_v2/`**，原始脚本输出位于 **`screen_output/`**。[VALIDATION_REPORT.md](VALIDATION_REPORT.md)给出验证与实际恢复命令；[ARTIFACT_VALIDATION.json](ARTIFACT_VALIDATION.json)核验历史保护、覆盖、真实缓存、头权重、每对比例及受体/聚类计算；[ARTIFACT_INDEX.csv](ARTIFACT_INDEX.csv)提供所有文件大小与SHA256（不自哈希）。较大特征/原始数值输入、缓存和不完整首次导出只留本地并由`.gitignore`忽略，提交可审查的代码、覆盖/匹配CSV、JSON、科学图和报告。最终提交SHA由同分支Git记录及交付消息提供。
