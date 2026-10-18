# Low-VS 尾类干净判别保持：三臂科学报告

## 1. 首要比较：C（TopCompetitor）− B（CleanCE）

固定第10epoch为唯一主终点。A=LowVS，B=A+尾类干净CE，C=A+尾类干净最强竞争类别softplus；两个辅助项均为0.5×尾类损失之和/完整batch大小。所有准确率来自同一1980张原始TRAIN留出验证，class1..99各20张，class0缺失，100维输出保留。

|seed|ΔOverall pp|ΔHead pp|ΔTail pp|ΔFew pp|Δ验证CE|
|---|---|---|---|---|---|
|124|+0.2020|+0.2857|-0.5172|-0.5000|-0.0068|
|125|+0.1515|+0.7143|-1.0345|-1.0000|-0.0073|

配对计数中的A/B指比较式左/右策略；wrong→right与right→wrong使用同一张验证图像的真实类别，不是只看预测类别组改变。

|seed|全体gain/loss/net|Head gain/loss/net|Tail gain/loss/net|Few gain/loss/net|
|---|---|---|---|---|
|124|10/6/+4|3/1/+2|2/5/-3|2/5/-3|
|125|10/7/+3|5/0/+5|0/6/-6|0/6/-6|

**预设规则判读：** C−B未满足两个seed均改善Tail且Overall不下降的预设方向条件；停止本轮候选，不扫描系数、margin或种子。

## 2. 六组固定epoch10绝对结果与辅助比较

|seed|组别|Overall|Head|OfficialMedium|Tail|Many|Medium|Few|Macro|validation_CE|
|---|---|---|---|---|---|---|---|---|---|---|
|124|LowVS|47.3232|59.8571|51.0000|27.7586|60.5882|50.7143|28.3333|47.3232|2.1530|
|124|LowVS-CleanCE|46.7172|58.8571|48.7143|29.6552|59.7059|48.2857|30.1667|46.7172|2.1632|
|124|LowVS-TopCompetitor|46.9192|59.1429|49.4286|29.1379|60.0000|49.0000|29.6667|46.9192|2.1564|
|125|LowVS|47.1717|59.5714|50.7143|27.9310|60.4412|50.2857|28.5000|47.1717|2.1532|
|125|LowVS-CleanCE|46.5152|58.0000|48.8571|29.8276|58.8235|48.4286|30.3333|46.5152|2.1645|
|125|LowVS-TopCompetitor|46.6667|58.7143|49.4286|28.7931|59.5588|49.0000|29.3333|46.6667|2.1572|

以下B−A、C−A用于区分额外干净监督与特定竞争约束，不替代首要C−B。

|seed|比较|ΔOverall|ΔHead|ΔTail|ΔFew|净正确图像|Tail净正确|Δ验证CE|
|---|---|---|---|---|---|---|---|---|
|124|LowVS-CleanCE - LowVS|-0.6061|-1.0000|+1.8966|+1.8333|-12|11|+0.0101|
|124|LowVS-TopCompetitor - LowVS|-0.4040|-0.7143|+1.3793|+1.3333|-8|8|+0.0033|
|125|LowVS-CleanCE - LowVS|-0.6566|-1.5714|+1.8966|+1.8333|-13|11|+0.0113|
|125|LowVS-TopCompetitor - LowVS|-0.5051|-0.8571|+0.8621|+0.8333|-10|5|+0.0040|

两seed均值仅作描述，不作显著性推断：

|比较|Overall|Head|OfficialMedium|Tail|Many|Medium|Few|Macro|validation_CE|
|---|---|---|---|---|---|---|---|---|---|
|LowVS-TopCompetitor - LowVS-CleanCE|+0.1768|+0.5000|+0.6429|-0.7759|+0.5147|+0.6429|-0.7500|+0.1768|-0.0070|
|LowVS-CleanCE - LowVS|-0.6313|-1.2857|-2.0714|+1.8966|-1.2500|-2.1429|+1.8333|-0.6313|+0.0107|
|LowVS-TopCompetitor - LowVS|-0.4545|-0.7857|-1.4286|+1.1207|-0.7353|-1.5000|+1.0833|-0.4545|+0.0037|

LowVS-TopCompetitor−LowVS-CleanCE：Overall同向；Head同向；Tail同向；Few同向。

LowVS-CleanCE−LowVS：Overall同向；Head同向；Tail同向；Few同向。

LowVS-TopCompetitor−LowVS：Overall同向；Head同向；Tail同向；Few同向。

## 3. 减少Tail→Head错误与真正判对Tail必须分开

当原先错误预测为Head的Tail图像转到另一个错误类别时，Tail→Head计数减少，但Tail准确率没有增加。下面直接分解同图变化，不把组别迁移当成正确识别。

|seed|比较|Tail gain/loss/net|Head错误→正确Tail|Head错误→其他错误|正确Tail→Head错误|其他错误→Head错误|Tail→Head Δ|
|---|---|---|---|---|---|---|---|
|124|LowVS-TopCompetitor - LowVS-CleanCE|2/5/-3|0|1|2|7|8|
|124|LowVS-CleanCE - LowVS|12/1/+11|8|12|0|0|-20|
|124|LowVS-TopCompetitor - LowVS|8/0/+8|6|6|0|0|-12|
|125|LowVS-TopCompetitor - LowVS-CleanCE|0/6/-6|0|1|2|2|3|
|125|LowVS-CleanCE - LowVS|12/1/+11|5|8|0|0|-13|
|125|LowVS-TopCompetitor - LowVS|6/1/+5|3|7|0|0|-10|

完整真值组×预测组混淆见 `group_confusion.csv`，其中Medium为OfficialMedium（ID36..70），不是训练频次Medium。预测class0算Head错误。

## 4. 逐类别得失与离散分辨率

全量99类、每类n=20的正负变化见 `per_class.csv`，每臂绝对正确数见 `per_class_accuracy.csv`，逐张变化及原TRAIN raw ID见 `paired_changed_images.csv`。不挑有利类别代表全部结果。

### LowVS-TopCompetitor − LowVS-CleanCE

- 两seed均净改善：全体3类；Head 1类；Tail 0类。Tail类别及两seed净正确数：无。
- 两seed均净下降：全体4类；Head 0类；Tail 4类。Tail类别及两seed净正确数：72(-1/-1), 89(-1/-1), 91(-1/-1), 95(-1/-1)。

### LowVS-CleanCE − LowVS

- 两seed均净改善：全体9类；Head 0类；Tail 8类。Tail类别及两seed净正确数：71(+1/+1), 72(+1/+1), 79(+1/+1), 85(+1/+1), 89(+3/+2), 91(+1/+1), 95(+1/+1), 96(+2/+2)。
- 两seed均净下降：全体14类；Head 5类；Tail 1类。Tail类别及两seed净正确数：94(-1/-1)。

### LowVS-TopCompetitor − LowVS

- 两seed均净改善：全体5类；Head 0类；Tail 4类。Tail类别及两seed净正确数：71(+1/+1), 79(+1/+1), 89(+2/+1), 96(+1/+2)。
- 两seed均净下降：全体8类；Head 3类；Tail 0类。Tail类别及两seed净正确数：无。

Overall一张=0.050505pp，Head一张=0.142857pp，Tail一张=0.172414pp，Few一张=0.166667pp；每类别一张=5pp。两seed平均很小的分数差不能掩盖只改判少数图片，亦不能将3960次预测当作独立验证样本。

## 5. 训练曲线与固定终点

以下显示epoch1/5/10；全部60行逐epoch指标、主CE、系数化辅助损失和Tail接收次数见 `curves.csv`。best epoch仅记录最高Overall的诊断值，不保存或选用其模型作为主结果。

|seed|组别|Overall 1/5/10|Head 1/5/10|Tail 1/5/10|best Overall epoch|best Overall|
|---|---|---|---|---|---|---|
|124|LowVS|46.4646/46.8182/47.3232|58.5714/60.0000/59.8571|27.0690/27.2414/27.7586|9|47.4242|
|124|LowVS-CleanCE|46.4646/46.5657/46.7172|57.8571/58.8571/58.8571|29.6552/29.4828/29.6552|6|46.8182|
|124|LowVS-TopCompetitor|46.4646/46.4646/46.9192|58.4286/59.2857/59.1429|28.4483/28.2759/29.1379|4|47.2222|
|125|LowVS|46.6667/47.1717/47.1717|58.4286/59.7143/59.5714|27.4138/27.0690/27.9310|6|47.2222|
|125|LowVS-CleanCE|46.4646/46.6667/46.5152|57.7143/58.4286/58.0000|30.0000/30.0000/29.8276|5|46.6667|
|125|LowVS-TopCompetitor|46.9192/46.9192/46.6667|58.4286/58.8571/58.7143|29.1379/29.1379/28.7931|1|46.9192|

epoch10训练损失组成（每个epoch按样本数加权；aux列已乘λ=0.5）：

|seed|组别|融合CE|系数化干净辅助项|总loss|Tail接收draws|
|---|---|---|---|---|---|
|124|LowVS|1.476298|0.000000|1.476298|3233|
|124|LowVS-CleanCE|1.461104|0.053999|1.515103|3233|
|124|LowVS-TopCompetitor|1.471975|0.024414|1.496389|3233|
|125|LowVS|1.448848|0.000000|1.448848|3142|
|125|LowVS-CleanCE|1.436302|0.049611|1.485914|3142|
|125|LowVS-TopCompetitor|1.445628|0.023475|1.469104|3142|

## 6. 协议、来源与正确性证据

- 锁定Clean StageI seed42 epoch184，SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`；原IF100统计SHA256 `e81a4897a965d642f832bd6429e0e5a7d136725daaa069d9ded96f87ab64aef9`。未训练StageI、未重算统计或改Bottom48/随机16/rho0.25/供体抽样。
- 原始TRAIN文件SHA256 `735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b`；10,847张LT的raw IDs严格匹配作者seed0，像素/标签对应官方TRAIN；与验证raw IDs和完整像素无重叠。验证缓存校验对应原历史模型，未读官方TEST或对TEST做哈希。
- ResNet32的463,504个参数及31个BN完全冻结；完整186项Backbone状态每epoch逐位核对，running_mean/var、affine及num_batches_tracked保持不变，BN计数均15,640。只优化原100维FC的6,500个参数，各臂weight和bias确有变化。
- 三臂均从同一StageI FC独立初始化、空SGD状态；lr0.1、lr_factor0.2，实际首epoch单FC组lr0.02，沿用历史cosine，momentum0.9、wd0.0005、batch128、末batch95、10epochs×85更新=850更新/组，总计6组5,100次更新。
- 每一臂每seed的850批、全部7项输入/掩码字段均与旧LowVS-Full真实记录匹配；共5,100个历史batch匹配、35,700项字段比较。4个辅助臂另与本轮A对照匹配3,400个batch/23,800项字段。字段包括索引/标签、实际两套增强像素、冻结特征、在线VS评分、Gumbel、Uniform与Low-VS通道。
- preflight在两个seed验证16个完整通道来自供体、未选通道逐位等于接收者；label始终为接收标签。损失CPU单测验证tail-only样本梯度、排除y且detach的最强竞争类别、m=0，以及无Tail batch精确退化到A。
- “干净”是同一接收图像crop/flip后的未融合特征，并非未增强图像。j由当前正在更新的FC在训练干净特征上产生，既不改供体也不改VS掩码。尾类样本辅助监督仍可更新Head输出行权重，因此Head存在代价并不违反tail-only样本定义。
- 各组最后checkpoint重载后，预测逐项相同，logits按提供代码rtol=atol=1e−6检查，全部指标差<1e−8；不把该logit容差检查描述成逐bit比较。独立报告脚本从已保存数组重算指标和配对净得失，没有额外前向或训练。
- 原提供losses.py和run_pilot.py逐字节未改，无阻塞bug修复。只新增日志/访问防护封装和本报告分析，原数据、模型、增强、在线评分、融合及优化函数均直接复用原文件。有效YAML残留的旧model_dir展示字段未使用；实际所有输出路径由新runner的RESULTS确定，历史写保护和前后校验均通过。

历史LowVS数值复核（本轮A−旧LowVS-Full，pp）：

|seed|ΔOverall|ΔHead|ΔTail|ΔFew|
|---|---|---|---|---|
|124|+0.0000|+0.0000|+0.0000|+0.0000|
|125|+0.0000|+0.0000|+0.0000|+0.0000|

## 7. 限制、警告与停止

- 同λ、同Tail样本和同全batch归一化不等于梯度预算完全匹配：全类CE与单最强竞争softplus的函数、梯度大小和方向不同。C−B检验两种给定辅助监督的整体差别，不能只归因于抽象“竞争关系”。
- 最强负类pairwise ranking/softplus是已有类型的目标；本轮不主张该loss本身首创，即使有效也仅为后续特征级模块提供方向证据。Backbone固定，本轮不构成表征学习提升。
- 一个StageI和两个相关StageII seed，共用同一1980张开发验证；该集已反复用于历史选择和分析，class0缺失。没有独立显著性、官方100类TEST性能或最终泛化证据；不能与论文TEST精度直接比较。
- CPU单测及preflight均触发提供代码float(loss)记录导致的requires_grad转标量UserWarning；已原样保存console日志，测试通过，未修改损失或训练策略处理该非阻塞警告。
- 本轮不增加lambda、margin、K、epoch或seed扫描，不新增第四组，不启动特征级训练或其他数据集。未执行任何版本控制操作。
- C−B未满足两个seed均改善Tail且Overall不下降的预设方向条件；停止本轮候选，不扫描系数、margin或种子。

## 8. 完整产物与复现入口

全部相对路径以 `H2T_repro_audit/lowvs_disc_preserve/results/` 为根：

- `preflight.json`、`dataset_attestation.json`、`execution_source_lock.json`、`historical_manifest.json`、`static_review.json`。
- `seed{124,125}/{三臂}/epochs.csv`、`batch_signatures.csv`、`effective.yaml`、`config.json`、`final_fc.pth`、`validation_epoch10.npz`、`summary.json`。
- `epoch10.csv`、`paired.csv`、`per_class.csv`、`curves.csv`、`group_confusion.csv`、`historical_replay.csv`、`scientific_summary.json`、`LOWVS_DISC_RESULTS.md`。
- `loss_components.csv`同时保存每epoch主CE、未乘λ的辅助项和乘λ辅助项；未乘λ值由原始加权记录除以固定0.5恢复，显式标为derived，归一化仍是完整batch而非尾类样本数。
- `paired_detailed_counts.csv`、`tail_error_transitions.csv`、`per_class_accuracy.csv`、`paired_changed_images.csv`、`mean_differences.csv`、`replay_verification.csv`、`classifier_state_verification.csv`、`scientific_verification.json`。
- `console/*.log`、`guards/*.json`、`execution_registry.json`保存每个完整命令、stdout/警告、起止时间、退出码、GPU环境及峰值显存。`provided_source/`和`runtime_source_snapshot/`保留原代码和锁定依赖；`delivery_manifest.json`记录原始文件SHA256。
- 使用 `D:\app\minicondaaa\envs\xiaoju\python.exe`，从仓库根按 `user_task_instructions.txt` 的losses→preflight→seed124三组→seed125三组→report顺序执行。本轮执行封装只加保护和日志，已有结果拒绝覆盖；没有训练重试。
