# 历史实验事实及重新解释

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
