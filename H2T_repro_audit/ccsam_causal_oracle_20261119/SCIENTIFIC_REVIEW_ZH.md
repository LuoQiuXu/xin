# CC-MIA 受控因果干预与 Oracle：科学审查报告

实际分析完成：2026-11-19T23:15:38+08:00

三个种子均完成30轮，六个实际组，共91,800次真实SGD更新。全部门禁通过，官方TEST零访问。
本轮停止在科学证据交付；Gate准入为 **HUMAN_REVIEW_ONLY**，没有实现或训练Gate。

## 三个种子的固定末轮原始指标

|Seed|组|Overall%|Macro%|Many%|Few%|固定Many Margin|
|---|---|---|---|---|---|---|
|123|C2|46.4646|46.4646|62.6471|23.0000|4.915808|
|123|BASE_C3|45.0505|45.0505|56.7647|27.6667|4.264955|
|123|TARGETED|44.7475|44.7475|58.5294|26.3333|4.766818|
|123|RANDOM_MATCHED|44.9495|44.9495|57.6471|27.0000|4.391490|
|123|FREQUENCY_MATCHED|44.9495|44.9495|56.7647|27.6667|4.306761|
|123|ORACLE_SELECTION|44.8485|44.8485|59.4118|26.3333|4.990144|
|124|C2|45.7576|45.7576|61.7647|24.0000|4.897622|
|124|BASE_C3|45.1515|45.1515|57.0588|28.3333|4.264147|
|124|TARGETED|45.1515|45.1515|58.5294|27.0000|4.765933|
|124|RANDOM_MATCHED|45.2525|45.2525|57.9412|28.0000|4.358571|
|124|FREQUENCY_MATCHED|45.2525|45.2525|57.9412|28.0000|4.333880|
|124|ORACLE_SELECTION|45.4545|45.4545|60.0000|26.6667|4.991300|
|125|C2|45.8586|45.8586|63.8235|23.3333|4.920338|
|125|BASE_C3|45.0505|45.0505|57.0588|28.3333|4.257823|
|125|TARGETED|45.0505|45.0505|58.8235|26.6667|4.762877|
|125|RANDOM_MATCHED|45.0505|45.0505|57.6471|28.0000|4.343251|
|125|FREQUENCY_MATCHED|44.8485|44.8485|57.3529|27.6667|4.340621|
|125|ORACLE_SELECTION|44.7475|44.7475|60.0000|25.6667|4.993893|

Evaluation每个已覆盖类恰好10张，故本轮Overall与Macro相同；不是两项独立收益。

## 配对效果量、样本SD和分层Bootstrap 95%区间

准确率差值单位为百分点。区间条件于三个固定种子、固定掩码和本开发集，不是总体显著性检验。

|对照|指标|均值差|样本SD|Bootstrap 95% CI|
|---|---|---|---|---|
|TARGETED_minus_BASE_C3|many_accuracy|+1.666667|0.169809|[+0.588235, +2.843137]|
|TARGETED_minus_BASE_C3|fixed_many_margin|+0.502901|0.001865|[+0.475595, +0.531079]|
|TARGETED_minus_BASE_C3|few_accuracy|-1.444444|0.192450|[-3.111111, +0.222222]|
|TARGETED_minus_BASE_C3|overall_accuracy|-0.101010|0.174955|[-0.841751, +0.673401]|
|TARGETED_minus_BASE_C3|macro_accuracy|-0.101010|0.174955|[-0.841751, +0.673401]|
|TARGETED_minus_RANDOM_MATCHED|many_accuracy|+0.882353|0.294118|[+0.196078, +1.764706]|
|TARGETED_minus_RANDOM_MATCHED|fixed_many_margin|+0.400772|0.022873|[+0.377805, +0.424292]|
|TARGETED_minus_RANDOM_MATCHED|few_accuracy|-1.000000|0.333333|[-2.555556, +0.555556]|
|TARGETED_minus_RANDOM_MATCHED|overall_accuracy|-0.101010|0.101010|[-0.740741, +0.505051]|
|TARGETED_minus_RANDOM_MATCHED|macro_accuracy|-0.101010|0.101010|[-0.740741, +0.505051]|
|TARGETED_minus_FREQUENCY_MATCHED|many_accuracy|+1.274510|0.612255|[+0.392157, +2.254902]|
|TARGETED_minus_FREQUENCY_MATCHED|fixed_many_margin|+0.438122|0.019617|[+0.413311, +0.463848]|
|TARGETED_minus_FREQUENCY_MATCHED|few_accuracy|-1.111111|0.192450|[-2.444444, +0.111111]|
|TARGETED_minus_FREQUENCY_MATCHED|overall_accuracy|-0.033670|0.210269|[-0.639731, +0.606061]|
|TARGETED_minus_FREQUENCY_MATCHED|macro_accuracy|-0.033670|0.210269|[-0.639731, +0.606061]|
|ORACLE_SELECTION_minus_BASE_C3|many_accuracy|+2.843137|0.169809|[+1.470588, +4.313725]|
|ORACLE_SELECTION_minus_BASE_C3|fixed_many_margin|+0.729471|0.005799|[+0.697340, +0.762460]|
|ORACLE_SELECTION_minus_BASE_C3|few_accuracy|-1.888889|0.693889|[-3.666667, -0.111111]|
|ORACLE_SELECTION_minus_BASE_C3|overall_accuracy|-0.067340|0.324702|[-0.942761, +0.808081]|
|ORACLE_SELECTION_minus_BASE_C3|macro_accuracy|-0.067340|0.324702|[-0.942761, +0.808081]|
|ORACLE_SELECTION_minus_TARGETED|many_accuracy|+1.176471|0.294118|[+0.392157, +2.156863]|
|ORACLE_SELECTION_minus_TARGETED|fixed_many_margin|+0.226570|0.003983|[+0.217695, +0.235965]|
|ORACLE_SELECTION_minus_TARGETED|few_accuracy|-0.444444|0.509175|[-1.333333, +0.222222]|
|ORACLE_SELECTION_minus_TARGETED|overall_accuracy|+0.033670|0.308591|[-0.370370, +0.404882]|
|ORACLE_SELECTION_minus_TARGETED|macro_accuracy|+0.033670|0.308591|[-0.370370, +0.404882]|

## 支持证据

TARGETED在3/3种子中同时超过两个匹配随机对照的Many Accuracy与固定Margin，达到预注册受控干预判据。
相对RANDOM_MATCHED的Many平均提升0.8824pp，相对FREQUENCY_MATCHED提升1.2745pp。
六个seed/control逐一删除单个Many类后，两个主指标方向仍为正，未由单个类单独决定。
Calibration/Evaluation参考影响排序相关约0.99855–0.99904；这些Evaluation评分只在封存后计算，未用于重选。

## 反对证据和代价

TARGETED/Oracle Suppression相对BASE_C3的Many平均提升1.6667pp，未达到2.0pp实用门槛；Few下降1.4444pp，Overall/Macro下降0.1010pp。
Oracle Selection的Many平均提升2.8431pp、Margin提升0.729471，Few均值下降1.8889pp，按提前冻结的均值口径实用效果标记通过。
但seed125的Few下降2.6667pp，超过单种子−2pp描述性边界；Few均值的条件Bootstrap区间为[-3.6667,-0.1111]pp，不能保证尾类损失不超过2pp。
Oracle Selection的Overall/Macro均值下降0.0673pp，三个种子方向混合；不得用Many或Margin单项提升宣布整体实验成功。
Top1收益在每个seed/control对照中仅覆盖2–6个Many类；Margin改善更广泛。这支持有限受控可操作性，不证明所有Many类均受益。

## Oracle约束与剩余空间

Oracle Suppression就是TARGETED：同一掩码、同一训练结果，二者差距为零，不能当作独立重复。
Oracle Selection固定保留221对，抑制其余1989对；Suppression固定抑制221对。两者约束不同，不能将Selection叫作全局理论上界。
Selection相对TARGETED的Many平均再提高1.1765pp、Margin再提高0.226570，同时Few平均再下降0.4444pp。

## 匹配和证据局限

RANDOM_MATCHED精确匹配source频次箱×target类计数；FREQUENCY_MATCHED匹配source频次箱×target频次箱。连续频次数值并非完全相同：source SMD最大0.1077，target SMD最大0.0544。
精确计数与连续平衡见analysis/matched_control_strata.csv和matched_control_balance.csv。
Oracle Selection的抑制集很大，不能将其抑制集Jaccard与221对保留集混同；保留集稳定性另见oracle_retained_mask_stability.csv。
原Stage-I在完整1980张holdout上选模，且父研究已查看该集合；本轮隔离不能撤销历史选择。Evaluation是TRAIN-derived development，不是独立测试集。
class0缺失，Many仅覆盖34/35类；每类Calibration只有10张。Bootstrap条件于封存掩码，不覆盖重新估计掩码的不确定性。
干预是source→classifier-row直接梯度的控制实验；clean路径保留。日志train_loss是原C3标量监测值，不声称新梯度来自新的端到端损失。

## 决策

**支持预注册范围内的受控负交互可操作性；存在明显Many/Few代价，整体收益尚未建立。**
Gate准入留待人工综合评价；本轮不自动准入、不设计新Gate，也不启动额外实验。
完整支持/反对证据见analysis/SCIENTIFIC_REVIEW_DECISION_ZH.json；该补充保留原DECISION.json及全部既有输出。
