# PI-H2T 可信复现科学结果

日期：2026-10-27；run-id：`pih2t_if100_seed123_42_v1`。状态：**200 + 10 + 10 epochs已实际完成；PAPER_ALIGNED_INDEPENDENT_RECONSTRUCTION_NOT_AUTHOR_VERIFIED**。

## 固定终点指标

|固定终点|Overall %|Many %|Medium %|Few %|实际平均donor mass|
|---|---:|---:|---:|---:|---:|
|Stage I epoch200|39.91|67.92|38.26|7.14|无融合|
|Stage II no_fusion epoch10|45.53|67.08|46.34|17.79|0.00000000|
|Stage II h2tf epoch10|36.50|29.86|43.29|36.55|0.77208249|
|H2TF − matched no_fusion（百分点）|-9.03|-37.22|-3.06|+18.76|—|

所有数值来自本run真实官方CIFAR100 TEST预测，不从历史报告抄入；以保存logits的argmax重新核算。每类100图，Many/Medium/Few为[0,36)/[36,71)/[71,100)。StageI只选固定epoch200，两组StageII只选固定epoch10，没有逐epoch TEST挑best或按准确率更改机制。

本轮H2TF相对匹配no_fusion出现局部Overall退化，差值为-9.03个百分点。这只适用于本checkpoint/这对seed及预先锁定的配置，不等于总体统计显著性、作者全流程精度匹配、独立测试集确认或创新模块收益。若存在退化，不自动归因论文错误；若有收益，也不据此填补尚未核实的作者配置。

相对StageI固定终点，no_fusion Overall变化=+5.62个百分点，H2TF变化=-3.41个百分点。StageI与StageII还同时改变了采样、优化目标输入和BN训练流，不能把这一跨stage差值视作只改变一个因素的归因结果。

## PIF与融合比例

PIF仅2个无bias参数，顺序[a_residual,b_feature]：初始=[0.2613041400909424, 0.616179883480072]；epoch200=[3.2602970600128174, 0.1977022886276245]；有效F系数a+b=3.457999349。完整200轮轨迹在stage1_epochs.csv，两组StageII均保持这套骨干/PIF参数哈希不变。

H2TF的训练接收者权重r均值=0.22791751，实际供体权重1-r均值=0.77208249，std=0.04177631，min/max=0.52489048/0.91532910。分位数见RATIO_DISTRIBUTIONS.json。这里汇总全部10轮共108470个训练样本出现次数；no_fusion实际donor mass为0，其candidate ratio只用于相同计算/日志，不参与混合。

每轮完整r、实际使用的donor mass、receiver/donor标签保存在stage2各组train_ratios_epochXX.npz；二进制留本地，SHA/大小见RESULTS_INDEX.json。未交换r与1-r，也未把论文约.65的接收者r作为待拟合目标。

## 逐类/逐图变化

H2TF净改善类别43，净下降57，不变0。在完全相同TEST标签顺序上，no_fusion错→H2TF对796张，no_fusion对→H2TF错1699张。每类计数/组别/胜负在per_class_paired_changes.csv；每类100图，因此该类净正确数差恰为百分点差。

两组checkpoint源SHA、初始FC、全部10轮真实图像/标签流SHA和最终BN buffers严格相同；骨干/PIF参数未更新。差异是StageII采用接收者向量还是Eq7融合向量。no_fusion也前向donor是预先声明的matched BN traffic控制，不能把它称作作者未改动的原DR实现。

本配对运行可支持“启用这套已锁定H2TF算子后的条件变化”判断；供体权重分布、Many/Medium/Few差值是可直接复核的事实。它不能单独证明某个比例定义、PIF初始化、冻结reference W、BN update或作者checkpoint选择是论文精度差距的根因。作者配置未定项与复现误差来源仍未解决，不能由一次总体下降反推论文错误，也没有据此再搜索配置。

## 与作者论文分别陈列

PI-H2T arXiv2506.00625v2（PDF p10 Table6、p12 Fig9）给CIFAR100-LT IF100/ResNet32/CE：PIF StageI42.19%，StageII best49.22%、last49.04%。这些是作者报告，不是本run指标；作者PIF精确初始化、Eq8 W来源/detach、BN buffers和checkpoint选择没有完全恢复。本run固定StageI epoch200，与作者可能使用的best路径不同，因此不以数值相等作为唯一parity门禁，不将两套协议混合汇总。

历史25参数PIF实验未用于本次模型初始化、选择、对照或指标合并。新checkpoint为本run内stage1_final_epoch200.pth，SHA256 `64f53e45a1b7f73f494f9c88374607e714b58dc6eb55118fa31154c4389f93ef`。

## 科学边界

核心Eq3/6/7/8的字面实现已核实；完整作者程序parity仍未建立。本轮是一个StageI seed123与StageII seed42的干净初步复现，没有追加seed、第三臂、参数搜索、TailOff、MarginProtect或新注意力机制。仅就本次已锁定重建报告现象，不据此宣称发表方法缺陷或新颖研究结论。
