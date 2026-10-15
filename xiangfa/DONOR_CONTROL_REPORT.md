# 相似／随机／远距离供体对照实验报告

## 要检验的强假设

在相同训练、表征提取、轨迹指标和供体数量下：

`similar-support > random-support > distant-support`

这里的 `>` 指头到尾频率退化轨迹的稳定支撑区域响应更相似，而不是单次分类准确率更高。

## 实验设计

- 数据：CIFAR-100 官方训练集；每类固定 400 张训练池、100 张校准池，二者无交集。
- 官方测试集：本阶段完全不加载。
- 频率节点：`400, 208, 56, 14, 4`。
- 拉丁频率设计：每个 seed 训练 5 个独立 ResNet-32；每个环境都有完全相同的频率直方图，每个类别恰好访问每个频率一次。
- 正式规模：3 个训练 seed（42、43、44），共 15 个 100-epoch 模型；每个模型每 epoch 抽样 40,000 张。
- 所有环境在同一 seed 内使用完全相同的初始化；不同 seed 使用不同初始化。
- 主轨迹指标：`angle_q50 + angle_q95 + tangent_trace`。
- 相似供体：平衡参考空间原型余弦 top-5。
- 随机供体：排除 top-5 和 bottom-5 后，按预先固定的随机种子抽取 5 类。
- 远距离供体：平衡参考空间原型余弦 bottom-5。
- 统计单位：先对同一目标类跨 seed 求平均，再对 100 个目标类进行配对 bootstrap 和单侧符号置换检验。

## 主要结果

| 训练 seed | similar | random | distant | 均值排序 |
|---:|---:|---:|---:|---|
| 42 | 0.4683 | 0.5218 | 0.3143 | random > similar > distant |
| 43 | 0.5419 | 0.5126 | 0.4704 | similar > random > distant |
| 44 | 0.5144 | 0.5149 | 0.4313 | random ≈ similar > distant |
| 三 seed 聚合 | 0.5082 | 0.5164 | 0.4053 | random ≈ similar > distant |

三 seed 聚合配对检验：

| 对比 | 平均差 | 95% bootstrap CI | 单侧置换 p | 判断 |
|---|---:|---:|---:|---|
| similar − random | -0.0082 | [-0.0336, 0.0170] | 0.739 | 无优势 |
| random − distant | +0.1111 | [0.0818, 0.1401] | < 0.00002 | 显著为正 |
| similar − distant | +0.1029 | [0.0715, 0.1339] | < 0.00002 | 显著为正 |

- 三 seed 聚合后，只有 36% 的目标类满足逐目标严格排序。
- 平衡原型相似度与所有 4,950 对轨迹相似度的 Spearman 分别为 `0.0271、0.0047、-0.0059`，基本为零。
- 因此，平衡原型相似度不是轨迹相似度的单调代理。

## 拉丁日程匹配稳健性检查

为了排除“同一拉丁组共享相同环境顺序”带来的混杂，只在与目标类具有完全相同频率日程的 19 个类别内重新选择 top-5、随机-5、bottom-5。

三 seed 聚合结果：

- similar：0.5157
- random：0.5270
- distant：0.4940
- similar − random：-0.0114
- random − distant：+0.0330
- 严格逐目标排序比例：23%

这个检查仍不支持 `similar > random`，因此主结果不是拉丁日程差异造成的。

## 其他相似度定义的诊断

在 seed 42 上，事后用 fusion、confusion、KNN 重新分组，也没有得到稳定的完整三段排序。该结果仅用于定位原因，不替换预注册的原型相似度主分析。

## 结论

当前证据**否定强结论** `similar-support > random-support > distant-support`。不能在论文中声称最近邻相似供体稳定优于随机供体。

当前可以支持的较弱结论是：

> 远距离供体的频率退化轨迹明显较不匹配；相似供体和普通非极端随机供体整体相当。平衡空间相似度更适合用于排除不合适供体，而不足以单独选出最优供体。

这意味着后续方法应从“最近邻迁移”调整为“候选供体筛除 + 轨迹可靠性／兼容性加权”。真正的恢复效用仍需在相同恢复算法下比较 similar、random、distant 对尾类 NCM、固定线性探针、oracle 几何距离和最终准确率的影响；即使该任务层比较偶然出现排序，也不能反向宣称平衡相似度与退化轨迹存在稳定单调关系。

## 可复现文件

- 实验程序：`latin_donor_trajectory_controls.py`
- 设计审计：`outputs/latin_donor_trajectory_controls/design_audit.json`
- 聚合统计：`outputs/latin_donor_trajectory_controls/aggregate_hypothesis_test.json`
- 各 seed 原始成对结果：`outputs/latin_donor_trajectory_controls/seed_*/all_pair_trajectory_similarity.csv`
- 各目标三组结果：`outputs/latin_donor_trajectory_controls/seed_*/target_group_comparisons.csv`
