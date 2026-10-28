# 下一阶段：H2T 增益是否超过“头/中/尾预测偏置再分配”？（IF200，两种子，零训练）

**这不是再做一个 H2T 改进模块。** 先用旧的、真正配对且已完成的 IF200 DR/H2T 结果，提出一个可以被直接反驳的限定性解释，再决定研究方向是否有继续价值。下一步若没有科学上清晰的新信号，就应 **STOP**，避免换名字继续开发 Low-VS / RAVS / GU / Risk-Off 等。

## 为什么这个问题现在有意义

历史已经证明：在同一 Stage I、冻结 BN 的 IF200 实验中，Original-BatchShared 相对 DR，seed124/125 的 Overall 都是正向、Tail 也是正向。但这不等于 H2T 实现了新的“语义表示学习”——StageII Backbone 已冻结，只有线性分类器改变。若收益只是把分数全局从 Head 推向 Tail，则与类别偏置调整差别有限。需要一个**不能由组级预测偏置独立造成的观察**。

**单个反事实：** 若只对类别组添加常数偏置，是否能解释 H2T 的每组性能？对同一真实类组中类别的排序是否因此改变？

形式化：`z'_c=z_c+b_{g(c)}`，其中 Head 偏置设零作为基准，中/尾各有一个常量。对某个真实组 g 内的样本：

`argmax_{c in g} (z_c+b_g) = argmax_{c in g} z_c`。

因此：
- **指标1：true-group restricted Top-1**。只允许在样本真实组内预测，逐张比较 DR 与 H2T。此指标对组常数偏置**严格不变**，可以证伪“仅仅改变头中尾三个组偏置”的狭义解释。
- **指标2：DR 组偏置 oracle envelope**。对 DR logits 在同一重复使用的开发集上枚举预先冻结的33×33组偏置，比较是否存在一点同时达到 H2T 的 Overall、Head、OfficialMedium、Tail。这个 oracle 明确**过于乐观，不是可用方法，不得报告其最大开发集结果为独立泛化性能**。

这两个指标也不能排除**每个类别不同的偏置、权重范数、方向、训练轨迹变化**等解释，所以绝不将通过理解为“证明 H2T 学到了有益语义”或可发表贡献。

## 复用而不是再训练

单一数据：`H2T_repro_audit/if200_stress/results/`（原分支 `experiment/h2t-if200-stress-20261017`，作为本分支的起点）。

需本地已有未上传的大文件，脚本不下载、重算或训练：

- `seed124/DR-NoFusion/validation_epoch10.npz`
- `seed124/Original-BatchShared/validation_epoch10.npz`
- `seed125/DR-NoFusion/validation_epoch10.npz`
- `seed125/Original-BatchShared/validation_epoch10.npz`

每臂还需 `summary.json`，以及历史真实 TRAIN 留出的 `validation_labels.npy`、`split_indices.npz`：

`H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/`

如果原始 NPZ 不在本地，不从官方 TEST 或别的 checkpoint 临时“补数据”，**直接停止**。不要因为作者确实有公开 CIFAR100-TEST 就转去测它。

### 固定条件

- CIFAR-100-LT IF200，Stage I seed42，固定第200轮；Stage II seeds 124/125，固定第10轮，750个 SGD step。
- 对照仅为 DR-NoFusion vs Original-BatchShared。其余 Low-VS、Uniform、RAVS 结果不能混入主比较。
- ResNet32、BN 完全冻结、同一初始FC和采样/增强计划。已保存日志可进一步证实输入配对。
- 开发验证为原始 CIFAR TRAIN 留出的1980张，class0缺失，class1..99各20张，100维 logits。
- 原始作者 `shift_bn=True`，本次固定 BN，故不是原作者协议的精确复现，不作作者方法失效结论。
- 融合后仅改变了分类器，不能宣称改善了冻结 Backbone 的表征。

### 不可改变的 oracle 范围

以 DR logits 每张图最大与第二大分数差的中位数作为正的归一化尺度 `s`。令 Head 组偏置0，Medium、Tail偏置各在 `[-4,-3.75,...,3.75,4] × s` 范围内枚举，共1089种。单个组内所有类别加同样偏置。**这是乐观解释上界，不是开发集选模。** 不改变网格、不放大范围、不能因失败继续扫描。

## 科学判断标准（预先锁定）

对每个 Stage II seed 同时要求：
1. H2T 普通 Overall 和普通 Tail 都严格大于 DR；
2. H2T true-group restricted Tail 准确率比 DR 至少多 1.0pp；
3. 在固定 grid 中，DR 的任何组偏置都不能同时达到 H2T 的 Overall、Head、OfficialMedium 和 Tail。

**两个 seed 全部通过**才记录 `EXPLORATORY_NON_GROUP_BIAS_SIGNAL`。任一不通过 → `NO_SUPPORT_STOP_THIS_HYPOTHESIS`，不要临时加 seed、换 IF、扫权重、发明模块“挽救”该假设。

PASS 仅表示值得在独立 Stage I/样本划分/数据集核验，不意味着已发现新颖的方法问题。FAIL 仅表示这个预定的狭义假设未过关，也不证明所有形式的分类器偏置都足够。

## 用法（只离线 NumPy）

仓库根目录执行：

```powershell
python -m unittest discover -s H2T_repro_audit/h2t_bias_discrimination_20261028 -p "test_*.py" -v

python H2T_repro_audit/h2t_bias_discrimination_20261028/analyze.py --output H2T_repro_audit/h2t_bias_discrimination_20261028/results/if200_bias_gate_01
```

若测试或 preflight 失败，不可放宽验证；修复程序型错误前记录真实信息。不运行 Stage I/II、新的训练、官方 TEST，拒绝覆盖旧结果目录。

输出4个小文件：`within_group.csv`、`oracle_grid.csv`、`decision.json`、`SCIENTIFIC_REPORT.md`。SHA256 记录所有消耗的源 NPZ/JSON/labels/split；输入失败**不会创建结果目录**。

## 最关键的解释限制

- 真正算的是“每个真实组内部分类排序”，**不是**有标签语义的正交表征、新类间语义距离，也不是新 loss。
- 即使组内排序增加，也可能来自类别细分决策边界、W 方向、W 范数或单类偏置，不说明 H2T 学到了新 Backbone 特征。
- oracle 在同一个被反复使用的 TRAIN 开发集上搜索，绝不能用它挑选新方法，也不能以此声称测试精度。过去多个研究已经反复看过此开发集，故整个阶段的结论只限于探索。
- 两个 StageII seed 共享同一个 StageI checkpoint 和验证图像，不是跨模型/数据集的独立显著性证据。
- 只有先从现有结果找到不能被简单再分配解释的稳定信号，才讨论一个新方法，否则不继续在旧方向做工程。

## 本轮 Codex 附加职责

严格检查脚本所依赖的历史 `summary.json` 的 `stage1_sha`、`classifier_initial_hash`、`backbone_unchanged`、`BN_frozen`、`TEST_used`，以及源代码中 `validation_epoch10.npz` 保存逻辑、保存的 checkpoint / split 与此处路径一致。JSON 内的声明只算日志证据；如真实 checkpoint 资产不在本地，必须报告未能独立复核 checkpoint，而不是写“已核验参数逐位一致”。

只提交小的离线 CSV、JSON、Markdown 报告到本分支，不提交模型、NPZ、原数据或任何测试图像。
