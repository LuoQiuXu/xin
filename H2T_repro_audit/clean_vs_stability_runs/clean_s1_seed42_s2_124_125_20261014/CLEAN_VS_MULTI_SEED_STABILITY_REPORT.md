# Clean Stage I 固定下 Low-VS vs Uniform 多种子稳定性验证

完成时间：2026-10-14T14:23:28+08:00。新增实验仅 seed124、seed125，各 Uniform-PerSample / Low-VS 10 epochs，共4次训练、3400次正式 SGD 更新。

## 1. 实验配置与无 TEST 选择的 Clean 来源

本轮固定已有 Clean Stage I seed42、200 epochs 中以99类 TRAIN 验证集 Overall 选出的 epoch184。没有重新训练 Stage I；四组均从该 Checkpoint 的同一 Backbone 和同一分类器权重独立初始化。dry-run 分类器及其优化器均丢弃，正式训练使用全新空 SGD 状态，未加载 seed123 或其他 Stage II 权重。

- 原 Clean Checkpoint：`D:\xiangmuxunlian\新\H2T_repro_audit\clean_vs_runs\clean_s1_seed42_vs_seed123_20261013\stage1\checkpoints\clean_stage1_best.pth`。
- 本轮逐字节副本：`D:\xiangmuxunlian\新\H2T_repro_audit\clean_vs_stability_runs\clean_s1_seed42_s2_124_125_20261014\locked\clean_stage1_best.pth`。
- SHA256：`7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。
- Clean 统计文件：`D:\xiangmuxunlian\新\H2T_repro_audit\clean_vs_stability_runs\clean_s1_seed42_s2_124_125_20261014\class_feature_statistics.npz`，SHA256 `e81a4897a965d642f832bd6429e0e5a7d136725daaa069d9ded96f87ab64aef9`。
- 统计文件与 seed123 完全相同，未重算均值、方差、shrinkage 或 epsilon；epsilon=`4.26226893568829e-07`。

历史 epoch198 模型曾用 TEST 选模；本轮加载的是重新训练的 Clean epoch184，没有加载旧模型或旧模型统计。Clean 的选模证据为 `clean_stage1_protocol.json`、复制的 `CLEAN_STAGE1_PROTOCOL.md` 与原 Clean 的 `test_guard_stage1_train.json`。数据划分和研究方向已有历史观察，不能把整个研究过程称为完全未受既有结果影响的盲测。

|项目|锁定设置|
|---|---|
|数据 / Backbone|真实 CIFAR-100-LT IF100；10847张、100类；ResNet-32|
|Stage II|10 epochs；batch128；85 batches/epoch，末batch95；每组850步|
|冻结|全部 Backbone、Conv、BN affine；BN.eval；全部 running_mean/var/counter 逐位固定|
|更新|只更新100类线性分类器，6500个参数；CE；无MixUp|
|SGD|base_lr0.1 × factor0.2，首epoch实际0.02；momentum0.9；weight_decay0.0005；dampening0；nesterov=False|
|调度|原Stage II cosine；factor0.2；同seed123，无额外warmup或搜索|
|随机性|Python / NumPy / Torch / CUDA 按124或125同步；cuDNN benchmark=False、deterministic=False，沿用seed123|
|采样|BS类别均衡接收者；IS供体；每epoch恰好遍历LT供体各一次；供体/标签规则不变|
|融合|[B,64,8,8]；rho0.25；每样本16个无重复完整通道复制；标签保持接收者|
|Uniform|64通道共享基础Gumbel流选16；均匀不放回|
|Low-VS|当前增强接收特征在线评分；Bottom48后均匀不放回选16|
|回退|完全沿用seed123：供体类n<20或64维评分极差<=1e-12时使用同一Uniform掩码；非有限评分立即中止|

评分为 `S=v_reg/[v_reg+(mu_h-current_augmented_receiver_GAP)^2+epsilon]`；`v_reg=(1-lambda)*v_ddof0+lambda*v_pooled`，`lambda=5/(n+5)`。池规模48、固定统计 epsilon、替换比例与训练预算均未调整。评分不依赖正在训练的分类器，不使用 Margin 或验证集定义分数。

逐epoch实际学习率：`0.0200000000, 0.0195105652, 0.0180901699, 0.0158778525, 0.0130901699, 0.0100000000, 0.0069098301, 0.0041221475, 0.0019098301, 0.0004894348`。精确 YAML、`experiment_config.json`、源代码快照及源文件校验均已保存。

## 2. 数据与指标解释

验证集是官方 TRAIN 中未进入10847张LT训练子集的1980张原始图像，类1..99各20张，缺class0（apple）；训练/验证raw ID交集0、像素哈希交集0。100维logits始终保留，预测缺失类0仍计错。该验证集已被前期实验反复使用，因此它仍是开发验证结果，不能视为新独立测试或直接对比论文的100类官方TEST准确率。

|指标|类ID / 训练频次定义|有效类数|验证图像|
|---|---|---|---|
|Overall / Macro|类1..99；每类20，因此两者相等|99|1980|
|Head|0..35，class0缺失|35|700|
|Medium（ID）|36..70|35|700|
|Tail|71..99|29|580|
|Many|LT训练数>100；0..34，class0缺失|34|680|
|Medium-n|20<=LT训练数<=100；35..69|35|700|
|Few|LT训练数<20；70..99|30|600|

本轮没有打开或解码官方TEST图像/标签，没有创建CIFAR100(train=False)，没有TEST指标、模型选择或评估。准备阶段仅有一次不解码文件的官方test MD5校验，其余阶段test文件打开次数为0。分离证据保存在 `test_guard_*.json`。

## 3. 固定第10 epoch 的主要结果（%）

|Seed|Method|Overall|Head|Medium（ID）|Tail|Few|
|---|---|---|---|---|---|---|
|124|Uniform-PerSample|47.0707|60.1429|50.7143|26.8966|27.5000|
|124|Low-VS|47.3232|59.8571|51.0000|27.7586|28.3333|
|125|Uniform-PerSample|46.8687|60.1429|50.1429|26.8966|27.5000|
|125|Low-VS|47.1717|59.5714|50.7143|27.9310|28.5000|

|Seed|Method|Many|Medium-n|Few|Macro|
|---|---|---|---|---|---|
|124|Uniform-PerSample|61.0294|50.2857|27.5000|47.0707|
|124|Low-VS|60.5882|50.7143|28.3333|47.3232|
|125|Uniform-PerSample|61.0294|49.7143|27.5000|46.8687|
|125|Low-VS|60.4412|50.2857|28.5000|47.1717|

## 4. Low-VS − Uniform 的配对增益

|Seed|ΔOverall pp|ΔTail pp|ΔFew pp|Overall净正确张数|Tail净正确张数|Few净正确张数|
|---|---|---|---|---|---|---|
|124|+0.2525|+0.8621|+0.8333|5|5|5|
|125|+0.3030|+1.0345|+1.0000|6|6|6|

|指标|新增两seed平均增益 pp|样本SD pp（ddof=1）|正增益seed|
|---|---|---|---|
|Overall|+0.2778|0.0357|2/2|
|Tail|+0.9483|0.1219|2/2|
|Few|+0.9167|0.1179|2/2|

整体提升对应每个seed仅净增加5张和6张正确图像。与此同时，Head两seed均下降，平均 -0.4286 pp；Many平均 -0.5147 pp。该结果包含类别组之间的权衡，不能描述为所有类别普遍改善。

这里只把两个新增 Stage II 种子作为本轮验证。n=2 的均值/样本标准差为描述统计，不构成显著性或稳定泛化证据；两个seed共用同一验证图像和同一个Stage I，不代表两个独立数据集。

### 已观察先导 seed123（不重训，单独列示）

|Method|Overall|Tail|Few|
|---|---|---|---|
|Uniform-PerSample|47.0707|26.3793|27.0000|
|Low-VS|47.3737|27.4138|28.0000|

先导Δ：Overall +0.3030 pp, Tail +1.0345 pp, Few +1.0000 pp。先导结果已用于提出本轮问题，不能把它与两新增种子共同当成三个预先独立确认实验。三种子合并描述另存 `results_summary.json/three_seed_descriptive_only`，主要判断只使用124/125。

## 5. Tail 类别级分析

|Seed|提升类数|下降类数|不变类数|
|---|---|---|---|
|124|3|0|26|
|125|6|1|22|

seed124 Tail提升ID：`[81, 94, 99]`；下降ID：`[]`；不变ID：`[71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98]`。

seed125 Tail提升ID：`[81, 84, 89, 94, 95, 99]`；下降ID：`[96]`；不变ID：`[71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 85, 86, 87, 88, 90, 91, 92, 93, 97, 98]`。

两个seed均提升ID：`[81, 94, 99]`；均下降ID：`[]`；一升一降ID：`[]`。

`per_class_delta.csv` 保存2×100行，包含类名、LT样本数、验证数、两方法正确数、准确率、Δpp、净正确数；class0保留一行、准确率空白而非0。每类只有20张，1张改变即5pp；不能把个别类别变化解释为可靠类别规律。`paired_changed_images_epoch10.csv` 逐张保存预测发生改变的验证索引，未挑选有利实例。

![固定epoch10增益](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/plots/paired_epoch10_deltas.png)

![全部Tail类别的增益](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/plots/tail_class_deltas.png)

## 6. 训练公平性、dry-run 与冻结核验

每个seed的Uniform和Low-VS共享分类器初始权重、BS接收者、IS供体、batch边界、实际增强像素、冻结特征图、GAP、64维评分以及Gumbel基础噪声。Low-VS每一步更新前对照同seed Uniform 的13个控制字段，850/850 batch 全部一致；两个不同seed的采样/增强本来就不同。`pairing_hash.json` 和两组 `batches.csv` 保存逐batch证据，不仅比较种子或数据配置名称。

私有流沿用seed123：BS Python seed；IS Torch seed+100003；训练增强 Torch seed+300009；通道Gumbel NumPy seed+700021。已先重建旧seed123全部10个epoch的BS/IS/Gumbel数组并逐位比较成功，未重新运行seed123训练。通道选择不消耗增强流；环境CPU/CUDA RNG哈希也逐batch匹配。CUDA未强制deterministic，因此不承诺跨机器/软件版本重新运行必然逐位一致，本轮实际输入和特征哈希匹配已经验证。

|seed|真实batch dry-run|Loss Uniform / Low|峰值分配MiB|峰值保留MiB|
|---|---|---|---|---|
|124|PASS|1.658290 / 1.984610|77.23|98.00|
|125|PASS|1.957921 / 2.325306|77.23|98.00|

Dry-run验证16个不重复完整通道、所选位置等于供体、未选位置等于接收者、标签不变、官方GAP/分类器调用等价、CE有限、分类器有有限梯度并更新、Backbone没有梯度。dry-run更新被丢弃。正式训练每epoch检查所有Backbone状态逐位一致；交付时再次加载8个best/final文件，对照Clean权重逐位比较全部93个Backbone参数张量（含Conv及62个BN affine张量）、93个BN缓冲张量（31组mean/var/counter）。31个counter均保持15640。分类器weight/bias均有非零变化，best/final重新加载后的同一TRAIN验证指标与保存记录完全一致。详见 `parameter_and_BN_verification.json`。

历史文件检查：2853个已存在文件SHA256全部不变，包含官方运行副本、Stage I/II历史权重与报告。本轮源码与运行输入分别由 `source_lock.json` 和 `source_snapshot/` 保护。

## 7. 掩码干预与统计组成（全部10 epochs）

|Seed|Method|均值V_selected|均值B_selected|均值V/(V+B+eps)|均值所选S|与Uniform重合通道|完全相同率|回退抽样数|
|---|---|---|---|---|---|---|---|---|
|124|Uniform-PerSample|3.7180|10.1229|0.3050|0.5316|16.0000|100.0000%|3050|
|124|Low-VS|3.8231|13.2857|0.2494|0.4072|12.1117|3.2756%|3050|
|125|Uniform-PerSample|3.7155|10.1383|0.3044|0.5312|16.0000|100.0000%|3050|
|125|Low-VS|3.8213|13.2948|0.2492|0.4069|12.1192|3.2673%|3050|

V为所选通道的正则化供体类方差之和，B为类均值与当前增强接收GAP的平方偏移之和，逐样本比例的均值不等于均值V与均值B的比例。回退同seed123，IS类别/实例分布没有删除小类。`sampling_diagnostics.json` 从保存的在线评分和基础噪声重新构造全部掩码，逐位匹配；每个掩码16个唯一索引，且IS每epoch为10847样本的全排列。

两新增seed均观察到Low-VS的B_selected高于Uniform、V/(V+B+epsilon)较低，平均每个掩码约3.88个所选通道发生改变。实际干预已经生效，不能由此把偏移增大解释为收益原因；这还同时改变了通道身份及其他相关特征。

低S仅表示在上述统计定义下方差相对偏移的占比低，不等于语义上更远、有害/有益或分类重要性。组成变化与精度是否改善是两类证据。

## 8. 最佳epoch与文件（仅描述，不替代固定epoch10）

|Seed|Method|最佳epoch|Overall|Head|Medium（ID）|Tail|Few|
|---|---|---|---|---|---|---|---|
|124|Uniform-PerSample|4|47.1212|60.5714|49.8571|27.5862|28.1667|
|124|Low-VS|9|47.4242|59.7143|51.4286|27.7586|28.3333|
|125|Uniform-PerSample|6|47.2222|60.0000|50.8571|27.4138|28.0000|
|125|Low-VS|6|47.1717|59.5714|50.4286|28.2759|28.8333|

最佳文件按1980张TRAIN验证集Overall严格提升保存（相等保留最早epoch），final固定epoch10。没有依据最佳epoch比较挑选本轮主结果。

最佳模型口径下的Low−Uniform Overall分别为 seed124 +0.3030 pp、seed125 -0.0505 pp；seed125并未在这一口径占优。因此“重复正增益”的结论明确限定在预先规定的固定第10 epoch，不能扩展为任何模型选择方式下都稳定领先。

|Seed|Method|文件|Epoch|Checkpoint|
|---|---|---|---|---|
|124|Uniform-PerSample|best|4|[model_best.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/uniform_seed124/checkpoints/model_best.pth.tar)|
|124|Uniform-PerSample|final|10|[model_final.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/uniform_seed124/checkpoints/model_final.pth.tar)|
|124|Low-VS|best|9|[model_best.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/lowvs_seed124/checkpoints/model_best.pth.tar)|
|124|Low-VS|final|10|[model_final.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/lowvs_seed124/checkpoints/model_final.pth.tar)|
|125|Uniform-PerSample|best|6|[model_best.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/uniform_seed125/checkpoints/model_best.pth.tar)|
|125|Uniform-PerSample|final|10|[model_final.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/uniform_seed125/checkpoints/model_final.pth.tar)|
|125|Low-VS|best|6|[model_best.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/lowvs_seed125/checkpoints/model_best.pth.tar)|
|125|Low-VS|final|10|[model_final.pth.tar](D:/xiangmuxunlian/新/H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/lowvs_seed125/checkpoints/model_final.pth.tar)|

## 9. 训练耗时与显存

|Seed|Method|训练秒|运行秒|峰值分配MiB|峰值保留MiB|
|---|---|---|---|---|---|
|124|Uniform-PerSample|95.59|96.79|78.98|98.00|
|124|Low-VS|92.64|93.93|79.45|116.00|
|125|Uniform-PerSample|94.55|95.95|78.98|98.00|
|125|Low-VS|94.06|95.26|79.45|116.00|

|Seed|Low−Uniform训练秒|训练耗时相对变化|Low−Uniform运行秒|
|---|---|---|---|
|124|-2.95|-3.09%|-2.86|
|125|-0.50|-0.52%|-0.69|

训练秒是10个训练epoch累计，包含在线评分、CPU增强、配对哈希、同步及统计，不包含epoch末验证；运行秒还包括保存和验证等。Uniform也计算在线分数及候选掩码用于公平核验，因此这里不能推断无审计工具生产实现的Low-VS净开销；两个顺序运行的时间差也受系统噪声影响。

## 10. 对四个问题的判断

1. **Low-VS是否在两个seed均优于Uniform？** 是，固定第10 epoch 的 Overall 在两个新增 seed 都高于 Uniform。

2. **Tail/Few提升是否稳定？** Tail 为 2/2 个 seed 正增益，Few 为 2/2。当前条件下重复了正向变化，但两个新增种子尚不足以证明广泛稳定性。

3. **是否支持“方向性扰动可能是H2T收益来源之一”？** 本轮直接检验的是在冻结特征空间中改变通道掩码后的分类器训练结果。即使Low-VS出现提升，也只说明某些通道统计组成变化可能值得研究；本轮没有无融合DR对照、没有独立控制同类/跨类供体或特征方向/扰动幅度，无法确定原H2T收益来源，也不能将B增大等同于有用的跨类别方向。结果不稳定时，该收益解释更缺少支持。

4. **是否值得继续论文方法设计？** 可以把机制作为有限的后续研究线索；当前结果只支持继续提出可证伪的受控假设，不支持宣称新方法有效或有论文级泛化收益。

不确定性还包括：只有一个Clean Backbone与一套固定类统计；验证每类20张且缺class0；验证集被多轮研究复用；新seed数仅2；cuDNN沿用非强制确定性设置。新seed确实改变BS/IS、增强和通道基础随机流，所以结果只度量这些Stage II随机性在该固定基础上的重复性。

## 11. 原始数据、配置与复现命令

实验根目录：`D:\xiangmuxunlian\新\H2T_repro_audit\clean_vs_stability_runs\clean_s1_seed42_s2_124_125_20261014`。关键文件：`epoch_validation_metrics.csv`（40行），`paired_epoch10_deltas.csv`，`per_class_delta.csv`（200行），`paired_epoch_deltas.csv`，`checkpoint_manifest.csv`（8行及SHA），`timing_and_memory.csv`，`results_summary.json`，`pairing_hash.json`，`sampling_diagnostics.json`，`parameter_and_BN_verification.json`。

每个 `uniform_seed124/`、`lowvs_seed124/`、`uniform_seed125/`、`lowvs_seed125/` 包含完整 `training.log`、`batches.csv`、`effective.yaml`、`experiment_config.json`、10个epoch的验证logits/labels/predictions、全部实际掩码与组成、best/final权重及加载核验。`plans/{seed}/` 包含固定BS/IS方案、共享Gumbel、在线64维评分、配对输入和两策略全部实际掩码以及dry-run记录。

索引：`split_indices.npz` SHA256 `c8579feef60d7705472ec00aeac85fa00a66c243761e0a8d76c462c663286ca1`；缓存验证特征SHA256 `a52852cd37f3daeb2116073253fb3e7cc5e798e0f4dd315849fb337064e34485`；源图像来自 `D:\xiangmuxunlian\新\H2T\data\cifar100\cifar-100-python\train`，预检查完整原始索引/像素比较见 `preflight.json`。

以下为本轮实际命令；训练入口拒绝覆盖已有目录。复现应在新的独立目标目录复制锁定源/输入后执行，不能在本目录重新训练。

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
$env:PYTHONIOENCODING="utf-8"
# cwd = 本报告的独立实验根目录；python为D:\app\anaconda\envs\pytorch\python.exe
# prepare_stability.py -> stability.py plans 为首次准备，已完成且禁止覆盖
$env:PYTHONHASHSEED="124"
& "D:\app\anaconda\envs\pytorch\python.exe" -B -u stability.py dryrun --seed 124
& "D:\app\anaconda\envs\pytorch\python.exe" -B -u stability.py train --seed 124
$env:PYTHONHASHSEED="125"
& "D:\app\anaconda\envs\pytorch\python.exe" -B -u stability.py dryrun --seed 125
& "D:\app\anaconda\envs\pytorch\python.exe" -B -u stability.py train --seed 125
# 只读结果分析；拒绝覆盖已生成结果
& "D:\app\anaconda\envs\pytorch\python.exe" -B -u analyze_stability.py
& "D:\app\anaconda\envs\pytorch\python.exe" -B -u write_stability_report.py
```

**已按停止条件结束：没有seed126，没有其他策略、超参数搜索、Stage I重训或官方TEST评估。新增实验等待下一步分析。**
