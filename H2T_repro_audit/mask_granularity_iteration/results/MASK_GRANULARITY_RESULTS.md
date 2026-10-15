# H2T 通道掩码粒度：三臂结果与配对限制

完成时间：2026-10-15T19:19:35+08:00。本轮只新增Uniform-BatchShared seed124、125，共两次10-epoch训练、1700次分类器更新；旧Uniform-PerSample和Low-VS均直接复用封存结果。

## 1. 准确科学命题

A：batch共用一个Uniform16/64掩码，与每样本独立Uniform16/64的训练结果是否不同？B：已有Low-VS相对Uniform-PerSample的差异是否仍存在？Shared采用每个真实minibatch第一个样本的Uniform Gumbel16通道，扩展至全batch。这是 **original-like mask granularity**；没有模拟原H2T的torch.randperm RNG路径，不能称完整官方原始复现。

## 2. 协议及用户后续授权

- CIFAR-100-LT IF100，10847张TRAIN；ResNet-32输出[B,64,8,8]。
- Clean Stage I seed42、epoch184，SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。
- 同一初始Backbone/分类器、同一BS接收者/IS供体方案、同一增强流和Gumbel；rho0.25，每样本复制16个完整通道；接收标签不变。
- Backbone及BN affine冻结；BN eval，全部running_mean/var/counter逐位固定；仅6500参数分类器SGD、CE。
- 10epochs、batch128、85batch/epoch（末batch95）。base_lr0.1、momentum0.9、wd0.0005、nesterovFalse；沿用原cosine factor0.2，首epoch实际lr0.02。固定epoch10为主，best严格按验证Overall提升、相等保留最早epoch。

预注册最初要求13字段逐位一致。preflight发现特征、GAP和评分哈希不一致后曾停止。用户随后明确授权：“继续执行，不管失败条件了，缺什么环境在我们的指定python环境里自己pip install”。据此解除历史哈希开跑阻断，并保留全部13字段逐batch比较和原始差异，**没有把不一致记作一致**。授权记录为 `results/continuation_authorization.json`。

解释器固定为 `D:\app\minicondaaa\envs\xiaoju\python.exe`；PyTorch2.14.1+cu132、CUDA13.2。只额外安装PyYAML6.0.3和yacs0.1.8两个配置依赖（--no-deps），没有更改指定PyTorch/CUDA。本轮使用新目录，未覆盖/重跑历史实验。

因此，本轮Shared与旧控制比较带有已知数值执行差异，不能宣称除掩码粒度外所有计算结果逐位相同。数值哈希不一致本身也不能量化误差大小。结果用于描述与提出后续假设，不作为完全消除计算差异的因果证明。

## 3. 真实数据与泄漏防护

使用官方CIFAR-100原始TRAIN，按锁定raw indices验证LT图像与标签，重新按官方seed0类内shuffle及 `int(500*0.01**(c/99))` 重建LT索引完全一致。原TRAIN留出1980张、类1..99各20张；class0缺失；训练/验证raw ID交集0，图像与原TRAIN索引逐项一致。100维logits保留，预测0仍计错。

数据/权重/统计/采样方案/Gumbel/旧控制CSV、验证NPZ和实际掩码都实际存在，并通过锁定文件校验。类别统计SHA256 `e81a4897a965d642f832bd6429e0e5a7d136725daaa069d9ded96f87ab64aef9`，未重新估计。

|指标|定义|有效类数|验证张数|
|---|---|---|---|
|Overall/Macro|class1..99，每类20|99|1980|
|Head|ID0..35，缺0|35|700|
|OfficialMedium / Medium-ID|ID36..70|35|700|
|Tail|ID71..99|29|580|
|Many|LT训练数>100，缺0|34|680|
|Medium / Medium-n|LT训练数20..100|35|700|
|Few|LT训练数<20|30|600|

OfficialMedium与频次Medium不同。本轮没有加载官方TEST图像/标签、没有CIFAR100(train=False)、TEST指标或TEST选模。评估使用封存的TRAIN验证特征，同一1980张图像在两个seed重复出现，不能将3960次预测当独立观测；该验证集曾用于前期开发和Clean选模，不能称盲测或与官方100类TEST准确率直接对比。

## 4. 固定epoch10及best（%）

|Seed|策略|Overall|Head|Medium-ID|Tail|Many|Medium-n|Few|Macro|best epoch|best Overall|
|---|---|---|---|---|---|---|---|---|---|---|---|
|124|Uniform-BatchShared|47.3232|60.2857|51.1429|27.0690|61.1765|50.7143|27.6667|47.3232|4|47.4747|
|124|Uniform-PerSample|47.0707|60.1429|50.7143|26.8966|61.0294|50.2857|27.5000|47.0707|4|47.1212|
|124|Low-VS|47.3232|59.8571|51.0000|27.7586|60.5882|50.7143|28.3333|47.3232|9|47.4242|
|125|Uniform-BatchShared|46.9697|59.8571|50.7143|26.8966|60.7353|50.2857|27.5000|46.9697|5|47.1717|
|125|Uniform-PerSample|46.8687|60.1429|50.1429|26.8966|61.0294|49.7143|27.5000|46.8687|6|47.2222|
|125|Low-VS|47.1717|59.5714|50.7143|27.9310|60.4412|50.2857|28.5000|47.1717|6|47.1717|

Uniform-PerSample/Low-VS四行是2026-10-14封存结果，未在本轮环境重新训练或重算预测；Shared两行是本轮训练。新best/final均重新加载，TRAIN验证指标与所存记录完全一致。

## 5. 逐图配对、Tail/Few得失

差值方向统一为“表中方法 − Shared”或明确注明Low−PerSample。每个对比均从相同1980张验证样本的预测逐图计算；净正确数÷图像数×100与各组准确率差逐项严格一致。

|Seed|对比|wrong→right|right→wrong|净正确|ΔOverall pp|Tail增|Tail失|Tail净|ΔTail pp|Few净|ΔFew pp|
|---|---|---|---|---|---|---|---|---|---|---|---|
|124|Uniform-PerSample - Uniform-BatchShared|7|12|-5|-0.2525|1|2|-1|-0.1724|-1|-0.1667|
|124|Low-VS - Uniform-BatchShared|16|16|0|+0.0000|5|1|4|+0.6897|4|+0.6667|
|124|Low-VS - Uniform-PerSample|18|13|5|+0.2525|5|0|5|+0.8621|5|+0.8333|
|125|Uniform-PerSample - Uniform-BatchShared|8|10|-2|-0.1010|3|3|0|+0.0000|0|+0.0000|
|125|Low-VS - Uniform-BatchShared|16|12|4|+0.2020|7|1|6|+1.0345|6|+1.0000|
|125|Low-VS - Uniform-PerSample|18|12|6|+0.3030|7|1|6|+1.0345|6|+1.0000|

Overall一张图像=0.050505pp；Tail一张=0.172414pp；Few一张=0.166667pp；单类别一张=5pp。少数几张图像的变化不夸大为可靠类别规律。完整600行类别对比在 `per_class_paired_differences.csv`，含class0空指标。

### 两个seed的均值和方向（描述统计）

|对比|指标|平均Δpp|样本SD ddof1|正向seed数|负向seed数|跨seed变号|
|---|---|---|---|---|---|---|
|Uniform-PerSample - Uniform-BatchShared|Overall|-0.1768|0.1071|0|2|否|
|Uniform-PerSample - Uniform-BatchShared|Tail|-0.0862|0.1219|0|1|否|
|Uniform-PerSample - Uniform-BatchShared|Few|-0.0833|0.1179|0|1|否|
|Low-VS - Uniform-BatchShared|Overall|+0.1010|0.1428|1|0|否|
|Low-VS - Uniform-BatchShared|Tail|+0.8621|0.2438|2|0|否|
|Low-VS - Uniform-BatchShared|Few|+0.8333|0.2357|2|0|否|
|Low-VS - Uniform-PerSample|Overall|+0.2778|0.0357|2|0|否|
|Low-VS - Uniform-PerSample|Tail|+0.9483|0.1219|2|0|否|
|Low-VS - Uniform-PerSample|Few|+0.9167|0.1179|2|0|否|

n=2只作描述，不宣称显著性；没有以3960预测作独立样本检验。

### Tail类别完整变化

124/Uniform-PerSample - Uniform-BatchShared：提升ID `[75]`；下降ID `[81, 93]`；不变ID `[71, 72, 73, 74, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99]`。

124/Low-VS - Uniform-BatchShared：提升ID `[75, 81, 94, 99]`；下降ID `[93]`；不变ID `[71, 72, 73, 74, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98]`。

124/Low-VS - Uniform-PerSample：提升ID `[81, 94, 99]`；下降ID `[]`；不变ID `[71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98]`。

125/Uniform-PerSample - Uniform-BatchShared：提升ID `[85, 90, 91]`；下降ID `[81, 89, 95]`；不变ID `[71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 86, 87, 88, 92, 93, 94, 96, 97, 98, 99]`。

125/Low-VS - Uniform-BatchShared：提升ID `[84, 85, 90, 91, 94, 99]`；下降ID `[96]`；不变ID `[71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 86, 87, 88, 89, 92, 93, 95, 97, 98]`。

125/Low-VS - Uniform-PerSample：提升ID `[81, 84, 89, 94, 95, 99]`；下降ID `[96]`；不变ID `[71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 85, 86, 87, 88, 90, 91, 92, 93, 97, 98]`。

## 6. best敏感性、Head损失及预注册判据

|Seed|对比|epoch10 ΔOverall pp|best ΔOverall pp|口径反转|
|---|---|---|---|---|
|124|Uniform-PerSample - Uniform-BatchShared|-0.2525|-0.3535|未发生正负反转|
|124|Low-VS - Uniform-BatchShared|+0.0000|-0.0505|未发生正负反转|
|124|Low-VS - Uniform-PerSample|+0.2525|+0.3030|未发生正负反转|
|125|Uniform-PerSample - Uniform-BatchShared|-0.1010|+0.0505|**反转**|
|125|Low-VS - Uniform-BatchShared|+0.2020|+0.0000|未发生正负反转|
|125|Low-VS - Uniform-PerSample|+0.3030|-0.0505|**反转**|

- **Overall**：PerSample−Shared净正确张数（124、125）为 `[-5, -2]`；Low-VS−PerSample为 `[5, 6]`。两个seed均为Shared高于PerSample；观察方向符合预注册判据(3)，不支持“逐样本随机更好”在该指标上的预测。
- **Tail**：PerSample−Shared净正确张数（124、125）为 `[-1, 0]`；Low-VS−PerSample为 `[5, 6]`。Shared与PerSample没有在两个seed形成严格同方向优势；不能挑选单seed认定粒度效应。
- **Few**：PerSample−Shared净正确张数（124、125）为 `[-1, 0]`；Low-VS−PerSample为 `[5, 6]`。Shared与PerSample没有在两个seed形成严格同方向优势；不能挑选单seed认定粒度效应。

所有Head/Many损失均保留在 `paired_differences.csv`。即使某组Tail/Few方向一致，也不能据此宣称所有类别全面获益；best口径的反转和固定epoch10并列报告。没有事后调整Top48、epsilon、rho或预算。

## 7. 旧控制13字段与模型状态审计

下表为850批中完全相同的批次数，分母均850；不是“已全部通过”。

|字段|seed124一致批数|seed125一致批数|
|---|---|---|
|epoch|850|850|
|batch|850|850|
|size|850|850|
|indices_sha256|850|850|
|images_sha256|850|850|
|features_sha256|0|0|
|augmentation_state_after_sha256|850|850|
|ambient_cpu_rng_sha256|850|850|
|ambient_cuda_rng_sha256|850|850|
|gumbel_sha256|850|850|
|online_gap_sha256|0|0|
|online_scores_sha256|0|0|
|all_two_masks_sha256|438|438|

所有新Shared实际掩码在850/850批与“封存Uniform首样本掩码向全batch扩展”逐位一致；16个通道无重复，所选完整8×8复制供体，未选位置保持接收，标签不变。`all_two_masks_sha256`同时包含当前Low-VS候选，所以它的不一致不等于Shared未按规定选掩码；Shared未使用该Low-VS候选。

BN/Backbone在每epoch及best/final检查全部186状态条目逐位不变，包含93个参数张量、31组running_mean/var/counter；counter均15640。分类器weight/bias均有非零变化，所有1700步损失与分类器梯度有限。原实验 277 个受保护文件内容未变。详情为 `parameter_BN_verification.json`、`pairing_hash.json` 和每组 `control_mismatches.csv`。

## 8. 掩码重合、选择次数与跨样本相关性

|Seed|策略|与PerSample平均重合通道|与PerSample全同率|同batch样本对平均共同通道|同batch样本对全同率|不同组合数|单通道选择次数范围|
|---|---|---|---|---|---|---|---|
|124|Uniform-BatchShared|4.0839|0.7836%|16.0000|100.0000%|850|23870..29758|
|124|Uniform-PerSample|16.0000|100.0000%|3.9996|0.0000%|108470|26886..27482|
|124|Low-VS|12.1117|3.2756%|4.0013|0.0000%|108470|25405..28179|
|125|Uniform-BatchShared|4.0955|0.7836%|16.0000|100.0000%|850|23486..30782|
|125|Uniform-PerSample|16.0000|100.0000%|4.0000|0.0000%|108470|26766..27371|
|125|Low-VS|12.1192|3.2673%|4.0017|0.0000%|108470|25558..28127|

使用每个实际95/128样本batch，样本对为不同样本的有序对并按对数加权；完整64通道频率见 `channel_selection_counts.csv`。Shared的同batch掩码完全相同，产生结构性的跨样本相关性；每个样本仍是Uniform16/64。独立样本策略通常约4个共同通道。不能仅用边际通道频率声称随机化粒度等价。

### 迁移组成与实际幅度记录

|Seed|策略|均值V|均值B|均值V/(V+B+eps)|均值所选S|GAP L2|特征图L2|
|---|---|---|---|---|---|---|---|
|124|Uniform-BatchShared|3.7164|10.1326|0.3045|0.5316|3.4806|61.3258|
|124|Uniform-PerSample|3.7180|10.1229|0.3050|0.5316|未存档|未存档|
|124|Low-VS|3.8231|13.2857|0.2494|0.4072|未存档|未存档|
|125|Uniform-BatchShared|3.7076|10.0586|0.3056|0.5317|3.4653|61.1106|
|125|Uniform-PerSample|3.7155|10.1383|0.3044|0.5312|未存档|未存档|
|125|Low-VS|3.8213|13.2948|0.2492|0.4069|未存档|未存档|

V、B来自固定类统计与该次训练使用的增强接收特征，逐样本比例取均值。旧控制没有存档实际GAP/特征图L2，不能拿新Shared的L2与空缺值比较。本轮未做范数匹配；现有证据不能认定“选择策略只改变范数”，也不能把即时幅度差当作最终分类收益原因。

## 9. 环境、耗时、异常及交付

|Seed|训练累计秒|总运行秒|峰值分配MiB|峰值保留MiB|
|---|---|---|---|---|
|124|93.1606|95.2705|128.0088|160.0000|
|125|92.6643|94.8491|128.0088|160.0000|

耗时包含增强、特征提取、掩码核验、配对记录和分类器更新；总运行另含保存/验证。显存为PyTorch分配器统计，不是整机GPU所有进程占用。

|失败attempt|原因 / 最后错误|完整错误栈路径|
|---|---|---|
|preflight_20261015T190439_590116|ModuleNotFoundError: No module named 'yacs'|results\execution\preflight_20261015T190439_590116\failure.json|
|preflight_20261015T190537_063217|RuntimeError: Strict historical 13-field pairing preflight failed; no training permitted. See D:\xiangmuxunlian\xin\H2T_repro_audit\mask_granularity_iteration\results\execution\preflight_20261015T190537_063217\pairing_preflight.csv|results\execution\preflight_20261015T190537_063217\failure.json|

两次正式训练均正常完成，没有清理或覆写失败训练目录。preflight失败记录保留，后续继续依据用户明确授权；授权后没有跳过冻结、合法掩码、有限损失或防覆盖检查。完整stdout见 `results/preflight_001_console.log`、`preflight_002_console.log`、`dependency_setup_console.log`、`train_seed124_console.log`、`train_seed125_console.log`、`report_console.log`；各执行状态/防护记录位于 `results/execution/`。

### best/final Checkpoint

|Seed|文件|Epoch|路径|SHA256|
|---|---|---|---|---|
|124|best|4|[model_best.pth.tar](D:/xiangmuxunlian/xin/H2T_repro_audit/mask_granularity_iteration/results/shared_seed124/checkpoints/model_best.pth.tar)|82abfc22a6f002fcf883a1fda8a5ac722db1b3065830e9c3daf7f90c0753626f|
|124|final|10|[model_final.pth.tar](D:/xiangmuxunlian/xin/H2T_repro_audit/mask_granularity_iteration/results/shared_seed124/checkpoints/model_final.pth.tar)|9bdcc7a68ca99331810b302c01176c5bcce6b40c460300ed011fd757b4b033b2|
|125|best|5|[model_best.pth.tar](D:/xiangmuxunlian/xin/H2T_repro_audit/mask_granularity_iteration/results/shared_seed125/checkpoints/model_best.pth.tar)|4511f1684aacf09c22c434b42ef50190cea0314e9c4f08e9a100e2b63c85e6ed|
|125|final|10|[model_final.pth.tar](D:/xiangmuxunlian/xin/H2T_repro_audit/mask_granularity_iteration/results/shared_seed125/checkpoints/model_final.pth.tar)|53fa8f01325b2c0f4b77b4c99fd28bc2c506d70bf5ec1f2da44445291a1b4235|

### seed123 独立历史pilot

|Seed|策略|Overall|Tail|Few|
|---|---|---|---|---|
|123|Uniform-PerSample|47.0707|26.3793|27.0000|
|123|Low-VS|47.3737|27.4138|28.0000|

不与124/125合成n=3同协议统计，也没有重跑seed123。

## 10. 下一轮唯一建议与停止条件

**唯一优先建议：在当前指定环境中，另行授权一次Shared/PerSample/Low-VS三臂同环境、同特征数值路径的配对复核，以确认粒度方向。** 本轮A比较混入了已披露的历史特征/评分数值差异，优先完成这一步，再决定是否值得进入norm-matched消融。这里没有建议新增网络、更多种子或超参数搜索；本轮也不执行该复核。

停止条件：本轮两次Shared训练、保存结果与上述报告完成后停止；不再训练其他seed、不重新训练Stage I或旧baseline、不评估官方TEST。后续是否开展同环境复核或幅度匹配由用户确认。请将本报告连同CSV、13字段差异和执行日志上传/提供给ChatGPT复核。

## 11. 原始文件与实际命令

新目录：`D:\xiangmuxunlian\xin\H2T_repro_audit\mask_granularity_iteration`。历史只读目录：`D:\xiangmuxunlian\xin\H2T_repro_audit\clean_vs_stability_runs\clean_s1_seed42_s2_124_125_20261014`。

关键输出均在 `results/`：`all_three_epoch10.csv`、`paired_differences.csv`、`paired_seed_summary.csv`、`per_class_paired_differences.csv`、`mask_diagnostics.csv`、`channel_selection_counts.csv`、`selected_variation_shift.csv`、`timing_and_memory.csv`、`checkpoint_manifest.csv`、`scientific_summary.json`、`STOP_COMPLETED.json`。`shared_seed124/`和`shared_seed125/`各含全部实际掩码、850行batch记录与差异、10个验证NPZ、best/final权重、完整配置/摘要/日志。

```powershell
# cwd为项目根目录；以下为本次实际执行入口，已存在输出时训练器拒绝覆盖。
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/mask_granularity_iteration/run_mask_granularity.py preflight
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/mask_granularity_iteration/run_mask_granularity.py train --seed 124
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/mask_granularity_iteration/run_mask_granularity.py train --seed 125
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/mask_granularity_iteration/run_mask_granularity.py report
```

本轮没有执行任何版本控制或仓库同步操作。

## 补充：最终封存异常记录

封存检查首次误读停止状态的字段名new_updates，实际字段为updates_total，产生KeyError。已改为核对原有正确字段；该错误发生在写入快照/封存文件之前，没有影响训练、Checkpoint或评估，没有重复训练。原始异常及修复记录见 `results/finalization_failure_001.json`。
