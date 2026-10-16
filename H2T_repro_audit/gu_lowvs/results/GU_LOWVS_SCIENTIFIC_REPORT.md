# GU-LowVS V1 科学实验报告

日期：2026-10-16（Asia/Shanghai）

## 1. 命题、范围与预注册结论

检验：固定 Low-VS Bottom48→随机16个完整8×8通道后，按干净 TRAIN 类别均衡风险梯度的对齐度，逐样本选择 α∈{0,0.5,1}，能否在本轮开发验证协议下优于固定完全替换、固定半强度和动作打乱对照。

**任务文字判据：NO-GO；提供的执行器数值判据：NO-GO。** 判据来自训练前的 `preregistered_decision_rules.json`；无事后调整。

数值执行器额外要求对 Full 平均 Overall 严格为正，并将最终 epoch 任意单一动作≥95%视为塌缩，比任务文字略严格；此处并列报告两者。Head 的“明显下降”按提供代码预先使用的 −0.15 pp 解释，同时展示各 seed 的实际值。

本轮固定epoch10：GU相对Full平均Overall −0.7323 pp、Head +3.7143 pp、Tail −4.0517 pp；相对Shuffled平均Overall −0.4798 pp、Tail −3.0172 pp。两个seed均出现总体下降与头尾取舍；当前不支持按样本训练效用路由带来收益。

## 2. 锁定模型、数据与算法

Stage I：Clean seed42 epoch184，SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`；参考库 SHA256 `74fa31ebbbf54b1fbbd1559f5c3161b1ddf9406f884d854bb3664bcec3c909d7`。同一 Stage I 的 Backbone、BN 和初始100类分类头用于8组独立初始化，没有加载其他 Stage II 权重。

真实 CIFAR-100 官方 TRAIN：原 LT 子集10847张、100类、IF100。逐个 raw index 核对缓存像素与标签和官方 TRAIN 完全一致。开发验证是官方 TRAIN 留出1980张，class1..99各20张，缺class0；与LT索引重叠0；始终输出100维logits。参考库仅为LT训练图像的官方 eval transform 特征，含class0的500张训练图像，没有加入开发验证图像。详见 `dataset_protocol.json` 和 `reference_bank_provenance.json`。

有效类别数：{'Head': 35, 'OfficialMedium': 35, 'Tail': 29, 'Many': 34, 'Medium': 35, 'Few': 30}。Head=0..35（有效1..35），OfficialMedium=36..70，Tail=71..99；Many按LT训练n>100，Medium按20≤n≤100，Few按n<20。OfficialMedium与频次Medium不是同一集合。验证各类等量，因此Overall=99类Macro。

原始BS/IS、增强、Gumbel和VS/小样本回退均复用冻结历史实现。ρ=.25对应预选16/64完整通道；α控制这些选中通道替换的幅度，α=0时实际不改变特征。每组10 epochs、850更新、batch128（每epoch最后一batch95），SGD lr=.1/momentum=.9/wd=.0005及原cos调度，硬接收者标签与CE。仅训练6500参数分类器，Backbone463504参数与全部BN固定。

每个epoch开头，以当前分类器在全部训练参考特征上计算100类别等权CE解析梯度G_ref；每batch各α的解析梯度gα取当前分类器值，路由no_grad，argmax平分时由固定顺序优先α0。score=<G_ref,gα>，仅选定特征的原CE参与反向传播。α0/1端点逐位一致。GU-Shuffled在自己当前分类器计算动作后，按独立确定性置换打乱样本分配；每batch打乱前后动作频数严格相同。

## 3. 正确性与执行证据

CPU解析梯度与autograd对照、GPU预检、TRAIN参考库检查全部PASS。8组共6800次optimizer.step；相对各seed新训Full基准，索引、增强图像、冻结特征、VS分数、Gumbel五字段各850/850一致；额外检查Low-VS实际掩码850/850一致。186个Backbone状态张量逐位不变，31个BN计数器均保持15640；分类器weight/bias均有实际更新；16个best/final checkpoint均重载验证并与原记录一致。

411个历史文件逐字节hash核对不变；运行时历史写保护及官方TEST guard均无事件。未重新训练Stage I、未运行官方TEST、未执行任何版本控制命令。输出只在本轮results目录。

`utility_policy.py`与提供版本逐字节不变。新增代码只补充结果路径约束、原始TRAIN核对、完整配对/掩码/动作频数断言、状态/耗时/配置/错误持久化，Diff见 `minimal_interface_and_evidence.patch`。没有改变参考风险、α候选、效用评分、VS或优化器。

## 4. 固定第10epoch主结果（%）

|seed|arm|Overall|Head|OfficialMedium|Tail|Many|Medium|Few|Macro|best epoch|best Overall|
|---|---|---|---|---|---|---|---|---|---|---|---|
|124|LowVS-Full|47.3232|59.8571|51.0000|27.7586|60.5882|50.7143|28.3333|47.3232|9|47.4242|
|124|LowVS-Half|47.0707|61.5714|49.8571|26.2069|62.5000|49.5714|26.6667|47.0707|9|47.1717|
|124|GU-LowVS|46.5152|63.5714|48.2857|23.7931|64.5588|48.1429|24.1667|46.5152|7|46.8687|
|124|GU-Shuffled|47.1717|61.1429|49.8571|27.0690|62.0588|49.5714|27.5000|47.1717|7|47.3232|
|125|LowVS-Full|47.1717|59.5714|50.7143|27.9310|60.4412|50.2857|28.5000|47.1717|6|47.2222|
|125|LowVS-Half|47.1717|61.7143|50.0000|26.2069|62.6471|49.7143|26.6667|47.1717|10|47.1717|
|125|GU-LowVS|46.5152|63.2857|48.5714|23.7931|64.2647|48.4286|24.1667|46.5152|7|46.7172|
|125|GU-Shuffled|46.8182|60.8571|49.5714|26.5517|61.7647|49.2857|27.0000|46.8182|8|47.0707|

## 5. GU与三个控制的逐图配对得失

差值=GU−控制；wrong→right与right→wrong按完全相同1980张验证图像配对。Overall/Tail/Head净数与准确率差已逐项严格核验（Tail580张、Head700张、Few600张）。

|seed|比较|wrong→right|right→wrong|净图数|ΔOverall pp|Tail净数|ΔTail pp|Head净数|ΔHead pp|ΔFew pp|
|---|---|---|---|---|---|---|---|---|---|---|
|124|GU-LowVS - LowVS-Full|46|62|-16|-0.8081|-23|-3.9655|26|+3.7143|-4.1667|
|124|GU-LowVS - LowVS-Half|19|30|-11|-0.5556|-14|-2.4138|14|+2.0000|-2.5000|
|124|GU-LowVS - GU-Shuffled|22|35|-13|-0.6566|-19|-3.2759|17|+2.4286|-3.3333|
|125|GU-LowVS - LowVS-Full|47|60|-13|-0.6566|-24|-4.1379|26|+3.7143|-4.3333|
|125|GU-LowVS - LowVS-Half|17|30|-13|-0.6566|-14|-2.4138|11|+1.5714|-2.5000|
|125|GU-LowVS - GU-Shuffled|25|31|-6|-0.3030|-16|-2.7586|17|+2.4286|-2.8333|

两seed均值（每seed等权，不将3960个重复验证观测当独立样本）：

|控制|ΔOverall pp|ΔHead pp|ΔTail pp|ΔFew pp|两个seed Overall方向|
|---|---|---|---|---|---|
|LowVS-Full|-0.7323|+3.7143|-4.0517|-4.2500|-0.8081/-0.6566|
|LowVS-Half|-0.6061|+1.7857|-2.4138|-2.5000|-0.5556/-0.6566|
|GU-Shuffled|-0.4798|+2.4286|-3.0172|-3.0833|-0.6566/-0.3030|

## 6. 动作比例与退化诊断

下表逐epoch报告选择α0/0.5/1的比例。Full恒为0/0/100%，Half恒为0/100/0%；原始整数频数见 `alpha_schedule.csv`、各组`epochs.csv`和`actions.npz`。

|seed|epoch|GU α0%|GU α.5%|GU α1%|Shuffled α0%|Shuffled α.5%|Shuffled α1%|Shuffled分配实际改变%|
|---|---|---|---|---|---|---|---|---|
|124|1|38.011|5.319|56.670|35.272|5.458|59.270|51.341|
|124|2|55.674|10.777|33.548|55.739|10.408|33.853|55.767|
|124|3|43.957|8.242|47.801|53.987|10.657|35.355|57.113|
|124|4|53.268|10.593|36.139|55.988|10.648|33.364|55.435|
|124|5|44.390|8.731|46.879|55.850|9.782|34.369|55.905|
|124|6|51.406|9.533|39.061|55.435|9.938|34.627|56.808|
|124|7|47.045|9.339|43.616|54.319|11.386|34.295|57.168|
|124|8|50.982|10.915|38.103|55.223|9.809|34.968|56.062|
|124|9|50.438|12.815|36.747|55.997|10.390|33.613|56.145|
|124|10|54.236|13.985|31.778|56.615|9.883|33.502|54.421|
|125|1|37.384|5.734|56.882|35.014|5.790|59.196|51.378|
|125|2|55.840|10.657|33.502|56.799|10.372|32.829|56.126|
|125|3|40.924|8.786|50.290|57.352|10.261|32.387|55.001|
|125|4|51.913|10.464|37.623|54.559|10.316|35.125|55.896|
|125|5|43.385|9.164|47.451|54.513|10.344|35.143|56.126|
|125|6|51.406|10.012|38.582|53.950|11.505|34.544|57.380|
|125|7|44.667|9.966|45.367|55.296|9.201|35.503|55.720|
|125|8|51.157|11.358|37.485|55.094|10.141|34.765|56.264|
|125|9|52.992|12.446|34.563|55.269|10.362|34.369|56.559|
|125|10|54.310|13.276|32.414|55.352|10.123|34.526|55.472|

按端点动作≥95%的预设诊断，GU任意epoch端点塌缩记录数量=0。最终epoch文字判据：True。统计显示是否退化见实际比例，不能把α0当作仍执行完全替换。

GU-Shuffled每batch在自身计算的动作和打乱后的动作之间850/850频数一致；但GU与Shuffled各自训练使分类器、参考梯度和动作直方图随epoch分离，跨组不构成完全强度匹配。逐epoch跨组直方图见 `cross_arm_action_histograms.csv`；这是该对照本身的限制。日志`mean_predicted_gain`是置换前argmax相对α0的预测分数优势，在Shuffled组不能当成置换后实际分配的效用。

## 7. 预注册筛选与解释

|文字判据数值化|通过|
|---|---|
|each_seed_Overall_vs_Full_nonnegative|False|
|mean_Tail_vs_Full_nonnegative|False|
|mean_Overall_vs_Shuffled_positive|False|
|mean_Head_vs_Shuffled_at_least_minus_0_15pp|True|
|no_endpoint_collapse_final_epoch|True|

GU未优于Shuffled的平均Overall：本轮缺少支持“按样本效用匹配”带来收益的证据，即使相对某个固定强度控制有变化，也不足以证明该效用路由有效。

GU相对Half平均Overall -0.6061 pp。Half控制显示固定柔化的效果；GU与Half不同并不能排除强度分布改变的解释，且本轮Shuffled也不是跨组严格直方图匹配。

## 8. best敏感性、耗时与显存

best只展示开发验证选出的辅助结果，不替代epoch10，不筛seed或epoch挽救规则。

- seed124 GU−LowVS-Full：epoch10 -0.8081 pp，双方各自best -0.5556 pp；方向反转=False。
- seed125 GU−LowVS-Full：epoch10 -0.6566 pp，双方各自best -0.5051 pp；方向反转=False。
- seed124 GU−LowVS-Half：epoch10 -0.5556 pp，双方各自best -0.3030 pp；方向反转=False。
- seed125 GU−LowVS-Half：epoch10 -0.6566 pp，双方各自best -0.4545 pp；方向反转=False。
- seed124 GU−GU-Shuffled：epoch10 -0.6566 pp，双方各自best -0.4545 pp；方向反转=False。
- seed125 GU−GU-Shuffled：epoch10 -0.3030 pp，双方各自best -0.3535 pp；方向反转=False。

|seed|arm|总进程工作秒|训练循环秒|相对Full额外秒|峰值allocated MiB|峰值reserved MiB|
|---|---|---|---|---|---|---|
|124|LowVS-Full|53.08|50.81|+0.00|120.7|150.0|
|124|LowVS-Half|52.92|50.76|-0.16|120.7|150.0|
|124|GU-LowVS|54.00|51.75|+0.92|120.7|150.0|
|124|GU-Shuffled|54.68|52.36|+1.60|120.7|150.0|
|125|LowVS-Full|53.39|51.22|+0.00|120.7|150.0|
|125|LowVS-Half|53.37|51.15|-0.01|120.7|150.0|
|125|GU-LowVS|54.56|52.27|+1.17|120.7|150.0|
|125|GU-Shuffled|53.99|51.76|+0.60|120.7|150.0|

总进程工作秒从train入口起计，包含加载、校验、验证、保存和重载；训练循环秒含增强、冻结双分支前向、效用/签名检查和分类器更新，不含epoch验证。Full/Half也计算每epoch参考梯度以使用同一运行引擎，因此差值不是相对于完全删掉参考梯度计算的生产基线；少量耗时差可能是系统噪声。


以下历史No-Fusion仅作背景，不是本轮严格配对控制，不计算跨轮逐图因果差值：

|历史seed|Overall|Head|Tail|Many|Medium|Few|
|---|---|---|---|---|---|---|
|124|46.3131|62.7143|24.4828|63.6765|48.0000|24.6667|
|125|46.5152|63.1429|24.1379|64.1176|48.4286|24.3333|

8组实际epoch1..10学习率完全相同：[0.020000000000000004, 0.019510565162979287, 0.018090169943855035, 0.015877852523142666, 0.013090169944091066, 0.010000000000448967, 0.006909830056762919, 0.004122147477583781, 0.0019098300566727567, 0.0004894348372981928]。有效配置保留历史YAML的model_dir/name描述字段，但resume已清空；所有实际保存路径由本轮执行器显式指定在results/seed/arm，历史写保护同时生效。

学习率细节：原Stage II函数对第0个参数组乘 `config.lr_scheduler.lr_factor=0.2`；当前优化器只有这个分类器参数组，因此配置 `lr=0.1` 对应实际初始步长0.02，再按10-epoch cosine衰减。八组完全沿用此原函数与原YAML，没有把实际学习率主动提高到0.1。

显存列为本进程PyTorch分配器allocated/reserved峰值，不含Windows桌面、驱动及CUDA context等整机占用。

## 9. 历史背景与证据限制

历史No-Fusion、Uniform和RAVS不是本轮新增的严格配对对照；不把它们混入本轮门槛。可只读追溯`../ravs_v1/results/all_arms_epoch10.csv`和历史科学报告。

1. G_ref是有限LT训练参考库的类别等权经验风险梯度，参考样本也参与分类器训练；不是独立验证梯度或未见总体梯度。
2. 一阶泰勒式需同一点梯度及小步长；本实现G_ref仅epoch开头刷新、gα逐batch更新，epoch内参考方向逐渐陈旧。真实SGD还有momentum和weight decay，均不在效用评分里；逐样本贪心也不等价于最优batch或长期训练。
3. 评分可全负，此时argmax只是选择相对较高的代理值，不保证降低真实风险。即时分数不等于未来测试效果。
4. 开发验证1980张被多轮研究反复使用，且Clean Stage I本身用此TRAIN留出选择epoch184，存在开发集方法选择偏差；并非独立最终测试。没有使用官方TEST规避不了该局限。
5. class0没有验证样本，100维输出中预测成class0仍算错，但无法衡量class0召回；99类Overall不可直接对照H2T论文官方100类TEST准确率。
6. 两个Stage II种子共享同一Stage I及同一批验证图像；不宣称n=2统计显著，不把3960条预测当独立样本。Overall一张图对应0.050505 pp，Tail一张对应0.172414 pp，Few一张对应0.166667 pp。
7. 只训练分类头，不能声称学习到更好Backbone表征。自适应α与已有比例自适应思想可能有概念重叠，本轮不作方法新颖性结论。

## 10. 失败、异常、环境和复现

报告脚本准备时有一次纯文本替换锚点不匹配的AssertionError；当次没有写入源文件或修改训练，随后仅修正报告生成脚本，训练无失败、无重跑。记录见analysis_preparation_notes.json。
训练失败=0；配对失败=0；TEST访问尝试=0；历史写入尝试=0；历史文件hash变化=0；Backbone/BN变化=0。没有重试、删除失败目录、自动续训、追加种子或更改α。首次静态检查尚未运行即补充的接口/证据修改有完整提供源码快照和Diff。

环境：

```json
{
  "python": "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe",
  "torch": "2.14.1+cu132",
  "torchvision": "0.29.1+cu132",
  "numpy": "2.5.2",
  "cuda": "13.2",
  "cudnn": 92400,
  "GPU": "NVIDIA GeForce RTX 5070 Laptop GPU",
  "total_mib": 8150.5625,
  "cudnn_benchmark": false,
  "cudnn_deterministic": true,
  "cudnn_allow_tf32": false,
  "matmul_allow_tf32": false
}
```

在本地项目根目录使用以下已实际执行的命令；输出目录存在时不得直接重复训练：

```powershell
$py = 'D:\app\minicondaaa\envs\xiaoju\python.exe'
& $py -B H2T_repro_audit/gu_lowvs/test_utility_policy.py
& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py preflight
& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py prepare
& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py train --seed 124 --arm LowVS-Full
& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py train --seed 124 --arm LowVS-Half
& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py train --seed 124 --arm GU-LowVS
& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py train --seed 124 --arm GU-Shuffled
& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py train --seed 125 --arm LowVS-Full
& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py train --seed 125 --arm LowVS-Half
& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py train --seed 125 --arm GU-LowVS
& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py train --seed 125 --arm GU-Shuffled
& $py -B H2T_repro_audit/gu_lowvs/run_gu_lowvs.py report
& $py -B H2T_repro_audit/gu_lowvs/analyze_gu_results.py
```

## 11. 原始文件与停止

本报告所在results目录：`epoch10.csv`/`epoch10_verified.csv`、`paired.csv`/`paired_verified.csv`、`per_class.csv`/`per_class_verified.csv`、`epochs_all.csv`、`alpha_schedule.csv`、`cross_arm_action_histograms.csv`、`scientific_summary.json`、`final_integrity.json`、`checkpoint_manifest.csv`；参考库`balanced_clean_train_bank.npz`及来源JSON；`console/`完整stdout/错误输出；`provided_source/`初始源码和补丁。各seed/arm目录有effective.yaml、config.json、epochs.csv、batch_signatures.csv、actions.npz、validation_epoch10.npz、train.log、summary.json和checkpoints/best.pth/final.pth。

未达到预注册探索门槛，停止GU-LowVS V1，不搜索α、参考风险、epoch或更多种子。本轮至此停止，后续由用户将产物提供给ChatGPT复核；没有自动启动其他训练。
