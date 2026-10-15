# RAVS-H2T V1 第一轮正式方法原型：科学报告

完成时间：2026-10-15T22:11:18+08:00

**结果状态：complete；本轮十组均重新训练，共100 epochs、8500次分类器更新。**

## 1. 准确算法与研究问题

固定原正则化VS分数：S=v_reg/(v_reg+(mu_donor-f_receiver_GAP)^2+epsilon)。沿用锁定类统计、shrinkage和epsilon，不重新估计。RAVS只调整随机候选池大小，不改变供体、融合比例或替换数量。

每步更新前，对当前分类器的未融合接收特征计算clean logits；margin=l_y-max(other logits)，h=sigmoid(-margin/(std(logits)+1e-8))。logits detach/no_grad，转float64，std使用总体标准差ddof=0。原实现保留标准化margin的[-20,20]数值保护，未调参。

q=clip(log(500/n_y)/log(100),0,1)；RAVS的omega=q*(0.5+0.5*h)，Frequency-Only固定h=0.5，因此omega=0.75*q。K=clip(floor(64-16*omega+0.5),48,64)，从原VS BottomK用同一Gumbel64均匀不放回选16个完整8×8通道。供体类n<20或评分平坦回退Uniform；非有限评分终止。

目的：相比固定Low-VS是否减少Head牺牲、提高Tail；相比Frequency-Only，困难度是否增加训练收益。没有额外网络、辅助损失、供体筛选或自适应rho。

## 2. 五组控制与固定协议

No-Fusion直接使用接收特征，但仍执行配对供体增强/forward。Uniform-PerSample和Low-VS直接使用原online_policy掩码；Frequency-Only与RAVS共享门控形式，仅h不同。五组均在本次指定Python环境重新独立初始化训练，不引用任何旧时期准确率作对照。

Clean Stage I seed42 epoch184 SHA256：`7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`；统计SHA256：`e81a4897a965d642f832bd6429e0e5a7d136725daaa069d9ded96f87ab64aef9`。

ResNet-32；BS receiver+IS donor；batch128（每epoch85批，末批95）；10epochs；6500参数的100类线性分类器；CE硬标签；SGD base_lr0.1、momentum0.9、wd0.0005、nesterovFalse，原cosine factor0.2，首epoch实际lr0.02。四组融合均固定16/64通道。

全部backbone、Conv与BN affine冻结，BN eval，running_mean/var/counter逐位不变。运行器预声明所有组cuDNN deterministic=True、benchmark=False，cuDNN及matmul TF32=False；配置已如实记录。effective.yaml的原base路径不决定实际保存位置，实际为results/seed<SEED>/<ARM>。

## 3. 数据协议与解释边界

真实CIFAR-100-LT IF100，10847张TRAIN；锁定原raw索引，预检训练/验证ID交集0。既有TRAIN留出1980张：类1..99各20张，class0缺失，保留100 logits，预测0计错。没有官方TEST图像、标签或TEST选模。

Head=ID0..35（有效35类/700张）；OfficialMedium=36..70（35/700）；Tail=71..99（29/580）。Many按LT数>100（有效34/680）；频次Medium为20..100（35/700）；Few为<20（30/600）。两个Medium定义不同；均衡验证使Overall等于99类Macro。

验证集已多轮使用并参与Clean Stage I选模，不是独立最终测试。两个seed使用相同1980张图像，不能把3960预测当独立观测。这里只调整线性分类器，不能声称Backbone表征得到训练改善；不是完整PI-H2T/PIF复现。

## 4. 固定epoch10与best（%）

|Seed|策略|Overall|Head|OfficialMedium|Tail|Many|Medium-n|Few|Macro|best epoch|best Overall|
|---|---|---|---|---|---|---|---|---|---|---|---|
|124|No-Fusion|46.3131|62.7143|48.0000|24.4828|63.6765|48.0000|24.6667|46.3131|5|46.4646|
|124|Uniform-PerSample|47.0707|60.1429|50.7143|26.8966|61.0294|50.2857|27.5000|47.0707|4|47.1212|
|124|Low-VS|47.3232|59.8571|51.0000|27.7586|60.5882|50.7143|28.3333|47.3232|9|47.4242|
|124|Frequency-Only|47.3232|59.8571|51.0000|27.7586|60.7353|50.5714|28.3333|47.3232|4|47.4747|
|124|RAVS-V1|47.2222|60.0000|50.7143|27.5862|60.8824|50.2857|28.1667|47.2222|7|47.4747|
|125|No-Fusion|46.5152|63.1429|48.4286|24.1379|64.1176|48.4286|24.3333|46.5152|4|46.6667|
|125|Uniform-PerSample|46.8687|60.1429|50.1429|26.8966|61.0294|49.7143|27.5000|46.8687|6|47.2222|
|125|Low-VS|47.1717|59.5714|50.7143|27.9310|60.4412|50.2857|28.5000|47.1717|6|47.2222|
|125|Frequency-Only|46.8687|59.8571|50.0000|27.4138|60.7353|49.5714|28.0000|46.8687|6|47.2222|
|125|RAVS-V1|46.7677|59.8571|49.8571|27.2414|60.7353|49.4286|27.8333|46.7677|5|47.2727|

best仅为敏感性分析，按验证Overall严格提升保存，相等保留最早epoch；主要判断始终固定epoch10。

## 5. RAVS对所有控制的逐图配对

|Seed|对比|ΔOverall pp|ΔHead pp|ΔTail pp|ΔFew pp|增正确|失正确|净正确|Tail净|Head净|
|---|---|---|---|---|---|---|---|---|---|---|
|124|RAVS-V1 - No-Fusion|+0.9091|-2.7143|+3.1034|+3.5000|56|38|18|18|-19|
|124|RAVS-V1 - Uniform-PerSample|+0.1515|-0.1429|+0.6897|+0.6667|8|5|3|4|-1|
|124|RAVS-V1 - Low-VS|-0.1010|+0.1429|-0.1724|-0.1667|8|10|-2|-1|1|
|124|RAVS-V1 - Frequency-Only|-0.1010|+0.1429|-0.1724|-0.1667|1|3|-2|-1|1|
|125|RAVS-V1 - No-Fusion|+0.2525|-3.2857|+3.1034|+3.5000|53|48|5|18|-23|
|125|RAVS-V1 - Uniform-PerSample|-0.1010|-0.2857|+0.3448|+0.3333|7|9|-2|2|-2|
|125|RAVS-V1 - Low-VS|-0.4040|+0.2857|-0.6897|-0.6667|5|13|-8|-4|2|
|125|RAVS-V1 - Frequency-Only|-0.1010|+0.0000|-0.1724|-0.1667|1|3|-2|-1|0|

全部20个无重复两两比较和所有类别组净得失保存在paired_epoch10.csv；每个净正确/组样本数×100与准确率差严格一致。Overall一张=0.050505pp，Head/OfficialMedium一张=0.142857pp，Tail一张=0.172414pp，Few一张=0.166667pp，单类别一张=5pp。

净收益/损失仅1～3张的RAVS比较：124 RAVS-V1 - Uniform-PerSample 净3张; 124 RAVS-V1 - Low-VS 净-2张; 124 RAVS-V1 - Frequency-Only 净-2张; 125 RAVS-V1 - Uniform-PerSample 净-2张; 125 RAVS-V1 - Frequency-Only 净-2张。这些不作稳定泛化证据。

## 6. best口径、Head牺牲与尾类变化

|Seed|RAVS对比|epoch10 ΔOverall pp|best ΔOverall pp|排序|
|---|---|---|---|---|
|124|RAVS-V1 - No-Fusion|+0.9091|+1.0101|未正负反转|
|124|RAVS-V1 - Uniform-PerSample|+0.1515|+0.3535|未正负反转|
|124|RAVS-V1 - Low-VS|-0.1010|+0.0505|反转|
|124|RAVS-V1 - Frequency-Only|-0.1010|+0.0000|未正负反转|
|125|RAVS-V1 - No-Fusion|+0.2525|+0.6061|未正负反转|
|125|RAVS-V1 - Uniform-PerSample|-0.1010|+0.0505|反转|
|125|RAVS-V1 - Low-VS|-0.4040|+0.0505|反转|
|125|RAVS-V1 - Frequency-Only|-0.1010|+0.0505|反转|

RAVS−Low-VS：seed124 Head +0.1429pp（净1张），Tail -0.1724pp（净-1张）; seed125 Head +0.2857pp（净2张），Tail -0.6897pp（净-4张）。

RAVS−Frequency-Only：seed124 Head +0.1429pp（净1张），Tail -0.1724pp（净-1张）; seed125 Head +0.0000pp（净0张），Tail -0.1724pp（净-1张）。

seed124 RAVS−Low-VS Tail提升类ID []；下降类ID [99]。每类仅20张，见paired_per_class.csv完整保留所有类，缺class0项为空。

seed124 RAVS−Frequency-Only Tail提升类ID []；下降类ID [99]。每类仅20张，见paired_per_class.csv完整保留所有类，缺class0项为空。

seed125 RAVS−Low-VS Tail提升类ID []；下降类ID [81, 84, 89, 99]。每类仅20张，见paired_per_class.csv完整保留所有类，缺class0项为空。

seed125 RAVS−Frequency-Only Tail提升类ID []；下降类ID [84]。每类仅20张，见paired_per_class.csv完整保留所有类，缺class0项为空。

## 7. 预先承诺的Go/No-Go

门槛1：两个seed的RAVS−Low-VS Overall均非负，平均Overall至少+0.15pp，平均Tail、Head均非负。

门槛2：RAVS−Frequency-Only方向基本一致：Overall路线要求两个seed均非负且平均至少+0.10pp；Tail路线要求两个seed的Tail均非负、平均Overall不降低、平均Tail至少+0.35pp。该对“方向一致”的执行定义已在训练前存档；两路线取或。

|指标|相对Low-VS平均Δpp|相对Frequency平均Δpp|
|---|---|---|
|Overall|-0.2525|-0.1010|
|Head|+0.2143|+0.0714|
|Tail|-0.4310|-0.1724|
|Few|-0.4167|-0.1667|

门槛1 **FAIL**；逐条件：`{'both_seed_Overall_nonnegative': False, 'mean_Overall_at_least_0_15': False, 'mean_Tail_nonnegative': False, 'mean_Head_nonnegative': True}`。

门槛2 **FAIL**；Overall路线：`{'both_seed_Overall_nonnegative': False, 'mean_Overall_at_least_0_10': False}`；Tail路线：`{'both_seed_Tail_nonnegative': False, 'mean_Overall_nonnegative': False, 'mean_Tail_at_least_0_35': False}`。

**最终判定：NO-GO：本版没有同时达到两道探索门槛，应停止当前V1规则，不调公式或候选池、不追加seed。**

门槛不等于统计显著性。所有阈值用未四舍五入准确率判断，两个Stage II种子只代表固定一套Stage I上的重复性；同一开发验证集反复使用会带来研究选择偏差。

## 8. 同环境配对与训练正确性

|seed/组|indices|增强图像|冻结特征|VS评分|Gumbel|
|---|---|---|---|---|---|
|124/No-Fusion|850|850|850|850|850|
|124/Uniform-PerSample|850|850|850|850|850|
|124/Low-VS|850|850|850|850|850|
|124/Frequency-Only|850|850|850|850|850|
|124/RAVS-V1|850|850|850|850|850|
|125/No-Fusion|850|850|850|850|850|
|125/Uniform-PerSample|850|850|850|850|850|
|125/Low-VS|850|850|850|850|850|
|125/Frequency-Only|850|850|850|850|850|
|125/RAVS-V1|850|850|850|850|850|

分母均850。五组均执行相同donor增强与forward，clean logits从当前分类器更新前取值且无梯度；RAVS门控不依赖验证预测或正在反传的计算图。梯度只来自融合/原特征的原CE。

五项签名全部850/850相同；没有使用上一轮历史哈希豁免。本轮严格控制输入、冻结特征、评分与噪声，策略之外没有跨环境对照混用。

每epoch检查186个backbone状态条目逐位不变（93参数张量与93 BN缓冲）；31组BN counter固定15640。分类器weight/bias均变化；所有8500步损失和分类器梯度有限。best/final重新加载后各指标与对应epoch存档差小于1e-9。Checkpoint只保存分类头及锁定Stage I哈希，推理需配合该冻结Backbone。

历史只读校验通过，277个原实验文件内容未变；TEST访问及历史写入事件均为0。

## 9. 门控诊断与额外耗时

|Seed|组|平均h|平均omega|实际K均值|K范围|与Uniform掩码相同比例|
|---|---|---|---|---|---|---|
|124|No-Fusion|0.2496|NA|NA|NA|NA|
|124|Uniform-PerSample|0.2506|NA|64.0000|64..64|1.0000|
|124|Low-VS|0.2531|NA|48.4499|48..64|0.0327|
|124|Frequency-Only|0.5000|0.3793|58.0932|52..64|0.2947|
|124|RAVS-V1|0.2514|0.3185|59.0542|49..64|0.3431|
|125|No-Fusion|0.2494|NA|NA|NA|NA|
|125|Uniform-PerSample|0.2509|NA|64.0000|64..64|1.0000|
|125|Low-VS|0.2533|NA|48.4499|48..64|0.0327|
|125|Frequency-Only|0.5000|0.3793|58.0961|52..64|0.2954|
|125|RAVS-V1|0.2517|0.3186|59.0558|49..64|0.3430|

Frequency-Only的h诊断严格为0.5；其他对照的h仅是在线观察，不作门控。Low-VS的K=48、回退64；No-Fusion没有通道候选池。candidate_pool_histogram.csv保存全部K48..64实际计数。难度数值有变化不等于它包含有用训练信息，需由RAVS对Frequency对照检验。

|Seed|组|训练秒|含验证保存总秒|峰值分配MiB|峰值保留MiB|
|---|---|---|---|---|---|
|124|No-Fusion|46.3032|46.8352|113.9604|148.0000|
|124|Uniform-PerSample|47.6799|48.2059|115.9683|148.0000|
|124|Low-VS|51.3637|51.8943|115.9683|148.0000|
|124|Frequency-Only|51.6287|52.1394|115.9683|148.0000|
|124|RAVS-V1|52.1785|52.7230|115.9683|148.0000|
|125|No-Fusion|46.4820|46.9605|113.9604|148.0000|
|125|Uniform-PerSample|51.6766|52.2015|115.9683|148.0000|
|125|Low-VS|51.7769|52.2795|115.9683|148.0000|
|125|Frequency-Only|52.0619|52.6129|115.9683|148.0000|
|125|RAVS-V1|51.9785|52.4824|115.9683|148.0000|

所有组都计算clean logits与原VS评分以维持计算/随机流一致，时间包括CPU增强、记录和核验，因此不能把顺序运行的时间差当作生产实现的纯门控开销。PyTorch显存统计不代表全机GPU总占用。

## 10. 训练失误、最小修复与强弱点

CPU策略单测和零更新preflight均PASS；十组按指定顺序各执行一次，没有失败后静默重跑或覆盖。修复仅涉及本轮目录：Frequency-Only h诊断改为实际0.5、Low-VS池大小诊断改为实际48/回退64；补齐冻结/有限CE预检、当前运行配置、日志/签名/计时及历史写保护。选择公式、Gumbel、VS统计、CE与预算未改变，详细对照为minimal_fixes.patch。

强点：包含No-Fusion、Uniform、Low-VS及频率门控，当前环境全部重训；每seed严格850批配对；只训练一个6500参数分类头。弱点：仅两个Stage II种子与一个Stage I；反复使用的99类开发验证集；门控反馈来自同一正在训练的分类器，并不保证困难样本适合更强VS限制；没有独立泛化测试。

## 11. 唯一下一轮建议与停止

唯一建议：先复核本轮逐图得失并由用户重新确定科学假说；当前RAVS规则停止，不继续搜索K、公式或新seed。

本轮已按预算停止：不追加seed、不训练新Backbone、不调V1、不评估官方TEST、不自行同步仓库。由用户本人决定上传哪些产物给ChatGPT进行深度分析。

## 12. 文件与准确运行命令

根目录：`D:\xiangmuxunlian\xin\H2T_repro_audit\ravs_v1`。每组结果在results/seed<SEED>/<ARM>/：config.json、effective.yaml、train.log、batch_signatures.csv、epoch_metrics.csv、validation_epoch10.npz、checkpoints/best.pth、checkpoints/final.pth、summary.json。保留了全部最终权重。

汇总：results/all_arms_epoch10.csv、paired_epoch10.csv、scientific_summary.json、paired_seed_summary.csv、paired_per_class.csv、gate_diagnostics.csv、candidate_pool_histogram.csv、checkpoint_manifest.csv；源修改对照minimal_fixes.patch，训练前判据preregistered_decision_rules.json，preflight.json，run_registry.json，console/全部stdout。

上述为实际执行入口；现有输出目录会拒绝重复训练，不能直接重跑或清除已完成结果。需要恢复实验时应先取得用户明确授权。


解释器：`D:\app\minicondaaa\envs\xiaoju\python.exe`；项目根目录的准确顺序命令：

```powershell
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/test_policy.py
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py preflight
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed 124 --arm No-Fusion
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed 124 --arm Uniform-PerSample
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed 124 --arm Low-VS
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed 124 --arm Frequency-Only
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed 124 --arm RAVS-V1
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed 125 --arm No-Fusion
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed 125 --arm Uniform-PerSample
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed 125 --arm Low-VS
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed 125 --arm Frequency-Only
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed 125 --arm RAVS-V1
& "D:\app\minicondaaa\envs\xiaoju\python.exe" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py report
```