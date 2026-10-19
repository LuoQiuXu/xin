# Low-VS 供体亲和度 × 间隔风险保护：2×2 实验结果

主终点：固定 epoch10。仅报告 TRAIN 留出集，不是官方 CIFAR-100 TEST 准确率。

## 协议与数据

冻结 Clean Stage-I seed42 epoch184，SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。
CIFAR-100-LT IF100 原 LT TRAIN 10847张；ResNet32；100-way FC；每组10epoch×85更新=850；batch128末批95。
原 SGD lr0.1/momentum0.9/wd0.0005、原 scheduler、原 CE。Backbone/BN affine和全部统计逐位冻结。
验证集是原始 TRAIN 的1980张留出，class1..99每类20，class0缺失，保留100维logits；历史重复使用，只支持探索性比较。
有效组：Head35类/700张，OfficialMedium35类/700张，Tail29类/580张。Many/Medium/Few按TRAIN数量>100、20..100、<20划分；与OfficialMedium定义不同。

|组|有效类别数|图像数|
|---|---:|---:|
|Many|34|680|
|Medium|35|700|
|Few|30|600|

## 四臂与配对

A=RandomDonor；B=HighAffinityDonor；C=A+MarginProtect；D=B+MarginProtect。仅y>=71接收者改变，其他draw沿用原供体和Low-VS。
A/B共享6个频次层的逐draw抽取；层内A按类频次随机、B从affinity最高2类按频次抽样；C供体逐项=A，D逐项=B。
所有臂共用接收者、增强随机流、Gumbel和非尾类输入；A/C与B/D的供体像素/特征/原VS分数逐batch完全一致。
正常路径保持原Bottom48，C/D保护其中q最小16个，再用同一Gumbel从余下32选16；fallback保持原mask，不执行保护。
每组850batch配对与整图通道复制检查通过；仅6500个FC参数更新，186项Backbone状态逐位不变，31个BN计数保持15640。
A/B同层内供体类别频次仍可不同，因此亲和度处理并未隔离全部类频率因素。

## 固定epoch10结果（%）

|seed|臂|Overall|Head|OfficialMedium|Tail|Many|Medium|Few|Macro|valCE|best epoch/Overall仅诊断|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|124|A|47.7273|59.8571|52.0000|27.9310|60.5882|51.7143|28.5000|47.7273|2.1534|10/47.7273|
|124|B|47.3737|59.2857|52.1429|27.2414|60.0000|51.8571|27.8333|47.3737|2.1612|8/47.4242|
|124|C|46.8182|61.5714|53.4286|21.0345|62.3529|53.1429|21.8333|46.8182|2.2341|9/46.9192|
|124|D|46.6162|61.4286|53.1429|20.8621|62.2059|52.8571|21.6667|46.6162|2.2341|5/46.7677|
|124|E|47.3232|59.8571|51.0000|27.7586|60.5882|50.7143|28.3333|47.3232|2.1530|9/47.4242|
|125|A|47.1212|59.1429|51.2857|27.5862|60.0000|50.8571|28.1667|47.1212|2.1529|6/47.2222|
|125|B|46.7172|58.7143|50.5714|27.5862|59.5588|50.1429|28.1667|46.7172|2.1582|5/47.4242|
|125|C|46.4646|61.2857|53.2857|20.3448|62.0588|53.0000|21.1667|46.4646|2.2310|5/46.7172|
|125|D|46.3636|60.7143|53.2857|20.6897|61.4706|53.0000|21.5000|46.3636|2.2299|5/46.5657|
|125|E|47.1717|59.5714|50.7143|27.9310|60.4412|50.2857|28.5000|47.1717|2.1532|6/47.2222|

E=历史LowVS-Full，仅作上下文，未重训。A将尾接收者供体限定为head，与E供体协议不同，E−A不作单因素因果解释。

## 逐图配对与真实Tail得失

|seed|比较|全体win/loss/net|Tail win/loss/net|ΔOverall pp|ΔHead pp|ΔTail pp|Tail wrong→wrong改预测|
|---|---|---|---|---:|---:|---:|---:|
|124|B-A|2/9/-7|0/4/-4|-0.3535|-0.5714|-0.6897|8|
|124|C-A|28/46/-18|1/41/-40|-0.9091|+1.7143|-6.8966|41|
|124|D-B|28/43/-15|1/38/-37|-0.7576|+2.1429|-6.3793|41|
|124|D-A|26/48/-22|1/42/-41|-1.1111|+1.5714|-7.0690|42|
|124|D-C|2/6/-4|1/2/-1|-0.2020|-0.1429|-0.1724|5|
|125|B-A|5/13/-8|1/1/+0|-0.4040|-0.4286|+0.0000|14|
|125|C-A|33/46/-13|1/43/-42|-0.6566|+2.1429|-7.2414|43|
|125|D-B|36/43/-7|1/41/-40|-0.3535|+2.0000|-6.8966|41|
|125|D-A|33/48/-15|1/41/-40|-0.7576|+1.5714|-6.8966|44|
|125|D-C|5/7/-2|4/2/+2|-0.1010|-0.5714|+0.3448|6|

Tail净收益只计真实标签正确与否的变化；wrong→wrong转移单独列出，没有计为准确率收益。
一张验证图像=0.050505pp Overall、0.142857pp Head、0.172414pp Tail；两个seed共享同一组图像，不能把3960次评估视为独立样本。

## Tail交互项

|seed|A正确|B正确|C正确|D正确|I=D−B−C+A（张）|I pp|
|---|---:|---:|---:|---:|---:|---:|
|124|162|158|122|121|+3|+0.5172|
|125|160|160|118|120|+2|+0.3448|

## 预注册判据

亲和度方向门槛（两seed B−A与D−C Tail均>0）：**False**。
保护方向门槛（两seed C−A与D−B Tail均>0）：**False**。
两seed交互为正且D>A：**False**；D对A完全无Overall/Head下降：**False**。
Tail跨seed变号比较：['D-C']；仅0–2张净变化：[{'seed': 124, 'contrast': 'D-C', 'tail_net_correct': -1}, {'seed': 125, 'contrast': 'B-A', 'tail_net_correct': 0}, {'seed': 125, 'contrast': 'D-C', 'tail_net_correct': 2}]。
按停止规则结果不确定：**True**。
协议没有给‘material regression’数值阈值，因此没有看结果后另设阈值。若有Overall/Head损失，列明成本，不擅自认定无实质退化。
n=2不做p值或显著性声明；即使方向门槛通过，也不能声称已建立泛化有效的新方法。

## 在线几何（更新之前；仅描述即时行为）

|seed|臂|Tail draws|保护draws|fallback|平均固定marginΔ|平均动态marginΔ|平均CEΔ|正margin跨界比例|保护相对同一当前FC原mask的平均q增益|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|124|A|31687|0|0|-0.838798|-1.871751|1.024443|0.323356|NA|
|124|B|31687|0|0|-0.771119|-1.779754|0.905150|0.298487|NA|
|124|C|31687|31687|0|1.823226|0.020522|0.128430|0.055060|2.694698|
|124|D|31687|31687|0|1.730829|0.010744|0.104102|0.055085|2.519794|
|125|A|31715|0|0|-0.845143|-1.857636|1.014193|0.322806|NA|
|125|B|31715|0|0|-0.773715|-1.782661|0.901417|0.297639|NA|
|125|C|31715|31715|0|1.820097|0.022346|0.131888|0.056608|2.690640|
|125|D|31715|31715|0|1.731995|0.004403|0.112028|0.060913|2.522849|

q是当前FC固定最强非真实竞争类别下的即时贡献，既不是语义重要性，也不是泛化效用。
保护路径逐draw验证确实剔除Bottom48中最小16个q，且同一当前FC下选中q和不低于原mask；不同训练臂的FC已分化，跨臂几何均值不视作纯掩码因果效应。

## 供体频率与亲和度

|seed|臂|Tail draws|平均供体训练样本数|affinity mean|affinity q05/q50/q95|6层实际draw数|
|---|---|---:|---:|---:|---|---|
|124|A|31687|303.1714|0.653705|0.524567/0.654462/0.779200|[9465, 7200, 5483, 4081, 3143, 2315]|
|124|B|31687|299.5142|0.712831|0.607823/0.714841/0.812840|[9465, 7200, 5483, 4081, 3143, 2315]|
|125|A|31715|303.3966|0.653067|0.523609/0.653689/0.779192|[9424, 7287, 5553, 4059, 3048, 2344]|
|125|B|31715|299.2738|0.711187|0.607104/0.713580/0.810968|[9424, 7287, 5553, 4059, 3048, 2344]|

完整类频次见 donor_plan_summary.csv；各层先验/实际次数见 donor_stratum_summary.csv；这些数据来自TRAIN，不依赖验证结果。

## 耗时与显存

|seed|臂|10epoch训练秒数（含诊断）|全程秒数|CUDA峰值allocated MiB|reserved MiB|
|---|---|---:|---:|---:|---:|
|124|A|64.34|64.93|116.05|148.00|
|124|B|65.27|65.86|116.05|148.00|
|124|C|68.30|68.91|116.05|148.00|
|124|D|69.19|69.82|116.05|148.00|
|125|A|64.64|65.24|116.05|148.00|
|125|B|65.17|65.76|116.05|148.00|
|125|C|67.34|67.89|116.05|148.00|
|125|D|80.58|81.23|116.05|148.00|

耗时包含新增在线诊断/CSV写入，不能直接与历史未记录相同诊断的用时比较为纯算法开销。

## 防护、异常与协议偏差

重新验证历史封存文件1117项，训练完成后哈希保持不变。官方TEST open/hash/构造guard事件0；无历史文件写入事件。
在首次打开任务说明之前，为定位任务文件执行过只读status/branch/log/ls-files查询；未改变版本状态。看到禁止条款后未再执行版本控制操作。此初始只读查询属于执行偏差，明确保留记录。
新入口静态审查阶段修正了草稿中的helper字段/参数名；正式CPU测试之前完成，没有改动任何历史源文件、数学定义或配对约束。
训练失败或其他异常以 FAILED_STOP.json 和 console_*.txt 为准；正常完成标记为 STOP_COMPLETED.json。没有自动重跑、补seed或超参数搜索。

## 可复查文件与运行命令

本地目录：`D:\xiangmuxunlian\xin\H2T_repro_audit\lowvs_factorial\results`。所有新文件仅位于此目录。源文件SHA见preflight.json/scientific_summary.json，完整环境见environment.json。
统计：epoch10.csv、paired.csv、per_class.csv、group_confusion.csv、paired_group_transitions.csv、factorial_effects.csv、geometry_summary.csv。
来源：historical_inputs.json、train_bank_provenance.json、head_prototypes.npz、train_gap_bank.npz、donor_plan_seed124.npz/125.npz、prepare.json。
每组：config.json、effective.yaml、epochs.csv、batch_signatures.csv、geometry_diagnostics.csv、final_fc.pth、validation_epoch10.npz、summary.json。
命令stdout和错误栈：results/console_unit.txt、console_preflight.txt、console_prepare.txt、console_train_seed{124,125}_{A,B,C,D}.txt、console_report.txt。

```powershell
Set-Location 'D:\xiangmuxunlian\xin'
$py = 'D:\app\minicondaaa\envs\xiaoju\python.exe'
$entry = 'H2T_repro_audit/lowvs_factorial/run_factorial.py'
& $py $entry unit
& $py $entry preflight
& $py $entry prepare
& $py $entry train --seed 124 --arm 'RandomDonor'
& $py $entry train --seed 124 --arm 'HighAffinityDonor'
& $py $entry train --seed 124 --arm 'RandomDonor+MarginProtect'
& $py $entry train --seed 124 --arm 'HighAffinityDonor+MarginProtect'
& $py $entry train --seed 125 --arm 'RandomDonor'
& $py $entry train --seed 125 --arm 'HighAffinityDonor'
& $py $entry train --seed 125 --arm 'RandomDonor+MarginProtect'
& $py $entry train --seed 125 --arm 'HighAffinityDonor+MarginProtect'
& $py $entry report
```

以上是首次执行顺序；输出存在时脚本拒绝覆写，不应直接在当前results上重跑。每条命令必须成功后才执行下一条。

## 停止与下一步

本轮8组、6800次更新完成后停止。不新增训练臂/seed，不重训Stage-I，不接触官方TEST，不自动执行版本控制。
若方向翻转或只涉及1–2张图，停止这一具体pilot，不在重复使用的验证集上扫参数。即使门槛通过，后续也仅值得在事先锁定、未使用的评估协议上独立复核，须另行确认。
