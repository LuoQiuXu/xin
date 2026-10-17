# Low-VS 扰动范数—方向归因：科学实验报告

日期：2026-10-16（Asia/Shanghai）

## 1. 问题与完成情况

检验在相同样本、IS供体、增强、Gumbel和候选通道下，逐样本配平GAP扰动范数后，Low-VS相对Uniform-PerSample是否仍有分类收益。本轮不是新方法，也不直接对照原H2T的batch共享随机掩码。

完成8组Stage II，每组10 epochs/850更新，共6800更新；没有追加训练。方向诊断：**C：Matched跨seed方向矛盾，暂停VS方法开发；若Full优势而Matched减弱，也仅有B的幅度线索，不能证明唯一范数机制**。下文同时列全部原始净图片数，方向成立不等于实质或显著收益。

固定epoch10均值：Full的Low-VS−Uniform为Overall +0.2778 pp、Tail +0.9483 pp；Matched为Overall +0.0505 pp、Tail +0.1724 pp。Matched逐seedOverall净图数=[5, -3]、Tail净图数=[3, -1]，必须保留这种变号，不用均值掩盖不稳定性。

## 2. 模型、训练与数据协议

Clean Stage I seed42 epoch184，SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。Backbone463504参数、31层BN完全eval冻结；同一初始6500参数100类FC用于全部独立训练。未加载其他Stage II权重。

原CIFAR-100-LT IF100训练子集10847张，真实官方TRAIN像素/标签按raw indices再次核对一致；开发验证1980张来自官方TRAIN，class1..99各20张，缺class0，索引与LT重叠0，输出保留100 logits。训练供体/接收者及扰动都不依赖开发验证。官方TEST没有读取或评估；W/b互换属于8组训练结束后的开发验证诊断。

验证组有效类别数：{'Head': 35, 'OfficialMedium': 35, 'Tail': 29, 'Many': 34, 'Medium': 35, 'Few': 30}。Head=0..35（验证有效1..35），OfficialMedium=36..70，Tail=71..99；Many按LT n>100，Medium按20≤n≤100，Few按n<20。训练alpha分组含Head class0（36类），与验证的有效35个Head类区别。

BS接收者、IS供体、增强与两种Gumbel采样通道均复用封存历史实现；16/64整通道；Low-VS Bottom48及小供体n<20等fallback不变。CE硬接收标签，batch128（每epoch最后batch95），原SGD momentum.9/wd.0005与原cos调度。配置lr=.1；原函数对唯一FC参数组乘lr_factor=.2，实际epoch1为.02，再cos衰减。8组相同，具体effective.yaml和epochs.csv已保存。

## 3. 干预与正确性

先形成两个完整硬替换δU/δL，用相同feature-map差先GAP再L2求nU/nL；公共r=min(nU,nL)。各自alpha=r/n，n≤1e−12安全取0，不放大。Matched是整个选中通道统一连续缩放，不再是原始硬替换；其余通道保持接收者。提供的norm_policy.py逐字节未变，增加的代码只保存缺失证据、保护旧文件并核查状态，Diff见evidence_recording.patch。

CPU数学测试及真实GPU预检PASS。全部8组均核对850 batch的5项输入签名、两种掩码哈希和额外范数诊断哈希；参考原始NPZ还能逐batch重算诊断hash。186个Backbone状态张量逐位不变，31个BN计数器全为15640；只有FC weight/bias实际变化。16个best/final均重载评估一致；543个历史文件hash不变，历史写保护及TEST guard无事件。

## 4. 固定epoch10主结果（%）

|seed|arm|Overall|Head|OfficialMedium|Tail|Many|Medium|Few|Macro|best epoch|best Overall|
|---|---|---|---|---|---|---|---|---|---|---|---|
|124|Uniform-Full|47.0707|60.1429|50.7143|26.8966|61.0294|50.2857|27.5000|47.0707|4|47.1212|
|124|LowVS-Full|47.3232|59.8571|51.0000|27.7586|60.5882|50.7143|28.3333|47.3232|9|47.4242|
|124|Uniform-Matched|47.0707|60.1429|50.7143|26.8966|61.0294|50.2857|27.5000|47.0707|4|47.1212|
|124|LowVS-Matched|47.3232|60.1429|51.0000|27.4138|60.8824|50.7143|28.0000|47.3232|9|47.3737|
|125|Uniform-Full|46.8687|60.1429|50.1429|26.8966|61.0294|49.7143|27.5000|46.8687|6|47.2222|
|125|LowVS-Full|47.1717|59.5714|50.7143|27.9310|60.4412|50.2857|28.5000|47.1717|6|47.2222|
|125|Uniform-Matched|46.8687|60.1429|50.1429|26.8966|61.0294|49.7143|27.5000|46.8687|6|47.2222|
|125|LowVS-Matched|46.7172|60.0000|50.0000|26.7241|60.8824|49.5714|27.3333|46.7172|5|46.9697|

## 5. 逐图配对与两个关键比较

wrong→right与right→wrong在相同1980张图像逐图计数；每一组的准确率差严格等于对应净图数/组样本数×100。

|seed|比较|wins|losses|net|ΔOverall pp|Tail net|ΔTail pp|Head net|ΔHead pp|ΔFew pp|
|---|---|---|---|---|---|---|---|---|---|---|
|124|LowVS-Full - Uniform-Full|18|13|5|+0.2525|5|+0.8621|-2|-0.2857|+0.8333|
|124|LowVS-Matched - Uniform-Matched|10|5|5|+0.2525|3|+0.5172|0|+0.0000|+0.5000|
|124|LowVS-Matched - LowVS-Full|9|9|0|+0.0000|-2|-0.3448|2|+0.2857|-0.3333|
|124|Uniform-Matched - Uniform-Full|1|1|0|+0.0000|0|+0.0000|0|+0.0000|+0.0000|
|125|LowVS-Full - Uniform-Full|18|12|6|+0.3030|6|+1.0345|-4|-0.5714|+1.0000|
|125|LowVS-Matched - Uniform-Matched|5|8|-3|-0.1515|-1|-0.1724|-1|-0.1429|-0.1667|
|125|LowVS-Matched - LowVS-Full|6|15|-9|-0.4545|-7|-1.2069|3|+0.4286|-1.1667|
|125|Uniform-Matched - Uniform-Full|0|0|0|+0.0000|0|+0.0000|0|+0.0000|+0.0000|

按seed等权的描述性均值：

|比较|ΔOverall pp|ΔHead pp|ΔTail pp|ΔFew pp|
|---|---|---|---|---|
|LowVS-Full - Uniform-Full|+0.2778|-0.4286|+0.9483|+0.9167|
|LowVS-Matched - Uniform-Matched|+0.0505|-0.0714|+0.1724|+0.1667|
|LowVS-Matched - LowVS-Full|-0.2273|+0.3571|-0.7759|-0.7500|
|Uniform-Matched - Uniform-Full|+0.0000|+0.0000|+0.0000|+0.0000|

Full与Matched是各自同环境、同预算新训练。Full差值减Matched差值不是可识别的“范数贡献百分比”：两次缩放同时改变训练增强分布、完整图范数和头部优化轨迹。

## 6. 逐样本配平，而非均值配平

每个seed包含108470个带增强的BS配对训练draw；相同原图会多次出现，不是108470个独立原图。浮点比较沿用提供策略容差 `1e-5+1e-4*r`，不宣称逐位完全相等。

|seed|draws|batch|max abs(U*-L*)|max abs(U*-r)|max abs(L*-r)|max误差/容差|失败数|U精确零|L精确零|任一≤epsilon|
|---|---|---|---|---|---|---|---|---|---|---|
|124|108470|850|9.54e-07|9.54e-07|9.54e-07|0.0023|0|0|0|0|
|125|108470|850|1.43e-06|9.54e-07|1.43e-06|0.0024|0|0|0|0|

所有epoch训练draw的分布：

|seed|量|均值|标准差|5%|中位数|95%|min|max|
|---|---|---|---|---|---|---|---|---|
|124|alpha_lowvs|0.89290|0.11117|0.66662|0.92600|1.00000|0.23225|1.00000|
|124|alpha_uniform|0.99609|0.01823|0.97698|1.00000|1.00000|0.49548|1.00000|
|124|full_map_norm_lowvs_matched|57.74797|11.04932|39.65584|57.59577|76.06738|14.67172|114.93980|
|124|full_map_norm_uniform_matched|60.99827|9.56119|46.32897|60.50547|77.48020|18.82435|116.77612|
|124|matched_norm_lowvs|3.46313|0.91372|2.09936|3.38763|5.09050|0.60989|8.76004|
|124|matched_norm_uniform|3.46313|0.91372|2.09936|3.38763|5.09050|0.60989|8.76004|
|124|original_norm_lowvs|3.89278|0.94230|2.46041|3.82906|5.54401|0.69899|8.86025|
|124|original_norm_uniform|3.47625|0.91320|2.11128|3.40036|5.10040|0.60989|8.76004|
|125|alpha_lowvs|0.89302|0.11088|0.66685|0.92596|1.00000|0.23173|1.00000|
|125|alpha_uniform|0.99619|0.01781|0.97766|1.00000|1.00000|0.51213|1.00000|
|125|full_map_norm_lowvs_matched|57.75033|11.05895|39.74980|57.62445|76.10910|12.36211|108.91171|
|125|full_map_norm_uniform_matched|60.99766|9.58706|46.21295|60.48228|77.59739|16.77015|110.18159|
|125|matched_norm_lowvs|3.46525|0.91774|2.09650|3.39017|5.09638|0.66005|9.27088|
|125|matched_norm_uniform|3.46525|0.91774|2.09650|3.39017|5.09638|0.66005|9.27088|
|125|original_norm_lowvs|3.89485|0.94881|2.45907|3.82837|5.56678|0.69480|9.27088|
|125|original_norm_uniform|3.47803|0.91722|2.11004|3.40222|5.10783|0.66005|9.34834|

Head/Tail接收者的alpha分布（训练含class0；逐组频次不是验证集有效类数）：

|seed|接收组|alpha|draws|均值|5%|中位数|95%|alpha=0%|alpha=1%|
|---|---|---|---|---|---|---|---|---|---|
|124|Head|alpha_lowvs|38971|0.89541|0.67739|0.92706|1.00000|0.000|16.587|
|124|Head|alpha_uniform|38971|0.99548|0.97249|1.00000|1.00000|0.000|86.729|
|124|Tail|alpha_lowvs|31687|0.89214|0.66293|0.92548|1.00000|0.000|14.943|
|124|Tail|alpha_uniform|31687|0.99652|0.97954|1.00000|1.00000|0.000|88.371|
|125|Head|alpha_lowvs|39005|0.89528|0.67516|0.92697|1.00000|0.000|16.154|
|125|Head|alpha_uniform|39005|0.99567|0.97358|1.00000|1.00000|0.000|87.084|
|125|Tail|alpha_lowvs|31715|0.89311|0.67005|0.92491|1.00000|0.000|15.368|
|125|Tail|alpha_uniform|31715|0.99619|0.97777|1.00000|1.00000|0.000|87.974|

完整分位数、每epoch及分组alpha见norm_distribution.csv、alpha_by_epoch_group.csv；逐batch最大误差见各组batch_norm_diagnostics.csv；逐样本原值及接收者/供体索引在各seed/Uniform-Full/norm_diagnostics.npz。另三组的诊断哈希逐batch一致，因此共用这份范数原始证据。完整8×8特征图L2并未被等化，已额外测量而未把它当作调参目标。

## 7. 分类头W/b交叉组合

以下为训练结束后，用固定开发验证特征将两套FC的W/b交叉组合。U代表相应Uniform组，L代表相应Low-VS组；Full和Matched分开比较。与原classifier_hybrids.csv逐行复核，底层float64线性运算也检查了相对训练时float32原模型的预测变化。

|seed|比较|组合|Overall|Head|Tail|Few|原生预测差异图数|
|---|---|---|---|---|---|---|---|
|124|LowVS-Full vs Uniform-Full|U_W+U_b|47.0707|60.1429|26.8966|27.5000|0|
|124|LowVS-Full vs Uniform-Full|L_W+L_b|47.3232|59.8571|27.7586|28.3333|0|
|124|LowVS-Full vs Uniform-Full|U_W+L_b|47.1212|60.1429|26.8966|27.5000|不适用|
|124|LowVS-Full vs Uniform-Full|L_W+U_b|47.3232|59.8571|27.7586|28.3333|不适用|
|124|LowVS-Matched vs Uniform-Matched|U_W+U_b|47.0707|60.1429|26.8966|27.5000|0|
|124|LowVS-Matched vs Uniform-Matched|L_W+L_b|47.3232|60.1429|27.4138|28.0000|0|
|124|LowVS-Matched vs Uniform-Matched|U_W+L_b|47.0707|60.1429|26.8966|27.5000|不适用|
|124|LowVS-Matched vs Uniform-Matched|L_W+U_b|47.3232|60.1429|27.4138|28.0000|不适用|
|125|LowVS-Full vs Uniform-Full|U_W+U_b|46.8687|60.1429|26.8966|27.5000|0|
|125|LowVS-Full vs Uniform-Full|L_W+L_b|47.1717|59.5714|27.9310|28.5000|0|
|125|LowVS-Full vs Uniform-Full|U_W+L_b|46.8687|60.1429|26.8966|27.5000|不适用|
|125|LowVS-Full vs Uniform-Full|L_W+U_b|47.1212|59.4286|27.9310|28.5000|不适用|
|125|LowVS-Matched vs Uniform-Matched|U_W+U_b|46.8687|60.1429|26.8966|27.5000|0|
|125|LowVS-Matched vs Uniform-Matched|L_W+L_b|46.7172|60.0000|26.7241|27.3333|0|
|125|LowVS-Matched vs Uniform-Matched|U_W+L_b|46.8687|60.1429|26.8966|27.5000|不适用|
|125|LowVS-Matched vs Uniform-Matched|L_W+U_b|46.7172|60.0000|26.7241|27.3333|不适用|

Tail的条件参数干预（相对U_W+U_b，单位pp）：

|seed|比较|完整L变化|只换b@U_W|只换W@U_b|非可加剩余|
|---|---|---|---|---|---|
|124|LowVS-Full vs Uniform-Full|+0.8621|+0.0000|+0.8621|+0.0000|
|124|LowVS-Matched vs Uniform-Matched|+0.5172|+0.0000|+0.5172|+0.0000|
|125|LowVS-Full vs Uniform-Full|+1.0345|+0.0000|+1.0345|+0.0000|
|125|LowVS-Matched vs Uniform-Matched|-0.1724|+0.0000|-0.1724|+0.0000|

若保持U_W仅换L_b主要复制Tail差值，分类器偏置需优先审视；但另一路径和非可加剩余必须同时看。W和b共同训练，分类边界及argmax是非线性的，互换模型可能不处于其训练轨迹，不可把条件替换效果分解成唯一训练因果贡献。完整Overall/Head/Few和参数L2差见hybrid_parameter_effects.csv。

实际W/b诊断：两个seed×Full/Matched共四个对比中，保持U_W只换L_b的Tail变化均为0；保持U_b只换L_W均复制对应完整L的Tail差值。本轮不支持“最终bias单独替换解释Tail差异”。变化在这项条件干预中跟随W，但W可编码类别尺度/边界等，不等于语义方向被证明；W/b共同训练，也不能排除bias在训练路径中的间接作用。

## 8. 预先情形与机制边界

|严格方向描述（不替代效应大小）|是否|
|---|---|
|Full_Tail_both_positive|True|
|Matched_Tail_both_positive|False|
|Full_Overall_both_nonnegative|True|
|Matched_Overall_both_nonnegative|False|

C：Matched跨seed方向矛盾，暂停VS方法开发；若Full优势而Matched减弱，也仅有B的幅度线索，不能证明唯一范数机制。未事后为“明显反向”或“很弱”新增数值阈值；请同时检查两个seed净图数、Head损失和best敏感性。

- 实际干预证据：相同训练draw下逐样本GAP范数在预声明容差内相同，alpha仅衰减；这一点可由原始NPZ和850个batch核查。
- 模型性能证据：只支持本表固定epoch10、99类开发验证协议下的差异；n=2不建立发表级显著性。
- 机制解释：若配平后正向差异保留，GAP幅度一个标量不足以完全解释；剩余还包括通道身份、完整空间结构、扰动方向、类别条件缩放分布和分类头优化差异，不能直接称“尾类语义通道有效”。
- 若Matched优势消失/反向，幅度解释更可信，但缩放改变增强分布，不能证明其为唯一原因。若仅少数图像或跨seed反向，应暂停VS算法开发。
- Backbone和BN始终未变，本轮不存在训练得到更强Backbone表征的证据。

## 9. best敏感性与运行开销

- seed124 LowVS-Full−Uniform-Full：epoch10 +0.2525 pp，各自best +0.3030 pp，方向反转=False。
- seed125 LowVS-Full−Uniform-Full：epoch10 +0.3030 pp，各自best +0.0000 pp，方向反转=False。
- seed124 LowVS-Matched−Uniform-Matched：epoch10 +0.2525 pp，各自best +0.2525 pp，方向反转=False。
- seed125 LowVS-Matched−Uniform-Matched：epoch10 -0.1515 pp，各自best -0.2525 pp，方向反转=False。

|seed|arm|总工作秒|训练循环秒|allocated峰值MiB|reserved峰值MiB|
|---|---|---|---|---|---|
|124|Uniform-Full|57.51|54.92|122.5|148.0|
|124|LowVS-Full|57.69|55.36|122.5|148.0|
|124|Uniform-Matched|56.87|54.59|122.5|148.0|
|124|LowVS-Matched|57.12|54.78|122.5|148.0|
|125|Uniform-Full|56.80|54.27|122.5|148.0|
|125|LowVS-Full|57.32|55.04|122.5|148.0|
|125|Uniform-Matched|56.93|54.64|122.5|148.0|
|125|LowVS-Matched|57.49|55.17|122.5|148.0|

总工作秒含加载、校验、保存和重载。训练循环秒包括增强、双分支冻结前向、四候选构建/诊断及分类器更新，不含epoch验证；所有组使用同样运行引擎和诊断，不表示生产优化后的纯算法耗时。显存为PyTorch分配器统计，不含Windows/驱动/CUDA context。effective.yaml保留历史model_dir/name元数据但resume清空；实际输出都显式指向本轮results并有运行时写保护。

## 10. 局限与异常

第一次离线分析已完成全部核验并写出报告，但外层日志进程在中文stdout处遇到GBK/UTF-8编码冲突（UnicodeEncodeError）。保留失败日志，统一PYTHONIOENCODING=utf-8后仅重跑一次离线汇总；8组训练无重跑。NPZ诊断一次读入内存，避免重复解压，数值与判据不变。详见analysis_stdout_error.json。
报告脚本准备有两次仅文本匹配的AssertionError，两次均未写入或影响训练。AST检查确认原Markdown换行正确，无需修复；之后仅完善了新报告的条件说明。详见analysis_preparation_notes.json。
验证只有99类且缺class0，保留100 logits不能补足class0召回；不能直接和论文官方100类TEST准确率比较。相同1980张官方TRAIN留出已经多轮用于开发，Clean Stage I的epoch184也由该留出选择，存在开发选择偏差；没有独立终检。两个Stage II种子共用一个Stage I和同一验证图像，不把3960次重复预测当作独立观测。Overall一张图0.050505 pp、Tail一张0.172414 pp、Few一张0.166667 pp；弱差值不夸大。

训练失败=0；配对失败=0；范数容差失败=0；TEST访问尝试=0；历史写入尝试=0；历史文件变化=0。训练无重跑、无续训、无删除失败目录。run_experiment.py和norm_policy.py在预检之后保持锁定；模型权重和原始数据未被改写。完整stdout在console/，执行状态/错误记录在run_registry.json。

## 11. 环境、复现与文件

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

从实际本地项目根目录运行；已有结果不可覆盖：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$py = 'D:\app\minicondaaa\envs\xiaoju\python.exe'
& $py -B H2T_repro_audit/lowvs_norm_direction/norm_policy.py
& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py preflight
& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 124 --arm Uniform-Full
& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 124 --arm LowVS-Full
& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 124 --arm Uniform-Matched
& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 124 --arm LowVS-Matched
& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 125 --arm Uniform-Full
& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 125 --arm LowVS-Full
& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 125 --arm Uniform-Matched
& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py train --seed 125 --arm LowVS-Matched
& $py -B H2T_repro_audit/lowvs_norm_direction/run_experiment.py report
& $py -B H2T_repro_audit/lowvs_norm_direction/analyze_results.py
```

本报告所在results目录保留：preflight.json、dataset_protocol.json、preregistered_decision_rules.json、historical_manifest.json、初始源码provided_source/和evidence_recording.patch；epoch10.csv/epoch10_verified.csv、paired.csv/paired_verified.csv、per_class.csv、classifier_hybrids.csv/classifier_hybrids_verified.csv、hybrid_parameter_effects.csv、scientific_summary.json、norm_distribution.csv、norm_matching_verification.csv、alpha_by_epoch_group.csv、epochs_all.csv、checkpoint_manifest.csv、final_integrity.json。每个seed/arm下epochs.csv、batch_signatures.csv、batch_norm_diagnostics.csv、validation_epoch10.npz、config.json、effective.yaml、summary.json和best/final.pth；每seed Uniform-Full下另存逐样本norm_diagnostics.npz。

## 12. 停止

一次协议已完整执行；无新α、K、供体策略、网络、损失、Stage I、额外seed或官方TEST。当前仅保守记录范数控制与分类头诊断，不认定发现语义机制。结果由用户提供给ChatGPT复核后，再决定研究方向；本轮到此停止。
