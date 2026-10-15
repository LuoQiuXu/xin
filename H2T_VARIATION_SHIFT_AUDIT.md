# H2T 类内变化 / 均值偏移通道机制审计

完成时间：2026-10-13T16:54:01+08:00。**纯冻结推理，无 Stage I/II 训练、无验证或测试评估、无超参数搜索。**

## 1. 三个问题的结论

**Q1：有明显通道区分力，Top48 池具有较好的稳定性；精确排序和 Top16 不可称为高度稳定。** 正则化评分的通道极差中位数 0.9771；两套增强的 Spearman 中位数 0.7822，High Top48 重合率中位数 93.7500%。High Top16 重合率均值仅 58.9708%；尾类供体统计仍不确定，按事先规则回退Uniform。

**Q2：确实改变实际掩码与迁移统计组成。** High 主要减少绝对均值偏移，类内方差绝对值没有增加；Low 呈相反方向。Shuffled 的统计组成接近 Uniform，说明通道身份与真实评分的对应有作用。它不是上一轮几乎不动掩码的弱干预。

**Q3：足以提出下一轮受控 Stage II 训练验证，尚不足以断言分类收益。** 建议只在另行确认后比较当前锁定的四个逐样本策略，继续保留小样本回退和配对控制。依据是可复现的采样与明显、跨两套增强一致的统计干预，不能把即时 Margin 或训练图像上的判对率当作训练收益。当前已停止。

## 2. 数据、模型和隔离

| 项目 | 实际使用 |
| --- | --- |
| 数据 | 原 CIFAR-100-LT IF100，100类，10847张LT训练图像；ResNet32 |
| 接收 / 供体 | 原BS规则10847次抽样，4793个不同接收图像；原IS单供体排列，每个LT实例恰好一次 |
| Stage I | D:\xiangmuxunlian\新\H2T_repro_audit\runtime\saved\cifar100_imb001_stage1_ce_mixup_formal_20261011_1429_r2_202610111428\ckps\model_best.pth.tar |
| Stage I SHA256 | 0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e |
| 参考Classifier | 固定BN DR 第10epoch；100维logits；独立冻结 |
| 参考Checkpoint | D:\xiangmuxunlian\新\H2T_repro_audit\bn_audit_runs\fixed_bn_20261011_2255\runtime\saved\cifar100_if100_stage2_fixed_bn_20261011_2255_dr_202610112255\ckps\model_final.pth.tar |
| 参考SHA256 | 1b054d8b45753c25dedd9897e3052e5f33c73b0c38b47acaa277c0f61c50283d |
| 参数状态 | Backbone、BN affine和全部running buffers、参考Classifier全部eval / requires_grad=False |
| 主视图 | 确定性官方eval transform；每图GAP64，一个图像为一个统计观测 |
| 额外视图 | 固定训练增强augA与augB；每个LT实例各增强一次，不是每个重复BS draw再独立增强 |
| 融合 | 每次16个完整8×8通道，rho0.25，接收标签不变 |
| 分组 | Head ID0..35，Medium-ID36..70，Tail71..99；Many n>100 / Medium-n20..100 / Few n<20 |
| 目录 | D:\xiangmuxunlian\新\H2T_repro_audit\variation_shift_runs\variation_shift_20261013_seed123 |


LT raw ID 逐项比对原始 CIFAR train 文件和既存 train 数组，不新建划分。[split_indices.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/split_indices.npz) 只取原 `train_raw_indices`；保存的其他成员不参与评分或评估。官方eval与train transform从复制的原数据加载类提取，保留其CIFAR10归一化常数。没有读取官方test图像、验证图像或用它们定义规则；历史文件哈希只用于保护既有输出。

Stage I Backbone与参考DR checkpoint的Backbone state逐位相同。新脚本与48个兼容源文件位于独立目录，原H2T方法不改；[preflight.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/preflight.json)、[source_lock.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/source_lock.json)、[git_run_provenance.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/git_run_provenance.json)、[model_initial_hashes.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/model_initial_hashes.json) 保存环境、raw ID与源码校验。

## 3. 恒等式、抽样分布和方差估计边界

固定接收特征 f_t 和供体类 c：`δ=f_h−f_t=r+b`，`r=f_h−μ_c`，`b=μ_c−f_t`。对同一个供体分布的真实均值/方差有 `E[r]=0`，故 `E[δ²]=Var(f_h)+(μ_c−f_t)²`。需要有限二阶矩，并对供体随机性取期望；这不是每一个供体样本的平方等式。

主统计把当前固定eval特征的类内均匀有限集合明确当作经验分布：`μ_c=mean(x)`，`v0_c=mean((x−μ_c)²)`（ddof0）。因此对这个经验集合，期望分解严格成立。原IS随机排列的单位置、给定类别的边际实例分布均匀；无放回导致位置相关，若条件化于已经抽出的实例，剩余池均值/方差可能改变，不能套用全类统计声称条件期望相同。

同时保存 ddof1 无偏样本方差。它只有在代表性IID图像等条件下才估计真实类别总体方差；仅5张图像无法证明总体可靠。不能混淆 `(f_h−μ_c)²` 的单供体残差平方、估计方差和真实总体方差。原始与正则化评分均保存，本轮未择优换版本。

独立二阶矩展开核验全部10847×64个条件：最大绝对误差 3.242e-14。逐实际融合核验 `δ²=r²+b²+2rb`：最大绝对误差 2.842e-14。样本交叉项未假定为0；[selected_variation_shift.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/selected_variation_shift.csv) 保留其值。

主恒等式对应eval经验分布。augA/B诊断继续用冻结eval均值和方差评分，不能将eval方差冒充训练增强后的真实分布方差；这些视图中的Vemp仍指eval经验方差。另存增强统计并独立分析统计漂移。正则化v是代理，`v_reg+B`也不是经验分布精确二阶矩。

## 4. 预先固定的统计方案

每图最后 `[64,8,8]` 特征执行原分类器 `avg_pool2d(map,8).view(B,64)`；神经网络float32推理，统计float64。不把64个空间位置当独立图像。

数值项：`epsilon=1e−6 × 全LT 64维无偏方差均值` = 4.42654486782e-07；尺度为 0.442654486782。选择集合的比例按用户定义用一个相同scalar epsilon，不乘16；它与平均单通道评分不同。

预声明收缩：`lambda_c=5/(n_c+5)`，`v_reg=(1−lambda_c)v0_c+lambda_c v_pool`；`v_pool=各类中心残差平方总和/(10847−100)`（逐通道），均值不收缩。池化目标更受多样本类影响，是先验代理，不是真实尾类方差。lambda从n500的0.0099到n5的0.5，未调参。供体n<20则High/Low/Shuffled均回退相同Uniform mask，IS序列不变。

30个Few供体类（ID70..99）共305个IS图像，占 2.8118%；全部仍参加融合。原始评分即使有区分力也不取消回退。未发生非有限或全等评分回退。

| 供体频次组 | 类数 | Raw方差bootstrap相对SD类均值 | 收缩后相对SD | Raw LOO相对变化均值 | 收缩后LOO均值 |
| --- | --- | --- | --- | --- | --- |
| Many | 35 | 0.1207 | 0.1166 | 0.0075 | 0.0073 |
| Medium | 35 | 0.2315 | 0.1942 | 0.0365 | 0.0335 |
| Few | 30 | 0.3712 | 0.1923 | 0.1783 | 0.1223 |


每类128次图像级有放回bootstrap，记录抽样索引、均值/方差及逐通道评分区间；不是总体置信覆盖保证。ddof0 bootstrap方差平均天然为经验var0的 `(1−1/n)` 倍，不能把bootstrap均值当无偏真实方差。删一图像时重新计算n−1的收缩系数，池化目标保持固定。Few原始方差相对SD平均约37.1%，收缩后约19.2%；收缩降低敏感性但不补足样本信息。

[class_feature_statistics.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/class_feature_statistics.npz)、[class_statistics_diagnostics.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/class_statistics_diagnostics.csv)、[bootstrap_class_statistics.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/bootstrap_class_statistics.npz)、[bootstrap_within_class_indices.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/bootstrap_within_class_indices.npz)、[score_uncertainty.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/score_uncertainty.npz) 保留数据与不确定性。

![小样本统计不确定性](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/figures/06_class_stat_uncertainty.png)

## 5. 评分区分能力

`S=v/[v+(mu_h−f_t)^2+epsilon]` 的运行版本使用v_reg；它仅表示代理预期平方幅度中的类内方差份额，不是语义相似度、分类器权重、判别重要性或安全性。

| 接收组 | draw数 | 通道S均值 | S的P05/中位/P95 | 每接收64通道极差中位 | 每接收std均值 |
| --- | --- | --- | --- | --- | --- |
| All | 10847 | 0.5238 | 0.0540/0.5062/0.9898 | 0.9771 | 0.2987 |
| ReceiverHead | 4014 | 0.5391 | 0.0626/0.5289/0.9907 | 0.9748 | 0.2960 |
| ReceiverMediumID | 3700 | 0.5136 | 0.0481/0.4924/0.9891 | 0.9798 | 0.3006 |
| ReceiverTail | 3133 | 0.5164 | 0.0518/0.4945/0.9894 | 0.9765 | 0.2998 |


原始/收缩前后、三视图、接收/供体频次分组的完整连续分布见 [score_distributions.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/score_distributions.json)；100个供体类别差异见 [scores_by_donor_class.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/scores_by_donor_class.csv)。未以阈值筛出有利接收者。

![评分连续分布](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/figures/01_score_separation.png)

## 6. 三种稳定性来源

两套固定训练增强先独立重提全部LT特征。以下表将接收者增强与类别统计变化拆开：固定eval类别统计比较f_t(A/B)；固定eval接收特征比较μ/v(A/B)；bootstrap固定eval接收特征只重抽供体类图像。

| 分组 | 接收增强ρ P05/中位/P95 | High48重合均值 | Low48重合均值 | 统计增强ρ中位 | bootstrapρ中位 |
| --- | --- | --- | --- | --- | --- |
| All | 0.5527/0.7822/0.9245 | 0.9312 | 0.8632 | 0.9957 | 0.9920 |
| ReceiverHead | 0.5383/0.7724/0.9232 | 0.9287 | 0.8595 | 0.9955 | 0.9914 |
| ReceiverMediumID | 0.5696/0.7913/0.9281 | 0.9336 | 0.8660 | 0.9960 | 0.9925 |
| ReceiverTail | 0.5512/0.7848/0.9228 | 0.9316 | 0.8648 | 0.9959 | 0.9922 |
| DonorMany | 0.5496/0.7785/0.9229 | 0.9304 | 0.8624 | 0.9963 | 0.9930 |
| DonorMediumFreq | 0.5735/0.7947/0.9292 | 0.9348 | 0.8668 | 0.9882 | 0.9769 |
| DonorFew | 0.5801/0.8018/0.9367 | 0.9354 | 0.8682 | 0.9650 | 0.9334 |


Top集合重合定义为交集数/16或/48，同时另存Jaccard。Tail接收者增强ρ中位约0.785，Top48均值约93.16%，与Head接近；**High Top16均值仅59.43%**，不可宣称尾部通道精确排序稳定。全体High Top16均值58.97%；不支持擅自改成直接固定Top16选择。

供体Few正则化bootstrapρ中位0.9334，低于Many的0.9930；高排序相关性不等于总体方差准确。原始评分与正则化评分、Top16/48的High/Low稳定性、组合增强、bootstrap有效秩相关次数及区间均在 [stability_analysis.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/stability_analysis.csv)、[stability_measures.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/stability_measures.npz)、[stability_group_summary.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/stability_group_summary.csv)。常数向量的Spearman保留未定义，不补成1。

![稳定性来源](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/figures/02_stability_sources.png)

## 7. 掩码规则、实际差异与打乱控制

四组均为逐样本：A在64通道中Uniform；B在S_reg最高48中Uniform选16；C在最低48中选16；D先打乱S与实际通道身份绑定，再按最高48选16。稳定排序对平分采用通道索引顺序；Top48是预设候选规模，不是最优证明。三种引导在预声明回退时与A逐位同mask。

每draw同一保存的Gumbel64基础噪声；在候选池外设−inf后取噪声Top16。等权候选池内这是均匀、无放回子集抽样。评分只决定池，未改变IS供体。Shuffled为每 `(接收raw ID,供体class ID)` 以独立SeedSequence固定置换，跨视图复用；只打乱评分绑定，不重排实际特征通道。

| 比较，主eval视图 | 总体共享通道均值/16 | 总体相同率 | 不回退draw共享均值 | 不回退相同率 |
| --- | --- | --- | --- | --- |
| High-Uniform | 12.1014 | 3.3097% | 11.9886 | 0.5122% |
| Low-Uniform | 12.1226 | 3.3281% | 12.0104 | 0.5312% |
| Shuffled-High | 11.4509 | 2.8395% | 11.3193 | 0.0285% |
| High-Low | 10.1427 | 2.8118% | 9.9732 | 0.0000% |


不回退时理论Uniform与任意独立Top48池共享通道期望=12，完全相同概率 0.4616%；实测High/Low分别约12.0及0.51/0.53%吻合。总体相同率约3.3%含2.81%的明确回退；与上轮SMART的约76.67%不同，本轮实际干预明显。

Shuffled的独立key数 10288，重复key 559；每对平均保持 1.0018 个索引不动，实际评分值改变约 62.9982 个位置，真实/打乱评分ρ均值 0.0004。置换重放与非恒等检查通过。

每组主视图分别覆盖10847种不同16通道组合；三视图逐通道实际计数/频率、分组重合、跨增强运行池/实际mask稳定性见 [channel_selection_frequency.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/channel_selection_frequency.csv)、[mask_overlap_group.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/mask_overlap_group.csv)、[operational_mask_stability.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/operational_mask_stability.csv)。原batch共享参照有85个不同组合，不可将其与逐样本的差异单归因S。

![掩码与频率](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/figures/03_actual_masks_and_frequency.png)

## 8. 实际选中集合的统计组成

V_reg=sum(v_reg)，V_emp=sum(eval类内经验var0)，B=sum((μ_eval−f_t)^2)。表中的比例是每draw `V/(V+B+epsilon)` 再平均，**不是平均V除以平均(V+B)**。单供体R=sum((f_h−μ_eval)^2)另存，不能拿R代替V。

| 策略，主eval | 所选S均值 | V_reg均值 | V_emp均值 | B均值 | V_reg份额均值 | V_emp份额均值 | 实际R均值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform-PerSample | 0.5236 | 3.7877 | 3.7941 | 11.9708 | 0.2815 | 0.2815 | 3.7774 |
| High-VS | 0.6464 | 3.7614 | 3.7654 | 3.5748 | 0.5429 | 0.5427 | 3.7782 |
| Low-VS | 0.3965 | 3.8735 | 3.8844 | 15.6833 | 0.2270 | 0.2271 | 3.8844 |
| Shuffled-VS | 0.5239 | 3.7897 | 3.7960 | 11.9670 | 0.2816 | 0.2816 | 3.7922 |
| Original-H2T-Shared | 0.5236 | 3.7573 | 3.7643 | 11.8365 | 0.2808 | 0.2808 | 3.7942 |


High的V略低于Uniform，B下降约70%，份额升高；Low的V略高但B更高、份额降低。High-VS不是单纯“增加类内方差”操作。Shuffled均值接近Uniform，支持真实身份绑定对这一统计组成的影响。全部结果包含回退样本，没有删去不利类别。

连续分布的P05/中位/P95：

| 策略 | V_reg | B | 比例 |
| --- | --- | --- | --- |
| Uniform-PerSample | 2.5015/3.7315/5.2970 | 3.6716/10.3726/25.9206 | 0.1193/0.2648/0.4965 |
| High-VS | 2.5745/3.7043/5.1690 | 1.3820/3.0441/6.9347 | 0.3609/0.5471/0.7116 |
| Low-VS | 2.5131/3.8105/5.4853 | 5.7822/14.0405/31.5380 | 0.1006/0.2142/0.3927 |
| Shuffled-VS | 2.5056/3.7230/5.3332 | 3.7278/10.3374/25.9172 | 0.1200/0.2668/0.4963 |


![绝对变化与比例](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/figures/04_selected_variation_shift.png)

按接收者Head / Medium-ID / Tail，主视图组成：

| 接收组 | 策略 | V_reg均值 | B均值 | 比例均值 | 所选S均值 |
| --- | --- | --- | --- | --- | --- |
| ReceiverHead | Uniform-PerSample | 3.7802 | 10.2893 | 0.3084 | 0.5400 |
| ReceiverHead | High-VS | 3.7591 | 3.3443 | 0.5627 | 0.6614 |
| ReceiverHead | Low-VS | 3.8620 | 13.3990 | 0.2518 | 0.4154 |
| ReceiverHead | Shuffled-VS | 3.7798 | 10.2901 | 0.3089 | 0.5397 |
| ReceiverMediumID | Uniform-PerSample | 3.7938 | 13.5821 | 0.2594 | 0.5145 |
| ReceiverMediumID | High-VS | 3.7564 | 3.7034 | 0.5307 | 0.6377 |
| ReceiverMediumID | Low-VS | 3.8929 | 18.0017 | 0.2053 | 0.3839 |
| ReceiverMediumID | Shuffled-VS | 3.8007 | 13.5983 | 0.2591 | 0.5145 |
| ReceiverTail | Uniform-PerSample | 3.7900 | 12.2223 | 0.2731 | 0.5133 |
| ReceiverTail | High-VS | 3.7703 | 3.7181 | 0.5320 | 0.6375 |
| ReceiverTail | Low-VS | 3.8654 | 15.8720 | 0.2207 | 0.3873 |
| ReceiverTail | Shuffled-VS | 3.7894 | 12.1888 | 0.2731 | 0.5147 |


两套固定增强继续使用主eval统计评分，效果方向仍一致：

| 视图 | 策略 | V_reg | B | 比例 |
| --- | --- | --- | --- | --- |
| augA | Uniform-PerSample | 3.7877 | 10.5274 | 0.3014 |
| augA | High-VS | 3.6917 | 3.3492 | 0.5505 |
| augA | Low-VS | 3.8966 | 13.7664 | 0.2463 |
| augA | Shuffled-VS | 3.7889 | 10.5064 | 0.3015 |
| augB | Uniform-PerSample | 3.7877 | 10.6081 | 0.3000 |
| augB | High-VS | 3.6948 | 3.3668 | 0.5505 |
| augB | Low-VS | 3.8989 | 13.8519 | 0.2457 |
| augB | Shuffled-VS | 3.7860 | 10.5454 | 0.3005 |


完整分布与配对差值见 [fusion_summary.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/fusion_summary.json)、[paired_fusion_contrasts.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/paired_fusion_contrasts.json)、[fusion_group_means.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/fusion_group_means.csv)。这是基于固定经验统计的迁移组成证据，尚未证明学习后泛化更好。

## 9. 融合后的几何与参考判别变化

参考模型是固定BN DR第10epoch分类器，全程eval。Margin=`z_y−max(z_j,j≠y)`，D=融合前Margin−融合后Margin；正D表示Margin下降。评分与候选池不调用Margin，所有分析都在选择后进行。

| 策略，主eval | GAP L2均值 | 完整map L2均值 | D均值 | D P05/中位/P95 | 预测类别改变率 |
| --- | --- | --- | --- | --- | --- |
| Uniform-PerSample | 3.8260 | 64.3186 | 1.7279 | -0.7900/1.4734/5.1409 | 26.0256% |
| High-VS | 2.6013 | 53.7533 | 0.5201 | -0.8630/0.4269/2.1716 | 6.9789% |
| Low-VS | 4.2892 | 68.1778 | 2.2967 | -0.5868/2.0414/5.9924 | 36.5354% |
| Shuffled-VS | 3.8299 | 64.3095 | 1.7168 | -0.8071/1.4503/5.1420 | 25.7767% |
| Original-H2T-Shared | 3.8267 | 64.2633 | 1.7638 | -0.7437/1.5001/5.1325 | 25.9427% |


High即时扰动较小、Margin下降较少，不等于适合Stage II训练；它也可能减少有用的增强难度或供体信息。当前是训练样本重复BS抽样，Backbone与参考Classifier已见过这些图像，不能把融合后即时判对率（CSV中correct_after）解释为测试泛化或最终训练准确率。未依Margin重新选择通道。

![融合后的几何和Margin](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/figures/05_fusion_changes.png)

## 10. 正确性、随机流与无训练证明

三视图×五组共162705条融合记录：每次16个不同索引；选中8×8通道逐位等于供体，未选逐位等于接收者；标签未变；原H2T共享mask与外部融合实现逐元素相同；GAP直接复制与真实融合后池化逐位相同。三视图每组10847条输入/标签/基础噪声哈希配对全部相同。

Backbone全部93个参数张量与93个BN缓冲区逐位不变，参考Classifier两个参数张量逐位不变，无参数梯度。原31个BN计数器保持16830。三个特征视图首batch重新提取逐位一致，主eval全部10847个GAP与历史锁定提取逐位相同；全噪声/置换/掩码重放正确。

运行期禁止Tensor/autograd.backward、autograd.grad与SGD/Adam.step；静态扫描无直接backward/step调用；没有构造Optimizer，所有计算处于no_grad。[no_training_api_calls.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/no_training_api_calls.json) 记录全部调用计数0；[correctness_verification.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/correctness_verification.json)、[feature_extraction_checks.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/feature_extraction_checks.json) 保存断言证据。

隔离随机流：seed123；BS原流、IS offset100003；augA810023 / augB820027；基础通道噪声700021；Shuffled830029；bootstrap840031。噪声复用并验证与以前SMART首epoch完全相同；Shuffled按rawID/class固定key，bootstrap按类固定key。任何额外采样不消耗增强流。

融合与CSV配对核验 166.83s，PyTorch峰值allocated 38.12MiB / reserved 80.00MiB。全部CPU/GPU审计耗时另见阶段日志和 [score_distributions.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/score_distributions.json)；这些不是训练耗时。

## 11. 原始统计与复查

独立目录 `D:\xiangmuxunlian\新\H2T_repro_audit\variation_shift_runs\variation_shift_20261013_seed123`。保存：

| 文件 | 内容 |
| --- | --- |
| [variation_shift_scores.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/variation_shift_scores.npz) | 三视图raw/reg评分、固定接收统计扰动评分、联合增强评分与raw IDs |
| [class_feature_statistics.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/class_feature_statistics.npz) | 每类n、eval/增强均值、ddof0/ddof1方差、收缩方差、目标与epsilon |
| [sampling_masks.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/sampling_masks.npz) | 四组逐样本子集/候选池、回退、基础噪声、Shuffled置换和原共享参照 |
| [stability_analysis.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/stability_analysis.csv) | 每个接收/供体关系的增强与bootstrap稳定性，区分raw/reg |
| [selected_variation_shift.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/selected_variation_shift.csv) | 162705行绝对V/B/比例、实际残差/交叉项、L2、Margin及输入哈希 |
| [fusion_statistics.npz](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/fusion_statistics.npz) | 全部三视图五组的连续几何、Margin与预测变化指标 |
| [class_statistics_diagnostics.csv](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/class_statistics_diagnostics.csv) | 100类方差范围、零方差、bootstrap与LOO风险 |
| [sampling_diagnostics.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/sampling_diagnostics.json) | 实际掩码差异、采样分布、重复key、回退和组合覆盖 |
| [full_console.log](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/full_console.log) | 完整运行日志 |
| [source_lock.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/source_lock.json) | 分析前的运行代码与配置校验 |
| [historical_verification.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/historical_verification.json) | 历史原文件逐项哈希保护 |
| [delivery_manifest.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/delivery_manifest.json) | 全部输出与源码SHA256清单 |


特征图与GAP银行在 `feature_banks/{eval,augA,augB}_maps.npy` / `*_GAP_float64.npy`；bootstrap索引与统计单独存储。六幅图同时保存PNG/SVG；任何单条报告数值均可回到CSV/NPZ复查。

复查流程命令（必须新建独立副本；当前已完成目录拒绝重复运行和覆盖）：

```powershell
$py = "D:\app\anaconda\envs\pytorch\python.exe"
$run = "D:\xiangmuxunlian\新\H2T_repro_audit\variation_shift_runs\variation_shift_20261013_seed123"
& $py "$run\build_helpers.py"
& $py "$run\audit.py" prepare
& $py "$run\lock_source.py"
& $py "$run\audit.py" run
& $py "$run\analyze_audit.py"
& $py "$run\finalize_audit.py"
```

## 12. 停止与局限

数学恒等式成立、采样按设计生效、对分类器训练有益是三个不同层面的证据。本轮只确认前两项及描述性的冻结模型反应，**没有第三项证据**。

建议下一阶段另行确认后才训练，原样锁定Top48、收缩、n<20回退、IS供体和四组逐样本机制；不据本轮结果改超参数。不建议将“High更小Margin扰动”直接升级为有效方法，也不应去掉Few回退或改成不稳定的固定Top16。

限制：单种子、两个固定增强而非完整增强分布、受原训练有限样本与头部池化先验影响、重复BS draw依赖、参考Classifier只代表一个冻结DR模型、真实训练中每draw独立增强的情况尚未验证、没有独立泛化指标。Top48稳定只能支持候选池机制，不能证明S是语义或判别重要性。

[AUDIT_COMPLETED_NO_TRAINING.json](D:/xiangmuxunlian/新/H2T_repro_audit/variation_shift_runs/variation_shift_20261013_seed123/AUDIT_COMPLETED_NO_TRAINING.json) 已锁定完成并停止；没有改原H2T、Stage I/II模型或历史报告，没有Stage II训练、供体筛选网络、验证/测试选策略或超参数搜索。
