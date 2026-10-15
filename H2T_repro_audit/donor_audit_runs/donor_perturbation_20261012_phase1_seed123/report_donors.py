"""Build the Chinese report after verified scoring and descriptive analysis."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import re

import numpy as np

from audit_donors import ROOT,BASE,N,K,SEEDS,read,dump,sha,now,historical_hashes,checkpoint_refs
from analyze_donors import GROUPS


def link(path,label=None):
    p = Path(path).resolve()
    return f'[{label or p.name}](<{p.as_posix()}>)'


def picture(name,label):
    return f'![{label}]({(ROOT/"figures"/(name+".png")).as_posix()})'


def table(header,rows):
    def cell(value):
        return str(value).replace('|','\\|').replace('\n',' ')
    return ['| '+' | '.join(map(cell,header))+' |','| '+' | '.join(['---']+['---:']*(len(header)-1))+' |',
        *['| '+' | '.join(map(cell,row))+' |' for row in rows]]


def f(value,digits=3):
    number = Decimal(str(value)).quantize(Decimal('1').scaleb(-digits),rounding=ROUND_HALF_UP)
    return format(number,f'.{digits}f')


def main():
    assert read(ROOT/'run_status.json')['state'] == 'completed'
    assert historical_hashes() == read(ROOT/'historical_before.json')
    preflight = read(ROOT/'preflight.json')
    checks = read(ROOT/'correctness_verification.json')
    summary = read(ROOT/'analysis_summary.json')
    assert checks['state_before'] == checks['state_after']
    assert checks['all_parameters_bn_buffers_bitwise_unchanged']
    assert checkpoint_refs() == (preflight['stage1_backbone_checkpoint'],preflight['reference_classifier_checkpoint'])
    assert sha(ROOT/'audit_donors.py') == preflight['script_sha256']
    main_stats = summary['analyses']['eval_preprocess']
    aug_stats = summary['analyses']['fixed_train_aug']
    sampling = summary['sampling']
    overall,tail = main_stats['groups']['All'],main_stats['groups']['Tail']
    seeds = '; '.join(f'{key}={value}' for key,value in SEEDS.items())
    lines = ['# H2T 供体判别扰动机制审计（第一阶段）','',
        f'完成时间：{now()}。独立审计目录：{link(ROOT,ROOT.name)}。','',
        '## 1. 回答本轮问题','',
        '**是：在这两个已锁定模型、训练子集、相同接收特征、相同候选供体类别、相同融合通道及 rho=0.25 下，不同候选供体产生了有实际判别含义的扰动差异。**','',
        f'主分析四候选 D 极差中位数为 **{overall["D_range"]["p50"]:.3f}**，95 分位为 **{overall["D_range"]["p95"]:.3f}**（logit units）；作为尺度参考，原始 |Margin| 中位数为 {overall["baseline_abs_margin"]["p50"]:.3f}。',
        f'**{overall["mixed_fused_correctness_groups"]["count"]:,}/{N:,} 组（{overall["mixed_fused_correctness_groups"]["percent"]:.2f}%）**在同一接收样本上，四个候选使冻结参考分类器出现部分判对、部分判错；Tail 为 **{tail["mixed_fused_correctness_groups"]["count"]:,}/{tail["receiver_groups"]:,} 组（{tail["mixed_fused_correctness_groups"]["percent"]:.2f}%）**。这是决策事件的差异，没有用人为选定的 D 阈值定义“明显”。','',
        '本轮结论支持进一步研究候选供体的判别差异，**没有证明 Low-D、Mid-D 或 High-D 的训练效果更好，也没有把正 D 或大 D 等同于有害融合**。没有训练任何新模型，没有选择或部署供体策略。','',
        '## 2. 实际模型、数据、特征及控制协议','',
        '### 锁定 Checkpoint','',
        '- 特征提取：Stage I epoch 198 最佳 Checkpoint（历史 Overall 40.49%），'+link(preflight['stage1_backbone_checkpoint']['path'],'Stage I 文件')+'。',
        '- SHA-256：`'+preflight['stage1_backbone_checkpoint']['sha256']+'`。',
        '- 参考分类器：固定 BN 的 DR 第 10 epoch Checkpoint，'+link(preflight['reference_classifier_checkpoint']['path'],'固定 BN DR epoch 10 文件')+'；仅取其中的 Classifier 权重。',
        '- SHA-256：`'+preflight['reference_classifier_checkpoint']['sha256']+'`。',
        '- ResNet-32 `resnet32_fe()`：[5,5,5] BasicBlocks，feature_tuning=False；输出最后一层 ReLU 后 **[B,64,8,8]** 特征图。',
        '- 使用原始 `Classifier(feat_in=64,num_classes=100)`：`F.avg_pool2d(x,x.size()[3])` → flatten [B,64] → `fc`；没有在审计中重新定义分类器或用别的投影代替。',
        '- 实例化并严格加载 DataParallel 模型，所有模块 `eval()`、参数 requires_grad=False、全程 no_grad；从未创建 optimizer、调用 backward 或训练。参考模型始终冻结。',
        '- 冻结 DR Checkpoint 的整个 Backbone state_dict 与 Stage I 逐位一致，加载时再次核验；两者的 BN counters 均为 16830。','',
        '### 数据与预处理','',
        '- 只实例化 CIFAR-100 的 train=True 长尾子集：IF=100，100 类，10,847 个实例，每类 500→5。子集像素字节 SHA-256 与此前复现记录一致。训练索引在本报告中均为重排后 LT 子集索引，**不是 CIFAR 原始 50,000 张序号**。',
        '- 未实例化测试 dataset，未解码或用于分析测试图像/标签；历史测试表现未参与候选构造、排序或参数选择。torchvision 的数据完整性检查可能读取测试文件字节校验 MD5，这不涉及测试样本评分或策略选择。',
        '- **主分析**：对训练子集使用官方 eval_transform（ToTensor + 原始 Normalize），不作随机裁剪/翻转，以隔离供体身份差异。',
        '- **敏感性核验**：每个训练实例使用一次固定 seed=129 的官方 train_transform（RandomCrop(32,padding=4) + RandomHorizontalFlip + ToTensor + 原始 Normalize）；固定缓存后沿用同一接收者、同一候选、同一掩码重新评分。',
        '- 两套 transform 表达式从未改动的官方 `CIFAR100_LT.__init__` AST 中直接提取并实例化。Normalize 仍为均值 (0.4914,0.4822,0.4465)、标准差 (0.2023,0.1994,0.2010)。',
        '- 特征图以 float32 保存，每套完整银行 [10847,64,8,8]；同一实例在一套银行内复用相同增强/特征，避免重复候选因不同增强而产生额外差异。这与实际 Stage II 每次抽样都重新增强的在线训练不同。','',
        '### 抽样与掩码','',
        '- 抽取 **20,000 个 BS 接收样本组**：Python Random 按官方顺序均匀抽类别，再在该类内随机抽实例（有放回）。20,000 是抽样次数，覆盖 '+str(sampling['unique_receiver_instances'])+' 个不同训练实例。',
        '- 每组先按 `P(c_h=c)=n_c/10847` 抽一个 IS 供体类别，再类内均匀、有放回抽 K=4 个供体实例。同组四个供体属于同一类别。允许重复、同类和自供体，全部保留。',
        '- 每次调用仍为 **原始 methods.H2T**，rho=0.25；64 通道随机排列取前 16，对接收特征做原地赋值，保留其余 48 通道，融合后标签保持接收者标签。没有改变融合函数签名或传入新选择规则。',
        '- 为保留官方函数的 batch 共享掩码，32 个接收组 × K4 合成 B=128 调用原始函数，一批 32 组共用一份 M；因此每套数据有 625 份实际随机掩码，而不是 20,000 份独立掩码。每组内 K4 完全共享通道，满足控制要求。',
        '- 通道 CPU RNG 单独用 seed=126，并在 fork_rng 中调用原始 H2T；用独立 predictor 核验原始 randperm 的 16 个通道及 RNG 消耗，两套预处理均重置同一通道流。环境数据 RNG 不被 H2T 扰动。',
        '- 固定随机流：`'+seeds+'`；PYTHONHASHSEED=123。后端配置：`'+str(checks['backend_settings'])+'`。','',
        '**与原始 BS+IS 在线过程的区别必须保留在解释中**：IS 的类别边缘概率相同，但原仓库 IS 来自每 epoch 打乱的实例遍历；本审计独立、有放回抽类别，并一次在类内构造四个候选，再对冻结缓存评分。类内候选联合选择方式、跨 batch 的遍历过程和增强复用均改变。随机从 K4 选一个的供体边缘分布仍是 IS；Low/Mid/High 条件排序后的类内实例分布可能改变。本审计不是原始训练轨迹的重放。','',
        '## 3. 正确性与保护核验','',
        '- 独立 runtime 复制已审计的 48 个文件，与原副本和历史 manifest 的 SHA-256 一致。核心源码新增修改为 0；沿用已批准 Windows 兼容性补丁。原 commit 标记 `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8` 来自原审计中的官方 archive。',
        '- Feature forward / 直接原始池化+fc 与 Classifier forward 的 logits **逐位相同**；B=128 的同一接收特征复用四份，融合前四个 Margin 一致，避免不同 batch shape 造成数值差异。四候选由相同基准特征独立复制后一起融合，不会在同一接收特征上累积四次原地替换。',
        '- 主分析与增强核验各 625 个融合 batch：每次检查 16 替换通道等于供体，48 保留通道等于接收者，供体及标签不变、环境 RNG 不变。1250 次原始 H2T 调用全部通过。',
        '- 重复供体索引在相同缓存和掩码下的 D 完全相同。'+str(sampling['self_donor_candidates']['count'])+' 个自供体候选的 D 与特征 L2 均恰为 0；额外身份融合检查亦为 0，未用数值噪声制造差异。',
        '- 同时验证 D=(真实类 logit 下降)+(最大竞争类 logit 上升)；最大 float32 代数误差：主分析 '+f(checks['scoring']['eval_preprocess']['margin_component_identity_max_float32_error'],9)+'，增强核验 '+f(checks['scoring']['fixed_train_aug']['margin_component_identity_max_float32_error'],9)+'。',
        '- Backbone 和参考 Classifier 的完整 state_dict 字节哈希前后相同，涵盖 Conv、BN affine、running_mean、running_var、num_batches_tracked 和分类器全部权重。',
        f'- {checks["historical_files_unchanged"]} 个历史文件（原始/固定 BN 实验的源码、日志、报告、Checkpoint 等）前后 SHA-256 一致；Stage I 和参考 DR Checkpoint 单独复核一致。',
        '- 证据：'+link(ROOT/'correctness_verification.json')+'；'+link(ROOT/'historical_before.json')+' / '+link(ROOT/'historical_after.json')+'。','',
        '## 4. K4 构造与采样分布验证','',
        f'BS 均匀类别分布：χ²={sampling["bs_chi_square"]:.3f}，df=99，p={sampling["bs_chi_square_p"]:.4f}；IS 的 n_c/10847 分布：χ²={sampling["is_chi_square"]:.3f}，df=99，p={sampling["is_chi_square_p"]:.4f}。类别频数与声明分布没有明显失配证据；这不等于证明有限抽样频数完全一致。概率来自实际子集 n_c，且逐候选验证标签确属抽取类别。','',
        *table(['接收组','BS 接收次数','不同接收实例','IS 供体类别次数'],[
            [group,sampling['receiver_group_counts'][group],main_stats['groups'][group]['unique_receiver_instances'],sampling['donor_group_counts'][group]]
            for group in ['Head','Medium','Tail']]),'',
        'Head=[0,36)，Medium=[36,71)，Tail=[71,100)。这里的供体次数为抽类别次数，每次对应 4 个候选；Tail 的 '+str(tail['receiver_groups'])+' 次接收只覆盖 '+str(tail['unique_receiver_instances'])+' 个不同训练实例，不能当作同等数量的独立图像。','',
        '625 个掩码中每通道出现 '+str(min(sampling['channel_selection_counts']))+'–'+str(max(sampling['channel_selection_counts']))+' 次，均匀抽样期望为 625×16/64=156.25；完整掩码及频数保存，没有按通道表现筛选。','',
        picture('01_sampling_distributions','BS 与 IS 类别分布验证'),'',
        '## 5. Margin 扰动总体与类别组分布','',
        '定义 m=z_y−max_{j≠y}z_j，D=m_before−m_after；D>0 为真实类别 Margin 下降，D<0 为上升，D=0 为不变。竞争类别在计算前明确排除 y。单位是锁定分类器的 logits，换分类器或权重尺度后不可直接跨模型比较绝对 D。','',
        *table(['接收组','候选数','D 均值','D P05','D P50','D P95','D>0 (%)','D<0 (%)','D=0 (%)'],[
            [g,main_stats['groups'][g]['candidate_rows'],f(main_stats['groups'][g]['D']['mean']),f(main_stats['groups'][g]['D']['p05']),
             f(main_stats['groups'][g]['D']['p50']),f(main_stats['groups'][g]['D']['p95']),
             f(main_stats['groups'][g]['D_positive']['percent'],2),f(main_stats['groups'][g]['D_negative']['percent'],2),f(main_stats['groups'][g]['D_exact_zero']['percent'],3)] for g in GROUPS]),'',
        *table(['接收组','Margin 前 P50','Margin 后 P50','前 |Margin| P50','前参考分类器判对 (%)'],[
            [g,f(main_stats['groups'][g]['baseline_margin']['p50']),f(main_stats['groups'][g]['fused_margin']['p50']),
             f(main_stats['groups'][g]['baseline_abs_margin']['p50']),f(main_stats['groups'][g]['baseline_correct']['percent'],2)] for g in GROUPS]),'',
        '融合前 Margin 每个接收组计一次；融合后四候选全部纳入。上表判对率为训练子集上的冻结参考分类器诊断值，非测试准确率，不能解释成泛化性能。主分析 D 的全部连续分位数、均值、标准差及极值保存在 analysis_summary.json。','',
        '## 6. 同一接收样本内部的候选差异','',
        *table(['接收组','极差 P05','极差 P25','极差 P50','极差 P75','极差 P95','极差 P99','候选 D_std P50'],[
            [g,*[f(main_stats['groups'][g]['D_range'][q]) for q in ['p05','p25','p50','p75','p95','p99']],f(main_stats['groups'][g]['D_std']['p50'])] for g in GROUPS]),'',
        *table(['接收组','极差>0 组 (%)','同时含 D<0 与 D>0 (%)','候选判对/判错混合组 (%)','候选预测类别不同组 (%)'],[
            [g,*[f(main_stats['groups'][g][key]['percent'],2) for key in ['nonzero_range_groups','mixed_D_sign_groups',
                'mixed_fused_correctness_groups','different_fused_predictions_groups']]] for g in GROUPS]),'',
        '主分析极差最小 '+f(overall['D_range']['min'],6)+'，最大 '+f(overall['D_range']['max'],6)+'；所有 20,000 组极差非零，但“非零”本身不作为有意义的唯一证据。候选组内标准差按 ddof=0 计算，描述这四个候选的离散程度，不将其当作总体标准差的无偏估计。',
        '每组保存 D_min、D_max、极差、标准差及完整四候选排序；并列使用原候选位置稳定排序，排序只用于描述。没有给 D 极差设定一个人为通过阈值。判对/判错混合以及 D 跨 0 使用的是原定义中的决策事件，具有直接解释。','',
        '**Tail 并非极差最大的一组**：其极差中位数低于 Head，但候选判对/判错混合比例更高。现有证据说明 Tail 也存在候选判别差异，不能据此推断差异仅发生于 Tail 或对 Tail 的训练收益最大。','',
        picture('02_margin_and_within_receiver_range','D 分布及同一接收样本候选极差 ECDF'),'',
        '## 7. 相似度、L2 距离与 D 的关系','',
        '主定义：余弦相似度为接收/供体完整 [64,8,8] 展平后的 cosine；L2 为融合前后完整特征图差值的欧氏范数。同时报告分类器所见 GAP 后 [64] 向量的 cosine / L2，避免把空间差异与分类器输入差异混为一谈。','',
        *table(['度量 × D（All）','Pearson','Spearman','组内去均值 Pearson'],[
            [metric,*[f(overall['correlations'][metric][key]) for key in ['pearson','spearman','within_receiver_centered_pearson']]]
            for metric in ['cosine_spatial','cosine_gap','l2_spatial','l2_gap']]),'',
        *table(['接收组','空间 cosine Pearson','空间 L2 Pearson','GAP cosine Pearson','GAP L2 Pearson'],[
            [g,*[f(main_stats['groups'][g]['correlations'][metric]['pearson']) for metric in ['cosine_spatial','l2_spatial','cosine_gap','l2_gap']]] for g in ['Head','Medium','Tail']]),'',
        '组内去均值 Pearson 先在每个 K4 组内减去该指标和 D 的各自均值，再合并计算，用于区分候选组内变化与不同接收者之间的变化。相关性为描述性结果，**不代表因果关系，也不能直接用 cosine 或 L2 替代 Margin 评分**。两者均由相同特征及掩码决定，可能存在数学耦合。重复接收实例、重复候选和共享 batch 掩码使行之间有关联，未把 80,000 行当作独立样本报告相关性显著性 p 值。','',
        picture('03_similarity_and_distance_vs_D','相似度和融合 L2 与 D 的关系'),'',
        '### 相似度接近但 D 不同的候选','',
        '对每组 6 个候选对，选择**空间 cosine 差最小的一对**，不依据 D 来选，也不用人工绝对相似度阈值。以下两列为同一批选中候选对的边缘统计，不暗示它们的中位数来自同一个候选对。','',
        *table(['接收组','最近对 |Δcos| P50','最近对 |ΔD| P50','最近对 |ΔD| P95'],[
            [g,f(main_stats['groups'][g]['nearest_cosine_pair_abs_cosine_difference']['p50'],5),
             f(main_stats['groups'][g]['nearest_cosine_pair_abs_D_difference']['p50']),
             f(main_stats['groups'][g]['nearest_cosine_pair_abs_D_difference']['p95'])] for g in GROUPS]),'',
        '因此相近的余弦值并不保证相近的判别 Margin 扰动。全部 120,000 个组内候选对按 |Δcos| 排成 10 个等数量区间后，也报告每区间 |ΔD| 连续分布（下图及 JSON），没有只保留一个有利阈值区间。','',
        picture('04_similar_cosine_different_D','同组候选对的余弦差与判别扰动差'),'',
        '## 8. 决策边界与最强竞争类别变化','',
        '“跨越参考分类器决策边界”主定义为 argmax 预测类别与融合前不同；另外区分正确→错误、错误→正确和真实类别 Margin 的符号跨越，后两类条件率使用各自融合前状态的候选数作分母。最强竞争类别是排除 y 后的 argmax，可能在整体预测仍正确时变化。','',
        *table(['接收组','预测类别变化 (%)','正确→错误 条件率 (%)','错误→正确 条件率 (%)','最强竞争类变化 (%)'],[
            [g,*[f(main_stats['groups'][g][key]['percent'],2) for key in ['prediction_boundary_cross','correct_to_incorrect','incorrect_to_correct','strongest_competitor_changed']]] for g in GROUPS]),'',
        f'All：预测类别变化 {overall["prediction_boundary_cross"]["count"]:,}/80,000；正确→错误 {overall["correct_to_incorrect"]["count"]:,}/{overall["correct_to_incorrect"]["denominator"]:,}；错误→正确 {overall["incorrect_to_correct"]["count"]:,}/{overall["incorrect_to_correct"]["denominator"]:,}；最强竞争类变化 {overall["strongest_competitor_changed"]["count"]:,}/80,000。',
        '这些是融合输入上的即时参考分类器行为。合成特征仍按接收者标签解释；模型误判与后续训练是否有益是不同的问题，本轮没有估计训练或泛化收益。完整 margin 符号跨越率、零 margin 数及分母均在 JSON 中保留。','',
        picture('06_decisions_and_augmentation_sensitivity','决策事件与预处理敏感性'),'',
        '## 9. 原始随机选择与 Low / Mid / High D 的描述性比较','',
        'Random 为每组用独立 RNG 在四候选位置中均匀选一个（位置计数 '+str(sampling['random_candidate_position_counts'])+'）；Low-D 为原始 D 最小的候选，High-D 为最大；Mid-D 在 D 升序第 2 或第 3 名中均匀选择，**不是最接近零的 D**。重复得分按稳定位置排序。',
        'Random 的供体边缘分布与本审计声明的 IS 相同，因此刻画原始随机融合抽样思想下的扰动；其类别/实例遍历节奏和冻结缓存仍与在线 Stage II 不同，不能声称这里重放了原始训练的每个 batch。','',
        *table(['选择（All）','D 均值','D P05','D P50','D P95','融合后参考判对 (%)','预测类别变化 (%)'],[
            [name,*[f(overall['choices'][name]['D'][key]) for key in ['mean','p05','p50','p95']],
             f(overall['choices'][name]['fused_correct']['percent'],2),f(overall['choices'][name]['boundary_cross']['percent'],2)] for name in ['Random','Low-D','Mid-D','High-D']]),'',
        *table(['选择（Tail）','D 均值','D P05','D P50','D P95'],[
            [name,*[f(tail['choices'][name]['D'][key]) for key in ['mean','p05','p50','p95']]] for name in ['Random','Low-D','Mid-D','High-D']]),'',
        'Low/Mid/High 的 D 分布顺序由排序定义产生，其参考判对率差异也受所用标签 Margin 直接影响，不能当作独立训练验证。特别是较大正 D 可能只是使冻结分类器面对更难的训练输入；较小或负 D 也不自动代表更好的增强。没有预设中等扰动最优。','',
        picture('05_random_low_mid_high_D','Random 与三种 D 排序选择的描述性分布'),'',
        '## 10. 重复候选、同类别和自供体','',
        *table(['项目','计数 / 分母','比例 (%)'],[
            ['存在重复候选的接收组',f'{sampling["duplicate_candidate_groups"]["count"]}/20000',f(sampling['duplicate_candidate_groups']['percent'],3)],
            ['超出不同实例数的重复候选位置',f'{sampling["repeated_candidate_slots"]["count"]}/80000',f(sampling['repeated_candidate_slots']['percent'],4)],
            ['供体类别=接收类别的组',f'{sampling["same_class_groups"]["count"]}/20000',f(sampling['same_class_groups']['percent'],3)],
            ['包含自供体的组',f'{sampling["self_donor_groups"]["count"]}/20000',f(sampling['self_donor_groups']['percent'],3)],
            ['自供体候选',f'{sampling["self_donor_candidates"]["count"]}/80000',f(sampling['self_donor_candidates']['percent'],5)]]),'',
        f'按实际 IS 类别概率和类内有放回采样公式，重复组率理论期望为 {sampling["expected_duplicate_group_probability_percent"]:.4f}%；实际 {sampling["duplicate_candidate_groups"]["percent"]:.4f}%。每组不同实例数直方图（1/2/3/4）为 '+str(sampling['candidate_unique_counts_histogram'])+'。本配置最少 5 个实例；代码允许更小类别有放回构造 K4，不会因不足 K 而排除。','',
        *table(['同类情况','接收组数','D 均值','D P50','极差 P50','预测类别变化 (%)'],[
            [name,main_stats['same_class_analysis'][name]['receiver_groups'],f(main_stats['same_class_analysis'][name]['D']['mean']),
             f(main_stats['same_class_analysis'][name]['D']['p50']),f(main_stats['same_class_analysis'][name]['D_range']['p50']),
             f(main_stats['same_class_analysis'][name]['boundary_cross']['percent'],2)] for name in ['same_class','different_class']]),'',
        '同类融合 Head / Medium / Tail 接收组数为 '+str([sampling['same_class_by_receiver_group'][g]['count'] for g in ['Head','Medium','Tail']])+'；Tail 同类只有 1 组，不能作稳定的 Tail 同类结论。所有同类和重复候选均包含在总体统计内，并单独标记，未删除。','',
        '## 11. 典型候选差异实例','',
        '示例按 Head / Medium / Tail 各自 **D 极差排序的 10%、50%、90% 位置**机械选取（9 组），未按正负 D、是否判对或某个候选“有利”选择。实例用于说明机制；全部总体统计才是结论依据。候选位置与排名均为 0-based 位置、1-based 名次，索引对应缓存 LT 子集。','',
        *table(['组 / 极差位置','接收组 ID','接收 / 供体类别','四个 D（原位置）','极差','D 升序位置'],[
            [f'{e["receiver_group"]} q{int(e["range_quantile_target"]*100)}',e['group_id'],f'{e["receiver_class"]} / {e["donor_class"]}',
             ', '.join(f(v) for v in e['D']),f(e['D_range']),str(e['ascending_D_order_zero_based'])] for e in main_stats['examples']]),'',
        '例如 Tail 的中位极差示例中，重复供体保持同分，同时另一个候选使 Margin 变为负值；图中也包含四候选都判对、都判错、Margin 上升等情况，展示不一致的扰动方向，而非选择一种结果证明策略有效。每例最近 cosine 候选对的 |Δcos| / |ΔD|、完整候选索引与 logits-derived Margin 存在 analysis_summary.json。','',
        picture('07_systematic_candidate_examples','按极差分位位置机械选择的九组供体候选实例'),'',
        '## 12. 固定官方训练增强的敏感性核验','',
        '接收索引、K4 供体索引、供体类别、Random/Mid 选择流和 625 份 M 均不变，仅将训练图像的预处理换为固定缓存的官方训练增强。不是另一个训练实验。','',
        *table(['预处理 / 接收组','D 均值','极差 P50','极差 P95','候选判对/判错混合 (%)'],[
            [f'{name} / {group}',f(collection['groups'][group]['D']['mean']),f(collection['groups'][group]['D_range']['p50']),
             f(collection['groups'][group]['D_range']['p95']),f(collection['groups'][group]['mixed_fused_correctness_groups']['percent'],2)]
            for name,collection in [('eval 主分析',main_stats),('固定 train augmentation',aug_stats)] for group in ['All','Tail']]),'',
        '两模式逐候选 D 的 Pearson='+f(summary['augmentation_sensitivity']['D_pearson_between_preprocessing_modes'])+'，逐组极差的 Pearson='+f(summary['augmentation_sensitivity']['range_pearson_between_preprocessing_modes'])+'。Low-D 被选候选**位置**一致率 '+f(summary['augmentation_sensitivity']['low_D_candidate_identity_agreement']['percent'],2)+'%，High-D 为 '+f(summary['augmentation_sensitivity']['high_D_candidate_identity_agreement']['percent'],2)+'%。','',
        '差异存在的总体证据在这次增强核验中仍成立，但具体候选排序明显依赖增强。没有证明相同实例在所有增强、所有随机通道或所有分类器下都有稳定的 Low/High 身份。','',
        '## 13. 是否值得第二阶段训练，以及尚不能归因的内容','',
        '**值得在获得下一步确认后检验 Low-D / Mid-D / High-D 的训练效果。**依据是连续极差的尺度、组内判别决策变化、增强核验中的差异延续和相近 cosine 候选仍有不同 D，说明候选排序具有可研究的空间。','',
        '这些证据只证明“供体不同会产生不同的冻结参考分类器扰动”。它们没有给出供体好坏的因果标签：参考分类器在该训练子集上训练过，训练 Margin 具有拟合偏差；对训练合成输入的即时判对/判错不是未来泛化效果；本轮只有一个抽样种子组合、一个通道流和一个固定增强缓存。重复接收与 batch 共用掩码也限制独立样本解释。',
        '后续若获授权，公平对照应包含同样 K4 构造下的 Random-of-4，控制 BS/IS 类别、接收者增强、候选增强、通道掩码、BN 条件、训练预算及参考分类器冻结方式，并报告多随机种子的变化。Low/Mid/High 均应作为待检验条件，不提前指定最优方向。当前没有启动这些训练，也没有修改官方 H2T。','',
        '## 14. 可复查产物、命令和资源','',
        '- 实际评分脚本：'+link(ROOT/'audit_donors.py')+'；统计脚本：'+link(ROOT/'analyze_donors.py')+'；绘图：'+link(ROOT/'plot_donors.py')+'；报告：'+link(ROOT/'report_donors.py')+'。',
        '- 全部抽样索引及 Random/Mid 位置：'+link(ROOT/'sampling_plan.npz')+'；16 通道掩码：'+link(ROOT/'channel_masks.npy')+'；每类分布：'+link(ROOT/'sampling_distribution.csv')+'。',
        '- 主分析原始候选：'+link(ROOT/'eval_preprocess_candidates.csv')+' / '+link(ROOT/'eval_preprocess_statistics.npz')+'；每组极差和排序：'+link(ROOT/'eval_preprocess_receiver_groups.csv')+' / '+link(ROOT/'eval_preprocess_ranked_groups.npz')+'。',
        '- 增强核验原始候选：'+link(ROOT/'fixed_train_aug_candidates.csv')+' / '+link(ROOT/'fixed_train_aug_statistics.npz')+'；每组排序：'+link(ROOT/'fixed_train_aug_receiver_groups.csv')+'。',
        '- 完整统计（全部分位数、相关性、分母、示例、分布验证）：'+link(ROOT/'analysis_summary.json')+'；完整 feature maps：'+link(ROOT/'eval_preprocess_feature_maps.npy')+' / '+link(ROOT/'fixed_train_aug_feature_maps.npy')+'。',
        '- 图表：7 组 PNG + SVG 位于 '+link(ROOT/'figures','图表目录')+'；评分日志：'+link(ROOT/'extraction_scoring.log')+'。',
        '- 配置及软件版本：'+link(ROOT/'preflight.json')+' / '+link(ROOT/'effective_source_config.yaml')+'。YAML 为加载的源配置，其中 Stage II 训练超参数没有执行；实际审计数量、预处理、K4、模型引用和 RNG 流以 preflight 为准。','',
        f'GPU：{preflight["gpu"]}；PyTorch {preflight["torch"]} / torchvision {preflight["torchvision"]} / CUDA {preflight["cuda"]}。两套提取、评分及结尾完整性核验耗时 {checks["elapsed_seconds"]:.2f} 秒；PyTorch peak allocated {checks["cuda_peak_allocated_mib"]:.2f} MiB，peak reserved {checks["cuda_peak_reserved_mib"]:.2f} MiB。这是进程 allocator 的峰值，不是含桌面的全 GPU 显存。','',
        '已执行命令（PowerShell；工作目录为本工作区）：','',
        '```powershell',"$env:PYTHONIOENCODING = 'utf-8'","$env:PYTHONHASHSEED = '123'",f"$auditRoot = '{ROOT}'",
        "& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -u \"$auditRoot\\audit_donors.py\" prepare",
        "& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -u \"$auditRoot\\audit_donors.py\" run",
        "& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -u \"$auditRoot\\analyze_donors.py\"",
        "& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -u \"$auditRoot\\plot_donors.py\"",
        "& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -u \"$auditRoot\\report_donors.py\"",'```','',
        '评分启动器拒绝覆盖已有运行和根目录报告。独立复查可利用保存的 feature maps、抽样计划、M、Classifier 权重重算；若复制脚本到新目录重新提取，应同时将复制脚本中的报告输出名及报告存在性保护改为新名称，保留本次记录。这只涉及独立审计脚本的输出保护，不涉及 H2T 核心源码。','',
        '**完成状态：第一阶段特征提取、K4 构造、Margin/距离计算、统计与图表全部完成。已停止，没有启动 Low-D / Mid-D / High-D 训练或任何其他改进实验。**','']
    report = '\n'.join(lines)
    for target in re.findall(r'\]\(<([^>]+)>\)',report):
        assert Path(target).exists(),target
    for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report):
        assert Path(target).exists(),target
    (BASE.parent/'H2T_DONOR_PERTURBATION_AUDIT.md').write_text(report,encoding='utf-8')
    (ROOT/'H2T_DONOR_PERTURBATION_AUDIT.md').write_text(report,encoding='utf-8')
    assert historical_hashes() == read(ROOT/'historical_before.json')
    files = {p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
        for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name != 'delivery_manifest.json'}
    dump(ROOT/'delivery_manifest.json',{'completed_at':now(),'files':files,
        'root_report_sha256':sha(BASE.parent/'H2T_DONOR_PERTURBATION_AUDIT.md'),
        'historical_files_unchanged':checks['historical_files_unchanged'],'training_performed':False})
    print(f'REPORT COMPLETE: {BASE.parent/"H2T_DONOR_PERTURBATION_AUDIT.md"}; {len(files)} artifact files checksummed; no training',flush=True)


if __name__ == '__main__':
    main()
