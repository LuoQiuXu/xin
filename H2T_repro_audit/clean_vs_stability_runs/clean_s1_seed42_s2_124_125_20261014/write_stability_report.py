"""Describe the locked fixed-epoch results; no training, no model selection, no TEST."""
from common import *
from analyze_stability import METRICS,rows
from stability_models import check_sources,plan_dir,outdir


def table(headers,values):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(str(v) for v in row)+'|' for row in values])


def f(x):
    return f'{x:.4f}'


def signed(x):
    return f'{x:+.4f}'


def main():
    install_test_guard('stability_report')
    check_sources()
    s = read(ROOT/'results_summary.json')
    diagnostics = read(ROOT/'sampling_diagnostics.json')
    report_path = ROOT/'CLEAN_VS_MULTI_SEED_STABILITY_REPORT.md'
    assert not report_path.exists(), 'Refuse replacing a delivered report'
    result = s['results']
    fixed,best,groups,delta,tail,timing,sampling,ck_rows = [],[],[],[],[],[],[],[]
    for seed in SEEDS:
        for arm in ARMS:
            r = result[str(seed)][arm]
            m = r['fixed_epoch10'];b = r['best_validation']
            fixed.append([seed,arm,*[f(m[k]) for k in ['Overall','Head','OfficialMedium','Tail','Few']]])
            groups.append([seed,arm,*[f(m[k]) for k in ['Many','Medium','Few','Macro']]])
            best.append([seed,arm,r['best_epoch'],*[f(b[k]) for k in ['Overall','Head','OfficialMedium','Tail','Few']]])
            for kind in ['best','final']:
                c = r['checkpoints'][kind]
                ck_rows.append([seed,arm,kind,c['epoch'],f'[{Path(c["path"]).name}]({Path(c["path"]).as_posix()})'])
        d = s['fixed_epoch10_deltas'][SEEDS.index(seed)]
        counts = result[str(seed)]['paired_change_counts']
        delta.append([seed,*[signed(d[f'delta_{k}_pp']) for k in ['Overall','Tail','Few']],
            *[counts[k]['net_correct'] for k in ['Overall','Tail','Few']]])
        t = s['tail_class_changes'][str(seed)]
        tail.append([seed,len(t['improved']),len(t['declined']),len(t['unchanged'])])
    for row in s['timing_and_memory']:
        timing.append([row['seed'],row['method'],f"{row['training_seconds']:.2f}",f"{row['run_seconds']:.2f}",
            f"{row['peak_allocated_mib']:.2f}",f"{row['peak_reserved_mib']:.2f}"])
    for row in s['composition']:
        sampling.append([row['seed'],row['method'],f(row['V']),f(row['B']),f(row['ratio']),f(row['score']),
            f(row['common_with_uniform']),f"{100*row['equal_Uniform_mask_rate']:.4f}%",row['fallback_draws']])
    timing_delta = []
    for seed in SEEDS:
        tu,tl = [next(t for t in s['timing_and_memory'] if t['seed'] == seed and t['method'] == a) for a in ARMS]
        timing_delta.append([seed,f"{tl['training_seconds']-tu['training_seconds']:+.2f}",
            f"{100*(tl['training_seconds']/tu['training_seconds']-1):+.2f}%",
            f"{tl['run_seconds']-tu['run_seconds']:+.2f}"])
    summary_rows = [[k,signed(s['new_seed_summary'][k]['mean_pp']),f(s['new_seed_summary'][k]['sample_sd_pp_ddof1']),
        f"{s['new_seed_summary'][k]['positive_seeds']}/2"] for k in ['Overall','Tail','Few']]
    positive = {k:s['new_seed_summary'][k]['positive_seeds'] == 2 for k in ['Overall','Tail','Few']}
    stable = all(positive.values())
    best_deltas = {str(seed):result[str(seed)][ARMS[1]]['best_validation']['Overall']-
        result[str(seed)][ARMS[0]]['best_validation']['Overall'] for seed in SEEDS}
    answer1 = ('是，固定第10 epoch 的 Overall 在两个新增 seed 都高于 Uniform。' if positive['Overall'] else
        '否，固定第10 epoch 的 Overall 没有在两个新增 seed 都高于 Uniform；不能称为稳定整体收益。')
    answer2 = f"Tail 为 {s['new_seed_summary']['Tail']['positive_seeds']}/2 个 seed 正增益，Few 为 {s['new_seed_summary']['Few']['positive_seeds']}/2。"
    answer2 += ('当前条件下重复了正向变化，但两个新增种子尚不足以证明广泛稳定性。' if positive['Tail'] and positive['Few'] else
        '本轮没有同时复现两个组在两个 seed 均提升，尚无稳定尾类增益的证据。')
    answer4 = ('可以把机制作为有限的后续研究线索；当前结果只支持继续提出可证伪的受控假设，不支持宣称新方法有效或有论文级泛化收益。' if stable else
        '当前证据不足以支撑以“稳定提升精度”为前提的论文方法设计。本轮应停止扩种子和调参，保留实际干预与不稳定收益的完整记录。')
    # Scientific figures read saved validation metrics only.
    os.environ['MPLCONFIGDIR'] = str(ROOT/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = ROOT/'plots';plots.mkdir()
    fig,axes = plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    for ax,key in zip(axes,['Overall','Tail','Few']):
        vals = s['new_seed_summary'][key]['values_pp']
        ax.bar(['124','125'],vals,color=['#35658b','#b56c3f'])
        ax.axhline(0,color='black',linewidth=.8)
        ax.margins(y=.3)
        ax.set_title(key);ax.set_xlabel('Stage II seed');ax.set_ylabel('Low-VS minus Uniform (pp)')
        for i,value in enumerate(vals):
            ax.annotate(f'{value:+.3f}',(i,value),xytext=(0,5 if value >= 0 else -13),textcoords='offset points',ha='center',fontsize=9)
    fig.savefig(plots/'paired_epoch10_deltas.png',dpi=180);plt.close(fig)
    class_rows = rows(ROOT/'per_class_delta.csv')
    matrix = np.array([[float(r['delta_pp']) for r in class_rows if int(r['seed']) == seed and r['id_group'] == 'Tail'] for seed in SEEDS])
    vmax = max(float(np.abs(matrix).max()),5.)
    fig,ax = plt.subplots(figsize=(12,2.6),layout='constrained')
    im = ax.imshow(matrix,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
    ax.set_yticks([0,1],['seed124','seed125']);ax.set_xticks(range(29),list(range(71,100)),fontsize=7)
    ax.set_xlabel('Tail class ID (20 validation images each)');ax.set_title('Fixed epoch10: Low-VS minus Uniform per class')
    fig.colorbar(im,ax=ax,label='Accuracy delta (pp)');fig.savefig(plots/'tail_class_deltas.png',dpi=180);plt.close(fig)
    source = read(ROOT/'source_lock.json')
    eps = read(ROOT/'protocol.json')['epsilon']
    dry = [read(plan_dir(seed)/'dryrun_verification.json') for seed in SEEDS]
    lrs = [float(r['lr']) for r in rows(outdir(SEEDS[0],ARMS[0])/'epoch_validation_metrics.csv')]
    text = [
        '# Clean Stage I 固定下 Low-VS vs Uniform 多种子稳定性验证',
        f'\n完成时间：{now()}。新增实验仅 seed124、seed125，各 Uniform-PerSample / Low-VS 10 epochs，共4次训练、3400次正式 SGD 更新。',
        '\n## 1. 实验配置与无 TEST 选择的 Clean 来源',
        '\n本轮固定已有 Clean Stage I seed42、200 epochs 中以99类 TRAIN 验证集 Overall 选出的 epoch184。没有重新训练 Stage I；四组均从该 Checkpoint 的同一 Backbone 和同一分类器权重独立初始化。dry-run 分类器及其优化器均丢弃，正式训练使用全新空 SGD 状态，未加载 seed123 或其他 Stage II 权重。',
        f'\n- 原 Clean Checkpoint：`{ORIGIN / "stage1/checkpoints/clean_stage1_best.pth"}`。\n- 本轮逐字节副本：`{ROOT / "locked/clean_stage1_best.pth"}`。\n- SHA256：`{S1_SHA}`。\n- Clean 统计文件：`{ROOT / "class_feature_statistics.npz"}`，SHA256 `{STAT_SHA}`。\n- 统计文件与 seed123 完全相同，未重算均值、方差、shrinkage 或 epsilon；epsilon=`{eps:.16g}`。',
        '\n历史 epoch198 模型曾用 TEST 选模；本轮加载的是重新训练的 Clean epoch184，没有加载旧模型或旧模型统计。Clean 的选模证据为 `clean_stage1_protocol.json`、复制的 `CLEAN_STAGE1_PROTOCOL.md` 与原 Clean 的 `test_guard_stage1_train.json`。数据划分和研究方向已有历史观察，不能把整个研究过程称为完全未受既有结果影响的盲测。',
        '\n'+table(['项目','锁定设置'],[
            ['数据 / Backbone','真实 CIFAR-100-LT IF100；10847张、100类；ResNet-32'],
            ['Stage II','10 epochs；batch128；85 batches/epoch，末batch95；每组850步'],
            ['冻结','全部 Backbone、Conv、BN affine；BN.eval；全部 running_mean/var/counter 逐位固定'],
            ['更新','只更新100类线性分类器，6500个参数；CE；无MixUp'],
            ['SGD','base_lr0.1 × factor0.2，首epoch实际0.02；momentum0.9；weight_decay0.0005；dampening0；nesterov=False'],
            ['调度','原Stage II cosine；factor0.2；同seed123，无额外warmup或搜索'],
            ['随机性','Python / NumPy / Torch / CUDA 按124或125同步；cuDNN benchmark=False、deterministic=False，沿用seed123'],
            ['采样','BS类别均衡接收者；IS供体；每epoch恰好遍历LT供体各一次；供体/标签规则不变'],
            ['融合','[B,64,8,8]；rho0.25；每样本16个无重复完整通道复制；标签保持接收者'],
            ['Uniform','64通道共享基础Gumbel流选16；均匀不放回'],
            ['Low-VS','当前增强接收特征在线评分；Bottom48后均匀不放回选16'],
            ['回退','完全沿用seed123：供体类n<20或64维评分极差<=1e-12时使用同一Uniform掩码；非有限评分立即中止']]),
        '\n评分为 `S=v_reg/[v_reg+(mu_h-current_augmented_receiver_GAP)^2+epsilon]`；`v_reg=(1-lambda)*v_ddof0+lambda*v_pooled`，`lambda=5/(n+5)`。池规模48、固定统计 epsilon、替换比例与训练预算均未调整。评分不依赖正在训练的分类器，不使用 Margin 或验证集定义分数。',
        '\n逐epoch实际学习率：`'+', '.join(f'{x:.10f}' for x in lrs)+'`。精确 YAML、`experiment_config.json`、源代码快照及源文件校验均已保存。',
        '\n## 2. 数据与指标解释',
        '\n验证集是官方 TRAIN 中未进入10847张LT训练子集的1980张原始图像，类1..99各20张，缺class0（apple）；训练/验证raw ID交集0、像素哈希交集0。100维logits始终保留，预测缺失类0仍计错。该验证集已被前期实验反复使用，因此它仍是开发验证结果，不能视为新独立测试或直接对比论文的100类官方TEST准确率。',
        '\n'+table(['指标','类ID / 训练频次定义','有效类数','验证图像'],[
            ['Overall / Macro','类1..99；每类20，因此两者相等',99,1980],
            ['Head','0..35，class0缺失',35,700],['Medium（ID）','36..70',35,700],['Tail','71..99',29,580],
            ['Many','LT训练数>100；0..34，class0缺失',34,680],['Medium-n','20<=LT训练数<=100；35..69',35,700],['Few','LT训练数<20；70..99',30,600]]),
        '\n本轮没有打开或解码官方TEST图像/标签，没有创建CIFAR100(train=False)，没有TEST指标、模型选择或评估。准备阶段仅有一次不解码文件的官方test MD5校验，其余阶段test文件打开次数为0。分离证据保存在 `test_guard_*.json`。',
        '\n## 3. 固定第10 epoch 的主要结果（%）',
        '\n'+table(['Seed','Method','Overall','Head','Medium（ID）','Tail','Few'],fixed),
        '\n'+table(['Seed','Method','Many','Medium-n','Few','Macro'],groups),
        '\n## 4. Low-VS − Uniform 的配对增益',
        '\n'+table(['Seed','ΔOverall pp','ΔTail pp','ΔFew pp','Overall净正确张数','Tail净正确张数','Few净正确张数'],delta),
        '\n'+table(['指标','新增两seed平均增益 pp','样本SD pp（ddof=1）','正增益seed'],summary_rows),
        '\n整体提升对应每个seed仅净增加5张和6张正确图像。与此同时，Head两seed均下降，平均 '+signed(s['new_seed_summary']['Head']['mean_pp'])+' pp；Many平均 '+signed(s['new_seed_summary']['Many']['mean_pp'])+' pp。该结果包含类别组之间的权衡，不能描述为所有类别普遍改善。',
        '\n这里只把两个新增 Stage II 种子作为本轮验证。n=2 的均值/样本标准差为描述统计，不构成显著性或稳定泛化证据；两个seed共用同一验证图像和同一个Stage I，不代表两个独立数据集。',
        '\n### 已观察先导 seed123（不重训，单独列示）',
        '\n'+table(['Method','Overall','Tail','Few'],[[a,*[f(s['pilot123'][a][k]) for k in ['Overall','Tail','Few']]] for a in ARMS]),
        '\n先导Δ：'+', '.join(k+' '+signed(s['pilot123_deltas'][k])+' pp' for k in ['Overall','Tail','Few'])+'。先导结果已用于提出本轮问题，不能把它与两新增种子共同当成三个预先独立确认实验。三种子合并描述另存 `results_summary.json/three_seed_descriptive_only`，主要判断只使用124/125。',
        '\n## 5. Tail 类别级分析',
        '\n'+table(['Seed','提升类数','下降类数','不变类数'],tail),
    ]
    for seed in SEEDS:
        t = s['tail_class_changes'][str(seed)]
        text += [f'\nseed{seed} Tail提升ID：`{t["improved"]}`；下降ID：`{t["declined"]}`；不变ID：`{t["unchanged"]}`。']
    text += [
        f'\n两个seed均提升ID：`{s["tail_consensus"]["improved_both"]}`；均下降ID：`{s["tail_consensus"]["declined_both"]}`；一升一降ID：`{s["tail_consensus"]["opposite_signs"]}`。',
        '\n`per_class_delta.csv` 保存2×100行，包含类名、LT样本数、验证数、两方法正确数、准确率、Δpp、净正确数；class0保留一行、准确率空白而非0。每类只有20张，1张改变即5pp；不能把个别类别变化解释为可靠类别规律。`paired_changed_images_epoch10.csv` 逐张保存预测发生改变的验证索引，未挑选有利实例。',
        '\n![固定epoch10增益]('+ (plots/'paired_epoch10_deltas.png').as_posix()+')',
        '\n![全部Tail类别的增益]('+ (plots/'tail_class_deltas.png').as_posix()+')',
        '\n## 6. 训练公平性、dry-run 与冻结核验',
        '\n每个seed的Uniform和Low-VS共享分类器初始权重、BS接收者、IS供体、batch边界、实际增强像素、冻结特征图、GAP、64维评分以及Gumbel基础噪声。Low-VS每一步更新前对照同seed Uniform 的13个控制字段，850/850 batch 全部一致；两个不同seed的采样/增强本来就不同。`pairing_hash.json` 和两组 `batches.csv` 保存逐batch证据，不仅比较种子或数据配置名称。',
        '\n私有流沿用seed123：BS Python seed；IS Torch seed+100003；训练增强 Torch seed+300009；通道Gumbel NumPy seed+700021。已先重建旧seed123全部10个epoch的BS/IS/Gumbel数组并逐位比较成功，未重新运行seed123训练。通道选择不消耗增强流；环境CPU/CUDA RNG哈希也逐batch匹配。CUDA未强制deterministic，因此不承诺跨机器/软件版本重新运行必然逐位一致，本轮实际输入和特征哈希匹配已经验证。',
        '\n'+table(['seed','真实batch dry-run','Loss Uniform / Low','峰值分配MiB','峰值保留MiB'],[[seed,'PASS',
            f"{d['arms'][ARMS[0]]['loss']:.6f} / {d['arms'][ARMS[1]]['loss']:.6f}",f"{d['peak_allocated_mib']:.2f}",f"{d['peak_reserved_mib']:.2f}"] for seed,d in zip(SEEDS,dry)]),
        '\nDry-run验证16个不重复完整通道、所选位置等于供体、未选位置等于接收者、标签不变、官方GAP/分类器调用等价、CE有限、分类器有有限梯度并更新、Backbone没有梯度。dry-run更新被丢弃。正式训练每epoch检查所有Backbone状态逐位一致；交付时再次加载8个best/final文件，对照Clean权重逐位比较全部93个Backbone参数张量（含Conv及62个BN affine张量）、93个BN缓冲张量（31组mean/var/counter）。31个counter均保持15640。分类器weight/bias均有非零变化，best/final重新加载后的同一TRAIN验证指标与保存记录完全一致。详见 `parameter_and_BN_verification.json`。',
        '\n历史文件检查：'+str(s['historical_files_unchanged'])+'个已存在文件SHA256全部不变，包含官方运行副本、Stage I/II历史权重与报告。本轮源码与运行输入分别由 `source_lock.json` 和 `source_snapshot/` 保护。',
        '\n## 7. 掩码干预与统计组成（全部10 epochs）',
        '\n'+table(['Seed','Method','均值V_selected','均值B_selected','均值V/(V+B+eps)','均值所选S','与Uniform重合通道','完全相同率','回退抽样数'],sampling),
        '\nV为所选通道的正则化供体类方差之和，B为类均值与当前增强接收GAP的平方偏移之和，逐样本比例的均值不等于均值V与均值B的比例。回退同seed123，IS类别/实例分布没有删除小类。`sampling_diagnostics.json` 从保存的在线评分和基础噪声重新构造全部掩码，逐位匹配；每个掩码16个唯一索引，且IS每epoch为10847样本的全排列。',
        '\n两新增seed均观察到Low-VS的B_selected高于Uniform、V/(V+B+epsilon)较低，平均每个掩码约3.88个所选通道发生改变。实际干预已经生效，不能由此把偏移增大解释为收益原因；这还同时改变了通道身份及其他相关特征。',
        '\n低S仅表示在上述统计定义下方差相对偏移的占比低，不等于语义上更远、有害/有益或分类重要性。组成变化与精度是否改善是两类证据。',
        '\n## 8. 最佳epoch与文件（仅描述，不替代固定epoch10）',
        '\n'+table(['Seed','Method','最佳epoch','Overall','Head','Medium（ID）','Tail','Few'],best),
        '\n最佳文件按1980张TRAIN验证集Overall严格提升保存（相等保留最早epoch），final固定epoch10。没有依据最佳epoch比较挑选本轮主结果。',
        '\n最佳模型口径下的Low−Uniform Overall分别为 seed124 '+signed(best_deltas['124'])+' pp、seed125 '+signed(best_deltas['125'])+' pp；seed125并未在这一口径占优。因此“重复正增益”的结论明确限定在预先规定的固定第10 epoch，不能扩展为任何模型选择方式下都稳定领先。',
        '\n'+table(['Seed','Method','文件','Epoch','Checkpoint'],ck_rows),
        '\n## 9. 训练耗时与显存',
        '\n'+table(['Seed','Method','训练秒','运行秒','峰值分配MiB','峰值保留MiB'],timing),
        '\n'+table(['Seed','Low−Uniform训练秒','训练耗时相对变化','Low−Uniform运行秒'],timing_delta),
        '\n训练秒是10个训练epoch累计，包含在线评分、CPU增强、配对哈希、同步及统计，不包含epoch末验证；运行秒还包括保存和验证等。Uniform也计算在线分数及候选掩码用于公平核验，因此这里不能推断无审计工具生产实现的Low-VS净开销；两个顺序运行的时间差也受系统噪声影响。',
        '\n## 10. 对四个问题的判断',
        '\n1. **Low-VS是否在两个seed均优于Uniform？** '+answer1,
        '\n2. **Tail/Few提升是否稳定？** '+answer2,
        '\n3. **是否支持“方向性扰动可能是H2T收益来源之一”？** 本轮直接检验的是在冻结特征空间中改变通道掩码后的分类器训练结果。即使Low-VS出现提升，也只说明某些通道统计组成变化可能值得研究；本轮没有无融合DR对照、没有独立控制同类/跨类供体或特征方向/扰动幅度，无法确定原H2T收益来源，也不能将B增大等同于有用的跨类别方向。结果不稳定时，该收益解释更缺少支持。',
        '\n4. **是否值得继续论文方法设计？** '+answer4,
        '\n不确定性还包括：只有一个Clean Backbone与一套固定类统计；验证每类20张且缺class0；验证集被多轮研究复用；新seed数仅2；cuDNN沿用非强制确定性设置。新seed确实改变BS/IS、增强和通道基础随机流，所以结果只度量这些Stage II随机性在该固定基础上的重复性。',
        '\n## 11. 原始数据、配置与复现命令',
        '\n实验根目录：`'+str(ROOT)+'`。关键文件：`epoch_validation_metrics.csv`（40行），`paired_epoch10_deltas.csv`，`per_class_delta.csv`（200行），`paired_epoch_deltas.csv`，`checkpoint_manifest.csv`（8行及SHA），`timing_and_memory.csv`，`results_summary.json`，`pairing_hash.json`，`sampling_diagnostics.json`，`parameter_and_BN_verification.json`。',
        '\n每个 `uniform_seed124/`、`lowvs_seed124/`、`uniform_seed125/`、`lowvs_seed125/` 包含完整 `training.log`、`batches.csv`、`effective.yaml`、`experiment_config.json`、10个epoch的验证logits/labels/predictions、全部实际掩码与组成、best/final权重及加载核验。`plans/{seed}/` 包含固定BS/IS方案、共享Gumbel、在线64维评分、配对输入和两策略全部实际掩码以及dry-run记录。',
        '\n索引：`split_indices.npz` SHA256 `'+source['data_files']['split_indices.npz']+'`；缓存验证特征SHA256 `'+source['data_files']['validation_feature_maps.npy']+'`；源图像来自 `'+str(DATA/'cifar-100-python/train')+'`，预检查完整原始索引/像素比较见 `preflight.json`。',
        '\n以下为本轮实际命令；训练入口拒绝覆盖已有目录。复现应在新的独立目标目录复制锁定源/输入后执行，不能在本目录重新训练。',
        '\n```powershell\n$env:PYTHONDONTWRITEBYTECODE="1"\n$env:PYTHONIOENCODING="utf-8"\n# cwd = 本报告的独立实验根目录；python为D:\\app\\anaconda\\envs\\pytorch\\python.exe\n# prepare_stability.py -> stability.py plans 为首次准备，已完成且禁止覆盖\n$env:PYTHONHASHSEED="124"\n& "D:\\app\\anaconda\\envs\\pytorch\\python.exe" -B -u stability.py dryrun --seed 124\n& "D:\\app\\anaconda\\envs\\pytorch\\python.exe" -B -u stability.py train --seed 124\n$env:PYTHONHASHSEED="125"\n& "D:\\app\\anaconda\\envs\\pytorch\\python.exe" -B -u stability.py dryrun --seed 125\n& "D:\\app\\anaconda\\envs\\pytorch\\python.exe" -B -u stability.py train --seed 125\n# 只读结果分析；拒绝覆盖已生成结果\n& "D:\\app\\anaconda\\envs\\pytorch\\python.exe" -B -u analyze_stability.py\n& "D:\\app\\anaconda\\envs\\pytorch\\python.exe" -B -u write_stability_report.py\n```',
        '\n**已按停止条件结束：没有seed126，没有其他策略、超参数搜索、Stage I重训或官方TEST评估。新增实验等待下一步分析。**\n',
    ]
    report_path.write_text('\n'.join(text),encoding='utf-8')
    guard_pass();check_history()
    print('REPORT WRITTEN: '+str(report_path),flush=True)
    print(answer1,flush=True);print(answer2,flush=True);print(answer4,flush=True)


if __name__ == '__main__':
    main()
