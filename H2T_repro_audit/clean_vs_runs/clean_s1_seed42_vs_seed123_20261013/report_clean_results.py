"""Seal TRAIN-only clean results; no training, no official TEST evaluation."""
import shutil
from common import *


def link(p, label=None):
    p = Path(p)
    assert p.exists(), p
    return f'[{label or p.name}]({p.as_posix()})'


def main():
    install_test_guard('final_report')
    stop = read(ROOT/'STOP_COMPLETED.json')
    assert stop['StageII_arms'] == ARMS and stop['StageII_seeds'] == [123] and not stop['TEST_used']
    check_sources();check_history()
    s1 = read(ROOT/'clean_stage1_protocol.json')
    statistics = read(ROOT/'clean_statistics_provenance.json')
    statuses = {a:read(ROOT/'arms'/a/'status.json') for a in ARMS}
    assert all(s['state'] == 'completed' and len(s['epochs']) == 10 for s in statuses.values())
    for phase in ['prepare','stage1_dryrun','stage1_train','clean_statistics','stage2_dryrun','stage2_train']:
        g = read(ROOT/f'test_guard_{phase}.json')
        assert not g['blocked_attempts'] and g['CIFAR100_trainFalse_calls'] == 0
        assert g['official_test_image_or_label_loads'] == 0
    final = {a:s['epochs'][9]['validation'] for a,s in statuses.items()}
    best = {a:s['best_validation'] for a,s in statuses.items()}
    baseline = final[ARMS[0]]
    rows = []
    for a,s in statuses.items():
        for kind,met,ep in [('epoch10',final[a],10),('best_validation',best[a],s['best_epoch'])]:
            rows.append({'arm':a,'seed':123,'checkpoint_kind':kind,'epoch':ep,
                **{k:met[k] for k in ['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro','validation_CE']}})
    csv_write(ROOT/'clean_stage2_metrics.csv',rows)
    per_class = []
    for c in range(100):
        row = {'class_id':c,'validation_samples':baseline['class_counts'][c],
               'LT_train_samples':read(ROOT/'preflight.json')['LT_class_counts'][c],
               'in_Head':c < 36,'in_OfficialMedium':36 <= c < 71,'in_Tail':c >= 71,
               'in_Few':read(ROOT/'preflight.json')['LT_class_counts'][c] < 20}
        for a in ARMS:
            value = final[a]['per_class_accuracy'][c]
            row[a] = '' if value is None else value
            if a != ARMS[0]:
                row[a+'_minus_Uniform_pp'] = '' if value is None else value-baseline['per_class_accuracy'][c]
        per_class.append(row)
    csv_write(ROOT/'per_class_epoch10.csv',per_class)
    csv_write(ROOT/'tail_class_changes_epoch10.csv',[r for r in per_class if r['in_Tail']])
    deltas = {a:{k:final[a][k]-baseline[k] for k in ['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro']}
              for a in ARMS[1:]}
    csv_write(ROOT/'paired_epoch10_contrasts.csv',[{'arm':a,**d} for a,d in deltas.items()])
    paired = []
    for a in ARMS[1:]:
        uniform = np.load(ROOT/'arms'/ARMS[0]/'validation_epoch10.npz')
        guided = np.load(ROOT/'arms'/a/'validation_epoch10.npz')
        assert np.array_equal(uniform['labels'],guided['labels'])
        y = uniform['labels'];c0 = uniform['predictions'] == y;c1 = guided['predictions'] == y
        for group,mask in [('Overall',np.ones(len(y),dtype=bool)),('Head',y < 36),
                           ('OfficialMedium',(y >= 36)&(y < 71)),('Tail',y >= 71),('Few',y >= 70)]:
            gained = int((~c0 & c1 & mask).sum());lost = int((c0 & ~c1 & mask).sum())
            paired.append({'arm':a,'group':group,'samples':int(mask.sum()),'gained_images':gained,
                'lost_images':lost,'net_images':gained-lost,'difference_pp':100*(gained-lost)/int(mask.sum())})
    csv_write(ROOT/'paired_changed_images_epoch10.csv',paired)
    masks = np.load(ROOT/'sampling_masks.npz',allow_pickle=False)
    labels = np.load(ROOT/'train_labels.npy')
    donor = masks['donors'];receiver = masks['receivers']
    diagnostic = {'draws_each_arm':10*N,'channels_replaced':16,'strategy_granularity':'per sample',
        'IS_counts_each_epoch_exact':all(np.array_equal(np.bincount(labels[d],minlength=100),np.bincount(labels,minlength=100)) for d in donor),
        'BS_draws_per_epoch':N,'same_BS_IS_plan_all_arms':True,'same_private_online_augmentation':True,
        'same_Gumbel_base':True,'old_model_statistics_used':False,'arms':{}}
    freq_rows,composition_rows,time_rows = [],[],[]
    uniform_masks = masks['Uniform_PerSample']
    for a in ARMS:
        chosen = masks[a.replace('-','_')]
        assert chosen.shape == (10,N,16) and np.all(np.diff(np.sort(chosen,axis=-1),axis=-1) > 0)
        bool_chosen = np.eye(64,dtype=bool)[chosen].any(-2)
        bool_uniform = np.eye(64,dtype=bool)[uniform_masks].any(-2)
        common = (bool_chosen & bool_uniform).sum(-1)
        comp = np.load(ROOT/'arms'/a/'sampling_composition.npz')
        diagnostic['arms'][a] = {'16_unique_channels_all_draws':True,
            'mean_overlap_Uniform':float(common.mean()),'exact_Uniform_mask_rate':float((common == 16).mean()),
            'fallback_draws':int((comp['fallback'] > 0).sum()),
            'fallback_rate':float((comp['fallback'] > 0).mean()),
            'unique_channel_sets':int(np.unique(np.sort(chosen.reshape(-1,16),axis=1),axis=0).shape[0])}
        for k in range(64):freq_rows.append({'arm':a,'channel':k,'selected_count':int(bool_chosen[...,k].sum()),'frequency':float(bool_chosen[...,k].mean())})
        composition_rows.append({'arm':a,**{k:float(comp[k].mean()) for k in ['V','B','ratio','score','common_with_uniform']},
            'fallback_rate':diagnostic['arms'][a]['fallback_rate']})
        s = statuses[a]
        time_rows.append({'arm':a,'training_seconds':s['total_training_seconds'],'total_run_seconds':s['total_run_seconds'],
            'scoring_seconds':sum(r['timing_components']['scoring_seconds'] for r in s['epochs']),
            'overhead_vs_Uniform_seconds':s['total_training_seconds']-statuses[ARMS[0]]['total_training_seconds'],
            'peak_allocated_mib':max(r['peak_allocated_mib'] for r in s['epochs']),
            'peak_reserved_mib':max(r['peak_reserved_mib'] for r in s['epochs'])})
    assert diagnostic['IS_counts_each_epoch_exact']
    dump(ROOT/'sampling_diagnostics.json',diagnostic)
    csv_write(ROOT/'channel_selection_frequency.csv',freq_rows)
    csv_write(ROOT/'selected_variation_shift.csv',composition_rows)
    csv_write(ROOT/'timing_and_memory.csv',time_rows)
    checkpoints = []
    for a in ARMS:
        for kind in ['best','final']:
            p = ROOT/'arms'/a/f'checkpoints/model_{kind}.pth.tar'
            q = torch.load(p,map_location='cpu',weights_only=False)
            assert q['Clean_StageI_sha256'] == s1['best_checkpoint_sha256'] and not q['TEST_used']
            checkpoints.append({'arm':a,'kind':kind,'path':str(p),'epoch':q['epoch'],'sha256':sha(p)})
    csv_write(ROOT/'checkpoint_manifest.csv',checkpoints)
    historical_low_gain = 0.35353535353536
    historical_low_tail = 1.5517241379310356
    signs = {a:{k:sum(statuses[a]['epochs'][ep]['validation'][k] > statuses[ARMS[0]]['epochs'][ep]['validation'][k]
                     for ep in range(10)) for k in ['Overall','Tail','Few']} for a in ARMS[1:]}
    conclusion = {'new_clean_backbone_no_TEST_selection':True,'old_backbone_seed':123,'new_backbone_seed':42,
        'StageII_seed':123,'epoch10_deltas':deltas,'positive_epochs_out_of10_descriptive_only':signs,
        'old_Low_Overall_gain_pp':historical_low_gain,'old_Low_Tail_gain_pp':historical_low_tail,
        'new_Low_Overall_positive':deltas['Low-VS']['Overall'] > 0,
        'new_Low_Tail_positive':deltas['Low-VS']['Tail'] > 0,'new_Low_Few_positive':deltas['Low-VS']['Few'] > 0,
        'causal_attribution_to_removing_TEST_selection_not_identified':True,
        'single_clean_backbone_and_single_StageII_seed':True,'same_validation_used_repeatedly':True}
    dump(ROOT/'clean_results_summary.json',{'StageI':s1,'StageII_final':final,'StageII_best':best,
        'statistics':statistics,'sampling':diagnostic,'timing':time_rows,'judgment':conclusion})
    plots()
    final_table = '\n'.join('| '+a+' | '+' | '.join(f'{final[a][k]:.4f}' for k in ['Overall','Head','OfficialMedium','Tail','Few'])+' |' for a in ARMS)
    best_table = '\n'.join(f'| {a} | {statuses[a]["best_epoch"]} | '+' | '.join(f'{best[a][k]:.4f}' for k in ['Overall','Head','OfficialMedium','Tail','Few'])+' |' for a in ARMS)
    delta_table = '\n'.join('| '+a+' | '+' | '.join(f'{deltas[a][k]:+.4f}' for k in ['Overall','Head','OfficialMedium','Tail','Few'])+' |' for a in ARMS[1:])
    all_group_table = '\n'.join('| '+a+' | '+' | '.join(f'{final[a][k]:.4f}' for k in ['Many','Medium','Few','Macro'])+' |' for a in ARMS)
    checkpoint_table = '\n'.join(f'| {r["arm"]} / {r["kind"]} | {r["epoch"]} | {link(r["path"])} | `{r["sha256"]}` |' for r in checkpoints)
    sample_table = '\n'.join(f'| {r["arm"]} | {r["V"]:.6f} | {r["B"]:.6f} | {r["ratio"]:.6f} | {r["score"]:.6f} | {r["common_with_uniform"]:.4f} |' for r in composition_rows)
    time_table = '\n'.join(f'| {r["arm"]} | {r["training_seconds"]:.2f} | {r["overhead_vs_Uniform_seconds"]:+.2f} | {r["peak_allocated_mib"]:.2f} | {r["peak_reserved_mib"]:.2f} |' for r in time_rows)
    low = deltas['Low-VS'];high = deltas['High-VS']
    verdict_low = '复现了固定第10epoch Overall的正向趋势' if low['Overall'] > 0 else '未复现固定第10epoch Overall的正向优势'
    report = f'''# Clean Backbone 上 VS Stage II 三组控制结果

完成时间：{now()}。独立实验目录：{link(ROOT,'全部源码、日志、数组与模型')}。

## 1. 结果与协议范围

完成一个**随机初始化的 seed42 Clean Stage I（200epochs）**，基于1980张官方TRAIN留出验证选最佳；之后完成**seed123 Uniform-PerSample、High-VS、Low-VS 各10epochs**。旧StageI、原VS、SMART和H2T结果完整保留。没有加载CIFAR100(train=False)、解码官方TEST图像或标签、计算TEST accuracy、使用TEST选模。prepare只对官方test文件作1次不透明MD5校验；训练/统计/StageII的TEST文件访问拦截记录均为0。

**固定epoch10结论：** High-VS相对Uniform ΔOverall={high['Overall']:+.4f}pp、ΔTail={high['Tail']:+.4f}pp、ΔFew={high['Few']:+.4f}pp；Low-VS相对Uniform ΔOverall={low['Overall']:+.4f}pp、ΔTail={low['Tail']:+.4f}pp、ΔFew={low['Few']:+.4f}pp。本轮{verdict_low}。该结论只针对一个Clean基座与一个StageII随机流，不能称多种子稳定改善或官方测试收益。

全部准确率为**99类、1980张官方TRAIN留出验证**，class1–99各20张，class0缺失；100维输出保留。Head有效35类、OfficialMedium35类、Tail29类；Many34类、频次Medium35类、Few30类。Overall与Macro99相同。不能与论文100类TEST准确率直接比较。

## 2. Clean Stage I

新模型从随机初始化训练，未载入旧epoch198；Python/NumPy/Torch seed42，LT的官方seed0索引在独立RandomState中核对，冻结数组加载不改变训练随机流。ResNet32、CE+MixUp alpha1、SGD lr0.1/momentum0.9/weight_decay0.0002、batch128、workers0、FP32、200epochs、官方5epoch warmup和160/180下降。实际train与scheduler从复制的兼容源码AST逐字编译复用。

RNG实现细节：旧官方Dataset构造会将NumPy全局状态重置到subset seed0，进而影响后续MixUp的Beta抽样序列。本轮加载已冻结数组，训练NumPy流明确按用户指定seed42运行，LT seed0只用于独立索引核对。MixUp公式与Beta分布保持不变；旧/新运行的差异还包括这一随机流路径，不能解释为只更换epoch选择数据。

训练10847张LT，100类，max500/min5，IF100；raw IDs、标签数量和像素SHA与历史StageI逐项一致。验证TRAIN raw ID和精确图像与LT均零重叠。

- 最佳epoch：**{s1['best_epoch']}**，按验证Overall、严格变大才替换、平局最早。没有early stopping。
- 最佳验证Overall={s1['validation']['Overall']:.4f}%、Head={s1['validation']['Head']:.4f}%、Medium={s1['validation']['OfficialMedium']:.4f}%、Tail={s1['validation']['Tail']:.4f}%。
- 最佳模型：{link(s1['best_checkpoint'])}，SHA256 `{s1['best_checkpoint_sha256']}`。
- 最终模型：{link(ROOT/'stage1/checkpoints/clean_stage1_final.pth')}，SHA256 `{s1['final_checkpoint_sha256']}`。
- 17000次正式更新，loss与全部梯度有限；验证不更新BN。真实数据1batch预检查使用丢弃模型，正式模型重新初始化。
- 协议：{link(ROOT/'CLEAN_STAGE1_PROTOCOL.md')}、{link(ROOT/'clean_stage1_protocol.json')}；200epoch指标/最佳间隔记录：{link(ROOT/'stage1/epoch_validation_metrics.csv')}。

## 3. 新统计与相同 VS 规则

根据用户确认，旧统计文件保持不动，仅替换统计来源。新μ、方差和epsilon全部从**本轮Clean最佳Backbone对10847张LT的官方eval transform特征**计算；类别统计没有使用验证或TEST。统计冻结SHA256 `{statistics['statistics_sha256']}`。

保持原定义：GAP64、float64统计；类内variance_ddof0；保存ddof1诊断；pooled=sum(n×var0)/(10847−100)；lambda=5/(n+5)；v_reg=(1−lambda)var0+lambda×pooled；epsilon=1e−6×mean(global LT per-channel variance ddof1)={statistics['epsilon']:.12g}。

在线评分 `S=v_reg/(v_reg+(mu_h−f_t_current_augmented_GAP)^2+epsilon)`；Uniform全64、High Top48、Low Bottom48，使用相同Gumbel基础随机数不放回取16个。rho0.25；每次完整复制供体对应8×8通道，标签始终为BS接收者。供体n<20或评分极差≤1e−12时使用同一Uniform掩码；非有限评分停止。IS类别分布没有过滤。

核心数学函数 {link(ROOT/'reused_vs_math.py')} 与前轮逐字相同；改动diff见 {link(ROOT/'online_policy.adaptation.diff')}（独立三组接口与Clean统计路径，评分/排序/Top48/fallback规则相同）。统计来源与提取正确性：{link(ROOT/'clean_statistics_provenance.json')}。Clean最佳分类器的缓存验证预测逐项复现；StageI与StageII的cudnn设置沿用各自原配置，记录logits/CE的数值差异。

## 4. 固定第10epoch主比较

Medium在此表指原编号分组36–70；Few按LT频次<20定义。单位%。

| Method | Overall | Head | Medium（编号） | Tail | Few |
|---|---:|---:|---:|---:|---:|
{final_table}

相对同一Clean Uniform的差值，单位pp：

| Method | ΔOverall | ΔHead | ΔMedium（编号） | ΔTail | ΔFew |
|---|---:|---:|---:|---:|---:|
{delta_table}

按训练频次及Macro99：

| Method | Many (>100) | Medium (20–100) | Few (<20) | Macro99 |
|---|---:|---:|---:|---:|
{all_group_table}

## 5. 最佳验证epoch（描述性）

每组只按同一验证Overall选择，平局最早；主要结论使用固定epoch10。

| Method | Best epoch | Overall | Head | Medium（编号） | Tail | Few |
|---|---:|---:|---:|---:|---:|---:|
{best_table}

完整逐类结果：{link(ROOT/'per_class_epoch10.csv')}；全部29个Tail类的变化：{link(ROOT/'tail_class_changes_epoch10.csv')}。每类20张，一张图对应该类5pp变化，class0留空，不将缺失类当成0%平均。配对得失图像数：{link(ROOT/'paired_changed_images_epoch10.csv')}；完整epoch与指标：{link(ROOT/'clean_stage2_metrics.csv')}和各arm的epoch_validation_metrics.csv。

## 6. 采样组成、实际掩码与公平性

三组独立加载同一个Clean最佳权重及fc，空SGD state；没有加载另一组StageII模型。共850个batch/组、108470个融合样本/组。沿用原seed123的BS/IS计划与Gumbel随机流；训练增强原RandomCrop32 padding4、随机翻转和Normalize，private CPU seed123+300009，IS先增强、BS后增强。评分与掩码不消耗增强流。

每batch在SGD前核查：BS/IS索引与标签、增强图像、冻结特征、增强状态、环境CPU/CUDA随机状态、Gumbel、当前GAP、S和全部三组掩码，均与Uniform逐项SHA匹配。{link(ROOT/'pairing_verification.json')}验证每组850个配对batch。所有掩码16个唯一通道；{link(ROOT/'sampling_masks.npz')}保存全部计划与三组实际掩码。预检查还验证选中通道逐位复制供体、未选通道逐位保留接收者、标签不变和原池化/fc调用一致。

以下平均所有10epochs的实际选择；V是**收缩方差和**，B是**平方均值偏移和**，ratio=V/(V+B+epsilon)，不把该比值解释成最终分类收益。

| Method | V_selected | B_selected | variance fraction | mean selected S | mean overlap Uniform /16 |
|---|---:|---:|---:|---:|---:|
{sample_table}

掩码完全相同率、组合覆盖、fallback和逐通道频率见 {link(ROOT/'sampling_diagnostics.json')}、{link(ROOT/'channel_selection_frequency.csv')}、{link(ROOT/'selected_variation_shift.csv')}。

全部Backbone可学习参数、BN affine、running_mean/running_var/num_batches_tracked在每epoch和保存模型中逐位保持Clean最佳状态，只更新fc。各组status记录classifier每参数最大变化和BB/BN检查；{link(ROOT/'stage2_dryrun_verification.json')}验证实际全通道复制和参数更新。每组最佳/最终checkpoint重新加载，仅在同一1980张TRAIN验证缓存上复核，指标完全匹配各组checkpoint_validation.json。

## 7. 时间与显存

| Method | 正式训练秒 | 相对Uniform秒 | peak allocated MiB | peak reserved MiB |
|---|---:|---:|---:|---:|
{time_table}

各组都计算三组掩码以核对输入与评分，计时包括线上增强、特征、评分、融合和SGD；差值会包含系统负载与执行顺序波动，不能当成纯算法额外成本。{link(ROOT/'timing_and_memory.csv')}保存分项计时及峰值；StageI对应显存见其epoch表。PyTorch张量峰值不等于nvidia-smi整个GPU占用。

## 8. 判断与限制

1. **High-VS是否优于Uniform？** 固定epoch10 ΔOverall={high['Overall']:+.4f}pp，ΔTail={high['Tail']:+.4f}pp，ΔFew={high['Few']:+.4f}pp；分别按指标判断，不能只看Overall。
2. **Low-VS是否优于Uniform？** 固定epoch10 ΔOverall={low['Overall']:+.4f}pp，ΔTail={low['Tail']:+.4f}pp，ΔFew={low['Few']:+.4f}pp。
3. **之前Low优势趋势是否保留？** 之前同99类协议seed123的ΔOverall约+0.3535pp、ΔTail约+1.5517pp；本轮{verdict_low}。新基座在不使用TEST选checkpoint时仍观察到的正向差值可作为进一步支持，但单次结果不能证明过去收益完全独立于选择偏差。若优势消失，本次也无法区分基座选择、seed42初始化、验证集复用和随机波动的作用。
4. **Tail/Few是否稳定提升？** Low在10个epoch中Tail正差值{signs['Low-VS']['Tail']}/10、Few正差值{signs['Low-VS']['Few']}/10；High为Tail {signs['High-VS']['Tail']}/10、Few {signs['High-VS']['Few']}/10。这些epoch高度相关，只是轨迹描述；只有一个新基座和一个StageII seed，**没有跨种子稳定性证据**。

本轮同时改变基座训练seed（旧123→新42）和StageI epoch选择数据，因此旧/新差值不是“只移除TEST选模”的因果估计。当前三组内部的共享控制可靠，仍受固定单基座、单StageII随机流、1980张验证规模、class0缺失及历史验证集反复使用影响。“Clean”仅说明新基座和统计的训练/选模数据链没有依赖旧TEST所选模型，不能将该验证集称为新独立最终测试集，也不能提前宣称提高官方TEST或论文成绩。

## 9. Checkpoint与复现

| Arm / kind | Epoch | 文件 | SHA256 |
|---|---:|---|---|
{checkpoint_table}

全部模型、日志与统计都在独立目录。命令在该目录执行，使用保存源码和现有冻结输入；入口拒绝覆盖已完成训练，重放须另建独立实验目录。

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONHASHSEED='42'
& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -B -u prepare_clean.py
& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -B -u clean_stage1.py dryrun
& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -B -u clean_stage1.py train
& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -B -u build_clean_statistics.py
$env:PYTHONHASHSEED='123'
& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -B -u clean_stage2.py dryrun
& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -B -u clean_stage2.py train
& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -B -u report_clean_results.py
```

{link(ROOT/'source_lock.json')}、{link(ROOT/'stage2_source_lock.json')}、两套source_snapshot及diff保存源码；{link(ROOT/'preflight.json')}、{link(ROOT/'protocol.json')}保存数据与完整配置；stage1_full_console.log、stage2_full_console.log保存完整控制台日志。历史完整性：{link(ROOT/'historical_verification.json')}。没有运行Original-H2T、Shuffled-VS、SMART、其他供体训练或额外seed。

本任务完成后停止，{link(ROOT/'STOP_COMPLETED.json')}锁定本轮结束状态。
'''
    (ROOT/'CLEAN_VS_STAGE2_RESULTS.md').write_text(report,encoding='utf-8')
    for name in ['CLEAN_STAGE1_PROTOCOL.md','CLEAN_VS_STAGE2_RESULTS.md']:
        dest = WORK/name
        assert not dest.exists(), f'Refuse overwriting historical root report: {dest}'
        shutil.copy2(ROOT/name,dest)
    guard_pass();check_history()
    files = {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name != 'delivery_manifest.json'}
    dump(ROOT/'delivery_manifest.json',{'sealed_at':now(),'files':files,'file_count':len(files),
        'StageI_epochs':200,'StageI_seed':42,'StageII_seed':123,'StageII_arms':ARMS,'StageII_epochs_each':10,
        'TEST_used':False,'historical_files_unchanged':True,'completed_and_stopped':True,
        'root_reports':{n:sha(WORK/n) for n in ['CLEAN_STAGE1_PROTOCOL.md','CLEAN_VS_STAGE2_RESULTS.md']}})
    print(json.dumps({'report':str(WORK/'CLEAN_VS_STAGE2_RESULTS.md'),'StageI_best_epoch':s1['best_epoch'],
        'fixed_epoch10_Overall':{a:final[a]['Overall'] for a in ARMS},'Low_VS_delta':low,'High_VS_delta':high,
        'files_sealed':len(files),'TEST_used':False,'stopped':True},ensure_ascii=False),flush=True)


def plots():
    os.environ['MPLCONFIGDIR'] = str(ROOT/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = ROOT/'plots';out.mkdir()
    fig,ax = plt.subplots(figsize=(7,4),layout='constrained')
    s1 = read(ROOT/'stage1/status.json')
    ax.plot(np.arange(1,201),[r['validation']['Overall'] for r in s1['epochs']])
    ax.axvline(s1['best_epoch'],ls='--',color='gray',label=f'Best epoch {s1["best_epoch"]}')
    ax.set(xlabel='Stage I epoch',ylabel='99-class TRAIN holdout accuracy (%)',title='Clean Stage I · seed 42')
    ax.legend();fig.savefig(out/'clean_stage1_validation.png',dpi=180);plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for a in ARMS:
        s = read(ROOT/'arms'/a/'status.json')
        for ax,key in zip(axes,['Overall','Tail','Few']):
            ax.plot(np.arange(1,11),[r['validation'][key] for r in s['epochs']],marker='o',ms=3,label=a)
            ax.set(xlabel='Stage II epoch',ylabel='Accuracy (%)',title=key)
    axes[0].legend(fontsize=8);fig.savefig(out/'clean_stage2_validation.png',dpi=180);plt.close(fig)


if __name__ == '__main__':
    main()
