"""Failure report and finite offline diagnostics from sealed control masks; no model execution."""
from pathlib import Path
import csv,json,math,sys
import numpy as np
from preflight_checks import dump,history_manifest


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def records(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def table(headers,values):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in values])


def report(r):
    if any((r.RESULTS/f'shared_seed{s}').exists() for s in r.SEEDS):
        raise RuntimeError('Unexpected formal run directory: preserve failed run and review separately')
    output=r.RESULTS/'MASK_GRANULARITY_RESULTS.md'
    if output.exists():raise FileExistsError('Refuse replacing a previously delivered report')
    attempts=sorted((r.RESULTS/'execution').glob('preflight_*'))
    latest=next(p for p in reversed(attempts) if (p/'preflight_summary.json').is_file())
    pref=read(latest/'preflight_summary.json')
    assert not pref['passed'] and pref['mismatches']
    baseline=read(r.RESULTS/'historical_readonly_baseline.json')
    assert history_manifest(r.OLD,r.sha)==baseline,'Historical source or controls changed'
    comparison=records(latest/'pairing_preflight.csv')
    grouped=[]
    for x in comparison:grouped.append([x['seed'],x['epoch'],x['batch'],x['field'],'一致' if x['matches']=='True' else '**不一致**'])
    metrics=['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro']
    csv_rows=[];md_rows=[];pilot_rows=[];contrast_rows=[];mask_rows=[];channel_rows=[]
    all_values={}
    for seed in r.SEEDS:
        for arm in ['Uniform-BatchShared','Uniform-PerSample','Low-VS']:
            if arm=='Uniform-BatchShared':
                row={'seed':seed,'arm':arm,'status':'not_run_preflight_failed',**{k:'' for k in metrics},'best_epoch':'','best_Overall':''}
                md_rows.append([seed,arm,'未运行',*['NA']*10])
            else:
                epochs=records(r.outdir(seed,arm)/'epoch_validation_metrics.csv')
                at10=next(x for x in epochs if int(x['epoch'])==10);best=max(epochs,key=lambda x:float(x['Overall']))
                row={'seed':seed,'arm':arm,'status':'sealed_historical_control_reused',**{k:float(at10[k]) for k in metrics},
                    'best_epoch':int(best['epoch']),'best_Overall':float(best['Overall'])}
                md_rows.append([seed,arm,'历史封存',*[f'{float(at10[k]):.4f}' for k in metrics],best['epoch'],f'{float(best["Overall"]):.4f}'])
                all_values[(seed,arm)]=row
            csv_rows.append(row)
        for arm in ['Uniform-PerSample','Low-VS']:
            contrast_rows.append({'seed':seed,'comparison':arm+' - Uniform-BatchShared','status':'not_available_new_arm_not_run',
                'gain_wrong_to_right':'','loss_right_to_wrong':'','net_correct':'','Tail_net_correct':'','delta_Overall_pp':'','delta_Tail_pp':'','delta_Few_pp':''})
        # Offline design diagnostics: sharing rule applied to historical masks only.
        u=np.load(r.outdir(seed,'Uniform-PerSample')/'actual_masks.npz',allow_pickle=False)['masks']
        l=np.load(r.outdir(seed,'Low-VS')/'actual_masks.npz',allow_pickle=False)['masks']
        assert u.shape==l.shape==(10,r.N,16)
        shared=np.empty_like(u)
        for e in range(10):
            for start in range(0,r.N,r.BATCH):
                stop=min(start+r.BATCH,r.N)
                trace={'masks':{'Uniform-PerSample':u[e,start:stop],'Low-VS':l[e,start:stop]}}
                shared[e,start:stop]=r.choose('Uniform-BatchShared',trace)
        arm_masks={'Uniform-BatchShared':shared,'Uniform-PerSample':u,'Low-VS':l}
        for arm,mask in arm_masks.items():
            assert np.all(np.diff(np.sort(mask,axis=-1),axis=-1)>0)
            frequency=np.bincount(mask.ravel(),minlength=64)
            overlap_sum=identical_to_u=pair_overlap_sum=identical_pair_sum=pair_count=0
            for e in range(10):
                for start in range(0,r.N,r.BATCH):
                    stop=min(start+r.BATCH,r.N);m=mask[e,start:stop];um=u[e,start:stop];b=len(m)
                    counts=np.bincount(m.ravel(),minlength=64)
                    pairs=b*(b-1)
                    pair_overlap_sum+=int((counts*(counts-1)).sum());pair_count+=pairs
                    _,n_same=np.unique(np.sort(m,axis=-1),axis=0,return_counts=True)
                    identical_pair_sum+=int((n_same*(n_same-1)).sum())
                    indicators=np.eye(64,dtype=bool)[m].any(1);ui=np.eye(64,dtype=bool)[um].any(1)
                    overlap=(indicators&ui).sum(1)
                    overlap_sum+=int(overlap.sum());identical_to_u+=int((overlap==16).sum())
            marginals=frequency/(10*r.N)
            mask_rows.append({'seed':seed,'arm':arm,'provenance':'offline_design_from_historical_masks' if arm==r.ARM else 'sealed_historical_masks',
                'classifier_training_executed_this_round':False,'draws':10*r.N,'batches':850,
                'mean_overlap_with_Uniform':overlap_sum/(10*r.N),'identical_to_Uniform_fraction':identical_to_u/(10*r.N),
                'within_batch_pair_mean_shared_channels':pair_overlap_sum/pair_count,
                'within_batch_pair_identical_mask_fraction':identical_pair_sum/pair_count,
                'global_marginal_independence_overlap':float(np.square(marginals).sum()),
                'unique_channel_combinations':len(np.unique(np.sort(mask.reshape(-1,16),axis=1),axis=0)),
                'min_channel_selected_count':int(frequency.min()),'max_channel_selected_count':int(frequency.max())})
            channel_rows += [{'seed':seed,'arm':arm,'channel':k,'selected_count':int(v),'draws':10*r.N,'selection_fraction':float(v/(10*r.N))} for k,v in enumerate(frequency)]
    r.csv_write(r.RESULTS/'all_three_epoch10.csv',csv_rows)
    r.csv_write(r.RESULTS/'paired_differences.csv',contrast_rows)
    r.csv_write(r.RESULTS/'mask_design_diagnostics.csv',mask_rows)
    r.csv_write(r.RESULTS/'channel_selection_counts.csv',channel_rows)
    old_summary=read(r.OLD/'results_summary.json')
    for a in ['Uniform-PerSample','Low-VS']:
        pilot_rows.append([123,a,*[f'{old_summary["pilot123"][a][k]:.4f}' for k in ['Overall','Tail','Few']]])
    historical_deltas=[]
    for seed in r.SEEDS:
        u,l=[all_values[(seed,a)] for a in ['Uniform-PerSample','Low-VS']]
        historical_deltas.append([seed,*[f'{l[k]-u[k]:+.4f}' for k in ['Overall','Tail','Few']],
            f'{l["best_Overall"]-u["best_Overall"]:+.4f}'])
    mask_md=[[x['seed'],x['arm'],f'{x["mean_overlap_with_Uniform"]:.4f}',f'{100*x["identical_to_Uniform_fraction"]:.4f}%',
        f'{x["within_batch_pair_mean_shared_channels"]:.4f}',f'{100*x["within_batch_pair_identical_mask_fraction"]:.4f}%',x['unique_channel_combinations']] for x in mask_rows]
    text=['# H2T 通道掩码粒度实验：严格 preflight 未通过，正式训练未启动',
        '\n日期：2026-10-15。**本轮状态为未完成科学对照，不是零增益或负结果。两次新增分类器训练均未启动；Stage I/旧baseline重训为0，优化器更新为0，官方TEST访问为0。**',
        '\n## 1. 准确命题与协议区别',
        '\n欲识别 A：Uniform每batch共享16/64掩码，相对于Uniform每样本独立掩码的效应；B：已有Low-VS相对于Uniform-PerSample的效应。Shared由每个真实minibatch第一个样本的Uniform Gumbel16通道扩展到全batch；仅为 original-like mask granularity，不是原始torch.randperm RNG路径的完整官方复现。',
        '\n固定Clean Stage I seed42 epoch184；固定BN/Backbone；只训练分类器；10epochs、batch128、rho0.25；SGD base_lr0.1、momentum0.9、wd0.0005，调用历史原cosine调度（factor0.2，首epoch实际0.02）；固定epoch10为主，best仅敏感性分析。只允许新增Shared seed124、125；PerSample与Low-VS直接读取封存结果。',
        '\n## 2. 固定环境与依赖处理',
        '\n解释器：`'+pref['environment']['python']+'`。\n\n```json\n'+json.dumps(pref['environment'],ensure_ascii=False,indent=2)+'\n```',
        '\nGPU正常可用，磁盘空间充足，未调整PyTorch/CUDA。首次预检在导入配置时缺少yacs；检查发现PyYAML也缺少。已使用指定解释器 `-m pip install --no-deps PyYAML==6.0.3 yacs==0.1.8` 补齐这两个配置依赖，完整安装输出在 `dependency_setup_console.log`。',
        '\n## 3. 历史输入完整性与真实数据',
        '\n所有必需历史大文件均存在并通过源锁定检查：`locked/clean_stage1_best.pth`、`class_feature_statistics.npz`、`train_images.npy`、`train_labels.npy`、`validation_feature_maps.npy`、`validation_labels.npy`、`split_indices.npz`、两个seed的 `plans/{seed}/sampling_plan.npz` 和 `gumbel_noise_float64.npy`、两控制每组10个验证NPZ、850行Uniform CSV、实际掩码与状态。没有伪造、重建或重新训练历史资料。',
        '\nClean权重SHA256：`7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`；统计SHA256：`'+r.sha(r.OLD/'class_feature_statistics.npz')+'`。模型加载后的Backbone和分类器初始状态哈希与旧控制完全一致。',
        '\n'+json.dumps(pref['dataset'],ensure_ascii=False,indent=2),
        '\n已按raw TRAIN索引核对10847张训练图像、1980张验证图像与标签；重建IF100类别数量公式和官方seed0按类shuffle的LT索引完全一致；训练/验证索引交集0。验证类1..99各20张，class0缺失；保留100维输出，预测0仍计错。严禁把这些结果称为官方100类TEST性能。',
        '\n## 4. 静态审查与仅新增文件的修改',
        '\n`mask_policy.py` 的Shared定义正确，1/7/95/128样本的选择自检通过。原运行器已有本地CSV函数和TEST防护，但preflight没有实际GPU/初始状态/13字段验证，失败状态也没有落盘。新增 `preflight_checks.py` 并修改本轮运行器，补齐指定配置依赖后进行了严格无更新预检；源码原稿、修改对照与静态审查记录保存在 `source_review/`。',
        '\n已禁止历史目录的写入/创建/删除/重命名，关闭历史字节码缓存写入；每一步状态、错误栈与防护记录写入新的 `results/execution/{action_timestamp}/`。未放松任何13字段校验，未更改旧模型、采样、VS公式、Bottom48、fallback、学习率或训练预算。',
        '\n## 5. 全部失败与配对证据',
        '\n1. `preflight_001_console.log`：缺少yacs，已补齐配置依赖。\n2. `preflight_002_console.log`：seed124、epoch1、batch0的严格13字段核验不通过。错误为 `Strict historical 13-field pairing preflight failed; no training permitted`。',
        '\n'+table(['Seed','Epoch','Batch','字段','与封存Uniform比较'],grouped),
        '\n不一致项是 `features_sha256`、`online_gap_sha256`、`online_scores_sha256`。样本索引、实际增强像素、增强后RNG状态、环境CPU/CUDA RNG、基础Gumbel及两候选掩码等10个字段完全一致。配对CSV保存全部实际/期望哈希。这里只能断言没有满足逐位相同条件；哈希本身不能量化数值差异大小，更不能据此宣称掩码粒度有性能收益。',
        '\n按失败即停止规则，只检查到seed124首个实际batch；没有执行其剩余849个batch或seed125模型重放，也没有把首批候选掩码相同外推成850批全部相同。预检使用冻结模型前向，不调用backward或optimizer.step；Backbone/BN核验不变。',
        '\n## 6. epoch10与best对照表（%）',
        '\nNA表示本轮新策略未运行，不能按0或未提升解释。其余4行全部是历史封存结果，不是本轮重新训练或重新评估。',
        '\n'+table(['Seed','策略','来源','Overall','Head','Medium-ID','Tail','Many','Medium-n','Few','Macro','best epoch','best Overall'],md_rows),
        '\nMedium-ID为OfficialMedium类36..70；Head为0..35（缺0后有效35类），Tail为71..99（29类）。Many按训练数>100（34个有效类），Medium-n为20..100（35类），Few为<20（30类）。两个Medium定义不同。',
        '\n历史Low-VS−PerSample（pp）：\n\n'+table(['Seed','ΔOverall','ΔTail','ΔFew','Δbest Overall'],historical_deltas),
        '\n这些只保留了旧B线索：固定epoch10两seed为正，但Head均损失，且seed125的best口径反转。不能用它们回答A，也不能替代Shared分类器结果。',
        '\n## 7. 逐图配对、Tail得失与统计边界',
        '\nShared预测不存在，因此Shared vs PerSample、Shared vs Low-VS的wrong→right、right→wrong、净正确、Tail净变化及准确率差均不可计算。`paired_differences.csv`明示 `not_available_new_arm_not_run`，所有数值留空；没有编造翻转数量或显著性。',
        '\n1980个验证样本在两个seed中相同，不可将3960次预测当成独立样本。Overall一张图像为0.050505pp，Tail一张为0.172414pp，Few一张为0.166667pp；单类一张为5pp。Shared的均值/方向均未知，不执行预注册成功/失败判据的性能判定。',
        '\n## 8. 掩码粒度的离线设计检查',
        '\n下表仅从旧Uniform/Low-VS实际掩码进行CPU离线分析；Shared是按指定规则推导的设计掩码，**不是已执行训练的actual_masks**。未创建虚假的Shared Checkpoint或actual_masks.npz。',
        '\n'+table(['Seed','策略','与Uniform平均重合通道','与Uniform完全同掩码','batch内样本对平均共同通道','batch内样本对同掩码率','不同通道组合数'],mask_md),
        '\n统计包括每个实际batch的95/128样本边界；样本对为batch内有序不重复样本对，按样本对数量加权。Shared同batch样本的16维选择完全相同，强相关是设计属性；它不能被解释为训练收益。每通道选中次数及边际频率完整保存于 `channel_selection_counts.csv`。',
        '\n## 9. seed123历史pilot',
        '\n'+table(['Seed','策略','Overall','Tail','Few'],pilot_rows),
        '\n单独列示，不与新实验合并为n=3；本轮新增训练数量实际为0。',
        '\n## 10. 唯一下一步与停止条件',
        '\n当前不能识别Shared与PerSample的因果差异，也没有证据决定进入扰动范数匹配。唯一前置步骤是请用户与ChatGPT复核这次真实哈希差异，并决定在固定当前环境下如何建立满足配对证据的新对照协议；如需新增同环境控制，必须另行明确授权，本轮不得自作主张重跑旧baseline。',
        '\n已停止：无两次正式Shared训练、无其他seed、无Stage I/旧控制重训、无官方TEST、无超参数搜索。不要将此报告误读为“Shared无效”或“PerSample更好”。',
        '\n## 11. 文件与可复查命令',
        '\n实验根目录：`'+str(r.HERE)+'`；历史只读目录：`'+str(r.OLD)+'`。',
        '\n- `results/preflight_001_console.log`、`preflight_002_console.log`：完整stdout/错误栈。\n- `results/dependency_setup_console.log`：配置依赖补齐记录。\n- `results/execution/'+latest.name+'/pairing_preflight.csv`：13项实际/预期哈希。\n- 同目录 `preflight_summary.json`、`environment.json`、`dataset_verification.json`、`locked_controls.json`、`failure.json`、`protection.json`。\n- `results/historical_readonly_baseline.json`：历史文件哈希基线。\n- `results/all_three_epoch10.csv`：6行，2行Shared未运行占位、4行封存结果。\n- `results/paired_differences.csv`：4个Shared对比，状态明确不可计算、数值为空。\n- `results/mask_design_diagnostics.csv`、`channel_selection_counts.csv`：仅离线掩码诊断。\n- `results/STOP_PREFLIGHT_FAILED.json`：停止状态。',
        '\n```powershell\n# 项目根目录运行；保持指定Python。现有失败记录已保留。\n& "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe" -B -u H2T_repro_audit/mask_granularity_iteration/run_mask_granularity.py preflight\n& "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe" -B -u H2T_repro_audit/mask_granularity_iteration/run_mask_granularity.py report\n```',
        '\n请把本报告及preflight的CSV/summary/错误日志提供给ChatGPT复核，再确认后续协议。本轮不自动继续。\n']
    output.write_text('\n'.join(text),encoding='utf-8')
    r.guard_check()
    assert history_manifest(r.OLD,r.sha)==baseline
    dump(r.RESULTS/'STOP_PREFLIGHT_FAILED.json',{'status':'stopped_preflight_failed','formal_training_runs':0,
        'optimizer_steps':0,'old_control_retraining':0,'StageI_retraining':0,'TEST_used':False,
        'paired_batch_checked':{'seed':124,'epoch':1,'batch':0},'unmatched_fields':pref['mismatches'][0]['mismatching_fields'],
        'historical_files_unchanged':len(baseline),'report':str(output)})
    print('REPORT: '+str(output),flush=True)
    print('STOP: strict13-field preflight failed; formal training runs=0; historical files unchanged='+str(len(baseline)),flush=True)
