"""Chinese scientific interpretation of recorded results, including authorization and limits."""
from pathlib import Path
import json
from datetime import datetime,timezone,timedelta


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])


def f(value):return f'{value:.4f}'


def signed(value):return f'{value:+.4f}'


def write(r,d):
    out=r.RESULTS/'MASK_GRANULARITY_RESULTS.md'
    assert not out.exists()
    primary=[[x['seed'],x['arm'],*[f(x[k]) for k in ['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro']],
        x['best_epoch'],f(x['best_Overall'])] for x in d['rows']]
    paired=[[x['seed'],x['comparison'],x['gain_wrong_to_right'],x['loss_right_to_wrong'],x['net_correct'],
        signed(x['delta_Overall_pp']),x['Tail_gain'],x['Tail_loss'],x['Tail_net_correct'],signed(x['delta_Tail_pp']),
        x['Few_net_correct'],signed(x['delta_Few_pp'])] for x in d['paired']]
    summary=[[x['comparison'],x['metric'],signed(x['mean_delta_pp']),f(x['sample_sd_pp_ddof1']),
        x['positive_seeds'],x['negative_seeds'],'是' if x['opposite_signs'] else '否']
        for x in d['seed_summary'] if x['metric'] in ['Overall','Tail','Few']]
    best=[[x['seed'],x['comparison'],signed(x['delta_Overall_pp']),signed(x['delta_best_Overall_pp']),
        '**反转**' if x['best_reversal'] else '未发生正负反转'] for x in d['paired']]
    fields=[[key,d['pairing']['124'][key],d['pairing']['125'][key]] for key in r.PAIR_KEYS]
    mask=[[x['seed'],x['arm'],f(x['mean_overlap_with_Uniform']),f(100*x['identical_to_Uniform_fraction'])+'%',
        f(x['within_batch_pair_mean_shared_channels']),f(100*x['within_batch_pair_identical_mask_fraction'])+'%',
        x['unique_combinations'],str(x['min_channel_selected_count'])+'..'+str(x['max_channel_selected_count'])] for x in d['mask_diagnostics']]
    comp=[[x['seed'],x['arm'],f(x['V']),f(x['B']),f(x['ratio']),f(x['score']),
        f(x['GAP_L2']) if x['GAP_L2']!='' else '未存档',f(x['feature_map_L2']) if x['feature_map_L2']!='' else '未存档'] for x in d['composition']]
    times=[[x['seed'],f(x['training_seconds']),f(x['run_seconds']),f(x['peak_allocated_mib']),f(x['peak_reserved_mib'])] for x in d['timing']]
    a=[x for x in d['paired'] if x['comparison']=='Uniform-PerSample - Uniform-BatchShared']
    b=[x for x in d['paired'] if x['comparison']=='Low-VS - Uniform-PerSample']
    statements=[]
    for key in ['Overall','Tail','Few']:
        na=[x[key+'_net_correct'] for x in a]
        nb=[x[key+'_net_correct'] for x in b]
        if all(v<0 for v in na):
            result='两个seed均为Shared高于PerSample；观察方向符合预注册判据(3)，不支持“逐样本随机更好”在该指标上的预测。'
        elif all(v>0 for v in na) and all(v>0 for v in nb):
            result='两个seed均为PerSample高于Shared且Low-VS高于PerSample；描述方向符合判据(1)，但仍受本轮非严格历史数值配对限制。'
        elif min(na)<0<max(na):
            result='PerSample−Shared跨seed变号，粒度效应没有稳定方向，须如实保留。'
        elif all(v==0 for v in na):
            result='两seed的Shared与PerSample均没有净正确差；本轮未预设等效界限，不能据此证明等效。'
        else:
            result='Shared与PerSample没有在两个seed形成严格同方向优势；不能挑选单seed认定粒度效应。'
        statements.append(f'- **{key}**：PerSample−Shared净正确张数（124、125）为 `{na}`；Low-VS−PerSample为 `{nb}`。'+result)
    pilot=[[123,arm,*[f(d['pilot123'][arm][k]) for k in ['Overall','Tail','Few']]] for arm in ['Uniform-PerSample','Low-VS']]
    all_fail=[]
    for p in sorted((r.RESULTS/'execution').glob('preflight_*')):
        if (p/'failure.json').is_file():
            q=json.loads((p/'failure.json').read_text(encoding='utf-8'))
            all_fail.append([p.name,q['traceback'].strip().splitlines()[-1],str((p/'failure.json').relative_to(r.HERE))])
    ck=[[x['seed'],x['kind'],x['epoch'],f'[{Path(x["path"]).name}]({Path(x["path"]).as_posix()})',x['sha256']] for x in d['checkpoints']]
    text=['# H2T 通道掩码粒度：三臂结果与配对限制',
        '\n完成时间：'+datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')+'。本轮只新增Uniform-BatchShared seed124、125，共两次10-epoch训练、1700次分类器更新；旧Uniform-PerSample和Low-VS均直接复用封存结果。',
        '\n## 1. 准确科学命题',
        '\nA：batch共用一个Uniform16/64掩码，与每样本独立Uniform16/64的训练结果是否不同？B：已有Low-VS相对Uniform-PerSample的差异是否仍存在？Shared采用每个真实minibatch第一个样本的Uniform Gumbel16通道，扩展至全batch。这是 **original-like mask granularity**；没有模拟原H2T的torch.randperm RNG路径，不能称完整官方原始复现。',
        '\n## 2. 协议及用户后续授权',
        '\n- CIFAR-100-LT IF100，10847张TRAIN；ResNet-32输出[B,64,8,8]。\n- Clean Stage I seed42、epoch184，SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。\n- 同一初始Backbone/分类器、同一BS接收者/IS供体方案、同一增强流和Gumbel；rho0.25，每样本复制16个完整通道；接收标签不变。\n- Backbone及BN affine冻结；BN eval，全部running_mean/var/counter逐位固定；仅6500参数分类器SGD、CE。\n- 10epochs、batch128、85batch/epoch（末batch95）。base_lr0.1、momentum0.9、wd0.0005、nesterovFalse；沿用原cosine factor0.2，首epoch实际lr0.02。固定epoch10为主，best严格按验证Overall提升、相等保留最早epoch。',
        '\n预注册最初要求13字段逐位一致。preflight发现特征、GAP和评分哈希不一致后曾停止。用户随后明确授权：“继续执行，不管失败条件了，缺什么环境在我们的指定python环境里自己pip install”。据此解除历史哈希开跑阻断，并保留全部13字段逐batch比较和原始差异，**没有把不一致记作一致**。授权记录为 `results/continuation_authorization.json`。',
        '\n解释器固定为 `D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe`；PyTorch2.14.1+cu132、CUDA13.2。只额外安装PyYAML6.0.3和yacs0.1.8两个配置依赖（--no-deps），没有更改指定PyTorch/CUDA。本轮使用新目录，未覆盖/重跑历史实验。',
        '\n因此，本轮Shared与旧控制比较带有已知数值执行差异，不能宣称除掩码粒度外所有计算结果逐位相同。数值哈希不一致本身也不能量化误差大小。结果用于描述与提出后续假设，不作为完全消除计算差异的因果证明。',
        '\n## 3. 真实数据与泄漏防护',
        '\n使用官方CIFAR-100原始TRAIN，按锁定raw indices验证LT图像与标签，重新按官方seed0类内shuffle及 `int(500*0.01**(c/99))` 重建LT索引完全一致。原TRAIN留出1980张、类1..99各20张；class0缺失；训练/验证raw ID交集0，图像与原TRAIN索引逐项一致。100维logits保留，预测0仍计错。',
        '\n数据/权重/统计/采样方案/Gumbel/旧控制CSV、验证NPZ和实际掩码都实际存在，并通过锁定文件校验。类别统计SHA256 `'+r.sha(r.OLD/'class_feature_statistics.npz')+'`，未重新估计。',
        '\n'+table(['指标','定义','有效类数','验证张数'],[
            ['Overall/Macro','class1..99，每类20',99,1980],['Head','ID0..35，缺0',35,700],
            ['OfficialMedium / Medium-ID','ID36..70',35,700],['Tail','ID71..99',29,580],
            ['Many','LT训练数>100，缺0',34,680],['Medium / Medium-n','LT训练数20..100',35,700],['Few','LT训练数<20',30,600]]),
        '\nOfficialMedium与频次Medium不同。本轮没有加载官方TEST图像/标签、没有CIFAR100(train=False)、TEST指标或TEST选模。评估使用封存的TRAIN验证特征，同一1980张图像在两个seed重复出现，不能将3960次预测当独立观测；该验证集曾用于前期开发和Clean选模，不能称盲测或与官方100类TEST准确率直接对比。',
        '\n## 4. 固定epoch10及best（%）',
        '\n'+table(['Seed','策略','Overall','Head','Medium-ID','Tail','Many','Medium-n','Few','Macro','best epoch','best Overall'],primary),
        '\nUniform-PerSample/Low-VS四行是2026-10-14封存结果，未在本轮环境重新训练或重算预测；Shared两行是本轮训练。新best/final均重新加载，TRAIN验证指标与所存记录完全一致。',
        '\n## 5. 逐图配对、Tail/Few得失',
        '\n差值方向统一为“表中方法 − Shared”或明确注明Low−PerSample。每个对比均从相同1980张验证样本的预测逐图计算；净正确数÷图像数×100与各组准确率差逐项严格一致。',
        '\n'+table(['Seed','对比','wrong→right','right→wrong','净正确','ΔOverall pp','Tail增','Tail失','Tail净','ΔTail pp','Few净','ΔFew pp'],paired),
        '\nOverall一张图像=0.050505pp；Tail一张=0.172414pp；Few一张=0.166667pp；单类别一张=5pp。少数几张图像的变化不夸大为可靠类别规律。完整600行类别对比在 `per_class_paired_differences.csv`，含class0空指标。',
        '\n### 两个seed的均值和方向（描述统计）',
        '\n'+table(['对比','指标','平均Δpp','样本SD ddof1','正向seed数','负向seed数','跨seed变号'],summary),
        '\nn=2只作描述，不宣称显著性；没有以3960预测作独立样本检验。',
        '\n### Tail类别完整变化',
    ]
    for key,value in d['tail_classes'].items():
        text.append('\n'+key+'：提升ID `'+str(value['improved'])+'`；下降ID `'+str(value['declined'])+'`；不变ID `'+str(value['unchanged'])+'`。')
    text += [
        '\n## 6. best敏感性、Head损失及预注册判据',
        '\n'+table(['Seed','对比','epoch10 ΔOverall pp','best ΔOverall pp','口径反转'],best),
        '\n'+ '\n'.join(statements),
        '\n所有Head/Many损失均保留在 `paired_differences.csv`。即使某组Tail/Few方向一致，也不能据此宣称所有类别全面获益；best口径的反转和固定epoch10并列报告。没有事后调整Top48、epsilon、rho或预算。',
        '\n## 7. 旧控制13字段与模型状态审计',
        '\n下表为850批中完全相同的批次数，分母均850；不是“已全部通过”。',
        '\n'+table(['字段','seed124一致批数','seed125一致批数'],fields),
        '\n所有新Shared实际掩码在850/850批与“封存Uniform首样本掩码向全batch扩展”逐位一致；16个通道无重复，所选完整8×8复制供体，未选位置保持接收，标签不变。`all_two_masks_sha256`同时包含当前Low-VS候选，所以它的不一致不等于Shared未按规定选掩码；Shared未使用该Low-VS候选。',
        '\nBN/Backbone在每epoch及best/final检查全部186状态条目逐位不变，包含93个参数张量、31组running_mean/var/counter；counter均15640。分类器weight/bias均有非零变化，所有1700步损失与分类器梯度有限。原实验 '+str(d['historical_files_unchanged'])+' 个受保护文件内容未变。详情为 `parameter_BN_verification.json`、`pairing_hash.json` 和每组 `control_mismatches.csv`。',
        '\n## 8. 掩码重合、选择次数与跨样本相关性',
        '\n'+table(['Seed','策略','与PerSample平均重合通道','与PerSample全同率','同batch样本对平均共同通道','同batch样本对全同率','不同组合数','单通道选择次数范围'],mask),
        '\n使用每个实际95/128样本batch，样本对为不同样本的有序对并按对数加权；完整64通道频率见 `channel_selection_counts.csv`。Shared的同batch掩码完全相同，产生结构性的跨样本相关性；每个样本仍是Uniform16/64。独立样本策略通常约4个共同通道。不能仅用边际通道频率声称随机化粒度等价。',
        '\n### 迁移组成与实际幅度记录',
        '\n'+table(['Seed','策略','均值V','均值B','均值V/(V+B+eps)','均值所选S','GAP L2','特征图L2'],comp),
        '\nV、B来自固定类统计与该次训练使用的增强接收特征，逐样本比例取均值。旧控制没有存档实际GAP/特征图L2，不能拿新Shared的L2与空缺值比较。本轮未做范数匹配；现有证据不能认定“选择策略只改变范数”，也不能把即时幅度差当作最终分类收益原因。',
        '\n## 9. 环境、耗时、异常及交付',
        '\n'+table(['Seed','训练累计秒','总运行秒','峰值分配MiB','峰值保留MiB'],times),
        '\n耗时包含增强、特征提取、掩码核验、配对记录和分类器更新；总运行另含保存/验证。显存为PyTorch分配器统计，不是整机GPU所有进程占用。',
        '\n'+table(['失败attempt','原因 / 最后错误','完整错误栈路径'],all_fail),
        '\n两次正式训练均正常完成，没有清理或覆写失败训练目录。preflight失败记录保留，后续继续依据用户明确授权；授权后没有跳过冻结、合法掩码、有限损失或防覆盖检查。完整stdout见 `results/preflight_001_console.log`、`preflight_002_console.log`、`dependency_setup_console.log`、`train_seed124_console.log`、`train_seed125_console.log`、`report_console.log`；各执行状态/防护记录位于 `results/execution/`。',
        '\n### best/final Checkpoint',
        '\n'+table(['Seed','文件','Epoch','路径','SHA256'],ck),
        '\n### seed123 独立历史pilot',
        '\n'+table(['Seed','策略','Overall','Tail','Few'],pilot),
        '\n不与124/125合成n=3同协议统计，也没有重跑seed123。',
        '\n## 10. 下一轮唯一建议与停止条件',
        '\n**唯一优先建议：在当前指定环境中，另行授权一次Shared/PerSample/Low-VS三臂同环境、同特征数值路径的配对复核，以确认粒度方向。** 本轮A比较混入了已披露的历史特征/评分数值差异，优先完成这一步，再决定是否值得进入norm-matched消融。这里没有建议新增网络、更多种子或超参数搜索；本轮也不执行该复核。',
        '\n停止条件：本轮两次Shared训练、保存结果与上述报告完成后停止；不再训练其他seed、不重新训练Stage I或旧baseline、不评估官方TEST。后续是否开展同环境复核或幅度匹配由用户确认。请将本报告连同CSV、13字段差异和执行日志上传/提供给ChatGPT复核。',
        '\n## 11. 原始文件与实际命令',
        '\n新目录：`'+str(r.HERE)+'`。历史只读目录：`'+str(r.OLD)+'`。',
        '\n关键输出均在 `results/`：`all_three_epoch10.csv`、`paired_differences.csv`、`paired_seed_summary.csv`、`per_class_paired_differences.csv`、`mask_diagnostics.csv`、`channel_selection_counts.csv`、`selected_variation_shift.csv`、`timing_and_memory.csv`、`checkpoint_manifest.csv`、`scientific_summary.json`、`STOP_COMPLETED.json`。`shared_seed124/`和`shared_seed125/`各含全部实际掩码、850行batch记录与差异、10个验证NPZ、best/final权重、完整配置/摘要/日志。',
        '\n```powershell\n# cwd为项目根目录；以下为本次实际执行入口，已存在输出时训练器拒绝覆盖。\n& "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe" -B -u H2T_repro_audit/mask_granularity_iteration/run_mask_granularity.py preflight\n& "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe" -B -u H2T_repro_audit/mask_granularity_iteration/run_mask_granularity.py train --seed 124\n& "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe" -B -u H2T_repro_audit/mask_granularity_iteration/run_mask_granularity.py train --seed 125\n& "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe" -B -u H2T_repro_audit/mask_granularity_iteration/run_mask_granularity.py report\n```',
        '\n本轮没有执行任何版本控制或仓库同步操作。\n',
    ]
    if (r.RESULTS/'finalization_failure_001.json').is_file():
        text.append('\n## 补充：最终封存异常记录\n\n封存检查首次误读停止状态的字段名new_updates，实际字段为updates_total，产生KeyError。已改为核对原有正确字段；该错误发生在写入快照/封存文件之前，没有影响训练、Checkpoint或评估，没有重复训练。原始异常及修复记录见 `results/finalization_failure_001.json`。\n')
    out.write_text('\n'.join(text),encoding='utf-8')
