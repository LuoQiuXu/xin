"""Chinese report of the fixed ten-run experiment."""
from datetime import datetime,timezone,timedelta

def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])

def f(x):return f'{x:.4f}'
def sg(x):return f'{x:+.4f}'

def write(r,d):
    rows=d['rows'];pairs=d['paired'];g=d['gate_decision']
    def selected(base):return [x for x in pairs if x['comparison']=='RAVS-V1 - '+base]
    primary=[[x['seed'],x['arm'],*[f(x[k]) for k in ['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro']],x['best_epoch'],f(x['best_Overall'])] for x in rows]
    critical=[x for x in pairs if x['comparison'].startswith('RAVS-V1 - ')]
    pairtable=[[x['seed'],x['comparison'],sg(x['delta_Overall_pp']),sg(x['delta_Head_pp']),sg(x['delta_Tail_pp']),sg(x['delta_Few_pp']),x['Overall_wins'],x['Overall_losses'],x['Overall_net_correct'],x['Tail_net_correct'],x['Head_net_correct']] for x in critical]
    low=selected('Low-VS');freq=selected('Frequency-Only')
    text=['# RAVS-H2T V1 第一轮正式方法原型：科学报告',
        '\n完成时间：'+datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        '\n**结果状态：'+d['status']+'；本轮十组均重新训练，共100 epochs、8500次分类器更新。**',
        '\n## 1. 准确算法与研究问题',
        '\n固定原正则化VS分数：S=v_reg/(v_reg+(mu_donor-f_receiver_GAP)^2+epsilon)。沿用锁定类统计、shrinkage和epsilon，不重新估计。RAVS只调整随机候选池大小，不改变供体、融合比例或替换数量。',
        '\n每步更新前，对当前分类器的未融合接收特征计算clean logits；margin=l_y-max(other logits)，h=sigmoid(-margin/(std(logits)+1e-8))。logits detach/no_grad，转float64，std使用总体标准差ddof=0。原实现保留标准化margin的[-20,20]数值保护，未调参。',
        '\nq=clip(log(500/n_y)/log(100),0,1)；RAVS的omega=q*(0.5+0.5*h)，Frequency-Only固定h=0.5，因此omega=0.75*q。K=clip(floor(64-16*omega+0.5),48,64)，从原VS BottomK用同一Gumbel64均匀不放回选16个完整8×8通道。供体类n<20或评分平坦回退Uniform；非有限评分终止。',
        '\n目的：相比固定Low-VS是否减少Head牺牲、提高Tail；相比Frequency-Only，困难度是否增加训练收益。没有额外网络、辅助损失、供体筛选或自适应rho。']

    text += ['\n## 2. 五组控制与固定协议',
        '\nNo-Fusion直接使用接收特征，但仍执行配对供体增强/forward。Uniform-PerSample和Low-VS直接使用原online_policy掩码；Frequency-Only与RAVS共享门控形式，仅h不同。五组均在本次指定Python环境重新独立初始化训练，不引用任何旧时期准确率作对照。',
        '\nClean Stage I seed42 epoch184 SHA256：`'+d['source_preflight']['checkpoint_sha256']+'`；统计SHA256：`'+d['source_preflight']['statistics_sha256']+'`。',
        '\nResNet-32；BS receiver+IS donor；batch128（每epoch85批，末批95）；10epochs；6500参数的100类线性分类器；CE硬标签；SGD base_lr0.1、momentum0.9、wd0.0005、nesterovFalse，原cosine factor0.2，首epoch实际lr0.02。四组融合均固定16/64通道。',
        '\n全部backbone、Conv与BN affine冻结，BN eval，running_mean/var/counter逐位不变。运行器预声明所有组cuDNN deterministic=True、benchmark=False，cuDNN及matmul TF32=False；配置已如实记录。effective.yaml的原base路径不决定实际保存位置，实际为results/seed<SEED>/<ARM>。',
        '\n## 3. 数据协议与解释边界',
        '\n真实CIFAR-100-LT IF100，10847张TRAIN；锁定原raw索引，预检训练/验证ID交集0。既有TRAIN留出1980张：类1..99各20张，class0缺失，保留100 logits，预测0计错。没有官方TEST图像、标签或TEST选模。',
        '\nHead=ID0..35（有效35类/700张）；OfficialMedium=36..70（35/700）；Tail=71..99（29/580）。Many按LT数>100（有效34/680）；频次Medium为20..100（35/700）；Few为<20（30/600）。两个Medium定义不同；均衡验证使Overall等于99类Macro。',
        '\n验证集已多轮使用并参与Clean Stage I选模，不是独立最终测试。两个seed使用相同1980张图像，不能把3960预测当独立观测。这里只调整线性分类器，不能声称Backbone表征得到训练改善；不是完整PI-H2T/PIF复现。',
        '\n## 4. 固定epoch10与best（%）',
        '\n'+table(['Seed','策略','Overall','Head','OfficialMedium','Tail','Many','Medium-n','Few','Macro','best epoch','best Overall'],primary),
        '\nbest仅为敏感性分析，按验证Overall严格提升保存，相等保留最早epoch；主要判断始终固定epoch10。',
        '\n## 5. RAVS对所有控制的逐图配对',
        '\n'+table(['Seed','对比','ΔOverall pp','ΔHead pp','ΔTail pp','ΔFew pp','增正确','失正确','净正确','Tail净','Head净'],pairtable),
        '\n全部20个无重复两两比较和所有类别组净得失保存在paired_epoch10.csv；每个净正确/组样本数×100与准确率差严格一致。Overall一张=0.050505pp，Head/OfficialMedium一张=0.142857pp，Tail一张=0.172414pp，Few一张=0.166667pp，单类别一张=5pp。']

    small=[x for x in critical if x['small_net_1_to_3_images']]
    text.append('\n净收益/损失仅1～3张的RAVS比较：'+('; '.join(str(x['seed'])+' '+x['comparison']+' 净'+str(x['Overall_net_correct'])+'张' for x in small) if small else '无')+'。这些不作稳定泛化证据。')
    best=[[x['seed'],x['comparison'],sg(x['delta_Overall_pp']),sg(x['delta_best_Overall_pp']),'反转' if x['best_order_reversal'] else '未正负反转'] for x in critical]
    text += ['\n## 6. best口径、Head牺牲与尾类变化',
        '\n'+table(['Seed','RAVS对比','epoch10 ΔOverall pp','best ΔOverall pp','排序'],best)]
    for base,values in [('Low-VS',low),('Frequency-Only',freq)]:
        text.append('\nRAVS−'+base+'：'+('; '.join('seed'+str(x['seed'])+' Head '+sg(x['delta_Head_pp'])+'pp（净'+str(x['Head_net_correct'])+'张），Tail '+sg(x['delta_Tail_pp'])+'pp（净'+str(x['Tail_net_correct'])+'张）' for x in values))+'。')
    for seed in r.SEEDS:
        for base in ['Low-VS','Frequency-Only']:
            t=d['tail_class_changes'][str(seed)+'/RAVS-V1 - '+base]
            text.append('\nseed'+str(seed)+' RAVS−'+base+' Tail提升类ID '+str(t['improved_ids'])+'；下降类ID '+str(t['declined_ids'])+'。每类仅20张，见paired_per_class.csv完整保留所有类，缺class0项为空。')
    text += ['\n## 7. 预先承诺的Go/No-Go',
        '\n门槛1：两个seed的RAVS−Low-VS Overall均非负，平均Overall至少+0.15pp，平均Tail、Head均非负。',
        '\n门槛2：RAVS−Frequency-Only方向基本一致：Overall路线要求两个seed均非负且平均至少+0.10pp；Tail路线要求两个seed的Tail均非负、平均Overall不降低、平均Tail至少+0.35pp。该对“方向一致”的执行定义已在训练前存档；两路线取或。',
        '\n'+table(['指标','相对Low-VS平均Δpp','相对Frequency平均Δpp'],[[k,sg(g['gate1']['mean_delta_pp'][k]),sg(g['gate2']['mean_delta_pp'][k])] for k in ['Overall','Head','Tail','Few']]),
        '\n门槛1 **'+('PASS' if g['gate1']['passed'] else 'FAIL')+'**；逐条件：`'+str(g['gate1']['conditions'])+'`。',
        '\n门槛2 **'+('PASS' if g['gate2']['passed'] else 'FAIL')+'**；Overall路线：`'+str(g['gate2']['Overall_route'])+'`；Tail路线：`'+str(g['gate2']['Tail_route'])+'`。',
        '\n**最终判定：'+('GO：仅允许提出下一轮独立验证申请，不代表论文已成立。' if g['go'] else 'NO-GO：本版没有同时达到两道探索门槛，应停止当前V1规则，不调公式或候选池、不追加seed。')+'**',
        '\n门槛不等于统计显著性。所有阈值用未四舍五入准确率判断，两个Stage II种子只代表固定一套Stage I上的重复性；同一开发验证集反复使用会带来研究选择偏差。']

    keys=['indices_sha256','images_sha256','features_sha256','scores_sha256','gumbel_sha256']
    matches=[[name,*[counts[k] for k in keys]] for name,counts in d['five_field_matching'].items()]
    text += ['\n## 8. 同环境配对与训练正确性',
        '\n'+table(['seed/组','indices','增强图像','冻结特征','VS评分','Gumbel'],matches),
        '\n分母均850。五组均执行相同donor增强与forward，clean logits从当前分类器更新前取值且无梯度；RAVS门控不依赖验证预测或正在反传的计算图。梯度只来自融合/原特征的原CE。']
    if not d['fully_matched']:
        text.append('\n**COMPARABILITY_WARNING：有组未达到850/850同环境配对，不能称严格单因素因果比较；仅呈现差异并停止。**')
    else:
        text.append('\n五项签名全部850/850相同；没有使用上一轮历史哈希豁免。本轮严格控制输入、冻结特征、评分与噪声，策略之外没有跨环境对照混用。')
    text += ['\n每epoch检查186个backbone状态条目逐位不变（93参数张量与93 BN缓冲）；31组BN counter固定15640。分类器weight/bias均变化；所有8500步损失和分类器梯度有限。best/final重新加载后各指标与对应epoch存档差小于1e-9。Checkpoint只保存分类头及锁定Stage I哈希，推理需配合该冻结Backbone。',
        '\n历史只读校验通过，'+str(d['historical_files_unchanged'])+'个原实验文件内容未变；TEST访问及历史写入事件均为0。',
        '\n## 9. 门控诊断与额外耗时',
        '\n'+table(['Seed','组','平均h','平均omega','实际K均值','K范围','与Uniform掩码相同比例'],[[x['seed'],x['arm'],f(x['mean_hardness']) if x['mean_hardness'] is not None else 'NA',f(x['mean_omega']) if x['mean_omega'] is not None else 'NA',f(x['K_mean']) if x['K_mean'] is not None else 'NA',str(x['K_min'])+'..'+str(x['K_max']) if x['K_min'] is not None else 'NA',f(x['same_Uniform_mask_fraction']) if x['same_Uniform_mask_fraction'] is not None else 'NA'] for x in d['gate_diagnostics']]),
        '\nFrequency-Only的h诊断严格为0.5；其他对照的h仅是在线观察，不作门控。Low-VS的K=48、回退64；No-Fusion没有通道候选池。candidate_pool_histogram.csv保存全部K48..64实际计数。难度数值有变化不等于它包含有用训练信息，需由RAVS对Frequency对照检验。',
        '\n'+table(['Seed','组','训练秒','含验证保存总秒','峰值分配MiB','峰值保留MiB'],[[x['seed'],x['arm'],f(x['training_seconds']),f(x['elapsed_seconds']),f(x['peak_allocated_mib']),f(x['peak_reserved_mib'])] for x in d['gate_diagnostics']]),
        '\n所有组都计算clean logits与原VS评分以维持计算/随机流一致，时间包括CPU增强、记录和核验，因此不能把顺序运行的时间差当作生产实现的纯门控开销。PyTorch显存统计不代表全机GPU总占用。',
        '\n## 10. 训练失误、最小修复与强弱点',
        '\nCPU策略单测和零更新preflight均PASS；十组按指定顺序各执行一次，没有失败后静默重跑或覆盖。修复仅涉及本轮目录：Frequency-Only h诊断改为实际0.5、Low-VS池大小诊断改为实际48/回退64；补齐冻结/有限CE预检、当前运行配置、日志/签名/计时及历史写保护。选择公式、Gumbel、VS统计、CE与预算未改变，详细对照为minimal_fixes.patch。',
        '\n强点：包含No-Fusion、Uniform、Low-VS及频率门控，当前环境全部重训；每seed严格850批配对；只训练一个6500参数分类头。弱点：仅两个Stage II种子与一个Stage I；反复使用的99类开发验证集；门控反馈来自同一正在训练的分类器，并不保证困难样本适合更强VS限制；没有独立泛化测试。']

    if g['go']:
        next_step='唯一建议：另行预注册新的独立验证数据上的同环境五组确认实验，锁定公式后再运行；当前结果只达到探索门槛，不继续在同一留出集调参。'
    elif g['gate1']['passed'] and not g['gate2']['passed']:
        next_step='唯一建议：用户先审阅频率条件门控的科学假说；目前只支持讨论类别条件门控潜力，不认定困难度模块成功，也不再训练当前RAVS规则。'
    else:
        next_step='唯一建议：先复核本轮逐图得失并由用户重新确定科学假说；当前RAVS规则停止，不继续搜索K、公式或新seed。'
    text += ['\n## 11. 唯一下一轮建议与停止',
        '\n'+next_step,
        '\n本轮已按预算停止：不追加seed、不训练新Backbone、不调V1、不评估官方TEST、不自行同步仓库。由用户本人决定上传哪些产物给ChatGPT进行深度分析。',
        '\n## 12. 文件与准确运行命令',
        '\n根目录：`'+str(r.HERE)+'`。每组结果在results/seed<SEED>/<ARM>/：config.json、effective.yaml、train.log、batch_signatures.csv、epoch_metrics.csv、validation_epoch10.npz、checkpoints/best.pth、checkpoints/final.pth、summary.json。保留了全部最终权重。',
        '\n汇总：results/all_arms_epoch10.csv、paired_epoch10.csv、scientific_summary.json、paired_seed_summary.csv、paired_per_class.csv、gate_diagnostics.csv、candidate_pool_histogram.csv、checkpoint_manifest.csv；源修改对照minimal_fixes.patch，训练前判据preregistered_decision_rules.json，preflight.json，run_registry.json，console/全部stdout。',
        '\n上述为实际执行入口；现有输出目录会拒绝重复训练，不能直接重跑或清除已完成结果。需要恢复实验时应先取得用户明确授权。\n']
    exe=d['source_preflight']['environment']['python']
    commands=[f'& "{exe}" -B -u H2T_repro_audit/ravs_v1/test_policy.py',f'& "{exe}" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py preflight']
    commands += [f'& "{exe}" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py train --seed {seed} --arm {arm}' for seed in r.SEEDS for arm in r.ARMS]
    commands += [f'& "{exe}" -B -u H2T_repro_audit/ravs_v1/run_ravs_v1.py report']
    text.append('\n解释器：`'+exe+'`；项目根目录的准确顺序命令：\n\n```powershell\n'+'\n'.join(commands)+'\n```')
    report='\n'.join(text)
    (r.RESULTS/'RAVS_V1_RESULTS.md').write_text(report,encoding='utf-8')
    (r.RESULTS/'RAVS_V1_SCIENTIFIC_REPORT.md').write_text(report,encoding='utf-8')
    (r.HERE/'RAVS_V1_SCIENTIFIC_REPORT.md').write_text(report,encoding='utf-8')
