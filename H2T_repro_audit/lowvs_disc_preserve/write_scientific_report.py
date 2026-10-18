"""Verify saved six-run artifacts and write the preregistered Chinese report.
No forward pass, optimizer, backward, model search or TEST access.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import csv,hashlib,json,sys
import numpy as np
import torch
from execute_step import install,verify
HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
WORK=HERE.parents[1]
OLD=WORK/'H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014'
HISTORY=WORK/'H2T_repro_audit/lowvs_norm_direction/results'
ARMS=('LowVS','LowVS-CleanCE','LowVS-TopCompetitor')
SEEDS=(124,125)
KEYS=('indices_sha256','images_sha256','features_sha256','scores_sha256','gumbel_sha256','uniform_mask_sha256','lowvs_mask_sha256')
METRICS=('Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro','validation_CE')
COMPARISONS=((ARMS[2],ARMS[1]),(ARMS[1],ARMS[0]),(ARMS[2],ARMS[0]))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def statehash(q):
    h=hashlib.sha256()
    for k,v in q.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def dump(p,obj):
    assert Path(p).resolve().is_relative_to(OUT.resolve())
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False,indent=2)
def csvwrite(p,data):
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def table(headers,data):
    return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(v) for v in r)+'|' for r in data]
def diffstr(x):return f'{float(x):+.4f}'

def main():
    install();verify()
    assert not (OUT/'LOWVS_DISC_SCIENTIFIC_REPORT.md').exists(),'Do not overwrite'
    reg=read(OUT/'execution_registry.json')
    wanted=[['losses'],['preflight']]+[['train','--seed',str(s),'--arm',a] for s in SEEDS for a in ARMS]+[['report']]
    assert [r['args'] for r in reg]==wanted and all(r['status']=='completed' and r['exit_code']==0 for r in reg)
    assert not (OUT/'FAILED_STOP.json').exists()
    guards={p.name:read(p) for p in (OUT/'guards').glob('*.json')}
    assert len(guards)==9 and all(r['completed'] and not r['guard_events'] and not r.get('reused_TEST_guard_events',[]) for r in guards.values())
    attest=read(OUT/'dataset_attestation.json');pre=read(OUT/'preflight.json')
    assert attest['passed'] and pre['passed'] and not pre['official_TEST_used'] and not pre['test_guard_events']
    assert sha(HERE/'losses.py')==sha(OUT/'provided_source/losses.py') and sha(HERE/'run_pilot.py')==sha(OUT/'provided_source/run_pilot.py')
    ck=torch.load(OLD/'locked/clean_stage1_best.pth',map_location='cpu',weights_only=False)
    assert ck['epoch']==184 and ck['seed']==42 and not ck['TEST_used'] and ck['selection_partition']=='official TRAIN holdout'
    init_c=statehash(ck['state_dict_classifier']);init_b=statehash(ck['state_dict_model'])
    bn={k:int(v) for k,v in ck['state_dict_model'].items() if k.endswith('num_batches_tracked')}
    assert len(ck['state_dict_model'])==186 and len(bn)==31 and set(bn.values())=={15640}
    traincounts=np.array(attest['train_counts']);vcounts=np.array(attest['validation_counts'])
    groups={'Head':np.arange(1,36),'OfficialMedium':np.arange(36,71),'Tail':np.arange(71,100),
       'Many':np.flatnonzero((traincounts>100)&(vcounts>0)),
       'Medium':np.flatnonzero((traincounts>=20)&(traincounts<=100)&(vcounts>0)),
       'Few':np.flatnonzero((traincounts<20)&(vcounts>0))}
    merged=rows(OUT/'epoch10.csv');pairs=rows(OUT/'paired.csv');perclass=rows(OUT/'per_class.csv')
    assert len(merged)==6 and len(pairs)==6 and len(perclass)==594 and len(rows(OUT/'curves.csv'))==60
    stats={(int(r['seed']),r['arm']):r for r in merged}
    truth=None;preds={};logits={};epochs={};summaries={};statechecks=[];classacc=[];replaychecks=[]
    for seed in SEEDS:
        historic=rows(HISTORY/f'seed{seed}/LowVS-Full/batch_signatures.csv');assert len(historic)==850
        plan=np.load(OLD/f'plans/{seed}/sampling_plan.npz',allow_pickle=False)
        ytrain=np.load(OLD/'train_labels.npy',allow_pickle=False)
        baseline=None
        for arm in ARMS:
            folder=OUT/f'seed{seed}'/arm;s=read(folder/'summary.json');cfg=read(folder/'config.json')
            assert s['completed'] and s['steps']==850 and s['pairing_historical_850'] and s['frozen_backbone_BN'] and not s['official_TEST_used']
            assert s['pairing_internal_850']==(arm!=ARMS[0])
            assert cfg['initial_state']=={'backbone':init_b,'classifier':init_c}
            assert cfg['stage1_sha']==attest['stage1_sha256'] and s['initial_classifier_sha']==init_c
            assert cfg['lambda']==.5 and cfg['margin']==0
            sig=rows(folder/'batch_signatures.csv');assert len(sig)==850
            if baseline is None:baseline=sig
            for i,(a,b,c) in enumerate(zip(sig,historic,baseline)):
                assert int(a['epoch'])==int(b['epoch'])==i//85+1 and int(a['batch'])==int(b['batch'])==i%85
                assert int(a['size'])==(95 if i%85==84 else 128)
                for key in KEYS:assert a[key]==b[key]==c[key],(seed,arm,i,key)
            replaychecks.append({'seed':seed,'arm':arm,'historical_batches':850,'internal_batches':850 if arm!=ARMS[0] else 0,**{key:850 for key in KEYS}})
            hist=rows(folder/'epochs.csv');assert len(hist)==10;epochs[seed,arm]=hist;summaries[seed,arm]=s
            for e,r in enumerate(hist):
                assert int(r['epoch'])==e+1 and abs(float(r['lr'])-.02*.5*(1+np.cos(e/10*3.1415926535)))<1e-12
                assert int(r['train_tail_receiver_draws'])==int((ytrain[plan['receivers'][e]]>=71).sum())
                assert abs(float(r['train_total_loss'])-float(r['train_fused_ce'])-float(r['train_tail_clean_aux_weighted_by_lambda']))<2e-6
                assert np.isfinite([float(r[k]) for k in ('train_total_loss','train_fused_ce','train_tail_clean_aux_weighted_by_lambda')]).all()
                if arm==ARMS[0]:assert float(r['train_tail_clean_aux_weighted_by_lambda'])==0
                if arm==ARMS[2]:assert sum(int(r[k]) for k in ('top_competitor_head','top_competitor_medium','top_competitor_tail'))==int(r['train_tail_receiver_draws'])
            path=folder/'final_fc.pth';assert sha(path)==s['final_checkpoint_sha']
            final=torch.load(path,map_location='cpu',weights_only=False)
            assert final['seed']==seed and final['arm']==arm and final['epoch']==10 and final['stage1_sha']==attest['stage1_sha256'] and not final['official_TEST_used']
            finalhash=statehash(final['classifier']);assert finalhash==s['final_classifier_sha'] and finalhash!=init_c
            changes={k:float((v-ck['state_dict_classifier'][k]).abs().max()) for k,v in final['classifier'].items()}
            assert all(v>0 for v in changes.values()) and sum(v.numel() for v in final['classifier'].values())==6500
            statechecks.append({'seed':seed,'arm':arm,'backbone_initial_hash':init_b,'classifier_initial_hash':init_c,'classifier_final_hash':finalhash,
                 'weight_max_change':changes['module.fc.weight'],'bias_max_change':changes['module.fc.bias'],'BN_counter_unchanged':15640,
                 'all_186_backbone_tensors_checked_each_epoch':True,'checkpoint_sha256':sha(path)})
            v=np.load(folder/'validation_epoch10.npz',allow_pickle=False);y=v['labels'];p=v['predictions'];z=v['logits']
            assert z.shape==(1980,100) and np.isfinite(z).all() and np.array_equal(p,z.argmax(1))
            assert np.array_equal(np.bincount(y,minlength=100),vcounts)
            if truth is None:truth=y.copy()
            else:assert np.array_equal(truth,y)
            preds[seed,arm]=p;logits[seed,arm]=z
            accuracy=np.array([100*float((p[y==c]==c).mean()) for c in range(1,100)])
            calc={'Overall':100*float((p==y).mean()),'Macro':float(accuracy.mean())}
            for name,ids in groups.items():sel=np.isin(y,ids);calc[name]=100*float((p[sel]==y[sel]).mean())
            zz=z.astype(np.float64);mx=zz.max(1);calc['validation_CE']=float((mx+np.log(np.exp(zz-mx[:,None]).sum(1))-zz[np.arange(len(y)),y]).mean())
            for key,value in calc.items():assert abs(value-float(stats[seed,arm][key]))<(1e-5 if key=='validation_CE' else 1e-9),(seed,arm,key)
            for c in range(1,100):classacc.append({'seed':seed,'arm':arm,'class_id':c,'n':20,'correct':int((p[y==c]==c).sum()),'accuracy':accuracy[c-1]})
    countchecks=[];transitions=[];diffrows=[];changed=[];pairlookup={}
    rawids=np.load(OLD/'split_indices.npz',allow_pickle=False)['validation_raw_indices']
    for seed in SEEDS:
        for a,b in COMPARISONS:
            pa,pb=preds[seed,a],preds[seed,b];wins=(pa==truth)&(pb!=truth);losses=(pa!=truth)&(pb==truth)
            row=next(r for r in pairs if int(r['seed'])==seed and r['comparison']==a+' - '+b)
            assert int(row['wins'])==wins.sum() and int(row['losses'])==losses.sum() and int(row['net_correct'])==int(wins.sum())-int(losses.sum())
            rec={'seed':seed,'comparison':a+' - '+b,'wins':int(wins.sum()),'losses':int(losses.sum()),'net':int(wins.sum())-int(losses.sum())}
            for name,ids in groups.items():
                sel=np.isin(truth,ids);gain=int(wins[sel].sum());loss=int(losses[sel].sum())
                rec[name+'_wins']=gain;rec[name+'_losses']=loss;rec[name+'_net']=gain-loss
                assert abs(100*(gain-loss)/int(sel.sum())-(float(stats[seed,a][name])-float(stats[seed,b][name])))<1e-9
            for name in ('Overall','Head','OfficialMedium','Tail','Few','validation_CE'):
                delta=float(stats[seed,a][name])-float(stats[seed,b][name]);assert abs(delta-float(row['delta_'+name]))<1e-9;rec['delta_'+name]=delta
            assert rec['Head_net']==int(row['head_net']) and rec['Tail_net']==int(row['tail_net'])
            tail=truth>=71
            head_to_correct=int((tail&(pb<=35)&(pa==truth)).sum())
            head_to_otherwrong=int((tail&(pb<=35)&(pa>=36)&(pa!=truth)).sum())
            correct_to_head=int((tail&(pb==truth)&(pa<=35)).sum())
            otherwrong_to_head=int((tail&(pb>=36)&(pb!=truth)&(pa<=35)).sum())
            delta_head=int((tail&(pa<=35)).sum())-int((tail&(pb<=35)).sum())
            assert delta_head==correct_to_head+otherwrong_to_head-head_to_correct-head_to_otherwrong==int(row['tail_true_to_head_pred_delta'])
            transitions.append({'seed':seed,'comparison':a+' - '+b,'Tail_correct_wins':rec['Tail_wins'],'Tail_correct_losses':rec['Tail_losses'],'Tail_correct_net':rec['Tail_net'],
                'Head_error_to_correct_Tail':head_to_correct,'Head_error_to_other_error':head_to_otherwrong,
                'correct_Tail_to_Head_error':correct_to_head,'other_error_to_Head_error':otherwrong_to_head,'Tail_to_Head_error_delta':delta_head})
            detail=[r for r in perclass if int(r['seed'])==seed and r['comparison']==a+' - '+b]
            assert len(detail)==99 and sum(int(r['net']) for r in detail)==rec['net']
            for r in detail:
                c=int(r['class_id']);sel=truth==c
                assert int(r['n'])==20 and int(r['wins'])==wins[sel].sum() and int(r['losses'])==losses[sel].sum()
            for i in np.flatnonzero(wins|losses):changed.append({'seed':seed,'comparison':a+' - '+b,'validation_row':int(i),'TRAIN_raw_id':int(rawids[i]),'true_class':int(truth[i]),'prediction_A':int(pa[i]),'prediction_B':int(pb[i]),'change':'wrong_to_right' if wins[i] else 'right_to_wrong'})
            pairlookup[seed,a,b]=rec;countchecks.append(rec)
    losscomponents=[{'seed':seed,'arm':arm,'epoch':int(r['epoch']),'fused_CE':float(r['train_fused_ce']),'aux_fullbatch_unweighted_derived':float(r['train_tail_clean_aux_weighted_by_lambda'])/.5,'lambda':.5,'aux_fullbatch_weighted':float(r['train_tail_clean_aux_weighted_by_lambda']),'total_loss':float(r['train_total_loss']),'tail_receiver_draws':int(r['train_tail_receiver_draws'])} for seed in SEEDS for arm in ARMS for r in epochs[seed,arm]]
    csvwrite(OUT/'loss_components.csv',losscomponents)
    csvwrite(OUT/'paired_detailed_counts.csv',countchecks);csvwrite(OUT/'tail_error_transitions.csv',transitions)
    csvwrite(OUT/'per_class_accuracy.csv',classacc);csvwrite(OUT/'classifier_state_verification.csv',statechecks)
    csvwrite(OUT/'replay_verification.csv',replaychecks)
    if changed:csvwrite(OUT/'paired_changed_images.csv',changed)
    for a,b in COMPARISONS:
        diffrows.append({'comparison':a+' - '+b,**{k:float(np.mean([float(stats[s,a][k])-float(stats[s,b][k]) for s in SEEDS])) for k in METRICS}})
    csvwrite(OUT/'mean_differences.csv',diffrows)
    primary=[pairlookup[s,ARMS[2],ARMS[1]] for s in SEEDS]
    tailnets=[r['Tail_net'] for r in primary];overallnets=[r['net'] for r in primary];headnets=[r['Head_net'] for r in primary]
    if all(n>3 for n in tailnets) and all(n>=0 for n in overallnets) and all(n>=0 for n in headnets):
        judgement='两seed均满足Tail净改善、Overall不降且Head不降的方向条件，可保留后续特征级研究候选；当前线性分类器pilot仍不构成方法创新或泛化证明。'
    elif all(abs(n)<=3 for n in tailnets) and all(abs(n)<=3 for n in overallnets):
        judgement='C−B仅少量图像差异，缺少超出额外干净CE的证据；按预设规则停止，不作等效性或特殊竞争机制有效性宣称。'
    elif any(n<=0 for n in tailnets) or any(n<0 for n in overallnets):
        judgement='C−B未满足两个seed均改善Tail且Overall不下降的预设方向条件；停止本轮候选，不扫描系数、margin或种子。'
    else:
        judgement='Tail/Overall方向部分满足，但Head存在代价或至少一seed只改善少量图片。未预设Head可接受损失的数值阈值，不能事后设门槛宣布通过；本轮停止并交付复核。'
    lines=['# Low-VS 尾类干净判别保持：三臂科学报告','',
      '## 1. 首要比较：C（TopCompetitor）− B（CleanCE）','',
      '固定第10epoch为唯一主终点。A=LowVS，B=A+尾类干净CE，C=A+尾类干净最强竞争类别softplus；两个辅助项均为0.5×尾类损失之和/完整batch大小。所有准确率来自同一1980张原始TRAIN留出验证，class1..99各20张，class0缺失，100维输出保留。','']
    lines+=table(['seed','ΔOverall pp','ΔHead pp','ΔTail pp','ΔFew pp','Δ验证CE'],[[r['seed'],*[diffstr(r['delta_'+k]) for k in ('Overall','Head','Tail','Few','validation_CE')]] for r in primary])
    lines+=['','配对计数中的A/B指比较式左/右策略；wrong→right与right→wrong使用同一张验证图像的真实类别，不是只看预测类别组改变。','']
    lines+=table(['seed','全体gain/loss/net','Head gain/loss/net','Tail gain/loss/net','Few gain/loss/net'],[[r['seed'],*[f"{r.get(g+'_wins',r['wins'])}/{r.get(g+'_losses',r['losses'])}/{r.get(g+'_net',r['net']):+d}" for g in ('All','Head','Tail','Few')]] for r in primary])
    lines+=['','**预设规则判读：** '+judgement,'',
      '## 2. 六组固定epoch10绝对结果与辅助比较','']
    lines+=table(['seed','组别',*METRICS],[[s,a,*[f'{float(stats[s,a][k]):.4f}' for k in METRICS]] for s in SEEDS for a in ARMS])
    lines+=['','以下B−A、C−A用于区分额外干净监督与特定竞争约束，不替代首要C−B。','']
    secondary=[pairlookup[s,a,b] for s in SEEDS for a,b in COMPARISONS[1:]]
    lines+=table(['seed','比较','ΔOverall','ΔHead','ΔTail','ΔFew','净正确图像','Tail净正确','Δ验证CE'],[[r['seed'],r['comparison'],*[diffstr(r['delta_'+k]) for k in ('Overall','Head','Tail','Few')],r['net'],r['Tail_net'],diffstr(r['delta_validation_CE'])] for r in secondary])
    lines+=['','两seed均值仅作描述，不作显著性推断：','']
    lines+=table(['比较',*METRICS],[[r['comparison'],*[diffstr(r[k]) for k in METRICS]] for r in diffrows])
    for a,b in COMPARISONS:
        d=[pairlookup[s,a,b] for s in SEEDS]
        signs=[]
        for name in ('Overall','Head','Tail','Few'):
            vs=[r['delta_'+name] for r in d]
            signs.append(name+('跨seed变号' if vs[0]*vs[1]<0 else '同向' if vs[0]*vs[1]>0 else '至少一seed持平'))
        lines+=['',a+'−'+b+'：'+'；'.join(signs)+'。']
    lines+=['','## 3. 减少Tail→Head错误与真正判对Tail必须分开','',
      '当原先错误预测为Head的Tail图像转到另一个错误类别时，Tail→Head计数减少，但Tail准确率没有增加。下面直接分解同图变化，不把组别迁移当成正确识别。','']
    lines+=table(['seed','比较','Tail gain/loss/net','Head错误→正确Tail','Head错误→其他错误','正确Tail→Head错误','其他错误→Head错误','Tail→Head Δ'],[[r['seed'],r['comparison'],f"{r['Tail_correct_wins']}/{r['Tail_correct_losses']}/{r['Tail_correct_net']:+d}",r['Head_error_to_correct_Tail'],r['Head_error_to_other_error'],r['correct_Tail_to_Head_error'],r['other_error_to_Head_error'],r['Tail_to_Head_error_delta']] for r in transitions])
    lines+=['','完整真值组×预测组混淆见 `group_confusion.csv`，其中Medium为OfficialMedium（ID36..70），不是训练频次Medium。预测class0算Head错误。','',
      '## 4. 逐类别得失与离散分辨率','',
      '全量99类、每类n=20的正负变化见 `per_class.csv`，每臂绝对正确数见 `per_class_accuracy.csv`，逐张变化及原TRAIN raw ID见 `paired_changed_images.csv`。不挑有利类别代表全部结果。','']
    for a,b in COMPARISONS:
        nets={s:{int(r['class_id']):int(r['net']) for r in perclass if int(r['seed'])==s and r['comparison']==a+' - '+b} for s in SEEDS}
        lines+=['### '+a+' − '+b,'']
        for sign,label in ((1,'两seed均净改善'),(-1,'两seed均净下降')):
            selected=[c for c in range(1,100) if all(sign*nets[s][c]>0 for s in SEEDS)]
            tail=[c for c in selected if c>=71];head=[c for c in selected if c<=35]
            lines +=[f'- {label}：全体{len(selected)}类；Head {len(head)}类；Tail {len(tail)}类。Tail类别及两seed净正确数：'+(', '.join(f'{c}({nets[124][c]:+d}/{nets[125][c]:+d})' for c in tail) if tail else '无')+'。']
        lines+=['']
    lines+=['Overall一张=0.050505pp，Head一张=0.142857pp，Tail一张=0.172414pp，Few一张=0.166667pp；每类别一张=5pp。两seed平均很小的分数差不能掩盖只改判少数图片，亦不能将3960次预测当作独立验证样本。','',
      '## 5. 训练曲线与固定终点','',
      '以下显示epoch1/5/10；全部60行逐epoch指标、主CE、系数化辅助损失和Tail接收次数见 `curves.csv`。best epoch仅记录最高Overall的诊断值，不保存或选用其模型作为主结果。','']
    lines+=table(['seed','组别','Overall 1/5/10','Head 1/5/10','Tail 1/5/10','best Overall epoch','best Overall'],[[s,a,*['/'.join(f'{float(epochs[s,a][e][k]):.4f}' for e in (0,4,9)) for k in ('Overall','Head','Tail')],summaries[s,a]['best_epoch_diagnostic'],f"{summaries[s,a]['best_Overall_diagnostic']:.4f}"] for s in SEEDS for a in ARMS])
    lines+=['','epoch10训练损失组成（每个epoch按样本数加权；aux列已乘λ=0.5）：','']
    lines+=table(['seed','组别','融合CE','系数化干净辅助项','总loss','Tail接收draws'],[[s,a,*[f'{float(epochs[s,a][-1][k]):.6f}' for k in ('train_fused_ce','train_tail_clean_aux_weighted_by_lambda','train_total_loss')],epochs[s,a][-1]['train_tail_receiver_draws']] for s in SEEDS for a in ARMS])
    lines+=['','## 6. 协议、来源与正确性证据','',
      f'- 锁定Clean StageI seed42 epoch184，SHA256 `{attest["stage1_sha256"]}`；原IF100统计SHA256 `{attest["statistics_sha256"]}`。未训练StageI、未重算统计或改Bottom48/随机16/rho0.25/供体抽样。',
      '- 原始TRAIN文件SHA256 `'+attest['official_TRAIN_sha256']+'`；10,847张LT的raw IDs严格匹配作者seed0，像素/标签对应官方TRAIN；与验证raw IDs和完整像素无重叠。验证缓存校验对应原历史模型，未读官方TEST或对TEST做哈希。',
      '- ResNet32的463,504个参数及31个BN完全冻结；完整186项Backbone状态每epoch逐位核对，running_mean/var、affine及num_batches_tracked保持不变，BN计数均15,640。只优化原100维FC的6,500个参数，各臂weight和bias确有变化。',
      '- 三臂均从同一StageI FC独立初始化、空SGD状态；lr0.1、lr_factor0.2，实际首epoch单FC组lr0.02，沿用历史cosine，momentum0.9、wd0.0005、batch128、末batch95、10epochs×85更新=850更新/组，总计6组5,100次更新。',
      '- 每一臂每seed的850批、全部7项输入/掩码字段均与旧LowVS-Full真实记录匹配；共5,100个历史batch匹配、35,700项字段比较。4个辅助臂另与本轮A对照匹配3,400个batch/23,800项字段。字段包括索引/标签、实际两套增强像素、冻结特征、在线VS评分、Gumbel、Uniform与Low-VS通道。',
      '- preflight在两个seed验证16个完整通道来自供体、未选通道逐位等于接收者；label始终为接收标签。损失CPU单测验证tail-only样本梯度、排除y且detach的最强竞争类别、m=0，以及无Tail batch精确退化到A。',
      '- “干净”是同一接收图像crop/flip后的未融合特征，并非未增强图像。j由当前正在更新的FC在训练干净特征上产生，既不改供体也不改VS掩码。尾类样本辅助监督仍可更新Head输出行权重，因此Head存在代价并不违反tail-only样本定义。',
      '- 各组最后checkpoint重载后，预测逐项相同，logits按提供代码rtol=atol=1e−6检查，全部指标差<1e−8；不把该logit容差检查描述成逐bit比较。独立报告脚本从已保存数组重算指标和配对净得失，没有额外前向或训练。',
      '- 原提供losses.py和run_pilot.py逐字节未改，无阻塞bug修复。只新增日志/访问防护封装和本报告分析，原数据、模型、增强、在线评分、融合及优化函数均直接复用原文件。有效YAML残留的旧model_dir展示字段未使用；实际所有输出路径由新runner的RESULTS确定，历史写保护和前后校验均通过。','',
      '历史LowVS数值复核（本轮A−旧LowVS-Full，pp）：','']
    history_drift=rows(OUT/'historical_replay.csv') if (OUT/'historical_replay.csv').exists() else []
    if history_drift:lines+=table(['seed','ΔOverall','ΔHead','ΔTail','ΔFew'],[[r['seed'],*[diffstr(r['delta_'+k+'_pp']) for k in ('Overall','Head','Tail','Few')]] for r in history_drift])
    else:lines+=['未提供旧epoch10数值总表；850批真实历史哈希仍全部检查，不伪造数值历史。']
    lines+=['','## 7. 限制、警告与停止','',
      '- 同λ、同Tail样本和同全batch归一化不等于梯度预算完全匹配：全类CE与单最强竞争softplus的函数、梯度大小和方向不同。C−B检验两种给定辅助监督的整体差别，不能只归因于抽象“竞争关系”。',
      '- 最强负类pairwise ranking/softplus是已有类型的目标；本轮不主张该loss本身首创，即使有效也仅为后续特征级模块提供方向证据。Backbone固定，本轮不构成表征学习提升。',
      '- 一个StageI和两个相关StageII seed，共用同一1980张开发验证；该集已反复用于历史选择和分析，class0缺失。没有独立显著性、官方100类TEST性能或最终泛化证据；不能与论文TEST精度直接比较。',
      '- CPU单测及preflight均触发提供代码float(loss)记录导致的requires_grad转标量UserWarning；已原样保存console日志，测试通过，未修改损失或训练策略处理该非阻塞警告。',
      '- 本轮不增加lambda、margin、K、epoch或seed扫描，不新增第四组，不启动特征级训练或其他数据集。未执行任何版本控制操作。',
      '- '+judgement,'',
      '## 8. 完整产物与复现入口','',
      '全部相对路径以 `H2T_repro_audit/lowvs_disc_preserve/results/` 为根：','',
      '- `preflight.json`、`dataset_attestation.json`、`execution_source_lock.json`、`historical_manifest.json`、`static_review.json`。',
      '- `seed{124,125}/{三臂}/epochs.csv`、`batch_signatures.csv`、`effective.yaml`、`config.json`、`final_fc.pth`、`validation_epoch10.npz`、`summary.json`。',
      '- `epoch10.csv`、`paired.csv`、`per_class.csv`、`curves.csv`、`group_confusion.csv`、`historical_replay.csv`、`scientific_summary.json`、`LOWVS_DISC_RESULTS.md`。',
      '- `loss_components.csv`同时保存每epoch主CE、未乘λ的辅助项和乘λ辅助项；未乘λ值由原始加权记录除以固定0.5恢复，显式标为derived，归一化仍是完整batch而非尾类样本数。',
      '- `paired_detailed_counts.csv`、`tail_error_transitions.csv`、`per_class_accuracy.csv`、`paired_changed_images.csv`、`mean_differences.csv`、`replay_verification.csv`、`classifier_state_verification.csv`、`scientific_verification.json`。',
      '- `console/*.log`、`guards/*.json`、`execution_registry.json`保存每个完整命令、stdout/警告、起止时间、退出码、GPU环境及峰值显存。`provided_source/`和`runtime_source_snapshot/`保留原代码和锁定依赖；`delivery_manifest.json`记录原始文件SHA256。',
      '- 使用 `D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe`，从仓库根按 `user_task_instructions.txt` 的losses→preflight→seed124三组→seed125三组→report顺序执行。本轮执行封装只加保护和日志，已有结果拒绝覆盖；没有训练重试。']
    report=OUT/'LOWVS_DISC_SCIENTIFIC_REPORT.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    verify()
    verification={'passed':True,'completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
      'formal_runs':6,'steps_per_run':850,'total_optimizer_steps':5100,'no_StageI_training':True,
      'historical_batches_matched':5100,'historical_7field_comparisons':35700,'internal_batches_matched':3400,'internal_7field_comparisons':23800,
      'backbone_tensors':186,'BN_modules':31,'BN_counter_frozen':15640,'classifier_parameters':6500,
      'all_provided_core_sources_bytewise_unchanged':True,'history_files_verified':len(read(OUT/'historical_manifest.json')),
      'guard_processes':9,'official_TEST_events':0,'historical_write_events':0,'saved_predictions_shape_per_run':[1980,100],
      'primary_C_minus_B':primary,'mean_differences':diffrows,'judgement':judgement,'reused_dev_set':True,
      'independent_significance_claimed':False,'no_further_training':True,'no_version_control_operations':True,
      'warnings':['Provided self_test float(loss) conversion UserWarning; unit/preflight passed; retained unchanged.'],
      'peak_allocated_MiB':max(g.get('peak_allocated_MiB',0) for g in guards.values()),'report_sha256':sha(report)}
    dump(OUT/'scientific_verification.json',verification)
    dump(OUT/'STOP_COMPLETED.json',{'status':'complete and stopped','additional_training':False,'TEST_used':False,'version_control_operations':False})
    dump(OUT/'final_analysis_source_lock.json',{p.relative_to(WORK).as_posix():sha(p) for p in HERE.iterdir() if p.is_file()})
    files={p.relative_to(HERE).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
    dump(OUT/'delivery_manifest.json',{'status':'complete and stopped','scientific_report':'results/LOWVS_DISC_SCIENTIFIC_REPORT.md','files':files})
    print(json.dumps({'report':str(report),'judgement':judgement,'primary_C_minus_B':primary,'mean_differences':diffrows,'files_sealed':len(files)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
