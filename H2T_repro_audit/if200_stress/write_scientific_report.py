"""Descriptive verification/report only. No inference, optimizer or new training."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone,timedelta
import csv,hashlib,json,sys
import numpy as np
import torch
from runtime_guard import install
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
OUT=HERE/'results'
ARMS=('DR-NoFusion','Original-BatchShared','Uniform-PerSample','LowVS-PerSample')
SEEDS=(124,125)
METRICS=('Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro','WorstClass')
CONTRASTS=((ARMS[1],ARMS[0]),(ARMS[2],ARMS[1]),(ARMS[3],ARMS[2]),(ARMS[3],ARMS[1]))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,obj):
    assert Path(p).resolve().is_relative_to(OUT.resolve())
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False,indent=2)
def writecsv(p,data):
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def statehash(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def array_hash(value):
    a=np.asarray(value);h=hashlib.sha256();h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes());return h.hexdigest()
def table(headers,data):
    return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(v) for v in row)+'|' for row in data]

def main():
    install(WORK,OUT)
    assert not (OUT/'IF200_STRESS_SCIENTIFIC_REPORT.md').exists(),'Do not overwrite report'
    from execute_step import verify
    verify()
    registry=read(OUT/'execution_registry.json')
    expected=[['policy'],['preflight'],['stage1'],['prepare']]+[['train','--seed',str(s),'--arm',a] for s in SEEDS for a in ARMS]+[['report']]
    completed=[r for r in registry if r['status']=='completed']
    assert [r['args'] for r in completed]==expected
    failed=[r for r in registry if r['status']=='failed']
    assert len(failed)==1 and failed[0]['args']==['preflight']
    assert read(OUT/'preflight_repair.json')['preflight_syntax_only']
    assert not (OUT/'FAILED_STOP_SECOND.json').exists()
    attest=read(OUT/'dataset_attestation.json');pre=read(OUT/'preflight.json')
    assert attest['passed'] and pre['passed']
    guards={p.name:read(p) for p in (OUT/'guards').glob('*.json')}
    assert len(guards)==12 and all(g['completed'] and not g['guard_events'] for g in guards.values())
    counts=np.array(attest['IF200_train_counts'])
    idx=np.load(OUT/'if200_split_indices.npz',allow_pickle=False)
    s1p=OUT/'stage1/stage1_fixed_epoch200.pth';s1=read(OUT/'stage1/summary.json')
    assert sha(s1p)==s1['checkpoint_sha256']
    ck=torch.load(s1p,map_location='cpu',weights_only=False)
    assert ck['seed']==42 and ck['epoch']==200 and ck['selection']=='fixed epoch200, not selected using validation'
    assert ck['IF']==200 and ck['official_TEST_used'] is False
    assert np.array_equal(ck['class_counts'],counts)
    assert ck['source_nested_raw_ids_sha256']==array_hash(idx['train_raw_indices'])
    assert statehash(ck['backbone'])==s1['backbone_hash'] and statehash(ck['classifier'])==s1['classifier_hash']
    bn={k:int(v) for k,v in ck['backbone'].items() if k.endswith('num_batches_tracked')}
    assert len(bn)==31 and set(bn.values())=={15000}
    hist=rows(OUT/'stage1/epochs.csv')
    assert len(hist)==200 and sum(int(r['steps']) for r in hist)==15000 and s1['updates']==15000
    for n,r in enumerate(hist,1):
        expectedlr=.1*n/5 if n<=5 else (.001 if n>180 else (.01 if n>160 else .1))
        assert int(r['epoch'])==n and int(r['steps'])==75 and abs(float(r['lr'])-expectedlr)<1e-12
        assert np.isfinite(float(r['train_CE_mixup']))
    stat=np.load(OUT/'stats.npz',allow_pickle=False);prov=read(OUT/'stats_provenance.json')
    assert sha(OUT/'stats.npz')==prov['stats_sha256'] and sha(OUT/'validation_maps.npy')==prov['validation_maps_sha256']
    assert prov['checkpoint_sha256']==s1['checkpoint_sha256'] and prov['backbone_unchanged'] and not prov['official_TEST_used']
    assert np.array_equal(stat['counts'],counts) and float(stat['epsilon'])>0
    assert np.allclose(stat['pooled'],(stat['variance_eval_ddof0']*counts[:,None]).sum(0)/(9502-100),rtol=0,atol=1e-14)
    assert np.array_equal(stat['shrinkage_lambda'],5/(counts+5))
    vv=(1-stat['shrinkage_lambda'][:,None])*stat['variance_eval_ddof0']+stat['shrinkage_lambda'][:,None]*stat['pooled']
    assert np.array_equal(vv,stat['variance_eval_regularized'])
    assert np.isfinite(stat['mean_eval']).all() and np.isfinite(vv).all() and (vv>=0).all()
    assert np.load(OUT/'validation_maps.npy',mmap_mode='r',allow_pickle=False).shape==(1980,64,8,8)
    metrics=rows(OUT/'epoch10.csv');pair=rows(OUT/'paired.csv');pc=rows(OUT/'per_class.csv')
    assert len(metrics)==8 and len(pair)==8 and len(pc)==792
    by={(int(r['seed']),r['arm']):r for r in metrics}
    predictions={};truth=None;statechecks=[];classacc=[];conf=[]
    validation_counts=np.array(attest['validation_counts'])
    groups={'Head':np.arange(1,36),'OfficialMedium':np.arange(36,71),'Tail':np.arange(71,100),
            'Many':np.flatnonzero((counts>100)&(validation_counts>0)),
            'Medium':np.flatnonzero((counts>=20)&(counts<=100)&(validation_counts>0)),
            'Few':np.flatnonzero((counts<20)&(validation_counts>0))}
    for seed in SEEDS:
        control=None
        for arm in ARMS:
            folder=OUT/f'seed{seed}'/arm;s=read(folder/'summary.json')
            assert s['completed'] and s['epochs']==10 and s['steps']==750 and s['BN_frozen'] and s['backbone_unchanged'] and not s['TEST_used']
            assert s['stage1_sha']==s1['checkpoint_sha256'] and s['classifier_initial_hash']==s1['classifier_hash']
            signatures=rows(folder/'paired_batches.csv')
            assert len(signatures)==750 and [int(r['step']) for r in signatures]==list(range(750))
            if control is None:control=signatures
            else:assert signatures==control
            eh=rows(folder/'epochs.csv');assert len(eh)==10
            for e,r in enumerate(eh):
                assert int(r['epoch'])==e+1 and abs(float(r['lr'])-.02*.5*(1+np.cos(e*np.pi/10)))<1e-12
                assert np.isfinite(float(r['train_CE']))
            checkpoint=torch.load(folder/'final_fc.pth',map_location='cpu',weights_only=False)
            assert checkpoint['stage1_sha']==s1['checkpoint_sha256'] and checkpoint['epoch']==10 and checkpoint['seed']==seed and checkpoint['arm']==arm and not checkpoint['official_TEST_used']
            fc=checkpoint['classifier'];assert list(fc)==list(ck['classifier'])
            assert sum(v.numel() for v in fc.values())==6500
            differences={k:float((v-ck['classifier'][k]).abs().max()) for k,v in fc.items()}
            assert all(d>0 for d in differences.values()) and all(torch.isfinite(v).all() for v in fc.values())
            finalhash=statehash(fc);assert finalhash!=s1['classifier_hash']
            v=np.load(folder/'validation_epoch10.npz',allow_pickle=False);y=v['labels'];p=v['predictions'];logits=v['logits']
            assert logits.shape==(1980,100) and np.isfinite(logits).all() and np.array_equal(p,logits.argmax(1))
            assert np.array_equal(np.bincount(y,minlength=100),validation_counts)
            if truth is None:truth=y.copy()
            else:assert np.array_equal(truth,y)
            predictions[(seed,arm)]=p
            per=np.array([100*float((p[y==c]==c).mean()) for c in range(1,100)])
            calculated={'Overall':100*float((p==y).mean()),'Macro':per.mean(),'WorstClass':per.min()}
            for group,ids in groups.items():calculated[group]=100*float((p[np.isin(y,ids)]==y[np.isin(y,ids)]).mean())
            for key,value in calculated.items():assert abs(value-float(by[(seed,arm)][key]))<1e-9,(seed,arm,key)
            for c in range(1,100):
                sel=y==c;classacc.append({'seed':seed,'arm':arm,'class':c,'n':20,'correct':int((p[sel]==c).sum()),'accuracy':per[c-1]})
            bounds=(('Head',0,36),('OfficialMedium',36,71),('Tail',71,100))
            for tg,lo,hi in bounds:
                for pg,plo,phi in bounds:
                    conf.append({'seed':seed,'arm':arm,'true_group':tg,'predicted_group':pg,'count':int(((y>=lo)&(y<hi)&(p>=plo)&(p<phi)).sum())})
            statechecks.append({'seed':seed,'arm':arm,'stage1_sha256':s1['checkpoint_sha256'],'classifier_initial_hash':s1['classifier_hash'],
                'classifier_final_hash':finalhash,'classifier_weight_max_change':differences['module.fc.weight'],
                'classifier_bias_max_change':differences['module.fc.bias'],'checkpoint_sha256':sha(folder/'final_fc.pth'),
                'backbone_state_tensors':len(ck['backbone']),'BN_modules':len(bn),'BN_counter':15000,
                'backbone_BN_all_state_hash_asserted_equal_each_epoch':True,'checkpoint_prediction_replay_verified_by_training_script':True})
    pairby={}
    for c in pair:
        seed=int(c['seed']);a,b=c['comparison'].split(' - ');pa=predictions[seed,a];pb=predictions[seed,b]
        wins=(pa==truth)&(pb!=truth);losses=(pa!=truth)&(pb==truth)
        assert int(c['wins'])==wins.sum() and int(c['losses'])==losses.sum() and int(c['net'])==int(wins.sum())-int(losses.sum())
        assert abs(100*int(c['net'])/1980-(float(by[seed,a]['Overall'])-float(by[seed,b]['Overall'])))<1e-9
        for group,ids in groups.items():
            if group not in ('Head','OfficialMedium','Tail'):continue
            selected=np.isin(truth,ids)
            assert int(c[group+'_net'])==int(wins[selected].sum())-int(losses[selected].sum())
            assert abs(float(c[group+'_delta_pp'])-100*int(c[group+'_net'])/selected.sum())<1e-9
        h2t=int(((truth<=35)&(pa>=71)).sum())-int(((truth<=35)&(pb>=71)).sum())
        t2h=int(((truth>=71)&(pa<=35)).sum())-int(((truth>=71)&(pb<=35)).sum())
        assert h2t==int(c['head_true_to_tail_pred_delta']) and t2h==int(c['tail_true_to_head_pred_delta'])
        matching=[r for r in pc if int(r['seed'])==seed and r['comparison']==c['comparison']]
        assert len(matching)==99 and sum(int(r['net']) for r in matching)==int(c['net']) and all(int(r['n'])==20 for r in matching)
        pairby[seed,a,b]=c
    for seed in SEEDS:
        for a,b in CONTRASTS:
            wa=predictions[seed,a]==truth;wb=predictions[seed,b]==truth
            for c in range(1,100):
                r=next(r for r in pc if int(r['seed'])==seed and r['comparison']==a+' - '+b and int(r['class'])==c)
                selected=truth==c
                assert int(r['net'])==int(wa[selected].sum())-int(wb[selected].sum())
    writecsv(OUT/'classifier_state_verification.csv',statechecks)
    writecsv(OUT/'per_class_accuracy.csv',classacc)
    writecsv(OUT/'head_tail_confusion.csv',conf)
    meanrows=[];consistent={}
    for a,b in CONTRASTS:
        for scope in ('All','Head','Tail'):
            ids=range(1,100) if scope=='All' else groups[scope]
            seednet={seed:{int(r['class']):int(r['net']) for r in pc if int(r['seed'])==seed and r['comparison']==a+' - '+b} for seed in SEEDS}
            consistent[a+' - '+b+' / '+scope]=[{'class':int(c),'net124':seednet[124][int(c)],'net125':seednet[125][int(c)]} for c in ids if all(seednet[s][int(c)]<0 for s in SEEDS)]
        delta={key:[float(by[s,a][key])-float(by[s,b][key]) for s in SEEDS] for key in METRICS}
        meanrows.append({'comparison':a+' - '+b,**{key:float(np.mean(values)) for key,values in delta.items()}})
    writecsv(OUT/'mean_paired_differences.csv',meanrows)
    candidate=all(int(pairby[s,ARMS[1],ARMS[0]]['Tail_net'])<0 for s in SEEDS)
    independent_vs=all(float(by[s,ARMS[3]]['Tail'])>float(by[s,ARMS[2]]['Tail']) for s in SEEDS)
    groupsummary={name:{'classes':len(ids),'samples':int(validation_counts[ids].sum()),'class_ids':[int(c) for c in ids]} for name,ids in groups.items()}
    history_inventory=read(OUT/'historical_manifest.json')
    assert not any(Path(k).name.casefold()=='test' and Path(k).parent.name.casefold()=='cifar-100-python' for k in history_inventory)
    max_peak=max(g.get('peak_allocated_MiB',0) for g in guards.values())
    print('All frozen-state, source, checkpoint, count and paired-prediction checks PASS',flush=True)
    report=['# IF200 固定 BN 四臂压力测试：科学报告','',
        '## 1. 准确命题与执行范围','',
        '本轮只检验：在 CIFAR-100-LT IF200、同一套新训 Stage I、固定 Backbone/BN、固定第10 epoch 的条件下，整 batch 共享随机通道替换相比 DR 是否出现尾类负迁移；以及 Low-VS 相比逐样本 Uniform 是否存在额外收益。',
        'Original-BatchShared 是作者式掩码粒度与 BS/IS 语义的对照，使用独立 Gumbel 流，不能称逐随机数完全复现。作者 IF200 YAML 为 shift_bn=True；本轮明确固定全部 BN，因此不能把本面板的结果写成作者完整方法成功或失败。','',
        '## 2. 数据、模型与固定预算','',
        f'- 真实 CIFAR-100 官方 TRAIN 文件 SHA256：`{attest["official_TRAIN_sha256"]}`。原文件仅解码 TRAIN，未读取 TEST，亦未对 TEST 做不透明哈希。',
        '- 官方 seed0 按类顺序 shuffle；IF200 每类取 int(500×0.005^(c/99)) 个，与该顺序的 IF100 类内前缀严格相等。原始 raw IDs、像素及标签均逐项核对；9,502 张、100 类、最少2/最多500张。',
        '- 沿用 1,980 张原始 TRAIN 留出验证，class1..99 各20张；class0 缺失，但分类器始终保留100维 logits。IF200 与验证的 raw ID 和完整像素重叠均为0。',
        '- Stage I：全新随机初始化 ResNet32+FC，seed42，CE+MixUp α=1；SGD lr0.1/momentum0.9/wd0.0002，batch128；200 epochs、15,000次更新，5epoch warmup，epoch161/181起下降。只保存固定 epoch200，不按验证选 best。',
        f'- 新 Stage I checkpoint：`stage1/stage1_fixed_epoch200.pth`，SHA256 `{s1["checkpoint_sha256"]}`。未载入旧 IF100 Stage I 权重。',
        '- Stage II：seed124/125各四臂，从上述同一 Stage I 的同一 FC 状态独立初始化，SGD状态为空；仅训练6,500个FC参数，冻结463,504个Backbone参数及31个BN。每组10epochs×75更新=750次，末batch30张，总共8组、6,000次更新。',
        '- 作者 IF200 YAML lr0.2，lr_factor0.1；单FC组初始有效lr0.02，10epoch cosine，momentum0.9、wd0.0002、CE，无新增损失或超参数搜索。',
        f'- 使用新模型的 IF200 TRAIN-only eval GAP 统计：ddof0类方差，pooled=Σn×var0/(9502−100)，λ=5/(n+5)，v_reg=(1−λ)var0+λpooled；epsilon=1e−6×全LT通道方差(ddof1)均值={float(stat["epsilon"]):.12g}。供体n<20的类别{int((counts<20).sum())}个，按既定规则回退到同一个逐样本Uniform掩码；不删除或改变IS供体类别分布。',
        f'- 环境：`{pre["GPU"]}`；Python `{sys.executable}`；PyTorch {torch.__version__} / CUDA {torch.version.cuda}。各步骤峰值显存见 `guards/*.json`；最大 allocated {max_peak:.2f} MiB。','',
        '### 验证分组定义（不能混用）','']
    report+=table(['分组','有效类别数','验证图像数','定义'],[[name,s['classes'],s['samples'],('类别ID 0..35，但0无样本' if name=='Head' else '类别ID 36..70' if name=='OfficialMedium' else '类别ID 71..99' if name=='Tail' else '训练数量>100' if name=='Many' else '训练数量20..100' if name=='Medium' else '训练数量<20')] for name,s in groupsummary.items()])
    report+=['','Overall 是1980张图像的 micro accuracy；每类恰好20张，所以等于99类 Macro。WorstClass 是这99类的最小准确率，不能表述为包含缺失class0的100类最差结果。','',
        '## 3. 四臂与逐 batch 配对','',
        '- DR-NoFusion：只将原接收特征送入FC，不执行替换；为配对审计仍计算相同供体特征、评分和全部掩码。',
        '- Original-BatchShared：每个真实minibatch共享一个随机16/64完整8×8通道掩码。',
        '- Uniform-PerSample：每样本独立随机16/64完整通道。',
        '- LowVS-PerSample：供体类统计的 Bottom48 池中随机取16个完整通道；采用同一逐样本Gumbel、相同fallback。接收者标签始终不变。',
        '- 四臂共享每seed的BS类别均匀接收序列、IS每epoch无放回全实例排列、隔离的crop/flip增强流、逐样本与batch共享Gumbel。每seed先DR建立750行基准；另三臂在更新前比较联合哈希，报告时再次逐行核对，总计4,500行非DR配对全部一致。',
        '- 联合哈希实际包含：接收/供体索引、两套增强像素、两套冻结特征、逐样本/共享Gumbel及三个掩码。评分本身未另列独立哈希；其全部输入（GAP、供体标签、同一锁定统计和代码）已固定，不可声称CSV单独存有评分哈希字段。',
        '- 每epoch比较完整Backbone state hash，包含Conv、BN affine、running_mean/var和num_batches_tracked；所有参数被冻结且特征提取在no_grad中。31个BN计数均保持Stage I结束时15,000。每组FC的weight与bias均有变化。',
        '- 每臂保存final_fc.pth、10行epochs.csv、750行paired_batches.csv、1980×100 logits；训练脚本重新加载final FC后验证全部预测和指标一致。额外报告脚本只核对已有数组和权重，不执行推理或训练。','',
        '## 4. 固定第10 epoch主结果（%）','']
    report+=table(['seed','策略',*METRICS],[[s,a,*[f'{float(by[s,a][k]):.4f}' for k in METRICS]] for s in SEEDS for a in ARMS])
    report+=['','新 Stage I 固定epoch200的描述性验证：'+', '.join(f'{k}={float(s1["validation_epoch200"][k]):.4f}%' for k in METRICS)+'。该结果不参与checkpoint选择。','',
        '## 5. 配对准确率差与净得失','',
        '以下均为 A−B，单位pp。两seed取算术平均仅供描述；同一1980张验证图像重复出现，不能将3960次预测当作独立观测，也不作显著性检验。','']
    report+=table(['seed','A − B','Overall pp','Head pp','Tail pp','wrong→right','right→wrong','净正确数','Tail净正确数','Tail→Head Δ','Head→Tail Δ'],
      [[s,a+' − '+b,f'{float(by[s,a]["Overall"])-float(by[s,b]["Overall"]):+.4f}',f'{float(by[s,a]["Head"])-float(by[s,b]["Head"]):+.4f}',f'{float(by[s,a]["Tail"])-float(by[s,b]["Tail"]):+.4f}',pairby[s,a,b]['wins'],pairby[s,a,b]['losses'],pairby[s,a,b]['net'],pairby[s,a,b]['Tail_net'],pairby[s,a,b]['tail_true_to_head_pred_delta'],pairby[s,a,b]['head_true_to_tail_pred_delta']] for s in SEEDS for a,b in CONTRASTS])
    report+=['','### 两seed平均差（pp，n=2不代表显著）','']
    report+=table(['比较',*METRICS],[[r['comparison'],*[f'{r[k]:+.4f}' for k in METRICS]] for r in meanrows])
    report+=['','### 核心判读','']
    if candidate:
        report+=['Original-BatchShared 相对 DR 在两个seed的Tail均下降，按预声明规则标记为“IF200固定BN负迁移候选”。该标签只适用于本次单一Stage I和重复使用的验证集；即使两个seed同向，也不能据此宣布作者原始H2T失败。']
    else:
        report+=['Original-BatchShared 相对 DR 未在两个seed均出现Tail下降，本轮未发现稳定的IF200固定BN负迁移。停止此单点失败假说，不追加IF50、外部数据或其他seed寻找负结果。']
    for a,b,title in [(ARMS[2],ARMS[1],'掩码粒度'),(ARMS[3],ARMS[2],'VS池身份'),(ARMS[3],ARMS[1],'混合对照')]:
        d=[float(by[s,a]['Tail'])-float(by[s,b]['Tail']) for s in SEEDS]
        o=[float(by[s,a]['Overall'])-float(by[s,b]['Overall']) for s in SEEDS]
        sign='Tail跨seed变号' if d[0]*d[1]<0 else 'Tail同向' if d[0]*d[1]>0 else 'Tail至少一seed持平'
        overall_sign='Overall跨seed变号' if o[0]*o[1]<0 else 'Overall同向' if o[0]*o[1]>0 else 'Overall至少一seed持平'
        report+=[f'- {title}（{a}−{b}）：Tail {d[0]:+.4f}/{d[1]:+.4f}pp；Overall {o[0]:+.4f}/{o[1]:+.4f}pp；{sign}，{overall_sign}。'+('此比较同时改变粒度与评分，不能独立归因于VS。' if title=='混合对照' else '只能解释该固定BN、固定预算下的受控比较。')]
    lowbeatsoriginal=[float(by[s,ARMS[3]]['Tail'])>float(by[s,ARMS[1]]['Tail']) for s in SEEDS]
    if independent_vs and not all(lowbeatsoriginal):report+=['LowVS虽然两个seed的Tail均超过逐样本Uniform，但未同时超过OriginalShared，不能称稳定超越原H2T式对照。']
    elif not independent_vs:report+=['LowVS未在两seed的Tail均超过逐样本Uniform；本轮不支持稳定的Tail VS额外收益，不据此设计新门控。']
    else:report+=['本轮LowVS相对逐样本Uniform和OriginalShared的Tail均同向为正，但只有一套Stage I、两个Stage II seed，且需同时检查Head及Overall，不能宣称稳定泛化或替代作者完整方法。']
    report+=['','## 6. 类别得失与Head/Tail混淆','',
        '每类别只有20张，单张改判即该类5pp。以下只按事先定义的负净正确数列出两seed持续受损类别，不筛选有利类别证明结论。所有99类的绝对准确率见 `per_class_accuracy.csv`，全部比较见 `per_class.csv`。','']
    for a,b in CONTRASTS:
        lost=consistent[a+' - '+b+' / Tail'];alllost=consistent[a+' - '+b+' / All']
        report+=[f'### {a} − {b}','',f'两seed都受损的Tail类别：{len(lost)}个；全部99类中{len(alllost)}个。','']
        if lost:report+=table(['类别ID','seed124净正确数','seed125净正确数'],[[d['class'],d['net124'],d['net125']] for d in lost])
        else:report+=['无两seed均负净正确数的Tail类别。']
        report+=['']
    report+=['绝对Head/Tail互相混淆计数如下；预测class0归入Head，尽管验证真值无class0。Tail→Head是错误去向计数，不等于Tail的全部错误；三组完整混淆表见 `head_tail_confusion.csv`。','']
    report+=table(['seed','策略','Tail真→Head预测','Head真→Tail预测'],[[s,a,next(r['count'] for r in conf if r['seed']==s and r['arm']==a and r['true_group']=='Tail' and r['predicted_group']=='Head'),next(r['count'] for r in conf if r['seed']==s and r['arm']==a and r['true_group']=='Head' and r['predicted_group']=='Tail')] for s in SEEDS for a in ARMS])
    report+=['','精度分辨率：Overall一张=0.050505pp；Head一张=0.142857pp；Tail一张=0.172414pp；类别内一张=5pp。小净变化不能夸大，最差类别为零也不能仅靠这一指标归因于融合。','',
        '## 7. 失败、修复与可复查证据','',
        '- 静态审查发现 cfg() 用 Path + str 拼接YAML路径；只对文件名加括号，不改任何配置值。',
        '- 首次preflight在导入前SyntaxError退出：报告两处Head/Tail混淆计数表达式缺失右括号。修复括号后全部新Python源通过AST解析；第二次preflight通过。失败日志 `console/preflight.log`、首次 `FAILED_STOP.json` 和 `preflight_repair.json` 原样保留；未删除/覆盖失败输出，也没有失败训练或恢复训练。',
        '- 原guard只匹配一条TEST绝对路径且无历史写保护；新增执行期审计防护，拦截任何cifar-100-python/test打开及工作区旧文件写操作，并保存各步骤状态、时间、显存和完整stdout。未放松数据、哈希、冻结或配对条件。',
        '- 一次只读rg搜索使用Windows不支持的字面通配目录而报错，后续改为真实路径。该异常未运行任何实验；已记入静态审查。',
        '- 原作者IF200 YAML的name字符串仍含imb001，但实际imb_factor为0.005，且9502张raw indices与官方seed0 IF200逐项相符。本轮按真实配置值、数据数量和索引判断IF；未修改历史YAML显示名称。',
        '- `blocking_fixes.patch` 与 `provided_source/` 保存新脚本的修复差异和原提供版本。`policy.py`与原提供文件逐字节一致；没有改变rho、池规模、供体分布、损失、学习率、训练预算或Backbone结构。',
        f'- 全部正式训练成功，未OOM、未出现配对不一致；{len(guards)}个成功主脚本进程的TEST/历史写拦截事件均为0。{len(read(OUT/"historical_manifest.json"))}个相关历史文件前后SHA256相同。源文件锁见 `execution_source_lock.json`；最终全文件校验见 `delivery_manifest.json`。','',
        '## 8. 因果解释和停止条件','',
        '- B−A识别本IF200固定模型/BN/预算下加入共享掩码特征替换的整体影响；不是跨IF因果效应，也不是原作者shift_bn=True策略的检验。',
        '- C−B识别本面板中掩码粒度改变；D−C识别既定VS池选择相对逐样本随机的整体影响。D−C仍包含通道身份引起的替换幅度与方向共同变化，不能称已匹配范数后的纯方向因果效应。',
        '- 历史IF100使用不同训练子集、不同Stage I训练执行及验证选best epoch184；IF200本轮为新模型固定epoch200。不能直接比较绝对准确率，也不能把跨实验差异全部归因于IF。',
        '- 只训了一套Stage I seed42；两个Stage II seed在同一模型、同一验证图像上，不能称独立Stage I复现或统计显著。该99类开发验证集已多次复用，class0缺失，不能称官方100类TEST精度或最终独立泛化证据。',
        '- Backbone与BN固定，任何本轮提升均是冻结特征上的分类器训练结果，不能称Backbone表征提升。没有使用验证图像估计类别均值/方差/epsilon，也没有按验证结果调参或选Stage I epoch。',
        ('- 唯一后续核验方向（本轮不执行）：确认该负迁移候选能否在作者shift_bn=True设置及独立Stage I seed下复现；在此之前停止缺陷宣称和新模块开发。' if candidate else '- 本轮停止：没有稳定Tail负迁移时，不追加IF、seed或外部数据寻找失败条件，不启动新门控或新模型。'),
        '- 已完成指定一套200epoch Stage I和8组10epoch Stage II，报告后停止。未执行任何版本控制操作。','',
        '## 9. 原始结果与可复现入口','',
        '本报告相对目录为 `H2T_repro_audit/if200_stress/results/`。正式模型、缓存和统计均保留在本地该目录，不覆盖历史文件。','',
        '- `stage1/stage1_fixed_epoch200.pth`、`stage1/epochs.csv`、`stage1/summary.json`。',
        '- `stats.npz`、`stats_provenance.json`、`validation_maps.npy`、`if200_split_indices.npz`、`dataset_attestation.json`。',
        '- `seed{124,125}/{四臂}/epochs.csv`、`paired_batches.csv`、`final_fc.pth`、`validation_epoch10.npz`、`summary.json`。',
        '- `epoch10.csv`、`paired.csv`、`per_class.csv`、`mean_paired_differences.csv`、`per_class_accuracy.csv`、`head_tail_confusion.csv`。',
        '- `scientific_summary.json`、`scientific_verification.json`、`classifier_state_verification.csv`、`execution_registry.json`、`guards/*.json`、`console/*.log`。',
        '- 从项目根目录使用指定Python，严格按 `user_task_instructions.txt` 命令顺序执行：policy.py → preflight → stage1 → prepare → seed124四臂（DR先）→ seed125四臂（DR先）→ report。既有results拒绝覆盖；此处只是记录本轮命令，不表示可重复覆盖封存输出。',
        '- 本轮完整命令、起止时间、退出码及所有日志路径见 `execution_registry.json`；日志封装 `execute_step.py` 只传递这些原始命令，不实现训练方法。']
    reportpath=OUT/'IF200_STRESS_SCIENTIFIC_REPORT.md'
    with reportpath.open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
    verify()
    verification={'passed':True,'finished_at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'dataset_attested':True,'StageI_epochs':200,'StageI_steps':15000,'StageII_runs':8,'StageII_epochs_per_run':10,
        'StageII_steps_per_run':750,'paired_nonDR_batches_identical':4500,'saved_logits_shape_per_run':[1980,100],
        'source_and_historical_hashes_unchanged':True,'historical_files_verified':len(read(OUT/'historical_manifest.json')),
        'guard_processes':len(guards),'official_TEST_events':0,'historical_write_events':0,
        'StageI_sha256':s1['checkpoint_sha256'],'backbone_state_tensors':len(ck['backbone']),
        'BN_modules':31,'BN_counter_frozen_at':15000,'classifier_parameters':6500,
        'class_group_denominators':groupsummary,'persistent_lost_classes':consistent,'mean_paired_differences':meanrows,
        'negative_transfer_candidate_fixed_BN':candidate,'two_StageII_seeds_not_significance':True,
        'initial_preflight_syntax_failure_preserved':True,'no_training_failed_or_retried':True,
        'official_original_recipe_run':False,'no_additional_experiments':True,'no_version_control_operations':True,
        'max_process_peak_allocated_MiB':max_peak,'scientific_report_sha256':sha(reportpath)}
    dump(OUT/'scientific_verification.json',verification)
    dump(OUT/'STOP_COMPLETED.json',{'status':'complete and stopped','additional_training':False,'TEST_used':False,'version_control_operations':False})
    lockfiles={p.relative_to(WORK).as_posix():sha(p) for p in here_source_files()}
    dump(OUT/'final_analysis_source_lock.json',lockfiles)
    files={}
    for p in OUT.rglob('*'):
        if p.is_file() and p.name!='delivery_manifest.json':files[p.relative_to(HERE).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
    dump(OUT/'delivery_manifest.json',{'status':'complete and stopped','scientific_report':'results/IF200_STRESS_SCIENTIFIC_REPORT.md','files':files})
    print(json.dumps({'report':str(reportpath),'candidate_negative_transfer':candidate,'mean_paired_differences':meanrows,'files_sealed':len(files)},ensure_ascii=False),flush=True)

def here_source_files():
    return list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+list(HERE.glob('*.txt'))
if __name__=='__main__':main()
