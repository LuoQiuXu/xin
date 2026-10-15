"""Fixed three-seed replication and competitor-direction ablation, no training."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import audit_h2t_transfer_oracle as base
from audit_h2t_real_shuffled_holdout import evaluate
from audit_h2t_direction_selectors import cosine


def main():
    here=Path(__file__).resolve().parent
    out=here/'outputs/h2t_direction_multiseed'; out.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(data_root=base.workspace_root()/'H2T/data/cifar100',limit_recipients=None,batch_size=256,workers=0)
    protocol=np.load(here/'outputs/h2t_real_shuffled_holdout_seed42/protocol_arrays.npz')
    pools=protocol['donor_pool_rows']; records=[]; competitor_records=[]; hashes={}
    for seed in (42,43,44):
        base.set_seed(20260916)
        ck=here/f'outputs/calibration_audit_official_h2t_seed{seed}/h2t/last_epoch_fixed.pth.tar'
        model,classifier,checkpoint=base.load_model(ck)
        assert checkpoint['rho']==.25 and checkpoint['epoch']==10
        hashes[str(seed)]=base.sha256(ck)
        for m in (model,classifier): m.requires_grad_(False)
        states={f'{k}.{key}':v.cpu().clone() for k,m in enumerate((model,classifier)) for key,v in m.state_dict().items()}
        tail,head=base.make_datasets(args)
        rec=base.extract_feature_bank(base.make_loader(tail,args),model,classifier,f'seed{seed} Few')
        don=base.extract_feature_bank(base.make_loader(head,args),model,classifier,f'seed{seed} Head')
        train=base.extract_feature_bank(base.make_loader(base.IndexedSubset(head.base,np.arange(len(head.base))),args),model,classifier,f'seed{seed} training prototypes')
        assert np.array_equal(rec.indices,protocol['recipient_test_indices'])
        f=rec.features.mean((2,3)).cpu().numpy(); h=don.features.mean((2,3)).cpu().numpy()
        tv=train.features.mean((2,3)).cpu().numpy()
        proto=np.stack([tv[train.labels==c].mean(0) for c in range(100)])
        y=rec.labels; n=len(y); rows=np.arange(n)
        w=classifier.module.fc.weight.detach().cpu().numpy()
        logits=rec.logits.cpu().numpy().copy(); logits[rows,y]=-np.inf; current=logits.argmax(1)
        # Most frequent highest-logit non-y competitor on TRAIN samples of each class.
        # Includes correctly classified samples; never chooses class competitor on test utility.
        tz=train.logits.cpu().numpy().copy(); tz[np.arange(len(tz)),train.labels]=-np.inf
        rivals=tz.argmax(1)
        clscomp=np.array([np.bincount(rivals[train.labels==c],minlength=100).argmax() for c in range(100)])
        for c in np.unique(y):
            competitor_records.append(dict(seed=seed,tail_class=int(c),train_class_competitor=int(clscomp[c]),
                train_count=int((train.labels==c).sum()),test_current_agreement_pct=float((current[y==c]==clscomp[c]).mean()*100)))
        classscore,_=cosine(h[pools],proto[y,None,:])
        top=np.argsort(-classscore,axis=1,kind='stable')[:,:10]
        td=np.take_along_axis(pools,top,1); displacement=h[td]-f[:,None,:]
        scores={'Class-aware':np.take_along_axis(classscore,top,1),
            'Center-direction':cosine(displacement,(proto[y]-f)[:,None,:])[0],
            'Boundary-direction':cosine(displacement,(w[y]-w[current])[:,None,:])[0],
            'Correct W_y':cosine(displacement,w[y,None,:])[0],
            'W_y-minus-mean-other':cosine(displacement,(w[y]-(w.sum(0)-w[y])/99)[:,None,:])[0],
            'Class-level competitor':cosine(displacement,(w[y]-w[clscomp[y]])[:,None,:])[0]}
        selected={name:s.argmax(1) for name,s in scores.items()}
        acc=np.zeros((n,10)); utility=np.zeros((n,10)); single=np.zeros((n,10))
        with torch.inference_mode():
            for i in rows:
                features=don.features[torch.as_tensor(td[i],device='cuda')]
                m,c,sm=evaluate(rec.features[i:i+1],features,y[i],[120260916+i*10+k for k in range(10)],classifier)
                acc[i]=c; utility[i]=m-rec.margins[i]; single[i]=sm
                if (i+1)%1000==0: print(f'seed {seed}: {i+1}/{n}',flush=True)
        classacc=acc[rows,selected['Class-aware']]
        for name,s in scores.items():
            weights=(s==s.max(1,keepdims=True)).astype(float); weights/=weights.sum(1,keepdims=True)
            accuracy=(weights*acc).sum(1)
            records.append(dict(seed=seed,rule=name,few_acc_pct=float(accuracy.mean()*100),
                mean_delta_margin=float((weights*utility).sum(1).mean()),single_mask_acc_pct=float((weights*single).sum(1).mean()*100),
                delta_class_pp=float((accuracy-classacc).mean()*100),
                current_competitor_disagreement_pct=float((selected[name]!=selected['Boundary-direction']).mean()*100)))
        if seed==42:
            expected={'Class-aware':35.06896551724138,'Center-direction':39.758620689655174,'Boundary-direction':42.10344827586207}
            for r in records:
                if r['rule'] in expected: assert abs(r['few_acc_pct']-expected[r['rule']])<1e-8, r
        for k,m in enumerate((model,classifier)):
            for key,v in m.state_dict().items(): assert torch.equal(states[f'{k}.{key}'],v.cpu())
        np.savez_compressed(out/f'seed{seed}_arrays.npz',top10_slots=top,top10_donor_rows=td,correct=acc,
            utility=utility,single_mask_accuracy=single,labels=y,current_competitor=current,class_competitor=clscomp,
            **{f'score_{name}':s for name,s in scores.items()})
        base.save_summary_csv(records,out/'all_results.csv')
        print(json.dumps([r for r in records if r['seed']==seed],indent=2),flush=True)
        del rec,don,train,model,classifier
        torch.cuda.empty_cache()
    base.save_summary_csv(competitor_records,out/'class_competitors.csv')
    aggregate=[]
    for name in scores:
        subset=[r for r in records if r['rule']==name]
        aggregate.append(dict(rule=name,mean_acc=float(np.mean([r['few_acc_pct'] for r in subset])),
            std_acc=float(np.std([r['few_acc_pct'] for r in subset],ddof=1)),
            mean_single_mask_acc=float(np.mean([r['single_mask_acc_pct'] for r in subset]))))
    base.save_summary_csv(aggregate,out/'aggregate.csv')
    lines=['# 三 seed 复现与竞争类别方向消融','',
        '已有 seed42/43/44 Stage-II H2T rho=0.25、epoch10 checkpoint；全部冻结，无训练或调参。相同 Few 测试样本、每人同一 50-donor pool；每个 checkpoint 单独计算训练 prototype，重新按 class cosine 筛选 Top10。共享同样 10 个 evaluation mask seeds。',
        'class-level competitor 定义为该 Tail 类训练样本的最高非真实类别 logit 对应类中，出现次数最多的类别；包括训练集中预测正确的样本，并列按类别 id 最小处理。仅用训练集，未利用测试集确定竞争类。',
        '所有方向均与原始 GAP 特征差 f_h-f_t 计算 cosine，最大者入选。W_mean 为除真实类别外 99 类权重均值。','',
        '| Seed | Class-aware | Center-direction | Boundary-direction |', '|---|---:|---:|---:|']
    for seed in (42,43,44):
        vals={r['rule']:r['few_acc_pct'] for r in records if r['seed']==seed}
        lines.append(f"| {seed} | {vals['Class-aware']:.2f} | {vals['Center-direction']:.2f} | {vals['Boundary-direction']:.2f} |")
    vals={r['rule']:r for r in aggregate}
    lines.append('| Mean±sample SD | '+' | '.join(f"{vals[k]['mean_acc']:.2f}±{vals[k]['std_acc']:.2f}" for k in ('Class-aware','Center-direction','Boundary-direction'))+' |')
    lines+=['','| Direction | seed42 | seed43 | seed44 | Mean±SD |','|---|---:|---:|---:|---:|']
    for name in ('Class-aware','Correct W_y','W_y-minus-mean-other','Class-level competitor','Boundary-direction'):
        rs=[r for r in records if r['rule']==name]; v=vals[name]
        lines.append('| '+name+' | '+' | '.join(f"{r['few_acc_pct']:.2f}" for r in rs)+f" | {v['mean_acc']:.2f}±{v['std_acc']:.2f} |")
    lines+=['','主表为 10-mask logits 集成 accuracy；完整结果另存单 mask 平均 accuracy。此处仅三个 Stage-II seeds，共享官方 Stage-I checkpoint、候选池与 mask，不代表三次独立 backbone 训练。',
        '三个 checkpoint 均检查全部参数及 BN buffers 无改变。seed42 三规则准确率严格复现上一轮。Oracle/测试 utility 未参与方向定义或 donor 排序；真实 recipient 标签用于类中心及正确类别方向，仍是诊断性干预。']
    (out/'MULTISEED_COMPETITOR_AUDIT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(dict(checkpoint_hashes=hashes,results=records,aggregate=aggregate),indent=2),encoding='utf-8')


if __name__=='__main__': main()
