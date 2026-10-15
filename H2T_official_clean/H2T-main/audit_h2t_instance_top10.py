"""Class-aware top-10 audit, with explicitly matched oracle protocols and donor identities."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau, rankdata
import audit_h2t_transfer_oracle as base
from audit_h2t_real_shuffled_holdout import evaluate


def main():
    here=Path(__file__).resolve().parent
    source=here/'outputs/h2t_donor_predictability_seed42'
    prior=here/'outputs/h2t_transfer_oracle_seed42_rho025'
    out=here/'outputs/h2t_instance_top10_seed42'
    out.mkdir(parents=True,exist_ok=True)
    a=np.load(source/'pair_arrays.npz')
    y=a['recipient_labels']; pools=a['donor_pool_rows']; u=a['utility']; correct=a['heldout_correct']
    score=a['score_Head-to-Tail center']; n=len(y); rows=np.arange(n)
    top=np.argsort(-score,axis=1,kind='stable')[:,:10]
    old=np.empty_like(u)
    with (prior/'pair_records.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            i,j=int(r['recipient_row']),int(r['donor_slot'])
            old[i,j]=float(r['mean_fused_true_margin'])
    top_u=np.take_along_axis(u,top,1)
    top_old=np.take_along_axis(old,top,1)
    selections={
        'Class-aware Top1':top[:,0],
        'Instance Oracle@10 (eval10 post-hoc)':top[rows,top_u.argmax(1)],
        'Global Oracle@50 (eval10 post-hoc)':u.argmax(1),
        'Instance Oracle@10 (select5 -> eval10)':top[rows,top_old.argmax(1)],
        'Global Oracle@50 (select5 -> eval10)':old.argmax(1),
    }
    baseline=correct[rows,selections['Class-aware Top1']]
    clean_margin=a['heldout_margins'][:,0]-u[:,0]
    clean=clean_margin>0
    summary=[dict(method='Clean no-fusion',few_acc_pct=float(clean.mean()*100),mean_delta_margin=0.)]
    per=[]
    for name,slots in selections.items():
        acc=correct[rows,slots]; delta=u[rows,slots]
        cls=np.array([(acc[y==c].astype(float)-baseline[y==c]).mean() for c in np.unique(y)])
        boot=np.random.default_rng(20260920).choice(cls,(10000,len(cls))).mean(1)*100
        summary.append(dict(method=name,few_acc_pct=float(acc.mean()*100),mean_delta_margin=float(delta.mean()),
            minus_top1_pp=float((acc.astype(float)-baseline).mean()*100),
            class_bootstrap_delta_low=float(np.quantile(boot,.025)),class_bootstrap_delta_high=float(np.quantile(boot,.975))))
        per.extend(dict(method=name,test_index=int(a['recipient_test_indices'][i]),label=int(y[i]),
            chosen_slot=int(slots[i]),chosen_donor_pool_row=int(pools[i,slots[i]]),correct=int(acc[i]),utility=float(delta[i])) for i in rows)
    base.save_summary_csv(summary,out/'top10_summary.csv')
    base.save_summary_csv(per,out/'recipient_choices.csv')
    # Ranking two different donor lists by position would not be a donor-preference correlation.
    donor_top=np.take_along_axis(pools,top,1)
    histogram=np.zeros(11,dtype=int); shared=[]
    for c in np.unique(y):
        ids=np.flatnonzero(y==c)
        for pos,i in enumerate(ids):
            for j in ids[pos+1:]:
                intersection,ii,jj=np.intersect1d(donor_top[i],donor_top[j],return_indices=True)
                k=len(intersection); histogram[k]+=1
                if k>=3:
                    v1,v2=top_u[i,ii],top_u[j,jj]
                    shared.append(dict(label=int(c),recipient_i=int(i),recipient_j=int(j),common_donors=k,
                        spearman=float(spearmanr(v1,v2).statistic),kendall=float(kendalltau(v1,v2).statistic)))
    base.save_summary_csv(shared,out/'original_pool_shared_donor_correlations.csv')
    print('Main summary:',json.dumps(summary),flush=True)
    print('Original pool overlap histogram:',histogram.tolist(),flush=True)
    # Controlled ranking panel: a common donor list and common masks within each class.
    # Choose the first recipient's class-aware Top10 before looking at any utility.
    manifest=json.loads((prior/'transfer_manifest.json').read_text(encoding='utf-8'))
    checkpoint=Path(manifest['checkpoint']); assert base.sha256(checkpoint)==manifest['checkpoint_sha256']
    args=SimpleNamespace(data_root=base.workspace_root()/'H2T/data/cifar100',limit_recipients=None,batch_size=256,workers=0)
    base.set_seed(20260916)
    model,classifier,_=base.load_model(checkpoint)
    for module in (model,classifier): module.requires_grad_(False)
    states={f'{k}.{key}':v.cpu().clone() for k,m in enumerate((model,classifier)) for key,v in m.state_dict().items()}
    tail,head=base.make_datasets(args)
    recipients=base.extract_feature_bank(base.make_loader(tail,args),model,classifier,'Few')
    donors=base.extract_feature_bank(base.make_loader(head,args),model,classifier,'Head')
    assert np.array_equal(y,recipients.labels)
    panel=np.empty((n,10)); panel_ids=np.empty((100,10),dtype=int); panel_ids.fill(-1)
    with torch.inference_mode():
        for c in np.unique(y):
            ids=np.flatnonzero(y==c); anchor=int(ids[0]); ds=donor_top[anchor]; panel_ids[c]=ds
            features=donors.features[torch.as_tensor(ds,device='cuda')]
            seeds=[120260916+anchor*10+k for k in range(10)]
            for i in ids:
                m,_,_=evaluate(recipients.features[i:i+1],features,y[i],seeds,classifier)
                panel[i]=m-recipients.margins[i]
            print(f'Common donor ranking panel: class {c}',flush=True)
    for k,m in enumerate((model,classifier)):
        for key,v in m.state_dict().items(): assert torch.equal(states[f'{k}.{key}'],v.cpu())
    corr=[]; paircorr=[]
    for c in np.unique(y):
        ids=np.flatnonzero(y==c); rr=rankdata(panel[ids],axis=1)
        mat=np.corrcoef(rr); vals=mat[np.triu_indices(len(ids),1)]
        tau=[]
        for pos,i in enumerate(ids):
            for q,j in enumerate(ids[pos+1:],pos+1):
                t=float(kendalltau(panel[i],panel[j]).statistic); tau.append(t)
                paircorr.append(dict(label=int(c),recipient_i=int(i),recipient_j=int(j),spearman=float(mat[pos,q]),kendall=t))
        winner=panel[ids].argmax(1); ct=np.bincount(winner,minlength=10)
        corr.append(dict(label=int(c),mean_spearman=float(np.nanmean(vals)),median_spearman=float(np.nanmedian(vals)),
            mean_kendall=float(np.nanmean(tau)),same_best_donor_pair_pct=float(np.sum(ct*(ct-1))/(len(ids)*(len(ids)-1))*100)))
    base.save_summary_csv(corr,out/'common_panel_class_correlations.csv')
    base.save_summary_csv(paircorr,out/'common_panel_pair_correlations.csv')
    np.savez_compressed(out/'common_panel_arrays.npz',utility=panel,donor_pool_rows_by_class=panel_ids,labels=y,top10_slots=top)
    meanrho=float(np.mean([r['mean_spearman'] for r in corr])); meantau=float(np.mean([r['mean_kendall'] for r in corr]))
    report=['# Class-aware Top1 与 Instance Oracle@10 审计','',
        'seed42 冻结 H2T rho=0.25；原 2900 Few recipient、每人原 50 Head donors。按训练集 Tail prototype cosine 排序取 Top10；未训练或拟合新选择器。','',
        '| 方法 | Few Acc | Mean Δmargin |', '|---|---:|---:|']
    for r in summary: report.append(f"| {r['method']} | {r['few_acc_pct']:.2f}% | {r['mean_delta_margin']:+.4f} |")
    report += ['', 'eval10 post-hoc 在这 10 个评估 masks 上既选择又评分，是图片要求的标签可见事后上限。select5 → eval10 保持此前的选择/评估分离协议。两种 Oracle 不混合比较；margin 最大的 donor 不保证平均 logits 的 accuracy 也最大。','',
        '## 同类 recipient 的 donor 排名','',
        f'原 Top10 交集大小 0..10 的 recipient-pair 数：{histogram.tolist()}。总 pair 数 {int(histogram.sum())}；至少 3 个共同 donor 的比例 {100*histogram[3:].sum()/histogram.sum():.4f}%。不能按各自 donor 列表的位置直接计算排名相关性。',
        '为获得可解释的比较，补充固定公共 donor 面板：每个 Tail 类预先取测试索引最小 recipient 的原 class-aware Top10，供同类全部 100 recipient 共同评估；所有人使用该 anchor 的同一组 10 evaluation masks，排除 donor 身份和 mask 差异。这是补充实验，不替换主实验的原候选池。',
        f'公共面板：平均同类 Spearman={meanrho:.4f}；平均 Kendall={meantau:.4f}。每类结果与每对样本结果已保存。',
        '公共面板每类只有一组 10 donor；其总体结论不能自动推广到所有候选池。原池稀疏交集相关性仅作为覆盖率审计，不据此推断总体偏好。','',
        '## 结果边界','',
        'Oracle gap 说明当前 class-aware 分数还未达到标签可见上限，单独不能证明缺失项一定是 instance-level 信息；也可能有更好的 class-level 排名。只有结合共享 donor/共享 mask 下的同类排序差异，才能判断样本差异是否明显。',
        '所有 accuracy 都是冻结模型的测试时干预诊断；真实标签仅用于定义类别、评价和 Oracle，尚无训练收益结论。']
    (out/'INSTANCE_TOP10_AUDIT_ZH.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    results=dict(summary=summary,original_overlap_histogram=histogram.tolist(),common_panel_mean_spearman=meanrho,
        common_panel_mean_kendall=meantau,common_panel_class_results=corr,checkpoint_sha256=manifest['checkpoint_sha256'])
    (out/'manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__': main()
