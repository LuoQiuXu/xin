"""Parameter-free center/boundary direction selection within the frozen Top10."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.stats import spearmanr
import audit_h2t_transfer_oracle as base


def cosine(a,b):
    denom=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.divide(np.sum(a*b,axis=-1),denom,out=np.zeros_like(denom),where=denom>1e-12),denom<=1e-12


def main():
    here=Path(__file__).resolve().parent
    out=here/'outputs/h2t_direction_selectors_seed42'; out.mkdir(parents=True,exist_ok=True)
    prev=here/'outputs/h2t_donor_predictability_seed42'
    topdir=here/'outputs/h2t_instance_top10_seed42'
    a=np.load(prev/'pair_arrays.npz'); top=np.load(topdir/'common_panel_arrays.npz')['top10_slots']
    manifest=json.loads((here/'outputs/h2t_transfer_oracle_seed42_rho025/transfer_manifest.json').read_text(encoding='utf-8'))
    checkpoint=Path(manifest['checkpoint']); assert base.sha256(checkpoint)==manifest['checkpoint_sha256']
    args=SimpleNamespace(data_root=base.workspace_root()/'H2T/data/cifar100',limit_recipients=None,batch_size=256,workers=0)
    base.set_seed(20260916)
    model,classifier,_=base.load_model(checkpoint)
    for m in (model,classifier): m.requires_grad_(False)
    states={f'{k}.{key}':v.cpu().clone() for k,m in enumerate((model,classifier)) for key,v in m.state_dict().items()}
    tail,head=base.make_datasets(args)
    rec=base.extract_feature_bank(base.make_loader(tail,args),model,classifier,'recipient')
    don=base.extract_feature_bank(base.make_loader(head,args),model,classifier,'donor')
    assert np.array_equal(rec.indices,a['recipient_test_indices'])
    y=rec.labels; n=len(y); rows=np.arange(n)
    f=rec.features.mean((2,3)).cpu().numpy()
    h=don.features.mean((2,3)).cpu().numpy()
    proto=a['train_prototypes']; w=classifier.module.fc.weight.detach().cpu().numpy()
    logits=rec.logits.cpu().numpy().copy(); logits[rows,y]=-np.inf
    competitor=logits.argmax(1)
    pool=a['donor_pool_rows']; topdon=np.take_along_axis(pool,top,1)
    direction=h[topdon]-f[:,None,:]  # Subtract raw GAP features, not unit-normalized vectors.
    center,z1=cosine(direction,(proto[y]-f)[:,None,:])
    boundary,z2=cosine(direction,(w[y]-w[competitor])[:,None,:])
    scores={'Class-aware Top1':np.take_along_axis(a['score_Head-to-Tail center'],top,1),
            'Center-direction':center,'Boundary-direction':boundary}
    u=np.take_along_axis(a['utility'],top,1); c=np.take_along_axis(a['heldout_correct'],top,1)
    single=np.take_along_axis(a['heldout_single_mask_correct'],top,1)
    clean=rec.predicted==y
    # Original select5 -> eval10 oracle choices, not a newly selected evaluation oracle.
    import csv
    oracle_slots=np.empty(n,dtype=int)
    with (topdir/'recipient_choices.csv').open(encoding='utf-8-sig',newline='') as handle:
        for r in csv.DictReader(handle):
            if r['method']=='Instance Oracle@10 (select5 -> eval10)':
                i=int(np.searchsorted(rec.indices,int(r['test_index'])))
                oracle_slots[i]=int(r['chosen_slot'])
    oracle_acc=a['heldout_correct'][rows,oracle_slots].mean()*100
    summary=[]; details=[]; baseline=None
    for name,s in scores.items():
        weights=(s==s.max(1,keepdims=True)).astype(float)
        weights/=weights.sum(1,keepdims=True)
        acc=(weights*c).sum(1); utility=(weights*u).sum(1)
        if baseline is None: baseline=acc.copy()
        diffs=acc-baseline
        classes=np.array([diffs[y==cls].mean() for cls in np.unique(y)])
        boots=np.random.default_rng(20260921).choice(classes,(10000,len(classes)),replace=True).mean(1)*100
        corrs=np.array([spearmanr(s[i],u[i]).statistic for i in rows])
        summary.append(dict(selector=name,few_acc_pct=float(acc.mean()*100),
            gap_to_oracle10_pp=float(oracle_acc-acc.mean()*100),mean_delta_margin=float(utility.mean()),
            minus_class_top1_pp=float(diffs.mean()*100),delta_class_bootstrap95_low=float(np.quantile(boots,.025)),
            delta_class_bootstrap95_high=float(np.quantile(boots,.975)),
            wrong_to_correct_pct=float(acc[~clean].mean()*100),correct_to_wrong_pct=float((1-acc[clean]).mean()*100),
            single_mask_acc_pct=float((weights*single).sum(1).mean()*100),
            mean_within_top10_spearman=float(np.nanmean(corrs)),
            changed_from_class_top1_pct=float((1-weights[:,0]).mean()*100),
            top1_became_correct_expected_count=float(((1-baseline)*acc).sum()),
            top1_became_wrong_expected_count=float((baseline*(1-acc)).sum())))
        for i in rows:
            chosen=int(s[i].argmax())
            details.append(dict(selector=name,recipient_test_index=int(rec.indices[i]),label=int(y[i]),
                competitor=int(competitor[i]),chosen_original_slot=int(top[i,chosen]),
                chosen_donor_pool_row=int(topdon[i,chosen]),accuracy_expectation=float(acc[i]),utility=float(utility[i])))
    summary.append(dict(selector='Instance Oracle@10 (select5 -> eval10)',few_acc_pct=float(oracle_acc),
        gap_to_oracle10_pp=0.,mean_delta_margin=float(a['utility'][rows,oracle_slots].mean())))
    assert abs(summary[0]['few_acc_pct']-35.06896551724138)<1e-8
    assert abs(oracle_acc-40.89655172413793)<1e-8
    for k,m in enumerate((model,classifier)):
        for key,v in m.state_dict().items(): assert torch.equal(states[f'{k}.{key}'],v.cpu())
    base.save_summary_csv(summary,out/'selector_summary.csv'); base.save_summary_csv(details,out/'recipient_choices.csv')
    np.savez_compressed(out/'direction_scores.npz',center=center,boundary=boundary,top10_slots=top,
        competitor=competitor,recipient_indices=rec.indices,center_zero_norm=z1,boundary_zero_norm=z2)
    checks=dict(checkpoint_sha256=manifest['checkpoint_sha256'],center_zero_norm_pairs=int(z1.sum()),
        boundary_zero_norm_pairs=int(z2.sum()),model_unchanged=True,summary=summary)
    (out/'manifest.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    lines=['# Top10 recipient 方向分数审计','',
        '同一 seed42 frozen H2T checkpoint；原 2900 Few recipient，每人原 class-aware Top10。无训练，无融合分数，无调参。',
        'f 为融合层 GAP 后原始 64 维向量，mu 为原长尾训练类中心。先求原始向量差再计算 cosine，未先单位归一化特征。',
        'Center = cos(f_h-f_t, mu_y-f_t)。Boundary = cos(f_h-f_t, W_y-W_j*)；j* 为 clean logits 中除真实 y 外最高分的竞争类。偏置用于 clean logits 决定 j*，方向 W_y-W_j* 不含偏置。',
        '所有规则在计算 utility 前即可选 donor。直接查用上一轮独立 10-mask 评估结果，Oracle 为原 select5→eval10 的 40.90%，不使用事后 eval10 oracle。','',
        '| 规则 | Few Acc | 与 Oracle@10 差距 | Mean Δmargin | 相对 class Top1 |',
        '|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['selector']} | {r['few_acc_pct']:.2f}% | {r['gap_to_oracle10_pp']:.2f}pp | {r['mean_delta_margin']:+.4f} | {r.get('minus_class_top1_pp',float('nan')):+.2f}pp |")
    lines+=['','完整 CSV 另含：按类别配对 bootstrap 差值区间、错转对/对转错、单 mask accuracy、Top10 内 Spearman、改选比例。类别 bootstrap 条件于当前 checkpoint/candidate pools，不能代替跨 seed 复现。',
        f'零范数 center pairs={z1.sum()}；boundary pairs={z2.sum()}。最高分并列均匀平均；零范数 cosine 定义为 0。',
        '该 Boundary 分数描述全向量移动方向；官方 H2T 实际只替换随机通道，因此不是对每个 mask 的精确 margin 增量预测。',
        'recipient 的真实类别用于 Top10 筛选、中心和边界方向，属于训练标签可用前提下的诊断；不代表无标签测试推理方案。']
    (out/'DIRECTION_SELECTORS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__': main()
