"""Within-recipient geometry/utility audit; no training or fitted selector."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.stats import rankdata
import audit_h2t_transfer_oracle as base
from audit_h2t_real_shuffled_holdout import evaluate


def normalized(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def main():
    here = Path(__file__).resolve().parent
    out = here/'outputs/h2t_donor_predictability_seed42'
    out.mkdir(parents=True, exist_ok=True)
    prior = here/'outputs/h2t_transfer_oracle_seed42_rho025'
    holdout = here/'outputs/h2t_real_shuffled_holdout_seed42'
    manifest = json.loads((prior/'transfer_manifest.json').read_text(encoding='utf-8'))
    checkpoint = Path(manifest['checkpoint'])
    assert base.sha256(checkpoint) == manifest['checkpoint_sha256']
    args = SimpleNamespace(data_root=base.workspace_root()/'H2T/data/cifar100',
                           limit_recipients=None, batch_size=256, workers=0)
    base.set_seed(20260916)
    model,classifier,_ = base.load_model(checkpoint)
    for module in (model,classifier):
        module.requires_grad_(False)
    before = {f'{k}.{name}': t.cpu().clone() for k,m in enumerate((model,classifier)) for name,t in m.state_dict().items()}
    tail,head = base.make_datasets(args)
    recipients = base.extract_feature_bank(base.make_loader(tail,args),model,classifier,'Few test')
    donors = base.extract_feature_bank(base.make_loader(head,args),model,classifier,'Head train')
    # Deterministic, original long-tailed training subset, including Few classes.
    all_train = base.IndexedSubset(head.base,np.arange(len(head.base)))
    train = base.extract_feature_bank(base.make_loader(all_train,args),model,classifier,'prototype train')
    train_vectors = train.features.mean((2,3)).cpu().numpy()
    prototypes = np.stack([train_vectors[train.labels==c].mean(0) for c in range(100)])
    counts = np.bincount(train.labels,minlength=100)
    rec = normalized(recipients.features.mean((2,3)).cpu().numpy())
    don = normalized(donors.features.mean((2,3)).cpu().numpy())
    proto = normalized(prototypes)
    weights = normalized(classifier.module.fc.weight.detach().cpu().numpy())
    protocol = np.load(holdout/'protocol_arrays.npz')
    pools = protocol['donor_pool_rows']
    assert np.array_equal(protocol['recipient_test_indices'],recipients.indices)
    n,d = pools.shape
    old = np.full((n,d),np.nan,dtype=np.float32)
    with (prior/'pair_records.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            i,j = int(row['recipient_row']),int(row['donor_slot'])
            assert pools[i,j] == int(row['donor_pool_row'])
            old[i,j] = float(row['mean_fused_true_margin'])
    assert np.isfinite(old).all()
    oracle = old.argmax(1)  # Only original selection masks, never evaluation utility.
    y = recipients.labels
    dy = donors.labels[pools]
    scores = {
        'Feature cosine':np.einsum('nc,ndc->nd',rec,don[pools]),
        'Class-center cosine':np.einsum('nc,ndc->nd',proto[y],proto[dy]),
        'Recipient-to-Head center':np.einsum('nc,ndc->nd',rec,proto[dy]),
        'Head-to-Tail center':np.einsum('nc,ndc->nd',proto[y],don[pools]),
        'Classifier geometry':np.einsum('nc,ndc->nd',weights[y],weights[dy]),
    }
    margins = np.zeros((n,d)); correct = np.zeros((n,d),bool); single = np.zeros((n,d))
    with torch.inference_mode():
        for i in range(n):
            features = donors.features[torch.as_tensor(pools[i],device='cuda')]
            margins[i],correct[i],single[i] = evaluate(recipients.features[i:i+1],features,y[i],
                [120260916+i*10+k for k in range(10)],classifier)
            if (i+1)%500 == 0 or i+1==n:
                print(f'Utility evaluated {i+1}/{n}',flush=True)
    utility = margins-recipients.margins[:,None]
    np.savez_compressed(out/'pair_arrays.npz',utility=utility,heldout_margins=margins,
        heldout_correct=correct,heldout_single_mask_correct=single,donor_pool_rows=pools,
        recipient_labels=y,recipient_test_indices=recipients.indices,train_prototypes=prototypes,
        train_class_counts=counts,selection_oracle_slots=oracle,**{f'score_{k}':v for k,v in scores.items()})
    clean = recipients.predicted==y
    rowids = np.arange(n)
    # Exact maximum-score ties are averaged uniformly, with no utility-based tie breaking.
    choices = {'Random':np.full((n,d),1/d)}
    choices.update({name:(s==s.max(1,keepdims=True))/np.sum(s==s.max(1,keepdims=True),axis=1,keepdims=True)
                    for name,s in scores.items()})
    oracle_weights = np.zeros((n,d)); oracle_weights[rowids,oracle] = 1
    choices['Oracle (5 selection masks)'] = oracle_weights
    rand = correct.mean()*100; ora = correct[rowids,oracle].mean()*100
    original = json.loads((holdout/'manifest.json').read_text())['summary'][0]
    assert abs(rand-original['random_few_acc_pct'])<1e-9
    assert abs(ora-original['oracle_few_acc_pct'])<1e-9
    selector_rows = []
    for name,w in choices.items():
        acc = (w*correct).sum(1)
        selector_rows.append(dict(selector=name,few_accuracy_pct=100*acc.mean(),
            mean_delta_margin=float((w*utility).sum(1).mean()),
            wrong_to_correct_pct=100*acc[~clean].mean(),
            correct_to_wrong_pct=100*(1-acc[clean]).mean(),
            oracle_gap_recovered_pct=(100*acc.mean()-rand)/(ora-rand)*100,
            single_mask_accuracy_pct=100*(w*single).sum(1).mean(),
            mean_number_of_tied_maxima=float((w>0).sum(1).mean())))
    base.save_summary_csv(selector_rows,out/'selector_performance.csv')
    urank = rankdata(utility,axis=1); urank -= urank.mean(1,keepdims=True)
    correlations = []; per_recipient = []; by_class = []
    rng = np.random.default_rng(20260919)
    for name,s in scores.items():
        rank = rankdata(s,axis=1); rank -= rank.mean(1,keepdims=True)
        denom = np.linalg.norm(rank,axis=1)*np.linalg.norm(urank,axis=1)
        rho = np.divide((rank*urank).sum(1),denom,out=np.full(n,np.nan),where=denom>0)
        cls = np.array([np.nanmean(rho[y==c]) for c in np.unique(y)])
        boot = rng.choice(cls,(10000,len(cls)),replace=True).mean(1)
        correlations.append(dict(score=name,mean_within_recipient_spearman=float(np.nanmean(rho)),
            median_spearman=float(np.nanmedian(rho)),p10=float(np.nanquantile(rho,.1)),
            p90=float(np.nanquantile(rho,.9)),positive_recipient_pct=float(np.nanmean(rho>0)*100),
            undefined_count=int(np.isnan(rho).sum()),class_bootstrap_mean_ci_low=float(np.quantile(boot,.025)),
            class_bootstrap_mean_ci_high=float(np.quantile(boot,.975))))
        per_recipient.extend(dict(score=name,recipient_test_index=int(recipients.indices[i]),label=int(y[i]),
                                  spearman=float(rho[i])) for i in range(n))
        by_class.extend(dict(score=name,label=int(c),mean_spearman=float(v)) for c,v in zip(np.unique(y),cls))
    base.save_summary_csv(correlations,out/'donor_predictability.csv')
    base.save_summary_csv(per_recipient,out/'within_recipient_correlations.csv')
    base.save_summary_csv(by_class,out/'class_correlations.csv')
    with (out/'pair_utility_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['recipient_test_index','recipient_label','donor_slot','donor_train_subset_index','donor_label','utility',*scores])
        for i in range(n):
            for j in range(d):
                writer.writerow([int(recipients.indices[i]),int(y[i]),j,int(donors.indices[pools[i,j]]),
                                 int(dy[i,j]),utility[i,j],*[s[i,j] for s in scores.values()]])
    for k,m in enumerate((model,classifier)):
        for name,value in m.state_dict().items():
            assert torch.equal(before[f'{k}.{name}'],value.cpu())
    lines = ['# Donor Utility Predictability 审计（seed42）','',
        '同一冻结 checkpoint、2900 个 Few recipient、每人原 50 个真实 Head donor。U 为独立 10 evaluation masks 的平均真实标签 margin 减 clean margin。复现 Random=24.1110%、原 5-mask 选择 Oracle=41.1379%。',
        '特征为 H2T 融合层经过全局平均池化后的 64 维向量（与 classifier 输入一致）。prototype 是原长尾训练子集的各类均值，未使用测试样本拟合。',
        '五个分数方向预先固定为越大越好；未根据测试 utility 翻转方向、拟合组合或挑参数。最高分完全并列时均匀平均其选择效果。','',
        '| 分数 | mean within-recipient Spearman | median | 正相关 recipient 比例 |',
        '|---|---:|---:|---:|']
    for r in correlations:
        lines.append(f"| {r['score']} | {r['mean_within_recipient_spearman']:.4f} | {r['median_spearman']:.4f} | {r['positive_recipient_pct']:.2f}% |")
    lines += ['', '| 选择规则 | Few Acc | Mean Δm | 错→对 (%) | 对→错 (%) | Oracle gap recovery |',
              '|---|---:|---:|---:|---:|---:|']
    for r in selector_rows:
        lines.append(f"| {r['selector']} | {r['few_accuracy_pct']:.2f}% | {r['mean_delta_margin']:+.4f} | {r['wrong_to_correct_pct']:.2f} | {r['correct_to_wrong_pct']:.2f} | {r['oracle_gap_recovered_pct']:.2f}% |")
    lines += ['', '错→对以 clean-wrong 为分母；对→错以 clean-correct 为分母。主 accuracy 为 10-mask 平均 logits 分类，Random 是候选池均匀期望。',
        'Oracle 仍只由原 5 selection masks 选择；没有用 10-mask utility 重选 Oracle。分数使用类别身份时采用真实类别，因此是宽松的诊断基线，不代表可直接部署测试推理。',
        '## 解释范围','',
        '相关性衡量各 recipient 内部的单分数单调排序关系。较弱结果不排除非线性关系、分数组合、其他简单几何或训练中有效性；较强结果也不等于证明语义因果机制。',
        '本轮只有 seed42；其他 checkpoint seeds 不自动扩展。原始 U 和五个 score 已保留，可复核每个 pair。']
    (out/'DONOR_PREDICTABILITY_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(dict(checkpoint_sha256=base.sha256(checkpoint),
        n_pairs=n*d,features='GAP 64D',prototype_source='original CIFAR100-LT IF100 train subset',
        model_state_unchanged=True,oracle_reproduced=True,correlations=correlations,selectors=selector_rows),indent=2),encoding='utf-8')
    print(json.dumps(dict(correlations=correlations,selectors=selector_rows),indent=2),flush=True)


if __name__ == '__main__':
    main()
