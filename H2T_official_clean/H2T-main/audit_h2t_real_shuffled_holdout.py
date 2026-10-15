"""Paired real/shuffled Head oracle audit, with disjoint selection/evaluation masks."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import audit_h2t_transfer_oracle as base


def evaluate(recipient, donors, label, seeds, classifier):
    targets = torch.full((len(donors),), int(label), device=donors.device, dtype=torch.long)
    recipient = recipient.expand(len(donors), -1, -1, -1)
    logits = []
    for seed in seeds:
        fused = base.h2t_with_fixed_mask_seed(recipient, donors, .25, int(seed))
        logits.append(classifier(fused))
    logits = torch.stack(logits)
    margins = torch.stack([base.signed_true_margin(z, targets) for z in logits])
    return (margins.mean(0).cpu().numpy(),
            logits.mean(0).argmax(1).eq(targets).cpu().numpy(),
            logits.argmax(2).eq(targets[None]).float().mean(0).cpu().numpy())


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit-recipients', type=int)
    p.add_argument('--output-dir', type=Path, default=here/'outputs/h2t_real_shuffled_holdout_seed42')
    opts = p.parse_args()
    out = opts.output_dir
    out.mkdir(parents=True, exist_ok=True)
    prior = here/'outputs/h2t_transfer_oracle_seed42_rho025'
    manifest = json.loads((prior/'transfer_manifest.json').read_text(encoding='utf-8'))
    checkpoint = Path(manifest['checkpoint'])
    assert base.sha256(checkpoint) == manifest['checkpoint_sha256']
    args = SimpleNamespace(data_root=base.workspace_root()/'H2T/data/cifar100',
                           limit_recipients=None, batch_size=256, workers=0)
    base.set_seed(20260916)
    model, classifier, _ = base.load_model(checkpoint)
    for module in (model, classifier):
        module.requires_grad_(False)
        assert not module.training
    before = {f'{k}.{n}': v.detach().cpu().clone() for k,m in enumerate((model,classifier)) for n,v in m.state_dict().items()}
    datasets = base.make_datasets(args)
    recipients, donors = [base.extract_feature_bank(base.make_loader(d,args),model,classifier,name)
                          for d,name in zip(datasets,('recipient','donor'))]
    if opts.limit_recipients:
        # Preserve extraction batch shapes to match the original CUDA numerics.
        recipients = base.FeatureBank(**{key: value[:opts.limit_recipients]
                                       for key,value in vars(recipients).items()})
    n = len(recipients.labels)
    # Read the actual prior candidate pools, rather than merely assuming RNG equivalence.
    pools = np.full((n,50),-1,dtype=np.int64)
    old_margins = np.full((n,50),np.nan,dtype=np.float32)
    with (prior/'pair_records.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            i,j = int(row['recipient_row']),int(row['donor_slot'])
            if i >= n:
                continue
            assert int(row['recipient_test_index']) == recipients.indices[i]
            assert int(row['recipient_label']) == recipients.labels[i]
            pool_row = int(row['donor_pool_row'])
            assert int(row['donor_train_index']) == donors.indices[pool_row]
            assert int(row['donor_label']) == donors.labels[pool_row]
            pools[i,j] = pool_row
            old_margins[i,j] = float(row['mean_fused_true_margin'])
    assert (pools >= 0).all()
    # One independent, fixed permutation per original donor, reused for all masks/recipients.
    rng = np.random.default_rng(20260917)
    channels = donors.features.shape[1]
    permutations = np.stack([rng.permutation(channels) for _ in donors.labels])
    np.savez_compressed(out/'protocol_arrays.npz',donor_pool_rows=pools,
                        channel_permutations=permutations,recipient_test_indices=recipients.indices)
    results = {name: {k:np.zeros(n) for k in ('random_acc','oracle_acc','random_delta','oracle_delta',
                 'random_single_mask_acc','oracle_single_mask_acc','selection_oracle_acc','selection_random_acc')}
               for name in ('Real Head','Shuffled Head')}
    records = []
    max_prior_error = 0.
    mask_collisions = 0
    with torch.inference_mode():
        for i in range(n):
            indices = pools[i]
            real = donors.features[torch.as_tensor(indices,device='cuda')]
            perm = torch.as_tensor(permutations[indices],device='cuda')
            shuffled = real.gather(1,perm[:,:,None,None].expand_as(real))
            if i == 0:
                # Exact preservation of values at each spatial position across channels.
                assert torch.equal(real.sort(dim=1).values, shuffled.sort(dim=1).values)
            selection = [20260916+i*5+k for k in range(5)]
            evaluation = [120260916+i*10+k for k in range(10)]
            # Independent seeds, plus an explicit check for identical channel sets.
            sets = []
            for seed in selection+evaluation:
                gen = torch.Generator().manual_seed(seed)
                sets.append(tuple(sorted(torch.randperm(channels,generator=gen)[:int(.25*channels)].tolist())))
            mask_collisions += len(set(sets[:5]) & set(sets[5:]))
            for name,features in (('Real Head',real),('Shuffled Head',shuffled)):
                sel_margin,sel_acc,_ = evaluate(recipients.features[i:i+1],features,recipients.labels[i],selection,classifier)
                chosen = int(sel_margin.argmax())  # Locked before any evaluation-mask results.
                ev_margin,ev_acc,ev_single = evaluate(recipients.features[i:i+1],features,recipients.labels[i],evaluation,classifier)
                if name == 'Real Head':
                    max_prior_error = max(max_prior_error,float(np.abs(sel_margin-old_margins[i]).max()))
                r = results[name]
                values = dict(random_acc=ev_acc.mean(),oracle_acc=ev_acc[chosen],
                    random_delta=ev_margin.mean()-recipients.margins[i],
                    oracle_delta=ev_margin[chosen]-recipients.margins[i],
                    random_single_mask_acc=ev_single.mean(),oracle_single_mask_acc=ev_single[chosen],
                    selection_oracle_acc=sel_acc[chosen],selection_random_acc=sel_acc.mean())
                for key,value in values.items():
                    r[key][i] = value
                records.append(dict(pool=name,recipient_row=i,test_index=int(recipients.indices[i]),
                    label=int(recipients.labels[i]),clean_correct=int(recipients.predicted[i]==recipients.labels[i]),
                    chosen_slot=chosen,chosen_donor_pool_row=int(indices[chosen]),
                    selection_mean_margin=float(sel_margin[chosen]),**{k:float(v) for k,v in values.items()}))
            if (i+1)%200 == 0 or i+1 == n:
                print(f'Completed {i+1}/{n} recipients',flush=True)
    assert mask_collisions == 0
    assert max_prior_error < 1e-5, max_prior_error
    for k,m in enumerate((model,classifier)):
        for name,value in m.state_dict().items():
            assert torch.equal(before[f'{k}.{name}'],value.cpu()), name
    base.save_summary_csv(records,out/'recipient_records.csv')
    summary = []
    for name,r in results.items():
        summary.append(dict(candidate_pool=name,random_few_acc_pct=100*r['random_acc'].mean(),
            oracle_few_acc_pct=100*r['oracle_acc'].mean(),
            oracle_minus_random_pp=100*(r['oracle_acc']-r['random_acc']).mean(),
            oracle_mean_delta_margin=r['oracle_delta'].mean(),random_mean_delta_margin=r['random_delta'].mean(),
            random_single_mask_acc_pct=100*r['random_single_mask_acc'].mean(),
            oracle_single_mask_acc_pct=100*r['oracle_single_mask_acc'].mean(),
            selection_oracle_acc_pct=100*r['selection_oracle_acc'].mean()))
    base.save_summary_csv(summary,out/'transfer_summary.csv')
    diff = results['Real Head']['oracle_acc']-results['Shuffled Head']['oracle_acc']
    # Paired class-cluster bootstrap: uncertainty across the observed 29 Few classes,
    # conditional on this checkpoint/pool/permutation; not across training seeds.
    class_diff = np.array([diff[recipients.labels==c].mean() for c in np.unique(recipients.labels)])
    boot = np.random.default_rng(20260918).choice(class_diff,(10000,len(class_diff)),replace=True).mean(1)*100
    ci = np.quantile(boot,[.025,.975]).tolist()
    checks = dict(n_recipients=n,checkpoint_sha256=base.sha256(checkpoint),
        selection_masks=5,evaluation_masks=10,selection_seed_base=20260916,evaluation_seed_base=120260916,
        shuffle_seed=20260917,mask_set_collisions=mask_collisions,max_prior_real_selection_margin_error=max_prior_error,
        model_state_unchanged=True,real_minus_shuffled_oracle_pp=float(diff.mean()*100),
        paired_class_bootstrap_95ci_pp=ci,summary=summary)
    (out/'manifest.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    lines = ['# Real Head / Shuffled Head：独立 mask 审计','',
        f'固定原 seed42、epoch10、rho=0.25 checkpoint；{n} 个相同 Few recipient，每个使用原来的 50 个训练集 Head donors。',
        '每个 donor 的 64 个通道使用独立固定排列，完整保留数值集合和各通道内部空间结构，但破坏其与 recipient/classifier 通道坐标的对应。排列在选取和评估时保持一致。',
        '原 5 个 mask 的平均真实标签 margin 用于选择 donor；选完锁定 donor，另外 10 个共享独立 mask 用于评估。',
        '主准确率沿用上一轮口径：每个 donor 的评估 mask logits 平均后分类；Random 为 50 个 donor 分类正确率的均匀期望。Mean Δm 为 Oracle 在 10 个评估 mask 上平均 margin 减 clean margin。','',
        '| Candidate pool | Random Few Acc | Oracle Few Acc | Oracle−Random | Oracle Mean Δm |',
        '|---|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['candidate_pool']} | {s['random_few_acc_pct']:.2f}% | {s['oracle_few_acc_pct']:.2f}% | {s['oracle_minus_random_pp']:+.2f}pp | {s['oracle_mean_delta_margin']:+.4f} |")
    lines += ['',f'Clean Few Acc：{100*np.mean(recipients.predicted==recipients.labels):.2f}%。',
        f'Real−Shuffled Oracle：{diff.mean()*100:+.2f}pp；配对类别 bootstrap 95% 区间 [{ci[0]:+.2f}, {ci[1]:+.2f}]pp。此区间不包含训练 seed 或通道排列重抽的不确定性。','',
        '## 复核与解释边界','',
        f'原 Real Head 5-mask 选择 margin 最大复现误差：{max_prior_error:.8g}；选取/评估 channel mask 集合重复数：{mask_collisions}；所有模型参数和 BN buffers 前后完全相同。',
        '平均 logits 的分类结果与单 mask 平均 accuracy 并不相同。以下补充单 mask 平均，检验结论是否依赖 mask 集成：']
    for s in summary:
        lines.append(f"- {s['candidate_pool']}：Random {s['random_single_mask_acc_pct']:.2f}%，Oracle {s['oracle_single_mask_acc_pct']:.2f}%；原选择 mask 集成 Oracle {s['selection_oracle_acc_pct']:.2f}%。")
    lines += ['', 'Oracle 在选择阶段仍使用真实测试标签。此实验衡量固定候选池内标签可见选择对新 mask 的泛化；不是可部署方法，也不直接证明训练改进。',
        '通道打乱同时改变了通道与分类器权重的对齐、逐通道分布和融合后激活分布；即使 Real 显著占优，也只能支持有结构的原始 Head 表征优于此打乱对照，不能单独将差异全部归因为语义迁移。']
    (out/'REAL_SHUFFLED_HOLDOUT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


if __name__ == '__main__':
    main()
