"""Three-level direction localization on frozen clean CIFAR-100 features."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from audit_h2t_transfer_oracle import load_model, eval_transform, set_seed, sha256
from swap_backbone_classifier import metrics, table, GROUPS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/h2t_direction_group_audit'
ARMS = {'None': (), 'Few': ('Few',), 'Medium': ('Medium',), 'Many': ('Many',),
        'Few+Medium': ('Few', 'Medium'), 'Few+Many': ('Few', 'Many'),
        'Many+Medium': ('Many', 'Medium'), 'All': ('Many', 'Medium', 'Few')}


def save(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def group(c):
    return 'Many' if c < 36 else 'Medium' if c < 71 else 'Few'


def angle(a, b):
    return float(torch.rad2deg(torch.acos(F.cosine_similarity(a.double()[None], b.double()[None]).clamp(-1, 1))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    torch.set_num_threads(4)
    dataset = CIFAR100(str(ROOT.parents[1] / 'H2T/data/cifar100'), train=False, download=False, transform=eval_transform())
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    rows, rotations, summaries, pairs, checks, interactions = [], [], [], [], [], []
    for seed in (42, 43, 44):
        paths = [ROOT / f'outputs/calibration_audit_official_h2t_seed{seed}/nofusion/last_epoch_fixed.pth.tar',
                 ROOT / f'outputs/calibration_audit_official_h2t_rho040_seed{seed}/h2t/last_epoch_fixed.pth.tar']
        m0, h0, c0 = load_model(paths[0])
        m1, h1, c1 = load_model(paths[1])
        assert c0['rho'] == 0 and c1['rho'] == .4 and c0['epoch'] == c1['epoch'] == 10
        assert c0['stage1_checkpoint_sha256'] == c1['stage1_checkpoint_sha256']
        assert all(torch.equal(v, c1['state_dict_model'][k]) for k, v in c0['state_dict_model'].items())
        w0, w1 = [h.module.fc.weight.detach().cpu() for h in (h0, h1)]
        b0, b1 = [h.module.fc.bias.detach().cpu() for h in (h0, h1)]
        r0 = w0.norm(dim=1, keepdim=True)
        u1 = w1 / w1.norm(dim=1, keepdim=True)
        features, labels = [], []
        with torch.inference_mode():
            for x, y in loader:
                f = m0(x.cuda())
                features.append(F.avg_pool2d(f, f.size(3)).flatten(1).cpu())
                labels.append(y.numpy())
        features = torch.cat(features)
        labels = np.concatenate(labels)
        weights, logits, predictions = {}, {}, {}
        for name, groups in ARMS.items():
            w = w0.clone()
            for g in groups:
                lo, hi = GROUPS[g]
                w[lo:hi] = (r0 * u1)[lo:hi]
            weights[name] = w
        weights['Full040'] = w1
        for name, w in weights.items():
            bias = b1 if name == 'Full040' else b0
            with torch.inference_mode():
                logits[name] = torch.cat([F.linear(f.cuda(), w.cuda(), bias.cuda()).cpu() for f in features.split(128)]).numpy()
            predictions[name] = logits[name].argmax(1)
            rows.append({'seed': seed, 'arm': name, **metrics(predictions[name], labels)})
        for name, idx in [('None', 0), ('Full040', 1)]:
            old = np.load(paths[idx].parent / 'clean_test_predictions.npz')
            assert np.array_equal(old['labels'], labels)
            assert np.array_equal(old['predicted'], predictions[name])
        previous = np.load(ROOT / f'outputs/h2t_module_swaps_0_vs_040/seed{seed}_factorial_predictions.npz')
        assert np.array_equal(previous['010'], predictions['All'])
        checks.append({'seed': seed, 'checkpoint_sha256': [sha256(p) for p in paths], 'baseline_full_and_direction_predictions_exact': True,
                       'max_swapped_norm_error': max(float((w.norm(dim=1)-r0[:, 0]).abs().max()) for n, w in weights.items() if n != 'Full040')})
        for c in range(100):
            take = labels == c
            acc0 = float((predictions['None'][take] == c).mean()*100)
            rotations.append({'seed': seed, 'class_id': c, 'group': group(c), 'rotation_deg': angle(w0[c], w1[c]),
                              'baseline_acc': acc0, 'full_delta_acc_pp': float((predictions['Full040'][take] == c).mean()*100-acc0),
                              'direction_delta_acc_pp': float((predictions['All'][take] == c).mean()*100-acc0)})
        for g in GROUPS:
            rr = [r for r in rotations if r['seed'] == seed and (g == 'Overall' or r['group'] == g)]
            summaries.append({'seed': seed, 'group': g, 'mean_rotation_deg': np.mean([r['rotation_deg'] for r in rr]),
                              'spearman_rotation_full_accuracy': float(spearmanr([r['rotation_deg'] for r in rr], [r['full_delta_acc_pp'] for r in rr]).statistic),
                              'spearman_rotation_direction_accuracy': float(spearmanr([r['rotation_deg'] for r in rr], [r['direction_delta_acc_pp'] for r in rr]).statistic)})
        # Complete group factorial: descriptive accuracy interactions, not fitted causal mediation.
        lookup = {r['arm']: r for r in rows if r['seed'] == seed}
        for a, b in [('Few', 'Many'), ('Few', 'Medium'), ('Many', 'Medium')]:
            name = a+'+'+b
            interactions.append({'seed': seed, 'groups': name, **{g: lookup[name][g]-lookup[a][g]-lookup[b][g]+lookup['None'][g] for g in GROUPS}})
        # Fixed baseline competitor, determined once before examining intervention outcomes.
        # Ties are resolved by smallest class ID and explicitly recorded.
        for y in range(71, 100):
            take = labels == y
            counts = np.bincount(predictions['None'][take], minlength=100)
            counts[y] = 0
            assert counts.max() > 0
            j = int(counts.argmax())
            base_margin = logits['None'][take, y] - logits['None'][take, j]
            for name in weights:
                mm = logits[name][take, y] - logits[name][take, j]
                pairs.append({'seed': seed, 'tail_class': y, 'competitor': j, 'competitor_group': group(j),
                              'baseline_confused_count': int(counts[j]), 'competitor_ties': int((counts == counts.max()).sum()),
                              'arm': name, 'angle_baseline_deg': angle(w0[y], w0[j]),
                              'angle_delta_deg': angle(weights[name][y], weights[name][j])-angle(w0[y], w0[j]),
                              'baseline_pair_margin': float(base_margin.mean()), 'pair_margin_delta': float((mm-base_margin).mean()),
                              'true_logit_delta': float((logits[name][take, y]-logits['None'][take, y]).mean()),
                              'competitor_logit_delta': float((logits[name][take, j]-logits['None'][take, j]).mean()),
                              'pair_margin_improved_fraction': float((mm > base_margin).mean()),
                              'confusion_change_pp': float(((predictions[name][take] == j).mean()-(predictions['None'][take] == j).mean())*100),
                              'class_accuracy_delta_pp': float(((predictions[name][take] == y).mean()-(predictions['None'][take] == y).mean())*100)})
        np.savez_compressed(OUT / f'seed{seed}_clean_logits.npz', labels=labels, **logits)
        print(f'seed {seed}: ' + str({n: round(lookup[n]['Few'], 3) for n in lookup}), flush=True)
    save('group_direction_swaps.csv', rows)
    save('class_rotation_accuracy.csv', rotations)
    save('rotation_group_summary.csv', summaries)
    save('group_interactions_pp.csv', interactions)
    save('tail_competitor_pairs.csv', pairs)
    pair_seed_summary = []
    for seed in (42, 43, 44):
        for name in weights:
            pp = [r for r in pairs if r['seed'] == seed and r['arm'] == name]
            pair_seed_summary.append({'seed': seed, 'arm': name,
                **{k: float(np.mean([r[k] for r in pp])) for k in ('angle_delta_deg', 'pair_margin_delta', 'true_logit_delta', 'competitor_logit_delta', 'confusion_change_pp')},
                'angle_increased_class_fraction': float(np.mean([r['angle_delta_deg'] > 1e-6 for r in pp])),
                'margin_improved_class_fraction': float(np.mean([r['pair_margin_delta'] > 1e-6 for r in pp]))})
    save('tail_competitor_seed_summary.csv', pair_seed_summary)
    (OUT/'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    report = '# H2T classifier 方向定位：三层审计\n\n'
    report += 'seed42/43/44，共享官方 Stage-I；固定 epoch10，rho0 vs 0.4。全程无训练、无 test fusion。Many=0–35，Medium=36–70，Few=71–99。\n\n'
    report += '## 第二层：分组方向干预（主要证据）\n\n固定 rho0 backbone、每行范数和 bias，仅交换指定类别行方向。Full040 为原完整 classifier 参照，不是方向单因素。额外补齐 Few+Many、Many+Medium，以免把缺失的联合组误判为无交互。\n\n'+table(rows, 'arm')
    report += '\n\n## 第一层：旋转幅度与类准确率（仅相关观察）\n\n|组|平均旋转角（度）|角度与完整模型 ΔAcc 的组内 Spearman（seed均值）|\n|---|---:|---:|\n'
    for g in ('Many', 'Medium', 'Few'):
        rr = [r for r in summaries if r['group'] == g]
        report += f"|{g}|{np.mean([r['mean_rotation_deg'] for r in rr]):.3f}|{np.mean([r['spearman_rotation_full_accuracy'] for r in rr]):.3f}|\n"
    report += '\n不能从角度大推断因果贡献大。完整模型 ΔAcc 同时包含范数和 bias 的变化，CSV 另存 direction-only 的相关性。\n\n'
    report += '## 第三层：固定主要竞争类的角度与 margin\n\n每个 Few 类取 baseline 误预测次数最多的类别；并列选最小 class ID，记录并列数。此处使用测试标签定位竞争类，只作描述，不用于训练或模型选择。表为 29 类 × 3 seed 的平均；不是 87 个独立复现。\n\n|干预|类间角度变化°|角度增大占比|pair margin变化|margin均值改善类占比|真实类logit变化|竞争类logit变化|误判到该竞争类变化pp|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for name in weights:
        pp = [r for r in pairs if r['arm'] == name]
        vals = [np.mean([r['angle_delta_deg'] for r in pp]), np.mean([r['angle_delta_deg'] > 1e-6 for r in pp])*100,
                np.mean([r['pair_margin_delta'] for r in pp]), np.mean([r['pair_margin_delta'] > 1e-6 for r in pp])*100,
                np.mean([r['true_logit_delta'] for r in pp]), np.mean([r['competitor_logit_delta'] for r in pp]), np.mean([r['confusion_change_pp'] for r in pp])]
        report += '|'+name+'|'+'|'.join(f'{v:.3f}' for v in vals)+'|\n'
    report += '\n角度增大不保证特征上的 margin 增大；margin 增大也不保证最终正确，因为其他竞争类仍可能胜出。Full040 margin 混有范数/bias 变化，应优先看方向干预。分组准确率干预是直接证据，竞争类分析不是独立验证的因果中介结论。\n\n'
    report += '## 核验与范围\n\n三 seed 的 baseline、完整 rho0.4 及 all-direction 逐样本预测均与上一轮完全一致。原始 checkpoint 未修改。分组结果和所有竞争对逐 seed 存 CSV，逐样本 logits 存 NPZ。本轮不扩展 rho0.1/0.25；三 seed 共用同一 Stage-I，不能当作独立 backbone 复现。\n'
    report += '\n## 当前结论与不能成立的解释\n\n'
    mean = {name: {g: float(np.mean([r[g] for r in rows if r['arm'] == name])) for g in GROUPS} for name in weights}
    report += f"1. Few 自身方向是主要来源：Few-only 相对 baseline 提升 {mean['Few']['Few']-mean['None']['Few']:.2f} pp，而 Many-only 为 {mean['Many']['Few']-mean['None']['Few']:.2f} pp。三个 seed 均如此。无需 Many 同时变化，Few-only 已有大幅提升，不能解释为必须 Head–Tail 协同才能起效。\n\n"
    report += f"2. Medium 方向存在组间竞争：Few-only 加入 Medium 后 Few 变化 {mean['Few+Medium']['Few']-mean['Few']['Few']:+.2f} pp；但 Medium accuracy 从 {mean['Few']['Medium']:.2f} 升到 {mean['Few+Medium']['Medium']:.2f}。不能因此把 Medium 方向称为整体有害。Few-only 的 Overall 也未超过完整模型。\n\n"
    report += '3. 不支持“Few 权重方向远离主要竞争类”：All-direction 的主要竞争对夹角平均缩小约 2 度，仅约 16% 增大。Few-only 也以夹角缩小为主。角度分离的预期链条未成立，应明确保留这个负结果。\n\n'
    report += '4. 支持的更窄解释：固定范数和 bias，Few-only 使真实类 logit 平均上升约 0.325，主要竞争对 margin 上升约 0.297；这意味着在所观测 Few 特征上，权重方向变化增加了真实类投影得分。它不是“与竞争类权重的角度越大越好”。此结论来自测试特征上的描述，不能直接推成新训练目标。\n\n'
    report += '下一步若继续，应验证该真实类投影/margin 变化能否随 rho 呈稳定关系，并保留 Many/Medium 代价；不应预设“转离竞争类”作为新方法目标。\n'
    (OUT/'DIRECTION_GROUPS_ZH.md').write_text(report, encoding='utf-8')
    print('Done: outputs/h2t_direction_group_audit/DIRECTION_GROUPS_ZH.md', flush=True)


if __name__ == '__main__':
    main()
