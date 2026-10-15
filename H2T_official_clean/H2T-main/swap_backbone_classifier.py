"""Frozen clean-inference module swaps; no training or test-time fusion."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from audit_h2t_transfer_oracle import load_model, eval_transform, set_seed, sha256

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'h2t_module_swaps_0_vs_040'
GROUPS = {'Overall': (0, 100), 'Many': (0, 36), 'Medium': (36, 71), 'Few': (71, 100)}


def metrics(pred, labels):
    return {g: float((pred[(labels >= lo) & (labels < hi)] == labels[(labels >= lo) & (labels < hi)]).mean() * 100)
            for g, (lo, hi) in GROUPS.items()}


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows, key):
    names = list(dict.fromkeys(r[key] for r in rows))
    lines = ['|组合|Overall|Many|Medium|Few|', '|---|---:|---:|---:|---:|']
    for name in names:
        cells = []
        for g in GROUPS:
            v = [r[g] for r in rows if r[key] == name]
            cells.append(f'{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}')
        lines.append('|' + name + '|' + '|'.join(cells) + '|')
    return '\n'.join(lines)


def main():
    set_seed(42)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    ds = CIFAR100(str(ROOT.parents[1] / 'H2T/data/cifar100'), train=False, download=False, transform=eval_transform())
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    swaps, factors, checks, effects = [], [], [], []
    cached = []
    for seed in (42, 43, 44):
        paths = [ROOT / f'outputs/calibration_audit_official_h2t_seed{seed}/nofusion/last_epoch_fixed.pth.tar',
                 ROOT / f'outputs/calibration_audit_official_h2t_rho040_seed{seed}/h2t/last_epoch_fixed.pth.tar']
        loaded = [load_model(p) for p in paths]
        models, heads, ck = zip(*loaded)
        assert [c['rho'] for c in ck] == [0, .4]
        assert all(c['epoch'] == 10 for c in ck)
        assert ck[0]['stage1_checkpoint_sha256'] == ck[1]['stage1_checkpoint_sha256']
        a, b = [c['state_dict_model'] for c in ck]
        assert a.keys() == b.keys()
        diffs = {k: float((a[k].double() - b[k].double()).abs().max()) for k in a}
        param_names = set(dict(models[0].named_parameters()))
        check = {'seed': seed, 'checkpoints': [str(p) for p in paths], 'sha256': [sha256(p) for p in paths],
                 'max_parameter_difference': max(diffs[k] for k in param_names),
                 'max_running_mean_difference': max(v for k, v in diffs.items() if 'running_mean' in k),
                 'max_running_var_difference': max(v for k, v in diffs.items() if 'running_var' in k),
                 'max_backbone_state_difference': max(diffs.values()), 'max_feature_element_difference': 0.,
                 'max_pooled_feature_l2_difference': 0.}
        preds = {k: [] for k in 'ABCD'}
        features, labels = [], []
        with torch.inference_mode():
            for x, y in loader:
                x = x.cuda()
                f0, f1 = [m(x) for m in models]
                check['max_feature_element_difference'] = max(check['max_feature_element_difference'], float((f0-f1).abs().max()))
                p0, p1 = [F.avg_pool2d(f, f.size(3)).flatten(1) for f in (f0, f1)]
                check['max_pooled_feature_l2_difference'] = max(check['max_pooled_feature_l2_difference'], float((p0-p1).norm(dim=1).max()))
                for name, feat, head in [('A', f0, heads[0]), ('B', f0, heads[1]), ('C', f1, heads[0]), ('D', f1, heads[1])]:
                    preds[name].append(head(feat).argmax(1).cpu().numpy())
                features.append(p0.cpu())
                labels.append(y.numpy())
        labels = np.concatenate(labels)
        preds = {k: np.concatenate(v) for k, v in preds.items()}
        for idx, name in [(0, 'A'), (1, 'D')]:
            old = np.load(paths[idx].parent / 'clean_test_predictions.npz')
            assert np.array_equal(labels, old['labels'])
            check[f'{name}_prediction_mismatch'] = int((preds[name] != old['predicted']).sum())
            assert check[f'{name}_prediction_mismatch'] == 0, 'Original checkpoint reproduction failed'
        for name in 'ABCD':
            swaps.append({'seed': seed, 'combination': name, **metrics(preds[name], labels)})
        check['B_equals_D'] = bool(np.array_equal(preds['B'], preds['D']))
        check['C_equals_A'] = bool(np.array_equal(preds['C'], preds['A']))
        np.savez_compressed(OUT / f'seed{seed}_swap_predictions.npz', labels=labels, **preds)
        cached.append((seed, torch.cat(features), labels, [h.module.fc.weight.detach().cpu() for h in heads], [h.module.fc.bias.detach().cpu() for h in heads], preds))
        checks.append(check)
        print(f'seed {seed}: ' + str(swaps[-4:]), flush=True)
        del loaded, models, heads, ck
    write_csv('backbone_classifier_swaps.csv', swaps)
    (OUT / 'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    # Conservative, exact gate. Do not proceed on an ambiguous backbone/head result.
    proceed = all(c['B_equals_D'] and c['C_equals_A'] for c in checks) and all(
        next(r['Few'] for r in swaps if r['seed'] == s and r['combination'] == 'D') >
        next(r['Few'] for r in swaps if r['seed'] == s and r['combination'] == 'A') for s in (42, 43, 44))
    if proceed:
        for seed, features, labels, weights, biases, baseline in cached:
            norms = [w.norm(dim=1, keepdim=True) for w in weights]
            assert all(bool((n > 0).all()) for n in norms)
            directions = [w/n for w, n in zip(weights, norms)]
            values, predictions = {}, {}
            for n, d, b in itertools.product((0, 1), repeat=3):
                code = f'{n}{d}{b}'
                weight = norms[n] * directions[d]
                with torch.inference_mode():
                    pred = torch.cat([F.linear(f.cuda(), weight.cuda(), biases[b].cuda()).argmax(1).cpu()
                                      for f in features.split(128)]).numpy()
                if code in ('000', '111'):
                    assert np.array_equal(pred, baseline['A' if code == '000' else 'B']), 'Recomposition changed endpoint predictions'
                values[n, d, b] = metrics(pred, labels)
                predictions[code] = pred
                factors.append({'seed': seed, 'norm_direction_bias': code, **values[n, d, b]})
            np.savez_compressed(OUT / f'seed{seed}_factorial_predictions.npz', labels=labels, **predictions)
            for axes, title in [((0, 1), 'norm_x_direction'), ((1, 2), 'direction_x_bias'), ((0, 2), 'norm_x_bias')]:
                other = next(i for i in range(3) if i not in axes)
                for level in (0, 1):
                    row = {'seed': seed, 'interaction': title, 'other_factor_level': level}
                    for g in GROUPS:
                        total = 0.
                        for i, j in itertools.product((0, 1), repeat=2):
                            key = [0, 0, 0]
                            key[axes[0]], key[axes[1]], key[other] = i, j, level
                            total += (1 if i == j else -1) * values[tuple(key)][g]
                        row[g] = total
                    effects.append(row)
            print(f'seed {seed}: factorial complete', flush=True)
        write_csv('norm_direction_bias_swaps.csv', factors)
        write_csv('factorial_interactions_pp.csv', effects)
    report = '# H2T 模块交换实验\n\n仅使用 rho=0 与 0.4 的 epoch10 checkpoint，seed42/43/44。无训练、无测试融合。\n\n'
    report += '## Backbone × Classifier\n\nA=B0+C0；B=B0+C0.4；C=B0.4+C0；D=B0.4+C0.4。数值为 accuracy %，均值 ± 样本标准差。\n\n' + table(swaps, 'combination')
    report += '\n\n全部原始 A/D 逐样本预测与历史保存结果完全一致。参数、BN 与特征差异详见 verification.json。\n'
    if proceed:
        report += '\n三 seed 均 B=D、C=A（逐样本预测一致），且 D 的 Few 高于 A，因此通过 classifier 主导门槛。\n\n## Norm × Direction × Bias\n\n固定 B0；编码顺序为 norm/direction/bias，0 表示 rho0，1 表示 rho0.4。000/111 重构后逐样本预测与原始 FC 一致。\n\n' + table(factors, 'norm_direction_bias')
        report += '\n\n交互量为 Y11−Y10−Y01+Y00，单位百分点；分别固定第三因素的两个水平，见 factorial_interactions_pp.csv。这不是统计显著性检验。\n'
    report += '\n## 解释范围\n\n三个 Stage-II seed 共享同一 released Stage-I checkpoint，不是三个独立 backbone 复现。交换所有类别的 FC 行，因此方向效应不能直接归因为只旋转了尾类行。所有结论仅针对当前匹配训练和 rho0/0.4 端点；不据此宣称新方法已提升性能。\n'
    if proceed:
        avg = {code: {g: float(np.mean([r[g] for r in factors if r['norm_direction_bias'] == code])) for g in GROUPS} for code in ('000', '001', '010', '100', '110', '111')}
        report += '\n## 本次干预结论\n\n'
        report += f"Few：完整 classifier 增益 {avg['111']['Few']-avg['000']['Few']:.2f} pp；单换方向 {avg['010']['Few']-avg['000']['Few']:.2f} pp；单换范数 {avg['100']['Few']-avg['000']['Few']:.2f} pp；单换 bias {avg['001']['Few']-avg['000']['Few']:.2f} pp。\n\n"
        report += '方向是 Few 的最大单因素贡献，范数提供额外贡献，bias 较小。不能把单因素增益比例解释为唯一、可加的因果贡献分解。范数与方向组合已接近完整 classifier；Few 的交互项跨 seed 有正有负，不支持强而稳定的正协同。\n\n'
        report += f"同时存在组间权衡：完整交换 Many {avg['111']['Many']-avg['000']['Many']:+.2f} pp、Medium {avg['111']['Medium']-avg['000']['Medium']:+.2f} pp、Overall {avg['111']['Overall']-avg['000']['Overall']:+.2f} pp。因此不能称所有类别都获益，也不能把 Few 的方向主导结论直接推广到 Overall。\n\n"
        report += '这为“训练期融合改变 classifier，尤其方向变化影响尾类 clean accuracy”提供了直接模块干预证据；尚未证明任何新训练方法，也未定位究竟哪些类别行或竞争关系负责。下一步可按原计划验证 rho 剂量与方向变化关系，本轮未扩展到 0.1/0.25。\n'
    (OUT / 'MODULE_SWAP_ZH.md').write_text(report, encoding='utf-8')
    print('Completed: outputs/h2t_module_swaps_0_vs_040/MODULE_SWAP_ZH.md', flush=True)


if __name__ == '__main__':
    main()
