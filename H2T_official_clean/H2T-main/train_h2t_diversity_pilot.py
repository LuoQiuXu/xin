"""Two preregistered diversity arms, T=1 only; reuse previous completed controls."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import audit_h2t_calibration as recipe
import audit_h2t_transfer_oracle as audit
from train_h2t_selected_donors import train_arm


def main():
    here=Path(__file__).resolve().parent
    old=here/'outputs/h2t_donor_training_seed42'; out=here/'outputs/h2t_diversity_training_seed42'
    out.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(data_root=recipe.default_workspace_root()/'H2T/data/cifar100',
        stage1_checkpoint=here/'pretrained/cifar100_imb001_stage1_official.pth.tar',seed=42,epochs=10,
        batch_size=128,workers=0,rho=.25,lr=.1,lr_factor=.2,momentum=.9,weight_decay=5e-4,ece_bins=15)
    protocol=json.loads((old/'protocol.json').read_text(encoding='utf-8'))
    protocol.update(arms=['class_top10_uniform','boundary_soft_t1'],temperature=1.,score_std_correction=1,epsilon=1e-6,
        sampling_seed=420001,sampling_rng='independent CUDA torch.Generator; does not consume training RNG')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results={'original_h2t':json.loads((here/'outputs/calibration_audit_official_h2t_seed42/h2t/metrics.json').read_text())}
    for name in ('head_bank_random','class_aware','boundary_aware'):
        results[name]=json.loads((old/name/'metrics.json').read_text())
    reference=results['head_bank_random']
    for name in protocol['arms']:
        results[name]=train_arm(name,args,out)
        for key in ('training_stream_sha256','candidate_stream_sha256'):
            assert results[name][key]==reference[key],f'{name}: mismatch {key}'
    rows=[]
    for name,r in results.items():
        row=dict(method=name,**{g:r['clean_test'][g]['accuracy_pct'] for g in ('Overall','Many','Medium','Few')})
        if name in protocol['arms']:
            hist=r['history']; total=sum(e['few_selected'] for e in hist)
            for key in ('selection_entropy_nats','normalized_entropy','mean_effective_donors','mean_max_probability'):
                row[key]=sum(e[key]*e['few_selected'] for e in hist)/total
            audit.save_summary_csv(hist,out/f'{name}_epoch_log.csv')
        rows.append(row)
    audit.save_summary_csv(rows,out/'clean_inference_summary.csv')
    lines=['# H2T donor 多样性训练验证：seed42','',
        '仅新增 Class-Top10 Uniform 与 Boundary Soft T=1 两组，均从同一官方 Stage-I checkpoint 开始完成 Stage-II 10 epochs。原四组复用已完成结果。',
        '其他训练配置沿用上一轮：Few-only selector、50 Head candidates、class-aware Top10、epoch 刷新的增强特征 bank、rho=.25、batch128、CE、SGD、原 cosine LR。测试为 clean inference，无 donor/fusion。',
        'Boundary Soft 在每个 recipient 的 10 个 boundary 分数内减均值，除以样本标准差（correction=1）加 1e-6，然后 softmax(T=1)，真正 multinomial 抽一个 donor。Uniform 的 10 项概率均为 0.1。未搜索温度。',
        '抽 donor 使用独立 CUDA RNG；两组训练图像/增强序列及候选池 SHA256 均与上一轮 Head-bank Random 完全一致；最终 backbone 参数及 BN buffers 逐项匹配官方 baseline 训练终点。','',
        '| Stage-II | Overall | Many | Medium | Few |','|---|---:|---:|---:|---:|']
    for r in rows: lines.append('| '+r['method']+' | '+' | '.join(f'{r[g]:.2f}' for g in ('Overall','Many','Medium','Few'))+' |')
    lines+=['','| 采样方法 | H(p), nats | H/log(10) | mean exp(H) | mean max(p) |','|---|---:|---:|---:|---:|']
    for r in rows[-2:]:
        lines.append('| '+r['method']+' | '+' | '.join(f'{r[k]:.4f}' for k in ('selection_entropy_nats','normalized_entropy','mean_effective_donors','mean_max_probability'))+' |')
    lookup={r['method']:r for r in rows}
    lines+=['','| 对比 | ΔOverall | ΔFew |','|---|---:|---:|']
    for a,b in [('class_top10_uniform','head_bank_random'),('boundary_soft_t1','class_top10_uniform'),
                ('boundary_soft_t1','boundary_aware'),('class_top10_uniform','class_aware')]:
        lines.append(f"| {a} − {b} | {lookup[a]['Overall']-lookup[b]['Overall']:+.2f} | {lookup[a]['Few']-lookup[b]['Few']:+.2f} |")
    lines+=['','选择熵是给定候选池下的条件随机性，不等于全训练过程 donor 身份覆盖率；原 Top1 也会随着候选池重抽和状态变化选到不同 donor。提高熵同时改变所选分数/难度分布，因此性能变化不能单独归因为纯粹多样性因果效应。',
        '本轮仅 seed42、固定 T=1；没有追加温度扫描或其它 seed。所有新组保存最后 epoch checkpoint、optimizer、逐 epoch entropy 日志及 clean predictions。']
    (out/'DIVERSITY_TRAINING_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__': main()
