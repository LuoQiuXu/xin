"""Verify two completed fixed-BN controls and render their mechanism audit."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import yaml

from bn_audit_helpers import check_sources, dump, protected_files, sha

BASE = Path(__file__).resolve().parent
OFFICIAL = BASE/'stage2_runs/official_20261011_1525'
METRICS = ['overall','head','medium','tail']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tensor_bytes(value):
    return value.detach().cpu().contiguous().numpy().tobytes()


def state_hash(state):
    h = hashlib.sha256()
    for key,value in state.items():
        h.update(key.encode())
        h.update(tensor_bytes(value))
    return h.hexdigest()


def compare(before, after):
    assert before.keys() == after.keys()
    result = {}
    for key, value in before.items():
        target = after[key]
        assert value.dtype == target.dtype and value.shape == target.shape
        assert torch.isfinite(target).all()
        equal = tensor_bytes(value) == tensor_bytes(target)
        result[key] = {'bitwise_equal':equal,'shape':list(value.shape),'dtype':str(value.dtype),
            'stage1_sha256':hashlib.sha256(tensor_bytes(value)).hexdigest(),
            'stage2_sha256':hashlib.sha256(tensor_bytes(target)).hexdigest(),
            'max_abs_delta':float((value.double()-target.double()).abs().max())}
    return {'changed_count':sum(not v['bitwise_equal'] for v in result.values()),'total_count':len(result),
        'max_abs_delta':max((v['max_abs_delta'] for v in result.values()),default=0.),'tensors':result}


def summarize(suite_id):
    folder = BASE/'bn_audit_runs'/suite_id
    preflight = read(folder/'preflight.json')
    source_manifest = check_sources(BASE,folder/'runtime')
    before_protected = read(folder/'protected_original_before.json')
    assert protected_files(BASE) == before_protected
    stage1_eval = read(folder/'stage1_reference.json')
    reference = stage1_eval['checkpoint']
    assert sha(reference['path']) == reference['sha256'] == '0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e'
    stage1 = torch.load(reference['path'],map_location='cpu',weights_only=False)
    assert stage1['epoch'] == 198
    bn_keys = [k for k in stage1['state_dict_model'] if k.endswith(('running_mean','running_var','num_batches_tracked'))]
    parameter_keys = [k for k in stage1['state_dict_model'] if k not in bn_keys]
    assert len(bn_keys) == 93 and len(parameter_keys) == 93
    bn_initial = {k:stage1['state_dict_model'][k] for k in bn_keys}
    param_initial = {k:stage1['state_dict_model'][k] for k in parameter_keys}
    assert all(int(v) == 16830 for k,v in bn_initial.items() if k.endswith('num_batches_tracked'))
    statuses = {arm:read(folder/arm/'status.json') for arm in ['dr','h2t']}
    evaluations = {arm:{kind:read(folder/arm/(kind+'_evaluation.json')) for kind in ['best','final']} for arm in statuses}
    historical = {arm:{kind:read(OFFICIAL/arm/(kind+'_evaluation.json')) for kind in ['best','final']} for arm in statuses}
    verification = {'suite_id':suite_id,'stage1_checkpoint':reference,'stage1_epoch':198,
        'source_files_equal_original':len(source_manifest),'protected_original_files_unchanged':len(before_protected),
        'bn_initial_counter_each_layer':16830,'arms':{}}
    checkpoint_objects = {}
    configs = {arm:yaml.safe_load((folder/arm/'effective.yaml').read_text(encoding='utf-8')) for arm in statuses}
    normalized = {arm:{k:v for k,v in cfg.items() if k not in ['name','H2T']} for arm,cfg in configs.items()}
    assert normalized['dr'] == normalized['h2t']
    for arm,status in statuses.items():
        assert status['state'] == 'completed' and status['completed_epochs'] == 10
        assert status['seed'] == 123 and status['shift_bn'] is False
        assert status['H2T'] == (0. if arm == 'dr' else .25)
        assert len(status['checkpoint_loads']) == 1 and status['checkpoint_loads'][0]['sha256'] == reference['sha256']
        assert status['h2t_calls'] == (0 if arm == 'dr' else 850)
        assert status['initial_backbone_buffers_sha256'] == state_hash(bn_initial)
        assert status['initial_backbone_parameters_sha256'] == state_hash(param_initial)
        assert status['initial_classifier_sha256'] == state_hash(stage1['state_dict_classifier'])
        assert status['channel_rng']['seed'] == 123
        assert (status['channel_rng']['initial_state_sha256'] == status['channel_rng']['final_state_sha256']) == (arm == 'dr')
        assert len(status['epochs']) == 10
        assert status['initial_bn_counters'] == {k:16830 for k in bn_initial if k.endswith('num_batches_tracked')}
        cfg = configs[arm]
        assert cfg['resume'] == reference['path'] and not cfg['shift_bn']
        old_cfg = yaml.safe_load((OFFICIAL/arm/'effective.yaml').read_text(encoding='utf-8'))
        differences = [k for k in cfg if cfg[k] != old_cfg[k]]
        assert set(differences) == {'name','shift_bn'}
        for i,epoch in enumerate(status['epochs']):
            assert epoch['epoch'] == i+1 and epoch['steps'] == 85 and epoch['samples'] == 10847
            assert abs(epoch['lr']-(.2*.5*.1*(1+math.cos(i/10*3.1415926535)))) < 1e-14
            assert epoch['finite_train_loss'] and np.isfinite(epoch['train_loss'])
            assert epoch['backbone_gradients_absent'] and epoch['classifier_gradients_finite'] and epoch['ce_labels_equal_bs']
            assert epoch['backbone_parameters_delta']['changed_count'] == 0 and epoch['bn_delta_this_epoch']['changed_count'] == 0
            assert epoch['bn_buffers_bitwise_equal_stage1'] and epoch['backbone_parameters_bitwise_equal_stage1']
            assert epoch['bn_buffers_sha256'] == state_hash(bn_initial)
            assert epoch['backbone_parameters_sha256'] == state_hash(param_initial)
            assert epoch['classifier_delta_from_stage1']['changed_count'] == 2
            assert len(epoch['bn_count_increment']) == 31 and set(epoch['bn_count_increment'].values()) == {0}
            assert epoch['bn_unchanged_during_eval'] and epoch['backbone_forward_calls'] == (85 if arm == 'dr' else 170)
            assert epoch['is_histogram'] == preflight['class_counts']
            assert len(epoch['batch_fingerprints']) == 85
            assert len(epoch['channel_masks']) == (0 if arm == 'dr' else 85)
            assert all(len(mask) == len(set(mask)) == 16 and min(mask) >= 0 and max(mask) < 64 for mask in epoch['channel_masks'])
        best_epoch = max(status['epochs'],key=lambda e:e['eval_overall_percent'])['epoch']
        assert status['stage2_best_epoch'] == best_epoch
        results = {}
        checkpoint_objects[arm] = {}
        for kind,evaluation in evaluations[arm].items():
            assert evaluation['test_samples'] == 10000 and evaluation['class_counts'] == [100]*100
            assert evaluation['bn_buffers_unchanged_during_eval'] and evaluation['matches_training_epoch_evaluation']
            path = Path(evaluation['checkpoint']['path'])
            assert path.resolve().is_relative_to((folder/'runtime/saved').resolve())
            assert sha(path) == evaluation['checkpoint']['sha256'] == status['checkpoints_sha256'][path.name]
            value = torch.load(path,map_location='cpu',weights_only=False)
            assert value['epoch'] == evaluation['epoch'] == (best_epoch if kind == 'best' else 10)
            checkpoint_objects[arm][kind] = value
            parameters = {k:value['state_dict_model'][k] for k in parameter_keys}
            buffers = {k:value['state_dict_model'][k] for k in bn_keys}
            parameter_delta = compare(param_initial,parameters)
            buffer_delta = compare(bn_initial,buffers)
            classifier_delta = compare(stage1['state_dict_classifier'],value['state_dict_classifier'])
            assert parameter_delta['changed_count'] == buffer_delta['changed_count'] == 0
            assert classifier_delta['changed_count'] == 2
            counts = np.array(evaluation['correct_counts'])
            expected_metrics = {'overall':float(counts.sum()/10000*100),'head':float(counts[:36].mean()),
                'medium':float(counts[36:71].mean()),'tail':float(counts[71:].mean())}
            assert all(abs(expected_metrics[k]-evaluation['metrics_percent'][k]) < 1e-10 for k in METRICS)
            results[kind] = {'checkpoint':evaluation['checkpoint'],'epoch':evaluation['epoch'],
                'metrics_percent':evaluation['metrics_percent'],'backbone_parameters':parameter_delta,
                'bn_buffers':buffer_delta,'classifier_parameters':classifier_delta}
        ckps = Path(status['model_dir'])
        assert sha(ckps/'model_final.pth.tar') == sha(ckps/'current.pth.tar')
        if 'model_best.pth.tar' in status['checkpoints_sha256']:
            assert sha(ckps/'model_best.pth.tar') == sha(ckps/'model_best_stage2.pth.tar')
        for rel,expected in source_manifest.items():
            assert sha(Path(status['code_dir'])/rel) == expected
        verification['arms'][arm] = {'checkpoints':results,'only_yaml_changes_from_historical_arm':differences,
            'backbone_parameters_unchanged_each_epoch':True,'bn_buffers_unchanged_each_epoch':True,
            'classifier_updates_each_epoch':True,'full_test_reloads_verified':2,'source_snapshot_verified':True}
        with (folder/arm/'epochs.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['epoch','lr','train_loss','overall','head','medium','tail','bn_buffers_changed','bn_counter_increment'])
            for e in status['epochs']:
                writer.writerow([e['epoch'],e['lr'],e['train_loss'],*[e['eval_metrics_percent_rounded_3dp'][k] for k in METRICS],0,0])
    assert statuses['dr']['initial_rng_hashes'] == statuses['h2t']['initial_rng_hashes']
    assert statuses['dr']['backend_settings'] == statuses['h2t']['backend_settings']
    paired = {'total_batches':850,'initial_rng_hashes_identical':True,'backend_settings_identical':True,
        'identical_batches_by_field':{},'epoch_start_rng_equal':{},'epoch_end_rng_equal':{}}
    fields = ['bs_labels_sha256','is_labels_sha256','bs_images_sha256','is_images_sha256','cpu_rng_at_batch_ready_sha256']
    for field in fields:
        paired['identical_batches_by_field'][field] = sum(a[field] == b[field]
            for ea,eb in zip(statuses['dr']['epochs'],statuses['h2t']['epochs'])
            for a,b in zip(ea['batch_fingerprints'],eb['batch_fingerprints']))
    assert set(paired['identical_batches_by_field'].values()) == {850}
    for stream in ['torch_cpu','torch_cuda','python','numpy']:
        for position in ['start','end']:
            key = 'rng_at_epoch_'+position
            value = sum(a[key][stream] == b[key][stream] for a,b in zip(statuses['dr']['epochs'],statuses['h2t']['epochs']))
            paired['epoch_'+position+'_rng_equal'][stream] = value
            assert value == 10
    paired['bn_state_and_backbone_parameters_identical_between_arms'] = all(
        tensor_bytes(checkpoint_objects['dr']['final']['state_dict_model'][k]) == tensor_bytes(checkpoint_objects['h2t']['final']['state_dict_model'][k])
        for k in stage1['state_dict_model'])
    assert paired['bn_state_and_backbone_parameters_identical_between_arms']
    assert statuses['h2t']['fusion_verified']['ambient_cpu_rng_bitwise_unchanged_every_call']
    verification['paired_inputs_and_rng'] = paired
    verification['historical_h2t_gain_pp'] = {kind:{k:historical['h2t'][kind]['metrics_percent'][k]-historical['dr'][kind]['metrics_percent'][k]
        for k in METRICS} for kind in ['best','final']}
    verification['fixed_bn_h2t_gain_pp'] = {kind:{k:evaluations['h2t'][kind]['metrics_percent'][k]-evaluations['dr'][kind]['metrics_percent'][k]
        for k in METRICS} for kind in ['best','final']}
    dump(folder/'artifact_verification.json',verification)
    dump(folder/'paired_inputs_verification.json',paired)
    dump(folder/'protected_original_after.json',protected_files(BASE))
    make_report(folder,preflight,statuses,evaluations,historical,stage1_eval,verification)
    shutil.copy2(__file__,folder/'summarize_bn_audit.py')
    files = {str(p.relative_to(folder)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':sha(p)}
        for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name != 'delivery_manifest.json'}
    dump(folder/'delivery_manifest.json',{'suite_id':suite_id,'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'files':files,'original_files_unchanged':True,'root_report_sha256':sha(BASE.parent/'H2T_BN_AUDIT.md')})
    assert protected_files(BASE) == before_protected
    print(json.dumps({'verification':'PASS','paired_batches':paired['identical_batches_by_field'],
        'fixed_bn_metrics':{a:{kind:e['metrics_percent'] for kind,e in ev.items()} for a,ev in evaluations.items()},
        'fixed_bn_h2t_gain_pp':verification['fixed_bn_h2t_gain_pp'],'report':str(BASE.parent/'H2T_BN_AUDIT.md')},ensure_ascii=False,indent=2))


def link(path,label=None):
    path = Path(path).resolve()
    return f'[{label or path.name}](<{path.as_posix()}>)'


def row(label,epoch,metrics):
    return '| '+label+' | '+str(epoch)+' | '+' | '.join(f'{metrics[k]:.2f}' for k in METRICS)+' |'


def make_report(folder,preflight,statuses,evaluations,historical,stage1_eval,verification):
    # Filled after all numerical and artifact checks; the report never substitutes for execution.
    stage1 = stage1_eval['metrics_percent']
    lines = ['# H2T BN 机制审计','',f'完成时间：{time.strftime("%Y-%m-%d %H:%M:%S %z")}；控制实验编号：`{folder.name}`。','',
        '本次完成两组独立的固定 BN 控制实验，每组 10 epochs，并分别重新加载最佳及第 10 epoch Checkpoint，在全部 10,000 张 CIFAR-100 测试图像上评估。原始 DR / DR+H2T 运行、源码和结果保留。本报告属于机制控制实验，不替代官方 `shift_bn=True` 复现。','',
        '## 1. 结果','',
        '所有准确率单位为 %；差值单位为百分点（pp）。Head 为类别 [0,36)，Medium 为 [36,71)，Tail 为 [71,100)，三组为各类准确率的平均值。固定 epoch 10 为等训练时长的主要对照，最佳 epoch 为补充。','',
        '### 最佳 Checkpoint','',
        '| 条件 | 最佳 epoch | Overall | Head | Medium | Tail |','|---|---:|---:|---:|---:|---:|',row('Stage I',198,stage1)]
    for arm,label in [('dr','DR'),('h2t','DR+H2T')]:
        e = historical[arm]['best']
        lines.append(row('原始 '+label,e['epoch'],e['metrics_percent']))
    for arm,label in [('dr','DR'),('h2t','DR+H2T')]:
        e = evaluations[arm]['best']
        lines.append(row('固定 BN '+label,e['epoch'],e['metrics_percent']))
    lines += ['','### 固定第 10 epoch','',
        '| 条件 | epoch | Overall | Head | Medium | Tail |','|---|---:|---:|---:|---:|---:|']
    for collection,prefix in [(historical,'原始'),(evaluations,'固定 BN')]:
        for arm,label in [('dr','DR'),('h2t','DR+H2T')]:
            e = collection[arm]['final']
            lines.append(row(prefix+' '+label,10,e['metrics_percent']))
    lines += ['','### H2T 相对 DR 的增益','',
        '| 条件 / 选择规则 | ΔOverall | ΔHead | ΔMedium | ΔTail |','|---|---:|---:|---:|---:|']
    for source,label in [('historical_h2t_gain_pp','原始 BN 更新'),('fixed_bn_h2t_gain_pp','固定 BN + 配对输入')]:
        for kind,rule in [('best','最佳'),('final','epoch 10')]:
            gain = verification[source][kind]
            lines.append('| '+label+' / '+rule+' | '+' | '.join(f'{gain[k]:+.2f}' for k in METRICS)+' |')
    difference = {k:verification['fixed_bn_h2t_gain_pp']['final'][k]-verification['historical_h2t_gain_pp']['final'][k] for k in METRICS}
    lines.append('| 固定条件增益 − 原始增益 / epoch 10 | '+' | '.join(f'{difference[k]:+.2f}' for k in METRICS)+' |')
    lines += ['','### 固定 BN 结果相对 Stage I 的变化','',
        '| 固定 BN 条件 / 选择规则 | ΔOverall | ΔHead | ΔMedium | ΔTail |','|---|---:|---:|---:|---:|']
    for arm,label in [('dr','DR'),('h2t','DR+H2T')]:
        for kind,rule in [('best','最佳'),('final','epoch 10')]:
            m = evaluations[arm][kind]['metrics_percent']
            lines.append('| '+label+' / '+rule+' | '+' | '.join(f'{m[k]-stage1[k]:+.2f}' for k in METRICS)+' |')
    lines += ['','## 2. 锁定配置与隔离方式','',
        f'- Stage I：epoch 198，Overall 40.49%；{link(preflight["stage1_checkpoint"]["path"],"锁定 Checkpoint")}。',
        f'- SHA-256：`{preflight["stage1_checkpoint"]["sha256"]}`。两组各自新进程只加载该文件一次，初始化 Backbone 和分类器与其一致。',
        '- CIFAR-100-LT IF=100，ResNet-32，真实训练子集 10,847 张，100 类，每类 500→5 张；每个 epoch 85 batches（128×84+95），workers=0，10 epochs；测试每类 100 张。',
        '- SGD 仅包含分类器 2 个可学习张量，共 6,500 参数；momentum=0.9、weight_decay=0.0005、nesterov=False。CE 不重加权；Stage II 不使用 MixUp。',
        '- 保留官方 cosine 调度：配置 lr=0.1、lr_factor=0.2，实际 epoch 1 lr=0.02，epoch 10 lr≈0.0004894348373。',
        '- 两组使用原始 YAML，命令行共同覆盖数据路径、epoch-198 resume、独立名称和 `shift_bn=False`。DR 额外设置 `H2T=0.0`；H2T 保留 `H2T=0.25`。',
        '- 独立运行目录：'+link(folder,'固定 BN 控制实验目录')+'；其中 `runtime/` 只复制已审计的 48 个源码文件，不复制原始 saved 目录。与已审计副本的 SHA-256 全部一致；核心源码新增修改为 0。',
        '- 原始 commit：`e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`（沿用原审计中官方 archive 的 commit 标记）；Windows 兼容补丁沿用已审计的五文件补丁，未新增算法补丁。',
        '- 使用代码已有的 `shift_bn=False` 分支，Backbone `eval()`，分类器 `train()`；保留原始 `torch.no_grad()` 及仅分类器 SGD。Conv / BN affine 的 `requires_grad` 标志保持原始值，但无梯度、未进入优化器，参数逐位不变。',
        '- DR 保留原始 dual_sample 数据准备：依然先准备 IS 供体和增强，再准备 BS 接收者；`H2T=0` 跳过供体 GPU forward 和融合。这样两组数据 RNG 的调用路径相同，未改变 BS/IS 采样规则。','',
        '### H2T 函数及 RNG 控制','',
        'H2T 仍调用原始 `methods.H2T`：`randperm(64)` 取前 `int(0.25×64)=16` 通道，在接收者 [B,64,8,8] 上原地复制供体同位置通道；保留 48 通道，整个 batch 共用掩码。BS 标签用于 CE，IS 标签不参与融合后的监督；无供体或通道筛选。','',
        '唯一额外执行控制为外部 observer 的 CPU RNG 隔离：独立 CPU Generator 种子 123；用 `torch.random.fork_rng(devices=[])` 暂存环境 CPU RNG，将独立状态交给原始函数，保存推进后的通道状态，再恢复环境状态。每次调用均核验原始随机排列的 RNG 消耗、16/48 通道替换关系、供体不变和环境 CPU RNG 逐位不变。抽样分布和融合公式保持不变，具体掩码序列与历史官方复现不同，因此明确列为新增控制条件。','',
        '两组 Python、NumPy、torch CPU、CUDA 与 PYTHONHASHSEED 均为 123；保留 YAML `deterministic=False`，未启用额外确定性算法。后端实际设置：`'+json.dumps(statuses['dr']['backend_settings'],ensure_ascii=False)+'`。','',
        '### 全训练数据配对实证','',
        '| 项目 | 相同 batch / 总 batch |','|---|---:|',
        '| BS 标签 | 850 / 850 |','| BS 增强后图像（完整 float tensor 字节哈希） | 850 / 850 |',
        '| IS 标签 | 850 / 850 |','| IS 增强后图像（完整 float tensor 字节哈希） | 850 / 850 |',
        '| batch 就绪时 torch CPU RNG 状态 | 850 / 850 |','',
        '两组每个 epoch 的开始和结束，Python / NumPy / torch CPU / CUDA RNG 状态均相同（各 10/10）。这里的“输入一致”经全部 batch 哈希确认，不仅依据相同 seed。BS 样本身份以图像与标签 tensor 的完整一致性核验；未改变 dataset 返回值加入索引。验证证据：'+link(folder/'paired_inputs_verification.json')+'；逐 batch 哈希和全部 H2T 掩码存于各组 status.json。','',
        '## 3. 参数与 BN 的逐位核验','',
        '| 状态项 | 原始 DR / 原始 H2T（epoch 10） | 固定 BN DR / 固定 BN H2T（最佳及 epoch 10） |','|---|---|---|',
        '| Backbone 可学习张量 | 均 0/93 改变 | 均 0/93 改变，原始字节一致 |',
        '| BN running_mean | 均 31/31 改变 | 均 0/31 改变，原始字节一致 |',
        '| BN running_var | 均 31/31 改变 | 均 0/31 改变，原始字节一致 |',
        '| BN num_batches_tracked | DR：16830→17680；H2T：16830→18530 | 两组全部 31 层保持 16830 |',
        '| 分类器可学习张量 | 均 2/2 改变 | 均 2/2 改变，每个 epoch 有有限梯度 |','',
        '训练时逐次 Backbone forward 检查 `no_grad` 及所有 BN 的 eval 状态；每个 epoch 后核验 Backbone 参数、93 个 BN buffers 的完整字节状态哈希与 Stage I 一致，计数增量为 0。每次验证也核验 BN 不变；最后对最佳与最终文件中的每个张量再次按 dtype、shape、原始字节比较，不依赖容差或仅最大差值。四次完整测试 reload 通过。','',
        'Conv/BN affine 的全部 463,504 参数和三个 BN buffer 集合均不变，两组最终 Backbone state_dict 也逐位相同。每组 850 个训练 loss 为有限值，CE 标签始终为 BS，分类器每个 epoch 均更新。详细逐张量 SHA-256 和差值：'+link(folder/'artifact_verification.json')+'。','',
        '| 固定 BN 组 | 最终分类器最大绝对变化（相对 Stage I） | GPU peak allocated MiB | GPU peak reserved MiB | 全 GPU 采样峰值 MiB |','|---|---:|---:|---:|---:|']
    for arm,label in [('dr','DR'),('h2t','DR+H2T')]:
        delta = verification['arms'][arm]['checkpoints']['final']['classifier_parameters']['max_abs_delta']
        s = statuses[arm]
        lines.append(f'| {label} | {delta:.9f} | {s["cuda_peak_allocated_mib"]:.2f} | {s["cuda_peak_reserved_mib"]:.2f} | {s["gpu_global_peak_sampled_mib"]:.0f} |')
    lines += ['','全 GPU 采样峰值包含桌面及其他进程，采样间隔 5 秒；allocated/reserved 为训练进程 PyTorch CUDA 的峰值。','',
        '## 4. 机制结论与归因边界','']
    final_gain = verification['fixed_bn_h2t_gain_pp']['final']['overall']
    best_gain = verification['fixed_bn_h2t_gain_pp']['best']['overall']
    conclusion = '带来额外收益' if final_gain > 0 else ('降低了准确率' if final_gain < 0 else '没有改变 Overall')
    lines += [f'在固定 Stage I BN、相同 Backbone、逐 batch 配对的 BS/IS 样本及增强条件下，H2T 在第 10 epoch 相对 DR 的 Overall 增益为 **{final_gain:+.2f} pp**，最佳模型增益为 **{best_gain:+.2f} pp**；本次单种子控制结果显示 H2T {conclusion}。',
        '',
        '这一固定 BN 条件内的差异可以归因于“加入原始随机通道融合”的训练干预及其对分类器学习轨迹的影响：BN、Conv/BN affine、数据和增强保持一致，测试时没有 H2T 融合，两组相同的固定 Backbone 只配合各自训练出的分类器。该结论以本次指定 seed、独立通道掩码流和软件/硬件环境为条件。',
        '',
        f'原始条件第 10 epoch H2T 增益为 +0.87 pp，本次固定条件为 {final_gain:+.2f} pp；二者相差 {difference["overall"]:+.2f} pp。这不是可单独识别的 BN 因果贡献：历史原始实验使用共享 CPU RNG，H2T 调用会改变后续增强及 IS 顺序；本次同时固定 BN 并隔离通道 RNG。历史与控制之间同时改变了这些因素。未执行“BN 更新 + 同一隔离 RNG/配对协议”实验，不能将这个差值全部归因于 BN。',
        '',
        '原始 `shift_bn=True` 使训练 forward 使用当前 batch 统计量，并更新用于测试的 running buffers；H2T 组每 batch 两次 Backbone forward，DR 一次。因此原始结果变化涉及训练特征归一化、BN 统计量迁移和分类器调整。固定 BN 不仅禁止 buffers 更新，还使训练特征使用 Stage I running statistics；不宜把原始全部性能变化解释为分类器权重调整。',
        '',
        '仍保留的随机性与选择影响：仅运行一个 seed 和一条随机通道流，无多种子方差或置信区间；未启用确定性 CUDA 算法，可能有数值非确定性。最佳 epoch 由同一测试集逐 epoch 选择，且两组可能不同；最佳对比带选择效应，固定第 10 epoch 为本次主要对比。本次证据证明了输入配对和状态冻结，不代表跨 seed 的普遍因果效应。','',
        '## 5. 产物、保护核验及完整配置','',
        f'原始套件的 delivery manifest、其 32 份交付文件、原始最佳/最终/当前 Checkpoint、源码快照、日志及已有报告等合计 {verification["protected_original_files_unchanged"]} 个文件，训练前后 SHA-256 一致；原始 runtime 的 48 个源码文件也保持一致。证据：'+link(folder/'protected_original_before.json')+'、'+link(folder/'protected_original_after.json')+'。原始报告：'+link(BASE.parent/'H2T_STAGE2_RESULTS.md')+'。','',
        '### 新增模型和日志','',
        '| 固定 BN 组 | 最佳 epoch | 最佳 Checkpoint | 第 10 epoch Checkpoint | 配置 / 逐 epoch 记录 / 完整日志 |','|---|---:|---|---|---|']
    for arm,label in [('dr','DR'),('h2t','DR+H2T')]:
        lines.append('| '+label+' | '+str(statuses[arm]['stage2_best_epoch'])+' | '+link(evaluations[arm]['best']['checkpoint']['path'],'best')+' | '+
            link(evaluations[arm]['final']['checkpoint']['path'],'epoch 10')+' | '+link(folder/arm/'effective.yaml','YAML')+' / '+link(folder/arm/'epochs.csv','CSV')+' / '+link(folder/arm/'stdout_stderr.log','日志')+' |')
    lines += ['','最佳模型使用 `model_best_stage2.pth.tar`；最终模型为 current.pth.tar 的字节副本 `model_final.pth.tar`。没有重置原始全局 best_acc1 的 Stage I 阈值，observer 单独保存 10 epochs 内的最大值；本次与原始 model_best.pth.tar 一致。源码快照位于各组 saved 运行目录的 codes/，48 个文件均验证。每个模型路径和 SHA-256 见 artifact_verification.json。','',
        '### 已执行命令','',
        '```powershell',"$env:PYTHONIOENCODING = 'utf-8'","$env:PYTHONHASHSEED = '123'", 
        f"$auditRoot = '{folder}'", 
        f"& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -u \"$auditRoot\\bn_audit_runner.py\" prepare --suite-id {folder.name}",
        f"& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -u \"$auditRoot\\bn_audit_runner.py\" run --suite-id {folder.name} --arm dr",
        f"& 'D:\\app\\anaconda\\envs\\pytorch\\python.exe' -u \"$auditRoot\\bn_audit_runner.py\" run --suite-id {folder.name} --arm h2t",'```','',
        '上述目录已完成，启动器会拒绝覆盖。重新执行需要通过 create_bn_audit.py 创建新的唯一 suite-id。每组原始训练入口 argv、加载路径和 SGD 参数保存在 status.json；训练入口仍为原始 train_stage2.py。','',
        '外部 observer 差异：'+link(folder/'control_observer.diff')+'；独立评估器差异：'+link(folder/'evaluation_observer.diff')+'。这些文件展示控制协议的改动，核心源码未修改。环境：'+preflight['python_version'].split('\n')[0]+f'；PyTorch {preflight["torch"]} / torchvision {preflight["torchvision"]} / CUDA {preflight["cuda_runtime"]}；{preflight["gpu"]}。','']
    for arm,label in [('dr','DR 固定 BN'),('h2t','DR+H2T 固定 BN')]:
        lines += ['### '+label+' 完整有效配置','','```yaml',(folder/arm/'effective.yaml').read_text(encoding='utf-8').rstrip(),'```','']
    lines += ['## 6. 完成状态','',
        '两组固定 BN 控制实验、四次 10,000 张测试集重新加载评估、原始产物保护核验、参数及 buffers 逐位核验、850 batches 配对核验均已完成。仅执行本次指定的两组控制；未开展其他改进实验。完成后停止，等待下一步指示。','']
    report = '\n'.join(lines)
    # All displayed local links resolve, including the locked reference and completed model files.
    import re
    for target in re.findall(r'\]\(<([^>]+)>\)',report):
        assert Path(target).exists(), target
    (BASE.parent/'H2T_BN_AUDIT.md').write_text(report,encoding='utf-8')
    (folder/'H2T_BN_AUDIT.md').write_text(report,encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite-id',required=True)
    args = parser.parse_args()
    summarize(args.suite_id)
