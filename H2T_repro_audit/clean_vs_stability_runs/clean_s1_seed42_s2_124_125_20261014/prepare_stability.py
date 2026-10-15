"""Freeze the approved Clean basis; reproduce seed123 RNG factory, then prepare124/125."""
from __future__ import annotations
import ast
import hashlib
import json
import pickle
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
WORK = BASE.parent
ORIGIN = BASE/'clean_vs_runs/clean_s1_seed42_vs_seed123_20261013'
S1_SHA = '7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88'
STAT_SHA = 'e81a4897a965d642f832bd6429e0e5a7d136725daaa069d9ded96f87ab64aef9'


def generate_local_helpers():
    # Read source text without importing any historical Python module.
    source = (ORIGIN/'common.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    names = ['now','read','dump','sha','array_hash','state_hash','csv_write','guard_flush','install_test_guard',
        'integrity_only','guard_pass','seed_all','metrics','evaluate','check_frozen','check_history']
    functions = {n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    # ast.get_source_segment omits decorators; restore the original contextmanager.
    functions['integrity_only'] = '@contextlib.contextmanager\n'+functions['integrity_only']
    header = '''"""Reviewed Clean helper functions with independent ROOT; no Stage I training entry."""
from __future__ import annotations
import ast,contextlib,csv,hashlib,json,math,os,random,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
WORK=BASE.parent
ORIGIN=BASE/'clean_vs_runs/clean_s1_seed42_vs_seed123_20261013'
RUNTIME=ROOT/'runtime'
DATA=WORK/'H2T/data/cifar100'
N,BATCH,STEPS,EPOCHS=10847,128,85,10
SEEDS=[124,125]
ARMS=['Uniform-PerSample','Low-VS']
S1_SHA='7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88'
STAT_SHA='e81a4897a965d642f832bd6429e0e5a7d136725daaa069d9ded96f87ab64aef9'
_GUARD=None
_INTEGRITY=False
'''
    (ROOT/'common.py').write_text(header+'\n\n'+'\n\n'.join(functions[n] for n in names)+'\n',encoding='utf-8')
    helpers = (ORIGIN/'clean_stage2.py').read_text(encoding='utf-8')
    helper_tree = ast.parse(helpers)
    selected = ['tensor_hash','augment','fuse','scheduler','optimizer_for']
    body = '\n\n'.join(ast.get_source_segment(helpers,n) for n in helper_tree.body
        if isinstance(n,ast.FunctionDef) and n.name in selected)
    (ROOT/'reused_train_helpers.py').write_text('"""Verbatim seed123 training helper functions, independent ROOT."""\nfrom common import *\n\n'+body+'\n',encoding='utf-8')
    shutil.copy2(ORIGIN/'reused_vs_math.py',ROOT/'reused_vs_math.py')
    policy = (ORIGIN/'clean_online_policy.py').read_text(encoding='utf-8')
    assert policy.count("        'High-VS':selected_bool(top_indices(s,48,True)),\n") == 1
    policy = policy.replace('Three-arm adaptation','Two-arm adaptation')
    policy = policy.replace("        'High-VS':selected_bool(top_indices(s,48,True)),\n",'')
    policy = policy.replace("for name in ['High-VS','Low-VS']:","for name in ['Low-VS']:")
    policy = policy.replace('all_three_masks_sha256','all_two_masks_sha256')
    assert 'High-VS' not in policy
    (ROOT/'online_policy.py').write_text(policy,encoding='utf-8')
    import difflib
    delta = ''.join(difflib.unified_diff((ORIGIN/'clean_online_policy.py').read_text(encoding='utf-8').splitlines(True),
        policy.splitlines(True),fromfile='clean_seed123/clean_online_policy.py',tofile='stability/online_policy.py'))
    (ROOT/'online_policy.adaptation.diff').write_text(delta,encoding='utf-8')


def main():
    assert not (ROOT/'preflight.json').exists(), 'Refuse overwriting prepared stability run'
    generate_local_helpers()
    from common import (sha,read,dump,now,install_test_guard,guard_pass,integrity_only,DATA,check_history,array_hash)
    install_test_guard('prepare')
    # All original/earlier experimental files remain protected, including pilot.
    history = [p for p in BASE.rglob('*') if p.is_file() and not p.is_relative_to(ROOT)]
    history += [p for p in WORK.glob('*.md') if p.is_file()]
    dump(ROOT/'historical_before.json',{str(p):sha(p) for p in sorted(set(history))})
    origin_manifest = read(ORIGIN/'delivery_manifest.json')
    assert all(sha(ORIGIN/r) == v for r,v in origin_manifest['files'].items())
    manifest = read(ORIGIN/'preflight.json')['runtime_files']
    for rel,expected in manifest.items():
        src = ORIGIN/'runtime'/rel
        assert sha(src) == expected
        dest = ROOT/'runtime'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest)
        assert sha(dest) == expected
    s1 = read(ORIGIN/'clean_stage1_protocol.json')
    statistics = read(ORIGIN/'clean_statistics_provenance.json')
    assert s1['passed'] and not s1['TEST_loaded'] and not s1['TEST_selected']
    assert s1['best_epoch'] == 184 and sha(s1['best_checkpoint']) == s1['best_checkpoint_sha256'] == S1_SHA
    assert statistics['Clean_StageI_sha256'] == S1_SHA and statistics['statistics_sha256'] == STAT_SHA
    for name in ['split_indices.npz','train_images.npy','train_labels.npy','validation_images.npy','validation_labels.npy',
        'validation_feature_maps.npy','class_feature_statistics.npz','validation_class_protocol.csv','fixed_sample_ids.csv',
        'clean_statistics_provenance.json','CLEAN_STAGE1_PROTOCOL.md','stage2_original_effective.yaml']:
        shutil.copy2(ORIGIN/name,ROOT/name)
        assert sha(ROOT/name) == sha(ORIGIN/name)
    (ROOT/'locked').mkdir()
    shutil.copy2(s1['best_checkpoint'],ROOT/'locked/clean_stage1_best.pth')
    assert sha(ROOT/'locked/clean_stage1_best.pth') == S1_SHA
    s1['original_checkpoint_path'] = s1['best_checkpoint']
    s1['best_checkpoint'] = str(ROOT/'locked/clean_stage1_best.pth')
    dump(ROOT/'clean_stage1_protocol.json',s1)
    assert sha(ROOT/'class_feature_statistics.npz') == STAT_SHA
    test_md5 = None
    with integrity_only():
        test_md5 = sha(DATA/'cifar-100-python/test','md5')
    assert test_md5 == 'f0ef6b0ae62326f3e7ffdfab6717acfc'
    assert sha(DATA/'cifar-100-python/train') == '735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b'
    with (DATA/'cifar-100-python/train').open('rb') as f:
        raw = pickle.load(f,encoding='bytes')
    images = raw[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1).copy()
    y = np.asarray(raw[b'fine_labels'],dtype=np.int64)
    split = np.load(ROOT/'split_indices.npz',allow_pickle=False)
    ti,vi = split['train_raw_indices'],split['validation_raw_indices']
    for name,ids in [('train',ti),('validation',vi)]:
        assert np.array_equal(images[ids],np.load(ROOT/f'{name}_images.npy'))
        assert np.array_equal(y[ids],np.load(ROOT/f'{name}_labels.npy'))
    assert len(ti) == 10847 and len(vi) == 1980 and np.intersect1d(ti,vi).size == 0
    assert np.bincount(y[vi],minlength=100).tolist() == [0]+[20]*99
    train_content = {hashlib.sha256(im.tobytes()).digest() for im in images[ti]}
    assert not any(hashlib.sha256(im.tobytes()).digest() in train_content for im in images[vi])
    stat = np.load(ROOT/'class_feature_statistics.npz',allow_pickle=False)
    assert np.array_equal(stat['counts'],np.bincount(y[ti],minlength=100))
    assert float(stat['epsilon']) == 4.2622689356882903e-7
    assert sha(ROOT/'validation_feature_maps.npy') == statistics['validation_maps_sha256']
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    probe = torch.randn((64,64),device='cuda')
    assert torch.isfinite(probe @ probe).all()
    dump(ROOT/'preflight.json',{'prepared_at':now(),'python':sys.executable,'python_version':sys.version,
        'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/1024**2,'runtime_files':manifest,
        'StageI_retrained':False,'Clean_StageI_sha256':S1_SHA,'statistics_sha256':STAT_SHA,'epsilon':float(stat['epsilon']),
        'LT_samples':10847,'LT_counts':stat['counts'].tolist(),'LT_pixel_hash':hashlib.sha256(images[ti].tobytes()).hexdigest(),
        'train_ID_hash':array_hash(ti),'validation_ID_hash':array_hash(vi),'split_sha256':sha(ROOT/'split_indices.npz'),
        'validation_samples':1980,'missing_classes':[0],'100_outputs_retained':True,
        'train_val_index_overlap':0,'train_val_pixel_overlap':0,'TEST_integrity_md5':test_md5,
        'classifier_init':'same Clean StageI best fc, never seed123 StageII',
        'old_statistics_loaded':False,'new_class_statistics_computed':False})
    dump(ROOT/'protocol.json',{'declared_at':now(),'seeds':[124,125],'methods':['Uniform-PerSample','Low-VS'],
        'epochs_each':10,'updates_each':850,'batch_size':128,'last_batch':95,'rho':.25,'channels':64,'replace':16,
        'Clean_StageI_sha256':S1_SHA,'statistics_sha256':STAT_SHA,'epsilon':float(stat['epsilon']),
        'classifier_init':'fresh independent Clean StageI best fc; empty SGD state after dryrun',
        'optimizer':'same original SGD lr.1,momentum.9,wd.0005,nesterovFalse',
        'scheduler':'verbatim original StageII cosine, factor.2',
        'score':'v_reg/(v_reg+(mu_h-current_augmented_receiver_GAP)^2+fixed_epsilon)',
        'candidate_pool':48,'Low-VS':'bottom48 then uniform16 without replacement using shared Gumbel',
        'Uniform':'all64 then uniform16 without replacement using shared Gumbel',
        'fallback':'donor n<20 or score ptp<=1e-12 -> same Uniform; nonfinite stop; identical seed123 rule',
        'RNG':'Python,NumPy,Torch,CUDA set per124/125; BS Python seed, IS Torch offset100003,augmentation Torch offset300009,Gumbel NumPy offset700021',
        'primary':'fixed epoch10 99-class TRAIN validation; best validation epoch descriptive only',
        'validation':'same locked1980 TRAIN holdout,20/class1..99,class0missing,100 logits; no TEST',
        'seed123':'already observed pilot only; no retrain,no weights initialization; report separately from new124/125',
        'stability_summary':'new124/125 paired deltas mean and sampleSD ddof1; n=2 not significance evidence',
        'TEST':'no image/label decode, no CIFAR100(train=False), no TEST metrics/selection; opaque preflightMD5 only',
        'no_hyperparameter_search':True,'no_extra_seeds':True,'stop_after124_125':True})
    guard_pass();check_history()
    print('STABILITY PREPARE PASS: same Clean checkpoint/statistics/data/validation; exactly124+125,Uniform+Low',flush=True)


if __name__ == '__main__':
    main()
