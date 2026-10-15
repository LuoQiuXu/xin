"""Offline K=4 donor mechanism audit; no optimizer, no training, no test data.

This script imports only a fresh byte-identical copy of the audited runtime.
Each actual fusion calls original methods.H2T on 128 feature maps. Four
candidates for a recipient share the original batch-wide 16-channel mask.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
from datetime import datetime, timezone, timedelta
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import sys
import textwrap
import time
import traceback

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
RUNTIME = ROOT/'runtime'
DATA = BASE.parent/'H2T/data/cifar100'
CFG = 'config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml'
N = 20000
K = 4
GROUP_BATCH = 32
SEEDS = {'bs':123, 'is_class':124, 'within_class':125, 'channel':126,
         'random_candidate':127, 'mid_candidate':128, 'train_augmentation':129}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def dump(path,value):
    path = Path(path)
    temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def state_hash(module):
    result = hashlib.sha256()
    for key,value in module.state_dict().items():
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def historical_hashes():
    # Do not import historical modules, write their caches, or touch their outputs.
    paths = [p for p in BASE.rglob('*') if p.is_file() and 'donor_audit_runs' not in p.relative_to(BASE).parts]
    paths += [BASE.parent/name for name in ['H2T_STAGE1_RESULTS.md','H2T_STAGE2_RESULTS.md','H2T_BN_AUDIT.md','H2T_REPRO_AUDIT.md']]
    return {str(p):sha(p) for p in sorted(paths)}


def checkpoint_refs():
    stage1 = read(BASE/'formal_runs/formal_20261011_1429_r2/best_evaluation.json')['checkpoint']
    dr_status = read(BASE/'bn_audit_runs/fixed_bn_20261011_2255/dr/status.json')
    assert dr_status['state'] == 'completed' and dr_status['shift_bn'] is False
    dr = {'path':str(Path(dr_status['model_dir'])/'model_final.pth.tar'),
          'sha256':dr_status['checkpoints_sha256']['model_final.pth.tar']}
    assert sha(stage1['path']) == stage1['sha256'] == '0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e'
    assert sha(dr['path']) == dr['sha256'] == '1b054d8b45753c25dedd9897e3052e5f33c73b0c38b47acaa277c0f61c50283d'
    return stage1,dr


def official_transforms(cls):
    # Instantiate the exact transform expressions without opening CIFAR100_LT.eval.
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    result = {}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ['train_transform','eval_transform']:
                    expression = ast.Expression(node.value)
                    result[target.id] = eval(compile(expression,'official_transform','eval'),{'transforms':transforms})
    assert set(result) == {'train_transform','eval_transform'}
    return result


def prepare():
    assert not (ROOT/'preflight.json').exists(), 'Refuse to overwrite an existing run'
    assert not (BASE.parent/'H2T_DONOR_PERTURBATION_AUDIT.md').exists(), 'Existing report needs a separate name'
    protected = historical_hashes()
    dump(ROOT/'historical_before.json',protected)
    manifest = read(BASE/'evidence/runtime_manifest.json')['files']
    for rel,expected in manifest.items():
        assert sha(BASE/'runtime'/rel) == expected
        target = RUNTIME/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/'runtime'/rel,target)
        assert sha(target) == expected
    for filename in ['runtime_manifest.json','source_manifest.json','compatibility.diff']:
        shutil.copy2(BASE/'evidence'/filename,ROOT/filename)
    stage1,dr = checkpoint_refs()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.ones(8,device='cuda').sum().item() == 8
    assert N % GROUP_BATCH == 0 and GROUP_BATCH*K == 128
    dump(ROOT/'preflight.json',{'started_at':now(),'phase':1,'seed_streams':SEEDS,
        'receivers':N,'K':K,'rho':.25,'channels':64,'replaced_channels':16,
        'recipients_per_fusion_batch':GROUP_BATCH,'fusion_batch_size':128,'backbone_batch_size':128,
        'stage1_backbone_checkpoint':stage1,'reference_classifier_checkpoint':dr,
        'python':sys.executable,'python_version':sys.version,'torch':torch.__version__,
        'torchvision':torchvision.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'data_path':str(DATA),'runtime_files':manifest,
        'historical_files_protected':len(protected),'script_sha256':sha(__file__),
        'primary_preprocess':'official eval_transform on training subset',
        'sensitivity_preprocess':'one cached official train_transform per instance; seed 129',
        'test_examples_loaded_for_analysis':False,'optimizer_created':False,'training_performed':False,
        'core_source_modifications':[]})
    print(f'PREPARE PASS: {len(protected)} historical files protected; 48 byte-identical source files',flush=True)


class FeatureImages(torch.utils.data.Dataset):
    def __init__(self,images,transform):
        self.images,self.transform = images,transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        return self.transform(Image.fromarray(self.images[index]))


def margin(logits,targets):
    true = logits.gather(1,targets[:,None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1,targets[:,None],float('-inf'))
    rival,rival_class = masked.max(1)
    return true-rival,true,rival,rival_class


def run():
    preflight = read(ROOT/'preflight.json')
    assert preflight['script_sha256'] == sha(__file__)
    assert historical_hashes() == read(ROOT/'historical_before.json')
    assert os.environ.get('PYTHONHASHSEED') == '123'
    sys.path.insert(0,str(RUNTIME))
    from utils import config, update_config
    from datasets.cifar100 import IMBALANCECIFAR100, CIFAR100_LT
    from models.resnet_cifar import resnet32_fe,Classifier
    import methods
    assert Path(methods.__file__).resolve().is_relative_to(RUNTIME.resolve())
    update_config(config,argparse.Namespace(cfg=str(RUNTIME/CFG),opts=['data_path',str(DATA),'shift_bn','False',
        'resume',preflight['stage1_backbone_checkpoint']['path'],'name',ROOT.name]))
    (ROOT/'effective_source_config.yaml').write_text(config.dump(),encoding='utf-8')
    source_transforms = official_transforms(CIFAR100_LT)
    dataset = IMBALANCECIFAR100(config,root=str(DATA),imb_factor=.01,rand_number=0,train=True,download=False,transform=None)
    labels = np.array(dataset.targets,dtype=np.int64)
    counts = np.bincount(labels,minlength=100)
    prior = read(BASE/'evidence/preflight.json')
    assert counts.tolist() == prior['class_counts'] and counts.sum() == 10847
    assert hashlib.sha256(dataset.data.tobytes()).hexdigest() == prior['subset_data_sha256']
    assert sha(DATA/'cifar-100-python/train') == prior['integrity']['train']['sha256']
    np.save(ROOT/'train_subset_images.npy',dataset.data)
    np.save(ROOT/'train_subset_labels.npy',labels)
    pools = [np.flatnonzero(labels == c) for c in range(100)]
    bs_rng = random.Random(SEEDS['bs'])
    receiver_classes = np.empty(N,dtype=np.int64)
    receiver_indices = np.empty(N,dtype=np.int64)
    for i in range(N):
        receiver_classes[i] = bs_rng.randint(0,99)
        receiver_indices[i] = bs_rng.choice(pools[receiver_classes[i]].tolist())
    donor_classes = np.random.default_rng(SEEDS['is_class']).choice(100,N,p=counts/counts.sum())
    donor_rng = np.random.default_rng(SEEDS['within_class'])
    donor_indices = np.vstack([donor_rng.choice(pools[c],K,replace=True) for c in donor_classes])
    assert np.array_equal(labels[receiver_indices],receiver_classes)
    assert np.array_equal(labels[donor_indices],np.broadcast_to(donor_classes[:,None],(N,K)))
    random_positions = np.random.default_rng(SEEDS['random_candidate']).integers(0,K,N)
    mid_sorted_positions = np.random.default_rng(SEEDS['mid_candidate']).integers(1,3,N)
    np.savez_compressed(ROOT/'sampling_plan.npz',receiver_indices=receiver_indices,receiver_classes=receiver_classes,
        donor_classes=donor_classes,donor_indices=donor_indices,random_positions=random_positions,mid_sorted_positions=mid_sorted_positions)
    stage1 = torch.load(preflight['stage1_backbone_checkpoint']['path'],map_location='cpu',weights_only=False)
    dr = torch.load(preflight['reference_classifier_checkpoint']['path'],map_location='cpu',weights_only=False)
    assert stage1['epoch'] == 198 and dr['epoch'] == 10
    assert stage1['state_dict_model'].keys() == dr['state_dict_model'].keys()
    assert all(a.numpy().tobytes() == dr['state_dict_model'][k].numpy().tobytes() for k,a in stage1['state_dict_model'].items())
    model = torch.nn.DataParallel(resnet32_fe()).cuda()
    classifier = torch.nn.DataParallel(Classifier(feat_in=64,num_classes=100)).cuda()
    model.load_state_dict(stage1['state_dict_model'],strict=True)
    classifier.load_state_dict(dr['state_dict_classifier'],strict=True)
    model.eval().requires_grad_(False)
    classifier.eval().requires_grad_(False)
    assert model.module.feature_tuning is False and not hasattr(model.module,'conv1d')
    initial = {'backbone':state_hash(model),'classifier':state_hash(classifier)}
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    checks = {'state_before':initial,'backbone_equals_fixed_bn_dr_checkpoint':True,'no_grad_eval_every_forward':True,
        'transform_expressions_evaluated_from_original_source':{k:repr(v) for k,v in source_transforms.items()},
        'subset_sha256':hashlib.sha256(dataset.data.tobytes()).hexdigest(),'class_counts':counts.tolist(),
        'backbone_shape':[64,8,8],'pool_shape':[64],'source_files':{'methods':methods.__file__,
        'backbone':inspect.getfile(resnet32_fe),'dataset':inspect.getfile(IMBALANCECIFAR100)},
        'test_dataset_instantiated':False,'test_images_or_labels_used':False,'optimizer_created':False,
        'backend_settings':{'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
        'feature_banks':{},'scoring':{}}

    def feature_hook(_module,_args):
        assert not torch.is_grad_enabled() and not model.training
        assert all(not layer.training for layer in model.modules() if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm))
    feature_handle = model.register_forward_pre_hook(feature_hook)
    started = time.perf_counter()
    mask_matrix = None
    for mode,transform_name in [('eval_preprocess','eval_transform'),('fixed_train_aug','train_transform')]:
        torch.manual_seed(SEEDS['train_augmentation'])
        random.seed(SEEDS['train_augmentation'])
        np.random.seed(SEEDS['train_augmentation'])
        loader = torch.utils.data.DataLoader(FeatureImages(dataset.data,source_transforms[transform_name]),
            batch_size=128,shuffle=False,num_workers=0,pin_memory=True)
        bank = np.lib.format.open_memmap(ROOT/(mode+'_feature_maps.npy'),mode='w+',dtype=np.float32,shape=(10847,64,8,8))
        input_hash = hashlib.sha256()
        offset = 0
        with torch.no_grad():
            for images in loader:
                input_hash.update(images.numpy().tobytes())
                features = model(images.cuda(non_blocking=True))
                assert features.shape[1:] == (64,8,8) and torch.isfinite(features).all()
                if offset == 0:
                    direct = classifier.module.fc(F.avg_pool2d(features,8).view(-1,64))
                    original = classifier(features)
                    assert torch.equal(direct,original)
                    ambient = torch.get_rng_state().clone()
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(999)
                        identity = methods.H2T(features.clone(),features,rho=.25)
                    assert torch.equal(ambient,torch.get_rng_state())
                    assert torch.equal(features,identity) and torch.equal(classifier(identity),original)
                    checks['classifier_pool_fc_verified'] = True
                    checks['identity_fusion_margin_error'] = 0.
                bank[offset:offset+len(images)] = features.detach().cpu().numpy()
                offset += len(images)
        bank.flush()
        assert offset == 10847 and state_hash(model) == initial['backbone']
        checks['feature_banks'][mode] = {'instances':offset,'shape':list(bank.shape),'dtype':str(bank.dtype),
            'input_tensors_sha256':input_hash.hexdigest(),'path':str(ROOT/(mode+'_feature_maps.npy')),
            'sha256':sha(ROOT/(mode+'_feature_maps.npy'))}
        print(f'{mode}: extracted {offset} training feature maps; parameters/BN unchanged',flush=True)
        result,mode_masks,mode_checks = score(bank,model,classifier,methods,receiver_indices,receiver_classes,donor_indices,mode)
        if mask_matrix is None:
            mask_matrix = mode_masks
            np.save(ROOT/'channel_masks.npy',mask_matrix)
        else:
            assert np.array_equal(mode_masks,mask_matrix)
        np.savez_compressed(ROOT/(mode+'_statistics.npz'),**result)
        checks['scoring'][mode] = mode_checks
        assert state_hash(model) == initial['backbone'] and state_hash(classifier) == initial['classifier']
        del bank
    feature_handle.remove()
    for rel,expected in preflight['runtime_files'].items():
        assert sha(RUNTIME/rel) == sha(BASE/'runtime'/rel) == expected
    assert checkpoint_refs() == (preflight['stage1_backbone_checkpoint'],preflight['reference_classifier_checkpoint'])
    after = historical_hashes()
    assert after == read(ROOT/'historical_before.json'), 'Historical experiment was modified'
    dump(ROOT/'historical_after.json',after)
    checks.update({'state_after':{'backbone':state_hash(model),'classifier':state_hash(classifier)},
        'all_parameters_bn_buffers_bitwise_unchanged':True,'masks_identical_across_preprocessing_modes':True,
        'historical_files_unchanged':len(after),'fusion_calls_each_mode':N//GROUP_BATCH,
        'elapsed_seconds':time.perf_counter()-started,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
        'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'finished_at':now()})
    dump(ROOT/'correctness_verification.json',checks)
    print(f'COMPLETE: {N} receivers x {K} candidates x 2 preprocessing modes; all checkpoints/state/history unchanged',flush=True)


def score(bank,model,classifier,methods,receiver_indices,receiver_classes,donor_indices,mode):
    fields = ['baseline_margin','fused_margin','D','l2_spatial','l2_gap','cosine_spatial','cosine_gap',
        'true_logit_drop','rival_max_increase','pred_before','pred_after','rival_before','rival_after']
    integers = {'pred_before','pred_after','rival_before','rival_after'}
    result = {key:np.empty((N,K),dtype=np.int64 if key in integers else np.float32) for key in fields}
    masks = np.empty((N//GROUP_BATCH,16),dtype=np.int64)
    mask_generator = torch.Generator(device='cpu').manual_seed(SEEDS['channel'])
    all_replacement_checks = 0
    component_error = 0.
    with torch.no_grad():
        for start in range(0,N,GROUP_BATCH):
            end = start+GROUP_BATCH
            recipient = torch.from_numpy(np.array(bank[receiver_indices[start:end]],copy=True)).cuda().repeat_interleave(K,0)
            donor = torch.from_numpy(np.array(bank[donor_indices[start:end].reshape(-1)],copy=True)).cuda()
            donor_before = donor.clone()
            targets = torch.from_numpy(receiver_classes[start:end]).cuda().repeat_interleave(K)
            target_copy = targets.clone()
            baseline_logits = classifier(recipient)
            m0,true0,rival0,rival_class0 = margin(baseline_logits,targets)
            assert torch.equal(m0.view(-1,K),m0.view(-1,K)[:,0:1].expand(-1,K))
            ambient = torch.get_rng_state().clone()
            predictor = torch.Generator(device='cpu')
            predictor.set_state(mask_generator.get_state())
            selected = torch.randperm(64,generator=predictor)[:16]
            masks[start//GROUP_BATCH] = selected.numpy()
            selected_gpu = selected.to(recipient.device)
            kept = torch.tensor([i for i in range(64) if i not in selected.tolist()],device=recipient.device)
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(mask_generator.get_state())
                fused = methods.H2T(recipient.clone(),donor,rho=.25)
                mask_generator.set_state(torch.get_rng_state())
            assert torch.equal(torch.get_rng_state(),ambient)
            assert torch.equal(mask_generator.get_state(),predictor.get_state())
            assert torch.equal(fused[:,selected_gpu],donor[:,selected_gpu])
            assert torch.equal(fused[:,kept],recipient[:,kept]) and torch.equal(targets,target_copy)
            assert torch.equal(donor,donor_before)
            all_replacement_checks += 1
            fused_logits = classifier(fused)
            m1,true1,rival1,rival_class1 = margin(fused_logits,targets)
            delta = m0-m1
            component_error = max(component_error,float((delta-((true0-true1)+(rival1-rival0))).abs().max()))
            r_pool = F.avg_pool2d(recipient,8).flatten(1)
            h_pool = F.avg_pool2d(donor,8).flatten(1)
            f_pool = F.avg_pool2d(fused,8).flatten(1)
            values = {'baseline_margin':m0,'fused_margin':m1,'D':delta,
                'l2_spatial':torch.linalg.vector_norm((fused-recipient).flatten(1),dim=1),
                'l2_gap':torch.linalg.vector_norm(f_pool-r_pool,dim=1),
                'cosine_spatial':F.cosine_similarity(recipient.flatten(1),donor.flatten(1),dim=1),
                'cosine_gap':F.cosine_similarity(r_pool,h_pool,dim=1),
                'true_logit_drop':true0-true1,'rival_max_increase':rival1-rival0,
                'pred_before':baseline_logits.argmax(1),'pred_after':fused_logits.argmax(1),
                'rival_before':rival_class0,'rival_after':rival_class1}
            for key,value in values.items():
                assert torch.isfinite(value).all()
                result[key][start:end] = value.detach().cpu().numpy().reshape(GROUP_BATCH,K)
            if end % 3200 == 0 or end == N:
                print(f'{mode}: scored {end}/{N} receiver groups',flush=True)
    for i in range(N):
        for a in range(K):
            for b in range(a+1,K):
                if donor_indices[i,a] == donor_indices[i,b]:
                    assert result['D'][i,a] == result['D'][i,b], 'Duplicate donors must have identical cached scores'
    self_mask = donor_indices == receiver_indices[:,None]
    assert np.all(result['D'][self_mask] == 0.) and np.all(result['l2_spatial'][self_mask] == 0.)
    return result,masks,{'receivers':N,'candidate_fusions':N*K,'fusion_batches':all_replacement_checks,
        'replaced_channels':16,'retained_channels':48,'replacement_correct_every_call':True,
        'same_mask_within_K4_and_within_original_batch':True,'ambient_rng_unchanged_every_call':True,
        'recipient_labels_unchanged':True,'duplicate_candidate_scores_exactly_equal':True,
        'donor_features_unchanged_every_call':True,
        'self_donor_candidates':int(self_mask.sum()),'self_donor_margin_and_distance_exactly_zero':True,
        'margin_component_identity_max_float32_error':component_error}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        assert not (ROOT/'run_status.json').exists()
        dump(ROOT/'run_status.json',{'state':'running','started_at':now()})
        with (ROOT/'extraction_scoring.log').open('w',encoding='utf-8',buffering=1) as stream:
            class Tee:
                def __init__(self,console): self.console = console
                def write(self,text): self.console.write(text); stream.write(text)
                def flush(self): self.console.flush(); stream.flush()
            with contextlib.redirect_stdout(Tee(sys.stdout)),contextlib.redirect_stderr(Tee(sys.stderr)):
                try:
                    run()
                    dump(ROOT/'run_status.json',{'state':'completed','finished_at':now(),'training_performed':False})
                except BaseException:
                    dump(ROOT/'run_status.json',{'state':'failed','finished_at':now(),'error':traceback.format_exc()})
                    traceback.print_exc()
                    raise
