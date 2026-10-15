"""Prepare the exact existing LT + TRAIN holdout protocol; no Dataset TEST load."""
import pickle
import shutil
from common import *


def main():
    install_test_guard('prepare')
    assert not (ROOT/'preflight.json').exists(), 'Refuse overwriting prepared experiment'
    protected = [p for p in BASE.rglob('*') if p.is_file() and not p.is_relative_to(ROOT)]
    protected += list(WORK.glob('H2T_*.md'))+list(WORK.glob('SMART_*.md'))
    protected += [WORK/'VS_STAGE2_RESULTS.md',WORK/'DATASET_PROTOCOL_AUDIT.md']
    dump(ROOT/'historical_before.json',{str(p):sha(p) for p in sorted(set(protected))})
    manifest = read(BASE/'evidence/runtime_manifest.json')['files']
    for rel, expected in manifest.items():
        assert sha(BASE/'runtime'/rel) == expected
        dest = RUNTIME/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/'runtime'/rel,dest)
        assert sha(dest) == expected
    for name in ['source_manifest.json','runtime_manifest.json','compatibility.diff']:
        shutil.copy2(BASE/'evidence'/name,ROOT/name)
    prior_s1 = BASE/'formal_runs/formal_20261011_1429_r2'
    shutil.copy2(prior_s1/'effective.yaml',ROOT/'stage1_original_effective.yaml')
    shutil.copy2(OLD/'arms/Uniform-PerSample/effective.yaml',ROOT/'stage2_original_effective.yaml')
    for name in ['split_indices.npz','train_images.npy','train_labels.npy','validation_images.npy',
                 'validation_labels.npy','validation_class_protocol.csv','fixed_sample_ids.csv','VALIDATION_PROTOCOL.md']:
        shutil.copy2(OLD/name,ROOT/name)
        assert sha(ROOT/name) == sha(OLD/name)
    # Only re-use input sampling/noise; no old model, statistics or feature bank.
    plan = np.load(OLD/'sampling_and_masks.npz',allow_pickle=False)
    np.savez_compressed(ROOT/'sampling_plan.npz',receivers=plan['receivers'],donors=plan['donors'])
    shutil.copy2(OLD/'gumbel_noise_float64.npy',ROOT/'gumbel_noise_float64.npy')
    shutil.copy2(OLD/'reused_vs_math.py',ROOT/'reused_vs_math.py')
    assert sha(ROOT/'reused_vs_math.py') == sha(OLD/'reused_vs_math.py')
    expected = read(BASE/'evidence/preflight.json')
    assert sha(DATA/'cifar-100-python/train') == expected['integrity']['train']['sha256']
    with integrity_only():
        md5_test = sha(DATA/'cifar-100-python/test','md5')
    assert md5_test == 'f0ef6b0ae62326f3e7ffdfab6717acfc'
    with (DATA/'cifar-100-python/train').open('rb') as f:
        raw = pickle.load(f,encoding='bytes')
    images = raw[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1).copy()
    labels = np.array(raw[b'fine_labels'],dtype=np.int64)
    split = np.load(ROOT/'split_indices.npz',allow_pickle=False)
    ti,vi = split['train_raw_indices'],split['validation_raw_indices']
    rng = np.random.RandomState(0)
    ids = []
    counts = np.array([int(500*(.01**(c/99.))) for c in range(100)])
    for c,n in enumerate(counts):
        pool = np.flatnonzero(labels == c)
        rng.shuffle(pool)
        ids.extend(pool[:n])
    assert np.array_equal(ids,ti) and len(ti) == 10847 and len(vi) == 1980
    for s,i in [('train',ti),('validation',vi)]:
        assert np.array_equal(images[i],np.load(ROOT/f'{s}_images.npy',allow_pickle=False))
        assert np.array_equal(labels[i],np.load(ROOT/f'{s}_labels.npy',allow_pickle=False))
    assert np.bincount(labels[ti],minlength=100).tolist() == counts.tolist() == expected['class_counts']
    assert np.bincount(labels[vi],minlength=100).tolist() == [0]+[20]*99
    assert np.intersect1d(ti,vi).size == 0
    train_hashes = {hashlib.sha256(im.tobytes()).digest() for im in images[ti]}
    assert not any(hashlib.sha256(im.tobytes()).digest() in train_hashes for im in images[vi])
    pixel_hash = hashlib.sha256(images[ti].tobytes()).hexdigest()
    assert pixel_hash == read(prior_s1/'preflight.json')['subset_sha256']
    assert all(np.array_equal(np.sort(d),np.arange(N)) for d in plan['donors'])
    assert int(plan['receivers'].min()) >= 0 and int(plan['receivers'].max()) < N
    cfg1,tr,_,_,_,_ = imports(1)
    cfg2,_,_,_,_,_ = imports(2)
    (ROOT/'clean_stage1_effective.yaml').write_text(cfg1.dump(),encoding='utf-8')
    (ROOT/'clean_stage2_effective.yaml').write_text(cfg2.dump(),encoding='utf-8')
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    p = torch.randn((64,64),device='cuda')
    assert torch.isfinite(p @ p).all()
    flight = {'prepared_at':now(),'python':sys.executable,'python_version':sys.version,
        'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/1024**2,
        'data_path':str(DATA),'StageI_seed':42,'StageII_seed':123,'runtime_files':manifest,
        'LT_samples':N,'LT_classes':100,'LT_class_counts':counts.tolist(),'LT_pixel_hash':pixel_hash,
        'train_raw_ID_sha256':array_hash(ti),'validation_raw_ID_sha256':array_hash(vi),
        'split_file_sha256':sha(ROOT/'split_indices.npz'),'validation_samples':1980,'missing_classes':[0],
        'train_val_ID_overlap':0,'train_val_pixel_overlap':0,'test_integrity_MD5':md5_test,
        'old_checkpoint_loading_enabled':False,'old_statistics_loading_enabled':False,
        'original_train_transform':str(tr['train_transform']),'original_eval_transform':str(tr['eval_transform'])}
    dump(ROOT/'preflight.json',flight)
    dump(ROOT/'protocol.json',{'declared_at':now(),'stage1_seed':42,'stage2_seed':123,
        'StageI_epochs':200,'StageII_epochs':10,'batch_size':128,'arms':ARMS,
        'validation':'same 1980 unseen TRAIN raw IDs, class1..99,20/class; 100 outputs retained',
        'StageI_selection':'max validation Overall over all200 epochs, earliest tie; no early stopping',
        'StageI_training_RNG':'Python/NumPy/Torch seed42; subset seed0 verified in isolated RandomState; frozen array load consumes no training RNG',
        'StageI_recipe':'original ResNet32,SGD lr.1,momentum.9,wd.0002,CE+MixUp alpha1,warmup5,milestones160/180,workers0,cudnn deterministicTrue',
        'StageII_recipe':'frozen Clean best Backbone/BN; fc init from same Clean best; original CE,SGD lr.1*cosine factor.2,momentum.9,wd.0005',
        'statistics':'USER CONFIRMED: preserve old files; recompute LT-only stats using Clean best Backbone, exact existing formulas/shrinkage/epsilon definition',
        'shrinkage':'lambda=5/(n+5), pooled=sum(n*var_ddof0)/(N-100)',
        'epsilon':'1e-6 * mean(per-channel global LT feature variance ddof1)',
        'primary':'fixed epoch10 99-class TRAIN holdout Overall; Head/OfficialMedium/Tail,Many/Medium/Few,Macro also report',
        'rho':.25,'channels':64,'replace_channels':16,'Top48':48,
        'fallback':'donor n<20; ptp(S)<=1e-12; nonfinite stop; unchanged IS class distribution',
        'sampling':'exact saved seed123 BS/IS indices and Gumbel noise; online native aug seed123+300009 isolated; identical three-arm input/control hashes',
        'TEST':'forbidden file/Dataset loads except preflight opaque MD5; guard violation aborts before StageII',
        'no_search':True,'no_extra_seeds':True,'historical_files_read_only':True})
    guard_pass()
    check_history()
    print(f'PREPARE PASS: exact10847 LT,1980 unseen TRAIN val,0 overlaps; fresh seed42 StageI; GPU={flight["gpu"]}',flush=True)


if __name__ == '__main__':
    main()
