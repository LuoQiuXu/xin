"""Exactly124/125; two methods, fixed Clean basis, no Stage I or TEST entry."""
import argparse
import shutil
from common import *
from stability_models import plan_dir,outdir,PAIR_KEYS,check_sources


def generate_training_loop():
    source = (ORIGIN/'clean_stage2.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['dryrun','train_one']]
    body = '\n\n'.join(ast.get_source_segment(source,n) for n in selected)
    changes = {
        'def dryrun():':'def dryrun(seed):',
        "    assert not (ROOT/'arms').exists()":"    assert not any(outdir(seed,a).exists() for a in ARMS)",
        '    lock_stage2()\n':'',
        '    check_stage2_sources()':'    check_sources()',
        'def train_one(arm):':'def train_one(seed,arm):',
        "ROOT/'arms'/arm":"outdir(seed,arm)",
        "ROOT/'arms'/ARMS[0]":"outdir(seed,ARMS[0])",
        'frozen_models()':'frozen_models(seed)',
        "ROOT/'sampling_plan.npz'":"plan_dir(seed)/'sampling_plan.npz'",
        "ROOT/'gumbel_noise_float64.npy'":"plan_dir(seed)/'gumbel_noise_float64.npy'",
        'S2_SEED+300009':'seed+300009',
        "'seed':123":"'seed':seed",
        "ROOT/'stage2_dryrun_verification.json'":"plan_dir(seed)/'dryrun_verification.json'",
        "ROOT/'online_clean_scores_float64.npy'":"plan_dir(seed)/'online_scores_float64.npy'",
        'all_three_masks_sha256':'all_two_masks_sha256',
        'CLEAN STAGE II DRYRUN PASS:three':'STABILITY DRYRUN PASS:two',
        "cfg.name = ROOT.name+'_'+arm":"cfg.name = ROOT.name+f'_seed{seed}_'+arm",
        "experiment.update({'arm':arm,":"experiment.update({'arm':arm,'seed':seed,",
    }
    for before,after in changes.items():
        assert before in body, before
        body = body.replace(before,after)
    header = '"""Reviewed seed123 training/dryrun loops; only routing,seed and two-arm trace adapted."""\nfrom common import *\nfrom stability_models import *\nfrom reused_train_helpers import *\nfrom online_policy import online_masks,selected_statistics\n\n'
    text = header+body+'\n'
    assert 'frozen_models()' not in text and 'S2_SEED' not in text and "'seed':123" not in text
    ast.parse(text)
    (ROOT/'training_loop.py').write_text(text,encoding='utf-8')
    import difflib
    original = '\n\n'.join(ast.get_source_segment(source,n) for n in selected)+'\n'
    diff = ''.join(difflib.unified_diff(original.splitlines(True),body.splitlines(True),
        fromfile='clean_seed123/dryrun_train_one',tofile='stability/dryrun_train_one'))
    (ROOT/'training_loop.adaptation.diff').write_text(diff,encoding='utf-8')


def sampling(seed,labels):
    # Identical seed123 generator definitions, using isolated streams.
    pools = [np.flatnonzero(labels == c).tolist() for c in range(100)]
    bs = random.Random(seed)
    receivers = np.empty((10,N),dtype=np.int64)
    donors = np.empty_like(receivers)
    is_rng = torch.Generator(device='cpu').manual_seed(seed+100003)
    for epoch in range(10):
        receivers[epoch] = [bs.choice(pools[bs.randint(0,99)]) for _ in range(N)]
        donors[epoch] = torch.randperm(N,generator=is_rng).numpy()
    noise_rng = np.random.default_rng(seed+700021)
    u = noise_rng.random((10,N,64),dtype=np.float64)
    u = np.clip(u,np.nextafter(0.,1.),np.nextafter(1.,0.))
    g = -np.log(-np.log(u))
    assert np.isfinite(g).all()
    return receivers,donors,g


def plans():
    assert not (ROOT/'source_lock.json').exists(), 'Immutable plans/source cannot be rebuilt'
    labels = np.load(ROOT/'train_labels.npy')
    # This is an input-stream verification, never seed123 model training.
    rec,don,noise = sampling(123,labels)
    origin_plan = np.load(ORIGIN/'sampling_plan.npz')
    assert np.array_equal(rec,origin_plan['receivers']) and np.array_equal(don,origin_plan['donors'])
    assert np.array_equal(noise,np.load(ORIGIN/'gumbel_noise_float64.npy',mmap_mode='r'))
    dump(ROOT/'RNG_factory_seed123_replay.json',{'verified_at':now(),'receivers_equal':True,'donors_equal':True,
        'Gumbel_float64_equal':True,'seed123_training_executed':False,'only_input_reconstruction':True})
    seeds_meta = {}
    for seed in SEEDS:
        path = plan_dir(seed);path.mkdir(parents=True)
        rec,don,noise = sampling(seed,labels)
        assert all(np.array_equal(np.sort(d),np.arange(N)) for d in don)
        assert int(rec.min()) >= 0 and int(rec.max()) < N
        np.savez_compressed(path/'sampling_plan.npz',receivers=rec,donors=don)
        np.save(path/'gumbel_noise_float64.npy',noise)
        seeds_meta[str(seed)] = {'BS_IS_array_hash':array_hash(rec,don),'Gumbel_array_hash':array_hash(noise),
            'draws_per_epoch':N,'IS_class_counts_each_epoch_exact':True,'augmentation_seed':seed+300009,
            'BS_seed':seed,'IS_seed':seed+100003,'Gumbel_seed':seed+700021}
    assert seeds_meta['124']['BS_IS_array_hash'] != seeds_meta['125']['BS_IS_array_hash']
    assert seeds_meta['124']['Gumbel_array_hash'] != seeds_meta['125']['Gumbel_array_hash']
    generate_training_loop()
    files = [p for p in ROOT.glob('*.py')]
    files += [ROOT/'stage2_original_effective.yaml',ROOT/'protocol.json']
    files += [RUNTIME/rel for rel in read(ROOT/'preflight.json')['runtime_files']]
    data = ['locked/clean_stage1_best.pth','class_feature_statistics.npz','validation_feature_maps.npy',
        'split_indices.npz','train_images.npy','train_labels.npy','validation_labels.npy']
    data += [f'plans/{s}/{n}' for s in SEEDS for n in ['sampling_plan.npz','gumbel_noise_float64.npy']]
    lock = {'locked_at':now(),'files':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'data_files':{p:sha(ROOT/p) for p in data},'seed_streams':seeds_meta}
    dump(ROOT/'source_lock.json',lock)
    snapshot = ROOT/'source_snapshot'
    for p in files:
        dest = snapshot/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
    dump(snapshot/'source_manifest.json',lock)
    guard_pass();check_history()
    print('STABILITY PLANS PASS:seed123 inputs reproduced exactly; independent124/125 streams frozen',flush=True)


def dryrun(seed):
    assert not (plan_dir(seed)/'dryrun_verification.json').exists()
    check_sources()
    from training_loop import dryrun as reviewed_dryrun
    torch.cuda.reset_peak_memory_stats()
    reviewed_dryrun(seed)
    result = read(plan_dir(seed)/'dryrun_verification.json')
    result.update({'seed':seed,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
        'fresh_formal_model_and_empty_SGD_required':True})
    assert result['passed'] and result['peak_allocated_mib'] < .8*torch.cuda.get_device_properties(0).total_memory/1024**2
    dump(plan_dir(seed)/'dryrun_verification.json',result)
    guard_pass();check_history()
    print(f'DRYRUN seed{seed} PASS:real inputs,BB/BN frozen,fc update,16 channels,peak={result["peak_allocated_mib"]:.2f}MiB',flush=True)


def train_pair(seed):
    assert read(plan_dir(seed)/'dryrun_verification.json')['passed']
    assert not (plan_dir(seed)/'STOP_COMPLETED.json').exists()
    from training_loop import train_one
    for arm in ARMS:
        check_sources();train_one(seed,arm)
    rows = {}
    for arm in ARMS:
        with (outdir(seed,arm)/'batches.csv').open(encoding='utf-8') as f:rows[arm] = list(csv.DictReader(f))
        assert len(rows[arm]) == 850
    equal = {k:sum(a[k] == b[k] for a,b in zip(rows[ARMS[0]],rows[ARMS[1]])) for k in PAIR_KEYS}
    assert all(n == 850 for n in equal.values())
    pairing = {'seed':seed,'batches_each':850,'equal_control_rows':equal,
        'same_initialization_hashes':read(outdir(seed,ARMS[0])/'status.json')['initial_state_hashes'],
        'same_inputs_BS_IS_augmentation_features_scores_noise':True,'only_mask_differs':True,
        'independent_empty_SGD':True,'verified_at':now()}
    dump(plan_dir(seed)/'pairing_hash.json',pairing)
    shared = read(ROOT/'pairing_hash.json') if (ROOT/'pairing_hash.json').exists() else {}
    shared[str(seed)] = pairing;dump(ROOT/'pairing_hash.json',shared)
    plan = np.load(plan_dir(seed)/'sampling_plan.npz')
    masks = {a.replace('-','_'):np.load(outdir(seed,a)/'actual_masks.npz')['masks'] for a in ARMS}
    assert all(m.shape == (10,N,16) and np.all(np.diff(np.sort(m,axis=-1),axis=-1) > 0) for m in masks.values())
    np.savez_compressed(plan_dir(seed)/'sampling_masks.npz',receivers=plan['receivers'],donors=plan['donors'],**masks)
    guard_pass();check_sources();check_history()
    dump(plan_dir(seed)/'STOP_COMPLETED.json',{'completed_at':now(),'seed':seed,'methods':ARMS,
        'epochs_each':10,'updates_each':850,'TEST_used':False,'StageI_retrained':False})
    if all((plan_dir(s)/'STOP_COMPLETED.json').exists() for s in SEEDS):
        dump(ROOT/'STOP_COMPLETED.json',{'completed_at':now(),'seeds':SEEDS,'methods':ARMS,
            'epochs_each':10,'total_formal_updates':3400,'TEST_used':False,'StageI_retrained':False,
            'extra_training_requires_explicit_authorization':True})
    print(f'COMPLETE STABILITY seed{seed}:exactly2methods,10epochs each; no TEST',flush=True)


def main():
    p = argparse.ArgumentParser();p.add_argument('action',choices=['plans','dryrun','train'])
    p.add_argument('--seed',type=int,choices=SEEDS)
    args = p.parse_args()
    assert args.action == 'plans' or args.seed in SEEDS
    install_test_guard(args.action+('_'+str(args.seed) if args.seed else ''))
    try:
        if args.action == 'plans':plans()
        elif args.action == 'dryrun':dryrun(args.seed)
        else:train_pair(args.seed)
    except BaseException:
        import traceback
        dump(ROOT/f'failure_{args.action}_{args.seed}.json',{'at':now(),'traceback':traceback.format_exc()})
        guard_flush();raise


if __name__ == '__main__':
    main()
