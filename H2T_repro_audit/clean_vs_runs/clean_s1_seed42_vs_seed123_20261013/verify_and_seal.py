"""Final artifact checks and seal after all training / report log streams close."""
from common import *


def main():
    install_test_guard('final_verification')
    previous = read(ROOT/'delivery_manifest.json')
    mismatches = [r for r,v in previous['files'].items() if not (ROOT/r).exists() or sha(ROOT/r) != v]
    # The report writes its final stdout message after its first seal; the Tee
    # stream is now closed, so include the complete reporting log in this seal.
    assert all(p == 'report_full_console.log' for p in mismatches), mismatches
    check_sources()
    s2 = read(ROOT/'stage2_source_lock.json')
    for group in ['files','data_files']:
        for p,value in s2[group].items():
            assert sha(ROOT/p) == value, ('changed training source or input',p)
    for name,value in previous['root_reports'].items():
        assert sha(WORK/name) == value == sha(ROOT/name)
    # Existing validation outputs and all saved checkpoint metadata were
    # verified by the training/reload gates. Recheck the report's CSV counts.
    with (ROOT/'stage1/epoch_validation_metrics.csv').open(encoding='utf-8') as f:
        epochs = list(csv.DictReader(f))
    assert len(epochs) == 200 and [int(r['epoch']) for r in epochs] == list(range(1,201))
    counts = {}
    for arm in ARMS:
        with (ROOT/'arms'/arm/'batches.csv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 850 and sum(int(r['size']) for r in rows) == 10*N
        assert all(np.isfinite(float(r['loss'])) for r in rows)
        checks = read(ROOT/'arms'/arm/'checkpoint_validation.json')
        assert all(x['validation_reload_exact'] and x['backbone_BN_saved_bitwise_exact'] for x in checks.values())
        counts[arm] = {'epochs':10,'updates':850,'sample_draws':10*N,'checkpoint_reloads_verified':2}
    masks = np.load(ROOT/'sampling_masks.npz',allow_pickle=False)
    for arm in ARMS:
        m = masks[arm.replace('-','_')]
        assert m.shape == (10,N,16) and np.all(np.diff(np.sort(m,axis=-1),axis=-1) > 0)
    with (ROOT/'per_class_epoch10.csv').open(encoding='utf-8') as f:
        per = list(csv.DictReader(f))
    assert len(per) == 100 and per[0]['validation_samples'] == '0'
    assert all(per[0][a] == '' for a in ARMS)
    assert all(r['validation_samples'] == '20' for r in per[1:])
    for p in ROOT.glob('test_guard_*.json'):
        g = read(p)
        assert not g['blocked_attempts'] and g['CIFAR100_trainFalse_calls'] == 0
        assert g['official_test_image_or_label_loads'] == 0
    guard_pass()
    historical = check_history()
    result = {'verified_at':now(),'passed':True,'StageI_epochs':200,'StageI_updates':17000,
        'StageII':counts,'all_masks16_unique_channels':True,'validation_classes':99,'validation_samples':1980,
        'class0_preserved_output_missing_validation':True,'TEST_used':False,
        'historical_protected_files':historical['protected_files'],'historical_files_unchanged':True,
        'previous_report_live_log_resealed':mismatches}
    dump(ROOT/'artifact_verification.json',result)
    files = {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
             if p.is_file() and p.name != 'delivery_manifest.json'}
    previous.update({'sealed_at':now(),'files':files,'file_count':len(files),'verification_passed':True})
    dump(ROOT/'delivery_manifest.json',previous)
    assert all(sha(ROOT/r) == v for r,v in files.items())
    print(json.dumps({'verified_files':len(files),'old_files_unchanged':historical['protected_files'],
        'mask_draws_each':10*N,'TEST_used':False,'stopped':True},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
