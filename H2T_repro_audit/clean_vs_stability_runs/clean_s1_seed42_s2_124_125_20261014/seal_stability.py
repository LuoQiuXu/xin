"""Final integrity seal of the completed four runs, analysis, report and new root report copy."""
from common import *
from stability_models import check_sources,outdir,plan_dir
import shutil


def main():
    install_test_guard('stability_final_verification')
    assert read(ROOT/'STOP_COMPLETED.json')['seeds'] == SEEDS
    assert not (ROOT/'delivery_manifest.json').exists(), 'Refuse changing a delivered seal'
    check_sources()
    report = ROOT/'CLEAN_VS_MULTI_SEED_STABILITY_REPORT.md'
    assert report.is_file()
    summary = read(ROOT/'results_summary.json')
    assert summary['TEST_used'] is False
    expected = {f'{prefix}_seed{seed}' for seed in SEEDS for prefix in ['uniform','lowvs']}
    actual = {p.name for p in ROOT.iterdir() if p.is_dir() and ('_seed124' in p.name or '_seed125' in p.name)}
    assert actual == expected
    for seed in SEEDS:
        assert read(plan_dir(seed)/'dryrun_verification.json')['passed']
        assert read(plan_dir(seed)/'STOP_COMPLETED.json')['seed'] == seed
        for arm in ARMS:
            assert read(outdir(seed,arm)/'status.json')['state'] == 'completed'
            assert len(read(outdir(seed,arm)/'status.json')['epochs']) == 10
    with (ROOT/'epoch_validation_metrics.csv').open(encoding='utf-8') as f:
        assert len(list(csv.DictReader(f))) == 40
    with (ROOT/'per_class_delta.csv').open(encoding='utf-8') as f:
        assert len(list(csv.DictReader(f))) == 200
    with (ROOT/'checkpoint_manifest.csv').open(encoding='utf-8') as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == 8 and all(sha(Path(r['path'])) == r['sha256'] for r in saved)
    analysis_sources = {}
    snapshot = ROOT/'source_snapshot/analysis'
    snapshot.mkdir()
    for name in ['analyze_stability.py','write_stability_report.py','seal_stability.py']:
        analysis_sources[name] = sha(ROOT/name)
        shutil.copy2(ROOT/name,snapshot/name)
    dump(ROOT/'analysis_source_lock.json',{'locked_at':now(),'files':analysis_sources})
    root_report = WORK/report.name
    assert not root_report.exists(), 'Refuse overwriting any historical report'
    shutil.copy2(report,root_report)
    assert sha(root_report) == sha(report)
    guard_pass()
    guards = {p.name:read(p) for p in ROOT.glob('test_guard_*.json')}
    for name,g in guards.items():
        assert not g['blocked_attempts'] and g['official_test_image_or_label_loads'] == 0
        assert g['CIFAR100_trainFalse_calls'] == 0
        if name != 'test_guard_prepare.json':
            assert g['allowed_integrity_open_count'] == 0
    history = check_history()
    dump(ROOT/'final_verification.json',{'verified_at':now(),'passed':True,
        'seeds':SEEDS,'methods':ARMS,'formal_runs':4,'epochs_each':10,'formal_updates':3400,
        'StageI_retrained':False,'additional_seeds_or_strategies':False,
        'TEST_images_or_labels_loaded':0,'TEST_evaluation_or_model_selection':False,
        'guard_phases':list(guards),'historical_files_unchanged':history['protected_files'],
        'fixed_checkpoint_sha256':S1_SHA,'fixed_statistics_sha256':STAT_SHA,
        'report_path':str(report),'root_report_copy':str(root_report),'report_sha256':sha(report),
        'all_sources_and_locked_inputs_unchanged':True,'all_saved_checkpoint_hashes_verified':True})
    # Run without a Tee log: the output below must not mutate any sealed file.
    files = {p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size}
        for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name != 'delivery_manifest.json'}
    dump(ROOT/'delivery_manifest.json',{'sealed_at':now(),'files':files,'root_report_sha256':sha(root_report),
        'report_copy':str(root_report),'immutable_after_seal':True})
    print(f'FINAL INTEGRITY PASS:{len(files)}new files sealed;{history["protected_files"]} historical files unchanged',flush=True)
    print(str(root_report),flush=True)


if __name__ == '__main__':
    main()
