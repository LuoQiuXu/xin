"""Final integrity and review package; no training or model evaluation."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile,importlib.metadata
from datetime import datetime,timezone,timedelta
root=Path(__file__).resolve().parent;res=root/'results'
sys.path.insert(0,str(root))
from preflight_checks import history_manifest
old=root.parent/'clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert not (res/'delivery_manifest.json').exists()
assert read(res/'STOP_COMPLETED.json')['updates_total']==1700
baseline=read(res/'historical_readonly_baseline.json');assert history_manifest(old,sha)==baseline
snapshot=res/'run_source_snapshot';snapshot.mkdir()
new_sources={}
for p in sorted(root.glob('*.py')):
    new_sources[p.name]=sha(p);shutil.copy2(p,snapshot/p.name)
for seed in [124,125]:
    run=res/f'shared_seed{seed}'
    q=read(run/'summary.json');protocol=read(run/'protocol.json')
    assert q['updates']==850 and q['epochs']==10 and q['status']=='complete'
    assert all(sha(root/k)==v for k,v in protocol['source_hashes'].items())
    for kind in ['best','final']:
        assert sha(run/f'checkpoints/model_{kind}.pth.tar')==q['checkpoints'][kind]['sha256']
    assert len(list(run.glob('validation_epoch*.npz')))==10
    assert (run/'actual_masks.npz').is_file()
for phase in (res/'execution').iterdir():
    if (phase/'protection.json').is_file():
        p=read(phase/'protection.json');assert not p['TEST_access_events'] and not p['historical_write_events']
versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','Pillow','PyYAML','yacs']}
assert versions['torch']=='2.14.1+cu132'
dump(res/'source_manifest.json',{'new_sources':new_sources,'historical_source_lock_sha256':sha(old/'source_lock.json'),
    'historical_checkpoint_sha256':sha(old/'locked/clean_stage1_best.pth'),'historical_statistics_sha256':sha(old/'class_feature_statistics.npz'),
    'python':sys.executable,'package_versions':versions})
dump(res/'final_verification.json',{'verified_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'passed':True,
    'formal_new_runs':2,'seeds':[124,125],'epochs_each':10,'updates_total':1700,
    'old_control_retraining':False,'StageI_retraining':False,'official_TEST_used':False,
    'historical_files_unchanged':len(baseline),'training_sources_unchanged_after_start':True,
    'all_13_historical_fields_exact':False,'continuation_authorized_by_user':True,
    'all_shared_masks_equal_historical_first_Uniform':True,'frozen_BB_BN_saved_states_exact':True,
    'four_best_final_checkpoint_hashes_verified':True,'new_source_manifest':str(res/'source_manifest.json')})
package=res/'ChatGPT_REVIEW_PACKAGE.zip';assert not package.exists()
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(res.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(res)
        if p.suffix.lower() in ['.md','.csv','.json','.log','.yaml','.py','.diff','.original']:
            archive.write(p,rel.as_posix())
    archive.write(root/'CODEX_MASK_GRANULARITY_INSTRUCTIONS.txt','instructions/CODEX_MASK_GRANULARITY_INSTRUCTIONS.txt')
files={p.relative_to(res).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(res.rglob('*')) if p.is_file()}
dump(res/'delivery_manifest.json',{'sealed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'files':files,'no_further_training':True,'review_package_sha256':sha(package)})
print('FINAL PASS:two Shared runs,1700 updates; '+str(len(baseline))+' historical files unchanged')
print('Report: '+str(res/'MASK_GRANULARITY_RESULTS.md'))
print('Review package: '+str(package)+' ('+str(package.stat().st_size)+' bytes)')
