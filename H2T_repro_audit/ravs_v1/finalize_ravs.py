"""Final compact integrity manifest and optional user review package; no training."""
from pathlib import Path
import json,hashlib,zipfile,csv,sys,importlib.metadata
from datetime import datetime,timezone,timedelta
from run_safeguards import history_manifest
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
OLD=ROOT.parent/'clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert not (RESULTS/'delivery_manifest.json').exists(),'Refuse resealing completed output'
stop=read(RESULTS/'STOP_COMPLETED.json');assert stop['updates_total']==8500 and stop['new_runs']==10
registry=read(RESULTS/'run_registry.json')
assert len([x for x in registry if x['action']=='train'])==10
assert all(x['status']=='completed' and not x['TEST_events'] and not x['historical_write_events'] for x in registry)
pre=read(RESULTS/'preflight.json')
assert pre['status']=='PASS' and all(sha(ROOT/k)==v for k,v in pre['source_hashes'].items())
baseline=read(RESULTS/'historical_manifest.json')
assert history_manifest(OLD,sha)==baseline
with (RESULTS/'checkpoint_manifest.csv').open(encoding='utf-8') as f:cp=list(csv.DictReader(f))
assert len(cp)==20 and all(sha(x['path'])==x['sha256'] for x in cp)
assert sum(1 for _ in RESULTS.glob('seed*/*/summary.json'))==10
summary=read(RESULTS/'scientific_summary.json')
assert len(summary['rows'])==10 and len(summary['paired'])==20
assert all(v==850 for counts in summary['five_field_matching'].values() for v in counts.values())
assert sha(ROOT/'RAVS_V1_SCIENTIFIC_REPORT.md')==sha(RESULTS/'RAVS_V1_SCIENTIFIC_REPORT.md')
versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','Pillow','PyYAML','yacs']}
dump(RESULTS/'final_integrity.json',{'verified_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'passed':True,'new_runs':10,'epochs_each':10,'updates_total':8500,'fully_matched':True,
    'historical_files_unchanged':len(baseline),'official_TEST_used':False,'StageI_retraining':False,
    'new_source_sha256':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},
    'checkpoint_files_verified':20,'Go':summary['gate_decision']['go'],'python':sys.executable,
    'package_versions':versions,'training_core_unchanged_after_preflight':True})
package=RESULTS/'RAVS_V1_REVIEW_PACKAGE.zip';assert not package.exists()
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(RESULTS.rglob('*')):
        if p.is_file() and (p.suffix.lower() in ['.md','.json','.csv','.log','.yaml','.patch'] or p.name=='validation_epoch10.npz'):
            z.write(p,p.relative_to(RESULTS).as_posix())
    for p in sorted(ROOT.glob('*.py')):z.write(p,'source/'+p.name)
    for name in ['README.md','CODEX_RAVS_H2T_V1_INSTRUCTIONS.txt']:
        z.write(ROOT/name,'source/'+name)
manifest={p.relative_to(RESULTS).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
          for p in sorted(RESULTS.rglob('*')) if p.is_file()}
dump(RESULTS/'delivery_manifest.json',{'sealed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'files':manifest,'review_package_sha256':sha(package),'completed':True,'no_further_training':True})
print('FINAL INTEGRITY PASS:10 runs,8500 updates,all5 fields850/850,20 checkpoint hashes; '+str(len(baseline))+' historical files unchanged')
print('Report: '+str(ROOT/'RAVS_V1_SCIENTIFIC_REPORT.md'))
print('Review package: '+str(package)+' ('+str(package.stat().st_size)+' bytes)')
print('STOP: Go='+str(summary['gate_decision']['go']))
