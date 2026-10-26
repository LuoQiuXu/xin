"""Validate the new report bundle and the read-only source boundary."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

RUN=Path(__file__).resolve().parent
REPO=RUN.parents[3]

def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def git(*args):
    p=subprocess.run(['git',*args],cwd=REPO,capture_output=True,text=True,encoding='utf-8')
    assert p.returncode==0,(args,p.stderr)
    return p.stdout

def save(name,value):
    with (RUN/name).open('x',encoding='utf-8',newline='\n') as h:
        json.dump(value,h,ensure_ascii=False,indent=2);h.write('\n')

def main():
    assert git('branch','--show-current').strip()=='research/h2t-pih2t-gap-20261026'
    assert git('rev-parse','HEAD').strip()=='de2198a85117a46972d294fb4ae09951b3dbb018'
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')
    assert not git('ls-files','--unmerged')
    git('diff','--check')
    required=['RESEARCH_REPORT.md','AUTHOR_MECHANISM_CROSSWALK.md','PI_PAPER_CODE_PARITY.md',
              'PI_PAPER_CODE_PARITY.csv','PRIOR_EXPERIMENT_EVIDENCE.md','ONE_QUESTION_DECISION.md',
              'EXECUTION_REPORT.md','EVIDENCE_INDEX.csv','SOURCE_MANIFEST.json','NEXT_EXPERIMENT_PREREG.md']
    assert all((RUN/name).is_file() for name in required)
    for path in RUN.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    for path in RUN.glob('*.json'):json.loads(path.read_text(encoding='utf-8'))
    manifest=json.loads((RUN/'SOURCE_REVIEW_MANIFEST.json').read_text(encoding='utf-8'))
    manifest+=json.loads((RUN/'ADDITIONAL_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    assert len(manifest)==44
    for item in manifest:assert sha(REPO/item['path'])==item['sha256'],item['path']
    source=json.loads((RUN/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for path,digest in source['sources'].items():assert sha(REPO/path)==digest,path
    with (RUN/'EVIDENCE_INDEX.csv').open(encoding='utf-8-sig',newline='') as h:
        assert len(list(csv.DictReader(h)))==10
    with (RUN/'PI_PAPER_CODE_PARITY.csv').open(encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h))
    assert len(rows)==23
    references=[]
    for row in rows:
        for key in ['historical_status','review_status']:assert row[key] in {'MATCH','MISMATCH','UNVERIFIED'}
        for key in ['historical_code','review_code']:
            path,number=row[key].rsplit(':',1);n=int(number)
            p=REPO/path;lines=p.read_text(encoding='utf-8-sig').splitlines()
            assert 1<=n<=len(lines)
            references.append({'field':row['field'],'kind':key,'path':path,'line':n,'source_line':lines[n-1]})
    recalc=json.loads((RUN/'HISTORICAL_RECALC.json').read_text(encoding='utf-8'))
    assert recalc['CE']['strict_gate_all_seeds_B_gt_A_and_C'] is True
    assert recalc['PIF']['strict_gate_all_seeds_B_gt_A_and_C'] is False
    assert abs(recalc['endpoint_pp']['T3-T2']-(-10.87))<1e-4
    static=json.loads((RUN/'STATIC_PARITY_CHECKS.json').read_text(encoding='utf-8'))
    assert static['historical_PIF_parameter_count']==25 and static['review_PIF_parameter_count']==2
    assert 'STOP_AND_REPAIR_PARITY' in (RUN/'ONE_QUESTION_DECISION.md').read_text(encoding='utf-8')
    assert 'WITHHELD' in (RUN/'NEXT_EXPERIMENT_PREREG.md').read_text(encoding='utf-8')
    with (RUN/'PAPER_PAGE_LEDGER.csv').open(encoding='utf-8-sig',newline='') as h:pages=list(csv.DictReader(h))
    assert len(pages)==93 and sum(p['visual_review']=='True' for p in pages)==11
    papers=json.loads((RUN/'PAPER_ACCESS.json').read_text(encoding='utf-8'))
    for item in papers:
        assert item['status']=='FULL_PDF_TEXT_OPENED'
        assert sha(RUN/item['local_pdf'])==item['sha256']
    baseline=json.loads((RUN/'paper_cache/PROTECTED_BASELINE.json').read_text(encoding='utf-8'))
    for item in baseline['tracked_text']:assert sha(REPO/item['path'])==item['sha256'],item['path']
    old={item['path']:item for item in baseline['historical_assets_metadata']}
    current={}
    for name in ['fusion_semantic_audit','lowvs_mask_parity','mdlt_domain_probe']:
        for p in (REPO/'H2T_repro_audit'/name).rglob('*'):
            if p.is_file():
                key=p.relative_to(REPO).as_posix()
                current[key]={'path':key,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
    assert current==old
    statuses=git('status','--porcelain=v1').splitlines()
    allowed=['H2T_repro_audit/research_gap_20261026/results/',
             'H2T_repro_audit/fusion_semantic_audit/','H2T_repro_audit/lowvs_mask_parity/','H2T_repro_audit/mdlt_domain_probe/']
    assert all(line[:3]=='?? ' and any(line[3:].startswith(prefix) for prefix in allowed) for line in statuses),statuses
    # Only small first-party audit artifacts are selected for publication; source
    # PDFs, their full transcriptions, renderings and integrity scratch stay local.
    publish=[p for p in RUN.iterdir() if p.is_file() and p.suffix in {'.md','.csv','.json','.py'} and not p.name.endswith('_PAGES.json')]
    assert all(p.stat().st_size<250_000 for p in publish)
    result={'passed':True,'branch':git('branch','--show-current').strip(),'audit_base_HEAD':git('rev-parse','HEAD').strip(),
            'unit_tests':{'passed':5,'initial_environment_error':True,'source_tests_changed':False},
            'local_anchors':10,'reviewed_sources_hashed':44,'parity_fields':23,'source_references':len(references),
            'paper_PDFs':6,'paper_pages':93,'visually_checked_pages':11,'history_rows':18,
            'CE_gate':True,'PIF_gate':False,'tracked_text_hashes_unchanged':len(baseline['tracked_text']),
            'historical_asset_metadata_unchanged':len(old),'historical_asset_content_hash_check':'not claimed; metadata+no-write scope only',
            'git_diff_check':'PASS','tracked_changes':[],'conflicts':[],'status_before_commit':statuses,
            'decision':'STOP_AND_REPAIR_PARITY','collapse_status':'REPRODUCTION_UNRESOLVED',
            'GPU_experiment':False,'model_forward':False,'checkpoint_deserialized':False,'official_TEST_accessed':False,
            'new_files_scope':str(RUN.relative_to(REPO)),'publication_byte_count_before_validation_files':sum(p.stat().st_size for p in publish)}
    save('PARITY_REFERENCE_VALIDATION.json',references)
    save('FINAL_VALIDATION.json',result)
    hashes={p.name:sha(p) for p in sorted(RUN.iterdir()) if p.is_file() and p.suffix in {'.md','.csv','.json','.py'} and not p.name.endswith('_PAGES.json')}
    save('OUTPUT_HASHES.json',hashes)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
