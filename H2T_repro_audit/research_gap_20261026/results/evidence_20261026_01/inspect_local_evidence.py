"""CPU/std-library evidence extraction: no training imports or data access."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics

RUN=Path(__file__).resolve().parent
REPO=RUN.parents[3]
FILES=set()

def read(name):
    FILES.add(name)
    return (REPO/name).read_text(encoding='utf-8-sig')

def save(name,value):
    with (RUN/name).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2);handle.write('\n')

def layers(name,cls):
    tree=ast.parse(read(name)); owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    result=[]
    for node in ast.walk(owner):
        if not isinstance(node,ast.Call) or ast.unparse(node.func)!='nn.Conv1d': continue
        ci,co=map(ast.literal_eval,node.args[:2]);kw={k.arg:ast.literal_eval(k.value) for k in node.keywords}
        kernel=kw['kernel_size'];bias=kw.get('bias',True)
        result.append(dict(line=node.lineno,input=ci,output=co,kernel=kernel,bias=bias,
                           parameters=ci*co*kernel+(co if bias else 0)))
    return result

def main():
    records=[]; results={}; methods=['pi_recipient_only','pair_stratified_rankmatch','pair_stratified_rankmatch_shuffled']
    for setting,template in [('CE','H2T/outputs/pi_h2t_ratio_only_ce_seed{seed}'),
                             ('PIF','H2T/outputs/pi_h2t_ratio_only_pif_stage1seed123_seed{seed}')]:
        series={k:[] for k in methods}; paired={'B-A':[],'B-C':[]}
        for seed in [42,43,44]:
            base=template.format(seed=seed)
            rows=list(csv.DictReader(read(base+'/summary.csv').splitlines()))
            exp=json.loads(read(base+'/experiment.json'))
            last={row['method']:row for row in rows if row['selection']=='last'}
            for method in methods:
                row=last[method]; value=100*float(row['overall']);series[method].append(value)
                records.append({'setting':setting,'seed':seed,'method':method,'last_overall_percent':value,
                                'donor_mass':float(row['donor_mass_mean']),
                                'recipient_mass':float(row['pi_recipient_mass_mean']),
                                'multiset_error':float(row['multiset_match_max_abs_error']),
                                'source':base+'/summary.csv'})
            for key,other in [('B-A',methods[0]),('B-C',methods[2])]:
                paired[key].append(100*(float(last[methods[1]]['overall'])-float(last[other]['overall'])))
            checksum_fields=['recipient_stream_checksum','donor_stream_checksum','target_stream_checksum','donor_label_stream_checksum']
            assert all(len({last[m][key] for m in methods})==1 for key in checksum_fields)
            assert all(float(last[m]['multiset_match_max_abs_error'])==0 for m in methods)
            assert exp['seed']==seed and exp['epochs']==10 and exp['geometry_head']=='frozen_stage1_classifier'
        results[setting]={'methods':{k:{'values':v,'mean':statistics.mean(v),'sample_sd':statistics.stdev(v)} for k,v in series.items()},
                          'paired':{k:{'values':v,'mean':statistics.mean(v),'sample_sd':statistics.stdev(v)} for k,v in paired.items()},
                          'strict_gate_all_seeds_B_gt_A_and_C':all(x>0 for v in paired.values() for x in v),
                          'scope':'three Stage-II seeds conditional on one Stage-I checkpoint; descriptive statistics only'}
    endpoints={}
    for key,base in [('T1','pif_endpoint_seed42_T1_k0_bnfixed'),('T2','pif_endpoint_seed42_T2_k0_bnshift'),
                     ('T3','pi_h2t_ratio_only_pif_stage1seed123_seed42'),('T4','pif_endpoint_seed42_T4_kappa_matched')]:
        path='H2T/outputs/'+base
        row=next(row for row in csv.DictReader(read(path+'/summary.csv').splitlines()) if row['selection']=='last' and row['method']=='pi_recipient_only')
        endpoints[key]={k:float(row[k]) for k in ['overall','many','medium','few','donor_mass_mean']}
        endpoints[key]['experiment']=json.loads(read(path+'/experiment.json'))
    results['endpoints']=endpoints
    results['endpoint_pp']={key:100*(endpoints[a]['overall']-endpoints[b]['overall']) for key,a,b in [('T2-T1','T2','T1'),('T3-T2','T3','T2'),('T4-T3','T4','T3'),('T4-T2','T4','T2')]}
    results['boundary_rows']=list(csv.DictReader(read('H2T/outputs/center_boundary_r_pif_seed42/summary.csv').splitlines()))
    results['cosine_audit']=json.loads(read('H2T/outputs/pif_matched_cosine_ratio_audit_seed42.json'))

    historic=layers('H2T/models/resnet_cifar.py','ResNet_fe')
    review=layers('PI_H2T_review/H2T-main/models/resnet_cifar.py','PermutationInvariantFusion')
    assert [(x['input'],x['output'],x['bias']) for x in historic]==[(2,4,True),(4,2,True),(2,1,True)]
    assert [(x['input'],x['output'],x['bias']) for x in review]==[(2,1,False)]
    hist_count=sum(x['parameters'] for x in historic);review_count=sum(x['parameters'] for x in review)
    assert hist_count==25 and review_count==2
    native=read('H2T/train_pi_h2t_ratio_only_stage2.py')
    assert '(1.0 - cosine) / 2.0' in native and 'donor_mass = 1.0 - recipient_mass' in native
    assert 'pi_weight = classifier_state["fc.weight"].to(device).detach().clone()' in native
    endpoint_cases=[]
    for cosine in [-1,0,1]:
        r=(1-cosine)/2;lam=1-r
        assert r+lam==1
        endpoint_cases.append({'cosine':cosine,'recipient_r':r,'donor_lambda':lam})
    checks={'historical_PIF_layers':historic,'historical_PIF_parameter_count':hist_count,
            'review_PIF_layers':review,'review_PIF_parameter_count':review_count,
            'ratio_boundary_arithmetic':endpoint_cases,
            'paper_ratio_direction':'Eq7 r=recipient; Eq8 + footnote2 implies r=(1-cos)/2',
            'affine_chain_identity':'A=W3*W2*W1; c=W3*W2*b1+W3*b2+b3; output=A0*F+A1*(F-mean)+c',
            'conclusion':'parameterization differs; absence of nonlinearities permits affine function-class reduction; not evidence that this caused accuracy collapse',
            'scope':'AST and scalar algebra only; no torch import, real tensors, checkpoint deserialization or model forward'}
    save('STATIC_PARITY_CHECKS.json',checks)
    save('HISTORICAL_RECALC.json',results)
    with (RUN/'HISTORICAL_CONTRASTS.csv').open('x',encoding='utf-8-sig',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

    for name in ['H2T_repro_audit/component_audit/results/static_20261026_164947/COMPONENT_AUDIT_REPORT.md',
                 'PI_H2T_review/H2T-main/PI_H2T_REVIEW_ZH.md','PI_H2T_review/H2T-main/review/upstream_provenance.json',
                 'PI_H2T_review/H2T-main/review/pi_h2t_vs_official_main.diff','PI_H2T_review/H2T-main/pi_h2t.py',
                 'PI_H2T_review/H2T-main/train_pi_h2t.py','H2T/train_boundary_h2t_stage2.py','H2T/train_center_boundary_r_stage2.py',
                 'H2T/audit_pif_cosine_ratio.py','H2T/datasets/cifar100.py','H2T/train_stage1.py',
                 'H2T/config/cifar100/cifar100_imb001_stage1_ce_mixup_pif.yaml',
                 'H2T/saved/cifar100_imb001_stage1_ce_mixup_pif_202608082320/codes/models/resnet_cifar.py',
                 'H2T/PI_H2T_PIF_CAUSAL_DIAGNOSTIC_REPORT.md','H2T/PI_H2T_RATIO_ONLY_PIF_FULL_REPORT.md',
                 'H2T/PI_H2T_RATIO_ONLY_CE_MULTISEED_REPORT.md','H2T/PI_H2T_CENTER_VS_BOUNDARY_R_SEED42_REPORT.md']:
        read(name)
    manifest=[dict(path=name,bytes=(REPO/name).stat().st_size,lines=len(read(name).splitlines()),
                   sha256=hashlib.sha256((REPO/name).read_bytes()).hexdigest()) for name in sorted(FILES)]
    save('SOURCE_REVIEW_MANIFEST.json',manifest)
    checkpoints=[]
    for suffix in ['cifar100_imb001_stage1_ce_mixup_pif_202608082320','cifar100_imb001_stage1_ce_mixup_pif_off_matched_202608092042']:
        path=REPO/'H2T/saved'/suffix/'ckps/model_best.pth.tar'
        checkpoints.append({'candidate_current_path':str(path.relative_to(REPO)),'exists':path.exists(),
                            'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'qualification':'Raw hash only. Historical absolute path used 新; byte identity to that original artifact unverified without prior hash. Not deserialized.'})
    save('CHECKPOINT_FILE_METADATA.json',checkpoints)
    print(json.dumps({'source_files':len(manifest),'historical_rows':len(records),'PIF_parameters':[hist_count,review_count],
                      'CE_gate':results['CE']['strict_gate_all_seeds_B_gt_A_and_C'],
                      'PIF_gate':results['PIF']['strict_gate_all_seeds_B_gt_A_and_C'],
                      'endpoint_pp':results['endpoint_pp']},ensure_ascii=False))

if __name__=='__main__':main()
