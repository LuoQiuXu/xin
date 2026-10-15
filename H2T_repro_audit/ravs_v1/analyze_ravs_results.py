"""Analyze only this iteration's ten saved runs; no training or new evaluation."""
import json,csv
from pathlib import Path
import numpy as np
import torch
from run_safeguards import check_history

KEYS=['indices_sha256','images_sha256','features_sha256','scores_sha256','gumbel_sha256']
METRICS=['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro']
PAIRS=[('Uniform-PerSample','No-Fusion'),('Low-VS','No-Fusion'),('Frequency-Only','No-Fusion'),('RAVS-V1','No-Fusion'),
       ('Low-VS','Uniform-PerSample'),('Frequency-Only','Uniform-PerSample'),('RAVS-V1','Uniform-PerSample'),
       ('Frequency-Only','Low-VS'),('RAVS-V1','Low-VS'),('RAVS-V1','Frequency-Only')]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def decision(pairs):
    tol=1e-12
    vs_low=[p for p in pairs if p['comparison']=='RAVS-V1 - Low-VS']
    vs_freq=[p for p in pairs if p['comparison']=='RAVS-V1 - Frequency-Only']
    def mean(rows,key):return float(np.mean([p['delta_'+key+'_pp'] for p in rows]))
    g1={'both_seed_Overall_nonnegative':all(p['delta_Overall_pp']>=-tol for p in vs_low),
        'mean_Overall_at_least_0_15':mean(vs_low,'Overall')>=.15-tol,
        'mean_Tail_nonnegative':mean(vs_low,'Tail')>=-tol,
        'mean_Head_nonnegative':mean(vs_low,'Head')>=-tol}
    g2a={'both_seed_Overall_nonnegative':all(p['delta_Overall_pp']>=-tol for p in vs_freq),
         'mean_Overall_at_least_0_10':mean(vs_freq,'Overall')>=.10-tol}
    g2b={'both_seed_Tail_nonnegative':all(p['delta_Tail_pp']>=-tol for p in vs_freq),
         'mean_Overall_nonnegative':mean(vs_freq,'Overall')>=-tol,
         'mean_Tail_at_least_0_35':mean(vs_freq,'Tail')>=.35-tol}
    return {'gate1':{'passed':all(g1.values()),'conditions':g1,
                    'mean_delta_pp':{k:mean(vs_low,k) for k in METRICS}},
            'gate2':{'passed':all(g2a.values()) or all(g2b.values()),'Overall_route':g2a,'Tail_route':g2b,
                    'mean_delta_pp':{k:mean(vs_freq,k) for k in METRICS}},
            'go':all(g1.values()) and (all(g2a.values()) or all(g2b.values())),
            'threshold_source':'results/preregistered_decision_rules.json; fixed before formal training'}


def report(r):
    from common import metrics
    final_report=r.RESULTS/'RAVS_V1_SCIENTIFIC_REPORT.md'
    if final_report.exists():raise FileExistsError('Refuse replacing a delivered report')
    expected_order=[(s,a) for s in r.SEEDS for a in r.ARMS]
    registry=read(r.RESULTS/'run_registry.json')
    finished=[(x['seed'],x['arm']) for x in registry if x['action']=='train' and x['status']=='completed']
    assert finished==expected_order,'Expected exactly ten successful ordered runs'
    assert not [x for x in registry if x['action']=='train' and x['status']!='completed']
    history_files=check_history(r)
    pre=read(r.RESULTS/'preflight.json');assert pre['status']=='PASS'
    rows=[];epoch_rows=[];pair_rows=[];class_rows=[];pool_rows=[];checkpoint_rows=[];diagnostics=[]
    predictions={};values={};summaries={};matching={};tail={};all_matched=True
    labels=np.load(r.OLD/'validation_labels.npy',allow_pickle=False)
    assert labels.shape==(1980,) and np.array_equal(np.bincount(labels,minlength=100),np.r_[0,np.repeat(20,99)])
    s1=torch.load(r.OLD/'locked/clean_stage1_best.pth',map_location='cpu',weights_only=False)
    fc0=s1['state_dict_classifier']
    for seed in r.SEEDS:
        reference=r.csv_read(r.RESULTS/f'seed{seed}/No-Fusion/batch_signatures.csv')
        assert len(reference)==850
        for arm in r.ARMS:
            folder=r.RESULTS/f'seed{seed}'/arm
            summary=read(folder/'summary.json');config=read(folder/'config.json')
            assert summary['status']=='completed' and summary['updates']==850 and not summary['TEST_used']
            assert summary['initial_hashes']==pre['seeds'][str(seed)]['initial_state_hashes']
            assert summary['final_backbone_hash']==summary['initial_hashes']['backbone']
            assert summary['backbone_state_entries_bitwise_unchanged']==186 and summary['BN_counters']==[15640]
            assert summary['validation_reload_exact']
            assert all(x>0 for x in summary['classifier_max_abs_changes'].values())
            assert config['environment']==pre['environment']
            epochs=r.csv_read(folder/'epoch_metrics.csv');sig=r.csv_read(folder/'batch_signatures.csv')
            assert len(epochs)==10 and [int(x['epoch']) for x in epochs]==list(range(1,11)) and len(sig)==850
            assert sum(int(x['size']) for x in sig)==10*r.N
            equal={k:sum(x[k]==y[k] for x,y in zip(reference,sig)) for k in KEYS}
            assert equal==summary['pairing_counts_vs_No_Fusion']
            matching[f'{seed}/{arm}']=equal
            all_matched=all_matched and all(x==850 for x in equal.values())
            at10=epochs[-1];best=max(epochs,key=lambda x:float(x['Overall']))
            assert int(best['epoch'])==summary['best_epoch']
            row={'seed':seed,'arm':arm,**{k:float(at10[k]) for k in METRICS},
                'best_epoch':int(best['epoch']),'best_Overall':float(best['Overall']),
                **{f'matched_{k}':v for k,v in equal.items()}}
            rows.append(row);values[(seed,arm)]=row;summaries[f'{seed}/{arm}']=summary
            v=np.load(folder/'validation_epoch10.npz',allow_pickle=False)
            assert np.array_equal(v['labels'],labels) and v['logits'].shape==(1980,100)
            assert np.array_equal(v['predictions'],v['logits'].argmax(1))
            predictions[(seed,arm)]=v['predictions'].copy()
            recomputed=metrics(v['predictions'],labels)
            assert all(abs(recomputed[k]-row[k])<1e-12 for k in METRICS)
            for e in epochs:
                assert np.isfinite(float(e['train_loss']))
                epoch_rows.append({'seed':seed,'arm':arm,'epoch':int(e['epoch']),'train_loss':float(e['train_loss']),
                    'lr':float(e['lr']),**{k:float(e[k]) for k in METRICS},
                    'training_seconds':float(e['training_seconds']),'peak_allocated_mib':float(e['peak_allocated_mib']),
                    'peak_reserved_mib':float(e['peak_reserved_mib'])})
            gate_stats={}
            for k in ['hardness','omega','frequency_need','fallback']:
                x=[float(e['mean_'+k]) for e in epochs if e['mean_'+k]!='']
                gate_stats['mean_'+k]=float(np.mean(x)) if x else None
            pool={k:sum(int(e[f'K_count_{k}']) for e in epochs) for k in range(48,65)} if arm!='No-Fusion' else {k:0 for k in range(48,65)}
            if arm!='No-Fusion':assert sum(pool.values())==10*r.N
            gate_stats.update({'K_mean':sum(k*n for k,n in pool.items())/(10*r.N) if arm!='No-Fusion' else None,
                'K_min':min([k for k,n in pool.items() if n],default=None),'K_max':max([k for k,n in pool.items() if n],default=None),
                'same_Uniform_mask_fraction':float(np.mean([float(e['same_Uniform_mask_fraction']) for e in epochs])) if arm!='No-Fusion' else None})
            if arm=='Frequency-Only':assert gate_stats['mean_hardness']==.5
            for k,n in pool.items():pool_rows.append({'seed':seed,'arm':arm,'K':k,'sample_draws':n,'total_draws':10*r.N})
            diagnostics.append({'seed':seed,'arm':arm,**gate_stats,
                'training_seconds':sum(float(e['training_seconds']) for e in epochs),'elapsed_seconds':summary['elapsed_seconds'],
                'peak_allocated_mib':max(float(e['peak_allocated_mib']) for e in epochs),
                'peak_reserved_mib':max(float(e['peak_reserved_mib']) for e in epochs)})
            for kind in ['best','final']:
                cp=folder/f'checkpoints/{kind}.pth';q=torch.load(cp,map_location='cpu',weights_only=False)
                assert q['seed']==seed and q['arm']==arm and not q['TEST_used']
                assert q['StageI_sha256']==pre['checkpoint_sha256']
                assert r.sha(cp)==summary['checkpoint_sha256'][kind]
                assert all(float((v-fc0[k]).abs().max())>0 for k,v in q['classifier'].items())
                checkpoint_rows.append({'seed':seed,'arm':arm,'kind':kind,'epoch':q['epoch'],
                    'sha256':r.sha(cp),'path':str(cp),'StageI_sha256':q['StageI_sha256']})
        for arm,base in PAIRS:
            p,ref=predictions[(seed,arm)],predictions[(seed,base)]
            m,rm=values[(seed,arm)],values[(seed,base)]
            win=(ref!=labels)&(p==labels);lose=(ref==labels)&(p!=labels)
            pair={'seed':seed,'comparison':arm+' - '+base,
                **{f'delta_{k}_pp':m[k]-rm[k] for k in METRICS},
                'delta_best_Overall_pp':m['best_Overall']-rm['best_Overall'],
                'changed_predictions':int((p!=ref).sum())}
            effective=metrics(p,labels)['effective_classes']
            effective['Overall']=list(range(1,100))
            for key in ['Overall','Head','OfficialMedium','Tail','Many','Medium','Few']:
                group=np.isin(labels,effective[key]);wins=int((win&group).sum());losses=int((lose&group).sum())
                pair[key+'_wins']=wins;pair[key+'_losses']=losses;pair[key+'_net_correct']=wins-losses;pair[key+'_samples']=int(group.sum())
                assert abs((wins-losses)*100/group.sum()-pair['delta_'+key+'_pp'])<1e-12
            pair['best_order_reversal']=bool(pair['delta_Overall_pp']*pair['delta_best_Overall_pp']<0)
            pair['small_net_1_to_3_images']=bool(1<=abs(pair['Overall_net_correct'])<=3)
            pair_rows.append(pair)
            gain_classes=[];lost_classes=[]
            for c in range(100):
                active=labels==c;n=int(active.sum());right=int((p[active]==c).sum());old=int((ref[active]==c).sum())
                class_rows.append({'seed':seed,'comparison':pair['comparison'],'class_id':c,'validation_samples':n,
                    'method_correct':right if n else '', 'reference_correct':old if n else '',
                    'net_correct':right-old if n else '', 'delta_pp':100*(right-old)/n if n else ''})
                if c>=71 and right>old:gain_classes.append(c)
                if c>=71 and right<old:lost_classes.append(c)
            tail[f'{seed}/{pair["comparison"]}']={'improved_ids':gain_classes,'declined_ids':lost_classes}
    aggregate=[]
    for arm,base in PAIRS:
        comp=arm+' - '+base;pair=[p for p in pair_rows if p['comparison']==comp]
        for key in METRICS:
            x=np.asarray([p['delta_'+key+'_pp'] for p in pair])
            aggregate.append({'comparison':comp,'metric':key,'n_seeds':2,'mean_delta_pp':float(x.mean()),
                'sample_sd_pp_ddof1':float(x.std(ddof=1)),'seed124_delta_pp':float(x[0]),'seed125_delta_pp':float(x[1]),
                'opposite_signs':bool(x.min()< -1e-12 and x.max()>1e-12)})
    gates=decision(pair_rows)
    status='complete' if all_matched else 'COMPARABILITY_WARNING'
    if not all_matched:gates['go']=False;gates['interpretation']='Five-field input parity failed; no causal gate decision'
    for name,data in [('all_arms_epoch10.csv',rows),('paired_epoch10.csv',pair_rows),('epoch_metrics_all.csv',epoch_rows),
        ('paired_per_class.csv',class_rows),('paired_seed_summary.csv',aggregate),('gate_diagnostics.csv',diagnostics),
        ('candidate_pool_histogram.csv',pool_rows),('checkpoint_manifest.csv',checkpoint_rows)]:r.csv_write(r.RESULTS/name,data)
    payload={'status':status,'fully_matched':all_matched,'rows':rows,'paired':pair_rows,'aggregate':aggregate,'gate_decision':gates,
        'five_field_matching':matching,'gate_diagnostics':diagnostics,'tail_class_changes':tail,
        'checkpoints':checkpoint_rows,'source_preflight':pre,'historical_files_unchanged':history_files,
        'new_classifier_runs':10,'new_optimizer_updates':8500,'old_accuracies_reused':False,'TEST_used':False,
        'preregistered_rules':read(r.RESULTS/'preregistered_decision_rules.json'),
        'execution_registry':registry,'model_summaries':summaries}
    r.json_write(r.RESULTS/'scientific_summary.json',payload)
    from write_ravs_report import write
    write(r,payload)
    r.check_test_guard();check_history(r)
    r.json_write(r.RESULTS/'STOP_COMPLETED.json',{'status':status,'new_runs':10,'seeds':list(r.SEEDS),'arms':list(r.ARMS),
        'epochs_each':10,'updates_total':8500,'fully_matched':all_matched,'Go':gates['go'],'TEST_used':False,
        'StageI_retraining':0,'no_further_training':True,'historical_files_unchanged':history_files})
    print('REPORT: '+str(final_report),flush=True)
    print('MATCHED FIVE-FIELD PARITY: '+str(all_matched),flush=True)
    print('PREREGISTERED GO/NO-GO: '+json.dumps(gates,ensure_ascii=False),flush=True)
