"""Saved-output analysis; no new model training or new data/evaluation."""
from pathlib import Path
import csv,json,math
import numpy as np
import torch
from preflight_checks import dump,history_manifest

METRICS=['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro']
ARMS=['Uniform-BatchShared','Uniform-PerSample','Low-VS']
COMPARISONS=[('Uniform-PerSample','Uniform-BatchShared'),('Low-VS','Uniform-BatchShared'),('Low-VS','Uniform-PerSample')]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def records(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def report(r):
    target=r.RESULTS/'MASK_GRANULARITY_RESULTS.md'
    if target.exists():raise FileExistsError('Refuse replacing a delivered scientific report')
    baseline=read(r.RESULTS/'historical_readonly_baseline.json')
    assert history_manifest(r.OLD,r.sha)==baseline
    r.check_sources()
    rows=[];pair_rows=[];class_rows=[];mask_rows=[];channels=[];composition=[];timing=[];checkpoints=[];bn=[]
    values={};predictions={};summaries={};pairing={};tail={};masks={}
    labels=np.load(r.OLD/'validation_labels.npy',allow_pickle=False)
    assert len(labels)==1980 and np.array_equal(np.bincount(labels,minlength=100),np.r_[0,np.repeat(20,99)])
    s1=torch.load(r.OLD/'locked/clean_stage1_best.pth',map_location='cpu',weights_only=False)
    frozen=s1['state_dict_model'];fc0=s1['state_dict_classifier']
    for seed in r.SEEDS:
        for arm in ARMS:
            folder=r.RESULTS/f'shared_seed{seed}' if arm==r.ARM else r.outdir(seed,arm)
            epoch_rows=records(folder/'epoch_validation_metrics.csv')
            assert [int(x['epoch']) for x in epoch_rows]==list(range(1,11))
            at10=epoch_rows[-1];best=max(epoch_rows,key=lambda x:float(x['Overall']))
            row={'seed':seed,'arm':arm,**{k:float(at10[k]) for k in METRICS},'best_epoch':int(best['epoch']),'best_Overall':float(best['Overall']),
                'source':'new_shared_training' if arm==r.ARM else 'sealed_historical_control'}
            rows.append(row);values[(seed,arm)]=row
            bundle=np.load(folder/'validation_epoch10.npz',allow_pickle=False)
            assert np.array_equal(labels,bundle['labels']) and bundle['logits'].shape==(1980,100)
            assert np.array_equal(bundle['logits'].argmax(1),bundle['predictions'])
            predictions[(seed,arm)]=bundle['predictions'].copy()
            m=r.metrics(bundle['predictions'],labels)
            assert all(abs(m[k]-row[k])<1e-12 for k in METRICS)
            actual=np.load(folder/'actual_masks.npz',allow_pickle=False)['masks']
            assert actual.shape==(10,r.N,16) and np.all(actual<64) and np.all(np.diff(np.sort(actual,axis=-1),axis=-1)>0)
            masks[(seed,arm)]=actual
            comp=np.load(folder/'sampling_composition.npz',allow_pickle=False)
            composition.append({'seed':seed,'arm':arm,**{k:float(comp[k].mean()) if k in comp.files else '' for k in ['V','B','ratio','score','GAP_L2','feature_map_L2']},
                'statistics_note':'new feature inference, fixed historical class statistics' if arm==r.ARM else 'sealed historical composition'})
            if arm==r.ARM:
                summary=read(folder/'summary.json');summaries[str(seed)]=summary
                assert summary['updates']==850 and summary['epochs']==10 and summary['stage1_frozen']
                assert summary['all_shared_masks_equal_sealed_Uniform_first_masks'] and not summary['TEST_used']
                b=records(folder/'batches.csv');ref=r.control_rows(seed)
                assert len(b)==len(ref)==850
                eq={k:sum(x[k]==y[k] for x,y in zip(b,ref)) for k in r.PAIR_KEYS}
                assert eq==summary['paired_control_fields_verified'];pairing[str(seed)]=eq
                for kind in ['best','final']:
                    cp=folder/f'checkpoints/model_{kind}.pth.tar';q=torch.load(cp,map_location='cpu',weights_only=False)
                    assert not q['TEST_used'] and q['seed']==seed
                    assert all(v.numpy().tobytes()==frozen[k].numpy().tobytes() for k,v in q['state_dict_model'].items())
                    fc_delta={k:float((v-fc0[k]).abs().max()) for k,v in q['state_dict_classifier'].items()}
                    assert all(v>0 for v in fc_delta.values())
                    checkpoints.append({'seed':seed,'arm':arm,'kind':kind,'epoch':q['epoch'],'path':str(cp),'sha256':r.sha(cp)})
                    bn.append({'seed':seed,'checkpoint':kind,'backbone_state_entries_unchanged':len(frozen),
                        'BN_running_mean_unchanged':sum(k.endswith('running_mean') for k in frozen),
                        'BN_running_var_unchanged':sum(k.endswith('running_var') for k in frozen),
                        'BN_num_batches_tracked_unchanged':sum(k.endswith('num_batches_tracked') for k in frozen),
                        'counter_values':sorted({int(v) for k,v in frozen.items() if k.endswith('num_batches_tracked')}),
                        'classifier_max_abs_changes':fc_delta,'validation_reload_exact':summary['checkpoints'][kind]['validation_reload_exact']})
                timing.append({'seed':seed,'arm':arm,'training_seconds':sum(float(x['training_seconds']) for x in epoch_rows),
                    'run_seconds':summary['run_seconds'],'peak_allocated_mib':max(float(x['peak_allocated_mib']) for x in epoch_rows),
                    'peak_reserved_mib':max(float(x['peak_reserved_mib']) for x in epoch_rows)})
        u=masks[(seed,'Uniform-PerSample')]
        for epoch in range(10):
            for start in range(0,r.N,r.BATCH):
                stop=min(start+r.BATCH,r.N)
                assert np.array_equal(masks[(seed,r.ARM)][epoch,start:stop],np.broadcast_to(u[epoch,start],(stop-start,16)))
        for arm,base in COMPARISONS:
            a,b=predictions[(seed,base)],predictions[(seed,arm)]
            av,bv=values[(seed,base)],values[(seed,arm)]
            gain=(a!=labels)&(b==labels);loss=(a==labels)&(b!=labels)
            pair={'seed':seed,'comparison':arm+' - '+base,'gain_wrong_to_right':int(gain.sum()),
                'loss_right_to_wrong':int(loss.sum()),'net_correct':int(gain.sum()-loss.sum()),'changed_prediction':int((a!=b).sum()),
                **{f'delta_{k}_pp':bv[k]-av[k] for k in METRICS},'delta_best_Overall_pp':bv['best_Overall']-av['best_Overall']}
            for key,ids in [('Overall',range(1,100)),('Tail',range(71,100)),('Few',range(70,100))]:
                active=np.isin(labels,list(ids));g,l=int((gain&active).sum()),int((loss&active).sum())
                pair[key+'_gain']=g;pair[key+'_loss']=l;pair[key+'_net_correct']=g-l;pair[key+'_samples']=int(active.sum())
                assert abs((g-l)*100/active.sum()-pair[f'delta_{key}_pp'])<1e-12
            pair['best_reversal']=bool(pair['delta_Overall_pp']*pair['delta_best_Overall_pp']<0)
            pair_rows.append(pair)
            class_delta=[]
            for c in range(100):
                active=labels==c;n=int(active.sum());ca=int((a[active]==c).sum());cb=int((b[active]==c).sum())
                rowc={'seed':seed,'comparison':pair['comparison'],'class_id':c,'validation_samples':n,
                    'id_group':'Head' if c<=35 else ('OfficialMedium' if c<=70 else 'Tail'),
                    'reference_correct':ca if n else '', 'method_correct':cb if n else '',
                    'net_correct':cb-ca if n else '', 'delta_pp':100*(cb-ca)/n if n else ''}
                class_rows.append(rowc)
                if c>=71:class_delta.append((c,cb-ca))
            tail[str(seed)+'/'+pair['comparison']]={'improved':[c for c,d in class_delta if d>0],
                'declined':[c for c,d in class_delta if d<0],'unchanged':[c for c,d in class_delta if d==0]}
        for arm in ARMS:
            mask=masks[(seed,arm)];frequency=np.bincount(mask.ravel(),minlength=64)
            overlap_sum=same_u=pair_overlap_sum=identical_pair_sum=pair_count=0
            for epoch in range(10):
                for start in range(0,r.N,r.BATCH):
                    stop=min(start+r.BATCH,r.N);m=mask[epoch,start:stop];b=len(m)
                    counts=np.bincount(m.ravel(),minlength=64);pair_count+=b*(b-1)
                    pair_overlap_sum+=int((counts*(counts-1)).sum())
                    _,n_same=np.unique(np.sort(m,axis=-1),axis=0,return_counts=True)
                    identical_pair_sum+=int((n_same*(n_same-1)).sum())
                    overlap=(np.eye(64,dtype=bool)[m].any(1)&np.eye(64,dtype=bool)[u[epoch,start:stop]].any(1)).sum(1)
                    overlap_sum+=int(overlap.sum());same_u+=int((overlap==16).sum())
            mask_rows.append({'seed':seed,'arm':arm,'draws':10*r.N,'batches':850,
                'mean_overlap_with_Uniform':overlap_sum/(10*r.N),'identical_to_Uniform_fraction':same_u/(10*r.N),
                'within_batch_pair_mean_shared_channels':pair_overlap_sum/pair_count,
                'within_batch_pair_identical_mask_fraction':identical_pair_sum/pair_count,
                'global_marginal_independence_overlap':float(np.square(frequency/(10*r.N)).sum()),
                'unique_combinations':len(np.unique(np.sort(mask.reshape(-1,16),axis=-1),axis=0)),
                'min_channel_selected_count':int(frequency.min()),'max_channel_selected_count':int(frequency.max())})
            channels += [{'seed':seed,'arm':arm,'channel':i,'selected_count':int(n),'selection_fraction':float(n/(10*r.N))} for i,n in enumerate(frequency)]
    summary_rows=[]
    for arm,base in COMPARISONS:
        comp=arm+' - '+base;matches=[p for p in pair_rows if p['comparison']==comp]
        for key in METRICS:
            x=np.asarray([m[f'delta_{key}_pp'] for m in matches])
            nets=[m.get(key+'_net_correct') for m in matches]
            summary_rows.append({'comparison':comp,'metric':key,'seeds':'124,125','n_seeds':2,'mean_delta_pp':float(x.mean()),
                'sample_sd_pp_ddof1':float(x.std(ddof=1)),'positive_seeds':int((x>1e-12).sum()),'negative_seeds':int((x< -1e-12).sum()),
                'opposite_signs':bool(x.min()< -1e-12 and x.max()>1e-12)})
    for name,data in [('all_three_epoch10.csv',rows),('paired_differences.csv',pair_rows),('per_class_paired_differences.csv',class_rows),
        ('mask_diagnostics.csv',mask_rows),('channel_selection_counts.csv',channels),('selected_variation_shift.csv',composition),
        ('timing_and_memory.csv',timing),('checkpoint_manifest.csv',checkpoints),('paired_seed_summary.csv',summary_rows)]:r.csv_write(r.RESULTS/name,data)
    dump(r.RESULTS/'pairing_hash.json',pairing);dump(r.RESULTS/'parameter_BN_verification.json',bn)
    dump(r.RESULTS/'tail_class_changes.json',tail)
    data={'rows':rows,'paired':pair_rows,'seed_summary':summary_rows,'mask_diagnostics':mask_rows,'composition':composition,
        'timing':timing,'checkpoints':checkpoints,'tail_classes':tail,'pairing':pairing,'summaries':summaries,
        'pilot123':read(r.OLD/'results_summary.json')['pilot123'],
        'historical_files_unchanged':len(baseline),'TEST_used':False,'new_training_runs':2,'new_updates':1700,
        'historical_exact_parity_waived_by_user':True}
    dump(r.RESULTS/'scientific_summary.json',data)
    from write_scientific_report import write
    write(r,data)
    r.guard_check();assert history_manifest(r.OLD,r.sha)==baseline
    dump(r.RESULTS/'STOP_COMPLETED.json',{'status':'completed_with_documented_historical_pairing_limit',
        'seeds':[124,125],'new_arm':r.ARM,'new_runs':2,'epochs_each':10,'updates_total':1700,
        'StageI_retraining':0,'old_control_retraining':0,'TEST_used':False,'historical_files_unchanged':len(baseline),
        'strict_original_13_field_parity':False,'user_authorized_continuation':True})
    print('REPORT COMPLETE: '+str(target),flush=True)
    for row in pair_rows:print(json.dumps(row,ensure_ascii=False),flush=True)
