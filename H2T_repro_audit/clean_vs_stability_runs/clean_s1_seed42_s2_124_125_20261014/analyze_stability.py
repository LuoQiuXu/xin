"""Read-only analysis of the four authorized completed runs; no training/evaluation entry."""
from common import *
from stability_models import outdir,plan_dir,check_sources,PAIR_KEYS
from reused_vs_math import top_indices,selected_bool

METRICS = ['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro']


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def identical(a,b):
    return a.detach().cpu().contiguous().numpy().tobytes() == b.detach().cpu().contiguous().numpy().tobytes()


def main():
    install_test_guard('stability_analysis')
    assert (ROOT/'STOP_COMPLETED.json').is_file(), 'Complete exactly the authorized four runs first'
    assert not (ROOT/'results_summary.json').exists(), 'Refuse replacing a completed analysis'
    check_sources()
    labels = np.load(ROOT/'validation_labels.npy',allow_pickle=False)
    train_labels = np.load(ROOT/'train_labels.npy',allow_pickle=False)
    train_counts = np.bincount(train_labels,minlength=100)
    val_counts = np.bincount(labels,minlength=100)
    assert len(labels) == 1980 and val_counts[0] == 0 and (val_counts[1:] == 20).all()
    import pickle
    with (DATA/'cifar-100-python/meta').open('rb') as f:
        names = pickle.load(f,encoding='latin1')['fine_label_names']
    s1 = torch.load(ROOT/'locked/clean_stage1_best.pth',map_location='cpu',weights_only=False)
    initial = s1['state_dict_model']
    initial_fc = s1['state_dict_classifier']
    bn_prefix = [k[:-len('running_mean')] for k in initial if k.endswith('running_mean')]
    bn_buffers = [p+n for p in bn_prefix for n in ['running_mean','running_var','num_batches_tracked']]
    bn_affine = [p+n for p in bn_prefix for n in ['weight','bias']]
    conv = [k for k,v in initial.items() if v.ndim == 4]
    parameter_keys = [k for k in initial if k not in bn_buffers]
    assert len(bn_prefix) == 31 and len(bn_buffers) == 93
    results,all_metrics,delta_rows,classes,timings,checkpoints,composition = {},[],[],[],[],[],[]
    changed_images,tail_summary,diagnostics,parameter_checks,epoch_pairs = [],{},{},{},[]
    for seed in SEEDS:
        pair = read(plan_dir(seed)/'pairing_hash.json')
        assert pair['batches_each'] == 850
        result = {}
        predictions = {}
        for arm in ARMS:
            path = outdir(seed,arm)
            status = read(path/'status.json')
            assert status['state'] == 'completed' and len(status['epochs']) == 10
            assert status['finite_loss_classifier_gradient_steps'] == 850 and not status['TEST_used']
            assert status['fresh_optimizer']
            assert status['initial_state_hashes']['backbone'] == status['final_state_hashes']['backbone']
            assert status['initial_state_hashes']['classifier'] != status['final_state_hashes']['classifier']
            for ep in status['epochs']:
                assert ep['backbone_BN_bitwise_unchanged'] and ep['steps'] == 85
                assert math.isfinite(ep['train_loss'])
                all_metrics.append({'seed':seed,'method':arm,'epoch':ep['epoch'],'train_loss':ep['train_loss'],
                    'lr':ep['lr'],**{k:ep['validation'][k] for k in METRICS},
                    'training_seconds':ep['training_seconds'],'peak_allocated_mib':ep['peak_allocated_mib'],
                    'peak_reserved_mib':ep['peak_reserved_mib']})
            final = status['epochs'][-1]['validation']
            data = np.load(path/'validation_epoch10.npz',allow_pickle=False)
            assert np.array_equal(labels,data['labels']) and data['logits'].shape == (1980,100)
            assert np.array_equal(data['logits'].argmax(1),data['predictions'])
            recomputed = metrics(data['predictions'],labels)
            assert all(recomputed[k] == final[k] for k in METRICS)
            predictions[arm] = data['predictions'].copy()
            reloads = read(path/'checkpoint_validation.json')
            ck = {}
            for kind in ['best','final']:
                cp = path/f'checkpoints/model_{kind}.pth.tar'
                q = torch.load(cp,map_location='cpu',weights_only=False)
                assert q['Clean_StageI_sha256'] == S1_SHA and q['VS_statistics_sha256'] == STAT_SHA
                assert not q['TEST_used'] and q['seed'] == seed and q['arm'] == arm
                assert set(q['state_dict_model']) == set(initial)
                assert all(identical(initial[k],v) for k,v in q['state_dict_model'].items())
                assert all(identical(initial[k],q['state_dict_model'][k]) for k in conv+bn_affine+bn_buffers)
                fc_change = {k:float((v-initial_fc[k]).abs().max()) for k,v in q['state_dict_classifier'].items()}
                assert all(v > 0 for v in fc_change.values())
                assert reloads[kind]['validation_reload_exact'] and reloads[kind]['backbone_BN_saved_bitwise_exact']
                assert reloads[kind]['checkpoint_sha256'] == sha(cp)
                ck[kind] = {'epoch':q['epoch'],'sha256':sha(cp),'path':str(cp),
                    'all_backbone_state_entries_bitwise_unchanged':len(initial),
                    'backbone_parameter_tensors_bitwise_unchanged':len(parameter_keys),
                    'conv_parameter_tensors_bitwise_unchanged':len(conv),
                    'BN_affine_parameter_tensors_bitwise_unchanged':len(bn_affine),
                    'BN_buffer_tensors_bitwise_unchanged':len(bn_buffers),
                    'num_batches_tracked_values':sorted({int(initial[p+'num_batches_tracked']) for p in bn_prefix}),
                    'classifier_max_abs_changes':fc_change,'validation_reload_exact':True}
                checkpoints.append({'seed':seed,'method':arm,'kind':kind,'epoch':q['epoch'],'sha256':sha(cp),'path':str(cp)})
            parameter_checks[f'{seed}/{arm}'] = ck
            result[arm] = {'fixed_epoch10':final,'best_epoch':status['best_epoch'],'best_validation':status['best_validation'],
                'checkpoints':ck,'initial_state_hashes':status['initial_state_hashes']}
            timing = {'seed':seed,'method':arm,'training_seconds':status['total_training_seconds'],
                'run_seconds':status['total_run_seconds'],
                'peak_allocated_mib':max(e['peak_allocated_mib'] for e in status['epochs']),
                'peak_reserved_mib':max(e['peak_reserved_mib'] for e in status['epochs'])}
            for k in ['augmentation_seconds','feature_seconds','scoring_seconds','fusion_seconds','update_seconds']:
                timing[k] = sum(e['timing_components'][k] for e in status['epochs'])
            timings.append(timing)
            comp = np.load(path/'sampling_composition.npz',allow_pickle=False)
            assert all(comp[k].shape == (10,N) for k in comp.files)
            composition.append({'seed':seed,'method':arm,**{k:float(comp[k].mean()) for k in comp.files},
                'equal_Uniform_mask_rate':float((comp['common_with_uniform'] == 16).mean()),
                'fallback_draws':int((comp['fallback'] > 0).sum()),'draws':10*N})
        assert result[ARMS[0]]['initial_state_hashes'] == result[ARMS[1]]['initial_state_hashes']
        u,l = [result[a]['fixed_epoch10'] for a in ARMS]
        deltas = {k:l[k]-u[k] for k in METRICS}
        delta_rows.append({'seed':seed,**{f'delta_{k}_pp':v for k,v in deltas.items()}})
        pu,pl = [predictions[a] for a in ARMS]
        gained,lost = (pu != labels)&(pl == labels),(pu == labels)&(pl != labels)
        predicted_different = pu != pl
        gains_by_group = {}
        for key,active in [('Overall',list(range(1,100))),('Tail',list(range(71,100))),('Few',list(range(70,100)))]:
            mask = np.isin(labels,active)
            gains_by_group[key] = {'samples':int(mask.sum()),'gained':int((gained&mask).sum()),
                'lost':int((lost&mask).sum()),'net_correct':int((gained&mask).sum()-(lost&mask).sum())}
            assert abs(gains_by_group[key]['net_correct']*100/mask.sum()-deltas[key]) < 1e-12
        result['paired_change_counts'] = gains_by_group
        for i in np.flatnonzero(predicted_different):
            changed_images.append({'seed':seed,'validation_local_index':int(i),'label':int(labels[i]),
                'Uniform_prediction':int(pu[i]),'LowVS_prediction':int(pl[i]),
                'Uniform_correct':int(pu[i] == labels[i]),'LowVS_correct':int(pl[i] == labels[i])})
        for c in range(100):
            active = labels == c
            cu,cl = int((pu[active] == c).sum()),int((pl[active] == c).sum())
            classes.append({'seed':seed,'class_id':c,'class_name':names[c],
                'id_group':'Head' if c <= 35 else ('Medium' if c <= 70 else 'Tail'),
                'frequency_group':'Many' if train_counts[c] > 100 else ('Few' if train_counts[c] < 20 else 'Medium-n'),
                'train_count':int(train_counts[c]),'validation_count':int(active.sum()),
                'Uniform_correct':cu if active.any() else None,'LowVS_correct':cl if active.any() else None,
                'Uniform_accuracy':cu*100/active.sum() if active.any() else None,
                'LowVS_accuracy':cl*100/active.sum() if active.any() else None,
                'delta_pp':(cl-cu)*100/active.sum() if active.any() else None,
                'net_correct':cl-cu if active.any() else None})
        tail = [r for r in classes if r['seed'] == seed and r['id_group'] == 'Tail']
        tail_summary[str(seed)] = {name:[r['class_id'] for r in tail if check(r['net_correct'])]
            for name,check in [('improved',lambda d:d>0),('declined',lambda d:d<0),('unchanged',lambda d:d==0)]}
        statuses = [read(outdir(seed,a)/'status.json') for a in ARMS]
        for epoch in range(10):
            eu,el = [s['epochs'][epoch]['validation'] for s in statuses]
            epoch_pairs.append({'seed':seed,'epoch':epoch+1,**{f'delta_{k}_pp':el[k]-eu[k] for k in METRICS}})
        # Verify saved masks from saved score/noise arrays, never recompute feature statistics.
        plans = np.load(plan_dir(seed)/'sampling_plan.npz',allow_pickle=False)
        gumbel = np.load(plan_dir(seed)/'gumbel_noise_float64.npy',mmap_mode='r',allow_pickle=False)
        score = np.load(plan_dir(seed)/'online_scores_float64.npy',mmap_mode='r',allow_pickle=False)
        masks = [np.load(outdir(seed,a)/'actual_masks.npz',allow_pickle=False)['masks'] for a in ARMS]
        for a in masks:
            assert a.shape == (10,N,16) and np.all(a < 64) and np.all(np.diff(np.sort(a,axis=-1),axis=-1) > 0)
        expected_fallbacks = sum(int((train_counts[train_labels[plans['donors'][e]]] < 20).sum()) for e in range(10))
        same = total_common = fallback_count = 0
        frequency = np.zeros((2,64),dtype=np.int64)
        for e in range(10):
            assert np.array_equal(np.sort(plans['donors'][e]),np.arange(N)), 'IS each sample exactly once per epoch'
            for start in range(0,N,BATCH):
                stop = min(start+BATCH,N)
                s,noise = np.asarray(score[e,start:stop]),np.asarray(gumbel[e,start:stop])
                assert np.isfinite(s).all() and ((s >= 0)&(s <= 1)).all()
                h = train_labels[plans['donors'][e,start:stop]]
                fallback = (train_counts[h] < 20)|(np.ptp(s,axis=1) <= 1e-12)
                pool = selected_bool(top_indices(s,48,False));pool[fallback] = True
                expected = [np.argsort(-noise,axis=1,kind='stable')[:,:16].astype(np.uint8),
                    np.argsort(-np.where(pool,noise,-np.inf),axis=1,kind='stable')[:,:16].astype(np.uint8)]
                for arm_index in range(2):
                    assert np.array_equal(masks[arm_index][e,start:stop],expected[arm_index]), 'Saved mask differs from fixed rule'
                    frequency[arm_index] += np.bincount(expected[arm_index].ravel(),minlength=64)
                common = (selected_bool(expected[0])&selected_bool(expected[1])).sum(1)
                same += int((common == 16).sum());total_common += int(common.sum());fallback_count += int(fallback.sum())
                assert np.array_equal(expected[0][fallback],expected[1][fallback])
        batch_rows = [rows(outdir(seed,a)/'batches.csv') for a in ARMS]
        assert all(len(r) == 850 for r in batch_rows)
        assert all(x[k] == y[k] for x,y in zip(*batch_rows) for k in PAIR_KEYS)
        diagnostics[str(seed)] = {'mask_rule_reconstruction_passed':True,'all_850_batch_pair_fields_equal':True,
            'masks_each':10*N,'channels_each':16,'mean_mask_overlap':total_common/(10*N),
            'identical_mask_count':same,'identical_mask_fraction':same/(10*N),
            'fallback_count':fallback_count,'small_class_fallback_expected':expected_fallbacks,
            'each_IS_epoch_is_permutation':True,'per_channel_frequency':{a:frequency[i].tolist() for i,a in enumerate(ARMS)}}
        results[str(seed)] = result
    deltas_summary = {}
    for metric in METRICS:
        x = np.array([r[f'delta_{metric}_pp'] for r in delta_rows])
        deltas_summary[metric] = {'new_seeds':SEEDS,'n':2,'mean_pp':float(x.mean()),
            'sample_sd_pp_ddof1':float(x.std(ddof=1)),'positive_seeds':int((x > 0).sum()),'values_pp':x.tolist()}
    pilot = {}
    for a in ARMS:
        status = read(ORIGIN/'arms'/a/'status.json')
        pilot[a] = {k:status['epochs'][-1]['validation'][k] for k in METRICS}
    pilot_delta = {k:pilot[ARMS[1]][k]-pilot[ARMS[0]][k] for k in METRICS}
    all_three = {k:{'mean_pp':float(np.mean([pilot_delta[k]]+deltas_summary[k]['values_pp'])),
        'sample_sd_pp_ddof1':float(np.std([pilot_delta[k]]+deltas_summary[k]['values_pp'],ddof=1)),
        'includes_already_observed_pilot':True} for k in METRICS}
    tail_consensus = {'improved_both':sorted(set(tail_summary['124']['improved'])&set(tail_summary['125']['improved'])),
        'declined_both':sorted(set(tail_summary['124']['declined'])&set(tail_summary['125']['declined'])),
        'opposite_signs':sorted((set(tail_summary['124']['improved'])&set(tail_summary['125']['declined']))|
            (set(tail_summary['124']['declined'])&set(tail_summary['125']['improved'])))}
    for filename,values in [('epoch_validation_metrics.csv',all_metrics),('paired_epoch10_deltas.csv',delta_rows),
        ('per_class_delta.csv',classes),('timing_and_memory.csv',timings),('checkpoint_manifest.csv',checkpoints),
        ('selected_variation_shift.csv',composition),('paired_epoch_deltas.csv',epoch_pairs)]:
        csv_write(ROOT/filename,values)
    if changed_images:csv_write(ROOT/'paired_changed_images_epoch10.csv',changed_images)
    dump(ROOT/'sampling_diagnostics.json',diagnostics)
    dump(ROOT/'parameter_and_BN_verification.json',parameter_checks)
    hist = check_history();guard_pass()
    summary = {'analyzed_at':now(),'results':results,'fixed_epoch10_deltas':delta_rows,'new_seed_summary':deltas_summary,
        'tail_class_changes':tail_summary,'tail_consensus':tail_consensus,'pilot123':pilot,'pilot123_deltas':pilot_delta,
        'three_seed_descriptive_only':all_three,'timing_and_memory':timings,'composition':composition,
        'historical_files_unchanged':hist['protected_files'],'TEST_used':False,'no_new_training_in_analysis':True}
    dump(ROOT/'results_summary.json',summary)
    for row in delta_rows:print(json.dumps(row,ensure_ascii=False),flush=True)
    print('STABILITY ANALYSIS PASS:all four completed runs,bitwise frozen BB/BN,850 paired batches per seed,exact saved mask rules',flush=True)


if __name__ == '__main__':
    main()
