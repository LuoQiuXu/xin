"""Only Uniform/High-VS/Low-VS, exact online shared controls, independent classifiers."""
import argparse
import shutil
from common import *
from build_clean_statistics import frozen_models
from clean_online_policy import online_masks, selected_statistics, stats

PAIR_KEYS = ['epoch','batch','size','indices_sha256','images_sha256','features_sha256',
    'augmentation_state_after_sha256','ambient_cpu_rng_sha256','ambient_cuda_rng_sha256',
    'gumbel_sha256','online_gap_sha256','online_scores_sha256','all_three_masks_sha256']


def tensor_hash(t):
    return array_hash(t.detach().cpu().numpy())


def augment(rec,don,generator,transform):
    images = np.load(ROOT/'train_images.npy',mmap_mode='r',allow_pickle=False)
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        ri,di = [],[]
        for r,d in zip(rec,don):
            di.append(transform(Image.fromarray(images[d])))
            ri.append(transform(Image.fromarray(images[r])))
        generator.set_state(torch.get_rng_state())
    assert torch.equal(before,torch.get_rng_state())
    return torch.stack(ri),torch.stack(di)


def fuse(receiver,donor,indices):
    i = torch.from_numpy(indices.astype(np.int64).copy()).cuda()
    mask = torch.zeros((len(indices),64),dtype=torch.bool,device='cuda')
    mask.scatter_(1,i,True)
    assert (mask.sum(1) == 16).all()
    assert np.all(np.diff(np.sort(indices,axis=1),axis=1) > 0)
    return torch.where(mask[:,:,None,None],donor,receiver),mask


def scheduler():
    tree = ast.parse((RUNTIME/'train_stage2.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'adjust_learning_rate')
    ns = {'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'original_stage2_scheduler','exec'),ns)
    return ns['adjust_learning_rate']


def optimizer_for(cfg,cls):
    opt = torch.optim.SGD([{'params':cls.parameters()}],cfg.lr,momentum=cfg.momentum,weight_decay=cfg.weight_decay)
    assert not opt.state
    assert {id(p) for group in opt.param_groups for p in group['params']} == {id(p) for p in cls.parameters()}
    return opt


def lock_stage2():
    files = ['common.py','build_clean_statistics.py','clean_online_policy.py','clean_stage2.py',
        'reused_vs_math.py','stage2_original_effective.yaml','clean_stage2_effective.yaml','protocol.json']
    files += ['runtime/'+r for r in read(ROOT/'preflight.json')['runtime_files']]
    files += [str(Path(read(ROOT/'clean_stage1_protocol.json')['best_checkpoint']).relative_to(ROOT))]
    data = ['sampling_plan.npz','gumbel_noise_float64.npy','class_feature_statistics.npz','validation_feature_maps.npy',
        'split_indices.npz','train_images.npy','train_labels.npy','validation_labels.npy']
    lock = {'locked_at':now(),'files':{p:sha(ROOT/p) for p in files},'data_files':{p:sha(ROOT/p) for p in data}}
    snapshot = ROOT/'stage2_source_snapshot'
    for rel in files:
        if rel.endswith('.pth'):
            continue
        dest = snapshot/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,dest)
    dump(ROOT/'stage2_source_lock.json',lock)
    dump(snapshot/'source_manifest.json',lock)
    # Record all local changes as reviewable text, preserving every original file.
    import difflib
    diff = ''.join(difflib.unified_diff((OLD/'online_policy.py').read_text(encoding='utf-8').splitlines(True),
        (ROOT/'clean_online_policy.py').read_text(encoding='utf-8').splitlines(True),
        fromfile='previous_VS/online_policy.py',tofile='clean_three_arm/clean_online_policy.py'))
    (ROOT/'online_policy.adaptation.diff').write_text(diff,encoding='utf-8')
    assert sha(ROOT/'reused_vs_math.py') == sha(OLD/'reused_vs_math.py')


def check_stage2_sources():
    check_sources()
    lock = read(ROOT/'stage2_source_lock.json')
    for group in ['files','data_files']:
        for p,value in lock[group].items():
            assert sha(ROOT/p) == value, ('StageII source/data changed',p)
    assert read(ROOT/'clean_stage1_protocol.json')['passed']
    for name in ['stage1_train','clean_statistics']:
        g = read(ROOT/f'test_guard_{name}.json')
        assert not g['blocked_attempts'] and g['CIFAR100_trainFalse_calls'] == 0


def dryrun():
    assert not (ROOT/'arms').exists()
    lock_stage2()
    check_stage2_sources()
    plan = np.load(ROOT/'sampling_plan.npz',allow_pickle=False)
    noise = np.load(ROOT/'gumbel_noise_float64.npy',mmap_mode='r',allow_pickle=False)
    labels = np.load(ROOT/'train_labels.npy',allow_pickle=False)
    rec,don = plan['receivers'][0,:BATCH],plan['donors'][0,:BATCH]
    results = {}
    expected = None
    for arm in ARMS:
        cfg,tr,bb,cls,CE,methods,p,q = frozen_models()
        initial = {'backbone':state_hash(bb),'classifier':state_hash(cls)}
        frozen = {k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
        aug = torch.Generator().manual_seed(S2_SEED+300009)
        ri,di = augment(rec,don,aug,tr['train_transform'])
        with torch.no_grad():
            receiver,donor = bb(ri.cuda()),bb(di.cuda())
        trace = online_masks(receiver,labels[don],noise[0,:BATCH])
        controls = {'initial':initial,'images':array_hash(ri.numpy(),di.numpy()),
            'features':array_hash(receiver.cpu().numpy(),donor.cpu().numpy()),
            'S':trace['score_sha256'],'all_masks':trace['all_three_masks_sha256']}
        if expected is None:
            expected = controls
        else:
            assert controls == expected
        with torch.no_grad():
            mixed,mask = fuse(receiver,donor,trace['masks'][arm])
            full = mask[:,:,None,None].expand_as(mixed)
            assert torch.equal(mixed[full],donor[full]) and torch.equal(mixed[~full],receiver[~full])
            gap = torch.nn.functional.avg_pool2d(mixed,8).view(-1,64)
            assert torch.equal(cls(mixed),cls.module.fc(gap))
        targets = torch.from_numpy(labels[rec].copy()).cuda()
        assert np.array_equal(targets.cpu().numpy(),labels[rec])
        optimizer = optimizer_for(cfg,cls)
        scheduler()(optimizer,0,cfg)
        criterion = CE(cfg,np.bincount(labels,minlength=100)).cuda()
        loss = criterion(cls(mixed),targets,extra_info=None)
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cls.parameters())
        optimizer.step()
        assert state_hash(cls) != initial['classifier']
        check_frozen(bb,frozen)
        results[arm] = {'loss':float(loss.detach()),'mask_sha256':array_hash(trace['masks'][arm]),
            '16_unique_full_channels':True,'copied_unselected_and_labels_exact':True,
            'classifier_updated':True,'backbone_BN_affine_and_buffers_bitwise_unchanged':True,
            'inputs_features_score_and_noise_paired':True}
    guard_pass()
    dump(ROOT/'stage2_dryrun_verification.json',{'passed':True,'completed_at':now(),'arms':results,
        'seed':123,'controls':expected,'no_old_checkpoint_or_statistics':True,
        'disposable_updates_discarded':True,'formal_classifiers_freshly_reinitialized':True})
    print('CLEAN STAGE II DRYRUN PASS:three paired masks,16 complete channels,labels exact,classifier-only updates,BB/BN unchanged',flush=True)


def train_one(arm):
    out = ROOT/'arms'/arm
    assert not out.exists(), 'Refuse overwrite or stage2 resume'
    (out/'checkpoints').mkdir(parents=True)
    cfg,tr,bb,cls,CE,methods,s1,q = frozen_models()
    cfg.name = ROOT.name+'_'+arm
    cfg.model_dir = str(out/'checkpoints')
    (out/'effective.yaml').write_text(cfg.dump(),encoding='utf-8')
    experiment = read(ROOT/'protocol.json')
    experiment.update({'arm':arm,'clean_stage1_sha256':sha(s1),'classifier_initialization':'same Clean StageI best fc; fresh empty independent SGD',
        'statistics_sha256':sha(ROOT/'class_feature_statistics.npz'),'output':str(out)})
    dump(out/'experiment_config.json',experiment)
    frozen = {k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
    initial_cls = {k:v.detach().cpu().clone() for k,v in cls.state_dict().items()}
    initial = {'backbone':state_hash(bb),'classifier':state_hash(cls)}
    optimizer = optimizer_for(cfg,cls)
    labels = np.load(ROOT/'train_labels.npy',allow_pickle=False)
    val_labels = np.load(ROOT/'validation_labels.npy',allow_pickle=False)
    val_bank = np.load(ROOT/'validation_feature_maps.npy',mmap_mode='r',allow_pickle=False)
    plan = np.load(ROOT/'sampling_plan.npz',allow_pickle=False)
    noise = np.load(ROOT/'gumbel_noise_float64.npy',mmap_mode='r',allow_pickle=False)
    aug = torch.Generator().manual_seed(S2_SEED+300009)
    criterion = CE(cfg,np.bincount(labels,minlength=100)).cuda()
    reference = None
    if arm != ARMS[0]:
        with (ROOT/'arms'/ARMS[0]/'batches.csv').open(encoding='utf-8') as f:
            reference = list(csv.DictReader(f))
        assert len(reference) == 850 and initial == read(ROOT/'arms'/ARMS[0]/'status.json')['initial_state_hashes']
    status = {'state':'running','started_at':now(),'arm':arm,'seed':123,'epochs':[],
        'best_epoch':None,'best_validation':None,'initial_state_hashes':initial,'TEST_used':False,'fresh_optimizer':True}
    dump(out/'status.json',status)
    # Identical disposable eval-mode GPU warmup in each arm.
    images = np.load(ROOT/'train_images.npy',mmap_mode='r',allow_pickle=False)[:BATCH]
    with torch.no_grad():
        warm = torch.stack([tr['eval_transform'](Image.fromarray(v)) for v in images]).cuda()
        bb(warm);bb(warm)
    torch.cuda.synchronize()
    actual = np.empty((10,N,16),dtype=np.uint8)
    composition = {k:np.empty((10,N),dtype=np.float64) for k in ['V','B','ratio','score','common_with_uniform','fallback']}
    saved_scores = np.lib.format.open_memmap(ROOT/'online_clean_scores_float64.npy',mode='w+',dtype=np.float64,shape=(10,N,64)) if arm == ARMS[0] else None
    headers = [*PAIR_KEYS,'loss','lr','mask_sha256','augmentation_seconds','feature_seconds','scoring_seconds',
        'fusion_seconds','update_seconds','mean_V','mean_B','mean_variance_fraction','mean_selected_score',
        'mean_common_Uniform','fallback_draws']
    schedule = scheduler()
    started = time.perf_counter()
    epoch_csv = []
    with (out/'batches.csv').open('w',encoding='utf-8',newline='') as f, (out/'training.log').open('w',encoding='utf-8',buffering=1) as log:
        writer = csv.DictWriter(f,fieldnames=headers);writer.writeheader()
        for epoch in range(10):
            cls.train();schedule(optimizer,epoch,cfg)
            torch.cuda.reset_peak_memory_stats()
            t_epoch = time.perf_counter()
            total_loss,seen = 0.,0
            counts = np.zeros(64,dtype=np.int64)
            timing = {k:0. for k in ['augmentation_seconds','feature_seconds','scoring_seconds','fusion_seconds','update_seconds']}
            for batch,start in enumerate(range(0,N,BATCH)):
                end = min(start+BATCH,N)
                rec,don = plan['receivers'][epoch,start:end],plan['donors'][epoch,start:end]
                target = torch.from_numpy(labels[rec].copy()).cuda()
                t = time.perf_counter();ri,di = augment(rec,don,aug,tr['train_transform'])
                aug_time = time.perf_counter()-t
                t = time.perf_counter()
                with torch.no_grad():
                    receiver,donor = bb(ri.cuda()),bb(di.cuda())
                torch.cuda.synchronize();feature_time = time.perf_counter()-t
                assert receiver.shape == (end-start,64,8,8) and not receiver.requires_grad
                t = time.perf_counter();trace = online_masks(receiver,labels[don],noise[epoch,start:end])
                score_time = time.perf_counter()-t
                if saved_scores is not None:
                    saved_scores[epoch,start:end] = trace['scores']
                row = {'epoch':epoch+1,'batch':batch,'size':end-start,
                    'indices_sha256':array_hash(rec,don,labels[rec],labels[don]),
                    'images_sha256':array_hash(ri.numpy(),di.numpy()),
                    'features_sha256':array_hash(receiver.cpu().numpy(),donor.cpu().numpy()),
                    'augmentation_state_after_sha256':tensor_hash(aug.get_state()),
                    'ambient_cpu_rng_sha256':tensor_hash(torch.get_rng_state()),
                    'ambient_cuda_rng_sha256':tensor_hash(torch.cuda.get_rng_state()),
                    'gumbel_sha256':array_hash(noise[epoch,start:end]),'online_gap_sha256':trace['GAP_sha256'],
                    'online_scores_sha256':trace['score_sha256'],'all_three_masks_sha256':trace['all_three_masks_sha256']}
                if reference is not None:
                    assert all(str(row[k]) == reference[epoch*85+batch][k] for k in PAIR_KEYS), ('paired batch differs',arm,epoch,batch)
                t = time.perf_counter()
                chosen = trace['masks'][arm]
                with torch.no_grad():
                    mixed,mask = fuse(receiver,donor,chosen)
                torch.cuda.synchronize();fusion_time = time.perf_counter()-t
                assert torch.isfinite(mixed).all() and not mixed.requires_grad
                assert np.array_equal(target.cpu().numpy(),labels[rec])
                actual[epoch,start:end] = chosen
                counts += np.bincount(chosen.astype(np.int64).ravel(),minlength=64)
                selected = selected_statistics(trace,chosen)
                for k,v in selected.items():
                    composition[k][epoch,start:end] = v
                t = time.perf_counter();output = cls(mixed)
                loss = criterion(output,target,extra_info=None)
                assert torch.isfinite(output).all() and torch.isfinite(loss)
                optimizer.zero_grad();loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cls.parameters())
                optimizer.step();torch.cuda.synchronize()
                row.update({'loss':float(loss.detach()),'lr':optimizer.param_groups[0]['lr'],
                    'mask_sha256':array_hash(chosen),'augmentation_seconds':aug_time,'feature_seconds':feature_time,
                    'scoring_seconds':score_time,'fusion_seconds':fusion_time,'update_seconds':time.perf_counter()-t,
                    'mean_V':float(selected['V'].mean()),'mean_B':float(selected['B'].mean()),
                    'mean_variance_fraction':float(selected['ratio'].mean()),'mean_selected_score':float(selected['score'].mean()),
                    'mean_common_Uniform':float(selected['common_with_uniform'].mean()),
                    'fallback_draws':int((selected['fallback'] > 0).sum())})
                writer.writerow(row)
                for k in timing:timing[k] += row[k]
                total_loss += row['loss']*(end-start);seen += end-start
                if batch % 40 == 0:
                    line = f'{arm} epoch{epoch+1}/10 batch{batch}/85 loss={row["loss"]:.6f}'
                    log.write(line+'\n');print(line,flush=True)
            train_seconds = time.perf_counter()-t_epoch
            assert seen == N and counts.sum() == 16*N
            check_frozen(bb,frozen)
            delta = {k:float((v.detach().cpu()-initial_cls[k]).abs().max()) for k,v in cls.state_dict().items()}
            assert all(d > 0 for d in delta.values())
            val,logits,pred = evaluate(cls,val_labels,feature_maps=val_bank)
            improved = status['best_validation'] is None or val['Overall'] > status['best_validation']['Overall']
            if improved:status['best_epoch'],status['best_validation'] = epoch+1,val
            entry = {'epoch':epoch+1,'lr':optimizer.param_groups[0]['lr'],'train_loss':total_loss/N,
                'validation':val,'steps':85,'training_seconds':train_seconds,'timing_components':timing,
                'classifier_parameter_max_abs_change':delta,'backbone_BN_bitwise_unchanged':True,
                'channel_selected_counts':counts.tolist(),'sampling_composition':{k:float(v[epoch].mean()) for k,v in composition.items()},
                'equal_Uniform_mask_rate':float((composition['common_with_uniform'][epoch] == 16).mean()),
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
            status['epochs'].append(entry)
            checkpoint = {'epoch':epoch+1,'state_dict_model':frozen,
                'state_dict_classifier':{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
                'optimizer':optimizer.state_dict(),'validation':val,'arm':arm,'seed':123,
                'Clean_StageI_sha256':sha(s1),'VS_statistics_sha256':sha(ROOT/'class_feature_statistics.npz'),
                'split_sha256':sha(ROOT/'split_indices.npz'),'TEST_used':False}
            if improved:torch.save(checkpoint,out/'checkpoints/model_best.pth.tar')
            if epoch == 9:torch.save(checkpoint,out/'checkpoints/model_final.pth.tar')
            np.savez_compressed(out/f'validation_epoch{epoch+1:02d}.npz',logits=logits,predictions=pred,labels=val_labels)
            epoch_csv.append({'arm':arm,'epoch':epoch+1,'train_loss':entry['train_loss'],'lr':entry['lr'],
                **{k:val[k] for k in ['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro','validation_CE']},
                'training_seconds':train_seconds,'peak_allocated_mib':entry['peak_allocated_mib'],'peak_reserved_mib':entry['peak_reserved_mib']})
            csv_write(out/'epoch_validation_metrics.csv',epoch_csv)
            guard_pass();dump(out/'status.json',status)
            line = f'CLEAN {arm} epoch{epoch+1}:valOverall={val["Overall"]:.4f} Head={val["Head"]:.4f} Medium={val["OfficialMedium"]:.4f} Tail={val["Tail"]:.4f} Few={val["Few"]:.4f}; train={train_seconds:.2f}s frozen PASS'
            log.write(line+'\n');print(line,flush=True)
    np.savez_compressed(out/'actual_masks.npz',masks=actual)
    np.savez_compressed(out/'sampling_composition.npz',**composition)
    if saved_scores is not None:saved_scores.flush()
    status.update({'state':'completed','finished_at':now(),'total_run_seconds':time.perf_counter()-started,
        'total_training_seconds':sum(x['training_seconds'] for x in status['epochs']),
        'finite_loss_classifier_gradient_steps':850,'paired_steps':850,
        'final_state_hashes':{'backbone':state_hash(bb),'classifier':state_hash(cls)},
        'checkpoint_sha256':{kind:sha(out/f'checkpoints/model_{kind}.pth.tar') for kind in ['best','final']}})
    checks = {}
    for kind in ['best','final']:
        checkpoint = torch.load(out/f'checkpoints/model_{kind}.pth.tar',map_location='cpu',weights_only=False)
        assert all(v.numpy().tobytes() == frozen[k].numpy().tobytes() for k,v in checkpoint['state_dict_model'].items())
        cls.load_state_dict(checkpoint['state_dict_classifier'],strict=True)
        reloaded,_,_ = evaluate(cls,val_labels,feature_maps=val_bank)
        assert reloaded == checkpoint['validation']
        checks[kind] = {'epoch':checkpoint['epoch'],'metrics':reloaded,'checkpoint_sha256':sha(out/f'checkpoints/model_{kind}.pth.tar'),
            'backbone_BN_saved_bitwise_exact':True,'validation_reload_exact':True}
    dump(out/'checkpoint_validation.json',checks);dump(out/'status.json',status)
    guard_pass()
    print(f'COMPLETE CLEAN {arm}:10epochs,850SGD updates; independent Clean initialization; no TEST',flush=True)


def train_three():
    assert read(ROOT/'stage2_dryrun_verification.json')['passed']
    assert not (ROOT/'STOP_COMPLETED.json').exists()
    for arm in ARMS:
        check_stage2_sources();train_one(arm)
    rows = {}
    for arm in ARMS:
        with (ROOT/'arms'/arm/'batches.csv').open(encoding='utf-8') as f:rows[arm] = list(csv.DictReader(f))
    matches = {arm:{k:sum(a[k] == b[k] for a,b in zip(rows[ARMS[0]],rows[arm])) for k in PAIR_KEYS} for arm in ARMS}
    assert all(v == 850 for d in matches.values() for v in d.values())
    dump(ROOT/'pairing_verification.json',{'verified_at':now(),'batches_each':850,'equal_control_rows':matches,
        'independent_same_Clean_initialization':True,'only_selection_mask_differs':True})
    plan = np.load(ROOT/'sampling_plan.npz')
    all_masks = {arm.replace('-','_'):np.load(ROOT/'arms'/arm/'actual_masks.npz')['masks'] for arm in ARMS}
    for mask in all_masks.values():
        assert mask.shape == (10,N,16) and np.all(np.diff(np.sort(mask,axis=-1),axis=-1) > 0)
    np.savez_compressed(ROOT/'sampling_masks.npz',receivers=plan['receivers'],donors=plan['donors'],**all_masks)
    guard_pass();check_stage2_sources();check_history()
    dump(ROOT/'STOP_COMPLETED.json',{'completed_at':now(),'StageI_seed':42,'StageI_epochs':200,
        'StageII_seeds':[123],'StageII_arms':ARMS,'epochs_each':10,'updates_each':850,'TEST_used':False,
        'new_training_requires_explicit_authorization':True})
    print('STOP:Clean StageI and exactly3 StageII arms completed; no extra seeds, no TEST',flush=True)


def main():
    p = argparse.ArgumentParser();p.add_argument('action',choices=['dryrun','train'])
    action = p.parse_args().action
    install_test_guard('stage2_'+action)
    try:
        {'dryrun':dryrun,'train':train_three}[action]()
    except BaseException:
        import traceback
        dump(ROOT/'failure_stage2.json',{'at':now(),'action':action,'traceback':traceback.format_exc()})
        guard_flush();raise


if __name__ == '__main__':
    main()
