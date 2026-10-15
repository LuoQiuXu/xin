"""Strict local preflight and historical write protection. No training or statistic estimation."""
from pathlib import Path
import csv,hashlib,json,os,pickle,sys,time,traceback
import numpy as np
import torch
import torchvision
from PIL import Image


def dump(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def protect_history(old,events):
    old=str(Path(old).resolve()).casefold().rstrip('\\/')
    def historical(value):
        if not isinstance(value,(str,bytes,os.PathLike)): return False
        text=os.path.abspath(os.fsdecode(value)).casefold()
        return text==old or text.startswith(old+os.sep)
    def audit(event,args):
        deny=False
        if event=='open' and args and historical(args[0]):
            mode=args[1] if len(args)>1 else None
            flags=args[2] if len(args)>2 else 0
            deny=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        elif event in ['os.remove','os.rmdir','os.mkdir','os.rename','os.link','os.symlink','os.chmod','os.utime']:
            deny=any(historical(a) for a in args[:2])
        if deny:
            events.append({'event':event,'path':str(args[0])})
            raise RuntimeError('Historical experiment is read-only: '+str(args[0]))
    sys.addaudithook(audit)


def history_manifest(old,sha):
    data={}
    for current,dirs,files in os.walk(old):
        dirs[:]=[d for d in dirs if d not in {'.git','__pycache__'}]
        for name in files:
            p=Path(current)/name
            data[p.relative_to(old).as_posix()]=sha(p)
    return data


def make_control_row(r,seed,bb,tr,labels,plan,noise,aug,epoch,batch):
    start=batch*r.BATCH;end=min(start+r.BATCH,r.N)
    rec,don=plan['receivers'][epoch,start:end],plan['donors'][epoch,start:end]
    ri,di=r.augment(rec,don,aug,tr['train_transform'])
    with torch.no_grad():
        receiver,donor=bb(ri.cuda()),bb(di.cuda())
    g=noise[epoch,start:end]
    trace=r.online_masks(receiver,labels[don],g)
    row={'epoch':epoch+1,'batch':batch,'size':end-start,
        'indices_sha256':r.array_hash(rec,don,labels[rec],labels[don]),
        'images_sha256':r.array_hash(ri.numpy(),di.numpy()),
        'features_sha256':r.array_hash(receiver.cpu().numpy(),donor.cpu().numpy()),
        'augmentation_state_after_sha256':r.tensor_hash(aug.get_state()),
        'ambient_cpu_rng_sha256':r.tensor_hash(torch.get_rng_state()),
        'ambient_cuda_rng_sha256':r.tensor_hash(torch.cuda.get_rng_state()),
        'gumbel_sha256':r.array_hash(g),'online_gap_sha256':trace['GAP_sha256'],
        'online_scores_sha256':trace['score_sha256'],'all_two_masks_sha256':trace['all_two_masks_sha256']}
    chosen=r.choose(r.ARM,trace)
    with torch.no_grad():
        mixed,mask=r.fuse(receiver,donor,chosen)
        full=mask[:,:,None,None].expand_as(mixed)
        assert torch.equal(mixed[full],donor[full]) and torch.equal(mixed[~full],receiver[~full])
    assert receiver.shape==(end-start,64,8,8) and not mixed.requires_grad
    return row


def preflight(r):
    r.self_test()
    assert torch.cuda.is_available(),'CUDA GPU required'
    env={'python':sys.executable,'python_version':sys.version,'torch':torch.__version__,
        'torchvision':torchvision.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0),
        'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/1024**2}
    dump(r.CURRENT/'environment.json',env)
    needed=[]
    for seed in r.SEEDS:
        needed+=r.ensure_inputs(seed)
        for arm in ['Uniform-PerSample','Low-VS']:
            p=r.outdir(seed,arm)
            for name in ['actual_masks.npz','status.json','checkpoint_validation.json']:
                if not (p/name).is_file(): raise FileNotFoundError(str(p/name))
            for epoch in range(1,11):
                if not (p/f'validation_epoch{epoch:02d}.npz').is_file(): raise FileNotFoundError(str(p/f'validation_epoch{epoch:02d}.npz'))
    before=history_manifest(r.OLD,r.sha)
    dump(r.RESULTS/'historical_readonly_baseline.json',before)
    labels=np.load(r.OLD/'train_labels.npy',allow_pickle=False)
    val_labels=np.load(r.OLD/'validation_labels.npy',allow_pickle=False)
    assert labels.shape==(r.N,) and set(labels)==set(range(100))
    assert val_labels.shape==(1980,) and np.array_equal(np.bincount(val_labels,minlength=100),np.r_[0,np.repeat(20,99)])
    split=np.load(r.OLD/'split_indices.npz',allow_pickle=False)
    train_ids,val_ids=split['train_raw_indices'],split['validation_raw_indices']
    assert len(np.intersect1d(train_ids,val_ids))==0
    from common import DATA
    raw_path=DATA/'cifar-100-python/train'
    if not raw_path.is_file(): raise FileNotFoundError(str(raw_path))
    with raw_path.open('rb') as f: raw=pickle.load(f,encoding='latin1')
    raw_y=np.asarray(raw['fine_labels']);raw_x=raw['data'].reshape(-1,3,32,32).transpose(0,2,3,1)
    assert np.array_equal(labels,raw_y[train_ids]) and np.array_equal(val_labels,raw_y[val_ids])
    assert np.array_equal(np.load(r.OLD/'train_images.npy',mmap_mode='r'),raw_x[train_ids])
    if (r.OLD/'validation_images.npy').is_file():
        assert np.array_equal(np.load(r.OLD/'validation_images.npy',mmap_mode='r'),raw_x[val_ids])
    actual_counts=np.bincount(labels,minlength=100)
    expected_counts=np.asarray([int(500*.01**(c/99)) for c in range(100)])
    assert np.array_equal(actual_counts,expected_counts)
    rs=np.random.RandomState(0);official_ids=[]
    for c,n in enumerate(expected_counts):
        ids=np.flatnonzero(raw_y==c);rs.shuffle(ids);official_ids.extend(ids[:n])
    assert np.array_equal(train_ids,np.asarray(official_ids))
    dataset={'LT':r.N,'validation':1980,'classes_outputs':100,'missing_validation_classes':[0],
        'raw_train_path':str(raw_path),'raw_train_sha256':r.sha(raw_path),'raw_ID_overlap':0,
        'train_and_validation_pixels_match_raw_TRAIN':True,'LT_formula_and_official_index_reconstruction':True,
        'split_sha256':r.sha(r.OLD/'split_indices.npz')}
    dump(r.CURRENT/'dataset_verification.json',dataset)
    # Verify stored controls before consuming any model input; never train/rebuild them.
    controls={}
    for seed in r.SEEDS:
        assert len(r.control_rows(seed))==850 and len(r.PAIR_KEYS)==13
        for arm in ['Uniform-PerSample','Low-VS']:
            p=r.outdir(seed,arm)
            with (p/'epoch_validation_metrics.csv').open(encoding='utf-8',newline='') as f: records=list(csv.DictReader(f))
            assert [int(x['epoch']) for x in records]==list(range(1,11))
            v=np.load(p/'validation_epoch10.npz',allow_pickle=False)
            assert np.array_equal(v['labels'],val_labels) and v['logits'].shape==(1980,100)
            assert np.array_equal(v['predictions'],v['logits'].argmax(1))
            m=r.metrics(v['predictions'],val_labels)
            assert all(abs(m[k]-float(records[-1][k]))<1e-12 for k in ['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro'])
            controls[f'{seed}/{arm}']={'status_sha256':r.sha(p/'status.json'),'epoch10_sha256':r.sha(p/'validation_epoch10.npz'),
                'metrics_sha256':r.sha(p/'epoch_validation_metrics.csv'),'actual_masks_sha256':r.sha(p/'actual_masks.npz')}
        plan=np.load(r.plan_dir(seed)/'sampling_plan.npz',allow_pickle=False)
        assert plan['receivers'].shape==plan['donors'].shape==(10,r.N)
        for e in range(10):assert np.array_equal(np.sort(plan['donors'][e]),np.arange(r.N))
        noise=np.load(r.plan_dir(seed)/'gumbel_noise_float64.npy',mmap_mode='r',allow_pickle=False)
        assert noise.shape==(10,r.N,64) and noise.dtype==np.float64 and np.isfinite(noise).all()
    dump(r.CURRENT/'locked_controls.json',controls)
    # Replay all 850 no-update batches per seed. Exact 13-field parity is a pre-training gate.
    comparisons=[];mismatches=[];passed={}
    for seed in r.SEEDS:
        cfg,tr,bb,cls,CE,methods,s1,q=r.frozen_models(seed)
        assert r.sha(s1)=='7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88'
        expected=json.loads((r.outdir(seed,'Uniform-PerSample')/'status.json').read_text(encoding='utf-8'))['initial_state_hashes']
        assert {'backbone':r.state_hash(bb),'classifier':r.state_hash(cls)}==expected
        frozen={k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
        # Same two eval warmup forwards as the original saved Uniform run.
        images=np.load(r.OLD/'train_images.npy',mmap_mode='r')[:r.BATCH]
        with torch.no_grad():
            warm=torch.stack([tr['eval_transform'](Image.fromarray(v)) for v in images]).cuda()
            bb(warm);bb(warm)
        plan=np.load(r.plan_dir(seed)/'sampling_plan.npz',allow_pickle=False)
        noise=np.load(r.plan_dir(seed)/'gumbel_noise_float64.npy',mmap_mode='r',allow_pickle=False)
        ref=r.control_rows(seed);aug=torch.Generator().manual_seed(seed+300009)
        eq={k:0 for k in r.PAIR_KEYS}
        for e in range(10):
            for b in range(85):
                actual=make_control_row(r,seed,bb,tr,labels,plan,noise,aug,e,b)
                wanted=ref[e*85+b]
                different=[k for k in r.PAIR_KEYS if str(actual[k])!=wanted[k]]
                for k in r.PAIR_KEYS:
                    ok=str(actual[k])==wanted[k];eq[k]+=int(ok)
                    comparisons.append({'seed':seed,'epoch':e+1,'batch':b,'field':k,'matches':ok,'expected':wanted[k],'actual':str(actual[k])})
                if different:
                    mismatches.append({'seed':seed,'epoch':e+1,'batch':b,'mismatching_fields':different})
                    print(f'PREFLIGHT PAIRING FAIL seed={seed} epoch={e+1} batch={b}: '+','.join(different),flush=True)
                    break
            r.check_frozen(bb,frozen)
            if mismatches:break
            print(f'PREFLIGHT replay seed={seed} epoch={e+1}:85 no-update batches matched',flush=True)
        passed[str(seed)]=eq
        del bb,cls;torch.cuda.empty_cache()
        if mismatches:break
    r.csv_write(r.CURRENT/'pairing_preflight.csv',comparisons)
    after=history_manifest(r.OLD,r.sha)
    assert before==after,'Historical experiment changed during read-only preflight'
    result={'environment':env,'dataset':dataset,'paired_fields':passed,'mismatches':mismatches,
        'historical_files_unchanged':len(before),'backward_calls':0,'optimizer_steps':0,
        'old_controls_retrained':False,'checkpoint_rebuilt':False,'TEST_used':False,
        'passed':not mismatches}
    dump(r.CURRENT/'preflight_summary.json',result)
    if mismatches:
        raise RuntimeError('Strict historical 13-field pairing preflight failed; no training permitted. See '+str(r.CURRENT/'pairing_preflight.csv'))
    dump(r.RESULTS/'preflight_pass.json',{'passed':True,'details':str(r.CURRENT/'preflight_summary.json'),
        'environment':env,'historical_manifest_sha256':r.sha(r.RESULTS/'historical_readonly_baseline.json')})
    print('PREFLIGHT PASS:two seeds x850 no-update paired batches,all13 fields,TRAIN-only data,frozen BB/BN',flush=True)
