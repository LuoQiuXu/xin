"""Original CE+MixUp Stage I functions, fresh seed42, TRAIN-only validation."""
import argparse
import logging
import shutil
from common import *


def original_functions():
    from utils import AverageMeter, ProgressMeter, accuracy
    import methods
    source = (RUNTIME/'train_stage1.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['train','adjust_learning_rate']]
    assert len(nodes) == 2
    scope = {'time':time,'math':math,'np':np,'torch':torch,'AverageMeter':AverageMeter,
        'ProgressMeter':ProgressMeter,'accuracy':accuracy,'mixup_data':methods.mixup_data,
        'mixup_criterion':methods.mixup_criterion}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(RUNTIME/'train_stage1.py'),'exec'),scope)
    return scope


def make_models():
    cfg,tr,ResNet,Classifier,CE,methods = imports(1)
    seed_all(S1_SEED,True)
    bb = torch.nn.DataParallel(ResNet()).cuda()
    cls = torch.nn.DataParallel(Classifier(feat_in=64,num_classes=100)).cuda()
    assert not bb.module.feature_tuning
    assert sum(p.numel() for p in bb.parameters()) == 463504
    assert sum(p.numel() for p in cls.parameters()) == 6500
    counts = np.bincount(np.load(ROOT/'train_labels.npy',allow_pickle=False),minlength=100)
    criterion = CE(config=cfg,cls_num_list=counts).to('cuda')
    optimizer = torch.optim.SGD([{'params':bb.parameters()},{'params':cls.parameters()}],cfg.lr,
        momentum=cfg.momentum,weight_decay=cfg.weight_decay)
    assert not optimizer.state
    return cfg,tr,bb,cls,criterion,optimizer


def logger_for(folder):
    log = logging.getLogger('clean_stage1')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    file = logging.FileHandler(folder/'training.log',encoding='utf-8')
    file.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    log.addHandler(file)
    log.addHandler(logging.StreamHandler(sys.stdout))
    return log


def snapshot_sources():
    files = ['common.py','prepare_clean.py','clean_stage1.py','stage1_original_effective.yaml',
        'clean_stage1_effective.yaml','protocol.json']
    files += ['runtime/'+p for p in read(ROOT/'preflight.json')['runtime_files']]
    snapshot = ROOT/'stage1/source_snapshot'
    for rel in files:
        dest = snapshot/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,dest)
    data = ['split_indices.npz','train_images.npy','train_labels.npy','validation_images.npy','validation_labels.npy']
    lock = {'locked_at':now(),'files':{p:sha(ROOT/p) for p in files},
        'data_files':{p:sha(ROOT/p) for p in data},'snapshot':str(snapshot),
        'source_functions':'verbatim AST train + adjust_learning_rate from copied compatibility source',
        'dataset_change':'frozen verified LT arrays and explicit TRAIN holdout; original Dataset constructor/main never called',
        'seed_change':'training Python,NumPy,Torch seed42; subset seed0 isolated in prepare',
        'checkpoint_loads_before_training':0}
    dump(ROOT/'source_lock.json',lock)
    dump(snapshot/'source_manifest.json',lock)


def observe_training(scope,optimizer,bb,cls):
    record = {}
    real_mixup = scope['mixup_criterion']
    real_step = optimizer.step

    def mixup(criterion,pred,y_a,y_b,lam,extra_info=None):
        loss = real_mixup(criterion,pred,y_a,y_b,lam,extra_info=extra_info)
        assert torch.isfinite(loss)
        value, size = float(loss.detach()),len(y_a)
        record['loss_sum'] += value*size
        record['samples'] += size
        record['loss_min'] = min(record['loss_min'],value)
        record['loss_max'] = max(record['loss_max'],value)
        return loss

    def step(*args,**kwargs):
        grads = [p.grad for p in list(bb.parameters())+list(cls.parameters())]
        assert all(g is not None for g in grads)
        assert torch.isfinite(torch.cat([g.flatten() for g in grads])).all()
        result = real_step(*args,**kwargs)
        record['steps'] += 1
        return result
    scope['mixup_criterion'] = mixup
    optimizer.step = step
    return record


def reset_record(record):
    record.clear()
    record.update({'steps':0,'samples':0,'loss_sum':0.,'loss_min':float('inf'),'loss_max':float('-inf')})


def dryrun():
    assert not (ROOT/'stage1').exists()
    folder = ROOT/'stage1_dryrun'
    folder.mkdir()
    cfg,tr,bb,cls,criterion,optimizer = make_models()
    log = logger_for(folder)
    initial = {'backbone':state_hash(bb),'classifier':state_hash(cls)}
    scope = original_functions()
    rec = observe_training(scope,optimizer,bb,cls)
    reset_record(rec)
    ds = ArrayDataset('train',tr['train_transform'])
    subset = torch.utils.data.Subset(ds,np.random.default_rng(42).choice(N,BATCH,replace=False))
    loader = torch.utils.data.DataLoader(subset,batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True)
    scope['adjust_learning_rate'](optimizer,0,cfg)
    scope['train'](loader,bb,cls,criterion,optimizer,0,cfg,log)
    assert rec['steps'] == 1 and rec['samples'] == 128
    assert state_hash(bb) != initial['backbone'] and state_hash(cls) != initial['classifier']
    bb.eval();cls.eval()
    val = ArrayDataset('validation',tr['eval_transform'])
    small = torch.utils.data.DataLoader(torch.utils.data.Subset(val,np.arange(128)),batch_size=128,num_workers=0)
    with torch.no_grad():
        x,y,_ = next(iter(small))
        logits = cls(bb(x.cuda()))
        assert logits.shape == (128,100) and torch.isfinite(logits).all()
    guard_pass()
    dump(ROOT/'stage1_dryrun_verification.json',{'passed':True,'completed_at':now(),
        'seed':42,'initial_state_hashes':initial,'original_functions_reused':True,
        'actual_LT_forward_backward_SGD_steps':1,'backbone_updated':True,'classifier_updated':True,
        'finite_loss_and_all_gradients':True,'loss':rec['loss_sum']/rec['samples'],
        'validation_source':'TRAIN holdout only; first128 checked; 100 logits',
        'disposable_model_discarded':True,'formal_training_reinitialized_fresh':True})
    print('CLEAN STAGE I DRYRUN PASS: real LT forward/backward/update; TRAIN holdout100 logits; no TEST',flush=True)


def audit_completed():
    out = ROOT/'stage1'
    status = read(out/'status.json')
    assert status['state'] == 'completed' and status['completed_epochs'] == 200
    assert status['gradient_verified_steps'] == 17000 and status['samples_per_epoch'] == N
    check_sources()
    # Verify current process and the whole Stage I training process guards.
    guard_pass()
    gate = read(ROOT/'test_guard_stage1_train.json')
    assert not gate['blocked_attempts'] and gate['CIFAR100_trainFalse_calls'] == 0
    assert gate['allowed_integrity_open_count'] == 0
    p = out/'checkpoints/clean_stage1_best.pth'
    q = torch.load(p,map_location='cpu',weights_only=False)
    assert q['seed'] == 42 and q['selection_partition'] == 'official TRAIN holdout'
    assert q['validation']['samples'] == 1980 and q['validation']['missing_classes'] == [0]
    assert q['epoch'] == status['best_epoch']
    assert abs(q['validation']['Overall']-max(t['validation']['Overall'] for t in status['epochs'])) < 1e-12
    protocol = {'audited_at':now(),'passed':True,'TEST_loaded':False,'TEST_selected':False,
        'training_test_guard':gate,'selected_metric':'99-class TRAIN holdout Overall; strict >; earliest tie',
        'best_epoch':status['best_epoch'],'best_checkpoint':str(p),'best_checkpoint_sha256':sha(p),
        'final_checkpoint_sha256':sha(out/'checkpoints/clean_stage1_final.pth'),
        'validation':q['validation'],'train_val_overlap':0,'train_val_pixel_overlap':0,
        'LT_class_counts':read(ROOT/'preflight.json')['LT_class_counts'],
        'LT_pixel_hash':read(ROOT/'preflight.json')['LT_pixel_hash'],'new_model_initialization':True,
        'old_checkpoint_loads':0,'epochs_completed':200,'training_updates':17000}
    dump(ROOT/'clean_stage1_protocol.json',protocol)
    text = f'''# Clean Stage I 数据与模型选择协议

完成时间：{now()}。独立目录：`{ROOT}`。

## 核心审计结论

1. **训练期间没有加载官方 TEST 图像/标签，没有调用 CIFAR100(train=False)，没有计算 TEST accuracy。** Python open 审计钩子及 Dataset 构造器拦截全程生效；训练阶段允许的 TEST 校验读取次数也为0。prepare阶段只进行了1次不透明字节 MD5 核查。
2. **最佳依据：** 同一1980张官方 TRAIN 留出验证的 Overall，200 epochs 内严格变大才替换最佳；平局取最早。最佳epoch={status['best_epoch']}，验证Overall={q['validation']['Overall']:.6f}%。固定200epoch预算，没有early stopping；逐epoch记录距最近最佳的epoch数。
3. **无重叠：** LT与验证raw ID交集=0，精确像素重叠=0；同一已冻结 split，SHA256 `{sha(ROOT/'split_indices.npz')}`。100类输出保留，class0无验证样本；99类各20张。
4. **LT正确：** 10847张、100类、max500/min5、IF100；官方seed0逐类指数采样索引、分布、像素与过去Stage I一致。训练自身Python/NumPy/Torch RNG均seed42；LT seed0在prepare中隔离验证，冻结数组加载不重置训练RNG。
5. **最佳Checkpoint：** [{p.name}]({p.as_posix()})；SHA256 `{sha(p)}`。从随机初始化训练，没有加载旧epoch198或旧StageII权重。最终Checkpoint另存 `{out/'checkpoints/clean_stage1_final.pth'}`。

## 保持的训练配置

ResNet32、100维classifier、CE+MixUp alpha1、SGD baseLR0.1/momentum0.9/weight_decay0.0002、batch128、workers0、200epochs；原5epoch warmup和160/180 multistep逐字复用。每epoch85次更新，最后batch95，共17000次有限loss与全部梯度检查。原train函数与scheduler以AST直接编译运行；没有调用会创建官方TEST loader的原main。DataParallel、FP32与StageI cudnn.deterministic=True保留。

## 可复核文件

- [epoch_validation_metrics.csv]({(out/'epoch_validation_metrics.csv').as_posix()})（train loss、Overall、Head、OfficialMedium、Tail、Many/Medium/Few、Macro与早停记录）。
- [protocol JSON]({(ROOT/'clean_stage1_protocol.json').as_posix()})、[原始数据preflight]({(ROOT/'preflight.json').as_posix()})、[TEST运行拦截记录]({(ROOT/'test_guard_stage1_train.json').as_posix()})。
- [源码锁]({(ROOT/'source_lock.json').as_posix()})、stage1/source_snapshot、完整training.log和full_console.log、200份validation_epoch预测。

## 解释范围

Clean指新基座的训练和checkpoint选择不依赖官方TEST。该TRAIN验证集仍是历史多轮复用的99类集合，并非全新未暴露数据；缺少class0且不能与100类官方测试准确率直接比较。新seed42与旧seed123的初始化也不同，所以后续变化不能仅归因于删除TEST选模。

协议核查通过后才允许进入本任务已授权的Uniform/High-VS/Low-VS三组StageII。发现任意TEST访问拦截或完整性失败时必须停止。
'''
    (ROOT/'CLEAN_STAGE1_PROTOCOL.md').write_text(text,encoding='utf-8')
    print(f'CLEAN STAGE I PROTOCOL PASS: best epoch{status["best_epoch"]},valOverall={q["validation"]["Overall"]:.4f}%,sha={sha(p)}',flush=True)


def train():
    assert read(ROOT/'stage1_dryrun_verification.json')['passed']
    assert not (ROOT/'stage1').exists(), 'Fresh training only; never resume old model'
    out = ROOT/'stage1'
    (out/'checkpoints').mkdir(parents=True)
    snapshot_sources()
    check_sources()
    cfg,tr,bb,cls,criterion,optimizer = make_models()
    log = logger_for(out)
    initial = {'backbone':state_hash(bb),'classifier':state_hash(cls)}
    assert initial == read(ROOT/'stage1_dryrun_verification.json')['initial_state_hashes']
    train_ds = ArrayDataset('train',tr['train_transform'])
    val_ds = ArrayDataset('validation',tr['eval_transform'])
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=128,shuffle=True,num_workers=0,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds,batch_size=128,shuffle=False,num_workers=0,pin_memory=True)
    labels = val_ds.targets
    scope = original_functions()
    record = observe_training(scope,optimizer,bb,cls)
    status = {'state':'running','started_at':now(),'pid':os.getpid(),'seed':42,'epochs':[],
        'completed_epochs':0,'best_epoch':None,'best_validation':None,'initial_state_hashes':initial,
        'checkpoint_loads_before_training':0,'samples_per_epoch':N,'gradient_verified_steps':0,'TEST_used':False}
    rows = []
    dump(out/'status.json',status)
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    for epoch in range(200):
        check_sources() if epoch == 0 else None
        reset_record(record)
        scope['adjust_learning_rate'](optimizer,epoch,cfg)
        t = time.perf_counter()
        scope['train'](train_loader,bb,cls,criterion,optimizer,epoch,cfg,log)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter()-t
        assert record['steps'] == 85 and record['samples'] == N
        bn_before = {k:v.detach().cpu().numpy().tobytes() for k,v in bb.state_dict().items()
                     if k.endswith(('running_mean','running_var','num_batches_tracked'))}
        t = time.perf_counter()
        val,logits,pred = evaluate(cls,labels,backbone=bb,loader=val_loader)
        assert all(v.detach().cpu().numpy().tobytes() == bn_before[k] for k,v in bb.state_dict().items() if k in bn_before)
        val_seconds = time.perf_counter()-t
        improved = status['best_validation'] is None or val['Overall'] > status['best_validation']['Overall']
        if improved:
            status['best_epoch'],status['best_validation'] = epoch+1,val
        entry = {'epoch':epoch+1,'lr':optimizer.param_groups[0]['lr'],'train_loss':record['loss_sum']/N,
            'loss_min':record['loss_min'],'loss_max':record['loss_max'],'steps':85,'samples':N,
            'train_seconds':train_seconds,'validation_seconds':val_seconds,'validation':val,
            'new_best':improved,'epochs_since_best':epoch+1-status['best_epoch'],
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
        status['epochs'].append(entry)
        status['completed_epochs'] = epoch+1
        status['gradient_verified_steps'] += 85
        checkpoint = {'epoch':epoch+1,'state_dict_model':{k:v.detach().cpu().clone() for k,v in bb.state_dict().items()},
            'state_dict_classifier':{k:v.detach().cpu().clone() for k,v in cls.state_dict().items()},
            'optimizer':optimizer.state_dict(),'seed':42,'selection_partition':'official TRAIN holdout',
            'selection_metric':'Overall','validation':val,'best_acc1':status['best_validation']['Overall'],
            'split_sha256':sha(ROOT/'split_indices.npz'),'TEST_used':False}
        torch.save(checkpoint,out/'checkpoints/clean_stage1_current.pth')
        if improved:
            torch.save(checkpoint,out/'checkpoints/clean_stage1_best.pth')
        if epoch == 199:
            torch.save(checkpoint,out/'checkpoints/clean_stage1_final.pth')
        np.savez_compressed(out/f'validation_epoch{epoch+1:03d}.npz',logits=logits,predictions=pred,labels=labels)
        rows.append({'epoch':epoch+1,'train_loss':entry['train_loss'],'lr':entry['lr'],
            **{k:val[k] for k in ['Overall','Head','OfficialMedium','Tail','Many','Medium','Few','Macro','validation_CE']},
            'new_best':improved,'epochs_since_best':entry['epochs_since_best'],'steps':85,
            'training_seconds':train_seconds,'validation_seconds':val_seconds,
            'peak_allocated_mib':entry['peak_allocated_mib'],'peak_reserved_mib':entry['peak_reserved_mib']})
        csv_write(out/'epoch_validation_metrics.csv',rows)
        guard_pass()
        dump(out/'status.json',status)
        log.info(f'CLEAN S1 epoch{epoch+1}/200 trainLoss={entry["train_loss"]:.6f} valOverall={val["Overall"]:.4f} Head={val["Head"]:.4f} Medium={val["OfficialMedium"]:.4f} Tail={val["Tail"]:.4f} bestEpoch={status["best_epoch"]}; train={train_seconds:.2f}s val={val_seconds:.2f}s')
    status.update({'state':'completed','finished_at':now(),'elapsed_seconds':time.perf_counter()-began,
        'final_state_hashes':{'backbone':state_hash(bb),'classifier':state_hash(cls)},
        'checkpoint_sha256':{kind:sha(out/f'checkpoints/clean_stage1_{kind}.pth') for kind in ['best','final']}})
    dump(out/'status.json',status)
    audit_completed()
    check_history()
    print('CLEAN STAGE I COMPLETE:200 epochs; protocol passed; no TEST; StageII permitted by task',flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=['dryrun','train','audit'])
    action = p.parse_args().action
    install_test_guard('stage1_'+action)
    try:
        {'dryrun':dryrun,'train':train,'audit':audit_completed}[action]()
    except BaseException:
        import traceback
        dump(ROOT/'failure_stage1.json',{'at':now(),'action':action,'traceback':traceback.format_exc()})
        guard_flush()
        raise


if __name__ == '__main__':
    main()
