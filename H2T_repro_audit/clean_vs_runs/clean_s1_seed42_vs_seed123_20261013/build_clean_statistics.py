"""Compute the unchanged VS primary statistics from the Clean best LT features."""
import argparse
import shutil
from common import *
from reused_vs_math import regularize_variance


def frozen_models():
    protocol = read(ROOT/'clean_stage1_protocol.json')
    assert protocol['passed'] and not protocol['TEST_loaded'] and not protocol['TEST_selected']
    cfg,tr,ResNet,Classifier,CE,methods = imports(2)
    seed_all(S2_SEED,False)
    p = Path(protocol['best_checkpoint'])
    assert p.resolve().is_relative_to(ROOT) and sha(p) == protocol['best_checkpoint_sha256']
    q = torch.load(p,map_location='cpu',weights_only=False)
    assert q['seed'] == 42 and q['selection_partition'] == 'official TRAIN holdout' and not q['TEST_used']
    bb,cls = torch.nn.DataParallel(ResNet()).cuda(),torch.nn.DataParallel(Classifier(64,100)).cuda()
    bb.load_state_dict(q['state_dict_model'],strict=True)
    cls.load_state_dict(q['state_dict_classifier'],strict=True)
    bb.eval().requires_grad_(False)
    cls.train()
    assert not bb.module.feature_tuning and not hasattr(bb.module,'conv1d')
    return cfg,tr,bb,cls,CE,methods,p,q


def extract(backbone,images,transform,path):
    shape = (len(images),64,8,8)
    bank = np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape)
    gap = np.empty((len(images),64),dtype=np.float64)
    with torch.no_grad():
        for start in range(0,len(images),BATCH):
            x = torch.stack([transform(Image.fromarray(v)) for v in images[start:start+BATCH]]).cuda()
            features = backbone(x)
            assert features.shape == (len(x),64,8,8) and torch.isfinite(features).all()
            bank[start:start+len(x)] = features.cpu().numpy()
            gap[start:start+len(x)] = torch.nn.functional.avg_pool2d(features,8).view(-1,64).cpu().numpy().astype(np.float64)
    bank.flush()
    return gap


def main():
    install_test_guard('clean_statistics')
    assert not (ROOT/'class_feature_statistics.npz').exists(), 'Frozen new statistics cannot be overwritten'
    check_sources()
    cfg,tr,bb,cls,CE,methods,p,q = frozen_models()
    initial = {'backbone':state_hash(bb),'classifier':state_hash(cls)}
    bb_state = {k:v.detach().cpu().clone() for k,v in bb.state_dict().items()}
    images = np.load(ROOT/'train_images.npy',mmap_mode='r',allow_pickle=False)
    labels = np.load(ROOT/'train_labels.npy',allow_pickle=False)
    validation_images = np.load(ROOT/'validation_images.npy',mmap_mode='r',allow_pickle=False)
    labels_val = np.load(ROOT/'validation_labels.npy',allow_pickle=False)
    bank_dir = ROOT/'feature_banks'
    bank_dir.mkdir()
    t = time.perf_counter()
    gap = extract(bb,images,tr['eval_transform'],bank_dir/'LT_eval_feature_maps.npy')
    np.save(bank_dir/'LT_eval_GAP_float64.npy',gap)
    extract(bb,validation_images,tr['eval_transform'],ROOT/'validation_feature_maps.npy')
    means,var0 = np.empty((100,64),dtype=np.float64),np.empty((100,64),dtype=np.float64)
    counts = np.bincount(labels,minlength=100)
    for c in range(100):
        x = gap[labels == c]
        means[c] = x.mean(0,dtype=np.float64)
        var0[c] = np.square(x-means[c]).mean(0,dtype=np.float64)
    var1 = var0*(counts/(counts-1))[:,None]
    pooled = (var0*counts[:,None]).sum(0)/(N-100)
    vreg = regularize_variance(var0,counts,pooled)
    scale = float(np.var(gap,axis=0,ddof=1).mean())
    eps = 1e-6*scale
    assert scale > 0 and np.isfinite(gap).all() and np.isfinite(vreg).all()
    np.savez_compressed(ROOT/'class_feature_statistics.npz',counts=counts,mean_eval=means,
        variance_eval_ddof0=var0,variance_eval_ddof1=var1,variance_eval_regularized=vreg,
        pooled_within_variance=pooled,shrinkage_lambda=5./(counts+5),
        epsilon=eps,global_variance_scale=scale)
    assert state_hash(bb) == initial['backbone'] and state_hash(cls) == initial['classifier']
    check_frozen(bb,bb_state)
    # Pool / classifier and map extraction correctness using TRAIN holdout only.
    probe = torch.from_numpy(np.load(ROOT/'validation_feature_maps.npy',mmap_mode='r')[:BATCH].copy()).cuda()
    with torch.no_grad():
        assert torch.equal(cls(probe),cls.module.fc(torch.nn.functional.avg_pool2d(probe,8).view(-1,64)))
    cached,logits,pred = evaluate(cls,labels_val,feature_maps=np.load(ROOT/'validation_feature_maps.npy',mmap_mode='r'))
    assert {k:v for k,v in cached.items() if k != 'validation_CE'} == {
        k:v for k,v in q['validation'].items() if k != 'validation_CE'}, 'Clean selected classification metrics changed'
    assert abs(cached['validation_CE']-q['validation']['validation_CE']) < 1e-5
    recorded = np.load(ROOT/'stage1'/f'validation_epoch{q["epoch"]:03d}.npz',allow_pickle=False)
    assert np.array_equal(pred,recorded['predictions'])
    provenance = {'created_at':now(),'Clean_StageI_checkpoint':str(p),'Clean_StageI_sha256':sha(p),
        'selected_epoch':q['epoch'],'source_partition':'same10847 LT TRAIN images only',
        'train_ID_sha256':read(ROOT/'preflight.json')['train_raw_ID_sha256'],
        'feature_shape':list(gap.shape),'class_counts':counts.tolist(),
        'mean_variance_rule':'exact existing mean/delta-square ddof0,ddof1 saved separately',
        'shrinkage_rule':'lambda5/(n+5); pooled sum(n*var0)/(N-100)',
        'epsilon_rule':'1e-6 * mean(global LT variance ddof1)','epsilon':eps,'global_variance_scale':scale,
        'statistics_sha256':sha(ROOT/'class_feature_statistics.npz'),
        'old_statistics_unchanged_sha256':sha(OLD/'class_feature_statistics.npz'),
        'old_statistics_loaded':False,'validation_used_in_class_statistics':False,'TEST_used':False,
        'validation_maps_sha256':sha(ROOT/'validation_feature_maps.npy'),'Clean_best_cached_validation':cached,
        'pool_fc_exact':True,'backbone_BN_classifier_unchanged':True,
        'selected_validation_predictions_exact':True,
        'cached_vs_StageI_validation_CE_abs_difference':abs(cached['validation_CE']-q['validation']['validation_CE']),
        'cached_vs_StageI_logits_max_abs_difference':float(np.max(np.abs(logits-recorded['logits']))),
        'extraction_seconds':time.perf_counter()-t}
    dump(ROOT/'clean_statistics_provenance.json',provenance)
    guard_pass()
    check_history()
    print(f'CLEAN STATISTICS PASS:LT={gap.shape},epsilon={eps:.12g}; class stats use LT only; validation cache exact',flush=True)


if __name__ == '__main__':
    main()
