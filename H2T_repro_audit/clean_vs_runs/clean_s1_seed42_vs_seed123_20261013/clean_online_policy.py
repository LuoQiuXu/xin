"""Three-arm adaptation of the reviewed VS policy; scoring / sampling rules unchanged."""
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,array_hash,ARMS
from reused_vs_math import scores,top_indices,selected_bool

_STAT = None


def stats():
    global _STAT
    if _STAT is None:
        q = np.load(ROOT/'class_feature_statistics.npz',allow_pickle=False)
        _STAT = {k:q[k] for k in q.files}
        for value in _STAT.values():
            value.flags.writeable = False
    return _STAT


def online_masks(receiver,donor_classes,noise):
    assert not receiver.requires_grad and torch.isfinite(receiver).all()
    gap = F.avg_pool2d(receiver,8).view(-1,64).detach().cpu().numpy().astype(np.float64)
    stat = stats()
    h = np.asarray(donor_classes,dtype=np.int64)
    s = scores(gap,h,stat['mean_eval'],stat['variance_eval_regularized'],float(stat['epsilon']))
    reason = np.zeros(len(h),dtype=np.uint8)
    reason[stat['counts'][h] < 20] = 1
    reason[np.ptp(s,axis=1) <= 1e-12] = 2
    reason[~np.isfinite(s).all(1)] = 3
    assert np.isfinite(s).all()
    assert ((s >= 0)&(s <= 1)).all()
    pools = {'Uniform-PerSample':np.ones((len(h),64),dtype=bool),
        'High-VS':selected_bool(top_indices(s,48,True)),
        'Low-VS':selected_bool(top_indices(s,48,False))}
    masks = {}
    for name,pool in pools.items():
        pool[reason > 0] = True
        choice = np.argsort(-np.where(pool,noise,-np.inf),axis=1,kind='stable')[:,:16].astype(np.uint8)
        assert np.all(np.diff(np.sort(choice,axis=1),axis=1) > 0)
        assert np.take_along_axis(pool,choice,axis=1).all()
        masks[name] = choice
    for name in ['High-VS','Low-VS']:
        assert np.array_equal(masks[name][reason > 0],masks['Uniform-PerSample'][reason > 0])
    return {'GAP':gap,'scores':s,'pools':pools,'masks':masks,'fallback':reason,'donor_classes':h,
        'GAP_sha256':array_hash(gap),'score_sha256':array_hash(s),
        'all_three_masks_sha256':array_hash(*[masks[k] for k in ARMS])}


def selected_statistics(trace,chosen):
    stat = stats()
    h = trace['donor_classes']
    bias = np.square(stat['mean_eval'][h]-trace['GAP'])
    V = np.take_along_axis(stat['variance_eval_regularized'][h],chosen,axis=1).sum(1)
    B = np.take_along_axis(bias,chosen,axis=1).sum(1)
    common = (selected_bool(chosen)&selected_bool(trace['masks']['Uniform-PerSample'])).sum(1)
    return {'V':V,'B':B,'ratio':V/(V+B+float(stat['epsilon'])),
        'score':np.take_along_axis(trace['scores'],chosen,axis=1).mean(1),
        'common_with_uniform':common,'fallback':trace['fallback']}
