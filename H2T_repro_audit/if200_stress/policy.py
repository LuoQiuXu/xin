"""Fixed, paired H2T masks for an IF=200 stress test.  No learned gates."""
from __future__ import annotations
import numpy as np
import torch

ARMS=("DR-NoFusion","Original-BatchShared","Uniform-PerSample","LowVS-PerSample")
EPS=1e-12

def masks(receiver_gap, donor_labels, counts, means, variances, epsilon,
          sample_noise, shared_noise):
    """All masks are derived on the SAME receiver/donor batch, independent of arm."""
    z=np.asarray(receiver_gap,dtype=np.float64)
    h=np.asarray(donor_labels,dtype=np.int64)
    g=np.asarray(sample_noise,dtype=np.float64)
    sh=np.asarray(shared_noise,dtype=np.float64)
    n=len(z)
    assert z.shape==(n,64) and g.shape==(n,64) and sh.shape==(64,)
    assert means.shape==variances.shape==(100,64) and len(counts)==100
    assert np.isfinite(z).all() and np.isfinite(g).all() and np.isfinite(sh).all()
    B=(means[h]-z)**2
    S=variances[h]/(variances[h]+B+float(epsilon))
    assert np.isfinite(S).all() and np.all((S>=0)&(S<=1))
    fallback=(counts[h]<20)|(np.ptp(S,axis=1)<=1e-12)
    low_pool=np.zeros((n,64),dtype=bool)
    low_pool[np.arange(n)[:,None],np.argsort(S,axis=1,kind="stable")[:,:48]]=True
    low_pool[fallback]=True
    full_pool=np.ones((n,64),dtype=bool)
    U=np.argsort(-np.where(full_pool,g,-np.inf),axis=1,kind="stable")[:,:16]
    L=np.argsort(-np.where(low_pool,g,-np.inf),axis=1,kind="stable")[:,:16]
    shared=np.broadcast_to(np.argsort(-sh,kind="stable")[:16],(n,16)).copy()
    assert np.array_equal(U[fallback],L[fallback])
    assert all(np.all(np.diff(np.sort(x,axis=1),axis=1)>0) for x in (U,L,shared))
    assert np.take_along_axis(low_pool,L,axis=1).all()
    return {"Original-BatchShared":shared.astype(np.uint8),
            "Uniform-PerSample":U.astype(np.uint8),
            "LowVS-PerSample":L.astype(np.uint8)}, {
            "fallback_count":int(fallback.sum()),
            "mean_original_VS":float(S.mean()),
            "mean_lowVS_selected_VS":float(np.take_along_axis(S,L,axis=1).mean()),
            "mean_uniform_selected_VS":float(np.take_along_axis(S,U,axis=1).mean()),
            "per_sample_uniform_selected_VS":np.take_along_axis(S,U,axis=1).mean(1),
            "per_sample_lowVS_selected_VS":np.take_along_axis(S,L,axis=1).mean(1),
            "mean_U_L_mask_overlap":float(np.mean([np.intersect1d(a,b).size for a,b in zip(U,L)]))
            }

@torch.no_grad()
def fuse(receiver,donor,indices):
    assert receiver.shape==donor.shape and receiver.shape[1:]==(64,8,8)
    sel=torch.as_tensor(np.asarray(indices,dtype=np.int64),device=receiver.device)
    mask=torch.zeros((len(receiver),64),dtype=torch.bool,device=receiver.device)
    mask.scatter_(1,sel,True)
    assert (mask.sum(1)==16).all()
    return torch.where(mask[...,None,None],donor,receiver)

def self_test():
    r=np.random.default_rng(20261017)
    n=19;z=r.normal(size=(n,64));m=r.normal(size=(100,64));v=np.abs(r.normal(size=(100,64)))+.01
    cnt=np.arange(2,102)
    d=np.r_[np.repeat(0,4),np.repeat(1,4),np.repeat(50,11)]
    g=r.gumbel(size=(n,64));sh=r.gumbel(size=64)
    masks_a,diag=masks(z,d,cnt,m,v,1e-6,g,sh)
    masks_b,_=masks(z,d,cnt,m,v,1e-6,g,sh)
    assert all(np.array_equal(masks_a[k],masks_b[k]) for k in masks_a)
    assert (masks_a["Original-BatchShared"]==masks_a["Original-BatchShared"][0]).all()
    assert np.array_equal(masks_a["Uniform-PerSample"][:8],masks_a["LowVS-PerSample"][:8])
    assert diag["fallback_count"]>=8
    x=torch.randn(n,64,8,8);h=torch.randn_like(x)
    y=fuse(x,h,masks_a["LowVS-PerSample"])
    sel=torch.zeros((n,64),dtype=torch.bool)
    sel.scatter_(1,torch.from_numpy(masks_a["LowVS-PerSample"].astype(np.int64)),True)
    assert torch.equal(y[sel[...,None,None].expand_as(x)],h[sel[...,None,None].expand_as(x)])
    assert torch.equal(y[~sel[...,None,None].expand_as(x)],x[~sel[...,None,None].expand_as(x)])
    print("PASS: shared/per-sample/low-VS masks, donor fallback, exact full-channel replacement")

if __name__=="__main__":self_test()
