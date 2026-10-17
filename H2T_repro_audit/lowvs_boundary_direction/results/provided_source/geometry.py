"""Boundary geometry of a PAIR of frozen-backbone GAP feature perturbations.

No fitting, no threshold search, no use of validation/TEST and no learned gate.
The competitor is chosen exactly ONCE on the unperturbed receiver using the
locked Clean Stage-I classifier (W,b), and never recomputed for interventions.
"""
from __future__ import annotations
import numpy as np

EPS=1e-12

def _loss(logits, labels):
    m=np.max(logits,axis=1)
    return m+np.log(np.exp(logits-m[:,None]).sum(1))-logits[np.arange(len(labels)),labels]

def probe(zr,zh,y,h,uniform,lowvs,W,b,fallback):
    zr=np.asarray(zr,dtype=np.float64);zh=np.asarray(zh,dtype=np.float64)
    y=np.asarray(y,dtype=np.int64);h=np.asarray(h,dtype=np.int64)
    uniform=np.asarray(uniform,dtype=np.int64);lowvs=np.asarray(lowvs,dtype=np.int64)
    W=np.asarray(W,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    fallback=np.asarray(fallback,dtype=np.uint8)
    n=len(y)
    assert zr.shape==zh.shape==(n,64) and uniform.shape==lowvs.shape==(n,16)
    assert W.shape==(100,64) and b.shape==(100,) and h.shape==fallback.shape==(n,)
    assert (y>=0).all() and (y<100).all() and (h>=0).all() and (h<100).all()
    assert np.isfinite(zr).all() and np.isfinite(zh).all() and np.isfinite(W).all()
    assert (uniform>=0).all() and (uniform<64).all() and (lowvs>=0).all() and (lowvs<64).all()
    assert np.all(np.diff(np.sort(uniform,axis=1),axis=1)>0)
    assert np.all(np.diff(np.sort(lowvs,axis=1),axis=1)>0)
    equal_mask=np.array([np.array_equal(np.sort(a),np.sort(b))
                         for a,b in zip(uniform,lowvs)],dtype=bool)
    assert equal_mask[fallback>0].all(), "Historic fallback must be uniform"
    D=zh-zr
    rows=np.arange(n)[:,None]
    U=np.zeros((n,64),dtype=np.float64);L=np.zeros_like(U)
    U[rows,uniform]=D[rows,uniform]
    L[rows,lowvs]=D[rows,lowvs]
    nu=np.linalg.norm(U,axis=1);nl=np.linalg.norm(L,axis=1)
    r=np.minimum(nu,nl)
    alpha_u=np.divide(r,nu,out=np.zeros(n),where=nu>EPS)
    alpha_l=np.divide(r,nl,out=np.zeros(n),where=nl>EPS)
    um=U*alpha_u[:,None];lm=L*alpha_l[:,None]
    assert np.allclose(np.linalg.norm(um,axis=1),np.linalg.norm(lm,axis=1),atol=1e-9,rtol=1e-9)
    assert np.allclose(np.linalg.norm(um,axis=1),r,atol=1e-9,rtol=1e-9)
    logits=zr@W.T+b
    comp_logits=logits.copy();comp_logits[np.arange(n),y]=-np.inf
    j=comp_logits.argmax(1)
    clean_margin=logits[np.arange(n),y]-logits[np.arange(n),j]
    wyj=W[y]-W[j]
    wh=W[y]-W[h]
    nyj=np.linalg.norm(wyj,axis=1)
    nh=np.linalg.norm(wh,axis=1)
    q_u=np.einsum("ij,ij->i",wyj,U)/np.maximum(nu,EPS)
    q_l=np.einsum("ij,ij->i",wyj,L)/np.maximum(nl,EPS)
    cos_u=q_u/np.maximum(nyj,EPS)
    cos_l=q_l/np.maximum(nyj,EPS)
    q_uh=np.einsum("ij,ij->i",wh,U)/np.maximum(nu,EPS)
    q_lh=np.einsum("ij,ij->i",wh,L)/np.maximum(nl,EPS)
    q_uh[h==y]=np.nan;q_lh[h==y]=np.nan
    assert (np.abs(cos_u)<=1+1e-10).all() and (np.abs(cos_l)<=1+1e-10).all()
    full_d_u=nu*q_u;full_d_l=nl*q_l
    matched_d_u=r*q_u;matched_d_l=r*q_l
    direction_part=matched_d_l-matched_d_u
    extra_magnitude_part=(nl-r)*q_l-(nu-r)*q_u
    assert np.allclose(full_d_l-full_d_u,direction_part+extra_magnitude_part,atol=1e-9,rtol=1e-9)
    # Diagnostic only: instantaneous frozen Stage-I classifier, not Stage-II improvement.
    loss_clean=_loss(logits,y)
    def effects(delta):
        out=logits+delta@W.T
        return _loss(out,y)-loss_clean,(out.argmax(1)==y),out.argmax(1)
    lu,cu,pu=effects(U);ll,cl,pl=effects(L)
    lmu,cmu,pmu=effects(um);lml,cml,pml=effects(lm)
    return {
        "receiver_class":y,"donor_class":h,"competitor_class":j,
        "fallback":fallback,"same_mask":equal_mask,
        "same_label":h==y,"donor_is_competitor":h==j,
        "clean_margin":clean_margin,"clean_correct":logits.argmax(1)==y,
        "nu":nu,"nl":nl,"r":r,"alpha_u":alpha_u,"alpha_l":alpha_l,
        "q_u":q_u,"q_l":q_l,"cos_u":cos_u,"cos_l":cos_l,
        "q_uh":q_uh,"q_lh":q_lh,
        "full_margin_delta_u":full_d_u,"full_margin_delta_l":full_d_l,
        "matched_margin_delta_u":matched_d_u,"matched_margin_delta_l":matched_d_l,
        "margin_diff_direction_at_r":direction_part,
        "margin_diff_extra_magnitude":extra_magnitude_part,
        "loss_delta_u":lu,"loss_delta_l":ll,
        "loss_delta_um":lmu,"loss_delta_lm":lml,
        "correct_full_u":cu,"correct_full_l":cl,
        "correct_matched_u":cmu,"correct_matched_l":cml,
        "prediction_change_full_u":pu!=logits.argmax(1),
        "prediction_change_full_l":pl!=logits.argmax(1),
        "prediction_change_matched_u":pmu!=logits.argmax(1),
        "prediction_change_matched_l":pml!=logits.argmax(1),
        "original_margin_cross_u":(clean_margin>0)&(clean_margin+full_d_u<0),
        "original_margin_cross_l":(clean_margin>0)&(clean_margin+full_d_l<0),
        "matched_margin_cross_u":(clean_margin>0)&(clean_margin+matched_d_u<0),
        "matched_margin_cross_l":(clean_margin>0)&(clean_margin+matched_d_l<0),
    }

def self_test():
    g=np.random.default_rng(17017); n=9
    x=g.normal(size=(n,64));h=g.normal(size=x.shape)
    W=g.normal(size=(100,64));b=g.normal(size=100)
    ys=np.arange(n)%100;donor=(ys+7)%100
    u=np.stack([g.choice(64,16,replace=False) for _ in range(n)])
    l=np.stack([g.choice(64,16,replace=False) for _ in range(n)])
    l[-1]=u[-1]
    fallback=np.zeros(n,dtype=np.uint8);fallback[-1]=1
    z=probe(x,h,ys,donor,u,l,W,b,fallback)
    assert len(z["q_u"])==n
    assert np.all(z["nu"]>=z["r"]) and np.all(z["nl"]>=z["r"])
    assert np.allclose(z["matched_margin_delta_l"]-z["matched_margin_delta_u"],
                       z["margin_diff_direction_at_r"],atol=1e-10)
    assert np.allclose(z["full_margin_delta_l"]-z["full_margin_delta_u"],
        z["margin_diff_direction_at_r"]+z["margin_diff_extra_magnitude"],atol=1e-10)
    assert np.isclose(z["q_u"][-1],z["q_l"][-1])
    assert z["correct_matched_l"].shape==(n,)
    # Same mask must mean identical action, including every border/margin diagnostic.
    assert np.isclose(z["full_margin_delta_u"][-1],z["full_margin_delta_l"][-1])
    x2=x.copy();h2=x.copy();z2=probe(x2,h2,ys,donor,u,l,W,b,fallback)
    assert np.all(z2["nu"]==0) and np.all(z2["nl"]==0)
    assert np.isfinite(z2["q_u"]).all() and np.isfinite(z2["q_l"]).all()
    print("PASS: frozen boundary margins, donor labels, zero norms, fallback and exact decomposition")

if __name__=="__main__":
    self_test()
