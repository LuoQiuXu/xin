"""Frozen-LowVS complementary supervision pilot: three fixed losses.

Do not edit any augmentation, channel mask, sampling or backbone.  The add-on
compares tail-only clean CE with a single most-competitive clean-class logit
constraint at identical tail examples, normalisation and fixed coefficient.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

ARMS=("LowVS", "LowVS-CleanCE", "LowVS-TopCompetitor")
LAMBDA=0.5
TAIL_START=71

def loss_terms(arm, fused_logits, clean_logits, targets):
    if arm not in ARMS:raise ValueError(arm)
    assert fused_logits.ndim==clean_logits.ndim==2
    assert fused_logits.shape==clean_logits.shape==(targets.numel(),100)
    assert targets.ndim==1 and targets.dtype==torch.long
    assert torch.all((targets>=0)&(targets<100))
    fused_ce=F.cross_entropy(fused_logits,targets)
    tail=(targets>=TAIL_START)
    count=int(tail.sum().item())
    # Auxiliary losses are summed on precisely the SAME tail subset and divided
    # by full minibatch size; no variable 1/n_tail multiplier and no CE rescaling.
    aux=fused_ce.new_zeros(())
    competition=None
    if arm!="LowVS" and count:
        z=clean_logits[tail]
        y=targets[tail]
        if arm=="LowVS-CleanCE":
            aux=F.cross_entropy(z,y,reduction="sum")/targets.numel()
        else:
            with torch.no_grad():
                other=z.detach().clone()
                other.scatter_(1,y[:,None],-torch.inf)
                competition=other.argmax(dim=1)
                assert bool((competition!=y).all())
            yy=z.gather(1,y[:,None]).squeeze(1)
            jj=z.gather(1,competition[:,None]).squeeze(1)
            # zero target margin: no arbitrary positive threshold or sample gate
            aux=F.softplus(jj-yy).sum()/targets.numel()
    total=fused_ce+LAMBDA*aux if arm!="LowVS" else fused_ce
    assert bool(torch.isfinite(total)) and bool(torch.isfinite(fused_ce))
    assert bool(torch.isfinite(aux))
    return total,{"fused_ce":float(fused_ce.detach()),"clean_tail_aux":float(aux.detach()),
                  "n_tail":count,
                  "n_competitor_head":int((competition<36).sum().item()) if competition is not None else None,
                  "n_competitor_medium":int(((competition>=36)&(competition<71)).sum().item()) if competition is not None else None,
                  "n_competitor_tail":int((competition>=71).sum().item()) if competition is not None else None}

def self_test():
    torch.manual_seed(18018)
    b=17
    y=torch.tensor([0,35,70,71,72,88,99,1,4,64,72,17,85,26,98,37,94],dtype=torch.long)
    f=torch.randn(b,100,requires_grad=True)
    z=torch.randn(b,100,requires_grad=True)
    results={}
    for arm in ARMS:
        loss,diag=loss_terms(arm,f,z,y)
        loss.backward(retain_graph=True)
        assert torch.isfinite(f.grad).all()
        if arm=="LowVS":
            assert torch.equal(z.grad,torch.zeros_like(z.grad)) if z.grad is not None else True
        else:
            assert z.grad is not None and torch.allclose(z.grad[y<71],torch.zeros_like(z.grad[y<71]))
            assert diag["n_tail"]==int((y>=71).sum())
        results[arm]=float(loss)
        f.grad=None;z.grad=None
    assert results["LowVS-CleanCE"]>results["LowVS"]
    assert results["LowVS-TopCompetitor"]>results["LowVS"]
    # No-tail batch must reduce EXACTLY to the original fused CE.
    no_tail=torch.zeros(3,dtype=torch.long)
    a=torch.randn(3,100,requires_grad=True)
    c=torch.randn(3,100,requires_grad=True)
    reference=F.cross_entropy(a,no_tail)
    for arm in ARMS:
        l,d=loss_terms(arm,a,c,no_tail)
        assert torch.equal(l,reference) and d["n_tail"]==0
    # The most competitive class is selected on CLEAN logits, excluding y.
    hard=torch.full((2,100),-5.,requires_grad=True)
    with torch.no_grad():
        hard[0,71]=5;hard[0,5]=4
        hard[1,89]=5;hard[1,11]=4
    yy=torch.tensor([71,89])
    _,d=loss_terms("LowVS-TopCompetitor",a[:2],hard,yy)
    assert d["n_competitor_head"]==2
    print("PASS: tail-only auxiliary gradient; zero-margin top competitor; no-tail identity")

if __name__=="__main__":self_test()
