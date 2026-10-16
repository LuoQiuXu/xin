"""First-order train-only gradient-utility routing for fixed Low-VS channels.

No tuning on validation. This is a *proxy*: positive alignment approximates
a local improvement of a class-balanced CLEAN-TRAIN risk for plain SGD, not
a guarantee about unseen-data utility. Frozen ResNet32; linear 100-class head.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

ALPHAS = (0.0, 0.5, 1.0)
ARMS = ("LowVS-Full", "LowVS-Half", "GU-LowVS", "GU-Shuffled")


@torch.no_grad()
def reference_gradient(clean_train_gap: torch.Tensor, labels: torch.Tensor,
                       classifier: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact gradient of clean, class-balanced empirical CE over TRAIN features.

    Every of 100 classes contributes 1/100. Includes trained classifier bias;
    excludes validation/TEST and L2 weight-decay/momentum from the proxy.
    """
    fc = classifier.module.fc if hasattr(classifier, "module") else classifier.fc
    x = clean_train_gap.to(fc.weight.device, dtype=fc.weight.dtype)
    y = labels.to(device=x.device, dtype=torch.long)
    assert x.ndim == 2 and x.shape[1] == 64 and y.shape == (len(x),)
    assert torch.isfinite(x).all() and ((y >= 0) & (y < 100)).all()
    n = torch.bincount(y, minlength=100)
    assert n.min().item() > 0
    w = (1.0 / (100.0 * n[y].to(x.dtype))).unsqueeze(1)
    e = F.softmax(F.linear(x, fc.weight.detach(), fc.bias.detach()), dim=1)
    e[torch.arange(len(x), device=x.device), y] -= 1.0
    e *= w
    gW = e.T @ x
    gb = e.sum(0)
    assert gW.shape == (100, 64) and gb.shape == (100,)
    assert torch.isfinite(gW).all() and torch.isfinite(gb).all()
    return gW, gb


@torch.no_grad()
def per_sample_alignment(gap: torch.Tensor, labels: torch.Tensor,
                         classifier: torch.nn.Module, gW: torch.Tensor,
                         gb: torch.Tensor) -> torch.Tensor:
    """Euclidean <reference_gradient, CE gradient from one sample>.

    For the linear classifier, sample gradient is (p - onehot(y)) outer x,
    plus bias gradient (p-onehot(y)). This avoids autograd per candidate.
    """
    fc = classifier.module.fc if hasattr(classifier, "module") else classifier.fc
    x = gap.to(device=fc.weight.device, dtype=fc.weight.dtype)
    y = labels.to(x.device, dtype=torch.long)
    r = F.softmax(F.linear(x, fc.weight.detach(), fc.bias.detach()), dim=1)
    r[torch.arange(len(x), device=x.device), y] -= 1.0
    score = ((r @ gW)*x).sum(1) + (r @ gb)
    assert torch.isfinite(score).all()
    return score


@torch.no_grad()
def utility_actions(receiver: torch.Tensor, full: torch.Tensor,
                    labels: torch.Tensor, classifier: torch.nn.Module,
                    gW: torch.Tensor, gb: torch.Tensor):
    """Per-receiver argmax across clean / half / full on train-only proxy.

    Returns integer 0,1,2 corresponding to alpha 0,.5,1. A separate
    shuffled-action control preserves the exact action histogram in a batch.
    """
    assert receiver.shape == full.shape and receiver.ndim == 4
    def gap(t):return F.avg_pool2d(t,t.shape[-1]).flatten(1)
    candidates=(receiver, 0.5*(receiver+full), full)
    scores=torch.stack([per_sample_alignment(gap(t),labels,classifier,gW,gb)
                        for t in candidates],dim=1)
    selected=scores.argmax(1).to(torch.uint8)
    assert selected.shape == (len(receiver),)
    return selected,scores


@torch.no_grad()
def apply_actions(receiver: torch.Tensor, full: torch.Tensor, actions: torch.Tensor):
    """Preserves exact endpoint tensors for 0 and 1; alpha=.5 mean between."""
    a=actions.to(receiver.device,dtype=torch.long)
    assert receiver.shape == full.shape and a.shape==(len(receiver),)
    assert ((a>=0)&(a<=2)).all()
    half=0.5*(receiver+full)
    out=torch.where((a==0)[:,None,None,None],receiver,
          torch.where((a==1)[:,None,None,None],half,full))
    assert torch.isfinite(out).all()
    return out


def test_policy() -> None:
    """CPU test: analytic balanced reference gradient matches autograd."""
    torch.manual_seed(20261016)
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc=torch.nn.Linear(64,100)
    head=Head()
    x=torch.randn(230,64)
    labels=(torch.arange(len(x))%100).long()
    y=labels.clone()
    n=torch.bincount(y,minlength=100)
    weights=1.0/(100*n[y].float())
    loss=(F.cross_entropy(head.fc(x),y,reduction="none")*weights).sum()
    aW,ab=torch.autograd.grad(loss, (head.fc.weight,head.fc.bias))
    gW,gb=reference_gradient(x,y,head)
    assert torch.allclose(aW,gW,atol=2e-6,rtol=2e-5)
    assert torch.allclose(ab,gb,atol=2e-6,rtol=2e-5)
    vals=per_sample_alignment(x[:5],y[:5],head,gW,gb)
    for i in range(5):
        li=F.cross_entropy(head.fc(x[i:i+1]),y[i:i+1])
        w,b=torch.autograd.grad(li,(head.fc.weight,head.fc.bias))
        exact=(w*gW).sum()+(b*gb).sum()
        assert torch.allclose(vals[i],exact,atol=1e-6,rtol=2e-5)
    recv=torch.randn(8,64,8,8);full=torch.randn_like(recv)
    actions,scores=utility_actions(recv,full,labels[:8],head,gW,gb)
    assert actions.shape==(8,) and scores.shape==(8,3)
    for k in range(3):
        z=apply_actions(recv,full,torch.full((8,),k))
        expected=[recv,0.5*(recv+full),full][k]
        assert torch.equal(z,expected)
    print("GU-LowVS CPU PASS: exact gradients, 3 actions, no learnable gate")
