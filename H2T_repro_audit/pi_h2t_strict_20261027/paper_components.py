"""Minimal PI-H2T formulas from the v2 manuscript, kept separate from training.

This is an independently reconstructed implementation, NOT claimed author code.
PIF: Eq (3)/(6) with [F-mean_channel(F), F], 2-to-1 Conv1d, bias=False.
H2TF: Eq (7)/(8) in literal cosine-distance reading.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PaperPIF(nn.Module):
    """Two-weight linear PIF. torch Conv1d default init is NOT author-verified."""
    def __init__(self):
        super().__init__()
        self.fuse = nn.Conv1d(2, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("PIF expects [B,C,H,W]")
        b,c,h,w=x.shape
        residual=x-x.mean(dim=1,keepdim=True)
        # [B,C,2,H,W] -> [B*C,2,H*W], parameters shared over channels.
        pair=torch.stack((residual,x),dim=2).reshape(b*c,2,h*w)
        out=self.fuse(pair).reshape(b,c,h,w)
        return out


class PIExtractor(nn.Module):
    """Attach PIF AFTER last conv stage, BEFORE global average pooling."""
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.extractor=feature_extractor
        self.pif=PaperPIF()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pif(self.extractor(x))


def pool_feature(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("expected spatial feature maps [B,C,H,W]")
    return F.adaptive_avg_pool2d(x,1).flatten(1)


def receiver_ratio(
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """PI-H2T v2 Eq (8): r=(1-cos(f_y,w_y))/2, RECIPIENT weight.

    The paper has unresolved reference-W / stop-grad details: caller must
    explicitly choose current-vs-stage1 W; caller controls detach.
    """
    if features.ndim != 2 or weights.ndim != 2 or labels.ndim != 1:
        raise ValueError("expected features [B,D], weights [K,D], labels [B]")
    if labels.numel()!=features.shape[0] or weights.shape[1]!=features.shape[1]:
        raise ValueError("feature/label/classifier dimension mismatch")
    if labels.dtype!=torch.long:
        raise TypeError("labels must be torch.long")
    if bool(torch.any(labels<0)) or bool(torch.any(labels>=weights.shape[0])):
        raise ValueError("label outside classifier range")
    cosine=F.cosine_similarity(features,weights[labels],dim=1,eps=1e-8)
    r=((1.0-cosine)/2.0).clamp(0.,1.)
    return r


def h2tf(
    recipient_feature: torch.Tensor,
    donor_feature: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Full-vector weighted mix; preserve recipient label for CE."""
    if recipient_feature.ndim!=2 or recipient_feature.shape!=donor_feature.shape:
        raise ValueError("recipient/donor must have matching [B,D]")
    if r.shape!=(recipient_feature.shape[0],):
        raise ValueError("r must be [B]")
    return r[:,None]*recipient_feature+(1-r[:,None])*donor_feature


def classify_vector(classifier: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Official CIFAR Classifier expects [B,64,H,W], not [B,64]."""
    if x.ndim!=2:
        raise ValueError("feature vector must be [B,D]")
    return classifier(x[:,:,None,None])


def group_stats(pred: torch.Tensor, y: torch.Tensor) -> dict:
    """Official H2T CIFAR group boundaries: [0,36),[36,71),[71,100)."""
    if pred.shape!=y.shape or y.ndim!=1: raise ValueError("expected [N] predictions/labels")
    masks={"Overall":torch.ones_like(y,dtype=torch.bool),
           "Many":y<36,"Medium":(y>=36)&(y<71),"Few":y>=71}
    # Compute percentages from integer counts; a float32 mean unnecessarily
    # rounds exact ratios such as 5/6 before conversion to Python float.
    return {name:(100.*int((pred[m]==y[m]).sum())/int(m.sum()) if bool(m.any()) else None)
            for name,m in masks.items()}
