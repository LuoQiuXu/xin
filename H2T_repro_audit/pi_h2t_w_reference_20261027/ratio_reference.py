"""Single-factor PI-H2T Stage-II classifier-reference parity controls.

f_r and w_y determine r=(1-cos)/2, unchanged from Eq (8).
Two meanings of W differ: frozen Stage-I vs live Stage-II classifier.
This is a diagnostic for an UNVERIFIED author implementation detail, not
a newly proposed H2T fusion method.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F

def cosine_receiver_ratio(features: torch.Tensor, labels: torch.Tensor,
                          weights: torch.Tensor) -> torch.Tensor:
    """Return RECIPIENT retention r, with no gradient through W or ratio."""
    if features.ndim!=2 or labels.ndim!=1 or weights.ndim!=2:
        raise ValueError("Expected feature[B,D], labels[B], weights[K,D]")
    if len(features)!=len(labels) or weights.shape[1]!=features.shape[1]:
        raise ValueError("Inconsistent feature/label/weights dimensions")
    if labels.dtype!=torch.long:raise TypeError("long labels required")
    if torch.any(labels<0) or torch.any(labels>=len(weights)):
        raise ValueError("label outside weight matrix")
    with torch.no_grad():
        cosine=F.cosine_similarity(features.detach(),weights.detach()[labels],dim=1,eps=1e-8)
        return ((1.-cosine)/2.).clamp(0.,1.)

def select_weights(initial_stage1_weights: torch.Tensor,
                   live_head_weights: torch.Tensor,
                   reference: str) -> torch.Tensor:
    """Select W WITHOUT changing fusion, gradient, optimizer or sampling."""
    if reference=="frozen_stage1":
        return initial_stage1_weights.detach()
    if reference=="current_stage2":
        return live_head_weights.detach()
    raise ValueError(f"Unknown W source {reference}")

def fuse_features(receiver: torch.Tensor, donor: torch.Tensor,
                  r: torch.Tensor) -> torch.Tensor:
    if receiver.ndim!=2 or donor.shape!=receiver.shape or r.shape!=(len(receiver),):
        raise ValueError("Bad feature/ratio shapes")
    return r[:,None]*receiver+(1-r[:,None])*donor
