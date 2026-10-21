"""Paper-form PIF and H2TF, independent of historical PI-H2T experiments.

PIF: Eqs (3),(4),(6) in the supplied PI-H2T manuscript.
H2TF: Eqs (7),(8). Authors' description of cosine *distance* conflicts
with Figure 9's reported r (~0.65 at positive feature-weight cosine).
This implementation makes BOTH mappings explicit instead of silently
choosing one after looking at benchmark accuracy.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "H2T"))
from models.resnet_cifar import resnet32_fe, Classifier  # noqa: E402


class PaperPIF(nn.Module):
    """Exact 2-weight affine fusion, no bias: a*(F-channel_mean(F)) + b*F.

    A bias-free Conv1d(2,1,1) is algebraically equivalent; a & b are
    explicitly exposed for formula checks and useful initialization.
    The paper does not prescribe an initializer. We declare (a,b)=(0,1)
    to start with identity features and avoid attributing this choice to
    the authors.
    """
    def __init__(self, a: float = 0.0, b: float = 1.0):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([a, b], dtype=torch.float32))

    @staticmethod
    def invariant_component(feature_maps: torch.Tensor) -> torch.Tensor:
        if feature_maps.ndim != 4:
            raise ValueError("PIF expects [B,C,H,W]")
        return feature_maps.mean(dim=1, keepdim=True)

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        mean = self.invariant_component(feature_maps)
        a, b = self.coefficients.unbind()
        return a * (feature_maps - mean) + b * feature_maps


class PIResNet32(nn.Module):
    def __init__(self, *, enabled: bool = True):
        super().__init__()
        self.backbone = resnet32_fe()
        self.pif = PaperPIF()
        self.pif_enabled = enabled

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        z = self.backbone(image)
        return self.pif(z) if self.pif_enabled else z


def classifier():
    return Classifier(64, 100)


def pool(feature_maps: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(feature_maps, 1).flatten(1)


def ratios(recipient: torch.Tensor, y: torch.Tensor,
           stage1_classifier_weights: torch.Tensor, *, mode: str) -> torch.Tensor:
    """Receiver retention r; mode declared before accuracy is inspected.

    distance_literal: r=(1-cos)/2, literal Eq.8 if cosine distance is 1-cos.
    figure_consistent: r=(1+cos)/2, consistent with paper Figure9 (r~.65)
       for the earlier observed cosine~+.30; this is NOT the literal
       cosine-distance reading. Both remain labelled in all artifacts.
    """
    if recipient.ndim != 2 or stage1_classifier_weights.ndim != 2:
        raise ValueError("Expected [B,D] features and [100,D] classifier weights")
    if stage1_classifier_weights.shape[0] != 100 or recipient.shape[1] != stage1_classifier_weights.shape[1]:
        raise ValueError("Classifier/feature dimension mismatch")
    cosine = F.cosine_similarity(recipient, stage1_classifier_weights[y], dim=1, eps=1e-8)
    if mode == "distance_literal":
        r = (1.0 - cosine) / 2.0
    elif mode == "figure_consistent":
        r = (1.0 + cosine) / 2.0
    else:
        raise ValueError(f"Undeclared ratio mode: {mode}")
    return r.clamp(0.0,1.0)


def h2tf(recipient: torch.Tensor, donor: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    if recipient.shape != donor.shape or r.shape != (recipient.shape[0],):
        raise ValueError("Invalid H2TF shapes")
    return r[:,None]*recipient + (1.0-r)[:,None]*donor


def group_metrics(pred: torch.Tensor, y: torch.Tensor) -> dict:
    if pred.shape != y.shape or pred.ndim != 1:
        raise ValueError("Expected flat prediction/target tensors")
    acc = {}
    for key, mask in {
        "overall": torch.ones_like(y,dtype=torch.bool),
        "head": y < 36,
        "medium": (y>=36)&(y<71),
        "tail": y>=71
    }.items():
        acc[key] = 100.0*float((pred[mask]==y[mask]).float().mean()) if mask.any() else None
    return acc
