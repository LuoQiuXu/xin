"""Small, testable target-policy primitives for the Low-VS target-semantics pilot.

This module is deliberately independent of dataset loading and model training.
The balanced cRT reference is used only to identify a first-crossed competitor;
it must not supply the Stage-II prediction logits or gradients.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

TargetMode = Literal["hard", "donor_mix", "first_crossed_competitor"]


@torch.no_grad()
def first_crossed_competitor(
    clean_reference_logits: torch.Tensor,
    fused_reference_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the first class boundary crossed along clean-to-fused logits.

    The reference head must be a frozen linear classifier over GAP features.
    Since GAP features change linearly along the clean-to-fused segment, logits
    also change linearly. For each non-target class j, let m0=z_y(clean)-z_j(clean)
    and m1=z_y(fused)-z_j(fused). A crossing exists when m0 > eps and m1 <= 0,
    with alpha_j = m0 / (m0 - m1).

    Returns:
        competitor_ids: [B] long tensor; -1 means no valid crossing.
        crossing_alpha: [B] float64 tensor; -1 means no valid crossing.
            Ties are resolved by the smallest class ID.

    Rows whose clean reference does not strictly classify the true label are
    treated as having no valid crossing. This is a geometry descriptor, not a
    claim that the reference's competitor is a correct semantic target.
    """
    if clean_reference_logits.ndim != 2:
        raise ValueError("clean_reference_logits must have shape [B, C]")
    if fused_reference_logits.shape != clean_reference_logits.shape:
        raise ValueError("clean and fused reference logits must have identical shape")
    if clean_reference_logits.shape[1] < 2:
        raise ValueError("at least two classes are required")
    if labels.ndim != 1 or labels.shape[0] != clean_reference_logits.shape[0]:
        raise ValueError("labels must have shape [B]")
    if not clean_reference_logits.is_floating_point() or not fused_reference_logits.is_floating_point():
        raise ValueError("reference logits must be floating-point tensors")
    if clean_reference_logits.device != fused_reference_logits.device:
        raise ValueError("clean and fused logits must be on the same device")
    if labels.device != clean_reference_logits.device:
        raise ValueError("labels and reference logits must be on the same device")
    if not torch.isfinite(clean_reference_logits).all().item():
        raise ValueError("clean reference logits contain non-finite values")
    if not torch.isfinite(fused_reference_logits).all().item():
        raise ValueError("fused reference logits contain non-finite values")
    if not torch.isfinite(labels.to(torch.float32)).all().item():
        raise ValueError("labels contain non-finite values")

    batch_size, n_classes = clean_reference_logits.shape
    y = labels.to(dtype=torch.long)
    if torch.any((y < 0) | (y >= n_classes)).item():
        raise ValueError("labels contain an out-of-range class ID")
    if eps <= 0:
        raise ValueError("eps must be positive")

    # Use float64 for crossing arithmetic while preserving the supplied logits
    # exactly as represented by their input tensors.
    clean = clean_reference_logits.to(dtype=torch.float64)
    fused = fused_reference_logits.to(dtype=torch.float64)
    rows = torch.arange(batch_size, device=clean.device)
    clean_y = clean[rows, y].unsqueeze(1)
    fused_y = fused[rows, y].unsqueeze(1)
    clean_margin = clean_y - clean
    fused_margin = fused_y - fused

    classes = torch.arange(n_classes, device=clean.device).unsqueeze(0)
    non_target = classes != y.unsqueeze(1)
    strictly_positive_clean = non_target & (clean_margin > eps)
    clean_strictly_correct = strictly_positive_clean.sum(dim=1) == (n_classes - 1)

    crossing = strictly_positive_clean & (fused_margin <= 0)
    denominator = clean_margin - fused_margin
    crossing = crossing & (denominator > eps)

    alpha = clean_margin / denominator.clamp_min(eps)
    alpha = torch.where(crossing, alpha, torch.full_like(alpha, float("inf")))
    first_alpha, competitor = alpha.min(dim=1)
    found = clean_strictly_correct & torch.isfinite(first_alpha) & (first_alpha <= 1.0 + 1e-12)

    competitor = torch.where(found, competitor, torch.full_like(competitor, -1))
    first_alpha = torch.where(found, first_alpha, torch.full_like(first_alpha, -1.0))
    return competitor.to(dtype=torch.long), first_alpha


def build_target_distribution(
    labels: torch.Tensor,
    num_classes: int,
    *,
    mode: TargetMode,
    donor_labels: torch.Tensor | None = None,
    crossed_competitors: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
    donor_mass: float = 0.25,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build normalized per-example targets for one of the three arms.

    active_mask allows the experiment to change targets only for tail receivers.
    Inactive rows always remain hard recipient labels.
    """
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B]")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if not 0.0 <= donor_mass <= 1.0:
        raise ValueError("donor_mass must be in [0, 1]")
    if mode not in ("hard", "donor_mix", "first_crossed_competitor"):
        raise ValueError(f"unsupported target mode: {mode}")

    y = labels.to(dtype=torch.long)
    if torch.any((y < 0) | (y >= num_classes)).item():
        raise ValueError("labels contain an out-of-range class ID")
    batch_size = y.shape[0]
    device = y.device
    hard = F.one_hot(y, num_classes=num_classes).to(dtype=dtype)

    if active_mask is None:
        active = torch.ones(batch_size, device=device, dtype=torch.bool)
    else:
        if active_mask.shape != (batch_size,):
            raise ValueError("active_mask must have shape [B]")
        active = active_mask.to(device=device, dtype=torch.bool)

    targets = hard.clone()
    if mode == "hard" or donor_mass == 0.0:
        return targets

    if mode == "donor_mix":
        if donor_labels is None or donor_labels.shape != (batch_size,):
            raise ValueError("donor_labels with shape [B] is required for donor_mix")
        donor = donor_labels.to(device=device, dtype=torch.long)
        if torch.any((donor < 0) | (donor >= num_classes)).item():
            raise ValueError("donor_labels contain an out-of-range class ID")
        selected = torch.nonzero(active, as_tuple=False).flatten()
        if selected.numel() == 0:
            return targets
        targets[selected] = hard[selected] * (1.0 - donor_mass)
        targets[selected, donor[selected]] += donor_mass
        return targets

    if crossed_competitors is None or crossed_competitors.shape != (batch_size,):
        raise ValueError("crossed_competitors with shape [B] is required for competitor targets")
    competitors = crossed_competitors.to(device=device, dtype=torch.long)
    valid_id = (competitors >= -1) & (competitors < num_classes)
    if not valid_id.all().item():
        raise ValueError("crossed_competitors contains an out-of-range class ID")
    if torch.any(active & (competitors == y)).item():
        raise ValueError("a crossed competitor cannot equal the true label")

    selected = torch.nonzero(active & (competitors >= 0), as_tuple=False).flatten()
    if selected.numel() == 0:
        return targets
    targets[selected] = hard[selected] * (1.0 - donor_mass)
    targets[selected, competitors[selected]] += donor_mass
    return targets


def soft_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """Cross-entropy with normalized probability targets, on standard CE scale."""
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits and targets must have identical shape [B, C]")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise ValueError("logits and targets must be floating-point tensors")
    if logits.device != targets.device:
        raise ValueError("logits and targets must be on the same device")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"unsupported reduction: {reduction}")
    if not torch.isfinite(logits).all().item() or not torch.isfinite(targets).all().item():
        raise ValueError("logits or targets contain non-finite values")
    if torch.any(targets < 0).item():
        raise ValueError("targets must be non-negative")
    row_sums = targets.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5):
        raise ValueError("each target distribution must sum to one")

    per_example = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    return per_example.mean()
