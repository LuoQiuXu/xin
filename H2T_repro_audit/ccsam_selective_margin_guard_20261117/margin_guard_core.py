"""Selective Many-to-Tail margin guard for CC-SAM Stage-II.

The guard protects only positive Many-vs-(Medium/Few) margins already present
in a frozen D0 teacher. It deliberately does not distill the full logit
distribution and does not constrain tail-input batches.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

MANY_THRESHOLD = 100
RETAINED_MARGIN_FRACTION = 0.5
EPS = 1e-6


def selective_many_tail_margin_guard(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    train_counts,
    *,
    many_threshold: int = MANY_THRESHOLD,
    retained_fraction: float = RETAINED_MARGIN_FRACTION,
    eps: float = EPS,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return a normalized hinge penalty and JSON-safe batch diagnostics.

    For each Many input whose D0 teacher has a positive margin over every
    non-Many output, retain at least retained_fraction of that teacher
    margin in the student:

        m_teacher = z_teacher[y] - max_{j: n_j <= threshold} z_teacher[j]
        m_student = z_student[y] - max_{j: n_j <= threshold} z_student[j]
        loss_i = relu(retained_fraction*m_teacher - m_student)^2
                 / (m_teacher + eps)^2

    Samples with a non-Many label or a non-positive teacher margin are
    intentionally excluded. Teacher logits are detached. The returned loss
    remains differentiable with respect to student_logits, including when
    the eligible set is empty (in which case it is exactly zero).
    """
    if student_logits.ndim != 2 or student_logits.shape[0] == 0:
        raise ValueError("student_logits must be a nonempty [B,C] tensor")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher_logits must have the same [B,C] shape")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise TypeError("student_logits and teacher_logits must be floating-point tensors")
    if student_logits.device != teacher_logits.device or student_logits.dtype != teacher_logits.dtype:
        raise ValueError("student and teacher logits must share device and dtype")
    if labels.shape != (student_logits.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must be int64 with shape [B]")
    if labels.device != student_logits.device:
        raise ValueError("labels and logits must share a device")
    if not bool(torch.isfinite(student_logits).all()):
        raise ValueError("student_logits contains non-finite values")
    if not bool(torch.isfinite(teacher_logits).all()):
        raise ValueError("teacher_logits contains non-finite values")
    if not isinstance(many_threshold, int) or many_threshold < 1:
        raise ValueError("many_threshold must be a positive integer")
    if not math.isfinite(float(retained_fraction)) or not 0.0 < float(retained_fraction) <= 1.0:
        raise ValueError("retained_fraction must be finite and in (0, 1]")
    if not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("eps must be finite and positive")

    batch_size, num_classes = student_logits.shape
    counts = torch.as_tensor(train_counts, dtype=torch.long, device=student_logits.device)
    if counts.shape != (num_classes,) or bool((counts <= 0).any()):
        raise ValueError("train_counts must contain one positive count per output class")
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError("labels contain an index outside the output class range")

    non_many_classes = counts <= many_threshold
    if not bool(non_many_classes.any()):
        raise ValueError("at least one non-Many output class is required")

    many_inputs = counts[labels] > many_threshold
    with torch.no_grad():
        teacher = teacher_logits.detach()
        teacher_true = teacher.gather(1, labels.view(-1, 1)).squeeze(1)
        teacher_non_many = teacher[:, non_many_classes].max(dim=1).values
        teacher_margin = teacher_true - teacher_non_many

    student_true = student_logits.gather(1, labels.view(-1, 1)).squeeze(1)
    student_non_many = student_logits[:, non_many_classes].max(dim=1).values
    student_margin = student_true - student_non_many

    eligible = many_inputs & (teacher_margin > float(eps))
    many_count = int(many_inputs.sum().item())
    eligible_count = int(eligible.sum().item())

    if eligible_count:
        reference_margin = teacher_margin[eligible].detach()
        target_margin = float(retained_fraction) * reference_margin
        normalized_shortfall = F.relu(target_margin - student_margin[eligible]) / (reference_margin + float(eps))
        loss = normalized_shortfall.square().mean()
        active = target_margin > student_margin[eligible].detach()
        active_count = int(active.sum().item())
        teacher_mean = float(reference_margin.mean().cpu())
        student_mean = float(student_margin[eligible].detach().mean().cpu())
        ratio_mean = float((student_margin[eligible].detach() / reference_margin).mean().cpu())
    else:
        loss = student_logits.sum() * 0.0
        active_count = 0
        teacher_mean = 0.0
        student_mean = 0.0
        ratio_mean = 0.0

    if not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite selective margin guard loss")

    diagnostics = {
        "many_inputs": many_count,
        "eligible_many_inputs": eligible_count,
        "active_guard_inputs": active_count,
        "eligible_fraction_of_many": eligible_count / max(many_count, 1),
        "active_fraction_of_eligible": active_count / max(eligible_count, 1),
        "teacher_margin_mean": teacher_mean,
        "student_margin_mean": student_mean,
        "student_to_teacher_margin_ratio_mean": ratio_mean,
        "retained_margin_fraction": float(retained_fraction),
        "guard_loss": float(loss.detach().cpu()),
    }
    return loss, diagnostics
