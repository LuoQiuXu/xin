"""Dimension-agnostic CC-SAM classifier-stage objectives for cross-dataset transfer.

This module intentionally contains only the frozen D0/D1/D2 mathematics. Dataset
loading, checkpoint orchestration, and final test evaluation live in separate
modules so the objective can be unit-tested without opening any benchmark data.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

D0 = "D0"
D1 = "D1"
D2 = "D2"
VALID_ARMS = (D0, D1, D2)
CLASS_WEIGHT_DIVISOR = 2.0
ADVER_REG = 1.0
ALPHA_RAMP_EPOCHS = 30
ALPHA_MAX = 0.3
HP_TEMPERATURE = 2.0
HP_LAMBDA = 1.0
MANY_THRESHOLD = 100


def class_weights_from_counts(counts, device: torch.device) -> torch.Tensor:
    """Return the exact class weights used by the sealed CIFAR CC-SAM transfer."""
    counts_np = np.asarray(counts, dtype=np.float64)
    if counts_np.ndim != 1 or counts_np.size < 2:
        raise ValueError("counts must be a one-dimensional vector with at least two classes")
    if not np.isfinite(counts_np).all() or np.any(counts_np <= 0):
        raise ValueError("all official training class counts must be finite and positive")
    max_count = float(counts_np.max())
    weights = np.log(max_count / counts_np + 0.01) / CLASS_WEIGHT_DIVISOR
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise RuntimeError("invalid CC-SAM class weights")
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def alpha_for_epoch(epoch_index: int) -> float:
    if not isinstance(epoch_index, int) or epoch_index < 0:
        raise ValueError("epoch_index must be a nonnegative integer")
    return min((epoch_index + 1) / float(ALPHA_RAMP_EPOCHS), ALPHA_MAX)


def class_conditional_feature_delta(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier: nn.Linear,
    class_weights: torch.Tensor,
    adver_reg: float = ADVER_REG,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Vectorized equivalent of per-class autograd.grad(mean CE_class, features).

    The class gradient is represented in the full [B,D] batch shape with zeros
    outside the class subset. Its L2 norm is taken over that full tensor, exactly
    as in the already validated CIFAR implementation.
    """
    if not isinstance(classifier, nn.Linear):
        raise TypeError("classifier must be a single nn.Linear layer")
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a nonempty [B,D] tensor")
    batch_size, feature_dim = features.shape
    num_classes = classifier.out_features
    if classifier.in_features != feature_dim:
        raise ValueError("feature dimension does not match classifier.in_features")
    if labels.shape != (batch_size,) or labels.dtype != torch.long:
        raise ValueError("labels must be int64 with shape [B]")
    if class_weights.shape != (num_classes,):
        raise ValueError("class_weights must have shape [num_classes]")
    if any(t.device != features.device for t in (labels, class_weights, classifier.weight)):
        raise ValueError("features, labels, weights, and classifier must share a device")
    if any(t.dtype != features.dtype for t in (classifier.weight, class_weights)):
        raise ValueError("features, classifier weights, and class weights must share a dtype")
    if not torch.isfinite(features).all() or not torch.isfinite(class_weights).all():
        raise RuntimeError("non-finite features or class weights")
    if bool(((labels < 0) | (labels >= num_classes)).any().item()):
        raise ValueError("label outside classifier output range")
    if not np.isfinite(float(adver_reg)) or float(adver_reg) < 0:
        raise ValueError("adver_reg must be finite and nonnegative")

    with torch.no_grad():
        logits = classifier(features.detach())
        probabilities = F.log_softmax(logits, dim=1).exp()
        one_hot = F.one_hot(labels, num_classes=num_classes).to(probabilities.dtype)
        counts_in_batch = torch.bincount(labels, minlength=num_classes)
        per_sample_counts = counts_in_batch[labels].to(probabilities.dtype).unsqueeze(1)
        scaled_logit_gradient = (probabilities - one_hot) / per_sample_counts
        per_sample_gradient = scaled_logit_gradient @ classifier.weight.detach()

        delta = torch.zeros_like(features)
        pre_norms = []
        zero_gradient_classes = 0
        for class_id in torch.unique(labels, sorted=True):
            mask = labels == class_id
            class_gradient_full = per_sample_gradient * mask.unsqueeze(1).to(per_sample_gradient.dtype)
            gradient_norm = torch.linalg.vector_norm(class_gradient_full)
            pre_norms.append(float(gradient_norm.cpu()))
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite class-conditional gradient norm")
            if float(gradient_norm.cpu()) <= 1e-12:
                zero_gradient_classes += 1
                continue
            delta[mask] = (
                float(adver_reg)
                * class_weights[class_id]
                * class_gradient_full[mask]
                / gradient_norm
            )

        if not torch.isfinite(delta).all():
            raise RuntimeError("non-finite CC-SAM feature perturbation")
        feature_norms = torch.linalg.vector_norm(features.detach(), dim=1).clamp_min(1e-12)
        delta_norms = torch.linalg.vector_norm(delta, dim=1)
        stats = {
            "num_classes_in_batch": int(torch.unique(labels).numel()),
            "zero_gradient_classes": int(zero_gradient_classes),
            "mean_class_gradient_norm_before_normalization": float(np.mean(pre_norms)) if pre_norms else 0.0,
            "mean_delta_l2_per_sample": float(delta_norms.mean().cpu()),
            "mean_relative_delta_l2": float((delta_norms / feature_norms).mean().cpu()),
            "max_delta_l2_per_sample": float(delta_norms.max().cpu()),
        }
    return delta.detach(), stats


def head_preservation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    train_counts,
    temperature: float = HP_TEMPERATURE,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """KL(teacher || student), applied to Many inputs only and all output classes."""
    if not np.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [B,C]")
    batch_size, num_classes = student_logits.shape
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if labels.shape != (batch_size,) or labels.dtype != torch.long:
        raise ValueError("labels must be int64 with shape [B]")
    if any(t.device != student_logits.device for t in (teacher_logits, labels)):
        raise ValueError("teacher, student, and labels must share a device")
    counts = torch.as_tensor(train_counts, dtype=torch.long, device=labels.device)
    if counts.shape != (num_classes,) or bool((counts <= 0).any().item()):
        raise ValueError("train_counts must contain one positive count per class")
    if bool(((labels < 0) | (labels >= num_classes)).any().item()):
        raise ValueError("label outside logits range")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise RuntimeError("non-finite logits passed to head-preservation KL")

    many_mask = counts[labels] > MANY_THRESHOLD
    many_count = int(many_mask.sum().item())
    if many_count == 0:
        return student_logits.sum() * 0.0, many_mask, 0

    student_log_prob = F.log_softmax(student_logits[many_mask] / float(temperature), dim=1)
    with torch.no_grad():
        teacher_prob = F.softmax(
            teacher_logits[many_mask].detach() / float(temperature), dim=1
        )
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    loss = loss * (float(temperature) ** 2)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite head-preservation KL")
    return loss, many_mask, many_count


def compute_method_loss(
    arm: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier: nn.Linear,
    class_weights: torch.Tensor,
    epoch_index: int,
    train_counts,
    teacher: Optional[nn.Linear] = None,
    temperature: float = HP_TEMPERATURE,
    lambda_hp: float = HP_LAMBDA,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute one of the frozen D0/D1/D2 objectives on clean cached features."""
    if arm not in VALID_ARMS:
        raise ValueError("arm must be one of " + ", ".join(VALID_ARMS))
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a nonempty [B,D] tensor")
    if labels.shape != (features.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must be int64 with shape [B]")
    if classifier.in_features != features.shape[1]:
        raise ValueError("feature dimension and classifier input dimension differ")
    if classifier.out_features != class_weights.numel():
        raise ValueError("classifier output dimension and class_weights differ")
    if not np.isfinite(float(lambda_hp)) or float(lambda_hp) < 0:
        raise ValueError("lambda_hp must be finite and nonnegative")
    if arm == D2 and teacher is None:
        raise ValueError("D2 requires the matching same-dataset/same-seed D0 teacher")
    if teacher is not None:
        if teacher.in_features != classifier.in_features or teacher.out_features != classifier.out_features:
            raise ValueError("teacher and student classifier dimensions differ")
        if any(p.requires_grad for p in teacher.parameters()):
            raise ValueError("teacher parameters must be frozen before loss computation")

    alpha = alpha_for_epoch(epoch_index)
    clean_features = features.detach()
    clean_logits = classifier(clean_features)
    clean_ce = F.cross_entropy(clean_logits, labels, reduction="none")
    zero_delta = torch.zeros_like(clean_features)
    delta_stats = {
        "num_classes_in_batch": int(torch.unique(labels).numel()),
        "zero_gradient_classes": 0,
        "mean_class_gradient_norm_before_normalization": 0.0,
        "mean_delta_l2_per_sample": 0.0,
        "mean_relative_delta_l2": 0.0,
        "max_delta_l2_per_sample": 0.0,
    }
    hp_loss = clean_logits.sum() * 0.0
    many_count = 0

    if arm == D0:
        delta = zero_delta
        adv_ce = clean_ce
        loss = alpha * (clean_ce * class_weights[labels]).mean() + (1.0 - alpha) * clean_ce.mean()
    else:
        delta, delta_stats = class_conditional_feature_delta(
            clean_features, labels, classifier, class_weights, adver_reg=ADVER_REG
        )
        adv_logits = classifier(clean_features + delta)
        adv_ce = F.cross_entropy(adv_logits, labels, reduction="none")
        loss = alpha * (adv_ce * class_weights[labels]).mean() + (1.0 - alpha) * clean_ce.mean()

        if arm == D2:
            teacher.eval()
            with torch.no_grad():
                teacher_logits = teacher(clean_features)
            hp_loss, many_mask, many_count = head_preservation_kl(
                clean_logits, teacher_logits, labels, train_counts, temperature=temperature
            )
            loss = loss + float(lambda_hp) * hp_loss

    if not torch.isfinite(loss):
        raise RuntimeError("non-finite total loss for " + arm)

    diagnostics = {
        "alpha": float(alpha),
        "clean_ce_mean": float(clean_ce.detach().mean().cpu()),
        "weighted_adv_ce_mean": float((adv_ce.detach() * class_weights[labels]).mean().cpu()),
        "head_preservation_kl": float(hp_loss.detach().cpu()),
        "many_inputs_in_batch": int(many_count),
        "delta_mean_l2": float(delta_stats["mean_delta_l2_per_sample"]),
        "delta_mean_relative_l2": float(delta_stats["mean_relative_delta_l2"]),
        "delta_zero_gradient_classes": int(delta_stats["zero_gradient_classes"]),
    }
    return loss, diagnostics
