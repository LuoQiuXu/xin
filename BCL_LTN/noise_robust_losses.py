from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def estimate_clean_class_counts(
    observed_counts: np.ndarray,
    *,
    noise_rate: float = 0.4,
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert a known symmetric K-class noise channel using observed counts only.

    Returns (unconstrained_inverse_counts, positive_renormalized_counts).
    No dataset imbalance ratio, clean labels, or corruption mask is used.
    epsilon is a small count-space floor before normalization, not a class-count
    prior: the returned counts are renormalized to sum to the observed total.
    """
    observed = np.asarray(observed_counts, dtype=np.float64)
    if observed.ndim != 1 or observed.size < 2:
        raise ValueError("observed_counts must be a 1-D vector with at least two classes")
    if not np.isfinite(observed).all() or np.any(observed < 0):
        raise ValueError("observed_counts must be finite and non-negative")
    if not np.isfinite(noise_rate) or not 0.0 <= noise_rate < (observed.size - 1) / observed.size:
        raise ValueError("Symmetric transition is not invertible at this noise rate")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    total = float(observed.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("observed_counts must have finite positive total mass")
    k = observed.size
    diagonal_scale = 1.0 - float(noise_rate) - float(noise_rate) / (k - 1)
    noise_floor = float(noise_rate) * total / (k - 1)
    raw = (observed - noise_floor) / diagonal_scale

    positive = np.maximum(raw, float(epsilon))
    corrected = positive * (total / float(positive.sum()))
    if (not np.isfinite(raw).all() or not np.isfinite(corrected).all()
            or np.any(corrected <= 0)
            or abs(float(corrected.sum()) - total) > max(1e-6, total * 1e-10)):
        raise RuntimeError("Clean-count inversion failed positivity or normalization checks")
    return raw, corrected


def symmetric_transition_matrix(
    num_classes: int,
    noise_rate: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """T[i,j] = P(observed=j | clean=i), with uniform off-diagonal noise."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if not 0.0 <= noise_rate < (num_classes - 1) / num_classes:
        raise ValueError("Symmetric transition must be invertible")
    eye = torch.eye(num_classes, device=device, dtype=dtype)
    off_diagonal = (1.0 - eye) * (float(noise_rate) / (num_classes - 1))
    return eye * (1.0 - float(noise_rate)) + off_diagonal


def soft_targets_from_three_view_logits(
    all_logits: torch.Tensor,
    batch_size: int,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Build detached q from mean raw logits of concatenated [v1, v2, v3].

    Deliberately does NOT add log class priors. The three-view logits must be
    ordered as [all v1, all v2, all v3], matching the BCL backbone forward.
    """
    if all_logits.ndim != 2 or all_logits.shape[0] != 3 * int(batch_size):
        raise ValueError("all_logits must have shape [3 * batch_size, num_classes]")
    if batch_size < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("batch_size and temperature must be finite and positive")
    if all_logits.shape[1] < 2 or not all_logits.is_floating_point() or not bool(torch.isfinite(all_logits).all()):
        raise ValueError("all_logits must be finite floating-point logits for at least two classes")
    three_views = all_logits.detach().reshape(3, int(batch_size), all_logits.shape[1])
    q = F.softmax(three_views.mean(dim=0) / float(temperature), dim=1)
    if not bool(torch.isfinite(q).all()):
        raise ValueError("Raw-logit soft targets must be finite")
    return q.detach()


class ForwardCorrectedLogitCompensation(nn.Module):
    """Balanced-softmax prior correction followed by forward label-noise correction."""

    def __init__(
        self,
        clean_class_counts: np.ndarray | torch.Tensor,
        *,
        noise_rate: float = 0.4,
    ):
        super().__init__()
        counts = torch.as_tensor(clean_class_counts, dtype=torch.float32).detach()
        if counts.ndim != 1 or counts.numel() < 2 or not bool(torch.isfinite(counts).all()):
            raise ValueError("clean_class_counts must be a finite 1-D vector")
        if bool(torch.any(counts <= 0)):
            raise ValueError("Every corrected class count must be positive")
        log_prior = torch.log(counts / counts.sum())
        if not bool(torch.isfinite(log_prior).all()):
            raise ValueError("The corrected log prior must be finite")
        transition = symmetric_transition_matrix(
            int(counts.numel()), noise_rate, dtype=torch.float32
        )
        self.register_buffer("log_prior", log_prior)
        self.register_buffer("transition", transition)
        self.noise_rate = float(noise_rate)

    def _adjusted_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != self.log_prior.numel():
            raise ValueError("logits must have shape [nonempty batch, num_classes]")
        if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
            raise ValueError("logits must be finite floating-point values")
        return logits + self.log_prior.to(logits)

    def clean_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self._adjusted_logits(logits), dim=1)

    def observed_log_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        log_p_clean = F.log_softmax(self._adjusted_logits(logits), dim=1)
        log_t = self.transition.to(logits).log()
        # log P(observed=j | x) = logsumexp_i(log P(clean=i | x) + log T[i,j]).
        return torch.logsumexp(log_p_clean.unsqueeze(2) + log_t.unsqueeze(0), dim=1)

    def forward(self, logits: torch.Tensor, observed_targets: torch.Tensor) -> torch.Tensor:
        log_p_clean = F.log_softmax(self._adjusted_logits(logits), dim=1)
        if observed_targets.ndim != 1 or observed_targets.shape[0] != logits.shape[0]:
            raise ValueError("observed_targets must have shape [batch]")
        if observed_targets.is_floating_point() or observed_targets.is_complex() or observed_targets.dtype == torch.bool:
            raise ValueError("Observed labels must be integer IDs")
        targets = observed_targets.to(device=logits.device, dtype=torch.long)
        if bool(torch.any((targets < 0) | (targets >= logits.shape[1]))):
            raise ValueError("Observed label outside class range")
        # Only one observed column per sample contributes to the NLL.
        log_t_for_target = self.transition.to(logits).T[targets].log()
        return -torch.logsumexp(log_p_clean + log_t_for_target, dim=1).mean()


class NoiseAwareSoftBalancedContrastiveLoss(nn.Module):
    """BCL-style class-complement loss with detached soft clean-class assignments.

    During cold start, semantic_enabled=False switches to instance-only NT-Xent:
    the paired v2-v3 view is the sole positive and no class prototypes are used.
    Once semantic links are enabled, q is detached and cross-sample/prototype
    positives are continuously attenuated by prediction confidence.
    """

    def __init__(self, num_classes: int = 100, temperature: float = 0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)

    @staticmethod
    def confidence_reliability(q: torch.Tensor) -> torch.Tensor:
        k = q.shape[1]
        min_probability, max_probability = torch.aminmax(q, dim=1)
        reliability = ((max_probability - 1.0 / k) / (1.0 - 1.0 / k)).clamp(0.0, 1.0)
        # A mathematically uniform row must have zero semantic weight despite normalization rounding.
        return reliability.masked_fill(max_probability == min_probability, 0.0)

    def _instance_only_loss(self, two_view_features: torch.Tensor) -> torch.Tensor:
        batch_size = int(two_view_features.shape[0])
        features = torch.cat(
            [two_view_features[:, 0, :], two_view_features[:, 1, :]], dim=0
        )
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        num_views = 2 * batch_size
        diagonal = torch.eye(num_views, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
        indices = torch.arange(num_views, device=logits.device)
        paired_indices = (indices + batch_size) % num_views
        # Standard instance-level NT-Xent: exactly one positive per anchor.
        return F.cross_entropy(logits, paired_indices)

    def _semantic_weights(self, soft_targets: torch.Tensor, features: torch.Tensor):
        # Critical: detach before any q_i q_j or prototype-positive calculation.
        q = soft_targets.detach().to(features)
        if not bool(torch.isfinite(q).all() & (q >= 0).all()):
            raise ValueError("soft_targets must be finite and non-negative")
        row_sums = q.sum(dim=1, keepdim=True)
        if not bool(torch.isfinite(row_sums).all() & (row_sums > 0).all()):
            raise ValueError("Every soft-target row must have finite positive mass")
        q = q / row_sums
        batch_size = q.shape[0]
        reliability = self.confidence_reliability(q)

        prototype_q = torch.eye(
            self.num_classes, device=q.device, dtype=q.dtype
        )
        candidate_q = torch.cat([q, q, prototype_q], dim=0)
        anchor_q = torch.cat([q, q], dim=0)
        candidate_reliability = torch.cat(
            [reliability, reliability,
             torch.ones(self.num_classes, device=q.device, dtype=q.dtype)],
            dim=0,
        )
        anchor_reliability = torch.cat([reliability, reliability], dim=0)

        num_anchors = 2 * batch_size
        logits_mask = torch.ones(
            (num_anchors, num_anchors + self.num_classes),
            device=q.device,
            dtype=q.dtype,
        )
        anchor_indices = torch.arange(num_anchors, device=q.device)
        logits_mask[anchor_indices, anchor_indices] = 0.0

        positive_weights = anchor_q @ candidate_q.T
        positive_weights = (
            positive_weights
            * anchor_reliability.unsqueeze(1)
            * candidate_reliability.unsqueeze(0)
        )
        # Preserve each image's v2-v3 self-positive regardless of confidence.
        paired_view_indices = (anchor_indices + batch_size) % num_anchors
        positive_weights[anchor_indices, paired_view_indices] = 1.0
        positive_weights = positive_weights * logits_mask

        # Expected class occurrences reduce exactly to BCL's integer counts
        # when q is one-hot; each candidate receives its expected class count.
        class_occurrences = candidate_q.sum(dim=0)
        candidate_class_mass = candidate_q @ class_occurrences
        per_entry_weight = candidate_class_mass.unsqueeze(0) - positive_weights
        if not bool(torch.isfinite(per_entry_weight).all() & (per_entry_weight > 0).all()):
            raise RuntimeError("Soft BCL produced a non-finite or non-positive denominator weight")
        return positive_weights, logits_mask, per_entry_weight, class_occurrences

    def forward(
        self,
        class_prototypes: Optional[torch.Tensor],
        two_view_features: torch.Tensor,
        soft_targets: Optional[torch.Tensor] = None,
        *,
        semantic_enabled: bool = True,
    ) -> torch.Tensor:
        if (two_view_features.ndim != 3 or two_view_features.shape[1] != 2
                or two_view_features.shape[0] == 0 or two_view_features.shape[2] == 0):
            raise ValueError("two_view_features must have shape [nonempty batch, 2, feature_dim]")
        if not two_view_features.is_floating_point() or not bool(torch.isfinite(two_view_features).all()):
            raise ValueError("two_view_features must contain finite floating-point values")

        if not semantic_enabled:
            return self._instance_only_loss(two_view_features)

        batch_size = two_view_features.shape[0]
        if class_prototypes is None or class_prototypes.shape != (self.num_classes, two_view_features.shape[2]):
            raise ValueError("class_prototypes must have shape [num_classes, feature_dim]")
        if soft_targets is None or soft_targets.shape != (batch_size, self.num_classes):
            raise ValueError("soft_targets must have shape [batch, num_classes]")
        sample_features = torch.cat([two_view_features[:, 0, :], two_view_features[:, 1, :]], dim=0)
        prototypes = class_prototypes.to(two_view_features)
        if not bool(torch.isfinite(prototypes).all()):
            raise ValueError("class_prototypes must be finite")
        all_features = torch.cat([sample_features, prototypes], dim=0)
        num_anchors = 2 * batch_size
        positive_weights, logits_mask, per_entry_weight, _ = self._semantic_weights(soft_targets, two_view_features)

        logits = (all_features[:num_anchors] @ all_features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * logits_mask
        denominator = (exp_logits / per_entry_weight).sum(dim=1, keepdim=True)
        if not bool(torch.isfinite(denominator).all()) or bool(torch.any(denominator <= 0)):
            raise RuntimeError("Soft BCL denominator is non-finite or non-positive")
        log_prob = logits - torch.log(denominator.clamp_min(torch.finfo(logits.dtype).tiny))

        positive_mass = positive_weights.sum(dim=1)
        if bool(torch.any(positive_mass <= 0)):
            raise RuntimeError("Every soft BCL anchor must retain its paired-view positive")
        mean_log_prob_positive = (
            positive_weights * log_prob
        ).sum(dim=1) / positive_mass
        loss = -mean_log_prob_positive.mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Soft BCL loss is non-finite")
        return loss
