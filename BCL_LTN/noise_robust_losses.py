from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def project_lower_bounded_simplex(
    values: np.ndarray,
    *,
    total: float,
    lower_bound: float,
) -> np.ndarray:
    """Euclidean projection onto {x: sum(x)=total, x_i>=lower_bound}."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0 or not np.isfinite(v).all():
        raise ValueError("values must be a finite non-empty vector")
    if not np.isfinite(total) or not np.isfinite(lower_bound):
        raise ValueError("total and lower_bound must be finite")
    remaining = float(total) - float(lower_bound) * v.size
    if remaining < 0:
        raise ValueError("total is smaller than the required lower-bound mass")
    shifted = v - float(lower_bound)
    ordered = np.sort(shifted)[::-1]
    cumulative = np.cumsum(ordered) - remaining
    indices = np.arange(1, v.size + 1, dtype=np.float64)
    valid = ordered - cumulative / indices > 0
    if not np.any(valid):
        theta = cumulative[-1] / v.size
    else:
        rho = int(np.flatnonzero(valid)[-1])
        theta = cumulative[rho] / float(rho + 1)
    projected = np.maximum(shifted - theta, 0.0) + float(lower_bound)
    # Remove floating-point sum drift without violating the lower bound.
    drift = float(total) - float(projected.sum())
    if abs(drift) > 1e-10:
        free = np.flatnonzero(projected > lower_bound + 1e-12)
        if free.size:
            projected[free] += drift / free.size
        elif abs(drift) > 1e-7:
            raise RuntimeError("Could not correct simplex projection sum drift")
    if (not np.isfinite(projected).all()
            or np.any(projected < lower_bound - 1e-8)
            or abs(float(projected.sum()) - float(total)) > 1e-6):
        raise RuntimeError("Lower-bounded simplex projection failed its constraints")
    return projected


def estimate_clean_class_counts(
    observed_counts: np.ndarray,
    *,
    noise_rate: float = 0.4,
    min_count: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert a known symmetric K-class noise channel, then project to the LT simplex.

    Returns (unconstrained_inverse_counts, projected_clean_count_estimate).
    The caller must provide only counts of observed noisy labels; no clean labels
    or corruption mask are used here.
    """
    observed = np.asarray(observed_counts, dtype=np.float64)
    if observed.ndim != 1 or observed.size < 2:
        raise ValueError("observed_counts must be a 1-D vector with at least two classes")
    if not np.isfinite(observed).all() or np.any(observed < 0):
        raise ValueError("observed_counts must be finite and non-negative")
    if not 0.0 <= noise_rate < (observed.size - 1) / observed.size:
        raise ValueError("Symmetric transition is not invertible at this noise rate")
    if not np.isfinite(min_count) or min_count < 0:
        raise ValueError("min_count must be finite and non-negative")
    total = float(observed.sum())
    if total <= min_count * observed.size:
        raise ValueError("Observed total is incompatible with the requested class floor")
    k = observed.size
    diagonal_scale = 1.0 - float(noise_rate) - float(noise_rate) / (k - 1)
    noise_floor = float(noise_rate) * total / (k - 1)
    raw = (observed - noise_floor) / diagonal_scale
    projected = project_lower_bounded_simplex(
        raw, total=total, lower_bound=float(min_count)
    )
    return raw, projected


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


class ForwardCorrectedLogitCompensation(nn.Module):
    """Balanced-softmax prior correction followed by forward label-noise correction."""

    def __init__(
        self,
        clean_class_counts: np.ndarray | torch.Tensor,
        *,
        noise_rate: float = 0.4,
    ):
        super().__init__()
        counts = torch.as_tensor(clean_class_counts, dtype=torch.float32)
        if counts.ndim != 1 or counts.numel() < 2 or not bool(torch.isfinite(counts).all()):
            raise ValueError("clean_class_counts must be a finite 1-D vector")
        if bool(torch.any(counts <= 0)):
            raise ValueError("Every corrected class count must be positive")
        log_prior = torch.log(counts / counts.sum())
        transition = symmetric_transition_matrix(
            int(counts.numel()), noise_rate, dtype=torch.float32
        )
        self.register_buffer("log_prior", log_prior)
        self.register_buffer("transition", transition)
        self.register_buffer("log_transition", transition.log())
        self.noise_rate = float(noise_rate)

    def clean_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.log_prior.numel():
            raise ValueError("logits must have shape [batch, num_classes]")
        return F.softmax(
            logits + self.log_prior.to(device=logits.device, dtype=logits.dtype),
            dim=1,
        )

    def observed_log_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        log_p_clean = F.log_softmax(
            logits + self.log_prior.to(device=logits.device, dtype=logits.dtype),
            dim=1,
        )
        log_t = self.log_transition.to(device=logits.device, dtype=logits.dtype)
        # log P(observed=j | x) = logsumexp_i(log P(clean=i | x) + log T[i,j]).
        return torch.logsumexp(log_p_clean.unsqueeze(2) + log_t.unsqueeze(0), dim=1)

    def forward(self, logits: torch.Tensor, observed_targets: torch.Tensor) -> torch.Tensor:
        if observed_targets.ndim != 1 or observed_targets.shape[0] != logits.shape[0]:
            raise ValueError("observed_targets must have shape [batch]")
        targets = observed_targets.to(device=logits.device, dtype=torch.long)
        if bool(torch.any((targets < 0) | (targets >= logits.shape[1]))):
            raise ValueError("Observed label outside class range")
        log_p_observed = self.observed_log_probabilities(logits)
        return F.nll_loss(log_p_observed, targets)


class NoiseAwareSoftBalancedContrastiveLoss(nn.Module):
    """BCL-style class-complement loss with detached soft clean-class assignments.

    q is a model-derived class distribution, not a clean-label target. The
    original two-view pair is always positive. Cross-sample and prototype
    positive links are continuously attenuated by prediction confidence.
    """

    def __init__(self, num_classes: int = 100, temperature: float = 0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)

    @staticmethod
    def confidence_reliability(q: torch.Tensor) -> torch.Tensor:
        k = q.shape[1]
        max_probability = q.max(dim=1).values
        return ((max_probability - 1.0 / k) / (1.0 - 1.0 / k)).clamp(0.0, 1.0)

    def forward(
        self,
        class_prototypes: torch.Tensor,
        two_view_features: torch.Tensor,
        soft_targets: torch.Tensor,
    ) -> torch.Tensor:
        if two_view_features.ndim != 3 or two_view_features.shape[1] != 2:
            raise ValueError("two_view_features must have shape [batch, 2, feature_dim]")
        batch_size = int(two_view_features.shape[0])
        if batch_size < 1:
            raise ValueError("batch must be non-empty")
        if class_prototypes.ndim != 2 or class_prototypes.shape[0] != self.num_classes:
            raise ValueError("class_prototypes must have shape [num_classes, feature_dim]")
        if two_view_features.shape[2] != class_prototypes.shape[1]:
            raise ValueError("Feature dimensions of samples and prototypes must match")
        if soft_targets.shape != (batch_size, self.num_classes):
            raise ValueError("soft_targets must have shape [batch, num_classes]")
        if not bool(torch.isfinite(soft_targets).all()) or bool(torch.any(soft_targets < 0)):
            raise ValueError("soft_targets must be finite and non-negative")

        # Detach prevents the contrastive objective from gaming its own targets.
        q = soft_targets.detach().to(
            device=two_view_features.device, dtype=two_view_features.dtype
        )
        row_sums = q.sum(dim=1, keepdim=True)
        if bool(torch.any(row_sums <= 0)):
            raise ValueError("Every soft-target row must have positive mass")
        q = q / row_sums
        reliability = self.confidence_reliability(q)

        sample_features = torch.cat(
            [two_view_features[:, 0, :], two_view_features[:, 1, :]], dim=0
        )
        prototypes = class_prototypes.to(
            device=two_view_features.device, dtype=two_view_features.dtype
        )
        all_features = torch.cat([sample_features, prototypes], dim=0)

        prototype_q = torch.eye(
            self.num_classes, device=q.device, dtype=q.dtype
        )
        candidate_q = torch.cat([q, q, prototype_q], dim=0)
        anchor_q = torch.cat([q, q], dim=0)
        candidate_reliability = torch.cat(
            [reliability, reliability, torch.ones(self.num_classes, device=q.device, dtype=q.dtype)],
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

        # Soft expected class counts reduce exactly to BCL's integer counts
        # when q is one-hot.
        class_occurrences = candidate_q.sum(dim=0)
        candidate_class_mass = candidate_q @ class_occurrences
        per_entry_weight = candidate_class_mass.unsqueeze(0) - positive_weights
        active_weights = per_entry_weight[logits_mask.bool()]
        if (not bool(torch.isfinite(active_weights).all())
                or bool(torch.any(active_weights <= 0))):
            raise RuntimeError("Soft BCL produced a non-positive denominator weight")

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
