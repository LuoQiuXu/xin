"""Core utilities for NTC-BCL v2 Stage-1 estimator and collapse audits.

Probability convention
----------------------
The estimator consumes BALANCED class evidence p_bal(c|x) = softmax(raw_logits).
It then applies the clean prior exactly once:
    gamma_ic ∝ p_bal(c|x_i) * pi_c * T[c, observed_i].
If callers already have a long-tail posterior softmax(logits + log(pi)), they
must not pass it as p_bal. Use q_from_logits(..., clean_prior=pi) for that
posterior, or recover p_bal from raw logits.

Transition orientation is always T[clean_class, observed_class].
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("probabilities must have shape [nonempty N, K>=2]")
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("probabilities must be finite and non-negative")
    sums = p.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("every probability row must have positive mass")
    return p / sums


def _validate_labels(labels: np.ndarray, n: int, k: int, name: str) -> np.ndarray:
    y = np.asarray(labels)
    if y.shape != (n,) or y.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer vector with shape [{n}]")
    y = y.astype(np.int64, copy=False)
    if np.any((y < 0) | (y >= k)):
        raise ValueError(f"{name} contains a class outside [0, {k})")
    return y


def _row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.maximum(matrix, 0.0)
    sums = matrix.sum(axis=1, keepdims=True)
    if np.any(sums <= eps):
        raise ValueError("cannot normalize a zero-mass transition row")
    return matrix / sums


def _global_transition_prior(
    weighted_counts: np.ndarray,
    class_support: np.ndarray,
    *,
    high_support_fraction: float,
    min_high_support_classes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a non-uniform global row prior from high-support pseudo-classes.

    The pooled diagonal retention is shared, while off-diagonal destination
    propensities are pooled by observed destination and re-normalized per row.
    No noise rate or symmetric transition matrix is supplied.
    """
    k = int(weighted_counts.shape[0])
    if not 0.0 < high_support_fraction <= 1.0:
        raise ValueError("high_support_fraction must be in (0, 1]")
    if min_high_support_classes < 1:
        raise ValueError("min_high_support_classes must be positive")

    n_high = min(k, max(min_high_support_classes, int(math.ceil(k * high_support_fraction))))
    high_ids = np.argsort(-class_support, kind="stable")[:n_high]
    high_counts = weighted_counts[high_ids]
    high_support = float(class_support[high_ids].sum())
    if high_support <= 1e-12:
        high_ids = np.arange(k)
        high_counts = weighted_counts
        high_support = float(class_support.sum())
    if high_support <= 1e-12:
        raise RuntimeError("No effective pseudo-class support for transition estimation")

    diag_by_class = np.diag(weighted_counts)
    diagonal_mass = float(diag_by_class[high_ids].sum())
    diagonal_rate = float(np.clip(diagonal_mass / high_support, 1e-4, 1.0 - 1e-4))

    offdiag_by_destination = high_counts.sum(axis=0)
    offdiag_by_destination = offdiag_by_destination.copy()
    offdiag_by_destination[high_ids] -= diag_by_class[high_ids]
    offdiag_by_destination = np.maximum(offdiag_by_destination, 0.0) + 1e-8

    global_prior = np.zeros((k, k), dtype=np.float64)
    for clean_class in range(k):
        destination = offdiag_by_destination.copy()
        destination[clean_class] = 0.0
        total = float(destination.sum())
        if total <= 1e-12:
            destination[:] = 1.0
            destination[clean_class] = 0.0
            total = float(destination.sum())
        global_prior[clean_class] = (1.0 - diagonal_rate) * destination / total
        global_prior[clean_class, clean_class] = diagonal_rate

    return _row_normalize(global_prior), high_ids.astype(np.int64), diagonal_rate


def posterior_from_balanced_probabilities(
    balanced_probabilities: np.ndarray,
    clean_prior: np.ndarray,
    transition: np.ndarray,
    observed_labels: np.ndarray,
) -> np.ndarray:
    """E-step: P(clean class | x, observed label), applying pi exactly once."""
    p_bal = _validate_probabilities(balanced_probabilities)
    n, k = p_bal.shape
    y = _validate_labels(observed_labels, n, k, "observed_labels")
    pi = np.asarray(clean_prior, dtype=np.float64)
    t = np.asarray(transition, dtype=np.float64)
    if pi.shape != (k,) or not np.isfinite(pi).all() or np.any(pi <= 0):
        raise ValueError("clean_prior must be a positive finite vector of length K")
    if t.shape != (k, k) or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("transition must be a finite non-negative KxK matrix")
    if not np.allclose(t.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("transition rows must sum to one")
    pi = pi / pi.sum()
    likelihood = t[:, y].T
    joint = p_bal * pi[None, :] * likelihood
    row_mass = joint.sum(axis=1, keepdims=True)
    if np.any(row_mass <= 1e-15):
        raise RuntimeError("E-step posterior has zero probability mass")
    return joint / row_mass


def estimate_transition_em(
    balanced_probabilities: np.ndarray,
    observed_labels: np.ndarray,
    *,
    shrinkage_mass: float = 20.0,
    prior_alpha: float = 0.5,
    high_support_fraction: float = 0.35,
    min_high_support_classes: int = 5,
    max_iter: int = 50,
    tolerance: float = 1e-6,
) -> Dict[str, object]:
    """Estimate clean prior pi, transition T and sample posterior gamma.

    Inputs contain only raw-logit balanced probabilities and observed labels.
    Clean labels, noise masks, true noise rates and true transition matrices
    are deliberately absent from this API.

    Tail rows shrink toward an empirical global transition structure estimated
    from high effective-support pseudo-classes, not toward a uniform matrix.
    """
    p_bal = _validate_probabilities(balanced_probabilities)
    n, k = p_bal.shape
    y = _validate_labels(observed_labels, n, k, "observed_labels")
    if not math.isfinite(shrinkage_mass) or shrinkage_mass < 0:
        raise ValueError("shrinkage_mass must be finite and non-negative")
    if not math.isfinite(prior_alpha) or prior_alpha < 0:
        raise ValueError("prior_alpha must be finite and non-negative")
    if max_iter < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("max_iter and tolerance must be positive")

    one_hot_observed = np.eye(k, dtype=np.float64)[y]
    gamma = p_bal.copy()
    support = gamma.sum(axis=0)
    weighted_counts = gamma.T @ one_hot_observed
    t_global, high_ids, global_diagonal = _global_transition_prior(
        weighted_counts,
        support,
        high_support_fraction=high_support_fraction,
        min_high_support_classes=min_high_support_classes,
    )
    transition = _row_normalize(
        weighted_counts + float(shrinkage_mass) * t_global
    )
    prior = np.full(k, 1.0 / k, dtype=np.float64)
    converged = False
    delta = float("inf")
    iterations = 0

    for iteration in range(1, max_iter + 1):
        gamma = posterior_from_balanced_probabilities(p_bal, prior, transition, y)
        new_support = gamma.sum(axis=0)
        new_counts = gamma.T @ one_hot_observed
        new_global, high_ids, global_diagonal = _global_transition_prior(
            new_counts,
            new_support,
            high_support_fraction=high_support_fraction,
            min_high_support_classes=min_high_support_classes,
        )
        new_transition = _row_normalize(
            new_counts + float(shrinkage_mass) * new_global
        )
        new_prior = (new_support + float(prior_alpha)) / (
            float(n) + float(k) * float(prior_alpha)
        )
        delta = max(
            float(np.max(np.abs(new_transition - transition))),
            float(np.max(np.abs(new_prior - prior))),
        )
        transition, prior = new_transition, new_prior
        iterations = iteration
        if delta <= tolerance:
            converged = True
            break

    gamma = posterior_from_balanced_probabilities(p_bal, prior, transition, y)
    support = gamma.sum(axis=0)
    weighted_counts = gamma.T @ one_hot_observed
    return {
        "clean_prior": prior,
        "transition": transition,
        "posterior": gamma,
        "effective_support": support,
        "weighted_transition_counts": weighted_counts,
        "global_transition_prior": t_global,
        "high_support_class_ids": high_ids,
        "global_diagonal_retention": float(global_diagonal),
        "estimated_noise_rate": float(1.0 - np.dot(prior, np.diag(transition))),
        "iterations": int(iterations),
        "converged": bool(converged),
        "final_delta": float(delta),
        "shrinkage_mass": float(shrinkage_mass),
        "prior_alpha_per_class": float(prior_alpha),
        "high_support_fraction": float(high_support_fraction),
    }


def transition_truth_from_labels(
    clean_labels: np.ndarray,
    observed_labels: np.ndarray,
    num_classes: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Offline-only helper: empirical T[clean, observed] and clean class prior."""
    clean = np.asarray(clean_labels)
    observed = np.asarray(observed_labels)
    if clean.ndim != 1 or observed.shape != clean.shape or clean.dtype.kind not in "iu" or observed.dtype.kind not in "iu":
        raise ValueError("clean_labels and observed_labels must be aligned integer vectors")
    k = int(num_classes or max(int(clean.max()), int(observed.max())) + 1)
    clean = _validate_labels(clean, len(clean), k, "clean_labels")
    observed = _validate_labels(observed, len(clean), k, "observed_labels")
    counts = np.bincount(clean, minlength=k).astype(np.float64)
    matrix = np.zeros((k, k), dtype=np.float64)
    np.add.at(matrix, (clean, observed), 1.0)
    transition = np.zeros_like(matrix)
    present = counts > 0
    transition[present] = matrix[present] / counts[present, None]
    return {
        "clean_prior": counts / max(float(counts.sum()), 1.0),
        "transition": transition,
        "class_counts": counts.astype(np.int64),
        "noise_mask": clean != observed,
    }


def evaluate_transition_estimate(
    estimate: Dict[str, object],
    clean_labels: np.ndarray,
    observed_labels: np.ndarray,
) -> Dict[str, object]:
    """Compare a frozen estimate with clean truth; call only after estimation."""
    truth = transition_truth_from_labels(clean_labels, observed_labels)
    pi_hat = np.asarray(estimate["clean_prior"], dtype=np.float64)
    t_hat = np.asarray(estimate["transition"], dtype=np.float64)
    pi_true = truth["clean_prior"]
    t_true = truth["transition"]
    present = truth["class_counts"] > 0
    row_tv = 0.5 * np.abs(t_hat[present] - t_true[present]).sum(axis=1)
    pi_eps = 1e-12
    pi_kl = float(np.sum(pi_true * np.log((pi_true + pi_eps) / (pi_hat + pi_eps))))
    estimated_rate = float(1.0 - np.dot(pi_hat, np.diag(t_hat)))
    true_rate = float(np.mean(truth["noise_mask"]))

    clean_counts = truth["class_counts"]
    groups = {
        "Many": np.flatnonzero(clean_counts > 100),
        "Medium": np.flatnonzero((clean_counts >= 20) & (clean_counts <= 100)),
        "Few": np.flatnonzero(clean_counts < 20),
    }
    group_results = {}
    clean = np.asarray(clean_labels, dtype=np.int64)
    noisy = np.asarray(observed_labels, dtype=np.int64)
    for name, class_ids in groups.items():
        if class_ids.size == 0:
            group_results[name] = {"classes": 0, "samples": 0, "estimated_noise_rate": None, "true_noise_rate": None}
            continue
        mask = np.isin(clean, class_ids)
        pi_mass = float(pi_hat[class_ids].sum())
        estimate_rate = (
            1.0 - float(np.dot(pi_hat[class_ids], np.diag(t_hat)[class_ids])) / max(pi_mass, 1e-12)
        )
        group_results[name] = {
            "classes": int(class_ids.size),
            "samples": int(mask.sum()),
            "estimated_noise_rate": estimate_rate,
            "true_noise_rate": float(np.mean(clean[mask] != noisy[mask])) if mask.any() else None,
        }

    return {
        "n": int(len(clean)),
        "num_classes": int(len(pi_hat)),
        "estimated_global_noise_rate": estimated_rate,
        "true_global_noise_rate": true_rate,
        "transition_mae": float(np.mean(np.abs(t_hat[present] - t_true[present]))),
        "transition_row_total_variation_mean": float(np.mean(row_tv)),
        "transition_diagonal_mae": float(np.mean(np.abs(np.diag(t_hat)[present] - np.diag(t_true)[present]))),
        "clean_prior_mae": float(np.mean(np.abs(pi_hat - pi_true))),
        "clean_prior_kl_true_to_estimated": pi_kl,
        "group_noise_rates": group_results,
        "present_clean_classes": int(present.sum()),
    }


def softmax_numpy(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("logits must be a finite 2-D array")
    x = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def q_from_logits(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    clean_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return raw q, or prior-calibrated q with pi added exactly once.

    For balanced raw logits, the calibrated posterior is
    softmax(logits / temperature + log(pi)). The prior is not temperature-scaled.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    x = np.asarray(logits, dtype=np.float64) / float(temperature)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("logits must be a finite 2-D array")
    if clean_prior is not None:
        pi = np.asarray(clean_prior, dtype=np.float64)
        if pi.shape != (x.shape[1],) or not np.isfinite(pi).all() or np.any(pi <= 0):
            raise ValueError("clean_prior must be a positive finite vector of length K")
        pi = pi / pi.sum()
        x = x + np.log(pi)[None, :]
    return softmax_numpy(x)


def probability_metrics(
    probabilities: np.ndarray,
    clean_labels: np.ndarray,
    class_counts: np.ndarray,
) -> Dict[str, object]:
    q = _validate_probabilities(probabilities)
    clean = _validate_labels(clean_labels, len(q), q.shape[1], "clean_labels")
    counts = np.asarray(class_counts, dtype=np.int64)
    if counts.shape != (q.shape[1],):
        raise ValueError("class_counts must have length K")
    prediction = q.argmax(axis=1)
    entropy = -(q * np.log(np.maximum(q, 1e-12))).sum(axis=1)
    nll = -np.log(np.maximum(q[np.arange(len(q)), clean], 1e-12))
    groups = {
        "Overall": np.ones(len(clean), dtype=bool),
        "Many": counts[clean] > 100,
        "Medium": (counts[clean] >= 20) & (counts[clean] <= 100),
        "Few": counts[clean] < 20,
    }
    return {
        name: {
            "n": int(mask.sum()),
            "accuracy_percent": 100.0 * float(np.mean(prediction[mask] == clean[mask])) if mask.any() else None,
            "nll": float(np.mean(nll[mask])) if mask.any() else None,
            "entropy": float(np.mean(entropy[mask])) if mask.any() else None,
        }
        for name, mask in groups.items()
    }


def soft_bcl_batch_diagnostics(
    probabilities: np.ndarray,
    *,
    support_floor_ratio: float = 0.0,
    denominator_epsilon: float = 1e-6,
    semantic_ramp: float = 1.0,
) -> Dict[str, np.ndarray | float]:
    """Mirror Soft-BCL's q-derived weights and expose the rare-prototype pressure.

    support_floor_ratio=0 reproduces the v1 class-occurrence calculation.
    A positive ratio floors class occurrence at ratio * mean occurrence. This
    is an explicit ablation, not a silent change to the v1 objective.
    """
    q = _validate_probabilities(probabilities)
    b, k = q.shape
    if not math.isfinite(support_floor_ratio) or support_floor_ratio < 0:
        raise ValueError("support_floor_ratio must be finite and non-negative")
    if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
        raise ValueError("denominator_epsilon must be finite and positive")
    if not math.isfinite(semantic_ramp) or not 0.0 < semantic_ramp <= 1.0:
        raise ValueError("semantic_ramp must be in (0, 1]")

    min_probability = q.min(axis=1)
    max_probability = q.max(axis=1)
    reliability = np.clip((max_probability - 1.0 / k) / (1.0 - 1.0 / k), 0.0, 1.0)
    reliability[max_probability == min_probability] = 0.0
    prototype_q = semantic_ramp * np.eye(k, dtype=np.float64)
    candidate_q = np.concatenate((q, q, prototype_q), axis=0)
    anchor_q = np.concatenate((q, q), axis=0)
    candidate_reliability = np.concatenate((reliability, reliability, np.ones(k)), axis=0)
    anchor_reliability = np.concatenate((reliability, reliability), axis=0)

    positive = anchor_q @ candidate_q.T
    positive *= anchor_reliability[:, None] * candidate_reliability[None, :]
    positive[:, : 2 * b] *= semantic_ramp
    positive[np.arange(2 * b), (np.arange(2 * b) + b) % (2 * b)] = 1.0

    mask = np.ones((2 * b, 3 * b + k), dtype=np.float64)
    mask[np.arange(2 * b), np.arange(2 * b)] = 0.0
    mask[:, 2 * b :] *= semantic_ramp
    positive *= mask

    class_occurrences = candidate_q.sum(axis=0)
    floor_value = 0.0
    effective_occurrences = class_occurrences.copy()
    if support_floor_ratio > 0:
        floor_value = max(1.0, float(support_floor_ratio) * float(class_occurrences.mean()))
        effective_occurrences = np.maximum(effective_occurrences, floor_value)
    candidate_class_mass = candidate_q @ effective_occurrences
    denominator_weights = np.maximum(
        candidate_class_mass[None, :] - positive, float(denominator_epsilon)
    )
    prototype_denominator = denominator_weights[:, 2 * b :]
    prototype_inverse = 1.0 / prototype_denominator
    prototype_positive = positive[:, 2 * b :]
    return {
        "class_occurrences": class_occurrences,
        "effective_class_occurrences": effective_occurrences,
        "class_mass_floor": float(floor_value),
        "prototype_positive_mass_by_class": prototype_positive.sum(axis=0),
        "prototype_inverse_denominator_by_class": prototype_inverse.mean(axis=0),
        "prototype_denominator_by_class": prototype_denominator.mean(axis=0),
        "positive_mass_per_anchor": positive.sum(axis=1),
        "denominator_weight_min": float(denominator_weights.min()),
        "denominator_weight_median": float(np.median(denominator_weights)),
        "denominator_weight_max": float(denominator_weights.max()),
    }


def prototype_geometry(
    classifier_weight: torch.Tensor,
    prototype_mlp: nn.Module,
) -> Dict[str, object]:
    """Compare prototypes from raw W and direction-normalized W without mutation."""
    if classifier_weight.ndim != 2 or not classifier_weight.is_floating_point():
        raise ValueError("classifier_weight must be a floating-point [K,D] tensor")
    if not bool(torch.isfinite(classifier_weight).all()):
        raise ValueError("classifier_weight contains non-finite values")
    was_training = prototype_mlp.training
    prototype_mlp.eval()
    try:
        with torch.inference_mode():
            raw = F.normalize(prototype_mlp(classifier_weight), dim=1)
            direction_input = F.normalize(classifier_weight, dim=1)
            directional = F.normalize(prototype_mlp(direction_input), dim=1)

            def summarize(prototypes):
                cosine = prototypes @ prototypes.T
                k = cosine.shape[0]
                offdiag = cosine[~torch.eye(k, device=cosine.device, dtype=torch.bool)]
                per_class_max = cosine.masked_fill(
                    torch.eye(k, device=cosine.device, dtype=torch.bool), -1.0
                ).max(dim=1).values
                return {
                    "mean_off_diagonal_cosine": float(offdiag.mean().cpu()),
                    "median_off_diagonal_cosine": float(offdiag.median().cpu()),
                    "max_off_diagonal_cosine": float(offdiag.max().cpu()),
                    "per_class_max_off_diagonal_cosine": per_class_max.cpu().tolist(),
                }

            norms = classifier_weight.norm(dim=1)
            return {
                "classifier_weight_norm": norms.cpu().tolist(),
                "classifier_weight_norm_min": float(norms.min().cpu()),
                "classifier_weight_norm_median": float(norms.median().cpu()),
                "classifier_weight_norm_max": float(norms.max().cpu()),
                "raw_weight_input": summarize(raw),
                "direction_normalized_weight_input": summarize(directional),
            }
    finally:
        prototype_mlp.train(was_training)


class EMATeacher:
    """EMA of parameters with student BatchNorm buffers copied exactly."""

    def __init__(self, student: nn.Module, decay: float = 0.999):
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        self.model = copy.deepcopy(student).eval()
        self.decay = float(decay)
        self.num_updates = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        student_parameters = dict(student.named_parameters())
        teacher_parameters = dict(self.model.named_parameters())
        if student_parameters.keys() != teacher_parameters.keys():
            raise ValueError("Student and EMA teacher parameter names differ")
        for name, teacher_parameter in teacher_parameters.items():
            source = student_parameters[name].detach().to(teacher_parameter)
            teacher_parameter.mul_(self.decay).add_(source, alpha=1.0 - self.decay)

        student_buffers = dict(student.named_buffers())
        teacher_buffers = dict(self.model.named_buffers())
        if student_buffers.keys() != teacher_buffers.keys():
            raise ValueError("Student and EMA teacher buffer names differ")
        for name, teacher_buffer in teacher_buffers.items():
            teacher_buffer.copy_(student_buffers[name].detach().to(teacher_buffer))
        self.num_updates += 1
        self.model.eval()

    def state_dict(self) -> Dict[str, object]:
        return {
            "model_state": self.model.state_dict(),
            "decay": self.decay,
            "num_updates": self.num_updates,
            "buffer_policy": "copy_student_buffers_each_update",
        }


class EstimatedTransitionForwardCorrection(nn.Module):
    """Forward correction using estimated clean prior pi and arbitrary row-stochastic T."""

    def __init__(self, clean_prior: np.ndarray, transition: np.ndarray, epsilon: float = 1e-8):
        super().__init__()
        pi = torch.as_tensor(clean_prior, dtype=torch.float64)
        t = torch.as_tensor(transition, dtype=torch.float64)
        if pi.ndim != 1 or pi.numel() < 2 or not bool(torch.isfinite(pi).all()) or bool(torch.any(pi <= 0)):
            raise ValueError("clean_prior must be a positive finite 1-D vector")
        k = int(pi.numel())
        if t.shape != (k, k) or not bool(torch.isfinite(t).all()) or bool(torch.any(t < 0)):
            raise ValueError("transition must be a finite non-negative KxK matrix")
        if not torch.allclose(t.sum(dim=1), torch.ones(k, dtype=t.dtype), atol=1e-7, rtol=0):
            raise ValueError("transition rows must sum to one")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        pi = pi / pi.sum()
        self.register_buffer("log_prior", pi.log().to(torch.float32))
        self.register_buffer("log_transition", t.clamp_min(float(epsilon)).log().to(torch.float32))

    def clean_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.log_prior.numel():
            raise ValueError("logits must have shape [batch, K]")
        return F.softmax(logits + self.log_prior.to(logits), dim=1)

    def observed_log_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        log_clean = F.log_softmax(logits + self.log_prior.to(logits), dim=1)
        log_t = self.log_transition.to(logits)
        return torch.logsumexp(log_clean.unsqueeze(2) + log_t.unsqueeze(0), dim=1)

    def forward(self, logits: torch.Tensor, observed_targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.log_prior.numel():
            raise ValueError("logits must have shape [batch, K]")
        if observed_targets.ndim != 1 or observed_targets.shape[0] != logits.shape[0]:
            raise ValueError("observed_targets must have shape [batch]")
        if observed_targets.is_floating_point() or observed_targets.is_complex() or observed_targets.dtype == torch.bool:
            raise ValueError("observed_targets must contain integer class IDs")
        targets = observed_targets.to(device=logits.device, dtype=torch.long)
        if bool(torch.any((targets < 0) | (targets >= logits.shape[1]))):
            raise ValueError("observed target outside class range")
        log_clean = F.log_softmax(logits + self.log_prior.to(logits), dim=1)
        selected_log_t = self.log_transition.to(logits)[:, targets].T
        return -torch.logsumexp(log_clean + selected_log_t, dim=1).mean()


class NoiseAwareSoftBalancedContrastiveLossV2(nn.Module):
    """Soft-BCL with explicit support floor and gradual semantic/prototype ramp.

    support_floor_ratio=0 and semantic_ramp=1 preserve v1's exact one-hot BCL
    equivalence. A positive support floor is an explicit v2 ablation that
    reduces the excessive inverse-occurrence weight of weakly supported classes.
    """

    def __init__(
        self,
        num_classes: int = 100,
        temperature: float = 0.1,
        *,
        support_floor_ratio: float = 0.0,
        denominator_epsilon: float = 1e-6,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(support_floor_ratio) or support_floor_ratio < 0:
            raise ValueError("support_floor_ratio must be finite and non-negative")
        if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
            raise ValueError("denominator_epsilon must be finite and positive")
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.support_floor_ratio = float(support_floor_ratio)
        self.denominator_epsilon = float(denominator_epsilon)

    @staticmethod
    def confidence_reliability(q: torch.Tensor) -> torch.Tensor:
        k = q.shape[1]
        minimum, maximum = torch.aminmax(q, dim=1)
        reliability = ((maximum - 1.0 / k) / (1.0 - 1.0 / k)).clamp(0.0, 1.0)
        return reliability.masked_fill(maximum == minimum, 0.0)

    def _instance_only_loss(self, two_view_features: torch.Tensor) -> torch.Tensor:
        b = int(two_view_features.shape[0])
        features = F.normalize(torch.cat((two_view_features[:, 0], two_view_features[:, 1]), dim=0), dim=1)
        logits = features @ features.T / self.temperature
        n = 2 * b
        diagonal = torch.eye(n, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
        indices = torch.arange(n, device=logits.device)
        positives = (indices + b) % n
        return F.cross_entropy(logits, positives)

    def forward(
        self,
        class_prototypes: Optional[torch.Tensor],
        two_view_features: torch.Tensor,
        soft_targets: Optional[torch.Tensor],
        *,
        semantic_ramp: float = 1.0,
    ) -> torch.Tensor:
        if not math.isfinite(semantic_ramp) or not 0.0 <= semantic_ramp <= 1.0:
            raise ValueError("semantic_ramp must be in [0, 1]")
        if two_view_features.ndim != 3 or two_view_features.shape[1] != 2 or two_view_features.shape[0] == 0:
            raise ValueError("two_view_features must have shape [nonempty batch, 2, D]")
        if not two_view_features.is_floating_point() or not bool(torch.isfinite(two_view_features).all()):
            raise ValueError("two_view_features must be finite floating-point values")
        if semantic_ramp == 0.0:
            return self._instance_only_loss(two_view_features)

        b, _, d = two_view_features.shape
        if class_prototypes is None or class_prototypes.shape != (self.num_classes, d):
            raise ValueError("class_prototypes must have shape [K, D]")
        if soft_targets is None or soft_targets.shape != (b, self.num_classes):
            raise ValueError("soft_targets must have shape [B, K]")
        if not bool(torch.isfinite(class_prototypes).all()):
            raise ValueError("class_prototypes must be finite")
        q = soft_targets.detach().to(two_view_features)
        if not bool(torch.isfinite(q).all() & (q >= 0).all()):
            raise ValueError("soft_targets must be finite and non-negative")
        q_mass = q.sum(dim=1, keepdim=True)
        if not bool((q_mass > 0).all()):
            raise ValueError("every soft-target row must have positive mass")
        q = q / q_mass
        reliability = self.confidence_reliability(q)

        prototype_q = semantic_ramp * torch.eye(self.num_classes, device=q.device, dtype=q.dtype)
        candidate_q = torch.cat((q, q, prototype_q), dim=0)
        anchor_q = torch.cat((q, q), dim=0)
        candidate_reliability = torch.cat((
            reliability, reliability,
            torch.ones(self.num_classes, device=q.device, dtype=q.dtype),
        ))
        anchor_reliability = torch.cat((reliability, reliability))

        n_anchors = 2 * b
        mask = torch.ones((n_anchors, n_anchors + self.num_classes), device=q.device, dtype=q.dtype)
        anchor_ids = torch.arange(n_anchors, device=q.device)
        mask[anchor_ids, anchor_ids] = 0.0
        mask[:, n_anchors:] *= semantic_ramp

        positive = anchor_q @ candidate_q.T
        positive = positive * anchor_reliability[:, None] * candidate_reliability[None, :]
        positive[:, :n_anchors] *= semantic_ramp
        paired = (anchor_ids + b) % n_anchors
        positive[anchor_ids, paired] = 1.0
        positive = positive * mask

        class_occurrences = candidate_q.sum(dim=0)
        effective_occurrences = class_occurrences
        if self.support_floor_ratio > 0:
            floor_value = torch.maximum(
                class_occurrences.new_tensor(1.0),
                class_occurrences.mean() * self.support_floor_ratio,
            )
            effective_occurrences = torch.maximum(class_occurrences, floor_value)
        candidate_class_mass = candidate_q @ effective_occurrences
        denominator_weight = (candidate_class_mass.unsqueeze(0) - positive).clamp_min(
            self.denominator_epsilon
        )

        samples = torch.cat((two_view_features[:, 0], two_view_features[:, 1]), dim=0)
        all_features = torch.cat((samples, class_prototypes.to(two_view_features)), dim=0)
        logits = samples @ all_features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * mask
        denominator = (exp_logits / denominator_weight).sum(dim=1, keepdim=True)
        if not bool(torch.isfinite(denominator).all()) or bool(torch.any(denominator <= 0)):
            raise RuntimeError("v2 Soft-BCL denominator is non-finite or non-positive")
        log_prob = logits - torch.log(denominator.clamp_min(torch.finfo(logits.dtype).tiny))
        positive_mass = positive.sum(dim=1)
        if bool(torch.any(positive_mass <= 0)):
            raise RuntimeError("Every v2 Soft-BCL anchor must retain its paired-view positive")
        loss = -((positive * log_prob).sum(dim=1) / positive_mass).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("v2 Soft-BCL loss is non-finite")
        return loss
