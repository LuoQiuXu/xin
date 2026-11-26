"""Unknown-transition estimation for NTC-BCL v2 (audit-stage implementation).

Convention: T[c, j] = P(observed label=j | clean class=c).
All estimation inputs are TRAIN-only teacher probabilities and observed labels.
No clean labels, corruption masks, test data, or supplied noise rate are accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


PosteriorKind = Literal["balanced_posterior", "long_tailed_posterior"]


@dataclass
class TransitionEstimate:
    transition: torch.Tensor
    clean_prior: torch.Tensor
    posterior_clean: torch.Tensor
    row_support: torch.Tensor
    global_diagonal_retention: torch.Tensor
    iterations: int


def _validate_inputs(probabilities: torch.Tensor, observed: torch.Tensor):
    if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [N,K], N>0, K>=2")
    if not probabilities.is_floating_point() or not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities must be finite floating-point values")
    if bool((probabilities < 0).any()):
        raise ValueError("probabilities must be non-negative")
    row_sum = probabilities.sum(1, keepdim=True)
    if bool((row_sum <= 0).any()):
        raise ValueError("every probability row must have positive mass")
    if observed.ndim != 1 or observed.shape[0] != probabilities.shape[0]:
        raise ValueError("observed must have shape [N]")
    if observed.is_floating_point() or observed.is_complex() or observed.dtype == torch.bool:
        raise ValueError("observed labels must be integer class IDs")
    observed = observed.to(device=probabilities.device, dtype=torch.long)
    k = probabilities.shape[1]
    if bool(((observed < 0) | (observed >= k)).any()):
        raise ValueError("observed label outside class range")
    return probabilities / row_sum, observed


def _normalize_rows(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    matrix = matrix.clamp_min(0)
    return matrix / matrix.sum(1, keepdim=True).clamp_min(eps)


def _global_template(row_counts: torch.Tensor, row_support: torch.Tensor, *,
                     min_global_support: float, eps: float):
    """Build a class-aware shrinkage template from supported rows."""
    reliable = row_support >= float(min_global_support)
    if not bool(reliable.any()):
        reliable = row_support > 0
    if not bool(reliable.any()):
        k = row_counts.shape[0]
        diag = torch.tensor(1.0 / k, device=row_counts.device, dtype=row_counts.dtype)
        eye = torch.eye(k, device=row_counts.device, dtype=row_counts.dtype)
        template = eye * diag + (1 - eye) * ((1 - diag) / (k - 1))
        return template, diag

    row_probs = _normalize_rows(row_counts)
    weights = row_support * reliable.to(row_support.dtype)
    diag = (row_probs.diag() * weights).sum() / weights.sum().clamp_min(eps)
    k = row_counts.shape[0]
    off_mass = row_probs * (1 - torch.eye(k, device=row_probs.device, dtype=row_probs.dtype))
    pooled = (off_mass * weights[:, None]).sum(0)
    template = torch.zeros_like(row_probs)
    for c in range(k):
        destinations = pooled.clone()
        destinations[c] = 0
        total = destinations.sum()
        if bool(total <= eps):
            destinations.fill_(1.0)
            destinations[c] = 0
            total = destinations.sum()
        template[c] = destinations / total.clamp_min(eps) * (1 - diag)
        template[c, c] = diag
    return _normalize_rows(template), diag.clamp(eps, 1 - eps)


@torch.no_grad()
def estimate_transition_and_prior(
    teacher_probabilities: torch.Tensor,
    observed_labels: torch.Tensor,
    *,
    posterior_kind: PosteriorKind,
    initial_prior: torch.Tensor | None = None,
    shrinkage_mass: float = 10.0,
    min_global_support: float = 20.0,
    iterations: int = 5,
    prior_pseudocount: float = 0.05,
    eps: float = 1e-8,
) -> TransitionEstimate:
    """Estimate T and clean prior by soft EM with empirical-Bayes shrinkage.

    balanced_posterior approximates p_bal(c|x), so multiply by clean prior once.
    long_tailed_posterior already approximates p_LT(c|x), so do not multiply it again.

    This is an estimator audit primitive, not yet wired into the v1 training loop.
    """
    probs, observed = _validate_inputs(teacher_probabilities.detach(), observed_labels)
    if posterior_kind not in ("balanced_posterior", "long_tailed_posterior"):
        raise ValueError("posterior_kind must explicitly name probability semantics")
    if iterations < 1 or shrinkage_mass < 0 or min_global_support < 0 or prior_pseudocount < 0:
        raise ValueError("invalid EM or shrinkage hyperparameter")
    n, k = probs.shape
    dtype, device = probs.dtype, probs.device

    if initial_prior is None:
        prior = torch.full((k,), 1.0 / k, dtype=dtype, device=device)
    else:
        prior = torch.as_tensor(initial_prior, dtype=dtype, device=device).detach().clone()
        if prior.shape != (k,) or not bool(torch.isfinite(prior).all()) or bool((prior <= 0).any()):
            raise ValueError("initial_prior must be a finite positive [K] vector")
        prior = prior / prior.sum()

    observed_one_hot = torch.nn.functional.one_hot(observed, k).to(dtype)
    eye = torch.eye(k, dtype=dtype, device=device)
    # Data-driven initialization: teacher mass against observed labels. No eta or
    # assumed diagonal-retention rate is injected into the estimator.
    seed_counts = probs.T @ observed_one_hot
    support = probs.sum(0)
    template, global_diag = _global_template(
        seed_counts, support, min_global_support=min_global_support, eps=eps
    )
    empirical = _normalize_rows(seed_counts)
    shrink = (support / (support + float(shrinkage_mass))).unsqueeze(1)
    transition = _normalize_rows(shrink * empirical + (1 - shrink) * template)

    for _ in range(iterations):
        clean_evidence = probs * prior.unsqueeze(0) if posterior_kind == "balanced_posterior" else probs
        posterior = clean_evidence * transition[:, observed].T
        posterior = posterior / posterior.sum(1, keepdim=True).clamp_min(eps)

        weighted_counts = posterior.T @ observed_one_hot
        support = posterior.sum(0)
        template, global_diag = _global_template(
            weighted_counts, support, min_global_support=min_global_support, eps=eps
        )
        empirical = _normalize_rows(weighted_counts)
        shrink = (support / (support + float(shrinkage_mass))).unsqueeze(1)
        transition = _normalize_rows(shrink * empirical + (1 - shrink) * template)

        prior = posterior.sum(0) + float(prior_pseudocount)
        prior = prior / prior.sum()

    clean_evidence = probs * prior.unsqueeze(0) if posterior_kind == "balanced_posterior" else probs
    posterior = clean_evidence * transition[:, observed].T
    posterior = posterior / posterior.sum(1, keepdim=True).clamp_min(eps)
    return TransitionEstimate(
        transition=transition,
        clean_prior=prior,
        posterior_clean=posterior,
        row_support=support,
        global_diagonal_retention=global_diag,
        iterations=int(iterations),
    )


@torch.no_grad()
def prototype_health_diagnostics(classifier_weight: torch.Tensor,
                                 prototype_embedding: torch.Tensor,
                                 *, small_weight_threshold: float = 0.15):
    """Measure W-norm and prototype angular collapse without changing the model."""
    if classifier_weight.ndim != 2 or prototype_embedding.ndim != 2:
        raise ValueError("classifier_weight and prototype_embedding must be matrices")
    if classifier_weight.shape[0] != prototype_embedding.shape[0]:
        raise ValueError("classifier and prototype class counts must match")
    w_norm = classifier_weight.detach().norm(dim=1)
    p = torch.nn.functional.normalize(prototype_embedding.detach(), dim=1)
    cosine = p @ p.T
    k = cosine.shape[0]
    offdiag = ~torch.eye(k, dtype=torch.bool, device=cosine.device)
    per_class_max = cosine.masked_fill(~offdiag, -torch.inf).max(1).values
    return {
        "classifier_weight_norm": w_norm,
        "small_weight_mask": w_norm < float(small_weight_threshold),
        "prototype_pairwise_cosine": cosine,
        "prototype_max_offdiag_cosine": per_class_max,
        "mean_offdiag_cosine": cosine[offdiag].mean(),
    }
