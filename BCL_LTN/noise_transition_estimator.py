"""Unknown-transition estimation for NTC-BCL v2 (audit-stage implementation).

Convention: T[c, j] = P(observed label=j | clean class=c).

The estimator accepts only teacher probabilities and observed labels. It never
accepts a noise rate, clean labels, a corruption mask, or a true transition
matrix. For balanced_posterior, the estimated clean prior is multiplied once
inside the E-step. For long_tailed_posterior, the supplied probabilities
already include a class prior and are not multiplied by pi again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


PosteriorKind = Literal["balanced_posterior", "long_tailed_posterior"]


@dataclass
class TransitionEstimate:
    transition: torch.Tensor
    clean_prior: torch.Tensor
    posterior_clean: torch.Tensor
    row_support: torch.Tensor
    global_diagonal_retention: torch.Tensor
    global_support_class_ids: torch.Tensor
    iterations: int
    converged: bool
    final_delta: float


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


def _global_template(
    row_counts: torch.Tensor,
    row_support: torch.Tensor,
    *,
    high_support_fraction: float,
    min_high_support_classes: int,
    eps: float,
):
    """Build an empirical-Bayes template from the highest-support TRAIN rows.

    Selection is proportional to the number of classes, not an absolute sample
    threshold. This keeps the rule meaningful when the estimator is applied to
    a full TRAIN set or a differently sized audit subset.
    """
    k = int(row_counts.shape[0])
    positive = row_support > eps
    if not bool(positive.any()):
        diag = torch.as_tensor(1.0 / k, device=row_counts.device, dtype=row_counts.dtype)
        eye = torch.eye(k, device=row_counts.device, dtype=row_counts.dtype)
        template = eye * diag + (1.0 - eye) * ((1.0 - diag) / (k - 1))
        return template, diag, torch.arange(k, device=row_counts.device)

    n_positive = int(positive.sum().item())
    n_high = min(
        n_positive,
        max(int(min_high_support_classes), int(math.ceil(k * float(high_support_fraction)))),
    )
    ranked_support = row_support.masked_fill(~positive, -torch.inf)
    high_ids = torch.topk(ranked_support, k=n_high, largest=True, sorted=True).indices
    reliable = torch.zeros(k, dtype=torch.bool, device=row_counts.device)
    reliable[high_ids] = True

    row_probs = _normalize_rows(row_counts, eps=eps)
    weights = row_support * reliable.to(row_support.dtype)
    weight_sum = weights.sum()
    if bool(weight_sum <= eps):
        # Defensive fallback: use all rows with positive support.
        reliable = positive
        high_ids = torch.nonzero(reliable, as_tuple=False).flatten()
        weights = row_support * reliable.to(row_support.dtype)
        weight_sum = weights.sum()
    if bool(weight_sum <= eps):
        raise RuntimeError("No effective support for the global transition template")

    eye = torch.eye(k, dtype=row_probs.dtype, device=row_probs.device)
    diagonal = (row_probs.diag() * weights).sum() / weight_sum.clamp_min(eps)
    diagonal = diagonal.clamp(eps, 1.0 - eps)

    off_diagonal = row_probs * (1.0 - eye)
    pooled_destinations = (off_diagonal * weights.unsqueeze(1)).sum(0)
    template = torch.zeros_like(row_probs)
    for clean_class in range(k):
        destinations = pooled_destinations.clone()
        destinations[clean_class] = 0.0
        destination_mass = destinations.sum()
        if bool(destination_mass <= eps):
            destinations.fill_(1.0)
            destinations[clean_class] = 0.0
            destination_mass = destinations.sum()
        template[clean_class] = (1.0 - diagonal) * destinations / destination_mass.clamp_min(eps)
        template[clean_class, clean_class] = diagonal

    return _normalize_rows(template, eps=eps), diagonal, high_ids


@torch.no_grad()
def estimate_transition_and_prior(
    teacher_probabilities: torch.Tensor,
    observed_labels: torch.Tensor,
    *,
    posterior_kind: PosteriorKind,
    initial_prior: torch.Tensor | None = None,
    shrinkage_mass: float = 10.0,
    high_support_fraction: float = 0.35,
    min_high_support_classes: int = 5,
    max_iter: int = 50,
    tolerance: float = 1e-6,
    prior_pseudocount: float = 0.05,
    eps: float = 1e-8,
) -> TransitionEstimate:
    """Estimate T, clean prior pi and gamma using soft EM + empirical Bayes.

    Args:
        teacher_probabilities: [N,K] probabilities with explicitly declared
            semantics. No logits or class priors are inferred implicitly.
        observed_labels: [N] noisy/observed class IDs.
        posterior_kind: balanced_posterior multiplies pi once in the E-step;
            long_tailed_posterior does not multiply pi again.
        high_support_fraction: fraction of supported classes used to construct
            the global transition template (default: highest-support 35%).
        max_iter/tolerance: convergence-controlled EM, capped at 50 by default.

    This function is deliberately TRAIN-only and has no API for clean truth,
    corruption masks, known noise rates, or oracle transition matrices.
    """
    probs, observed = _validate_inputs(teacher_probabilities.detach(), observed_labels)
    if posterior_kind not in ("balanced_posterior", "long_tailed_posterior"):
        raise ValueError("posterior_kind must explicitly name probability semantics")
    scalar_values = (shrinkage_mass, high_support_fraction, tolerance, prior_pseudocount, eps)
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise ValueError("estimator hyperparameters must be finite")
    if shrinkage_mass < 0 or prior_pseudocount < 0:
        raise ValueError("shrinkage_mass and prior_pseudocount must be non-negative")
    if not 0.0 < high_support_fraction <= 1.0:
        raise ValueError("high_support_fraction must be in (0,1]")
    if min_high_support_classes < 1 or max_iter < 1 or tolerance <= 0 or eps <= 0:
        raise ValueError("min_high_support_classes, max_iter, tolerance and eps must be positive")

    n, k = probs.shape
    dtype, device = probs.dtype, probs.device
    if initial_prior is None:
        prior = torch.full((k,), 1.0 / k, dtype=dtype, device=device)
    else:
        prior = torch.as_tensor(initial_prior, dtype=dtype, device=device).detach().clone()
        if (prior.shape != (k,) or not bool(torch.isfinite(prior).all())
                or bool((prior <= 0).any()) or bool(prior.sum() <= 0)):
            raise ValueError("initial_prior must be a finite positive [K] vector")
        prior = prior / prior.sum()

    observed_one_hot = F.one_hot(observed, k).to(dtype=dtype)
    seed_counts = probs.T @ observed_one_hot
    support = probs.sum(0)
    template, global_diagonal, high_ids = _global_template(
        seed_counts,
        support,
        high_support_fraction=high_support_fraction,
        min_high_support_classes=min_high_support_classes,
        eps=eps,
    )
    empirical = _normalize_rows(seed_counts, eps=eps)
    shrink = support / (support + float(shrinkage_mass)).clamp_min(eps)
    transition = _normalize_rows(
        shrink.unsqueeze(1) * empirical + (1.0 - shrink).unsqueeze(1) * template,
        eps=eps,
    )

    converged = False
    final_delta = float("inf")
    actual_iterations = 0

    for iteration in range(1, int(max_iter) + 1):
        previous_prior = prior.clone()
        previous_transition = transition.clone()

        clean_evidence = (
            probs * prior.unsqueeze(0)
            if posterior_kind == "balanced_posterior"
            else probs
        )
        likelihood = transition[:, observed].T
        joint = clean_evidence * likelihood
        posterior_mass = joint.sum(1, keepdim=True)
        if bool((posterior_mass <= eps).any()):
            raise RuntimeError("E-step posterior has zero probability mass")
        posterior = joint / posterior_mass

        weighted_counts = posterior.T @ observed_one_hot
        support = posterior.sum(0)
        template, global_diagonal, high_ids = _global_template(
            weighted_counts,
            support,
            high_support_fraction=high_support_fraction,
            min_high_support_classes=min_high_support_classes,
            eps=eps,
        )
        empirical = _normalize_rows(weighted_counts, eps=eps)
        shrink = support / (support + float(shrinkage_mass)).clamp_min(eps)
        transition = _normalize_rows(
            shrink.unsqueeze(1) * empirical + (1.0 - shrink).unsqueeze(1) * template,
            eps=eps,
        )

        prior = posterior.sum(0) + float(prior_pseudocount)
        prior = prior.clamp_min(eps)
        prior = prior / prior.sum()

        transition_delta = (transition - previous_transition).abs().max()
        prior_delta = (prior - previous_prior).abs().max()
        final_delta = float(torch.maximum(transition_delta, prior_delta).item())
        actual_iterations = iteration
        if final_delta <= float(tolerance):
            converged = True
            break

    # Recompute gamma/support against the final (pi,T), not the preceding EM state.
    clean_evidence = probs * prior.unsqueeze(0) if posterior_kind == "balanced_posterior" else probs
    joint = clean_evidence * transition[:, observed].T
    posterior_mass = joint.sum(1, keepdim=True)
    if bool((posterior_mass <= eps).any()):
        raise RuntimeError("Final posterior has zero probability mass")
    posterior = joint / posterior_mass
    support = posterior.sum(0)

    return TransitionEstimate(
        transition=transition,
        clean_prior=prior,
        posterior_clean=posterior,
        row_support=support,
        global_diagonal_retention=global_diagonal,
        global_support_class_ids=high_ids,
        iterations=actual_iterations,
        converged=converged,
        final_delta=final_delta,
    )


@torch.no_grad()
def prototype_health_diagnostics(
    classifier_weight: torch.Tensor,
    prototype_embedding: torch.Tensor,
    *,
    small_weight_threshold: float = 0.15,
):
    """Measure classifier row norms and prototype angular collapse without mutation."""
    if classifier_weight.ndim != 2 or prototype_embedding.ndim != 2:
        raise ValueError("classifier_weight and prototype_embedding must be matrices")
    if classifier_weight.shape[0] != prototype_embedding.shape[0]:
        raise ValueError("classifier and prototype class counts must match")
    if not bool(torch.isfinite(classifier_weight).all() & torch.isfinite(prototype_embedding).all()):
        raise ValueError("classifier weights and prototype embeddings must be finite")
    w_norm = classifier_weight.detach().norm(dim=1)
    # ProjectionMLP.forward() already returns unit-normalized prototypes.
    # Reuse that output directly; do not add another normalization stage.
    p = prototype_embedding.detach()
    prototype_norm = p.norm(dim=1)
    if not torch.allclose(
        prototype_norm,
        torch.ones_like(prototype_norm),
        atol=1e-3,
        rtol=1e-3,
    ):
        raise ValueError("prototype_embedding must already be unit-normalized by ProjectionMLP")
    cosine = p @ p.T
    k = cosine.shape[0]
    offdiag = ~torch.eye(k, dtype=torch.bool, device=cosine.device)
    per_class_max = cosine.masked_fill(~offdiag, -torch.inf).max(1).values
    return {
        "classifier_weight_norm": w_norm,
        "small_weight_mask": w_norm < float(small_weight_threshold),
        "prototype_output_norm": prototype_norm,
        "prototype_pairwise_cosine": cosine,
        "prototype_max_offdiag_cosine": per_class_max,
        "mean_offdiag_cosine": cosine[offdiag].mean(),
    }
