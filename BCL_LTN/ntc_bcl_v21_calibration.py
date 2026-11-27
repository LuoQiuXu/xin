"""Calibration and zero-intervention diagnostics for NTC-BCL v2.1 G3.

The prior objective implemented here is exactly:
    KL(mu_obs || T.T @ pi) + lambda_h * ||pi - pi_EM||_2^2
subject to sum(pi)=1 and pi_c >= prior_min.

Important terminology note: the written quadratic tether is an L2 penalty, not
a Hellinger distance. The code preserves the requested objective rather than
silently replacing its mathematics.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch


@torch.no_grad()
def observed_fit_statistics(
    balanced_probabilities: torch.Tensor,
    observed_labels: torch.Tensor,
) -> Dict[str, object]:
    """Macro observed-label fit rates, with confidence-ranked within-class subsets.

    For each observed class j, rank its examples by max_c p_bal(c|x), then
    compute the fraction whose argmax equals j among its top 20% / top 50%.
    The all-sample statistic is also macro-averaged over observed classes.
    """
    probs = torch.as_tensor(balanced_probabilities).detach().to(torch.float64).cpu().numpy()
    observed = torch.as_tensor(observed_labels).detach().to(torch.long).cpu().numpy()
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] < 2:
        raise ValueError("balanced_probabilities must have shape [N,K], N>0, K>=2")
    if observed.shape != (probs.shape[0],):
        raise ValueError("observed_labels must have shape [N]")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("balanced_probabilities must be finite and non-negative")
    row_sum = probs.sum(axis=1)
    if np.any(row_sum <= 0):
        raise ValueError("every probability row must have positive mass")
    probs = probs / row_sum[:, None]
    k = probs.shape[1]
    if np.any((observed < 0) | (observed >= k)):
        raise ValueError("observed label outside class range")

    prediction = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    all_rates = []
    top20_rates = []
    top50_rates = []
    top20_count = 0
    top50_count = 0
    classes_seen = 0

    for cls in range(k):
        ids = np.flatnonzero(observed == cls)
        if ids.size == 0:
            continue
        classes_seen += 1
        correct = prediction[ids] == cls
        all_rates.append(float(correct.mean()))
        order = np.argsort(-confidence[ids], kind="stable")
        n20 = max(1, int(math.ceil(0.20 * ids.size)))
        n50 = max(1, int(math.ceil(0.50 * ids.size)))
        top20_ids = ids[order[:n20]]
        top50_ids = ids[order[:n50]]
        top20_rates.append(float((prediction[top20_ids] == cls).mean()))
        top50_rates.append(float((prediction[top50_ids] == cls).mean()))
        top20_count += int(n20)
        top50_count += int(n50)

    if classes_seen == 0:
        raise ValueError("no observed class has any samples")
    return {
        "f_top20": float(np.mean(top20_rates)),
        "f_top50": float(np.mean(top50_rates)),
        "f_all_macro": float(np.mean(all_rates)),
        "fit_num_observed_classes": int(classes_seen),
        "fit_top20_samples": int(top20_count),
        "fit_top50_samples": int(top50_count),
    }


def _validate_prior_calibration_inputs(
    observed_counts: torch.Tensor,
    transition: torch.Tensor,
    em_prior: torch.Tensor,
    prior_min: float,
    lambda_h: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.as_tensor(observed_counts, dtype=torch.float64).detach()
    t = torch.as_tensor(transition, dtype=torch.float64).detach().to(counts.device)
    pi_em = torch.as_tensor(em_prior, dtype=torch.float64).detach().to(counts.device)
    if counts.ndim != 1 or counts.numel() < 2:
        raise ValueError("observed_counts must be a 1-D vector with K>=2")
    k = counts.numel()
    if t.shape != (k, k) or pi_em.shape != (k,):
        raise ValueError("transition and em_prior must have shapes [K,K] and [K]")
    if (
        not bool(torch.isfinite(counts).all())
        or bool((counts < 0).any())
        or bool(counts.sum() <= 0)
    ):
        raise ValueError("observed_counts must be finite, non-negative and non-empty")
    if (
        not bool(torch.isfinite(t).all())
        or bool((t < 0).any())
        or not torch.allclose(t.sum(1), torch.ones(k, dtype=t.dtype, device=t.device), atol=1e-8, rtol=0)
    ):
        raise ValueError("transition must be finite, non-negative and row-stochastic")
    if (
        not bool(torch.isfinite(pi_em).all())
        or bool((pi_em <= 0).any())
        or bool(pi_em.sum() <= 0)
    ):
        raise ValueError("em_prior must be finite and strictly positive")
    if not math.isfinite(float(prior_min)) or prior_min < 0 or k * prior_min >= 1:
        raise ValueError("prior_min must be finite, non-negative and satisfy K*prior_min < 1")
    if not math.isfinite(float(lambda_h)) or lambda_h <= 0:
        raise ValueError("lambda_h must be finite and strictly positive for strict convexity")
    return counts, t, pi_em / pi_em.sum()


def calibrate_clean_prior(
    observed_counts: torch.Tensor,
    transition: torch.Tensor,
    em_prior: torch.Tensor,
    *,
    prior_min: float,
    lambda_h: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-8,
    eps: float = 1e-15,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Solve constrained KL marginal matching with a quadratic EM-prior tether.

    The objective is strictly convex in pi because lambda_h > 0. A softmax
    residual-mass parameterization enforces the simplex and lower bound exactly;
    optimization is performed with float64 torch L-BFGS.
    """
    if max_iter < 1 or not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise ValueError("max_iter and tolerance must be positive")
    counts, t, pi_em = _validate_prior_calibration_inputs(
        observed_counts, transition, em_prior, prior_min, lambda_h
    )
    k = counts.numel()
    floor = float(prior_min)
    residual_mass = 1.0 - k * floor
    mu_obs = counts / counts.sum()

    residual_start = (pi_em - floor).clamp_min(0.0)
    if bool(residual_start.sum() <= eps):
        residual_start = torch.full_like(residual_start, 1.0 / k)
    else:
        residual_start = residual_start / residual_start.sum()
    # pi = floor + residual_mass * softmax(theta) guarantees pi_c >= floor
    # and sum(pi)=1 for every finite theta.
    theta = torch.log(residual_start.clamp_min(eps)).detach().clone().requires_grad_(True)

    def make_prior(parameter: torch.Tensor) -> torch.Tensor:
        return floor + residual_mass * torch.softmax(parameter, dim=0)

    def objective(parameter: torch.Tensor) -> torch.Tensor:
        pi = make_prior(parameter)
        observed_marginal = t.transpose(0, 1).matmul(pi).clamp_min(eps)
        positive = mu_obs > 0
        kl = torch.sum(
            mu_obs[positive]
            * (torch.log(mu_obs[positive]) - torch.log(observed_marginal[positive]))
        )
        quadratic = torch.sum((pi - pi_em) ** 2)
        return kl + float(lambda_h) * quadratic

    initial_objective = float(objective(theta).detach().item())
    optimizer = torch.optim.LBFGS(
        [theta],
        lr=1.0,
        max_iter=int(max_iter),
        tolerance_grad=float(tolerance),
        tolerance_change=float(tolerance) * 0.1,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        optimizer.zero_grad(set_to_none=True)
        loss = objective(theta)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Prior calibration objective became non-finite")
        loss.backward()
        closure_calls += 1
        return loss

    optimizer.step(closure)
    final_loss = objective(theta)
    optimizer.zero_grad(set_to_none=True)
    final_loss.backward()
    grad_norm = float(theta.grad.detach().norm().item())
    calibrated = make_prior(theta.detach()).detach()
    final_objective = float(final_loss.detach().item())
    state = optimizer.state.get(theta, {})
    iterations = int(state.get("n_iter", 0))
    # The gradient is measured in unconstrained logits, not directly in pi.
    # A 1e-5 norm is a conservative practical stationarity check for K=100;
    # an already-optimal initialization (zero LBFGS iterations) is also converged.
    converged = bool(
        math.isfinite(grad_norm)
        and grad_norm <= max(100.0 * float(tolerance), 1e-5)
    )

    simplex_error = abs(float(calibrated.sum().item()) - 1.0)
    min_prior = float(calibrated.min().item())
    if (
        not bool(torch.isfinite(calibrated).all())
        or simplex_error > 1e-10
        or min_prior < floor - 1e-12
        or final_objective > initial_objective + 1e-10
    ):
        raise RuntimeError(
            "Constrained prior calibration failed invariant/objective checks: "
            f"sum_error={simplex_error:.3e}, min_prior={min_prior:.3e}, "
            f"objective={initial_objective:.8g}->{final_objective:.8g}"
        )
    if not converged:
        raise RuntimeError(
            "Constrained prior calibration did not converge: "
            f"grad_norm={grad_norm:.3e}, iterations={iterations}, "
            f"max_iter={max_iter}"
        )

    predicted_observed = t.transpose(0, 1).matmul(calibrated)
    positive = mu_obs > 0
    kl_final = float(
        torch.sum(
            mu_obs[positive]
            * (torch.log(mu_obs[positive]) - torch.log(predicted_observed[positive].clamp_min(eps)))
        ).item()
    )
    quadratic_final = float(torch.sum((calibrated - pi_em) ** 2).item())
    diagnostics: Dict[str, object] = {
        "solver": "torch.optim.LBFGS_float64_strong_wolfe",
        "objective": "KL(mu_obs || T.T @ pi) + lambda_h * ||pi - pi_EM||_2^2",
        "lambda_h": float(lambda_h),
        "prior_min": floor,
        "max_iter": int(max_iter),
        "iterations": iterations,
        "closure_calls": closure_calls,
        "tolerance": float(tolerance),
        "gradient_norm": grad_norm,
        "objective_initial": initial_objective,
        "objective_final": final_objective,
        "kl_final": kl_final,
        "quadratic_tether_final": quadratic_final,
        "simplex_error": simplex_error,
        "minimum_prior": min_prior,
        "converged": converged,
    }
    return calibrated, diagnostics
