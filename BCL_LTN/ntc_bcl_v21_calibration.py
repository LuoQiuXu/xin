"""Calibration and zero-intervention diagnostics for NTC-BCL v2.1 G3.

The prior objective implemented here is exactly:
    KL(mu_obs || T.T @ pi) + lambda_h * sum_c (sqrt(pi_c) - sqrt(pi_EM_c))^2
subject to sum(pi)=1 and pi_c >= prior_min.

Note: the squared Hellinger tether is used directly without a 1/2 prefactor,
providing scale-invariant relative protection for weak tail-class priors.
Optimization runs in float64 with Stage-1 strong-Wolfe L-BFGS followed, when
boundary constraints are active or primal KKT conditions are not yet satisfied,
by Stage-2 exact primal Projected Newton on the capped simplex.
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


def _project_capped_simplex(values: torch.Tensor, floor: float) -> torch.Tensor:
    """Exact Euclidean projection onto {sum(pi)=1, pi_c >= floor} in float64."""
    k = values.numel()
    residual_mass = 1.0 - k * float(floor)
    shifted = values - float(floor)
    descending, _ = torch.sort(shifted, descending=True)
    cumsum = torch.cumsum(descending, dim=0) - residual_mass
    indices = torch.arange(1, k + 1, dtype=values.dtype, device=values.device)
    positive_support = (descending - cumsum / indices) > 0
    nz = torch.nonzero(positive_support, as_tuple=False)
    rho = int(nz[-1].item()) + 1 if nz.numel() > 0 else 1
    tau = cumsum[rho - 1] / float(rho)
    projected = float(floor) + (shifted - tau).clamp_min(0.0)
    sum_err = 1.0 - float(projected.sum().item())
    if abs(sum_err) > 0.0:
        # Repair tiny float64 rounding solely on the largest slack coordinate so
        # exact active floor entries stay identically at `floor`.
        interior_idx = int(torch.argmax(projected - float(floor)).item())
        candidate = float(projected[interior_idx].item()) + sum_err
        if candidate >= float(floor):
            projected[interior_idx] = candidate
    return projected


def _evaluate_hellinger_objective_and_derivatives(
    pi: torch.Tensor,
    mu_obs: torch.Tensor,
    transition: torch.Tensor,
    pi_em: torch.Tensor,
    lambda_h: float,
    eps: float = 1e-15,
) -> Tuple[float, torch.Tensor, torch.Tensor, float, float]:
    """Compute objective, primal gradient, and analytic Hessian wrt pi."""
    sqrt_em = torch.sqrt(pi_em)
    positive = mu_obs > 0
    observed_marginal = transition.transpose(0, 1).matmul(pi)
    om_pos = observed_marginal[positive].clamp_min(eps)
    kl_tensor = torch.sum(mu_obs[positive] * (torch.log(mu_obs[positive]) - torch.log(om_pos)))

    pi_safe = pi.clamp_min(1e-30)
    sqrt_pi = torch.sqrt(pi_safe)
    hellinger_tensor = torch.sum((sqrt_pi - sqrt_em) ** 2)
    obj = float((kl_tensor + float(lambda_h) * hellinger_tensor).item())

    ratio = torch.zeros_like(mu_obs)
    active_om = observed_marginal[positive] > eps
    ratio[positive] = torch.where(active_om, mu_obs[positive] / om_pos, torch.zeros_like(om_pos))
    grad_kl = -transition.matmul(ratio)

    inv_sqrt_pi = torch.where(pi > 1e-30, 1.0 / sqrt_pi, torch.zeros_like(pi))
    grad_h = float(lambda_h) * (1.0 - sqrt_em * inv_sqrt_pi)
    grad = grad_kl + grad_h

    inv_var = torch.zeros_like(mu_obs)
    inv_var[positive] = torch.where(
        active_om, mu_obs[positive] / (om_pos * om_pos), torch.zeros_like(om_pos)
    )
    hess_kl = (transition * inv_var.unsqueeze(0)).matmul(transition.transpose(0, 1))
    hess_h_diag = 0.5 * float(lambda_h) * sqrt_em / (pi_safe ** 1.5)
    hess = hess_kl + torch.diag(hess_h_diag)
    return obj, grad, hess, float(kl_tensor.item()), float(hellinger_tensor.item())


def _primal_kkt_metrics(
    pi: torch.Tensor,
    grad: torch.Tensor,
    floor: float,
    active_tol: float = 1e-12,
) -> Dict[str, object]:
    """Measure primal constrained KKT optimality on {sum(pi)=1, pi_c >= floor}."""
    g_min = float(grad.min().item())
    slack = (pi - float(floor)).clamp_min(0.0)
    frank_wolfe_gap = float(torch.sum(slack * (grad - g_min)).item())

    free_mask = pi > float(floor) + active_tol
    active_mask = ~free_mask
    if bool(free_mask.any()):
        nu = float(grad[free_mask].mean().item())
        free_residual = float((grad[free_mask] - nu).abs().max().item())
    else:
        nu = g_min
        free_residual = 0.0
    if bool(active_mask.any()):
        active_violation = float((nu - grad[active_mask]).clamp_min(0.0).max().item())
    else:
        active_violation = 0.0
    kkt_residual = max(free_residual, active_violation)
    return {
        "dual_multiplier_nu": nu,
        "kkt_free_residual": free_residual,
        "kkt_active_violation": active_violation,
        "kkt_residual": kkt_residual,
        "frank_wolfe_gap": frank_wolfe_gap,
        "active_floor_count": int(active_mask.sum().item()),
    }


def _refine_projected_newton(
    pi_init: torch.Tensor,
    mu_obs: torch.Tensor,
    transition: torch.Tensor,
    pi_em: torch.Tensor,
    *,
    floor: float,
    lambda_h: float,
    max_iter: int,
    kkt_target: float,
    fw_target: float,
    eps: float,
) -> Tuple[torch.Tensor, int]:
    """Refine solution in primal simplex space with active-set Projected Newton."""
    pi = _project_capped_simplex(pi_init, floor)
    obj, grad, hess, _, _ = _evaluate_hellinger_objective_and_derivatives(
        pi, mu_obs, transition, pi_em, lambda_h, eps=eps
    )
    iters = 0
    for it in range(1, int(max_iter) + 1):
        kkt = _primal_kkt_metrics(pi, grad, floor)
        if kkt["kkt_residual"] <= kkt_target and kkt["frank_wolfe_gap"] <= fw_target:
            break
        iters = it

        free_for_nu = pi > float(floor) + 1e-12
        nu = (
            float(grad[free_for_nu].mean().item())
            if bool(free_for_nu.any())
            else float(grad.min().item())
        )
        max_slack = float((pi - float(floor)).clamp_min(0.0).max().item())
        eps_act = min(1e-11, max_slack * 0.1)
        binding = (pi <= float(floor) + eps_act) & (grad > nu)
        free = ~binding

        direction = torch.zeros_like(pi)
        if int(free.sum().item()) >= 2:
            h_ff = hess[free][:, free]
            g_f = grad[free]
            ones_f = torch.ones_like(g_f)
            try:
                uv = torch.linalg.solve(h_ff, torch.stack([-g_f, ones_f], dim=1))
                u, v = uv[:, 0], uv[:, 1]
                denom = float(ones_f.dot(v).item())
                if denom > 1e-15:
                    d_f = u - (float(ones_f.dot(u).item()) / denom) * v
                    if float(g_f.dot(d_f).item()) < 0.0:
                        direction[free] = d_f
            except RuntimeError:
                pass

        if float(grad.dot(direction).item()) >= -1e-16:
            lipschitz = max(float(torch.linalg.norm(hess, ord=2).item()), 1.0)
            direction = _project_capped_simplex(pi - grad / lipschitz, floor) - pi

        accepted = False
        alpha = 1.0
        for _ in range(45):
            candidate = _project_capped_simplex(pi + alpha * direction, floor)
            obj_c, grad_c, hess_c, _, _ = _evaluate_hellinger_objective_and_derivatives(
                candidate, mu_obs, transition, pi_em, lambda_h, eps=eps
            )
            slope = float(grad.dot(candidate - pi).item())
            if math.isfinite(obj_c) and obj_c <= obj + 1e-4 * slope + 1e-15:
                pi, obj, grad, hess = candidate, obj_c, grad_c, hess_c
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # Projected gradient arc backup guarantees descent whenever away from KKT.
            lipschitz = max(float(torch.linalg.norm(hess, ord=2).item()), 1.0)
            for scale in (1.0, 0.1, 0.01, 1e-4, 1e-6):
                pg_cand = _project_capped_simplex(pi - (scale / lipschitz) * grad, floor)
                obj_p, grad_p, hess_p, _, _ = _evaluate_hellinger_objective_and_derivatives(
                    pg_cand, mu_obs, transition, pi_em, lambda_h, eps=eps
                )
                if math.isfinite(obj_p) and obj_p < obj:
                    pi, obj, grad, hess = pg_cand, obj_p, grad_p, hess_p
                    accepted = True
                    break
            if not accepted:
                break

    return pi, iters


def calibrate_clean_prior(
    observed_counts: torch.Tensor,
    transition: torch.Tensor,
    em_prior: torch.Tensor,
    *,
    prior_min: float,
    lambda_h: float = 1.0,
    max_iter: int = 200,
    tolerance: float = 1e-8,
    eps: float = 1e-15,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Solve constrained KL marginal matching with a squared Hellinger tether.

    Objective (strictly convex for lambda_h > 0):
        KL(mu_obs || T.T @ pi) + lambda_h * sum_c (sqrt(pi_c) - sqrt(pi_EM_c))^2
    subject to sum(pi)=1 and pi_c >= prior_min.

    Stage 1 runs float64 L-BFGS on a softmax residual-mass parameterization.
    Because active floor constraints drive softmax logits toward -inf while
    driving logit gradients artificially small, Stage 2 verifies primal KKT
    optimality and refines via exact primal Projected Newton when needed.
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
    theta = torch.log(residual_start.clamp_min(eps)).detach().clone().requires_grad_(True)
    sqrt_em = torch.sqrt(pi_em)

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
        hellinger_sq = torch.sum((torch.sqrt(pi.clamp_min(1e-30)) - sqrt_em) ** 2)
        return kl + float(lambda_h) * hellinger_sq

    pi_start = _project_capped_simplex(make_prior(theta.detach()).detach(), floor)
    initial_objective, _, _, _, _ = _evaluate_hellinger_objective_and_derivatives(
        pi_start, mu_obs, t, pi_em, lambda_h, eps=eps
    )

    lbfgs_tol_grad = min(float(tolerance) * 1e-4, 1e-12)
    lbfgs_tol_change = min(float(tolerance) * 1e-6, 1e-14)
    optimizer = torch.optim.LBFGS(
        [theta],
        lr=1.0,
        max_iter=int(max_iter),
        tolerance_grad=lbfgs_tol_grad,
        tolerance_change=lbfgs_tol_change,
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
    stage1_state = optimizer.state.get(theta, {})
    stage1_iterations = int(stage1_state.get("n_iter", 0))
    pi_stage1 = _project_capped_simplex(make_prior(theta.detach()).detach(), floor)
    stage1_obj, stage1_grad, _, _, _ = _evaluate_hellinger_objective_and_derivatives(
        pi_stage1, mu_obs, t, pi_em, lambda_h, eps=eps
    )
    if not math.isfinite(stage1_obj) or stage1_obj > initial_objective + 1e-12:
        pi_stage1 = pi_start
        stage1_obj, stage1_grad, _, _, _ = _evaluate_hellinger_objective_and_derivatives(
            pi_stage1, mu_obs, t, pi_em, lambda_h, eps=eps
        )

    stage1_kkt = _primal_kkt_metrics(pi_stage1, stage1_grad, floor)
    kkt_target = min(float(tolerance) * 1e-2, 1e-11)
    fw_target = min(float(tolerance) * 1e-3, 1e-12)

    stage2_iterations = 0
    calibrated = pi_stage1
    solver_name = "torch.optim.LBFGS_float64_strong_wolfe"
    if stage1_kkt["kkt_residual"] > kkt_target or stage1_kkt["frank_wolfe_gap"] > fw_target:
        calibrated, stage2_iterations = _refine_projected_newton(
            pi_stage1,
            mu_obs,
            t,
            pi_em,
            floor=floor,
            lambda_h=lambda_h,
            max_iter=max_iter,
            kkt_target=kkt_target,
            fw_target=fw_target,
            eps=eps,
        )
        solver_name = "lbfgs_stage1+projected_newton_kkt_float64"

    final_objective, final_grad, _, kl_final, hellinger_final = (
        _evaluate_hellinger_objective_and_derivatives(
            calibrated, mu_obs, t, pi_em, lambda_h, eps=eps
        )
    )
    final_kkt = _primal_kkt_metrics(calibrated, final_grad, floor)

    simplex_error = abs(float(calibrated.sum().item()) - 1.0)
    min_prior = float(calibrated.min().item())
    kkt_tol = float(tolerance)
    fw_tol = max(float(tolerance), 1e-10)
    converged = bool(
        math.isfinite(final_objective)
        and math.isfinite(final_kkt["kkt_residual"])
        and math.isfinite(final_kkt["frank_wolfe_gap"])
        and final_kkt["kkt_residual"] <= kkt_tol
        and final_kkt["frank_wolfe_gap"] <= fw_tol
    )

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
            "Constrained prior calibration failed primal KKT check: "
            f"kkt_residual={final_kkt['kkt_residual']:.3e} (tol={kkt_tol:.1e}), "
            f"fw_gap={final_kkt['frank_wolfe_gap']:.3e} (tol={fw_tol:.1e}), "
            f"stage1_iter={stage1_iterations}, stage2_iter={stage2_iterations}, max_iter={max_iter}"
        )

    diagnostics: Dict[str, object] = {
        "solver": solver_name,
        "objective": "KL(mu_obs || T.T @ pi) + lambda_h * sum((sqrt(pi) - sqrt(pi_EM))^2)",
        "lambda_h": float(lambda_h),
        "prior_min": floor,
        "max_iter": int(max_iter),
        "iterations": stage1_iterations + stage2_iterations,
        "stage1_iterations": stage1_iterations,
        "stage1_closure_calls": closure_calls,
        "stage2_iterations": stage2_iterations,
        "lbfgs_tolerance_grad": lbfgs_tol_grad,
        "lbfgs_tolerance_change": lbfgs_tol_change,
        "kkt_tolerance": kkt_tol,
        "frank_wolfe_tolerance": fw_tol,
        "kkt_free_residual": final_kkt["kkt_free_residual"],
        "kkt_active_violation": final_kkt["kkt_active_violation"],
        "kkt_residual": final_kkt["kkt_residual"],
        "frank_wolfe_gap": final_kkt["frank_wolfe_gap"],
        "active_floor_count": final_kkt["active_floor_count"],
        "dual_multiplier_nu": final_kkt["dual_multiplier_nu"],
        "objective_initial": initial_objective,
        "objective_final": final_objective,
        "kl_final": kl_final,
        "hellinger_tether_final": hellinger_final,
        "simplex_error": simplex_error,
        "minimum_prior": min_prior,
        "converged": converged,
    }
    return calibrated, diagnostics
