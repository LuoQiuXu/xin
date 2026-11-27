from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from BCL_LTN.ntc_bcl_v2_losses import estimate_anchored_transition_and_prior
from BCL_LTN.ntc_bcl_v21_calibration import (
    _evaluate_hellinger_objective_and_derivatives,
    calibrate_clean_prior,
    observed_fit_statistics,
)


def test_observed_fit_statistics_are_macro_and_confidence_ranked():
    # Class 0 has 5 samples; class 1 has 3. The highest-confidence observed
    # examples are ranked by max_c p_bal(c|x) independently within each observed class.
    probs = torch.tensor(
        [
            [0.95, 0.05],  # y=0, correct, conf=0.95
            [0.80, 0.20],  # y=0, correct, conf=0.80
            [0.55, 0.45],  # y=0, correct, conf=0.55
            [0.51, 0.49],  # y=0, correct, conf=0.51
            [0.99, 0.01],  # y=1, wrong, conf=0.99
            [0.60, 0.40],  # y=1, wrong, conf=0.60
            [0.10, 0.90],  # y=1, correct, conf=0.90
            [0.20, 0.80],  # y=0, wrong, conf=0.80 (enters top-50% of y=0)
        ],
        dtype=torch.float64,
    )
    observed = torch.tensor([0, 0, 0, 0, 1, 1, 1, 0])
    stats = observed_fit_statistics(probs, observed)
    assert stats["fit_num_observed_classes"] == 2
    assert stats["f_all_macro"] == pytest.approx((4 / 5 + 1 / 3) / 2)
    # Top-20%: one sample per class. Class 0's most confident is correct;
    # class 1's most confident is wrong.
    assert stats["f_top20"] == pytest.approx(0.5)
    # Top-50%: ceil(5/2)=3 for class 0 ([0.95 ok, 0.80 ok, 0.80 wrong] -> 2/3);
    # ceil(3/2)=2 for class 1 ([0.99 wrong, 0.90 ok] -> 1/2). Macro mean = 7/12.
    assert stats["f_top50"] == pytest.approx(7.0 / 12.0)


def test_piw_anchor_support_uses_reference_prior_and_raw_soft_support():
    probs = torch.tensor(
        [
            [0.90, 0.08, 0.02],
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.20, 0.70],
        ],
        dtype=torch.float64,
    )
    observed = torch.tensor([0, 1, 1, 2])
    pi_ref = torch.tensor([0.60, 0.30, 0.10], dtype=torch.float64)
    result = estimate_anchored_transition_and_prior(
        probs,
        observed,
        reference_prior=pi_ref,
        confidence_power=2.0,
        shrinkage_mass=10.0,
        prior_pseudocount=1.0,
        max_iter=200,
    )
    q = probs * pi_ref.unsqueeze(0)
    q = q / q.sum(1, keepdim=True)
    max_prob = probs.max(1).values
    reliability = ((max_prob - 1 / 3) / (1 - 1 / 3)).clamp_min(0)
    weights = reliability.pow(2)
    expected_support = (q * weights.unsqueeze(1)).sum(0)
    torch.testing.assert_close(result.anchor_support, expected_support, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        result.transition.sum(1), torch.ones(3, dtype=torch.float64), atol=1e-10, rtol=0
    )
    assert torch.isfinite(result.anchor_support).all()


def test_prior_calibration_uses_exact_hellinger_and_kkt_optimality():
    counts = torch.tensor([700.0, 220.0, 80.0], dtype=torch.float64)
    transition = torch.tensor(
        [[0.90, 0.08, 0.02], [0.10, 0.82, 0.08], [0.05, 0.15, 0.80]],
        dtype=torch.float64,
    )
    em_prior = torch.tensor([0.80, 0.19, 0.01], dtype=torch.float64)
    prior_min = 3.0 / float(counts.sum())
    prior, diag = calibrate_clean_prior(
        counts,
        transition,
        em_prior,
        prior_min=prior_min,
        lambda_h=1.0,
        max_iter=200,
        tolerance=1e-9,
    )
    assert diag["converged"] is True
    assert "hellinger_tether_final" in diag
    assert "quadratic_tether_final" not in diag
    assert diag["objective_final"] <= diag["objective_initial"] + 1e-10
    assert diag["kkt_residual"] <= 1e-9
    assert diag["frank_wolfe_gap"] <= 1e-10
    assert float(prior.min()) >= prior_min - 1e-12
    assert float(prior.sum()) == pytest.approx(1.0, abs=1e-10)
    assert prior.dtype == torch.float64
    assert torch.isfinite(prior).all()

    # Verify exact Hellinger value (no 1/2 factor) and verify it differs from L2 solution.
    expected_hel = float(torch.sum((torch.sqrt(prior) - torch.sqrt(em_prior)) ** 2).item())
    assert diag["hellinger_tether_final"] == pytest.approx(expected_hel, abs=1e-12)

    mu = counts.numpy() / counts.numpy().sum()
    t_np = transition.numpy()
    sqrt_em = np.sqrt(em_prior.numpy())
    ref = minimize(
        lambda x: float(
            np.sum(mu * (np.log(mu) - np.log(np.clip(t_np.T @ x, 1e-15, None))))
            + np.sum((np.sqrt(x) - sqrt_em) ** 2)
        ),
        em_prior.numpy(),
        method="SLSQP",
        bounds=[(prior_min, 1.0)] * 3,
        constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert ref.success
    np.testing.assert_allclose(prior.numpy(), ref.x, atol=1e-6, rtol=1e-6)


def test_hellinger_analytic_gradient_matches_autograd_and_handles_active_floor():
    counts = torch.tensor([900.0, 95.0, 5.0], dtype=torch.float64)
    mu_obs = counts / counts.sum()
    transition = torch.tensor(
        [[0.92, 0.06, 0.02], [0.15, 0.80, 0.05], [0.25, 0.25, 0.50]],
        dtype=torch.float64,
    )
    # Deliberately set EM prior of class 2 very small while mu_obs[2] is mostly
    # accounted for by leakage from classes 0 and 1 so class 2 hits the floor.
    em_prior = torch.tensor([0.85, 0.1499, 0.0001], dtype=torch.float64)
    pi_var = torch.tensor([0.80, 0.15, 0.05], dtype=torch.float64, requires_grad=True)
    om = transition.T @ pi_var
    loss = torch.sum(mu_obs * (torch.log(mu_obs) - torch.log(om))) + 1.5 * torch.sum(
        (torch.sqrt(pi_var) - torch.sqrt(em_prior)) ** 2
    )
    loss.backward()
    obj, grad, _, _, _ = _evaluate_hellinger_objective_and_derivatives(
        pi_var.detach(), mu_obs, transition, em_prior, lambda_h=1.5
    )
    assert obj == pytest.approx(float(loss.item()), abs=1e-14)
    torch.testing.assert_close(grad, pi_var.grad.detach(), atol=1e-13, rtol=1e-13)

    floor = 0.02
    calibrated, diag = calibrate_clean_prior(
        counts, transition, em_prior, prior_min=floor, lambda_h=1.5, max_iter=200, tolerance=1e-10
    )
    assert diag["converged"] is True
    assert diag["active_floor_count"] >= 1
    assert diag["kkt_residual"] <= 1e-10
    assert diag["frank_wolfe_gap"] <= 1e-10
    assert float(calibrated.min()) >= floor - 1e-15


def test_prior_calibration_rejects_infeasible_floor_and_nonpositive_lambda():
    counts = torch.tensor([5.0, 3.0, 2.0], dtype=torch.float64)
    transition = torch.eye(3, dtype=torch.float64)
    prior = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    with pytest.raises(ValueError, match="K\\*prior_min"):
        calibrate_clean_prior(counts, transition, prior, prior_min=0.34)
    with pytest.raises(ValueError, match="strictly positive"):
        calibrate_clean_prior(counts, transition, prior, prior_min=0.01, lambda_h=0.0)


def test_piw_anchor_defaults_to_uniform_prior_for_backward_compatibility():
    probs = torch.tensor(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        dtype=torch.float64,
    )
    observed = torch.tensor([0, 1, 2])
    old_style = estimate_anchored_transition_and_prior(probs, observed)
    uniform = estimate_anchored_transition_and_prior(
        probs, observed, reference_prior=torch.full((3,), 1 / 3, dtype=torch.float64)
    )
    torch.testing.assert_close(old_style.transition, uniform.transition)
    torch.testing.assert_close(old_style.anchor_support, uniform.anchor_support)


def test_prior_calibration_scales_to_100_classes_in_float64():
    k = 100
    counts = torch.logspace(math.log10(500), math.log10(5), k, dtype=torch.float64).round()
    transition = torch.full((k, k), 0.4 / (k - 1), dtype=torch.float64)
    transition.fill_diagonal_(0.6)
    generator = torch.Generator().manual_seed(20261127)
    em_prior = torch.softmax(torch.randn(k, generator=generator, dtype=torch.float64) * 2.0, dim=0)
    prior, diag = calibrate_clean_prior(
        counts,
        transition,
        em_prior,
        prior_min=3.0 / 10847.0,
        lambda_h=1.0,
        max_iter=200,
        tolerance=1e-8,
    )
    assert prior.shape == (k,)
    assert prior.dtype == torch.float64
    assert torch.isfinite(prior).all()
    assert float(prior.min()) >= 3.0 / 10847.0 - 1e-12
    assert float(prior.sum()) == pytest.approx(1.0, abs=1e-10)
    assert diag["converged"] is True
    assert diag["kkt_residual"] <= 1e-8
    assert diag["frank_wolfe_gap"] <= 1e-8
