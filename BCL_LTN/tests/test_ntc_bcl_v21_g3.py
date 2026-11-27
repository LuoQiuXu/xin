from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from BCL_LTN.ntc_bcl_v2_losses import estimate_anchored_transition_and_prior
from BCL_LTN.ntc_bcl_v21_calibration import (
    calibrate_clean_prior,
    observed_fit_statistics,
)


def test_observed_fit_statistics_are_macro_and_confidence_ranked():
    # Class 0 has 5 samples; class 1 has 3. The highest-confidence observed
    # examples are selected independently within each observed class.
    probs = torch.tensor(
        [
            [0.95, 0.05],  # y=0, correct, high confidence
            [0.80, 0.20],  # y=0, correct
            [0.55, 0.45],  # y=0, correct
            [0.51, 0.49],  # y=0, correct
            [0.99, 0.01],  # y=1, wrong but highest confidence for y=1
            [0.60, 0.40],  # y=1, wrong
            [0.10, 0.90],  # y=1, correct
            [0.20, 0.80],  # y=0, wrong
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
    # Top-50%: ceil(5/2)=3 for class 0, ceil(3/2)=2 for class 1.
    assert stats["f_top50"] == pytest.approx((1.0 + 0.5) / 2)


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


def test_prior_calibration_respects_floor_simplex_and_reduces_objective():
    counts = torch.tensor([700.0, 220.0, 80.0], dtype=torch.float64)
    transition = torch.tensor(
        [[0.90, 0.08, 0.02], [0.10, 0.82, 0.08], [0.05, 0.15, 0.80]],
        dtype=torch.float64,
    )
    em_prior = torch.tensor([0.80, 0.19, 0.01], dtype=torch.float64)
    prior, diag = calibrate_clean_prior(
        counts,
        transition,
        em_prior,
        prior_min=3.0 / float(counts.sum()),
        lambda_h=1.0,
        max_iter=200,
        tolerance=1e-9,
    )
    assert diag["converged"] is True
    assert diag["objective_final"] <= diag["objective_initial"] + 1e-10
    assert float(prior.min()) >= 3.0 / float(counts.sum()) - 1e-12
    assert float(prior.sum()) == pytest.approx(1.0, abs=1e-10)
    assert prior.dtype == torch.float64
    assert torch.isfinite(prior).all()


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
        max_iter=150,
        tolerance=1e-8,
    )
    assert prior.shape == (k,)
    assert prior.dtype == torch.float64
    assert torch.isfinite(prior).all()
    assert float(prior.min()) >= 3.0 / 10847.0 - 1e-12
    assert float(prior.sum()) == pytest.approx(1.0, abs=1e-10)
    assert diag["converged"] is True
