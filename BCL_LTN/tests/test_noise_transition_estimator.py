import math

import pytest
import torch

from BCL_LTN.noise_transition_estimator import (
    estimate_transition_and_prior,
    prototype_health_diagnostics,
)


def make_fixture():
    clean = torch.arange(3).repeat_interleave(120)
    transition = torch.tensor(
        [
            [0.80, 0.15, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.15, 0.80],
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(31)
    observed = torch.multinomial(transition[clean], 1, generator=generator).squeeze(1)
    probs = torch.full((len(clean), 3), 0.02, dtype=torch.float64)
    probs[torch.arange(len(clean)), clean] = 0.96
    return probs, observed, transition


def test_unknown_transition_estimator_returns_valid_rows_and_converges():
    probs, observed, _ = make_fixture()
    result = estimate_transition_and_prior(
        probs,
        observed,
        posterior_kind="long_tailed_posterior",
        shrinkage_mass=5.0,
        high_support_fraction=0.35,
        max_iter=50,
        tolerance=1e-6,
    )
    torch.testing.assert_close(
        result.transition.sum(1), torch.ones(3, dtype=torch.float64), atol=1e-8, rtol=0
    )
    torch.testing.assert_close(
        result.clean_prior.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-8, rtol=0
    )
    torch.testing.assert_close(
        result.posterior_clean.sum(1),
        torch.ones(len(observed), dtype=torch.float64),
        atol=1e-8,
        rtol=0,
    )
    assert torch.isfinite(result.transition).all()
    assert torch.isfinite(result.posterior_clean).all()
    assert (result.transition >= 0).all()
    assert 1 <= result.iterations <= 50
    assert result.converged
    assert result.final_delta <= 1e-6


def test_posterior_semantics_do_not_double_multiply_prior():
    probs = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float64)
    observed = torch.tensor([0, 1])
    prior = torch.tensor([0.9, 0.1], dtype=torch.float64)
    lt = estimate_transition_and_prior(
        probs,
        observed,
        posterior_kind="long_tailed_posterior",
        initial_prior=prior,
        max_iter=1,
    )
    balanced = estimate_transition_and_prior(
        probs,
        observed,
        posterior_kind="balanced_posterior",
        initial_prior=prior,
        max_iter=1,
    )
    assert not torch.allclose(lt.posterior_clean, balanced.posterior_clean)


def test_empirical_bayes_shrinkage_preserves_positive_tail_rows():
    probs = torch.tensor(
        [
            [0.98, 0.01, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
        ]
        * 4,
        dtype=torch.float64,
    )
    observed = torch.tensor([0, 1, 2] * 4)
    initial_prior = torch.tensor([0.8, 0.19, 0.01], dtype=torch.float64)
    result = estimate_transition_and_prior(
        probs,
        observed,
        posterior_kind="long_tailed_posterior",
        initial_prior=initial_prior,
        shrinkage_mass=50.0,
        max_iter=3,
    )
    assert torch.allclose(
        result.transition.sum(1), torch.ones(3, dtype=torch.float64), atol=1e-8, rtol=0
    )
    assert (result.clean_prior > 0).all()
    assert torch.isfinite(result.clean_prior).all()


def test_high_support_template_uses_a_fraction_of_supported_classes():
    counts = torch.arange(20, 0, -1, dtype=torch.long)
    clean = torch.repeat_interleave(torch.arange(20), counts)
    probs = torch.full((len(clean), 20), 1e-4, dtype=torch.float64)
    probs[torch.arange(len(clean)), clean] = 1.0 - 19e-4
    observed = clean.clone()
    result = estimate_transition_and_prior(
        probs,
        observed,
        posterior_kind="long_tailed_posterior",
        high_support_fraction=0.35,
        min_high_support_classes=5,
        max_iter=10,
    )
    assert result.global_support_class_ids.numel() == 7
    assert set(result.global_support_class_ids.cpu().tolist()) == set(range(7))
    assert result.row_support[0] > result.row_support[-1]
    assert result.converged


def test_estimator_requires_convergence_controls_to_be_valid():
    probs, observed, _ = make_fixture()
    with pytest.raises(ValueError, match="tolerance"):
        estimate_transition_and_prior(
            probs, observed, posterior_kind="balanced_posterior", tolerance=0
        )
    with pytest.raises(ValueError, match="high_support_fraction"):
        estimate_transition_and_prior(
            probs, observed, posterior_kind="balanced_posterior", high_support_fraction=0
        )


def test_prototype_health_diagnostics_detect_small_classifier_weights_and_cosine_collapse():
    weight = torch.tensor([[1.0, 0.0], [0.01, 0.0], [0.02, 0.0]])
    prototypes = torch.tensor([[1.0, 0.0], [1.0, 0.001], [1.0, -0.001]])
    result = prototype_health_diagnostics(weight, prototypes, small_weight_threshold=0.15)
    assert result["small_weight_mask"].tolist() == [False, True, True]
    assert torch.allclose(
        result["prototype_output_norm"], torch.ones(3), atol=1e-3, rtol=1e-3
    )
    assert result["prototype_max_offdiag_cosine"].min() > 0.99
    assert result["mean_offdiag_cosine"] > 0.99


@pytest.mark.parametrize("kind", ["unknown", None])
def test_estimator_requires_explicit_probability_semantics(kind):
    with pytest.raises(ValueError, match="posterior_kind"):
        estimate_transition_and_prior(
            torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
            torch.tensor([0, 1]),
            posterior_kind=kind,
        )
