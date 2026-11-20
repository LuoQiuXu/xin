import numpy as np
import pytest

from run_logit_geometry_diagnostics import (
    compute_metrics, decompose_vector, is_dominated, logit_adjustment,
    pareto_frontier, select_tau,
)


def test_tau_zero_is_identity():
    logits = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
    priors = np.array([0.8, 0.2])
    np.testing.assert_allclose(logit_adjustment(logits, priors, 0.0), logits, atol=0.0, rtol=0.0)


def test_positive_tau_boosts_rare_class():
    adjusted = logit_adjustment(np.zeros((1, 2)), np.array([0.9, 0.1]), 1.0)
    assert adjusted[0, 1] > adjusted[0, 0]


def test_logit_adjustment_rejects_invalid_priors():
    with pytest.raises(ValueError):
        logit_adjustment(np.zeros((2, 2)), np.array([1.0, 0.0]), 1.0)


def test_metrics_include_many_medium_few_and_balanced_macro():
    counts = np.array([500, 100, 10], dtype=np.int64)
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    logits = np.zeros((6, 3), dtype=np.float64)
    logits[np.arange(6), labels] = 1.0
    result = compute_metrics(logits, labels, counts)
    assert result["many_accuracy"] == result["medium_accuracy"] == result["few_accuracy"] == 100.0
    assert result["n_many"] == result["n_medium"] == result["n_few"] == 2
    assert result["overall_accuracy"] == result["macro_accuracy"] == 100.0


def test_metrics_reject_missing_group():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros((2, 3)), np.array([0, 0]), np.array([500, 100, 10]))


def test_tau_selection_tie_break_is_deterministic():
    rows = [
        {"tau": -0.5, "overall_accuracy": 90.0},
        {"tau": 0.5, "overall_accuracy": 90.0},
        {"tau": 0.0, "overall_accuracy": 89.0},
    ]
    assert select_tau(rows)["tau"] == -0.5
    rows.append({"tau": 0.0, "overall_accuracy": 90.0})
    assert select_tau(rows)["tau"] == 0.0


def test_radial_tangential_reconstruct_and_are_orthogonal():
    reference = np.array([2.0, -1.0, 3.0])
    delta = np.array([1.5, 0.25, -2.0])
    result = decompose_vector(delta, reference)
    np.testing.assert_allclose(result["radial_vector"] + result["tangential_vector"],
                               delta, atol=1e-12, rtol=1e-12)
    assert abs(np.dot(result["radial_vector"], result["tangential_vector"])) < 1e-10


def test_zero_delta_has_zero_components():
    result = decompose_vector(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]))
    assert result["total_displacement_norm"] == 0.0
    assert result["tangential_component_norm"] == 0.0


def test_geometry_rejects_zero_reference_and_nonfinite():
    with pytest.raises(ValueError):
        decompose_vector(np.ones(3), np.zeros(3))
    with pytest.raises(ValueError):
        decompose_vector(np.array([np.nan, 0.0]), np.ones(2))


def test_pareto_dominance():
    point = {"many_accuracy": 60.0, "few_accuracy": 25.0}
    candidates = [{"many_accuracy": 61.0, "few_accuracy": 26.0},
                  {"many_accuracy": 58.0, "few_accuracy": 30.0}]
    assert is_dominated(point, candidates)
    assert not is_dominated(candidates[0], [point])


def test_pareto_frontier_removes_dominated_points_and_duplicates():
    rows = [
        {"many_accuracy": 60.0, "few_accuracy": 25.0, "tau": 0.0},
        {"many_accuracy": 61.0, "few_accuracy": 26.0, "tau": 0.5},
        {"many_accuracy": 61.0, "few_accuracy": 26.0, "tau": 0.6},
        {"many_accuracy": 59.0, "few_accuracy": 28.0, "tau": 1.0},
    ]
    frontier = pareto_frontier(rows)
    assert len(frontier) == 2
    assert all(not is_dominated(point, frontier) for point in frontier)
