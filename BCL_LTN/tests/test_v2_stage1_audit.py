import numpy as np

from BCL_LTN.audit_v2_transition_estimator import (
    aggregate_soft_bcl_diagnostics,
    soft_bcl_batch_diagnostics,
    softmax_numpy,
)


def test_softmax_numpy_returns_finite_normalized_probabilities():
    logits = np.asarray([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    probabilities = softmax_numpy(logits)
    assert probabilities.shape == logits.shape
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-10)


def test_soft_bcl_batch_diagnostics_keep_instance_positive_and_finite_denominators():
    q = np.zeros((4, 100), dtype=np.float32)
    q[np.arange(4), np.asarray([0, 1, 2, 3])] = 1.0
    result = soft_bcl_batch_diagnostics(q, support_floor_ratio=0.5)
    assert np.isfinite(result["positive_mass_per_anchor"]).all()
    assert np.isfinite(result["negative_inverse_denominator_mass_per_anchor"]).all()
    assert (result["positive_mass_per_anchor"] > 0).all()
    assert result["denominator_weight_min"] > 0
    assert result["prototype_positive_mass_by_class"].shape == (100,)


def test_soft_bcl_aggregate_handles_last_partial_batch():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(7, 100)).astype(np.float32)
    q = softmax_numpy(logits)
    result = aggregate_soft_bcl_diagnostics(
        q, batch_size=4, support_floor_ratio=0.5
    )
    assert result["num_batches"] == 2
    assert result["num_anchors"] == 14
    assert np.isfinite(result["mean_positive_mass_per_anchor"])
    assert np.isfinite(result["mean_negative_inverse_denominator_mass_per_anchor"])
    assert len(result["class_occurrences_per_sample"]) == 100
