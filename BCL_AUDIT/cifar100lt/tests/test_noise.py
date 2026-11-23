import numpy as np
import pytest

from BCL_AUDIT.cifar100lt.noise import make_symmetric_noisy_labels, validate_noise_result


def test_zero_noise_is_exact_identity():
    labels = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
    result = make_symmetric_noisy_labels(labels, 0.0, seed=7, num_classes=3)
    assert np.array_equal(result.noisy_labels, labels)
    assert result.flipped_mask.sum() == 0
    assert result.realized_rate == 0.0
    validate_noise_result(result)


@pytest.mark.parametrize("rate", [0.2, 0.4])
def test_exact_symmetric_noise_count_and_determinism(rate):
    labels = np.tile(np.arange(10, dtype=np.int64), 100)
    a = make_symmetric_noisy_labels(labels, rate, seed=123, num_classes=10)
    b = make_symmetric_noisy_labels(labels, rate, seed=123, num_classes=10)
    assert np.array_equal(a.noisy_labels, b.noisy_labels)
    assert np.array_equal(a.flipped_mask, b.flipped_mask)
    assert int(a.flipped_mask.sum()) == round(rate * len(labels))
    assert np.all(a.noisy_labels[a.flipped_mask] != labels[a.flipped_mask])
    assert np.array_equal(a.noisy_labels[~a.flipped_mask], labels[~a.flipped_mask])
    assert np.all(a.observed_class_counts > 0)
    validate_noise_result(a, expected_clean_counts=np.full(10, 100))


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        make_symmetric_noisy_labels(np.array([0, 1]), 1.0, seed=1, num_classes=2)
    with pytest.raises(ValueError):
        make_symmetric_noisy_labels(np.array([0, 2]), 0.2, seed=1, num_classes=2)
    with pytest.raises(ValueError):
        make_symmetric_noisy_labels(np.array([[0, 1]]), 0.2, seed=1, num_classes=2)


def test_cifar100lt_count_vector_is_preserved_before_corruption():
    counts = np.asarray([int(500 * (0.01 ** (i / 99.0))) for i in range(100)])
    labels = np.repeat(np.arange(100, dtype=np.int64), counts)
    result = make_symmetric_noisy_labels(labels, 0.4, seed=2026, num_classes=100)
    validate_noise_result(result, expected_clean_counts=counts)
    assert np.array_equal(result.clean_class_counts, counts)
    assert result.observed_class_counts.sum() == counts.sum()
    assert result.observed_class_counts.shape == (100,)
