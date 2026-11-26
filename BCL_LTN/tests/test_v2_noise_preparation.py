import numpy as np

from BCL_LTN.prepare_ntc_bcl_v2_noise import (
    coarse_cycle_mapping,
    make_asymmetric_coarse_cycle,
)


def test_coarse_cycle_mapping_stays_inside_each_cifar100_superclass():
    fine = np.repeat(np.arange(100, dtype=np.int64), 500)
    coarse_by_class = np.repeat(np.arange(20, dtype=np.int64), 5)
    coarse = np.repeat(coarse_by_class, 500)
    mapping = coarse_cycle_mapping(fine, coarse)
    assert mapping.shape == (100,)
    assert np.all(mapping != np.arange(100))
    for superclass in range(20):
        ids = np.flatnonzero(coarse_by_class == superclass)
        assert set(mapping[ids].tolist()) == set(ids.tolist())


def test_asymmetric_coarse_cycle_uses_one_deterministic_destination_per_flipped_class():
    fine = np.tile(np.arange(100, dtype=np.int64), 20)
    target_map = np.roll(np.arange(100, dtype=np.int64), -1)
    observed, mask, transition = make_asymmetric_coarse_cycle(
        fine, target_map, noise_rate=0.4, seed=42
    )
    assert observed.shape == fine.shape
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, observed != fine)
    assert np.all(observed[mask] == target_map[fine[mask]])
    assert np.all(observed[~mask] == fine[~mask])
    np.testing.assert_allclose(transition.sum(axis=1), np.ones(100), atol=1e-12)
    assert np.allclose(np.diag(transition), 0.6)
