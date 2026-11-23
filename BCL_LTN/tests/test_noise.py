import numpy as np

from BCL_LTN.make_noisy_lt import make_symmetric_noise, group_class_ids


def test_noise_is_per_example_bernoulli_and_reproducible():
    clean = np.repeat(np.arange(100), [int(500 * (0.01 ** (i / 99.0))) for i in range(100)])
    a = make_symmetric_noise(clean, 0.4, 42)
    b = make_symmetric_noise(clean, 0.4, 42)
    noisy, original, mask = a
    assert np.array_equal(noisy, b[0])
    assert np.array_equal(mask, b[2])
    assert np.array_equal(original, clean)
    assert int(mask.sum()) != 0
    assert np.all(noisy[mask] != clean[mask])
    assert np.array_equal(noisy[~mask], clean[~mask])
    assert abs(mask.mean() - 0.4) < 0.02


def test_many_medium_few_groups_match_if100_definition():
    counts = np.asarray([int(500 * (0.01 ** (i / 99.0))) for i in range(100)])
    groups = group_class_ids(counts)
    assert (len(groups["Many"]), len(groups["Medium"]), len(groups["Few"])) == (35, 35, 30)
