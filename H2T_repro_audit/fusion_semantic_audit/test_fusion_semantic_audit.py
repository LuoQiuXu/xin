import unittest
import numpy as np

from audit_fusion_semantics import (
    apply_channel_mask,
    average_ranks,
    margin_and_competitor,
    pair_excluded_ncm_scores,
    spearman_rho,
    normalize_rows,
)


class FusionSemanticAuditUnitTests(unittest.TestCase):
    def test_channel_mask_replaces_only_selected_coordinates(self):
        receiver = np.zeros((3, 64), dtype=np.float32)
        donor = np.ones((3, 64), dtype=np.float32)
        mask = np.tile(np.arange(16, dtype=np.uint8), (3, 1))
        fused = apply_channel_mask(receiver, donor, mask)
        self.assertTrue(np.all(fused[:, :16] == 1.0))
        self.assertTrue(np.all(fused[:, 16:] == 0.0))

    def test_margin_and_competitor(self):
        scores = np.zeros((2, 100), dtype=np.float64)
        labels = np.array([3, 7], dtype=np.int64)
        scores[0, 3], scores[0, 4] = 2.0, 1.0
        scores[1, 7], scores[1, 8] = 0.5, 1.0
        pred, margin, competitor = margin_and_competitor(scores, labels)
        np.testing.assert_array_equal(pred, np.array([3, 8]))
        np.testing.assert_allclose(margin, np.array([1.0, -0.5]))
        np.testing.assert_array_equal(competitor, np.array([4, 8]))

    def test_average_ranks_handles_ties(self):
        ranks = average_ranks(np.array([10.0, 10.0, 20.0, 30.0]))
        np.testing.assert_allclose(ranks, np.array([1.5, 1.5, 3.0, 4.0]))

    def test_spearman_is_rank_correlation(self):
        self.assertAlmostEqual(spearman_rho(np.array([1, 2, 3, 4]), np.array([2, 4, 6, 8])), 1.0)
        self.assertAlmostEqual(spearman_rho(np.array([1, 2, 3, 4]), np.array([8, 6, 4, 2])), -1.0)
        self.assertIsNone(spearman_rho(np.array([1, 1, 1]), np.array([1, 2, 3])))

    def test_pair_excluded_ncm_scores_are_finite(self):
        features = np.array([
            [1.0, 0.0], [0.9, 0.1], [0.8, 0.2],
            [0.0, 1.0], [0.1, 0.9], [0.2, 0.8],
        ], dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        normalized = normalize_rows(features)
        sums = np.stack([normalized[labels == c].sum(axis=0) for c in range(2)] + [
            np.ones(2, dtype=np.float64) for _ in range(98)
        ])
        counts = np.bincount(labels, minlength=100).astype(np.int64)
        counts[2:] = 3
        z = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float64)
        rec = np.array([0, 3], dtype=np.int64)
        don = np.array([3, 0], dtype=np.int64)
        # The synthetic bank uses nonzero placeholder prototypes for unused classes.
        scores = pair_excluded_ncm_scores(z, rec, don, labels, normalized, sums, counts)
        self.assertEqual(scores.shape, (2, 100))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
