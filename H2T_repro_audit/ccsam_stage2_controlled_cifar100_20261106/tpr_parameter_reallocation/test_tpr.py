"""CPU-only semantic tests for CC-SAM-TPR."""
import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from train_tpr import (
    LowRankTailClassifier,
    more_discrepancy_loss,
    sinusoidal_more_weight,
    MORE_PEAK,
    NUM_CLASSES,
    FEATURE_DIM,
)


class TPRSemanticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261110)
        self.counts = np.full(NUM_CLASSES, 5, dtype=np.int64)
        self.counts[0] = 500
        self.counts[1] = 101
        self.counts[2] = 100
        self.tail_ids = torch.as_tensor(
            np.flatnonzero(self.counts <= 100), dtype=torch.long
        )
        self.base_weight = torch.randn(NUM_CLASSES, FEATURE_DIM)
        self.base_bias = torch.randn(NUM_CLASSES)
        self.model = LowRankTailClassifier(
            self.base_weight, self.base_bias, self.tail_ids, rank=8
        )

    def test_many_output_rows_are_bitwise_unchanged(self):
        effective = self.model.weight.detach()
        many_ids = torch.tensor([0, 1], dtype=torch.long)
        self.assertTrue(torch.equal(effective[many_ids], self.model.base_weight[many_ids]))
        tail_ids = self.model.tail_indices
        self.assertTrue(torch.equal(
            effective[many_ids],
            self.base_weight[many_ids],
        ))
        self.assertEqual(len(tail_ids), 98)

    def test_low_rank_residual_has_expected_shape_and_rank_bound(self):
        residual = self.model.tail_residual()
        self.assertEqual(tuple(residual.shape), (98, FEATURE_DIM))
        self.assertLessEqual(int(torch.linalg.matrix_rank(residual)), 8)

    def test_more_loss_matches_explicit_class_prior_weighted_l2(self):
        features = torch.randn(11, FEATURE_DIM)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long)
        loss, delta = more_discrepancy_loss(
            self.model, features, labels, self.counts
        )
        prior = torch.as_tensor(self.counts, dtype=torch.float32)
        prior = prior / prior.sum()
        expected = (
            prior[labels] * delta.square().sum(dim=1)
        ).mean()
        self.assertTrue(torch.allclose(loss, expected, rtol=1e-6, atol=1e-8))
        self.assertTrue(torch.isfinite(loss))

    def test_more_loss_updates_only_low_rank_parameters(self):
        features = torch.randn(16, FEATURE_DIM)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7] * 2, dtype=torch.long)
        loss, _ = more_discrepancy_loss(self.model, features, labels, self.counts)
        loss.backward()
        self.assertIsNotNone(self.model.A.grad)
        self.assertIsNotNone(self.model.B.grad)
        self.assertTrue(torch.isfinite(self.model.A.grad).all())
        self.assertTrue(torch.isfinite(self.model.B.grad).all())
        self.assertFalse(self.model.base_weight.requires_grad)
        self.assertFalse(self.model.base_bias.requires_grad)

    def test_sinusoidal_schedule_starts_and_ends_at_zero_and_peaks_midway(self):
        total = 101
        self.assertAlmostEqual(sinusoidal_more_weight(0, total), 0.0, places=7)
        self.assertAlmostEqual(
            sinusoidal_more_weight(total - 1, total), 0.0, places=7
        )
        self.assertAlmostEqual(
            sinusoidal_more_weight(total // 2, total), MORE_PEAK, places=5
        )
        self.assertGreater(sinusoidal_more_weight(25, total), 0.0)
        self.assertGreater(sinusoidal_more_weight(75, total), 0.0)

    def test_invalid_schedule_indices_are_rejected(self):
        with self.assertRaises(ValueError):
            sinusoidal_more_weight(-1, 10)
        with self.assertRaises(ValueError):
            sinusoidal_more_weight(10, 10)
        with self.assertRaises(ValueError):
            sinusoidal_more_weight(0, 1)

    def test_rowwise_orthogonality_and_neighbor_diversity_are_differentiable(self):
        edges = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
        orth = self.model.orthogonality_loss()
        diversity = self.model.diversity_loss(edges)
        total = orth + diversity
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(self.model.A.grad)
        self.assertIsNotNone(self.model.B.grad)
        self.assertTrue(torch.isfinite(self.model.A.grad).all())
        self.assertTrue(torch.isfinite(self.model.B.grad).all())

    def test_merged_classifier_matches_adapter_forward(self):
        features = torch.randn(13, FEATURE_DIM)
        logits = self.model(features)
        merged = self.model.merged_state_dict()
        merged_logits = F.linear(features, merged["fc.weight"], merged["fc.bias"])
        self.assertTrue(torch.allclose(logits, merged_logits, rtol=0.0, atol=1e-7))

    def test_invalid_neighbor_edges_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model.diversity_loss(torch.tensor([[0, 98]], dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model.diversity_loss(torch.tensor([0, 1, 2], dtype=torch.long))


if __name__ == "__main__":
    unittest.main(verbosity=2)
