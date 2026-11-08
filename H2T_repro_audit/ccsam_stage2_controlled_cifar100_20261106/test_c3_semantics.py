"""CPU-only deterministic algebraic checks; no datasets, checkpoint, or TEST access."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from train_ccsam import C3_ARM, compute_arm_loss, mask_many_feature_delta


class C3SelectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261108)
        self.gap = torch.randn(64, 64, dtype=torch.float32)
        self.fc = torch.nn.Linear(64, 100)
        self.weights = torch.linspace(0.01, 2.3, steps=100)
        # Synthetic counts verify STRICT >100 rather than >=100 or ID range.
        self.counts = np.array([500] * 34 + [101, 100] + [98] * 34 + [19] * 30)
        assert self.counts.shape == (100,)
        self.labels = torch.tensor(([0, 34, 35, 36, 69, 70, 99, 1] * 8),
                                   dtype=torch.long)

    def loss(self, arm, labels=None, train_counts=None):
        return compute_arm_loss(
            arm, self.gap, self.labels if labels is None else labels,
            self.fc, self.weights, 0, train_counts=train_counts,
        )

    def test_strict_threshold_mixed_class_and_repeated_class(self):
        counts = torch.tensor(self.counts)
        expected_many = counts[self.labels] > 100
        self.assertTrue(bool(expected_many.any()))
        self.assertTrue(bool((~expected_many).any()))
        c1 = self.loss("C1_CCSAM_FEATURE_GRAD")
        c3 = self.loss(C3_ARM, train_counts=self.counts)
        delta1, delta3 = c1[4], c3[4]
        self.assertTrue(torch.equal(delta3[expected_many],
                                    torch.zeros_like(delta3[expected_many])))
        self.assertTrue(torch.equal(delta3[~expected_many], delta1[~expected_many]))
        self.assertGreater(float(delta3[~expected_many].abs().sum()), 0)
        self.assertEqual(c3[5]["c3_many_examples_masked"], int(expected_many.sum()))
        self.assertEqual(c3[5]["c3_nonmany_examples_kept"], int((~expected_many).sum()))
        self.assertTrue(torch.isfinite(c3[0]))

    def test_all_many_c3_equals_c2_exactly(self):
        ys = torch.zeros(64, dtype=torch.long)
        c3 = self.loss(C3_ARM, labels=ys, train_counts=self.counts)
        c2 = self.loss("C2_WEIGHTED_CE_NO_PERTURB", labels=ys)
        self.assertTrue(torch.equal(c3[4], torch.zeros_like(c3[4])))
        self.assertTrue(torch.equal(c3[0], c2[0]))

    def test_all_nonmany_c3_equals_c1_exactly(self):
        ys = torch.full((64,), 99, dtype=torch.long)
        c3 = self.loss(C3_ARM, labels=ys, train_counts=self.counts)
        c1 = self.loss("C1_CCSAM_FEATURE_GRAD", labels=ys)
        self.assertTrue(torch.equal(c3[4], c1[4]))
        self.assertTrue(torch.equal(c3[0], c1[0]))

    def test_exact_100_perturbed_and_101_not(self):
        ys = torch.tensor([34, 35], dtype=torch.long)
        gap = self.gap[:2]
        logits = self.fc(gap)
        w = self.weights
        from train_ccsam import analytic_classwise_feature_delta
        raw, _ = analytic_classwise_feature_delta(logits, gap, ys, w, self.fc.weight)
        masked, metadata = mask_many_feature_delta(raw, gap, ys, self.counts)
        self.assertTrue(torch.equal(masked[0], torch.zeros_like(masked[0])))
        self.assertTrue(torch.equal(masked[1], raw[1]))
        self.assertEqual(metadata["c3_many_examples_masked"], 1)

    def test_missing_counts_rejected_and_c1_unchanged(self):
        with self.assertRaises(ValueError):
            self.loss(C3_ARM)
        c1 = self.loss("C1_CCSAM_FEATURE_GRAD")
        c1_with_counts = self.loss("C1_CCSAM_FEATURE_GRAD", train_counts=self.counts)
        self.assertTrue(torch.equal(c1[0], c1_with_counts[0]))
        self.assertTrue(torch.equal(c1[4], c1_with_counts[4]))

    def test_classifier_receives_c3_gradient(self):
        loss = self.loss(C3_ARM, train_counts=self.counts)[0]
        self.fc.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in self.fc.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
