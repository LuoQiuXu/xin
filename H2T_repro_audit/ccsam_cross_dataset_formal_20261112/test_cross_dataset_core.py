from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cross_dataset_core import (
    D0, D1, D2, alpha_for_epoch, class_conditional_feature_delta,
    class_weights_from_counts, compute_method_loss, head_preservation_kl,
)


class CrossDatasetCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261112)
        self.device = torch.device("cpu")
        self.num_classes = 7
        self.feature_dim = 13
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        self.features = torch.randn(19, self.feature_dim)
        self.labels = torch.tensor(
            [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6],
            dtype=torch.long,
        )
        self.counts = np.array([250, 120, 100, 55, 19, 8, 5], dtype=np.int64)
        self.weights = class_weights_from_counts(self.counts, self.device)

    def test_alpha_schedule(self):
        self.assertAlmostEqual(alpha_for_epoch(0), 1.0 / 30.0)
        self.assertAlmostEqual(alpha_for_epoch(29), 0.3)
        self.assertAlmostEqual(alpha_for_epoch(89), 0.3)

    def test_class_weights_exact_formula(self):
        expected = np.log(self.counts.max() / self.counts + 0.01) / 2.0
        np.testing.assert_allclose(self.weights.numpy(), expected.astype(np.float32), rtol=1e-6)

    def test_vectorized_gradient_matches_autograd_reference(self):
        delta, _ = class_conditional_feature_delta(
            self.features, self.labels, self.classifier, self.weights
        )
        reference_features = self.features.detach().clone().requires_grad_(True)
        reference_logits = self.classifier(reference_features)
        reference = torch.zeros_like(reference_features)
        for class_id in torch.unique(self.labels, sorted=True):
            mask = self.labels == class_id
            class_mean_ce = F.cross_entropy(
                reference_logits[mask], self.labels[mask], reduction="mean"
            )
            class_gradient = torch.autograd.grad(
                class_mean_ce, reference_features, retain_graph=True
            )[0]
            norm = torch.linalg.vector_norm(class_gradient)
            if float(norm) > 1e-12:
                reference[mask] = self.weights[class_id] * class_gradient[mask] / norm
        self.assertTrue(torch.allclose(delta, reference, rtol=1e-5, atol=1e-7))

    def test_many_only_kl_uses_strict_gt_100_and_full_output_support(self):
        student = torch.randn(3, self.num_classes, requires_grad=True)
        teacher = torch.randn(3, self.num_classes)
        labels = torch.tensor([0, 1, 4], dtype=torch.long)
        counts = torch.tensor(self.counts, dtype=torch.long)
        loss, mask, n_many = head_preservation_kl(
            student, teacher, labels, counts, temperature=2.0
        )
        self.assertEqual(mask.tolist(), [True, True, False])
        self.assertEqual(n_many, 2)
        expected = F.kl_div(
            F.log_softmax(student[:2] / 2.0, dim=1),
            F.softmax(teacher[:2] / 2.0, dim=1),
            reduction="batchmean",
        ) * 4.0
        self.assertTrue(torch.allclose(loss, expected, rtol=1e-6, atol=1e-7))
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad[:2].abs().sum()), 0.0)
        self.assertEqual(float(student.grad[2].abs().sum()), 0.0)

    def test_no_many_batch_returns_differentiable_zero(self):
        student = torch.randn(2, self.num_classes, requires_grad=True)
        teacher = torch.randn(2, self.num_classes)
        labels = torch.tensor([4, 5], dtype=torch.long)
        loss, mask, n_many = head_preservation_kl(
            student, teacher, labels, self.counts, temperature=2.0
        )
        self.assertEqual(n_many, 0)
        self.assertFalse(bool(mask.any()))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(student.grad)

    def test_d0_d1_d2_losses_are_finite_and_teacher_is_frozen(self):
        teacher = nn.Linear(self.feature_dim, self.num_classes)
        teacher.requires_grad_(False)
        for arm in (D0, D1, D2):
            student = nn.Linear(self.feature_dim, self.num_classes)
            student.load_state_dict(self.classifier.state_dict())
            student.zero_grad(set_to_none=True)
            loss, diagnostics = compute_method_loss(
                arm, self.features, self.labels, student, self.weights,
                epoch_index=4, train_counts=self.counts, teacher=teacher if arm == D2 else None,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("alpha", diagnostics)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_d2_requires_matching_teacher(self):
        with self.assertRaises(ValueError):
            compute_method_loss(
                D2, self.features, self.labels, self.classifier, self.weights,
                epoch_index=0, train_counts=self.counts, teacher=None,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
