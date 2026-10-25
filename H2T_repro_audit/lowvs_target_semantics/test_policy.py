"""CPU unit tests for the Low-VS target-semantics policy primitives."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from policy import (
    build_target_distribution,
    first_crossed_competitor,
    soft_target_cross_entropy,
)


class TargetSemanticsPolicyTests(unittest.TestCase):
    def test_hard_target_loss_matches_torch_cross_entropy(self) -> None:
        logits = torch.tensor([[1.2, -0.1, 0.5], [0.2, 0.7, -0.4]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        targets = build_target_distribution(labels, 3, mode="hard")
        self.assertTrue(torch.equal(targets, F.one_hot(labels, num_classes=3).float()))
        self.assertTrue(torch.allclose(
            soft_target_cross_entropy(logits, targets),
            F.cross_entropy(logits, labels),
            atol=1e-7,
            rtol=1e-7,
        ))

    def test_donor_mix_target_has_fixed_mass(self) -> None:
        labels = torch.tensor([0, 1, 2])
        donor = torch.tensor([1, 1, 2])
        targets = build_target_distribution(labels, 3, mode="donor_mix", donor_labels=donor)
        self.assertTrue(torch.allclose(targets[0], torch.tensor([0.75, 0.25, 0.0])))
        self.assertTrue(torch.allclose(targets[1], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.allclose(targets[2], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.allclose(targets.sum(dim=1), torch.ones(3)))

    def test_active_mask_keeps_non_tail_rows_hard(self) -> None:
        labels = torch.tensor([3, 4, 5])
        donor = torch.tensor([1, 2, 0])
        active = torch.tensor([False, True, False])
        targets = build_target_distribution(
            labels, 6, mode="donor_mix", donor_labels=donor, active_mask=active
        )
        self.assertTrue(torch.equal(targets[0], F.one_hot(labels[0], 6).float()))
        self.assertTrue(torch.allclose(targets[1], torch.tensor([0, 0, 0, 0, 0.75, 0.25])))
        self.assertTrue(torch.equal(targets[2], F.one_hot(labels[2], 6).float()))

    def test_first_crossed_competitor_and_alpha(self) -> None:
        clean = torch.tensor([[3.0, 2.0, 0.0]], dtype=torch.float32)
        fused = torch.tensor([[1.0, 2.5, 0.0]], dtype=torch.float32)
        labels = torch.tensor([0])
        competitor, alpha = first_crossed_competitor(clean, fused, labels)
        self.assertEqual(int(competitor.item()), 1)
        self.assertAlmostEqual(float(alpha.item()), 0.5, places=6)

    def test_no_crossing_uses_sentinel(self) -> None:
        clean = torch.tensor([[3.0, 2.0, 0.0]], dtype=torch.float32)
        fused = torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float32)
        competitor, alpha = first_crossed_competitor(clean, fused, torch.tensor([0]))
        self.assertEqual(int(competitor.item()), -1)
        self.assertEqual(float(alpha.item()), -1.0)

    def test_clean_reference_misclassification_has_no_valid_crossing(self) -> None:
        clean = torch.tensor([[2.0, 3.0, 0.0]], dtype=torch.float32)
        fused = torch.tensor([[0.0, 3.5, 0.0]], dtype=torch.float32)
        competitor, alpha = first_crossed_competitor(clean, fused, torch.tensor([0]))
        self.assertEqual(int(competitor.item()), -1)
        self.assertEqual(float(alpha.item()), -1.0)

    def test_competitor_target_uses_first_crossed_class(self) -> None:
        labels = torch.tensor([0, 1])
        crossed = torch.tensor([2, -1])
        targets = build_target_distribution(
            labels, 3, mode="first_crossed_competitor", crossed_competitors=crossed
        )
        self.assertTrue(torch.allclose(targets[0], torch.tensor([0.75, 0.0, 0.25])))
        self.assertTrue(torch.equal(targets[1], torch.tensor([0.0, 1.0, 0.0])))

    def test_inactive_or_uncrossed_competitor_rows_stay_hard(self) -> None:
        labels = torch.tensor([0, 1, 2])
        crossed = torch.tensor([1, 2, -1])
        active = torch.tensor([False, True, True])
        targets = build_target_distribution(
            labels,
            3,
            mode="first_crossed_competitor",
            crossed_competitors=crossed,
            active_mask=active,
        )
        self.assertTrue(torch.equal(targets[0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(targets[1], torch.tensor([0.0, 0.75, 0.25])))
        self.assertTrue(torch.equal(targets[2], torch.tensor([0.0, 0.0, 1.0])))

    def test_invalid_target_rows_are_rejected(self) -> None:
        logits = torch.zeros((2, 3))
        targets = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.25, 0.0]])
        with self.assertRaises(ValueError):
            soft_target_cross_entropy(logits, targets)


    def test_endpoint_crossing_alpha_is_one(self) -> None:
        clean = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
        fused = torch.tensor([[2.0, 2.0, 0.0]], dtype=torch.float32)
        competitor, alpha = first_crossed_competitor(clean, fused, torch.tensor([0]))
        self.assertEqual(int(competitor.item()), 1)
        self.assertAlmostEqual(float(alpha.item()), 1.0, places=6)

    def test_equal_crossing_times_choose_smallest_class_id(self) -> None:
        clean = torch.tensor([[2.0, 1.0, 1.0]], dtype=torch.float32)
        fused = torch.tensor([[1.0, 2.0, 2.0]], dtype=torch.float32)
        competitor, alpha = first_crossed_competitor(clean, fused, torch.tensor([0]))
        self.assertEqual(int(competitor.item()), 1)
        self.assertAlmostEqual(float(alpha.item()), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
