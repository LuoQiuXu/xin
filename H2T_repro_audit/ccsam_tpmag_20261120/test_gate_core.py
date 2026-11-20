import unittest

import torch

from gate_core import (
    apply_tail_preserving_gate,
    build_margin_reference,
    compute_tail_preserving_gate,
)


class TailPreservingMarginGateTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        # Class 0 and 1 are Many; classes 2 and 3 are non-Many.
        self.features = torch.tensor(
            [[3.0], [3.0], [2.0], [4.0], [2.0], [4.0], [2.0], [4.0]],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        self.counts = torch.tensor([200, 150, 10, 5], dtype=torch.long)
        # For source class 2, p(target=0 | Many) is 0.5 on both examples.
        self.initial_logits = torch.tensor(
            [
                [2.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 1.0, 0.0],
                [0.0, 2.0, 1.0, 0.0],
                [0.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.reference = build_margin_reference(
            self.features, self.labels, self.initial_logits, self.counts
        )

    def test_reference_has_expected_pair_family_and_competitors(self):
        self.assertEqual(int(self.reference["eligible"].sum()), 4)
        self.assertEqual(self.reference["source_ids"].tolist(), [2, 3])
        self.assertEqual(self.reference["target_ids"].tolist(), [0, 1])
        self.assertEqual(int(self.reference["competitor_ids"][0]), 2)
        self.assertEqual(int(self.reference["competitor_ids"][1]), 2)

    def test_gate_uses_head_benefit_and_tail_risk_ratio(self):
        source_w = torch.zeros((4, 4, 1), dtype=torch.float32)
        source_b = torch.zeros((4, 4), dtype=torch.float32)
        source_w[2, 0, 0] = -1.0
        gate, diagnostic = compute_tail_preserving_gate(
            source_w, source_b, self.reference
        )
        # head benefit = 3; tail risk = 1.5; gate = 3 / (3 + 1.5).
        self.assertAlmostEqual(float(gate[2, 0]), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(diagnostic["head_benefit"][2, 0]), 3.0, places=6)
        self.assertAlmostEqual(float(diagnostic["tail_risk"][2, 0]), 1.5, places=6)

    def test_exactly_zero_eligible_pair_is_identity(self):
        source_w = torch.zeros((4, 4, 1), dtype=torch.float32)
        source_b = torch.zeros((4, 4), dtype=torch.float32)
        gate, diagnostic = compute_tail_preserving_gate(
            source_w, source_b, self.reference
        )
        self.assertEqual(float(gate[2, 0]), 1.0)
        self.assertFalse(bool(diagnostic["active_pairs"][2, 0]))

    def test_nonbeneficial_head_pair_is_closed(self):
        source_w = torch.zeros((4, 4, 1), dtype=torch.float32)
        source_b = torch.zeros((4, 4), dtype=torch.float32)
        source_w[2, 0, 0] = 1.0
        gate, diagnostic = compute_tail_preserving_gate(
            source_w, source_b, self.reference
        )
        self.assertEqual(float(gate[2, 0]), 0.0)
        self.assertLess(float(diagnostic["head_margin_gain_raw"][2, 0]), 0.0)

    def test_tail_safe_beneficial_pair_is_fully_retained(self):
        source_w = torch.zeros((4, 4, 1), dtype=torch.float32)
        source_b = torch.zeros((4, 4), dtype=torch.float32)
        source_w[2, 0, 0] = -1.0
        # Move source-2 Calibration features to negative values in the frozen
        # tail reference, making the same head-beneficial update tail-safe.
        reference = dict(self.reference)
        reference["tail_weighted_features"] = self.reference["tail_weighted_features"].clone()
        reference["tail_weighted_features"][2, 0, 0] = -1.0
        gate, diagnostic = compute_tail_preserving_gate(source_w, source_b, reference)
        self.assertEqual(float(diagnostic["tail_risk"][2, 0]), 0.0)
        self.assertAlmostEqual(float(gate[2, 0]), 1.0, places=6)

    def test_ineligible_components_are_exact_identity(self):
        source_w = torch.randn((4, 4, 1), dtype=torch.float32)
        source_b = torch.randn((4, 4), dtype=torch.float32)
        gate, _ = compute_tail_preserving_gate(source_w, source_b, self.reference)
        self.assertTrue(torch.equal(
            gate[~self.reference["eligible"]],
            torch.ones_like(gate[~self.reference["eligible"]]),
        ))

    def test_gate_and_gradient_shapes_and_finiteness(self):
        source_w = torch.randn((4, 4, 1), dtype=torch.float32)
        source_b = torch.randn((4, 4), dtype=torch.float32)
        clean_w = torch.randn((4, 1), dtype=torch.float32)
        clean_b = torch.randn((4,), dtype=torch.float32)
        gate, _ = compute_tail_preserving_gate(source_w, source_b, self.reference)
        grad_w, grad_b, gated_w, gated_b = apply_tail_preserving_gate(
            clean_w, clean_b, source_w, source_b, gate
        )
        self.assertEqual(tuple(grad_w.shape), (4, 1))
        self.assertEqual(tuple(grad_b.shape), (4,))
        self.assertTrue(torch.isfinite(grad_w).all())
        self.assertTrue(torch.isfinite(grad_b).all())
        self.assertTrue(torch.isfinite(gated_w).all())
        self.assertTrue(torch.isfinite(gated_b).all())

    def test_all_one_gate_reconstructs_original_direct_gradient(self):
        source_w = torch.randn((4, 4, 1), dtype=torch.float32)
        source_b = torch.randn((4, 4), dtype=torch.float32)
        clean_w = torch.randn((4, 1), dtype=torch.float32)
        clean_b = torch.randn((4,), dtype=torch.float32)
        ones = torch.ones((4, 4), dtype=torch.float32)
        grad_w, grad_b, gated_w, gated_b = apply_tail_preserving_gate(
            clean_w, clean_b, source_w, source_b, ones
        )
        self.assertTrue(torch.equal(gated_w, source_w))
        self.assertTrue(torch.equal(gated_b, source_b))
        self.assertTrue(torch.equal(grad_w, clean_w + source_w.sum(dim=0)))
        self.assertTrue(torch.equal(grad_b, clean_b + source_b.sum(dim=0)))

    def test_rejects_nonfinite_source_gradient(self):
        source_w = torch.zeros((4, 4, 1), dtype=torch.float32)
        source_b = torch.zeros((4, 4), dtype=torch.float32)
        source_w[2, 0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            compute_tail_preserving_gate(source_w, source_b, self.reference)


if __name__ == "__main__":
    unittest.main()
