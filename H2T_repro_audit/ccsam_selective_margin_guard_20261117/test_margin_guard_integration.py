import unittest

import torch

from train_margin_guard import combine_d3_objective


class SelectiveMarginGuardIntegrationTests(unittest.TestCase):
    def test_d3_is_exactly_d1_plus_guard(self):
        teacher = torch.tensor(
            [[4.0, 0.0, 1.0, -1.0], [0.0, 3.0, 2.0, 0.0]],
            dtype=torch.float32,
        )
        student = torch.tensor(
            [[1.0, 0.0, 2.0, -1.0], [0.0, 3.0, 2.0, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.tensor([0, 1], dtype=torch.long)
        d1_loss = torch.tensor(1.25, requires_grad=True)
        total, guard, diagnostics = combine_d3_objective(
            d1_loss, student, teacher, labels, [500, 300, 50, 5]
        )
        self.assertGreater(float(guard), 0.0)
        self.assertAlmostEqual(float(total), float(d1_loss + guard), places=7)
        self.assertEqual(diagnostics["many_inputs"], 2)
        total.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNotNone(d1_loss.grad)

    def test_no_margin_violation_leaves_d1_objective_unchanged(self):
        teacher = torch.tensor(
            [[4.0, 0.0, 1.0, -1.0], [0.0, 3.0, 2.0, 0.0]],
            dtype=torch.float32,
        )
        student = teacher.clone().requires_grad_(True)
        labels = torch.tensor([0, 1], dtype=torch.long)
        d1_loss = torch.tensor(0.75, requires_grad=True)
        total, guard, diagnostics = combine_d3_objective(
            d1_loss, student, teacher, labels, [500, 300, 50, 5]
        )
        self.assertEqual(float(guard), 0.0)
        self.assertAlmostEqual(float(total), float(d1_loss), places=7)
        self.assertEqual(diagnostics["active_guard_inputs"], 0)

    def test_tail_only_batch_does_not_activate_guard(self):
        teacher = torch.randn(2, 4)
        student = torch.randn(2, 4, requires_grad=True)
        labels = torch.tensor([2, 3], dtype=torch.long)
        d1_loss = student.square().mean()
        total, guard, diagnostics = combine_d3_objective(
            d1_loss, student, teacher, labels, [500, 300, 50, 5]
        )
        self.assertEqual(float(guard), 0.0)
        self.assertEqual(diagnostics["many_inputs"], 0)
        total.backward()
        self.assertIsNotNone(student.grad)


if __name__ == "__main__":
    unittest.main()
