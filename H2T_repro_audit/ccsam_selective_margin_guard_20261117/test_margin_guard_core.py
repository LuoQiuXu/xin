import unittest

import torch

from margin_guard_core import selective_many_tail_margin_guard


COUNTS = [500, 300, 50, 5]


class SelectiveMarginGuardTests(unittest.TestCase):
    def setUp(self):
        self.teacher = torch.tensor(
            [[4.0, 0.0, 1.0, -1.0], [0.0, 3.0, 2.0, 0.0]],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 1], dtype=torch.long)

    def test_student_equal_to_teacher_has_zero_guard(self):
        student = self.teacher.clone().requires_grad_(True)
        teacher = self.teacher.clone().requires_grad_(True)
        loss, stats = selective_many_tail_margin_guard(
            student, teacher, self.labels, COUNTS
        )
        self.assertAlmostEqual(float(loss), 0.0, places=7)
        self.assertEqual(stats["many_inputs"], 2)
        self.assertEqual(stats["eligible_many_inputs"], 2)
        self.assertEqual(stats["active_guard_inputs"], 0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertEqual(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_guard_penalizes_margin_collapse_and_backpropagates_to_student(self):
        student = torch.tensor(
            [[1.0, 0.0, 2.0, -1.0], [0.0, 3.0, 2.0, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher = self.teacher.clone().requires_grad_(True)
        loss, stats = selective_many_tail_margin_guard(
            student, teacher, self.labels, COUNTS
        )
        self.assertGreater(float(loss), 0.0)
        self.assertEqual(stats["active_guard_inputs"], 1)
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_only_many_inputs_with_positive_teacher_margin_are_eligible(self):
        logits = torch.tensor(
            [[0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 3.0, 0.0]],
            dtype=torch.float32,
        )
        labels = torch.tensor([2, 0], dtype=torch.long)
        teacher = torch.tensor(
            [[0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 3.0, 0.0]],
            dtype=torch.float32,
        )
        loss, stats = selective_many_tail_margin_guard(
            logits.requires_grad_(True), teacher, labels, COUNTS
        )
        self.assertEqual(stats["many_inputs"], 1)
        self.assertEqual(stats["eligible_many_inputs"], 0)
        self.assertEqual(float(loss), 0.0)

    def test_tail_only_batch_returns_differentiable_zero(self):
        student = torch.randn(2, 4, requires_grad=True)
        teacher = torch.randn(2, 4)
        labels = torch.tensor([2, 3], dtype=torch.long)
        loss, stats = selective_many_tail_margin_guard(
            student, teacher, labels, COUNTS
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(stats["many_inputs"], 0)
        loss.backward()
        self.assertIsNotNone(student.grad)

    def test_student_can_improve_margin_without_penalty(self):
        teacher = self.teacher
        student = torch.tensor(
            [[8.0, 0.0, 1.0, -1.0], [0.0, 7.0, 2.0, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        loss, stats = selective_many_tail_margin_guard(
            student, teacher, self.labels, COUNTS
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(stats["active_guard_inputs"], 0)

    def test_rejects_invalid_shapes_and_class_counts(self):
        with self.assertRaises(ValueError):
            selective_many_tail_margin_guard(
                torch.randn(2, 4), torch.randn(3, 4), self.labels, COUNTS
            )
        with self.assertRaises(ValueError):
            selective_many_tail_margin_guard(
                torch.randn(2, 4), torch.randn(2, 4), self.labels, [500, 300]
            )


if __name__ == "__main__":
    unittest.main()
