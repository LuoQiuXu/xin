"""CPU-only tests for the Many-only head-preservation KL term."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from train_ab import (
    DISTILL_TEMPERATURE,
    head_preservation_kl,
)


class HeadPreservationKLDistillationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261109)
        self.counts = torch.full((100,), 10, dtype=torch.long)
        self.counts[0] = 101
        self.counts[1] = 100
        self.counts[2] = 500
        self.labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        self.student = torch.randn(4, 100, dtype=torch.float32, requires_grad=True)
        self.teacher = torch.randn(4, 100, dtype=torch.float32, requires_grad=True)

    def test_strict_many_threshold_101_included_100_excluded(self):
        loss, mask, n_many = head_preservation_kl(
            self.student, self.teacher, self.labels, self.counts, 2.0
        )
        self.assertEqual(mask.tolist(), [True, False, True, False])
        self.assertEqual(n_many, 2)
        self.assertTrue(torch.isfinite(loss))

    def test_kl_matches_explicit_teacher_to_student_formula(self):
        loss, mask, n_many = head_preservation_kl(
            self.student, self.teacher, self.labels, self.counts, 2.0
        )
        expected = F.kl_div(
            F.log_softmax(self.student[mask] / 2.0, dim=1),
            F.softmax(self.teacher[mask].detach() / 2.0, dim=1),
            reduction="batchmean",
        ) * 4.0
        self.assertEqual(n_many, 2)
        self.assertTrue(torch.allclose(loss, expected, rtol=1e-6, atol=1e-7))

    def test_nonmany_rows_do_not_contribute_to_kd_or_student_gradient(self):
        loss, mask, _ = head_preservation_kl(
            self.student, self.teacher, self.labels, self.counts, 2.0
        )
        loss.backward()
        self.assertIsNotNone(self.student.grad)
        self.assertTrue(torch.isfinite(self.student.grad).all())
        self.assertGreater(float(self.student.grad[mask].abs().sum()), 0.0)
        self.assertTrue(torch.equal(
            self.student.grad[~mask],
            torch.zeros_like(self.student.grad[~mask]),
        ))
        self.assertIsNone(self.teacher.grad)

    def test_no_many_batch_returns_differentiable_zero(self):
        labels = torch.tensor([1, 3, 4, 5], dtype=torch.long)
        student = torch.randn(4, 100, requires_grad=True)
        teacher = torch.randn(4, 100, requires_grad=True)
        loss, mask, n_many = head_preservation_kl(
            student, teacher, labels, self.counts, DISTILL_TEMPERATURE
        )
        self.assertEqual(n_many, 0)
        self.assertFalse(bool(mask.any()))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student.grad)))
        self.assertIsNone(teacher.grad)

    def test_equal_teacher_and_student_logits_have_zero_kl(self):
        logits = torch.randn(3, 100, dtype=torch.float32, requires_grad=True)
        labels = torch.tensor([0, 2, 3], dtype=torch.long)
        loss, mask, n_many = head_preservation_kl(
            logits, logits.detach().clone(), labels, self.counts, 2.0
        )
        self.assertEqual(n_many, 2)
        self.assertTrue(bool(mask.tolist() == [True, True, False]))
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)

    def test_bad_temperature_and_bad_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            head_preservation_kl(
                self.student, self.teacher, self.labels, self.counts, 0.0
            )
        with self.assertRaises(ValueError):
            head_preservation_kl(
                self.student, self.teacher, self.labels, torch.ones(99), 2.0
            )


if __name__ == "__main__":
    unittest.main()
