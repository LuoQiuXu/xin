"""Fast CPU unit tests for the validation-only transfer-risk probe."""
import unittest

import numpy as np
import torch

from transfer_risk_probe import (
    ARMS,
    CANDIDATES_PER_RECEIVER,
    build_receiver_masks,
    binary_auc,
    evaluate_arm,
    make_candidate_donors,
    average_ranks,
    target_margin,
)


class TransferRiskProbeTests(unittest.TestCase):
    def test_average_ranks_and_auc_ties(self):
        np.testing.assert_allclose(average_ranks(np.array([4.0, 2.0, 2.0, 9.0])), [3.0, 1.5, 1.5, 4.0])
        self.assertAlmostEqual(binary_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(binary_auc(np.array([0, 0, 1, 1]), np.array([0.8, 0.9, 0.1, 0.2])), 0.0)
        self.assertIsNone(binary_auc(np.array([0, 1]), np.array([1.0, 1.0])))
        self.assertIsNone(binary_auc(np.array([1, 1]), np.array([0.1, 0.9])))

    def test_target_margin_uses_true_label_and_excludes_it_from_competitor(self):
        logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 2.0]])
        labels = torch.tensor([0, 2])
        torch.testing.assert_close(target_margin(logits, labels), torch.tensor([1.0, -1.0]))

    def _synthetic_inputs(self):
        # Every class occurs in every domain, so conditional pools are non-empty.
        train_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        train_domains = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        val_labels = np.array([0, 1, 0, 1], dtype=np.int64)
        val_domains = np.array([0, 1, 2, 3], dtype=np.int64)
        global_plan = np.tile(np.array([0, 4, 1, 5], dtype=np.int64), 4)[None, :]
        mask_plan = np.arange(128, dtype=np.int64)[None, None, :]
        inputs = {
            "train": {
                "labels": train_labels,
                "domains": train_domains,
                "features": np.random.default_rng(11).normal(size=(8, 512)).astype(np.float32),
                "raw_indices": np.arange(100, 108, dtype=np.int64),
            },
            "val": {
                "labels": val_labels,
                "domains": val_domains,
                "features": np.random.default_rng(12).normal(size=(4, 512)).astype(np.float32),
                "raw_indices": np.arange(200, 204, dtype=np.int64),
            },
            "plan": {"global": global_plan, "masks": mask_plan},
            "head": torch.nn.Linear(512, 65),
        }
        return inputs

    def test_candidate_donors_preserve_donor_class_and_domain_constraints(self):
        inputs = self._synthetic_inputs()
        donors = make_candidate_donors(inputs, seed=42)
        train_labels = inputs["train"]["labels"]
        train_domains = inputs["train"]["domains"]
        val_domains = inputs["val"]["domains"]
        classes = train_labels[donors["H2T-Global"]]
        for arm in ARMS:
            np.testing.assert_array_equal(train_labels[donors[arm]], classes)
        recipient_domains = np.repeat(val_domains[:, None], CANDIDATES_PER_RECEIVER, axis=1)
        self.assertTrue(np.all(train_domains[donors["H2T-DomainMatched"]] == recipient_domains))
        self.assertTrue(np.all(train_domains[donors["H2T-CrossDomain"]] != recipient_domains))
        self.assertFalse(donors["H2T-DomainMatched_fallback"].any())
        self.assertFalse(donors["H2T-CrossDomain_fallback"].any())
        masks = build_receiver_masks(donors["mask_plan"], len(val_domains))
        self.assertEqual(masks.shape, (4, 512))
        np.testing.assert_array_equal(masks.sum(axis=1), np.full(4, 128))

    def test_evaluation_returns_finite_paired_risk_scores(self):
        torch.manual_seed(13)
        inputs = self._synthetic_inputs()
        donors = make_candidate_donors(inputs, seed=43)
        masks = build_receiver_masks(donors["mask_plan"], len(inputs["val"]["labels"]))
        result = evaluate_arm(
            inputs, 43, "H2T-Global", donors["H2T-Global"],
            np.zeros_like(donors["H2T-Global"], dtype=bool), masks,
        )
        self.assertEqual(result["delta_ce"].shape, (4, CANDIDATES_PER_RECEIVER))
        self.assertEqual(result["delta_margin"].shape, (4, CANDIDATES_PER_RECEIVER))
        self.assertTrue(np.isfinite(result["delta_ce"]).all())
        self.assertTrue(np.isfinite(result["predictors"]["directional_predicted_delta_ce"]).all())
        self.assertTrue(np.isfinite(result["predictors"]["masked_relative_l2"]).all())
        self.assertEqual(result["donor_indices"].shape, (4, CANDIDATES_PER_RECEIVER))


if __name__ == "__main__":
    unittest.main(verbosity=2)
