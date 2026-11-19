from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from causal_core import (
    ROOT, ALIASES, apply_pair_mask, array_hash, balance_diagnostics,
    calibration_competitors, canonical_hash, construct_masks,
    decompose_perturbation_gradient_by_source, deterministic_split,
    eligible_pairs, fixed_k, keep_mask, masked_optimizer_gradient,
)
from isolated_data import HoldoutVault, file_sha, read_npy_rows, seal_masks
from analyze_causal_oracle import metric_samples, stratified_draws


class SplitAndIsolationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".test_tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".test_tmp")
        self.root = Path(self.temp.name)
        labels = np.repeat(np.arange(1, 4, dtype=np.int64), 4)
        self.split = deterministic_split(np.arange(100, 112), labels, per_half=2)
        np.save(self.root / "validation_labels.npy", labels)
        np.save(self.root / "validation_gap_features.npy", np.arange(60, dtype=np.float32).reshape(12, 5))
        self.vault = HoldoutVault(self.root, self.split, self.root / "seal.json")

    def tearDown(self):
        self.vault.close()
        self.temp.cleanup()

    def make_seal(self):
        artifact = self.root / "mask.json"
        artifact.write_text('{"calibration_only":true}', encoding="utf-8")
        payload = {"seeds": [123, 124, 125], "split_sha256": canonical_hash(self.split),
                   "dependencies": {str(artifact): file_sha(artifact)},
                   "evaluation_unopened_before_seal": True}
        return seal_masks(self.root / "seal.json", payload)

    def test_split_deterministic_and_disjoint(self):
        other = deterministic_split(np.arange(100, 112), np.repeat(np.arange(1, 4), 4), per_half=2)
        self.assertEqual(self.split, other)
        self.assertFalse(set(other["calibration_raw_train_ids"]) & set(other["evaluation_raw_train_ids"]))
        self.assertEqual(len(other["calibration_positions"]), 6)

    def test_shuffled_row_order_preserves_raw_id_membership(self):
        permutation = np.random.default_rng(4).permutation(12)
        other = deterministic_split(np.arange(100, 112)[permutation],
                                    np.repeat(np.arange(1, 4), 4)[permutation], per_half=2)
        self.assertEqual(set(self.split["calibration_raw_train_ids"]), set(other["calibration_raw_train_ids"]))

    def test_wrong_class_count_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            deterministic_split([1, 2, 3], [1, 1, 1], per_half=2)
        with self.assertRaises(ValueError):
            deterministic_split([1, 1, 2, 3], [1] * 4, per_half=2)

    def test_calibration_read_without_evaluation(self):
        features, labels = self.vault.read("calibration")
        self.assertEqual(features.shape, (6, 5))
        self.assertEqual(np.bincount(labels).tolist(), [0, 2, 2, 2])
        self.assertEqual([e["partition"] for e in self.vault.events], ["calibration"])

    def test_evaluation_blocked_before_seal(self):
        with self.assertRaisesRegex(RuntimeError, "no mask seal"):
            self.vault.read("evaluation")

    def test_legacy_full_loader_and_raw_reader_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "isolated row"):
            np.load(self.root / "validation_labels.npy")
        with self.assertRaisesRegex(RuntimeError, "partition authorization"):
            read_npy_rows(self.root / "validation_labels.npy", [0, 1])
        with self.assertRaisesRegex(RuntimeError, "isolated row"):
            file_sha(self.root / "validation_gap_features.npy")

    def test_evaluation_allowed_only_after_valid_all_seed_seal(self):
        self.make_seal()
        _, labels = self.vault.read("evaluation")
        self.assertEqual(np.bincount(labels).tolist(), [0, 2, 2, 2])
        self.assertEqual(len(self.vault.sealed_integrity_hash("validation_labels.npy")), 64)

    def test_mask_tampering_blocks_evaluation(self):
        self.make_seal()
        (self.root / "mask.json").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "dependency changed"):
            self.vault.read("evaluation")

    def test_incomplete_seal_and_overwrite_rejected(self):
        with self.assertRaises(ValueError):
            seal_masks(self.root / "seal.json", {"seeds": [123]})
        self.make_seal()
        with self.assertRaises(FileExistsError):
            self.make_seal()


class MaskAndGradientTests(unittest.TestCase):
    def setUp(self):
        self.counts = np.array([500, 400, 300, 180, 95, 60, 45, 35, 18, 12, 9, 5])
        self.pairs = eligible_pairs(self.counts, np.arange(1, 12))
        self.scores = {p: float(np.sin(p[0] * 0.8 + p[1])) for p in self.pairs}

    def test_eligible_absent_class_and_half_up(self):
        self.assertTrue(all(s >= 4 and 1 <= t <= 3 for s, t in self.pairs))
        self.assertEqual([fixed_k(n) for n in (0, 4, 5, 14, 15, 2210)], [0, 0, 1, 1, 2, 221])

    def test_matched_controls_deterministic_frequency_and_target_counts(self):
        first = construct_masks(self.scores, self.pairs, self.counts, 123, k=12)
        second = construct_masks(self.scores, list(reversed(self.pairs)), self.counts, 123, k=12)
        self.assertEqual(first, second)
        balance = balance_diagnostics(first, self.counts)
        self.assertTrue(balance)
        for arm in ("TARGETED", "RANDOM_MATCHED", "FREQUENCY_MATCHED"):
            self.assertEqual(len(first["suppressed_pairs"][arm]), 12)
        self.assertNotEqual(first["random_seed"], first["frequency_seed"])

    def test_oracle_objective_and_alias(self):
        bundle = construct_masks(self.scores, self.pairs, self.counts, 123, k=3)
        retained = {tuple(p) for p in bundle["oracle_retained_pairs"]}
        expected = set(sorted(self.pairs, key=lambda p: (-self.scores[p], *p))[:3])
        self.assertEqual(retained, expected)
        self.assertEqual(bundle["aliases"], ALIASES)
        self.assertNotEqual(bundle["suppressed_pairs"]["TARGETED"], bundle["suppressed_pairs"]["ORACLE_SELECTION"])

    def test_k_zero_and_all(self):
        zero = construct_masks(self.scores, self.pairs, self.counts, 123, k=0)
        full = construct_masks(self.scores, self.pairs, self.counts, 123, k=len(self.pairs))
        self.assertFalse(zero["suppressed_pairs"]["TARGETED"])
        self.assertEqual(len(zero["suppressed_pairs"]["ORACLE_SELECTION"]), len(self.pairs))
        self.assertEqual(len(full["suppressed_pairs"]["TARGETED"]), len(self.pairs))
        self.assertFalse(full["suppressed_pairs"]["ORACLE_SELECTION"])

    def test_source_target_orientation_and_non_square_dimensions(self):
        weight = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7)
        bias = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        mask = keep_mask(3, 5, [(1, 4)])
        w, b = apply_pair_mask(weight, bias, mask)
        self.assertEqual(torch.count_nonzero(w[1, 4]).item(), 0)
        self.assertEqual(b[1, 4].item(), 0)
        self.assertTrue(torch.equal(w[2], weight[2]))
        self.assertTrue(torch.equal(w[1, 3], weight[1, 3]))

    def test_invalid_masks_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            keep_mask(3, 5, [(1, 4), (1, 4)])
        with self.assertRaises(ValueError):
            keep_mask(3, 5, [(4, 1)])
        bad = torch.zeros(3, 5, 7)
        bad[1, 2, 3] = float("nan")
        with self.assertRaises(RuntimeError):
            apply_pair_mask(bad, torch.zeros(3, 5), keep_mask(3, 5, []))
        bad[1, 2, 3] = float("inf")
        with self.assertRaises(RuntimeError):
            apply_pair_mask(bad, torch.zeros(3, 5), keep_mask(3, 5, []))

    def mathematical_batch(self, dtype):
        torch.manual_seed(19)
        x = torch.randn(13, 5, dtype=dtype)
        labels = torch.tensor([0, 1, 2, 4, 3, 1, 2, 3, 0, 4, 2, 4, 1])
        delta = torch.randn_like(x) * 0.2
        delta[labels == 0] = 0
        fc = torch.nn.Linear(5, 5, dtype=dtype)
        weights = torch.linspace(0.2, 1.4, 5, dtype=dtype)
        alpha = 0.3
        clean, adv = fc(x), fc(x + delta)
        loss2 = ((1 - alpha) * F.cross_entropy(clean, labels, reduction="none")
                 + alpha * weights[labels] * F.cross_entropy(clean, labels, reduction="none")).mean()
        loss3 = ((1 - alpha) * F.cross_entropy(clean, labels, reduction="none")
                 + alpha * weights[labels] * F.cross_entropy(adv, labels, reduction="none")).mean()
        g2 = torch.autograd.grad(loss2, (fc.weight, fc.bias), retain_graph=True)
        g3 = torch.autograd.grad(loss3, (fc.weight, fc.bias), retain_graph=True)
        sw, sb = decompose_perturbation_gradient_by_source(x, labels, delta, clean, adv, weights, alpha)
        return x, labels, delta, fc, weights, alpha, clean, adv, g2, g3, sw, sb

    def test_fp64_source_additivity_and_masked_autograd_reference(self):
        x, y, d, fc, weights, alpha, clean, adv, g2, g3, sw, sb = self.mathematical_batch(torch.float64)
        torch.testing.assert_close(sw.sum(0), g3[0] - g2[0], rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(sb.sum(0), g3[1] - g2[1], rtol=1e-12, atol=1e-14)
        self.assertEqual(torch.count_nonzero(sw[0]).item(), 0)
        keep = keep_mask(5, 5, [(2, 1), (4, 0)])
        actual = masked_optimizer_gradient(*g2, sw, sb, keep)
        # Independent autograd for each source's direct LOSS, then select only
        # the declared classifier rows. This is a gradient intervention.
        ref_w, ref_b = g2[0].clone(), g2[1].clone()
        for source in range(5):
            subset = y == source
            direct_loss = (alpha / len(y)) * (weights[y[subset]] * (
                F.cross_entropy(adv[subset], y[subset], reduction="none")
                - F.cross_entropy(clean[subset], y[subset], reduction="none"))).sum()
            rw, rb = torch.autograd.grad(direct_loss, (fc.weight, fc.bias), retain_graph=True)
            ref_w += rw * keep[source].unsqueeze(1)
            ref_b += rb * keep[source]
        torch.testing.assert_close(actual[0], ref_w, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(actual[1], ref_b, rtol=1e-12, atol=1e-14)

    def test_fp32_gradient_reconstruction_original_tolerances(self):
        *_, g2, g3, sw, sb = self.mathematical_batch(torch.float32)
        predicted = masked_optimizer_gradient(*g2, sw, sb, keep_mask(5, 5, []))
        torch.testing.assert_close(predicted[0], g3[0], rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(predicted[1], g3[1], rtol=2e-4, atol=2e-5)

    def test_all_direct_zero_preserves_clean_gradient(self):
        *_, g2, g3, sw, sb = self.mathematical_batch(torch.float64)
        zero_mask = torch.zeros(5, 5, dtype=torch.bool)
        actual = masked_optimizer_gradient(*g2, sw, sb, zero_mask)
        self.assertTrue(torch.equal(actual[0], g2[0]))
        self.assertTrue(torch.equal(actual[1], g2[1]))

    def test_competitors_calibration_class_map_not_evaluation_selection(self):
        counts = np.array([500, 400, 20, 5])
        logits = np.array([[1, 1, 8, 1], [1, 1, 2, 7], [1, 1, 1, 9], [1, 1, 1, 8]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        result = calibration_competitors(logits, labels, counts)
        self.assertEqual(result, {0: 2, 1: 3})
        self.assertEqual(canonical_hash(result), canonical_hash({1: 3, 0: 2}))


class BootstrapAndProductionTests(unittest.TestCase):
    def test_cpu_imports_actual_production_modules(self):
        import run_causal_oracle
        import oracle_runtime
        self.assertEqual(run_causal_oracle.NSTEPS, 5100)
        self.assertEqual(oracle_runtime.STAGE2_LR, 0.01)
        self.assertEqual(oracle_runtime.WEIGHT_DECAY, 0.0002)

    def test_stratified_draws_deterministic_and_preserve_counts(self):
        labels = np.array([1, 3, 1, 2, 3, 1, 3, 2, 3])
        a = stratified_draws(labels, repetitions=31, seed=19)
        b = stratified_draws(labels, repetitions=31, seed=19)
        self.assertTrue(np.array_equal(a, b))
        for row in a:
            self.assertTrue(np.array_equal(np.bincount(labels[row]), np.bincount(labels)))

    def test_paired_identical_predictions_have_exact_zero_effect_and_interval(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        counts = np.array([500, 35, 5])
        logits = np.array([[2, 1, 0], [1, 2, 0], [1, 2, 0], [0, 1, 2], [0, 1, 2], [2, 1, 0]], dtype=float)
        draws = stratified_draws(labels, 43, 2)
        point, bootstrap = metric_samples(logits, labels, counts, {"0": 1}, draws)
        other_point, other_bootstrap = metric_samples(logits.copy(), labels, counts, {"0": 1}, draws)
        for key in point:
            self.assertEqual(point[key] - other_point[key], 0)
            self.assertTrue(np.array_equal(bootstrap[key] - other_bootstrap[key], np.zeros(43)))

    def test_bootstrap_macro_is_class_mean_with_unequal_class_counts(self):
        labels = np.array([0, 0, 0, 0, 1, 2])
        logits = np.tile(np.array([2, 0, 0]), (6, 1))
        points, _ = metric_samples(logits, labels, np.array([500, 35, 5]), {"0": 1},
                                   stratified_draws(labels, 20, 9))
        self.assertAlmostEqual(points["overall_accuracy"], 200 / 3)
        self.assertAlmostEqual(points["macro_accuracy"], 100 / 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
