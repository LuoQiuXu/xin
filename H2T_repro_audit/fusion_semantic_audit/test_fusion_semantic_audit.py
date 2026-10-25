import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch

import audit_fusion_semantics as audit

from audit_fusion_semantics import (
    apply_channel_mask,
    average_ranks,
    margin_and_competitor,
    pair_excluded_ncm_scores,
    spearman_rho,
    normalize_rows,
)


class FusionSemanticAuditUnitTests(unittest.TestCase):
    def test_channel_mask_replaces_only_selected_coordinates(self):
        receiver = np.zeros((3, 64), dtype=np.float32)
        donor = np.ones((3, 64), dtype=np.float32)
        mask = np.tile(np.arange(16, dtype=np.uint8), (3, 1))
        fused = apply_channel_mask(receiver, donor, mask)
        self.assertTrue(np.all(fused[:, :16] == 1.0))
        self.assertTrue(np.all(fused[:, 16:] == 0.0))

    def test_margin_and_competitor(self):
        scores = np.zeros((2, 100), dtype=np.float64)
        labels = np.array([3, 7], dtype=np.int64)
        scores[0, 3], scores[0, 4] = 2.0, 1.0
        scores[1, 7], scores[1, 8] = 0.5, 1.0
        pred, margin, competitor = margin_and_competitor(scores, labels)
        np.testing.assert_array_equal(pred, np.array([3, 8]))
        np.testing.assert_allclose(margin, np.array([1.0, -0.5]))
        np.testing.assert_array_equal(competitor, np.array([4, 8]))

    def test_average_ranks_handles_ties(self):
        ranks = average_ranks(np.array([10.0, 10.0, 20.0, 30.0]))
        np.testing.assert_allclose(ranks, np.array([1.5, 1.5, 3.0, 4.0]))

    def test_spearman_is_rank_correlation(self):
        self.assertAlmostEqual(spearman_rho(np.array([1, 2, 3, 4]), np.array([2, 4, 6, 8])), 1.0)
        self.assertAlmostEqual(spearman_rho(np.array([1, 2, 3, 4]), np.array([8, 6, 4, 2])), -1.0)
        self.assertIsNone(spearman_rho(np.array([1, 1, 1]), np.array([1, 2, 3])))

    def test_pair_excluded_ncm_scores_are_finite(self):
        features = np.array([
            [1.0, 0.0], [0.9, 0.1], [0.8, 0.2],
            [0.0, 1.0], [0.1, 0.9], [0.2, 0.8],
        ], dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        normalized = normalize_rows(features)
        sums = np.stack([normalized[labels == c].sum(axis=0) for c in range(2)] + [
            np.ones(2, dtype=np.float64) for _ in range(98)
        ])
        counts = np.bincount(labels, minlength=100).astype(np.int64)
        counts[2:] = 3
        z = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float64)
        rec = np.array([0, 3], dtype=np.int64)
        don = np.array([3, 0], dtype=np.int64)
        # The synthetic bank uses nonzero placeholder prototypes for unused classes.
        scores = pair_excluded_ncm_scores(z, rec, don, labels, normalized, sums, counts)
        self.assertEqual(scores.shape, (2, 100))
        self.assertTrue(np.isfinite(scores).all())


class ReplayMismatchDiagnosticTests(unittest.TestCase):
    """Synthetic CPU arrays only; never replay a historical batch or model."""

    def setUp(self):
        self.receiver = np.arange(512, dtype=np.float32).reshape(2, 64, 2, 2)
        self.donor = self.receiver + np.float32(1)
        self.temp_parent = audit.HERE / "diagnostics"
        self.temp_parent.mkdir(exist_ok=True)

    def verify(self, expected, observed, receiver=None):
        receiver = self.receiver if receiver is None else receiver
        return audit.verify_replay_signatures(
            seed=124, epoch=10, batch=0, expected=expected, observed=observed,
            feature_arrays={"receiver": receiver, "donor": self.donor},
            feature_tensors={"receiver": torch.from_numpy(receiver),
                             "donor": torch.from_numpy(self.donor)},
        )

    def signatures(self, receiver=None):
        receiver = self.receiver if receiver is None else receiver
        observed = dict.fromkeys(audit.REPLAY_FIELDS, "a" * 64)
        observed["features_sha256"] = audit.array_hash(receiver, self.donor)
        expected = dict(observed)
        expected["features_sha256"] = "0" * 64
        return expected, observed

    def test_hash_matches_locked_historical_function(self):
        tree = ast.parse((audit.OLD / "common.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "array_hash")
        namespace = {"np": np, "hashlib": hashlib}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "historical_array_hash", "exec"), namespace)
        for arrays in ((self.receiver, self.donor),
                       (self.receiver[:, :, :, ::-1], self.donor.astype(np.float64))):
            self.assertEqual(audit.array_hash(*arrays), namespace["array_hash"](*arrays))

    def test_mismatch_is_persisted_before_exception(self):
        expected, observed = self.signatures()
        with tempfile.TemporaryDirectory(dir=self.temp_parent, prefix="unit_test_") as temp, \
                mock.patch.object(audit, "DIAGNOSTICS", Path(temp)), \
                mock.patch.object(audit, "replay_runtime_diagnostic", return_value={"scope": "synthetic unit test"}):
            with self.assertRaisesRegex(RuntimeError, "features_sha256.*Diagnostics persisted"):
                self.verify(expected, observed)
            paths = list(Path(temp).glob("replay_mismatch_*/replay_mismatch.json"))
            self.assertEqual(len(paths), 1)
            record = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(record["expected_sha256"], expected["features_sha256"])
            self.assertEqual(record["observed_sha256"], observed["features_sha256"])
            self.assertEqual(record["feature_arrays"]["receiver"]["shape"], [2, 64, 2, 2])
            self.assertEqual(record["feature_arrays"]["receiver"]["dtype"], "float32")
            self.assertTrue(record["feature_arrays"]["receiver"]["all_finite"])
            self.assertEqual(record["feature_tensors"]["receiver"]["device"], "cpu")
            self.assertEqual([row["match"] for row in record["checks_through_first_mismatch"]], [True, True, False])
            self.assertFalse(record["exact_replay_passed"])

    def test_near_equal_values_still_fail_exact_gate(self):
        changed = self.receiver.copy()
        changed.flat[0] = np.float32(1e-9)
        self.assertTrue(np.allclose(self.receiver, changed))
        observed = dict.fromkeys(audit.REPLAY_FIELDS, "a" * 64)
        observed["features_sha256"] = audit.array_hash(changed, self.donor)
        expected = dict(observed)
        expected["features_sha256"] = audit.array_hash(self.receiver, self.donor)
        with tempfile.TemporaryDirectory(dir=self.temp_parent, prefix="unit_test_") as temp, \
                mock.patch.object(audit, "DIAGNOSTICS", Path(temp)), \
                mock.patch.object(audit, "replay_runtime_diagnostic", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "Exact replay mismatch"):
                self.verify(expected, observed, changed)

    def test_nonfinite_features_are_recorded_without_invalid_json(self):
        nonfinite = self.receiver.copy()
        nonfinite.flat[:2] = [np.nan, np.inf]
        expected, observed = self.signatures(nonfinite)
        with tempfile.TemporaryDirectory(dir=self.temp_parent, prefix="unit_test_") as temp, \
                mock.patch.object(audit, "DIAGNOSTICS", Path(temp)), \
                mock.patch.object(audit, "replay_runtime_diagnostic", return_value={}):
            with self.assertRaises(RuntimeError):
                self.verify(expected, observed, nonfinite)
            path = next(Path(temp).glob("replay_mismatch_*/replay_mismatch.json"))
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(record["feature_arrays"]["receiver"]["all_finite"])
            self.assertEqual(record["feature_arrays"]["receiver"]["nonfinite_elements"], 2)

    def test_repeated_failures_never_overwrite_existing_diagnostic(self):
        expected, observed = self.signatures()
        with tempfile.TemporaryDirectory(dir=self.temp_parent, prefix="unit_test_") as temp, \
                mock.patch.object(audit, "DIAGNOSTICS", Path(temp)), \
                mock.patch.object(audit, "replay_runtime_diagnostic", return_value={}):
            with self.assertRaises(RuntimeError):
                self.verify(expected, observed)
            first = next(Path(temp).glob("replay_mismatch_*/replay_mismatch.json"))
            first_bytes = first.read_bytes()
            with self.assertRaises(RuntimeError):
                self.verify(expected, observed)
            self.assertEqual(len(list(Path(temp).glob("replay_mismatch_*/replay_mismatch.json"))), 2)
            self.assertEqual(first.read_bytes(), first_bytes)

    def test_matching_signatures_do_not_create_diagnostics(self):
        _, observed = self.signatures()
        with mock.patch.object(audit, "persist_replay_mismatch") as persist:
            rows = self.verify(dict(observed), observed)
            self.assertEqual(len(rows), len(audit.REPLAY_FIELDS))
            self.assertTrue(all(row["match"] for row in rows))
            persist.assert_not_called()

    def test_first_mismatch_order_is_unchanged(self):
        expected, observed = self.signatures()
        expected["indices_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=self.temp_parent, prefix="unit_test_") as temp, \
                mock.patch.object(audit, "DIAGNOSTICS", Path(temp)), \
                mock.patch.object(audit, "replay_runtime_diagnostic", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "field=indices_sha256"):
                self.verify(expected, observed)
            path = next(Path(temp).glob("replay_mismatch_*/replay_mismatch.json"))
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(record["checks_through_first_mismatch"]), 1)

    def test_diagnostic_io_failure_still_stops_replay(self):
        expected, observed = self.signatures()
        with mock.patch.object(Path, "mkdir", side_effect=PermissionError("synthetic write failure")) as mkdir, \
                mock.patch.object(audit, "replay_runtime_diagnostic", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "Diagnostic persistence failed; strict replay remains stopped"):
                self.verify(expected, observed)
            self.assertEqual(mkdir.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
