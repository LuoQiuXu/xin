"""CPU-only synthetic contracts; these tests are not a semantic experiment."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from coverage import TASK, DEFAULT_OUT, official_mapping, class_availability
from extract_features import draw_candidates


class Contracts(unittest.TestCase):
    def setUp(self):
        self.root = DEFAULT_OUT / "primary_cache"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=self.root)
        self.path = Path(self.tmp.name).resolve()
        assert self.path.is_relative_to(self.root.resolve())

    def tearDown(self):
        assert self.path.is_relative_to(self.root.resolve())
        self.tmp.cleanup()

    def test_official_mapping_requires_consistent_fine_coarse_pairs(self):
        np.testing.assert_array_equal(official_mapping(np.array([0, 0, 1, 1]),
            np.array([1, 1, 0, 0]), classes=2), [1, 0])
        with self.assertRaises(ValueError):
            official_mapping(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 0]), classes=2)

    def test_mapping_rejects_noninteger_labels_and_missing_fine_class(self):
        with self.assertRaises(ValueError):
            official_mapping(np.array([0., 1.]), np.array([0, 1]), classes=2)
        with self.assertRaises(ValueError):
            official_mapping(np.array([0, 0]), np.array([0, 0]), classes=2)

    def test_few_many_eligibility_does_not_borrow_medium_donors(self):
        counts = np.array([101, 100, 20, 19, 5, 150])
        mapping = np.array([0, 1, 1, 0, 1, 2])
        same, different, eligible = class_availability(counts, mapping)
        self.assertEqual(int(same[3]), 101)
        self.assertEqual(int(same[4]), 0)
        np.testing.assert_array_equal(eligible, [False, False, False, True, False, False])
        self.assertEqual(int(different[4]), 251)

    def test_fixed_candidates_preserve_semantic_arms_and_raw_ids(self):
        fine = np.array([0, 0, 0, 1, 1, 1, 2, 3])
        mapping = np.array([0, 1, 0, 1])
        recipients, many = np.array([6, 7]), np.arange(6)
        same, different = draw_candidates(recipients, many, fine, mapping, 2)
        for row, rid in enumerate(recipients):
            self.assertEqual(len(np.unique(same[row])), 2)
            self.assertEqual(len(np.unique(different[row])), 2)
            self.assertTrue(np.isin(same[row], many).all())
            self.assertTrue((mapping[fine[same[row]]] == mapping[fine[rid]]).all())
            self.assertTrue((mapping[fine[different[row]]] != mapping[fine[rid]]).all())
            self.assertFalse(np.isin(rid, np.r_[same[row], different[row]]))

    def test_candidate_draw_is_order_independent_and_does_not_broaden(self):
        fine, mapping = np.array([0, 0, 0, 1, 1, 1, 2, 3]), np.array([0, 1, 0, 1])
        first = draw_candidates(np.array([6, 7]), np.arange(6), fine, mapping, 2)
        second = draw_candidates(np.array([7, 6]), np.arange(6), fine, mapping, 2)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b[::-1])
        with self.assertRaises(ValueError):
            draw_candidates(np.array([6, 7]), np.arange(6), fine, mapping, 4)

    def make_pairs(self):
        # Exact within-recipient distance equality gives a known zero contrast.
        r = np.array([.2, .4, .6, .8])
        rec = np.tile(np.array([2., 0.]), (8, 1))
        donor = np.concatenate([np.column_stack([2-r, np.zeros(4)]),
                                np.column_stack([np.full(4, 2.), r])])
        return dict(rec_id=np.full(8, 1, dtype=np.int64), rec_label=np.zeros(8, dtype=np.int64),
            donor_id=np.arange(100, 108, dtype=np.int64), donor_label=np.array([1]*4+[2]*4),
            rec_feat=rec, donor_feat=donor, coarse_label=np.array([0, 0, 1, 1]),
            weight=np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]), bias=np.zeros(4))

    def run_screen(self, arrays=None, mass="0.25"):
        p = self.path / "pairs.npz"
        np.savez_compressed(p, **(self.make_pairs() if arrays is None else arrays))
        output = self.path / "output"
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(TASK / "screen.py"),
            "--input", str(p), "--output", str(output), "--lambda-fixed", mass],
            capture_output=True, text=True, encoding="utf-8")
        return result, output

    def test_provided_screen_known_null_at_exactly_matched_distance(self):
        result, out = self.run_screen()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["n_matched_pairs"], 4)
        self.assertEqual(summary["n_matched_recipients"], 1)
        self.assertEqual(summary["matched_same_minus_diff_accuracy_pp"], 0.)
        self.assertAlmostEqual(summary["matched_same_minus_diff_margin"], 0.)
        self.assertAlmostEqual(summary["mean_absolute_distance_gap"], 0.)
        self.assertEqual(summary["mean_mass_gap"], 0.)
        self.assertFalse(summary["screen_sufficient_pairs"])

    def test_repeated_pair_is_rejected(self):
        x = self.make_pairs()
        x["donor_id"][1] = x["donor_id"][0]
        result, _ = self.run_screen(x)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("repeated pair", result.stderr)

    def test_changing_cached_recipient_is_rejected(self):
        x = self.make_pairs()
        x["rec_feat"][1, 0] += 1
        result, _ = self.run_screen(x)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("recipient feature/label differs", result.stderr)

    def test_nonfinite_feature_and_illegal_mass_are_rejected(self):
        x = self.make_pairs()
        x["donor_feat"][0, 0] = np.nan
        result, _ = self.run_screen(x)
        self.assertNotEqual(result.returncode, 0)
        result, _ = self.run_screen(mass="1.2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid donor mass", result.stderr)


if __name__ == "__main__":
    unittest.main()
