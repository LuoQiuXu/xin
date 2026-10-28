"""Offline logic tests; synthetic data are never presented as scientific results."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as a


def demo_logits():
    y = np.repeat(np.arange(1, 100, dtype=np.int64), 20)
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(len(y), 100))
    logits[np.arange(len(y)), y] += 2
    return y, logits


class GroupBiasTests(unittest.TestCase):
    def test_group_shift_invariant_to_within_group_rank(self):
        y, logits = demo_logits()
        shifted = logits.copy()
        shifted[:, 36:71] += 40
        shifted[:, 71:100] -= 25
        self.assertEqual(a.within_group(logits, y), a.within_group(shifted, y))
        self.assertNotEqual(a.group_metrics(logits.argmax(1), y),
                            a.group_metrics(shifted.argmax(1), y))

    def test_only_within_tail_correction_changes_tail_rank(self):
        y, dr = demo_logits()
        h2t = dr.copy()
        where = np.flatnonzero(y == 71)[:9]
        dr[where, 71:100] = -10
        h2t[where, 71:100] = -10
        dr[where, 71], dr[where, 72] = 0.0, 1.0
        h2t[where, 71], h2t[where, 72] = 2.0, 1.0
        d = a.within_group(dr, y)["Tail"]
        h = a.within_group(h2t, y)["Tail"]
        self.assertEqual(h["correct"] - d["correct"], 9)
        self.assertGreater(h["accuracy"], d["accuracy"])

    def test_grid_is_bounded_and_zero_offset_reproduces_dr(self):
        y, dr = demo_logits()
        h = dr.copy()
        grid, info = a.oracle_grid(dr, h, y)
        self.assertEqual(info["grid_points"], 1089)
        self.assertTrue(info["oracle_covers_h2t"])
        at_zero = next(r for r in grid if r["medium_shift_in_gap_units"] == 0
                       and r["tail_shift_in_gap_units"] == 0)
        true = a.group_metrics(dr.argmax(1), y)
        for k in true:
            self.assertAlmostEqual(at_zero[k], true[k], places=10)

    def test_cannot_claim_general_calibrator_from_oracle(self):
        y, dr = demo_logits()
        _, meta = a.oracle_grid(dr, dr.copy(), y)
        self.assertTrue(meta["group_bias_search_is_validation_oracle"])

    def test_validation_labels_must_be_train_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            np.save(d/"validation_labels.npy", np.zeros(1980, dtype=np.int64))
            np.savez(d/"split_indices.npz",
                     validation_raw_indices=np.arange(1980),
                     train_raw_indices=np.arange(1980, 5000))
            with self.assertRaises(ValueError):
                a.validate_holdout(d)

    def test_split_overlap_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            np.save(d/"validation_labels.npy", np.repeat(np.arange(1, 100), 20))
            np.savez(d/"split_indices.npz",
                     validation_raw_indices=np.arange(1980),
                     train_raw_indices=np.arange(500, 2600))
            with self.assertRaises(ValueError):
                a.validate_holdout(d)

    def test_run_missing_model_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/"H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014"
            old.mkdir(parents=True)
            y, _ = demo_logits()
            np.save(old/"validation_labels.npy", y)
            np.savez(old/"split_indices.npz",
                     train_raw_indices=np.arange(3000, 4000),
                     validation_raw_indices=np.arange(1980))
            out = Path("H2T_repro_audit/h2t_bias_discrimination_20261028/results/test")
            with self.assertRaises(FileNotFoundError):
                a.run(root, out)
            self.assertFalse((root/out).exists())

    def test_actual_synthetic_schema_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/"H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014"
            old.mkdir(parents=True)
            y, dr = demo_logits()
            h2t = dr.copy()
            h2t[np.arange(len(y)), y] += 0.2
            np.save(old/"validation_labels.npy", y)
            np.savez(old/"split_indices.npz",
                     train_raw_indices=np.arange(3000, 4000),
                     validation_raw_indices=np.arange(1980))
            base = root/"H2T_repro_audit/if200_stress/results"
            for seed in a.SEEDS:
                for arm, logits in zip(a.ARMS, (dr, h2t)):
                    f = base/f"seed{seed}"/arm
                    f.mkdir(parents=True)
                    metrics = a.group_metrics(logits.argmax(1), y)
                    np.savez(f/"validation_epoch10.npz",
                             labels=y, logits=logits,
                             predictions=logits.argmax(1))
                    (f/"summary.json").write_text(
                        __import__("json").dumps({
                            "completed": True, "seed": seed, "arm": arm,
                            "stage1_sha": a.PINNED_S1,
                            "classifier_initial_hash": a.PINNED_INIT,
                            "steps": 750, "epochs": 10,
                            "backbone_unchanged": True, "BN_frozen": True,
                            "TEST_used": False,
                            "paired_all_batches": arm != a.ARMS[0],
                            "final": metrics,
                        }), encoding="utf-8")
            out = Path("H2T_repro_audit/h2t_bias_discrimination_20261028/results/fixture")
            z = a.run(root, out)
            self.assertEqual(z["status"], "COMPLETE")
            self.assertEqual(len(z["per_seed"]), 2)
            self.assertTrue((root/out/"SCIENTIFIC_REPORT.md").exists())
            with self.assertRaises(FileExistsError):
                a.run(root, out)


if __name__ == "__main__":
    unittest.main()
