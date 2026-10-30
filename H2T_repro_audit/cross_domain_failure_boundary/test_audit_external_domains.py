import unittest
from pathlib import Path

from audit_external_domains import (
    allocate_group_splits,
    infer_f4k_class_id,
    merge_leakage_groups,
)


class GroupAuditTests(unittest.TestCase):
    def test_f4k_folder_id_parser(self):
        self.assertEqual(infer_f4k_class_id(Path("fish_01/123_0001.jpg")), 1)
        self.assertEqual(infer_f4k_class_id(Path("01.Dascyllus reticulatus/123_0001.jpg")), 1)
        self.assertIsNone(infer_f4k_class_id(Path("mask_01/123_0001.jpg")))

    def test_group_split_keeps_groups_disjoint(self):
        records = []
        for label in ("A", "B"):
            for group in range(20):
                records.append({"label": label, "group_id": f"{label}_{group}"})
        assignments, eligible, reasons = allocate_group_splits(
            records, {"train": 0.6, "val": 0.2, "test": 0.2},
            seed=42, min_train=9, min_val=3, min_test=3, primary=True)
        self.assertEqual(set(eligible), {"A", "B"})
        self.assertFalse(reasons)
        for label in ("A", "B"):
            split_groups = {}
            for record in records:
                if record["label"] == label:
                    split_groups.setdefault(assignments[record["group_id"]], set()).add(record["group_id"])
            self.assertEqual(len(split_groups["train"]), 12)
            self.assertEqual(len(split_groups["val"]), 4)
            self.assertEqual(len(split_groups["test"]), 4)

    def test_rare_classes_are_excluded_from_primary_not_silently_dropped(self):
        records = []
        for group in range(20):
            records.append({"label": "supported", "group_id": f"s_{group}"})
        for group in range(15):
            records.append({"label": "supported2", "group_id": f"t_{group}"})
        for group in range(6):
            records.append({"label": "rare", "group_id": f"r_{group}"})
        assignments, eligible, reasons = allocate_group_splits(
            records, {"train": 0.6, "val": 0.2, "test": 0.2},
            seed=42, min_train=9, min_val=3, min_test=3, primary=True)
        self.assertEqual(set(eligible), {"supported", "supported2"})
        self.assertIn("rare", reasons)
        self.assertNotIn("r_0", assignments)

    def test_duplicate_image_hashes_merge_groups(self):
        records = [
            {"image_id": "a", "label": "A", "base_group": "track:1", "image_sha256": "same"},
            {"image_id": "b", "label": "A", "base_group": "track:2", "image_sha256": "same"},
            {"image_id": "c", "label": "A", "base_group": "track:3", "image_sha256": "other"},
        ]
        links = merge_leakage_groups(records)
        self.assertEqual(links, 1)
        self.assertEqual(records[0]["group_id"], records[1]["group_id"])
        self.assertNotEqual(records[0]["group_id"], records[2]["group_id"])

    def test_conflicting_labels_inside_duplicate_group_fail(self):
        records = [
            {"image_id": "a", "label": "A", "base_group": "track:1", "image_sha256": "same"},
            {"image_id": "b", "label": "B", "base_group": "track:2", "image_sha256": "same"},
        ]
        with self.assertRaises(RuntimeError):
            merge_leakage_groups(records)


if __name__ == "__main__":
    unittest.main()
