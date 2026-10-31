import unittest
import csv
import io
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from audit_external_domains import (
    allocate_group_splits,
    infer_f4k_class_id,
    merge_leakage_groups,
    inspect_image,
    load_isic,
    load_f4k,
    reserve_output_directory,
    run_audit,
    main,
    ISIC_LABELS,
    F4K_CLASSES,
    verify_f4k_class_mapping,
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

    def test_truncated_jpeg_requires_real_pixel_decode(self):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (12, 52, 129)).save(buffer, format="JPEG")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.jpg"
            path.write_bytes(buffer.getvalue()[:-20])
            with self.assertRaises(OSError):
                inspect_image(path)

    def test_isic_exact_one_hot_duplicate_ids_and_missing_lesion_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (4, 4), "red").save(images / "valid.png")
            metadata, ground_truth = root / "meta.csv", root / "gt.csv"
            with metadata.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image", "lesion_id"])
                writer.writeheader()
                writer.writerow({"image": "valid", "lesion_id": ""})
            with ground_truth.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image", *ISIC_LABELS])
                writer.writeheader()
                row = dict.fromkeys(ISIC_LABELS, 0)
                row.update(image="valid", MEL=1)
                writer.writerow(row)
                writer.writerow(row)
                for image_id, value in (("fraction", 0.6), ("nan", "nan"), ("negative", -1)):
                    row = dict.fromkeys(ISIC_LABELS, 0)
                    row.update(image=image_id, MEL=value)
                    writer.writerow(row)
            records, issues, _, _ = load_isic(images, metadata, ground_truth)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["group_source"], "image_id_fallback")
            self.assertEqual(issues["missing_lesion_ids"], ["valid"])
            self.assertEqual(len(issues["invalid_rows"]), 4)

    def test_numeric_fish_id_remains_global_across_zero_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fish_01").mkdir()
            for name in ("fish_123_00001.png", "fish_456_1.png"):
                Image.new("RGB", (4, 4), "red").save(root / "fish_01" / name)
            records, issues, _, _ = load_f4k(root)
            self.assertEqual(len(records), 1)
            self.assertIn("duplicate global fish ID", issues["invalid_rows"][0])

    def test_release_track_counts_and_deduplicated_support_are_separate(self):
        records = [
            {"image_id": str(i), "relative_path": f"{i}.png", "label": "A",
             "base_group": f"track:{i}", "group_source": "tracking_id",
             "image_sha256": "shared" if i < 2 else str(i), "width": 1, "height": 1}
            for i in range(15)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            args = Namespace(dataset="fish4knowledge", images_root=str(images),
                             output_dir=str(root / "output"), seed=42)
            with patch("audit_external_domains.load_f4k", return_value=(records, {}, {"A": 15}, {"A": 15})):
                with patch("builtins.print"):
                    summary = run_audit(args)
            self.assertTrue(summary["expected_release_counts_match"])
            self.assertEqual(summary["raw_group_ids_per_class"]["A"], 15)
            self.assertEqual(summary["independent_groups_per_class"]["A"], 14)
            self.assertEqual(summary["status"], "NO_GO_INDEPENDENT_GROUP_SUPPORT")
            self.assertTrue((root / "output" / "manifest_all.csv").exists())

    def test_output_reservation_never_overwrites_or_writes_in_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            output = root / "output"
            reserve_output_directory(images, output)
            sentinel = output / "existing.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                reserve_output_directory(images, output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            with self.assertRaises(RuntimeError):
                reserve_output_directory(images, Path(__file__).resolve().parent / "forbidden_output")

    def test_missing_input_persists_failure_without_invented_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(dataset="isic2019", images_root=str(root / "missing_images"),
                             output_dir=str(root / "output"), seed=42,
                             metadata_csv=str(root / "missing_meta.csv"),
                             ground_truth_csv=str(root / "missing_gt.csv"))
            with patch("audit_external_domains.parse_args", return_value=args):
                with patch("sys.stderr", new=io.StringIO()):
                    self.assertEqual(main(), 2)
            summary = json.loads((root / "output" / "audit_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "BLOCKED_MISSING_DATA")
            self.assertIsNone(summary["n_valid_images"])
            self.assertFalse(summary["audit_complete"])
            self.assertTrue((root / "output" / "audit_issues.json").is_file())
            for name in ("manifest_all.csv", "manifest_primary.csv", "class_counts.csv"):
                with (root / "output" / name).open(newline="", encoding="utf-8") as handle:
                    self.assertEqual(list(csv.DictReader(handle)), [])

    def test_mapping_decimal_padding_matches_without_changing_source_ids(self):
        records = [
            {"image_id": "fish_000420000101_00001", "label": F4K_CLASSES[1][0], "base_group": "track:fish_000420000101"},
            {"image_id": "fish_000004210001_00013", "label": F4K_CLASSES[6][0], "base_group": "track:fish_000004210001"},
        ]
        original = [record.copy() for record in records]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "class_id.csv"
            path.write_text("file_name,cluster_id\n00420000101_00001,1\n00004210001_00013,6\n", encoding="utf-8")
            result = verify_f4k_class_mapping(records, path)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["n_matched_rows"], 2)
        self.assertEqual(result["matches_requiring_leading_zero_normalization"], 2)
        self.assertEqual(records, original)

    def test_mapping_same_fish_id_with_wrong_track_is_rejected(self):
        records = [{"image_id": "fish_000420000101_00001", "label": F4K_CLASSES[1][0]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "class_id.csv"
            path.write_text("file_name,cluster_id\n999_00001,1\n", encoding="utf-8")
            result = verify_f4k_class_mapping(records, path)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["n_matched_rows"], 0)
        self.assertEqual(result["unmatched_images"], 1)

    def test_mapping_missing_duplicate_and_wrong_category_fail(self):
        records = [
            {"image_id": "fish_000420000101_00001", "label": F4K_CLASSES[1][0]},
            {"image_id": "fish_000004210001_00013", "label": F4K_CLASSES[6][0]},
        ]
        bodies = ("00420000101_00001,1\n", "00420000101_00001,1\n00420000101_00001,1\n",
                  "00420000101_00001,6\n00004210001_00013,6\n")
        for body in bodies:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "class_id.csv"
                path.write_text("file_name,cluster_id\n" + body, encoding="utf-8")
                result = verify_f4k_class_mapping(records, path)
                self.assertEqual(result["status"], "FAIL")

    def test_mapping_mismatch_reaches_strict_integrity_gate(self):
        label = F4K_CLASSES[1][0]
        records = [
            {"image_id": f"fish_{i + 1:012d}_{i + 1:05d}", "relative_path": f"{i}.png", "label": label,
             "base_group": f"track:{i}", "group_source": "tracking_id",
             "image_sha256": str(i), "width": 1, "height": 1} for i in range(15)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            path = root / "class_id.csv"
            path.write_text("file_name,cluster_id\n" + "".join(
                f"{i + 1}_{i + 1},{6 if i == 0 else 1}\n" for i in range(15)), encoding="utf-8")
            args = Namespace(dataset="fish4knowledge", images_root=str(images),
                             output_dir=str(root / "output"), seed=42, class_id_csv=str(path))
            with patch("audit_external_domains.load_f4k", return_value=(records, {}, {label: 15}, {label: 15})):
                with patch("builtins.print"):
                    summary = run_audit(args)
            self.assertEqual(summary["class_mapping_status"], "FAIL")
            self.assertEqual(summary["status"], "NO_GO_DATA_INTEGRITY")


if __name__ == "__main__":
    unittest.main()
