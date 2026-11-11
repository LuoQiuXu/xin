import tempfile
import unittest
from collections import Counter
from pathlib import Path

from audit_cross_dataset_inputs import (
    classify_parquet_split,
    parse_annotation_file,
    resolve_image_path,
    resolve_label,
    summarize_class_counts,
)


class AuditHelpersTest(unittest.TestCase):
    def test_parse_annotation_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.txt"
            path.write_text("data_256/a/airfield/0001.jpg 0\nval_256/test.jpg 364\n", encoding="utf-8")
            parsed = parse_annotation_file(path)
            self.assertEqual(parsed["rows"], 2)
            self.assertEqual(parsed["unique_paths"], 2)
            self.assertEqual(parsed["class_counts"], Counter({0: 1, 364: 1}))
            self.assertEqual(parsed["malformed_count"], 0)

    def test_resolve_numeric_and_wnid_labels(self):
        mapping = {"n01440764": 0, "n15075141": 999}
        self.assertEqual(resolve_label(12, mapping), 12)
        self.assertEqual(resolve_label("12", mapping), 12)
        self.assertEqual(resolve_label("n01440764", mapping), 0)
        self.assertEqual(resolve_label("Class n15075141", mapping), 999)
        self.assertIsNone(resolve_label(None, mapping))

    def test_classify_parquet_split(self):
        self.assertEqual(classify_parquet_split(Path("train-00000-of-00025.parquet")), "train")
        self.assertEqual(classify_parquet_split(Path("validation-00000-of-00005.parquet")), "validation")
        self.assertEqual(classify_parquet_split(Path("test-00000-of-00015.parquet")), "test")
        self.assertIsNone(classify_parquet_split(Path("other-00000.parquet")))

    def test_class_summary_keeps_missing_classes_visible(self):
        summary = summarize_class_counts(Counter({0: 10, 1: 2}), 3)
        self.assertEqual(summary["num_classes_present"], 2)
        self.assertEqual(summary["missing_class_ids"], [2])
        self.assertEqual(summary["min_count_including_missing"], 0)

    def test_suffix_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / "nested" / "data_256" / "a" / "airfield" / "0001.jpg"
            actual.parent.mkdir(parents=True)
            actual.write_bytes(b"placeholder")
            inventory = {
                "exact": {},
                "suffix2": {"airfield/0001.jpg": [actual]},
                "suffix3": {"a/airfield/0001.jpg": [actual]},
                "basename": {"0001.jpg": [actual]},
            }
            resolved, mode = resolve_image_path("data_256/a/airfield/0001.jpg", root, inventory)
            self.assertEqual(resolved, actual)
            self.assertEqual(mode, "suffix3")


if __name__ == "__main__":
    unittest.main()
