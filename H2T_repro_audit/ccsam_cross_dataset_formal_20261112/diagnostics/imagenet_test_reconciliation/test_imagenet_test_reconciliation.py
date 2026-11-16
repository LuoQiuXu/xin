import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from diagnose_imagenet_test import ImageRow, filename_identity, match_metadata, parquet_metadata

WNIDS = {"n00000001": 0, "n00000002": 1}


def row(number, label, wnid, *, official=False, index=0):
    name = f"ILSVRC2012_val_{number:08d}.JPEG"
    if official:
        return ImageRow(f"val/{wnid}/{name}", name, name, label, label, wnid, None, None, index)
    path = f"other/{Path(name).stem}_{wnid}.JPEG"
    raw, normalized, suffix = filename_identity(path)
    return ImageRow(path, raw, normalized, label, label, suffix, "test.parquet", 1, index)


class MetadataReconciliationTest(unittest.TestCase):
    def test_exact_rule_does_not_remove_unverified_suffixes(self):
        name = "ILSVRC2012_val_00029930.JPEG"
        self.assertEqual(filename_identity("a\\" + name), (name, name, None))
        self.assertEqual(filename_identity("ILSVRC2012_val_00029930_n00000001.JPEG")[1:], (name, "n00000001"))
        invalid = "ILSVRC2012_val_00029930_n00000001_copy.JPEG"
        self.assertEqual(filename_identity(invalid), (invalid, invalid, None))

    def test_pairing_preserves_official_order_with_shuffled_physical_rows(self):
        official = [row(2, 0, "n00000001", official=True), row(1, 1, "n00000002", official=True, index=1)]
        parquet = [row(1, 1, "n00000002", index=4), row(2, 0, "n00000001", index=7)]
        report, matches, unresolved = match_metadata(official, parquet, WNIDS)
        self.assertEqual([b.row_index for _, b in matches], [7, 4])
        self.assertEqual(report["matches"]["removed_verified_wnid_suffix"], 2)
        self.assertEqual(report["raw_integer_minus_official_id_by_image"], {0: 2})
        self.assertEqual(unresolved, [])

    def test_rejects_label_conflict_even_when_filename_matches(self):
        official = row(1, 0, "n00000001", official=True)
        observed = row(1, 1, "n00000002")
        report, matches, unresolved = match_metadata([official], [observed], WNIDS)
        self.assertFalse(matches)
        self.assertEqual(unresolved, [official])
        self.assertEqual(report["matches"]["different_labels_after_normalization"], 1)
        self.assertEqual(report["matches"]["wnid_suffix_inconsistencies"], 1)

    def test_rejects_normalization_collision_even_if_one_label_agrees(self):
        official = row(1, 0, "n00000001", official=True)
        variants = [row(1, 0, "n00000001"), row(1, 1, "n00000002", index=1)]
        report, matches, unresolved = match_metadata([official], variants, WNIDS)
        self.assertEqual(report["matches"]["parquet_normalization_collisions"], 1)
        self.assertFalse(matches)
        self.assertEqual(unresolved, [official])

    def test_metadata_projection_and_hf_order_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "test.parquet"
            table = pa.table({
                "image": pa.array([{"bytes": b"invalid image", "path": "ILSVRC2012_val_00000001_n00000001.JPEG"}],
                                  type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                "label": pa.array([0], type=pa.int64()),
            }).replace_schema_metadata({
                b"huggingface": json.dumps({"info": {"features": {"label": {
                    "_type": "ClassLabel", "names": ["n00000002", "n00000001"],
                }}}}).encode(),
            })
            pq.write_table(table, shard)
            rows, inventory = parquet_metadata(root, [shard], WNIDS)
            self.assertEqual(rows[0].row_index, 0)
            self.assertEqual(rows[0].row_group, 0)
            self.assertEqual(rows[0].raw_label, 0)
            self.assertFalse(inventory[0]["class_label_order_matches"])
            self.assertEqual(inventory[0]["class_label_mismatch_examples"], [0, 1])


if __name__ == "__main__":
    unittest.main()
