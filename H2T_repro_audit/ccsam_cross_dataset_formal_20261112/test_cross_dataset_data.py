from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cross_dataset_data import (
    ImageNetLTParquetDataset,
    PlacesLTManifestDataset,
    places_class_folders,
)


def write_rgb(path: Path, color=(80, 120, 160)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (8, 8), color)
    image.save(path, format="JPEG")


class PlacesManifestDatasetTests(unittest.TestCase):
    def test_official_train_and_test_mapping_uses_only_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "places-lt"
            train_manifest = Path(tmp) / "Places_LT_train.txt"
            test_manifest = Path(tmp) / "Places_LT_test.txt"
            train_lines = []
            test_lines = []
            for class_id in range(365):
                if class_id == 0:
                    original_folder = "data_256/train/arena/hockey"
                    disk_folder = "arena-hockey"
                else:
                    original_folder = "data_256/train/class_%03d" % class_id
                    disk_folder = "class_%03d" % class_id
                train_name = "train_%03d.jpg" % class_id
                test_name = "test_%03d.jpg" % class_id
                train_lines.append("%s/%s %d" % (original_folder, train_name, class_id))
                test_lines.append("val_256/%s %d" % (test_name, class_id))
                write_rgb(root / "train" / disk_folder / train_name)
                write_rgb(root / "val" / disk_folder / test_name)

            # An extra image must never enter the manifest-backed dataset.
            write_rgb(root / "train" / "unreferenced" / "extra.jpg")
            train_manifest.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
            test_manifest.write_text("\n".join(test_lines) + "\n", encoding="utf-8")

            train_ds = PlacesLTManifestDataset(
                root, train_manifest, split="train", expected_rows=365
            )
            test_ds = PlacesLTManifestDataset(
                root, test_manifest, split="test",
                class_folders=train_ds.class_folders, expected_rows=365
            )
            self.assertEqual(len(train_ds), 365)
            self.assertEqual(len(test_ds), 365)
            self.assertEqual(train_ds.class_folders[0], "arena-hockey")
            self.assertEqual(int(train_ds.class_counts.sum()), 365)
            self.assertEqual(int(test_ds.class_counts.sum()), 365)
            self.assertEqual(train_ds[0][1], 0)
            self.assertEqual(test_ds[0][1], 0)
            self.assertNotIn(root / "train" / "unreferenced" / "extra.jpg", train_ds.paths)

    def test_duplicate_label_folder_mapping_is_rejected(self):
        entries = [
            ("data_256/train/class_a/a.jpg", 0),
            ("data_256/train/class_b/b.jpg", 0),
        ]
        with self.assertRaises(ValueError):
            places_class_folders(entries)


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is not installed")
class ImageNetParquetDatasetTests(unittest.TestCase):
    def test_row_group_random_access_and_rgb_decode(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
            rows = []
            for index, label in enumerate([0, 1, 0]):
                image_path = root / ("unused_%d.jpg" % index)
                write_rgb(image_path, color=(index * 30, 50, 90))
                from io import BytesIO
                buffer = BytesIO()
                with Image.open(image_path) as image:
                    image.save(buffer, format="JPEG")
                rows.append({"bytes": buffer.getvalue(), "path": None})
            table = pa.table({
                "image": pa.array(rows, type=image_struct),
                "label": pa.array([0, 1, 0], type=pa.int64()),
            })
            pq.write_table(table, root / "train-00000-of-00001.parquet", row_group_size=2)

            dataset = ImageNetLTParquetDataset(
                root, split="train", expected_rows=3, expected_shards=1,
                num_classes=2, row_group_cache_size=1,
            )
            self.assertEqual(len(dataset), 3)
            np.testing.assert_array_equal(dataset.class_counts, np.array([2, 1]))
            image_2, label_2 = dataset[2]
            image_0, label_0 = dataset[0]
            self.assertEqual(label_2, 0)
            self.assertEqual(label_0, 0)
            self.assertEqual(image_2.mode, "RGB")
            self.assertEqual(image_0.mode, "RGB")


if __name__ == "__main__":
    unittest.main(verbosity=2)
