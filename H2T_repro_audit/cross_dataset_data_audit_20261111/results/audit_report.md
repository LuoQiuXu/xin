# ImageNet-LT / Places-LT local data audit

- Run time: 2026-11-11T23:15:02+08:00
- Script version: 1.0.2
- Mode: read-only; no training, download, extraction, or source-data modification.

## Overall result

| Dataset | Status | Main observations |
|---|---|---|
| ImageNet-LT | **pass** | train: 115846 rows / 25 shards; validation: 20000 rows / 5 shards; test: 50000 rows / 15 shards |
| Places-LT | **pass** | train: 62500/62500 paths resolved; test: 36500/36500 paths resolved; disk images: 106300 |

## Expected protocol references

| Dataset split | Expected images | Classes | Expected class-count range |
|---|---:|---:|---:|
| ImageNet-LT train | 115,846 | 1,000 | 5–1,280 |
| ImageNet-LT test | 50,000 | 1,000 | 50/class |
| Places-LT train | 62,500 | 365 | 5–4,980 |
| Places-LT test | 36,500 | 365 | 100/class |

Expected counts are derived from the checked-in official H2T annotation lists. The Hugging Face validation split is reported independently; it is not silently treated as the official test split.

## ImageNet-LT Parquet details

- Root: D:\datasets\ImageNet-LT
- Parquet shards found: 45
- Official train distribution exact match: True
- Official test distribution exact match: True

### train

- Rows: 115846
- Shards: 25
- Columns: image, label
- Label column: label; image column: image
- Label null/unparsed: 0
- Class summary: {"num_classes_expected": 1000, "num_classes_present": 1000, "missing_class_ids": [], "min_count_including_missing": 5, "max_count": 1280, "total_rows": 115846, "many_gt_100": 385, "medium_20_to_100": 479, "few_lt_20": 136}
- Image decode sample: {"attempted": 24, "decoded": 24, "failed": 0, "skipped": 0, "failures": []}

### validation

- Rows: 20000
- Shards: 5
- Columns: image, label
- Label column: label; image column: image
- Label null/unparsed: 0
- Class summary: {"num_classes_expected": 1000, "num_classes_present": 1000, "missing_class_ids": [], "min_count_including_missing": 20, "max_count": 20, "total_rows": 20000, "many_gt_100": 0, "medium_20_to_100": 1000, "few_lt_20": 0}
- Image decode sample: {"attempted": 24, "decoded": 24, "failed": 0, "skipped": 0, "failures": []}

### test

- Rows: 50000
- Shards: 15
- Columns: image, label
- Label column: label; image column: image
- Label null/unparsed: 0
- Class summary: {"num_classes_expected": 1000, "num_classes_present": 1000, "missing_class_ids": [], "min_count_including_missing": 50, "max_count": 50, "total_rows": 50000, "many_gt_100": 0, "medium_20_to_100": 1000, "few_lt_20": 0}
- Image decode sample: {"attempted": 24, "decoded": 24, "failed": 0, "skipped": 0, "failures": []}

### ImageNet-LT notes

- Validation is reported separately; it is not substituted for the official 50,000-image test split.

## Places-LT image-folder details

- Root: D:\datasets\places-lt
- Image inventory: {"image_files": 106300, "total_bytes": 1441761320, "extension_counts": {".jpg": 106300}, "unreferenced_images": 7300}
- Archive: {"path": "D:\\datasets\\places-lt.tar.gz", "size_bytes": 1380563300}
- Manifest files: {"train": "D:\\xiangmuxunlian\\xin\\H2T\\datasets\\data_txt\\Places_LT_train.txt", "test": "D:\\xiangmuxunlian\\xin\\H2T\\datasets\\data_txt\\Places_LT_test.txt"}
- Train/test annotation path overlap: 0
- Train/test resolved path overlap: 0
- Extracted layout: {"relative_root": "places-lt", "manifest_to_extracted_split": {"train": "train", "test": "val"}, "class_folders_mapped": 365, "method": "Exact class folder and filename using the official label. Nested class components are joined with '-'."}

### train

- Manifest rows: 62500
- Unique classes: 365
- Class summary: {"num_classes_expected": 365, "num_classes_present": 365, "missing_class_ids": [], "min_count_including_missing": 5, "max_count": 4980, "total_rows": 62500, "many_gt_100": 131, "medium_20_to_100": 163, "few_lt_20": 71}
- Resolved local image paths: 62500/62500
- Exact/direct matches 0, class-folder matches 62500, fallback matches 0
- Sample image decode: {"attempted": 64, "decoded": 64, "failed": 0, "failures": []}

### test

- Manifest rows: 36500
- Unique classes: 365
- Class summary: {"num_classes_expected": 365, "num_classes_present": 365, "missing_class_ids": [], "min_count_including_missing": 100, "max_count": 100, "total_rows": 36500, "many_gt_100": 0, "medium_20_to_100": 365, "few_lt_20": 0}
- Resolved local image paths: 36500/36500
- Exact/direct matches 0, class-folder matches 36500, fallback matches 0
- Sample image decode: {"attempted": 64, "decoded": 64, "failed": 0, "failures": []}

## Limitations

- ImageNet-LT labels are scanned for every Parquet row, but image decoding is a deterministic sample from selected shards, not a full decode of every embedded image.
- Places-LT checks every manifest path for local file existence, but decodes only a deterministic sample of images.
- This audit does not compute a full pixel-level duplicate scan across splits.
- No model, CUDA context, optimizer, checkpoint, or training loop is loaded or executed.

## All issues

- No blocking issues detected by the checks in this report.

## Independent run verification

- Python 3.12.12, PyArrow 26.0.1 and Pillow 12.3.0. All 7 CPU unit tests passed.
- The first audit exited with code 1 after resolving 53,592 of 62,500 Places-LT train paths. Its reports and log are preserved locally under `initial_audit_attempt/`. The extracted folders use a hyphen for nested category components, such as `arena/hockey` to `arena-hockey`. The corrected exact class-folder lookup resolved the remaining 8,908 paths. The final audit exited with code 0.
- The 1,000 WNID-to-label assignments from the official ImageNet-LT manifests agree with the local dataset card and the embedded label metadata in all 45 Parquet shards. Each split has contiguous shard indices.
- The extracted Places-LT train directory contains 7,300 additional images outside the official lists. The extracted val directory contains exactly the 36,500 images selected by the official test manifest.
- All four official manifest SHA256 hashes, the Places-LT tree fingerprint and the Places-LT archive size and modification time matched the pre-run checks. The ImageNet-LT tree metadata fingerprint differed, with the origin of the discrepancy unresolved. All 45 Parquet shard records agreed across the two audit runs. A separate byte-level check matched all 45 Parquet SHA256 digests against the existing local download records (22,397,166,280 bytes checked).
