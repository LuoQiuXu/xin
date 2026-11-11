#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read-only integrity audit for local ImageNet-LT (HF Parquet) and Places-LT
(image files + official TXT manifests). No training, downloads, extraction,
or source-dataset modification. Only small audit reports are written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_VERSION = "1.0.2"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXCLUDED_DIRS = {".git", ".cache", "__pycache__", ".ipynb_checkpoints"}

EXPECTED = {
    "imagenet_lt": {
        "train_rows": 115846, "test_rows": 50000, "num_classes": 1000,
        "train_min": 5, "train_max": 1280, "test_per_class": 50,
        "train_shards": 25, "validation_shards": 5, "test_shards": 15,
    },
    "places_lt": {
        "train_rows": 62500, "test_rows": 36500, "num_classes": 365,
        "train_min": 5, "train_max": 4980, "test_per_class": 100,
    },
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_relpath(value: str) -> str:
    value = str(value).strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return str(PurePosixPath(value)).lstrip("/").lower()


def find_manifest(repo_root: Path, filename: str) -> Optional[Path]:
    candidates = [
        repo_root / "H2T" / "datasets" / "data_txt" / filename,
        repo_root / "H2T_official_clean" / "H2T-main" / "datasets" / "data_txt" / filename,
        repo_root / "H2T_repro_audit" / "official" / "datasets" / "data_txt" / filename,
        repo_root / "H2T_repro_audit" / "runtime" / "datasets" / "data_txt" / filename,
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def parse_annotation_file(path: Path) -> Dict[str, Any]:
    entries: List[Tuple[str, int]] = []
    malformed: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                malformed.append({"line": line_no, "text": line[:200]})
                continue
            try:
                label = int(parts[1])
            except ValueError:
                malformed.append({"line": line_no, "text": line[:200]})
                continue
            entries.append((normalize_relpath(parts[0]), label))

    labels = [label for _, label in entries]
    paths = [rel for rel, _ in entries]
    counts = Counter(labels)
    return {
        "path": str(path.resolve()),
        "entries": entries,
        "rows": len(entries),
        "unique_paths": len(set(paths)),
        "duplicate_path_rows": len(paths) - len(set(paths)),
        "unique_labels": len(counts),
        "label_min": min(labels) if labels else None,
        "label_max": max(labels) if labels else None,
        "malformed_count": len(malformed),
        "malformed_examples": malformed[:10],
        "class_counts": counts,
        "path_set": set(paths),
    }


def summarize_class_counts(counts: Counter, num_classes: int) -> Dict[str, Any]:
    values = [int(counts.get(i, 0)) for i in range(num_classes)]
    nonzero = [v for v in values if v > 0]
    return {
        "num_classes_expected": num_classes,
        "num_classes_present": len(nonzero),
        "missing_class_ids": [i for i, value in enumerate(values) if value == 0],
        "min_count_including_missing": min(values) if values else 0,
        "max_count": max(values) if values else 0,
        "total_rows": sum(values),
        "many_gt_100": sum(1 for v in values if v > 100),
        "medium_20_to_100": sum(1 for v in values if 20 <= v <= 100),
        "few_lt_20": sum(1 for v in values if 0 < v < 20),
    }


def make_wnid_map(annotation: Dict[str, Any]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for rel, label in annotation.get("entries", []):
        for part in rel.split("/"):
            if re.fullmatch(r"n\d{8}", part):
                mapping[part] = label
                break
    return mapping


def resolve_label(value: Any, wnid_to_id: Dict[str, int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        for key in ("id", "label", "class_id", "category_id"):
            if key in value:
                result = resolve_label(value[key], wnid_to_id)
                if result is not None:
                    return result
        return None
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    if text in wnid_to_id:
        return wnid_to_id[text]
    match = re.search(r"(n\d{8})", text)
    if match and match.group(1) in wnid_to_id:
        return wnid_to_id[match.group(1)]
    match = re.match(r"^(\d+)\s+", text)
    if match:
        return int(match.group(1))
    return None


def classify_parquet_split(path: Path) -> Optional[str]:
    name = path.name.lower()
    match = re.match(r"^(train|validation|valid|val|test)(?:[-_.]|$)", name)
    if match:
        raw = match.group(1)
    else:
        raw = path.parent.name.lower()
        if raw not in {"train", "validation", "valid", "val", "test"}:
            return None
    return "validation" if raw in {"valid", "val"} else raw


def image_payload_to_bytes_or_path(value: Any) -> Tuple[Optional[bytes], Optional[str]]:
    if value is None:
        return None, None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value), None
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
        if raw is not None:
            return bytes(raw), str(path) if path else None
        return None, str(path) if path else None
    return None, None


def decode_image_bytes(payload: bytes) -> Tuple[bool, Optional[str]]:
    try:
        from PIL import Image
        with Image.open(BytesIO(payload)) as im:
            im.load()
        return True, None
    except Exception as exc:
        return False, "%s: %s" % (type(exc).__name__, str(exc)[:180])


def audit_imagenet_parquet(root: Path, official_train: Dict[str, Any],
                           official_test: Dict[str, Any], sample_per_split: int = 24) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "dataset": "ImageNet-LT", "root": str(root.resolve()), "status": "blocked",
        "expected": EXPECTED["imagenet_lt"], "splits": {}, "issues": [], "notes": [],
    }
    if not root.is_dir():
        result["issues"].append("ImageNet-LT root directory does not exist.")
        result["status"] = "fail"
        return result

    parquet_files = sorted(p for p in root.rglob("*.parquet")
                           if not any(part in EXCLUDED_DIRS for part in p.parts))
    if not parquet_files:
        result["issues"].append("No .parquet files found recursively under ImageNet-LT root.")
        result["status"] = "fail"
        return result

    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        result["issues"].append(
            "pyarrow is not importable (%s). Install it in the active conda environment: "
            "python -m pip install pyarrow" % str(exc)
        )
        result["status"] = "blocked"
        result["parquet_files_found"] = len(parquet_files)
        return result

    by_split: Dict[str, List[Path]] = defaultdict(list)
    unclassified: List[str] = []
    for path in parquet_files:
        split = classify_parquet_split(path)
        if split is None:
            unclassified.append(str(path))
        else:
            by_split[split].append(path)

    result["parquet_files_found"] = len(parquet_files)
    result["unclassified_parquet_files"] = unclassified[:30]
    if unclassified:
        result["issues"].append("%d Parquet file(s) could not be assigned to train/validation/test by filename." % len(unclassified))

    wnid_to_id = make_wnid_map(official_train)
    official_train_counts = official_train.get("class_counts", Counter())
    official_test_counts = official_test.get("class_counts", Counter())
    split_totals: Dict[str, int] = {}

    for split in ("train", "validation", "test"):
        files = sorted(by_split.get(split, []))
        split_result: Dict[str, Any] = {
            "shard_count": len(files), "files": [], "rows": 0,
            "label_column": None, "image_column": None, "schema": None,
            "class_counts": {}, "label_null_or_unparsed": 0,
            "image_decode_sample": {"attempted": 0, "decoded": 0, "failed": 0, "skipped": 0, "failures": []},
        }
        if not files:
            result["issues"].append("Missing Parquet split: %s" % split)
            result["splits"][split] = split_result
            continue

        label_col: Optional[str] = None
        image_col: Optional[str] = None
        schema_names: List[str] = []
        metadata_errors: List[str] = []
        for path in files:
            try:
                pf = pq.ParquetFile(str(path))
                names = list(pf.schema_arrow.names)
                schema_names = names
                if label_col is None:
                    label_col = next((c for c in ("label", "labels", "target", "fine_label", "class", "category_id") if c in names), None)
                if image_col is None:
                    image_col = next((c for c in ("image", "images", "img", "pixel_values") if c in names), None)
                split_result["rows"] += int(pf.metadata.num_rows)
                split_result["files"].append({
                    "name": str(path.relative_to(root)),
                    "rows": int(pf.metadata.num_rows),
                    "row_groups": int(pf.metadata.num_row_groups),
                    "size_bytes": int(path.stat().st_size),
                })
            except Exception as exc:
                metadata_errors.append("%s: %s" % (path.name, str(exc)[:180]))

        split_result["schema"] = schema_names
        split_result["label_column"] = label_col
        split_result["image_column"] = image_col
        if metadata_errors:
            result["issues"].extend(["Parquet metadata error: " + item for item in metadata_errors[:10]])
        if label_col is None:
            result["issues"].append("No recognized label column in ImageNet-LT %s split. Columns: %s" % (split, schema_names))
            result["splits"][split] = split_result
            continue
        if image_col is None:
            result["issues"].append("No recognized image column in ImageNet-LT %s split. Columns: %s" % (split, schema_names))

        # Stream only labels for every row; image payloads are not loaded in this pass.
        label_counter: Counter = Counter()
        unparsed_labels = 0
        for path in files:
            try:
                pf = pq.ParquetFile(str(path))
                for rg_index in range(pf.metadata.num_row_groups):
                    table = pf.read_row_group(rg_index, columns=[label_col], use_threads=False)
                    for value in table.column(0).to_pylist():
                        label_id = resolve_label(value, wnid_to_id)
                        if label_id is None:
                            unparsed_labels += 1
                        else:
                            label_counter[label_id] += 1
            except Exception as exc:
                result["issues"].append("Could not read labels from %s: %s" % (path.name, str(exc)[:180]))

        split_result["label_null_or_unparsed"] = unparsed_labels
        split_result["class_counts"] = {str(k): int(v) for k, v in sorted(label_counter.items())}
        split_result["class_summary"] = summarize_class_counts(label_counter, EXPECTED["imagenet_lt"]["num_classes"])
        split_totals[split] = int(split_result["rows"])
        if unparsed_labels:
            result["issues"].append("ImageNet-LT %s has %d null/unrecognized labels." % (split, unparsed_labels))
        if split_result["rows"] and sum(label_counter.values()) != split_result["rows"]:
            result["issues"].append("ImageNet-LT %s label count (%d) does not equal Parquet row count (%d)." %
                                    (split, sum(label_counter.values()), split_result["rows"]))

        # Decode a deterministic sample from first, middle and last shard only.
        if image_col is not None:
            chosen_files = [files[i] for i in sorted(set([0, len(files) // 2, len(files) - 1]))]
            remaining = max(1, sample_per_split)
            per_file = max(1, sample_per_split // max(1, len(chosen_files)))
            for path in chosen_files:
                if remaining <= 0:
                    break
                try:
                    pf = pq.ParquetFile(str(path))
                    batch = next(iter(pf.iter_batches(batch_size=max(16, per_file * 4),
                                                      columns=[image_col], use_threads=False)), None)
                    if batch is None:
                        continue
                    payloads = batch.column(0).to_pylist()
                    if not payloads:
                        continue
                    pick_count = min(per_file, remaining, len(payloads))
                    indexes = sorted(set(int(round(i * (len(payloads) - 1) / max(1, pick_count - 1)))
                                          for i in range(pick_count)))
                    for idx in indexes:
                        raw_bytes, local_path = image_payload_to_bytes_or_path(payloads[idx])
                        sample = split_result["image_decode_sample"]
                        if raw_bytes is not None:
                            sample["attempted"] += 1
                            ok, error = decode_image_bytes(raw_bytes)
                            if ok:
                                sample["decoded"] += 1
                            else:
                                sample["failed"] += 1
                                sample["failures"].append({"file": path.name, "row_in_batch": idx, "error": error})
                            remaining -= 1
                        elif local_path:
                            candidate = Path(local_path)
                            if not candidate.is_absolute():
                                candidate = root / candidate
                            if candidate.is_file():
                                sample["attempted"] += 1
                                try:
                                    from PIL import Image
                                    with Image.open(candidate) as im:
                                        im.load()
                                    sample["decoded"] += 1
                                except Exception as exc:
                                    sample["failed"] += 1
                                    sample["failures"].append({"file": path.name, "path": local_path, "error": str(exc)[:180]})
                                remaining -= 1
                            else:
                                sample["skipped"] += 1
                        else:
                            sample["skipped"] += 1
                except Exception as exc:
                    split_result["image_decode_sample"]["failed"] += 1
                    split_result["image_decode_sample"]["failures"].append(
                        {"file": path.name, "error": "sample read failed: %s" % str(exc)[:180]})
            sample = split_result["image_decode_sample"]
            if sample["failed"] or sample["skipped"] or not sample["decoded"]:
                result["issues"].append(
                    "ImageNet-LT %s sample decode: %d decoded, %d failures, %d skipped."
                    % (split, sample["decoded"], sample["failed"], sample["skipped"])
                )

        result["splits"][split] = split_result

    expected = EXPECTED["imagenet_lt"]
    train = result["splits"].get("train", {})
    validation = result["splits"].get("validation", {})
    test = result["splits"].get("test", {})
    if train.get("rows") != expected["train_rows"]:
        result["issues"].append("ImageNet-LT train rows expected %d, observed %s." % (expected["train_rows"], train.get("rows")))
    if test.get("rows") != expected["test_rows"]:
        result["issues"].append("ImageNet-LT test rows expected %d, observed %s." % (expected["test_rows"], test.get("rows")))
    for split, expected_shards in (("train", expected["train_shards"]),
                                   ("validation", expected["validation_shards"]),
                                   ("test", expected["test_shards"])):
        observed = result["splits"].get(split, {}).get("shard_count")
        if observed != expected_shards:
            result["notes"].append("Download-plan shard count for %s: expected %d, observed %s." % (split, expected_shards, observed))

    train_summary = train.get("class_summary", {})
    if train_summary:
        if train_summary.get("num_classes_present") != expected["num_classes"]:
            result["issues"].append("ImageNet-LT train should contain 1000 classes; observed %s." % train_summary.get("num_classes_present"))
        if train_summary.get("min_count_including_missing") != expected["train_min"]:
            result["issues"].append("ImageNet-LT train minimum class count expected 5; observed %s." % train_summary.get("min_count_including_missing"))
        if train_summary.get("max_count") != expected["train_max"]:
            result["issues"].append("ImageNet-LT train maximum class count expected 1280; observed %s." % train_summary.get("max_count"))
        observed = Counter({int(k): int(v) for k, v in train.get("class_counts", {}).items()})
        if observed:
            result["official_train_distribution_match"] = (observed == official_train_counts)
            if observed != official_train_counts:
                result["issues"].append("ImageNet-LT Parquet train class counts do not exactly match checked-in official ImageNet_LT_train.txt counts.")

    test_summary = test.get("class_summary", {})
    if test_summary:
        if test_summary.get("num_classes_present") != expected["num_classes"]:
            result["issues"].append("ImageNet-LT test should contain 1000 classes; observed %s." % test_summary.get("num_classes_present"))
        if any(int(test.get("class_counts", {}).get(str(i), 0)) != expected["test_per_class"]
               for i in range(expected["num_classes"])):
            result["issues"].append("ImageNet-LT test should contain exactly 50 images per class (50,000 total).")
        observed_test = Counter({int(k): int(v) for k, v in test.get("class_counts", {}).items()})
        result["official_test_distribution_match"] = (observed_test == official_test_counts) if observed_test else False

    if validation.get("rows", 0):
        result["notes"].append("Validation is reported separately; it is not substituted for the official 50,000-image test split.")
        if validation.get("class_summary", {}).get("num_classes_present") not in (None, 1000):
            result["issues"].append("ImageNet-LT validation split does not cover all 1000 classes.")

    result["split_row_total"] = sum(split_totals.values())
    result["status"] = "fail" if result["issues"] else "pass"
    return result


def build_image_inventory(root: Path) -> Dict[str, Any]:
    exact: Dict[str, Path] = {}
    suffix2: Dict[str, List[Path]] = defaultdict(list)
    suffix3: Dict[str, List[Path]] = defaultdict(list)
    basename: Dict[str, List[Path]] = defaultdict(list)
    total_images = 0
    total_bytes = 0
    ext_counts = Counter()
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for filename in files:
            path = Path(current) / filename
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                rel = normalize_relpath(str(path.relative_to(root)))
                size = int(path.stat().st_size)
            except OSError:
                continue
            total_images += 1
            total_bytes += size
            ext_counts[path.suffix.lower()] += 1
            exact[rel] = path
            parts = rel.split("/")
            basename[parts[-1]].append(path)
            if len(parts) >= 2:
                suffix2["/".join(parts[-2:])].append(path)
            if len(parts) >= 3:
                suffix3["/".join(parts[-3:])].append(path)
    return {"exact": exact, "suffix2": suffix2, "suffix3": suffix3,
            "basename": basename, "total_images": total_images,
            "total_bytes": total_bytes, "extension_counts": dict(ext_counts)}


def resolve_image_path(rel: str, root: Path, inventory: Dict[str, Any]) -> Tuple[Optional[Path], str]:
    normalized = normalize_relpath(rel)
    exact = inventory["exact"].get(normalized)
    if exact is not None:
        return exact, "exact"
    direct = root / Path(normalized)
    if direct.is_file():
        return direct, "direct"
    parts = normalized.split("/")
    if len(parts) >= 3:
        matches = inventory["suffix3"].get("/".join(parts[-3:]), [])
        if len(matches) == 1:
            return matches[0], "suffix3"
        if len(matches) > 1:
            return None, "ambiguous_suffix3"
    if len(parts) >= 2:
        matches = inventory["suffix2"].get("/".join(parts[-2:]), [])
        if len(matches) == 1:
            return matches[0], "suffix2"
        if len(matches) > 1:
            return None, "ambiguous_suffix2"
    matches = inventory["basename"].get(parts[-1], [])
    if len(matches) == 1:
        return matches[0], "basename"
    if len(matches) > 1:
        return None, "ambiguous_basename"
    return None, "missing"


def places_class_folders(entries: List[Tuple[str, int]]) -> Dict[int, str]:
    folders = {}
    for rel, label in entries:
        parts = PurePosixPath(rel).parts
        if len(parts) < 4 or parts[0] != "data_256":
            raise ValueError(f"Unexpected official TRAIN path {rel!r}")
        folder = "-".join(parts[2:-1])
        if folders.setdefault(label, folder) != folder:
            raise ValueError(f"Label {label} maps to multiple class folders")
    if len(set(folders.values())) != len(folders):
        raise ValueError("Multiple labels map to the same class folder")
    return folders


def audit_places(root: Path, repo_root: Path, sample_per_split: int = 64) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "dataset": "Places-LT", "root": str(root.resolve()), "status": "blocked",
        "expected": EXPECTED["places_lt"], "splits": {}, "issues": [], "notes": [],
    }
    if not root.is_dir():
        result["issues"].append("Places-LT image directory does not exist.")
        result["status"] = "fail"
        return result

    train_path = find_manifest(repo_root, "Places_LT_train.txt")
    test_path = find_manifest(repo_root, "Places_LT_test.txt")
    if train_path is None or test_path is None:
        result["issues"].append("Could not locate both repository annotation files Places_LT_train.txt and Places_LT_test.txt.")
        result["status"] = "blocked"
        return result

    train = parse_annotation_file(train_path)
    test = parse_annotation_file(test_path)
    result["manifest_files"] = {"train": str(train_path.resolve()), "test": str(test_path.resolve())}
    result["manifest_summary"] = {
        "train": {k: v for k, v in train.items() if k not in {"entries", "class_counts", "path_set"}},
        "test": {k: v for k, v in test.items() if k not in {"entries", "class_counts", "path_set"}},
    }

    expected = EXPECTED["places_lt"]
    for split_name, data, expected_rows in (("train", train, expected["train_rows"]),
                                             ("test", test, expected["test_rows"])):
        summary = summarize_class_counts(data["class_counts"], expected["num_classes"])
        if data["rows"] != expected_rows:
            result["issues"].append("Places-LT %s manifest rows expected %d, observed %d." % (split_name, expected_rows, data["rows"]))
        if data["unique_labels"] != expected["num_classes"]:
            result["issues"].append("Places-LT %s manifest should contain 365 classes; observed %d." % (split_name, data["unique_labels"]))
        if data["malformed_count"]:
            result["issues"].append("Places-LT %s manifest has %d malformed lines." % (split_name, data["malformed_count"]))
        if data["duplicate_path_rows"]:
            result["issues"].append("Places-LT %s manifest has %d duplicate image paths." % (split_name, data["duplicate_path_rows"]))
        if summary["num_classes_present"] != expected["num_classes"]:
            result["issues"].append("Places-LT %s manifest has missing class IDs." % split_name)
        if split_name == "train":
            if summary["min_count_including_missing"] != expected["train_min"] or summary["max_count"] != expected["train_max"]:
                result["issues"].append("Places-LT train class-count range expected 5..4980; observed %s..%s." %
                                        (summary["min_count_including_missing"], summary["max_count"]))
        else:
            if any(int(data["class_counts"].get(i, 0)) != expected["test_per_class"] for i in range(expected["num_classes"])):
                result["issues"].append("Places-LT test should contain exactly 100 images per class.")
        result["splits"][split_name] = {
            "manifest_rows": data["rows"], "unique_paths": data["unique_paths"],
            "unique_classes": data["unique_labels"], "class_summary": summary,
            "class_counts": {str(k): int(v) for k, v in sorted(data["class_counts"].items())},
            "manifest_path": data["path"],
        }

    overlap = train["path_set"] & test["path_set"]
    result["train_test_manifest_path_overlap"] = len(overlap)
    if overlap:
        result["issues"].append("Places-LT train/test manifests share %d image paths." % len(overlap))

    archive = root.parent / "places-lt.tar.gz"
    if archive.is_file():
        result["archive"] = {"path": str(archive.resolve()), "size_bytes": int(archive.stat().st_size)}
    else:
        result["notes"].append("places-lt.tar.gz was not found next to the extracted image directory; this does not invalidate extracted files.")

    inventory = build_image_inventory(root)
    result["image_inventory"] = {
        "image_files": inventory["total_images"], "total_bytes": inventory["total_bytes"],
        "extension_counts": inventory["extension_counts"],
    }

    layout_roots = [candidate for candidate in (root, root / "places-lt")
                    if (candidate / "train").is_dir() and (candidate / "val").is_dir()]
    layout_root = layout_roots[0] if len(layout_roots) == 1 else None
    class_folders = {}
    if len(layout_roots) > 1:
        result["issues"].append("Found multiple extracted Places-LT train/val roots.")
    if layout_root is not None:
        try:
            class_folders = places_class_folders(train["entries"])
        except ValueError as exc:
            result["issues"].append("Invalid Places-LT class folder mapping: %s" % exc)
        result["extracted_layout"] = {
            "relative_root": layout_root.relative_to(root).as_posix(),
            "manifest_to_extracted_split": {"train": "train", "test": "val"},
            "class_folders_mapped": len(class_folders),
            "method": "Exact class folder and filename using the official label. Nested class components are joined with '-'.",
        }

    resolved_sets = {}
    for split_name, data in (("train", train), ("test", test)):
        resolved = 0
        exact = 0
        fallback = 0
        layout_mapped = 0
        missing_examples = []
        ambiguous_examples = []
        resolved_paths: List[Path] = []
        for rel, label in data["entries"]:
            if layout_root is None:
                path, mode = resolve_image_path(rel, root, inventory)
            else:
                folder = class_folders.get(label)
                path, mode = None, "missing"
                parts = PurePosixPath(rel).parts
                valid_path = (len(parts) >= 4 and parts[0] == "data_256" if split_name == "train"
                              else len(parts) == 2 and parts[0] == "val_256")
                if folder and valid_path:
                    split_folder = "train" if split_name == "train" else "val"
                    expected_path = layout_root / split_folder / folder / parts[-1]
                    path = inventory["exact"].get(normalize_relpath(expected_path.relative_to(root).as_posix()))
                    if path is not None:
                        mode = "class_folder"
            if path is not None:
                resolved += 1
                resolved_paths.append(path)
                if mode in ("exact", "direct"):
                    exact += 1
                elif mode == "class_folder":
                    layout_mapped += 1
                else:
                    fallback += 1
            elif mode.startswith("ambiguous"):
                if len(ambiguous_examples) < 10:
                    ambiguous_examples.append({"annotation": rel, "resolution": mode})
            elif len(missing_examples) < 10:
                missing_examples.append(rel)

        split_result = result["splits"][split_name]
        resolved_sets[split_name] = set(resolved_paths)
        split_result["image_paths_resolved"] = resolved
        split_result["image_paths_exact_or_direct"] = exact
        split_result["image_paths_layout_mapped"] = layout_mapped
        split_result["image_paths_fallback"] = fallback
        split_result["image_paths_missing_or_ambiguous"] = data["rows"] - resolved
        split_result["missing_examples"] = missing_examples
        split_result["ambiguous_examples"] = ambiguous_examples
        if resolved != data["rows"]:
            result["issues"].append("Places-LT %s: only %d/%d annotation paths resolve to local image files." %
                                    (split_name, resolved, data["rows"]))
        if len(resolved_sets[split_name]) != resolved:
            result["issues"].append("Places-LT %s resolves multiple annotation rows to the same file." % split_name)

        sample = {"attempted": 0, "decoded": 0, "failed": 0, "failures": []}
        if resolved_paths:
            if len(resolved_paths) <= sample_per_split:
                sample_paths = resolved_paths
            else:
                indexes = sorted(set(int(round(i * (len(resolved_paths) - 1) / max(1, sample_per_split - 1)))
                                     for i in range(sample_per_split)))
                sample_paths = [resolved_paths[i] for i in indexes]
            for path in sample_paths:
                sample["attempted"] += 1
                try:
                    from PIL import Image
                    with Image.open(path) as im:
                        im.load()
                    sample["decoded"] += 1
                except Exception as exc:
                    sample["failed"] += 1
                    if len(sample["failures"]) < 10:
                        sample["failures"].append({"path": str(path), "error": str(exc)[:180]})
        split_result["image_decode_sample"] = sample
        if sample["failed"]:
            result["issues"].append("Places-LT %s sample contains %d undecodable image(s)." % (split_name, sample["failed"]))

    resolved_overlap = len(resolved_sets["train"] & resolved_sets["test"])
    result["train_test_resolved_path_overlap"] = resolved_overlap
    if resolved_overlap:
        result["issues"].append("Places-LT train/test annotations resolve to %d shared local files." % resolved_overlap)
    result["image_inventory"]["unreferenced_images"] = len(
        set(inventory["exact"].values()) - resolved_sets["train"] - resolved_sets["test"]
    )
    result["status"] = "fail" if result["issues"] else "pass"
    return result


def write_imagenet_class_csv(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "split", "class_id", "count"])
        writer.writeheader()
        for split in ("train", "validation", "test"):
            counts = result.get("splits", {}).get(split, {}).get("class_counts", {})
            for class_id in range(EXPECTED["imagenet_lt"]["num_classes"]):
                writer.writerow({"dataset": "ImageNet-LT", "split": split,
                                 "class_id": class_id, "count": int(counts.get(str(class_id), 0))})


def write_places_class_csv(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "split", "class_id", "count"])
        writer.writeheader()
        for split in ("train", "test"):
            counts = result.get("splits", {}).get(split, {}).get("class_counts", {})
            for class_id in range(EXPECTED["places_lt"]["num_classes"]):
                writer.writerow({"dataset": "Places-LT", "split": split,
                                 "class_id": class_id, "count": int(counts.get(str(class_id), 0))})


def markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# ImageNet-LT / Places-LT local data audit", "",
        "- Run time: %s" % report["run_time"],
        "- Script version: %s" % report["script_version"],
        "- Mode: read-only; no training, download, extraction, or source-data modification.", "",
        "## Overall result", "",
        "| Dataset | Status | Main observations |",
        "|---|---|---|",
    ]
    for key in ("imagenet_lt", "places_lt"):
        item = report.get(key, {})
        summary = []
        if key == "imagenet_lt":
            for split in ("train", "validation", "test"):
                s = item.get("splits", {}).get(split, {})
                if s:
                    summary.append("%s: %s rows / %s shards" % (split, s.get("rows"), s.get("shard_count")))
        else:
            for split in ("train", "test"):
                s = item.get("splits", {}).get(split, {})
                if s:
                    summary.append("%s: %s/%s paths resolved" % (split, s.get("image_paths_resolved"), s.get("manifest_rows")))
            inv = item.get("image_inventory", {})
            if inv:
                summary.append("disk images: %s" % inv.get("image_files"))
        lines.append("| %s | **%s** | %s |" %
                     (item.get("dataset", key), item.get("status", "unknown"), "; ".join(summary) or "see details"))

    lines.extend([
        "", "## Expected protocol references", "",
        "| Dataset split | Expected images | Classes | Expected class-count range |",
        "|---|---:|---:|---:|",
        "| ImageNet-LT train | 115,846 | 1,000 | 5–1,280 |",
        "| ImageNet-LT test | 50,000 | 1,000 | 50/class |",
        "| Places-LT train | 62,500 | 365 | 5–4,980 |",
        "| Places-LT test | 36,500 | 365 | 100/class |", "",
        "Expected counts are derived from the checked-in official H2T annotation lists. The Hugging Face validation split is reported independently; it is not silently treated as the official test split.",
        "", "## ImageNet-LT Parquet details", "",
    ])
    img = report.get("imagenet_lt", {})
    lines.append("- Root: %s" % img.get("root"))
    lines.append("- Parquet shards found: %s" % img.get("parquet_files_found"))
    lines.append("- Official train distribution exact match: %s" % img.get("official_train_distribution_match", "not determined"))
    lines.append("- Official test distribution exact match: %s" % img.get("official_test_distribution_match", "not determined"))
    for split in ("train", "validation", "test"):
        s = img.get("splits", {}).get(split, {})
        if not s:
            continue
        lines.extend([
            "", "### %s" % split, "",
            "- Rows: %s" % s.get("rows"),
            "- Shards: %s" % s.get("shard_count"),
            "- Columns: %s" % ", ".join(s.get("schema") or []),
            "- Label column: %s; image column: %s" % (s.get("label_column"), s.get("image_column")),
            "- Label null/unparsed: %s" % s.get("label_null_or_unparsed"),
            "- Class summary: %s" % json.dumps(s.get("class_summary", {}), ensure_ascii=False),
            "- Image decode sample: %s" % json.dumps(s.get("image_decode_sample", {}), ensure_ascii=False),
        ])
    if img.get("issues"):
        lines.extend(["", "### ImageNet-LT issues", ""])
        lines.extend(["- " + x for x in img["issues"]])
    if img.get("notes"):
        lines.extend(["", "### ImageNet-LT notes", ""])
        lines.extend(["- " + x for x in img["notes"]])

    places = report.get("places_lt", {})
    lines.extend([
        "", "## Places-LT image-folder details", "",
        "- Root: %s" % places.get("root"),
        "- Image inventory: %s" % json.dumps(places.get("image_inventory", {}), ensure_ascii=False),
        "- Archive: %s" % json.dumps(places.get("archive", {}), ensure_ascii=False),
        "- Manifest files: %s" % json.dumps(places.get("manifest_files", {}), ensure_ascii=False),
        "- Train/test annotation path overlap: %s" % places.get("train_test_manifest_path_overlap"),
        "- Train/test resolved path overlap: %s" % places.get("train_test_resolved_path_overlap"),
    ])
    if places.get("extracted_layout"):
        lines.append("- Extracted layout: %s" % json.dumps(places["extracted_layout"], ensure_ascii=False))
    for split in ("train", "test"):
        s = places.get("splits", {}).get(split, {})
        if s:
            lines.extend([
                "", "### %s" % split, "",
                "- Manifest rows: %s" % s.get("manifest_rows"),
                "- Unique classes: %s" % s.get("unique_classes"),
                "- Class summary: %s" % json.dumps(s.get("class_summary", {}), ensure_ascii=False),
                "- Resolved local image paths: %s/%s" % (s.get("image_paths_resolved"), s.get("manifest_rows")),
                "- Exact/direct matches %s, class-folder matches %s, fallback matches %s" %
                (s.get("image_paths_exact_or_direct"), s.get("image_paths_layout_mapped"), s.get("image_paths_fallback")),
                "- Sample image decode: %s" % json.dumps(s.get("image_decode_sample", {}), ensure_ascii=False),
            ])
            if s.get("missing_examples"):
                lines.append("- Missing examples: %s" % json.dumps(s["missing_examples"], ensure_ascii=False))
            if s.get("ambiguous_examples"):
                lines.append("- Ambiguous examples: %s" % json.dumps(s["ambiguous_examples"], ensure_ascii=False))
    if places.get("issues"):
        lines.extend(["", "### Places-LT issues", ""])
        lines.extend(["- " + x for x in places["issues"]])
    if places.get("notes"):
        lines.extend(["", "### Places-LT notes", ""])
        lines.extend(["- " + x for x in places["notes"]])

    lines.extend([
        "", "## Limitations", "",
        "- ImageNet-LT labels are scanned for every Parquet row, but image decoding is a deterministic sample from selected shards, not a full decode of every embedded image.",
        "- Places-LT checks every manifest path for local file existence, but decodes only a deterministic sample of images.",
        "- This audit does not compute a full pixel-level duplicate scan across splits.",
        "- No model, CUDA context, optimizer, checkpoint, or training loop is loaded or executed.",
        "", "## All issues", "",
    ])
    all_issues = []
    for key in ("imagenet_lt", "places_lt"):
        for issue in report.get(key, {}).get("issues", []):
            all_issues.append("[%s] %s" % (key, issue))
    lines.extend(["- " + x for x in all_issues] if all_issues else ["- No blocking issues detected by the checks in this report."])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Read-only ImageNet-LT and Places-LT data audit.")
    parser.add_argument("--imagenet-root", type=Path, default=Path(r"D:\datasets\ImageNet-LT"))
    parser.add_argument("--places-root", type=Path, default=Path(r"D:\datasets\places-lt"))
    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--imagenet-image-samples", type=int, default=24)
    parser.add_argument("--places-image-samples", type=int, default=64)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    report: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION, "run_time": now_iso(),
        "repo_root": str(repo_root), "python": sys.version,
        "platform": sys.platform, "read_only": True,
    }
    imagenet_train_path = find_manifest(repo_root, "ImageNet_LT_train.txt")
    imagenet_test_path = find_manifest(repo_root, "ImageNet_LT_test.txt")
    official_train: Dict[str, Any] = {"entries": [], "class_counts": Counter()}
    official_test: Dict[str, Any] = {"entries": [], "class_counts": Counter()}
    if imagenet_train_path and imagenet_test_path:
        official_train = parse_annotation_file(imagenet_train_path)
        official_test = parse_annotation_file(imagenet_test_path)
        report["imagenet_official_manifests"] = {
            "train": {k: v for k, v in official_train.items() if k not in {"entries", "class_counts", "path_set"}},
            "test": {k: v for k, v in official_test.items() if k not in {"entries", "class_counts", "path_set"}},
        }
    else:
        report["imagenet_official_manifests"] = {"error": "Official ImageNet-LT TXT manifests not found in repository."}

    report["imagenet_lt"] = audit_imagenet_parquet(
        args.imagenet_root.resolve(), official_train, official_test,
        sample_per_split=max(1, args.imagenet_image_samples))
    if not imagenet_train_path or not imagenet_test_path:
        report["imagenet_lt"]["issues"].append("Official ImageNet-LT train/test TXT manifests were not found; distribution comparison unavailable.")
        if report["imagenet_lt"]["status"] == "pass":
            report["imagenet_lt"]["status"] = "fail"

    report["places_lt"] = audit_places(args.places_root.resolve(), repo_root,
                                       sample_per_split=max(1, args.places_image_samples))
    report["duration_seconds"] = round(time.time() - started, 2)
    report["overall_status"] = (
        "pass" if report["imagenet_lt"].get("status") == "pass" and report["places_lt"].get("status") == "pass"
        else "blocked" if "blocked" in (report["imagenet_lt"].get("status"), report["places_lt"].get("status"))
        else "fail"
    )

    json_path = output_dir / "audit_report.json"
    md_path = output_dir / "audit_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda x: dict(x)), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_imagenet_class_csv(output_dir / "imagenet_lt_class_counts.csv", report["imagenet_lt"])
    write_places_class_csv(output_dir / "places_lt_class_counts.csv", report["places_lt"])

    artifacts = [json_path, md_path, output_dir / "imagenet_lt_class_counts.csv",
                 output_dir / "places_lt_class_counts.csv"]
    with (output_dir / "ARTIFACT_SHA256.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "size_bytes", "sha256"])
        writer.writeheader()
        for path in artifacts:
            writer.writerow({"file": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})

    print("Overall status:", report["overall_status"])
    print("ImageNet-LT:", report["imagenet_lt"].get("status"))
    print("Places-LT:", report["places_lt"].get("status"))
    print("Report:", md_path)
    print("JSON:", json_path)
    print("Duration seconds:", report["duration_seconds"])
    if report["overall_status"] != "pass":
        print("\nIssues:")
        for key in ("imagenet_lt", "places_lt"):
            for issue in report.get(key, {}).get("issues", []):
                print(" - [%s] %s" % (key, issue))
        return 2 if report["overall_status"] == "blocked" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
