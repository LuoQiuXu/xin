#!/usr/bin/env python
"""Group-aware data audit for ISIC 2019 and Fish4Knowledge.

This script is deliberately CPU-only. It does not train models or alter the
original H2T / PI-H2T implementations. It builds leakage-safe manifests and
reports whether each external dataset is suitable for a confirmatory screen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image
import PIL

ISIC_LABELS = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
ISIC_EXPECTED = {
    "MEL": 4522, "NV": 12875, "BCC": 3323, "AK": 867,
    "BKL": 2624, "DF": 239, "VASC": 253, "SCC": 628,
}

# Official Fish4Knowledge FRGT counts and independent trajectory counts.
F4K_CLASSES = {
    1: ("Dascyllus reticulatus", 12112, 4240),
    2: ("Plectroglyphidodon dickii", 2683, 1225),
    3: ("Chromis chrysura", 3593, 1175),
    4: ("Amphiprion clarkii", 4049, 1021),
    5: ("Chaetodon lunulatus", 2534, 536),
    6: ("Chaetodon trifascialis", 190, 79),
    7: ("Myripristis kuntee", 450, 71),
    8: ("Acanthurus nigrofuscus", 218, 71),
    9: ("Hemigymnus fasciatus", 241, 58),
    10: ("Neoniphon sammara", 299, 53),
    11: ("Abudefduf vaigiensis", 98, 42),
    12: ("Canthigaster valentini", 147, 28),
    13: ("Pomacentrus moluccensis", 181, 27),
    14: ("Zebrasoma scopas", 90, 23),
    15: ("Hemigymnus melapterus", 42, 16),
    16: ("Lutjanus fulvus", 206, 15),
    17: ("Scolopsis bilineata", 49, 8),
    18: ("Scaridae", 56, 5),
    19: ("Pempheris vanicolensis", 29, 6),
    20: ("Zanclus cornutus", 21, 6),
    21: ("Neoglyphidodon nigroris", 16, 8),
    22: ("Balistapus undulatus", 41, 6),
    23: ("Siganus fuscescens", 25, 6),
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MANIFEST_FIELDS = [
    "image_id", "relative_path", "label", "group_id", "base_group",
    "group_source", "group_size", "image_sha256", "width", "height",
    "split_all", "split_primary",
]


class AuditError(RuntimeError):
    pass


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def clean(value) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"", "nan", "none", "null", "na"} else value


def read_csv_rows(path: Path) -> Tuple[List[str], List[dict]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise AuditError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    # verify() alone does not decode JPEG pixels and can accept truncated files.
    with Image.open(path) as image:
        image.load()
    if width <= 0 or height <= 0:
        raise AuditError(f"Invalid image dimensions: {path}")
    return width, height


def reserve_output_directory(images_root: Path, output_dir: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    if output_dir.is_relative_to(repository_root):
        raise AuditError("Audit outputs must be outside the Git repository")
    if output_dir.is_relative_to(images_root):
        raise AuditError("Audit outputs must be outside the source image directory")
    output_dir.mkdir(parents=True, exist_ok=False)


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def environment_record() -> dict:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "pillow_version": PIL.__version__,
        "operating_system": platform.platform(),
        "execution_scope": "CPU data audit only; no model or torch import",
    }


def image_file_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    ambiguous = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = {part.lower() for part in path.parts}
        if any(part.startswith("mask") or part in {"masks", "segmentation"} for part in parts):
            continue
        stem = path.stem
        if stem in index:
            ambiguous[stem].append(str(path))
        else:
            index[stem] = path
    if ambiguous:
        examples = list(ambiguous.items())[:5]
        raise AuditError(f"Ambiguous image IDs (duplicate stems in image root): {examples}")
    return index


def record_image(path: Path, root: Path, image_id: str, label: str,
                 base_group: str, group_source: str) -> dict:
    width, height = inspect_image(path)
    return {
        "image_id": image_id,
        "relative_path": path.relative_to(root).as_posix(),
        "label": label,
        "base_group": base_group,
        "group_source": group_source,
        "image_sha256": sha256_file(path),
        "width": width,
        "height": height,
    }


def load_isic(images_root: Path, metadata_csv: Path, ground_truth_csv: Path):
    index = image_file_index(images_root)
    gt_headers, gt_rows = read_csv_rows(ground_truth_csv)
    meta_headers, meta_rows = read_csv_rows(metadata_csv)

    def choose(headers, candidates, description):
        for candidate in candidates:
            if candidate in headers:
                return candidate
        raise AuditError(f"Cannot find {description}; expected one of {candidates}, got {headers}")

    gt_id_col = choose(gt_headers, ("image", "isic_id", "image_id"), "ISIC image ID column")
    meta_id_col = choose(meta_headers, ("image", "isic_id", "image_id"), "ISIC metadata image ID column")
    missing_label_cols = [name for name in ISIC_LABELS if name not in gt_headers]
    if missing_label_cols:
        raise AuditError(f"ISIC ground-truth CSV missing class columns: {missing_label_cols}")
    if "lesion_id" not in meta_headers:
        raise AuditError("ISIC metadata CSV missing lesion_id column")

    records, missing_images, invalid_rows, missing_lesion_ids = [], [], [], []
    metadata = {}
    for row in meta_rows:
        image_id = clean(row.get(meta_id_col))
        if not image_id or image_id in metadata:
            invalid_rows.append(f"metadata: empty or duplicate image ID {image_id!r}")
            continue
        metadata[image_id] = row
    seen_ids = set()
    for row in gt_rows:
        image_id = clean(row.get(gt_id_col))
        if not image_id or image_id in seen_ids:
            invalid_rows.append(f"ground truth: empty or duplicate image ID {image_id!r}")
            continue
        seen_ids.add(image_id)
        try:
            values = [float(row.get(label, "")) for label in ISIC_LABELS]
            unknown = float(row.get("UNK", "0"))
        except (TypeError, ValueError):
            values, unknown = [], -1
        if (len(values) != len(ISIC_LABELS) or any(v not in {0.0, 1.0} for v in values)
                or sum(values) != 1.0 or unknown != 0.0):
            invalid_rows.append(f"{image_id}: expected exact 8-class one-hot label and UNK=0")
            continue
        active = [ISIC_LABELS[values.index(1.0)]]
        path = index.get(image_id)
        if path is None:
            missing_images.append(image_id)
            continue
        meta = metadata.get(image_id, {})
        if image_id not in metadata:
            invalid_rows.append(f"{image_id}: missing metadata row")
        lesion_id = clean(meta.get("lesion_id"))
        if lesion_id:
            base_group, group_source = f"lesion:{lesion_id}", "lesion_id"
        else:
            base_group, group_source = f"image:{image_id}", "image_id_fallback"
            missing_lesion_ids.append(image_id)
        try:
            records.append(record_image(path, images_root, image_id, active[0],
                                         base_group, group_source))
            if len(records) % 1000 == 0:
                print(f"ISIC decoded and hashed: {len(records)}/{len(gt_rows)}", file=sys.stderr, flush=True)
        except Exception as exc:
            invalid_rows.append(f"{image_id}: {type(exc).__name__}: {exc}")

    issues = {
        "missing_images": missing_images,
        "invalid_rows": invalid_rows,
        "missing_lesion_ids": missing_lesion_ids,
        "metadata_rows_without_match": sorted(set(metadata) - {clean(r.get(gt_id_col)) for r in gt_rows}),
        "unlabelled_image_ids": sorted(set(index) - seen_ids),
        "csv_statistics": {
            "ground_truth_rows": len(gt_rows), "metadata_rows": len(meta_rows),
            "ground_truth_headers": gt_headers, "metadata_headers": meta_headers,
            "missing_lesion_id_count": len(missing_lesion_ids),
            "missing_lesion_fraction_of_ground_truth": len(missing_lesion_ids) / len(gt_rows) if gt_rows else None,
        },
    }
    return records, issues, ISIC_EXPECTED, None


def infer_f4k_class_id(relative_path: Path):
    if any(part.lower().startswith("mask") for part in relative_path.parts[:-1]):
        return None
    for part in reversed(relative_path.parts[:-1]):
        match = re.match(r"^(?:fish[_ -]?)?(\d{1,2})(?:[._\s-].*)?$", part, re.IGNORECASE)
        if match:
            class_id = int(match.group(1))
            if class_id in F4K_CLASSES:
                return class_id
    return None


def load_f4k(images_root: Path):
    records, invalid_rows = [], []
    seen_fish_ids = set()
    for path in sorted(images_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(images_root)
        if any(part.lower().startswith("mask") or part.lower() in {"masks", "segmentation"}
               for part in rel.parts[:-1]):
            continue
        class_id = infer_f4k_class_id(rel)
        if class_id is None:
            invalid_rows.append(f"{rel.as_posix()}: cannot infer class folder (expected fish_01 or 01.Species)")
            continue
        if "_" not in path.stem:
            invalid_rows.append(f"{rel.as_posix()}: expected tracking_id_fish_id filename")
            continue
        track_id, fish_id = path.stem.rsplit("_", 1)
        if not track_id or not fish_id.isdigit():
            invalid_rows.append(f"{rel.as_posix()}: invalid tracking_id_fish_id filename")
            continue
        fish_id_number = int(fish_id)
        if fish_id_number < 1 or fish_id_number > sum(v[1] for v in F4K_CLASSES.values()):
            invalid_rows.append(f"{rel.as_posix()}: global fish ID out of official range {fish_id}")
            continue
        if fish_id_number in seen_fish_ids:
            invalid_rows.append(f"{rel.as_posix()}: duplicate global fish ID {fish_id}")
            continue
        seen_fish_ids.add(fish_id_number)
        label, _, _ = F4K_CLASSES[class_id]
        try:
            records.append(record_image(path, images_root, path.stem, label,
                                         f"track:{track_id}", "tracking_id"))
            if len(records) % 1000 == 0:
                print(f"Fish4Knowledge decoded and hashed: {len(records)}", file=sys.stderr, flush=True)
        except Exception as exc:
            invalid_rows.append(f"{rel.as_posix()}: {type(exc).__name__}: {exc}")

    expected = {F4K_CLASSES[i][0]: F4K_CLASSES[i][1] for i in F4K_CLASSES}
    expected_groups = {F4K_CLASSES[i][0]: F4K_CLASSES[i][2] for i in F4K_CLASSES}
    issues = {"invalid_rows": invalid_rows}
    return records, issues, expected, expected_groups


def verify_f4k_class_mapping(records: List[dict], mapping_csv: Path) -> dict:
    """Check the official filename/cluster reverse table without changing IDs.

    Only the mapping comparison removes the literal fish_ prefix and compares
    decimal IDs numerically. Split groups continue to use original track tokens.
    """
    headers, rows = read_csv_rows(mapping_csv)
    result = {
        "csv_path": str(mapping_csv), "csv_sha256": sha256_file(mapping_csv),
        "headers": headers, "first_20_data_rows": rows[:20], "n_mapping_rows": len(rows),
        "n_images": len(records), "issues": [],
        "comparison_rule": "Exact decimal tracking-ID and fish-ID equality after literal fish_ prefix removal and leading-zero normalization; original split IDs unchanged",
        "category_semantics": "Release cluster/category IDs cross-checked with class folders and official representative-category table; not a proof of biological species purity",
    }
    problems = result["issues"]
    if headers != ["file_name", "cluster_id"]:
        problems.append(f"UNVERIFIED mapping schema: {headers}")
        result["status"] = "UNVERIFIED_SCHEMA"
        return result
    images = {}
    for record in records:
        match = re.fullmatch(r"fish_(\d+)_(\d+)", record["image_id"])
        if not match:
            problems.append(f"Cannot compare original image ID: {record['image_id']}")
            continue
        key = tuple(map(int, match.groups()))
        if key in images:
            problems.append(f"Ambiguous numeric image key: {record['image_id']}")
        images[key] = record
    seen_keys, seen_fish_ids = set(), set()
    normalized, matches, examples, cluster_counts = 0, 0, [], Counter()
    for row_number, row in enumerate(rows, 2):
        raw_name = clean(row.get("file_name"))
        match = re.fullmatch(r"(\d+)_(\d+)", raw_name)
        cluster = clean(row.get("cluster_id"))
        if not match or not cluster.isdigit() or int(cluster) not in F4K_CLASSES:
            problems.append(f"row {row_number}: unsupported filename or category {row}")
            continue
        key, class_id = tuple(map(int, match.groups())), int(cluster)
        if key in seen_keys or key[1] in seen_fish_ids:
            problems.append(f"row {row_number}: duplicate mapping ID {raw_name}")
        seen_keys.add(key)
        seen_fish_ids.add(key[1])
        cluster_counts[str(class_id)] += 1
        record = images.get(key)
        if record is None:
            problems.append(f"row {row_number}: no image with matching tracking and fish IDs: {raw_name}")
            continue
        if F4K_CLASSES[class_id][0] != record["label"]:
            problems.append(f"row {row_number}: category mismatch for {record['image_id']}")
            continue
        matches += 1
        normalized += ("fish_" + raw_name != record["image_id"])
        if len(examples) < 3:
            examples.append({"csv_file_name": raw_name, "actual_image_id": record["image_id"], "cluster_id": class_id})
    unmatched_images = sorted(set(images) - seen_keys)
    if unmatched_images:
        problems.append(f"{len(unmatched_images)} images have no CSV mapping; examples {unmatched_images[:5]}")
    if len(rows) != len(records):
        problems.append(f"Mapping/image row count mismatch: {len(rows)} vs {len(records)}")
    result.update({
        "n_matched_rows": matches, "unmatched_images": len(unmatched_images),
        "matches_requiring_leading_zero_normalization": normalized,
        "cluster_counts": dict(sorted(cluster_counts.items(), key=lambda item: int(item[0]))),
        "examples": examples, "status": "PASS" if not problems else "FAIL",
    })
    return result


def merge_leakage_groups(records: List[dict]) -> int:
    """Merge same lesion/track and exact duplicate image bytes into one split group."""
    dsu = DisjointSet(len(records))
    first_by_key = {}
    duplicate_hash_links = 0
    for i, record in enumerate(records):
        for key in (f"base:{record['base_group']}", f"sha256:{record['image_sha256']}"):
            if key in first_by_key:
                dsu.union(i, first_by_key[key])
                if key.startswith("sha256:"):
                    duplicate_hash_links += 1
            else:
                first_by_key[key] = i

    components = defaultdict(list)
    for i in range(len(records)):
        components[dsu.find(i)].append(i)

    for indices in components.values():
        labels = {records[i]["label"] for i in indices}
        if len(labels) != 1:
            ids = [records[i]["image_id"] for i in indices[:8]]
            raise AuditError(f"One leakage group contains conflicting labels {labels}: {ids}")
        stable_ids = sorted(records[i]["image_id"] for i in indices)
        group_id = "grp_" + hashlib.sha256("|".join(stable_ids).encode("utf-8")).hexdigest()[:16]
        for i in indices:
            records[i]["group_id"] = group_id
            records[i]["group_size"] = len(indices)
    return duplicate_hash_links


def allocate_group_splits(records: List[dict], ratios: dict, seed: int,
                          min_train: int, min_val: int, min_test: int,
                          primary: bool):
    by_class = defaultdict(lambda: defaultdict(int))
    for record in records:
        by_class[record["label"]][record["group_id"]] += 1

    assignments, eligible, reasons = {}, [], {}
    for label in sorted(by_class):
        group_ids = list(by_class[label])
        n_groups = len(group_ids)
        if primary and n_groups < min_train + min_val + min_test:
            reasons[label] = f"only {n_groups} independent groups; need at least {min_train + min_val + min_test}"
            continue
        if not primary and n_groups < 3:
            reasons[label] = f"only {n_groups} independent groups; cannot make 3 disjoint splits"
            continue

        if primary:
            n_test = max(min_test, round(n_groups * ratios["test"]))
            n_val = max(min_val, round(n_groups * ratios["val"]))
            n_train = n_groups - n_test - n_val
            if n_train < min_train:
                reasons[label] = f"only {n_groups} groups after minimum split constraints"
                continue
            eligible.append(label)
        else:
            n_test = max(1, round(n_groups * ratios["test"]))
            n_val = max(1, round(n_groups * ratios["val"]))
            if n_test + n_val >= n_groups:
                n_test, n_val = 1, 1
            n_train = n_groups - n_test - n_val

        label_seed = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed + label_seed)
        rng.shuffle(group_ids)
        train_ids = group_ids[:n_train]
        val_ids = group_ids[n_train:n_train + n_val]
        test_ids = group_ids[n_train + n_val:]
        for group_id in train_ids:
            assignments[group_id] = "train"
        for group_id in val_ids:
            assignments[group_id] = "val"
        for group_id in test_ids:
            assignments[group_id] = "test"

    return assignments, eligible, reasons


def verify_no_split_leakage(records: List[dict], split_field: str) -> dict:
    group_splits = defaultdict(set)
    hash_splits = defaultdict(set)
    for record in records:
        split = record.get(split_field)
        if split not in {"train", "val", "test"}:
            continue
        group_splits[record["group_id"]].add(split)
        hash_splits[record["image_sha256"]].add(split)
    group_leaks = [key for key, values in group_splits.items() if len(values) > 1]
    hash_leaks = [key for key, values in hash_splits.items() if len(values) > 1]
    if group_leaks or hash_leaks:
        raise AuditError(
            f"Split leakage detected for {split_field}: "
            f"{len(group_leaks)} groups, {len(hash_leaks)} exact image hashes"
        )
    return {"group_leaks": 0, "exact_hash_leaks": 0, "status": "PASS"}


def frequency_tertiles(train_counts: dict, labels: Iterable[str]) -> dict:
    ranked = sorted(labels, key=lambda label: (-train_counts.get(label, 0), label))
    n = len(ranked)
    head_n = (n + 2) // 3
    medium_n = (n + 1) // 3
    result = {}
    for index, label in enumerate(ranked):
        if index < head_n:
            result[label] = "Head"
        elif index < head_n + medium_n:
            result[label] = "Medium"
        else:
            result[label] = "Tail"
    return result


def write_csv(path: Path, rows: List[dict], fields: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(args) -> dict:
    images_root = Path(args.images_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    reserve_output_directory(images_root, output_dir)
    args._output_reserved = True
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root does not exist: {images_root}")

    if args.dataset == "isic2019":
        if not args.metadata_csv or not args.ground_truth_csv:
            raise AuditError("ISIC2019 requires --metadata-csv and --ground-truth-csv")
        records, issues, expected, expected_groups = load_isic(
            images_root, Path(args.metadata_csv).expanduser().resolve(),
            Path(args.ground_truth_csv).expanduser().resolve())
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
        min_train, min_val, min_test = 5, 3, 5
        min_primary_classes = 8
    else:
        records, issues, expected, expected_groups = load_f4k(images_root)
        class_id_csv = getattr(args, "class_id_csv", None)
        if class_id_csv:
            mapping = verify_f4k_class_mapping(records, Path(class_id_csv).expanduser().resolve())
            issues["class_mapping_issues"] = mapping["issues"]
            write_json(output_dir / "class_mapping_audit.json", mapping)
        ratios = {"train": 0.60, "val": 0.20, "test": 0.20}
        min_train, min_val, min_test = 9, 3, 3
        min_primary_classes = 12

    if not records:
        raise AuditError("No valid images were found. Check extraction layout and paths.")

    # Preserve real measurements even if conflicting labels make grouping fail.
    integrity_fields = [field for field in MANIFEST_FIELDS if field not in {
        "group_id", "group_size", "split_all", "split_primary"}]
    write_csv(output_dir / "image_integrity_inventory.csv", records, integrity_fields)
    write_json(output_dir / "input_statistics.json", {
        "n_decoded_and_hashed_images": len(records),
        "image_counts": dict(Counter(record["label"] for record in records)),
        "group_source_counts": dict(Counter(record["group_source"] for record in records)),
        "issues": issues,
    })
    duplicate_hash_links = merge_leakage_groups(records)
    all_assignments, _, all_reasons = allocate_group_splits(
        records, ratios, args.seed, 1, 1, 1, primary=False)
    primary_assignments, primary_eligible, primary_reasons = allocate_group_splits(
        records, ratios, args.seed, min_train, min_val, min_test, primary=True)

    for record in records:
        record["split_all"] = all_assignments.get(record["group_id"], "not_split")
        record["split_primary"] = primary_assignments.get(record["group_id"], "excluded")

    all_leakage_check = verify_no_split_leakage(records, "split_all")
    primary_leakage_check = verify_no_split_leakage(records, "split_primary")

    all_counts = Counter(record["label"] for record in records)
    all_groups = defaultdict(set)
    raw_groups = defaultdict(set)
    for record in records:
        all_groups[record["label"]].add(record["group_id"])
        raw_groups[record["label"]].add(record["base_group"])

    primary_labels = set(primary_eligible)
    primary_train_counts = Counter(
        record["label"] for record in records
        if record["label"] in primary_labels and record["split_primary"] == "train")
    frequency_groups = frequency_tertiles(primary_train_counts, primary_eligible)

    class_rows = []
    for label in sorted(set(expected) | set(all_counts)):
        label_records = [r for r in records if r["label"] == label]
        primary_records = [r for r in label_records if r["split_primary"] != "excluded"]
        row = {
            "label": label,
            "all_images": all_counts[label],
            "all_independent_groups": len(all_groups[label]),
            "raw_lesion_or_tracking_ids": len(raw_groups[label]),
            "primary_eligible": label in primary_labels,
            "primary_reason_if_excluded": primary_reasons.get(label, "missing class" if label not in all_counts else ""),
            "frequency_group_primary": frequency_groups.get(label, "excluded"),
        }
        for split in ("train", "val", "test"):
            row[f"all_{split}_images"] = sum(r["split_all"] == split for r in label_records)
            row[f"all_{split}_groups"] = len({r["group_id"] for r in label_records if r["split_all"] == split})
            row[f"primary_{split}_images"] = sum(r["split_primary"] == split for r in primary_records)
            row[f"primary_{split}_groups"] = len({r["group_id"] for r in primary_records if r["split_primary"] == split})
        class_rows.append(row)

    manifest_rows = sorted(records, key=lambda r: (r["label"], r["image_id"]))
    write_csv(output_dir / "manifest_all.csv", manifest_rows, MANIFEST_FIELDS)
    primary_rows = [r for r in manifest_rows if r["split_primary"] in {"train", "val", "test"}]
    write_csv(output_dir / "manifest_primary.csv", primary_rows, MANIFEST_FIELDS)
    write_csv(output_dir / "class_counts.csv", class_rows, list(class_rows[0].keys()))

    expected_mismatches = {
        label: {"expected": count, "observed": all_counts.get(label, 0)}
        for label, count in expected.items() if all_counts.get(label, 0) != count
    }
    missing_expected_labels = sorted(set(expected) - set(all_counts))
    extra_labels = sorted(set(all_counts) - set(expected))
    expected_group_mismatches = {}
    if expected_groups is not None:
        expected_group_mismatches = {
            label: {"expected": count, "observed": len(raw_groups.get(label, set()))}
            for label, count in expected_groups.items()
            if len(raw_groups.get(label, set())) != count
        }
    release_complete = (
        not expected_mismatches and not expected_group_mismatches
        and not missing_expected_labels and not extra_labels
    )
    critical_issues = sum(len(issues.get(key, [])) for key in (
        "missing_images", "invalid_rows", "unlabelled_image_ids", "metadata_rows_without_match", "class_mapping_issues"))
    n_primary_classes = len(primary_eligible)
    primary_counts = Counter(r["label"] for r in primary_rows if r["split_primary"] == "train")
    primary_if = (
        max(primary_counts.values()) / min(primary_counts.values())
        if primary_counts and min(primary_counts.values()) > 0 else None
    )

    if critical_issues or not release_complete:
        status = "NO_GO_DATA_INTEGRITY"
    elif n_primary_classes < min_primary_classes:
        status = "NO_GO_INDEPENDENT_GROUP_SUPPORT"
    elif args.dataset == "fish4knowledge" and n_primary_classes < len(expected):
        status = "PASS_LIMITED_SCOPE"
    else:
        status = "PASS"

    summary = {
        "dataset": args.dataset,
        "status": status,
        "seed": args.seed,
        "environment": environment_record(),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "images_root": str(images_root),
        "n_valid_images": len(records),
        "n_classes_observed": len(all_counts),
        "n_classes_primary": n_primary_classes,
        "primary_classes": sorted(primary_eligible),
        "excluded_classes": [
            {"label": label, "reason": primary_reasons.get(label, "missing class" if label not in all_counts else "not eligible")}
            for label in sorted((set(expected) | set(all_counts)) - primary_labels)
        ],
        "all_class_counts": dict(sorted(all_counts.items())),
        "independent_groups_per_class": {
            label: len(all_groups[label]) for label in sorted(all_groups)
        },
        "raw_group_ids_per_class": {label: len(raw_groups[label]) for label in sorted(raw_groups)},
        "official_trajectory_comparison": "Raw tracking IDs before duplicate-byte merging; primary support uses merged groups",
        "missing_lesion_id_limit": "image_id_fallback is not a verified independent lesion; same-lesion leakage remains unverified for these images",
        "split_ratios": ratios,
        "minimum_primary_groups": {
            "train": min_train, "val": min_val, "test": min_test,
            "minimum_primary_classes": min_primary_classes,
        },
        "primary_train_imbalance_ratio": primary_if,
        "frequency_group_definition": "Head/Medium/Tail frequency tertiles computed from primary training image counts",
        "expected_release_counts_match": release_complete,
        "expected_count_mismatches": expected_mismatches,
        "expected_group_count_mismatches": expected_group_mismatches,
        "missing_expected_labels": missing_expected_labels,
        "unexpected_labels": extra_labels,
        "exact_duplicate_hash_links_merged": duplicate_hash_links,
        "issue_counts": {key: len(value) for key, value in issues.items() if isinstance(value, list)},
        "isic_csv_statistics": issues.get("csv_statistics"),
        "class_mapping_status": mapping["status"] if args.dataset == "fish4knowledge" and class_id_csv else "NOT_REQUESTED",
        "issues_examples": {
            key: value[:20] for key, value in issues.items()
            if isinstance(value, list) and value
        },
        "all_split_classes_without_three_groups": all_reasons,
        "split_leakage_checks": {"all": all_leakage_check, "primary": primary_leakage_check},
        "interpretation_limit": (
            "This is a data-readiness audit, not evidence that H2T or PI-H2T fails. "
            "Fish4Knowledge primary results, if run, apply only to the trajectory-supported class subset."
        ),
    }
    write_json(output_dir / "audit_summary.json", summary)
    write_json(output_dir / "audit_issues.json", issues)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Group-aware external-domain dataset audit")
    parser.add_argument("--dataset", required=True, choices=("isic2019", "fish4knowledge"))
    parser.add_argument("--images-root", required=True, help="Extracted image directory")
    parser.add_argument("--output-dir", required=True, help="Write audit outputs outside source data")
    parser.add_argument("--metadata-csv", help="ISIC2019 metadata CSV")
    parser.add_argument("--ground-truth-csv", help="ISIC2019 one-hot ground-truth CSV")
    parser.add_argument("--class-id-csv", help="Fish4Knowledge file_name,cluster_id reverse table; numeric comparison only, no relabelling")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        summary = run_audit(args)
    except Exception as exc:
        print(f"AUDIT_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "_output_reserved", False):
            # Persist a failed attempt without inventing counts or overwriting an old run.
            output_dir = Path(args.output_dir).expanduser().resolve()
            summary = {
                "dataset": args.dataset,
                "status": ("BLOCKED_MISSING_DATA" if isinstance(exc, FileNotFoundError)
                           else "NO_GO_DATA_INTEGRITY" if isinstance(exc, AuditError)
                           else "NO_GO_AUDIT_ERROR"),
                "error_type": type(exc).__name__, "error": str(exc),
                "images_root": str(Path(args.images_root).expanduser().resolve()),
                "environment": environment_record(), "seed": args.seed,
                "n_valid_images": None, "n_classes_observed": None,
                "n_classes_primary": None, "audit_complete": False,
                "interpretation_limit": "Incomplete data audit; no model failure inference or training eligibility",
            }
            for filename, value in (
                ("audit_summary.json", summary),
                ("audit_issues.json", {"fatal_errors": [f"{type(exc).__name__}: {exc}"]}),
            ):
                if not (output_dir / filename).exists():
                    write_json(output_dir / filename, value)
            for filename in ("manifest_all.csv", "manifest_primary.csv"):
                if not (output_dir / filename).exists():
                    write_csv(output_dir / filename, [], MANIFEST_FIELDS)
            if not (output_dir / "class_counts.csv").exists():
                write_csv(output_dir / "class_counts.csv", [], ["label", "all_images", "all_independent_groups", "primary_eligible"])
        return 2
    return 0 if summary["status"] in {"PASS", "PASS_LIMITED_SCOPE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
