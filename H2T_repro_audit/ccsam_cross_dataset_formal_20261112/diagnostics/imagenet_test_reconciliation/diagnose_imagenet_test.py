"""Reconcile official ImageNet-LT TEST rows with the local Parquet metadata."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import re
import sys

import pyarrow.parquet as pq

EXPERIMENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT))

from cross_dataset_data import discover_imagenet_shards, imagenet_wnid_map, parse_annotation_file
from experiment_common import REPO, json_sha256, read_json, record_run, sha256_file, write_json

TEST_MANIFEST = REPO / "H2T/datasets/data_txt/ImageNet_LT_test.txt"
TRAIN_MANIFEST = REPO / "H2T/datasets/data_txt/ImageNet_LT_train.txt"
TEST_ROWS, TEST_SHARDS, CLASS_COUNT = 50_000, 15, 1_000
PROJECTED_COLUMNS = ["label", "image.path"]
FOCUS_FILENAME = "ILSVRC2012_val_00029930.JPEG"
FILENAME = re.compile(r"(ILSVRC2012_val_[0-9]{8})(?:_(n[0-9]{8}))?\.JPEG")
WNID = re.compile(r"n[0-9]{8}")


@dataclass(frozen=True, slots=True)
class ImageRow:
    raw_path: str | None
    filename: str | None
    normalized_filename: str | None
    raw_label: int | str | None
    class_id: int | None
    wnid: str | None
    shard: str | None
    row_group: int | None
    row_index: int


def filename_identity(path: str | None) -> tuple[str | None, str | None, str | None]:
    if not path or not path.strip():
        return None, None, None
    name = PurePosixPath(path.replace("\\", "/")).name
    m = FILENAME.fullmatch(name)
    return (name, m[1] + ".JPEG", m[2]) if m else (name, name, None)


def official_rows(manifest: Path, wnids: dict[str, int]) -> list[ImageRow]:
    rows = []
    for index, (path, label) in enumerate(parse_annotation_file(manifest)):
        parsed = PurePosixPath(path)
        name, normalized, suffix = filename_identity(path)
        class_wnid = parsed.parent.name
        if (len(parsed.parts) != 3 or parsed.parts[0] != "val" or suffix is not None
            or not FILENAME.fullmatch(name or "") or wnids.get(class_wnid) != label):
            raise ValueError(f"Invalid ImageNet-LT official TEST identity at row {index}: {path}, {label}")
        rows.append(ImageRow(path, name, normalized, label, label, class_wnid, None, None, index))
    return rows


def parquet_metadata(root: Path, shards: list[Path], wnids: dict[str, int]) -> tuple[list[ImageRow], list[dict]]:
    ordered_wnids = [wnid for wnid, _ in sorted(wnids.items(), key=lambda item: item[1])]
    rows, shards_info = [], []
    for path in shards:
        relative = path.relative_to(root).as_posix()
        with pq.ParquetFile(path) as parquet:
            arrow_schema = parquet.schema_arrow
            hf = json.loads((arrow_schema.metadata or {}).get(b"huggingface", b"{}"))
            label_spec = hf.get("info", {}).get("features", {}).get("label", {})
            names = label_spec.get("names") or []
            mismatch = [i for i, wnid in enumerate(ordered_wnids) if i >= len(names) or names[i] != wnid]
            stat = path.stat()
            shards_info.append({
                "path": relative, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "rows": parquet.metadata.num_rows, "row_groups": parquet.num_row_groups,
                "label_arrow_type": str(arrow_schema.field("label").type),
                "class_label_names_count": len(names),
                "class_label_names_sha256": json_sha256(names),
                "class_label_order_matches": not mismatch and len(names) == len(ordered_wnids),
                "class_label_mismatch_examples": mismatch[:10],
            })
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=PROJECTED_COLUMNS, use_threads=False)
                if (set(table.column_names) != {"image", "label"}
                    or [field.name for field in table.schema.field("image").type] != ["path"]):
                    raise ValueError(f"{relative}: projection must expose only label and image.path")
                paths = table["image"].to_pylist()
                labels = table["label"].to_pylist()
                for row_index, (item, raw_label) in enumerate(zip(paths, labels, strict=True)):
                    raw_path = item["path"] if item else None
                    name, normalized, suffix = filename_identity(raw_path)
                    if type(raw_label) is int:
                        class_id = raw_label
                    else:
                        class_id = wnids.get(raw_label) if isinstance(raw_label, str) and WNID.fullmatch(raw_label) else None
                    rows.append(ImageRow(raw_path, name, normalized, raw_label, class_id, suffix,
                                         relative, group, row_index))
    return rows, shards_info


def names_by(rows: list[ImageRow], key) -> dict[str, list[ImageRow]]:
    result = defaultdict(list)
    for row in rows:
        name = key(row)
        if name:
            result[name].append(row)
    return result


def inventory(rows: list[ImageRow], names: dict[str, list[ImageRow]]) -> dict:
    counts = Counter(row.class_id for row in rows if row.class_id is not None)
    return {
        "rows": len(rows), "unique_filenames": len(names),
        "duplicate_filenames": sum(len(group) > 1 for group in names.values()),
        "duplicate_filename_rows": sum(len(group) - 1 for group in names.values()),
        "empty_path_rows": sum(not row.filename for row in rows),
        "classes": len(counts), "class_counts": dict(sorted(counts.items())),
        "raw_label_types": dict(Counter(type(row.raw_label).__name__ for row in rows)),
        "unresolved_labels": sum(row.class_id is None for row in rows),
    }


def match_metadata(official: list[ImageRow], parquet: list[ImageRow], wnids: dict[str, int]):
    left, right = (names_by(rows, lambda row: row.filename) for rows in (official, parquet))
    normalized_left, normalized_right = (
        names_by(rows, lambda row: row.normalized_filename) for rows in (official, parquet)
    )
    same_names = left.keys() & right.keys()
    raw_pairs = [(left[name][0], right[name][0]) for name in same_names if len(left[name]) == len(right[name]) == 1]
    cases = names_by(parquet, lambda row: row.filename.casefold() if row.filename else None)
    stems = names_by(parquet, lambda row: PurePosixPath(row.filename).stem if row.filename else None)
    only_case, only_extension = 0, 0
    for name in left.keys() - right.keys():
        if len(left[name]) != 1:
            continue
        label = left[name][0].class_id
        case = cases.get(name.casefold(), [])
        ext = stems.get(PurePosixPath(name).stem, [])
        if len(case) == 1 and case[0].class_id == label:
            only_case += 1
        elif len(ext) == 1 and ext[0].class_id == label:
            only_extension += int(PurePosixPath(ext[0].filename).suffix.casefold() != PurePosixPath(name).suffix.casefold())

    matches = []
    problems = []
    examples = []
    label_offsets = Counter()
    label_mismatches = suffix_mismatches = suffix_matches = 0
    for row in official:
        candidates = normalized_right.get(row.normalized_filename, [])
        candidate = candidates[0] if len(candidates) == 1 else None
        reason = None
        if candidate is None or len(normalized_left[row.normalized_filename]) != 1:
            reason = "missing_or_ambiguous_normalized_filename"
        else:
            if type(candidate.raw_label) is int:
                label_offsets[candidate.raw_label - row.class_id] += 1
            bad_label = candidate.class_id != row.class_id
            bad_suffix = candidate.wnid is not None and (
                candidate.wnid != row.wnid or wnids.get(candidate.wnid) != candidate.class_id
            )
            label_mismatches += bad_label
            suffix_mismatches += bad_suffix
            if bad_label or bad_suffix:
                reason = "class_id_or_wnid_mismatch"
            elif candidate.filename != row.filename and candidate.wnid is None:
                reason = "unverified_filename_rule"
            else:
                matches.append((row, candidate))
                if candidate.filename != row.filename:
                    suffix_matches += 1
        if reason:
            problems.append(row)
        if len(examples) < 100 and (reason or row.raw_path != candidate.raw_path):
            examples.append({
                "reason": reason or ("wnid_suffix" if candidate.filename != row.filename else "path_prefix"),
                "official": asdict(row),
                "parquet_candidates": [asdict(item) for item in candidates[:5]],
            })

    focus_ref = left.get(FOCUS_FILENAME, [])
    focus_labels = {row.class_id for row in focus_ref}
    focus = {
        "official": [asdict(row) for row in focus_ref],
        "exact_file_path_string": [asdict(row) for row in parquet if row.raw_path == FOCUS_FILENAME],
        "exact_basename_any_path": [asdict(row) for row in right.get(FOCUS_FILENAME, [])],
        "exact_basename_different_label": [
            asdict(row) for row in right.get(FOCUS_FILENAME, []) if row.class_id not in focus_labels
        ],
        "normalized_filename": [asdict(row) for row in normalized_right.get(FOCUS_FILENAME, [])],
    }
    return {
        "official": inventory(official, left), "parquet": inventory(parquet, right),
        "matches": {
            "exact_filename": len(same_names),
            "same_filename_different_label": sum(a.class_id != b.class_id for a, b in raw_pairs),
            "official_missing_raw_filename": len(left.keys() - right.keys()),
            "parquet_extra_raw_filename": len(right.keys() - left.keys()),
            "path_prefix_only": sum(a.class_id == b.class_id and a.raw_path != b.raw_path for a, b in raw_pairs),
            "case_only": only_case, "extension_only": only_extension,
            "normalized_filename_intersection": len(normalized_left.keys() & normalized_right.keys()),
            "official_missing_normalized_filename": len(normalized_left.keys() - normalized_right.keys()),
            "parquet_extra_normalized_filename": len(normalized_right.keys() - normalized_left.keys()),
            "official_normalization_collisions": sum(len(group) > 1 for group in normalized_left.values()),
            "parquet_normalization_collisions": sum(len(group) > 1 for group in normalized_right.values()),
            "different_labels_after_normalization": label_mismatches,
            "wnid_suffix_inconsistencies": suffix_mismatches,
            "removed_verified_wnid_suffix": suffix_matches,
            "matched_one_to_one": len(matches),
        },
        "raw_integer_minus_official_id_by_image": dict(sorted(label_offsets.items())),
        "focus": focus, "first_100_differences": examples,
    }, matches, problems


def mapping_digest(matches: list[tuple[ImageRow, ImageRow]]) -> str:
    return json_sha256([[a.row_index, a.raw_path, a.class_id, a.wnid, b.raw_path, b.raw_label,
                         b.shard, b.row_group, b.row_index] for a, b in matches])


def reconcile(root: Path, train_manifest: Path, test_manifest: Path):
    wnids = imagenet_wnid_map(parse_annotation_file(train_manifest))
    if set(wnids.values()) != set(range(CLASS_COUNT)):
        raise ValueError("Official ImageNet-LT TRAIN WNIDs do not cover IDs 0 through 999")
    shards = discover_imagenet_shards(root, "test")
    if [item.name for item in shards] != [f"test-{i:05d}-of-00015.parquet" for i in range(TEST_SHARDS)]:
        raise ValueError(f"Unexpected ImageNet-LT test shard inventory: {[item.name for item in shards]}")
    official = official_rows(test_manifest, wnids)
    parquet, shard_info = parquet_metadata(root, shards, wnids)
    report, matches, unresolved = match_metadata(official, parquet, wnids)
    errors = []
    for side, rows in (("official", official), ("parquet", parquet)):
        data = report[side]
        if data["rows"] != TEST_ROWS or data["unique_filenames"] != TEST_ROWS or data["unresolved_labels"]:
            errors.append(f"{side} must have 50,000 identified rows with unique filenames and valid labels")
        if data["class_counts"] != dict.fromkeys(range(CLASS_COUNT), 50):
            errors.append(f"{side} must contain 50 records per original class")
        if any(not FILENAME.fullmatch(row.filename or "") for row in rows):
            errors.append(f"{side} filenames do not all match the validated ImageNet-LT rule")
    if not all(item["class_label_order_matches"] for item in shard_info):
        errors.append("Hugging Face ClassLabel.names differs from the official TRAIN WNID order")
    counts = report["matches"]
    if (counts["matched_one_to_one"] != TEST_ROWS or counts["official_normalization_collisions"]
        or counts["parquet_normalization_collisions"] or counts["parquet_extra_normalized_filename"]):
        errors.append("A complete unique official filename and label correspondence could not be proved")
    if len({(row.shard, row.row_group, row.row_index) for _, row in matches}) != len(matches):
        errors.append("A physical Parquet row was matched more than once")
    report.update({
        "passed": not errors, "errors": errors, "projected_columns": PROJECTED_COLUMNS,
        "parquet_image_bytes_read": False, "images_decoded": False, "root": str(root),
        "train_manifest_sha256": sha256_file(train_manifest), "test_manifest_sha256": sha256_file(test_manifest),
        "wnid_by_official_id": [wnid for wnid, _ in sorted(wnids.items(), key=lambda item: item[1])],
        "shards": shard_info, "mapping_sha256": mapping_digest(matches),
    })
    return report, official, parquet, matches, unresolved


def write_records(path: Path, records) -> None:
    records = iter(records)
    first = next(records)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parent):
        parser.error("diagnostic output must be inside the reconciliation directory")
    protocol = read_json(EXPERIMENT / "experiment_protocol.json")
    root = Path(protocol["datasets"]["imagenet_lt"]["root"]).resolve()
    config = {"phase": "imagenet-test-metadata", "source_sha256": sha256_file(Path(__file__)), "root": str(root)}
    with record_run(output, config) as status:
        status["official_test_access"] = True
        status["projected_columns"] = PROJECTED_COLUMNS
        report, official, parquet, matches, unresolved = reconcile(root, TRAIN_MANIFEST, TEST_MANIFEST)
        differences = report.pop("first_100_differences")
        write_json(output / "reconciliation.json", report)
        write_json(output / "first_100_differences.json", differences)
        if official:
            write_records(output / "official_rows.csv", (asdict(row) for row in official))
        if parquet:
            write_records(output / "parquet_rows.csv", (asdict(row) for row in parquet))
        if matches:
            write_records(output / "official_to_parquet_indices.csv", ({
                "official_row": a.row_index, "official_path": a.raw_path, "filename": a.filename,
                "class_id": a.class_id, "wnid": a.wnid, "parquet_path": b.raw_path, "raw_parquet_label": b.raw_label,
                "shard": b.shard, "row_group": b.row_group, "row_index": b.row_index,
            } for a, b in matches))
        if not report["passed"]:
            needed = unresolved or official
            with (output / "official_test_images_needing_verification.txt").open("x", encoding="utf-8") as handle:
                handle.writelines(f"{row.raw_path} {row.class_id}\n" for row in needed)
            raise RuntimeError("; ".join(report["errors"]))
        status.update(passed=True, matches=report["matches"]["matched_one_to_one"], mapping_sha256=report["mapping_sha256"])
    print(f"Metadata-only reconciliation PASS for {status['matches']:,} images: {output}")


if __name__ == "__main__":
    main()
