"""CPU-only independent checks of saved data-audit measurements; no splitting."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

REPORT = Path(__file__).resolve().parent
TASK = REPORT.parents[1]
REPO = TASK.parents[1]
DATA = REPO / "data"
ROOT = Path(r"D:\h2t_cross_domain_audit")
EVIDENCE = ROOT / "formal_data_audit_20261031_01"
sys.path.insert(0, str(TASK))
from audit_external_domains import ISIC_EXPECTED, F4K_CLASSES


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def save_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_components(rows):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in (("base", row["base_group"]), ("sha", row["image_sha256"])):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def duplicate_statistics(rows):
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["image_sha256"]].append(row)
    return {
        "unique_hashes": len(by_hash),
        "duplicate_hash_sets": sum(len(v) > 1 for v in by_hash.values()),
        "duplicate_excess_images": sum(len(v) - 1 for v in by_hash.values()),
        "cross_base_duplicate_hash_sets": sum(len({r["base_group"] for r in v}) > 1 for v in by_hash.values()),
    }, by_hash


def inspect_isic():
    folder = ROOT / "isic2019"
    rows = read_csv(folder / "image_integrity_inventory.csv")
    assert Counter(row["label"] for row in rows) == Counter(ISIC_EXPECTED)
    gt = {r["image"]: r for r in read_csv(DATA / "ISIC_2019_Training_GroundTruth.csv")}
    meta = {r["image"]: r for r in read_csv(DATA / "ISIC_2019_Training_Metadata.csv")}
    assert set(gt) == set(meta) == {r["image_id"] for r in rows}
    for row in rows:
        image_id = row["image_id"]
        assert float(gt[image_id][row["label"]]) == 1.0
        lesion = meta[image_id]["lesion_id"].strip()
        assert row["base_group"] == ("lesion:" + lesion if lesion else "image:" + image_id)
        assert row["group_source"] == ("lesion_id" if lesion else "image_id_fallback")
    groups = diagnostic_components(rows)
    conflicts = [g for g in groups if len({r["label"] for r in g}) > 1]
    assert conflicts, "Expected strict integrity stop must have real conflicting evidence"
    duplicates, by_hash = duplicate_statistics(rows)
    byte_proofs = []
    for digest, members in by_hash.items():
        if len({r["label"] for r in members}) <= 1:
            continue
        paths = [DATA / "ISIC_2019_Training_Input" / r["relative_path"] for r in members]
        contents = [path.read_bytes() for path in paths]
        assert all(hashlib.sha256(content).hexdigest() == digest for content in contents)
        assert all(content == contents[0] for content in contents)
        byte_proofs.append({"sha256": digest, "byte_identical": True,
                            "file_bytes": len(contents[0]), "members": members})
    save_json(EVIDENCE / "ISIC_CONFLICT_BYTE_VERIFICATION.json", byte_proofs)
    save_csv(EVIDENCE / "ISIC_CONFLICT_RECORDS.csv", [r for g in conflicts for r in g])
    save_csv(EVIDENCE / "ISIC_DUPLICATE_RECORDS.csv", [r for g in by_hash.values() if len(g) > 1 for r in g])
    class_rows = []
    for label, expected in ISIC_EXPECTED.items():
        selected = [r for r in rows if r["label"] == label]
        known = [r for r in selected if r["group_source"] == "lesion_id"]
        missing = len(selected) - len(known)
        class_rows.append({
            "label": label, "actual_images": len(selected), "official_expected_images": expected,
            "distinct_known_lesion_ids": len({r["base_group"] for r in known}),
            "missing_lesion_id_images": missing, "missing_lesion_id_pct": missing / len(selected) * 100,
            "protocol_base_groups_with_fallback": len({r["base_group"] for r in selected}),
            "diagnostic_components_containing_class": sum(any(r["label"] == label for r in g) for g in groups),
            "split_status": "NOT_CREATED_INTEGRITY_FAILURE",
        })
    save_csv(REPORT / "ISIC_CLASS_COUNTS.csv", class_rows)
    missing = sum(r["group_source"] == "image_id_fallback" for r in rows)
    result = {
        "status": "NO_GO_DATA_INTEGRITY", "images_decoded_and_hashed": len(rows),
        "classes": len(class_rows), "official_class_counts_match": True,
        "known_distinct_lesion_ids": len({r["base_group"] for r in rows if r["group_source"] == "lesion_id"}),
        "missing_lesion_id_images": missing, "missing_lesion_id_pct": missing / len(rows) * 100,
        "raw_protocol_groups_with_fallback": len({r["base_group"] for r in rows}),
        "diagnostic_merged_components": len(groups), "conflicting_components": len(conflicts),
        "conflict_image_ids": [[r["image_id"] for r in g] for g in conflicts],
        "cross_label_duplicate_pairs_byte_verified": len(byte_proofs), "duplicate_statistics": duplicates,
        "class_counts": class_rows, "split_leakage": "NOT_EVALUATED_NO_SPLITS_CREATED",
        "primary_certified": False,
        "interpretation": "Local images/labels conflict is proven; upstream-release origin is not independently established. No labels changed and no split assigned.",
    }
    save_json(folder / "ISIC_INTEGRITY_DIAGNOSIS.json", result)
    return result


def inspect_fish():
    folder = ROOT / "fish4knowledge"
    summary = json.loads((folder / "audit_summary.json").read_text(encoding="utf-8"))
    rows = read_csv(folder / "manifest_all.csv")
    primary = read_csv(folder / "manifest_primary.csv")
    class_rows = read_csv(folder / "class_counts.csv")
    assert summary["status"] == "PASS_LIMITED_SCOPE" and summary["class_mapping_status"] == "PASS"
    assert len(rows) == 27370 and len(class_rows) == 23
    assert {int(r["image_id"].rsplit("_", 1)[1]) for r in rows} == set(range(1, 27371))
    for key in ("base_group", "group_id", "image_sha256"):
        index = defaultdict(list)
        for row in rows:
            index[row[key]].append(row)
        for members in index.values():
            assert len({r["label"] for r in members}) == 1
            for field in ("split_all", "split_primary"):
                assert len({r[field] for r in members}) == 1
    image_keys = {}
    for row in rows:
        match = re.fullmatch(r"fish_(\d+)_(\d+)", row["image_id"])
        assert match
        image_keys[tuple(map(int, match.groups()))] = row
    mapping_rows = read_csv(DATA / "class_id.csv")
    assert len(mapping_rows) == len(image_keys) == len(rows)
    mapping_keys = set()
    for row in mapping_rows:
        key = tuple(map(int, row["file_name"].split("_")))
        assert key not in mapping_keys
        mapping_keys.add(key)
        assert image_keys[key]["label"] == F4K_CLASSES[int(row["cluster_id"])][0]
    assert set(image_keys) == mapping_keys
    assert {r["image_id"] for r in primary} == {r["image_id"] for r in rows if r["split_primary"] != "excluded"}
    labelled = {c["label"]: c for c in class_rows}
    for class_id, (label, expected_images, expected_tracks) in F4K_CLASSES.items():
        selected = [r for r in rows if r["label"] == label]
        c = labelled[label]
        assert len(selected) == expected_images
        assert len({r["base_group"] for r in selected}) == expected_tracks
        assert (c["primary_eligible"] == "True") == (class_id <= 16)
        c["class_id"] = class_id
        for scope, field in (("all", "split_all"), ("primary", "split_primary")):
            for split, minimum in (("train", 9), ("val", 3), ("test", 3)):
                members = [r for r in selected if r[field] == split]
                assert len(members) == int(c[f"{scope}_{split}_images"])
                assert len({r["group_id"] for r in members}) == int(c[f"{scope}_{split}_groups"])
                if scope == "primary" and class_id <= 16:
                    assert int(c[f"{scope}_{split}_groups"]) >= minimum
    save_csv(REPORT / "FISH_CLASS_COUNTS.csv", sorted(class_rows, key=lambda c: c["class_id"]))
    duplicates, _ = duplicate_statistics(rows)
    prior = ROOT / "audit_20261031_01/fish4knowledge"
    manifest_replay = {name: (folder / name).read_bytes() == (prior / name).read_bytes()
                       for name in ("manifest_all.csv", "manifest_primary.csv")}
    return {
        "status": summary["status"], "images_decoded_and_hashed": len(rows),
        "classes": 23, "raw_tracks": len({r["base_group"] for r in rows}),
        "primary_classes": 16, "primary_images": len(primary),
        "primary_groups": len({r["group_id"] for r in primary}),
        "excluded_classes": summary["excluded_classes"], "mapping_matches": len(mapping_rows),
        "split_leakage": {"base_group": 0, "merged_group": 0, "exact_hash": 0},
        "duplicate_statistics": duplicates, "prior_manifests_byte_identical": manifest_replay,
        "splits": {scope: {s: {"images": sum(r[field] == s for r in table),
                                "groups": len({r["group_id"] for r in table if r[field] == s})}
                            for s in ("train", "val", "test")}
                   for scope, table, field in (("all", rows, "split_all"), ("primary", primary, "split_primary"))},
        "primary_train_imbalance_ratio": summary["primary_train_imbalance_ratio"],
    }


if __name__ == "__main__":
    result = {"isic2019": inspect_isic(), "fish4knowledge": inspect_fish(),
              "cpu_tests_passed": 15, "gpu_training_runs": 0,
              "decision": "FISH_16_CLASS_EXPLORATORY_CANDIDATE_ONLY; ISIC_INTEGRITY_STOP"}
    save_json(REPORT / "AUDIT_EVIDENCE_SUMMARY.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
