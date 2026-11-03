"""Read-only Fish16 data and approved-adapter preflight.

TRAIN and VAL image bytes are verified against the locked manifest.
TEST rows are checked only as manifest metadata: no TEST path is resolved,
no TEST file is stat'ed/opened, and no TEST image bytes are read.
The model gate validates the existing 19c8e357 adapter/cache without downloads.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import traceback

REPO = Path(__file__).resolve().parents[3]
from protocol_config import BRANCH, LOCKED_UPSTREAM_SOURCE_HASHES

MANIFEST = Path(r"D:\\h2t_cross_domain_audit\\fish4knowledge\\manifest_primary.csv")
IMAGES = REPO / "data/fishRecognition_GT/fish_image"
EXPECTED_SHA = "62d9a2e9fffaaca22ef0c99c0ce4134b4d0e1ecc308333a7cb57b10cf95f0f30"
EXPECTED_SPLITS = {"train": (16269, 5208), "val": (5396, 1736), "test": (5468, 1736)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(row):
    relative = Path(row["relative_path"])
    assert not relative.is_absolute(), row["relative_path"]
    assert ".." not in relative.parts, row["relative_path"]
    assert len(relative.parts) >= 2, row["relative_path"]
    assert not any("mask" in part.lower() for part in relative.parts), row["relative_path"]
    class_id = int(relative.parts[0].removeprefix("fish_"))
    assert class_id in range(1, 17), row["relative_path"]
    return relative, class_id


def check_data():
    manifest_sha = sha(MANIFEST)
    assert manifest_sha == EXPECTED_SHA, "Locked manifest SHA mismatch"
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows, headers = list(reader), reader.fieldnames
    assert len(rows) == 27133
    labels = sorted({row["label"] for row in rows})
    assert len(labels) == 16

    for key in ("base_group", "group_id", "image_sha256"):
        assignments = defaultdict(set)
        for row in rows:
            assignments[row[key]].add(row["split_primary"])
        assert all(len(splits) == 1 for splits in assignments.values()), key

    support = {}
    for split, (images, groups) in EXPECTED_SPLITS.items():
        selected = [row for row in rows if row["split_primary"] == split]
        support[split] = {
            "images": len(selected),
            "groups": len({row["group_id"] for row in selected}),
        }
        assert support[split] == {"images": images, "groups": groups}

    per_class = []
    class_ids = {}
    for label in labels:
        selected = [row for row in rows if row["label"] == label]
        ids = {int(Path(row["relative_path"]).parts[0].removeprefix("fish_")) for row in selected}
        assert len(ids) == 1
        class_id = next(iter(ids))
        assert class_id in range(1, 17)
        class_ids[class_id] = label
        per_class.append({
            "class_id": class_id,
            "label": label,
            **{
                split: {
                    "images": sum(row["split_primary"] == split for row in selected),
                    "groups": len({row["group_id"] for row in selected if row["split_primary"] == split}),
                }
                for split in EXPECTED_SPLITS
            },
        })
    assert set(class_ids) == set(range(1, 17))

    image_root = IMAGES.resolve()
    train_val_verified = 0
    test_metadata_only = 0
    for row in rows:
        relative, _ = _safe_relative_path(row)
        split = row["split_primary"]
        if split == "test":
            # Deliberately do not resolve, stat, open, or hash TEST image paths.
            test_metadata_only += 1
            continue
        assert split in ("train", "val"), split
        path = (IMAGES / relative).resolve()
        assert path.is_relative_to(image_root), str(path)
        assert path.is_file(), str(path)
        assert sha(path) == row["image_sha256"], str(path)
        train_val_verified += 1
        if train_val_verified % 3000 == 0:
            print(
                f"Verified TRAIN/VAL image bytes: {train_val_verified}/21665",
                flush=True,
            )

    assert train_val_verified == EXPECTED_SPLITS["train"][0] + EXPECTED_SPLITS["val"][0]
    assert test_metadata_only == EXPECTED_SPLITS["test"][0]

    train = {row["label"]: row["train"]["images"] for row in per_class}
    ranked = sorted(labels, key=lambda label: (-train[label], label))
    return {
        "status": "PASS",
        "manifest": str(MANIFEST),
        "manifest_sha256": manifest_sha,
        "headers": headers,
        "rows": len(rows),
        "classes": sorted(per_class, key=lambda row: row["class_id"]),
        "split_support": support,
        "frequency_classes": {
            "Head": ranked[:6],
            "Medium": ranked[6:11],
            "Tail": ranked[11:],
        },
        "cross_split_leaks": 0,
        "train_val_image_sha256_verified": train_val_verified,
        "test_rows_metadata_only": test_metadata_only,
        "test_image_bytes_read": 0,
        "manifest_regenerated_or_modified": False,
    }


def check_approved_model_adapter():
    import torch
    import models

    actual_sources = models.source_hashes()
    assert actual_sources == LOCKED_UPSTREAM_SOURCE_HASHES, "Upstream H2T/PI-H2T source fingerprint mismatch"
    assert models.sha(models.ORIGINAL_WEIGHT) == models.ORIGINAL_SHA, "Official 19c8e357 weight SHA mismatch"
    assert models.SAFE_WEIGHT.is_file(), "Converted strict adapter tensor state is missing"
    assert models.sha(models.SAFE_WEIGHT), "Converted adapter state is empty or unreadable"

    provenance_path = models.CACHE / "ADAPTER_PROVENANCE.json"
    assert provenance_path.is_file(), "Adapter provenance is missing"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["source_sha256"] == models.ORIGINAL_SHA
    assert provenance["safe_sha256"] == models.sha(models.SAFE_WEIGHT)
    assert provenance["strict_load"] is True
    assert provenance["dropped_only"] == ["fc.weight", "fc.bias"]
    assert provenance["learned_or_floating_keys_filled"] == []

    # The approved adapter must load only the already-verified local tensor state.
    # Any attempt to download pretrained weights during this gate is a hard failure.
    original_download = torch.hub.load_state_dict_from_url
    try:
        def forbidden_download(*args, **kwargs):
            raise RuntimeError("Network weight download is forbidden during preflight")
        torch.hub.load_state_dict_from_url = forbidden_download
        plain = models.FishModel(False).eval()
        pif = models.FishModel(True).eval()
    finally:
        torch.hub.load_state_dict_from_url = original_download

    assert all(
        torch.equal(value, pif.extractor.backbone.state_dict()[name])
        for name, value in plain.extractor.backbone.state_dict().items()
    )
    assert all(
        torch.equal(value, pif.classifier.state_dict()[name])
        for name, value in plain.classifier.state_dict().items()
    )
    assert sum(parameter.numel() for parameter in pif.extractor.pif.parameters()) == 2
    del plain, pif

    return {
        "status": "PASS",
        "adapter": "models.FishModel / local strict adapter",
        "official_weight_sha256": models.ORIGINAL_SHA,
        "converted_adapter_sha256": models.sha(models.SAFE_WEIGHT),
        "adapter_provenance_sha256": sha(provenance_path),
        "upstream_source_hashes": actual_sources,
        "plain_and_pif_common_initialization": True,
        "pif_parameter_count": 2,
        "network_weight_downloads": 0,
        "legacy_ResNet50_FE_probe": "REMOVED_NOT_APPLICABLE",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    assert branch == BRANCH, f"Wrong branch: {branch}"
    assert not dirty, "Git worktree must be clean before preflight"

    output = Path(args.output_dir).resolve()
    assert not output.is_relative_to(REPO), "Preflight output must be outside Git"
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "branch": branch,
        "head": head,
        "worktree_clean": True,
        "training_runs": 0,
        "weights_downloaded": False,
        "original_sources_modified": False,
        "test_image_bytes_read": 0,
    }
    try:
        result["data_gate"] = check_data()
        result["model_gate"] = check_approved_model_adapter()
        result["status"] = "PASS_PREFLIGHT_ONLY"
    except Exception as exc:
        result.update(
            status="FAILED_PREFLIGHT",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    (output / "preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "PASS_PREFLIGHT_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
