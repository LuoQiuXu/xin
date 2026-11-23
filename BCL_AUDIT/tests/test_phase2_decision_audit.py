from __future__ import annotations

import csv
import json

import numpy as np

from BCL_AUDIT.scripts.phase2_decision_audit import (
    _matched_pair_effects,
    _partial_spearman,
    _standardized_beta,
    load_and_validate_dino_topology,
)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_partial_spearman_and_adjusted_beta_recover_positive_association():
    rng = np.random.RandomState(7)
    control = np.linspace(-1.0, 1.0, 30)
    exposure = control + rng.normal(0, 0.25, size=30)
    outcome = 1.8 * exposure + 0.3 * control + rng.normal(0, 0.15, size=30)
    controls = control[:, None]
    assert _partial_spearman(exposure, outcome, controls) > 0.7
    assert _standardized_beta(exposure, outcome, controls) > 0.7


def test_nearest_baseline_matching_removes_baseline_difference():
    rows = [
        {"same_superclass": "true", "LC_confusion_rate": "0.00",
         "delta_confusion_BCL_minus_LC": "0.20",
         "delta_raw_margin_BCL_minus_LC": "-0.5",
         "delta_adjusted_margin_BCL_minus_LC": "-0.4"},
        {"same_superclass": "true", "LC_confusion_rate": "0.10",
         "delta_confusion_BCL_minus_LC": "0.30",
         "delta_raw_margin_BCL_minus_LC": "-0.6",
         "delta_adjusted_margin_BCL_minus_LC": "-0.5"},
        {"same_superclass": "false", "LC_confusion_rate": "0.00",
         "delta_confusion_BCL_minus_LC": "0.00",
         "delta_raw_margin_BCL_minus_LC": "0.0",
         "delta_adjusted_margin_BCL_minus_LC": "0.0"},
        {"same_superclass": "false", "LC_confusion_rate": "0.10",
         "delta_confusion_BCL_minus_LC": "0.10",
         "delta_raw_margin_BCL_minus_LC": "-0.1",
         "delta_adjusted_margin_BCL_minus_LC": "-0.1"},
    ]
    result = _matched_pair_effects(rows, "same_superclass", np.ones(4))
    assert result is not None
    assert abs(result["baseline_smd_after"]) < 1e-12
    assert abs(result["delta_confusion_BCL_minus_LC__matched_did"] - 0.20) < 1e-12


def test_dino_topology_validator_reconstructs_symmetric_union(tmp_path):
    class_meta = {}
    for class_id in range(100):
        group = "many" if class_id < 35 else "medium" if class_id < 70 else "few"
        class_meta[class_id] = {"coarse_id": str(class_id % 20), "shot_group": group}

    similarity = np.zeros((100, 100), dtype=np.float32)
    np.fill_diagonal(similarity, 1.0)
    near_pairs = set()
    for class_id in range(100):
        row = similarity[class_id].copy()
        row[class_id] = -np.inf
        for neighbor in np.argsort(-row)[:5]:
            near_pairs.add(tuple(sorted((class_id, int(neighbor)))))

    many_ids = list(range(35))
    few_ids = list(range(70, 100))
    sidecar = []
    for head in many_ids:
        for tail in few_ids:
            sidecar.append({
                "head_fine_id": head,
                "tail_fine_id": tail,
                "same_superclass": class_meta[head]["coarse_id"] == class_meta[tail]["coarse_id"],
                "dino_cosine_similarity": float(similarity[head, tail]),
                "dino_topk_union": tuple(sorted((head, tail))) in near_pairs,
            })

    np.save(tmp_path / "DINO_CLASS_SIMILARITY.npy", similarity)
    np.save(tmp_path / "DINO_CLASS_MEAN_FEATURES.npy", np.ones((100, 4), dtype=np.float32))
    with (tmp_path / "dino_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "test_split_loaded": False,
            "source_split": "original balanced CIFAR-100 training split only",
            "topk": 5,
            "topk_pair_rule": "symmetric union: either class lists the other in its Top-K",
        }, stream)
    near_count = sum(tuple(sorted((h, t))) in near_pairs for h in many_ids for t in few_ids)
    with (tmp_path / "data_audit_dino_summary.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "dino_enrichment_completed": True,
            "test_split_loaded": False,
            "source_split": "original balanced CIFAR-100 training split only",
            "dino_topk_union_many_few_pairs": near_count,
        }, stream)
    _write_csv(tmp_path / "head_tail_pair_topology_dino.csv", sidecar)
    _write_csv(tmp_path / "DINO_CLASS_TOPOLOGY.csv", [
        {"fine_id": i, "coarse_id": class_meta[i]["coarse_id"]} for i in range(100)
    ])

    result = load_and_validate_dino_topology(tmp_path, class_meta)
    assert len(result["many_ids"]) == 35
    assert len(result["few_ids"]) == 30
    assert len(result["sidecar_rows"]) == 1050
    assert result["near_pairs"] == near_pairs


def test_dino_topology_validator_rejects_test_derived_metadata(tmp_path):
    class_meta = {
        i: {
            "coarse_id": str(i % 20),
            "shot_group": "many" if i < 35 else "medium" if i < 70 else "few",
        }
        for i in range(100)
    }
    for filename in (
        "DINO_CLASS_MEAN_FEATURES.npy",
        "DINO_CLASS_SIMILARITY.npy",
        "dino_metadata.json",
        "head_tail_pair_topology_dino.csv",
        "DINO_CLASS_TOPOLOGY.csv",
        "data_audit_dino_summary.json",
    ):
        (tmp_path / filename).write_text("placeholder", encoding="utf-8")
    with (tmp_path / "dino_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump({"test_split_loaded": True}, stream)
    with (tmp_path / "data_audit_dino_summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"test_split_loaded": True}, stream)
    try:
        load_and_validate_dino_topology(tmp_path, class_meta)
    except RuntimeError as exc:
        assert "TRAIN-only provenance" in str(exc)
    else:
        raise AssertionError("Test-derived DINO metadata must be rejected")
