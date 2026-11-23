from __future__ import annotations

import csv
import json

import numpy as np

from BCL_AUDIT.scripts.phase2_decision_audit import (
    _build_decision_brief,
    _build_h2_associations,
    _describe_exposure_groups,
    _leave_one_few_class_out,
    _loco_supports_direction,
    _matched_pair_effects,
    _partial_spearman,
    _standardized_beta,
    load_and_validate_dino_topology,
)


def _few_classes():
    return [{"class_id": i, "fine_name": f"few_{i}"} for i in range(70, 100)]


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
    assert _standardized_beta(exposure, outcome, controls) > 0.2


def test_loco_keeps_the_adjusted_direction_for_a_diffuse_association():
    rng = np.random.RandomState(17)
    controls = rng.normal(size=(30, 2))
    exposure = controls[:, 0] + rng.normal(size=30)
    outcome = 2 * exposure + 0.3 * controls[:, 1] + rng.normal(scale=0.1, size=30)
    full = {
        "standardized_beta": _standardized_beta(exposure, outcome, controls),
        "partial_spearman": _partial_spearman(exposure, outcome, controls),
    }
    loco = _leave_one_few_class_out(exposure, outcome, controls, _few_classes(), full)
    for stat in ("beta", "rho"):
        assert loco[f"loco_{stat}_valid_count"] == 30
        assert 0 < loco[f"loco_{stat}_min"] <= loco[f"loco_{stat}_max"]
        assert loco[f"loco_{stat}_same_sign_fraction"] == 1.0
        assert loco[f"loco_{stat}_sign_reversals"] == 0
    assert _loco_supports_direction({**full, **loco}, 1)


def test_loco_identifies_the_class_that_reverses_beta():
    ordinary = np.linspace(-2, 2, 29)
    exposure = np.r_[ordinary, 30.0]
    outcome = np.r_[ordinary, -30.0]
    controls = np.empty((30, 0))
    full = {
        "standardized_beta": _standardized_beta(exposure, outcome, controls),
        "partial_spearman": _partial_spearman(exposure, outcome, controls),
    }
    loco = _leave_one_few_class_out(exposure, outcome, controls, _few_classes(), full)
    beta_without_outlier = _standardized_beta(exposure[:-1], outcome[:-1], controls[:-1])
    assert full["standardized_beta"] < 0 < beta_without_outlier
    assert loco["loco_beta_sign_reversals"] == 1
    assert loco["loco_beta_same_sign_fraction"] == 29 / 30
    assert loco["loco_beta_largest_change_class_id"] == 99
    assert loco["loco_beta_largest_change_class_name"] == "few_99"
    assert loco["loco_beta_max_abs_change"] == abs(beta_without_outlier - full["standardized_beta"])
    assert not _loco_supports_direction({**full, **loco}, -1)


def test_loco_marks_nonfinite_degenerate_associations_unavailable():
    exposure = np.ones(30)
    exposure[-2:] = (np.inf, np.nan)
    outcome = np.linspace(0.0, 1.0, 30)
    outcome[5] = np.nan
    controls = np.zeros((30, 1))
    controls[8, 0] = np.inf
    full = {
        "standardized_beta": _standardized_beta(exposure, outcome, controls),
        "partial_spearman": _partial_spearman(exposure, outcome, controls),
    }
    loco = _leave_one_few_class_out(exposure, outcome, controls, _few_classes(), full)
    assert np.isnan(full["standardized_beta"]) and np.isnan(full["partial_spearman"])
    for stat in ("beta", "rho"):
        assert loco[f"loco_{stat}_valid_count"] == 0
        assert loco[f"loco_{stat}_min"] is None
        assert loco[f"loco_{stat}_max"] is None
        assert loco[f"loco_{stat}_same_sign_fraction"] is None
        assert loco[f"loco_{stat}_sign_reversals"] is None
        assert loco[f"loco_{stat}_max_abs_change"] is None


def test_loco_has_no_direction_statistic_for_a_zero_or_nonfinite_full_estimate():
    exposure = np.linspace(-2, 2, 30)
    outcome = exposure ** 2
    controls = np.empty((30, 0))
    loco = _leave_one_few_class_out(
        exposure, outcome, controls, _few_classes(),
        {"standardized_beta": 0.0, "partial_spearman": np.nan},
    )
    for stat in ("beta", "rho"):
        assert loco[f"loco_{stat}_valid_count"] == 30
        assert loco[f"loco_{stat}_same_sign_fraction"] is None
        assert loco[f"loco_{stat}_sign_reversals"] is None
    assert loco["loco_beta_max_abs_change"] > 0
    assert loco["loco_rho_max_abs_change"] is None


def test_binary_exposure_descriptives_use_finite_outcomes_and_separate_groups():
    exposure = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, np.nan])
    outcome = np.array([5.0, 1.0, 7.0, 2.0, 9.0, np.inf, 100.0])
    summary = _describe_exposure_groups(exposure, outcome, np.arange(7), continuous=False)
    assert summary["descriptive_exposed_n"] == 2
    assert summary["descriptive_unexposed_n"] == 3
    assert summary["descriptive_exposed_mean"] == 6.0
    assert summary["descriptive_unexposed_mean"] == 4.0
    assert summary["descriptive_exposed_median"] == 6.0
    assert summary["descriptive_unexposed_median"] == 2.0
    assert summary["descriptive_mean_difference_exposed_minus_unexposed"] == 2.0


def test_continuous_exposure_split_has_fifteen_per_side_and_breaks_ties_by_id():
    class_ids = np.arange(99, 69, -1)
    exposure = np.full(30, 0.7)
    outcome = class_ids.astype(float)
    summary = _describe_exposure_groups(exposure, outcome, class_ids, continuous=True)
    assert summary["descriptive_unexposed_n"] == summary["descriptive_exposed_n"] == 15
    assert summary["descriptive_unexposed_mean"] == summary["descriptive_unexposed_median"] == 77.0
    assert summary["descriptive_exposed_mean"] == summary["descriptive_exposed_median"] == 92.0
    assert summary["descriptive_mean_difference_exposed_minus_unexposed"] == 15.0


def test_h2_additions_keep_six_rows_and_the_brief_stops_for_heterogeneity_alone():
    exposure_rows = [
        {
            **few,
            "max_dino_similarity_to_many": 0.7,
            "has_dino_many_neighbor": True,
            "has_same_superclass_many": False,
            "LC_only_test_accuracy_pct": 25.0,
            "log_train_count": 3.0,
            "LC_tail_to_head_mean_confusion_pp": 1.0,
            "delta_accuracy_pp": float(i - 15),
            "delta_tail_to_head_mean_confusion_pp": float(15 - i),
        }
        for i, few in enumerate(_few_classes())
    ]
    associations = _build_h2_associations(exposure_rows)
    assert len(associations) == 6
    assert all(r["bootstrap_reps_requested"] == 2000 and r["loco_omissions_requested"] == 30
               and r["loco_beta_valid_count"] == r["loco_rho_valid_count"] == 0 for r in associations)
    gain = associations[0]
    assert gain["descriptive_exposed_n"] == gain["descriptive_unexposed_n"] == 15
    assert gain["descriptive_mean_difference_exposed_minus_unexposed"] == 15.0

    h1 = {
        "by_shot_group": {"few": {
            "num_classes": 30, "mean_gain_pp": 0.0, "median_gain_pp": 0.0, "std_gain_pp": 3.0,
            "q25_gain_pp": -3.0, "q75_gain_pp": 3.0, "min_gain_pp": -5.0, "max_gain_pp": 5.0,
            "positive_classes": 15, "zero_classes": 0, "negative_classes": 15,
        }},
        "few_gain_mixed_sign": True, "few_top10": [], "few_bottom10": [],
    }
    pairs = [
        {
            "relation": group, "direction": "Tail_to_Head", "metric": "delta_confusion_BCL_minus_LC",
            "bootstrap_ci_2_5": -0.1, "bootstrap_ci_97_5": 0.1, "matched_did_exposed_minus_control": 0.0,
            "exposed_pair_count": 10, "control_pair_count": 1040, "matched_exposed_pair_count": 10,
            "unadjusted_did_exposed_minus_control": 0.0, "valid_two_way_class_cluster_bootstrap_reps": 2000,
            "bootstrap_reps_requested": 2000, "baseline_confusion_smd_before_matching": 0.0,
            "baseline_confusion_smd_after_matching": 0.0,
        }
        for group in ("same_superclass", "dino_topk_union")
    ]
    brief, decision = _build_decision_brief(h1, associations, pairs, [], [])
    assert decision["evidence_level"] == "UNSUPPORTED"
    assert decision["candidate_intervention"] == decision["stop_condition"]
    assert "Valid omissions" in brief and "Exposed minus unexposed mean" in brief


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
