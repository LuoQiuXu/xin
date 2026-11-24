import numpy as np
import pandas as pd

from BCL_LTN.analyze_epoch160_source_attribution import (
    binary_ranking_metrics,
    class_group_lookup,
    compute_gate_reference,
)


def _synthetic_clean_counts():
    return np.asarray([200] * 35 + [50] * 35 + [10] * 30, dtype=np.int64)


def _synthetic_score_only_frame(clean_counts):
    lookup = class_group_lookup(clean_counts)
    observed = []
    raw = []
    for class_id in range(100):
        n = 1 if lookup[class_id] != "Few" else 51 + (1 if class_id >= 70 and class_id < 76 else 0)
        values = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        observed.extend([class_id] * n)
        raw.extend(values.tolist())
    observed = np.asarray(observed, dtype=np.int64)
    raw = np.asarray(raw, dtype=np.float64)
    normalized = np.empty_like(raw)
    for class_id in range(100):
        mask = observed == class_id
        values = raw[mask]
        normalized[mask] = (values - values.mean()) / (values.std(ddof=0) + 1e-8)
    frame = pd.DataFrame({
        "observed_label": observed,
        "raw_margin": raw,
        "class_normalized_raw_margin": normalized,
    })
    assert int(np.sum(np.asarray([lookup[c] == "Few" for c in observed]))) == 1536
    return frame, lookup


def test_class_group_lookup_uses_clean_if100_frequency_cutoffs():
    lookup = class_group_lookup(_synthetic_clean_counts())
    assert np.sum(lookup == "Many") == 35
    assert np.sum(lookup == "Medium") == 35
    assert np.sum(lookup == "Few") == 30
    assert lookup[0] == "Many"
    assert lookup[35] == "Medium"
    assert lookup[70] == "Few"


def test_binary_ranking_metrics_perfect_and_tied_scores():
    perfect = binary_ranking_metrics(
        np.asarray([1, 1, 0, 0]),
        np.asarray([0.9, 0.8, 0.2, 0.1]),
    )
    assert perfect["roc_auc"] == 1.0
    assert perfect["average_precision"] == 1.0

    tied = binary_ranking_metrics(
        np.asarray([1, 0]),
        np.asarray([0.5, 0.5]),
    )
    assert tied["roc_auc"] == 0.5
    assert tied["average_precision"] == 0.5


def test_gate_reference_is_score_only_and_uses_few_q90():
    frame, lookup = _synthetic_score_only_frame(_synthetic_clean_counts())
    reference = compute_gate_reference(frame, lookup)

    assert reference["score"] == "class_normalized_raw_margin"
    assert reference["scope"] == "observed Few labels only; Many and Medium are not gated"
    assert reference["n_observed_few"] == 1536
    assert len(reference["class_stats"]) == 100
    assert np.isfinite(reference["few_q90_threshold"])

    with_truth_columns = frame.copy()
    with_truth_columns["clean_label"] = np.arange(len(frame), dtype=np.int64) % 100
    with_truth_columns["is_clean"] = np.arange(len(frame), dtype=np.int64) % 2
    reference_with_truth_columns = compute_gate_reference(with_truth_columns, lookup)
    assert reference_with_truth_columns["few_q90_threshold"] == reference["few_q90_threshold"]
    assert reference_with_truth_columns["class_stats"] == reference["class_stats"]


def test_gate_reference_rejects_inconsistent_saved_zscores():
    frame, lookup = _synthetic_score_only_frame(_synthetic_clean_counts())
    frame.loc[0, "class_normalized_raw_margin"] += 0.25
    try:
        compute_gate_reference(frame, lookup)
    except RuntimeError as exc:
        assert "does not match reconstructed" in str(exc)
    else:
        raise AssertionError("Expected inconsistent class-normalized margins to be rejected")
