import numpy as np
import pytest

from BCL_LTN.audit_epoch160_margin_reliability import (
    EvalOnlyTrainDataset,
    binary_ranking_metrics,
    build_margin_scores,
    group_masks,
    stratified_bootstrap_ci,
    target_margin,
    threshold_metrics,
)


def test_target_margin_uses_observed_target_against_best_competitor():
    logits = np.asarray([[3.0, 2.0, 1.0], [0.0, 4.0, 2.0]])
    targets = np.asarray([0, 2])
    np.testing.assert_allclose(target_margin(logits, targets), [1.0, -2.0])


def test_four_margin_scores_use_observed_prior_and_opposite_signs():
    logits = np.asarray([
        [2.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 2.0],
        [2.0, 0.0, 1.0],
    ])
    observed = np.asarray([0, 1, 2, 0])
    counts = np.asarray([2, 1, 1])
    scores, log_prior, stats = build_margin_scores(logits, observed, counts)

    np.testing.assert_allclose(scores["raw_margin"], target_margin(logits, observed))
    np.testing.assert_allclose(
        scores["lc_plus_log_prior_margin"],
        target_margin(logits + log_prior[None, :], observed),
    )
    np.testing.assert_allclose(
        scores["inverse_prior_margin"],
        target_margin(logits - log_prior[None, :], observed),
    )
    assert len(stats) == 3
    assert np.all(np.isfinite(scores["class_normalized_raw_margin"]))


def test_auc_and_pr_metrics_handle_perfect_and_tied_scores():
    perfect = binary_ranking_metrics(
        np.asarray([1, 0, 1, 0]),
        np.asarray([0.9, 0.1, 0.8, 0.2]),
    )
    assert perfect["roc_auc"] == pytest.approx(1.0)
    assert perfect["average_precision"] == pytest.approx(1.0)
    assert perfect["pr_auc_trapezoid"] == pytest.approx(1.0)

    tied = binary_ranking_metrics(np.asarray([1, 0]), np.asarray([1.0, 1.0]))
    assert tied["roc_auc"] == pytest.approx(0.5)
    assert tied["average_precision"] == pytest.approx(0.5)
    assert tied["pr_auc_trapezoid"] == pytest.approx(0.75)


def test_class_normalization_preserves_within_class_ranking_metrics():
    observed = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    raw = np.asarray([3.0, 2.0, 1.0, 0.0, 10.0, 8.0, 6.0, 4.0])
    clean = np.asarray([True, False, True, False, True, False, True, False])
    normalized = np.empty_like(raw)
    for class_id in (0, 1):
        mask = observed == class_id
        normalized[mask] = (raw[mask] - raw[mask].mean()) / raw[mask].std()

    for class_id in (0, 1):
        mask = observed == class_id
        labels = clean[mask].astype(np.int64)
        raw_metrics = binary_ranking_metrics(labels, raw[mask])
        norm_metrics = binary_ranking_metrics(labels, normalized[mask])
        for metric in raw_metrics:
            assert norm_metrics[metric] == pytest.approx(raw_metrics[metric], abs=1e-12)


def test_group_masks_are_based_on_observed_labels():
    clean_counts = np.asarray([500, 100, 5])
    observed = np.asarray([0, 2, 2, 1, 0, 2])
    masks, groups = group_masks(observed, clean_counts)
    np.testing.assert_array_equal(np.flatnonzero(masks["Many"]), [0, 4])
    np.testing.assert_array_equal(np.flatnonzero(masks["Medium"]), [3])
    np.testing.assert_array_equal(np.flatnonzero(masks["Few"]), [1, 2, 5])
    assert groups["Few"].tolist() == [2]


def test_threshold_metrics_reports_noisy_acceptance_separately():
    labels = np.asarray([1, 1, 0, 0])
    scores = np.asarray([0.8, -0.1, 0.7, -0.2])
    result = threshold_metrics(labels, scores, 0.0)
    assert result["precision_clean"] == pytest.approx(0.5)
    assert result["recall_clean"] == pytest.approx(0.5)
    assert result["noisy_acceptance_rate"] == pytest.approx(0.5)
    assert result["clean_rejection_rate"] == pytest.approx(0.5)


def test_stratified_bootstrap_is_reproducible_and_preserves_prevalence():
    labels = np.asarray([1, 1, 0, 0, 0])
    scores = np.asarray([0.9, 0.7, 0.6, 0.2, -0.1])
    first = stratified_bootstrap_ci(labels, scores, reps=20, seed=17)
    second = stratified_bootstrap_ci(labels, scores, reps=20, seed=17)
    assert first == second
    assert set(first) == {"roc_auc", "average_precision", "pr_auc_trapezoid"}


def test_class_normalization_handles_constant_within_class_margins():
    logits = np.asarray([
        [2.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 1.0],
        [0.0, 2.0, 1.0],
        [0.0, 1.0, 2.0],
    ])
    observed = np.asarray([0, 0, 1, 1, 2])
    counts = np.asarray([2, 2, 1])
    scores, _, stats = build_margin_scores(logits, observed, counts)
    assert np.all(np.isfinite(scores["class_normalized_raw_margin"]))
    assert stats[0]["zero_or_near_zero_std"] is True
    np.testing.assert_allclose(scores["class_normalized_raw_margin"][:2], [0.0, 0.0])


def test_eval_dataset_returns_only_image_and_position():
    raw = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    dataset = EvalOnlyTrainDataset(raw, np.asarray([1]), transform=lambda image: np.asarray(image))
    sample = dataset[0]
    assert len(sample) == 2
    assert sample[1] == 0
    assert sample[0].shape == (32, 32, 3)
