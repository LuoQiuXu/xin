"""Unit tests for Exact-V2 alignment audit metrics, truth isolation, and provenance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from BCL_LTN.audit_v2_exact_estimator_alignment import (
    coefficient_of_variation,
    compute_detailed_prior_and_group_metrics,
    spearman_rank_correlation,
)
from BCL_LTN.make_noisy_lt import group_class_ids
from BCL_LTN.train_ntc_bcl_v2 import collect_source_provenance


def test_spearman_and_cv_handles_ties_and_monotone_order():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman_rank_correlation(x, x * 2.0) == pytest.approx(1.0)
    assert spearman_rank_correlation(x, -x) == pytest.approx(-1.0)
    tied = np.array([10.0, 10.0, 20.0, 30.0])
    assert spearman_rank_correlation(np.array([1.0, 2.0, 3.0, 4.0]), tied) > 0.9
    assert coefficient_of_variation(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0)
    assert coefficient_of_variation(np.array([10.0, 20.0])) == pytest.approx(5.0 / 15.0)


def test_detailed_group_metrics_reports_mass_ratio_and_polarization():
    clean_counts = np.array([200, 150, 50, 30, 15, 5], dtype=np.int64)
    pi_true = clean_counts / clean_counts.sum()
    groups = group_class_ids(clean_counts)
    # Deliberately collapse class 4 (<0.5x) and inflate class 5 (>2x)
    pi_hat = np.array([200, 150, 50, 30, 2, 18], dtype=np.float64)
    pi_hat = pi_hat / pi_hat.sum()
    t = np.eye(6, dtype=np.float64)
    metrics = compute_detailed_prior_and_group_metrics(
        pi_hat, t, pi_true, t, t, clean_counts, groups
    )
    few = metrics["groups"]["Few"]
    assert few["classes_under_0_5x_true"] == 1
    assert few["classes_over_2_0x_true"] == 1
    assert few["spearman_prior"] == pytest.approx(-1.0)


def test_collect_source_provenance_records_v2_file_hashes():
    prov = collect_source_provenance()
    assert prov["git_branch"]
    assert prov["git_head"]
    hashes = prov["v2_source_sha256"]
    assert "BCL_LTN/ntc_bcl_v2_losses.py" in hashes
    assert "BCL_LTN/train_ntc_bcl_v2.py" in hashes
    assert len(hashes["BCL_LTN/train_ntc_bcl_v2.py"]) == 64
