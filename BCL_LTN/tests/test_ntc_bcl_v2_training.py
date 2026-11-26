"""Unit tests for NTC-BCL v2 losses, anchored EM estimator, EMA teacher, and gates."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from BCL_AUDIT.cifar100lt.losses import BalancedContrastiveLoss
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_LTN.ntc_bcl_v2_losses import (
    EMATeacher,
    EstimatedTransitionForwardCorrection,
    NoiseAwareSoftBalancedContrastiveLossV2,
    blend_transition_and_prior,
    estimate_anchored_transition_and_prior,
    measure_prototype_and_classifier_health,
)
from BCL_LTN.train_ntc_bcl_v2 import RegimeTruthGate, semantic_ramp_for_epoch


def test_soft_bcl_v2_matches_hard_bcl_when_onehot_and_no_floor():
    torch.manual_seed(0)
    k, b, d = 6, 12, 8
    protos = F.normalize(torch.randn(k, d), dim=1)
    two_views = F.normalize(torch.randn(b, 2, d), dim=2)
    targets = torch.randint(0, k, (b,))
    onehot = F.one_hot(targets, k).to(two_views.dtype)

    hard = BalancedContrastiveLoss([10] * k, temperature=0.1)
    soft_v2 = NoiseAwareSoftBalancedContrastiveLossV2(
        k, temperature=0.1, support_floor_ratio=0.0, denominator_epsilon=1e-8
    )
    loss_hard = hard(protos, two_views, targets)
    loss_soft, diag = soft_v2(
        protos, two_views, onehot, semantic_ramp=1.0, return_diagnostics=True
    )
    assert torch.allclose(loss_soft, loss_hard, atol=1e-5, rtol=1e-5)
    assert diag["semantic_ramp"] == pytest.approx(1.0)


def test_soft_bcl_v2_warmup_and_continuity():
    torch.manual_seed(1)
    k, b, d = 5, 10, 8
    protos = F.normalize(torch.randn(k, d), dim=1)
    two_views = F.normalize(torch.randn(b, 2, d), dim=2)
    q = F.softmax(torch.randn(b, k), dim=1)

    loss_fn = NoiseAwareSoftBalancedContrastiveLossV2(
        k, temperature=0.1, support_floor_ratio=0.5, denominator_epsilon=1e-6
    )
    l_zero = loss_fn(None, two_views, None, semantic_ramp=0.0)
    l_tiny = loss_fn(protos, two_views, q, semantic_ramp=1e-6)
    l_small = loss_fn(protos, two_views, q, semantic_ramp=1e-4)
    assert torch.isfinite(l_zero) and torch.isfinite(l_tiny) and torch.isfinite(l_small)
    assert abs(float(l_small - l_tiny)) < 1e-3


def test_soft_bcl_v2_support_floor_bounds_tail_denominator():
    torch.manual_seed(2)
    k, b, d = 10, 16, 8
    protos = F.normalize(torch.randn(k, d), dim=1)
    two_views = F.normalize(torch.randn(b, 2, d), dim=2)
    q = torch.zeros(b, k)
    q[:, 0] = 1.0  # classes 1..9 have zero sample mass

    no_floor = NoiseAwareSoftBalancedContrastiveLossV2(k, support_floor_ratio=0.0)
    with_floor = NoiseAwareSoftBalancedContrastiveLossV2(k, support_floor_ratio=0.5)
    _, d0 = no_floor(protos, two_views, q, return_diagnostics=True)
    _, d1 = with_floor(protos, two_views, q, return_diagnostics=True)
    assert d0["denominator_weight_min"] == pytest.approx(1.0)
    assert d1["denominator_weight_min"] > d0["denominator_weight_min"]


def test_ema_teacher_tracks_params_and_copies_buffers():
    torch.manual_seed(3)
    student = BCLCIFARModel(num_classes=10)
    teacher = EMATeacher(student, decay=0.9, warmup_ramp=False)
    assert not teacher.model.training
    assert all(not p.requires_grad for p in teacher.model.parameters())

    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)
        for name, buf in student.named_buffers():
            if buf.is_floating_point():
                buf.fill_(7.0)
            else:
                buf.fill_(42)
    alpha = teacher.update(student)
    assert alpha == pytest.approx(0.9)
    assert teacher.num_updates == 1
    for (n, ts), (_, tt) in zip(student.named_buffers(), teacher.model.named_buffers()):
        assert torch.equal(ts, tt), f"Buffer {n} not copied"
    for p_s, p_t in zip(student.parameters(), teacher.model.parameters()):
        assert torch.allclose(p_t, p_s - 0.9, atol=1e-6)


def test_forward_correction_inplace_update_and_identity_equivalence():
    torch.manual_seed(4)
    k, b = 8, 12
    counts = torch.tensor([50, 40, 30, 20, 15, 10, 8, 5], dtype=torch.float64)
    pi = counts / counts.sum()
    eye = torch.eye(k, dtype=torch.float64)

    corr = EstimatedTransitionForwardCorrection(pi, eye)
    logits = torch.randn(b, k)
    targets = torch.randint(0, k, (b,))
    loss_id = corr(logits, targets)
    ce_ref = F.cross_entropy(logits + pi.float().log(), targets)
    assert torch.allclose(loss_id, ce_ref, atol=1e-5)

    t_noisy = torch.full((k, k), 0.4 / (k - 1), dtype=torch.float64)
    t_noisy.fill_diagonal_(0.6)
    new_pi = torch.full((k,), 1.0 / k, dtype=torch.float64)
    corr.update_transition_and_prior(new_pi, t_noisy)
    assert torch.allclose(corr.prior, new_pi.float())
    assert torch.allclose(corr.transition, t_noisy.float())


@pytest.mark.parametrize("regime", ["symmetric", "asymmetric"])
def test_anchored_em_recovers_long_tail_prior_and_transition(regime: str):
    rng = np.random.default_rng(123)
    k = 10
    clean_counts = np.array([400, 300, 220, 150, 100, 70, 45, 30, 18, 12])
    n = int(clean_counts.sum())
    pi_true = clean_counts / n
    clean_y = np.repeat(np.arange(k), clean_counts)

    t_true = np.zeros((k, k), dtype=np.float64)
    if regime == "symmetric":
        t_true[:] = 0.4 / (k - 1)
        np.fill_diagonal(t_true, 0.6)
    else:
        for c in range(k):
            t_true[c, c] = 0.6
            t_true[c, (c + 1) % k] = 0.4

    obs_y = np.array([rng.choice(k, p=t_true[c]) for c in clean_y], dtype=np.int64)
    # Realistic confident balanced teacher probabilities with slight noise
    probs = np.full((n, k), 0.015 / (k - 1), dtype=np.float64)
    probs[np.arange(n), clean_y] = 0.985

    est = estimate_anchored_transition_and_prior(
        torch.from_numpy(probs),
        torch.from_numpy(obs_y),
        confidence_power=2.0,
        shrinkage_mass=5.0,
        prior_pseudocount=1.0,
        max_iter=200,
    )
    assert est.converged
    t_hat = est.transition.numpy()
    pi_hat = est.clean_prior.numpy()
    assert np.mean(np.abs(t_hat - t_true)) < 0.03
    assert np.mean(np.abs(pi_hat - pi_true)) < 0.005
    assert pi_hat.min() > 0.5 * pi_true.min()


def test_anchored_em_resists_diag_collapse_under_diffuse_predictions():
    rng = np.random.default_rng(999)
    k, n = 20, 4000
    clean_y = rng.integers(0, k, size=n)
    t_true = np.full((k, k), 0.4 / (k - 1))
    np.fill_diagonal(t_true, 0.6)
    obs_y = np.array([rng.choice(k, p=t_true[c]) for c in clean_y])

    # Simulated 65% clean teacher with 10% wrong-label leakage into noisy labels
    probs = np.full((n, k), 0.005, dtype=np.float64)
    probs[np.arange(n), clean_y] += 0.65
    probs[np.arange(n), obs_y] += 0.15
    probs = probs / probs.sum(axis=1, keepdims=True)

    est = estimate_anchored_transition_and_prior(
        torch.from_numpy(probs),
        torch.from_numpy(obs_y),
        confidence_power=2.0,
        shrinkage_mass=10.0,
        prior_pseudocount=1.0,
    )
    assert 0.25 < est.estimated_noise_rate < 0.55


def test_blend_and_prototype_health_preserve_invariants():
    k = 6
    pi1 = torch.full((k,), 1.0 / k, dtype=torch.float64)
    t1 = torch.eye(k, dtype=torch.float64)
    pi2 = torch.tensor([0.4, 0.2, 0.15, 0.1, 0.1, 0.05], dtype=torch.float64)
    t2 = torch.full((k, k), 0.1 / (k - 1), dtype=torch.float64)
    t2.fill_diagonal_(0.9)
    new_pi, new_t = blend_transition_and_prior(pi1, t1, pi2, t2, blend_alpha=0.5)
    assert math.isclose(float(new_pi.sum()), 1.0, abs_tol=1e-7)
    assert torch.allclose(new_t.sum(dim=1), torch.ones(k, dtype=torch.float64))

    model = BCLCIFARModel(num_classes=k)
    model.prototype_mlp.train()
    before_bn = model.prototype_mlp.net[1].running_mean.clone()
    health = measure_prototype_and_classifier_health(
        model.classifier.weight, model.prototype_mlp
    )
    assert model.prototype_mlp.training
    assert torch.equal(model.prototype_mlp.net[1].running_mean, before_bn)
    assert "dir_mean_offdiag_cosine" in health and "raw_mean_offdiag_cosine" in health


def test_semantic_ramp_schedule_and_truth_gate(tmp_path: Path):
    assert semantic_ramp_for_epoch(0, 20, 20) == 0.0
    assert semantic_ramp_for_epoch(19, 20, 20) == 0.0
    assert semantic_ramp_for_epoch(20, 20, 20) == pytest.approx(1.0 / 20.0)
    assert semantic_ramp_for_epoch(39, 20, 20) == pytest.approx(1.0)
    assert semantic_ramp_for_epoch(150, 20, 20) == pytest.approx(1.0)

    regime_dir = tmp_path / "sym04"
    (regime_dir / "truth").mkdir(parents=True)
    secret = regime_dir / "truth" / "clean_labels.npy"
    secret.write_bytes(b"secret")
    gate = RegimeTruthGate(regime_dir, legacy_noise_root=tmp_path / "legacy")
    gate.install()
    with pytest.raises(RuntimeError, match="before prediction freeze"):
        secret.read_bytes()
    assert gate.denied == 1
    gate.denied = 0
    with gate.offline_only():
        assert secret.read_bytes() == b"secret"
    assert gate.opens == 1


def test_official_test_audit_blocks_outside_final_evaluation(tmp_path: Path):
    from BCL_LTN.train_noisy_bcl import OfficialTestAudit

    data_root = tmp_path / "cifar100"
    test_dir = data_root / "cifar-100-python"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "test"
    test_file.write_bytes(b"test-bytes")

    gate = OfficialTestAudit(data_root)
    with pytest.raises(RuntimeError, match="outside final evaluation"):
        test_file.read_bytes()
    assert gate.denied == 1
    gate.denied = 0
    with gate.final_evaluation():
        assert test_file.read_bytes() == b"test-bytes"
    assert gate.opens == 1
