import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from BCL_AUDIT.cifar100lt.losses import BalancedContrastiveLoss
from BCL_LTN.ntc_bcl_v2_core import (
    EMATeacher,
    EstimatedTransitionForwardCorrection,
    NoiseAwareSoftBalancedContrastiveLossV2,
    estimate_transition_em,
    posterior_from_balanced_probabilities,
    prototype_geometry,
    soft_bcl_batch_diagnostics,
)


def test_em_estimates_unknown_symmetric_noise_and_clean_prior_without_truth_inputs():
    rng = np.random.RandomState(17)
    clean = np.repeat(np.arange(4), [600, 300, 100, 30])
    n, k = len(clean), 4
    balanced = np.full((n, k), 0.005, dtype=np.float64)
    balanced[np.arange(n), clean] = 0.985
    transition = np.full((k, k), 0.2 / (k - 1), dtype=np.float64)
    np.fill_diagonal(transition, 0.8)
    observed = np.asarray([rng.choice(k, p=transition[c]) for c in clean], dtype=np.int64)

    estimate = estimate_transition_em(
        balanced, observed, shrinkage_mass=10.0, prior_alpha=0.1,
        high_support_fraction=0.5, max_iter=100,
    )
    assert estimate["transition"].shape == (k, k)
    assert estimate["posterior"].shape == (n, k)
    np.testing.assert_allclose(estimate["transition"].sum(1), 1.0, atol=1e-10)
    np.testing.assert_allclose(estimate["posterior"].sum(1), 1.0, atol=1e-10)
    assert abs(estimate["estimated_noise_rate"] - np.mean(clean != observed)) < 0.035
    assert np.mean(np.abs(estimate["clean_prior"] - np.bincount(clean, minlength=k) / n)) < 0.025


def test_em_probability_convention_applies_clean_prior_exactly_once():
    p_bal = np.array([[0.5, 0.5]], dtype=np.float64)
    prior = np.array([0.9, 0.1], dtype=np.float64)
    identity = np.eye(2, dtype=np.float64)
    gamma = posterior_from_balanced_probabilities(
        p_bal, prior, identity, np.array([0], dtype=np.int64)
    )
    np.testing.assert_allclose(gamma, [[0.9, 0.1]], atol=1e-12)


def test_low_support_transition_rows_shrink_to_nonuniform_global_structure():
    rng = np.random.RandomState(21)
    clean = np.concatenate((np.zeros(1000, dtype=np.int64),
                            np.ones(1000, dtype=np.int64),
                            np.full(5, 2, dtype=np.int64),
                            np.full(5, 3, dtype=np.int64)))
    k, n = 4, len(clean)
    balanced = np.full((n, k), 0.002, dtype=np.float64)
    balanced[np.arange(n), clean] = 0.994
    observed = clean.copy()
    # High-support classes have a shared but strongly non-uniform destination pattern.
    flip0 = np.flatnonzero(clean == 0)[::5]
    flip1 = np.flatnonzero(clean == 1)[::5]
    observed[flip0] = 1
    observed[flip1] = 0
    observed[-5:] = 3

    estimate = estimate_transition_em(
        balanced, observed, shrinkage_mass=20.0, prior_alpha=0.1,
        high_support_fraction=0.5, max_iter=100,
    )
    row2 = estimate["transition"][2]
    assert row2[0] + row2[1] > row2[3]
    assert abs(row2[0] - row2[1]) < 0.15
    assert np.allclose(estimate["transition"].sum(1), 1.0)


def test_v2_soft_bcl_preserves_one_hot_equivalence_when_repairs_are_off():
    rng = torch.Generator().manual_seed(9)
    batch, classes, dim = 5, 7, 11
    features = F.normalize(torch.randn(batch, 2, dim, generator=rng, dtype=torch.float64), dim=-1)
    prototypes = F.normalize(torch.randn(classes, dim, generator=rng, dtype=torch.float64), dim=-1)
    labels = torch.tensor([0, 3, 3, 5, 1])
    q = F.one_hot(labels, classes).to(torch.float64)
    views_a = features.clone().requires_grad_()
    proto_a = prototypes.clone().requires_grad_()
    views_b = features.clone().requires_grad_()
    proto_b = prototypes.clone().requires_grad_()

    reference = BalancedContrastiveLoss(np.ones(classes), temperature=0.1)
    candidate = NoiseAwareSoftBalancedContrastiveLossV2(
        classes, temperature=0.1, support_floor_ratio=0.0, denominator_epsilon=1e-6
    )
    expected = reference(proto_a, views_a, labels)
    actual = candidate(proto_b, views_b, q, semantic_ramp=1.0)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    expected_grads = torch.autograd.grad(expected, (views_a, proto_a))
    actual_grads = torch.autograd.grad(actual, (views_b, proto_b))
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)


def test_semantic_ramp_zero_is_instance_only_nt_xent():
    features = F.normalize(torch.randn(4, 2, 9), dim=-1)
    prototypes = F.normalize(torch.randn(5, 9), dim=-1)
    q = torch.full((4, 5), 0.2)
    criterion = NoiseAwareSoftBalancedContrastiveLossV2(5)
    expected = criterion._instance_only_loss(features)
    actual = criterion(prototypes, features, q, semantic_ramp=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_occurrence_floor_reduces_rare_prototype_inverse_denominator_pressure():
    q = np.full((8, 4), 0.01, dtype=np.float64)
    q[:, 0] = 0.97
    q[-1] = np.array([0.70, 0.10, 0.10, 0.10])
    no_floor = soft_bcl_batch_diagnostics(q, support_floor_ratio=0.0)
    with_floor = soft_bcl_batch_diagnostics(q, support_floor_ratio=0.5)
    assert with_floor["class_mass_floor"] > 0
    assert (
        with_floor["prototype_inverse_denominator_by_class"][3]
        < no_floor["prototype_inverse_denominator_by_class"][3]
    )


def test_ema_teacher_averages_parameters_but_copies_batchnorm_buffers():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)
            self.bn = nn.BatchNorm1d(2)

    student = TinyModel()
    with torch.no_grad():
        student.linear.weight.fill_(1.0)
        student.bn.running_mean.fill_(0.0)
    ema = EMATeacher(student, decay=0.5)
    with torch.no_grad():
        student.linear.weight.fill_(3.0)
        student.bn.running_mean.fill_(4.0)
    ema.update(student)
    torch.testing.assert_close(ema.model.linear.weight, torch.full((2, 2), 2.0))
    torch.testing.assert_close(ema.model.bn.running_mean, torch.full((2,), 4.0))
    assert ema.num_updates == 1
    assert all(not p.requires_grad for p in ema.model.parameters())


def test_estimated_transition_forward_correction_matches_manual_probability():
    prior = np.array([0.75, 0.25], dtype=np.float64)
    transition = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    logits = torch.tensor([[1.2, -0.4], [-0.2, 0.8]], dtype=torch.float64, requires_grad=True)
    observed = torch.tensor([1, 0], dtype=torch.long)
    criterion = EstimatedTransitionForwardCorrection(prior, transition)
    p_clean = torch.softmax(logits + torch.log(torch.tensor(prior)), dim=1)
    t = torch.tensor(transition, dtype=torch.float64)
    p_observed = p_clean @ t
    expected = -torch.log(p_observed[torch.arange(2), observed]).mean()
    actual = criterion(logits, observed)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_prototype_geometry_compares_raw_and_direction_inputs_without_bn_mutation():
    from BCL_AUDIT.cifar100lt.models import ProjectionMLP

    mlp = ProjectionMLP(input_dim=4, hidden_dim=8, output_dim=3)
    mlp.train()
    weight = torch.randn(5, 4)
    with torch.no_grad():
        weight[0].mul_(0.001)
        weight[1].mul_(0.01)
    before = {k: v.clone() for k, v in mlp.state_dict().items()}
    result = prototype_geometry(weight, mlp)
    after = mlp.state_dict()
    assert "raw_weight_input" in result and "direction_normalized_weight_input" in result
    assert len(result["classifier_weight_norm"]) == 5
    assert mlp.training is True
    for key in before:
        torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)
