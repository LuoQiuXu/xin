from __future__ import annotations

import torch

from tpr_model import (
    TailLowRankResidual,
    dominant_right_subspace,
    dominant_subspace_overlap,
    select_confusion_pairs,
    confusion_aware_diversity_loss,
    many_head_preservation_kl,
)


def counts_fixture():
    # 20 Many classes, 80 residual-eligible classes.
    return torch.tensor([500 - 20 * i for i in range(20)] + [100 - i for i in range(80)])


def test_zero_init_identity_and_mask():
    model = TailLowRankResidual(counts_fixture(), rank=8, seed=7)
    w = torch.randn(100, 64)
    b = torch.randn(100)
    x = torch.randn(12, 64)
    assert torch.equal(model.delta_weight(), torch.zeros_like(w))
    assert torch.equal(model.logits(x, w, b), torch.nn.functional.linear(x, w, b))
    model.assert_many_rows_zero()
    assert torch.equal(model.tail_mask[:20], torch.zeros(20))
    assert torch.equal(model.tail_mask[20:], torch.ones(80))


def test_two_step_factor_gradients_and_update():
    model = TailLowRankResidual(counts_fixture(), rank=8, seed=11)
    w = torch.randn(100, 64)
    x = torch.randn(16, 64)
    y = torch.arange(16) + 20
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    # At exact zero residual, B receives the first nonzero gradient; A's
    # gradient is mathematically zero until B becomes nonzero.
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model.logits(x, w), y)
        loss.backward()
        assert model.B.grad is not None and torch.isfinite(model.B.grad).all()
        if step == 0:
            assert model.A.grad is not None and torch.isfinite(model.A.grad).all()
            assert torch.count_nonzero(model.B.grad) > 0
        else:
            assert torch.count_nonzero(model.A.grad) > 0
        opt.step()
    assert torch.count_nonzero(model.B) > 0
    assert torch.count_nonzero(model.A) > 0
    model.assert_many_rows_zero()


def test_subspace_and_diversity_losses():
    model = TailLowRankResidual(counts_fixture(), rank=8, seed=13)
    with torch.no_grad():
        model.B.normal_(0, 0.01)
    w = torch.randn(100, 64)
    basis, singular = dominant_right_subspace(w, k=8)
    assert basis.shape == (64, 8)
    assert singular.shape == (64,)
    overlap = dominant_subspace_overlap(model.delta_weight(), basis)
    assert torch.isfinite(overlap) and overlap >= 0
    confusion = torch.zeros(100, 100)
    confusion[20, 21] = 4
    confusion[21, 20] = 2
    confusion[20, 22] = 1
    pairs = select_confusion_pairs(confusion, model.tail_mask, top_k=5)
    assert pairs[0][0:2] == (20, 21)
    div = confusion_aware_diversity_loss(model.delta_weight(), pairs)
    assert torch.isfinite(div)
    empty = confusion_aware_diversity_loss(model.delta_weight(), [])
    assert empty.requires_grad


def test_many_kd_full_distribution_and_empty_batch():
    model = TailLowRankResidual(counts_fixture(), rank=8)
    x = torch.randn(10, 100, requires_grad=True)
    teacher = torch.randn(10, 100)
    labels = torch.tensor([0, 1, 2, 3, 20, 21, 22, 23, 24, 25])
    loss, mask, n = many_head_preservation_kl(x, teacher, labels, counts_fixture(), 2.0)
    assert n == 4
    assert mask.tolist() == [True, True, True, True, False, False, False, False, False, False]
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None
    tail_labels = torch.full((5,), 25, dtype=torch.long)
    tail_student = torch.randn(5, 100, requires_grad=True)
    tail_teacher = torch.randn(5, 100)
    zero, empty_mask, n0 = many_head_preservation_kl(
        tail_student, tail_teacher, tail_labels, counts_fixture(), 2.0
    )
    assert n0 == 0 and not empty_mask.any() and zero.requires_grad



def test_diversity_uses_unweighted_cosines_and_zero_has_finite_backward():
    delta = torch.zeros(100, 64)
    delta[20, 0] = delta[21, 0] = delta[22, 1] = 1
    delta.requires_grad_()
    pairs = [(20, 21, 0.25), (20, 22, 0.75)]
    div = confusion_aware_diversity_loss(delta, pairs)
    assert div.item() == 0.5
    div.backward()
    assert torch.isfinite(delta.grad).all()
    for pairs in (pairs, []):
        zero = torch.zeros_like(delta, requires_grad=True)
        loss = confusion_aware_diversity_loss(zero, pairs)
        loss.backward()
        assert loss.item() == 0 and torch.equal(zero.grad, torch.zeros_like(zero))


def test_confusion_symmetry_normalization_top_five_and_ties():
    confusion = torch.eye(100, dtype=torch.int64) * 8
    for i, j in ((20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31)):
        confusion[i, j] = confusion[j, i] = 2
    confusion[0, 20] = 90
    mask = counts_fixture() <= 100
    pairs = select_confusion_pairs(confusion, mask)
    assert pairs == [(i, i + 1, 0.2) for i in (20, 22, 24, 26, 28)]
    assert select_confusion_pairs(confusion, mask) == pairs
    assert select_confusion_pairs(torch.eye(100), mask) == []


def test_subspace_ratio_is_scale_normalized_with_safe_zero():
    basis = torch.eye(64)[:, :8].contiguous()
    delta = torch.zeros(100, 64)
    delta[20, 0], delta[20, 8] = 3.0, 4.0
    delta.requires_grad_()
    penalty = dominant_subspace_overlap(delta, basis)
    assert torch.allclose(penalty, torch.tensor(9.0 / 25.0))
    assert torch.equal(penalty, dominant_subspace_overlap(delta * 4.0, basis))
    penalty.backward()
    assert torch.isfinite(delta.grad).all()
    zero = torch.zeros_like(delta, requires_grad=True)
    penalty = dominant_subspace_overlap(zero, basis)
    penalty.backward()
    assert penalty.item() == 0 and torch.equal(zero.grad, torch.zeros_like(zero))


def test_full_kl_matches_teacher_to_student_formula_and_detaches_teacher():
    generator = torch.Generator().manual_seed(5)
    student = torch.randn(4, 100, generator=generator, requires_grad=True)
    teacher = torch.randn(4, 100, generator=generator, requires_grad=True)
    counts = counts_fixture()
    counts[19] = 101
    labels = torch.tensor([19, 20, 0, 21])
    loss, mask, count = many_head_preservation_kl(student, teacher, labels, counts)
    assert mask.tolist() == [True, False, True, False] and count == 2
    expected = torch.nn.functional.kl_div(
        (student[mask] / 2).log_softmax(1), (teacher.detach()[mask] / 2).softmax(1),
        reduction="batchmean",
    ) * 4
    assert torch.equal(loss, expected)
    loss.backward()
    assert teacher.grad is None
    assert torch.equal(student.grad[~mask], torch.zeros_like(student.grad[~mask]))


def test_production_objective_preserves_base_and_activates_both_factors():
    from train_tpr import EffectiveLinear, OPTIMIZER, bitwise_equal, objective, verify_analytic_gradient

    torch.manual_seed(17)
    counts = counts_fixture()
    residual = TailLowRankResidual(counts, seed=123)
    copy = TailLowRankResidual(counts, seed=123)
    assert bitwise_equal(residual.A, copy.A)
    base = torch.nn.Linear(64, 100).requires_grad_(False).eval()
    before_weight, before_bias = base.weight.clone(), base.bias.clone()
    basis, _ = dominant_right_subspace(base.weight)
    pairs = [(20, 21, 0.25), (22, 23, 0.75)]
    optimizer = torch.optim.SGD(residual.parameters(), **OPTIMIZER)
    labels = torch.tensor([0, 20, 21, 22, 23, 24, 25, 26]).repeat(2)
    weights = torch.ones(100)
    for step in (0, 1):
        gap = torch.randn(len(labels), 64)
        head = EffectiveLinear(residual.effective_weight(base.weight).detach(), base.bias)
        assert verify_analytic_gradient(gap, labels, head, weights)["passed"]
        optimizer.zero_grad(set_to_none=True)
        loss, stats, student, teacher = objective(
            residual, base, basis, pairs, counts, weights, gap, labels, 0
        )
        if step == 0:
            assert bitwise_equal(student, teacher)
            assert stats["sub"] == stats["div"] == 0.0
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in residual.parameters())
        assert bool(torch.count_nonzero(residual.B.grad)) and bool(torch.count_nonzero(residual.A.grad)) == bool(step)
        previous_a = residual.A.detach().clone()
        optimizer.step()
        if step == 1:
            assert not bitwise_equal(previous_a, residual.A)
        residual.assert_many_rows_zero()
        assert bitwise_equal(base.weight, before_weight) and bitwise_equal(base.bias, before_bias)
        assert all(p.grad is None for p in base.parameters())
