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
