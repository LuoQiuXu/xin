"""CPU unit tests for the donor-affinity × margin-protection policy."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from policy import (
    build_frequency_strata,
    build_head_prototypes,
    receiver_prototype_affinity,
    choose_donor_classes,
    sample_donor_indices,
    channel_margin_delta,
    protected_lowvs_mask,
)


class TinyGapClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 100)

    def forward(self, maps):
        return self.fc(maps.mean(dim=(-2, -1)))


def test_frequency_strata_and_common_random_numbers():
    counts = np.asarray([500 - 4 * i for i in range(100)], dtype=np.int64)
    strata = build_frequency_strata(counts)
    assert len(strata) == 6
    assert all(len(s) == 6 for s in strata)
    assert sorted(np.concatenate(strata).tolist()) == list(range(36))

    head_ids = np.arange(36, dtype=np.int64)
    affinity = np.tile(head_ids.astype(np.float32), (12, 1))
    u_stratum = np.linspace(0.01, 0.99, 12)
    u_class = np.linspace(0.07, 0.93, 12)
    random_c, random_s, _ = choose_donor_classes(
        affinity, head_ids, counts, strata, u_stratum, u_class, "random"
    )
    high_c, high_s, _ = choose_donor_classes(
        affinity, head_ids, counts, strata, u_stratum, u_class, "high"
    )
    bottom_c, bottom_s, _ = choose_donor_classes(
        affinity, head_ids, counts, strata, u_stratum, u_class, "bottom"
    )
    assert np.array_equal(random_s, high_s) and np.array_equal(high_s, bottom_s)
    for i, si in enumerate(high_s):
        s = strata[int(si)]
        assert high_c[i] in s[-2:]
        assert bottom_c[i] in s[:2]
        assert random_c[i] in s
    print("PASS: frequency strata and paired random numbers")


def test_prototypes_and_donor_indices_are_train_label_consistent():
    rng = np.random.default_rng(9)
    labels = np.repeat(np.arange(100), 3)
    features = rng.normal(size=(len(labels), 64)).astype(np.float32)
    head_ids, prototypes = build_head_prototypes(features, labels)
    assert head_ids.shape == (36,) and prototypes.shape == (36, 64)
    assert np.allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1e-5)
    affinity = receiver_prototype_affinity(features[:8], head_ids, prototypes)
    assert affinity.shape == (8, 36) and np.isfinite(affinity).all()
    donor_classes = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
    donor_idx = sample_donor_indices(donor_classes, labels, np.linspace(0.01, 0.99, 8))
    assert np.array_equal(labels[donor_idx], donor_classes)
    print("PASS: train-only prototypes and donor-label consistency")


def test_channel_delta_is_exact_for_fixed_competitor():
    torch.manual_seed(20261019)
    model = TinyGapClassifier()
    receiver = torch.randn(7, 64, 8, 8)
    donor = torch.randn_like(receiver)
    target = torch.tensor([1, 7, 9, 20, 33, 72, 99])
    q, j_star, clean_logits = channel_margin_delta(receiver, donor, target, model)
    donor_logits = model(donor)
    rows = torch.arange(len(target))
    exact_delta = (
        donor_logits[rows, target] - donor_logits[rows, j_star]
        - clean_logits[rows, target] + clean_logits[rows, j_star]
    )
    assert torch.allclose(q.sum(dim=1), exact_delta, atol=2e-5, rtol=2e-5)
    print("PASS: channel contributions sum to exact fixed-competitor margin delta")


def test_protection_preserves_bottom48_and_replacement_count():
    torch.manual_seed(17)
    b = 11
    candidates = torch.zeros(b, 64, dtype=torch.bool)
    for i in range(b):
        # Different Bottom48 sets per sample.
        candidates[i, torch.randperm(64)[:48]] = True
    q = torch.randn(b, 64)
    gumbel = torch.randn(b, 64)
    selected, protected = protected_lowvs_mask(candidates, q, gumbel)
    assert torch.all(selected.sum(1) == 16)
    assert torch.all(protected.sum(1) == 16)
    assert not torch.any(selected & protected)
    assert not torch.any(selected & ~candidates)
    assert not torch.any(protected & ~candidates)
    for i in range(b):
        cand = torch.where(candidates[i])[0]
        expected = cand[torch.argsort(q[i, cand], stable=True)[:16]]
        assert torch.equal(torch.where(protected[i])[0], expected)
    print("PASS: protected mask is paired, fixed-size, and remains inside Bottom48")


if __name__ == "__main__":
    test_frequency_strata_and_common_random_numbers()
    test_prototypes_and_donor_indices_are_train_label_consistent()
    test_channel_delta_is_exact_for_fixed_competitor()
    test_protection_preserves_bottom48_and_replacement_count()
    print("ALL POLICY TESTS PASSED")
