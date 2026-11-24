import hashlib
import pickle

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from BCL_AUDIT.cifar100lt.losses import BalancedContrastiveLoss, LogitCompensatedCE
from BCL_LTN import train_ntc_bcl as training
from BCL_LTN.noise_robust_losses import (
    ForwardCorrectedLogitCompensation,
    NoiseAwareSoftBalancedContrastiveLoss,
    estimate_clean_class_counts,
    soft_targets_from_three_view_logits,
    symmetric_transition_matrix,
)
from BCL_LTN.train_noisy_bcl import OfficialTestAudit


def feature_pairs(batch, classes, dtype):
    generator = torch.Generator().manual_seed(83)
    samples = F.normalize(torch.randn(batch, 2, 7, dtype=dtype, generator=generator), dim=-1)
    prototypes = F.normalize(torch.randn(classes, 7, dtype=dtype, generator=generator), dim=-1)
    return samples.requires_grad_(), prototypes.requires_grad_()


def test_zero_noise_forward_correction_matches_original_lc_value_and_gradient():
    counts = np.array([11.0, 7.0, 2.0])
    logits = torch.tensor([[3.0, -2.0, 4.0], [0.2, 3.2, -0.1]], dtype=torch.float64, requires_grad=True)
    observed = torch.tensor([0, 2])
    expected = LogitCompensatedCE(counts)(logits, observed)
    actual = ForwardCorrectedLogitCompensation(counts, noise_rate=0)(logits, observed)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    want = torch.autograd.grad(expected, logits)[0]
    got = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)


def test_transition_orientation_and_log_space_selected_target_loss():
    matrix = symmetric_transition_matrix(100, 0.4, dtype=torch.float64)
    off_diagonal = ~torch.eye(100, dtype=torch.bool)
    torch.testing.assert_close(matrix.sum(1), torch.ones(100, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(matrix.diag(), torch.full((100,), 0.6, dtype=torch.float64), atol=1e-15, rtol=0)
    torch.testing.assert_close(matrix[off_diagonal], torch.full((9900,), 0.4 / 99, dtype=torch.float64), atol=1e-15, rtol=0)

    loss = ForwardCorrectedLogitCompensation(np.arange(1, 101))
    logits = torch.zeros((2, 100), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        logits[0, 0], logits[1, 99] = 10_000, 10_000
    observed = torch.tensor([99, 99])
    expected = loss.clean_probabilities(logits) @ loss.transition.to(logits)
    torch.testing.assert_close(loss.observed_log_probabilities(logits).exp(), expected, atol=1e-12, rtol=1e-12)
    value = loss(logits, observed)
    torch.testing.assert_close(value, -expected[torch.arange(2), observed].log().mean(), atol=1e-12, rtol=1e-12)
    assert {"log_prior", "transition"} <= dict(loss.named_buffers()).keys()
    value.backward()
    assert torch.isfinite(logits.grad).all()


def test_prior_uses_unit_floor_then_normalizes_without_if100_constraint():
    observed = np.array([0, 1, 4, 30, 100], dtype=np.int64)
    raw, corrected = estimate_clean_class_counts(observed, noise_rate=0.4, epsilon=1.0)
    expected_raw = (observed - 0.4 * observed.sum() / 4) / (0.6 - 0.4 / 4)
    floored = np.maximum(expected_raw, 1.0)
    np.testing.assert_allclose(raw, expected_raw, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(corrected, floored * (observed.sum() / floored.sum()), rtol=1e-14, atol=1e-14)
    assert 0.0 < corrected.min() < 1.0
    assert corrected.sum() == pytest.approx(observed.sum(), abs=1e-12)


def test_prior_inverts_known_observed_mass_without_floor_activating():
    expected = np.array([110.0, 64.0, 22.0, 4.0])
    observed = expected @ symmetric_transition_matrix(4, 0.4, dtype=torch.float64).numpy()
    raw, corrected = estimate_clean_class_counts(observed)
    np.testing.assert_allclose(raw, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(corrected, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("observed", [[0, 0, 0], [4, np.nan, 3], [4, -1, 3]])
def test_prior_rejects_invalid_observed_counts(observed):
    with pytest.raises(ValueError):
        estimate_clean_class_counts(np.array(observed))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("labels", [[2], [2, 0, 2, 4, 0]])
def test_one_hot_soft_bcl_matches_original_bcl_relations_loss_and_gradients(dtype, labels):
    classes = 100
    targets = torch.tensor(labels)
    views, prototypes = feature_pairs(len(labels), classes, dtype)
    q = F.one_hot(targets, classes).to(dtype)
    soft = NoiseAwareSoftBalancedContrastiveLoss(classes)
    positives, mask, denominator_weights, counts = soft._semantic_weights(q, views)
    candidate_classes = torch.cat((targets, targets, torch.arange(classes)))
    want_counts = torch.bincount(candidate_classes, minlength=classes).to(dtype)
    want_mask = torch.ones_like(mask)
    want_mask.fill_diagonal_(0)
    want_positives = (candidate_classes[:2 * len(labels), None] == candidate_classes[None, :]).to(dtype) * want_mask
    want_weights = want_counts[candidate_classes][None, :] - want_positives
    for got, want in ((mask, want_mask), (positives, want_positives), (counts, want_counts), (denominator_weights, want_weights)):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    bcl = BalancedContrastiveLoss(np.ones(classes))
    expected = bcl(prototypes, views, targets)
    actual = soft(prototypes, views, q)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    old_grads = torch.autograd.grad(expected, (views, prototypes))
    new_grads = torch.autograd.grad(actual, (views, prototypes))
    for got, want in zip(new_grads, old_grads):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("classes", [3, 100])
def test_uniform_detached_q_has_only_self_view_positives_and_finite_loss(classes):
    batch = 3
    views, prototypes = feature_pairs(batch, classes, torch.float32)
    q = torch.full((batch, classes), 1 / classes, requires_grad=True)
    soft = NoiseAwareSoftBalancedContrastiveLoss(classes)
    weights, mask, denominator, _ = soft._semantic_weights(q, views)
    expected = torch.zeros_like(weights)
    expected[torch.arange(2 * batch), (torch.arange(2 * batch) + batch) % (2 * batch)] = 1.0
    assert torch.equal(weights, expected)
    assert torch.all(denominator[mask.bool()] > 0)
    loss = soft(prototypes, views, q)
    assert loss.isfinite()
    grad_view, grad_proto, grad_q = torch.autograd.grad(loss, (views, prototypes, q), allow_unused=True)
    assert grad_q is None
    assert torch.isfinite(grad_view).all() and torch.isfinite(grad_proto).all()


@pytest.mark.parametrize("batch", [1, 4])
def test_instance_only_warmup_matches_nt_xent_and_does_not_consult_q_or_prototypes(batch):
    views, _ = feature_pairs(batch, classes=5, dtype=torch.float64)
    flat = F.normalize(torch.cat((views[:, 0], views[:, 1]), dim=0), dim=1)
    rows = []
    for i, anchor in enumerate(flat):
        logits = (anchor * flat).sum(1) / 0.1
        nonself = torch.arange(len(flat)) != i
        positive = logits[(i + batch) % len(flat)]
        rows.append(torch.logsumexp(logits[nonself], dim=0) - positive)
    expected = torch.stack(rows).mean()
    got = NoiseAwareSoftBalancedContrastiveLoss(5)(
        object(), views, object(), semantic_enabled=False,
    )
    torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)
    got.backward()
    assert torch.isfinite(views.grad).all()


def test_three_view_soft_targets_average_raw_logits_before_softmax_and_detach():
    raw = torch.tensor([
        [4.0, 0.0, 0.0], [-10000.0, 10000.0, 0.0],
        [1.0, 2.0, 0.0], [-10000.0, 10000.0, 0.0],
        [2.0, 0.0, 1.0], [-10000.0, 10000.0, 0.0],
    ], dtype=torch.float64, requires_grad=True)
    q = soft_targets_from_three_view_logits(raw, 2, temperature=1.0)
    expected = ((raw[:2] + raw[2:4] + raw[4:6]) / 3).softmax(1)
    torch.testing.assert_close(q, expected, atol=1e-12, rtol=1e-12)
    assert q.grad_fn is None and not q.requires_grad
    assert q[1].tolist() == [0.0, 1.0, 0.0]
    for counts in ([1, 1, 1], [99, 1, 1]):
        lc = ForwardCorrectedLogitCompensation(np.array(counts))
        assert torch.equal(q, soft_targets_from_three_view_logits(raw, 2))
        if counts[0] == 99:
            assert not torch.equal(q[0], (raw[:2] + lc.log_prior)[0].softmax(0))
    views, prototypes = feature_pairs(batch=2, classes=3, dtype=torch.float64)
    # An attached q passed by a caller must still be detached by the loss.
    loss = NoiseAwareSoftBalancedContrastiveLoss(3)(prototypes, views, expected)
    assert torch.autograd.grad(loss, raw, allow_unused=True)[0] is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 0.0, -1.0])
def test_nonfinite_or_nonpositive_temperature_rejected(invalid):
    with pytest.raises(ValueError, match="temperature"):
        soft_targets_from_three_view_logits(torch.zeros(6, 3), 2, temperature=invalid)
    with pytest.raises(ValueError, match="temperature"):
        NoiseAwareSoftBalancedContrastiveLoss(3, temperature=invalid)


@pytest.mark.parametrize("unsupported", ["features", "prototypes", "q", "q_zero"])
def test_semantic_loss_rejects_nonfinite_or_missing_mass(unsupported):
    views, prototypes = feature_pairs(2, 3, torch.float32)
    q = torch.full((2, 3), 1 / 3)
    with torch.no_grad():
        if unsupported == "features":
            views[0, 0, 0] = torch.nan
        elif unsupported == "prototypes":
            prototypes[0, 0] = torch.inf
        elif unsupported == "q":
            q[0, 0] = torch.nan
        else:
            q[0] = 0
    with pytest.raises(ValueError):
        NoiseAwareSoftBalancedContrastiveLoss(3)(prototypes, views, q)


def test_forward_corrected_loss_rejects_nonfinite_logits_and_noninteger_labels():
    loss = ForwardCorrectedLogitCompensation(np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        loss(torch.tensor([[0.0, torch.nan, 0.0]]), torch.tensor([1]))
    with pytest.raises(ValueError, match="integer"):
        loss(torch.zeros(1, 3), torch.tensor([1.5]))
    with pytest.raises(ValueError, match="finite"):
        soft_targets_from_three_view_logits(torch.full((3, 3), torch.nan), 1)


class TracedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 6)
        self.classifier = nn.Linear(6, 5)
        self.sample_mlp = nn.Linear(6, 4)
        self.prototype_mlp = nn.Linear(6, 4)
        self.inputs = []

    def forward(self, views, compute_contrastive=True):
        self.inputs.append((views.detach().clone(), compute_contrastive))
        features = self.backbone(views)
        logits = self.classifier(features)
        if not compute_contrastive:
            return features, logits, None, None
        return features, logits, self.sample_mlp(features), self.prototype_mlp(self.classifier.weight)


@pytest.mark.parametrize(("epoch_index", "semantic"), [(0, False), (19, False), (20, True), (199, True)])
def test_training_switches_to_semantics_at_epoch_index_20(epoch_index, semantic, monkeypatch):
    model = TracedModel()
    calls = []
    model.prototype_mlp.register_forward_hook(lambda *_: calls.append("prototype"))
    model.sample_mlp.register_forward_hook(lambda *_: calls.append("sample"))
    captured = []
    real_q = training.soft_targets_from_three_view_logits

    def trace_q(raw, batch_size, *, temperature):
        target = real_q(raw, batch_size, temperature=temperature)
        captured.append((raw.detach().clone(), target, temperature))
        return target

    monkeypatch.setattr(training, "soft_targets_from_three_view_logits", trace_q)
    views = [torch.randn(3, 3) + float(view_id) for view_id in range(3)]
    observed = torch.tensor([0, 1, 0])
    criterion = ForwardCorrectedLogitCompensation(np.array([500.0, 20.0, 10.0, 2.0, 1.0]))
    soft = NoiseAwareSoftBalancedContrastiveLoss(5)
    lc, bcl, raw = training.three_view_objective(model, training.ARM_C, epoch_index, views, observed, criterion, soft)
    assert len(model.inputs) == 1
    torch.testing.assert_close(model.inputs[0][0], torch.cat(views), atol=0, rtol=0)
    assert model.inputs[0][1] is semantic
    assert calls == (["sample", "prototype"] if semantic else ["sample"])
    assert len(captured) == int(semantic)
    if semantic:
        traced_raw, q, temperature = captured[0]
        assert temperature == 1.0
        torch.testing.assert_close(traced_raw, raw, atol=0, rtol=0)
        torch.testing.assert_close(q, raw.reshape(3, 3, 5).mean(0).softmax(1), atol=0, rtol=0)
        assert not q.requires_grad
    assert torch.autograd.grad(bcl, raw, allow_unused=True, retain_graph=True)[0] is None
    (2.0 * lc + 0.6 * bcl).backward()
    for name, parameter in model.named_parameters():
        if name.startswith("prototype_mlp.") and not semantic:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_semantic_bcl_ignores_observed_labels_and_lc_prior():
    model = TracedModel()
    views = [torch.randn(3, 3) for _ in range(3)]
    bcl = NoiseAwareSoftBalancedContrastiveLoss(5)
    _, first, _ = training.three_view_objective(
        model, training.ARM_C, 20, views, torch.tensor([0, 1, 0]),
        ForwardCorrectedLogitCompensation(np.ones(5)), bcl,
    )
    _, second, _ = training.three_view_objective(
        model, training.ARM_C, 20, views, torch.tensor([4, 2, 3]),
        ForwardCorrectedLogitCompensation(np.array([100., 50., 25., 10., 1.])), bcl,
    )
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert {training.bcl_mode(training.ARM_B, p) for p in (0, 19, 20, 199)} == {"hard_observed"}


def test_single_open_test_reader_with_synthetic_fixture(tmp_path, monkeypatch):
    split = tmp_path / "cifar-100-python"
    split.mkdir()
    targets = np.repeat(np.arange(100), 100)
    payload = pickle.dumps({"data": np.zeros((10_000, 3072), dtype=np.uint8), "fine_labels": targets.tolist()})
    (split / "test").write_bytes(payload)
    monkeypatch.setattr(training, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(training.torchvision.datasets.CIFAR100, "test_list", [["test", hashlib.md5(payload).hexdigest()]])
    guard = OfficialTestAudit(tmp_path)
    assert guard.opens == 0
    with guard.final_evaluation():
        pixels, read_targets, sha256 = training.official_test_one_open()
    assert pixels.shape == (10_000, 32, 32, 3)
    np.testing.assert_array_equal(targets, read_targets)
    assert sha256 == hashlib.sha256(payload).hexdigest()
    assert guard.opens == 1 and guard.denied == 0
