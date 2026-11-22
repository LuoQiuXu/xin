import numpy as np
import torch
import torch.nn.functional as F

from BCL_AUDIT.cifar100lt.data import (
    cifar100lt_class_counts,
    make_cifar100lt_indices,
)
from BCL_AUDIT.cifar100lt.losses import BalancedContrastiveLoss, LogitCompensatedCE
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel


def test_if100_count_vector_and_indices():
    counts = cifar100lt_class_counts()
    assert counts.shape == (100,)
    assert counts[0] == 500
    assert counts[-1] == 5
    assert int(counts.sum()) == 10_847

    labels = np.repeat(np.arange(100, dtype=np.int64), 500)
    counts_from_split, indices = make_cifar100lt_indices(labels, split_seed=0)
    assert np.array_equal(counts, counts_from_split)
    assert indices.shape == (10_847,)
    assert np.array_equal(np.bincount(labels[indices], minlength=100), counts)


def test_resnet32_and_bcl_forward_shapes():
    model = BCLCIFARModel()
    model.eval()
    images = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        features, logits, embeddings, prototypes = model(
            torch.cat([images, images, images], dim=0),
            compute_contrastive=True,
        )
    assert features.shape == (12, 64)
    assert logits.shape == (12, 100)
    assert embeddings.shape == (12, 128)
    assert prototypes.shape == (100, 128)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(12), atol=1e-5)
    assert torch.allclose(prototypes.norm(dim=1), torch.ones(100), atol=1e-5)


def test_bcl_loss_is_finite_and_differentiable_with_all_prototypes():
    counts = cifar100lt_class_counts()
    criterion = BalancedContrastiveLoss(counts, temperature=0.1)

    feature_raw = torch.randn(6, 2, 128, requires_grad=True)
    prototype_raw = torch.randn(100, 128, requires_grad=True)
    features = F.normalize(feature_raw, dim=-1)
    prototypes = F.normalize(prototype_raw, dim=-1)
    targets = torch.tensor([0, 0, 1, 2, 2, 4], dtype=torch.long)

    loss = criterion(prototypes, features, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert feature_raw.grad is not None
    assert prototype_raw.grad is not None
    assert torch.isfinite(feature_raw.grad).all()
    assert torch.isfinite(prototype_raw.grad).all()
    # All class prototypes are in the class-complement denominator.
    assert torch.count_nonzero(prototype_raw.grad.norm(dim=1)).item() == 100


def test_gradient_routes_match_the_declared_computation_graph():
    counts = cifar100lt_class_counts()
    model = BCLCIFARModel()
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 0, 1, 2], dtype=torch.long)

    # LC path: Backbone and W receive gradients; projection MLPs do not.
    _, logits, _, _ = model(images, compute_contrastive=False)
    lc = LogitCompensatedCE(counts)(logits, labels)
    lc_grads = torch.autograd.grad(
        lc,
        [
            model.backbone.conv1.weight,
            model.classifier.weight,
            model.sample_mlp.net[0].weight,
            model.prototype_mlp.net[0].weight,
        ],
        allow_unused=True,
    )
    assert lc_grads[0] is not None and lc_grads[0].abs().sum() > 0
    assert lc_grads[1] is not None and lc_grads[1].abs().sum() > 0
    assert lc_grads[2] is None
    assert lc_grads[3] is None

    # BCL path: Backbone and W receive gradients through samples / prototypes;
    # both MLPs are BCL-only modules.
    all_views = torch.cat([images, images + 0.05, images - 0.05], dim=0)
    _, all_logits, all_embeddings, prototypes = model(
        all_views,
        compute_contrastive=True,
    )
    batch_size = labels.shape[0]
    z2 = all_embeddings[batch_size : 2 * batch_size]
    z3 = all_embeddings[2 * batch_size :]
    two_views = torch.stack([z2, z3], dim=1)
    bcl = BalancedContrastiveLoss(counts, temperature=0.1)(
        prototypes, two_views, labels
    )
    bcl_grads = torch.autograd.grad(
        bcl,
        [
            model.backbone.conv1.weight,
            model.classifier.weight,
            model.sample_mlp.net[0].weight,
            model.prototype_mlp.net[0].weight,
        ],
        allow_unused=True,
    )
    for gradient in bcl_grads:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
