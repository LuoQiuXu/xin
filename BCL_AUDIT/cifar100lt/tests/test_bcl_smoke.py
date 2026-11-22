import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from BCL_AUDIT.cifar100lt import data as data_module
from BCL_AUDIT.cifar100lt.data import (
    CIFAR100LT,
    CIFAR100TrainOnly,
    cifar100lt_class_counts,
    make_cifar100lt_indices,
)
from BCL_AUDIT.cifar100lt.train import BatchNormAudit
from BCL_AUDIT.scripts.analyze_paired_bcl import summarize_did
from BCL_AUDIT.scripts.audit_cifar100lt_topology import _load_existing_static_audit
from BCL_AUDIT.cifar100lt.losses import BalancedContrastiveLoss, LogitCompensatedCE
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel


def test_cifar100_train_only_integrity_never_checks_test_files(monkeypatch, tmp_path):
    loader = CIFAR100TrainOnly.__new__(CIFAR100TrainOnly)
    loader.root = str(tmp_path)
    loader.base_folder = "cifar-100-python"
    loader.train_list = [("train", "train-md5")]
    loader.test_list = [("test", "test-md5")]

    checked_files = []

    def fake_check_integrity(path, md5):
        checked_files.append(Path(path).name)
        return True

    monkeypatch.setattr(data_module, "check_integrity", fake_check_integrity)
    assert loader._check_integrity() is True
    assert checked_files == ["train"]


def test_reuse_static_audit_loads_without_rewriting(tmp_path):
    counts = cifar100lt_class_counts()
    selected_indices = np.arange(int(counts.sum()), dtype=np.int64)
    np.save(tmp_path / "cifar100lt_if100_train_indices_seed0.npy", selected_indices)
    (tmp_path / "class_counts.csv").write_text("placeholder\n", encoding="utf-8")
    (tmp_path / "superclass_counts.csv").write_text("placeholder\n", encoding="utf-8")
    pair_path = tmp_path / "head_tail_pair_topology.csv"
    pair_path.write_text(
        "head_fine_id,tail_fine_id,head_coarse_id,tail_coarse_id,"
        "same_superclass,dino_cosine_similarity,dino_topk_union\n"
        "0,99,0,19,False,,\n",
        encoding="utf-8",
    )
    summary = {
        "source_split": "original balanced CIFAR-100 training split only",
        "test_split_loaded": False,
        "split_seed": 0,
        "imbalance_factor": 0.01,
        "class_counts": counts.tolist(),
        "long_tail_train_size": int(counts.sum()),
        "total_many_few_pairs": 1,
        "dino_similarity_computed": False,
    }
    summary_path = tmp_path / "data_audit_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    before = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }
    loaded_summary, pair_rows = _load_existing_static_audit(
        tmp_path, counts, selected_indices, split_seed=0, imb_factor=0.01
    )
    after = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }

    assert loaded_summary == summary
    assert pair_rows[0]["same_superclass"] is False
    assert before == after


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
    all_views = all_views.detach().requires_grad_(True)
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
            all_views,
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

    # Train-mode BatchNorm2d / BatchNorm1d couple all three views. Although
    # BCL directly uses z2/z3, its gradient can reach the v1 input rows.
    assert bcl_grads[0][:batch_size].abs().sum() > 0



def test_lc_only_and_bcl_share_three_view_dataset_contract():
    for mode in ("lc", "bcl"):
        dataset = CIFAR100LT.__new__(CIFAR100LT)
        dataset.mode = mode
        dataset.indices = np.asarray([0], dtype=np.int64)
        dataset.data = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        dataset.targets = np.asarray([7], dtype=np.int64)
        dataset.transform_v1 = lambda image: torch.zeros(3, 32, 32)
        dataset.transform_sim = lambda image: torch.ones(3, 32, 32)

        views, label = dataset[0]
        assert len(views) == 3
        assert views[0].shape == (3, 32, 32)
        assert views[1].shape == (3, 32, 32)
        assert views[2].shape == (3, 32, 32)
        assert label == 7


def test_batchnorm_audit_captures_backbone_and_bcl_sample_projection_views():
    model = BCLCIFARModel()
    audit = BatchNormAudit(model)
    audit.arm()
    images = torch.randn(6, 3, 32, 32)  # three views, two examples per view
    with torch.no_grad():
        model(images, compute_contrastive=True)
    audit.finish_batch()
    record = audit.snapshot("BCL", 40)
    audit.close()

    first_batch = record["first_minibatch_per_view_input_stats"]
    assert "backbone.bn1" in first_batch
    assert "sample_mlp.net.1" in first_batch
    assert set(first_batch["backbone.bn1"]["views"]) == {"v1", "v2", "v3"}
    assert set(first_batch["sample_mlp.net.1"]["views"]) == {"v1", "v2", "v3"}


def test_head_tail_did_is_paired_and_direction_specific():
    effects = [
        {
            "direction": "Head_to_Tail",
            "same_superclass": True,
            "dino_topk_union": True,
            "delta_confusion_BCL_minus_LC": 0.20,
            "delta_raw_margin_BCL_minus_LC": -0.10,
            "delta_adjusted_margin_BCL_minus_LC": -0.08,
        },
        {
            "direction": "Head_to_Tail",
            "same_superclass": False,
            "dino_topk_union": False,
            "delta_confusion_BCL_minus_LC": 0.05,
            "delta_raw_margin_BCL_minus_LC": -0.02,
            "delta_adjusted_margin_BCL_minus_LC": -0.01,
        },
        {
            "direction": "Tail_to_Head",
            "same_superclass": True,
            "dino_topk_union": False,
            "delta_confusion_BCL_minus_LC": 0.12,
            "delta_raw_margin_BCL_minus_LC": -0.04,
            "delta_adjusted_margin_BCL_minus_LC": -0.03,
        },
        {
            "direction": "Tail_to_Head",
            "same_superclass": False,
            "dino_topk_union": True,
            "delta_confusion_BCL_minus_LC": 0.02,
            "delta_raw_margin_BCL_minus_LC": 0.01,
            "delta_adjusted_margin_BCL_minus_LC": 0.02,
        },
    ]
    rows = summarize_did(effects)
    head_row = next(
        row for row in rows
        if row["relation"] == "same_superclass"
        and row["direction"] == "Head_to_Tail"
        and row["metric"] == "delta_confusion_BCL_minus_LC"
        and row["slice"] == "SameSuperclass"
    )
    assert abs(head_row["DID_positive_minus_negative"] - 0.15) < 1e-12

    tail_row = next(
        row for row in rows
        if row["relation"] == "same_superclass"
        and row["direction"] == "Tail_to_Head"
        and row["metric"] == "delta_confusion_BCL_minus_LC"
        and row["slice"] == "SameSuperclass"
    )
    assert abs(tail_row["DID_positive_minus_negative"] - 0.10) < 1e-12

def _official_balscl_reference(
    class_prototypes: torch.Tensor,
    two_view_features: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Numerical reference matching the released BCL BalSCL source formula."""
    batch_size = int(two_view_features.shape[0])
    num_classes = int(class_prototypes.shape[0])
    device = two_view_features.device

    sample_targets = targets.to(device=device, dtype=torch.long).reshape(-1, 1)
    prototype_targets = torch.arange(num_classes, device=device).reshape(-1, 1)
    all_targets = torch.cat(
        [sample_targets.repeat(2, 1), prototype_targets], dim=0
    ).reshape(-1)

    class_counts_in_batch = torch.zeros(
        num_classes, device=device, dtype=torch.float32
    )
    class_counts_in_batch.scatter_add_(
        0, all_targets, torch.ones_like(all_targets, dtype=torch.float32)
    )

    positive_mask = (
        all_targets[: 2 * batch_size, None] == all_targets[None, :]
    ).to(dtype=two_view_features.dtype)
    logits_mask = torch.ones_like(positive_mask)
    logits_mask.scatter_(
        1,
        torch.arange(2 * batch_size, device=device).reshape(-1, 1),
        0,
    )
    positive_mask = positive_mask * logits_mask

    sample_features = torch.cat(
        [two_view_features[:, 0], two_view_features[:, 1]], dim=0
    )
    all_features = torch.cat(
        [sample_features, class_prototypes.to(device=device, dtype=two_view_features.dtype)],
        dim=0,
    )
    logits = all_features[: 2 * batch_size] @ all_features.T
    logits = logits / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    exp_logits = torch.exp(logits) * logits_mask
    per_entry_weight = class_counts_in_batch[all_targets].to(
        dtype=two_view_features.dtype
    ).reshape(1, -1).expand_as(exp_logits) - positive_mask
    denominator = (exp_logits / per_entry_weight).sum(dim=1, keepdim=True)
    log_prob = logits - torch.log(denominator)
    mean_log_prob_positive = (
        positive_mask * log_prob
    ).sum(dim=1) / positive_mask.sum(dim=1)
    return -mean_log_prob_positive.reshape(2, batch_size).mean()


def test_bcl_loss_matches_released_official_balscl_formula():
    torch.manual_seed(20261122)
    counts = cifar100lt_class_counts()
    criterion = BalancedContrastiveLoss(counts, temperature=0.1)

    features = F.normalize(torch.randn(9, 2, 128), dim=-1)
    prototypes = F.normalize(torch.randn(100, 128), dim=-1)
    targets = torch.tensor([0, 0, 1, 2, 2, 4, 7, 9, 9], dtype=torch.long)

    actual = criterion(prototypes, features, targets)
    reference = _official_balscl_reference(
        prototypes, features, targets, temperature=0.1
    )
    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-6)

