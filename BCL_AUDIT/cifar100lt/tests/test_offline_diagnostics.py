import numpy as np
import torch

from BCL_AUDIT.cifar100lt.data import cifar100lt_class_counts
from BCL_AUDIT.cifar100lt.losses import BalancedContrastiveLoss, LogitCompensatedCE
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_AUDIT.scripts.analyze_paired_bcl import (
    compute_bcl_gradient_rows,
    compute_native_three_view_gradient_diagnostics,
    compute_test_margin_rows,
)


def _class_meta():
    return {
        class_id: {
            "coarse_id": str(class_id // 5),
            "shot_group": "many" if class_id < 35 else ("medium" if class_id < 70 else "few"),
        }
        for class_id in range(100)
    }


def test_test_margin_is_read_only_and_class_conditional():
    labels = np.arange(100, dtype=np.int64)
    logits = np.zeros((100, 100), dtype=np.float32)
    logits[np.arange(100), labels] = 2.0
    adjusted = logits.copy()
    adjusted[np.arange(100), labels] = 3.0

    rows = compute_test_margin_rows(
        group="BCL",
        labels=labels,
        logits=logits,
        adjusted_logits=adjusted,
        class_meta=_class_meta(),
    )
    assert len(rows) == 100
    assert rows[0]["source_split"] == "balanced_TEST_read_only"
    assert rows[0]["raw_margin_mean"] == 2.0
    assert rows[0]["adjusted_margin_mean"] == 3.0
    assert rows[99]["test_count"] == 1


def test_common_input_eval_gradient_probe_reports_objective_weighted_conflicts():
    torch.manual_seed(321)
    model = BCLCIFARModel()
    model.eval()
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    probe = {
        "view1": torch.randn(4, 3, 32, 32),
        "view2": torch.randn(4, 3, 32, 32),
        "view3": torch.randn(4, 3, 32, 32),
        "labels": labels,
        "metadata": {
            "source_split": "original CIFAR-100 training split, fixed LT subset",
            "test_split_loaded": False,
        },
    }
    bn_before = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }

    rows = compute_bcl_gradient_rows(
        model=model,
        epoch=40,
        probe=probe,
        class_meta=_class_meta(),
        criterion_lc=LogitCompensatedCE(cifar100lt_class_counts()),
        criterion_bcl=BalancedContrastiveLoss(cifar100lt_class_counts(), temperature=0.1),
        device=torch.device("cpu"),
        batch_size=4,
    )

    assert len(rows) == 100
    observed = [row for row in rows if row["class_id"] in (0, 1)]
    assert all(row["probe_mode"] == "common_input_eval" for row in observed)
    assert all(row["bn_mode"] == "eval" for row in observed)
    assert all(np.isfinite(row["W_y_gradient_cosine"]) for row in observed)
    for row in observed:
        raw_ratio = row["W_y_BCL_to_LC_norm_ratio"]
        weighted_ratio = row["W_y_objective_weighted_BCL_to_LC_norm_ratio"]
        assert np.isclose(weighted_ratio, raw_ratio * (0.6 / 2.0), rtol=1e-5, atol=1e-7)
    bn_after = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    for name in bn_before:
        assert torch.equal(bn_before[name], bn_after[name])


def test_native_three_view_train_probe_reports_gradients_and_restores_bn():
    torch.manual_seed(123)
    model = BCLCIFARModel()
    model.eval()
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    probe = {
        "view1": torch.randn(4, 3, 32, 32),
        "view2": torch.randn(4, 3, 32, 32),
        "view3": torch.randn(4, 3, 32, 32),
        "labels": labels,
        "metadata": {
            "source_split": "original CIFAR-100 training split, fixed LT subset",
            "test_split_loaded": False,
        },
    }
    bn_before = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }

    class_rows, global_row = compute_native_three_view_gradient_diagnostics(
        model=model,
        epoch=40,
        probe=probe,
        class_meta=_class_meta(),
        criterion_lc=LogitCompensatedCE(cifar100lt_class_counts()),
        criterion_bcl=BalancedContrastiveLoss(cifar100lt_class_counts(), temperature=0.1),
        device=torch.device("cpu"),
        batch_size=4,
    )

    assert len(class_rows) == 2
    assert global_row["probe_mode"] == "native_three_view_train"
    assert global_row["bn_running_buffers_restored"] is True
    assert np.isfinite(global_row["backbone_gradient_cosine"])
    assert global_row["lc_objective_weight"] == 2.0
    assert global_row["bcl_objective_weight"] == 0.6
    raw_ratio = global_row["backbone_BCL_to_LC_norm_ratio"]
    weighted_ratio = global_row["backbone_objective_weighted_BCL_to_LC_norm_ratio"]
    assert np.isclose(weighted_ratio, raw_ratio * (0.6 / 2.0), rtol=1e-5, atol=1e-7)
    assert all(np.isfinite(row["W_y_gradient_cosine"]) for row in class_rows)
    assert all("W_y_objective_weighted_BCL_to_LC_norm_ratio" in row for row in class_rows)
    bn_after = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    assert bn_before.keys() == bn_after.keys()
    for name in bn_before:
        assert torch.equal(bn_before[name], bn_after[name])
    assert model.training is False
