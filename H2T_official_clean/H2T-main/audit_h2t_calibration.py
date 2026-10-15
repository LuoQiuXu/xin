"""Paired accuracy--calibration audit for the published H2T Stage-II recipe.

This is deliberately an audit harness, not a new H2T variant.  It imports the
released H2T operator from ``methods.py`` and keeps the published CIFAR-100-LT
Stage-II recipe fixed.  The only training intervention is the number of
feature channels replaced by that operator:

* ``rho=0``: a *matched no-fusion control*.  The donor image is still sampled,
  transformed, passed through the frozen backbone and passed to ``H2T``.  Thus
  it also preserves dual sampling, random-channel-mask RNG consumption and BN
  updates.  No channel is replaced.
* ``rho=0.25``: the released H2T setting.

The test endpoint is the fixed final Stage-II epoch (10), never a checkpoint
chosen on test accuracy.  Clean-test calibration is primary.  A separately
labelled held-out synthetic fusion probe is auxiliary and must not be confused
with ordinary test calibration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets.cifar100 import CIFAR100_LT
from methods import H2T
from models import resnet_cifar


NUM_CLASSES = 100
GROUPS: Mapping[str, Tuple[int, int]] = {
    "Many": (0, 36),
    "Medium": (36, 71),
    "Few": (71, 100),
}


def default_workspace_root() -> Path:
    # .../workspace/H2T_official_clean/H2T-main/audit_h2t_calibration.py
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    workspace = default_workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=workspace / "H2T" / "data" / "cifar100",
        help="Existing CIFAR-100 root used by the local H2T experiments.",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=here / "pretrained" / "cifar100_imb001_stage1_official.pth.tar",
        help="Released H2T CIFAR-100-LT IF=100 Stage-I checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "outputs" / "calibration_audit_official_h2t_seed42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("nofusion", "h2t"),
        default=("nofusion", "h2t"),
        help="Run one or both paired Stage-II arms.",
    )
    parser.add_argument(
        "--skip-fusion-probe",
        action="store_true",
        help="Skip the auxiliary held-out synthetic fusion probe.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify data, checkpoint and the released H2T operator on one batch without training.",
    )
    return parser.parse_args()


def set_paired_seed(seed: int) -> None:
    """Make the two arms replay the same host-side data/RNG stream."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # This is a pairing safeguard; it is not a model or H2T-method change.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_dataset_config() -> SimpleNamespace:
    """Only the fields read by the released CIFAR100_LT implementation."""
    return SimpleNamespace(
        sampler=SimpleNamespace(
            type="weighted sampler",
            weighted_sampler=SimpleNamespace(type="balance"),
            dual_sample=SimpleNamespace(enable=True, type="long-tailed"),
        )
    )


def build_loaders(args: argparse.Namespace) -> Tuple[CIFAR100_LT, DataLoader, DataLoader]:
    dataset = CIFAR100_LT(
        make_dataset_config(),
        distributed=False,
        root=str(args.data_root),
        imb_factor=0.01,
        batch_size=args.batch_size,
        num_works=args.workers,
    )
    # These are the exact published Stage-II loader choices: train_instance
    # plus its internal weighted-recipient / long-tailed-donor mechanism.
    return dataset, dataset.train_instance, dataset.eval


def load_released_stage1(checkpoint_path: Path) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage-I checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"state_dict_model", "state_dict_classifier", "epoch", "best_acc1", "its_ece"}
    absent = required.difference(checkpoint)
    if absent:
        raise KeyError(f"Checkpoint does not have released Stage-I fields: {sorted(absent)}")

    model = torch.nn.DataParallel(resnet_cifar.resnet32_fe()).cuda()
    classifier = torch.nn.DataParallel(
        resnet_cifar.Classifier(feat_in=64, num_classes=NUM_CLASSES)
    ).cuda()
    model.load_state_dict(checkpoint["state_dict_model"], strict=True)
    classifier.load_state_dict(checkpoint["state_dict_classifier"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, classifier, checkpoint


def stage2_lr(epoch: int, args: argparse.Namespace) -> float:
    # Exact cosine schedule in the released train_stage2.py, including the
    # classifier learning-rate factor used by the official config.
    base = 0.5 * args.lr * (1.0 + math.cos(epoch / args.epochs * math.pi))
    return args.lr_factor * base


def group_name(label: np.ndarray) -> np.ndarray:
    groups = np.empty(label.shape[0], dtype=object)
    groups[(label >= 0) & (label < 36)] = "Many"
    groups[(label >= 36) & (label < 71)] = "Medium"
    groups[(label >= 71) & (label < 100)] = "Few"
    return groups


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, bins: int
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lo, hi = float(edges[index]), float(edges[index + 1])
        if index == 0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence > lo) & (confidence <= hi)
        count = int(mask.sum())
        if count:
            mean_conf = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            gap = mean_conf - accuracy
        else:
            mean_conf = None
            accuracy = None
            gap = None
        rows.append(
            {
                "bin": index + 1,
                "lower": lo,
                "upper": hi,
                "count": count,
                "confidence": mean_conf,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    return rows


def ece_from_arrays(confidence: np.ndarray, correct: np.ndarray, bins: int) -> float:
    if confidence.size == 0:
        return float("nan")
    total = float(confidence.size)
    ece = 0.0
    for row in reliability_bins(confidence, correct, bins):
        if row["count"]:
            ece += abs(float(row["gap"])) * float(row["count"]) / total
    return float(ece)


def metric_block(
    labels: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    nll: np.ndarray,
    brier: np.ndarray,
    bins: int,
) -> Dict[str, Any]:
    correct = predicted == labels
    return {
        "n": int(labels.size),
        "accuracy_pct": float(correct.mean() * 100.0),
        "ece15_pct": float(ece_from_arrays(confidence, correct, bins) * 100.0),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "mean_confidence_pct": float(confidence.mean() * 100.0),
        "correct_confidence_pct": (
            float(confidence[correct].mean() * 100.0) if correct.any() else None
        ),
        "wrong_confidence_pct": (
            float(confidence[~correct].mean() * 100.0) if (~correct).any() else None
        ),
        "wrong_rate_pct": float((~correct).mean() * 100.0),
        "reliability_bins": reliability_bins(confidence, correct, bins),
    }


def summarise_predictions(
    labels: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    nll: np.ndarray,
    brier: np.ndarray,
    bins: int,
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {
        "Overall": metric_block(labels, predicted, confidence, nll, brier, bins)
    }
    for name, (lo, hi) in GROUPS.items():
        mask = (labels >= lo) & (labels < hi)
        summary[name] = metric_block(
            labels[mask], predicted[mask], confidence[mask], nll[mask], brier[mask], bins
        )
    return summary


def evaluate_clean(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    model.eval()
    classifier.eval()
    labels_parts: List[np.ndarray] = []
    prediction_parts: List[np.ndarray] = []
    confidence_parts: List[np.ndarray] = []
    nll_parts: List[np.ndarray] = []
    brier_parts: List[np.ndarray] = []
    with torch.no_grad():
        for images, target in loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits = classifier(model(images))
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            one_hot = F.one_hot(target, num_classes=NUM_CLASSES).float()
            labels_parts.append(target.cpu().numpy())
            prediction_parts.append(predicted.cpu().numpy())
            confidence_parts.append(confidence.cpu().numpy())
            nll_parts.append(F.cross_entropy(logits, target, reduction="none").cpu().numpy())
            brier_parts.append(((probabilities - one_hot).pow(2).sum(dim=1)).cpu().numpy())

    raw = {
        "labels": np.concatenate(labels_parts),
        "predicted": np.concatenate(prediction_parts),
        "confidence": np.concatenate(confidence_parts),
        "nll": np.concatenate(nll_parts),
        "brier": np.concatenate(brier_parts),
    }
    return summarise_predictions(**raw, bins=args.ece_bins), raw


class HeldOutFusionProbe(Dataset):
    """A deterministic, recipient-labelled test-to-test H2T probe.

    Donor *classes* are drawn with the released long-tailed train frequency.
    This is intentionally reported separately from clean-test calibration:
    it is a testable probe of the synthetic distribution H2T creates, not a
    replacement for ordinary CIFAR-100 test accuracy.
    """

    def __init__(self, base_dataset: Dataset, donor_indices: np.ndarray):
        self.base_dataset = base_dataset
        self.donor_indices = donor_indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.donor_indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        recipient, label = self.base_dataset[index]
        donor, _ = self.base_dataset[int(self.donor_indices[index])]
        return recipient, donor, label


def make_fusion_probe_loader(
    dataset: CIFAR100_LT, args: argparse.Namespace
) -> DataLoader:
    base = dataset.eval.dataset
    labels = np.asarray(base.targets, dtype=np.int64)
    class_indices = [np.flatnonzero(labels == class_id) for class_id in range(NUM_CLASSES)]
    probabilities = np.asarray(dataset.cls_num_list, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(args.seed + 20260916)
    donor_classes = rng.choice(NUM_CLASSES, size=len(labels), p=probabilities)
    donor_indices = np.asarray(
        [rng.choice(class_indices[int(class_id)]) for class_id in donor_classes], dtype=np.int64
    )
    return DataLoader(
        HeldOutFusionProbe(base, donor_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )


def evaluate_fusion_probe(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    # Resetting before the probe makes the sequence of official H2T channel
    # masks identical for the two trained classifiers.
    set_paired_seed(args.seed + 314159)
    model.eval()
    classifier.eval()
    labels_parts: List[np.ndarray] = []
    prediction_parts: List[np.ndarray] = []
    confidence_parts: List[np.ndarray] = []
    nll_parts: List[np.ndarray] = []
    brier_parts: List[np.ndarray] = []
    with torch.no_grad():
        for recipient, donor, target in loader:
            recipient = recipient.cuda(non_blocking=True)
            donor = donor.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            features = H2T(model(recipient), model(donor), rho=args.rho)
            logits = classifier(features)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            one_hot = F.one_hot(target, num_classes=NUM_CLASSES).float()
            labels_parts.append(target.cpu().numpy())
            prediction_parts.append(predicted.cpu().numpy())
            confidence_parts.append(confidence.cpu().numpy())
            nll_parts.append(F.cross_entropy(logits, target, reduction="none").cpu().numpy())
            brier_parts.append(((probabilities - one_hot).pow(2).sum(dim=1)).cpu().numpy())

    raw = {
        "labels": np.concatenate(labels_parts),
        "predicted": np.concatenate(prediction_parts),
        "confidence": np.concatenate(confidence_parts),
        "nll": np.concatenate(nll_parts),
        "brier": np.concatenate(brier_parts),
    }
    return summarise_predictions(**raw, bins=args.ece_bins), raw


def train_arm(
    name: str,
    rho: float,
    args: argparse.Namespace,
    arm_dir: Path,
) -> Dict[str, Any]:
    set_paired_seed(args.seed)
    dataset, train_loader, test_loader = build_loaders(args)
    model, classifier, source_checkpoint = load_released_stage1(args.stage1_checkpoint)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    arm_dir.mkdir(parents=True, exist_ok=True)
    train_history: List[Dict[str, float]] = []
    started_at = time.time()
    for epoch in range(args.epochs):
        lr = stage2_lr(epoch, args)
        optimizer.param_groups[0]["lr"] = lr
        # Release config: shift_bn=True.  The backbone stays no-grad/frozen,
        # but its BN running statistics update in both matched arms.
        model.train()
        classifier.train()
        loss_total = 0.0
        correct_total = 0
        seen_total = 0
        for images, target, meta in train_loader:
            images = images.cuda(non_blocking=True)
            donor_images = meta["sample_image"].cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                recipient_features = model(images)
                donor_features = model(donor_images)
                # Call the published H2T function in BOTH arms.  At rho=0 its
                # channel index is empty, so no feature channel is replaced.
                fused_features = H2T(recipient_features, donor_features, rho=rho)
            logits = classifier(fused_features)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = int(target.size(0))
            loss_total += float(loss.item()) * batch_size
            correct_total += int((logits.argmax(dim=1) == target).sum().item())
            seen_total += batch_size
        record = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": loss_total / seen_total,
            "train_accuracy_pct": correct_total / seen_total * 100.0,
            "elapsed_seconds": time.time() - started_at,
        }
        train_history.append(record)
        print(
            f"[{name}] epoch {epoch + 1:02d}/{args.epochs}  "
            f"lr={lr:.6f} loss={record['train_loss']:.4f} "
            f"acc={record['train_accuracy_pct']:.2f}%",
            flush=True,
        )

    # The test set is read exactly once for this arm, after the pre-fixed final
    # epoch.  No validation score determines a selected checkpoint.
    clean_metrics, clean_raw = evaluate_clean(model, classifier, test_loader, args)
    torch.save(
        {
            "epoch": args.epochs,
            "rho": rho,
            "arm": name,
            "state_dict_model": model.state_dict(),
            "state_dict_classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_history": train_history,
            "stage1_checkpoint_sha256": sha256(args.stage1_checkpoint),
        },
        arm_dir / "last_epoch_fixed.pth.tar",
    )
    np.savez_compressed(arm_dir / "clean_test_predictions.npz", **clean_raw)

    fusion_metrics = None
    if not args.skip_fusion_probe:
        probe_loader = make_fusion_probe_loader(dataset, args)
        fusion_metrics, fusion_raw = evaluate_fusion_probe(model, classifier, probe_loader, args)
        np.savez_compressed(arm_dir / "heldout_fusion_probe_predictions.npz", **fusion_raw)

    payload: Dict[str, Any] = {
        "arm": name,
        "rho": rho,
        "stage1_checkpoint_epoch": int(source_checkpoint["epoch"]),
        "stage1_checkpoint_accuracy_pct": float(source_checkpoint["best_acc1"]),
        "stage1_checkpoint_ece_pct": float(source_checkpoint["its_ece"]),
        "train_history": train_history,
        "clean_test": clean_metrics,
        "heldout_fusion_probe": fusion_metrics,
        "duration_seconds": time.time() - started_at,
    }
    with (arm_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def as_float(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def make_flat_rows(arms: Mapping[str, Mapping[str, Any]], area: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for arm_name, arm in arms.items():
        part = arm.get(area)
        if not part:
            continue
        for group, metrics in part.items():
            rows.append(
                {
                    "area": area,
                    "arm": arm_name,
                    "rho": arm["rho"],
                    "group": group,
                    "n": metrics["n"],
                    "accuracy_pct": metrics["accuracy_pct"],
                    "ece15_pct": metrics["ece15_pct"],
                    "nll": metrics["nll"],
                    "brier": metrics["brier"],
                    "mean_confidence_pct": metrics["mean_confidence_pct"],
                    "correct_confidence_pct": metrics["correct_confidence_pct"],
                    "wrong_confidence_pct": metrics["wrong_confidence_pct"],
                    "wrong_rate_pct": metrics["wrong_rate_pct"],
                }
            )
    return rows


def bootstrap_deltas(
    nofusion: Mapping[str, np.ndarray],
    h2t: Mapping[str, np.ndarray],
    bins: int,
    repeats: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap test-set sampling uncertainty, not training-seed variability."""
    if repeats <= 0:
        return {}
    if not np.array_equal(nofusion["labels"], h2t["labels"]):
        raise ValueError("The clean test labels differ between paired arms.")
    rng = np.random.default_rng(seed + 99)
    n = len(nofusion["labels"])
    values: Dict[str, List[float]] = {"accuracy_pct": [], "ece15_pct": [], "nll": [], "brier": []}
    correct_a = nofusion["predicted"] == nofusion["labels"]
    correct_b = h2t["predicted"] == h2t["labels"]
    for _ in range(repeats):
        index = rng.integers(0, n, size=n)
        values["accuracy_pct"].append(float((correct_b[index].mean() - correct_a[index].mean()) * 100.0))
        values["ece15_pct"].append(
            float(
                (ece_from_arrays(h2t["confidence"][index], correct_b[index], bins)
                - ece_from_arrays(nofusion["confidence"][index], correct_a[index], bins))
                * 100.0
            )
        )
        values["nll"].append(float(h2t["nll"][index].mean() - nofusion["nll"][index].mean()))
        values["brier"].append(float(h2t["brier"][index].mean() - nofusion["brier"][index].mean()))
    return {
        key: {
            "mean": float(np.mean(value)),
            "ci95_low": float(np.quantile(value, 0.025)),
            "ci95_high": float(np.quantile(value, 0.975)),
        }
        for key, value in values.items()
    }


def plot_reliability(
    arms: Mapping[str, Mapping[str, Any]], output_path: Path, group: str, title: str
) -> None:
    plt.figure(figsize=(5.2, 4.4), dpi=180)
    plt.plot([0, 1], [0, 1], "--", color="0.45", linewidth=1.2, label="perfect calibration")
    colors = {"nofusion": "#4C78A8", "h2t": "#E45756"}
    labels = {"nofusion": "matched no-fusion (ρ=0)", "h2t": "H2T (ρ=0.25)"}
    for arm_name, arm in arms.items():
        metrics = arm["clean_test"][group]
        rows = metrics["reliability_bins"]
        xs = [row["confidence"] for row in rows if row["count"]]
        ys = [row["accuracy"] for row in rows if row["count"]]
        sizes = [max(15, row["count"] / 22) for row in rows if row["count"]]
        if xs:
            display = f"{labels.get(arm_name, arm_name)}; ECE={metrics['ece15_pct']:.2f}%"
            plt.plot(xs, ys, marker="o", color=colors.get(arm_name), label=display)
            plt.scatter(xs, ys, s=sizes, color=colors.get(arm_name), alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("mean confidence")
    plt.ylabel("empirical accuracy")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7.4, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = "| 组别 | Acc (%) | ECE-15 (%) | NLL | Brier | 错误样本置信度 (%) |"
    separator = "|---|---:|---:|---:|---:|---:|"
    body = [header, separator]
    for row in rows:
        wrong = row["wrong_confidence_pct"]
        wrong_text = "—" if wrong is None else f"{float(wrong):.2f}"
        body.append(
            f"| {row['group']} | {float(row['accuracy_pct']):.2f} | "
            f"{float(row['ece15_pct']):.2f} | {float(row['nll']):.4f} | "
            f"{float(row['brier']):.4f} | {wrong_text} |"
        )
    return "\n".join(body)


def write_phase1_report(
    args: argparse.Namespace,
    output_dir: Path,
    arms: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
) -> None:
    lines = [
        "# H2T 原始 Stage-II：accuracy–calibration 先验审计",
        "",
        "## 审计边界",
        "",
        "- 共同起点：作者 README 提供的 CIFAR-100-LT IF=100 Stage-I checkpoint。",
        "- 训练流程：发布版 H2T 的 CIFAR-100-LT Stage-II 配置（ResNet-32 特征提取器、CE、balanced recipient sampler、long-tailed dual donor、shift-BN、10 epoch、batch 128、SGD）。",
        "- 配对控制：两臂均抽取并前向 donor，均调用发布版 `H2T`；仅 `rho=0` 的通道集合为空、`rho=0.25` 替换 16/64 个通道。因此 dual sampling、随机通道掩码的 RNG 消耗与 BN 更新均被保留。",
        "- 终点：固定第 10 个 Stage-II epoch 的干净 CIFAR-100 测试集；没有依据测试 Accuracy 选择 best checkpoint。ECE 固定为 15 个等宽置信度 bin。",
        "- 限制：这是单个训练 seed 的筛查，不能用“稳定”描述训练随机性；bootstrap 区间只反映固定测试集的抽样不确定性。",
        "",
    ]
    for arm_name, arm in arms.items():
        lines.extend(
            [
                f"## {arm_name}（rho={arm['rho']}）——干净测试集",
                "",
                markdown_table(make_flat_rows({arm_name: arm}, "clean_test")),
                "",
            ]
        )
        if arm.get("heldout_fusion_probe"):
            lines.extend(
                [
                    f"### {arm_name}——辅助的 held-out fusion probe",
                    "",
                    "该 probe 用测试集 recipient 与按训练长尾频率抽取的测试集 donor 构造；它是 synthetic H2T 输入的诊断，不是普通测试校准结果。",
                    "",
                    markdown_table(make_flat_rows({arm_name: arm}, "heldout_fusion_probe")),
                    "",
                ]
            )

    if "nofusion" in arms and "h2t" in arms:
        base = arms["nofusion"]["clean_test"]
        treated = arms["h2t"]["clean_test"]
        lines.extend(["## H2T − matched no-fusion（干净测试集）", ""])
        delta_rows = []
        for group in ("Overall", "Many", "Medium", "Few"):
            delta_rows.append(
                {
                    "group": group,
                    "accuracy_pct": treated[group]["accuracy_pct"] - base[group]["accuracy_pct"],
                    "ece15_pct": treated[group]["ece15_pct"] - base[group]["ece15_pct"],
                    "nll": treated[group]["nll"] - base[group]["nll"],
                    "brier": treated[group]["brier"] - base[group]["brier"],
                    "wrong_confidence_pct": (
                        None
                        if base[group]["wrong_confidence_pct"] is None
                        or treated[group]["wrong_confidence_pct"] is None
                        else treated[group]["wrong_confidence_pct"] - base[group]["wrong_confidence_pct"]
                    ),
                }
            )
        lines.extend(
            [
                "| 组别 | ΔAcc (百分点) | ΔECE-15 (百分点) | ΔNLL | ΔBrier | Δ错误样本置信度 (百分点) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in delta_rows:
            wrong = row["wrong_confidence_pct"]
            wrong_text = "—" if wrong is None else f"{float(wrong):+.2f}"
            lines.append(
                f"| {row['group']} | {float(row['accuracy_pct']):+.2f} | "
                f"{float(row['ece15_pct']):+.2f} | {float(row['nll']):+.4f} | "
                f"{float(row['brier']):+.4f} | {wrong_text} |"
            )
        lines.extend(["", "### 测试集 bootstrap（Overall；不是 seed 稳定性）", ""])
        if bootstrap:
            lines.extend(["| 指标 | Δ均值 | 95% CI |", "|---|---:|---:|"])
            for metric, summary in bootstrap.items():
                lines.append(
                    f"| {metric} | {summary['mean']:+.4f} | "
                    f"[{summary['ci95_low']:+.4f}, {summary['ci95_high']:+.4f}] |"
                )
        else:
            lines.append("未执行 bootstrap。")

        overall_delta_acc = treated["Overall"]["accuracy_pct"] - base["Overall"]["accuracy_pct"]
        overall_delta_ece = treated["Overall"]["ece15_pct"] - base["Overall"]["ece15_pct"]
        few_delta_acc = treated["Few"]["accuracy_pct"] - base["Few"]["accuracy_pct"]
        few_delta_ece = treated["Few"]["ece15_pct"] - base["Few"]["ece15_pct"]
        lines.extend(
            [
                "",
                "## 预先规定的筛查解释",
                "",
                f"- Overall: ΔAcc={overall_delta_acc:+.2f}pp，ΔECE={overall_delta_ece:+.2f}pp。",
                f"- Few: ΔAcc={few_delta_acc:+.2f}pp，ΔECE={few_delta_ece:+.2f}pp。",
                "- 只有在 Accuracy 上升且 ECE/NLL/Brier 的恶化方向一致时，才把“accuracy–calibration trade-off”作为下一阶段 rho 扫描的候选机制。若不是，不能把该路线表述为 H2T 的稳定缺陷。",
                "- 即使本 seed 满足方向性，仍须用独立的 Stage-I/Stage-II seeds 才能回答“稳定恶化”；固定测试集的 bootstrap 不能替代该重复。",
                "",
            ]
        )

    (output_dir / "CALIBRATION_AUDIT_REPORT_ZH.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def save_summary(
    args: argparse.Namespace,
    output_dir: Path,
    arms: Mapping[str, Mapping[str, Any]],
) -> None:
    manifest = {
        "purpose": "Published H2T Stage-II paired accuracy-calibration audit",
        "primary_endpoint": "fixed final Stage-II epoch clean CIFAR-100 test",
        "seed": args.seed,
        "epochs": args.epochs,
        "rho_h2t": args.rho,
        "rho_nofusion": 0.0,
        "ece_bins": args.ece_bins,
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "stage1_checkpoint_sha256": sha256(args.stage1_checkpoint),
        "h2t_operator": str((Path(__file__).resolve().parent / "methods.py").resolve()),
        "h2t_operator_sha256": sha256(Path(__file__).resolve().parent / "methods.py"),
        "source_note": (
            "The published H2T operator and Stage-II mechanics are unchanged. "
            "The current upstream ResNet file contained an incompatible PIF insertion; "
            "the audit uses the standard ResNet-32 feature extractor required by the released checkpoint."
        ),
        "paired_control": (
            "Both arms execute the dual donor forward pass and H2T call; rho=0 has an empty selected-channel set."
        ),
        "arms": arms,
    }
    with (output_dir / "audit_manifest_and_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    rows = make_flat_rows(arms, "clean_test") + make_flat_rows(arms, "heldout_fusion_probe")
    if rows:
        with (output_dir / "metrics_flat.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def verify_dry_run(args: argparse.Namespace) -> None:
    set_paired_seed(args.seed)
    dataset, train_loader, _ = build_loaders(args)
    model, classifier, checkpoint = load_released_stage1(args.stage1_checkpoint)
    model.train()
    classifier.train()
    images, target, meta = next(iter(train_loader))
    images = images.cuda()
    donor = meta["sample_image"].cuda()
    with torch.no_grad():
        features = H2T(model(images), model(donor), rho=args.rho)
    logits = classifier(features)
    print(
        json.dumps(
            {
                "dry_run": "passed",
                "batch_shape": list(images.shape),
                "feature_shape": list(features.shape),
                "logit_shape": list(logits.shape),
                "stage1_epoch": int(checkpoint["epoch"]),
                "stage1_accuracy_pct": float(checkpoint["best_acc1"]),
                "class_counts_head_medium_few": [
                    int(sum(dataset.cls_num_list[:36])),
                    int(sum(dataset.cls_num_list[36:71])),
                    int(sum(dataset.cls_num_list[71:])),
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires CUDA because the released H2T operator calls .cuda().")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        verify_dry_run(args)
        return

    rho_by_arm = {"nofusion": 0.0, "h2t": args.rho}
    arms: Dict[str, Dict[str, Any]] = {}
    for name in args.arms:
        print(f"\n=== Starting paired arm: {name} (rho={rho_by_arm[name]}) ===", flush=True)
        arms[name] = train_arm(name, rho_by_arm[name], args, args.output_dir / name)

    save_summary(args, args.output_dir, arms)
    bootstrap: Dict[str, Dict[str, float]] = {}
    if set(args.arms) == {"nofusion", "h2t"}:
        nofusion_raw = dict(np.load(args.output_dir / "nofusion" / "clean_test_predictions.npz"))
        h2t_raw = dict(np.load(args.output_dir / "h2t" / "clean_test_predictions.npz"))
        bootstrap = bootstrap_deltas(
            nofusion_raw,
            h2t_raw,
            args.ece_bins,
            args.bootstrap_repeats,
            args.seed,
        )
        with (args.output_dir / "bootstrap_overall_deltas.json").open("w", encoding="utf-8") as handle:
            json.dump(bootstrap, handle, ensure_ascii=False, indent=2)
        plot_reliability(
            arms,
            args.output_dir / "reliability_overall_clean_test.png",
            "Overall",
            "Clean test reliability: Overall",
        )
        plot_reliability(
            arms,
            args.output_dir / "reliability_few_clean_test.png",
            "Few",
            "Clean test reliability: Few classes",
        )
    write_phase1_report(args, args.output_dir, arms, bootstrap)
    print(f"\nAudit completed. Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
