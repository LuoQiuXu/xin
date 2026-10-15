from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Counterfactual class-frequency reversal pilot on CIFAR-100."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\xiangmuxunlian\hafe_experiment\data"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cfrr_pilot"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--imb-factor",
        type=float,
        default=0.01,
        help="N_min/N_max. 0.01 corresponds to imbalance ratio 100.",
    )
    parser.add_argument("--subset-seed", type=int, default=20260729)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--recovery-folds", type=int, default=5)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse final checkpoints already present in output-dir.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two epochs and fewer workers for a fast pipeline check.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(data), handle, indent=2, ensure_ascii=False)


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), eps)


def normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), eps)


def make_count_vectors(
    num_classes: int, max_count: int, imb_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array(
        [
            max(1, int(max_count * (imb_factor ** (rank / (num_classes - 1.0)))))
            for rank in range(num_classes)
        ],
        dtype=np.int64,
    )
    reverse = forward[::-1].copy()
    return forward, reverse


def make_nested_class_indices(
    targets: np.ndarray, num_classes: int, subset_seed: int
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(targets == class_id).copy()
        class_rng = np.random.default_rng(subset_seed + class_id * 1009)
        class_rng.shuffle(class_indices)
        result.append(class_indices)
    return result


def select_indices(class_orders: list[np.ndarray], counts: np.ndarray) -> np.ndarray:
    selected = [
        class_orders[class_id][: int(counts[class_id])]
        for class_id in range(len(class_orders))
    ]
    return np.concatenate(selected).astype(np.int64)


def intersection_indices(
    class_orders: list[np.ndarray], counts_a: np.ndarray, counts_b: np.ndarray
) -> np.ndarray:
    selected = [
        class_orders[class_id][
            : int(min(counts_a[class_id], counts_b[class_id]))
        ]
        for class_id in range(len(class_orders))
    ]
    return np.concatenate(selected).astype(np.int64)


class ArrayCIFAR100(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray | None,
        transform,
        return_index: bool = False,
    ):
        self.data = data
        self.targets = targets
        self.indices = (
            np.arange(len(data), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.transform = transform
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        original_index = int(self.indices[item])
        image = Image.fromarray(self.data[original_index])
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.targets[original_index])
        if self.return_index:
            return image, label, original_index
        return image, label


def train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        outputs = outputs + self.shortcut(inputs)
        return F.relu(outputs)


class ResNetCifar(nn.Module):
    def __init__(self, depth: int = 32, num_classes: int = 100):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("CIFAR ResNet depth must be 6n+2.")
        blocks_per_stage = (depth - 2) // 6
        self.in_planes = 16
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(32, blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(64, blocks_per_stage, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.feature_dim = 64
        self._initialize()

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.relu(self.bn1(self.conv1(inputs)))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.avgpool(outputs)
        return torch.flatten(outputs, 1)

    def forward(
        self, inputs: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(inputs)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


def learning_rate(base_lr: float, epoch: int, epochs: int) -> float:
    if epoch >= int(0.9 * epochs):
        return base_lr * 0.01
    if epoch >= int(0.8 * epochs):
        return base_lr * 0.1
    return base_lr


@torch.no_grad()
def classification_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray]:
    model.eval()
    correct = np.zeros(100, dtype=np.int64)
    total = np.zeros(100, dtype=np.int64)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        for class_id in np.unique(labels_np):
            mask = labels_np == class_id
            total[class_id] += int(mask.sum())
            correct[class_id] += int((predictions_np[mask] == class_id).sum())
    per_class = correct / np.maximum(total, 1)
    return float(correct.sum() / max(total.sum(), 1)), per_class


def train_one_environment(
    name: str,
    seed: int,
    train_dataset: Dataset,
    test_loader: DataLoader,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    checkpoint_path = output_dir / f"model_{name}_final.pth"
    if args.skip_train and checkpoint_path.exists():
        print(f"[seed {seed}] Reusing {checkpoint_path}")
        return checkpoint_path

    set_seed(seed)
    model = ResNetCifar(depth=32, num_classes=100).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.time()
    history = []

    for epoch in range(args.epochs):
        current_lr = learning_rate(args.lr, epoch, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * labels.numel()
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total_seen += int(labels.numel())

        row = {
            "epoch": epoch + 1,
            "loss": total_loss / max(total_seen, 1),
            "train_accuracy": total_correct / max(total_seen, 1),
            "lr": current_lr,
        }
        history.append(row)
        report_every = max(1, min(20, args.epochs // 5))
        if (
            epoch == 0
            or epoch + 1 == args.epochs
            or (epoch + 1) % report_every == 0
        ):
            elapsed = time.time() - started
            print(
                f"[seed {seed} {name}] epoch {epoch + 1:03d}/{args.epochs}, "
                f"loss={row['loss']:.4f}, train={row['train_accuracy'] * 100:.2f}%, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    test_accuracy, test_class_accuracy = classification_accuracy(
        model, test_loader, device
    )
    checkpoint = {
        "model": model.state_dict(),
        "seed": seed,
        "environment": name,
        "epochs": args.epochs,
        "test_accuracy": test_accuracy,
        "test_class_accuracy": test_class_accuracy,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    save_json(history, output_dir / f"history_{name}.json")
    print(
        f"[seed {seed} {name}] final test accuracy={test_accuracy * 100:.2f}% "
        f"saved={checkpoint_path}",
        flush=True,
    )
    return checkpoint_path


def load_model(checkpoint_path: Path, device: torch.device) -> ResNetCifar:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ResNetCifar(depth=32, num_classes=100)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    features = []
    logits_all = []
    labels_all = []
    indices_all = []
    model.eval()
    for batch in loader:
        if len(batch) == 3:
            images, labels, indices = batch
        else:
            images, labels = batch
            indices = torch.arange(len(labels))
        images = images.to(device, non_blocking=True)
        logits, batch_features = model(images, return_features=True)
        features.append(batch_features.float().cpu().numpy())
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(labels.numpy())
        indices_all.append(indices.numpy())
    return (
        np.concatenate(features),
        np.concatenate(logits_all),
        np.concatenate(labels_all),
        np.concatenate(indices_all),
    )


def orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = l2_normalize(source.astype(np.float64))
    target_norm = l2_normalize(target.astype(np.float64))
    left, _, right_t = np.linalg.svd(source_norm.T @ target_norm, full_matrices=False)
    return (left @ right_t).astype(np.float32)


def effective_rank(features: np.ndarray, eps: float = 1e-10) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(features) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    positive = eigenvalues[eigenvalues > eps]
    if positive.size == 0:
        return 0.0
    probabilities = positive / positive.sum()
    return float(np.exp(-(probabilities * np.log(probabilities + eps)).sum()))


def covariance_trace(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(features) - 1, 1)
    return float(np.trace(covariance))


def safe_wilcoxon(head: np.ndarray, tail: np.ndarray) -> dict:
    differences = np.asarray(head) - np.asarray(tail)
    if np.allclose(differences, 0.0):
        return {"statistic": 0.0, "pvalue": 1.0}
    result = wilcoxon(differences)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def compute_environment_geometry(
    features: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    classifier_weight: np.ndarray,
) -> dict[str, np.ndarray]:
    normalized = l2_normalize(features.astype(np.float64))
    normalized_weights = l2_normalize(classifier_weight.astype(np.float64))
    prototypes = np.zeros((100, normalized.shape[1]), dtype=np.float64)
    radii = np.zeros(100, dtype=np.float64)
    traces = np.zeros(100, dtype=np.float64)
    ranks = np.zeros(100, dtype=np.float64)
    alignments = np.zeros(100, dtype=np.float64)
    accuracies = np.zeros(100, dtype=np.float64)
    for class_id in range(100):
        class_features = normalized[labels == class_id]
        prototype = normalize_vector(class_features.mean(axis=0))
        prototypes[class_id] = prototype
        radii[class_id] = float(np.mean(1.0 - class_features @ prototype))
        traces[class_id] = covariance_trace(class_features)
        ranks[class_id] = effective_rank(class_features)
        alignments[class_id] = float(normalized_weights[class_id] @ prototype)
        class_logits = logits[labels == class_id]
        accuracies[class_id] = float(
            (class_logits.argmax(axis=1) == class_id).mean()
        )

    margins = np.zeros(100, dtype=np.float64)
    similarity = normalized @ prototypes.T
    for class_id in range(100):
        mask = labels == class_id
        true_similarity = similarity[mask, class_id]
        other_similarity = similarity[mask].copy()
        other_similarity[:, class_id] = -np.inf
        margins[class_id] = float(
            np.mean(true_similarity - other_similarity.max(axis=1))
        )
    return {
        "features": normalized,
        "prototypes": prototypes,
        "radius": radii,
        "trace": traces,
        "effective_rank": ranks,
        "alignment": alignments,
        "margin": margins,
        "accuracy": accuracies,
    }


def role_ordered(
    values_a: np.ndarray,
    values_b: np.ndarray,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a_is_head = counts_a > counts_b
    head = np.where(a_is_head, values_a, values_b)
    tail = np.where(a_is_head, values_b, values_a)
    return head, tail


def class_geometry_rows(
    geometry_a: dict[str, np.ndarray],
    geometry_b: dict[str, np.ndarray],
    counts_a: np.ndarray,
    counts_b: np.ndarray,
) -> tuple[list[dict], dict]:
    prototypes_a = geometry_a["prototypes"]
    prototypes_b = geometry_b["prototypes"]
    prototype_drift = 1.0 - np.sum(prototypes_a * prototypes_b, axis=1)
    feature_drift = np.zeros(100, dtype=np.float64)
    for class_id in range(100):
        class_mask = TEST_LABELS_FOR_GEOMETRY == class_id
        feature_drift[class_id] = float(
            np.mean(
                1.0
                - np.sum(
                    geometry_a["features"][class_mask]
                    * geometry_b["features"][class_mask],
                    axis=1,
                )
            )
        )

    metric_names = [
        "accuracy",
        "radius",
        "trace",
        "effective_rank",
        "alignment",
        "margin",
    ]
    ordered = {}
    for metric_name in metric_names:
        ordered[metric_name] = role_ordered(
            geometry_a[metric_name],
            geometry_b[metric_name],
            counts_a,
            counts_b,
        )

    rows = []
    for class_id in range(100):
        row = {
            "class_id": class_id,
            "count_a": int(counts_a[class_id]),
            "count_b": int(counts_b[class_id]),
            "head_environment": "A" if counts_a[class_id] > counts_b[class_id] else "B",
            "tail_environment": "B" if counts_a[class_id] > counts_b[class_id] else "A",
            "frequency_ratio": float(
                max(counts_a[class_id], counts_b[class_id])
                / max(min(counts_a[class_id], counts_b[class_id]), 1)
            ),
            "prototype_drift": float(prototype_drift[class_id]),
            "same_image_feature_drift": float(feature_drift[class_id]),
        }
        for metric_name in metric_names:
            head, tail = ordered[metric_name]
            row[f"{metric_name}_head"] = float(head[class_id])
            row[f"{metric_name}_tail"] = float(tail[class_id])
            row[f"{metric_name}_delta_head_minus_tail"] = float(
                head[class_id] - tail[class_id]
            )
        rows.append(row)

    non_equal = counts_a != counts_b
    strong = non_equal & (
        np.maximum(counts_a, counts_b) / np.maximum(np.minimum(counts_a, counts_b), 1)
        >= 10.0
    )
    summary = {}
    for subset_name, subset_mask in [("all_swapped", non_equal), ("ratio_ge_10", strong)]:
        subset_summary = {"num_classes": int(subset_mask.sum())}
        for metric_name in metric_names:
            head, tail = ordered[metric_name]
            subset_summary[metric_name] = {
                "head_mean": float(head[subset_mask].mean()),
                "tail_mean": float(tail[subset_mask].mean()),
                "delta_head_minus_tail": float(
                    (head[subset_mask] - tail[subset_mask]).mean()
                ),
                "head_better_fraction": float(
                    (head[subset_mask] > tail[subset_mask]).mean()
                ),
                "wilcoxon": safe_wilcoxon(head[subset_mask], tail[subset_mask]),
            }
        subset_summary["prototype_drift_mean"] = float(
            prototype_drift[subset_mask].mean()
        )
        subset_summary["same_image_feature_drift_mean"] = float(
            feature_drift[subset_mask].mean()
        )
        summary[subset_name] = subset_summary

    log_ratio = np.log(
        np.maximum(counts_a, counts_b)
        / np.maximum(np.minimum(counts_a, counts_b), 1)
    )
    summary["correlations"] = {
        "log_frequency_ratio_vs_prototype_drift_spearman": float(
            spearmanr(log_ratio[non_equal], prototype_drift[non_equal]).statistic
        ),
        "log_frequency_ratio_vs_feature_drift_spearman": float(
            spearmanr(log_ratio[non_equal], feature_drift[non_equal]).statistic
        ),
        "log_frequency_ratio_vs_accuracy_gain_spearman": float(
            spearmanr(
                log_ratio[non_equal],
                (
                    ordered["accuracy"][0][non_equal]
                    - ordered["accuracy"][1][non_equal]
                ),
            ).statistic
        ),
    }
    return rows, summary


def align_features(features_b: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return l2_normalize(l2_normalize(features_b) @ rotation)


def paired_recovery_cross_validation(
    train_features_a: np.ndarray,
    train_features_b: np.ndarray,
    train_labels: np.ndarray,
    train_indices: np.ndarray,
    test_features_a: np.ndarray,
    test_features_b: np.ndarray,
    test_labels: np.ndarray,
    anchor_features_a: np.ndarray,
    anchor_features_b: np.ndarray,
    anchor_labels: np.ndarray,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    folds: int,
    alpha: float,
    seed: int,
) -> dict:
    results = []
    directions = [
        ("A_tail_to_B_head", np.flatnonzero(counts_a < counts_b)),
        ("B_tail_to_A_head", np.flatnonzero(counts_b < counts_a)),
    ]

    for direction_name, direction_classes in directions:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold_id, (source_positions, target_positions) in enumerate(
            splitter.split(direction_classes)
        ):
            source_classes = direction_classes[source_positions]
            target_classes = direction_classes[target_positions]
            alignment_mask = ~np.isin(anchor_labels, target_classes)
            rotation = orthogonal_procrustes(
                anchor_features_b[alignment_mask],
                anchor_features_a[alignment_mask],
            )

            train_a = l2_normalize(train_features_a)
            train_b = align_features(train_features_b, rotation)
            test_a = l2_normalize(test_features_a)
            test_b = align_features(test_features_b, rotation)

            source_train_mask = np.isin(train_labels, source_classes)
            target_test_mask = np.isin(test_labels, target_classes)
            if direction_name == "A_tail_to_B_head":
                source_x = train_a[source_train_mask]
                source_y = train_b[source_train_mask]
                target_x = test_a[target_test_mask]
                target_y = test_b[target_test_mask]
            else:
                source_x = train_b[source_train_mask]
                source_y = train_a[source_train_mask]
                target_x = test_b[target_test_mask]
                target_y = test_a[target_test_mask]

            source_class_labels = train_labels[source_train_mask]
            target_class_labels = test_labels[target_test_mask]
            source_tail_counts = np.minimum(
                counts_a[source_class_labels], counts_b[source_class_labels]
            )
            source_head_counts = np.maximum(
                counts_a[source_class_labels], counts_b[source_class_labels]
            )
            target_tail_counts = np.minimum(
                counts_a[target_class_labels], counts_b[target_class_labels]
            )
            target_head_counts = np.maximum(
                counts_a[target_class_labels], counts_b[target_class_labels]
            )
            source_condition = np.stack(
                [
                    np.log(source_tail_counts / 500.0 + 1e-8),
                    np.log(source_head_counts / 500.0 + 1e-8),
                ],
                axis=1,
            )
            target_condition = np.stack(
                [
                    np.log(target_tail_counts / 500.0 + 1e-8),
                    np.log(target_head_counts / 500.0 + 1e-8),
                ],
                axis=1,
            )
            residual_target = source_y - source_x
            raw_sample_error = 1.0 - np.sum(target_x * target_y, axis=1)
            variants = {
                "unconditional": (source_x, target_x),
                "frequency_conditioned": (
                    np.concatenate([source_x, source_condition], axis=1),
                    np.concatenate([target_x, target_condition], axis=1),
                ),
            }
            variant_results = {}
            for variant_name, (source_design, target_design) in variants.items():
                regressor = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=alpha),
                )
                regressor.fit(source_design, residual_target)
                predicted_y = l2_normalize(
                    target_x + regressor.predict(target_design)
                )
                recovered_sample_error = 1.0 - np.sum(
                    predicted_y * target_y, axis=1
                )
                class_raw_errors = []
                class_recovered_errors = []
                for class_id in target_classes:
                    class_mask = target_class_labels == class_id
                    raw_prototype = normalize_vector(
                        target_x[class_mask].mean(axis=0)
                    )
                    recovered_prototype = normalize_vector(
                        predicted_y[class_mask].mean(axis=0)
                    )
                    oracle_prototype = normalize_vector(
                        target_y[class_mask].mean(axis=0)
                    )
                    class_raw_errors.append(
                        1.0 - raw_prototype @ oracle_prototype
                    )
                    class_recovered_errors.append(
                        1.0 - recovered_prototype @ oracle_prototype
                    )
                variant_results[variant_name] = {
                    "sample_error": float(recovered_sample_error.mean()),
                    "sample_error_reduction_fraction": float(
                        1.0
                        - recovered_sample_error.mean()
                        / max(raw_sample_error.mean(), 1e-12)
                    ),
                    "sample_improved_fraction": float(
                        (recovered_sample_error < raw_sample_error).mean()
                    ),
                    "prototype_error": float(np.mean(class_recovered_errors)),
                    "prototype_error_reduction_fraction": float(
                        1.0
                        - np.mean(class_recovered_errors)
                        / max(np.mean(class_raw_errors), 1e-12)
                    ),
                    "prototype_improved_fraction": float(
                        (
                            np.asarray(class_recovered_errors)
                            < np.asarray(class_raw_errors)
                        ).mean()
                    ),
                }

            recovered = variant_results["frequency_conditioned"]
            unconditional = variant_results["unconditional"]

            result = {
                "direction": direction_name,
                "fold": fold_id,
                "source_classes": source_classes.tolist(),
                "target_classes": target_classes.tolist(),
                "raw_sample_cosine_error": float(raw_sample_error.mean()),
                "unconditional_sample_cosine_error": unconditional["sample_error"],
                "recovered_sample_cosine_error": recovered["sample_error"],
                "sample_error_reduction_fraction": recovered[
                    "sample_error_reduction_fraction"
                ],
                "sample_improved_fraction": recovered[
                    "sample_improved_fraction"
                ],
                "frequency_conditioning_sample_gain": float(
                    unconditional["sample_error"] - recovered["sample_error"]
                ),
                "raw_prototype_cosine_error": float(np.mean(class_raw_errors)),
                "unconditional_prototype_cosine_error": unconditional[
                    "prototype_error"
                ],
                "recovered_prototype_cosine_error": recovered[
                    "prototype_error"
                ],
                "prototype_error_reduction_fraction": recovered[
                    "prototype_error_reduction_fraction"
                ],
                "prototype_improved_fraction": recovered[
                    "prototype_improved_fraction"
                ],
                "frequency_conditioning_prototype_gain": float(
                    unconditional["prototype_error"] - recovered["prototype_error"]
                ),
            }
            results.append(result)

    aggregate_keys = [
        "raw_sample_cosine_error",
        "unconditional_sample_cosine_error",
        "recovered_sample_cosine_error",
        "sample_error_reduction_fraction",
        "sample_improved_fraction",
        "frequency_conditioning_sample_gain",
        "raw_prototype_cosine_error",
        "unconditional_prototype_cosine_error",
        "recovered_prototype_cosine_error",
        "prototype_error_reduction_fraction",
        "prototype_improved_fraction",
        "frequency_conditioning_prototype_gain",
    ]
    aggregate = {}
    for key in aggregate_keys:
        values = np.array([result[key] for result in results], dtype=np.float64)
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return {"folds": results, "aggregate": aggregate}


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_report(seed_summaries: list[dict], output_dir: Path) -> None:
    geometry_acc_delta = [
        summary["geometry"]["ratio_ge_10"]["accuracy"]["delta_head_minus_tail"]
        for summary in seed_summaries
    ]
    geometry_proto_drift = [
        summary["geometry"]["ratio_ge_10"]["prototype_drift_mean"]
        for summary in seed_summaries
    ]
    recovery_sample = [
        summary["recovery"]["aggregate"]["sample_error_reduction_fraction"]["mean"]
        for summary in seed_summaries
    ]
    recovery_proto = [
        summary["recovery"]["aggregate"][
            "prototype_error_reduction_fraction"
        ]["mean"]
        for summary in seed_summaries
    ]
    recovery_proto_success = [
        summary["recovery"]["aggregate"]["prototype_improved_fraction"]["mean"]
        for summary in seed_summaries
    ]
    conditioning_proto_gain = [
        summary["recovery"]["aggregate"][
            "frequency_conditioning_prototype_gain"
        ]["mean"]
        for summary in seed_summaries
    ]

    acc_delta_mean = float(np.mean(geometry_acc_delta))
    sample_recovery_mean = float(np.mean(recovery_sample))
    prototype_recovery_mean = float(np.mean(recovery_proto))
    proto_success_mean = float(np.mean(recovery_proto_success))
    conditioning_proto_gain_mean = float(np.mean(conditioning_proto_gain))
    mechanism_pass = acc_delta_mean > 0.05
    transfer_pass = (
        prototype_recovery_mean > 0.05
        and proto_success_mean > 0.6
        and conditioning_proto_gain_mean > 0.0
    )
    if mechanism_pass and transfer_pass:
        verdict = "支持：频率干预产生明显变化，且简单跨类别恢复器已有正迁移。"
    elif mechanism_pass:
        verdict = "部分支持：频率效应明显，但当前线性恢复器尚未证明可迁移恢复。"
    else:
        verdict = "暂不支持：本轮实验没有观察到足够稳定的同类头尾频率效应。"

    lines = [
        "# 反事实头尾表征恢复：最小实验报告",
        "",
        f"**结论：{verdict}**",
        "",
        "## 实验设置",
        "",
        "- CIFAR-100 仅作为受控实验载体。",
        "- 两个环境使用完全相同的指数长尾样本直方图（IF=100），类别—频率排序完全反转。",
        "- 同一类别的小样本集合严格嵌套在大样本集合中。",
        "- ResNet-32、交叉熵、相同初始化与训练超参数。",
        "- 所有几何指标都在同一平衡测试集上计算。",
        "- B→A 特征空间使用公共训练图像做正交 Procrustes 对齐。",
        "- 恢复器采用类别不相交的交叉验证；目标类别不参与恢复器训练或空间对齐。",
        "",
        "## 核心结果",
        "",
        f"- 强角色互换类别的头类准确率相对尾类平均变化：**{acc_delta_mean * 100:.2f} 个百分点**。",
        f"- 强角色互换类别的平均原型漂移（1-cos）：**{np.mean(geometry_proto_drift):.4f}**。",
        f"- 留出类别同图像余弦误差相对下降：**{sample_recovery_mean * 100:.2f}%**。",
        f"- 留出类别原型误差相对下降：**{prototype_recovery_mean * 100:.2f}%**。",
        f"- 留出类别原型改善比例：**{proto_success_mean * 100:.2f}%**。",
        f"- 加入频率条件后相对无条件恢复器的原型误差绝对改善：**{conditioning_proto_gain_mean:.6f}**。",
        "",
        "## 判定门槛",
        "",
        "- 机制门槛：强角色互换类别头类准确率平均至少高5个百分点。",
        "- 迁移门槛：留出类原型误差至少下降5%，至少60%的留出类获得改善，且频率条件优于无条件恢复。",
        "",
        "详细的逐类结果和每折恢复结果见同目录 CSV/JSON。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


TEST_LABELS_FOR_GEOMETRY = np.empty(0, dtype=np.int64)


def run_seed(
    seed: int,
    raw_train,
    raw_test,
    class_orders: list[np.ndarray],
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    global TEST_LABELS_FOR_GEOMETRY
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    selected_a = select_indices(class_orders, counts_a)
    selected_b = select_indices(class_orders, counts_b)
    shared_indices = intersection_indices(class_orders, counts_a, counts_b)

    train_a = ArrayCIFAR100(
        raw_train.data,
        np.asarray(raw_train.targets),
        selected_a,
        train_transform(),
    )
    train_b = ArrayCIFAR100(
        raw_train.data,
        np.asarray(raw_train.targets),
        selected_b,
        train_transform(),
    )
    test_dataset = ArrayCIFAR100(
        raw_test.data,
        np.asarray(raw_test.targets),
        None,
        eval_transform(),
        return_index=True,
    )
    test_loader = DataLoader(
        ArrayCIFAR100(
            raw_test.data,
            np.asarray(raw_test.targets),
            None,
            eval_transform(),
        ),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    checkpoint_a = train_one_environment(
        "A", seed, train_a, test_loader, seed_dir, args, device
    )
    checkpoint_b = train_one_environment(
        "B", seed, train_b, test_loader, seed_dir, args, device
    )
    model_a = load_model(checkpoint_a, device)
    model_b = load_model(checkpoint_b, device)

    anchor_dataset = ArrayCIFAR100(
        raw_train.data,
        np.asarray(raw_train.targets),
        shared_indices,
        eval_transform(),
        return_index=True,
    )
    full_train_dataset = ArrayCIFAR100(
        raw_train.data,
        np.asarray(raw_train.targets),
        None,
        eval_transform(),
        return_index=True,
    )

    anchor_a, _, anchor_labels, _ = extract_features(
        model_a, anchor_dataset, args, device
    )
    anchor_b, _, _, _ = extract_features(model_b, anchor_dataset, args, device)
    rotation = orthogonal_procrustes(anchor_b, anchor_a)
    alignment_before = float(
        np.mean(np.sum(l2_normalize(anchor_a) * l2_normalize(anchor_b), axis=1))
    )
    alignment_after = float(
        np.mean(
            np.sum(
                l2_normalize(anchor_a) * align_features(anchor_b, rotation), axis=1
            )
        )
    )

    test_a, logits_a, test_labels, _ = extract_features(
        model_a, test_dataset, args, device
    )
    test_b, logits_b, _, _ = extract_features(model_b, test_dataset, args, device)
    TEST_LABELS_FOR_GEOMETRY = test_labels.copy()
    aligned_test_b = align_features(test_b, rotation)

    weight_a = model_a.fc.weight.detach().cpu().numpy()
    weight_b = l2_normalize(model_b.fc.weight.detach().cpu().numpy()) @ rotation
    geometry_a = compute_environment_geometry(
        test_a, logits_a, test_labels, weight_a
    )
    geometry_b = compute_environment_geometry(
        aligned_test_b, logits_b, test_labels, weight_b
    )
    rows, geometry_summary = class_geometry_rows(
        geometry_a, geometry_b, counts_a, counts_b
    )
    write_csv(rows, seed_dir / "per_class_geometry.csv")

    train_a_features, _, train_labels, train_indices = extract_features(
        model_a, full_train_dataset, args, device
    )
    train_b_features, _, _, _ = extract_features(
        model_b, full_train_dataset, args, device
    )
    recovery = paired_recovery_cross_validation(
        train_features_a=train_a_features,
        train_features_b=train_b_features,
        train_labels=train_labels,
        train_indices=train_indices,
        test_features_a=test_a,
        test_features_b=test_b,
        test_labels=test_labels,
        anchor_features_a=anchor_a,
        anchor_features_b=anchor_b,
        anchor_labels=anchor_labels,
        counts_a=counts_a,
        counts_b=counts_b,
        folds=args.recovery_folds,
        alpha=args.ridge_alpha,
        seed=seed,
    )
    summary = {
        "seed": seed,
        "alignment": {
            "num_anchor_images": int(len(anchor_labels)),
            "mean_same_image_cosine_before": alignment_before,
            "mean_same_image_cosine_after": alignment_after,
        },
        "geometry": geometry_summary,
        "recovery": recovery,
    }
    save_json(summary, seed_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 2
        args.num_workers = 0
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), args.output_dir / "args.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    raw_train = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False
    )
    raw_test = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=False
    )
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    class_orders = make_nested_class_indices(
        train_targets, num_classes=100, subset_seed=args.subset_seed
    )
    counts_a, counts_b = make_count_vectors(
        num_classes=100,
        max_count=500,
        imb_factor=args.imb_factor,
    )
    save_json(
        {
            "counts_a": counts_a,
            "counts_b": counts_b,
            "total_a": int(counts_a.sum()),
            "total_b": int(counts_b.sum()),
            "histograms_identical": bool(
                np.array_equal(np.sort(counts_a), np.sort(counts_b))
            ),
        },
        args.output_dir / "frequency_design.json",
    )
    print(
        f"Frequency design: total={counts_a.sum()}, "
        f"A range={counts_a.min()}..{counts_a.max()}, "
        f"histograms identical={np.array_equal(np.sort(counts_a), np.sort(counts_b))}"
    )

    summaries = []
    for seed in args.seeds:
        summaries.append(
            run_seed(
                seed,
                raw_train,
                raw_test,
                class_orders,
                counts_a,
                counts_b,
                args,
                device,
            )
        )
    save_json(summaries, args.output_dir / "all_seed_summaries.json")
    generate_report(summaries, args.output_dir)
    print(f"Finished. Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
