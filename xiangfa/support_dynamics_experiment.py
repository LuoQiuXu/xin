from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


NUM_CLASSES = 100
MAX_CLASS_COUNT = 500
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

ANGLE_NAMES = [
    "angle_q50",
    "angle_q80",
    "angle_q90",
    "angle_q95",
    "angle_mean",
    "angle_std",
]
SHAPE_NAMES = [
    "tangent_trace",
    "tangent_effective_rank",
    *[f"tangent_eig_{rank}" for rank in range(1, 9)],
]
RELATION_NAMES = [
    "neighbor_cos_1",
    "neighbor_cos_3mean",
    "neighbor_cos_5mean",
    "feature_margin_mean",
    "feature_overlap_rate",
    "cap_overlap_fraction",
    "cap_overlap_excess_mean",
    "cap_overlap_excess_max",
]
DESCRIPTOR_NAMES = ANGLE_NAMES + SHAPE_NAMES + RELATION_NAMES
DESCRIPTOR_INDEX = {name: index for index, name in enumerate(DESCRIPTOR_NAMES)}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual class-support dynamics under controlled class-frequency "
            "interventions on CIFAR-100."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\xiangmuxunlian\hafe_experiment\data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "support_dynamics",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[5, 20, 50, 200, 500],
        help="Per-class frequency levels. The number of environments equals the number of levels.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--subset-seed", type=int, default=20260729)
    parser.add_argument("--design-seed", type=int, default=20260730)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--recovery-folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--tree-min-leaf", type=int, default=5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse compatible final checkpoints in output-dir.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch per environment with a small tree ensemble.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), eps)


def normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), eps)


def make_nested_class_orders(
    targets: np.ndarray, subset_seed: int
) -> list[np.ndarray]:
    orders = []
    for class_id in range(NUM_CLASSES):
        indices = np.flatnonzero(targets == class_id).copy()
        generator = np.random.default_rng(subset_seed + class_id * 1009)
        generator.shuffle(indices)
        orders.append(indices)
    return orders


def make_latin_frequency_design(
    levels: list[int], design_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(levels) < 3:
        raise ValueError("At least three frequency levels are required.")
    if NUM_CLASSES % len(levels) != 0:
        raise ValueError(
            f"{NUM_CLASSES} classes must be divisible by {len(levels)} levels."
        )
    if sorted(set(levels)) != sorted(levels):
        raise ValueError("Frequency levels must be unique and sorted ascending.")
    if min(levels) < 2 or max(levels) > MAX_CLASS_COUNT:
        raise ValueError(f"Levels must lie in [2, {MAX_CLASS_COUNT}].")

    generator = np.random.default_rng(design_seed)
    permutation = generator.permutation(NUM_CLASSES)
    group = np.empty(NUM_CLASSES, dtype=np.int64)
    group[permutation] = np.arange(NUM_CLASSES) % len(levels)

    levels_array = np.asarray(levels, dtype=np.int64)
    counts = np.zeros((len(levels), NUM_CLASSES), dtype=np.int64)
    for environment in range(len(levels)):
        counts[environment] = levels_array[(group + environment) % len(levels)]

    expected_histogram = np.sort(counts[0])
    if not all(np.array_equal(np.sort(row), expected_histogram) for row in counts):
        raise AssertionError("Environment frequency histograms are not identical.")
    for class_id in range(NUM_CLASSES):
        if not np.array_equal(np.sort(counts[:, class_id]), levels_array):
            raise AssertionError("A class does not visit every frequency level once.")
    return counts, group


def select_indices(
    class_orders: list[np.ndarray], counts: np.ndarray
) -> np.ndarray:
    selected = [
        class_orders[class_id][: int(counts[class_id])]
        for class_id in range(NUM_CLASSES)
    ]
    return np.concatenate(selected).astype(np.int64)


def common_anchor_indices(
    class_orders: list[np.ndarray], levels: list[int]
) -> np.ndarray:
    count = min(levels)
    return np.concatenate(
        [class_orders[class_id][:count] for class_id in range(NUM_CLASSES)]
    ).astype(np.int64)


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
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
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
    def __init__(self, depth: int = 32, num_classes: int = NUM_CLASSES):
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
        self._initialize()

    def _make_layer(
        self, planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(
                BasicBlock(self.in_planes, planes, block_stride)
            )
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

    def forward(self, inputs: torch.Tensor, return_features: bool = False):
        features = self.forward_features(inputs)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


@torch.no_grad()
def classification_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += int((logits.argmax(dim=1) == labels).sum())
        seen += int(labels.numel())
    return correct / max(seen, 1)


def cosine_learning_rate(base_lr: float, epoch: int, epochs: int) -> float:
    warmup = min(5, max(1, epochs // 10))
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_environment(
    environment: int,
    seed: int,
    counts: np.ndarray,
    train_dataset: Dataset,
    test_loader: DataLoader,
    seed_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    checkpoint_path = seed_dir / f"model_env_{environment}_final.pth"
    if args.skip_train and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        saved_counts = np.asarray(checkpoint.get("counts", []), dtype=np.int64)
        if (
            checkpoint.get("epochs") == args.epochs
            and np.array_equal(saved_counts, counts)
        ):
            print(f"[seed {seed} env {environment}] reusing {checkpoint_path}")
            return checkpoint_path
        raise ValueError(
            f"Incompatible checkpoint at {checkpoint_path}; "
            "epochs or frequency design differs."
        )

    set_seed(seed)
    model = ResNetCifar(depth=32).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not args.no_amp
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        lr = cosine_learning_rate(args.lr, epoch, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and not args.no_amp,
            ):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += int(labels.numel())
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "loss": loss_sum / max(seen, 1),
            "train_accuracy": correct / max(seen, 1),
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print_every = max(1, args.epochs // 5)
        if epoch == 0 or (epoch + 1) % print_every == 0 or epoch + 1 == args.epochs:
            print(
                f"[seed {seed} env {environment}] epoch {epoch + 1:03d}/"
                f"{args.epochs}, loss={row['loss']:.4f}, "
                f"train={row['train_accuracy'] * 100:.2f}%, "
                f"elapsed={row['elapsed_seconds']:.1f}s"
            )

    test_accuracy = classification_accuracy(model, test_loader, device)
    checkpoint = {
        "model": model.state_dict(),
        "seed": seed,
        "environment": environment,
        "epochs": args.epochs,
        "counts": counts,
        "test_accuracy": test_accuracy,
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    save_json(history, seed_dir / f"history_env_{environment}.json")
    print(
        f"[seed {seed} env {environment}] final test accuracy="
        f"{test_accuracy * 100:.2f}% saved={checkpoint_path}"
    )
    return checkpoint_path


def load_model(path: Path, device: torch.device) -> ResNetCifar:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ResNetCifar(depth=32)
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
    all_features = []
    all_logits = []
    all_labels = []
    all_indices = []
    model.eval()
    for batch in loader:
        if len(batch) == 3:
            images, labels, indices = batch
        else:
            images, labels = batch
            indices = torch.arange(len(labels))
        images = images.to(device, non_blocking=True)
        logits, features = model(images, return_features=True)
        all_features.append(features.float().cpu().numpy())
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.numpy())
        all_indices.append(indices.numpy())
    return (
        np.concatenate(all_features),
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        np.concatenate(all_indices),
    )


def orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = l2_normalize(source.astype(np.float64))
    target_norm = l2_normalize(target.astype(np.float64))
    left, _, right_t = np.linalg.svd(
        source_norm.T @ target_norm, full_matrices=False
    )
    return (left @ right_t).astype(np.float32)


def effective_rank_from_eigenvalues(
    eigenvalues: np.ndarray, eps: float = 1e-12
) -> float:
    positive = np.clip(eigenvalues, 0.0, None)
    positive = positive[positive > eps]
    if positive.size == 0:
        return 0.0
    probabilities = positive / positive.sum()
    return float(np.exp(-(probabilities * np.log(probabilities + eps)).sum()))


@dataclass
class SupportDescription:
    descriptors: np.ndarray
    prototypes: np.ndarray
    class_accuracy: np.ndarray | None


def describe_supports(
    features: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray | None = None,
) -> SupportDescription:
    normalized = l2_normalize(features.astype(np.float64))
    feature_dim = normalized.shape[1]
    prototypes = np.zeros((NUM_CLASSES, feature_dim), dtype=np.float64)
    class_features: list[np.ndarray] = []
    angles_by_class: list[np.ndarray] = []
    eigs_by_class: list[np.ndarray] = []

    for class_id in range(NUM_CLASSES):
        current = normalized[labels == class_id]
        if len(current) < 2:
            raise ValueError(f"Class {class_id} has fewer than two support samples.")
        prototype = normalize_vector(current.mean(axis=0))
        prototypes[class_id] = prototype
        class_features.append(current)
        cosines = np.clip(current @ prototype, -1.0, 1.0)
        angles = np.arccos(cosines)
        angles_by_class.append(angles)
        tangent = current - cosines[:, None] * prototype[None, :]
        covariance = tangent.T @ tangent / max(len(current) - 1, 1)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        eigs_by_class.append(np.clip(eigenvalues, 0.0, None))

    center_cosine = np.clip(prototypes @ prototypes.T, -1.0, 1.0)
    center_angles = np.arccos(center_cosine)
    theta95 = np.array(
        [np.quantile(angles, 0.95) for angles in angles_by_class],
        dtype=np.float64,
    )
    descriptors = np.zeros(
        (NUM_CLASSES, len(DESCRIPTOR_NAMES)), dtype=np.float64
    )
    class_accuracy = (
        np.zeros(NUM_CLASSES, dtype=np.float64) if logits is not None else None
    )

    for class_id in range(NUM_CLASSES):
        current = class_features[class_id]
        angles = angles_by_class[class_id]
        eigenvalues = eigs_by_class[class_id]
        own_similarity = current @ prototypes[class_id]
        all_similarity = current @ prototypes.T
        all_similarity[:, class_id] = -np.inf
        margins = own_similarity - all_similarity.max(axis=1)

        other_center_cosine = np.delete(center_cosine[class_id], class_id)
        sorted_center_cosine = np.sort(other_center_cosine)[::-1]
        cap_excess = (
            theta95[class_id]
            + np.delete(theta95, class_id)
            - np.delete(center_angles[class_id], class_id)
        )
        positive_excess = np.maximum(cap_excess, 0.0)
        top_eigenvalues = np.zeros(8, dtype=np.float64)
        top_eigenvalues[: min(8, len(eigenvalues))] = eigenvalues[:8]

        values = {
            "angle_q50": np.quantile(angles, 0.50),
            "angle_q80": np.quantile(angles, 0.80),
            "angle_q90": np.quantile(angles, 0.90),
            "angle_q95": np.quantile(angles, 0.95),
            "angle_mean": angles.mean(),
            "angle_std": angles.std(ddof=1) if len(angles) > 1 else 0.0,
            "tangent_trace": eigenvalues.sum(),
            "tangent_effective_rank": effective_rank_from_eigenvalues(
                eigenvalues
            ),
            **{
                f"tangent_eig_{rank + 1}": top_eigenvalues[rank]
                for rank in range(8)
            },
            "neighbor_cos_1": sorted_center_cosine[0],
            "neighbor_cos_3mean": sorted_center_cosine[:3].mean(),
            "neighbor_cos_5mean": sorted_center_cosine[:5].mean(),
            "feature_margin_mean": margins.mean(),
            "feature_overlap_rate": np.mean(margins <= 0.0),
            "cap_overlap_fraction": np.mean(cap_excess > 0.0),
            "cap_overlap_excess_mean": positive_excess.mean(),
            "cap_overlap_excess_max": positive_excess.max(),
        }
        descriptors[class_id] = [values[name] for name in DESCRIPTOR_NAMES]
        if class_accuracy is not None:
            current_logits = logits[labels == class_id]
            class_accuracy[class_id] = np.mean(
                current_logits.argmax(axis=1) == class_id
            )

    return SupportDescription(
        descriptors=descriptors,
        prototypes=prototypes,
        class_accuracy=class_accuracy,
    )


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    result = spearmanr(x, y)
    return float(result.statistic) if np.isfinite(result.statistic) else 0.0


def safe_wilcoxon(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, 0.0):
        return {"statistic": 0.0, "pvalue": 1.0}
    result = wilcoxon(values)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def summarize_mechanism(
    test_descriptors: list[np.ndarray],
    aligned_test_features: list[np.ndarray],
    test_labels: np.ndarray,
    class_accuracies: list[np.ndarray],
    counts_matrix: np.ndarray,
    levels: list[int],
) -> tuple[dict, list[dict]]:
    rows = []
    per_metric_rhos = {name: [] for name in DESCRIPTOR_NAMES}
    accuracy_rhos = []
    for class_id in range(NUM_CLASSES):
        for environment in range(len(levels)):
            row = {
                "class_id": class_id,
                "environment": environment,
                "count": int(counts_matrix[environment, class_id]),
                "accuracy": float(class_accuracies[environment][class_id]),
            }
            row.update(
                {
                    name: float(
                        test_descriptors[environment][
                            class_id, DESCRIPTOR_INDEX[name]
                        ]
                    )
                    for name in DESCRIPTOR_NAMES
                }
            )
            rows.append(row)

        class_counts = counts_matrix[:, class_id].astype(np.float64)
        accuracy_values = np.array(
            [values[class_id] for values in class_accuracies]
        )
        accuracy_rhos.append(
            safe_spearman(np.log(class_counts), accuracy_values)
        )
        for name in DESCRIPTOR_NAMES:
            values = np.array(
                [
                    matrix[class_id, DESCRIPTOR_INDEX[name]]
                    for matrix in test_descriptors
                ]
            )
            per_metric_rhos[name].append(
                safe_spearman(np.log(class_counts), values)
            )

    low_count = min(levels)
    high_count = max(levels)
    low_env = np.argmax(counts_matrix == low_count, axis=0)
    high_env = np.argmax(counts_matrix == high_count, axis=0)
    minmax = {}
    for name in DESCRIPTOR_NAMES:
        index = DESCRIPTOR_INDEX[name]
        low_values = np.array(
            [
                test_descriptors[low_env[class_id]][class_id, index]
                for class_id in range(NUM_CLASSES)
            ]
        )
        high_values = np.array(
            [
                test_descriptors[high_env[class_id]][class_id, index]
                for class_id in range(NUM_CLASSES)
            ]
        )
        delta = high_values - low_values
        minmax[name] = {
            "low_mean": float(low_values.mean()),
            "high_mean": float(high_values.mean()),
            "delta_high_minus_low": float(delta.mean()),
            "high_greater_fraction": float(np.mean(delta > 0.0)),
            "wilcoxon": safe_wilcoxon(delta),
            "mean_within_class_spearman_log_count": float(
                np.mean(per_metric_rhos[name])
            ),
            "median_within_class_spearman_log_count": float(
                np.median(per_metric_rhos[name])
            ),
        }

    accuracy_low = np.array(
        [
            class_accuracies[low_env[class_id]][class_id]
            for class_id in range(NUM_CLASSES)
        ]
    )
    accuracy_high = np.array(
        [
            class_accuracies[high_env[class_id]][class_id]
            for class_id in range(NUM_CLASSES)
        ]
    )
    accuracy_delta = accuracy_high - accuracy_low

    prototype_drift = []
    same_image_drift = []
    for class_id in range(NUM_CLASSES):
        low_features = aligned_test_features[low_env[class_id]][
            test_labels == class_id
        ]
        high_features = aligned_test_features[high_env[class_id]][
            test_labels == class_id
        ]
        low_prototype = normalize_vector(low_features.mean(axis=0))
        high_prototype = normalize_vector(high_features.mean(axis=0))
        prototype_drift.append(1.0 - low_prototype @ high_prototype)
        same_image_drift.append(
            np.mean(1.0 - np.sum(low_features * high_features, axis=1))
        )

    summary = {
        "low_count": low_count,
        "high_count": high_count,
        "accuracy": {
            "low_mean": float(accuracy_low.mean()),
            "high_mean": float(accuracy_high.mean()),
            "delta_high_minus_low": float(accuracy_delta.mean()),
            "high_better_fraction": float(np.mean(accuracy_delta > 0.0)),
            "wilcoxon": safe_wilcoxon(accuracy_delta),
            "mean_within_class_spearman_log_count": float(
                np.mean(accuracy_rhos)
            ),
        },
        "prototype_drift_mean": float(np.mean(prototype_drift)),
        "same_image_feature_drift_mean": float(np.mean(same_image_drift)),
        "descriptors": minmax,
    }
    return summary, rows


def build_pair_dataset(
    source_descriptors: list[np.ndarray],
    target_descriptors: list[np.ndarray],
    counts_matrix: np.ndarray,
    levels: list[int],
) -> dict[str, np.ndarray]:
    class_ids = []
    source_counts = []
    target_counts = []
    source_rows = []
    target_rows = []
    environments = []
    for class_id in range(NUM_CLASSES):
        environment_for_count = {
            int(counts_matrix[environment, class_id]): environment
            for environment in range(len(levels))
        }
        for source_count, target_count in combinations(levels, 2):
            source_environment = environment_for_count[source_count]
            target_environment = environment_for_count[target_count]
            class_ids.append(class_id)
            source_counts.append(source_count)
            target_counts.append(target_count)
            source_rows.append(source_descriptors[source_environment][class_id])
            target_rows.append(target_descriptors[target_environment][class_id])
            environments.append((source_environment, target_environment))
    return {
        "class_id": np.asarray(class_ids, dtype=np.int64),
        "source_count": np.asarray(source_counts, dtype=np.int64),
        "target_count": np.asarray(target_counts, dtype=np.int64),
        "source": np.asarray(source_rows, dtype=np.float64),
        "target": np.asarray(target_rows, dtype=np.float64),
        "environment": np.asarray(environments, dtype=np.int64),
    }


def count_features(
    source_count: np.ndarray, target_count: np.ndarray
) -> np.ndarray:
    source = source_count.astype(np.float64)
    target = target_count.astype(np.float64)
    return np.stack(
        [
            np.log(source / MAX_CLASS_COUNT),
            np.log(target / MAX_CLASS_COUNT),
            np.log(target / source),
        ],
        axis=1,
    )


def scaled_ridge_prediction(
    train_x: np.ndarray,
    train_residual: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(train_residual)
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    regressor.fit(train_x, scaled_target)
    return target_scaler.inverse_transform(regressor.predict(test_x))


def extra_trees_prediction(
    train_x: np.ndarray,
    train_residual: np.ndarray,
    test_x: np.ndarray,
    trees: int,
    min_leaf: int,
    seed: int,
) -> np.ndarray:
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(train_residual)
    regressor = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=min_leaf,
        max_features=0.8,
        random_state=seed,
        n_jobs=-1,
    )
    regressor.fit(train_x, scaled_target)
    return target_scaler.inverse_transform(regressor.predict(test_x))


def group_errors(
    prediction: np.ndarray, target: np.ndarray, scale: np.ndarray
) -> dict[str, np.ndarray]:
    normalized_absolute_error = np.abs(prediction - target) / scale[None, :]
    groups = {
        "all": np.arange(len(DESCRIPTOR_NAMES)),
        "angle": np.array(
            [DESCRIPTOR_INDEX[name] for name in ANGLE_NAMES], dtype=np.int64
        ),
        "shape": np.array(
            [DESCRIPTOR_INDEX[name] for name in SHAPE_NAMES], dtype=np.int64
        ),
        "relation": np.array(
            [DESCRIPTOR_INDEX[name] for name in RELATION_NAMES], dtype=np.int64
        ),
    }
    return {
        group: normalized_absolute_error[:, indices].mean(axis=1)
        for group, indices in groups.items()
    }


def evaluate_recovery(
    dataset: dict[str, np.ndarray],
    folds: int,
    alpha: float,
    trees: int,
    min_leaf: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    class_ids = dataset["class_id"]
    source = dataset["source"]
    target = dataset["target"]
    source_count = dataset["source_count"]
    target_count = dataset["target_count"]
    unique_classes = np.arange(NUM_CLASSES)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)

    prediction_records = []
    fold_summaries = []
    model_names = [
        "identity",
        "count_only",
        "prototype_ridge",
        "cap_ridge",
        "full_support_no_count_ridge",
        "full_support_ridge",
        "full_support_extra_trees",
    ]
    all_model_errors: dict[str, dict[str, list[np.ndarray]]] = {
        model: {group: [] for group in ["all", "angle", "shape", "relation"]}
        for model in model_names
    }
    all_model_minmax: dict[str, dict[str, list[np.ndarray]]] = {
        model: {group: [] for group in ["all", "angle", "shape", "relation"]}
        for model in model_names
    }
    all_identity_errors = []
    all_identity_minmax = []

    prototype_indices = np.array(
        [
            DESCRIPTOR_INDEX["neighbor_cos_1"],
            DESCRIPTOR_INDEX["neighbor_cos_3mean"],
            DESCRIPTOR_INDEX["neighbor_cos_5mean"],
        ],
        dtype=np.int64,
    )
    cap_indices = np.concatenate(
        [
            np.array([DESCRIPTOR_INDEX["angle_q95"]], dtype=np.int64),
            prototype_indices,
        ]
    )

    for fold_id, (train_positions, test_positions) in enumerate(
        splitter.split(unique_classes)
    ):
        train_classes = unique_classes[train_positions]
        test_classes = unique_classes[test_positions]
        train_mask = np.isin(class_ids, train_classes)
        test_mask = np.isin(class_ids, test_classes)

        train_source = source[train_mask]
        train_target = target[train_mask]
        test_source = source[test_mask]
        test_target = target[test_mask]
        train_source_count = source_count[train_mask]
        train_target_count = target_count[train_mask]
        test_source_count = source_count[test_mask]
        test_target_count = target_count[test_mask]
        train_condition = count_features(
            train_source_count, train_target_count
        )
        test_condition = count_features(test_source_count, test_target_count)
        residual = train_target - train_source
        scale = np.maximum(train_target.std(axis=0, ddof=1), 1e-3)

        count_prediction = np.zeros_like(test_target)
        for current_source in np.unique(test_source_count):
            for current_target in np.unique(test_target_count):
                current_test = (
                    (test_source_count == current_source)
                    & (test_target_count == current_target)
                )
                if not np.any(current_test):
                    continue
                current_train = (
                    (train_source_count == current_source)
                    & (train_target_count == current_target)
                )
                mean_residual = residual[current_train].mean(axis=0)
                count_prediction[current_test] = (
                    test_source[current_test] + mean_residual
                )

        prototype_train_x = np.concatenate(
            [train_source[:, prototype_indices], train_condition], axis=1
        )
        prototype_test_x = np.concatenate(
            [test_source[:, prototype_indices], test_condition], axis=1
        )
        cap_train_x = np.concatenate(
            [train_source[:, cap_indices], train_condition], axis=1
        )
        cap_test_x = np.concatenate(
            [test_source[:, cap_indices], test_condition], axis=1
        )
        full_train_x = np.concatenate(
            [train_source, train_condition], axis=1
        )
        full_test_x = np.concatenate([test_source, test_condition], axis=1)

        predictions = {
            "identity": test_source.copy(),
            "count_only": count_prediction,
            "prototype_ridge": test_source
            + scaled_ridge_prediction(
                prototype_train_x,
                residual,
                prototype_test_x,
                alpha,
            ),
            "cap_ridge": test_source
            + scaled_ridge_prediction(
                cap_train_x, residual, cap_test_x, alpha
            ),
            "full_support_no_count_ridge": test_source
            + scaled_ridge_prediction(
                train_source, residual, test_source, alpha
            ),
            "full_support_ridge": test_source
            + scaled_ridge_prediction(
                full_train_x, residual, full_test_x, alpha
            ),
            "full_support_extra_trees": test_source
            + extra_trees_prediction(
                full_train_x,
                residual,
                full_test_x,
                trees=trees,
                min_leaf=min_leaf,
                seed=seed + fold_id,
            ),
        }
        errors = {
            name: group_errors(prediction, test_target, scale)
            for name, prediction in predictions.items()
        }
        identity_all = errors["identity"]["all"]
        minmax_mask = (
            (test_source_count == test_source_count.min())
            & (test_target_count == test_target_count.max())
        )
        all_identity_errors.append(identity_all)
        all_identity_minmax.append(identity_all[minmax_mask])

        fold_row = {
            "fold": fold_id,
            "train_classes": train_classes.tolist(),
            "test_classes": test_classes.tolist(),
            "num_test_pairs": int(test_mask.sum()),
            "num_minmax_pairs": int(minmax_mask.sum()),
            "models": {},
        }
        for model_name in model_names:
            fold_row["models"][model_name] = {}
            for group, values in errors[model_name].items():
                all_model_errors[model_name][group].append(values)
                all_model_minmax[model_name][group].append(values[minmax_mask])
                fold_row["models"][model_name][group] = float(values.mean())
                fold_row["models"][model_name][f"{group}_minmax"] = float(
                    values[minmax_mask].mean()
                )

        test_indices = np.flatnonzero(test_mask)
        for local_index, global_index in enumerate(test_indices):
            row = {
                "fold": fold_id,
                "class_id": int(class_ids[global_index]),
                "source_count": int(source_count[global_index]),
                "target_count": int(target_count[global_index]),
            }
            for model_name in model_names:
                row[f"{model_name}_nmae"] = float(
                    errors[model_name]["all"][local_index]
                )
            prediction_records.append(row)
        fold_summaries.append(fold_row)

    identity_error = np.concatenate(all_identity_errors)
    identity_minmax = np.concatenate(all_identity_minmax)
    aggregate = {}
    for model_name in model_names:
        model_result = {}
        for group in ["all", "angle", "shape", "relation"]:
            values = np.concatenate(all_model_errors[model_name][group])
            minmax_values = np.concatenate(
                all_model_minmax[model_name][group]
            )
            model_result[group] = {
                "nmae": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
            model_result[f"{group}_minmax"] = {
                "nmae": float(minmax_values.mean()),
                "std": float(minmax_values.std(ddof=1)),
            }
        model_all = np.concatenate(all_model_errors[model_name]["all"])
        model_minmax = np.concatenate(
            all_model_minmax[model_name]["all"]
        )
        model_result["relative_improvement_vs_identity"] = float(
            1.0 - model_all.mean() / max(identity_error.mean(), 1e-12)
        )
        model_result["better_fraction_vs_identity"] = float(
            np.mean(model_all < identity_error)
        )
        model_result["minmax_relative_improvement_vs_identity"] = float(
            1.0
            - model_minmax.mean() / max(identity_minmax.mean(), 1e-12)
        )
        model_result["minmax_better_fraction_vs_identity"] = float(
            np.mean(model_minmax < identity_minmax)
        )
        aggregate[model_name] = model_result

    return (
        {
            "descriptor_names": DESCRIPTOR_NAMES,
            "aggregate": aggregate,
            "folds": fold_summaries,
        },
        prediction_records,
    )


def build_report(seed_summaries: list[dict], output_dir: Path) -> None:
    mechanism_accuracy = np.array(
        [
            summary["mechanism"]["accuracy"]["delta_high_minus_low"]
            for summary in seed_summaries
        ]
    )
    angle_delta = np.array(
        [
            summary["mechanism"]["descriptors"]["angle_q95"][
                "delta_high_minus_low"
            ]
            for summary in seed_summaries
        ]
    )
    trace_delta = np.array(
        [
            summary["mechanism"]["descriptors"]["tangent_trace"][
                "delta_high_minus_low"
            ]
            for summary in seed_summaries
        ]
    )
    cap_overlap_delta = np.array(
        [
            summary["mechanism"]["descriptors"]["cap_overlap_fraction"][
                "delta_high_minus_low"
            ]
            for summary in seed_summaries
        ]
    )

    def recovery_mean(task: str, model: str, field: str) -> float:
        values = []
        for summary in seed_summaries:
            value = summary["recovery"][task]["aggregate"][model][field]
            if isinstance(value, dict):
                value = value["nmae"]
            values.append(value)
        return float(np.mean(values))

    primary_fewshot_improvement = recovery_mean(
        "fewshot_to_head",
        "full_support_ridge",
        "minmax_relative_improvement_vs_identity",
    )
    primary_fewshot_better = recovery_mean(
        "fewshot_to_head",
        "full_support_ridge",
        "minmax_better_fraction_vs_identity",
    )
    cap_nmae = recovery_mean(
        "fewshot_to_head", "cap_ridge", "all_minmax"
    )
    full_nmae = recovery_mean(
        "fewshot_to_head", "full_support_ridge", "all_minmax"
    )
    no_count_nmae = recovery_mean(
        "fewshot_to_head",
        "full_support_no_count_ridge",
        "all_minmax",
    )
    full_vs_cap = 1.0 - full_nmae / max(cap_nmae, 1e-12)
    frequency_condition_gain = (
        1.0 - full_nmae / max(no_count_nmae, 1e-12)
    )

    mechanism_supported = float(mechanism_accuracy.mean()) >= 0.05
    recovery_supported = (
        primary_fewshot_improvement >= 0.05
        and primary_fewshot_better >= 0.60
        and full_vs_cap > 0.0
        and frequency_condition_gain > 0.0
    )
    if mechanism_supported and recovery_supported:
        verdict = "支持：支撑动力学显著，且完整支撑描述提升未知尾类恢复。"
    elif mechanism_supported:
        verdict = "部分支持：支撑动力学显著，但当前恢复器尚未通过预设门槛。"
    else:
        verdict = "暂不支持：当前设置下频率造成的支撑变化不够稳定。"

    lines = [
        "# 反事实类别支撑动力学实验报告",
        "",
        f"**结论：{verdict}**",
        "",
        "## 设计",
        "",
        "- CIFAR-100 仅作为受控验证载体。",
        "- 五个环境具有完全相同的类别频率直方图；每类依次经历 5、20、50、200、500 个训练样本。",
        "- 小样本集合严格嵌套在大样本集合中。",
        "- 所有机制指标使用同一平衡测试集，避免支撑估计样本量的机械混淆。",
        "- 恢复器采用类别不相交五折验证。",
        "- fewshot_to_head 只用尾类实际可见的少量训练样本描述支撑，并预测 500 样本状态的固定测试支撑。",
        "",
        "## 机制结果：500 样本状态减去 5 样本状态",
        "",
        f"- 类别准确率：**{mechanism_accuracy.mean() * 100:+.2f} 个百分点**。",
        f"- 95% 角半径：**{angle_delta.mean():+.5f} 弧度**。",
        f"- 切空间协方差迹：**{trace_delta.mean():+.5f}**。",
        f"- 球面帽重叠比例：**{cap_overlap_delta.mean():+.5f}**。",
        "",
        "## 类别不相交恢复：5 → 500",
        "",
        "| 模型 | 固定评估支撑 NMAE | 少样本到头类 NMAE | 少样本相对 identity 改善 |",
        "|---|---:|---:|---:|",
    ]
    for model in [
        "identity",
        "count_only",
        "prototype_ridge",
        "cap_ridge",
        "full_support_no_count_ridge",
        "full_support_ridge",
        "full_support_extra_trees",
    ]:
        fixed_error = recovery_mean(
            "fixed_eval_dynamics", model, "all_minmax"
        )
        fewshot_error = recovery_mean(
            "fewshot_to_head", model, "all_minmax"
        )
        improvement = recovery_mean(
            "fewshot_to_head",
            model,
            "minmax_relative_improvement_vs_identity",
        )
        lines.append(
            f"| {model} | {fixed_error:.4f} | {fewshot_error:.4f} | "
            f"{improvement * 100:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 预设门槛",
            "",
            "- 机制：5→500 的类别准确率平均提升至少 5 个百分点。",
            "- 恢复：full_support_ridge 在 fewshot 5→500 上相对 identity 的 NMAE 至少下降 5%，至少 60% 类别改善，并同时优于 cap_ridge 和不输入频率的完整支撑模型。",
            "",
            f"- full_support_ridge 改善比例：**{primary_fewshot_improvement * 100:+.2f}%**。",
            f"- 改善类别比例：**{primary_fewshot_better * 100:.2f}%**。",
            f"- 相对 cap_ridge 改善：**{full_vs_cap * 100:+.2f}%**。",
            f"- 输入频率相对不输入频率的改善：**{frequency_condition_gain * 100:+.2f}%**。",
            "",
            "逐类—频率支撑描述见 `support_states.csv`，逐对恢复误差见各 seed 目录中的 CSV。",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_seed(
    seed: int,
    raw_train,
    raw_test,
    class_orders: list[np.ndarray],
    counts_matrix: np.ndarray,
    levels: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    test_targets = np.asarray(raw_test.targets, dtype=np.int64)
    test_dataset = ArrayCIFAR100(
        raw_test.data,
        test_targets,
        None,
        eval_transform(),
        return_index=True,
    )
    test_loader = DataLoader(
        ArrayCIFAR100(
            raw_test.data, test_targets, None, eval_transform()
        ),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    anchor_indices = common_anchor_indices(class_orders, levels)
    anchor_dataset = ArrayCIFAR100(
        raw_train.data,
        train_targets,
        anchor_indices,
        eval_transform(),
        return_index=True,
    )

    checkpoints = []
    selected_by_environment = []
    for environment in range(len(levels)):
        selected = select_indices(
            class_orders, counts_matrix[environment]
        )
        selected_by_environment.append(selected)
        train_dataset = ArrayCIFAR100(
            raw_train.data,
            train_targets,
            selected,
            train_transform(),
        )
        checkpoints.append(
            train_environment(
                environment=environment,
                seed=seed,
                counts=counts_matrix[environment],
                train_dataset=train_dataset,
                test_loader=test_loader,
                seed_dir=seed_dir,
                args=args,
                device=device,
            )
        )

    environment_states = []
    for environment, checkpoint in enumerate(checkpoints):
        model = load_model(checkpoint, device)
        selected_eval_dataset = ArrayCIFAR100(
            raw_train.data,
            train_targets,
            selected_by_environment[environment],
            eval_transform(),
            return_index=True,
        )
        test_features, test_logits, labels, _ = extract_features(
            model, test_dataset, args, device
        )
        train_features, train_logits, train_labels, _ = extract_features(
            model, selected_eval_dataset, args, device
        )
        anchor_features, _, anchor_labels, _ = extract_features(
            model, anchor_dataset, args, device
        )
        test_support = describe_supports(
            test_features, labels, test_logits
        )
        train_support = describe_supports(
            train_features, train_labels, train_logits
        )
        environment_states.append(
            {
                "test_features": l2_normalize(test_features.astype(np.float64)),
                "test_labels": labels,
                "anchor_features": l2_normalize(
                    anchor_features.astype(np.float64)
                ),
                "anchor_labels": anchor_labels,
                "test_support": test_support,
                "train_support": train_support,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reference_anchor = environment_states[0]["anchor_features"]
    aligned_test_features = []
    alignment_rows = []
    for environment, state in enumerate(environment_states):
        if environment == 0:
            rotation = np.eye(reference_anchor.shape[1], dtype=np.float64)
        else:
            rotation = orthogonal_procrustes(
                state["anchor_features"], reference_anchor
            )
        aligned_anchor = l2_normalize(
            state["anchor_features"] @ rotation
        )
        aligned_test = l2_normalize(state["test_features"] @ rotation)
        aligned_test_features.append(aligned_test)
        alignment_rows.append(
            {
                "environment": environment,
                "num_anchor_images": int(len(reference_anchor)),
                "same_image_cosine_after": float(
                    np.mean(np.sum(reference_anchor * aligned_anchor, axis=1))
                ),
            }
        )

    test_descriptors = [
        state["test_support"].descriptors for state in environment_states
    ]
    train_descriptors = [
        state["train_support"].descriptors for state in environment_states
    ]
    class_accuracies = [
        state["test_support"].class_accuracy for state in environment_states
    ]
    mechanism, support_rows = summarize_mechanism(
        test_descriptors=test_descriptors,
        aligned_test_features=aligned_test_features,
        test_labels=environment_states[0]["test_labels"],
        class_accuracies=class_accuracies,
        counts_matrix=counts_matrix,
        levels=levels,
    )
    write_csv(support_rows, seed_dir / "support_states.csv")
    write_csv(alignment_rows, seed_dir / "alignment.csv")

    fixed_dataset = build_pair_dataset(
        source_descriptors=test_descriptors,
        target_descriptors=test_descriptors,
        counts_matrix=counts_matrix,
        levels=levels,
    )
    fewshot_dataset = build_pair_dataset(
        source_descriptors=train_descriptors,
        target_descriptors=test_descriptors,
        counts_matrix=counts_matrix,
        levels=levels,
    )
    fixed_recovery, fixed_rows = evaluate_recovery(
        fixed_dataset,
        folds=args.recovery_folds,
        alpha=args.ridge_alpha,
        trees=args.trees,
        min_leaf=args.tree_min_leaf,
        seed=seed,
    )
    fewshot_recovery, fewshot_rows = evaluate_recovery(
        fewshot_dataset,
        folds=args.recovery_folds,
        alpha=args.ridge_alpha,
        trees=args.trees,
        min_leaf=args.tree_min_leaf,
        seed=seed + 10000,
    )
    write_csv(
        fixed_rows, seed_dir / "fixed_eval_recovery_pairs.csv"
    )
    write_csv(
        fewshot_rows, seed_dir / "fewshot_recovery_pairs.csv"
    )
    summary = {
        "seed": seed,
        "levels": levels,
        "alignment": alignment_rows,
        "mechanism": mechanism,
        "recovery": {
            "fixed_eval_dynamics": fixed_recovery,
            "fewshot_to_head": fewshot_recovery,
        },
    }
    save_json(summary, seed_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    args.levels = sorted(args.levels)
    if args.smoke:
        args.epochs = 1
        args.num_workers = 0
        args.trees = min(args.trees, 40)
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), args.output_dir / "args.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
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
    class_orders = make_nested_class_orders(
        train_targets, subset_seed=args.subset_seed
    )
    counts_matrix, class_groups = make_latin_frequency_design(
        args.levels, design_seed=args.design_seed
    )
    save_json(
        {
            "levels": args.levels,
            "counts_matrix": counts_matrix,
            "class_groups": class_groups,
            "environment_totals": counts_matrix.sum(axis=1),
            "identical_histograms": all(
                np.array_equal(
                    np.sort(counts_matrix[0]),
                    np.sort(counts_matrix[environment]),
                )
                for environment in range(len(args.levels))
            ),
            "every_class_visits_every_level": all(
                np.array_equal(
                    np.sort(counts_matrix[:, class_id]),
                    np.asarray(args.levels),
                )
                for class_id in range(NUM_CLASSES)
            ),
        },
        args.output_dir / "frequency_design.json",
    )
    print(
        f"Frequency design: levels={args.levels}, "
        f"environments={len(args.levels)}, "
        f"total/environment={counts_matrix[0].sum()}"
    )

    summaries = []
    for seed in args.seeds:
        summaries.append(
            run_seed(
                seed=seed,
                raw_train=raw_train,
                raw_test=raw_test,
                class_orders=class_orders,
                counts_matrix=counts_matrix,
                levels=args.levels,
                args=args,
                device=device,
            )
        )
    save_json(summaries, args.output_dir / "all_seed_summaries.json")
    build_report(summaries, args.output_dir)
    print(f"Finished. Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
