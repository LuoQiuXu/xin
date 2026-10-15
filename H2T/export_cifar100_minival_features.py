#!/usr/bin/env python
"""Export fixed CIFAR-100-LT feature pairs for ``fusion_risk_minival.py``.

The export reproduces the Stage-II sampling roles without training anything:

* recipient: class-balanced sampling from the long-tailed training subset,
* donor: instance-uniform sampling from that subset (therefore head-biased),
* reference head: the frozen Stage-I classifier from the checkpoint,
* independent probe data: a balanced subset of CIFAR-100's test images.

The balanced test subset is used only to train a diagnostic probe for this
go/no-go experiment.  It must not be reused as the final paper test protocol;
replace it with a dedicated validation split before reporting final results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import resnet_cifar


def strip_parallel_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_stage1(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = ["state_dict_model", "state_dict_classifier"]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint is missing keys: {missing}")

    backbone = resnet_cifar.resnet32_fe()
    classifier = resnet_cifar.Classifier(feat_in=64, num_classes=100)
    backbone.load_state_dict(strip_parallel_prefix(checkpoint["state_dict_model"]), strict=True)
    classifier.load_state_dict(
        strip_parallel_prefix(checkpoint["state_dict_classifier"]), strict=True
    )
    backbone.to(device).eval()
    classifier.to(device).eval()
    return backbone, classifier, checkpoint


def long_tailed_indices(
    targets: np.ndarray,
    imbalance_factor: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Match IMBALANCECIFAR100.gen_imbalanced_data's legacy NumPy sampling."""
    num_classes = 100
    image_max = targets.size / num_classes
    counts = np.asarray(
        [
            int(image_max * (imbalance_factor ** (class_id / (num_classes - 1.0))))
            for class_id in range(num_classes)
        ],
        dtype=np.int64,
    )
    rng = np.random.RandomState(seed)
    selected: list[np.ndarray] = []
    for class_id, count in enumerate(counts):
        indices = np.flatnonzero(targets == class_id)
        rng.shuffle(indices)
        selected.append(indices[:count])
    return np.concatenate(selected), counts


@torch.inference_mode()
def extract_features(
    backbone: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    indices: np.ndarray,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        feature_map = backbone(images)
        feature = F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
        feature_batches.append(feature.cpu().numpy())
        label_batches.append(labels.numpy())
    return np.concatenate(feature_batches), np.concatenate(label_batches)


def balanced_subset_indices(targets: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in range(100):
        indices = np.flatnonzero(targets == class_id)
        if indices.size < per_class:
            raise ValueError(
                f"class {class_id} has only {indices.size} images, fewer than {per_class}"
            )
        selected.append(rng.choice(indices, size=per_class, replace=False))
    return np.concatenate(selected)


def sample_pairs(
    pool_features: np.ndarray,
    pool_labels: np.ndarray,
    pool_dataset_indices: np.ndarray,
    num_pairs: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(pool_labels == class_id) for class_id in range(100)]
    if any(indices.size == 0 for indices in by_class):
        raise ValueError("long-tailed pool contains an empty class")

    recipient_labels = rng.integers(0, 100, size=num_pairs)
    recipient_pool_indices = np.fromiter(
        (rng.choice(by_class[class_id]) for class_id in recipient_labels),
        dtype=np.int64,
        count=num_pairs,
    )
    donor_pool_indices = rng.integers(0, pool_features.shape[0], size=num_pairs)
    return {
        "recipient_features": pool_features[recipient_pool_indices],
        "recipient_labels": pool_labels[recipient_pool_indices],
        "recipient_dataset_indices": pool_dataset_indices[recipient_pool_indices],
        "donor_features": pool_features[donor_pool_indices],
        "donor_labels": pool_labels[donor_pool_indices],
        "donor_dataset_indices": pool_dataset_indices[donor_pool_indices],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export fixed H2T CIFAR-100-LT pairs for offline fusion-risk validation"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--imbalance-factor", type=float, default=0.01)
    parser.add_argument("--num-pairs", type=int, default=30000)
    parser.add_argument("--probe-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.imbalance_factor <= 1.0:
        raise ValueError("--imbalance-factor must lie in (0, 1]")
    if args.num_pairs <= 0 or args.probe_per_class <= 0:
        raise ValueError("--num-pairs and --probe-per-class must be positive")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = datasets.CIFAR100(
        root=args.data_root,
        train=True,
        transform=transform,
        download=args.download,
    )
    test_dataset = datasets.CIFAR100(
        root=args.data_root,
        train=False,
        transform=transform,
        download=args.download,
    )

    backbone, classifier, checkpoint = load_stage1(args.checkpoint, device)
    train_targets = np.asarray(train_dataset.targets, dtype=np.int64)
    lt_indices, class_counts = long_tailed_indices(
        train_targets, args.imbalance_factor, seed=0
    )
    print(
        f"Extracting {lt_indices.size} long-tailed train features on {device}; "
        f"class counts {class_counts.max()} -> {class_counts.min()}"
    )
    pool_features, pool_labels = extract_features(
        backbone,
        train_dataset,
        lt_indices,
        args.batch_size,
        args.workers,
        device,
    )

    test_targets = np.asarray(test_dataset.targets, dtype=np.int64)
    probe_indices = balanced_subset_indices(test_targets, args.probe_per_class, args.seed)
    print(f"Extracting {probe_indices.size} balanced diagnostic-probe features")
    probe_features, probe_labels = extract_features(
        backbone,
        test_dataset,
        probe_indices,
        args.batch_size,
        args.workers,
        device,
    )

    pairs = sample_pairs(
        pool_features,
        pool_labels,
        lt_indices,
        args.num_pairs,
        args.seed,
    )
    fc = classifier.fc
    reference_weight = fc.weight.detach().cpu().numpy()
    reference_bias = (
        fc.bias.detach().cpu().numpy()
        if fc.bias is not None
        else np.zeros(reference_weight.shape[0], dtype=np.float32)
    )

    with torch.inference_mode():
        pool_logits = pool_features @ reference_weight.T + reference_bias
        pool_accuracy = float((pool_logits.argmax(axis=1) == pool_labels).mean())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **pairs,
        reference_weight=reference_weight,
        reference_bias=reference_bias,
        class_centers=reference_weight,
        probe_train_features=probe_features,
        probe_train_labels=probe_labels,
        long_tailed_class_counts=class_counts,
        checkpoint_epoch=np.asarray(checkpoint.get("epoch", -1)),
    )
    print(f"Frozen reference accuracy on LT feature pool: {pool_accuracy:.4f}")
    print(f"Wrote {args.num_pairs} fixed pairs to {args.output.resolve()}")
    print(
        "Diagnostic warning: the probe uses balanced CIFAR-100 test images. "
        "Use a dedicated validation split for final reported experiments."
    )


if __name__ == "__main__":
    main()
