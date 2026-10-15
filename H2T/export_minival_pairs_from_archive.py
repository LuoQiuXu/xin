#!/usr/bin/env python
"""Convert an existing fixed-feature archive into minival recipient/donor pairs.

The source archive is expected to contain the fixed CIFAR100-LT Stage-I
features already produced by the local HAFE experiments.  No network or
backbone execution is needed:

* recipients are class-balanced samples from the LT training feature pool,
* donors are instance-uniform samples from the same pool (therefore head-biased),
* the Stage-I linear head is kept frozen as the reference classifier,
* 50 balanced test features per class train an independent diagnostic probe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REQUIRED_KEYS = (
    "train_features",
    "train_labels",
    "test_features",
    "test_labels",
    "classifier_weight",
    "classifier_bias",
)


def balanced_subset_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    chosen: list[np.ndarray] = []
    for class_id in classes:
        indices = np.flatnonzero(labels == class_id)
        if indices.size < per_class:
            raise ValueError(
                f"class {int(class_id)} has {indices.size} samples, fewer than {per_class}"
            )
        chosen.append(rng.choice(indices, size=per_class, replace=False))
    return np.concatenate(chosen)


def sample_pairs(
    features: np.ndarray,
    labels: np.ndarray,
    dataset_indices: np.ndarray,
    num_pairs: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    expected = np.arange(classes.size)
    if not np.array_equal(classes, expected):
        raise ValueError("training labels must be contiguous [0, C)")
    by_class = [np.flatnonzero(labels == class_id) for class_id in classes]

    recipient_labels = rng.integers(0, classes.size, size=num_pairs)
    recipient_pool_indices = np.fromiter(
        (rng.choice(by_class[class_id]) for class_id in recipient_labels),
        dtype=np.int64,
        count=num_pairs,
    )
    donor_pool_indices = rng.integers(0, features.shape[0], size=num_pairs)
    return {
        "recipient_features": features[recipient_pool_indices],
        "recipient_labels": labels[recipient_pool_indices],
        "recipient_dataset_indices": dataset_indices[recipient_pool_indices],
        "donor_features": features[donor_pool_indices],
        "donor_labels": labels[donor_pool_indices],
        "donor_dataset_indices": dataset_indices[donor_pool_indices],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed H2TF minival pairs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=30000)
    parser.add_argument("--probe-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_pairs <= 0 or args.probe_per_class <= 0:
        raise ValueError("--num-pairs and --probe-per-class must be positive")

    with np.load(args.input, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"source archive is missing keys: {missing}")
        source = {key: archive[key] for key in archive.files}

    train_features = np.asarray(source["train_features"], dtype=np.float32)
    train_labels = np.asarray(source["train_labels"], dtype=np.int64)
    test_features = np.asarray(source["test_features"], dtype=np.float32)
    test_labels = np.asarray(source["test_labels"], dtype=np.int64)
    reference_weight = np.asarray(source["classifier_weight"], dtype=np.float32)
    reference_bias = np.asarray(source["classifier_bias"], dtype=np.float32)

    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("train_features and test_features must be matrices")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("train and test feature dimensions differ")
    if train_labels.shape != (train_features.shape[0],):
        raise ValueError("train_labels shape does not match train_features")
    if test_labels.shape != (test_features.shape[0],):
        raise ValueError("test_labels shape does not match test_features")
    num_classes = int(np.unique(train_labels).size)
    if reference_weight.shape != (num_classes, train_features.shape[1]):
        raise ValueError("classifier_weight has an incompatible shape")
    if reference_bias.shape != (num_classes,):
        raise ValueError("classifier_bias has an incompatible shape")

    if "train_selected_indices" in source:
        dataset_indices = np.asarray(source["train_selected_indices"], dtype=np.int64)
    else:
        dataset_indices = np.arange(train_features.shape[0], dtype=np.int64)
    if dataset_indices.shape != (train_features.shape[0],):
        raise ValueError("train_selected_indices has an incompatible shape")

    probe_indices = balanced_subset_indices(test_labels, args.probe_per_class, args.seed)
    pairs = sample_pairs(
        train_features,
        train_labels,
        dataset_indices,
        args.num_pairs,
        args.seed,
    )

    reference_logits = train_features @ reference_weight.T + reference_bias
    reference_accuracy = float((reference_logits.argmax(axis=1) == train_labels).mean())
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **pairs,
        reference_weight=reference_weight,
        reference_bias=reference_bias,
        class_centers=reference_weight,
        probe_train_features=test_features[probe_indices],
        probe_train_labels=test_labels[probe_indices],
        long_tailed_class_counts=class_counts,
    )
    print(
        f"source train={train_features.shape}, test={test_features.shape}, "
        f"reference train accuracy={reference_accuracy:.4f}"
    )
    print(
        f"pairs={args.num_pairs}, probe={probe_indices.size}, "
        f"class counts={class_counts.max()}->{class_counts.min()}"
    )
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
