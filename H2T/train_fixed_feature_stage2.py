#!/usr/bin/env python
"""Classifier-only Stage-II validation for boundary-controlled feature fusion.

This experiment is the causal bridge between the fixed-feature minival and a
full PI-H2T run.  Every method shares:

* the same frozen Stage-I features and classifier initialization,
* the same class-balanced recipient / instance-wise donor pair schedule,
* the official CIFAR100-LT Stage-II optimizer and cosine schedule,
* the same number of updates.

The frozen reference head is used only to compute fusion geometry.  A separate
classifier is optimized on the fused features with the recipient label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fusion_risk_minival import (
    cosine_center_distance,
    distribution_matched_recipient_ratio,
    donor_conditioned_critical_ratio,
    pairwise_boundary_distances,
    pi_h2t_donor_ratio,
)


BASE_METHODS = (
    "nofusion",
    "fixed_matched",
    "pi_h2t_center_native",
    "pi_h2t_center_matched",
    "recipient_boundary_native",
    "recipient_boundary_matched",
)


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def gamma_method_name(gamma: float) -> str:
    return f"donor_conditioned_g{int(round(100 * gamma)):03d}"


def load_feature_archive(path: Path) -> dict[str, np.ndarray]:
    required = (
        "train_features",
        "train_labels",
        "test_features",
        "test_labels",
        "classifier_weight",
        "classifier_bias",
        "class_counts",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(f"feature archive is missing keys: {missing}")
        data = {key: archive[key] for key in archive.files}

    train_features = np.asarray(data["train_features"], dtype=np.float32)
    test_features = np.asarray(data["test_features"], dtype=np.float32)
    train_labels = np.asarray(data["train_labels"], dtype=np.int64)
    test_labels = np.asarray(data["test_labels"], dtype=np.int64)
    weight = np.asarray(data["classifier_weight"], dtype=np.float32)
    bias = np.asarray(data["classifier_bias"], dtype=np.float32)
    counts = np.asarray(data["class_counts"], dtype=np.int64)
    num_classes = weight.shape[0]

    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("train_features and test_features must be matrices")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("train and test feature dimensions differ")
    if train_labels.shape != (train_features.shape[0],):
        raise ValueError("train_labels shape does not match train_features")
    if test_labels.shape != (test_features.shape[0],):
        raise ValueError("test_labels shape does not match test_features")
    if weight.shape != (num_classes, train_features.shape[1]):
        raise ValueError("classifier_weight has an incompatible shape")
    if bias.shape != (num_classes,) or counts.shape != (num_classes,):
        raise ValueError("classifier_bias or class_counts has an incompatible shape")
    if not np.array_equal(np.unique(train_labels), np.arange(num_classes)):
        raise ValueError("training labels must be contiguous [0, C)")

    return {
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels,
        "reference_weight": weight,
        "reference_bias": bias,
        "class_counts": counts,
    }


def make_pair_schedule(
    labels: np.ndarray,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_samples = labels.size
    num_classes = int(labels.max()) + 1
    total = epochs * num_samples
    by_class = [np.flatnonzero(labels == class_id) for class_id in range(num_classes)]
    recipient_classes = rng.integers(0, num_classes, size=total)
    recipient_indices = np.fromiter(
        (rng.choice(by_class[class_id]) for class_id in recipient_classes),
        dtype=np.int64,
        count=total,
    )
    donor_indices = rng.integers(0, num_samples, size=total, dtype=np.int64)
    return recipient_indices, donor_indices, recipient_classes.astype(np.int64)


def compute_geometry(
    train_features: np.ndarray,
    labels: np.ndarray,
    recipient_indices: np.ndarray,
    donor_indices: np.ndarray,
    reference_weight: np.ndarray,
    reference_bias: np.ndarray,
    lambda_cap: float,
    delta: float,
    chunk_size: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recipient = train_features[recipient_indices]
    recipient_labels = labels[recipient_indices]
    center_distance = cosine_center_distance(
        recipient, recipient_labels, reference_weight, eps
    )
    center_ratio = pi_h2t_donor_ratio(center_distance, lambda_cap)

    margins = np.empty(recipient_indices.size, dtype=np.float64)
    critical = np.empty(recipient_indices.size, dtype=np.float64)
    for start in range(0, recipient_indices.size, chunk_size):
        stop = min(start + chunk_size, recipient_indices.size)
        batch_labels = recipient_labels[start:stop]
        recipient_distance = pairwise_boundary_distances(
            train_features[recipient_indices[start:stop]],
            batch_labels,
            reference_weight,
            reference_bias,
            eps,
        )
        donor_distance = pairwise_boundary_distances(
            train_features[donor_indices[start:stop]],
            batch_labels,
            reference_weight,
            reference_bias,
            eps,
        )
        batch_critical, batch_margin, _ = donor_conditioned_critical_ratio(
            recipient_distance, donor_distance, delta, eps
        )
        critical[start:stop] = batch_critical
        margins[start:stop] = batch_margin

    donor_base = np.minimum(critical, lambda_cap)
    recipient_ratio = distribution_matched_recipient_ratio(margins, center_ratio)
    return (
        center_ratio.astype(np.float32),
        recipient_ratio.astype(np.float32),
        donor_base.astype(np.float32),
    )


def match_ratio_mean(base: np.ndarray, target: float, cap: float) -> np.ndarray:
    if target < 0.0 or target > cap + 1e-8:
        raise ValueError(f"target mean {target} lies outside [0, {cap}]")
    if target <= 0.0:
        return np.zeros_like(base, dtype=np.float32)
    if np.all(base <= 0.0):
        raise ValueError("cannot match a positive mean from an all-zero base ratio")

    low = 0.0
    high = max(1.0, target / max(float(base.mean()), 1e-12))
    while float(np.minimum(high * base, cap).mean()) < target:
        high *= 2.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        mean = float(np.minimum(middle * base, cap).mean())
        if mean < target:
            low = middle
        else:
            high = middle
    return np.minimum(high * base, cap).astype(np.float32)


def match_ratio_mean_within_strata(
    base: np.ndarray,
    strata: np.ndarray,
    target: float,
    cap: float,
) -> np.ndarray:
    matched = np.empty_like(base, dtype=np.float32)
    for stratum in np.unique(strata):
        mask = strata == stratum
        matched[mask] = match_ratio_mean(base[mask], target, cap)
    return matched


def shuffle_within_recipient_class_and_epoch(
    ratios: np.ndarray,
    recipient_classes: np.ndarray,
    samples_per_epoch: int,
    epochs: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = ratios.copy()
    num_classes = int(recipient_classes.max()) + 1
    for epoch in range(epochs):
        start = epoch * samples_per_epoch
        stop = start + samples_per_epoch
        epoch_classes = recipient_classes[start:stop]
        for class_id in range(num_classes):
            local = np.flatnonzero(epoch_classes == class_id) + start
            if local.size > 1:
                shuffled[local] = ratios[rng.permutation(local)]
    return shuffled


def build_method_ratios(
    center_ratio: np.ndarray,
    recipient_ratio: np.ndarray,
    donor_base: np.ndarray,
    recipient_classes: np.ndarray,
    samples_per_epoch: int,
    epochs: int,
    gammas: list[float],
    main_gamma: float,
    lambda_cap: float,
    seed: int,
) -> dict[str, np.ndarray]:
    target_mean = float((main_gamma * donor_base).mean())
    ratios: dict[str, np.ndarray] = {
        "nofusion": np.zeros_like(center_ratio),
        "fixed_matched": np.full_like(center_ratio, target_mean),
        "pi_h2t_center_native": center_ratio,
        "pi_h2t_center_matched": match_ratio_mean(
            center_ratio, target_mean, lambda_cap
        ),
        "recipient_boundary_native": recipient_ratio,
        "recipient_boundary_matched": match_ratio_mean(
            recipient_ratio, target_mean, lambda_cap
        ),
    }
    for gamma in sorted(set(gammas + [main_gamma])):
        ratios[gamma_method_name(gamma)] = np.minimum(
            gamma * donor_base, lambda_cap
        ).astype(np.float32)

    num_classes = int(recipient_classes.max()) + 1
    first_cut = num_classes // 3
    second_cut = 2 * num_classes // 3
    frequency_groups = np.where(
        recipient_classes < first_cut,
        0,
        np.where(recipient_classes < second_cut, 1, 2),
    )
    ratios["donor_conditioned_group_mass_matched"] = match_ratio_mean_within_strata(
        donor_base, frequency_groups, target_mean, lambda_cap
    )
    ratios["donor_conditioned_class_mass_matched"] = match_ratio_mean_within_strata(
        donor_base, recipient_classes, target_mean, lambda_cap
    )
    for blend in (0.25, 0.50, 0.75):
        suffix = f"b{int(round(100 * blend)):03d}"
        ratios[f"donor_conditioned_group_blend_{suffix}"] = (
            blend * ratios["donor_conditioned_group_mass_matched"]
            + (1.0 - blend) * ratios["fixed_matched"]
        ).astype(np.float32)
        ratios[f"donor_conditioned_class_blend_{suffix}"] = (
            blend * ratios["donor_conditioned_class_mass_matched"]
            + (1.0 - blend) * ratios["fixed_matched"]
        ).astype(np.float32)

    main_name = gamma_method_name(main_gamma)
    ratios[f"{main_name}_shuffled"] = shuffle_within_recipient_class_and_epoch(
        ratios[main_name],
        recipient_classes,
        samples_per_epoch,
        epochs,
        seed + 100_003,
    )
    ratios["donor_conditioned_group_mass_matched_shuffled"] = (
        shuffle_within_recipient_class_and_epoch(
            ratios["donor_conditioned_group_mass_matched"],
            recipient_classes,
            samples_per_epoch,
            epochs,
            seed + 100_019,
        )
    )
    ratios["donor_conditioned_class_mass_matched_shuffled"] = (
        shuffle_within_recipient_class_and_epoch(
            ratios["donor_conditioned_class_mass_matched"],
            recipient_classes,
            samples_per_epoch,
            epochs,
            seed + 100_043,
        )
    )
    return ratios


def cosine_lr(epoch: int, epochs: int, base_lr: float, lr_factor: float) -> float:
    return lr_factor * base_lr * 0.5 * (1.0 + math.cos(epoch / epochs * math.pi))


@torch.inference_mode()
def evaluate_classifier(
    classifier: nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
    class_counts: np.ndarray,
) -> dict[str, float]:
    classifier.eval()
    logits = classifier(features)
    predictions = logits.argmax(dim=1)
    correct = predictions.eq(labels)
    num_classes = int(logits.shape[1])
    totals = torch.bincount(labels, minlength=num_classes).float()
    class_correct = torch.bincount(labels[correct], minlength=num_classes).float()
    class_accuracy = class_correct / totals.clamp_min(1.0)

    probabilities = logits.softmax(dim=1)
    confidence, _ = probabilities.max(dim=1)
    ece = torch.zeros((), device=logits.device)
    boundaries = torch.linspace(0.0, 1.0, 16, device=logits.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = confidence.gt(lower) & confidence.le(upper)
        if mask.any():
            ece += mask.float().mean() * (
                confidence[mask].mean() - correct[mask].float().mean()
            ).abs()

    # CIFAR100-LT reporting in the source feature project uses equal class-rank
    # thirds (0:33, 33:66, 66:100), with classes already sorted by frequency.
    class_rank = torch.arange(num_classes, device=class_accuracy.device)
    first_cut = num_classes // 3
    second_cut = 2 * num_classes // 3
    many_mask = class_rank < first_cut
    medium_mask = (class_rank >= first_cut) & (class_rank < second_cut)
    few_mask = class_rank >= second_cut
    return {
        "overall": float(correct.float().mean().item()),
        "balanced_accuracy": float(class_accuracy.mean().item()),
        "many": float(class_accuracy[many_mask].mean().item()),
        "medium": float(class_accuracy[medium_mask].mean().item()),
        "few": float(class_accuracy[few_mask].mean().item()),
        "ece": float(ece.item()),
    }


@torch.inference_mode()
def unfused_train_accuracy(
    classifier: nn.Linear,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
) -> float:
    classifier.eval()
    return float(classifier(train_features).argmax(1).eq(train_labels).float().mean().item())


def train_one_method(
    method: str,
    ratio: np.ndarray,
    data: dict[str, np.ndarray],
    recipient_indices: np.ndarray,
    donor_indices: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    base_lr: float,
    lr_factor: float,
    momentum: float,
    weight_decay: float,
    max_batches: int,
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, torch.Tensor]]:
    train_features = torch.from_numpy(data["train_features"]).to(device)
    train_labels = torch.from_numpy(data["train_labels"]).to(device)
    test_features = torch.from_numpy(data["test_features"]).to(device)
    test_labels = torch.from_numpy(data["test_labels"]).to(device)
    recipient_schedule = torch.from_numpy(recipient_indices).to(device)
    donor_schedule = torch.from_numpy(donor_indices).to(device)
    ratio_schedule = torch.from_numpy(ratio).to(device)

    feature_dim = train_features.shape[1]
    num_classes = data["reference_weight"].shape[0]
    classifier = nn.Linear(feature_dim, num_classes, bias=True).to(device)
    with torch.no_grad():
        classifier.weight.copy_(torch.from_numpy(data["reference_weight"]).to(device))
        classifier.bias.copy_(torch.from_numpy(data["reference_bias"]).to(device))

    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=base_lr * lr_factor,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    samples_per_epoch = data["train_labels"].size
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        classifier.train()
        lr = cosine_lr(epoch, epochs, base_lr, lr_factor)
        for group in optimizer.param_groups:
            group["lr"] = lr
        epoch_start = epoch * samples_per_epoch
        epoch_stop = epoch_start + samples_per_epoch
        loss_sum = 0.0
        correct_sum = 0
        seen = 0
        for batch_number, start in enumerate(range(epoch_start, epoch_stop, batch_size)):
            if max_batches > 0 and batch_number >= max_batches:
                break
            stop = min(start + batch_size, epoch_stop)
            recipient_batch = recipient_schedule[start:stop]
            donor_batch = donor_schedule[start:stop]
            lambda_batch = ratio_schedule[start:stop]
            recipient_features = train_features[recipient_batch]
            donor_features = train_features[donor_batch]
            fused = (
                (1.0 - lambda_batch[:, None]) * recipient_features
                + lambda_batch[:, None] * donor_features
            )
            targets = train_labels[recipient_batch]
            logits = classifier(fused)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            count = stop - start
            loss_sum += float(loss.item()) * count
            correct_sum += int(logits.detach().argmax(1).eq(targets).sum().item())
            seen += count

        evaluation = evaluate_classifier(
            classifier, test_features, test_labels, data["class_counts"]
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "lr": lr,
                "train_loss": loss_sum / seen,
                "fused_train_accuracy": correct_sum / seen,
                **evaluation,
            }
        )

    final = dict(history[-1])
    final["unfused_train_accuracy"] = unfused_train_accuracy(
        classifier, train_features, train_labels
    )
    state = {
        "weight": classifier.weight.detach().cpu(),
        "bias": classifier.bias.detach().cpu(),
    }
    return final, history, state


def ratio_diagnostics(
    ratio: np.ndarray,
    data: dict[str, np.ndarray],
    recipient_indices: np.ndarray,
    donor_indices: np.ndarray,
    chunk_size: int = 16_384,
) -> dict[str, float]:
    correct = 0
    total = ratio.size
    weight = data["reference_weight"]
    bias = data["reference_bias"]
    labels = data["train_labels"]
    features = data["train_features"]
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        lam = ratio[start:stop, None]
        fused = (
            (1.0 - lam) * features[recipient_indices[start:stop]]
            + lam * features[donor_indices[start:stop]]
        )
        prediction = (fused @ weight.T + bias).argmax(axis=1)
        correct += int((prediction == labels[recipient_indices[start:stop]]).sum())
    return {
        "ratio_mean": float(ratio.mean()),
        "ratio_std": float(ratio.std()),
        "ratio_q10": float(np.quantile(ratio, 0.10)),
        "ratio_q50": float(np.quantile(ratio, 0.50)),
        "ratio_q90": float(np.quantile(ratio, 0.90)),
        "reference_fused_accuracy": correct / total,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "overall",
        "balanced_accuracy",
        "many",
        "medium",
        "few",
        "ece",
        "unfused_train_accuracy",
        "ratio_mean",
        "ratio_std",
        "reference_fused_accuracy",
    )
    methods = list(dict.fromkeys(row["method"] for row in rows))
    aggregate: list[dict[str, Any]] = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        result: dict[str, Any] = {"method": method, "runs": len(selected)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        aggregate.append(result)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-feature Stage-II fusion training")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stage2_fixed_feature")
    )
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("42,43,44"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--gammas", type=parse_float_list, default=parse_float_list("0.75,0.8,0.85")
    )
    parser.add_argument("--main-gamma", type=float, default=0.8)
    parser.add_argument("--lambda-cap", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--geometry-chunk-size", type=int, default=8192)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--methods",
        default="all",
        help="all or a comma-separated subset of generated method names",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if not 0.0 < args.lambda_cap <= 1.0:
        raise ValueError("--lambda-cap must lie in (0, 1]")
    if args.delta < 0.0:
        raise ValueError("--delta must be non-negative")
    if any(not 0.0 < gamma <= 1.5 for gamma in args.gammas + [args.main_gamma]):
        raise ValueError("all gamma values must lie in (0, 1.5]")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data = load_feature_archive(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    all_ratio_rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float]]] = {}

    source_weight = data["reference_weight"]
    source_bias = data["reference_bias"]
    source_test_prediction = (
        data["test_features"] @ source_weight.T + source_bias
    ).argmax(axis=1)
    source_test_accuracy = float((source_test_prediction == data["test_labels"]).mean())
    print(
        f"device={device}, train={data['train_features'].shape}, "
        f"test={data['test_features'].shape}, source test acc={source_test_accuracy:.4f}",
        flush=True,
    )

    requested_methods = None
    if args.methods != "all":
        requested_methods = {item.strip() for item in args.methods.split(",") if item.strip()}

    for seed in args.seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        recipient_indices, donor_indices, recipient_classes = make_pair_schedule(
            data["train_labels"], args.epochs, seed
        )
        print(f"seed {seed}: computing pair geometry", flush=True)
        center_ratio, recipient_ratio, donor_base = compute_geometry(
            data["train_features"],
            data["train_labels"],
            recipient_indices,
            donor_indices,
            data["reference_weight"],
            data["reference_bias"],
            args.lambda_cap,
            args.delta,
            args.geometry_chunk_size,
            1e-12,
        )
        method_ratios = build_method_ratios(
            center_ratio,
            recipient_ratio,
            donor_base,
            recipient_classes,
            data["train_labels"].size,
            args.epochs,
            args.gammas,
            args.main_gamma,
            args.lambda_cap,
            seed,
        )
        if requested_methods is not None:
            unknown = requested_methods - set(method_ratios)
            if unknown:
                raise ValueError(f"unknown requested methods: {sorted(unknown)}")
            method_ratios = {
                method: ratio
                for method, ratio in method_ratios.items()
                if method in requested_methods
            }

        for method, ratio in method_ratios.items():
            diagnostics = ratio_diagnostics(
                ratio, data, recipient_indices, donor_indices
            )
            all_ratio_rows.append({"seed": seed, "method": method, **diagnostics})
            print(
                f"seed {seed} {method}: mean_lambda={diagnostics['ratio_mean']:.4f}, "
                f"ref_fused_acc={diagnostics['reference_fused_accuracy']:.4f}",
                flush=True,
            )
            final, history, state = train_one_method(
                method,
                ratio,
                data,
                recipient_indices,
                donor_indices,
                device,
                args.epochs,
                args.batch_size,
                args.lr,
                args.lr_factor,
                args.momentum,
                args.weight_decay,
                args.max_batches,
            )
            row = {
                "seed": seed,
                "method": method,
                **final,
                **diagnostics,
                "source_test_accuracy": source_test_accuracy,
            }
            all_results.append(row)
            histories[f"seed{seed}/{method}"] = history
            print(
                f"  final overall={final['overall']:.4f}, many={final['many']:.4f}, "
                f"medium={final['medium']:.4f}, few={final['few']:.4f}",
                flush=True,
            )
            if args.save_checkpoints:
                checkpoint_dir = args.output_dir / "checkpoints" / f"seed{seed}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "method": method,
                        "seed": seed,
                        "classifier_weight": state["weight"],
                        "classifier_bias": state["bias"],
                        "final_metrics": final,
                        "ratio_diagnostics": diagnostics,
                    },
                    checkpoint_dir / f"{method}.pt",
                )

    aggregate = aggregate_results(all_results)
    write_csv(args.output_dir / "per_run.csv", all_results)
    write_csv(args.output_dir / "ratio_diagnostics.csv", all_ratio_rows)
    write_csv(args.output_dir / "aggregate.csv", aggregate)
    with (args.output_dir / "histories.json").open("w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)
    with (args.output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "input": str(args.input.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "device": str(device),
                "source_test_accuracy": source_test_accuracy,
                "note": (
                    "Exploratory classifier-only Stage-II run.  The current test features "
                    "were previously used by the minival probe and are not publication-clean."
                ),
            },
            handle,
            indent=2,
            default=str,
        )

    print("\nFinal Stage-II leaderboard (last epoch)")
    for row in sorted(aggregate, key=lambda item: item["overall_mean"], reverse=True):
        print(
            f"{row['method']:<38} overall={row['overall_mean']:.4f}+-{row['overall_std']:.4f} "
            f"many={row['many_mean']:.4f} medium={row['medium_mean']:.4f} "
            f"few={row['few_mean']:.4f}",
            flush=True,
        )
    print(f"wrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
