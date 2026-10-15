from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler

from lt_frequency_sweep_experiment import DEFAULT_RANKS, make_sweep_design
from lt_rank_swap_experiment import (
    exponential_counts,
    read_class_names,
    select_indices,
)
from support_dynamics_experiment import (
    CIFAR_MEAN,
    CIFAR_STD,
    NUM_CLASSES,
    ArrayCIFAR100,
    ResNetCifar,
    cosine_learning_rate,
    eval_transform,
    make_nested_class_orders,
    save_json,
    set_seed,
    train_transform,
    write_csv,
)


METHODS = [
    "classifier_only",
    "augmented_ft",
    "self_support",
    "similar_support",
    "oracle_support",
]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Trainable layer3 support recovery. Recovery targets use only the "
            "tail training images and donor trajectories estimated on the held-out "
            "calibration split. The official test set is final evaluation only."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\xiangmuxunlian\hafe_experiment\data"),
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=project_dir / "outputs" / "leakage_free_frequency_sweep",
    )
    parser.add_argument(
        "--balanced-root",
        type=Path,
        default=project_dir / "outputs" / "balanced_neighbor_trajectory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "trainable_support_recovery",
    )
    parser.add_argument("--class-a", type=int, default=47)
    parser.add_argument("--class-b", type=int, default=52)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--train-pool-per-class", type=int, default=400)
    parser.add_argument("--calibration-per-class", type=int, default=100)
    parser.add_argument("--imbalance-factor", type=float, default=100.0)
    parser.add_argument("--split-seed", type=int, default=20260731)
    parser.add_argument("--rank-seed", type=int, default=20260801)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--support-views", type=int, default=8)
    parser.add_argument("--target-ce-weight", type=float, default=0.5)
    parser.add_argument("--support-weight", type=float, default=1.0)
    parser.add_argument("--preserve-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--layer3-lr", type=float, default=0.005)
    parser.add_argument("--fc-lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--probe-lr", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Reuse completed method metrics/checkpoints and continue missing runs.",
    )
    return parser.parse_args()


def strong_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.08),
            transforms.RandomGrayscale(p=0.08),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def split_data(
    targets: np.ndarray, args: argparse.Namespace
) -> tuple[list[np.ndarray], np.ndarray]:
    orders = make_nested_class_orders(targets, args.split_seed)
    train_orders = [
        order[: args.train_pool_per_class].copy() for order in orders
    ]
    calibration = np.concatenate(
        [
            order[
                args.train_pool_per_class :
                args.train_pool_per_class + args.calibration_per_class
            ]
            for order in orders
        ]
    ).astype(np.int64)
    if set(np.concatenate(train_orders)).intersection(calibration.tolist()):
        raise AssertionError("training/calibration leakage")
    return train_orders, calibration


def load_pair_similarity(args: argparse.Namespace) -> float:
    path = args.balanced_root / "seed_42" / "all_class_pairs.csv"
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    scores: dict[int, dict[int, float]] = {}
    for row in rows:
        target = int(row["target_class"])
        candidate = int(row["candidate_class"])
        scores.setdefault(target, {})[candidate] = float(row["prototype"])
    if max(scores[args.class_a], key=scores[args.class_a].get) != args.class_b:
        raise AssertionError("B is not A's nearest balanced prototype class")
    if max(scores[args.class_b], key=scores[args.class_b].get) != args.class_a:
        raise AssertionError("A is not B's nearest balanced prototype class")
    return scores[args.class_a][args.class_b]


def make_design(args: argparse.Namespace) -> dict:
    curve = exponential_counts(
        args.train_pool_per_class, args.imbalance_factor
    )
    namespace = argparse.Namespace(
        class_a=args.class_a,
        class_b=args.class_b,
        ranks=DEFAULT_RANKS,
        rank_seed=args.rank_seed,
    )
    return make_sweep_design(namespace, curve)


def model_path(root: Path, seed: int, state: dict) -> Path:
    return root / f"seed_{seed}" / f"model_{state['name']}_final.pth"


def load_model(
    path: Path,
    device: torch.device,
    expected_counts: np.ndarray | None = None,
    args: argparse.Namespace | None = None,
) -> ResNetCifar:
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory checkpoint: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if expected_counts is not None:
        checks = {
            "counts": np.array_equal(
                np.asarray(saved.get("counts", []), dtype=np.int64),
                np.asarray(expected_counts, dtype=np.int64),
            ),
            "split_seed": int(saved.get("split_seed", -1)) == int(args.split_seed),
            "train_pool": int(saved.get("train_pool_per_class", -1))
            == int(args.train_pool_per_class),
            "calibration": int(saved.get("calibration_per_class", -1))
            == int(args.calibration_per_class),
        }
        if not all(checks.values()):
            raise ValueError(f"Incompatible trajectory checkpoint {path}: {checks}")
    model = ResNetCifar(depth=32)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    raw,
    indices: np.ndarray | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray(raw.targets, dtype=np.int64)
    dataset = ArrayCIFAR100(
        raw.data, targets, indices, eval_transform()
    )
    # Spawning Windows workers dominates tiny per-class calibration extracts.
    # Keep those synchronous; retain workers for the 10k-image evaluation pass.
    eval_workers = 0 if indices is not None and len(indices) <= 1000 else args.num_workers
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=eval_workers > 0,
    )
    features, logits, labels = [], [], []
    model.eval()
    for images, current_labels in loader:
        images = images.to(device, non_blocking=True)
        current_logits, current_features = model(images, return_features=True)
        features.append(current_features.float().cpu().numpy())
        logits.append(current_logits.float().cpu().numpy())
        labels.append(current_labels.numpy())
    return (
        np.concatenate(features).astype(np.float32),
        np.concatenate(logits).astype(np.float32),
        np.concatenate(labels).astype(np.int64),
    )


def support_stats(features: np.ndarray) -> dict:
    z = features.astype(np.float64)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    prototype = z.mean(axis=0)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
    cosine = np.clip(z @ prototype, -1.0, 1.0)
    angles = np.arccos(cosine)
    tangent = z - cosine[:, None] * prototype[None, :]
    covariance = tangent.T @ tangent / max(len(tangent) - 1, 1)
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) * 0.5)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    trace = max(float(eigenvalues.sum()), 1e-12)
    shares = eigenvalues[:4] / trace
    positive = eigenvalues[eigenvalues > 1e-12]
    probabilities = positive / max(float(positive.sum()), 1e-12)
    effective_rank = float(
        np.exp(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    )
    return {
        "prototype": prototype,
        "trace": trace,
        "angle_q50": float(np.quantile(angles, 0.50)),
        "angle_q80": float(np.quantile(angles, 0.80)),
        "angle_q95": float(np.quantile(angles, 0.95)),
        "eigshare_top4": shares,
        "effective_rank": effective_rank,
    }


def trajectory_stats(
    seed: int,
    raw_train,
    calibration: np.ndarray,
    design: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    targets = np.asarray(raw_train.targets, dtype=np.int64)
    result = {"A": [], "B": []}
    for state in design["states"]:
        model = load_model(
            model_path(args.sweep_root, seed, state),
            device,
            state["counts"],
            args,
        )
        for role, class_id in [("A", args.class_a), ("B", args.class_b)]:
            indices = calibration[targets[calibration] == class_id]
            features, _, _ = extract_features(
                model, raw_train, indices, args, device
            )
            result[role].append(support_stats(features))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["A_head_to_tail"] = result["A"]
    result["B_head_to_tail"] = list(reversed(result["B"]))
    return result


def tail_views(
    raw_train,
    indices: np.ndarray,
    views: int,
    transform,
) -> torch.Tensor:
    images = []
    for _ in range(views):
        for index in indices:
            images.append(transform(Image.fromarray(raw_train.data[int(index)])))
    return torch.stack(images)


@torch.no_grad()
def initial_train_support(
    model: nn.Module,
    raw_train,
    indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    set_seed(97000 + int(indices[0]))
    images = tail_views(raw_train, indices, max(32, args.support_views), strong_transform())
    _, features = model(images.to(device), return_features=True)
    return support_stats(features.float().cpu().numpy())


def scaled_target(current: dict, donor_head: dict, donor_tail: dict) -> dict:
    target = {}
    for name in ["trace", "angle_q50", "angle_q95"]:
        ratio = donor_head[name] / max(donor_tail[name], 1e-12)
        target[name] = float(current[name] * np.clip(ratio, 0.25, 4.0))
    target["source"] = "target few-shot support times similar donor head/tail ratios"
    return target


def direct_target(stats: dict, source: str) -> dict:
    return {
        "trace": float(stats["trace"]),
        "angle_q50": float(stats["angle_q50"]),
        "angle_q95": float(stats["angle_q95"]),
        "source": source,
    }


def torch_support(features: torch.Tensor) -> dict[str, torch.Tensor]:
    z = F.normalize(features.float(), dim=1)
    prototype = F.normalize(z.mean(dim=0), dim=0)
    cosine = torch.clamp(z @ prototype, -1.0 + 1e-6, 1.0 - 1e-6)
    angles = torch.acos(cosine)
    tangent = z - cosine[:, None] * prototype[None, :]
    covariance = tangent.T @ tangent / max(len(tangent) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh((covariance + covariance.T) * 0.5)
    eigenvalues = torch.clamp(eigenvalues.flip(0), min=0.0)
    trace = torch.clamp(eigenvalues.sum(), min=1e-8)
    return {
        "prototype": prototype,
        "trace": trace,
        "angle_q50": torch.quantile(angles, 0.50),
        "angle_q95": torch.quantile(angles, 0.95),
        "eigshare_top4": eigenvalues[:4] / trace,
    }


def support_loss(current: dict[str, torch.Tensor], target: dict) -> torch.Tensor:
    device = current["trace"].device
    loss = (
        torch.log(current["trace"] / float(target["trace"])) ** 2
        + torch.log(current["angle_q50"] / float(target["angle_q50"])) ** 2
        + torch.log(current["angle_q95"] / float(target["angle_q95"])) ** 2
    )
    return loss


def configure_trainable(model: ResNetCifar, method: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if method != "classifier_only":
        for parameter in model.layer3.parameters():
            parameter.requires_grad = True


def keep_frozen_eval(model: ResNetCifar, method: str) -> None:
    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    # A classifier-only control must not even update layer3 BatchNorm buffers.
    if method == "classifier_only":
        model.layer3.eval()
    else:
        # Tail support batches contain many correlated views of only 4 images.
        # Updating running moments with them shifts every class, independent of
        # the small optimizer LR. Keep BN moments fixed while still allowing
        # convolution and BN affine parameters in layer3 to receive gradients.
        for module in model.layer3.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def relative_parameter_delta(
    before: dict[str, torch.Tensor], model: nn.Module, prefix: str
) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, value in model.state_dict().items():
        if name.startswith(prefix) and torch.is_floating_point(value):
            numerator += float(torch.sum((value.cpu() - before[name]) ** 2))
            denominator += float(torch.sum(before[name] ** 2))
    return math.sqrt(numerator / max(denominator, 1e-20))


def train_recovery(
    baseline: ResNetCifar,
    method: str,
    target_class: int,
    donor_class: int,
    target_indices: np.ndarray,
    train_indices: np.ndarray,
    raw_train,
    target_support: dict,
    args: argparse.Namespace,
    device: torch.device,
    run_seed: int,
) -> tuple[ResNetCifar, list[dict], dict]:
    set_seed(run_seed)
    teacher = copy.deepcopy(baseline).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    model = copy.deepcopy(baseline).to(device)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    configure_trainable(model, method)
    groups = [{"params": model.fc.parameters(), "lr": args.fc_lr}]
    if method != "classifier_only":
        groups.insert(0, {"params": model.layer3.parameters(), "lr": args.layer3_lr})
    optimizer = torch.optim.SGD(
        groups, momentum=0.9, nesterov=True, weight_decay=args.weight_decay
    )
    targets = np.asarray(raw_train.targets, dtype=np.int64)
    dataset = ArrayCIFAR100(
        raw_train.data, targets, train_indices, train_transform()
    )
    generator = torch.Generator().manual_seed(run_seed)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=args.steps_per_epoch * args.batch_size,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    transform = strong_transform()
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        keep_frozen_eval(model, method)
        scale = 0.5 * (1.0 + math.cos(math.pi * epoch / max(args.epochs - 1, 1)))
        for group, base_lr in zip(optimizer.param_groups, [args.layer3_lr, args.fc_lr] if method != "classifier_only" else [args.fc_lr]):
            group["lr"] = base_lr * scale
        sums = {name: 0.0 for name in ["total", "classification", "support", "preserve", "margin"]}
        finite = True
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            support_images = tail_views(
                raw_train, target_indices, args.support_views, transform
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits, teacher_features = teacher(
                    images, return_features=True
                )
            logits, features = model(images, return_features=True)
            target_logits, target_features = model(
                support_images, return_features=True
            )
            target_labels = torch.full(
                (len(support_images),), target_class, device=device, dtype=torch.long
            )
            classification = F.cross_entropy(logits, labels) + args.target_ce_weight * F.cross_entropy(target_logits, target_labels)
            preserve = (1.0 - F.cosine_similarity(features, teacher_features.detach(), dim=1).mean()) + 0.25 * F.kl_div(
                F.log_softmax(logits, dim=1),
                F.softmax(teacher_logits.detach(), dim=1),
                reduction="batchmean",
            )
            current = torch_support(target_features)
            geometry = torch.zeros((), device=device)
            if method in {"self_support", "similar_support", "oracle_support"}:
                geometry = support_loss(current, target_support)
            classifier_weights = F.normalize(model.fc.weight, dim=1)
            cos_target = current["prototype"] @ classifier_weights[target_class]
            cos_donor = current["prototype"] @ classifier_weights[donor_class]
            margin = F.relu(cos_donor - cos_target + args.margin)
            total = classification + args.preserve_weight * preserve + args.margin_weight * margin
            if method in {"self_support", "similar_support", "oracle_support"}:
                total = total + args.support_weight * geometry
            if not torch.isfinite(total):
                finite = False
                raise FloatingPointError(f"Non-finite recovery loss: {method}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 10.0
            )
            optimizer.step()
            for name, value in [
                ("total", total),
                ("classification", classification),
                ("support", geometry),
                ("preserve", preserve),
                ("margin", margin),
            ]:
                sums[name] += float(value.detach())
        row = {
            "epoch": epoch + 1,
            **{name: value / max(len(loader), 1) for name, value in sums.items()},
            "finite": finite,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"[{method}] epoch {epoch + 1:02d}/{args.epochs} "
            f"loss={row['total']:.4f} support={row['support']:.4f} "
            f"preserve={row['preserve']:.4f}",
            flush=True,
        )
    diagnostics = {
        "layer3_relative_delta": relative_parameter_delta(before, model, "layer3."),
        "fc_relative_delta": relative_parameter_delta(before, model, "fc."),
    }
    return model.eval(), history, diagnostics


def class_metrics(logits: np.ndarray, labels: np.ndarray, target: int, donor: int) -> dict:
    predictions = logits.argmax(axis=1)
    correct = predictions == labels
    target_mask = labels == target
    donor_mask = labels == donor
    other_mask = ~target_mask
    return {
        "overall_accuracy": float(correct.mean()),
        "target_accuracy": float(correct[target_mask].mean()),
        "donor_accuracy": float(correct[donor_mask].mean()),
        "other99_accuracy": float(correct[other_mask].mean()),
    }


def representation_geometry(
    features: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
    target: int,
    donor: int,
    classifier_weight: np.ndarray,
) -> dict:
    target_features = features[labels == target]
    donor_features = features[labels == donor]
    target_stats = support_stats(target_features)
    donor_stats = support_stats(donor_features)
    target_prototype = target_stats["prototype"]
    donor_prototype = donor_stats["prototype"]
    weights = classifier_weight / np.maximum(
        np.linalg.norm(classifier_weight, axis=1, keepdims=True), 1e-12
    )
    result = {
        key: value
        for key, value in target_stats.items()
        if key not in {"prototype", "eigshare_top4"}
    }
    result.update(
        {
            "eigshare_top4": target_stats["eigshare_top4"],
            "target_donor_prototype_cosine": float(target_prototype @ donor_prototype),
            "target_classifier_alignment": float(target_prototype @ weights[target]),
            "target_vs_donor_classifier_margin": float(
                target_prototype @ weights[target] - target_prototype @ weights[donor]
            ),
            **class_metrics(logits, labels, target, donor),
        }
    )
    return result


def ncm_accuracy(
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    calibration_features = calibration_features / np.maximum(
        np.linalg.norm(calibration_features, axis=1, keepdims=True), 1e-12
    )
    test_features = test_features / np.maximum(
        np.linalg.norm(test_features, axis=1, keepdims=True), 1e-12
    )
    prototypes = np.stack(
        [calibration_features[calibration_labels == c].mean(axis=0) for c in range(NUM_CLASSES)]
    )
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    predictions = (test_features @ prototypes.T).argmax(axis=1)
    return float((predictions == test_labels).mean()), predictions


def linear_probe(
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    test_features: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    x = torch.as_tensor(calibration_features, dtype=torch.float32, device=device)
    x = F.normalize(x, dim=1)
    y = torch.as_tensor(calibration_labels, dtype=torch.long, device=device)
    probe = nn.Linear(x.shape[1], NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(args.probe_steps):
        indices = torch.randint(0, len(x), (min(2048, len(x)),), generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(x[indices]), y[indices])
        loss.backward()
        optimizer.step()
    outputs = []
    probe.eval()
    with torch.no_grad():
        for start in range(0, len(test_features), 2048):
            current = torch.as_tensor(test_features[start : start + 2048], dtype=torch.float32, device=device)
            outputs.append(probe(F.normalize(current, dim=1)).argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def probe_metrics(predictions: np.ndarray, labels: np.ndarray, target: int) -> dict:
    correct = predictions == labels
    mask = labels == target
    return {
        "overall": float(correct.mean()),
        "target": float(correct[mask].mean()),
    }


def evaluate_representation(
    model: ResNetCifar,
    raw_train,
    raw_test,
    calibration: np.ndarray,
    target: int,
    donor: int,
    args: argparse.Namespace,
    device: torch.device,
    probe_seed: int,
) -> dict:
    calibration_features, _, calibration_labels = extract_features(
        model, raw_train, calibration, args, device
    )
    test_features, test_logits, test_labels = extract_features(
        model, raw_test, None, args, device
    )
    geometry = representation_geometry(
        test_features,
        test_labels,
        test_logits,
        target,
        donor,
        model.fc.weight.detach().cpu().numpy(),
    )
    ncm_overall, ncm_predictions = ncm_accuracy(
        calibration_features, calibration_labels, test_features, test_labels
    )
    probe_predictions = linear_probe(
        calibration_features,
        calibration_labels,
        test_features,
        args,
        device,
        probe_seed,
    )
    geometry["ncm"] = probe_metrics(ncm_predictions, test_labels, target)
    geometry["ncm"]["overall"] = ncm_overall
    geometry["linear_probe"] = probe_metrics(
        probe_predictions, test_labels, target
    )
    return geometry


def oracle_distance(metrics: dict, oracle: dict) -> float:
    values = []
    for name in ["trace", "angle_q50", "angle_q95"]:
        values.append(math.log(max(metrics[name], 1e-12) / max(oracle[name], 1e-12)))
    return float(np.linalg.norm(values))


def main() -> None:
    args = parse_args()
    unknown = set(args.methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if args.smoke:
        args.seeds = args.seeds[:1]
        args.epochs = 1
        args.steps_per_epoch = min(args.steps_per_epoch, 4)
        args.support_views = min(args.support_views, 2)
        args.probe_steps = min(args.probe_steps, 10)
        args.num_workers = min(args.num_workers, 2)
        args.sweep_root = args.sweep_root / "smoke"
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    class_names = read_class_names(args.data_root)
    similarity = load_pair_similarity(args)
    raw_train = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False
    )
    raw_test = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=False
    )
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    train_orders, calibration = split_data(train_targets, args)
    design = make_design(args)
    audit = {
        "official_test_usage": "final representation and classifier evaluation only",
        "recovery_target_uses_test": False,
        "target_calibration_support_statistics_used_by_non_oracle_recovery": False,
        "balanced_pair_selection_uses_calibration_reference_space": True,
        "train_calibration_overlap": 0,
        "pair": [args.class_a, args.class_b],
        "pair_names": [class_names[args.class_a], class_names[args.class_b]],
        "balanced_prototype_similarity": similarity,
        "trainable_stages": "layer3 + fc (classifier_only freezes layer3)",
        "arguments": vars(args),
    }
    save_json(audit, args.output_dir / "audit.json")
    all_rows = []
    for seed in args.seeds:
        print(f"[seed {seed}] estimating donor trajectory on calibration split", flush=True)
        trajectory = trajectory_stats(
            seed, raw_train, calibration, design, args, device
        )
        compact_trajectory = {
            role: [
                {key: value for key, value in state.items() if key != "prototype"}
                for state in trajectory[f"{role}_head_to_tail"]
            ]
            for role in ["A", "B"]
        }
        save_json(compact_trajectory, args.output_dir / f"seed_{seed}" / "calibration_trajectory.json")
        scenarios = [
            {
                "name": "A_tail_from_B",
                "target": args.class_a,
                "donor": args.class_b,
                "endpoint": 7,
                "target_role": "A",
                "donor_role": "B",
            },
            {
                "name": "B_tail_from_A",
                "target": args.class_b,
                "donor": args.class_a,
                "endpoint": 0,
                "target_role": "B",
                "donor_role": "A",
            },
        ]
        for scenario_index, scenario in enumerate(scenarios):
            state = design["states"][scenario["endpoint"]]
            baseline = load_model(
                model_path(args.sweep_root, seed, state),
                device,
                state["counts"],
                args,
            ).eval()
            counts = state["counts"]
            train_indices = select_indices(train_orders, counts)
            target_indices = train_orders[scenario["target"]][: int(counts[scenario["target"]])]
            if len(target_indices) != int(counts[scenario["target"]]):
                raise AssertionError("target tail subset mismatch")
            current = initial_train_support(
                baseline, raw_train, target_indices, args, device
            )
            donor_curve = trajectory[f"{scenario['donor_role']}_head_to_tail"]
            target_curve = trajectory[f"{scenario['target_role']}_head_to_tail"]
            method_targets = {
                "self_support": direct_target(current, "initial target few-shot support"),
                "similar_support": scaled_target(current, donor_curve[0], donor_curve[-1]),
                "oracle_support": scaled_target(current, target_curve[0], target_curve[-1]),
            }
            method_targets["oracle_support"]["source"] = (
                "target few-shot support times oracle same-class head/tail ratios"
            )
            scenario_dir = args.output_dir / f"seed_{seed}" / scenario["name"]
            save_json(method_targets, scenario_dir / "support_targets.json")
            baseline_eval = evaluate_representation(
                baseline,
                raw_train,
                raw_test,
                calibration,
                scenario["target"],
                scenario["donor"],
                args,
                device,
                8000 + scenario_index,
            )
            baseline_eval["oracle_geometry_distance"] = oracle_distance(
                baseline_eval, target_curve[0]
            )
            save_json(baseline_eval, scenario_dir / "baseline_metrics.json")
            for method_index, method in enumerate(args.methods):
                print(f"[{scenario['name']}] method={method}", flush=True)
                target = method_targets.get(method, method_targets["self_support"])
                method_dir = scenario_dir / method
                metrics_path = method_dir / "metrics.json"
                checkpoint_path = method_dir / "recovered_final.pth"
                if args.skip_completed and metrics_path.exists() and checkpoint_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    row = {
                        "seed": seed,
                        "scenario": scenario["name"],
                        "target_class": scenario["target"],
                        "method": method,
                        "target_accuracy": metrics["target_accuracy"],
                        "delta_target_accuracy": metrics["delta_target_accuracy"],
                        "other99_accuracy": metrics["other99_accuracy"],
                        "delta_other99_accuracy": metrics["delta_other99_accuracy"],
                        "ncm_target": metrics["ncm"]["target"],
                        "delta_ncm_target": metrics["delta_ncm_target"],
                        "probe_target": metrics["linear_probe"]["target"],
                        "delta_probe_target": metrics["delta_probe_target"],
                        "oracle_geometry_distance": metrics["oracle_geometry_distance"],
                        "layer3_relative_delta": metrics["layer3_relative_delta"],
                    }
                    all_rows.append(row)
                    write_csv(all_rows, args.output_dir / "results.csv")
                    print(f"[{scenario['name']}] reusing completed {method}", flush=True)
                    continue
                recovered, history, diagnostics = train_recovery(
                    baseline,
                    method,
                    scenario["target"],
                    scenario["donor"],
                    target_indices,
                    train_indices,
                    raw_train,
                    target,
                    args,
                    device,
                    seed * 100 + scenario_index * 10 + method_index,
                )
                metrics = evaluate_representation(
                    recovered,
                    raw_train,
                    raw_test,
                    calibration,
                    scenario["target"],
                    scenario["donor"],
                    args,
                    device,
                    8000 + scenario_index,
                )
                metrics.update(diagnostics)
                metrics["oracle_geometry_distance"] = oracle_distance(
                    metrics, target_curve[0]
                )
                metrics["delta_target_accuracy"] = metrics["target_accuracy"] - baseline_eval["target_accuracy"]
                metrics["delta_other99_accuracy"] = metrics["other99_accuracy"] - baseline_eval["other99_accuracy"]
                metrics["delta_ncm_target"] = metrics["ncm"]["target"] - baseline_eval["ncm"]["target"]
                metrics["delta_probe_target"] = metrics["linear_probe"]["target"] - baseline_eval["linear_probe"]["target"]
                method_dir.mkdir(parents=True, exist_ok=True)
                save_json(history, method_dir / "history.json")
                save_json(metrics, method_dir / "metrics.json")
                torch.save(
                    {
                        "model": recovered.state_dict(),
                        "seed": seed,
                        "scenario": scenario["name"],
                        "method": method,
                        "support_target": target,
                        "metrics": metrics,
                    },
                    checkpoint_path,
                )
                row = {
                    "seed": seed,
                    "scenario": scenario["name"],
                    "target_class": scenario["target"],
                    "method": method,
                    "target_accuracy": metrics["target_accuracy"],
                    "delta_target_accuracy": metrics["delta_target_accuracy"],
                    "other99_accuracy": metrics["other99_accuracy"],
                    "delta_other99_accuracy": metrics["delta_other99_accuracy"],
                    "ncm_target": metrics["ncm"]["target"],
                    "delta_ncm_target": metrics["delta_ncm_target"],
                    "probe_target": metrics["linear_probe"]["target"],
                    "delta_probe_target": metrics["delta_probe_target"],
                    "oracle_geometry_distance": metrics["oracle_geometry_distance"],
                    "layer3_relative_delta": metrics["layer3_relative_delta"],
                }
                all_rows.append(row)
                write_csv(all_rows, args.output_dir / "results.csv")
                del recovered
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del baseline
    save_json(all_rows, args.output_dir / "results.json")
    print(f"Finished: {args.output_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
