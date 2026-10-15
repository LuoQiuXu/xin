"""PI-H2T donor-blindness validation with recipient-OOF geometry.

This experiment changes only the H2TF donor-mass ratio.  Backbone, Stage-I
classifier initialization, balanced recipient sampling, instance donor
sampling, CE loss, optimizer, epochs and BN-shift stream remain fixed.

Arms:
  A    original PI-H2T recipient-only ratio
  T2   no-fusion endpoint under the same Stage-II/BN stream
  D0   recipient-OOF strict pair boundary
  E0   exact D0 ratio multiset, allocated by corrupted donor geometry
  Dcc  D0 plus OOF-reliability controlled crossing
  Ecc  exact Dcc ratio multiset, allocated by corrupted donor geometry

E0/Ecc corrupt geometry only.  Actual fusion always uses the original true
recipient--donor pair.  Allocation is epoch-level within
Many/Medium/Few x native-ratio-bin x reliability-bin strata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from datasets.cifar100 import CIFAR100_LT
from models import resnet_cifar
from train_boundary_h2t_stage2 import (
    dataset_config,
    frequency_groups,
    load_stage1_checkpoint,
    seed_everything,
)
from train_oof_balanced_reference import balanced_schedule
from train_pi_h2t_ratio_only_stage2 import (
    build_models,
    classify_vector,
    evaluate,
    official_pi_h2t_ratios,
)


METHODS = ("A", "T2", "D0", "E0", "Dcc", "Ecc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF controlled-crossing Stage-II")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oof-reference", required=True)
    parser.add_argument("--data-path", default="./data/cifar100")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--donor-mass-scale", type=float, default=0.9267667644209227)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--alpha-grid", default="0,0.5,1")
    parser.add_argument("--ratio-bins", type=int, default=4)
    parser.add_argument("--q-bins", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-shift-bn", action="store_true")
    parser.add_argument("--skip-alpha-cv", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choices are {METHODS}")
    args.alpha_grid = [float(item) for item in args.alpha_grid.split(",")]
    if any(not 0.0 <= value <= 1.0 for value in args.alpha_grid):
        raise ValueError("alpha grid values must be in [0,1]")
    if args.alpha is not None and not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0,1]")
    if not 0.0 <= args.donor_mass_scale <= 1.0:
        raise ValueError("--donor-mass-scale must be in [0,1]")
    return args


def load_oof(path: Path, device: torch.device) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "fold_ids", "reference_weight", "reference_bias", "temperature",
        "q_true_class", "train_labels",
    }
    missing = required - set(raw)
    if missing:
        raise KeyError(f"OOF checkpoint missing {sorted(missing)}")
    return {
        **raw,
        "fold_ids": raw["fold_ids"].long().to(device),
        "reference_weight": raw["reference_weight"].float().to(device),
        "reference_bias": raw["reference_bias"].float().to(device),
        "temperature": raw["temperature"].float().to(device),
        "q_true_class": raw["q_true_class"].float().to(device),
        "train_labels": raw["train_labels"].long().to(device),
    }


def batched_reference_geometry(
    recipient: torch.Tensor,
    donor: torch.Tensor,
    target: torch.Tensor,
    fold: torch.Tensor,
    all_weight: torch.Tensor,
    all_bias: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Worst normalized margin and donor-conditioned safe mass per sample."""
    weight = all_weight[fold]
    bias = all_bias[fold]
    recipient_logits = torch.einsum("bcd,bd->bc", weight, recipient) + bias
    donor_logits = torch.einsum("bcd,bd->bc", weight, donor) + bias
    recipient_num = recipient_logits.gather(1, target[:, None]) - recipient_logits
    donor_num = donor_logits.gather(1, target[:, None]) - donor_logits
    selected = weight[torch.arange(target.numel(), device=target.device), target]
    denominator = torch.linalg.vector_norm(selected[:, None, :] - weight, dim=2).clamp_min(eps)
    recipient_distance = recipient_num / denominator
    donor_distance = donor_num / denominator
    row = torch.arange(target.numel(), device=target.device)
    recipient_distance[row, target] = torch.inf
    donor_distance[row, target] = torch.inf
    margin = recipient_distance.min(1).values
    floor = torch.minimum(margin, torch.zeros_like(margin))
    decrease = recipient_distance - donor_distance
    dangerous = torch.isfinite(decrease) & (decrease > eps)
    bound = torch.full_like(decrease, torch.inf)
    raw = (recipient_distance - floor[:, None]) / decrease.clamp_min(eps)
    bound[dangerous] = raw[dangerous]
    critical = bound.min(1).values
    critical = torch.where(torch.isfinite(critical), critical, torch.ones_like(critical))
    return margin, critical.clamp(0.0, 1.0)


def batched_worst_margin(
    feature: torch.Tensor,
    target: torch.Tensor,
    fold: torch.Tensor,
    all_weight: torch.Tensor,
    all_bias: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    weight = all_weight[fold]
    bias = all_bias[fold]
    logits = torch.einsum("bcd,bd->bc", weight, feature) + bias
    numerator = logits.gather(1, target[:, None]) - logits
    selected = weight[torch.arange(target.numel(), device=target.device), target]
    denominator = torch.linalg.vector_norm(selected[:, None, :] - weight, dim=2).clamp_min(eps)
    distance = numerator / denominator
    distance[torch.arange(target.numel(), device=target.device), target] = torch.inf
    return distance.min(1).values


def strata_ids(
    target: torch.Tensor,
    lambda0: torch.Tensor,
    q: torch.Tensor,
    ratio_bins: int,
    q_bins: int,
) -> torch.Tensor:
    group = frequency_groups(target)
    ratio_bin = (lambda0 * ratio_bins).long().clamp(0, ratio_bins - 1)
    q_bin = (q * q_bins).long().clamp(0, q_bins - 1)
    return (group * ratio_bins + ratio_bin) * q_bins + q_bin


def corrupt_geometry_allocate(
    source_ratio: torch.Tensor,
    recipient: torch.Tensor,
    donor: torch.Tensor,
    target: torch.Tensor,
    fold: torch.Tensor,
    strata: torch.Tensor,
    stratum_count: int,
    reference_weight: torch.Tensor,
    reference_bias: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reallocate an exact ratio multiset using only wrong-donor geometry."""
    result = source_ratio.clone()
    donor_permutation = torch.arange(source_ratio.numel(), device=source_ratio.device)
    eligible = 0
    moved = 0
    for stratum in range(stratum_count):
        indices = torch.nonzero(strata == stratum, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        # A random cyclic shift is a derangement and therefore has an auditable
        # 100% moved fraction for every eligible stratum.
        offset = int(torch.randint(
            1, indices.numel(), (1,), generator=generator,
            device=source_ratio.device,
        ))
        permuted = indices.roll(offset)
        donor_permutation[indices] = permuted
        eligible += int(indices.numel())
        moved += int((permuted != indices).sum())

    _, wrong_critical = batched_reference_geometry(
        recipient,
        donor[donor_permutation],
        target,
        fold,
        reference_weight,
        reference_bias,
    )
    ratio_reassigned = 0
    maximum_error = 0.0
    for stratum in range(stratum_count):
        indices = torch.nonzero(strata == stratum, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        score_order = torch.argsort(wrong_critical[indices], stable=True)
        masses = torch.sort(source_ratio[indices], stable=True).values
        allocated_indices = indices[score_order]
        result[allocated_indices] = masses
        ratio_reassigned += int((result[indices] != source_ratio[indices]).sum())
        maximum_error = max(
            maximum_error,
            float((torch.sort(result[indices]).values - masses).abs().max()),
        )
    return result, {
        "donor_corruption_moved_fraction": moved / max(eligible, 1),
        "donor_corruption_eligible_fraction": eligible / source_ratio.numel(),
        "ratio_reassigned_fraction": ratio_reassigned / source_ratio.numel(),
        "multiset_match_max_abs_error": maximum_error,
    }


def metric_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    prediction = logits.argmax(1)
    correct = torch.zeros(100, device=labels.device)
    count = torch.zeros(100, device=labels.device)
    correct.scatter_add_(0, labels, prediction.eq(labels).float())
    count.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    acc = correct / count.clamp_min(1.0)
    many, medium, few = float(acc[:36].mean()), float(acc[36:71].mean()), float(acc[71:].mean())
    hmean = 2.0 * many * few / max(many + few, 1e-12)
    return {
        "overall": float(correct.sum() / count.sum()),
        "many": many,
        "medium": medium,
        "few": few,
        "hmean": hmean,
    }


def select_alpha_cv(
    feature_archive: Path,
    classifier_state: dict[str, torch.Tensor],
    oof: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, list[dict[str, Any]]]:
    """Training-only fixed-feature 5-fold Stage-II calibration of alpha."""
    archive = np.load(feature_archive, allow_pickle=False)
    features_np = archive["train_features"].astype(np.float32, copy=False)
    labels_np = archive["train_labels"].astype(np.int64, copy=False)
    features = torch.from_numpy(features_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    folds = oof["fold_ids"]
    q = oof["q_true_class"]
    reference_weight = oof["reference_weight"]
    reference_bias = oof["reference_bias"]
    pi_weight = classifier_state["fc.weight"].to(device)
    fold_count = int(oof["fold_count"])
    rows: list[dict[str, Any]] = []

    for alpha in args.alpha_grid:
        prediction_logits = torch.empty(labels.numel(), 100, device=device)
        for held_out_fold in range(fold_count):
            allowed = np.flatnonzero(oof["fold_ids"].cpu().numpy() != held_out_fold)
            schedule = balanced_schedule(
                labels_np, allowed, args.epochs, labels.numel(),
                args.seed + 70001 + held_out_fold,
            )
            rng = np.random.default_rng(args.seed + 81013 + held_out_fold)
            donor_schedule = rng.choice(allowed, size=schedule.shape, replace=True)
            seed_everything(args.seed + held_out_fold)
            classifier = resnet_cifar.Classifier(feat_in=64, num_classes=100).to(device)
            classifier.load_state_dict(classifier_state)
            optimizer = torch.optim.SGD(
                classifier.parameters(), lr=args.lr * args.lr_factor,
                momentum=args.momentum, weight_decay=args.weight_decay,
            )
            for epoch in range(args.epochs):
                lr = args.lr_factor * args.lr * 0.5 * (
                    1.0 + math.cos(epoch / args.epochs * math.pi)
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                for start in range(0, labels.numel(), args.batch_size):
                    recipient_index = torch.from_numpy(
                        schedule[epoch, start : start + args.batch_size]
                    ).to(device)
                    donor_index = torch.from_numpy(
                        donor_schedule[epoch, start : start + args.batch_size]
                    ).to(device)
                    recipient = features[recipient_index]
                    donor = features[donor_index]
                    target = labels[recipient_index]
                    fold = folds[recipient_index]
                    _, lambda_pi = official_pi_h2t_ratios(recipient, target, pi_weight)
                    lambda0 = lambda_pi * args.donor_mass_scale
                    _, critical = batched_reference_geometry(
                        recipient, donor, target, fold, reference_weight, reference_bias
                    )
                    safe = torch.minimum(lambda0, critical)
                    ratio = safe + alpha * (1.0 - q[recipient_index]) * (lambda0 - safe)
                    fused = (1.0 - ratio[:, None]) * recipient + ratio[:, None] * donor
                    logits = classify_vector(classifier, fused)
                    loss = F.cross_entropy(logits, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            held_out = folds == held_out_fold
            with torch.no_grad():
                prediction_logits[held_out] = classify_vector(classifier, features[held_out])
        metrics = metric_from_logits(prediction_logits, labels)
        row = {"alpha": alpha, **metrics}
        rows.append(row)
        print(f"alpha CV {alpha:.2f}: {metrics}")

    selected = max(rows, key=lambda row: (row["hmean"], row["few"], row["overall"]))
    return float(selected["alpha"]), rows


@torch.no_grad()
def cache_epoch_stream(
    loader: torch.utils.data.DataLoader,
    backbone: torch.nn.Module,
    pi_weight: torch.Tensor,
    oof: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor | float | int]:
    if args.no_shift_bn:
        backbone.eval()
    else:
        backbone.train()
    buffers: dict[str, list[torch.Tensor]] = {
        key: [] for key in (
            "recipient", "donor", "target", "donor_label", "recipient_index",
            "fold", "q", "recipient_mass_pi", "lambda_pi", "lambda0",
            "margin", "critical", "safe", "controlled",
        )
    }
    recipient_checksum = 0.0
    donor_checksum = 0.0
    for batch_index, (images, target, meta) in enumerate(loader):
        if args.max_train_batches and batch_index >= args.max_train_batches:
            break
        images = images.to(device, non_blocking=True)
        donor_images = meta["sample_image"].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        donor_label = meta["sample_label"].to(device, non_blocking=True)
        recipient_index = meta["recipient_index"].to(device, non_blocking=True).long()
        expected = oof["train_labels"][recipient_index]
        if not torch.equal(expected, target):
            raise RuntimeError("OOF archive/sample index mismatch")
        fold = oof["fold_ids"][recipient_index]
        q = oof["q_true_class"][recipient_index]
        recipient = F.adaptive_avg_pool2d(backbone(images), 1).flatten(1)
        donor = F.adaptive_avg_pool2d(backbone(donor_images), 1).flatten(1)
        recipient_mass_pi, lambda_pi = official_pi_h2t_ratios(
            recipient, target, pi_weight
        )
        lambda0 = lambda_pi * args.donor_mass_scale
        margin, critical = batched_reference_geometry(
            recipient, donor, target, fold,
            oof["reference_weight"], oof["reference_bias"],
        )
        safe = torch.minimum(lambda0, critical)
        controlled = safe + args.alpha * (1.0 - q) * (lambda0 - safe)
        values = {
            "recipient": recipient, "donor": donor, "target": target,
            "donor_label": donor_label, "recipient_index": recipient_index,
            "fold": fold, "q": q, "recipient_mass_pi": recipient_mass_pi,
            "lambda_pi": lambda_pi, "lambda0": lambda0, "margin": margin,
            "critical": critical, "safe": safe, "controlled": controlled,
        }
        for key, value in values.items():
            buffers[key].append(value.detach())
        recipient_checksum += float(images.sum())
        donor_checksum += float(donor_images.sum())
    result: dict[str, torch.Tensor | float | int] = {
        key: torch.cat(value) for key, value in buffers.items()
    }
    result["recipient_stream_checksum"] = recipient_checksum
    result["donor_stream_checksum"] = donor_checksum
    return result


def train_cached_epoch(
    stream: dict[str, torch.Tensor | float | int],
    backbone: torch.nn.Module,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    method: str,
    oof: dict[str, Any],
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    recipient = stream["recipient"]
    donor = stream["donor"]
    target = stream["target"]
    fold = stream["fold"]
    q = stream["q"]
    lambda0 = stream["lambda0"]
    safe = stream["safe"]
    controlled = stream["controlled"]
    assert isinstance(recipient, torch.Tensor)
    assert isinstance(donor, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(fold, torch.Tensor)
    assert isinstance(q, torch.Tensor)
    assert isinstance(lambda0, torch.Tensor)
    assert isinstance(safe, torch.Tensor)
    assert isinstance(controlled, torch.Tensor)
    strata = strata_ids(target, lambda0, q, args.ratio_bins, args.q_bins)
    stratum_count = 3 * args.ratio_bins * args.q_bins
    control_audit = {
        "donor_corruption_moved_fraction": 0.0,
        "donor_corruption_eligible_fraction": 0.0,
        "ratio_reassigned_fraction": 0.0,
        "multiset_match_max_abs_error": 0.0,
    }
    if method == "A":
        ratio = lambda0
    elif method == "T2":
        ratio = torch.zeros_like(lambda0)
    elif method == "D0":
        ratio = safe
    elif method == "Dcc":
        ratio = controlled
    elif method in {"E0", "Ecc"}:
        source = safe if method == "E0" else controlled
        generator = torch.Generator(device=source.device.type)
        generator.manual_seed(args.seed + 900001 + epoch)
        ratio, control_audit = corrupt_geometry_allocate(
            source, recipient, donor, target, fold, strata, stratum_count,
            oof["reference_weight"], oof["reference_bias"], generator,
        )
    else:
        raise ValueError(method)

    fused = (1.0 - ratio[:, None]) * recipient + ratio[:, None] * donor
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    for start in range(0, target.numel(), args.batch_size):
        stop = min(start + args.batch_size, target.numel())
        logits = classify_vector(classifier, fused[start:stop])
        loss = F.cross_entropy(logits, target[start:stop])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * (stop - start)
        total_correct += int((logits.argmax(1) == target[start:stop]).sum())

    with torch.no_grad():
        fused_margin_parts = []
        for start in range(0, target.numel(), 512):
            stop = min(start + 512, target.numel())
            fused_margin_parts.append(
                batched_worst_margin(
                    fused[start:stop], target[start:stop], fold[start:stop],
                    oof["reference_weight"], oof["reference_bias"],
                )
            )
        fused_margin = torch.cat(fused_margin_parts)
        floor = torch.minimum(stream["margin"], torch.zeros_like(stream["margin"]))
        violation_depth = (floor - fused_margin).clamp_min(0.0)
        groups = frequency_groups(target)
        group_mass = [float(ratio[groups == group].mean()) for group in range(3)]
        q_group = [float(q[groups == group].mean()) for group in range(3)]
        relaxed = (ratio - safe).clamp_min(0.0)
    return {
        "loss": total_loss / target.numel(),
        "accuracy": total_correct / target.numel(),
        "pi_donor_mass_mean": float(stream["lambda_pi"].mean()),
        "lambda0_mean": float(lambda0.mean()),
        "safe_mean": float(safe.mean()),
        "controlled_mean": float(controlled.mean()),
        "donor_mass_mean": float(ratio.mean()),
        "donor_mass_std": float(ratio.std(unbiased=False)),
        "donor_mass_many": group_mass[0],
        "donor_mass_medium": group_mass[1],
        "donor_mass_few": group_mass[2],
        "q_mean": float(q.mean()),
        "q_many": q_group[0],
        "q_medium": q_group[1],
        "q_few": q_group[2],
        "relaxation_mean": float(relaxed.mean()),
        "boundary_violation_rate": float((violation_depth > 5e-6).float().mean()),
        "crossing_depth_mean": float(violation_depth.mean()),
        "crossing_depth_p90": float(torch.quantile(violation_depth, 0.9)),
        "crossing_depth_max": float(violation_depth.max()),
        "recipient_stream_checksum": float(stream["recipient_stream_checksum"]),
        "donor_stream_checksum": float(stream["donor_stream_checksum"]),
        **control_audit,
    }


def run_method(
    method: str,
    backbone_state: dict[str, torch.Tensor],
    classifier_state: dict[str, torch.Tensor],
    pif_active: bool,
    oof: dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = method_dir / "metrics.json"
    current_path = method_dir / "current.pth"
    if args.resume and metrics_path.exists():
        print(f"{method}: completed; skipping")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    seed_everything(args.seed)
    backbone, classifier, uses_pif = build_models(
        backbone_state, classifier_state, device, pif_active=pif_active
    )
    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=args.lr * args.lr_factor,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if args.resume and current_path.exists():
        saved = torch.load(current_path, map_location=device, weights_only=False)
        backbone.load_state_dict(saved["backbone"])
        classifier.load_state_dict(saved["classifier"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        start_epoch = int(saved["epoch"])
        print(f"{method}: resumed at epoch {start_epoch}")
    pi_weight = classifier_state["fc.weight"].to(device)
    for epoch in range(start_epoch, args.epochs):
        lr = args.lr_factor * args.lr * 0.5 * (
            1.0 + math.cos(epoch / args.epochs * math.pi)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        stream = cache_epoch_stream(
            train_loader, backbone, pi_weight, oof, args, device
        )
        train_metrics = train_cached_epoch(
            stream, backbone, classifier, optimizer, method, oof, args, epoch
        )
        history.append({"epoch": epoch + 1, "lr": lr, "train": train_metrics})
        torch.save(
            {
                "epoch": epoch + 1, "method": method, "alpha": args.alpha,
                "backbone": backbone.state_dict(), "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history,
            },
            current_path,
        )
        print(
            f"{method} epoch {epoch + 1:02d}/{args.epochs}: "
            f"train={train_metrics['accuracy']:.4f}, lambda={train_metrics['donor_mass_mean']:.4f}, "
            f"M/Md/Fmass={train_metrics['donor_mass_many']:.3f}/"
            f"{train_metrics['donor_mass_medium']:.3f}/{train_metrics['donor_mass_few']:.3f}, "
            f"violation={train_metrics['boundary_violation_rate']:.3f}, "
            f"match={train_metrics['multiset_match_max_abs_error']:.1e}"
        )
    evaluation = evaluate(eval_loader, backbone, classifier, device, args.max_eval_batches)
    evaluation["hmean"] = (
        2.0 * evaluation["many"] * evaluation["few"]
        / max(evaluation["many"] + evaluation["few"], 1e-12)
    )
    result = {
        "method": method, "seed": args.seed, "alpha": args.alpha,
        "uses_pif": uses_pif, "selection": "fixed_last_epoch",
        "last": {"epoch": args.epochs, "train": history[-1]["train"], "eval": evaluation},
        "history": history,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{method} endpoint: {evaluation}")
    return result


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        train = result["last"]["train"]
        evaluation = result["last"]["eval"]
        rows.append({
            "method": result["method"], "seed": result["seed"], "alpha": result["alpha"],
            **evaluation, **{f"train_{key}": value for key, value in train.items()},
        })
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    oof_path = Path(args.oof_reference).resolve()
    backbone_state, classifier_state, metadata = load_stage1_checkpoint(
        checkpoint_path, device
    )
    oof = load_oof(oof_path, device)
    feature_archive = Path(oof["feature_archive"])
    stage1_has_pif = any(key.startswith("conv1d.") for key in backbone_state)
    pif_active = bool(metadata.get("pif_active", stage1_has_pif))
    if args.alpha is None:
        if args.skip_alpha_cv:
            raise ValueError("give --alpha when --skip-alpha-cv is used")
        args.alpha, alpha_rows = select_alpha_cv(
            feature_archive, classifier_state, oof, args, device
        )
    else:
        alpha_rows = []
    experiment = {
        **vars(args),
        "checkpoint": str(checkpoint_path),
        "oof_reference": str(oof_path),
        "feature_archive": str(feature_archive),
        "pif_active": pif_active,
        "alpha_cv": alpha_rows,
        "q_definition": "cross-fitted-temperature p_T^OOF(y_i|f_i), fixed per sample",
        "geometry_definition": "recipient-fold-specific OOF head; true donor for D, corrupted donor only for E allocation",
        "control_definition": "E preserves exact D ratio multiset within frequency x lambda0-bin x q-bin; true pair is fused",
        "selection": "alpha by pooled training-only OOF H-mean, Few, Overall; test evaluated once per arm at fixed last epoch",
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    dataset = CIFAR100_LT(
        dataset_config(), distributed=False, root=args.data_path, imb_factor=0.01,
        batch_size=args.batch_size, num_works=args.workers,
    )
    if len(dataset.train_instance.dataset) != oof["fold_ids"].numel():
        raise ValueError("OOF archive and CIFAR100-LT split lengths differ")
    print(
        f"device={device}, train={len(dataset.train_instance.dataset)}, alpha={args.alpha}, "
        f"methods={args.methods}, PIF={pif_active}"
    )
    results = []
    for method in args.methods:
        seed_everything(args.seed)
        results.append(
            run_method(
                method, backbone_state, classifier_state, pif_active, oof,
                dataset.train_instance, dataset.eval, output_dir, args, device,
            )
        )
        write_summary(output_dir, results)
    print("final endpoints")
    for result in results:
        print(result["method"], result["last"]["eval"])
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
