"""Recipient-only center-r versus boundary-r validation for PI-H2T.

This is intentionally narrower than the prior donor-conditioned experiments.
The H2T recipient/donor stream, frozen PIF Stage-I backbone, Stage-II BN
shift, classifier initialization, optimizer and CE loss are fixed.  Only the
recipient-level scalar H2TF ratio changes:

    A  calibrated original PI-H2T center-based ratio
    B  signed-boundary-margin rank-matched ratio
    C  exact B ratio multiset shuffled across recipients

B uses neither donor feature nor donor label.  C does not recompute or clip a
ratio; it only destroys the margin-to-recipient correspondence inside the same
strata.  ``frequency_only`` strata are the pure boundary-r protocol: each
Many/Medium/Few group retains A's ratio multiset as an identity-free bag, but
no recipient-level center-ratio bin remains.  Thus A/B compares center
information with boundary information, while B/C tests whether the boundary
information is assigned to the correct recipient rather than merely changing a
distribution.
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

from datasets.cifar100 import CIFAR100_LT
from train_boundary_h2t_stage2 import (
    dataset_config,
    frequency_groups,
    load_stage1_checkpoint,
    seed_everything,
)
from train_pi_h2t_ratio_only_stage2 import (
    build_models,
    classify_vector,
    evaluate,
    official_pi_h2t_ratios,
)


METHODS = (
    "center_r",
    "boundary_r",
    "boundary_r_shuffled",
    "nearest_boundary_correction",
    "mbe_correction",
    "mbe_correction_shuffled",
)

CORRECTION_METHODS = (
    "nearest_boundary_correction",
    "mbe_correction",
    "mbe_correction_shuffled",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PI-H2T center-r versus recipient-boundary-r validation"
    )
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument(
        "--donor-mass-scale",
        type=float,
        default=0.9267667644209227,
        help="Frozen PIF matched global scale kappa applied identically to A/B/C.",
    )
    parser.add_argument(
        "--ratio-bins",
        type=int,
        default=4,
        help="A donor-mass bins used only by legacy frequency_lambda_bins stratification.",
    )
    parser.add_argument(
        "--stratification",
        choices=("frequency_lambda_bins", "frequency_only", "frequency_pi_bins"),
        default="frequency_lambda_bins",
        help=(
            "frequency_only is the pure recipient-boundary protocol; "
            "frequency_lambda_bins reproduces the earlier donor-mass-conditioned check; "
            "frequency_pi_bins is the preregistered PI-center-conditioned correction."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-shift-bn",
        action="store_true",
        help="Disable released H2T Stage-II BatchNorm running-statistic shift.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Smoke-test limit; zero means all training batches.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Smoke-test limit; zero evaluates the complete test set.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; choices are {METHODS}")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.ratio_bins < 1:
        raise ValueError("--ratio-bins must be positive")
    if not 0.0 <= args.donor_mass_scale <= 1.0:
        raise ValueError("--donor-mass-scale must be in [0, 1]")
    if any(method in CORRECTION_METHODS for method in args.methods):
        if args.stratification != "frequency_pi_bins":
            raise ValueError(
                "nearest/MBE correction methods require --stratification frequency_pi_bins"
            )
    return args


def boundary_denominator(weight: torch.Tensor) -> torch.Tensor:
    """Pairwise ||w_y-w_j|| used by the normalized signed classifier margin."""
    squared = (weight * weight).sum(dim=1)
    return torch.sqrt(
        torch.clamp(
            squared[:, None] + squared[None, :] - 2.0 * (weight @ weight.t()),
            min=0.0,
        )
    ).clamp_min(1e-8)


def signed_boundary_margin(
    recipient: torch.Tensor,
    target: torch.Tensor,
    reference_weight: torch.Tensor,
    reference_bias: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """m_i=min_j!=y signed normalized recipient-to-boundary distance."""
    logits = F.linear(recipient, reference_weight, reference_bias)
    numerator = logits.gather(1, target[:, None]) - logits
    distance = numerator / denominator[target]
    distance[torch.arange(target.numel(), device=target.device), target] = torch.inf
    return distance.min(dim=1).values


def multi_boundary_exposure(
    recipient: torch.Tensor,
    target: torch.Tensor,
    reference_weight: torch.Tensor,
    reference_bias: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """Hyperparameter-free MBE over every competing frozen-head boundary.

    S_i = logmeanexp_j!=y(-m_ij).  The mean only subtracts the constant
    log(C-1), so it does not alter recipient ranking.  Larger S means greater
    exposure: a smaller nearest margin and/or several similarly close margins.
    """
    logits = F.linear(recipient, reference_weight, reference_bias)
    numerator = logits.gather(1, target[:, None]) - logits
    distance = numerator / denominator[target]
    distance[torch.arange(target.numel(), device=target.device), target] = torch.inf
    return torch.logsumexp(-distance, dim=1) - math.log(reference_weight.shape[0] - 1)


def stratum_ids(
    target: torch.Tensor,
    lambda_a: torch.Tensor,
    recipient_mass_pi: torch.Tensor,
    ratio_bins: int,
    stratification: str,
) -> torch.Tensor:
    """Choose the exact-distribution control stratum, never using boundary/donor."""
    group = frequency_groups(target)
    if stratification == "frequency_only":
        # Pure boundary-r: lambda_A values are an unlabeled group-level bag.
        # The particular recipient's center-derived lambda_A never constrains
        # which boundary rank it can receive.
        return group
    if stratification == "frequency_lambda_bins":
        ratio_value = lambda_a
    elif stratification == "frequency_pi_bins":
        # The correction hypothesis explicitly retains PI-H2T's centre score
        # as the coarse first-order proxy and only reorders ratios locally.
        ratio_value = recipient_mass_pi
    else:
        raise ValueError(f"unknown stratification {stratification}")
    ratio_bin = (ratio_value * ratio_bins).long().clamp(0, ratio_bins - 1)
    return group * ratio_bins + ratio_bin


def stratum_count(args: argparse.Namespace) -> int:
    return 3 if args.stratification == "frequency_only" else 3 * args.ratio_bins


def rank_match_boundary(
    lambda_a: torch.Tensor,
    margin: torch.Tensor,
    strata: torch.Tensor,
    stratum_count: int,
) -> torch.Tensor:
    """Give larger donor mass to recipients with larger signed boundary margin."""
    result = torch.empty_like(lambda_a)
    for stratum in range(stratum_count):
        indices = torch.nonzero(strata == stratum, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        margin_order = torch.argsort(margin[indices], stable=True)
        masses = torch.sort(lambda_a[indices], stable=True).values
        result[indices[margin_order]] = masses
    return result


def shuffle_ratio_correspondence(
    ratio: torch.Tensor,
    strata: torch.Tensor,
    stratum_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use a random derangement, preserving the exact ratio multiset per stratum."""
    result = ratio.clone()
    moved = 0
    eligible = 0
    value_reassigned = 0
    for stratum in range(stratum_count):
        indices = torch.nonzero(strata == stratum, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        # Randomize the order first, then cyclically rotate it.  Every eligible
        # source identity moves, unlike a plain randperm with fixed points.
        shuffled_indices = indices[torch.randperm(
            indices.numel(), device=ratio.device, generator=generator
        )]
        result[shuffled_indices] = ratio[shuffled_indices.roll(1)]
        moved += int(indices.numel())
        eligible += int(indices.numel())
        value_reassigned += int((result[indices] != ratio[indices]).sum())
    return result, {
        "shuffle_moved_fraction": moved / max(eligible, 1),
        "shuffle_eligible_fraction": eligible / ratio.numel(),
        "ratio_reassigned_fraction": value_reassigned / ratio.numel(),
    }


def ratio_multiset_audit(
    expected: torch.Tensor,
    actual: torch.Tensor,
    strata: torch.Tensor,
    stratum_count: int,
) -> float:
    maximum_error = 0.0
    for stratum in range(stratum_count):
        indices = torch.nonzero(strata == stratum, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        error = (torch.sort(expected[indices]).values - torch.sort(actual[indices]).values).abs().max()
        maximum_error = max(maximum_error, float(error))
    return maximum_error


def monotonicity_violations(
    ratio: torch.Tensor,
    margin: torch.Tensor,
    strata: torch.Tensor,
    stratum_count: int,
) -> int:
    violations = 0
    for stratum in range(stratum_count):
        indices = torch.nonzero(strata == stratum, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        ordered = indices[torch.argsort(margin[indices], stable=True)]
        allocation = ratio[ordered]
        violations += int((allocation[1:] + 1e-7 < allocation[:-1]).sum())
    return violations


def spearman_rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Stable-rank Spearman audit; all experiment groups contain >1 sample."""
    if left.numel() < 2:
        return 0.0
    left_rank = torch.argsort(
        torch.argsort(left, stable=True), stable=True
    ).float()
    right_rank = torch.argsort(
        torch.argsort(right, stable=True), stable=True
    ).float()
    left_rank = left_rank - left_rank.mean()
    right_rank = right_rank - right_rank.mean()
    denominator = torch.sqrt(
        (left_rank * left_rank).sum() * (right_rank * right_rank).sum()
    ).clamp_min(1e-12)
    return float((left_rank * right_rank).sum() / denominator)


def recipient_assignment_diagnostics(
    target: torch.Tensor,
    margin: torch.Tensor,
    lambda_a: torch.Tensor,
    assigned_lambda: torch.Tensor,
) -> dict[str, float]:
    """Verify that boundary-r replaced, rather than locally retained, center-r."""
    groups = frequency_groups(target)
    group_names = ("many", "medium", "few")
    result: dict[str, float] = {
        "mean_abs_lambda_change": float((assigned_lambda - lambda_a).abs().mean()),
        "ratio_changed_fraction": float(
            (assigned_lambda.sub(lambda_a).abs() > 1e-7).float().mean()
        ),
    }
    margin_assigned: list[float] = []
    center_assigned: list[float] = []
    margin_center: list[float] = []
    for group_id, group_name in enumerate(group_names):
        mask = groups == group_id
        group_margin = margin[mask]
        group_center = lambda_a[mask]
        group_assigned = assigned_lambda[mask]
        rho_margin_assigned = spearman_rank_correlation(
            group_margin, group_assigned
        )
        rho_center_assigned = spearman_rank_correlation(
            group_center, group_assigned
        )
        rho_margin_center = spearman_rank_correlation(
            group_margin, group_center
        )
        result[f"spearman_margin_assigned_{group_name}"] = rho_margin_assigned
        result[f"spearman_center_assigned_{group_name}"] = rho_center_assigned
        result[f"spearman_margin_center_{group_name}"] = rho_margin_center
        result[f"mean_abs_lambda_change_{group_name}"] = float(
            (group_assigned - group_center).abs().mean()
        )
        result[f"ratio_changed_fraction_{group_name}"] = float(
            (group_assigned.sub(group_center).abs() > 1e-7).float().mean()
        )
        margin_assigned.append(rho_margin_assigned)
        center_assigned.append(rho_center_assigned)
        margin_center.append(rho_margin_center)
    result["spearman_margin_assigned_macro"] = float(np.mean(margin_assigned))
    result["spearman_center_assigned_macro"] = float(np.mean(center_assigned))
    result["spearman_margin_center_macro"] = float(np.mean(margin_center))
    return result


def mbe_assignment_diagnostics(
    target: torch.Tensor,
    mbe: torch.Tensor,
    assigned_lambda: torch.Tensor,
) -> dict[str, float]:
    """Audit MBE correspondence; larger risk must receive less donor mass."""
    groups = frequency_groups(target)
    group_names = ("many", "medium", "few")
    result: dict[str, float] = {}
    group_rhos: list[float] = []
    for group_id, group_name in enumerate(group_names):
        mask = groups == group_id
        rho = spearman_rank_correlation(mbe[mask], assigned_lambda[mask])
        result[f"spearman_mbe_assigned_{group_name}"] = rho
        group_rhos.append(rho)
    result["spearman_mbe_assigned_macro"] = float(np.mean(group_rhos))
    return result


@torch.no_grad()
def cache_epoch_stream(
    loader: torch.utils.data.DataLoader,
    backbone: torch.nn.Module,
    pi_weight: torch.Tensor,
    reference_weight: torch.Tensor,
    reference_bias: torch.Tensor,
    denominator: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor | float | int]:
    """Freeze one identical Stage-II feature/data stream for all ratio arms."""
    if args.no_shift_bn:
        backbone.eval()
    else:
        backbone.train()
    fields = (
        "recipient",
        "donor",
        "target",
        "donor_label",
        "recipient_mass",
        "lambda_pi",
        "lambda_a",
        "margin",
        "mbe",
    )
    buffers: dict[str, list[torch.Tensor]] = {field: [] for field in fields}
    recipient_checksum = 0.0
    donor_checksum = 0.0
    target_checksum = 0
    donor_label_checksum = 0
    for batch_index, (images, target, meta) in enumerate(loader):
        if args.max_train_batches and batch_index >= args.max_train_batches:
            break
        images = images.to(device, non_blocking=True)
        donor_images = meta["sample_image"].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        donor_label = meta["sample_label"].to(device, non_blocking=True)
        recipient = F.adaptive_avg_pool2d(backbone(images), 1).flatten(1)
        donor = F.adaptive_avg_pool2d(backbone(donor_images), 1).flatten(1)
        recipient_mass, lambda_pi = official_pi_h2t_ratios(recipient, target, pi_weight)
        lambda_a = lambda_pi * args.donor_mass_scale
        margin = signed_boundary_margin(
            recipient, target, reference_weight, reference_bias, denominator
        )
        mbe = multi_boundary_exposure(
            recipient, target, reference_weight, reference_bias, denominator
        )
        values = {
            "recipient": recipient,
            "donor": donor,
            "target": target,
            "donor_label": donor_label,
            "recipient_mass": recipient_mass,
            "lambda_pi": lambda_pi,
            "lambda_a": lambda_a,
            "margin": margin,
            "mbe": mbe,
        }
        for field, value in values.items():
            buffers[field].append(value.detach())
        recipient_checksum += float(images.sum())
        donor_checksum += float(donor_images.sum())
        target_checksum += int(target.sum())
        donor_label_checksum += int(donor_label.sum())
    result: dict[str, torch.Tensor | float | int] = {
        field: torch.cat(values) for field, values in buffers.items()
    }
    result.update(
        recipient_stream_checksum=recipient_checksum,
        donor_stream_checksum=donor_checksum,
        target_stream_checksum=target_checksum,
        donor_label_stream_checksum=donor_label_checksum,
    )
    return result


def choose_ratio(
    method: str,
    stream: dict[str, torch.Tensor | float | int],
    args: argparse.Namespace,
    epoch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    target = stream["target"]
    lambda_a = stream["lambda_a"]
    margin = stream["margin"]
    recipient_mass = stream["recipient_mass"]
    mbe = stream["mbe"]
    assert isinstance(target, torch.Tensor)
    assert isinstance(lambda_a, torch.Tensor)
    assert isinstance(margin, torch.Tensor)
    assert isinstance(recipient_mass, torch.Tensor)
    assert isinstance(mbe, torch.Tensor)
    strata = stratum_ids(
        target, lambda_a, recipient_mass, args.ratio_bins, args.stratification
    )
    num_strata = stratum_count(args)
    lambda_b = rank_match_boundary(lambda_a, margin, strata, num_strata)
    # Larger MBE means more risk.  rank_match_boundary expects a safety score
    # whose larger values receive more donor mass, hence the minus sign.
    mbe_safety = -mbe
    lambda_mbe = rank_match_boundary(lambda_a, mbe_safety, strata, num_strata)
    audit = {
        "ratio_multiset_match_max_abs_error": 0.0,
        "boundary_monotonicity_violations": 0.0,
        "allocation_score_monotonicity_violations": 0.0,
        "shuffle_moved_fraction": 0.0,
        "shuffle_eligible_fraction": 0.0,
        "ratio_reassigned_fraction": 0.0,
    }
    if method == "center_r":
        ratio = lambda_a
    elif method == "boundary_r":
        ratio = lambda_b
        audit["ratio_multiset_match_max_abs_error"] = ratio_multiset_audit(
            lambda_a, ratio, strata, num_strata
        )
        audit["boundary_monotonicity_violations"] = float(
            monotonicity_violations(ratio, margin, strata, num_strata)
        )
    elif method == "boundary_r_shuffled":
        generator = torch.Generator(device=device.type)
        generator.manual_seed(args.seed + 900001 + epoch)
        ratio, shuffle_audit = shuffle_ratio_correspondence(
            lambda_b, strata, num_strata, generator
        )
        audit.update(shuffle_audit)
        audit["ratio_multiset_match_max_abs_error"] = ratio_multiset_audit(
            lambda_b, ratio, strata, num_strata
        )
        audit["boundary_monotonicity_violations"] = float(
            monotonicity_violations(ratio, margin, strata, num_strata)
        )
    elif method == "nearest_boundary_correction":
        ratio = lambda_b
        audit["ratio_multiset_match_max_abs_error"] = ratio_multiset_audit(
            lambda_a, ratio, strata, num_strata
        )
        violations = float(monotonicity_violations(
            ratio, margin, strata, num_strata
        ))
        audit["boundary_monotonicity_violations"] = violations
        audit["allocation_score_monotonicity_violations"] = violations
    elif method == "mbe_correction":
        ratio = lambda_mbe
        audit["ratio_multiset_match_max_abs_error"] = ratio_multiset_audit(
            lambda_a, ratio, strata, num_strata
        )
        audit["allocation_score_monotonicity_violations"] = float(
            monotonicity_violations(ratio, mbe_safety, strata, num_strata)
        )
    elif method == "mbe_correction_shuffled":
        generator = torch.Generator(device=device.type)
        generator.manual_seed(args.seed + 1_700_003 + epoch)
        ratio, shuffle_audit = shuffle_ratio_correspondence(
            lambda_mbe, strata, num_strata, generator
        )
        audit.update(shuffle_audit)
        audit["ratio_multiset_match_max_abs_error"] = ratio_multiset_audit(
            lambda_mbe, ratio, strata, num_strata
        )
        audit["allocation_score_monotonicity_violations"] = float(
            monotonicity_violations(ratio, mbe_safety, strata, num_strata)
        )
    else:
        raise ValueError(method)
    return ratio, strata, audit


def class_audit_rows(
    target: torch.Tensor,
    margin: torch.Tensor,
    mbe: torch.Tensor,
    lambda_a: torch.Tensor,
    assigned_lambda: torch.Tensor,
) -> list[dict[str, float | int]]:
    groups = frequency_groups(target)
    rows: list[dict[str, float | int]] = []
    for class_id in range(100):
        mask = target == class_id
        if not bool(mask.any()):
            continue
        rows.append(
            {
                "class_id": class_id,
                "frequency_group": int(groups[mask][0]),
                "count": int(mask.sum()),
                "boundary_margin_mean": float(margin[mask].mean()),
                "boundary_margin_p10": float(torch.quantile(margin[mask], 0.1)),
                "mbe_mean": float(mbe[mask].mean()),
                "mbe_p90": float(torch.quantile(mbe[mask], 0.9)),
                "lambda_a_mean": float(lambda_a[mask].mean()),
                "assigned_lambda_mean": float(assigned_lambda[mask].mean()),
            }
        )
    return rows


def train_cached_epoch(
    stream: dict[str, torch.Tensor | float | int],
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    method: str,
    args: argparse.Namespace,
    epoch: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    recipient = stream["recipient"]
    donor = stream["donor"]
    target = stream["target"]
    margin = stream["margin"]
    mbe = stream["mbe"]
    lambda_a = stream["lambda_a"]
    assert isinstance(recipient, torch.Tensor)
    assert isinstance(donor, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(margin, torch.Tensor)
    assert isinstance(mbe, torch.Tensor)
    assert isinstance(lambda_a, torch.Tensor)
    ratio, _, audit = choose_ratio(method, stream, args, epoch, device)
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
        groups = frequency_groups(target)
        group_mass = [float(ratio[groups == group].mean()) for group in range(3)]
        group_margin = [float(margin[groups == group].mean()) for group in range(3)]
        assignment_diagnostics = recipient_assignment_diagnostics(
            target, margin, lambda_a, ratio
        )
        mbe_diagnostics = mbe_assignment_diagnostics(target, mbe, ratio)
        train_metrics = {
            "loss": total_loss / target.numel(),
            "accuracy": total_correct / target.numel(),
            "center_recipient_mass_mean": float(stream["recipient_mass"].mean()),
            "native_pi_donor_mass_mean": float(stream["lambda_pi"].mean()),
            "center_calibrated_donor_mass_mean": float(lambda_a.mean()),
            "donor_mass_mean": float(ratio.mean()),
            "donor_mass_std": float(ratio.std(unbiased=False)),
            "donor_mass_many": group_mass[0],
            "donor_mass_medium": group_mass[1],
            "donor_mass_few": group_mass[2],
            "boundary_margin_mean": float(margin.mean()),
            "boundary_margin_many": group_margin[0],
            "boundary_margin_medium": group_margin[1],
            "boundary_margin_few": group_margin[2],
            "mbe_mean": float(mbe.mean()),
            "mbe_many": float(mbe[groups == 0].mean()),
            "mbe_medium": float(mbe[groups == 1].mean()),
            "mbe_few": float(mbe[groups == 2].mean()),
            "recipient_stream_checksum": float(stream["recipient_stream_checksum"]),
            "donor_stream_checksum": float(stream["donor_stream_checksum"]),
            "target_stream_checksum": float(stream["target_stream_checksum"]),
            "donor_label_stream_checksum": float(stream["donor_label_stream_checksum"]),
            **audit,
            **assignment_diagnostics,
            **mbe_diagnostics,
        }
    return train_metrics, class_audit_rows(target, margin, mbe, lambda_a, ratio)


def rng_epoch_seed(base_seed: int, epoch: int) -> int:
    """Make augmentation/sampling independent of method and evaluation order."""
    return base_seed + 1_000_003 * (epoch + 1)


def run_method(
    method: str,
    backbone_state: dict[str, torch.Tensor],
    classifier_state: dict[str, torch.Tensor],
    pif_active: bool,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    metric_path = method_dir / "metrics.json"
    current_path = method_dir / "current.pth"
    class_audit_path = method_dir / "last_epoch_class_audit.csv"
    if args.resume and metric_path.exists():
        print(f"{method}: completed result exists; skipping")
        return json.loads(metric_path.read_text(encoding="utf-8"))

    seed_everything(args.seed)
    backbone, classifier, uses_pif = build_models(
        backbone_state, classifier_state, device, pif_active=pif_active
    )
    pi_weight = classifier_state["fc.weight"].to(device).detach().clone()
    reference_weight = classifier_state["fc.weight"].to(device).detach().clone()
    reference_bias = classifier_state["fc.bias"].to(device).detach().clone()
    denominator = boundary_denominator(reference_weight)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.lr * args.lr_factor,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if args.resume and current_path.exists():
        saved = torch.load(current_path, map_location=device, weights_only=False)
        if saved.get("method") != method:
            raise ValueError("resume checkpoint method mismatch")
        backbone.load_state_dict(saved["backbone"])
        classifier.load_state_dict(saved["classifier"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        start_epoch = int(saved["epoch"])
        print(f"{method}: resumed at epoch {start_epoch}")

    last_class_audit: list[dict[str, float | int]] = []
    for epoch in range(start_epoch, args.epochs):
        # This reset is deliberately before the loader iterator.  A/B/C get
        # bitwise-identical recipient/donor/augmentation streams each epoch.
        seed_everything(rng_epoch_seed(args.seed, epoch))
        learning_rate = args.lr_factor * args.lr * 0.5 * (
            1.0 + math.cos(epoch / args.epochs * math.pi)
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        stream = cache_epoch_stream(
            train_loader,
            backbone,
            pi_weight,
            reference_weight,
            reference_bias,
            denominator,
            args,
            device,
        )
        train_metrics, last_class_audit = train_cached_epoch(
            stream, classifier, optimizer, method, args, epoch, device
        )
        history.append({"epoch": epoch + 1, "lr": learning_rate, "train": train_metrics})
        torch.save(
            {
                "epoch": epoch + 1,
                "method": method,
                "backbone": backbone.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
            },
            current_path,
        )
        print(
            f"{method} epoch {epoch + 1:02d}/{args.epochs}: "
            f"train={train_metrics['accuracy']:.4f}, "
            f"lambda={train_metrics['donor_mass_mean']:.4f}, "
            f"m={train_metrics['boundary_margin_mean']:.4f}, "
            f"match={train_metrics['ratio_multiset_match_max_abs_error']:.1e}, "
            f"mono={train_metrics['allocation_score_monotonicity_violations']:.0f}, "
            f"rho(m,lambda)={train_metrics['spearman_margin_assigned_macro']:.3f}, "
            f"rho(MBE,lambda)={train_metrics['spearman_mbe_assigned_macro']:.3f}, "
            f"rho(A,lambda)={train_metrics['spearman_center_assigned_macro']:.3f}, "
            f"moved={train_metrics['shuffle_moved_fraction']:.3f}"
        )

    evaluation = evaluate(eval_loader, backbone, classifier, device, args.max_eval_batches)
    evaluation["hmean"] = (
        2.0 * evaluation["many"] * evaluation["few"]
        / max(evaluation["many"] + evaluation["few"], 1e-12)
    )
    result = {
        "method": method,
        "seed": args.seed,
        "uses_pif": uses_pif,
        "selection": "fixed_last_epoch_only",
        "last": {"epoch": args.epochs, "train": history[-1]["train"], "eval": evaluation},
        "history": history,
    }
    metric_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with class_audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(last_class_audit[0]))
        writer.writeheader()
        writer.writerows(last_class_audit)
    print(f"{method} endpoint: {evaluation}")
    return result


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        train = result["last"]["train"]
        evaluation = result["last"]["eval"]
        rows.append(
            {
                "method": result["method"],
                "seed": result["seed"],
                **evaluation,
                **{f"train_{key}": value for key, value in train.items()},
            }
        )
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
    backbone_state, classifier_state, metadata = load_stage1_checkpoint(
        checkpoint_path, device
    )
    stage1_has_pif = any(key.startswith("conv1d.") for key in backbone_state)
    pif_active = bool(metadata.get("pif_active", stage1_has_pif))
    experiment = {
        **vars(args),
        "checkpoint": str(checkpoint_path),
        "pif_active": pif_active,
        "center_ratio": "released PI-H2T r_PI; lambda_A=donor_mass_scale*(1-r_PI)",
        "boundary_ratio": (
            "m_i=min_j!=y ((w_y-w_j)^T f_i+(b_y-b_j))/"
            "(||w_y-w_j||_2+eps); m ascending receives lambda_A ascending "
            "within the configured exact-distribution stratum"
        ),
        "nearest_boundary_correction": (
            "within each Many/Medium/Few x native PI-recipient-r bin, preserve "
            "the exact official PI donor-mass multiset and assign smaller donor "
            "mass to smaller signed nearest-boundary margin"
        ),
        "mbe": (
            "S_i=logmeanexp_j!=y(-m_ij), where m_ij is the normalized signed "
            "frozen-head boundary margin; larger exposure receives smaller donor mass"
        ),
        "mbe_shuffled_control": (
            "preserve the exact MBE-correction ratio multiset in every PI-centre "
            "stratum, but derange ratio-to-recipient correspondence"
        ),
        "shuffled_control": (
            "exact B ratio multiset shuffled only across recipients inside the same "
            "configured stratum; no recomputation, clipping, donor geometry, q, OOF, or delta"
        ),
        "stratification_interpretation": (
            "frequency_only: pure boundary assignment within Many/Medium/Few, "
            "where lambda_A is only an identity-free group-level multiset; "
            "frequency_lambda_bins: legacy donor-mass-conditioned ablation; "
            "frequency_pi_bins: PI center-r is retained as the coarse proxy and "
            "boundary information only corrects local correspondence"
        ),
        "fixed_protocol": (
            "same PIF Stage-I backbone, original CE reference head, H2T sampling, "
            "BN shift, classifier initialization, CE, optimizer and fixed final epoch"
        ),
        "primary_success": (
            "mbe_correction > center_r, nearest_boundary_correction, and "
            "mbe_correction_shuffled on Few/H-mean, without material Medium/Overall loss"
        ),
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    dataset = CIFAR100_LT(
        dataset_config(),
        distributed=False,
        root=args.data_path,
        imb_factor=0.01,
        batch_size=args.batch_size,
        num_works=args.workers,
    )
    print(
        f"device={device}, train={len(dataset.train_instance.dataset)}, "
        f"test={len(dataset.eval.dataset)}, PIF={pif_active}, "
        f"kappa={args.donor_mass_scale:.6f}, stratification={args.stratification}, "
        f"methods={args.methods}"
    )
    results: list[dict[str, Any]] = []
    for method in args.methods:
        results.append(
            run_method(
                method,
                backbone_state,
                classifier_state,
                pif_active,
                dataset.train_instance,
                dataset.eval,
                output_dir,
                args,
                device,
            )
        )
        write_summary(output_dir, results)
    print("fixed endpoint A/B/C")
    for result in results:
        print(result["method"], result["last"]["eval"])
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
