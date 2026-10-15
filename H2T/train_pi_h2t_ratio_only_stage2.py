"""Paper-faithful PI-H2T H2TF ratio-only experiments.

This deliberately contains no donor selection, balanced reference head,
risk-weighted loss, fixed donor mass, or additional supervision.  Every arm
uses the released H2T sampling protocol:

* class-balanced sample ``f_B`` as the labelled/fused recipient;
* instance-wise sample ``f_I`` as the donor/fusing sample;
* a frozen Stage-I backbone with Stage-II BatchNorm shift and a trainable
  classifier initialized from the same Stage-I checkpoint.

The only intervention is how the scalar H2TF ratio is allocated:

``pi_recipient_only``
    Original PI-H2T Eq. (7)--(8): r_PI is the recipient mass calculated from
    the recipient's cosine distance to its Stage-I classifier weight.

``pair_stratified_rankmatch``
    Preserve the *exact* donor-mass multiset from PI-H2T inside every
    (Many/Medium/Few, PI-ratio-bin) stratum, but allocate larger donor mass to
    recipient--donor pairs with larger Stage-I boundary critical ratio.

``pair_stratified_rankmatch_shuffled``
    Preserve B's same ratio multiset and the same data pairs, then shuffle
    ratio-to-pair correspondence only inside the same strata.

Therefore B>A tests recipient--donor conditioning, and B>C tests that an
improvement comes from the correct correspondence rather than a new ratio
distribution.

``pair_boundary_clip``
    Start from the globally calibrated PI-H2T donor mass ``lambda_0`` and clip
    it to the largest donor contribution that does not reduce the recipient's
    current worst frozen-Stage-I reference margin.  If the recipient is already
    misclassified by the reference head, the current worst margin is preserved
    rather than requiring the fused feature to become correctly classified.

``pair_boundary_clip_shuffled``
    Preserve D's exact clipped-ratio multiset inside every
    (Many/Medium/Few, lambda_0-bin) stratum, but shuffle its correspondence to
    the unchanged recipient--donor pairs.  The shuffled ratios are used
    directly and are never projected again.
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
    ResNetFeatureOptionB,
    dataset_config,
    frequency_groups,
    load_stage1_checkpoint,
    reference_geometry,
    seed_everything,
)


METHODS = (
    "pi_recipient_only",
    "pair_stratified_rankmatch",
    "pair_stratified_rankmatch_shuffled",
    "pair_boundary_clip",
    "pair_boundary_clip_shuffled",
)

BOUNDARY_CLIP_METHODS = (
    "pair_boundary_clip",
    "pair_boundary_clip_shuffled",
)


def json_scalar(value: Any) -> Any:
    """Convert legacy checkpoint scalar tensors before writing experiment JSON."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("checkpoint metric expected to be scalar")
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PI-H2T H2TF recipient-only versus pair-conditioned ratio"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--geometry-reference-head",
        default=None,
        help=(
            "Optional frozen classifier checkpoint used only for boundary geometry. "
            "The trainable Stage-II classifier is still initialized from --checkpoint."
        ),
    )
    parser.add_argument("--data-path", default="./data/cifar100")
    parser.add_argument(
        "--output-dir",
        default="./outputs/pi_h2t_ratio_only_seed42",
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated subset of the fixed A/B/C protocol.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--ratio-bins",
        type=int,
        default=4,
        help="Number of fixed [0,1] bins of the native PI-H2T recipient ratio.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
        help="Boundary-budget tolerance used only to compute pair critical ratios.",
    )
    parser.add_argument(
        "--donor-mass-scale",
        type=float,
        default=1.0,
        help=(
            "Pre-registered global scale kappa applied after ratio allocation. "
            "Use zero for the no-fusion endpoint while retaining both backbone "
            "forwards and the native Stage-II data/BN stream."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-shift-bn",
        action="store_true",
        help="Keep frozen backbone BatchNorm buffers fixed instead of H2T Stage-II shift.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Smoke-test limit; zero means a complete epoch.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Smoke-test limit; zero evaluates all test batches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted arm from its current checkpoint.",
    )
    args = parser.parse_args()
    args.methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; choices are {METHODS}")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.ratio_bins < 1:
        raise ValueError("--ratio-bins must be at least one")
    if args.delta < 0.0:
        raise ValueError("--delta must be non-negative")
    if any(method in BOUNDARY_CLIP_METHODS for method in args.methods) and args.delta != 0.0:
        raise ValueError("boundary-clip D/E are frozen at --delta 0")
    if not 0.0 <= args.donor_mass_scale <= 1.0:
        raise ValueError("--donor-mass-scale must lie inside [0, 1]")
    return args


def load_geometry_reference(
    reference_path: Path | None,
    fallback_classifier_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Load a frozen linear geometry head without changing classifier init."""
    if reference_path is None:
        return {
            "fc.weight": fallback_classifier_state["fc.weight"].detach().cpu().clone(),
            "fc.bias": fallback_classifier_state["fc.bias"].detach().cpu().clone(),
        }
    raw = torch.load(reference_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError("geometry reference checkpoint must be a dictionary")
    candidates = []
    if "classifier" in raw:
        candidates.append(raw["classifier"])
    if "state_dict" in raw:
        candidates.append(raw["state_dict"])
    if "state_dict_classifier" in raw:
        candidates.append(raw["state_dict_classifier"])
    for candidate in candidates:
        state = {
            key.removeprefix("module."): value
            for key, value in candidate.items()
        }
        if "fc.weight" in state and "fc.bias" in state:
            result = {
                "fc.weight": state["fc.weight"].detach().cpu().clone(),
                "fc.bias": state["fc.bias"].detach().cpu().clone(),
            }
            if result["fc.weight"].shape != fallback_classifier_state["fc.weight"].shape:
                raise ValueError("geometry reference weight shape does not match Stage-I head")
            if result["fc.bias"].shape != fallback_classifier_state["fc.bias"].shape:
                raise ValueError("geometry reference bias shape does not match Stage-I head")
            if not all(torch.isfinite(value).all() for value in result.values()):
                raise ValueError("geometry reference contains non-finite values")
            return result
    raise KeyError(
        "geometry reference checkpoint needs classifier/state_dict/"
        "state_dict_classifier with fc.weight and fc.bias"
    )


def build_models(
    backbone_state: dict[str, torch.Tensor],
    classifier_state: dict[str, torch.Tensor],
    device: torch.device,
    pif_active: bool | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module, bool]:
    """Recreate Stage-I, distinguishing PIF parameters from PIF activation."""
    uses_projection_shortcut = any(
        ".shortcut.0.weight" in key for key in backbone_state
    )
    has_pif_parameters = any(key.startswith("conv1d.") for key in backbone_state)
    uses_pif = has_pif_parameters if pif_active is None else bool(pif_active)
    if uses_pif and not has_pif_parameters:
        raise ValueError("checkpoint metadata activates PIF but has no PIF parameters")
    if uses_projection_shortcut:
        backbone = ResNetFeatureOptionB(
            resnet_cifar.BasicBlock,
            [5, 5, 5],
            feature_tuning=has_pif_parameters,
        ).to(device)
    else:
        backbone = resnet_cifar.ResNet_fe(
            resnet_cifar.BasicBlock,
            [5, 5, 5],
            feature_tuning=has_pif_parameters,
        ).to(device)
    classifier = resnet_cifar.Classifier(feat_in=64, num_classes=100).to(device)
    backbone.load_state_dict(backbone_state, strict=True)
    if hasattr(backbone, "set_pif_active"):
        backbone.set_pif_active(uses_pif)
    classifier.load_state_dict(classifier_state, strict=True)
    backbone.requires_grad_(False)
    return backbone, classifier, uses_pif


def pooled(feature_map: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)


def classify_vector(
    classifier: torch.nn.Module,
    feature: torch.Tensor,
) -> torch.Tensor:
    """The classifier part of the paper's post-pooling H2TF Eq. (7)."""
    return F.linear(feature, classifier.fc.weight, classifier.fc.bias)


def official_pi_h2t_ratios(
    recipient_feature: torch.Tensor,
    target: torch.Tensor,
    stage1_weight: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native PI-H2T recipient mass r and donor mass lambda=1-r.

    Eq. (8) uses cosine distance in [0, 2] and linear min-max normalization,
    so r_PI=(1-cos(f_B,w_y))/2.  This calculation intentionally has no donor
    input, which is the defect being tested.
    """
    cosine = F.cosine_similarity(
        recipient_feature,
        stage1_weight[target],
        dim=1,
        eps=eps,
    )
    recipient_mass = torch.clamp((1.0 - cosine) / 2.0, min=0.0, max=1.0)
    donor_mass = 1.0 - recipient_mass
    return recipient_mass, donor_mass


def stratum_ids(
    target: torch.Tensor,
    recipient_mass_pi: torch.Tensor,
    ratio_bins: int,
) -> torch.Tensor:
    """Many/Medium/Few x fixed native PI-H2T-ratio bins."""
    group = frequency_groups(target)
    bin_id = torch.clamp(
        (recipient_mass_pi * ratio_bins).to(torch.long),
        min=0,
        max=ratio_bins - 1,
    )
    return group * ratio_bins + bin_id


def donor_mass_stratum_ids(
    target: torch.Tensor,
    base_donor_mass: torch.Tensor,
    ratio_bins: int,
) -> torch.Tensor:
    """Many/Medium/Few x fixed lambda_0 bins for the D/E causal control."""
    group = frequency_groups(target)
    bin_id = torch.clamp(
        (base_donor_mass * ratio_bins).to(torch.long),
        min=0,
        max=ratio_bins - 1,
    )
    return group * ratio_bins + bin_id


def rank_match_within_strata(
    pi_donor_mass: torch.Tensor,
    critical: torch.Tensor,
    strata: torch.Tensor,
    stratum_count: int,
) -> torch.Tensor:
    """Allocate the native PI donor-mass multiset by pair critical-ratio rank.

    Both the input PI ratios and the output ratios have exactly the same values
    in every stratum.  High critical ratio means the concrete donor admits more
    donor mass, hence critical ascending maps to donor mass ascending.
    """
    result = torch.empty_like(pi_donor_mass)
    for stratum in range(stratum_count):
        indices = torch.nonzero(strata == stratum, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        score_order = torch.argsort(critical[indices], stable=True)
        masses = torch.sort(pi_donor_mass[indices], stable=True).values
        result[indices[score_order]] = masses
    return result


def shuffle_within_strata(
    ratio: torch.Tensor,
    strata: torch.Tensor,
    stratum_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int, int]:
    """Destroy pair correspondence but preserve every stratum's ratio multiset."""
    result = ratio.clone()
    moved = 0
    eligible = 0
    for stratum in range(stratum_count):
        indices = torch.nonzero(strata == stratum, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        permutation = torch.randperm(
            indices.numel(),
            device=ratio.device,
            generator=generator,
        )
        result[indices] = ratio[indices[permutation]]
        moved += int((permutation != torch.arange(indices.numel(), device=ratio.device)).sum())
        eligible += int(indices.numel())
    return result, moved, eligible


def assignment_audit(
    pi_donor_mass: torch.Tensor,
    assigned_donor_mass: torch.Tensor,
    critical: torch.Tensor,
    strata: torch.Tensor,
    stratum_count: int,
) -> tuple[float, int]:
    """Return multiset mismatch and monotonicity violations per batch."""
    maximum_error = 0.0
    monotonicity_violations = 0
    for stratum in range(stratum_count):
        indices = torch.nonzero(strata == stratum, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        expected = torch.sort(pi_donor_mass[indices], stable=True).values
        actual = torch.sort(assigned_donor_mass[indices], stable=True).values
        maximum_error = max(maximum_error, float((expected - actual).abs().max()))
        if indices.numel() > 1:
            order = torch.argsort(critical[indices], stable=True)
            allocation = assigned_donor_mass[indices[order]]
            monotonicity_violations += int(
                (allocation[1:] + 1e-7 < allocation[:-1]).sum()
            )
    return maximum_error, monotonicity_violations


def make_local_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def reference_worst_margin(
    feature: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """Normalized worst y-vs-j margin under the frozen linear reference head."""
    logits = F.linear(feature, weight, bias)
    numerator = logits.gather(1, target[:, None]) - logits
    distance = numerator / denominator[target]
    row = torch.arange(target.numel(), device=target.device)
    distance[row, target] = torch.inf
    return distance.min(dim=1).values


@torch.no_grad()
def evaluate(
    loader: torch.utils.data.DataLoader,
    backbone: torch.nn.Module,
    classifier: torch.nn.Module,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    backbone.eval()
    classifier.eval()
    correct = torch.zeros(100, device=device)
    count = torch.zeros(100, device=device)
    total_loss = 0.0
    total = 0
    for batch_index, (images, target) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = classifier(backbone(images))
        total_loss += float(F.cross_entropy(logits, target, reduction="sum"))
        prediction = logits.argmax(dim=1)
        count.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float32))
        correct.scatter_add_(0, target, (prediction == target).float())
        total += target.numel()
    class_accuracy = correct / torch.clamp(count, min=1.0)
    return {
        "loss": total_loss / max(total, 1),
        "overall": float(correct.sum() / torch.clamp(count.sum(), min=1.0)),
        "many": float(class_accuracy[:36].mean()),
        "medium": float(class_accuracy[36:71].mean()),
        "few": float(class_accuracy[71:].mean()),
    }


def train_epoch(
    loader: torch.utils.data.DataLoader,
    backbone: torch.nn.Module,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    method: str,
    pi_weight: torch.Tensor,
    reference_weight: torch.Tensor,
    reference_bias: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    shuffle_generator: torch.Generator,
) -> dict[str, float]:
    if args.no_shift_bn:
        backbone.eval()
    else:
        backbone.train()
    classifier.train()

    total_loss = 0.0
    total_correct = 0
    total = 0
    pi_recipient_mass_sum = 0.0
    pi_donor_mass_sum = 0.0
    base_donor_mass_sum = 0.0
    projected_donor_mass_sum = 0.0
    donor_mass_sum = 0.0
    donor_mass_square_sum = 0.0
    critical_sum = 0.0
    group_donor_mass_sum = np.zeros(3, dtype=np.float64)
    group_count = np.zeros(3, dtype=np.int64)
    reference_trusted = 0
    group_reference_trusted = np.zeros(3, dtype=np.int64)
    multiset_error = 0.0
    monotonicity_violations = 0
    shuffle_moved = 0
    shuffle_eligible = 0
    projection_clipped = 0
    zero_donor_mass = 0
    reference_violations = 0
    reference_margin_before_sum = 0.0
    reference_margin_after_sum = 0.0
    reference_floor_gap_sum = 0.0
    recipient_checksum = 0.0
    donor_checksum = 0.0
    target_checksum = 0
    donor_label_checksum = 0
    stratum_count = 3 * args.ratio_bins
    weight_sq = (reference_weight * reference_weight).sum(dim=1)
    reference_denominator = torch.sqrt(
        torch.clamp(
            weight_sq[:, None]
            + weight_sq[None, :]
            - 2.0 * (reference_weight @ reference_weight.t()),
            min=0.0,
        )
    ).clamp_min(1e-8)

    for batch_index, (images, target, meta) in enumerate(loader):
        if args.max_train_batches and batch_index >= args.max_train_batches:
            break
        donor_images = meta["sample_image"].to(device, non_blocking=True)
        donor_labels = meta["sample_label"].to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # The two frozen-backbone forwards, their order, and Stage-II BN shift
        # match the released H2T recipe.  The only later intervention is ratio.
        with torch.no_grad():
            recipient_vector = pooled(backbone(images))
            donor_vector = pooled(backbone(donor_images))
            recipient_mass_pi, donor_mass_pi = official_pi_h2t_ratios(
                recipient_vector,
                target,
                pi_weight,
            )
            _, recipient_margin, critical = reference_geometry(
                recipient_vector,
                donor_vector,
                target,
                reference_weight,
                reference_bias,
                args.delta,
            )
            strata = stratum_ids(target, recipient_mass_pi, args.ratio_bins)
            pair_donor_mass = rank_match_within_strata(
                donor_mass_pi,
                critical,
                strata,
                stratum_count,
            )
            base_donor_mass = donor_mass_pi * args.donor_mass_scale
            projected_donor_mass = torch.minimum(base_donor_mass, critical)
            projected_strata = donor_mass_stratum_ids(
                target,
                base_donor_mass,
                args.ratio_bins,
            )
            if method == "pi_recipient_only":
                donor_mass = base_donor_mass
                audit_expected = base_donor_mass
                audit_strata = strata
            elif method == "pair_stratified_rankmatch":
                donor_mass = pair_donor_mass * args.donor_mass_scale
                audit_expected = base_donor_mass
                audit_strata = strata
            elif method == "pair_stratified_rankmatch_shuffled":
                donor_mass, moved, eligible = shuffle_within_strata(
                    pair_donor_mass,
                    strata,
                    stratum_count,
                    shuffle_generator,
                )
                donor_mass = donor_mass * args.donor_mass_scale
                audit_expected = base_donor_mass
                audit_strata = strata
                shuffle_moved += moved
                shuffle_eligible += eligible
            elif method == "pair_boundary_clip":
                donor_mass = projected_donor_mass
                audit_expected = projected_donor_mass
                audit_strata = projected_strata
            elif method == "pair_boundary_clip_shuffled":
                donor_mass, moved, eligible = shuffle_within_strata(
                    projected_donor_mass,
                    projected_strata,
                    stratum_count,
                    shuffle_generator,
                )
                audit_expected = projected_donor_mass
                audit_strata = projected_strata
                shuffle_moved += moved
                shuffle_eligible += eligible
            else:
                raise ValueError(f"unknown method: {method}")
            batch_multiset_error, batch_violations = assignment_audit(
                audit_expected,
                donor_mass,
                critical,
                audit_strata,
                stratum_count,
            )
            multiset_error = max(multiset_error, batch_multiset_error)
            monotonicity_violations += batch_violations
            # Eq. (7) written in donor-mass form: f~= (1-lambda) f_B+lambda f_I.
            fused_vector = (
                (1.0 - donor_mass[:, None]) * recipient_vector
                + donor_mass[:, None] * donor_vector
            )
            fused_margin = reference_worst_margin(
                fused_vector,
                target,
                reference_weight,
                reference_bias,
                reference_denominator,
            )
            reference_floor = torch.minimum(
                recipient_margin,
                torch.zeros_like(recipient_margin),
            ) - args.delta
            reference_trust_mask = recipient_margin >= 0.0

        logits = classify_vector(classifier, fused_vector)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = target.numel()
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == target).sum())
        pi_recipient_mass_sum += float(recipient_mass_pi.sum())
        pi_donor_mass_sum += float(donor_mass_pi.sum())
        base_donor_mass_sum += float(base_donor_mass.sum())
        projected_donor_mass_sum += float(projected_donor_mass.sum())
        donor_mass_sum += float(donor_mass.sum())
        donor_mass_square_sum += float((donor_mass * donor_mass).sum())
        critical_sum += float(critical.sum())
        projection_clipped += int(
            (projected_donor_mass + 1e-7 < base_donor_mass).sum()
        )
        zero_donor_mass += int((donor_mass <= 1e-8).sum())
        reference_violations += int(
            (fused_margin + 5e-6 < reference_floor).sum()
        )
        reference_margin_before_sum += float(recipient_margin.sum())
        reference_margin_after_sum += float(fused_margin.sum())
        reference_floor_gap_sum += float((fused_margin - reference_floor).sum())
        reference_trusted += int(reference_trust_mask.sum())
        groups = frequency_groups(target)
        for group_id in range(3):
            mask = groups == group_id
            group_donor_mass_sum[group_id] += float(donor_mass[mask].sum())
            group_count[group_id] += int(mask.sum())
            group_reference_trusted[group_id] += int(reference_trust_mask[mask].sum())
        # These values audit that A/B/C were offered identical data streams;
        # donor identity is never changed by this experiment.
        recipient_checksum += float(images.detach().sum())
        donor_checksum += float(donor_images.detach().sum())
        target_checksum += int(target.detach().sum())
        donor_label_checksum += int(donor_labels.detach().sum())
        total += batch_size

    donor_mass_mean = donor_mass_sum / max(total, 1)
    donor_mass_variance = donor_mass_square_sum / max(total, 1) - donor_mass_mean**2
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "pi_recipient_mass_mean": pi_recipient_mass_sum / max(total, 1),
        "pi_donor_mass_mean": pi_donor_mass_sum / max(total, 1),
        "base_donor_mass_mean": base_donor_mass_sum / max(total, 1),
        "projected_donor_mass_mean": projected_donor_mass_sum / max(total, 1),
        "donor_mass_mean": donor_mass_mean,
        "donor_mass_std": math.sqrt(max(donor_mass_variance, 0.0)),
        "donor_mass_many": group_donor_mass_sum[0] / max(group_count[0], 1),
        "donor_mass_medium": group_donor_mass_sum[1] / max(group_count[1], 1),
        "donor_mass_few": group_donor_mass_sum[2] / max(group_count[2], 1),
        "critical_mean": critical_sum / max(total, 1),
        "multiset_match_max_abs_error": multiset_error,
        "critical_allocation_monotonicity_violations": monotonicity_violations,
        "shuffle_moved_fraction": shuffle_moved / max(shuffle_eligible, 1),
        "projection_clipping_fraction": projection_clipped / max(total, 1),
        "zero_donor_mass_fraction": zero_donor_mass / max(total, 1),
        "reference_boundary_violation_rate": reference_violations / max(total, 1),
        "reference_margin_before_mean": reference_margin_before_sum / max(total, 1),
        "reference_margin_after_mean": reference_margin_after_sum / max(total, 1),
        "reference_floor_gap_mean": reference_floor_gap_sum / max(total, 1),
        "reference_trusted_fraction": reference_trusted / max(total, 1),
        "reference_trusted_many": group_reference_trusted[0] / max(group_count[0], 1),
        "reference_trusted_medium": group_reference_trusted[1] / max(group_count[1], 1),
        "reference_trusted_few": group_reference_trusted[2] / max(group_count[2], 1),
        "recipient_stream_checksum": recipient_checksum,
        "donor_stream_checksum": donor_checksum,
        "target_stream_checksum": target_checksum,
        "donor_label_stream_checksum": donor_label_checksum,
    }


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_method(
    method: str,
    backbone_state: dict[str, torch.Tensor],
    classifier_state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
    pif_active: bool,
) -> dict[str, Any]:
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    result_path = method_dir / "metrics.json"
    current_path = method_dir / "current.pth"
    best_path = method_dir / "best.pth"
    if args.resume and result_path.exists():
        print(f"{method}: completed result exists; skipping")
        return json.loads(result_path.read_text(encoding="utf-8"))

    seed_everything(args.seed)
    backbone, classifier, uses_pif = build_models(
        backbone_state,
        classifier_state,
        device,
        pif_active=pif_active,
    )
    # Geometry is immutable and never follows the trainable Stage-II classifier.
    # Importantly, the trainable classifier above still starts from the original
    # Stage-I state even when a balanced geometry reference is supplied.
    pi_weight = classifier_state["fc.weight"].to(device).detach().clone()
    reference_weight = reference_state["fc.weight"].to(device).detach().clone()
    reference_bias = reference_state["fc.bias"].to(device).detach().clone()
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.lr * args.lr_factor,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    shuffle_generator = make_local_generator(device, args.seed + 900_001)
    start_epoch = 0
    best_overall = -1.0
    history: list[dict[str, Any]] = []

    if args.resume and current_path.exists():
        saved = torch.load(current_path, map_location=device, weights_only=False)
        if saved.get("method") != method:
            raise ValueError("resume checkpoint method does not match requested method")
        if saved.get("ratio_bins") != args.ratio_bins:
            raise ValueError("resume checkpoint ratio-bin setting does not match")
        if saved.get("donor_mass_scale", 1.0) != args.donor_mass_scale:
            raise ValueError("resume checkpoint donor-mass scale does not match")
        backbone.load_state_dict(saved["backbone"])
        classifier.load_state_dict(saved["classifier"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"])
        best_overall = float(saved["best_overall"])
        history = saved["history"]
        restore_rng_state(saved["rng"])
        if saved.get("shuffle_generator") is not None:
            shuffle_generator.set_state(saved["shuffle_generator"])
        print(f"{method}: resumed at epoch {start_epoch}")

    initial = evaluate(eval_loader, backbone, classifier, device, args.max_eval_batches)
    print(
        f"{method}: initial overall={initial['overall']:.4f}, "
        f"many={initial['many']:.4f}, medium={initial['medium']:.4f}, "
        f"few={initial['few']:.4f}, pif={uses_pif}"
    )
    for epoch in range(start_epoch, args.epochs):
        learning_rate = args.lr_factor * args.lr * 0.5 * (
            1.0 + math.cos(epoch / args.epochs * math.pi)
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train_metrics = train_epoch(
            train_loader,
            backbone,
            classifier,
            optimizer,
            method,
            pi_weight,
            reference_weight,
            reference_bias,
            device,
            args,
            shuffle_generator,
        )
        evaluation = evaluate(eval_loader, backbone, classifier, device, args.max_eval_batches)
        row = {"epoch": epoch + 1, "lr": learning_rate, "train": train_metrics, "eval": evaluation}
        history.append(row)
        is_best = evaluation["overall"] > best_overall
        best_overall = max(best_overall, evaluation["overall"])
        state = {
            "epoch": epoch + 1,
            "method": method,
            "ratio_bins": args.ratio_bins,
            "donor_mass_scale": args.donor_mass_scale,
            "uses_pif": uses_pif,
            "backbone": backbone.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_overall": best_overall,
            "history": history,
            "rng": rng_state(),
            "shuffle_generator": shuffle_generator.get_state(),
        }
        torch.save(state, current_path)
        if is_best:
            torch.save(state, best_path)
        print(
            f"  epoch {epoch + 1:02d}/{args.epochs}: "
            f"train={train_metrics['accuracy']:.4f}, test={evaluation['overall']:.4f}, "
            f"M/Md/F={evaluation['many']:.4f}/{evaluation['medium']:.4f}/"
            f"{evaluation['few']:.4f}, lambda={train_metrics['donor_mass_mean']:.4f}, "
            f"clip={train_metrics['projection_clipping_fraction']:.3f}, "
            f"ref_violation={train_metrics['reference_boundary_violation_rate']:.3e}, "
            f"match_err={train_metrics['multiset_match_max_abs_error']:.2e}, "
            f"violations={train_metrics['critical_allocation_monotonicity_violations']}"
        )

    result = {
        "method": method,
        "uses_pif": uses_pif,
        "geometry_head": (
            "frozen_stage1_classifier"
            if args.geometry_reference_head is None
            else "frozen_external_balanced_classifier"
        ),
        "ratio_bins": args.ratio_bins,
        "donor_mass_scale": args.donor_mass_scale,
        "initial": initial,
        "last": history[-1],
        "best": max(history, key=lambda item: item["eval"]["overall"]),
        "history": history,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        for selection in ("last", "best"):
            row = result[selection]
            train = row["train"]
            evaluation = row["eval"]
            rows.append(
                {
                    "method": result["method"],
                    "selection": selection,
                    "epoch": row["epoch"],
                    "overall": evaluation["overall"],
                    "many": evaluation["many"],
                    "medium": evaluation["medium"],
                    "few": evaluation["few"],
                    "train_accuracy": train["accuracy"],
                    "pi_recipient_mass_mean": train["pi_recipient_mass_mean"],
                    "pi_donor_mass_mean": train["pi_donor_mass_mean"],
                    "donor_mass_scale": result["donor_mass_scale"],
                    "base_donor_mass_mean": train["base_donor_mass_mean"],
                    "projected_donor_mass_mean": train["projected_donor_mass_mean"],
                    "donor_mass_mean": train["donor_mass_mean"],
                    "donor_mass_std": train["donor_mass_std"],
                    "donor_mass_many": train["donor_mass_many"],
                    "donor_mass_medium": train["donor_mass_medium"],
                    "donor_mass_few": train["donor_mass_few"],
                    "critical_mean": train["critical_mean"],
                    "multiset_match_max_abs_error": train[
                        "multiset_match_max_abs_error"
                    ],
                    "critical_allocation_monotonicity_violations": train[
                        "critical_allocation_monotonicity_violations"
                    ],
                    "shuffle_moved_fraction": train["shuffle_moved_fraction"],
                    "projection_clipping_fraction": train[
                        "projection_clipping_fraction"
                    ],
                    "zero_donor_mass_fraction": train["zero_donor_mass_fraction"],
                    "reference_boundary_violation_rate": train[
                        "reference_boundary_violation_rate"
                    ],
                    "reference_margin_before_mean": train[
                        "reference_margin_before_mean"
                    ],
                    "reference_margin_after_mean": train[
                        "reference_margin_after_mean"
                    ],
                    "reference_floor_gap_mean": train["reference_floor_gap_mean"],
                    "reference_trusted_fraction": train["reference_trusted_fraction"],
                    "reference_trusted_many": train["reference_trusted_many"],
                    "reference_trusted_medium": train["reference_trusted_medium"],
                    "reference_trusted_few": train["reference_trusted_few"],
                    "recipient_stream_checksum": train["recipient_stream_checksum"],
                    "donor_stream_checksum": train["donor_stream_checksum"],
                    "target_stream_checksum": train["target_stream_checksum"],
                    "donor_label_stream_checksum": train["donor_label_stream_checksum"],
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
    backbone_state, classifier_state, checkpoint_metadata = load_stage1_checkpoint(
        checkpoint_path,
        device,
    )
    reference_path = (
        Path(args.geometry_reference_head).resolve()
        if args.geometry_reference_head is not None
        else None
    )
    reference_state = load_geometry_reference(reference_path, classifier_state)
    stage1_has_pif_parameters = any(
        key.startswith("conv1d.") for key in backbone_state
    )
    stage1_pif_active = bool(
        checkpoint_metadata.get("pif_active", stage1_has_pif_parameters)
    )
    experiment = vars(args).copy()
    experiment.update(
        {
            "checkpoint": str(checkpoint_path),
            "output_dir": str(output_dir),
            "geometry_head": (
                "frozen_stage1_classifier"
                if reference_path is None
                else "frozen_external_balanced_classifier"
            ),
            "geometry_reference_head": (
                None if reference_path is None else str(reference_path)
            ),
            "pi_ratio_head": "original_frozen_stage1_classifier_weight",
            "trainable_classifier_init": "original_stage1_classifier",
            "ratio_definition": (
                "T4: lambda_0=kappa*lambda_PI; D: lambda=min(lambda_0, "
                "relative-worst-margin critical ratio); E: shuffle D lambda "
                "within Many/Medium/Few x lambda_0-bin without reprojection"
            ),
            "donor_mass_intervention": (
                "lambda_0=donor_mass_scale*lambda_PI; boundary-clip methods "
                "apply only pair-specific downward projection"
            ),
            "stage1_has_pif_parameters": stage1_has_pif_parameters,
            "stage1_pif_active": stage1_pif_active,
            "checkpoint_epoch": json_scalar(checkpoint_metadata.get("epoch")),
            "checkpoint_best_acc": json_scalar(
                checkpoint_metadata.get("best_acc", checkpoint_metadata.get("best_acc1"))
            ),
            "protocol_note": (
                "No donor selection, no fixed lambda, no risk-weighted loss, "
                "and CE on only the H2TF fused feature. An optional balanced "
                "head affects frozen reference geometry only; it never changes "
                "the trainable classifier initialization or its loss."
            ),
        }
    )
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
        f"test={len(dataset.eval.dataset)}, stage1_pif={stage1_pif_active}, "
        f"geometry_head={'stage1' if reference_path is None else 'external_balanced'}, "
        f"ratio_bins={args.ratio_bins}, methods={args.methods}"
    )
    results: list[dict[str, Any]] = []
    for method in args.methods:
        # Reset all global streams.  A/B/C thus receive bitwise-identical
        # recipient/donor transforms and sampling; C's local shuffle generator
        # is the only extra random source and does not advance those streams.
        seed_everything(args.seed)
        results.append(
            run_method(
                method,
                backbone_state,
                classifier_state,
                reference_state,
                dataset.train_instance,
                dataset.eval,
                device,
                output_dir,
                args,
                stage1_pif_active,
            )
        )
        write_summary(output_dir, results)
    print("\nFinal ratio-only A/B/C leaderboard (last epoch)")
    for result in sorted(results, key=lambda item: item["last"]["eval"]["overall"], reverse=True):
        metric = result["last"]["eval"]
        print(
            f"{result['method']:42s} overall={metric['overall']:.4f} "
            f"many={metric['many']:.4f} medium={metric['medium']:.4f} "
            f"few={metric['few']:.4f}"
        )
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
