"""Stage-II test of donor-conditioned boundary geometry as donor selection.

This script deliberately keeps the fusion amount and training loss fixed.  For
each balanced recipient it draws K long-tailed donor candidates, scores every
recipient--donor pair with a frozen Stage-I backbone plus balanced cRT reference
head, then chooses a donor according to one of the paired policies:

* random donor;
* best-of-K donor by the donor-conditioned boundary crossing ratio;
* shuffled boundary-score ranking (negative control);
* official H2T channel replacement (contextual reference).

The candidate-scoring backbone is a separate frozen/eval copy.  Consequently
unselected candidates never update Stage-II BatchNorm statistics.  The train
backbone sees only the recipient and the donor actually selected for fusion.
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
from train_boundary_h2t_stage2 import (
    build_models,
    dataset_config,
    evaluate,
    load_reference_head,
    load_stage1_checkpoint,
    pooled,
    reference_geometry,
    restore_rng_state,
    rng_state,
    seed_everything,
)


METHODS = (
    "random_donor_fixed025",
    "best_of_k_boundary_donor_fixed025",
    "hardest_safe_boundary_donor_fixed025",
    "shuffled_hardest_safe_boundary_score_fixed025",
    "relative_hardest_safe_boundary_donor_fixed025",
    "shuffled_relative_hardest_safe_boundary_score_fixed025",
    "shuffled_boundary_ranking_fixed025",
    "official_h2t_r025",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full Stage-II donor-selection experiment for boundary-aware H2T"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--reference-head",
        required=True,
        help="Frozen balanced cRT head used only for donor-pair geometry.",
    )
    parser.add_argument("--data-path", default="./data/cifar100")
    parser.add_argument("--output-dir", default="./outputs/donor_selection")
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
        "--num-donor-candidates",
        type=int,
        default=4,
        help="K independently drawn long-tailed candidates per recipient.",
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=0.25,
        help="Fixed scalar donor mass for all scalar-fusion selection methods.",
    )
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument(
        "--safe-margin-tolerance",
        type=float,
        default=0.0,
        help=(
            "Permit signed boundary margins down to -tolerance when selecting "
            "a hardest-safe donor; zero requires no frozen-reference crossing."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-shift-bn",
        action="store_true",
        help="Keep the train-backbone BatchNorm buffers fixed during Stage-II.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Smoke-test limit; zero uses each complete epoch.",
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
        help="Resume a method checkpoint, or skip a method with metrics.json.",
    )
    args = parser.parse_args()
    args.methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; choices are {METHODS}")
    if args.num_donor_candidates < 1:
        raise ValueError("--num-donor-candidates must be at least one")
    if not 0.0 <= args.lambda_value <= 1.0:
        raise ValueError("--lambda-value must lie inside [0, 1]")
    if args.safe_margin_tolerance < 0.0:
        raise ValueError("--safe-margin-tolerance must be non-negative")
    return args


def make_local_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def select_candidate(
    method: str,
    critical: torch.Tensor,
    signed_margin: torch.Tensor,
    recipient_signed_margin: torch.Tensor,
    generator: torch.Generator,
    safe_margin_tolerance: float,
) -> torch.Tensor:
    """Choose one column from the B x K pair-geometry matrix.

    The local generator never advances the global transform/sampler RNG.  Thus
    every method consumes exactly the same recipient/candidate image stream.
    """
    batch_size, candidate_count = critical.shape
    if method in {"random_donor_fixed025", "official_h2t_r025"}:
        return torch.randint(
            candidate_count,
            (batch_size,),
            device=critical.device,
            generator=generator,
        )
    if method == "best_of_k_boundary_donor_fixed025":
        return critical.argmax(dim=1)
    if method == "hardest_safe_boundary_donor_fixed025":
        return hardest_safe_candidate(
            signed_margin,
            absolute_safety_threshold(signed_margin, safe_margin_tolerance),
        )
    if method == "shuffled_hardest_safe_boundary_score_fixed025":
        shuffle = torch.rand(
            signed_margin.shape,
            device=signed_margin.device,
            generator=generator,
        ).argsort(dim=1)
        shuffled_margin = signed_margin.gather(1, shuffle)
        return hardest_safe_candidate(
            shuffled_margin,
            absolute_safety_threshold(signed_margin, safe_margin_tolerance),
        )
    if method == "relative_hardest_safe_boundary_donor_fixed025":
        return hardest_safe_candidate(
            signed_margin,
            recipient_relative_safety_threshold(
                recipient_signed_margin,
                safe_margin_tolerance,
            ),
        )
    if method == "shuffled_relative_hardest_safe_boundary_score_fixed025":
        shuffle = torch.rand(
            signed_margin.shape,
            device=signed_margin.device,
            generator=generator,
        ).argsort(dim=1)
        shuffled_margin = signed_margin.gather(1, shuffle)
        return hardest_safe_candidate(
            shuffled_margin,
            recipient_relative_safety_threshold(
                recipient_signed_margin,
                safe_margin_tolerance,
            ),
        )
    if method == "shuffled_boundary_ranking_fixed025":
        # Reassign the K real boundary scores to donor slots by a random
        # permutation, then choose the donor with the highest assigned score.
        # This keeps the score multiset but destroys score--donor alignment.
        shuffle = torch.rand(
            critical.shape,
            device=critical.device,
            generator=generator,
        ).argsort(dim=1)
        shuffled_scores = critical.gather(1, shuffle)
        return shuffled_scores.argmax(dim=1)
    raise ValueError(f"unknown donor-selection method: {method}")


def absolute_safety_threshold(
    signed_margin: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    return torch.full_like(signed_margin[:, 0], -tolerance)


def recipient_relative_safety_threshold(
    recipient_signed_margin: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Do not make an already-misclassified recipient more unsafe."""
    return torch.minimum(
        recipient_signed_margin,
        torch.zeros_like(recipient_signed_margin),
    ) - tolerance


def safety_threshold_for_method(
    method: str,
    signed_margin: torch.Tensor,
    recipient_signed_margin: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    if method in {
        "relative_hardest_safe_boundary_donor_fixed025",
        "shuffled_relative_hardest_safe_boundary_score_fixed025",
    }:
        return recipient_relative_safety_threshold(recipient_signed_margin, tolerance)
    return absolute_safety_threshold(signed_margin, tolerance)


def hardest_safe_candidate(
    signed_margin: torch.Tensor,
    threshold: torch.Tensor,
) -> torch.Tensor:
    """Pick the closest allowed candidate, else the least unsafe candidate."""
    valid = signed_margin >= threshold[:, None]
    closest_valid = signed_margin.masked_fill(~valid, torch.inf).argmin(dim=1)
    least_unsafe = signed_margin.argmax(dim=1)
    return torch.where(valid.any(dim=1), closest_valid, least_unsafe)


def official_h2t_with_generator(
    recipient_map: torch.Tensor,
    donor_map: torch.Tensor,
    generator: torch.Generator,
    rho: float = 0.25,
) -> torch.Tensor:
    channels = recipient_map.shape[1]
    selected_channels = torch.randperm(
        channels,
        device=recipient_map.device,
        generator=generator,
    )[: int(rho * channels)]
    fused = recipient_map.clone()
    fused[:, selected_channels] = donor_map[:, selected_channels]
    return fused


@torch.no_grad()
def score_candidate_pairs(
    geometry_backbone: torch.nn.Module,
    recipient_images: torch.Tensor,
    candidate_images: torch.Tensor,
    target: torch.Tensor,
    reference_weight: torch.Tensor,
    reference_bias: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return B x K crossing ratios and frozen-reference feature vectors."""
    batch_size, candidate_count = candidate_images.shape[:2]
    recipient_vector = pooled(geometry_backbone(recipient_images))
    flat_candidates = candidate_images.flatten(0, 1)
    donor_vectors = pooled(geometry_backbone(flat_candidates)).view(
        batch_size,
        candidate_count,
        -1,
    )
    repeated_recipient = recipient_vector[:, None, :].expand(
        -1,
        candidate_count,
        -1,
    ).reshape(batch_size * candidate_count, -1)
    repeated_target = target[:, None].expand(-1, candidate_count).reshape(-1)
    _, _, critical = reference_geometry(
        repeated_recipient,
        donor_vectors.reshape(batch_size * candidate_count, -1),
        repeated_target,
        reference_weight,
        reference_bias,
        delta,
    )
    return critical.view(batch_size, candidate_count), recipient_vector, donor_vectors


def signed_boundary_margin(
    feature: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Closest signed distance to a target-vs-competitor reference boundary.

    Positive means every competitor boundary is still on the recipient side;
    zero is the nearest boundary; negative means at least one has been crossed.
    """
    logits = F.linear(feature, weight, bias)
    numerator = logits.gather(1, target[:, None]) - logits
    weight_sq = (weight * weight).sum(dim=1)
    denominator = torch.sqrt(
        torch.clamp(
            weight_sq[:, None] + weight_sq[None, :] - 2.0 * (weight @ weight.t()),
            min=0.0,
        )
    )[target]
    distance = numerator / torch.clamp(denominator, min=eps)
    row = torch.arange(target.numel(), device=target.device)
    distance[row, target] = torch.inf
    return distance.min(dim=1).values


def candidate_rank(critical: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """One is the best (largest safe crossing ratio); ties receive best rank."""
    selected_value = critical.gather(1, selected[:, None])
    return 1 + (critical > selected_value).sum(dim=1)


def train_epoch(
    loader: torch.utils.data.DataLoader,
    train_backbone: torch.nn.Module,
    geometry_backbone: torch.nn.Module,
    classifier: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    method: str,
    reference_weight: torch.Tensor,
    reference_bias: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
    args: argparse.Namespace,
) -> dict[str, float]:
    if args.no_shift_bn:
        train_backbone.eval()
    else:
        train_backbone.train()
    geometry_backbone.eval()
    classifier.train()

    total_loss = 0.0
    total_correct = 0
    total = 0
    selected_critical_sum = 0.0
    candidate_max_critical_sum = 0.0
    selected_rank_sum = 0.0
    selected_safe_count = 0
    selected_signed_margin_sum = 0.0
    selected_abs_signed_margin_sum = 0.0
    candidate_has_safe_count = 0
    safety_threshold_sum = 0.0
    selected_same_class_count = 0
    candidate_same_class_count = 0
    candidate_count = 0
    reference_fused_correct = 0
    selected_index_count: torch.Tensor | None = None
    recipient_checksum = 0.0
    candidate_checksum = 0.0
    target_checksum = 0
    candidate_label_checksum = 0

    for batch_index, (images, target, meta) in enumerate(loader):
        if args.max_train_batches and batch_index >= args.max_train_batches:
            break
        if "sample_images" not in meta or "sample_labels" not in meta:
            raise RuntimeError(
                "multi-candidate donor metadata is missing; use num_donor_candidates > 1"
            )
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        candidates = meta["sample_images"].to(device, non_blocking=True)
        candidate_labels = meta["sample_labels"].to(device, non_blocking=True)
        batch_size, k = candidates.shape[:2]

        critical, geometry_recipient, geometry_donors = score_candidate_pairs(
            geometry_backbone,
            images,
            candidates,
            target,
            reference_weight,
            reference_bias,
            args.delta,
        )
        candidate_fused_geometry = (
            (1.0 - args.lambda_value) * geometry_recipient[:, None, :]
            + args.lambda_value * geometry_donors
        )
        repeated_target = target[:, None].expand(-1, k).reshape(-1)
        candidate_signed_margin = signed_boundary_margin(
            candidate_fused_geometry.reshape(batch_size * k, -1),
            repeated_target,
            reference_weight,
            reference_bias,
        ).view(batch_size, k)
        recipient_signed_margin = signed_boundary_margin(
            geometry_recipient,
            target,
            reference_weight,
            reference_bias,
        )
        safety_threshold = safety_threshold_for_method(
            method,
            candidate_signed_margin,
            recipient_signed_margin,
            args.safe_margin_tolerance,
        )
        selected = select_candidate(
            method,
            critical,
            candidate_signed_margin,
            recipient_signed_margin,
            generator,
            args.safe_margin_tolerance,
        )
        row = torch.arange(batch_size, device=device)
        chosen_images = candidates[row, selected]
        chosen_labels = candidate_labels[row, selected]
        chosen_critical = critical[row, selected]
        chosen_signed_margin = candidate_signed_margin[row, selected]
        chosen_geometry_donor = geometry_donors[row, selected]

        with torch.no_grad():
            recipient_map = train_backbone(images)
            donor_map = train_backbone(chosen_images)
            if method == "official_h2t_r025":
                fused_map = official_h2t_with_generator(
                    recipient_map,
                    donor_map,
                    generator,
                    rho=0.25,
                )
            else:
                fused_map = (
                    (1.0 - args.lambda_value) * recipient_map
                    + args.lambda_value * donor_map
                )
            geometry_fused = (
                (1.0 - args.lambda_value) * geometry_recipient
                + args.lambda_value * chosen_geometry_donor
            )
            reference_prediction = F.linear(
                geometry_fused,
                reference_weight,
                reference_bias,
            ).argmax(dim=1)

        output = classifier(fused_map)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach()) * batch_size
        total_correct += int((output.argmax(dim=1) == target).sum())
        reference_fused_correct += int((reference_prediction == target).sum())
        selected_critical_sum += float(chosen_critical.sum())
        candidate_max_critical_sum += float(critical.max(dim=1).values.sum())
        selected_rank_sum += float(candidate_rank(critical, selected).sum())
        selected_safe_count += int(
            (chosen_signed_margin >= safety_threshold).sum()
        )
        selected_signed_margin_sum += float(chosen_signed_margin.sum())
        selected_abs_signed_margin_sum += float(chosen_signed_margin.abs().sum())
        candidate_has_safe_count += int(
            (candidate_signed_margin >= safety_threshold[:, None]).any(dim=1).sum()
        )
        safety_threshold_sum += float(safety_threshold.sum())
        selected_same_class_count += int((chosen_labels == target).sum())
        candidate_same_class_count += int(
            (candidate_labels == target[:, None]).sum()
        )
        candidate_count += batch_size * k
        if selected_index_count is None:
            selected_index_count = torch.zeros(k, dtype=torch.long, device=device)
        selected_index_count.scatter_add_(
            0,
            selected,
            torch.ones_like(selected, dtype=torch.long),
        )
        # These stream checksums audit that every policy was offered the same
        # recipient and K-candidate batches.  They intentionally exclude the
        # selected donor, which is the method's causal intervention.
        recipient_checksum += float(images.detach().sum())
        candidate_checksum += float(candidates.detach().sum())
        target_checksum += int(target.detach().sum())
        candidate_label_checksum += int(candidate_labels.detach().sum())
        total += batch_size

    if selected_index_count is None:
        selected_index_count = torch.zeros(
            args.num_donor_candidates,
            dtype=torch.long,
            device=device,
        )
    selection_histogram = (
        selected_index_count.detach().cpu().to(torch.float64) / max(total, 1)
    ).tolist()
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "lambda_mean": args.lambda_value,
        "selected_critical_mean": selected_critical_sum / max(total, 1),
        "candidate_max_critical_mean": candidate_max_critical_sum / max(total, 1),
        "selected_candidate_rank_mean": selected_rank_sum / max(total, 1),
        "selected_safe_fraction": selected_safe_count / max(total, 1),
        "selected_signed_margin_mean": selected_signed_margin_sum / max(total, 1),
        "selected_abs_signed_margin_mean": selected_abs_signed_margin_sum
        / max(total, 1),
        "candidate_has_safe_fraction": candidate_has_safe_count / max(total, 1),
        "safety_threshold_mean": safety_threshold_sum / max(total, 1),
        "selected_same_class_fraction": selected_same_class_count / max(total, 1),
        "candidate_same_class_fraction": candidate_same_class_count
        / max(candidate_count, 1),
        "reference_fused_accuracy": reference_fused_correct / max(total, 1),
        "reference_flip_rate": 1.0 - reference_fused_correct / max(total, 1),
        "selection_histogram": [float(value) for value in selection_histogram],
        "recipient_stream_checksum": recipient_checksum,
        "candidate_stream_checksum": candidate_checksum,
        "target_stream_checksum": target_checksum,
        "candidate_label_stream_checksum": candidate_label_checksum,
    }


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
    train_backbone, classifier = build_models(
        backbone_state,
        classifier_state,
        device,
    )
    geometry_backbone, _ = build_models(
        backbone_state,
        classifier_state,
        device,
    )
    geometry_backbone.eval()
    reference_weight = reference_state["fc.weight"].to(device).detach().clone()
    reference_bias = reference_state["fc.bias"].to(device).detach().clone()
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.lr * args.lr_factor,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # Separate generator: selection randomness never perturbs dataset transforms
    # or DataLoader sampling, which use the global seeded streams.
    selection_generator = make_local_generator(device, args.seed + 104729)
    start_epoch = 0
    best_overall = -1.0
    history: list[dict[str, Any]] = []
    initial: dict[str, float] | None = None

    if args.resume and current_path.exists():
        saved = torch.load(current_path, map_location=device, weights_only=False)
        if saved.get("method") != method:
            raise ValueError("checkpoint method does not match requested method")
        train_backbone.load_state_dict(saved["backbone"])
        classifier.load_state_dict(saved["classifier"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"])
        best_overall = float(saved["best_overall"])
        history = saved["history"]
        initial = saved.get("initial")
        restore_rng_state(saved["rng"])
        selection_generator.set_state(saved["selection_generator_state"])
        print(f"{method}: resumed at epoch {start_epoch}")

    if initial is None:
        initial = evaluate(
            eval_loader,
            train_backbone,
            classifier,
            device,
            args.max_eval_batches,
        )
    print(
        f"{method}: initial overall={initial['overall']:.4f}, "
        f"many={initial['many']:.4f}, medium={initial['medium']:.4f}, "
        f"few={initial['few']:.4f}"
    )

    for epoch in range(start_epoch, args.epochs):
        cosine = 0.5 * (1.0 + math.cos(epoch / args.epochs * math.pi))
        learning_rate = args.lr_factor * args.lr * cosine
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train_metrics = train_epoch(
            train_loader,
            train_backbone,
            geometry_backbone,
            classifier,
            optimizer,
            method,
            reference_weight,
            reference_bias,
            device,
            selection_generator,
            args,
        )
        evaluation = evaluate(
            eval_loader,
            train_backbone,
            classifier,
            device,
            args.max_eval_batches,
        )
        row = {
            "epoch": epoch + 1,
            "lr": learning_rate,
            "train": train_metrics,
            "eval": evaluation,
        }
        history.append(row)
        is_best = evaluation["overall"] > best_overall
        best_overall = max(best_overall, evaluation["overall"])
        state = {
            "epoch": epoch + 1,
            "method": method,
            "backbone": train_backbone.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_overall": best_overall,
            "initial": initial,
            "history": history,
            "rng": rng_state(),
            "selection_generator_state": selection_generator.get_state(),
            "num_donor_candidates": args.num_donor_candidates,
            "lambda_value": args.lambda_value,
            "reference_head": str(Path(args.reference_head).resolve()),
            "safe_margin_tolerance": args.safe_margin_tolerance,
        }
        torch.save(state, current_path)
        if is_best:
            torch.save(state, best_path)
        print(
            f"  epoch {epoch + 1:02d}/{args.epochs}: "
            f"train={train_metrics['accuracy']:.4f}, "
            f"test={evaluation['overall']:.4f}, "
            f"M/Md/F={evaluation['many']:.4f}/{evaluation['medium']:.4f}/"
            f"{evaluation['few']:.4f}, crit={train_metrics['selected_critical_mean']:.4f}, "
            f"margin={train_metrics['selected_signed_margin_mean']:.4f}, "
            f"safe={train_metrics['selected_safe_fraction']:.4f}, "
            f"threshold={train_metrics['safety_threshold_mean']:.4f}, "
            f"ref_flip={train_metrics['reference_flip_rate']:.4f}"
        )

    best_row = max(history, key=lambda item: item["eval"]["overall"])
    result = {
        "method": method,
        "initial": initial,
        "last": history[-1],
        "best": best_row,
        "history": history,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        for selection in ("last", "best"):
            epoch_row = result[selection]
            train = epoch_row["train"]
            evaluation = epoch_row["eval"]
            rows.append(
                {
                    "method": result["method"],
                    "selection": selection,
                    "epoch": epoch_row["epoch"],
                    "overall": evaluation["overall"],
                    "many": evaluation["many"],
                    "medium": evaluation["medium"],
                    "few": evaluation["few"],
                    "train_accuracy": train["accuracy"],
                    "lambda_mean": train["lambda_mean"],
                    "selected_critical_mean": train["selected_critical_mean"],
                    "candidate_max_critical_mean": train[
                        "candidate_max_critical_mean"
                    ],
                    "selected_candidate_rank_mean": train[
                        "selected_candidate_rank_mean"
                    ],
                    "selected_safe_fraction": train["selected_safe_fraction"],
                    "selected_signed_margin_mean": train[
                        "selected_signed_margin_mean"
                    ],
                    "selected_abs_signed_margin_mean": train[
                        "selected_abs_signed_margin_mean"
                    ],
                    "candidate_has_safe_fraction": train[
                        "candidate_has_safe_fraction"
                    ],
                    "safety_threshold_mean": train["safety_threshold_mean"],
                    "selected_same_class_fraction": train[
                        "selected_same_class_fraction"
                    ],
                    "candidate_same_class_fraction": train[
                        "candidate_same_class_fraction"
                    ],
                    "reference_fused_accuracy": train["reference_fused_accuracy"],
                    "reference_flip_rate": train["reference_flip_rate"],
                    "recipient_stream_checksum": train["recipient_stream_checksum"],
                    "candidate_stream_checksum": train["candidate_stream_checksum"],
                    "target_stream_checksum": train["target_stream_checksum"],
                    "candidate_label_stream_checksum": train[
                        "candidate_label_stream_checksum"
                    ],
                    "selection_histogram": json.dumps(train["selection_histogram"]),
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
    reference_path = Path(args.reference_head).resolve()
    backbone_state, classifier_state, checkpoint_metadata = load_stage1_checkpoint(
        checkpoint_path,
        device,
    )
    reference_state = load_reference_head(reference_path, classifier_state)

    experiment = vars(args).copy()
    experiment.update(
        {
            "checkpoint": str(checkpoint_path),
            "reference_head": str(reference_path),
            "output_dir": str(output_dir),
            "device_resolved": str(device),
            "checkpoint_epoch": checkpoint_metadata.get("epoch"),
            "checkpoint_best_acc": checkpoint_metadata.get("best_acc"),
            "candidate_policy_note": (
                "All policies consume the same K candidate donor stream. "
                "Only selected donor identity changes."
            ),
        }
    )
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2),
        encoding="utf-8",
    )

    dataset = CIFAR100_LT(
        dataset_config(args.num_donor_candidates),
        distributed=False,
        root=args.data_path,
        imb_factor=0.01,
        batch_size=args.batch_size,
        num_works=args.workers,
    )
    print(
        f"device={device}, train={len(dataset.train_instance.dataset)}, "
        f"test={len(dataset.eval.dataset)}, K={args.num_donor_candidates}, "
        f"lambda={args.lambda_value:.4f}, methods={args.methods}"
    )

    results = []
    for method in args.methods:
        # Reset global streams.  The local selection RNG is recreated per
        # method and never consumes from them, so the candidate stream is paired.
        seed_everything(args.seed)
        result = run_method(
            method,
            backbone_state,
            classifier_state,
            reference_state,
            dataset.train_instance,
            dataset.eval,
            device,
            output_dir,
            args,
        )
        results.append(result)
        write_summary(output_dir, results)

    print("\nFinal donor-selection leaderboard (last epoch)")
    for result in sorted(
        results,
        key=lambda item: item["last"]["eval"]["overall"],
        reverse=True,
    ):
        evaluation = result["last"]["eval"]
        print(
            f"{result['method']:42s} overall={evaluation['overall']:.4f} "
            f"many={evaluation['many']:.4f} medium={evaluation['medium']:.4f} "
            f"few={evaluation['few']:.4f}"
        )
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
