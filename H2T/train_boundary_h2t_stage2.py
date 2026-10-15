"""Full image-based Stage-II validation for boundary-aware H2T fusion.

The backbone and reference classifier are loaded from a Stage-I checkpoint.
As in the official H2T Stage-II recipe, only a fresh classifier is optimized;
the frozen backbone is still put in train mode so that BatchNorm statistics can
shift on the Stage-II image stream.

This script accepts the Stage-I checkpoints produced by the local
``hafe_experiment`` project (a single ResNet-32 state dict containing ``fc``).
It uses the official H2T CIFAR100-LT dataset/sampling implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from datasets.cifar100 import CIFAR100_LT
from models import resnet_cifar


METHODS = (
    "nofusion",
    "official_h2t_r025",
    "fixed_matched",
    "pi_h2t_center_matched",
    "recipient_boundary_matched",
    "boundary_direct_g110",
    "boundary_group_b025",
    "boundary_group_b050",
    "boundary_group_b075",
    "boundary_group_b100",
    "boundary_group_b050_shuffled",
    "boundary_reliability_blend",
)

SUPERVISIONS = (
    "fused_only",
    "clean_anchor",
    "risk_weighted_anchor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full image-based Stage-II boundary-aware H2T experiment"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--reference-head",
        default="",
        help=(
            "Optional frozen geometry-head checkpoint. The trainable output "
            "head is still initialized from --checkpoint."
        ),
    )
    parser.add_argument("--data-path", default="./data/cifar100")
    parser.add_argument("--output-dir", default="./outputs/full_stage2_boundary")
    parser.add_argument(
        "--methods",
        default=(
            "nofusion,official_h2t_r025,fixed_matched,"
            "pi_h2t_center_matched,recipient_boundary_matched,"
            "boundary_group_b050"
        ),
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
        "--target-mean",
        type=float,
        default=0.49647045,
        help="Matched mean donor mass selected from the train-feature pilot.",
    )
    parser.add_argument("--gamma", type=float, default=1.1)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--lambda-cap", type=float, default=1.0)
    parser.add_argument(
        "--supervision",
        choices=SUPERVISIONS,
        default="fused_only",
        help=(
            "Stage-II label supervision: fused CE only; clean-recipient CE plus "
            "fused CE; or clean CE plus boundary-risk-weighted fused CE."
        ),
    )
    parser.add_argument(
        "--anchor-alpha",
        type=float,
        default=1.0,
        help="Multiplier on the fused term for clean/risk-weighted anchors.",
    )
    parser.add_argument(
        "--risk-weight-min",
        type=float,
        default=0.1,
        help=(
            "Minimum fused-loss weight for risk_weighted_anchor. The raw "
            "weight is (lambda_critical - lambda)/(lambda_critical + eps)."
        ),
    )
    parser.add_argument(
        "--normalize-supervision-loss",
        action="store_true",
        help=(
            "Normalize multi-term anchor objectives to the scale of one CE: "
            "divide clean_anchor by 1+alpha and risk_weighted_anchor by "
            "1+alpha*mean(risk_weight)."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-shift-bn",
        action="store_true",
        help="Keep frozen backbone BatchNorm statistics fixed.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Smoke-test limit; zero uses the complete epoch.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Smoke-test limit; zero evaluates the full test set.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume method-level current checkpoints and skip completed methods.",
    )
    args = parser.parse_args()

    args.methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; choices are {METHODS}")
    if not 0.0 <= args.target_mean <= args.lambda_cap:
        raise ValueError("--target-mean must lie inside [0, lambda-cap]")
    if not 0.0 < args.lambda_cap <= 1.0:
        raise ValueError("--lambda-cap must lie inside (0, 1]")
    if args.anchor_alpha < 0.0:
        raise ValueError("--anchor-alpha must be non-negative")
    if not 0.0 <= args.risk_weight_min <= 1.0:
        raise ValueError("--risk-weight-min must lie inside [0, 1]")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dataset_config(num_donor_candidates: int = 1) -> SimpleNamespace:
    if num_donor_candidates < 1:
        raise ValueError("num_donor_candidates must be at least one")
    return SimpleNamespace(
        sampler=SimpleNamespace(
            type="weighted sampler",
            weighted_sampler=SimpleNamespace(type="balance"),
            dual_sample=SimpleNamespace(
                enable=True,
                type="long-tailed",
                num_donor_candidates=num_donor_candidates,
            ),
        )
    )


def strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def load_stage1_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError("Stage-I checkpoint must be a dictionary")

    if "model" in raw:
        state = strip_module_prefix(raw["model"])
    elif "state_dict" in raw:
        state = strip_module_prefix(raw["state_dict"])
    elif "state_dict_model" in raw and "state_dict_classifier" in raw:
        backbone_state = strip_module_prefix(raw["state_dict_model"])
        classifier_state = strip_module_prefix(raw["state_dict_classifier"])
        return backbone_state, classifier_state, raw
    else:
        raise KeyError("unrecognized Stage-I checkpoint layout")

    if "fc.weight" not in state or "fc.bias" not in state:
        raise KeyError("checkpoint model state has no fc.weight/fc.bias")
    backbone_state = {
        key: value for key, value in state.items() if not key.startswith("fc.")
    }
    classifier_state = {
        "fc.weight": state["fc.weight"],
        "fc.bias": state["fc.bias"],
    }
    return backbone_state, classifier_state, raw


def load_reference_head(
    reference_path: Path | None,
    fallback_classifier_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if reference_path is None:
        return {
            "fc.weight": fallback_classifier_state["fc.weight"].detach().cpu().clone(),
            "fc.bias": fallback_classifier_state["fc.bias"].detach().cpu().clone(),
        }

    raw = torch.load(reference_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError("reference-head checkpoint must be a dictionary")
    if "reference_weight" in raw and "reference_bias" in raw:
        return {
            "fc.weight": raw["reference_weight"].detach().cpu().clone(),
            "fc.bias": raw["reference_bias"].detach().cpu().clone(),
        }
    if "state_dict" in raw:
        state = strip_module_prefix(raw["state_dict"])
        if "fc.weight" in state and "fc.bias" in state:
            return {
                "fc.weight": state["fc.weight"].detach().cpu().clone(),
                "fc.bias": state["fc.bias"].detach().cpu().clone(),
            }
    raise KeyError(
        "reference checkpoint needs reference_weight/reference_bias or "
        "state_dict[fc.weight/fc.bias]"
    )


class ResNetFeatureOptionB(resnet_cifar.ResNet_fe):
    """Official feature extractor with learnable projection shortcuts.

    The local 200-epoch Stage-I checkpoints use the common CIFAR ResNet
    Option-B shortcut, while the released H2T model defaults to Option A.
    All other block/layer names and tensors are compatible.
    """

    def _make_layer(
        self,
        block: type[torch.nn.Module],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> torch.nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(
                block(self.in_planes, planes, current_stride, option="B")
            )
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)


def build_models(
    backbone_state: dict[str, torch.Tensor],
    classifier_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    uses_projection_shortcut = any(
        ".shortcut.0.weight" in key for key in backbone_state
    )
    if uses_projection_shortcut:
        backbone = ResNetFeatureOptionB(
            resnet_cifar.BasicBlock,
            [5, 5, 5],
        ).to(device)
    else:
        backbone = resnet_cifar.resnet32_fe().to(device)
    classifier = resnet_cifar.Classifier(feat_in=64, num_classes=100).to(device)
    backbone.load_state_dict(backbone_state, strict=True)
    classifier.load_state_dict(classifier_state, strict=True)
    backbone.requires_grad_(False)
    return backbone, classifier


def pooled(feature_map: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)


def match_ratio_mean(
    base: torch.Tensor,
    target: float,
    cap: float,
) -> torch.Tensor:
    if base.numel() == 0:
        return base
    if target <= 0.0:
        return torch.zeros_like(base)
    if not torch.any(base > 0.0):
        return torch.full_like(base, min(target, cap))

    low = 0.0
    high = max(1.0, target / max(float(base.mean()), 1e-12))
    while float(torch.clamp(high * base, max=cap).mean()) < target:
        high *= 2.0
    for _ in range(50):
        middle = 0.5 * (low + high)
        if float(torch.clamp(middle * base, max=cap).mean()) < target:
            low = middle
        else:
            high = middle
    return torch.clamp(high * base, min=0.0, max=cap)


def frequency_groups(target: torch.Tensor) -> torch.Tensor:
    # Official H2T CIFAR100-LT splits: [0, 36), [36, 71), [71, 100).
    return torch.where(target < 36, 0, torch.where(target < 71, 1, 2))


def match_within_frequency_groups(
    base: torch.Tensor,
    target_labels: torch.Tensor,
    target_mean: float,
    cap: float,
) -> torch.Tensor:
    result = torch.empty_like(base)
    groups = frequency_groups(target_labels)
    for group_id in range(3):
        mask = groups == group_id
        if torch.any(mask):
            result[mask] = match_ratio_mean(base[mask], target_mean, cap)
    return result


def reference_geometry(
    recipient: torch.Tensor,
    donor: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    delta: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return PI center ratio, recipient margin, and donor-conditioned crossing."""
    selected_weight = weight[target]
    cosine = F.cosine_similarity(recipient, selected_weight, dim=1, eps=eps)
    center_ratio = torch.clamp((1.0 + cosine) / 2.0, min=0.0, max=1.0)

    recipient_logits = F.linear(recipient, weight, bias)
    donor_logits = F.linear(donor, weight, bias)
    recipient_numerator = recipient_logits.gather(1, target[:, None]) - recipient_logits
    donor_numerator = donor_logits.gather(1, target[:, None]) - donor_logits

    weight_sq = (weight * weight).sum(dim=1)
    denominator = torch.sqrt(
        torch.clamp(
            weight_sq[:, None] + weight_sq[None, :] - 2.0 * (weight @ weight.t()),
            min=0.0,
        )
    )[target]
    denominator = torch.clamp(denominator, min=eps)
    recipient_distance = recipient_numerator / denominator
    donor_distance = donor_numerator / denominator
    row = torch.arange(target.numel(), device=target.device)
    recipient_distance[row, target] = torch.inf
    donor_distance[row, target] = torch.inf

    recipient_margin = recipient_distance.min(dim=1).values
    threshold = torch.minimum(recipient_margin, torch.zeros_like(recipient_margin)) - delta
    decrease = recipient_distance - donor_distance
    dangerous = torch.isfinite(decrease) & (decrease > eps)
    bounds = torch.full_like(decrease, torch.inf)
    raw = (recipient_distance - threshold[:, None]) / torch.clamp(decrease, min=eps)
    bounds[dangerous] = raw[dangerous]
    critical = bounds.min(dim=1).values
    critical = torch.where(torch.isfinite(critical), critical, torch.ones_like(critical))
    critical = torch.clamp(critical, min=0.0, max=1.0)
    return center_ratio, recipient_margin, critical


def distribution_matched_recipient_ratio(
    recipient_margin: torch.Tensor,
    center_ratio: torch.Tensor,
) -> torch.Tensor:
    """Assign larger PI-scale donor ratios to safer recipient features."""
    sorted_ratio = torch.sort(center_ratio).values
    order = torch.argsort(recipient_margin)
    result = torch.empty_like(center_ratio)
    result[order] = sorted_ratio
    return result


def shuffled_within_group(
    ratio: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    result = ratio.clone()
    groups = frequency_groups(labels)
    for group_id in range(3):
        indices = torch.nonzero(groups == group_id, as_tuple=False).flatten()
        if indices.numel() > 1:
            result[indices] = ratio[indices[torch.randperm(indices.numel(), device=ratio.device)]]
    return result


def scalar_ratio(
    method: str,
    recipient_vector: torch.Tensor,
    donor_vector: torch.Tensor,
    target: torch.Tensor,
    reference_weight: torch.Tensor,
    reference_bias: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    center, margin, critical = reference_geometry(
        recipient_vector,
        donor_vector,
        target,
        reference_weight,
        reference_bias,
        args.delta,
    )
    target_mean = float(args.target_mean)

    if method == "nofusion":
        ratio = torch.zeros_like(critical)
    elif method == "fixed_matched":
        ratio = torch.full_like(critical, target_mean)
    elif method == "pi_h2t_center_matched":
        ratio = match_ratio_mean(center, target_mean, args.lambda_cap)
    elif method == "recipient_boundary_matched":
        recipient_only = distribution_matched_recipient_ratio(margin, center)
        ratio = match_ratio_mean(recipient_only, target_mean, args.lambda_cap)
    elif method == "boundary_direct_g110":
        ratio = torch.clamp(
            args.gamma * critical,
            min=0.0,
            max=args.lambda_cap,
        )
    elif method == "boundary_reliability_blend":
        geometry_ratio = match_within_frequency_groups(
            critical,
            target,
            target_mean,
            args.lambda_cap,
        )
        groups = frequency_groups(target)
        # Boundary estimates from a long-tailed reference head are most reliable
        # for head classes.  Preserve the same mean donor mass in every group,
        # but decay pair-specific geometry from Many to Medium to Few.
        blend = torch.where(
            groups == 0,
            torch.full_like(critical, 0.50),
            torch.where(
                groups == 1,
                torch.full_like(critical, 0.25),
                torch.zeros_like(critical),
            ),
        )
        ratio = blend * geometry_ratio + (1.0 - blend) * target_mean
    elif method.startswith("boundary_group_b"):
        suffix = method.split("boundary_group_b", maxsplit=1)[1]
        shuffled = suffix.endswith("_shuffled")
        if shuffled:
            suffix = suffix[: -len("_shuffled")]
        blend = int(suffix) / 100.0
        geometry_ratio = match_within_frequency_groups(
            critical,
            target,
            target_mean,
            args.lambda_cap,
        )
        ratio = blend * geometry_ratio + (1.0 - blend) * target_mean
        if shuffled:
            ratio = shuffled_within_group(ratio, target)
    else:
        raise ValueError(f"{method} is not a scalar-ratio method")
    return ratio, critical


def official_h2t(
    recipient_map: torch.Tensor,
    donor_map: torch.Tensor,
    rho: float = 0.25,
) -> torch.Tensor:
    channels = recipient_map.shape[1]
    selected = torch.randperm(channels, device=recipient_map.device)[: int(rho * channels)]
    fused = recipient_map.clone()
    fused[:, selected] = donor_map[:, selected]
    return fused


def boundary_risk_weight(
    ratio: torch.Tensor,
    critical: torch.Tensor,
    minimum: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fused-loss confidence and a monotone boundary-risk score.

    ``critical`` is the first donor interpolation ratio predicted to reach the
    closest reference boundary.  The confidence is the fraction of that safe
    path still unused by the actual interpolation.  The uncapped ratio to the
    boundary is retained as a diagnostic risk score (one means boundary reach).
    """
    remaining = (critical - ratio) / (critical + eps)
    weight = torch.clamp(remaining, min=minimum, max=1.0)
    risk_score = torch.nan_to_num(
        ratio / (critical + eps),
        nan=0.0,
        posinf=1e8,
        neginf=0.0,
    )
    return weight, torch.clamp(risk_score, min=0.0, max=1e8)


def summarize_risk_diagnostics(
    risk_score: list[float],
    clean_loss: list[float],
    fused_loss: list[float],
    gradient_cosine: list[float],
) -> dict[str, Any]:
    """Summarize equal-count low/medium/high risk bins for one epoch."""
    if not risk_score:
        return {}
    risk = np.asarray(risk_score, dtype=np.float64)
    clean = np.asarray(clean_loss, dtype=np.float64)
    fused = np.asarray(fused_loss, dtype=np.float64)
    cosine = np.asarray(gradient_cosine, dtype=np.float64)
    order = np.argsort(risk, kind="stable")
    chunks = np.array_split(order, 3)
    result: dict[str, Any] = {}
    for name, indices in zip(("low", "medium", "high"), chunks):
        result[name] = {
            "count": int(indices.size),
            "risk_mean": float(risk[indices].mean()),
            "risk_min": float(risk[indices].min()),
            "risk_max": float(risk[indices].max()),
            "clean_loss": float(clean[indices].mean()),
            "fused_loss": float(fused[indices].mean()),
            "fused_minus_clean": float((fused[indices] - clean[indices]).mean()),
            "gradient_cosine": float(cosine[indices].mean()),
            "gradient_conflict_rate": float((cosine[indices] < 0.0).mean()),
        }
    return result


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
        output = classifier(backbone(images))
        total_loss += float(F.cross_entropy(output, target, reduction="sum"))
        prediction = output.argmax(dim=1)
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
    reference_weight: torch.Tensor,
    reference_bias: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    if args.no_shift_bn:
        backbone.eval()
    else:
        backbone.train()
    classifier.train()

    total_loss = 0.0
    total_correct = 0
    total = 0
    ratio_sum = 0.0
    ratio_square_sum = 0.0
    ratio_group_sum = np.zeros(3, dtype=np.float64)
    ratio_group_count = np.zeros(3, dtype=np.int64)
    reference_fused_correct = 0
    clean_correct = 0
    clean_loss_sum = 0.0
    fused_loss_sum = 0.0
    risk_weight_sum = 0.0
    risk_weight_floor_count = 0
    risk_values: list[float] = []
    clean_loss_values: list[float] = []
    fused_loss_values: list[float] = []
    gradient_cosine_values: list[float] = []

    for batch_index, (images, target, meta) in enumerate(loader):
        if args.max_train_batches and batch_index >= args.max_train_batches:
            break
        donor_images = meta["sample_image"]
        images = images.to(device, non_blocking=True)
        donor_images = donor_images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            recipient_map = backbone(images)
            donor_map = backbone(donor_images)
            recipient_vector = pooled(recipient_map)
            donor_vector = pooled(donor_map)
            if method == "official_h2t_r025":
                fused_map = official_h2t(recipient_map, donor_map, rho=0.25)
                ratio = torch.full(
                    (target.numel(),),
                    0.25,
                    device=device,
                    dtype=recipient_map.dtype,
                )
                critical = torch.zeros_like(ratio)
            else:
                ratio, critical = scalar_ratio(
                    method,
                    recipient_vector,
                    donor_vector,
                    target,
                    reference_weight,
                    reference_bias,
                    args,
                )
                fused_map = (
                    (1.0 - ratio[:, None, None, None]) * recipient_map
                    + ratio[:, None, None, None] * donor_map
                )
            fused_vector = pooled(fused_map)
            reference_prediction = F.linear(
                fused_vector,
                reference_weight,
                reference_bias,
            ).argmax(dim=1)

        clean_output = classifier(recipient_map)
        output = classifier(fused_map)
        clean_per_sample = F.cross_entropy(
            clean_output,
            target,
            reduction="none",
        )
        fused_per_sample = F.cross_entropy(
            output,
            target,
            reduction="none",
        )
        risk_weight, risk_score = boundary_risk_weight(
            ratio,
            critical,
            args.risk_weight_min,
        )
        if args.supervision == "fused_only":
            loss = fused_per_sample.mean()
        elif args.supervision == "clean_anchor":
            loss = clean_per_sample.mean() + args.anchor_alpha * fused_per_sample.mean()
            if args.normalize_supervision_loss:
                loss = loss / (1.0 + args.anchor_alpha)
        elif args.supervision == "risk_weighted_anchor":
            loss = clean_per_sample.mean() + args.anchor_alpha * (
                risk_weight * fused_per_sample
            ).mean()
            if args.normalize_supervision_loss:
                loss = loss / (
                    1.0 + args.anchor_alpha * risk_weight.detach().mean()
                )
        else:
            raise ValueError(f"unknown supervision: {args.supervision}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # For a linear classifier, the per-example weight-gradient is an outer
        # product: (softmax(logits)-one_hot) x feature.  Its clean/fused cosine
        # therefore factors into the two vector cosines, avoiding extra backward
        # passes while measuring the exact classifier-weight gradient conflict.
        with torch.no_grad():
            clean_error = F.softmax(clean_output.detach(), dim=1)
            fused_error = F.softmax(output.detach(), dim=1)
            row = torch.arange(target.numel(), device=device)
            clean_error[row, target] -= 1.0
            fused_error[row, target] -= 1.0
            gradient_cosine = F.cosine_similarity(
                clean_error,
                fused_error,
                dim=1,
            ) * F.cosine_similarity(
                recipient_vector,
                fused_vector,
                dim=1,
            )

        batch_size = target.numel()
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((output.argmax(dim=1) == target).sum())
        clean_correct += int((clean_output.argmax(dim=1) == target).sum())
        clean_loss_sum += float(clean_per_sample.detach().sum())
        fused_loss_sum += float(fused_per_sample.detach().sum())
        risk_weight_sum += float(risk_weight.sum())
        risk_weight_floor_count += int(
            (risk_weight <= args.risk_weight_min + 1e-7).sum()
        )
        reference_fused_correct += int((reference_prediction == target).sum())
        ratio_sum += float(ratio.sum())
        ratio_square_sum += float((ratio * ratio).sum())
        groups = frequency_groups(target)
        for group_id in range(3):
            mask = groups == group_id
            ratio_group_sum[group_id] += float(ratio[mask].sum())
            ratio_group_count[group_id] += int(mask.sum())
        risk_values.extend(risk_score.detach().cpu().tolist())
        clean_loss_values.extend(clean_per_sample.detach().cpu().tolist())
        fused_loss_values.extend(fused_per_sample.detach().cpu().tolist())
        gradient_cosine_values.extend(gradient_cosine.detach().cpu().tolist())
        total += batch_size

    ratio_mean = ratio_sum / max(total, 1)
    ratio_variance = ratio_square_sum / max(total, 1) - ratio_mean**2
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "clean_accuracy": clean_correct / max(total, 1),
        "clean_loss": clean_loss_sum / max(total, 1),
        "fused_loss": fused_loss_sum / max(total, 1),
        "ratio_mean": ratio_mean,
        "ratio_std": math.sqrt(max(ratio_variance, 0.0)),
        "ratio_many": ratio_group_sum[0] / max(ratio_group_count[0], 1),
        "ratio_medium": ratio_group_sum[1] / max(ratio_group_count[1], 1),
        "ratio_few": ratio_group_sum[2] / max(ratio_group_count[2], 1),
        "reference_fused_accuracy": reference_fused_correct / max(total, 1),
        "risk_weight_mean": risk_weight_sum / max(total, 1),
        "risk_weight_floor_fraction": risk_weight_floor_count / max(total, 1),
        "gradient_cosine": float(np.mean(gradient_cosine_values)),
        "gradient_conflict_rate": float(
            np.mean(np.asarray(gradient_cosine_values) < 0.0)
        ),
        "risk_bins": summarize_risk_diagnostics(
            risk_values,
            clean_loss_values,
            fused_loss_values,
            gradient_cosine_values,
        ),
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
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    method_dir = output_root / method
    method_dir.mkdir(parents=True, exist_ok=True)
    result_path = method_dir / "metrics.json"
    current_path = method_dir / "current.pth"
    best_path = method_dir / "best.pth"
    if args.resume and result_path.exists():
        print(f"{method}: completed result exists; skipping")
        return json.loads(result_path.read_text(encoding="utf-8"))

    seed_everything(args.seed)
    backbone, classifier = build_models(backbone_state, classifier_state, device)
    reference_weight = reference_state["fc.weight"].to(device).detach().clone()
    reference_bias = reference_state["fc.bias"].to(device).detach().clone()
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.lr * args.lr_factor,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    best_overall = -1.0
    history: list[dict[str, Any]] = []

    if args.resume and current_path.exists():
        saved = torch.load(current_path, map_location=device, weights_only=False)
        saved_supervision = saved.get("supervision", "fused_only")
        if saved_supervision != args.supervision:
            raise ValueError(
                f"checkpoint supervision {saved_supervision!r} does not match "
                f"requested {args.supervision!r}"
            )
        saved_normalization = saved.get("normalize_supervision_loss", False)
        if saved_normalization != args.normalize_supervision_loss:
            raise ValueError(
                "checkpoint loss-normalization setting does not match request"
            )
        backbone.load_state_dict(saved["backbone"])
        classifier.load_state_dict(saved["classifier"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"])
        best_overall = float(saved["best_overall"])
        history = saved["history"]
        restore_rng_state(saved["rng"])
        print(f"{method}: resumed at epoch {start_epoch}")

    initial = evaluate(
        eval_loader,
        backbone,
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
            backbone,
            classifier,
            optimizer,
            method,
            reference_weight,
            reference_bias,
            device,
            args,
        )
        evaluation = evaluate(
            eval_loader,
            backbone,
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
            "supervision": args.supervision,
            "anchor_alpha": args.anchor_alpha,
            "risk_weight_min": args.risk_weight_min,
            "normalize_supervision_loss": args.normalize_supervision_loss,
            "backbone": backbone.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_overall": best_overall,
            "history": history,
            "rng": rng_state(),
        }
        torch.save(state, current_path)
        if is_best:
            torch.save(state, best_path)
        print(
            f"  epoch {epoch + 1:02d}/{args.epochs}: "
            f"train={train_metrics['accuracy']:.4f}, "
            f"clean={train_metrics['clean_accuracy']:.4f}, "
            f"test={evaluation['overall']:.4f}, "
            f"M/Md/F={evaluation['many']:.4f}/"
            f"{evaluation['medium']:.4f}/{evaluation['few']:.4f}, "
            f"lambda={train_metrics['ratio_mean']:.4f}, "
            f"risk_w={train_metrics['risk_weight_mean']:.4f}, "
            f"ref_fused={train_metrics['reference_fused_accuracy']:.4f}"
        )

    best_epoch_row = max(history, key=lambda item: item["eval"]["overall"])
    result = {
        "method": method,
        "supervision": args.supervision,
        "anchor_alpha": args.anchor_alpha,
        "risk_weight_min": args.risk_weight_min,
        "normalize_supervision_loss": args.normalize_supervision_loss,
        "initial": initial,
        "last": history[-1],
        "best": best_epoch_row,
        "history": history,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        for selection in ("last", "best"):
            epoch_row = result[selection]
            evaluation = epoch_row["eval"]
            train_metrics = epoch_row["train"]
            rows.append(
                {
                    "method": result["method"],
                    "supervision": result.get("supervision", "fused_only"),
                    "normalize_supervision_loss": result.get(
                        "normalize_supervision_loss",
                        False,
                    ),
                    "selection": selection,
                    "epoch": epoch_row["epoch"],
                    "overall": evaluation["overall"],
                    "many": evaluation["many"],
                    "medium": evaluation["medium"],
                    "few": evaluation["few"],
                    "train_accuracy": train_metrics["accuracy"],
                    "clean_accuracy": train_metrics["clean_accuracy"],
                    "clean_loss": train_metrics["clean_loss"],
                    "fused_loss": train_metrics["fused_loss"],
                    "ratio_mean": train_metrics["ratio_mean"],
                    "ratio_std": train_metrics["ratio_std"],
                    "ratio_many": train_metrics["ratio_many"],
                    "ratio_medium": train_metrics["ratio_medium"],
                    "ratio_few": train_metrics["ratio_few"],
                    "reference_fused_accuracy": train_metrics[
                        "reference_fused_accuracy"
                    ],
                    "risk_weight_mean": train_metrics["risk_weight_mean"],
                    "risk_weight_floor_fraction": train_metrics[
                        "risk_weight_floor_fraction"
                    ],
                    "gradient_cosine": train_metrics["gradient_cosine"],
                    "gradient_conflict_rate": train_metrics[
                        "gradient_conflict_rate"
                    ],
                    "low_risk_fused_minus_clean": train_metrics["risk_bins"]["low"][
                        "fused_minus_clean"
                    ],
                    "medium_risk_fused_minus_clean": train_metrics["risk_bins"][
                        "medium"
                    ]["fused_minus_clean"],
                    "high_risk_fused_minus_clean": train_metrics["risk_bins"]["high"][
                        "fused_minus_clean"
                    ],
                }
            )
    fieldnames = list(rows[0])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
        Path(args.reference_head).resolve() if args.reference_head else None
    )
    reference_state = load_reference_head(reference_path, classifier_state)

    experiment = vars(args).copy()
    experiment["methods"] = args.methods
    experiment["checkpoint"] = str(checkpoint_path)
    experiment["reference_head"] = (
        str(reference_path) if reference_path is not None else "stage1_ce_head"
    )
    experiment["output_dir"] = str(output_dir)
    experiment["device_resolved"] = str(device)
    experiment["checkpoint_epoch"] = checkpoint_metadata.get("epoch")
    experiment["checkpoint_best_acc"] = checkpoint_metadata.get(
        "best_acc",
        checkpoint_metadata.get("best_acc1"),
    )
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2),
        encoding="utf-8",
    )

    # Re-create loaders for every method indirectly through deterministic RNG.
    # With workers=0, the official Python-random dual sampler is reproducible.
    config = dataset_config()
    dataset = CIFAR100_LT(
        config,
        distributed=False,
        root=args.data_path,
        imb_factor=0.01,
        batch_size=args.batch_size,
        num_works=args.workers,
    )
    print(
        f"device={device}, train={len(dataset.train_instance.dataset)}, "
        f"test={len(dataset.eval.dataset)}, methods={args.methods}"
    )

    results = []
    for method in args.methods:
        # Reset all random streams before each paired method. Dataset transforms,
        # balanced recipient sampling, and long-tailed donor sampling therefore
        # see the same random stream in every run.
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
            )
        )
        write_summary(output_dir, results)

    print("\nFinal full Stage-II leaderboard (last epoch)")
    for result in sorted(
        results,
        key=lambda item: item["last"]["eval"]["overall"],
        reverse=True,
    ):
        evaluation = result["last"]["eval"]
        print(
            f"{result['method']:36s} overall={evaluation['overall']:.4f} "
            f"many={evaluation['many']:.4f} medium={evaluation['medium']:.4f} "
            f"few={evaluation['few']:.4f}"
        )
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
