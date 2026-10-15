"""Read-only audit of why recipient-relative hardest-safe donor selection behaves differently by group."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from datasets.cifar100 import CIFAR100_LT
from train_boundary_h2t_stage2 import (
    build_models,
    dataset_config,
    load_reference_head,
    load_stage1_checkpoint,
    pooled,
    seed_everything,
)
from train_donor_selection_stage2 import (
    recipient_relative_safety_threshold,
    select_candidate,
    signed_boundary_margin,
)


CHECKPOINT = Path(
    r"D:\xiangmuxunlian\hafe_experiment\checkpoints\stage1"
    r"\cifar100_rho100_resnet32_ce_seed42\best.pth"
)
REFERENCE = Path(
    r"D:\xiangmuxunlian\新\H2T\outputs\balanced_reference"
    r"\ce_backbone_crt_reference_seed42.pth"
)
LAMBDA = 0.25


def group_id(target: torch.Tensor) -> torch.Tensor:
    return torch.where(target < 36, 0, torch.where(target < 71, 1, 2))


def main() -> None:
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_state, classifier_state, _ = load_stage1_checkpoint(CHECKPOINT, device)
    reference_state = load_reference_head(REFERENCE, classifier_state)
    backbone, _ = build_models(backbone_state, classifier_state, device)
    backbone.eval()
    weight = reference_state["fc.weight"].to(device)
    bias = reference_state["fc.bias"].to(device)
    dataset = CIFAR100_LT(
        dataset_config(4),
        distributed=False,
        root="./data/cifar100",
        imb_factor=0.01,
        batch_size=128,
        num_works=0,
    )

    methods = (
        "random_donor_fixed025",
        "relative_hardest_safe_boundary_donor_fixed025",
        "shuffled_relative_hardest_safe_boundary_score_fixed025",
    )
    names = ("random", "relative_hardest_safe", "shuffled_score")
    # Separate generators reproduce the policy-local randomness and leave the
    # dataset transform/sampling stream identical across policies.
    generators = {
        method: torch.Generator(device=device.type).manual_seed(420_000)
        for method in methods
    }
    # [method, group] accumulators: count, clean-positive, selected-positive,
    # selected-relative-safe, selected margin, threshold, reference correct.
    stats = torch.zeros(len(methods), 3, 7, device=device)

    with torch.no_grad():
        for images, target, meta in dataset.train_instance:
            images = images.to(device)
            target = target.to(device)
            candidates = meta["sample_images"].to(device)
            batch_size, candidate_count = candidates.shape[:2]
            recipient = pooled(backbone(images))
            donors = pooled(backbone(candidates.flatten(0, 1))).view(
                batch_size, candidate_count, -1
            )
            fused = (
                (1.0 - LAMBDA) * recipient[:, None, :] + LAMBDA * donors
            )
            repeated_target = target[:, None].expand(-1, candidate_count).reshape(-1)
            margins = signed_boundary_margin(
                fused.reshape(batch_size * candidate_count, -1),
                repeated_target,
                weight,
                bias,
            ).view(batch_size, candidate_count)
            clean_margin = signed_boundary_margin(recipient, target, weight, bias)
            threshold = recipient_relative_safety_threshold(clean_margin, 0.0)
            critical_placeholder = torch.zeros_like(margins)
            groups = group_id(target)

            for method_index, method in enumerate(methods):
                selected = select_candidate(
                    method,
                    critical_placeholder,
                    margins,
                    clean_margin,
                    generators[method],
                    0.0,
                )
                row = torch.arange(batch_size, device=device)
                selected_margin = margins[row, selected]
                fused_feature = fused[row, selected]
                reference_correct = (
                    F.linear(fused_feature, weight, bias).argmax(dim=1) == target
                )
                for current_group in range(3):
                    mask = groups == current_group
                    if not mask.any():
                        continue
                    values = stats[method_index, current_group]
                    values[0] += mask.sum()
                    values[1] += (clean_margin[mask] >= 0.0).sum()
                    values[2] += (selected_margin[mask] >= 0.0).sum()
                    values[3] += (selected_margin[mask] >= threshold[mask]).sum()
                    values[4] += selected_margin[mask].sum()
                    values[5] += threshold[mask].sum()
                    values[6] += reference_correct[mask].sum()

    print(
        "method,group,count,clean_ref_correct,selected_ref_correct,"
        "selected_absolute_safe,meets_relative_constraint,"
        "selected_margin_mean,relative_threshold_mean"
    )
    for method_index, name in enumerate(names):
        for current_group, group_name in enumerate(("many", "medium", "few")):
            count = stats[method_index, current_group, 0]
            print(
                name,
                group_name,
                int(count),
                f"{float(stats[method_index, current_group, 1] / count):.4f}",
                f"{float(stats[method_index, current_group, 6] / count):.4f}",
                f"{float(stats[method_index, current_group, 2] / count):.4f}",
                f"{float(stats[method_index, current_group, 3] / count):.4f}",
                f"{float(stats[method_index, current_group, 4] / count):.4f}",
                f"{float(stats[method_index, current_group, 5] / count):.4f}",
                sep=",",
            )


if __name__ == "__main__":
    main()
