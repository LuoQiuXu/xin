"""Audit whether absolute signed-margin safety is available by frequency group."""

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
    reference_geometry,
    seed_everything,
)
from train_donor_selection_stage2 import signed_boundary_margin


CHECKPOINT = Path(
    r"D:\xiangmuxunlian\hafe_experiment\checkpoints\stage1"
    r"\cifar100_rho100_resnet32_ce_seed42\best.pth"
)
REFERENCE = Path(
    r"D:\xiangmuxunlian\新\H2T\outputs\balanced_reference"
    r"\ce_backbone_crt_reference_seed42.pth"
)


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

    count = torch.zeros(3, device=device)
    clean_correct = torch.zeros(3, device=device)
    has_safe = torch.zeros(3, device=device)
    selected_safe = torch.zeros(3, device=device)
    pair_critical_std_sum = 0.0
    pair_margin_std_sum = 0.0
    mixed_critical_safety = 0
    recipient_total = 0
    with torch.no_grad():
        for images, target, meta in dataset.train_instance:
            images = images.to(device)
            target = target.to(device)
            candidates = meta["sample_images"].to(device)
            batch_size, candidate_count = candidates.shape[:2]
            recipient = pooled(backbone(images))
            donors = pooled(backbone(candidates.flatten(0, 1))).view(
                batch_size,
                candidate_count,
                -1,
            )
            clean_margin = signed_boundary_margin(recipient, target, weight, bias)
            fused = (
                0.75 * recipient[:, None, :] + 0.25 * donors
            ).reshape(batch_size * candidate_count, -1)
            repeated_target = target[:, None].expand(-1, candidate_count).reshape(-1)
            margin = signed_boundary_margin(
                fused,
                repeated_target,
                weight,
                bias,
            ).view(batch_size, candidate_count)
            repeated_recipient = recipient[:, None, :].expand(
                -1,
                candidate_count,
                -1,
            ).reshape(batch_size * candidate_count, -1)
            _, _, critical = reference_geometry(
                repeated_recipient,
                donors.reshape(batch_size * candidate_count, -1),
                repeated_target,
                weight,
                bias,
                delta=0.0,
            )
            critical = critical.view(batch_size, candidate_count)
            group = torch.where(target < 36, 0, torch.where(target < 71, 1, 2))
            valid = (margin >= 0.0).any(dim=1)
            selection = torch.where(
                valid,
                margin.masked_fill(~(margin >= 0.0), torch.inf).argmin(dim=1),
                margin.argmax(dim=1),
            )
            selected_margin = margin[
                torch.arange(batch_size, device=device),
                selection,
            ]
            pair_critical_std_sum += float(critical.std(dim=1, unbiased=False).sum())
            pair_margin_std_sum += float(margin.std(dim=1, unbiased=False).sum())
            mixed_critical_safety += int(
                ((critical >= 0.25).any(dim=1) & (critical < 0.25).any(dim=1)).sum()
            )
            recipient_total += batch_size
            for group_id in range(3):
                mask = group == group_id
                count[group_id] += mask.sum()
                clean_correct[group_id] += (clean_margin[mask] >= 0.0).sum()
                has_safe[group_id] += valid[mask].sum()
                selected_safe[group_id] += (selected_margin[mask] >= 0.0).sum()

    print("group,count,clean_ref_correct,has_any_safe,hardest_safe_selected_safe")
    for name, group_id in (("many", 0), ("medium", 1), ("few", 2)):
        print(
            name,
            int(count[group_id]),
            f"{float(clean_correct[group_id] / count[group_id]):.4f}",
            f"{float(has_safe[group_id] / count[group_id]):.4f}",
            f"{float(selected_safe[group_id] / count[group_id]):.4f}",
            sep=",",
        )
    print(
        "pair_geometry,mean_within_recipient_std_lambda_critical,"
        "mean_within_recipient_std_signed_margin,mixed_safe_unsafe_at_lambda025"
    )
    print(
        "all",
        f"{pair_critical_std_sum / recipient_total:.4f}",
        f"{pair_margin_std_sum / recipient_total:.4f}",
        f"{mixed_critical_safety / recipient_total:.4f}",
        sep=",",
    )
    print(
        "pi_h2t_center_score_within_recipient_std,0.0000 "
        "(by construction: it is a function of recipient and class center only)"
    )


if __name__ == "__main__":
    main()
