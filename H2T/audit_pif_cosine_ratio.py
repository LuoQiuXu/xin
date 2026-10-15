"""Matched PIF OFF/ON cosine and PI-H2T ratio audit.

The two checkpoints are evaluated on the exact same tensors.  The audit covers
both the deterministic CIFAR-100 test set and the operational class-balanced
Stage-II recipient stream (with its training augmentation), while keeping both
backbones in eval mode so that the audit itself cannot mutate BN buffers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from datasets.cifar100 import CIFAR100_LT
from train_boundary_h2t_stage2 import (
    dataset_config,
    frequency_groups,
    load_stage1_checkpoint,
    seed_everything,
)
from train_pi_h2t_ratio_only_stage2 import build_models, classify_vector, pooled


QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
THRESHOLDS = (0.50, 0.75, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched PIF cosine/ratio audit")
    parser.add_argument("--pif-off-checkpoint", required=True)
    parser.add_argument("--pif-on-checkpoint", required=True)
    parser.add_argument("--data-path", default="./data/cifar100")
    parser.add_argument("--output", default="./outputs/pif_matched_cosine_ratio_audit.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    return parser.parse_args()


def scalar_metadata(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else None
    return value


def distribution(values: torch.Tensor) -> dict[str, Any]:
    values = values.float().cpu()
    if values.numel() == 0:
        return {"n": 0}
    quantiles = torch.quantile(values, torch.tensor(QUANTILES))
    result: dict[str, Any] = {
        "n": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    result.update(
        {f"p{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, quantiles)}
    )
    return result


def ratio_distribution(values: torch.Tensor) -> dict[str, Any]:
    result = distribution(values)
    if values.numel():
        result.update(
            {
                f"fraction_gt_{threshold:.2f}": float((values > threshold).float().mean())
                for threshold in THRESHOLDS
            }
        )
    return result


@torch.no_grad()
def audit_loader(
    loader: torch.utils.data.DataLoader,
    off_backbone: torch.nn.Module,
    off_classifier: torch.nn.Module,
    on_backbone: torch.nn.Module,
    on_classifier: torch.nn.Module,
    device: torch.device,
    max_batches: int,
) -> dict[str, Any]:
    off_backbone.eval()
    off_classifier.eval()
    on_backbone.eval()
    on_classifier.eval()

    stored: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "target",
            "off_cosine",
            "on_cosine",
            "off_lambda",
            "on_lambda",
            "off_correct",
            "on_correct",
        )
    }
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images, target = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        off_feature = pooled(off_backbone(images))
        on_feature = pooled(on_backbone(images))
        off_weight = off_classifier.fc.weight
        on_weight = on_classifier.fc.weight
        off_cosine = F.cosine_similarity(off_feature, off_weight[target], dim=1)
        on_cosine = F.cosine_similarity(on_feature, on_weight[target], dim=1)
        off_lambda = torch.clamp((1.0 + off_cosine) / 2.0, 0.0, 1.0)
        on_lambda = torch.clamp((1.0 + on_cosine) / 2.0, 0.0, 1.0)
        off_correct = classify_vector(off_classifier, off_feature).argmax(1) == target
        on_correct = classify_vector(on_classifier, on_feature).argmax(1) == target

        for key, value in (
            ("target", target),
            ("off_cosine", off_cosine),
            ("on_cosine", on_cosine),
            ("off_lambda", off_lambda),
            ("on_lambda", on_lambda),
            ("off_correct", off_correct),
            ("on_correct", on_correct),
        ):
            stored[key].append(value.detach().cpu())

    values = {key: torch.cat(parts) for key, parts in stored.items()}
    target = values["target"]
    off_correct = values["off_correct"].bool()
    on_correct = values["on_correct"].bool()
    groups = frequency_groups(target)

    result: dict[str, Any] = {
        "n": int(target.numel()),
        "accuracy": {
            "pif_off": float(off_correct.float().mean()),
            "pif_on": float(on_correct.float().mean()),
            "on_minus_off": float(on_correct.float().mean() - off_correct.float().mean()),
        },
        "overall": {
            "pif_off_cosine": distribution(values["off_cosine"]),
            "pif_on_cosine": distribution(values["on_cosine"]),
            "on_minus_off_cosine": distribution(
                values["on_cosine"] - values["off_cosine"]
            ),
            "pif_off_lambda": ratio_distribution(values["off_lambda"]),
            "pif_on_lambda": ratio_distribution(values["on_lambda"]),
            "on_minus_off_lambda": distribution(
                values["on_lambda"] - values["off_lambda"]
            ),
        },
        "frequency": {},
        "own_correctness": {},
        "joint_correctness": {},
    }

    for group_id, name in enumerate(("many", "medium", "few")):
        mask = groups == group_id
        result["frequency"][name] = {
            "pif_off_lambda": ratio_distribution(values["off_lambda"][mask]),
            "pif_on_lambda": ratio_distribution(values["on_lambda"][mask]),
            "on_minus_off_lambda": distribution(
                values["on_lambda"][mask] - values["off_lambda"][mask]
            ),
        }

    for model_name, value_prefix, correct in (
        ("pif_off", "off", off_correct),
        ("pif_on", "on", on_correct),
    ):
        lam = values[f"{value_prefix}_lambda"]
        result["own_correctness"][model_name] = {
            "correct": ratio_distribution(lam[correct]),
            "wrong": ratio_distribution(lam[~correct]),
        }

    joint_masks = {
        "off_correct_on_correct": off_correct & on_correct,
        "off_correct_on_wrong": off_correct & ~on_correct,
        "off_wrong_on_correct": ~off_correct & on_correct,
        "off_wrong_on_wrong": ~off_correct & ~on_correct,
    }
    for name, mask in joint_masks.items():
        result["joint_correctness"][name] = {
            "n": int(mask.sum()),
            "pif_off_lambda": ratio_distribution(values["off_lambda"][mask]),
            "pif_on_lambda": ratio_distribution(values["on_lambda"][mask]),
            "on_minus_off_lambda": distribution(
                values["on_lambda"][mask] - values["off_lambda"][mask]
            ),
        }

    off_mean = result["overall"]["pif_off_lambda"]["mean"]
    on_mean = result["overall"]["pif_on_lambda"]["mean"]
    result["kappa_mean_match"] = off_mean / on_mean
    return result


def load_arm(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any], bool]:
    backbone_state, classifier_state, metadata = load_stage1_checkpoint(
        checkpoint_path, device
    )
    has_pif_parameters = any(key.startswith("conv1d.") for key in backbone_state)
    pif_active = bool(metadata.get("pif_active", has_pif_parameters))
    backbone, classifier, resolved_active = build_models(
        backbone_state,
        classifier_state,
        device,
        pif_active=pif_active,
    )
    return backbone, classifier, metadata, resolved_active


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    off_path = Path(args.pif_off_checkpoint).resolve()
    on_path = Path(args.pif_on_checkpoint).resolve()
    off_backbone, off_classifier, off_meta, off_active = load_arm(off_path, device)
    on_backbone, on_classifier, on_meta, on_active = load_arm(on_path, device)
    if off_active:
        raise ValueError("--pif-off-checkpoint metadata says PIF is active")
    if not on_active:
        raise ValueError("--pif-on-checkpoint metadata says PIF is inactive")

    dataset = CIFAR100_LT(
        dataset_config(),
        distributed=False,
        root=args.data_path,
        imb_factor=0.01,
        batch_size=args.batch_size,
        num_works=args.workers,
    )
    result: dict[str, Any] = {
        "protocol": {
            "seed": args.seed,
            "pif_off_checkpoint": str(off_path),
            "pif_on_checkpoint": str(on_path),
            "pif_off_epoch": scalar_metadata(off_meta.get("epoch")),
            "pif_on_epoch": scalar_metadata(on_meta.get("epoch")),
            "pif_off_best_acc": scalar_metadata(off_meta.get("best_acc1")),
            "pif_on_best_acc": scalar_metadata(on_meta.get("best_acc1")),
            "audit_bn_mode": "eval_for_both",
            "ratio": "lambda_PI=(1+cos(f,w_y))/2",
        }
    }
    print("auditing deterministic test stream")
    result["test"] = audit_loader(
        dataset.eval,
        off_backbone,
        off_classifier,
        on_backbone,
        on_classifier,
        device,
        args.max_test_batches,
    )

    # Reset every input RNG before the operational balanced recipient stream.
    # Each sampled/augmented tensor is then passed to both models in one batch.
    seed_everything(args.seed)
    print("auditing operational balanced Stage-II recipient stream")
    result["stage2_balanced_recipient"] = audit_loader(
        dataset.train_instance,
        off_backbone,
        off_classifier,
        on_backbone,
        on_classifier,
        device,
        args.max_train_batches,
    )
    result["pre_registered_kappa_match"] = result["stage2_balanced_recipient"][
        "kappa_mean_match"
    ]

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "test_accuracy": result["test"]["accuracy"],
        "test_lambda_off": result["test"]["overall"]["pif_off_lambda"]["mean"],
        "test_lambda_on": result["test"]["overall"]["pif_on_lambda"]["mean"],
        "train_lambda_off": result["stage2_balanced_recipient"]["overall"]["pif_off_lambda"]["mean"],
        "train_lambda_on": result["stage2_balanced_recipient"]["overall"]["pif_on_lambda"]["mean"],
        "kappa_match": result["pre_registered_kappa_match"],
    }, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
