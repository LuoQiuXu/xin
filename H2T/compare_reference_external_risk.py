"""Compare CE and balanced reference geometry against an external evaluator.

The external evaluator is a cosine nearest-class-centroid classifier built from
the frozen Stage-I train features.  It is not either reference head.  Harmful
fusion is defined at a fixed donor mass as an external evaluator flip from a
correct recipient to an incorrect fused feature.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from fusion_risk_minival import (
    donor_conditioned_critical_ratio,
    pairwise_boundary_distances,
)
from train_fixed_feature_stage2 import match_ratio_mean_within_strata


GROUPS = (
    ("many", 0, 36),
    ("medium", 36, 71),
    ("few", 71, 100),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External reference-risk comparison")
    parser.add_argument("--input", required=True)
    parser.add_argument("--balanced-reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--fixed-lambda", type=float, default=0.25)
    parser.add_argument("--target-mean", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()
    args.seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    return args


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)


def centroid_logits(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return normalize_rows(features) @ centers.T


def margins_and_predictions(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(labels.size)
    target_logits = logits[rows, labels]
    competitors = logits.copy()
    competitors[rows, labels] = -np.inf
    margin = target_logits - competitors.max(axis=1)
    return margin, logits.argmax(axis=1)


def critical_ratios(
    recipient: np.ndarray,
    donor: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    result = np.empty(labels.size, dtype=np.float64)
    for start in range(0, labels.size, chunk_size):
        stop = min(start + chunk_size, labels.size)
        local_labels = labels[start:stop]
        recipient_distance = pairwise_boundary_distances(
            recipient[start:stop],
            local_labels,
            weight,
            bias,
            1e-12,
        )
        donor_distance = pairwise_boundary_distances(
            donor[start:stop],
            local_labels,
            weight,
            bias,
            1e-12,
        )
        local, _, _ = donor_conditioned_critical_ratio(
            recipient_distance,
            donor_distance,
            delta=0.0,
            eps=1e-12,
        )
        result[start:stop] = local
    return result


def safe_binary_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(
        average_precision_score(labels, scores)
    )


def group_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    result = {"overall": np.ones(labels.size, dtype=bool)}
    for name, start, stop in GROUPS:
        result[name] = (labels >= start) & (labels < stop)
    return result


def policy_metrics(
    recipient: np.ndarray,
    donor: np.ndarray,
    labels: np.ndarray,
    critical: np.ndarray,
    centers: np.ndarray,
    recipient_correct: np.ndarray,
    recipient_margin: np.ndarray,
    target_mean: float,
    beta: float,
) -> tuple[dict[str, float], np.ndarray]:
    frequency_group = np.where(labels < 36, 0, np.where(labels < 71, 1, 2))
    geometry_ratio = match_ratio_mean_within_strata(
        critical,
        frequency_group,
        target_mean,
        1.0,
    )
    ratio = beta * geometry_ratio + (1.0 - beta) * target_mean
    fused = (1.0 - ratio[:, None]) * recipient + ratio[:, None] * donor
    fused_margin, fused_prediction = margins_and_predictions(
        centroid_logits(fused, centers),
        labels,
    )
    masks = group_masks(labels)
    result: dict[str, float] = {
        "ratio_mean": float(ratio.mean()),
        "ratio_std": float(ratio.std()),
    }
    for name, mask in masks.items():
        correct = fused_prediction[mask] == labels[mask]
        eligible = mask & recipient_correct
        harmful = eligible & (fused_prediction != labels)
        result[f"{name}_accuracy"] = float(correct.mean())
        result[f"{name}_margin_drop"] = float(
            (recipient_margin[mask] - fused_margin[mask]).mean()
        )
        result[f"{name}_flip_rate"] = float(
            harmful.sum() / max(int(eligible.sum()), 1)
        )
    return result, ratio


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(Path(args.input).resolve(), allow_pickle=False)
    train_features = archive["train_features"].astype(np.float64)
    train_labels = archive["train_labels"].astype(np.int64)
    recipient = archive["test_features"].astype(np.float64)
    labels = archive["test_labels"].astype(np.int64)

    centers = np.stack(
        [train_features[train_labels == class_id].mean(axis=0) for class_id in range(100)]
    )
    centers = normalize_rows(centers)
    recipient_margin, recipient_prediction = margins_and_predictions(
        centroid_logits(recipient, centers),
        labels,
    )
    recipient_correct = recipient_prediction == labels

    balanced_raw = torch.load(
        Path(args.balanced_reference).resolve(),
        map_location="cpu",
        weights_only=False,
    )
    references = {
        "ce": (
            archive["classifier_weight"].astype(np.float64),
            archive["classifier_bias"].astype(np.float64),
        ),
        "balanced": (
            balanced_raw["reference_weight"].cpu().numpy().astype(np.float64),
            balanced_raw["reference_bias"].cpu().numpy().astype(np.float64),
        ),
    }

    prediction_rows: list[dict[str, float | int | str]] = []
    policy_rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed + 91_003)
        donor_indices = rng.integers(0, train_features.shape[0], size=recipient.shape[0])
        donor = train_features[donor_indices]
        fixed_fused = (
            (1.0 - args.fixed_lambda) * recipient
            + args.fixed_lambda * donor
        )
        _, fixed_prediction = margins_and_predictions(
            centroid_logits(fixed_fused, centers),
            labels,
        )
        harmful = recipient_correct & (fixed_prediction != labels)
        masks = group_masks(labels)

        for reference_name, (weight, bias) in references.items():
            critical = critical_ratios(
                recipient,
                donor,
                labels,
                weight,
                bias,
                args.chunk_size,
            )
            risk_score = 1.0 - critical
            for group_name, group_mask in masks.items():
                eligible = group_mask & recipient_correct
                auc, average_precision = safe_binary_metrics(
                    harmful[eligible].astype(np.int64),
                    risk_score[eligible],
                )
                prediction_rows.append(
                    {
                        "seed": seed,
                        "reference": reference_name,
                        "group": group_name,
                        "eligible": int(eligible.sum()),
                        "harmful": int(harmful[eligible].sum()),
                        "auroc": auc,
                        "auprc": average_precision,
                        "critical_mean": float(critical[group_mask].mean()),
                    }
                )

            metrics, _ = policy_metrics(
                recipient,
                donor,
                labels,
                critical,
                centers,
                recipient_correct,
                recipient_margin,
                args.target_mean,
                args.beta,
            )
            policy_rows.append(
                {
                    "seed": seed,
                    "reference": reference_name,
                    **metrics,
                }
            )

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "prediction.csv", prediction_rows)
    write_csv(output_dir / "policy.csv", policy_rows)

    summary: dict[str, object] = {
        "config": vars(args),
        "external_evaluator": "cosine nearest train-class centroid",
        "recipient_accuracy": float(recipient_correct.mean()),
        "prediction": {},
        "policy": {},
    }
    for reference_name in references:
        summary["prediction"][reference_name] = {}
        for group_name in group_masks(labels):
            rows = [
                row
                for row in prediction_rows
                if row["reference"] == reference_name and row["group"] == group_name
            ]
            summary["prediction"][reference_name][group_name] = {
                metric: {
                    "mean": float(np.nanmean([row[metric] for row in rows])),
                    "std": float(np.nanstd([row[metric] for row in rows], ddof=1)),
                }
                for metric in ("auroc", "auprc", "critical_mean")
            }
        rows = [row for row in policy_rows if row["reference"] == reference_name]
        summary["policy"][reference_name] = {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "std": float(np.std([row[key] for row in rows], ddof=1)),
            }
            for key in rows[0]
            if key not in ("seed", "reference")
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
