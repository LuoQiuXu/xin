"""Audit whether PI-H2T centre distance is a coarse multi-boundary proxy.

This is a frozen-geometry diagnostic, not a Stage-II training experiment.
It keeps the selected Stage-I PIF backbone and its original CE classifier head
fixed, extracts features from the complete CIFAR100 test set, and asks five
recipient-only questions:

1. How strongly does the official PI centre distance ``d_c`` correlate with
   the signed distance ``m_1`` to the nearest *among all 99* competing class
   boundaries?
2. At nearly equal centre distance (within-class centre-distance deciles), how
   broad is the distribution of ``m_1``?
3. Is one nearest boundary enough, or do ``m_1, m_2, m_3`` reveal a richer
   local boundary profile?
4. At equal centre distance, does the identity of the nearest competitor vary?
5. How often do pairs with similar centre ranks have substantially different
   nearest-boundary ranks?

The geometry is always measured with the frozen original Stage-I CE head.  No
donor feature, fusion ratio, Stage-II classifier, OOF head, or test-set model
selection is involved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from datasets.cifar100 import CIFAR100_LT
from train_boundary_h2t_stage2 import dataset_config, frequency_groups, load_stage1_checkpoint
from train_pi_h2t_ratio_only_stage2 import build_models


GROUP_NAMES = ("Many", "Medium", "Few")
GROUP_COLORS = ("#2a6fbb", "#e28b24", "#b44e99")
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen PI-H2T centre-distance versus multi-boundary geometry audit"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="./data/cifar100")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--center-bins",
        type=int,
        default=10,
        help="Within-class quantile bins of the official PI centre distance.",
    )
    parser.add_argument(
        "--center-rank-tolerance",
        type=float,
        default=0.05,
        help="Maximum within-class percentile-rank difference for a centre-similar pair.",
    )
    parser.add_argument(
        "--boundary-rank-gap",
        type=float,
        default=0.30,
        help="Minimum within-class percentile-rank difference for a boundary-divergent pair.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Smoke-test limit; zero means the full 10,000-image test set.",
    )
    args = parser.parse_args()
    if args.center_bins < 2:
        raise ValueError("--center-bins must be at least 2")
    if not 0.0 <= args.center_rank_tolerance <= 1.0:
        raise ValueError("--center-rank-tolerance must be in [0, 1]")
    if not 0.0 <= args.boundary_rank_gap <= 1.0:
        raise ValueError("--boundary-rank-gap must be in [0, 1]")
    return args


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def normalized_pairwise_weight_distance(weight: torch.Tensor) -> torch.Tensor:
    squared = (weight * weight).sum(dim=1)
    return torch.sqrt(
        torch.clamp(
            squared[:, None] + squared[None, :] - 2.0 * (weight @ weight.t()),
            min=0.0,
        )
    ).clamp_min(EPS)


def stable_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Ordinal rank in [0, 1]; deterministic and sufficient for continuous scores."""
    count = values.size
    if count <= 1:
        return np.zeros(count, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    rank = np.empty(count, dtype=np.float64)
    rank[order] = np.arange(count, dtype=np.float64)
    return rank / float(count - 1)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return float("nan")
    x = stable_percentile_rank(left)
    y = stable_percentile_rank(right)
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denominator) if denominator > 0.0 else float("nan")


def quantile(values: np.ndarray, level: float) -> float:
    return float(np.quantile(values, level)) if values.size else float("nan")


def basic_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(values.size),
        "mean": float(values.mean()) if values.size else float("nan"),
        "std": float(values.std(ddof=0)) if values.size else float("nan"),
        "min": float(values.min()) if values.size else float("nan"),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "max": float(values.max()) if values.size else float("nan"),
    }


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def finite_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=0)) if array.size else float("nan")


def rows_to_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def extract_geometry(args: argparse.Namespace) -> dict[str, np.ndarray]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).resolve()
    backbone_state, classifier_state, metadata = load_stage1_checkpoint(checkpoint_path, device)
    stage1_has_pif = any(key.startswith("conv1d.") for key in backbone_state)
    pif_active = bool(metadata.get("pif_active", stage1_has_pif))
    backbone, classifier, uses_pif = build_models(
        backbone_state, classifier_state, device, pif_active=pif_active
    )
    backbone.eval()
    classifier.eval()
    reference_weight = classifier_state["fc.weight"].to(device).detach()
    reference_bias = classifier_state["fc.bias"].to(device).detach()
    denominator = normalized_pairwise_weight_distance(reference_weight)

    dataset = CIFAR100_LT(
        dataset_config(),
        distributed=False,
        root=args.data_path,
        imb_factor=0.01,
        batch_size=args.batch_size,
        num_works=args.workers,
    )
    buffers: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("label", "group", "dc", "m1", "m2", "m3", "jstar", "prediction", "correct")
    }
    processed = 0
    for batch_index, (images, target) in enumerate(dataset.eval):
        if args.max_eval_batches and batch_index >= args.max_eval_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        feature_map = backbone(images)
        feature = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        logits = F.linear(feature, reference_weight, reference_bias)

        # The exact PI-H2T centre quantity: r_PI=(1-cos(f,w_y))/2.
        cosine = F.cosine_similarity(feature, reference_weight[target], dim=1, eps=EPS)
        dc = torch.clamp((1.0 - cosine) / 2.0, min=0.0, max=1.0)

        numerator = logits.gather(1, target[:, None]) - logits
        distance = numerator / denominator[target]
        row = torch.arange(target.numel(), device=device)
        distance[row, target] = torch.inf
        sorted_distance, sorted_competitor = torch.sort(distance, dim=1)
        prediction = logits.argmax(dim=1)
        values = {
            "label": target,
            "group": frequency_groups(target),
            "dc": dc,
            "m1": sorted_distance[:, 0],
            "m2": sorted_distance[:, 1],
            "m3": sorted_distance[:, 2],
            "jstar": sorted_competitor[:, 0],
            "prediction": prediction,
            "correct": prediction.eq(target),
        }
        for name, value in values.items():
            buffers[name].append(value.detach().cpu().numpy())
        processed += target.numel()
        print(f"geometry batch {batch_index + 1:02d}: processed={processed}", flush=True)

    data = {name: np.concatenate(parts) for name, parts in buffers.items()}
    if data["label"].size == 0:
        raise RuntimeError("no evaluation samples were processed")
    data["delta12"] = data["m2"] - data["m1"]
    data["delta23"] = data["m3"] - data["m2"]
    data["pif_active"] = np.asarray([int(uses_pif)])
    data["checkpoint"] = np.asarray([str(checkpoint_path)])
    return data


def add_within_class_ranks(data: dict[str, np.ndarray], center_bins: int) -> None:
    count = data["label"].size
    dc_rank = np.empty(count, dtype=np.float64)
    m1_rank = np.empty(count, dtype=np.float64)
    dc_bin = np.empty(count, dtype=np.int64)
    for label in np.unique(data["label"]):
        indices = np.flatnonzero(data["label"] == label)
        ranked_dc = stable_percentile_rank(data["dc"][indices])
        ranked_m1 = stable_percentile_rank(data["m1"][indices])
        dc_rank[indices] = ranked_dc
        m1_rank[indices] = ranked_m1
        # Right-most value 1.0 belongs to the final bin.
        dc_bin[indices] = np.minimum((ranked_dc * center_bins).astype(int), center_bins - 1)
    data["dc_rank_within_class"] = dc_rank
    data["m1_rank_within_class"] = m1_rank
    data["dc_bin_within_class"] = dc_bin


def make_sample_rows(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(data["label"].size):
        group = int(data["group"][index])
        rows.append(
            {
                "sample_id": index,
                "class_id": int(data["label"][index]),
                "frequency_group": GROUP_NAMES[group],
                "pi_center_distance_dc": float(data["dc"][index]),
                "nearest_margin_m1": float(data["m1"][index]),
                "second_margin_m2": float(data["m2"][index]),
                "third_margin_m3": float(data["m3"][index]),
                "delta12_m2_minus_m1": float(data["delta12"][index]),
                "nearest_competitor_jstar": int(data["jstar"][index]),
                "reference_prediction": int(data["prediction"][index]),
                "reference_correct": int(data["correct"][index]),
                "dc_rank_within_class": float(data["dc_rank_within_class"][index]),
                "m1_rank_within_class": float(data["m1_rank_within_class"][index]),
                "dc_bin_within_class": int(data["dc_bin_within_class"][index]),
            }
        )
    return rows


def correlation_summary(data: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for group_id, group_name in enumerate(GROUP_NAMES):
        group_indices = np.flatnonzero(data["group"] == group_id)
        class_rhos: list[float] = []
        for label in np.unique(data["label"][group_indices]):
            indices = group_indices[data["label"][group_indices] == label]
            rho = spearman(data["dc"][indices], data["m1"][indices])
            class_rhos.append(rho)
            class_rows.append(
                {
                    "class_id": int(label),
                    "frequency_group": group_name,
                    "n": int(indices.size),
                    "spearman_dc_m1": rho,
                    "dc_mean": float(data["dc"][indices].mean()),
                    "m1_mean": float(data["m1"][indices].mean()),
                    "m1_negative_fraction": float((data["m1"][indices] < 0.0).mean()),
                }
            )
        group_rows.append(
            {
                "frequency_group": group_name,
                "n_samples": int(group_indices.size),
                "n_classes": int(np.unique(data["label"][group_indices]).size),
                "reference_accuracy": float(data["correct"][group_indices].mean()),
                "pooled_spearman_dc_m1": spearman(data["dc"][group_indices], data["m1"][group_indices]),
                "macro_class_spearman_dc_m1": finite_mean(class_rhos),
                "class_spearman_std": finite_std(class_rhos),
                "dc_mean": float(data["dc"][group_indices].mean()),
                "m1_mean": float(data["m1"][group_indices].mean()),
                "m1_negative_fraction": float((data["m1"][group_indices] < 0.0).mean()),
            }
        )
    return group_rows, class_rows


def center_bin_summary(data: dict[str, np.ndarray], center_bins: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_id, group_name in enumerate(GROUP_NAMES):
        for bin_id in range(center_bins):
            mask = (data["group"] == group_id) & (data["dc_bin_within_class"] == bin_id)
            values = data["m1"][mask]
            row: dict[str, Any] = {
                "frequency_group": group_name,
                "center_bin": bin_id + 1,
                "n_samples": int(values.size),
                "n_classes": int(np.unique(data["label"][mask]).size),
                "m1_iqr": quantile(values, 0.75) - quantile(values, 0.25),
                "m1_negative_fraction": float((values < 0.0).mean()) if values.size else float("nan"),
            }
            row.update({f"m1_{key}": value for key, value in basic_stats(values).items()})
            rows.append(row)
    return rows


def boundary_profile_summary(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_id, group_name in enumerate(GROUP_NAMES):
        mask = data["group"] == group_id
        for name in ("m1", "m2", "m3", "delta12", "delta23"):
            row: dict[str, Any] = {"frequency_group": group_name, "metric": name}
            row.update(basic_stats(data[name][mask]))
            rows.append(row)
    return rows


def entropy_effective_count(values: np.ndarray) -> tuple[float, float, int]:
    _, counts = np.unique(values, return_counts=True)
    probability = counts / counts.sum()
    entropy = float(-(probability * np.log(np.clip(probability, EPS, 1.0))).sum())
    return entropy, float(np.exp(entropy)), int(counts.max())


def competitor_diversity(data: dict[str, np.ndarray], center_bins: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_rows: list[dict[str, Any]] = []
    for label in np.unique(data["label"]):
        label_indices = np.flatnonzero(data["label"] == label)
        group_id = int(data["group"][label_indices[0]])
        for bin_id in range(center_bins):
            indices = label_indices[data["dc_bin_within_class"][label_indices] == bin_id]
            if not indices.size:
                continue
            competitor = data["jstar"][indices]
            entropy, effective, largest = entropy_effective_count(competitor)
            cell_rows.append(
                {
                    "class_id": int(label),
                    "frequency_group": GROUP_NAMES[group_id],
                    "center_bin": bin_id + 1,
                    "n_samples": int(indices.size),
                    "unique_nearest_competitors": int(np.unique(competitor).size),
                    "nearest_competitor_entropy": entropy,
                    "effective_nearest_competitors": effective,
                    "top_competitor_share": largest / float(indices.size),
                    "mean_m1": float(data["m1"][indices].mean()),
                }
            )

    group_rows: list[dict[str, Any]] = []
    for group_name in GROUP_NAMES:
        rows = [row for row in cell_rows if row["frequency_group"] == group_name]
        for bin_id in range(center_bins):
            matching = [row for row in rows if row["center_bin"] == bin_id + 1]
            if not matching:
                continue
            group_rows.append(
                {
                    "frequency_group": group_name,
                    "center_bin": bin_id + 1,
                    "n_class_bin_cells": len(matching),
                    "mean_unique_nearest_competitors": float(np.mean([row["unique_nearest_competitors"] for row in matching])),
                    "mean_effective_nearest_competitors": float(np.mean([row["effective_nearest_competitors"] for row in matching])),
                    "mean_top_competitor_share": float(np.mean([row["top_competitor_share"] for row in matching])),
                }
            )
    return cell_rows, group_rows


def divergent_pairs(
    data: dict[str, np.ndarray],
    center_tolerance: float,
    boundary_gap: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    class_rows: list[dict[str, Any]] = []
    for label in np.unique(data["label"]):
        indices = np.flatnonzero(data["label"] == label)
        q_center = data["dc_rank_within_class"][indices]
        q_boundary = data["m1_rank_within_class"][indices]
        # Upper triangle: every unordered same-class pair exactly once.
        center_diff = np.abs(q_center[:, None] - q_center[None, :])
        boundary_diff = np.abs(q_boundary[:, None] - q_boundary[None, :])
        upper = np.triu(np.ones((indices.size, indices.size), dtype=bool), k=1)
        center_similar = upper & (center_diff <= center_tolerance + 1e-12)
        divergent = center_similar & (boundary_diff >= boundary_gap - 1e-12)
        denominator = int(center_similar.sum())
        numerator = int(divergent.sum())
        group_id = int(data["group"][indices[0]])
        class_rows.append(
            {
                "class_id": int(label),
                "frequency_group": GROUP_NAMES[group_id],
                "n_samples": int(indices.size),
                "center_similar_pair_count": denominator,
                "boundary_divergent_pair_count": numerator,
                "conditional_divergent_pair_rate": numerator / denominator if denominator else float("nan"),
            }
        )

    group_rows: list[dict[str, Any]] = []
    for group_name in GROUP_NAMES:
        rows = [row for row in class_rows if row["frequency_group"] == group_name]
        denominator = sum(int(row["center_similar_pair_count"]) for row in rows)
        numerator = sum(int(row["boundary_divergent_pair_count"]) for row in rows)
        rate_values = np.asarray([row["conditional_divergent_pair_rate"] for row in rows], dtype=float)
        group_rows.append(
            {
                "frequency_group": group_name,
                "n_classes": len(rows),
                "center_similar_pair_count": denominator,
                "boundary_divergent_pair_count": numerator,
                "pooled_conditional_divergent_pair_rate": numerator / denominator if denominator else float("nan"),
                "macro_class_conditional_divergent_pair_rate": finite_mean(rate_values),
                "macro_rate_std": finite_std(rate_values),
                "macro_rate_sem": finite_std(rate_values) / math.sqrt(max(np.isfinite(rate_values).sum(), 1)),
            }
        )
    return class_rows, group_rows


def plot_scatter(data: dict[str, np.ndarray], group_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for group_id, axis in enumerate(axes):
        mask = data["group"] == group_id
        axis.scatter(
            data["dc"][mask], data["m1"][mask], s=9, alpha=0.28,
            color=GROUP_COLORS[group_id], linewidths=0,
        )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        result = group_rows[group_id]
        axis.set_title(
            f"{GROUP_NAMES[group_id]}\n"
            f"pooled $\\rho$={result['pooled_spearman_dc_m1']:.3f}; "
            f"class-macro $\\rho$={result['macro_class_spearman_dc_m1']:.3f}"
        )
        axis.set_xlabel(r"PI centre distance $d_c=(1-\cos(f,w_y))/2$")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel(r"nearest signed boundary distance $m_1$")
    fig.suptitle("Centre distance is compared with the nearest of all 99 competing boundaries", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_center_bin_distribution(data: dict[str, np.ndarray], center_bins: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for group_id, axis in enumerate(axes):
        samples = [
            data["m1"][(data["group"] == group_id) & (data["dc_bin_within_class"] == bin_id)]
            for bin_id in range(center_bins)
        ]
        box = axis.boxplot(samples, patch_artist=True, showfliers=False, widths=0.65)
        for patch in box["boxes"]:
            patch.set(facecolor=GROUP_COLORS[group_id], alpha=0.62)
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(GROUP_NAMES[group_id])
        axis.set_xlabel("within-class $d_c$ decile (low → high)")
        axis.set_xticks(range(1, center_bins + 1))
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel(r"nearest signed boundary distance $m_1$")
    fig.suptitle("Even samples with comparable PI centre distance can occupy different boundary states", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_profile(data: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for group_id, axis in enumerate(axes):
        mask = data["group"] == group_id
        values = [data[name][mask] for name in ("m1", "m2", "m3", "delta12")]
        box = axis.boxplot(values, patch_artist=True, showfliers=False, widths=0.62)
        colors = ("#d44a3a", "#e8a43a", "#4e9a74", "#7668aa")
        for patch, color in zip(box["boxes"], colors):
            patch.set(facecolor=color, alpha=0.68)
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        axis.set_xticklabels((r"$m_1$", r"$m_2$", r"$m_3$", r"$\Delta_{12}$"))
        axis.set_title(GROUP_NAMES[group_id])
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("normalised signed distance")
    fig.suptitle("Local boundary profile: nearest, second-nearest, third-nearest, and the first gap", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_competitor_diversity(summary_rows: list[dict[str, Any]], center_bins: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    for group_id, axis in enumerate(axes):
        group_rows = [row for row in summary_rows if row["frequency_group"] == GROUP_NAMES[group_id]]
        by_bin = {int(row["center_bin"]): row for row in group_rows}
        # Missing rows occur only in deliberately partial smoke runs.  Full
        # CIFAR100 test evaluation has every class and every decile.
        unique = [by_bin.get(bin_id + 1, {}).get("mean_unique_nearest_competitors", np.nan) for bin_id in range(center_bins)]
        effective = [by_bin.get(bin_id + 1, {}).get("mean_effective_nearest_competitors", np.nan) for bin_id in range(center_bins)]
        top_share = [by_bin.get(bin_id + 1, {}).get("mean_top_competitor_share", np.nan) for bin_id in range(center_bins)]
        x = np.arange(center_bins)
        axis.bar(x - 0.18, unique, width=0.36, color=GROUP_COLORS[group_id], alpha=0.74, label="unique $j^*$")
        axis.bar(x + 0.18, effective, width=0.36, color="#64748b", alpha=0.64, label="effective $j^*$")
        axis.set_xticks(x)
        axis.set_xticklabels(range(1, center_bins + 1))
        axis.set_xlabel("within-class $d_c$ decile")
        axis.set_title(GROUP_NAMES[group_id])
        axis.set_ylabel("competitor diversity per class × decile")
        axis.grid(axis="y", alpha=0.18)
        right = axis.twinx()
        right.plot(x, top_share, color="#8b1e3f", marker="o", linewidth=1.8, label="top share")
        right.set_ylim(0.0, 1.05)
        right.set_ylabel("top-$j^*$ share")
        if group_id == 0:
            axis.legend(loc="upper left", fontsize=8)
            right.legend(loc="upper right", fontsize=8)
    fig.suptitle("Nearest-competitor identity within the same class and centre-distance decile", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_divergent_pairs(group_rows: list[dict[str, Any]], path: Path, center_tolerance: float, boundary_gap: float) -> None:
    labels = [row["frequency_group"] for row in group_rows]
    rates = np.asarray([row["macro_class_conditional_divergent_pair_rate"] for row in group_rows])
    errors = np.asarray([1.96 * row["macro_rate_sem"] for row in group_rows])
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    bars = axis.bar(labels, rates * 100.0, yerr=errors * 100.0, capsize=5, color=GROUP_COLORS, alpha=0.78)
    axis.set_ylabel("conditional pair rate (%)")
    finite_upper = (rates + errors)[np.isfinite(rates + errors)]
    upper = float(finite_upper.max() * 125.0) if finite_upper.size else 5.0
    axis.set_ylim(0.0, max(5.0, upper))
    axis.set_title(
        "Centre-similar yet boundary-divergent pairs\n"
        + rf"$|\Delta q(d_c)|\leq {center_tolerance:.2f}$ and $|\Delta q(m_1)|\geq {boundary_gap:.2f}$ within the same class"
    )
    axis.grid(axis="y", alpha=0.18)
    label_offset = max(0.2, float(np.nanmax(errors) * 35.0) if np.isfinite(errors).any() else 0.2)
    for bar, rate, row in zip(bars, rates, group_rows):
        if not np.isfinite(rate):
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + label_offset,
            f"{rate * 100.0:.1f}%\nN={row['center_similar_pair_count']:,}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_float(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.4f}"


def write_report(
    path: Path,
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    correlation_rows: list[dict[str, Any]],
    center_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    diversity_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# PI-H2T 中心距离与多竞争边界几何审计",
        "",
        "## 协议",
        "",
        "- 冻结 PIF Stage-I backbone 与同一 checkpoint 的原始 CE classifier head。",
        "- 使用完整 CIFAR100 测试集；不训练 Stage-II、不使用 donor、不计算 fusion ratio，也不做测试集选参。",
        "- 中心量严格采用 PI-H2T 的 `d_c=(1-cos(f,w_y))/2`。其中 `w_y` 是 Stage-I classifier weight，因此它是 PI 的类中心代理，而不是真实样本均值。",
        "- 对真实类 `y` 的每个竞争类 `j!=y`，计算归一化有符号边界距离：",
        "",
        "  `d_{y,j}(f)=((w_y-w_j)^T f+(b_y-b_j))/(||w_y-w_j||_2+eps)`。",
        "",
        "  `m_1<=m_2<=m_3` 是 99 个 `d_{y,j}` 的前三个顺序统计量；`m_1<0` 代表至少跨过一条真实类边界。",
        "- `center-distance decile` 始终在**同一真实类别内部**分位，避免类别间尺度差异制造伪相关。",
        "",
        "## 1. 中心距离与最近边界距离",
        "",
        "| 组别 | N | reference acc. | pooled Spearman ρ(d_c,m1) | 类内 ρ 宏平均 | m1<0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in correlation_rows:
        lines.append(
            f"| {row['frequency_group']} | {row['n_samples']:,} | {row['reference_accuracy']:.3%} | "
            f"{format_float(row['pooled_spearman_dc_m1'])} | {format_float(row['macro_class_spearman_dc_m1'])} | "
            f"{row['m1_negative_fraction']:.3%} |"
        )
    lines.extend([
        "",
        "解释：若 PI 的中心距离完全等价于局部边界位置，类内相关应接近单调确定关系，且同一 `d_c` decile 内的 `m1` 不应很分散。此审计把相关性、条件分布和配对不一致同时报告，不能仅凭一个相关系数下结论。",
        "",
        "## 2. 相近中心距离下的最近边界分散度",
        "",
        "| 组别 | decile | N | m1 median | m1 IQR | m1<0 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in center_rows:
        lines.append(
            f"| {row['frequency_group']} | {row['center_bin']} | {row['n_samples']:,} | "
            f"{row['m1_median']:.4f} | {row['m1_iqr']:.4f} | {row['m1_negative_fraction']:.3%} |"
        )
    lines.extend([
        "",
        "## 3. 多边界 profile",
        "",
        "| 组别 | 指标 | mean | median | P10 | P90 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in profile_rows:
        lines.append(
            f"| {row['frequency_group']} | {row['metric']} | {row['mean']:.4f} | {row['median']:.4f} | {row['p10']:.4f} | {row['p90']:.4f} |"
        )
    lines.extend([
        "",
        "`Δ12=m2-m1` 很小表示样本附近不止一条边界都接近；很大表示存在一条主导的最近边界。仅使用 `m1` 会丢弃这一区分。",
        "",
        "## 4. 同一中心距离区间内的最近竞争类多样性",
        "",
        "以下为每个 `真实类 × center decile` 单元的平均值；`effective` 是 `exp(entropy)`，`top share` 越低说明最近竞争类越不稳定、越多样。",
        "",
        "| 组别 | decile | 平均 unique j* | 平均 effective j* | 平均 top-j* share |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in diversity_rows:
        lines.append(
            f"| {row['frequency_group']} | {row['center_bin']} | {row['mean_unique_nearest_competitors']:.3f} | "
            f"{row['mean_effective_nearest_competitors']:.3f} | {row['mean_top_competitor_share']:.3f} |"
        )
    lines.extend([
        "",
        "## 5. center 相近但 boundary 差异大的样本对",
        "",
        f"在同一真实类内，对无序样本对计算分位 rank。条件是 `|Δq(d_c)|≤{args.center_rank_tolerance:.2f}`；在满足该条件的 pair 中，若 `|Δq(m1)|≥{args.boundary_rank_gap:.2f}`，则记为 boundary-divergent。下面先给 pooled pair rate，再给类宏平均。",
        "",
        "| 组别 | center-similar pair 数 | boundary-divergent 数 | pooled rate | 类宏平均 rate |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in pair_rows:
        lines.append(
            f"| {row['frequency_group']} | {row['center_similar_pair_count']:,} | {row['boundary_divergent_pair_count']:,} | "
            f"{row['pooled_conditional_divergent_pair_rate']:.3%} | {row['macro_class_conditional_divergent_pair_rate']:.3%} |"
        )
    lines.extend([
        "",
        "## 对论文动机可支持的判据",
        "",
        "下面三项共同出现，才构成较强的‘有效但粗粒度 proxy’证据：",
        "",
        "1. `d_c` 与 `m1` 不是近乎确定的类内单调关系；",
        "2. 同一 `d_c` decile 内 `m1` 仍有明显 IQR / 正负状态混合；",
        "3. 同一 decile 内 `j*` 多样，或存在可观比例的 center-similar / boundary-divergent pairs。",
        "",
        "若 Few 的上述不一致性强于 Many，则可以进一步使用‘这种信息缺失在尾类更严重’；若 Few 没有更强，只能稳妥主张它是多类局部几何的普遍信息缺失，不能把尾类强化写成结论。",
        "",
        "## 产物",
        "",
        "- `01_center_distance_vs_nearest_boundary.png`：输出 1。",
        "- `02_m1_by_within_class_center_bin.png`：输出 2。",
        "- `03_multiboundary_profile.png`：输出 3。",
        "- `04_nearest_competitor_diversity.png`：输出 4。",
        "- `05_center_similar_boundary_divergence.png`：输出 5。",
        "- `sample_geometry.csv`、各 summary CSV：可复核的原始逐样本和聚合统计。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Starting frozen multi-boundary geometry audit", flush=True)
    data = extract_geometry(args)
    add_within_class_ranks(data, args.center_bins)

    correlation_rows, class_correlation_rows = correlation_summary(data)
    center_rows = center_bin_summary(data, args.center_bins)
    profile_rows = boundary_profile_summary(data)
    diversity_cell_rows, diversity_rows = competitor_diversity(data, args.center_bins)
    pair_class_rows, pair_rows = divergent_pairs(
        data, args.center_rank_tolerance, args.boundary_rank_gap
    )

    rows_to_csv(output_dir / "sample_geometry.csv", make_sample_rows(data))
    rows_to_csv(output_dir / "center_boundary_correlation_summary.csv", correlation_rows)
    rows_to_csv(output_dir / "center_boundary_correlation_by_class.csv", class_correlation_rows)
    rows_to_csv(output_dir / "m1_by_center_distance_bin.csv", center_rows)
    rows_to_csv(output_dir / "multiboundary_profile_summary.csv", profile_rows)
    rows_to_csv(output_dir / "nearest_competitor_diversity_by_class_bin.csv", diversity_cell_rows)
    rows_to_csv(output_dir / "nearest_competitor_diversity_summary.csv", diversity_rows)
    rows_to_csv(output_dir / "center_similar_boundary_divergent_pairs_by_class.csv", pair_class_rows)
    rows_to_csv(output_dir / "center_similar_boundary_divergent_pairs_summary.csv", pair_rows)

    plot_scatter(data, correlation_rows, output_dir / "01_center_distance_vs_nearest_boundary.png")
    plot_center_bin_distribution(data, args.center_bins, output_dir / "02_m1_by_within_class_center_bin.png")
    plot_boundary_profile(data, output_dir / "03_multiboundary_profile.png")
    plot_competitor_diversity(diversity_rows, args.center_bins, output_dir / "04_nearest_competitor_diversity.png")
    plot_divergent_pairs(
        pair_rows,
        output_dir / "05_center_similar_boundary_divergence.png",
        args.center_rank_tolerance,
        args.boundary_rank_gap,
    )
    write_report(
        output_dir / "MULTIBOUNDARY_GEOMETRY_AUDIT_REPORT.md",
        args,
        data,
        correlation_rows,
        center_rows,
        profile_rows,
        diversity_rows,
        pair_rows,
    )
    protocol = {
        **vars(args),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "geometry_reference": "same checkpoint original Stage-I CE classifier head",
        "center_distance": "PI-H2T d_c=(1-cos(f,w_y))/2",
        "all_competing_boundaries": "d_yj=((w_y-w_j)^T f+(b_y-b_j))/(||w_y-w_j||_2+eps), j!=y",
        "data": "CIFAR100 full test set; no Stage-II training, donor, fusion, OOF, or test-set model selection",
        "n_samples": int(data["label"].size),
        "pif_active": bool(int(data["pif_active"][0])),
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(protocol, indent=2, default=json_value), encoding="utf-8"
    )
    print(f"Wrote audit to {output_dir}", flush=True)
    for row in correlation_rows:
        print(
            f"{row['frequency_group']}: pooled rho={row['pooled_spearman_dc_m1']:.4f}, "
            f"class-macro rho={row['macro_class_spearman_dc_m1']:.4f}",
            flush=True,
        )
    for row in pair_rows:
        print(
            f"{row['frequency_group']}: divergent pair rate="
            f"{row['pooled_conditional_divergent_pair_rate']:.3%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
