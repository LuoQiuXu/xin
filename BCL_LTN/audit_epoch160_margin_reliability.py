from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_AUDIT.cifar100lt.data import CIFAR100LT
from BCL_AUDIT.cifar100lt.train import state_dict_sha256
from BCL_LTN.make_noisy_lt import group_class_ids
from BCL_LTN.train_noisy_bcl import OfficialTestAudit, load_verified_noise, safe_torch_load


SCORE_NAMES = (
    "raw_margin",
    "lc_plus_log_prior_margin",
    "inverse_prior_margin",
    "class_normalized_raw_margin",
)
GROUP_NAMES = ("Overall", "Many", "Medium", "Few")
THRESHOLD_QUANTILES = (10, 25, 50, 75, 90)
EXPECTED_CHECKPOINT_SHA256 = "10f0cb41db390f04a9d401433cc121396ab67fad7eb11dee0095c40350d75428"
EXPECTED_MODEL_STATE_SHA256 = "ef1a96e6d018a0dc2f6888b8816643935ab100856b6ad41655371a4ec5498102"
EXPECTED_MANIFEST_SHA256 = "0903d755db1e490cdf7a507f38a2c05ccdb32c7ecbd01d029487ea95205bd5e8"
EXPECTED_INDICES_SHA256 = "be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7"
EXPECTED_NOISY_LABELS_SHA256 = "691b8c8bf5758811d54c33c213e3bf2db12f586aa3d857f37da309a55d03554f"
EXPECTED_TRAIN_SOURCE_SHA256 = {
    "train": "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b",
    "meta": "a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class EvalOnlyTrainDataset(Dataset):
    """TRAIN-only deterministic view. Returns images and positions, never labels."""

    def __init__(self, raw_data: np.ndarray, indices: np.ndarray, transform):
        self.raw_data = raw_data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, position: int):
        source_index = int(self.indices[position])
        image = Image.fromarray(self.raw_data[source_index])
        return self.transform(image), int(position)


def target_margin(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("Expected logits [N,K] and targets [N]")
    if np.any((targets < 0) | (targets >= logits.shape[1])):
        raise ValueError("Observed target outside the logit class range")
    rows = np.arange(len(targets))
    target_logits = logits[rows, targets]
    competitors = logits.copy()
    competitors[rows, targets] = -np.inf
    return target_logits - competitors.max(axis=1)


def build_margin_scores(
    logits: np.ndarray,
    observed_labels: np.ndarray,
    observed_class_counts: np.ndarray,
    eps: float = 1e-8,
):
    """Build the four predeclared scores without reading clean labels or flip masks."""
    logits = np.asarray(logits, dtype=np.float64)
    observed_labels = np.asarray(observed_labels, dtype=np.int64)
    counts = np.asarray(observed_class_counts, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] != observed_labels.size:
        raise ValueError("Logit and observed-label dimensions do not match")
    if counts.shape != (logits.shape[1],) or np.any(counts <= 0):
        raise ValueError("Observed class counts must be a positive K-vector")
    if int(counts.sum()) != len(observed_labels):
        raise ValueError("Observed class counts do not sum to the audit sample count")
    actual_counts = np.bincount(observed_labels, minlength=logits.shape[1])
    if not np.array_equal(actual_counts, counts):
        raise ValueError("Observed class counts do not match the observed labels")

    log_prior = np.log(counts.astype(np.float64) / float(counts.sum()))
    raw = target_margin(logits, observed_labels)
    # Exact LogitCompensatedCE training convention: logits + log(pi).
    lc_plus = target_margin(logits + log_prior[None, :], observed_labels)
    # Reverse-prior diagnostic: logits - log(pi), evaluated as a hypothesis only.
    inverse_prior = target_margin(logits - log_prior[None, :], observed_labels)

    normalized = np.empty_like(raw)
    class_stats = []
    for class_id in range(logits.shape[1]):
        selected = observed_labels == class_id
        values = raw[selected]
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        scale = std + eps
        normalized[selected] = (values - mean) / scale
        class_stats.append({
            "class_id": int(class_id),
            "observed_count": int(selected.sum()),
            "raw_margin_mean": mean,
            "raw_margin_std": std,
            "normalization_scale": float(scale),
            "zero_or_near_zero_std": bool(std < eps),
        })

    scores = {
        "raw_margin": raw,
        "lc_plus_log_prior_margin": lc_plus,
        "inverse_prior_margin": inverse_prior,
        "class_normalized_raw_margin": normalized,
    }
    return scores, log_prior, class_stats


def _tie_group_ends(sorted_scores: np.ndarray) -> np.ndarray:
    if sorted_scores.size == 0:
        return np.empty(0, dtype=np.int64)
    changes = np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])
    return np.concatenate((changes, np.asarray([sorted_scores.size - 1], dtype=np.int64)))


def roc_curve_points(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires at least one clean and one noisy sample")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = _tie_group_ends(sorted_scores)
    cumulative_tp = np.cumsum(sorted_labels)[ends]
    cumulative_fp = (ends + 1) - cumulative_tp
    tpr = np.concatenate(([0.0], cumulative_tp / positives))
    fpr = np.concatenate(([0.0], cumulative_fp / negatives))
    return fpr.astype(np.float64), tpr.astype(np.float64)


def pr_curve_points(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        raise ValueError("PR metrics require both clean and noisy samples")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = _tie_group_ends(sorted_scores)
    cumulative_tp = np.cumsum(sorted_labels)[ends]
    cumulative_fp = (ends + 1) - cumulative_tp
    recall = np.concatenate(([0.0], cumulative_tp / positives))
    precision = np.concatenate(([1.0], cumulative_tp / (cumulative_tp + cumulative_fp)))
    return recall.astype(np.float64), precision.astype(np.float64)


def binary_ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """ROC-AUC (rank statistic), average precision and trapezoidal PR-AUC."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Binary labels and scores must be matching 1-D arrays")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Ranking metrics require both clean and noisy samples")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    starts = np.concatenate(([0], np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1))
    ends = np.concatenate((starts[1:], [scores.size]))
    for start, end in zip(starts, ends):
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
    roc_auc = (
        ranks[labels == 1].sum() - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    recall, precision = pr_curve_points(labels, scores)
    delta_recall = np.diff(recall)
    average_precision = float(np.sum(delta_recall * precision[1:]))
    pr_auc_trapezoid = float(
        np.sum(delta_recall * (precision[1:] + precision[:-1]) * 0.5)
    )
    return {
        "roc_auc": float(roc_auc),
        "average_precision": average_precision,
        "pr_auc_trapezoid": pr_auc_trapezoid,
    }


def stratified_bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    reps: int,
    seed: int,
) -> dict:
    """Percentile CI with clean/noisy counts fixed in every bootstrap replicate."""
    if reps <= 0:
        return {}
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    if positive_scores.size == 0 or negative_scores.size == 0:
        raise ValueError("Stratified bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    samples = {"roc_auc": [], "average_precision": [], "pr_auc_trapezoid": []}
    for _ in range(reps):
        pos = rng.choice(positive_scores, size=positive_scores.size, replace=True)
        neg = rng.choice(negative_scores, size=negative_scores.size, replace=True)
        boot_labels = np.concatenate((
            np.ones(pos.size, dtype=np.int64),
            np.zeros(neg.size, dtype=np.int64),
        ))
        boot_scores = np.concatenate((pos, neg))
        metrics = binary_ranking_metrics(boot_labels, boot_scores)
        for name in samples:
            samples[name].append(metrics[name])
    return {
        name: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for name, values in samples.items()
    }


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    predicted_clean = np.asarray(scores, dtype=np.float64) > float(threshold)
    clean = labels == 1
    noisy = labels == 0
    tp = int(np.sum(predicted_clean & clean))
    fp = int(np.sum(predicted_clean & noisy))
    fn = int(np.sum((~predicted_clean) & clean))
    tn = int(np.sum((~predicted_clean) & noisy))
    return {
        "threshold": float(threshold),
        "predicted_clean": int(predicted_clean.sum()),
        "true_positive_clean": tp,
        "false_positive_noisy_accepted": fp,
        "false_negative_clean_rejected": fn,
        "true_negative_noisy_rejected": tn,
        "precision_clean": float(tp / max(tp + fp, 1)),
        "recall_clean": float(tp / max(tp + fn, 1)),
        "noisy_acceptance_rate": float(fp / max(int(noisy.sum()), 1)),
        "clean_rejection_rate": float(fn / max(int(clean.sum()), 1)),
    }


def group_masks(observed_labels: np.ndarray, clean_class_counts: np.ndarray):
    groups = group_class_ids(np.asarray(clean_class_counts, dtype=np.int64))
    masks = {"Overall": np.ones(len(observed_labels), dtype=np.bool_)}
    for name in ("Many", "Medium", "Few"):
        masks[name] = np.isin(observed_labels, groups[name])
    return masks, groups


def per_class_metrics(
    observed_labels: np.ndarray,
    is_noisy: np.ndarray,
    scores: dict[str, np.ndarray],
):
    rows = []
    for class_id in range(100):
        selected = observed_labels == class_id
        labels = (~is_noisy[selected]).astype(np.int64)
        row = {
            "observed_class_id": int(class_id),
            "observed_count": int(selected.sum()),
            "clean_count": int(labels.sum()),
            "noisy_count": int(labels.size - labels.sum()),
            "clean_prevalence": float(labels.mean()) if labels.size else None,
        }
        for name, values in scores.items():
            if labels.size and labels.sum() > 0 and labels.sum() < labels.size:
                metrics = binary_ranking_metrics(labels, values[selected])
                for metric_name, value in metrics.items():
                    row[f"{name}__{metric_name}"] = value
            else:
                for metric_name in ("roc_auc", "average_precision", "pr_auc_trapezoid"):
                    row[f"{name}__{metric_name}"] = None
        rows.append(row)
    return rows


def summarize_metrics(
    observed_labels: np.ndarray,
    is_noisy: np.ndarray,
    clean_class_counts: np.ndarray,
    scores: dict[str, np.ndarray],
    bootstrap_reps: int,
    bootstrap_seed: int,
):
    masks, class_groups = group_masks(observed_labels, clean_class_counts)
    clean_indicator = (~is_noisy).astype(np.int64)
    metric_rows = []
    threshold_rows = []
    macro_by_group = {}

    for group_index, group_name in enumerate(GROUP_NAMES):
        selected = masks[group_name]
        labels = clean_indicator[selected]
        group_scores = {name: values[selected] for name, values in scores.items()}
        clean_count = int(labels.sum())
        noisy_count = int(labels.size - clean_count)
        group_summary = {
            "samples": int(labels.size),
            "clean_samples": clean_count,
            "noisy_samples": noisy_count,
            "clean_prevalence": float(clean_count / max(labels.size, 1)),
            "classes": int(len(class_groups[group_name])) if group_name != "Overall" else 100,
        }

        for score_index, score_name in enumerate(SCORE_NAMES):
            values = group_scores[score_name]
            metrics = binary_ranking_metrics(labels, values)
            ci = stratified_bootstrap_ci(
                labels, values, bootstrap_reps,
                bootstrap_seed + group_index * 1009 + score_index * 9176,
            )
            metric_rows.append({
                "group": group_name,
                "score": score_name,
                **group_summary,
                **metrics,
                "roc_auc_ci95_low": ci.get("roc_auc", [None, None])[0],
                "roc_auc_ci95_high": ci.get("roc_auc", [None, None])[1],
                "average_precision_ci95_low": ci.get("average_precision", [None, None])[0],
                "average_precision_ci95_high": ci.get("average_precision", [None, None])[1],
                "pr_auc_trapezoid_ci95_low": ci.get("pr_auc_trapezoid", [None, None])[0],
                "pr_auc_trapezoid_ci95_high": ci.get("pr_auc_trapezoid", [None, None])[1],
            })

            operating_points = [("zero", 0.0)]
            operating_points.extend(
                (f"q{q}", float(np.quantile(values, q / 100.0)))
                for q in THRESHOLD_QUANTILES
            )
            for threshold_name, threshold in operating_points:
                threshold_rows.append({
                    "group": group_name,
                    "score": score_name,
                    "threshold_rule": threshold_name,
                    **group_summary,
                    **threshold_metrics(labels, values, threshold),
                })

        class_ids = range(100) if group_name == "Overall" else class_groups[group_name]
        class_rows = []
        for class_id in class_ids:
            class_selected = observed_labels == class_id
            class_labels = clean_indicator[class_selected]
            if class_labels.sum() == 0 or class_labels.sum() == class_labels.size:
                continue
            class_rows.append({
                name: binary_ranking_metrics(class_labels, values[class_selected])
                for name, values in scores.items()
            })
        if not class_rows:
            raise RuntimeError(f"No observed classes with both clean and noisy samples in {group_name}")
        macro_by_group[group_name] = {
            name: {
                metric: float(np.mean([row[name][metric] for row in class_rows]))
                for metric in ("roc_auc", "average_precision", "pr_auc_trapezoid")
            }
            for name in SCORE_NAMES
        }
        macro_by_group[group_name]["classes_with_both_clean_and_noisy"] = len(class_rows)

    return metric_rows, threshold_rows, macro_by_group, masks, class_groups


def write_dict_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields = list(rows[0].keys())
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def plot_distributions(path: Path, masks: dict, scores: dict, is_noisy: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
    for row_index, group_name in enumerate(("Many", "Medium", "Few")):
        selected_group = masks[group_name]
        for col_index, score_name in enumerate(SCORE_NAMES):
            ax = axes[row_index, col_index]
            selected_clean = selected_group & (~is_noisy)
            selected_noisy = selected_group & is_noisy
            ax.hist(scores[score_name][selected_clean], bins=40, density=True,
                    alpha=0.55, label=f"clean n={int(selected_clean.sum())}")
            ax.hist(scores[score_name][selected_noisy], bins=40, density=True,
                    alpha=0.55, label=f"noisy n={int(selected_noisy.sum())}")
            ax.set_title(f"{group_name} / {score_name}", fontsize=8)
            ax.set_xlabel("score", fontsize=8)
            ax.set_ylabel("density", fontsize=8)
            ax.legend(fontsize=7)
    fig.suptitle("Epoch-160 observed-label margin distributions", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc_pr(path: Path, masks: dict, scores: dict, is_noisy: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 13), constrained_layout=True)
    clean_indicator = (~is_noisy).astype(np.int64)
    for row_index, group_name in enumerate(("Many", "Medium", "Few")):
        selected = masks[group_name]
        labels = clean_indicator[selected]
        for score_name in SCORE_NAMES:
            values = scores[score_name][selected]
            fpr, tpr = roc_curve_points(labels, values)
            recall, precision = pr_curve_points(labels, values)
            metrics = binary_ranking_metrics(labels, values)
            axes[row_index, 0].plot(fpr, tpr, label=f"{score_name} AUC={metrics['roc_auc']:.3f}")
            axes[row_index, 1].plot(recall, precision, label=f"{score_name} AP={metrics['average_precision']:.3f}")
        axes[row_index, 0].plot([0, 1], [0, 1], linestyle="--")
        axes[row_index, 0].set_title(f"{group_name} ROC")
        axes[row_index, 0].set_xlabel("false positive rate")
        axes[row_index, 0].set_ylabel("true positive rate")
        axes[row_index, 1].set_title(f"{group_name} Precision–Recall")
        axes[row_index, 1].set_xlabel("recall")
        axes[row_index, 1].set_ylabel("precision")
        axes[row_index, 0].legend(fontsize=6)
        axes[row_index, 1].legend(fontsize=6)
    fig.suptitle("Epoch-160 clean-vs-noisy ranking diagnostics", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_report(summary: dict, metric_rows: list[dict], threshold_rows: list[dict]) -> str:
    lines = [
        "# Epoch-160 Margin Reliability Audit",
        "",
        "## Scope and leakage controls",
        "- Frozen BCL-LTN epoch-160-pre-decay checkpoint; inference only, no optimizer or BN updates.",
        "- Fixed CIFAR-100-LT IF=100 TRAIN subset only; official TEST is not instantiated or read.",
        "- The model receives images only. Observed labels are used after inference to calculate margins.",
        "- Clean labels and the corruption mask are used only after prediction, for offline metric evaluation.",
        "- Group membership is determined by the OBSERVED LABEL's class-frequency group, not the original source class.",
        "",
        "## Score definitions",
        "- raw_margin: z[y_obs] - max(z[k], k != y_obs).",
        "- lc_plus_log_prior_margin: the exact training-loss adjustment z + log(pi), then the observed-label margin.",
        "- inverse_prior_margin: z - log(pi), an alternative scoring hypothesis, not a training change.",
        "- class_normalized_raw_margin: (raw_margin - observed-class mean) / observed-class standard deviation.",
        "- pi is computed from the observed noisy-label class counts used by LC/BCL.",
        "- Class normalization uses all scores in each observed class and never reads clean/noisy status.",
        "",
        "## Observed-label group composition",
        "",
        "| Group | Samples | Clean | Noisy | Clean prevalence |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in GROUP_NAMES:
        row = summary["group_composition"][group]
        lines.append(
            f"| {group} | {row['samples']} | {row['clean_samples']} | {row['noisy_samples']} | "
            f"{100 * row['clean_prevalence']:.2f}% |"
        )
    lines += [
        "",
        "## Pooled ranking metrics",
        "Positive class = unflipped/clean observed label; negative class = flipped/noisy observed label.",
        "ROC-AUC is rank-based. average_precision is step-wise AP; pr_auc_trapezoid is trapezoidal PR area.",
        "Confidence intervals are 95% stratified bootstrap intervals preserving the clean/noisy counts.",
        "",
        "| Group | Score | ROC-AUC [95% CI] | AP [95% CI] | PR-AUC trapezoid |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['group']} | {row['score']} | {row['roc_auc']:.4f} "
            f"[{row['roc_auc_ci95_low'] if row['roc_auc_ci95_low'] is not None else 'NA'}, "
            f"{row['roc_auc_ci95_high'] if row['roc_auc_ci95_high'] is not None else 'NA'}] | "
            f"{row['average_precision']:.4f} "
            f"[{row['average_precision_ci95_low'] if row['average_precision_ci95_low'] is not None else 'NA'}, "
            f"{row['average_precision_ci95_high'] if row['average_precision_ci95_high'] is not None else 'NA'}] | "
            f"{row['pr_auc_trapezoid']:.4f} |"
        )
    lines += [
        "",
        "## Macro per-observed-class ranking metrics",
        "Each observed class is evaluated separately, then metrics are macro-averaged within its frequency group.",
        "",
        "| Group | Score | Macro ROC-AUC | Macro AP | Macro PR-AUC trapezoid | Classes evaluated |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group_name in GROUP_NAMES:
        for score_name in SCORE_NAMES:
            row = summary["macro_per_observed_class_metrics"][group_name][score_name]
            count = summary["macro_per_observed_class_metrics"][group_name]["classes_with_both_clean_and_noisy"]
            lines.append(
                f"| {group_name} | {score_name} | {row['roc_auc']:.4f} | "
                f"{row['average_precision']:.4f} | {row['pr_auc_trapezoid']:.4f} | {count} |"
            )
    lines += [
        "",
        "## Threshold operating points",
        "Thresholds use score > threshold at zero and at quantiles of the unlabeled score distribution; no clean/noisy labels select a threshold.",
        "",
        "| Group | Score | Rule | Threshold | Precision(clean) | Recall(clean) | Noisy acceptance |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in threshold_rows:
        lines.append(
            f"| {row['group']} | {row['score']} | {row['threshold_rule']} | {row['threshold']:.5f} | "
            f"{row['precision_clean']:.4f} | {row['recall_clean']:.4f} | "
            f"{row['noisy_acceptance_rate']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "- AUC measures ranking, not whether a usable fixed threshold exists.",
        "- The class-normalized score is a positive affine transform within each observed class, so within-class ROC-AUC/AP must match raw margin. Pooled group metrics may differ because classes are rescaled relative to one another.",
        "- z + log(pi) is included only as a diagnostic of the training convention; z - log(pi) is a competing hypothesis. Neither is assumed superior.",
        "- Do not use clean labels, flip masks, AUC, or PR-AUC to choose a threshold for a later training run.",
        "- This audit does not change BCL, does not train a noise filter, and does not establish real-world label-noise robustness.",
        "",
        "## Provenance",
        f"- Checkpoint SHA256: {summary['checkpoint_sha256']}",
        f"- Checkpoint model-state SHA256: {summary['model_state_sha256']}",
        f"- Noise-manifest SHA256: {summary['noise_manifest_sha256']}",
        f"- Fixed indices SHA256: {summary['indices_sha256']}",
        f"- Inference seconds: {summary['inference_seconds']:.2f}",
        f"- Device: {summary['device']}",
        f"- TEST file opens: {summary['official_test_opens']} (denied: {summary['official_test_denied']})",
        "",
    ]
    return "\n".join(lines)


def run_audit(args):
    data_root = args.data_root.expanduser().resolve()
    indices_path = args.indices_file.expanduser().resolve()
    noise_dir = args.noise_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = (args.checkpoint or (run_dir / "checkpoints" / "epoch_160_pre_decay.pth")).expanduser().resolve()
    training_summary_path = (args.training_summary or (run_dir / "training_summary.json")).expanduser().resolve()
    diagnostic_path = run_dir / "train_diagnostics" / "epoch_160.json"
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Epoch-160 checkpoint not found: {checkpoint_path}")
    if not training_summary_path.is_file():
        raise FileNotFoundError(f"Training summary not found: {training_summary_path}")
    if not diagnostic_path.is_file():
        raise FileNotFoundError(f"Epoch-160 TRAIN diagnostic not found: {diagnostic_path}")
    if args.batch_size <= 0 or args.num_workers != 0 or args.bootstrap_reps < 0:
        raise ValueError("Require positive batch size, num_workers=0 and bootstrap_reps>=0")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu only for a fallback audit")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")

    # Load only protocol metadata and observed noisy labels before inference.
    # Clean labels and the flip mask are deliberately deferred until predictions are fixed.
    test_guard = OfficialTestAudit(data_root)
    manifest_path = noise_dir / "noise_manifest.json"
    manifest = read_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Noise-manifest SHA256 differs from the locked experiment receipt")
    indices_sha = sha256_file(indices_path)
    if indices_sha != EXPECTED_INDICES_SHA256 or indices_sha != manifest.get("indices_sha256"):
        raise RuntimeError("Fixed IF=100 index SHA256 differs from the locked receipt/manifest")
    if (manifest.get("seed") != 42 or manifest.get("noise_rate_requested") != 0.4
            or manifest.get("num_train_samples") != 10_847 or manifest.get("num_classes") != 100
            or manifest.get("num_flipped") != 4_379 or manifest.get("num_unflipped") != 6_468):
        raise RuntimeError("Noise manifest does not match the locked NR=0.4 seed-42 protocol")
    if manifest.get("train_source_sha256") != EXPECTED_TRAIN_SOURCE_SHA256:
        raise RuntimeError("TRAIN/meta source hashes differ from the locked data receipt")
    train_source_dir = data_root / "cifar-100-python"
    for source_name, expected_sha in EXPECTED_TRAIN_SOURCE_SHA256.items():
        if sha256_file(train_source_dir / source_name) != expected_sha:
            raise RuntimeError(f"CIFAR-100 TRAIN source hash mismatch: {source_name}")
    noisy_path = noise_dir / manifest.get("noisy_labels_file", "")
    if not noisy_path.is_file():
        raise RuntimeError("Observed noisy-label file is missing")
    noisy_sha = sha256_file(noisy_path)
    if noisy_sha != EXPECTED_NOISY_LABELS_SHA256 or noisy_sha != manifest.get("noisy_labels_sha256"):
        raise RuntimeError("Observed noisy-label SHA256 differs from the locked receipt/manifest")
    noisy_labels = np.load(noisy_path, allow_pickle=False)
    if noisy_labels.shape != (10_847,) or noisy_labels.dtype != np.int64:
        raise RuntimeError("Observed noisy-label array must be int64 with exactly 10,847 entries")
    if np.any((noisy_labels < 0) | (noisy_labels >= 100)):
        raise RuntimeError("Observed noisy labels are outside [0,99]")
    observed_counts = np.bincount(noisy_labels, minlength=100).astype(np.int64)
    if observed_counts.tolist() != manifest.get("observed_noisy_class_counts"):
        raise RuntimeError("Observed noisy-label counts differ from the noise manifest")
    clean_counts = np.asarray(manifest["clean_class_counts"], dtype=np.int64)

    # The dataset object supplies TRAIN pixels, fixed indices and the eval transform.
    # EvalOnlyTrainDataset returns no labels, and the model receives only images.
    base = CIFAR100LT(
        root=data_root, mode="bcl", indices_file=indices_path,
        split_seed=0, download=False,
    )
    if len(base.indices) != 10_847 or np.unique(base.indices).size != 10_847:
        raise RuntimeError("Fixed IF=100 TRAIN indices are not a unique 10,847-element vector")
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Epoch-160 checkpoint SHA256 differs from the locked experiment receipt")
    training_summary = read_json(training_summary_path)
    diagnostic = read_json(diagnostic_path)
    checkpoint_inventory_sha = training_summary.get("checkpoint_sha256", {}).get(checkpoint_path.name)
    if checkpoint_inventory_sha != checkpoint_sha:
        raise RuntimeError("Checkpoint SHA256 does not match the committed training_summary receipt")

    checkpoint = safe_torch_load(checkpoint_path, torch.device("cpu"))
    if checkpoint.get("epoch") != 160 or checkpoint.get("seed") != 42:
        raise RuntimeError("Checkpoint is not the locked epoch-160 seed-42 checkpoint")
    if checkpoint.get("indices_sha256") != manifest.get("indices_sha256"):
        raise RuntimeError("Checkpoint and noise manifest use different fixed indices")
    if checkpoint.get("noise_manifest_sha256") != manifest_sha:
        raise RuntimeError("Checkpoint and local noise manifest SHA256 differ")
    observed_counts = np.bincount(noisy_labels, minlength=100).astype(np.int64)
    clean_counts = np.asarray(manifest["clean_class_counts"], dtype=np.int64)
    if checkpoint.get("class_counts_for_lc_and_bcl") != observed_counts.tolist():
        raise RuntimeError("Checkpoint LC/BCL class counts do not match observed noisy labels")
    if (checkpoint.get("model_state_sha256") != EXPECTED_MODEL_STATE_SHA256
            or checkpoint.get("model_state_sha256") != diagnostic.get("model_state_sha256")):
        raise RuntimeError("Checkpoint model state does not match the locked epoch-160 diagnostic")
    protocol = checkpoint.get("protocol", {})
    if protocol.get("precision") != "fp32" or protocol.get("amp") is not False or protocol.get("tf32") is not False:
        raise RuntimeError("Unexpected checkpoint precision/AMP/TF32 protocol")

    model = BCLCIFARModel()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Frozen audit model must remain FP32")
    model_hash_before = state_dict_sha256(model.state_dict())
    if model_hash_before != checkpoint["model_state_sha256"]:
        raise RuntimeError("Loaded model state SHA256 differs from the checkpoint receipt")
    model = model.to(device)
    model.eval()

    eval_set = EvalOnlyTrainDataset(base.data, base.indices, base.transform_eval)
    loader = DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    logits = np.empty((len(eval_set), 100), dtype=np.float32)
    seen = np.zeros(len(eval_set), dtype=np.bool_)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, positions in loader:
            images = images.to(device, non_blocking=True)
            if images.dtype != torch.float32:
                raise RuntimeError("Eval transform must provide FP32 image tensors")
            _, batch_logits, _, _ = model(images, compute_contrastive=False)
            if not bool(torch.isfinite(batch_logits).all()):
                raise RuntimeError("Nonfinite logits during epoch-160 TRAIN-only audit")
            locations = positions.numpy().astype(np.int64, copy=False)
            if seen[locations].any():
                raise RuntimeError("Duplicate sample positions during TRAIN-only inference")
            logits[locations] = batch_logits.detach().cpu().numpy()
            seen[locations] = True
    if not seen.all():
        raise RuntimeError("TRAIN-only inference did not cover every fixed IF=100 sample")
    inference_seconds = time.perf_counter() - started
    model_hash_after = state_dict_sha256(model.state_dict())
    if model_hash_after != model_hash_before:
        raise RuntimeError("Frozen model state changed during inference")
    if test_guard.opens != 0 or test_guard.denied != 0:
        raise RuntimeError("Official TEST was accessed during the margin audit")

    # Only now load clean labels and the corruption mask, strictly for offline evaluation.
    verified_base, verified_manifest, verified_noisy, clean_labels, is_noisy = load_verified_noise(
        data_root, indices_path, noise_dir
    )
    if (not np.array_equal(verified_noisy, noisy_labels)
            or not np.array_equal(verified_base.indices, base.indices)
            or sha256_file(manifest_path) != manifest_sha):
        raise RuntimeError("Post-inference label-truth verification differs from pre-inference inputs")

    scores, log_prior, class_stats = build_margin_scores(logits, noisy_labels, observed_counts)
    metric_rows, threshold_rows, macro_by_group, masks, class_groups = summarize_metrics(
        noisy_labels, is_noisy, clean_counts, scores,
        args.bootstrap_reps, args.bootstrap_seed,
    )

    group_composition = {}
    manifest_groups = manifest["many_medium_few"]
    for group_name in GROUP_NAMES:
        selected = masks[group_name]
        clean_n = int(np.sum(selected & (~is_noisy)))
        noisy_n = int(np.sum(selected & is_noisy))
        group_composition[group_name] = {
            "samples": int(selected.sum()),
            "clean_samples": clean_n,
            "noisy_samples": noisy_n,
            "clean_prevalence": float(clean_n / max(int(selected.sum()), 1)),
        }
        if group_name != "Overall":
            expected_total = int(manifest_groups[group_name]["observed_noisy_label_samples"])
            expected_clean = int(manifest_groups[group_name]["clean_source_samples_unflipped"])
            if group_composition[group_name]["samples"] != expected_total or clean_n != expected_clean:
                raise RuntimeError(f"{group_name} observed-label bucket differs from the noise manifest")

    per_class = per_class_metrics(noisy_labels, is_noisy, scores)
    for row in per_class:
        for metric_name in ("roc_auc", "average_precision", "pr_auc_trapezoid"):
            raw_value = row[f"raw_margin__{metric_name}"]
            normalized_value = row[f"class_normalized_raw_margin__{metric_name}"]
            if (raw_value is not None and normalized_value is not None
                    and not math.isclose(raw_value, normalized_value, abs_tol=1e-12)):
                raise RuntimeError(
                    f"Class normalization unexpectedly changed within-class {metric_name}"
                )

    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "group_metrics.csv"
    thresholds_path = output_dir / "threshold_metrics.csv"
    per_class_path = output_dir / "per_observed_class_metrics.csv"
    write_dict_csv(metrics_path, metric_rows)
    write_dict_csv(thresholds_path, threshold_rows)
    write_dict_csv(per_class_path, per_class)

    sample_path = output_dir / "sample_scores.csv"
    with sample_path.open("x", newline="", encoding="utf-8-sig") as stream:
        fields = [
            "position", "original_train_index", "observed_label", "clean_label",
            "is_clean", "observed_label_group", *SCORE_NAMES,
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for position in range(len(noisy_labels)):
            observed = int(noisy_labels[position])
            group_name = next(
                name for name in ("Many", "Medium", "Few")
                if observed in class_groups[name]
            )
            writer.writerow({
                "position": position,
                "original_train_index": int(base.indices[position]),
                "observed_label": observed,
                "clean_label": int(clean_labels[position]),
                "is_clean": int(not is_noisy[position]),
                "observed_label_group": group_name,
                **{name: float(values[position]) for name, values in scores.items()},
            })

    plot_distributions(output_dir / "margin_distributions.png", masks, scores, is_noisy)
    plot_roc_pr(output_dir / "roc_pr_curves.png", masks, scores, is_noisy)

    summary = {
        "audit": "epoch160_margin_reliability",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "model_state_sha256": model_hash_before,
        "noise_manifest_sha256": manifest_sha,
        "indices_sha256": manifest["indices_sha256"],
        "noisy_labels_sha256": manifest["noisy_labels_sha256"],
        "clean_labels_sha256_offline_only": manifest["clean_labels_sha256"],
        "is_noisy_sha256_offline_only": manifest["is_noisy_sha256"],
        "epoch": 160,
        "seed": 42,
        "device": str(device),
        "precision": "fp32",
        "amp": False,
        "tf32": False,
        "batch_size": int(args.batch_size),
        "num_workers": 0,
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.bootstrap_seed),
        "samples_inferred": int(seen.sum()),
        "inference_seconds": float(inference_seconds),
        "model_state_unchanged": model_hash_after == model_hash_before,
        "official_test_opens": int(test_guard.opens),
        "official_test_denied": int(test_guard.denied),
        "log_prior_definition": "log(observed_noisy_class_count / 10847)",
        "log_prior": log_prior.tolist(),
        "score_definitions": {
            "raw_margin": "z[y_obs] - max_{k!=y_obs} z[k]",
            "lc_plus_log_prior_margin": "margin computed from z + log(pi), matching LogitCompensatedCE training convention",
            "inverse_prior_margin": "margin computed from z - log(pi), diagnostic hypothesis only",
            "class_normalized_raw_margin": "(raw_margin - observed-class mean) / observed-class std; computed without clean/noisy status",
        },
        "class_normalization_stats": class_stats,
        "group_composition": group_composition,
        "pooled_metrics": metric_rows,
        "macro_per_observed_class_metrics": macro_by_group,
        "threshold_metrics": threshold_rows,
        "output_files": {
            "group_metrics": str(metrics_path),
            "threshold_metrics": str(thresholds_path),
            "per_observed_class_metrics": str(per_class_path),
            "sample_scores_offline_only": str(sample_path),
            "margin_distributions": str(output_dir / "margin_distributions.png"),
            "roc_pr_curves": str(output_dir / "roc_pr_curves.png"),
        },
        "guardrails": [
            "No training, optimizer, scheduler, or BatchNorm update was performed.",
            "No official TEST data was instantiated or read.",
            "Clean labels and the flip mask were used only after inference for offline evaluation.",
            "Group membership is based on observed labels, not original clean source labels.",
            "No threshold was selected using clean/noisy truth.",
        ],
    }
    write_json(output_dir / "audit_summary.json", summary)
    report = build_report(summary, metric_rows, threshold_rows)
    (output_dir / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "passed": True,
        "branch_scope": "BCL_LTN only; BCL_AUDIT read-only",
        "checkpoint_sha256": checkpoint_sha,
        "model_state_sha256": model_hash_before,
        "samples_inferred": int(seen.sum()),
        "inference_seconds": round(inference_seconds, 2),
        "Few_observed_bucket": group_composition["Few"],
        "official_test_opens": test_guard.opens,
        "output_dir": str(output_dir),
        "metrics": str(metrics_path),
        "report": str(output_dir / "AUDIT_REPORT.md"),
    }, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Frozen epoch-160 BCL-LTN margin reliability audit (TRAIN only)"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--indices-file", type=Path, required=True)
    parser.add_argument("--noise-dir", type=Path, default=Path("BCL_LTN/data"))
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("BCL_LTN/outputs/BCL_sym04_seed42/BCL_sym04_seed42"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--training-summary", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("BCL_LTN/outputs/epoch160_margin_reliability_audit_20261124"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-reps", type=int, default=300)
    parser.add_argument("--bootstrap-seed", type=int, default=20261124)
    return parser.parse_args(argv)


def main():
    run_audit(parse_args())


if __name__ == "__main__":
    main()
