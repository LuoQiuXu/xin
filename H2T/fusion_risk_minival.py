#!/usr/bin/env python
"""Offline minimum validation for boundary-controlled feature fusion.

The script deliberately does not train a backbone or the reference classifier.
It compares three feature-fusion ratios on fixed recipient/donor pairs:

1. PI-H2T center ratio (the mapping used by ``H2Tv2`` in ``methods.py``),
2. recipient-only boundary ratio, distribution-matched to the PI-H2T ratio,
3. donor-conditioned closed-form boundary ratio.

Harm is measured by an *independent* frozen probe.  Reusing the reference head
as the probe would make the donor-conditioned result circular, so the script
rejects identical reference/probe heads unless explicitly overridden.

Input ``.npz`` schema
---------------------
Required arrays:

``recipient_features``: [N, D]
``recipient_labels``:   [N]
``donor_features``:     [N, D]
``reference_weight``:   [C, D] (or [D, C])

One of the following probe specifications is also required:

``probe_weight``:       [C, D] (or [D, C]), with optional ``probe_bias``

or

``probe_train_features``: [M, D]
``probe_train_labels``:   [M]

Optional arrays:

``reference_bias``: [C]
``class_centers``:  [C, D] (defaults to reference classifier weights)
``donor_labels``:   [N]

Example:

    python fusion_risk_minival.py --input fixed_pairs.npz \
        --output-dir outputs/fusion_risk_minival

Run ``--synthetic`` first to smoke-test the complete evaluation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


METHOD_CENTER = "pi_h2t_center"
METHOD_RECIPIENT = "recipient_boundary"
METHOD_DONOR = "donor_conditioned"


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def orient_weight(weight: np.ndarray, feature_dim: int, name: str) -> np.ndarray:
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {weight.shape}")
    if weight.shape[1] == feature_dim:
        return weight
    if weight.shape[0] == feature_dim:
        return weight.T
    raise ValueError(
        f"cannot orient {name} with shape {weight.shape} for feature dim {feature_dim}"
    )


def validate_labels(labels: np.ndarray, num_classes: int, name: str) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError(f"{name} is empty")
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"{name} must lie in [0, {num_classes - 1}], got "
            f"[{labels.min()}, {labels.max()}]"
        )
    return labels


def get_bias(data: dict[str, np.ndarray], key: str, num_classes: int) -> np.ndarray:
    if key not in data:
        return np.zeros(num_classes, dtype=np.float64)
    bias = np.asarray(data[key], dtype=np.float64).reshape(-1)
    if bias.shape != (num_classes,):
        raise ValueError(f"{key} must have shape ({num_classes},), got {bias.shape}")
    return bias


def fit_balanced_probe(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = validate_labels(labels, num_classes, "probe_train_labels")
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("probe_train_features and probe_train_labels have incompatible shapes")
    present = np.unique(labels)
    if present.size != num_classes:
        missing = np.setdiff1d(np.arange(num_classes), present)
        raise ValueError(f"probe training data is missing classes: {missing.tolist()}")

    probe = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed,
    )
    probe.fit(np.asarray(features, dtype=np.float64), labels)
    if not np.array_equal(probe.classes_, np.arange(num_classes)):
        raise ValueError("probe class order is not contiguous [0, C)")
    return probe.coef_.astype(np.float64), probe.intercept_.astype(np.float64)


def load_npz(
    path: Path,
    probe_max_iter: int,
    seed: int,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    required = [
        "recipient_features",
        "recipient_labels",
        "donor_features",
        "reference_weight",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"input archive is missing required arrays: {missing}")

    recipient = np.asarray(data["recipient_features"], dtype=np.float64)
    donor = np.asarray(data["donor_features"], dtype=np.float64)
    if recipient.ndim != 2 or donor.shape != recipient.shape:
        raise ValueError(
            "recipient_features and donor_features must both have shape [N, D], "
            f"got {recipient.shape} and {donor.shape}"
        )

    reference_weight = orient_weight(
        data["reference_weight"], recipient.shape[1], "reference_weight"
    )
    num_classes = reference_weight.shape[0]
    labels = validate_labels(data["recipient_labels"], num_classes, "recipient_labels")
    if labels.shape[0] != recipient.shape[0]:
        raise ValueError("recipient_labels length does not match recipient_features")

    reference_bias = get_bias(data, "reference_bias", num_classes)

    if "probe_weight" in data:
        probe_weight = orient_weight(data["probe_weight"], recipient.shape[1], "probe_weight")
        if probe_weight.shape[0] != num_classes:
            raise ValueError("probe and reference classifiers have different class counts")
        probe_bias = get_bias(data, "probe_bias", num_classes)
    elif "probe_train_features" in data and "probe_train_labels" in data:
        probe_weight, probe_bias = fit_balanced_probe(
            np.asarray(data["probe_train_features"], dtype=np.float64),
            data["probe_train_labels"],
            num_classes,
            probe_max_iter,
            seed,
        )
    else:
        raise ValueError(
            "an independent probe is required: provide probe_weight (and optional "
            "probe_bias), or probe_train_features plus probe_train_labels"
        )

    if "class_centers" in data:
        centers = orient_weight(data["class_centers"], recipient.shape[1], "class_centers")
        if centers.shape[0] != num_classes:
            raise ValueError("class_centers and reference_weight have different class counts")
    else:
        centers = reference_weight.copy()

    result = {
        "recipient_features": recipient,
        "recipient_labels": labels,
        "donor_features": donor,
        "reference_weight": reference_weight,
        "reference_bias": reference_bias,
        "probe_weight": probe_weight,
        "probe_bias": probe_bias,
        "class_centers": centers,
    }
    if "donor_labels" in data:
        donor_labels = validate_labels(data["donor_labels"], num_classes, "donor_labels")
        if donor_labels.shape[0] != recipient.shape[0]:
            raise ValueError("donor_labels length does not match donor_features")
        result["donor_labels"] = donor_labels
    return result


def make_synthetic(seed: int, n: int = 12000, c: int = 10, d: int = 32) -> dict[str, np.ndarray]:
    """Create a directional-harm toy problem for an end-to-end smoke test."""
    rng = np.random.default_rng(seed)
    semantic_weight = rng.normal(size=(c, d))
    semantic_weight /= np.linalg.norm(semantic_weight, axis=1, keepdims=True)

    labels = rng.integers(0, c, size=n)
    recipient_strength = rng.uniform(0.55, 2.2, size=(n, 1))
    recipient = recipient_strength * semantic_weight[labels] + rng.normal(0.0, 0.55, (n, d))

    # Long-tailed donor prior: low-index classes are sampled more often.
    donor_prior = np.geomspace(1.0, 0.08, c)
    donor_prior /= donor_prior.sum()
    donor_labels = rng.choice(c, size=n, p=donor_prior)
    donor = 2.1 * semantic_weight[donor_labels] + rng.normal(0.0, 0.45, (n, d))

    reference_weight = semantic_weight + rng.normal(0.0, 0.025, (c, d))
    probe_weight = semantic_weight + rng.normal(0.0, 0.035, (c, d))
    reference_bias = rng.normal(0.0, 0.02, c)
    probe_bias = rng.normal(0.0, 0.02, c)
    return {
        "recipient_features": recipient,
        "recipient_labels": labels,
        "donor_features": donor,
        "donor_labels": donor_labels,
        "reference_weight": reference_weight,
        "reference_bias": reference_bias,
        "probe_weight": probe_weight,
        "probe_bias": probe_bias,
        "class_centers": reference_weight.copy(),
    }


def logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return features @ weight.T + bias


def cross_entropy_per_sample(class_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shifted = class_logits - class_logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    return logsumexp - shifted[np.arange(labels.size), labels]


def cosine_center_distance(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    eps: float,
) -> np.ndarray:
    selected = centers[labels]
    numerator = np.einsum("nd,nd->n", features, selected)
    denominator = np.linalg.norm(features, axis=1) * np.linalg.norm(selected, axis=1)
    cosine = numerator / np.maximum(denominator, eps)
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def pi_h2t_donor_ratio(center_distance: np.ndarray, lambda_cap: float) -> np.ndarray:
    """PI-H2T Eq. (7)-(8), expressed as donor rather than recipient mass.

    PI-H2T uses cosine distance to the ground-truth classifier weight and
    linearly normalizes it with the theoretical cosine-distance bounds [0, 2].
    Its ``r`` is recipient mass, whereas this evaluator consistently denotes
    donor mass by lambda = 1 - r.
    """
    recipient_ratio = np.clip(center_distance / 2.0, 0.0, 1.0)
    return np.clip(1.0 - recipient_ratio, 0.0, lambda_cap)


def pairwise_boundary_distances(
    features: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Signed normalized distances from class y to every y-vs-j hyperplane."""
    class_logits = logits(features, weight, bias)
    numerator = class_logits[np.arange(labels.size), labels, None] - class_logits

    weight_sq = np.einsum("cd,cd->c", weight, weight)
    distance_sq = weight_sq[:, None] + weight_sq[None, :] - 2.0 * (weight @ weight.T)
    denominator = np.sqrt(np.maximum(distance_sq, 0.0))
    denominator_for_rows = denominator[labels]
    result = numerator / np.maximum(denominator_for_rows, eps)
    result[np.arange(labels.size), labels] = np.inf
    return result


def donor_conditioned_critical_ratio(
    recipient_distance: np.ndarray,
    donor_distance: np.ndarray,
    delta: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form maximal ratio before exceeding the recipient-specific budget."""
    recipient_margin = np.min(recipient_distance, axis=1)
    threshold = np.minimum(recipient_margin, 0.0) - delta
    valid = np.isfinite(recipient_distance) & np.isfinite(donor_distance)
    decrease = np.zeros_like(recipient_distance)
    np.subtract(recipient_distance, donor_distance, out=decrease, where=valid)
    dangerous = valid & (decrease > eps)

    bounds = np.full_like(recipient_distance, np.inf)
    threshold_matrix = threshold[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_bounds = (recipient_distance - threshold_matrix) / decrease
    bounds[dangerous] = raw_bounds[dangerous]
    critical = np.min(bounds, axis=1)
    critical[~np.isfinite(critical)] = 1.0
    critical = np.clip(critical, 0.0, 1.0)
    return critical, recipient_margin, threshold


def distribution_matched_recipient_ratio(
    recipient_safety: np.ndarray,
    target_ratio: np.ndarray,
) -> np.ndarray:
    """Map safer recipients to more donor mass while matching PI-H2T's ratio scale."""
    quantiles = (rankdata(recipient_safety, method="average") - 0.5) / recipient_safety.size
    return np.quantile(target_ratio, quantiles)


def fused_features(
    recipient: np.ndarray,
    donor: np.ndarray,
    donor_ratio: np.ndarray | float,
) -> np.ndarray:
    ratio = np.asarray(donor_ratio, dtype=np.float64)
    if ratio.ndim == 0:
        ratio = np.full(recipient.shape[0], float(ratio), dtype=np.float64)
    return (1.0 - ratio[:, None]) * recipient + ratio[:, None] * donor


def safe_spearman(score: np.ndarray, target: np.ndarray) -> float:
    if score.size < 2 or np.all(score == score[0]) or np.all(target == target[0]):
        return math.nan
    value = spearmanr(score, target).statistic
    return float(value) if np.isfinite(value) else math.nan


def bootstrap_binary_metrics(
    labels: np.ndarray,
    score: np.ndarray,
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if labels.size == 0 or np.unique(labels).size < 2:
        return {
            "roc_auc": math.nan,
            "roc_auc_low": math.nan,
            "roc_auc_high": math.nan,
            "average_precision": math.nan,
            "average_precision_low": math.nan,
            "average_precision_high": math.nan,
        }

    roc = float(roc_auc_score(labels, score))
    ap = float(average_precision_score(labels, score))
    if bootstrap <= 0:
        return {
            "roc_auc": roc,
            "roc_auc_low": math.nan,
            "roc_auc_high": math.nan,
            "average_precision": ap,
            "average_precision_low": math.nan,
            "average_precision_high": math.nan,
        }

    roc_samples: list[float] = []
    ap_samples: list[float] = []
    for _ in range(bootstrap):
        index = rng.integers(0, labels.size, size=labels.size)
        sampled_labels = labels[index]
        if np.unique(sampled_labels).size < 2:
            continue
        sampled_score = score[index]
        roc_samples.append(float(roc_auc_score(sampled_labels, sampled_score)))
        ap_samples.append(float(average_precision_score(sampled_labels, sampled_score)))
    if not roc_samples:
        return {
            "roc_auc": roc,
            "roc_auc_low": math.nan,
            "roc_auc_high": math.nan,
            "average_precision": ap,
            "average_precision_low": math.nan,
            "average_precision_high": math.nan,
        }
    return {
        "roc_auc": roc,
        "roc_auc_low": float(np.quantile(roc_samples, 0.025)),
        "roc_auc_high": float(np.quantile(roc_samples, 0.975)),
        "average_precision": ap,
        "average_precision_low": float(np.quantile(ap_samples, 0.025)),
        "average_precision_high": float(np.quantile(ap_samples, 0.975)),
    }


def prediction_rows(
    recipient: np.ndarray,
    donor: np.ndarray,
    labels: np.ndarray,
    probe_weight: np.ndarray,
    probe_bias: np.ndarray,
    candidate_lambdas: Iterable[float],
    center_ratio: np.ndarray,
    recipient_ratio: np.ndarray,
    critical_ratio: np.ndarray,
    loss_harm_threshold: float,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    before_logits = logits(recipient, probe_weight, probe_bias)
    before_prediction = before_logits.argmax(axis=1)
    before_correct = before_prediction == labels
    before_loss = cross_entropy_per_sample(before_logits, labels)

    rows: list[dict[str, Any]] = []
    for candidate in candidate_lambdas:
        candidate_features = fused_features(recipient, donor, candidate)
        after_logits = logits(candidate_features, probe_weight, probe_bias)
        after_prediction = after_logits.argmax(axis=1)
        after_loss = cross_entropy_per_sample(after_logits, labels)
        loss_delta = after_loss - before_loss

        method_scores = {
            # For all three methods, risk is the amount by which the proposed
            # candidate exceeds that method's pairwise donor allowance.
            METHOD_CENTER: candidate - center_ratio,
            METHOD_RECIPIENT: candidate - recipient_ratio,
            # Positive means the candidate exceeds the closed-form safe ratio.
            METHOD_DONOR: candidate - critical_ratio,
        }

        flip_mask = before_correct
        flip_labels = (after_prediction[flip_mask] != labels[flip_mask]).astype(np.int64)
        loss_labels = (loss_delta > loss_harm_threshold).astype(np.int64)

        for method, score in method_scores.items():
            flip_metrics = bootstrap_binary_metrics(
                flip_labels, score[flip_mask], bootstrap, rng
            )
            rows.append(
                {
                    "candidate_lambda": candidate,
                    "harm_definition": "probe_correct_to_wrong",
                    "method": method,
                    "n": int(flip_labels.size),
                    "harm_n": int(flip_labels.sum()),
                    "harm_prevalence": float(flip_labels.mean()) if flip_labels.size else math.nan,
                    **flip_metrics,
                    "spearman_probe_loss_delta": safe_spearman(score[flip_mask], loss_delta[flip_mask]),
                }
            )

            loss_metrics = bootstrap_binary_metrics(loss_labels, score, bootstrap, rng)
            rows.append(
                {
                    "candidate_lambda": candidate,
                    "harm_definition": f"probe_loss_increase_gt_{loss_harm_threshold:g}",
                    "method": method,
                    "n": int(loss_labels.size),
                    "harm_n": int(loss_labels.sum()),
                    "harm_prevalence": float(loss_labels.mean()),
                    **loss_metrics,
                    "spearman_probe_loss_delta": safe_spearman(score, loss_delta),
                }
            )
    return rows


def evaluate_control(
    recipient: np.ndarray,
    donor: np.ndarray,
    labels: np.ndarray,
    ratio: np.ndarray,
    probe_weight: np.ndarray,
    probe_bias: np.ndarray,
    before_prediction: np.ndarray,
    before_loss: np.ndarray,
) -> dict[str, float]:
    after_logits = logits(fused_features(recipient, donor, ratio), probe_weight, probe_bias)
    after_prediction = after_logits.argmax(axis=1)
    after_loss = cross_entropy_per_sample(after_logits, labels)
    before_correct = before_prediction == labels
    before_wrong = ~before_correct
    loss_delta = after_loss - before_loss
    return {
        "mean_lambda": float(ratio.mean()),
        "std_lambda": float(ratio.std()),
        "active_pair_rate": float((ratio > 1e-12).mean()),
        "probe_accuracy": float((after_prediction == labels).mean()),
        "harmful_flip_rate": (
            float((after_prediction[before_correct] != labels[before_correct]).mean())
            if before_correct.any()
            else math.nan
        ),
        "beneficial_repair_rate": (
            float((after_prediction[before_wrong] == labels[before_wrong]).mean())
            if before_wrong.any()
            else math.nan
        ),
        "mean_probe_loss_delta": float(loss_delta.mean()),
        "probe_loss_increase_rate": float((loss_delta > 0.0).mean()),
    }


def control_rows(
    recipient: np.ndarray,
    donor: np.ndarray,
    labels: np.ndarray,
    probe_weight: np.ndarray,
    probe_bias: np.ndarray,
    center_ratio: np.ndarray,
    recipient_ratio: np.ndarray,
    critical_ratio: np.ndarray,
    lambda_cap: float,
    aggressiveness_grid: Iterable[float],
) -> list[dict[str, Any]]:
    before_logits = logits(recipient, probe_weight, probe_bias)
    before_prediction = before_logits.argmax(axis=1)
    before_loss = cross_entropy_per_sample(before_logits, labels)
    donor_base_ratio = np.minimum(critical_ratio, lambda_cap)
    rows: list[dict[str, Any]] = []

    for aggressiveness in aggressiveness_grid:
        method_ratios = {
            METHOD_CENTER: np.clip(aggressiveness * center_ratio, 0.0, lambda_cap),
            METHOD_RECIPIENT: np.clip(aggressiveness * recipient_ratio, 0.0, lambda_cap),
            # Apply the same global intensity scaling to every base policy.  This
            # preserves the pairwise ordering encoded by the closed-form ratio.
            METHOD_DONOR: np.clip(
                aggressiveness * donor_base_ratio, 0.0, lambda_cap
            ),
        }
        for method, ratio in method_ratios.items():
            rows.append(
                {
                    "aggressiveness": aggressiveness,
                    "method": method,
                    **evaluate_control(
                        recipient,
                        donor,
                        labels,
                        ratio,
                        probe_weight,
                        probe_bias,
                        before_prediction,
                        before_loss,
                    ),
                }
            )
    return rows


def interpolate_control(
    rows: list[dict[str, Any]], budgets: list[float] | None
) -> tuple[list[dict[str, Any]], float]:
    methods = [METHOD_CENTER, METHOD_RECIPIENT, METHOD_DONOR]
    curves: dict[str, list[dict[str, Any]]] = {
        method: sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["mean_lambda"],
        )
        for method in methods
    }
    common_max = min(curve[-1]["mean_lambda"] for curve in curves.values())
    if budgets is None:
        budgets = [common_max * fraction for fraction in (0.25, 0.5, 0.75, 1.0)]
    budgets = [value for value in budgets if 0.0 <= value <= common_max + 1e-12]

    metric_names = [
        "probe_accuracy",
        "harmful_flip_rate",
        "beneficial_repair_rate",
        "mean_probe_loss_delta",
        "probe_loss_increase_rate",
        "std_lambda",
        "active_pair_rate",
    ]
    matched: list[dict[str, Any]] = []
    for budget in budgets:
        for method, curve in curves.items():
            x = np.asarray([row["mean_lambda"] for row in curve])
            result: dict[str, Any] = {"target_mean_lambda": budget, "method": method}
            for metric in metric_names:
                y = np.asarray([row[metric] for row in curve])
                result[metric] = float(np.interp(budget, x, y))
            matched.append(result)
    return matched, float(common_max)


def area_under_control_curve(
    control_rows_: list[dict[str, Any]], common_max: float
) -> dict[str, float]:
    result: dict[str, float] = {}
    if common_max <= 0.0:
        return {method: math.nan for method in (METHOD_CENTER, METHOD_RECIPIENT, METHOD_DONOR)}
    grid = np.linspace(0.0, common_max, 101)
    for method in (METHOD_CENTER, METHOD_RECIPIENT, METHOD_DONOR):
        curve = sorted(
            (row for row in control_rows_ if row["method"] == method),
            key=lambda row: row["mean_lambda"],
        )
        x = np.asarray([row["mean_lambda"] for row in curve])
        y = np.asarray([row["harmful_flip_rate"] for row in curve])
        interpolated = np.interp(grid, x, y)
        trapezoid = getattr(np, "trapezoid", np.trapz)
        result[method] = float(trapezoid(interpolated, grid) / common_max)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def print_prediction_summary(rows: list[dict[str, Any]]) -> None:
    flip_rows = [row for row in rows if row["harm_definition"] == "probe_correct_to_wrong"]
    if not flip_rows:
        return
    last_lambda = max(row["candidate_lambda"] for row in flip_rows)
    selected = [row for row in flip_rows if row["candidate_lambda"] == last_lambda]
    print(f"\nPrediction at candidate lambda={last_lambda:g} (probe correct -> wrong)")
    print("method                         AUROC       AUPRC       prevalence")
    for row in selected:
        print(
            f"{row['method']:<30} "
            f"{row['roc_auc']:>8.4f}    {row['average_precision']:>8.4f}    "
            f"{row['harm_prevalence']:>8.4f}"
        )


def print_control_summary(matched: list[dict[str, Any]], common_max: float) -> None:
    if not matched:
        return
    budget = max(row["target_mean_lambda"] for row in matched)
    selected = [row for row in matched if row["target_mean_lambda"] == budget]
    print(f"\nControl at matched mean lambda={budget:.4f} (common max={common_max:.4f})")
    print("method                         flip rate   mean delta CE   probe acc")
    for row in selected:
        print(
            f"{row['method']:<30} "
            f"{row['harmful_flip_rate']:>8.4f}    "
            f"{row['mean_probe_loss_delta']:>12.4f}    "
            f"{row['probe_accuracy']:>8.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline go/no-go validation for donor-conditioned fusion control"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="fixed feature-pair .npz archive")
    source.add_argument("--synthetic", action="store_true", help="run a synthetic smoke test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fusion_risk_minival"))
    parser.add_argument("--lambda-cap", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument(
        "--candidate-lambdas",
        type=parse_float_list,
        default=parse_float_list("0.1,0.2,0.3,0.4"),
        help="common fixed lambdas used for harmful-fusion prediction",
    )
    parser.add_argument(
        "--aggressiveness-grid",
        type=parse_float_list,
        default=[float(value) for value in np.linspace(0.0, 1.0, 21)],
        help="multipliers/caps used to build matched-control curves",
    )
    parser.add_argument(
        "--budgets",
        type=parse_float_list,
        default=None,
        help="optional target mean-lambda budgets; defaults to four common quantiles",
    )
    parser.add_argument("--loss-harm-threshold", type=float, default=0.1)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--probe-max-iter", type=int, default=2000)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-shared-probe",
        action="store_true",
        help="debug only: allow reference and probe classifiers to be identical",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.lambda_cap <= 1.0:
        raise ValueError("--lambda-cap must lie in (0, 1]")
    if args.delta < 0.0:
        raise ValueError("--delta must be non-negative")
    if any(value < 0.0 or value > 1.0 for value in args.candidate_lambdas):
        raise ValueError("candidate lambdas must lie in [0, 1]")
    if any(value < 0.0 for value in args.aggressiveness_grid):
        raise ValueError("aggressiveness values must be non-negative")

    data = make_synthetic(args.seed) if args.synthetic else load_npz(
        args.input, args.probe_max_iter, args.seed
    )
    recipient = data["recipient_features"]
    donor = data["donor_features"]
    labels = data["recipient_labels"]
    reference_weight = data["reference_weight"]
    reference_bias = data["reference_bias"]
    probe_weight = data["probe_weight"]
    probe_bias = data["probe_bias"]

    if (
        reference_weight.shape == probe_weight.shape
        and np.allclose(reference_weight, probe_weight)
        and np.allclose(reference_bias, probe_bias)
        and not args.allow_shared_probe
    ):
        raise ValueError(
            "reference and probe heads are identical; this makes boundary-risk validation "
            "circular. Supply an independent probe or use --allow-shared-probe for debugging only."
        )

    center_distance = cosine_center_distance(recipient, labels, data["class_centers"], args.eps)
    center_ratio = pi_h2t_donor_ratio(center_distance, args.lambda_cap)
    recipient_distance = pairwise_boundary_distances(
        recipient, labels, reference_weight, reference_bias, args.eps
    )
    donor_distance = pairwise_boundary_distances(
        donor, labels, reference_weight, reference_bias, args.eps
    )
    critical_ratio, recipient_margin, threshold = donor_conditioned_critical_ratio(
        recipient_distance, donor_distance, args.delta, args.eps
    )
    donor_ratio = np.minimum(critical_ratio, args.lambda_cap)
    recipient_ratio = distribution_matched_recipient_ratio(recipient_margin, center_ratio)

    predictions = prediction_rows(
        recipient,
        donor,
        labels,
        probe_weight,
        probe_bias,
        args.candidate_lambdas,
        center_ratio,
        recipient_ratio,
        critical_ratio,
        args.loss_harm_threshold,
        args.bootstrap,
        args.seed,
    )
    controls = control_rows(
        recipient,
        donor,
        labels,
        probe_weight,
        probe_bias,
        center_ratio,
        recipient_ratio,
        critical_ratio,
        args.lambda_cap,
        args.aggressiveness_grid,
    )
    matched, common_max = interpolate_control(controls, args.budgets)
    control_auc = area_under_control_curve(controls, common_max)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "prediction.csv", predictions)
    write_csv(args.output_dir / "control_curve.csv", controls)
    write_csv(args.output_dir / "matched_control.csv", matched)
    diagnostics: dict[str, np.ndarray] = {
        "recipient_labels": labels,
        "center_distance": center_distance,
        "recipient_margin": recipient_margin,
        "risk_threshold": threshold,
        "lambda_pi_h2t_center": center_ratio,
        "lambda_recipient_boundary": recipient_ratio,
        "lambda_donor_conditioned": donor_ratio,
        "lambda_donor_critical": critical_ratio,
    }
    if "donor_labels" in data:
        diagnostics["donor_labels"] = data["donor_labels"]
    np.savez_compressed(args.output_dir / "pair_diagnostics.npz", **diagnostics)

    summary = {
        "source": "synthetic" if args.synthetic else str(args.input.resolve()),
        "num_pairs": int(recipient.shape[0]),
        "feature_dim": int(recipient.shape[1]),
        "num_classes": int(reference_weight.shape[0]),
        "lambda_cap": args.lambda_cap,
        "delta": args.delta,
        "mean_base_ratio": {
            METHOD_CENTER: float(center_ratio.mean()),
            METHOD_RECIPIENT: float(recipient_ratio.mean()),
            METHOD_DONOR: float(donor_ratio.mean()),
        },
        "common_max_mean_lambda": common_max,
        "normalized_harmful_flip_area": control_auc,
        "primary_decision_rule": (
            "At matched mean donor mass, donor_conditioned should improve AUPRC and "
            "reduce harmful_flip_rate without degrading probe accuracy; probe-loss delta "
            "is a secondary severity check. Otherwise do not proceed to full training."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"pairs={recipient.shape[0]}, dim={recipient.shape[1]}, classes={reference_weight.shape[0]}")
    print(
        "base mean lambda: "
        f"center={center_ratio.mean():.4f}, "
        f"recipient={recipient_ratio.mean():.4f}, "
        f"donor={donor_ratio.mean():.4f}"
    )
    print_prediction_summary(predictions)
    print_control_summary(matched, common_max)
    print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
