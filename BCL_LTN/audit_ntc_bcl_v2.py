"""Offline audit for NTC-BCL v2 transition/prior estimates and Soft-BCL collapse.

The estimator is completed and its outputs are persisted before clean labels are
opened. Clean labels are used only for offline scoring, class grouping and
prototype/q diagnostics. No official CIFAR TEST split is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_LTN.ntc_bcl_v2_core import (
    estimate_transition_em,
    evaluate_transition_estimate,
    probability_metrics,
    prototype_geometry,
    q_from_logits,
    soft_bcl_batch_diagnostics,
    softmax_numpy,
    transition_truth_from_labels,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def class_group(count: int) -> str:
    if count > 100:
        return "Many"
    if count >= 20:
        return "Medium"
    return "Few"


def per_class_probability_rows(q: np.ndarray, clean: np.ndarray, counts: np.ndarray):
    prediction = q.argmax(axis=1)
    entropy = -(q * np.log(np.maximum(q, 1e-12))).sum(axis=1)
    rows = []
    for c in range(q.shape[1]):
        mask = clean == c
        rows.append({
            "accuracy_percent": 100.0 * float(np.mean(prediction[mask] == c)) if mask.any() else None,
            "entropy": float(entropy[mask].mean()) if mask.any() else None,
            "mean_probability_for_true_class": float(q[mask, c].mean()) if mask.any() else None,
            "n": int(mask.sum()),
        })
    return rows


def aggregate_soft_bcl_diagnostics(
    q: np.ndarray,
    *,
    batch_size: int,
    support_floor_ratio: float,
) -> dict:
    k = q.shape[1]
    sums = {
        "class_occurrences": np.zeros(k, dtype=np.float64),
        "effective_class_occurrences": np.zeros(k, dtype=np.float64),
        "prototype_positive_mass_by_class": np.zeros(k, dtype=np.float64),
        "prototype_inverse_denominator_by_class": np.zeros(k, dtype=np.float64),
        "prototype_denominator_by_class": np.zeros(k, dtype=np.float64),
    }
    total_anchors = 0
    batch_count = 0
    positive_mass_sum = 0.0
    minimums, medians, maximums, floors = [], [], [], []
    for start in range(0, len(q), batch_size):
        batch = q[start:start + batch_size]
        diag = soft_bcl_batch_diagnostics(
            batch, support_floor_ratio=support_floor_ratio, semantic_ramp=1.0
        )
        anchors = 2 * len(batch)
        for key in sums:
            if "inverse_denominator" in key or "denominator_by_class" in key:
                sums[key] += np.asarray(diag[key], dtype=np.float64) * anchors
            elif "positive_mass" in key:
                sums[key] += np.asarray(diag[key], dtype=np.float64)
            else:
                sums[key] += np.asarray(diag[key], dtype=np.float64)
        total_anchors += anchors
        batch_count += 1
        positive_mass_sum += float(np.asarray(diag["positive_mass_per_anchor"]).sum())
        minimums.append(float(diag["denominator_weight_min"]))
        medians.append(float(diag["denominator_weight_median"]))
        maximums.append(float(diag["denominator_weight_max"]))
        floors.append(float(diag["class_mass_floor"]))

    for key in sums:
        if "inverse_denominator" in key or "denominator_by_class" in key or "positive_mass" in key:
            sums[key] /= max(total_anchors, 1)
        else:
            sums[key] /= max(batch_count, 1)
    return {
        **{key: value.tolist() for key, value in sums.items()},
        "mean_positive_mass_per_anchor": positive_mass_sum / max(total_anchors, 1),
        "mean_batch_denominator_weight_min": float(np.mean(minimums)),
        "mean_batch_denominator_weight_median": float(np.mean(medians)),
        "mean_batch_denominator_weight_max": float(np.mean(maximums)),
        "mean_class_mass_floor": float(np.mean(floors)),
        "num_batches": batch_count,
        "num_anchors": total_anchors,
        "support_floor_ratio": float(support_floor_ratio),
    }


def load_prototype_geometry(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("ema_model_state")
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain ema_model_state")
    model = BCLCIFARModel()
    model.load_state_dict(state, strict=True)
    model.eval()
    result = prototype_geometry(model.classifier.weight, model.prototype_mlp)
    result["checkpoint_sha256"] = sha256(checkpoint_path)
    result["ema_state_sha256"] = checkpoint.get("ema_state_sha256")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True, help="EMA raw classifier logits [N,100]")
    parser.add_argument("--observed-labels", type=Path, required=True)
    parser.add_argument("--clean-labels", type=Path, default=None, help="Offline-only truth; opened after estimator outputs are sealed")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional EMA checkpoint for prototype geometry")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shrinkage-mass", type=float, default=20.0)
    parser.add_argument("--prior-alpha", type=float, default=0.5)
    parser.add_argument("--high-support-fraction", type=float, default=0.35)
    parser.add_argument("--support-floor-ratios", type=float, nargs="+", default=[0.0, 0.5])
    args = parser.parse_args()

    for path in (args.logits, args.observed_labels):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.clean_labels is not None and not args.clean_labels.is_file():
        raise FileNotFoundError(args.clean_labels)
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit directory: {args.output_dir}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    logits = np.load(args.logits, allow_pickle=False)
    observed = np.load(args.observed_labels, allow_pickle=False)
    if logits.ndim != 2 or logits.shape[1] != 100 or logits.shape[0] != len(observed):
        raise RuntimeError("EMA logits and observed labels must align as [N,100] and [N]")
    if logits.dtype.kind != "f" or not np.isfinite(logits).all():
        raise RuntimeError("EMA logits must be finite floating-point values")
    if observed.dtype != np.int64 or np.any((observed < 0) | (observed >= 100)):
        raise RuntimeError("Observed labels must be int64 class IDs in [0,100)")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    balanced_probabilities = softmax_numpy(logits)
    estimate = estimate_transition_em(
        balanced_probabilities,
        observed,
        shrinkage_mass=args.shrinkage_mass,
        prior_alpha=args.prior_alpha,
        high_support_fraction=args.high_support_fraction,
    )
    np.save(args.output_dir / "estimated_clean_prior.npy", estimate["clean_prior"], allow_pickle=False)
    np.save(args.output_dir / "estimated_transition.npy", estimate["transition"], allow_pickle=False)
    np.save(args.output_dir / "estimated_posterior_gamma.npy", estimate["posterior"], allow_pickle=False)
    estimator_only = {
        "n": int(len(observed)),
        "num_classes": int(logits.shape[1]),
        "logits_sha256": sha256(args.logits),
        "observed_labels_sha256": sha256(args.observed_labels),
        "estimated_noise_rate": float(estimate["estimated_noise_rate"]),
        "iterations": int(estimate["iterations"]),
        "converged": bool(estimate["converged"]),
        "final_delta": float(estimate["final_delta"]),
        "shrinkage_mass": float(estimate["shrinkage_mass"]),
        "prior_alpha_per_class": float(estimate["prior_alpha_per_class"]),
        "high_support_fraction": float(estimate["high_support_fraction"]),
        "high_support_class_ids": np.asarray(estimate["high_support_class_ids"]).tolist(),
        "global_diagonal_retention": float(estimate["global_diagonal_retention"]),
        "estimated_clean_prior": np.asarray(estimate["clean_prior"]).tolist(),
        "effective_support": np.asarray(estimate["effective_support"]).tolist(),
        "truth_loaded": False,
    }
    write_json(args.output_dir / "estimator_only.json", estimator_only)

    if args.clean_labels is None:
        print(json.dumps(estimator_only, indent=2, ensure_ascii=False))
        return

    # Truth access starts only after T, pi and gamma have been saved and hashed.
    clean = np.load(args.clean_labels, allow_pickle=False)
    if clean.dtype != np.int64 or clean.shape != observed.shape:
        raise RuntimeError("Offline clean labels must be int64 and align with observed labels")
    clean_counts = np.bincount(clean, minlength=100)
    truth = transition_truth_from_labels(clean, observed)
    transition_metrics = evaluate_transition_estimate(estimate, clean, observed)

    q_raw = q_from_logits(logits)
    q_prior = q_from_logits(logits, clean_prior=estimate["clean_prior"])
    q_gamma = np.asarray(estimate["posterior"], dtype=np.float64)
    q_metrics = {
        "raw_balanced_q": probability_metrics(q_raw, clean, clean_counts),
        "prior_calibrated_q": probability_metrics(q_prior, clean, clean_counts),
        "em_posterior_gamma": probability_metrics(q_gamma, clean, clean_counts),
    }

    bcl_diagnostics = {}
    q_variants = {"raw_balanced_q": q_raw, "prior_calibrated_q": q_prior, "em_posterior_gamma": q_gamma}
    for q_name, q_value in q_variants.items():
        bcl_diagnostics[q_name] = {}
        for floor_ratio in args.support_floor_ratios:
            key = f"floor_{floor_ratio:g}"
            bcl_diagnostics[q_name][key] = aggregate_soft_bcl_diagnostics(
                q_value, batch_size=args.batch_size, support_floor_ratio=floor_ratio
            )

    prototype = load_prototype_geometry(args.checkpoint) if args.checkpoint else None
    if prototype is not None:
        write_json(args.output_dir / "prototype_geometry.json", prototype)

    raw_class = per_class_probability_rows(q_raw, clean, clean_counts)
    prior_class = per_class_probability_rows(q_prior, clean, clean_counts)
    gamma_class = per_class_probability_rows(q_gamma, clean, clean_counts)
    rows = []
    for c in range(100):
        row = {
            "class_id": c,
            "clean_count": int(clean_counts[c]),
            "group": class_group(int(clean_counts[c])),
            "pi_true": float(clean_counts[c] / clean_counts.sum()),
            "pi_estimated": float(estimate["clean_prior"][c]),
            "T_diag_true": float(truth["transition"][c, c]),
            "T_diag_estimated": float(np.asarray(estimate["transition"])[c, c]),
            "effective_support": float(np.asarray(estimate["effective_support"])[c]),
            "raw_q_accuracy_pct": raw_class[c]["accuracy_percent"],
            "prior_q_accuracy_pct": prior_class[c]["accuracy_percent"],
            "gamma_accuracy_pct": gamma_class[c]["accuracy_percent"],
            "raw_q_entropy": raw_class[c]["entropy"],
            "prior_q_entropy": prior_class[c]["entropy"],
            "gamma_entropy": gamma_class[c]["entropy"],
            "raw_q_true_class_probability": raw_class[c]["mean_probability_for_true_class"],
            "prior_q_true_class_probability": prior_class[c]["mean_probability_for_true_class"],
            "gamma_true_class_probability": gamma_class[c]["mean_probability_for_true_class"],
        }
        if prototype is not None:
            row["classifier_weight_norm"] = prototype["classifier_weight_norm"][c]
            row["raw_prototype_max_cosine"] = prototype["raw_weight_input"]["per_class_max_off_diagonal_cosine"][c]
            row["direction_prototype_max_cosine"] = prototype["direction_normalized_weight_input"]["per_class_max_off_diagonal_cosine"][c]
        for q_name in q_variants:
            for floor_ratio in args.support_floor_ratios:
                diag = bcl_diagnostics[q_name][f"floor_{floor_ratio:g}"]
                row[f"{q_name}_floor{floor_ratio:g}_class_occurrence"] = diag["class_occurrences"][c]
                row[f"{q_name}_floor{floor_ratio:g}_prototype_positive_mass"] = diag["prototype_positive_mass_by_class"][c]
                row[f"{q_name}_floor{floor_ratio:g}_prototype_inverse_denominator"] = diag["prototype_inverse_denominator_by_class"][c]
        rows.append(row)

    csv_path = args.output_dir / "per_class_metrics.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "estimator_only": estimator_only,
        "transition_metrics": transition_metrics,
        "q_metrics": q_metrics,
        "soft_bcl_diagnostics": bcl_diagnostics,
        "prototype_geometry_included": prototype is not None,
        "truth_loaded_after_estimator_sealed": True,
        "official_test_opened": False,
        "interpretation_guardrails": [
            "A fixed EMA teacher is used to audit the estimator across observed-label regimes; this is not a per-regime end-to-end training result.",
            "The clean prior and transition estimates are generated without clean labels; clean labels are loaded only for this offline evaluation.",
            "support_floor_ratio changes the v2 denominator weighting and is an explicit ablation; it is not a numerical epsilon.",
            "Direction-normalized prototype input is a diagnostic counterfactual and does not mutate BCL_AUDIT model structure.",
        ],
    }
    write_json(args.output_dir / "offline_evaluation.json", report)
    print(json.dumps({
        "estimated_noise_rate": estimator_only["estimated_noise_rate"],
        "true_noise_rate": transition_metrics["true_global_noise_rate"],
        "T_MAE": transition_metrics["transition_mae"],
        "T_diag_MAE": transition_metrics["transition_diagonal_mae"],
        "pi_MAE": transition_metrics["clean_prior_mae"],
        "group_noise_rates": transition_metrics["group_noise_rates"],
        "q_metrics": q_metrics,
        "prototype_geometry": prototype,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
