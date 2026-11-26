"""Two-phase TRAIN-only estimator audit for NTC-BCL v2.

Phase 1 (estimate) reads only ordered TRAIN logits, the public protocol manifest
and observed labels. It writes and seals pi, T and gamma before any truth path is
opened. Phase 2 (evaluate) verifies that seal, then opens truth/ for offline
metrics. Neither phase reads the official CIFAR-100 TEST split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_AUDIT.cifar100lt.train import state_dict_sha256
from BCL_LTN.noise_transition_estimator import (
    estimate_transition_and_prior,
    prototype_health_diagnostics,
)
from BCL_LTN.train_noisy_bcl import safe_torch_load


K = 100
N = 10_847
DEFAULT_SUPPORT_FLOOR_RATIOS = (0.0, 0.5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("logits must be a finite [N,K] matrix")
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def estimator_summary(result, posterior_kind: str) -> dict:
    prior = result.clean_prior.detach().cpu().numpy()
    transition = result.transition.detach().cpu().numpy()
    support = result.row_support.detach().cpu().numpy()
    return {
        "posterior_kind": posterior_kind,
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
        "final_delta": float(result.final_delta),
        "estimated_noise_rate": float(1.0 - np.dot(prior, np.diag(transition))),
        "prior_min": float(prior.min()),
        "prior_max": float(prior.max()),
        "transition_diagonal_min": float(np.diag(transition).min()),
        "transition_diagonal_median": float(np.median(np.diag(transition))),
        "transition_diagonal_max": float(np.diag(transition).max()),
        "row_support_min": float(support.min()),
        "row_support_median": float(np.median(support)),
        "row_support_max": float(support.max()),
        "high_support_class_ids": result.global_support_class_ids.detach().cpu().tolist(),
        "clean_prior": prior.tolist(),
        "transition_diagonal": np.diag(transition).tolist(),
    }


def load_checkpoint_prototype_geometry(
    checkpoint_path: Path,
    expected_state_sha: str | None,
    state_key: str,
) -> dict:
    checkpoint = safe_torch_load(checkpoint_path, torch.device("cpu"))
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint payload must be a dictionary")
    if state_key not in ("model_state", "ema_model_state"):
        raise RuntimeError("Unsupported checkpoint state key for prototype diagnostics")
    state = checkpoint.get(state_key)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint does not contain the requested {state_key}")
    actual_state_sha = state_dict_sha256(state)
    if expected_state_sha and actual_state_sha != expected_state_sha:
        raise RuntimeError("Prototype checkpoint state differs from the logits checkpoint state")
    model = BCLCIFARModel()
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        prototype_embedding = model.prototype_mlp(model.classifier.weight)
        diagnostics = prototype_health_diagnostics(model.classifier.weight, prototype_embedding)
    return {
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_state_key": state_key,
        "model_state_sha256": actual_state_sha,
        "classifier_weight_norm": diagnostics["classifier_weight_norm"].cpu().tolist(),
        "small_weight_class_ids": torch.nonzero(
            diagnostics["small_weight_mask"], as_tuple=False
        ).flatten().cpu().tolist(),
        "prototype_max_offdiag_cosine": diagnostics["prototype_max_offdiag_cosine"].cpu().tolist(),
        "mean_offdiag_cosine": float(diagnostics["mean_offdiag_cosine"].cpu()),
        "prototype_mlp_was_eval": True,
        "prototype_mlp_bn_running_stats_mutated": False,
    }


def estimate_phase(args) -> None:
    protocol_dir = args.protocol_dir.resolve()
    output_dir = args.output_dir.resolve()
    logits_path = args.logits.resolve()
    if not protocol_dir.is_dir() or not logits_path.is_file():
        raise FileNotFoundError("protocol-dir and logits file must exist")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite estimate output: {output_dir}")

    manifest_path = protocol_dir / "manifest.json"
    observed_path = protocol_dir / "observed_labels.npy"
    if not manifest_path.is_file() or not observed_path.is_file():
        raise FileNotFoundError("Protocol requires manifest.json and observed_labels.npy")
    manifest = read_json(manifest_path)
    if manifest.get("observed_labels_file") != "observed_labels.npy":
        raise RuntimeError("Unexpected observed-label filename in public manifest")
    if manifest.get("num_samples") != N or manifest.get("num_classes") != K:
        raise RuntimeError("Protocol manifest does not describe CIFAR-100-LT IF=100")
    observed_sha = sha256(observed_path)
    if manifest.get("observed_labels_sha256") != observed_sha:
        raise RuntimeError("Observed-label SHA256 does not match the public protocol manifest")

    logits_receipt_path = logits_path.with_suffix(logits_path.suffix + ".json")
    if not logits_receipt_path.is_file():
        raise FileNotFoundError(f"Missing logit extraction receipt: {logits_receipt_path}")
    logits_receipt = read_json(logits_receipt_path)
    logits_sha = sha256(logits_path)
    if (logits_receipt.get("state") != "complete"
            or logits_receipt.get("logits_sha256") != logits_sha
            or logits_receipt.get("clean_labels_loaded") is not False
            or logits_receipt.get("observed_labels_loaded") is not False
            or logits_receipt.get("official_test_opened") is not False):
        raise RuntimeError("Logit extraction receipt is invalid or reports forbidden data access")

    logits = np.load(logits_path, allow_pickle=False)
    observed = np.load(observed_path, allow_pickle=False)
    if (logits.shape != (N, K) or logits.dtype.kind != "f" or not np.isfinite(logits).all()
            or observed.shape != (N,) or observed.dtype != np.int64
            or np.any((observed < 0) | (observed >= K))):
        raise RuntimeError("Logits/observed labels must align as finite [10847,100] and int64 [10847]")
    observed_counts = np.bincount(observed, minlength=K)
    if np.any(observed_counts <= 0):
        raise RuntimeError("Every observed class must have positive support")

    # The v1 checkpoint was trained with LogitCompensatedCE(logits + log n_obs).
    # Thus softmax(raw logits) is the balanced-evidence interpretation. The
    # second setting is an explicit sensitivity analysis using the observed
    # noisy-label prior as a long-tail posterior proxy; it is not a clean prior.
    q_balanced = softmax_numpy(logits)
    observed_prior = observed_counts.astype(np.float64) / float(N)
    q_long_tailed_observed_prior = softmax_numpy(
        logits.astype(np.float64) + np.log(observed_prior)[None, :]
    )

    estimator_kwargs = {
        "shrinkage_mass": float(args.shrinkage_mass),
        "high_support_fraction": float(args.high_support_fraction),
        "min_high_support_classes": int(args.min_high_support_classes),
        "max_iter": int(args.max_iter),
        "tolerance": float(args.tolerance),
        "prior_pseudocount": float(args.prior_pseudocount),
    }
    observed_tensor = torch.from_numpy(observed)
    balanced = estimate_transition_and_prior(
        torch.from_numpy(q_balanced),
        observed_tensor,
        posterior_kind="balanced_posterior",
        **estimator_kwargs,
    )
    long_tailed = estimate_transition_and_prior(
        torch.from_numpy(q_long_tailed_observed_prior),
        observed_tensor,
        posterior_kind="long_tailed_posterior",
        **estimator_kwargs,
    )

    pi_bal = balanced.clean_prior.cpu().numpy()
    pi_lt = long_tailed.clean_prior.cpu().numpy()
    gamma_bal = balanced.posterior_clean.cpu().numpy()
    gamma_lt = long_tailed.posterior_clean.cpu().numpy()
    q_prior_calibrated = softmax_numpy(
        logits.astype(np.float64) + np.log(pi_bal)[None, :]
    )

    prototype = None
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if sha256(checkpoint_path) != logits_receipt.get("checkpoint_sha256"):
            raise RuntimeError("Prototype checkpoint must be the same file used for logit extraction")
        prototype = load_checkpoint_prototype_geometry(
            checkpoint_path,
            logits_receipt.get("checkpoint_state_sha256"),
            logits_receipt.get("checkpoint_state_key", "model_state"),
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = output_dir / "estimate_bundle.npz"
    with bundle_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            logits=logits.astype(np.float32, copy=False),
            q_raw_balanced=q_balanced.astype(np.float32),
            q_long_tailed_observed_prior=q_long_tailed_observed_prior.astype(np.float32),
            q_prior_calibrated_balanced=q_prior_calibrated.astype(np.float32),
            transition_balanced=balanced.transition.cpu().numpy(),
            prior_balanced=pi_bal,
            gamma_balanced=gamma_bal.astype(np.float32),
            support_balanced=balanced.row_support.cpu().numpy(),
            high_support_ids_balanced=balanced.global_support_class_ids.cpu().numpy(),
            transition_long_tailed=long_tailed.transition.cpu().numpy(),
            prior_long_tailed=pi_lt,
            gamma_long_tailed=gamma_lt.astype(np.float32),
            support_long_tailed=long_tailed.row_support.cpu().numpy(),
            high_support_ids_long_tailed=long_tailed.global_support_class_ids.cpu().numpy(),
            observed_prior=observed_prior,
        )

    estimator_only = {
        "state": "sealed_estimate",
        "n": N,
        "num_classes": K,
        "protocol": manifest.get("protocol"),
        "regime": manifest.get("regime"),
        "logits_sha256": logits_sha,
        "logits_receipt_sha256": sha256(logits_receipt_path),
        "observed_labels_sha256": observed_sha,
        "protocol_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": logits_receipt.get("checkpoint_sha256"),
        "checkpoint_state_sha256": logits_receipt.get("checkpoint_state_sha256"),
        "checkpoint_state_key": logits_receipt.get("checkpoint_state_key"),
        "teacher_type": logits_receipt.get("teacher_type"),
        "posterior_semantics": {
            "balanced_posterior": "softmax(raw logits); LogitCompensatedCE trained logits with log observed counts, so raw logits are treated as balanced class evidence; multiply estimated clean pi exactly once in the E-step",
            "long_tailed_posterior": "softmax(raw logits + log observed noisy-label prior); sensitivity proxy only; do not multiply estimated clean pi again",
        },
        "primary_estimator": "balanced_posterior",
        "estimator_hyperparameters": estimator_kwargs,
        "balanced_posterior_estimate": estimator_summary(balanced, "balanced_posterior"),
        "long_tailed_posterior_estimate": estimator_summary(long_tailed, "long_tailed_posterior"),
        "prototype_geometry": prototype,
        "truth_loaded": False,
        "truth_directory_opened": False,
        "official_test_opened": False,
        "clean_labels_loaded": False,
        "corruption_mask_loaded": False,
    }
    write_json(output_dir / "estimator_only.json", estimator_only)
    seal = {
        "state": "immutable_estimator_outputs",
        "estimate_bundle_sha256": sha256(bundle_path),
        "estimator_only_sha256": sha256(output_dir / "estimator_only.json"),
        "logits_sha256": logits_sha,
        "observed_labels_sha256": observed_sha,
        "protocol_manifest_sha256": sha256(manifest_path),
        "truth_loaded_before_seal": False,
        "official_test_opened": False,
    }
    write_json(output_dir / "seal.json", seal)
    print(json.dumps(estimator_only, indent=2, ensure_ascii=False))


def probability_metrics(probabilities: np.ndarray, clean: np.ndarray, counts: np.ndarray) -> dict:
    q = np.asarray(probabilities, dtype=np.float64)
    prediction = q.argmax(axis=1)
    entropy = -(q * np.log(np.maximum(q, 1e-12))).sum(axis=1)
    nll = -np.log(np.maximum(q[np.arange(len(clean)), clean], 1e-12))
    group_masks = {
        "Overall": np.ones(len(clean), dtype=bool),
        "Many": counts[clean] > 100,
        "Medium": (counts[clean] >= 20) & (counts[clean] <= 100),
        "Few": counts[clean] < 20,
    }
    result = {}
    for name, mask in group_masks.items():
        result[name] = {
            "n": int(mask.sum()),
            "accuracy_percent": float(100.0 * np.mean(prediction[mask] == clean[mask])) if mask.any() else None,
            "nll": float(np.mean(nll[mask])) if mask.any() else None,
            "entropy": float(np.mean(entropy[mask])) if mask.any() else None,
        }
    return result


def per_class_probability_rows(probabilities: np.ndarray, clean: np.ndarray) -> list[dict]:
    prediction = probabilities.argmax(axis=1)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
    rows = []
    for class_id in range(K):
        mask = clean == class_id
        rows.append({
            "accuracy_percent": float(100.0 * np.mean(prediction[mask] == class_id)) if mask.any() else None,
            "entropy": float(entropy[mask].mean()) if mask.any() else None,
            "mean_probability_for_true_class": float(probabilities[mask, class_id].mean()) if mask.any() else None,
        })
    return rows


def soft_bcl_batch_diagnostics(probabilities: np.ndarray, support_floor_ratio: float) -> dict:
    """Mirror the current Soft-BCL q-derived weights; diagnostic only."""
    q = np.asarray(probabilities, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != K or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("Soft-BCL diagnostics require finite non-negative [B,100] probabilities")
    q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
    batch_size = q.shape[0]
    reliability = np.clip(
        (q.max(axis=1) - 1.0 / K) / (1.0 - 1.0 / K), 0.0, 1.0
    ).astype(np.float32)
    reliability[q.max(axis=1) == q.min(axis=1)] = 0.0

    prototype_q = np.eye(K, dtype=np.float32)
    candidate_q = np.concatenate((q, q, prototype_q), axis=0)
    anchor_q = np.concatenate((q, q), axis=0)
    candidate_reliability = np.concatenate(
        (reliability, reliability, np.ones(K, dtype=np.float32)), axis=0
    )
    anchor_reliability = np.concatenate((reliability, reliability), axis=0)

    num_anchors = 2 * batch_size
    positive = anchor_q @ candidate_q.T
    positive *= anchor_reliability[:, None] * candidate_reliability[None, :]
    paired = (np.arange(num_anchors) + batch_size) % num_anchors
    positive[np.arange(num_anchors), paired] = 1.0
    mask = np.ones((num_anchors, 2 * batch_size + K), dtype=np.float32)
    mask[np.arange(num_anchors), np.arange(num_anchors)] = 0.0
    positive *= mask

    class_occurrences = candidate_q.sum(axis=0)
    floor_value = 0.0
    effective_occurrences = class_occurrences.copy()
    if support_floor_ratio > 0:
        floor_value = max(1.0, float(support_floor_ratio) * float(class_occurrences.mean()))
        effective_occurrences = np.maximum(effective_occurrences, floor_value)
    candidate_class_mass = candidate_q @ effective_occurrences
    denominator_weights = np.maximum(candidate_class_mass[None, :] - positive, 1e-6)
    prototype_denominator = denominator_weights[:, 2 * batch_size:]
    prototype_inverse = 1.0 / prototype_denominator
    negative_mask = (mask > 0) & (positive <= 1e-12)
    negative_inverse_mass = (
        (1.0 / denominator_weights) * negative_mask.astype(np.float32)
    ).sum(axis=1)
    return {
        "class_occurrences": class_occurrences.astype(np.float64),
        "prototype_positive_mass_by_class": positive[:, 2 * batch_size:].sum(axis=0).astype(np.float64),
        "prototype_inverse_denominator_by_class": prototype_inverse.mean(axis=0).astype(np.float64),
        "positive_mass_per_anchor": positive.sum(axis=1).astype(np.float64),
        "negative_inverse_denominator_mass_per_anchor": negative_inverse_mass.astype(np.float64),
        "denominator_weight_min": float(denominator_weights.min()),
        "denominator_weight_median": float(np.median(denominator_weights)),
        "denominator_weight_max": float(denominator_weights.max()),
        "class_mass_floor": float(floor_value),
    }


def aggregate_soft_bcl_diagnostics(
    probabilities: np.ndarray,
    *,
    batch_size: int,
    support_floor_ratio: float,
) -> dict:
    sums_occ = np.zeros(K, dtype=np.float64)
    sums_positive = np.zeros(K, dtype=np.float64)
    sums_inverse = np.zeros(K, dtype=np.float64)
    total_samples = 0
    total_anchors = 0
    positive_sum = 0.0
    negative_inverse_sum = 0.0
    min_values, median_values, max_values, floors = [], [], [], []
    batch_count = 0
    for start in range(0, len(probabilities), batch_size):
        batch = probabilities[start:start + batch_size]
        diag = soft_bcl_batch_diagnostics(batch, support_floor_ratio)
        anchors = 2 * len(batch)
        sums_occ += diag["class_occurrences"] * len(batch)
        sums_positive += diag["prototype_positive_mass_by_class"]
        sums_inverse += diag["prototype_inverse_denominator_by_class"] * anchors
        positive_sum += float(diag["positive_mass_per_anchor"].sum())
        negative_inverse_sum += float(diag["negative_inverse_denominator_mass_per_anchor"].sum())
        total_samples += len(batch)
        total_anchors += anchors
        min_values.append(diag["denominator_weight_min"])
        median_values.append(diag["denominator_weight_median"])
        max_values.append(diag["denominator_weight_max"])
        floors.append(diag["class_mass_floor"])
        batch_count += 1
    return {
        "class_occurrences_per_sample": (sums_occ / max(total_samples, 1)).tolist(),
        "prototype_positive_mass_per_anchor_by_class": (sums_positive / max(total_anchors, 1)).tolist(),
        "prototype_inverse_denominator_mean_by_class": (sums_inverse / max(total_anchors, 1)).tolist(),
        "mean_positive_mass_per_anchor": positive_sum / max(total_anchors, 1),
        "mean_negative_inverse_denominator_mass_per_anchor": negative_inverse_sum / max(total_anchors, 1),
        "mean_batch_denominator_weight_min": float(np.mean(min_values)),
        "mean_batch_denominator_weight_median": float(np.mean(median_values)),
        "mean_batch_denominator_weight_max": float(np.mean(max_values)),
        "mean_class_mass_floor": float(np.mean(floors)),
        "num_batches": batch_count,
        "num_anchors": total_anchors,
        "support_floor_ratio": float(support_floor_ratio),
    }


def transition_and_prior_metrics(
    pi_hat: np.ndarray,
    transition_hat: np.ndarray,
    clean: np.ndarray,
    observed: np.ndarray,
    transition_reference: np.ndarray,
    empirical_transition: np.ndarray,
    reference_kind: str,
) -> dict:
    clean_counts = np.bincount(clean, minlength=K).astype(np.float64)
    pi_true = clean_counts / clean_counts.sum()
    present = clean_counts > 0
    delta_t = transition_hat - transition_reference
    row_tv = 0.5 * np.abs(delta_t).sum(axis=1)
    pi_eps = 1e-12
    groups = {
        "Many": np.flatnonzero(clean_counts > 100),
        "Medium": np.flatnonzero((clean_counts >= 20) & (clean_counts <= 100)),
        "Few": np.flatnonzero(clean_counts < 20),
    }
    group_errors = {}
    for name, class_ids in groups.items():
        if class_ids.size == 0:
            group_errors[name] = {"classes": 0, "prior_mass_true": None, "prior_mass_estimated": None}
            continue
        group_errors[name] = {
            "classes": int(class_ids.size),
            "prior_mass_true": float(pi_true[class_ids].sum()),
            "prior_mass_estimated": float(pi_hat[class_ids].sum()),
            "prior_mass_absolute_error": float(abs(pi_hat[class_ids].sum() - pi_true[class_ids].sum())),
            "transition_mae": float(np.abs(delta_t[class_ids]).mean()),
            "transition_frobenius": float(np.linalg.norm(delta_t[class_ids])),
            "transition_diagonal_mae": float(np.abs(np.diag(transition_hat)[class_ids] - np.diag(transition_reference)[class_ids]).mean()),
            "transition_row_tv_mean": float(row_tv[class_ids].mean()),
        }
    empirical_delta = transition_hat - empirical_transition
    return {
        "reference_kind": reference_kind,
        "estimated_noise_rate": float(1.0 - np.dot(pi_hat, np.diag(transition_hat))),
        "realized_noise_rate": float(np.mean(clean != observed)),
        "transition_reference": {
            "mae": float(np.abs(delta_t[present]).mean()),
            "frobenius_norm": float(np.linalg.norm(delta_t[present])),
            "frobenius_rmse": float(np.sqrt(np.mean(delta_t[present] ** 2))),
            "diagonal_mae": float(np.abs(np.diag(transition_hat)[present] - np.diag(transition_reference)[present]).mean()),
            "row_total_variation_mean": float(row_tv[present].mean()),
        },
        "transition_empirical_finite_sample": {
            "mae": float(np.abs(empirical_delta[present]).mean()),
            "frobenius_norm": float(np.linalg.norm(empirical_delta[present])),
            "diagonal_mae": float(np.abs(np.diag(transition_hat)[present] - np.diag(empirical_transition)[present]).mean()),
        },
        "clean_prior": {
            "mae": float(np.abs(pi_hat - pi_true).mean()),
            "l1": float(np.abs(pi_hat - pi_true).sum()),
            "l2": float(np.linalg.norm(pi_hat - pi_true)),
            "kl_true_to_estimated": float(np.sum(pi_true * np.log((pi_true + pi_eps) / (pi_hat + pi_eps)))),
            "minimum_true": float(pi_true.min()),
            "minimum_estimated": float(pi_hat.min()),
            "maximum_true": float(pi_true.max()),
            "maximum_estimated": float(pi_hat.max()),
        },
        "frequency_groups": group_errors,
    }


def evaluate_phase(args) -> None:
    protocol_dir = args.protocol_dir.resolve()
    output_dir = args.output_dir.resolve()
    seal_path = output_dir / "seal.json"
    estimator_path = output_dir / "estimator_only.json"
    bundle_path = output_dir / "estimate_bundle.npz"
    for path in (seal_path, estimator_path, bundle_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing sealed estimator artifact: {path}")
    seal = read_json(seal_path)
    estimator_only = read_json(estimator_path)
    if (seal.get("state") != "immutable_estimator_outputs"
            or seal.get("truth_loaded_before_seal") is not False
            or seal.get("estimate_bundle_sha256") != sha256(bundle_path)
            or seal.get("estimator_only_sha256") != sha256(estimator_path)):
        raise RuntimeError("Estimator seal is invalid; refusing to open truth")
    manifest_path = protocol_dir / "manifest.json"
    observed_path = protocol_dir / "observed_labels.npy"
    if (sha256(manifest_path) != seal.get("protocol_manifest_sha256")
            or sha256(observed_path) != seal.get("observed_labels_sha256")):
        raise RuntimeError("Protocol manifest or observed-label file changed after estimation")

    # Truth is first accessed here, after the immutable estimator receipt verifies.
    truth_dir = protocol_dir / "truth"
    truth_manifest_path = truth_dir / "truth_manifest.json"
    clean_path = truth_dir / "clean_labels.npy"
    empirical_transition_path = truth_dir / "true_transition.npy"
    for path in (truth_manifest_path, clean_path, empirical_transition_path):
        if not path.is_file():
            raise FileNotFoundError(f"Offline truth artifact missing: {path}")
    truth_manifest = read_json(truth_manifest_path)
    if sha256(clean_path) != truth_manifest.get("clean_labels_sha256"):
        raise RuntimeError("Offline clean-label SHA256 mismatch")
    if sha256(empirical_transition_path) != truth_manifest.get("true_transition_sha256"):
        raise RuntimeError("Offline empirical-transition SHA256 mismatch")

    clean = np.load(clean_path, allow_pickle=False)
    observed = np.load(observed_path, allow_pickle=False)
    empirical_transition = np.load(empirical_transition_path, allow_pickle=False)
    if (clean.shape != (N,) or clean.dtype != np.int64 or observed.shape != (N,)
            or empirical_transition.shape != (K, K)):
        raise RuntimeError("Offline truth arrays have invalid shapes or dtypes")
    clean_counts = np.bincount(clean, minlength=K)
    if truth_manifest.get("clean_class_counts") != clean_counts.tolist():
        raise RuntimeError("Offline clean class counts do not match the truth manifest")
    if not np.allclose(empirical_transition.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Empirical truth transition is not row stochastic")

    theoretical_path = truth_dir / "theoretical_transition.npy"
    theoretical_transition = None
    if truth_manifest.get("theoretical_transition_sha256") is not None:
        if not theoretical_path.is_file() or sha256(theoretical_path) != truth_manifest["theoretical_transition_sha256"]:
            raise RuntimeError("Theoretical transition SHA256 mismatch")
        theoretical_transition = np.load(theoretical_path, allow_pickle=False)
        if theoretical_transition.shape != (K, K) or not np.allclose(
            theoretical_transition.sum(axis=1), 1.0, atol=1e-8
        ):
            raise RuntimeError("Theoretical transition is invalid")
    transition_reference = theoretical_transition if theoretical_transition is not None else empirical_transition
    reference_kind = "theoretical_protocol_matrix" if theoretical_transition is not None else "empirical_finite_sample_matrix"

    with np.load(bundle_path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    pi_bal = arrays["prior_balanced"].astype(np.float64)
    pi_lt = arrays["prior_long_tailed"].astype(np.float64)
    t_bal = arrays["transition_balanced"].astype(np.float64)
    t_lt = arrays["transition_long_tailed"].astype(np.float64)
    q_raw = arrays["q_raw_balanced"].astype(np.float64)
    q_prior = arrays["q_prior_calibrated_balanced"].astype(np.float64)
    gamma_bal = arrays["gamma_balanced"].astype(np.float64)
    q_lt = arrays["q_long_tailed_observed_prior"].astype(np.float64)
    gamma_lt = arrays["gamma_long_tailed"].astype(np.float64)

    tpi_metrics = {
        "balanced_posterior": transition_and_prior_metrics(
            pi_bal, t_bal, clean, observed, transition_reference, empirical_transition, reference_kind
        ),
        "long_tailed_posterior_sensitivity": transition_and_prior_metrics(
            pi_lt, t_lt, clean, observed, transition_reference, empirical_transition, reference_kind
        ),
    }
    q_variants = {
        "raw_balanced_q": q_raw,
        "prior_calibrated_q": q_prior,
        "em_posterior_gamma": gamma_bal,
        "observed_prior_long_tailed_q": q_lt,
        "long_tailed_posterior_gamma": gamma_lt,
    }
    q_metrics = {
        name: probability_metrics(q, clean, clean_counts)
        for name, q in q_variants.items()
    }

    soft_bcl_diagnostics = {}
    for q_name in ("raw_balanced_q", "prior_calibrated_q", "em_posterior_gamma"):
        soft_bcl_diagnostics[q_name] = {}
        for floor_ratio in args.support_floor_ratios:
            key = f"floor_{floor_ratio:g}"
            soft_bcl_diagnostics[q_name][key] = aggregate_soft_bcl_diagnostics(
                q_variants[q_name],
                batch_size=args.batch_size,
                support_floor_ratio=float(floor_ratio),
            )

    class_rows = {}
    for name, q in q_variants.items():
        class_rows[name] = per_class_probability_rows(q, clean)
    rows = []
    pi_true = clean_counts / clean_counts.sum()
    t_true_diag = np.diag(transition_reference)
    for class_id in range(K):
        row = {
            "class_id": class_id,
            "clean_count": int(clean_counts[class_id]),
            "frequency_group": (
                "Many" if clean_counts[class_id] > 100
                else "Medium" if clean_counts[class_id] >= 20
                else "Few"
            ),
            "pi_true": float(pi_true[class_id]),
            "pi_estimated_balanced": float(pi_bal[class_id]),
            "pi_estimated_long_tailed": float(pi_lt[class_id]),
            "T_diag_reference": float(t_true_diag[class_id]),
            "T_diag_empirical": float(np.diag(empirical_transition)[class_id]),
            "T_diag_estimated_balanced": float(np.diag(t_bal)[class_id]),
            "T_diag_estimated_long_tailed": float(np.diag(t_lt)[class_id]),
            "effective_support_balanced": float(arrays["support_balanced"][class_id]),
            "effective_support_long_tailed": float(arrays["support_long_tailed"][class_id]),
        }
        for name, per_class in class_rows.items():
            row[f"{name}_accuracy_pct"] = per_class[class_id]["accuracy_percent"]
            row[f"{name}_entropy"] = per_class[class_id]["entropy"]
        rows.append(row)

    csv_path = output_dir / "per_class_metrics.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "state": "offline_evaluation_complete",
        "regime": estimator_only.get("regime"),
        "estimator_only": estimator_only,
        "transition_and_prior_metrics": tpi_metrics,
        "probability_metrics": q_metrics,
        "soft_bcl_diagnostics": soft_bcl_diagnostics,
        "truth_loaded_after_estimator_sealed": True,
        "truth_manifest_sha256": sha256(truth_manifest_path),
        "clean_labels_sha256": sha256(clean_path),
        "empirical_transition_sha256": sha256(empirical_transition_path),
        "theoretical_transition_sha256": (
            sha256(theoretical_path) if theoretical_transition is not None else None
        ),
        "official_test_opened": False,
        "corruption_mask_loaded": False,
        "interpretation_guardrails": [
            "The frozen v1 BCL checkpoint is a fixed teacher, not an EMA teacher.",
            "softmax(raw logits) is treated as balanced evidence because the v1 LC objective adds log observed class counts during training.",
            "The long-tailed posterior sensitivity uses the observed noisy-label prior as a proxy; it is not the unknown clean prior.",
            "The clean prior and transition estimates were sealed before the offline truth directory was opened.",
            "Soft-BCL positive/negative denominator statistics are diagnostic-only; no loss or training behavior is changed.",
            "The coarse-superclass cyclic asymmetric protocol is an explicit synthetic stress test, not a claim that it is the only standard asymmetric CIFAR-100 noise definition.",
            "CIFAR-100N has no theoretical transition matrix here; its reference is the empirical TRAIN transition matrix.",
        ],
    }
    write_json(output_dir / "offline_evaluation.json", report)

    lines = [
        f"# NTC-BCL v2 Stage-1 estimator audit — {estimator_only.get('regime')}",
        "",
        f"- Estimator sealed before truth access: yes",
        f"- Posterior semantics: {estimator_only['primary_estimator']} (primary), long-tailed observed-prior proxy (sensitivity)",
        f"- Transition reference: {reference_kind}",
        f"- Estimated balanced EM converged: {estimator_only['balanced_posterior_estimate']['converged']}",
        f"- Balanced EM iterations / final delta: {estimator_only['balanced_posterior_estimate']['iterations']} / {estimator_only['balanced_posterior_estimate']['final_delta']:.3e}",
        "",
        "## Transition and prior estimation",
        "",
        "| Estimator | T MAE | T Frobenius | T diagonal MAE | Prior MAE | Prior KL | Estimated noise | Realized noise |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in tpi_metrics.items():
        t = values["transition_reference"]
        p = values["clean_prior"]
        lines.append(
            f"| {name} | {t['mae']:.6f} | {t['frobenius_norm']:.6f} | {t['diagonal_mae']:.6f} "
            f"| {p['mae']:.6f} | {p['kl_true_to_estimated']:.6f} "
            f"| {values['estimated_noise_rate']:.4f} | {values['realized_noise_rate']:.4f} |"
        )
    lines.extend([
        "",
        "## Probability accuracy",
        "",
        "| q variant | Overall | Many | Medium | Few |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name, metrics in q_metrics.items():
        vals = [metrics[g]["accuracy_percent"] for g in ("Overall", "Many", "Medium", "Few")]
        lines.append("| " + name + " | " + " | ".join(f"{v:.3f}%" for v in vals) + " |")
    lines.extend([
        "",
        "## Soft-BCL denominator diagnostics",
        "",
        "| q variant | support floor | positive mass / anchor | negative inverse-denominator mass / anchor | denominator min / median / max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for q_name, variants in soft_bcl_diagnostics.items():
        for key, values in variants.items():
            lines.append(
                f"| {q_name} | {values['support_floor_ratio']:.2f} "
                f"| {values['mean_positive_mass_per_anchor']:.5f} "
                f"| {values['mean_negative_inverse_denominator_mass_per_anchor']:.5f} "
                f"| {values['mean_batch_denominator_weight_min']:.5g} / "
                f"{values['mean_batch_denominator_weight_median']:.5g} / "
                f"{values['mean_batch_denominator_weight_max']:.5g} |"
            )
    lines.extend([
        "",
        "## Limitations",
        "",
        "- This is a fixed-teacher estimator audit, not an end-to-end v2 training result.",
        "- The epoch-200 v1 noisy-BCL teacher is known to have weak Few accuracy; do not interpret poor estimates as an isolated estimator failure without the q diagnostics.",
        "- Transition rows for five-example classes are weakly identified and are shrunk toward the high-support empirical template.",
        "- No clean labels, corruption masks, or official TEST data were used during estimation.",
        "",
    ])
    report_path = output_dir / "STAGE1_AUDIT_REPORT.md"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))

    print(json.dumps({
        "state": "offline_evaluation_complete",
        "regime": estimator_only.get("regime"),
        "balanced_transition_metrics": tpi_metrics["balanced_posterior"]["transition_reference"],
        "balanced_prior_metrics": tpi_metrics["balanced_posterior"]["clean_prior"],
        "q_accuracy": {
            name: {group: value["accuracy_percent"] for group, value in metrics.items()}
            for name, metrics in q_metrics.items()
        },
        "official_test_opened": False,
    }, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-phase NTC-BCL v2 estimator audit")
    parser.add_argument("--phase", choices=("estimate", "evaluate"), required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logits", type=Path, default=None, help="Required only for --phase estimate")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional same v1 checkpoint for prototype geometry")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shrinkage-mass", type=float, default=10.0)
    parser.add_argument("--high-support-fraction", type=float, default=0.35)
    parser.add_argument("--min-high-support-classes", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--prior-pseudocount", type=float, default=0.05)
    parser.add_argument("--support-floor-ratios", type=float, nargs="+", default=list(DEFAULT_SUPPORT_FLOOR_RATIOS))
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.support_floor_ratios or any(
        not math.isfinite(value) or value < 0 for value in args.support_floor_ratios
    ):
        raise ValueError("--support-floor-ratios must contain non-negative finite values")
    if args.phase == "estimate":
        if args.logits is None:
            raise ValueError("--logits is required for --phase estimate")
        estimate_phase(args)
    else:
        evaluate_phase(args)


if __name__ == "__main__":
    main()
