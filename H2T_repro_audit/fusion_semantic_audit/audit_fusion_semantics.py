#!/usr/bin/env python
"""Exact replay audit of recipient-label consistency under historical H2T channel fusion.

This is a diagnostic-only experiment: no optimizer, no parameter update, and no
official CIFAR-100 test data access. It replays only epoch 10 augmented features
and validates them against sealed batch signatures before computing any result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AUDIT_ROOT = REPO / "H2T_repro_audit"
OLD = AUDIT_ROOT / "clean_vs_stability_runs" / "clean_s1_seed42_s2_124_125_20261014"
FACTORIAL = AUDIT_ROOT / "lowvs_factorial"
NORM = AUDIT_ROOT / "lowvs_norm_direction"
DISC = AUDIT_ROOT / "lowvs_disc_preserve"
RESULTS = HERE / "results"
DIAGNOSTICS = HERE / "diagnostics"

SEEDS = (124, 125)
N = 10847
BATCH = 128
EPOCHS = 10
BATCHES_PER_EPOCH = math.ceil(N / BATCH)
NUM_CLASSES = 100
FEATURE_DIM = 64
MASK_SIZE = 16
PRIMARY_EPOCH = 10

MASK_ARMS = ("Uniform-PerSample", "Low-VS")
MASK_KEYS = {
    "Uniform-PerSample": "Uniform_PerSample",
    "Low-VS": "Low_VS",
}
MASK_FOLDERS = {
    "Uniform-PerSample": "uniform",
    "Low-VS": "lowvs",
}
VALIDATION_ARMS = ("LowVS", "LowVS-CleanCE", "LowVS-TopCompetitor")
VALIDATION_CLASSES = tuple(range(1, 100))
GROUPS = ("All", "Head", "OfficialMedium", "Tail", "Tail-cross-class-donor", "Tail-same-class-donor")
REPLAY_FIELDS = (
    "indices_sha256",
    "images_sha256",
    "features_sha256",
    "gumbel_sha256",
    "uniform_mask_sha256",
    "lowvs_mask_sha256",
)
OUTPUT_FILES = (
    "replay_verification.csv",
    "group_summary.csv",
    "class_summary.csv",
    "mask_contrast.csv",
    "tail_draws_epoch10.csv",
    "validation_class_deltas.csv",
    "class_level_association.csv",
    "FUSION_SEMANTIC_AUDIT_REPORT.md",
    "run_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_write_new(path: Path, value: Any) -> None:
    path = Path(path).resolve()
    if not path.is_relative_to(RESULTS.resolve()):
        raise RuntimeError(f"Refusing output outside this experiment results/: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def csv_read(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path).resolve()
    if not path.is_relative_to(RESULTS.resolve()):
        raise RuntimeError(f"Refusing output outside this experiment results/: {path}")
    if not rows:
        raise ValueError(f"Refusing to create an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def array_hash(*arrays: np.ndarray) -> str:
    """Match the historical common.array_hash implementation exactly."""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.asarray(array)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def array_diagnostic(array: np.ndarray) -> dict[str, Any]:
    """Describe the exact hash input without casting or changing its bytes."""
    array = np.asarray(array)
    finite = np.isfinite(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "dtype_str": array.dtype.str,
        "strides_bytes": list(array.strides),
        "c_contiguous": bool(array.flags.c_contiguous),
        "elements": int(array.size),
        "all_finite": bool(finite.all()),
        "nonfinite_elements": int(array.size - np.count_nonzero(finite)),
        "single_array_sha256": array_hash(array),
    }


def replay_runtime_diagnostic(backbone: torch.nn.Module | None) -> dict[str, Any]:
    """Read numerical settings at failure time; never change them."""
    cuda_available = torch.cuda.is_available()
    info: dict[str, Any] = {
        "python": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "GPU": torch.cuda.get_device_name(0) if cuda_available else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "grad_enabled_at_diagnostic_capture": torch.is_grad_enabled(),
        "feature_forward_context_in_source": "torch.no_grad()",
        "relevant_environment": {
            name: os.environ.get(name)
            for name in ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
                         "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
        },
    }
    if backbone is not None:
        info["backbone_training"] = backbone.training
        info["trainable_backbone_parameters"] = sum(
            parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
        )
        info["batchnorm_training"] = {
            name: module.training for name, module in backbone.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        }
    return info


def persist_replay_mismatch(
    *, seed: int, epoch: int, batch: int, field: str,
    expected: dict[str, str], observed: dict[str, str],
    checked_rows: list[dict[str, Any]], feature_arrays: dict[str, np.ndarray],
    feature_tensors: dict[str, torch.Tensor],
    backbone: torch.nn.Module | None, checkpoint_sha256: str | None,
) -> Path:
    """Persist failure evidence separately, without touching sealed results."""
    payload: dict[str, Any] = {
        "status": "EXACT_REPLAY_MISMATCH",
        "seed": seed, "epoch": epoch, "batch": batch, "field": field,
        "expected_sha256": expected[field], "observed_sha256": observed[field],
        "features_sha256": {
            "expected": expected["features_sha256"],
            "observed": observed["features_sha256"],
        },
        "checkpoint_sha256": checkpoint_sha256,
        "historical_signatures": str(
            NORM / "results" / f"seed{seed}" / "LowVS-Full" / "batch_signatures.csv"
        ),
        "checks_through_first_mismatch": checked_rows,
        "expected_hashes": dict(expected), "observed_hashes": dict(observed),
        "hash_contract": "receiver then donor; str(shape), str(dtype), numpy.tobytes(order='C'); no casting or tolerance",
        "feature_arrays": {}, "feature_tensors": {},
        "historical_feature_value_comparison": None,
        "historical_feature_value_comparison_note": "No historical augmented feature values supplied; no numerical difference is inferred from hashes.",
        "exact_replay_passed": False,
    }
    for name, array in feature_arrays.items():
        try:
            payload["feature_arrays"][name] = array_diagnostic(array)
        except Exception as error:
            payload["feature_arrays"][name] = {"diagnostic_error": repr(error)}
    for name, tensor in feature_tensors.items():
        payload["feature_tensors"][name] = {
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "stride": list(tensor.stride()),
            "requires_grad": tensor.requires_grad, "contiguous": tensor.is_contiguous(),
        }
    try:
        payload["runtime"] = replay_runtime_diagnostic(backbone)
    except Exception as error:
        payload["runtime"] = {"diagnostic_error": repr(error)}
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    folder = DIAGNOSTICS / f"replay_mismatch_s{seed}_e{epoch}_b{batch}_{uuid.uuid4().hex}"
    folder.mkdir()  # Permission errors must stop immediately, without tempfile's Windows retries.
    path = folder / "replay_mismatch.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def verify_replay_signatures(
    *, seed: int, epoch: int, batch: int,
    expected: dict[str, str], observed: dict[str, str],
    feature_arrays: dict[str, np.ndarray], feature_tensors: dict[str, torch.Tensor],
    backbone: torch.nn.Module | None = None, checkpoint_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Keep the original ordered, exact-equality gate; save evidence before raising."""
    checked_rows: list[dict[str, Any]] = []
    for field in REPLAY_FIELDS:
        matched = expected[field] == observed[field]
        checked_rows.append({
            "seed": seed, "epoch": epoch, "batch": batch, "field": field,
            "match": matched, "expected_sha256": expected[field],
            "observed_sha256": observed[field],
        })
        if not matched:
            message = f"Exact replay mismatch: seed={seed}, epoch={epoch}, batch={batch}, field={field}."
            try:
                path = persist_replay_mismatch(
                    seed=seed, epoch=epoch, batch=batch, field=field,
                    expected=expected, observed=observed, checked_rows=checked_rows,
                    feature_arrays=feature_arrays, feature_tensors=feature_tensors,
                    backbone=backbone, checkpoint_sha256=checkpoint_sha256,
                )
            except Exception as error:
                raise RuntimeError(
                    f"{message} Diagnostic persistence failed; strict replay remains stopped. "
                    "Do not relax the hash check."
                ) from error
            raise RuntimeError(
                f"{message} Diagnostics persisted: {path}. Stop; do not relax the hash check."
            )
    return checked_rows


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norms).all():
        raise ValueError("Non-finite feature or norm encountered")
    if np.any(norms <= eps):
        raise ValueError("Zero/near-zero feature or prototype encountered")
    return values / norms


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Rank input must be a finite one-dimensional array")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        # One-based average rank for tied values.
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Spearman inputs must be equal-length vectors")
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    if np.std(rx) <= 0 or np.std(ry) <= 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def apply_channel_mask(receiver: np.ndarray, donor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """GAP-equivalent of replacing the selected complete feature channels."""
    receiver = np.asarray(receiver)
    donor = np.asarray(donor)
    mask = np.asarray(mask)
    if receiver.shape != donor.shape or receiver.ndim != 2 or receiver.shape[1] != FEATURE_DIM:
        raise ValueError("Expected matching [N,64] receiver/donor GAP features")
    if mask.shape != (len(receiver), MASK_SIZE):
        raise ValueError("Expected [N,16] channel-index mask")
    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError("Channel mask must contain integer channel indices")
    if np.any(mask < 0) or np.any(mask >= FEATURE_DIM):
        raise ValueError("Channel index outside [0,63]")
    if not np.all(np.diff(np.sort(mask, axis=1), axis=1) > 0):
        raise ValueError("Each mask row must contain 16 unique channels")
    output = receiver.copy()
    rows = np.arange(len(receiver))[:, None]
    cols = mask.astype(np.int64, copy=False)
    output[rows, cols] = donor[rows, cols]
    return output


def margin_and_competitor(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or scores.shape != (len(labels), NUM_CLASSES):
        raise ValueError("Expected [N,100] evaluator scores")
    if not np.isfinite(scores).all() or np.any((labels < 0) | (labels >= NUM_CLASSES)):
        raise ValueError("Invalid scores or labels")
    rows = np.arange(len(labels))
    predictions = scores.argmax(axis=1)
    alternatives = scores.copy()
    alternatives[rows, labels] = -np.inf
    competitor = alternatives.argmax(axis=1)
    margin = scores[rows, labels] - alternatives.max(axis=1)
    return predictions.astype(np.int64), margin.astype(np.float64), competitor.astype(np.int64)


def pair_excluded_ncm_scores(
    evaluated_features: np.ndarray,
    receiver_indices: np.ndarray,
    donor_indices: np.ndarray,
    train_labels: np.ndarray,
    normalized_eval_bank: np.ndarray,
    prototype_sums: np.ndarray,
    class_counts: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Cosine-NCM scores with each current receiver/donor identity left out.

    Prototypes are built from deterministic eval-transform TRAIN features.
    For each draw, the eval-bank vectors of that receiver and donor are removed
    from their classes; augmented features are the points being scored. Thus
    the evaluated images do not contribute their own stored prototype vectors.
    """
    z = normalize_rows(np.asarray(evaluated_features, dtype=np.float64))
    receiver_indices = np.asarray(receiver_indices, dtype=np.int64)
    donor_indices = np.asarray(donor_indices, dtype=np.int64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    if not (len(z) == len(receiver_indices) == len(donor_indices)):
        raise ValueError("Pairwise NCM inputs have inconsistent lengths")
    base_prototypes = normalize_rows(prototype_sums)
    result = np.empty((len(z), NUM_CLASSES), dtype=np.float64)

    for start in range(0, len(z), batch_size):
        stop = min(start + batch_size, len(z))
        rows = np.arange(stop - start)
        rec = receiver_indices[start:stop]
        don = donor_indices[start:stop]
        y_rec = train_labels[rec]
        y_don = train_labels[don]
        current = z[start:stop] @ base_prototypes.T

        rec_sum = prototype_sums[y_rec] - normalized_eval_bank[rec]
        rec_count = class_counts[y_rec].astype(np.int64) - 1
        same_class_distinct = (y_don == y_rec) & (don != rec)
        if same_class_distinct.any():
            rec_sum[same_class_distinct] -= normalized_eval_bank[don[same_class_distinct]]
            rec_count[same_class_distinct] -= 1
        if np.any(rec_count <= 0):
            raise ValueError("Leave-pair-out receiver prototype became empty")
        rec_proto = normalize_rows(rec_sum)
        current[rows, y_rec] = np.sum(z[start:stop] * rec_proto, axis=1)

        different_class = y_don != y_rec
        if different_class.any():
            d_sum = prototype_sums[y_don[different_class]] - normalized_eval_bank[don[different_class]]
            d_count = class_counts[y_don[different_class]].astype(np.int64) - 1
            if np.any(d_count <= 0):
                raise ValueError("Leave-pair-out donor prototype became empty")
            d_proto = normalize_rows(d_sum)
            current[rows[different_class], y_don[different_class]] = np.sum(
                z[start:stop][different_class] * d_proto, axis=1
            )
        result[start:stop] = current
    if not np.isfinite(result).all():
        raise ValueError("Non-finite leave-pair-out NCM scores")
    return result


def mask_source_paths(seed: int) -> list[Path]:
    combined = OLD / "plans" / str(seed) / "sampling_masks.npz"
    if combined.is_file():
        return [combined]
    fallback = [
        OLD / f"{MASK_FOLDERS[arm]}_seed{seed}" / "actual_masks.npz"
        for arm in MASK_ARMS
    ]
    if all(path.is_file() for path in fallback):
        return fallback
    missing = [str(path) for path in fallback if not path.is_file()]
    raise FileNotFoundError(
        f"Exact saved channel masks are missing for seed {seed}. "
        f"Expected {combined} or both actual_masks.npz files. Missing: {missing}. "
        "Do not reconstruct or guess historical masks."
    )


def fixed_input_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {
        "stage1_checkpoint": OLD / "locked" / "clean_stage1_best.pth",
        "stage1_protocol": OLD / "clean_stage1_protocol.json",
        "source_lock": OLD / "source_lock.json",
        "train_images": OLD / "train_images.npy",
        "train_labels": OLD / "train_labels.npy",
        "split_indices": OLD / "split_indices.npz",
        "feature_bank": FACTORIAL / "results" / "train_gap_bank.npz",
        "feature_bank_provenance": FACTORIAL / "results" / "train_bank_provenance.json",
    }
    for relative in (
        "common.py",
        "stability_models.py",
        "reused_train_helpers.py",
        "online_policy.py",
        "reused_vs_math.py",
        "runtime/datasets/cifar100.py",
        "runtime/models/resnet_cifar.py",
    ):
        paths[f"historical_source::{relative}"] = OLD / relative
    for seed in SEEDS:
        paths[f"sampling_plan_seed{seed}"] = OLD / "plans" / str(seed) / "sampling_plan.npz"
        paths[f"gumbel_seed{seed}"] = OLD / "plans" / str(seed) / "gumbel_noise_float64.npy"
        paths[f"historical_batch_signatures_seed{seed}"] = NORM / "results" / f"seed{seed}" / "LowVS-Full" / "batch_signatures.csv"
        for index, path in enumerate(mask_source_paths(seed)):
            paths[f"mask_source_seed{seed}_{index}::{path.parent.name}/{path.name}"] = path
        for arm in VALIDATION_ARMS:
            paths[f"validation_seed{seed}_{arm}"] = (
                DISC / "results" / f"seed{seed}" / arm / "validation_epoch10.npz"
            )
    return paths


def build_input_manifest() -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for label, path in fixed_input_paths().items():
        if not path.is_file():
            raise FileNotFoundError(f"Required historical input is missing: {label}: {path}")
        manifest[label] = {
            "path": path.resolve().relative_to(REPO.resolve()).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest["audit_code_self"] = {
        "path": HERE.joinpath("audit_fusion_semantics.py").resolve().relative_to(REPO.resolve()).as_posix(),
        "sha256": sha256_file(HERE / "audit_fusion_semantics.py"),
        "size_bytes": (HERE / "audit_fusion_semantics.py").stat().st_size,
    }
    return manifest


def load_masks(seed: int, receivers: np.ndarray, donors: np.ndarray) -> tuple[dict[str, np.ndarray], list[Path]]:
    source_paths = mask_source_paths(seed)
    if len(source_paths) == 1 and source_paths[0].name == "sampling_masks.npz":
        with np.load(source_paths[0], allow_pickle=False) as archive:
            if "receivers" not in archive or "donors" not in archive:
                raise KeyError(f"Combined mask archive lacks receiver/donor arrays: {source_paths[0]}")
            if not np.array_equal(archive["receivers"], receivers) or not np.array_equal(archive["donors"], donors):
                raise ValueError(f"Combined mask archive pairs do not match sampling_plan.npz for seed {seed}")
            masks = {}
            for arm, key in MASK_KEYS.items():
                if key not in archive:
                    raise KeyError(f"Mask key {key} missing from {source_paths[0]}; keys={archive.files}")
                masks[arm] = archive[key].copy()
    else:
        masks = {}
        for arm, path in zip(MASK_ARMS, source_paths):
            with np.load(path, allow_pickle=False) as archive:
                if "masks" not in archive:
                    raise KeyError(f"Expected key masks in {path}; keys={archive.files}")
                masks[arm] = archive["masks"].copy()

    for arm, mask in masks.items():
        if mask.shape != (EPOCHS, N, MASK_SIZE):
            raise ValueError(f"{arm} mask shape mismatch: {mask.shape}")
        if not np.issubdtype(mask.dtype, np.integer):
            raise TypeError(f"{arm} mask dtype is not integer: {mask.dtype}")
        if np.any(mask < 0) or np.any(mask >= FEATURE_DIM):
            raise ValueError(f"{arm} has invalid channel index")
        if not np.all(np.diff(np.sort(mask, axis=-1), axis=-1) > 0):
            raise ValueError(f"{arm} masks contain duplicate channel indices")
    return masks, source_paths


def load_checkpoint_head(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (("epoch", 184), ("seed", 42)):
        if checkpoint.get(key) != expected:
            raise ValueError(f"Stage-I checkpoint {key}={checkpoint.get(key)}, expected {expected}")
    if bool(checkpoint.get("TEST_used", True)):
        raise ValueError("Locked Stage-I checkpoint does not attest TEST_used=False")
    if checkpoint.get("selection_partition") != "official TRAIN holdout":
        raise ValueError("Unexpected Stage-I checkpoint-selection partition")
    state = checkpoint.get("state_dict_classifier")
    if not isinstance(state, dict):
        raise KeyError("Stage-I checkpoint lacks state_dict_classifier")
    normalized = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[len("module."):]
        normalized[key] = value
    if "fc.weight" not in normalized or "fc.bias" not in normalized:
        raise KeyError("Could not find fc.weight/fc.bias in Stage-I classifier state")
    weight = normalized["fc.weight"].detach().cpu().numpy().astype(np.float64, copy=True)
    bias = normalized["fc.bias"].detach().cpu().numpy().astype(np.float64, copy=True)
    if weight.shape != (NUM_CLASSES, FEATURE_DIM) or bias.shape != (NUM_CLASSES,):
        raise ValueError(f"Unexpected Stage-I FC shapes: {weight.shape}, {bias.shape}")
    return checkpoint, weight, bias


def load_static_data() -> dict[str, Any]:
    stage1_protocol = json_read(OLD / "clean_stage1_protocol.json")
    provenance = json_read(FACTORIAL / "results" / "train_bank_provenance.json")
    checkpoint_path = OLD / "locked" / "clean_stage1_best.pth"
    checkpoint_sha = sha256_file(checkpoint_path)
    expected_sha = stage1_protocol.get("best_checkpoint_sha256")
    if checkpoint_sha != expected_sha:
        raise ValueError("Locked Stage-I checkpoint SHA does not match clean_stage1_protocol.json")
    if provenance.get("StageI_sha256") != checkpoint_sha:
        raise ValueError("Feature-bank Stage-I SHA does not match the locked checkpoint")
    if provenance.get("validation_features_used_for_bank_or_prototypes") is not False:
        raise ValueError("Feature-bank provenance does not confirm TRAIN-only prototypes")
    if sha256_file(FACTORIAL / "results" / "train_gap_bank.npz") != provenance.get("bank_sha256"):
        raise ValueError("Feature-bank SHA does not match train_bank_provenance.json")

    labels = np.load(OLD / "train_labels.npy", allow_pickle=False).astype(np.int64, copy=True)
    split = np.load(OLD / "split_indices.npz", allow_pickle=False)
    train_raw = split["train_raw_indices"].astype(np.int64, copy=True)
    val_raw = split["validation_raw_indices"].astype(np.int64, copy=True)
    if labels.shape != (N,) or train_raw.shape != (N,):
        raise ValueError(f"Expected {N} TRAIN labels/raw IDs, got {labels.shape}/{train_raw.shape}")
    if np.intersect1d(train_raw, val_raw).size:
        raise ValueError("TRAIN and original-TRAIN holdout raw IDs overlap")

    with np.load(FACTORIAL / "results" / "train_gap_bank.npz", allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=True)
        bank_labels = archive["labels"].astype(np.int64, copy=True)
        bank_raw = archive["raw_indices"].astype(np.int64, copy=True)
        bank_stage1_sha = str(np.asarray(archive["stage1_sha256"]).item())
        split_name = str(np.asarray(archive["split"]).item()) if "split" in archive else ""
    if features.shape != (N, FEATURE_DIM) or not np.isfinite(features).all():
        raise ValueError(f"Unexpected/non-finite feature bank shape: {features.shape}")
    if not np.array_equal(bank_labels, labels) or not np.array_equal(bank_raw, train_raw):
        raise ValueError("Feature bank labels/raw IDs do not exactly match the sealed TRAIN order")
    if bank_stage1_sha != checkpoint_sha:
        raise ValueError("Feature-bank embedded Stage-I SHA mismatch")
    if split_name and "TRAIN" not in split_name.upper():
        raise ValueError(f"Feature bank is not explicitly TRAIN-only: {split_name}")
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    if counts.min() <= 0 or not np.array_equal(counts, np.asarray(provenance["class_counts"], dtype=np.int64)):
        raise ValueError("TRAIN class counts do not match feature-bank provenance")

    checkpoint, ce_weight, ce_bias = load_checkpoint_head(checkpoint_path)
    plan_data = {}
    mask_data = {}
    mask_sources = {}
    signatures = {}
    gumbels = {}
    for seed in SEEDS:
        with np.load(OLD / "plans" / str(seed) / "sampling_plan.npz", allow_pickle=False) as archive:
            receivers = archive["receivers"].astype(np.int64, copy=True)
            donors = archive["donors"].astype(np.int64, copy=True)
        if receivers.shape != (EPOCHS, N) or donors.shape != (EPOCHS, N):
            raise ValueError(f"Seed {seed} sampling plan shape mismatch")
        if np.any((receivers < 0) | (receivers >= N)) or np.any((donors < 0) | (donors >= N)):
            raise ValueError(f"Seed {seed} sampling plan has out-of-range indices")
        expected_permutation = np.broadcast_to(np.arange(N), donors.shape)
        if not np.array_equal(np.sort(donors, axis=1), expected_permutation):
            raise ValueError(f"Seed {seed} donor plan is not a per-epoch permutation")
        masks, sources = load_masks(seed, receivers, donors)
        plan_data[seed] = {"receivers": receivers, "donors": donors}
        mask_data[seed] = masks
        mask_sources[seed] = sources
        gumbels[seed] = np.load(
            OLD / "plans" / str(seed) / "gumbel_noise_float64.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if gumbels[seed].shape != (EPOCHS, N, FEATURE_DIM) or not np.isfinite(gumbels[seed]).all():
            raise ValueError(f"Seed {seed} Gumbel array shape/value mismatch")
        sig_path = NORM / "results" / f"seed{seed}" / "LowVS-Full" / "batch_signatures.csv"
        sig = csv_read(sig_path)
        if len(sig) != EPOCHS * BATCHES_PER_EPOCH:
            raise ValueError(f"Seed {seed} expected {EPOCHS * BATCHES_PER_EPOCH} batch signatures, got {len(sig)}")
        signatures[seed] = sig

    validation = {}
    shared_labels = None
    for seed in SEEDS:
        validation[seed] = {}
        for arm in VALIDATION_ARMS:
            path = DISC / "results" / f"seed{seed}" / arm / "validation_epoch10.npz"
            with np.load(path, allow_pickle=False) as archive:
                y = archive["labels"].astype(np.int64, copy=True)
                pred = archive["predictions"].astype(np.int64, copy=True)
                logits = archive["logits"].astype(np.float32, copy=True)
            if y.shape != (1980,) or pred.shape != (1980,) or logits.shape != (1980, NUM_CLASSES):
                raise ValueError(f"Unexpected saved validation shape for seed{seed}/{arm}")
            if not np.isfinite(logits).all() or not np.array_equal(pred, logits.argmax(axis=1)):
                raise ValueError(f"Saved validation logits/predictions mismatch for seed{seed}/{arm}")
            class_counts = np.bincount(y, minlength=NUM_CLASSES)
            if class_counts[0] != 0 or not np.all(class_counts[1:] == 20):
                raise ValueError("Expected reused original-TRAIN holdout: class0 absent, classes1..99 x20")
            if shared_labels is None:
                shared_labels = y
            elif not np.array_equal(shared_labels, y):
                raise ValueError("Validation labels differ across seeds/arms")
            validation[seed][arm] = {"labels": y, "predictions": pred, "logits": logits}

    return {
        "features": features,
        "labels": labels,
        "raw_indices": train_raw,
        "class_counts": counts,
        "ce_weight": ce_weight,
        "ce_bias": ce_bias,
        "checkpoint": checkpoint,
        "plan_data": plan_data,
        "mask_data": mask_data,
        "mask_sources": mask_sources,
        "gumbels": gumbels,
        "signatures": signatures,
        "validation": validation,
        "provenance": provenance,
        "checkpoint_sha256": checkpoint_sha,
    }


def import_historical_helpers():
    if str(OLD) not in sys.path:
        sys.path.insert(0, str(OLD))
    import importlib
    common = importlib.import_module("common")
    stability_models = importlib.import_module("stability_models")
    train_helpers = importlib.import_module("reused_train_helpers")
    stability_models.check_sources()
    return common, stability_models, train_helpers


def replay_epoch10_features(data: dict[str, Any]) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]], str]:
    if not torch.cuda.is_available():
        raise RuntimeError("Exact feature-hash replay requires CUDA, matching the sealed historical run; do not silently downgrade to CPU.")
    common, stability_models, train_helpers = import_historical_helpers()
    cfg, transforms, ResNet, Classifier, CE, methods = stability_models.imports()
    checkpoint_path = OLD / "locked" / "clean_stage1_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch.backends.cudnn.benchmark = False
    # common.seed_all(seed, False) below matches the historical run flags.
    backbone = torch.nn.DataParallel(ResNet()).cuda()
    backbone.load_state_dict(checkpoint["state_dict_model"], strict=True)
    backbone.eval().requires_grad_(False)
    transform = transforms["train_transform"]

    gap_by_seed: dict[int, dict[str, np.ndarray]] = {}
    replay_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        common.seed_all(seed, False)
        generator = torch.Generator().manual_seed(seed + 300009)
        plan = data["plan_data"][seed]
        masks = data["mask_data"][seed]
        noise = data["gumbels"][seed]
        signatures = data["signatures"][seed]
        receiver_gap = np.empty((N, FEATURE_DIM), dtype=np.float32)
        donor_gap = np.empty((N, FEATURE_DIM), dtype=np.float32)

        # Advance the exact historical augmentation RNG through epochs 1..9.
        # The frozen backbone is in eval mode, so skipping these forward passes
        # cannot alter BN state or any random stream used by augmentation.
        for epoch in range(EPOCHS):
            for batch, start in enumerate(range(0, N, BATCH)):
                stop = min(start + BATCH, N)
                rec = plan["receivers"][epoch, start:stop]
                don = plan["donors"][epoch, start:stop]
                ri, di = train_helpers.augment(rec, don, generator, transform)
                if epoch != PRIMARY_EPOCH - 1:
                    continue

                with torch.no_grad():
                    receiver_map = backbone(ri.cuda())
                    donor_map = backbone(di.cuda())
                receiver_np = receiver_map.detach().cpu().numpy()
                donor_np = donor_map.detach().cpu().numpy()
                sig = signatures[epoch * BATCHES_PER_EPOCH + batch]
                expected = {
                    "indices_sha256": sig["indices_sha256"],
                    "images_sha256": sig["images_sha256"],
                    "features_sha256": sig["features_sha256"],
                    "gumbel_sha256": sig["gumbel_sha256"],
                    "uniform_mask_sha256": sig["uniform_mask_sha256"],
                    "lowvs_mask_sha256": sig["lowvs_mask_sha256"],
                }
                observed = {
                    "indices_sha256": common.array_hash(rec, don, data["labels"][rec], data["labels"][don]),
                    "images_sha256": common.array_hash(ri.numpy(), di.numpy()),
                    "features_sha256": common.array_hash(receiver_np, donor_np),
                    "gumbel_sha256": common.array_hash(noise[epoch, start:stop]),
                    "uniform_mask_sha256": common.array_hash(masks["Uniform-PerSample"][epoch, start:stop]),
                    "lowvs_mask_sha256": common.array_hash(masks["Low-VS"][epoch, start:stop]),
                }
                replay_rows.extend(verify_replay_signatures(
                    seed=seed, epoch=PRIMARY_EPOCH, batch=batch,
                    expected=expected, observed=observed,
                    feature_arrays={"receiver": receiver_np, "donor": donor_np},
                    feature_tensors={"receiver": receiver_map, "donor": donor_map},
                    backbone=backbone, checkpoint_sha256=data["checkpoint_sha256"],
                ))
                receiver_gap[start:stop] = F.avg_pool2d(receiver_map, 8).view(-1, FEATURE_DIM).detach().cpu().numpy()
                donor_gap[start:stop] = F.avg_pool2d(donor_map, 8).view(-1, FEATURE_DIM).detach().cpu().numpy()

                # Verify that replacing 16 complete maps and replacing the same
                # 16 GAP coordinates are exactly equivalent up to float tolerance.
                for arm in MASK_ARMS:
                    idx = torch.from_numpy(masks[arm][epoch, start:stop].astype(np.int64, copy=True)).cuda()
                    bool_mask = torch.zeros((stop - start, FEATURE_DIM), dtype=torch.bool, device="cuda")
                    bool_mask.scatter_(1, idx, True)
                    fused_map = torch.where(bool_mask[:, :, None, None], donor_map, receiver_map)
                    fused_gap_map = F.avg_pool2d(fused_map, 8).view(-1, FEATURE_DIM).detach().cpu().numpy()
                    fused_gap_vector = apply_channel_mask(
                        receiver_gap[start:stop], donor_gap[start:stop], masks[arm][epoch, start:stop]
                    )
                    if not np.allclose(fused_gap_map, fused_gap_vector, rtol=1e-6, atol=1e-6):
                        raise RuntimeError("GAP/channel-replacement equivalence check failed")

        gap_by_seed[seed] = {"receiver": receiver_gap, "donor": donor_gap}
    return gap_by_seed, replay_rows, str(torch.cuda.get_device_name(0))


def build_ncm_bank(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = normalize_rows(data["features"].astype(np.float64))
    labels = data["labels"]
    sums = np.stack([normalized[labels == class_id].sum(axis=0) for class_id in range(NUM_CLASSES)])
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.int64)
    if not np.isfinite(sums).all() or np.any(counts <= 0):
        raise ValueError("Invalid TRAIN-only NCM prototype statistics")
    return normalized, sums, counts


def scores_for_evaluator(
    evaluator: str,
    features: np.ndarray,
    receiver_indices: np.ndarray,
    donor_indices: np.ndarray,
    data: dict[str, Any],
    ncm_bank: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    if evaluator == "StageI-CE":
        return features.astype(np.float64) @ data["ce_weight"].T + data["ce_bias"]
    if evaluator == "LOO-NCM":
        normalized_eval, sums, counts = ncm_bank
        return pair_excluded_ncm_scores(
            features,
            receiver_indices,
            donor_indices,
            data["labels"],
            normalized_eval,
            sums,
            counts,
        )
    raise ValueError(f"Unknown evaluator: {evaluator}")


def receiver_group_masks(receiver_labels: np.ndarray, donor_labels: np.ndarray) -> dict[str, np.ndarray]:
    tail = receiver_labels >= 71
    return {
        "All": np.ones(len(receiver_labels), dtype=bool),
        "Head": receiver_labels <= 35,
        "OfficialMedium": (receiver_labels >= 36) & (receiver_labels <= 70),
        "Tail": tail,
        "Tail-cross-class-donor": tail & (donor_labels != receiver_labels),
        "Tail-same-class-donor": tail & (donor_labels == receiver_labels),
    }


def summarize_subset(
    seed: int,
    arm: str,
    evaluator: str,
    group: str,
    selection: np.ndarray,
    receiver_labels: np.ndarray,
    donor_labels: np.ndarray,
    clean_pred: np.ndarray,
    clean_margin: np.ndarray,
    clean_competitor: np.ndarray,
    fused_pred: np.ndarray,
    fused_margin: np.ndarray,
) -> dict[str, Any]:
    n = int(selection.sum())
    if n == 0:
        raise ValueError(f"Empty summary subset: {group}")
    y = receiver_labels[selection]
    yd = donor_labels[selection]
    cp = clean_pred[selection]
    fp = fused_pred[selection]
    cm = clean_margin[selection]
    fm = fused_margin[selection]
    cc = cp == y
    fc = fp == y
    harmful = cc & ~fc
    beneficial = ~cc & fc
    changed = fp != cp
    harmful_cross = harmful & (yd != y)
    donor_takeover = harmful_cross & (fp == yd)
    competitor_takeover = harmful & (fp == clean_competitor[selection])
    positive_clean_margin = cm > 0
    crossed = positive_clean_margin & (fm <= 0)
    return {
        "seed": seed,
        "epoch": PRIMARY_EPOCH,
        "mask_policy": arm,
        "evaluator": evaluator,
        "receiver_group": group,
        "n_draws": n,
        "clean_correct_n": int(cc.sum()),
        "fused_correct_n": int(fc.sum()),
        "clean_correct_pct": 100.0 * float(cc.mean()),
        "fused_correct_pct": 100.0 * float(fc.mean()),
        "net_correct_delta_pp": 100.0 * float(fc.mean() - cc.mean()),
        "harmful_flip_n": int(harmful.sum()),
        "harmful_flip_rate_pct": 100.0 * float(harmful.sum() / cc.sum()) if cc.any() else None,
        "beneficial_flip_n": int(beneficial.sum()),
        "beneficial_flip_rate_pct": 100.0 * float(beneficial.sum() / (~cc).sum()) if (~cc).any() else None,
        "prediction_changed_n": int(changed.sum()),
        "prediction_changed_rate_pct": 100.0 * float(changed.mean()),
        "mean_margin_delta": float(np.mean(fm - cm)),
        "median_margin_delta": float(np.median(fm - cm)),
        "positive_clean_margin_n": int(positive_clean_margin.sum()),
        "margin_crossing_n": int(crossed.sum()),
        "margin_crossing_rate_pct": 100.0 * float(crossed.sum() / positive_clean_margin.sum()) if positive_clean_margin.any() else None,
        "harmful_cross_class_donor_n": int(harmful_cross.sum()),
        "donor_label_takeover_among_harmful_cross_pct": (
            100.0 * float(donor_takeover.sum() / harmful_cross.sum()) if harmful_cross.any() else None
        ),
        "clean_competitor_takeover_among_harmful_pct": (
            100.0 * float(competitor_takeover.sum() / harmful.sum()) if harmful.any() else None
        ),
    }


def compute_audit_tables(data: dict[str, Any], gap_by_seed: dict[int, dict[str, np.ndarray]]) -> dict[str, list[dict[str, Any]]]:
    ncm_bank = build_ncm_bank(data)
    group_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    tail_draw_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        rec = data["plan_data"][seed]["receivers"][PRIMARY_EPOCH - 1]
        don = data["plan_data"][seed]["donors"][PRIMARY_EPOCH - 1]
        y = data["labels"][rec]
        yd = data["labels"][don]
        receiver = gap_by_seed[seed]["receiver"]
        donor = gap_by_seed[seed]["donor"]
        masks = data["mask_data"][seed]
        group_masks = receiver_group_masks(y, yd)
        clean_outputs: dict[str, dict[str, np.ndarray]] = {}
        fused_outputs: dict[str, dict[str, dict[str, np.ndarray]]] = {}

        for evaluator in ("StageI-CE", "LOO-NCM"):
            clean_r_scores = scores_for_evaluator(evaluator, receiver, rec, don, data, ncm_bank)
            clean_d_scores = scores_for_evaluator(evaluator, donor, rec, don, data, ncm_bank)
            clean_pred, clean_margin, clean_comp = margin_and_competitor(clean_r_scores, y)
            donor_pred, donor_margin, donor_comp = margin_and_competitor(clean_d_scores, yd)
            clean_outputs[evaluator] = {
                "pred": clean_pred,
                "margin": clean_margin,
                "competitor": clean_comp,
                "donor_pred": donor_pred,
                "donor_margin": donor_margin,
                "donor_competitor": donor_comp,
            }
            fused_outputs[evaluator] = {}
            for arm in MASK_ARMS:
                fused = apply_channel_mask(receiver, donor, masks[arm][PRIMARY_EPOCH - 1])
                fused_scores = scores_for_evaluator(evaluator, fused, rec, don, data, ncm_bank)
                fused_pred, fused_margin, fused_comp = margin_and_competitor(fused_scores, y)
                fused_outputs[evaluator][arm] = {
                    "pred": fused_pred,
                    "margin": fused_margin,
                    "competitor": fused_comp,
                }
                for group, selection in group_masks.items():
                    group_rows.append(summarize_subset(
                        seed, arm, evaluator, group, selection, y, yd,
                        clean_pred, clean_margin, clean_comp, fused_pred, fused_margin
                    ))
                for class_id in range(NUM_CLASSES):
                    selection = y == class_id
                    if not selection.any():
                        continue
                    class_rows.append(summarize_subset(
                        seed, arm, evaluator, f"class:{class_id}", selection, y, yd,
                        clean_pred, clean_margin, clean_comp, fused_pred, fused_margin
                    ) | {"class_id": class_id, "train_class_count": int(data["class_counts"][class_id])})

        if (y >= 71).any():
            tail_indices = np.flatnonzero(y >= 71)
            for arm in MASK_ARMS:
                for i in tail_indices:
                    row: dict[str, Any] = {
                        "seed": seed,
                        "epoch": PRIMARY_EPOCH,
                        "draw_index": int(i),
                        "mask_policy": arm,
                        "receiver_index": int(rec[i]),
                        "donor_index": int(don[i]),
                        "receiver_raw_index": int(data["raw_indices"][rec[i]]),
                        "donor_raw_index": int(data["raw_indices"][don[i]]),
                        "receiver_label": int(y[i]),
                        "donor_label": int(yd[i]),
                        "cross_class_donor": bool(yd[i] != y[i]),
                        "selected_channel_count": MASK_SIZE,
                    }
                    for ev in ("StageI-CE", "LOO-NCM"):
                        cp = clean_outputs[ev]["pred"][i]
                        fp = fused_outputs[ev][arm]["pred"][i]
                        cc = bool(cp == y[i])
                        fc = bool(fp == y[i])
                        row.update({
                            f"{ev}_clean_pred": int(cp),
                            f"{ev}_clean_margin": float(clean_outputs[ev]["margin"][i]),
                            f"{ev}_clean_competitor": int(clean_outputs[ev]["competitor"][i]),
                            f"{ev}_donor_pred": int(clean_outputs[ev]["donor_pred"][i]),
                            f"{ev}_fused_pred": int(fp),
                            f"{ev}_fused_margin": float(fused_outputs[ev][arm]["margin"][i]),
                            f"{ev}_margin_delta": float(fused_outputs[ev][arm]["margin"][i] - clean_outputs[ev]["margin"][i]),
                            f"{ev}_clean_correct": cc,
                            f"{ev}_fused_correct": fc,
                            f"{ev}_harmful_flip": bool(cc and not fc),
                            f"{ev}_beneficial_flip": bool((not cc) and fc),
                            f"{ev}_semantic_prediction_changed": bool(cp != fp),
                            f"{ev}_donor_label_takeover": bool(yd[i] != y[i] and fp == yd[i]),
                            f"{ev}_clean_competitor_takeover": bool(cc and fp == clean_outputs[ev]["competitor"][i]),
                        })
                    tail_draw_rows.append(row)

        for evaluator in ("StageI-CE", "LOO-NCM"):
            base_clean_pred = clean_outputs[evaluator]["pred"]
            base_clean_margin = clean_outputs[evaluator]["margin"]
            base_competitor = clean_outputs[evaluator]["competitor"]
            uniform = fused_outputs[evaluator]["Uniform-PerSample"]
            lowvs = fused_outputs[evaluator]["Low-VS"]
            for group, selection in group_masks.items():
                if not selection.any():
                    continue
                cc = base_clean_pred[selection] == y[selection]
                u_pred = uniform["pred"][selection]
                l_pred = lowvs["pred"][selection]
                ys = y[selection]
                ds = yd[selection]
                u_harm = cc & (u_pred != ys)
                l_harm = cc & (l_pred != ys)
                u_benefit = (~cc) & (u_pred == ys)
                l_benefit = (~cc) & (l_pred == ys)
                u_correct = u_pred == ys
                l_correct = l_pred == ys
                contrast_rows.append({
                    "seed": seed,
                    "epoch": PRIMARY_EPOCH,
                    "evaluator": evaluator,
                    "receiver_group": group,
                    "n_draws": int(selection.sum()),
                    "uniform_harmful_n": int(u_harm.sum()),
                    "lowvs_harmful_n": int(l_harm.sum()),
                    "harmful_flip_delta_lowvs_minus_uniform_pp": (
                        100.0 * float(l_harm.sum() - u_harm.sum()) / int(cc.sum()) if cc.any() else None
                    ),
                    "uniform_beneficial_n": int(u_benefit.sum()),
                    "lowvs_beneficial_n": int(l_benefit.sum()),
                    "beneficial_flip_delta_lowvs_minus_uniform_pp": (
                        100.0 * float(l_benefit.sum() - u_benefit.sum()) / int((~cc).sum()) if (~cc).any() else None
                    ),
                    "uniform_net_correct_delta_pp": 100.0 * float(u_correct.mean() - cc.mean()),
                    "lowvs_net_correct_delta_pp": 100.0 * float(l_correct.mean() - cc.mean()),
                    "net_correct_delta_lowvs_minus_uniform_pp": 100.0 * float(l_correct.mean() - u_correct.mean()),
                    "fused_prediction_disagreement_n": int((u_pred != l_pred).sum()),
                    "uniform_harmful_lowvs_not_n": int((u_harm & ~(l_pred != ys)).sum()),
                    "lowvs_harmful_uniform_not_n": int((l_harm & ~(u_pred != ys)).sum()),
                    "uniform_beneficial_lowvs_not_n": int((u_benefit & ~(l_pred == ys)).sum()),
                    "lowvs_beneficial_uniform_not_n": int((l_benefit & ~(u_pred == ys)).sum()),
                    "cross_class_donor_n": int((selection & (yd != y)).sum()),
                })

    validation_delta_rows = build_validation_class_deltas(data["validation"])
    association_rows = build_class_level_associations(class_rows, validation_delta_rows)
    return {
        "group_summary": group_rows,
        "class_summary": class_rows,
        "mask_contrast": contrast_rows,
        "tail_draws": tail_draw_rows,
        "validation_class_deltas": validation_delta_rows,
        "class_level_association": association_rows,
    }


def build_validation_class_deltas(validation: dict[int, dict[str, dict[str, np.ndarray]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        labels = validation[seed]["LowVS"]["labels"]
        baseline = validation[seed]["LowVS"]["predictions"]
        for arm in VALIDATION_ARMS[1:]:
            predictions = validation[seed][arm]["predictions"]
            for class_id in VALIDATION_CLASSES:
                select = labels == class_id
                if int(select.sum()) != 20:
                    raise ValueError(f"Validation class {class_id} does not have 20 examples")
                base_correct = int((baseline[select] == labels[select]).sum())
                arm_correct = int((predictions[select] == labels[select]).sum())
                rows.append({
                    "seed": seed,
                    "class_id": class_id,
                    "arm_vs_baseline": f"{arm}-minus-LowVS",
                    "n_validation": int(select.sum()),
                    "baseline_correct_n": base_correct,
                    "arm_correct_n": arm_correct,
                    "delta_correct_n": arm_correct - base_correct,
                    "delta_accuracy_pp": 100.0 * (arm_correct - base_correct) / int(select.sum()),
                })
    return rows


def build_class_level_associations(
    class_rows: list[dict[str, Any]],
    validation_delta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation_map = {
        (int(r["seed"]), int(r["class_id"]), str(r["arm_vs_baseline"])): float(r["delta_accuracy_pp"])
        for r in validation_delta_rows
    }
    risk_map = {
        (int(r["seed"]), str(r["evaluator"]), int(r["class_id"])): r
        for r in class_rows
        if str(r["mask_policy"]) == "Low-VS"
    }
    for seed in SEEDS:
        for evaluator in ("StageI-CE", "LOO-NCM"):
            for scope, allowed in (
                ("all_validation_classes_1_99", set(VALIDATION_CLASSES)),
                ("tail_classes_71_99", set(range(71, 100))),
            ):
                for arm in VALIDATION_ARMS[1:]:
                    outcome_key = f"{arm}-minus-LowVS"
                    for risk_name in ("harmful_flip_rate_pct", "net_correct_delta_pp"):
                        pairs = []
                        for class_id in sorted(allowed):
                            risk = risk_map.get((seed, evaluator, class_id))
                            if risk is None:
                                continue
                            x = risk.get(risk_name)
                            y = validation_map.get((seed, class_id, outcome_key))
                            if x is None or y is None:
                                continue
                            pairs.append((float(x), float(y)))
                        rho = spearman_rho(
                            np.asarray([p[0] for p in pairs], dtype=np.float64),
                            np.asarray([p[1] for p in pairs], dtype=np.float64),
                        ) if pairs else None
                        rows.append({
                            "seed": seed,
                            "evaluator": evaluator,
                            "class_scope": scope,
                            "validation_arm_vs_LowVS": outcome_key,
                            "semantic_risk_metric": risk_name,
                            "n_classes_used": len(pairs),
                            "spearman_rho_descriptive_only": rho,
                            "p_value": None,
                            "interpretation_limit": "class-level association on a repeatedly used original-TRAIN holdout; not causal or independent-test evidence",
                        })
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        output.append("|" + "|".join(str(value) for value in row) + "|")
    return "\n".join(output)


def build_report(tables: dict[str, list[dict[str, Any]]], device_name: str, replay_rows: list[dict[str, Any]]) -> str:
    groups = tables["group_summary"]
    contrasts = tables["mask_contrast"]
    associations = tables["class_level_association"]
    primary_group_rows = [
        r for r in groups
        if r["receiver_group"] in ("Tail", "Tail-cross-class-donor")
        and r["mask_policy"] in MASK_ARMS
    ]
    primary_table = [[
        r["seed"], r["evaluator"], r["receiver_group"], r["mask_policy"], r["n_draws"],
        fmt(r["harmful_flip_rate_pct"]), fmt(r["beneficial_flip_rate_pct"]),
        fmt(r["net_correct_delta_pp"]), fmt(r["mean_margin_delta"], 5),
        fmt(r["donor_label_takeover_among_harmful_cross_pct"]),
    ] for r in primary_group_rows]
    contrast_rows = [
        r for r in contrasts if r["receiver_group"] in ("Tail", "Tail-cross-class-donor")
    ]
    contrast_table = [[
        r["seed"], r["evaluator"], r["receiver_group"],
        fmt(r["harmful_flip_delta_lowvs_minus_uniform_pp"]),
        fmt(r["beneficial_flip_delta_lowvs_minus_uniform_pp"]),
        fmt(r["net_correct_delta_lowvs_minus_uniform_pp"]),
        r["fused_prediction_disagreement_n"],
    ] for r in contrast_rows]
    assoc_table = [[
        r["seed"], r["evaluator"], r["class_scope"], r["validation_arm_vs_LowVS"],
        r["semantic_risk_metric"], r["n_classes_used"], fmt(r["spearman_rho_descriptive_only"], 4),
    ] for r in associations]
    replay_batches = len({(int(r["seed"]), int(r["batch"])) for r in replay_rows})
    replay_fields = len(replay_rows)
    return "\n".join([
        "# Fusion Semantic Audit — fixed epoch 10",
        "",
        "## 1. Scope and integrity",
        "",
        f"- Exact augmented-feature replay: passed for {replay_batches} epoch-10 batches across two seeds; {replay_fields} field-level hashes matched.",
        f"- Device used for exact replay: {device_name}.",
        "- No optimizer, backward pass, Stage-I retraining, classifier training, or parameter update was performed.",
        "- No official CIFAR-100 test images or labels were loaded.",
        "- Main comparison: the same historical receiver/donor pairs under Uniform-PerSample versus Low-VS channel masks; each mask replaces 16 of 64 complete feature channels.",
        "- Evaluators: the locked Stage-I CE head, and leave-pair-out cosine nearest-class-centroid (LOO-NCM) built from deterministic TRAIN-only GAP features.",
        "",
        "## 2. Primary tail-recipient semantic-consistency diagnostics",
        "",
        "Harmful flip means the frozen evaluator predicts the true recipient class on the clean augmented recipient, but no longer predicts that class after channel fusion. Beneficial flip is the reverse. These are evaluator-defined proxies, not ground-truth labels for a hybrid feature.",
        "",
        markdown_table(
            ["Seed", "Evaluator", "Receiver group", "Mask policy", "Draws", "Harmful rate %", "Beneficial rate %", "Net correct Δ pp", "Mean target-margin Δ", "Donor takeover among harmful cross-class %"],
            primary_table,
        ),
        "",
        "Margin units differ between StageI-CE logits and LOO-NCM cosine scores; compare within evaluator, not across evaluator types.",
        "",
        "## 3. Paired mask contrast: Low-VS minus Uniform-PerSample",
        "",
        markdown_table(
            ["Seed", "Evaluator", "Receiver group", "Δ harmful rate pp", "Δ beneficial rate pp", "Δ net evaluator accuracy pp", "Fused prediction disagreement n"],
            contrast_table,
        ),
        "",
        "Both policies share the same receiver/donor pair and augmented features. This contrast isolates the saved channel mask at the feature-evaluation level; it does not isolate the complete end-to-end training effect.",
        "",
        "## 4. Class-level association with prior supervision results",
        "",
        markdown_table(
            ["Seed", "Evaluator", "Class scope", "Validation arm − LowVS", "Semantic risk metric", "Classes used", "Spearman ρ"],
            assoc_table,
        ),
        "",
        "The association is descriptive only: class-level risk on epoch-10 training draws is compared with per-class changes on the repeatedly used 1,980-image original-TRAIN holdout. There are no p-values, no independent-test claim, and no causal interpretation.",
        "",
        "## 5. Interpretation boundary",
        "",
        "- Consistent harmful flips under both evaluators would make recipient-label mismatch plausible, but would not prove it is the cause of downstream accuracy loss.",
        "- Strong disagreement between StageI-CE and LOO-NCM means the diagnosis depends on the evaluator and should be treated as inconclusive.",
        "- A class-level association can guide the next intervention but cannot prove that changing the target would improve generalization.",
        "- This report intentionally does not emit an automatic pass/fail verdict or select a new method. The scientific decision should be made after reviewing the paired counts, class-wise results, evaluator agreement, and seed consistency.",
        "",
        "## 6. Full outputs",
        "",
        "- replay_verification.csv: exact pair/image/feature/Gumbel/mask hash checks for each epoch-10 batch.",
        "- group_summary.csv: receiver-group metrics for each seed, evaluator, and mask policy.",
        "- class_summary.csv: per-class semantic risk and net evaluator-correctness change.",
        "- mask_contrast.csv: paired Low-VS versus Uniform-PerSample differences.",
        "- tail_draws_epoch10.csv: row-level tail-receiver predictions, margins, donor takeover, and harmful/beneficial flags.",
        "- validation_class_deltas.csv: saved per-class epoch-10 changes for LowVS-CleanCE and LowVS-TopCompetitor relative to LowVS.",
        "- class_level_association.csv: descriptive Spearman associations; p_value is intentionally blank.",
        "",
    ])


def validate_static_inputs() -> dict[str, Any]:
    # Importing this historical helper installs no guard and starts no training.
    common, stability_models, train_helpers = import_historical_helpers()
    data = load_static_data()
    return data


def preflight() -> None:
    if (RESULTS / "preflight.json").exists():
        raise FileExistsError("Refusing to overwrite existing preflight.json")
    for name in OUTPUT_FILES:
        if (RESULTS / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {RESULTS / name}")
    data = validate_static_inputs()
    manifest = build_input_manifest()
    summary = {
        "status": "passed",
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "primary_epoch": PRIMARY_EPOCH,
        "seeds": list(SEEDS),
        "train_samples": N,
        "feature_dim": FEATURE_DIM,
        "classes": NUM_CLASSES,
        "mask_channels": MASK_SIZE,
        "checkpoint_sha256": data["checkpoint_sha256"],
        "feature_bank_shape": [N, FEATURE_DIM],
        "validation_protocol": "reused original-TRAIN holdout; class0 absent; classes1..99 x20",
        "input_manifest": manifest,
        "model_training_performed": False,
        "official_TEST_used": False,
        "cuda_required_for_exact_replay": True,
        "interpretation": "preflight only; no semantic results computed",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_write_new(RESULTS / "preflight.json", summary)
    print("PREFLIGHT PASS: source/data locks and required artifacts verified; no training or official TEST access.", flush=True)


def run_audit() -> None:
    preflight_path = RESULTS / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("Run preflight first")
    for name in OUTPUT_FILES:
        if (RESULTS / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {RESULTS / name}")
    pre = json_read(preflight_path)
    if pre.get("status") != "passed":
        raise RuntimeError("preflight.json does not record a passed preflight")
    data = validate_static_inputs()
    current_manifest = build_input_manifest()
    if current_manifest != pre.get("input_manifest"):
        raise RuntimeError("One or more source/input hashes changed after preflight; stop and rerun only after review")

    started = time.perf_counter()
    gap_by_seed, replay_rows, device_name = replay_epoch10_features(data)
    # Do not create any result files until all exact replay checks have passed.
    tables = compute_audit_tables(data, gap_by_seed)
    report = build_report(tables, device_name, replay_rows)

    csv_write_new(RESULTS / "replay_verification.csv", replay_rows)
    csv_write_new(RESULTS / "group_summary.csv", tables["group_summary"])
    csv_write_new(RESULTS / "class_summary.csv", tables["class_summary"])
    csv_write_new(RESULTS / "mask_contrast.csv", tables["mask_contrast"])
    csv_write_new(RESULTS / "tail_draws_epoch10.csv", tables["tail_draws"])
    csv_write_new(RESULTS / "validation_class_deltas.csv", tables["validation_class_deltas"])
    csv_write_new(RESULTS / "class_level_association.csv", tables["class_level_association"])
    with (RESULTS / "FUSION_SEMANTIC_AUDIT_REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write(report)
    output_hashes = {
        path.name: sha256_file(path)
        for path in RESULTS.iterdir()
        if path.is_file() and path.name not in ("run_manifest.json",)
    }
    manifest = {
        "status": "completed",
        "primary_epoch": PRIMARY_EPOCH,
        "seeds": list(SEEDS),
        "device": device_name,
        "elapsed_seconds": time.perf_counter() - started,
        "exact_replay_fields_matched": len(replay_rows),
        "exact_replay_batches": len({(int(r["seed"]), int(r["batch"])) for r in replay_rows}),
        "model_training_performed": False,
        "optimizer_steps": 0,
        "official_TEST_used": False,
        "source_input_manifest": current_manifest,
        "output_sha256": output_hashes,
        "automatic_scientific_verdict": None,
    }
    json_write_new(RESULTS / "run_manifest.json", manifest)
    print(
        f"AUDIT COMPLETE: exact replay verified for {manifest['exact_replay_batches']} batches; "
        f"no training; outputs at {RESULTS}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic-only H2T fusion semantic audit")
    parser.add_argument("action", choices=("preflight", "run"))
    args = parser.parse_args()
    if args.action == "preflight":
        preflight()
    else:
        run_audit()


if __name__ == "__main__":
    main()
